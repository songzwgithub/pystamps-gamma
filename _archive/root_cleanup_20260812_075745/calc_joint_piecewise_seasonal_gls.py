#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint continuous piecewise-linear annual LOS velocity estimation for pySTAMPS
SBAS time series.

Model per PS
------------
    phase(t) = intercept
             + sum_y slope_y * exposure_y(t)
             + annual_sin * sin(2*pi*t/T)
             + annual_cos * cos(2*pi*t/T)
             [+ semiannual harmonics]
             + error

`exposure_y(t)` is the elapsed fraction of a 365.25-day year inside calendar
year y, clipped before/after that year. This parameterization is continuous at
calendar-year boundaries and each `slope_y` is directly interpretable as
rad/year. All years and seasonal coefficients are solved jointly from the full
acquisition time series.

Estimator
---------
  * Stage-8 acquisition phase (`uw_space_time.mat/ph_uw_ts`)
  * SBAS acquisition covariance (`scla2.mat/ifg_vcm`, or network fallback)
  * covariance whitening
  * Huber IRLS robust generalized least squares
  * sandwich covariance with model-based fallback

Primary output is at the original PS coordinates. No raster aggregation is
performed.

LOS sign convention:
    positive velocity = toward satellite
    negative velocity = away from satellite
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class JointModelError(RuntimeError):
    """Fatal joint-model error."""


@dataclass(slots=True)
class Design:
    matrix: np.ndarray
    parameter_names: list[str]
    model_years: list[int]
    year_column_indices: np.ndarray
    annual_sin_index: int
    annual_cos_index: int
    semiannual_sin_index: int | None
    semiannual_cos_index: int | None
    origin: datetime
    t_days: np.ndarray
    year_coverage: list[dict[str, Any]]


@dataclass(slots=True)
class WhitenedContext:
    mask: np.ndarray
    X: np.ndarray
    Xw: np.ndarray
    whitener: np.ndarray
    covariance_meta: dict[str, Any]
    rank: int


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise JointModelError(
            f"Missing package '{name}'. Install it without replacing the "
            f"working NumPy build, for example:\n"
            f"  python -m pip install {pip_name or name}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def load_support(repo_root: Path):
    path = repo_root / "calc_annual_velocity_gls.py"
    if not path.exists():
        raise JointModelError(f"Missing support module: {path}")
    spec = importlib.util.spec_from_file_location("joint_gls_support", path)
    if spec is None or spec.loader is None:
        raise JointModelError(f"Unable to load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def datetime_days(date: datetime, origin: datetime) -> float:
    return (date - origin).total_seconds() / 86400.0


def build_design(
    dates: list[datetime],
    *,
    seasonal_harmonics: int,
    seasonal_period_days: float,
) -> Design:
    if len(dates) < 3:
        raise JointModelError("At least three acquisitions are required")
    if seasonal_harmonics not in {0, 1, 2}:
        raise JointModelError("seasonal_harmonics must be 0, 1 or 2")

    origin = datetime(dates[0].year, 1, 1)
    t_days = np.asarray(
        [datetime_days(date, origin) for date in dates],
        dtype=np.float64,
    )
    model_years = list(range(min(d.year for d in dates), max(d.year for d in dates) + 1))

    columns: list[np.ndarray] = [np.ones(len(dates), dtype=np.float64)]
    names = ["intercept_rad"]
    year_column_indices: list[int] = []
    year_coverage: list[dict[str, Any]] = []

    for year in model_years:
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        start_day = datetime_days(start, origin)
        end_day = datetime_days(end, origin)
        exposure_years = np.clip(t_days - start_day, 0.0, end_day - start_day) / 365.25
        year_column_indices.append(len(columns))
        columns.append(exposure_years)
        names.append(f"slope_{year}_rad_yr")

        year_dates = [date for date in dates if date.year == year]
        span = (
            (max(year_dates) - min(year_dates)).total_seconds() / 86400.0
            if len(year_dates) >= 2
            else 0.0
        )
        year_coverage.append(
            {
                "year": int(year),
                "acquisition_count": int(len(year_dates)),
                "period_start": min(year_dates).strftime("%Y-%m-%d") if year_dates else None,
                "period_end": max(year_dates).strftime("%Y-%m-%d") if year_dates else None,
                "span_days": float(span),
            }
        )

    annual_sin_index = -1
    annual_cos_index = -1
    semiannual_sin_index: int | None = None
    semiannual_cos_index: int | None = None

    if seasonal_harmonics >= 1:
        omega = 2.0 * math.pi / float(seasonal_period_days)
        annual_sin_index = len(columns)
        columns.append(np.sin(omega * t_days))
        names.append("annual_sin_rad")
        annual_cos_index = len(columns)
        columns.append(np.cos(omega * t_days))
        names.append("annual_cos_rad")

    if seasonal_harmonics >= 2:
        omega2 = 4.0 * math.pi / float(seasonal_period_days)
        semiannual_sin_index = len(columns)
        columns.append(np.sin(omega2 * t_days))
        names.append("semiannual_sin_rad")
        semiannual_cos_index = len(columns)
        columns.append(np.cos(omega2 * t_days))
        names.append("semiannual_cos_rad")

    X = np.column_stack(columns).astype(np.float64)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise JointModelError(
            f"Joint design matrix is rank deficient: rank={rank}, parameters={X.shape[1]}"
        )

    return Design(
        matrix=X,
        parameter_names=names,
        model_years=model_years,
        year_column_indices=np.asarray(year_column_indices, dtype=np.int64),
        annual_sin_index=annual_sin_index,
        annual_cos_index=annual_cos_index,
        semiannual_sin_index=semiannual_sin_index,
        semiannual_cos_index=semiannual_cos_index,
        origin=origin,
        t_days=t_days,
        year_coverage=year_coverage,
    )


def pattern_groups(valid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    packed = np.packbits(valid, axis=1, bitorder="little")
    packed = np.ascontiguousarray(packed)
    dtype = np.dtype((np.void, packed.shape[1]))
    keys = packed.view(dtype).reshape(-1)
    _, inverse = np.unique(keys, return_inverse=True)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for group_id in range(int(np.max(inverse)) + 1):
        rows = np.flatnonzero(inverse == group_id)
        groups.append((rows, valid[rows[0], :].copy()))
    return groups


def weighted_fit_general(
    yw: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B = int(yw.shape[0])
    p = int(Xw.shape[1])
    A = np.einsum("bn,np,nq->bpq", weights, Xw, Xw, optimize=True)
    rhs = np.einsum("bn,bn,np->bp", weights, yw, Xw, optimize=True)

    trace = np.trace(A, axis1=1, axis2=2)
    ridge = np.maximum(trace / max(p, 1) * 1.0e-12, 1.0e-15)
    A_reg = A + ridge[:, None, None] * np.eye(p, dtype=np.float64)[None, :, :]

    beta = np.full((B, p), np.nan, dtype=np.float64)
    inverse = np.full((B, p, p), np.nan, dtype=np.float64)
    valid = np.ones(B, dtype=bool)
    try:
        beta = np.linalg.solve(A_reg, rhs[..., None]).squeeze(-1)
        inverse = np.linalg.inv(A_reg)
    except np.linalg.LinAlgError:
        for row in range(B):
            try:
                beta[row] = np.linalg.solve(A_reg[row], rhs[row])
                inverse[row] = np.linalg.inv(A_reg[row])
            except np.linalg.LinAlgError:
                valid[row] = False

    eig = np.linalg.eigvalsh(A_reg)
    condition = np.divide(
        eig[:, -1],
        eig[:, 0],
        out=np.full(B, np.inf, dtype=np.float64),
        where=eig[:, 0] > 0,
    )
    valid &= np.all(np.isfinite(beta), axis=1) & np.isfinite(condition)
    return beta, inverse, condition, valid


def robust_joint_gls_batch(
    y_original: np.ndarray,
    context: WhitenedContext,
    *,
    robust: bool,
    huber_c: float,
    weight_floor: float,
    max_iterations: int,
    convergence: float,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_original, dtype=np.float64)
    X = context.X
    Xw = context.Xw
    yw = y @ context.whitener.T
    B, n = yw.shape
    p = Xw.shape[1]

    weights = np.ones_like(yw, dtype=np.float64)
    beta, A_inv, condition, fit_valid = weighted_fit_general(yw, Xw, weights)
    beta_gls = beta.copy()
    iterations = np.zeros(B, dtype=np.uint8)

    if robust:
        previous = beta.copy()
        for iteration in range(max(1, int(max_iterations))):
            residual = yw - beta @ Xw.T
            center = np.median(residual, axis=1)
            mad = np.median(np.abs(residual - center[:, None]), axis=1)
            scale = 1.4826 * mad
            rms = np.sqrt(np.mean(residual * residual, axis=1))
            scale = np.where(np.isfinite(scale) & (scale > 1.0e-8), scale, rms)
            scale = np.maximum(scale, 1.0e-8)

            standardized = np.abs(residual) / (float(huber_c) * scale[:, None])
            new_weights = np.ones_like(standardized)
            large = standardized > 1.0
            new_weights[large] = 1.0 / standardized[large]
            new_weights = np.clip(new_weights, float(weight_floor), 1.0)

            beta_new, A_inv_new, condition_new, valid_new = weighted_fit_general(
                yw, Xw, new_weights
            )
            denominator = np.maximum(1.0e-8, np.abs(previous))
            relative = np.max(np.abs(beta_new - previous) / denominator, axis=1)
            converged = np.isfinite(relative) & (relative <= float(convergence))
            iterations[~converged] = np.uint8(min(iteration + 1, 255))

            beta = beta_new
            A_inv = A_inv_new
            condition = condition_new
            fit_valid &= valid_new
            weights = new_weights
            previous = beta_new.copy()
            if np.all(converged | ~fit_valid):
                break

    residual_w = yw - beta @ Xw.T
    residual_original = y - beta @ X.T

    psi2 = (weights * residual_w) ** 2
    meat = np.einsum("bn,np,nq->bpq", psi2, Xw, Xw, optimize=True)
    covariance_beta = A_inv @ meat @ A_inv
    dof = np.maximum(np.sum(weights, axis=1) - p, 1.0)
    correction = n / np.maximum(n - p, 1)
    covariance_beta *= correction

    reduced = np.sum(weights * residual_w * residual_w, axis=1) / dof
    fallback = A_inv * reduced[:, None, None]
    diagonal = np.diagonal(covariance_beta, axis1=1, axis2=2).copy()
    fallback_diagonal = np.diagonal(fallback, axis1=1, axis2=2)
    bad = ~np.isfinite(diagonal) | (diagonal <= 0)
    diagonal[bad] = fallback_diagonal[bad]
    beta_se = np.sqrt(np.maximum(diagonal, 0.0))

    effective_n = np.divide(
        np.sum(weights, axis=1) ** 2,
        np.sum(weights * weights, axis=1),
        out=np.zeros(B, dtype=np.float64),
        where=np.sum(weights * weights, axis=1) > 0,
    )
    downweighted = np.sum(weights < 0.999, axis=1).astype(np.uint16)
    fit_valid &= (
        np.all(np.isfinite(beta), axis=1)
        & np.all(np.isfinite(beta_se), axis=1)
        & np.isfinite(condition)
        & (condition < 1.0e14)
    )

    return {
        "beta": beta,
        "beta_gls": beta_gls,
        "beta_se": beta_se,
        "rmse_rad": np.sqrt(np.mean(residual_original * residual_original, axis=1)),
        "whitened_rmse": np.sqrt(np.mean(residual_w * residual_w, axis=1)),
        "effective_n": effective_n,
        "downweighted_mode_count": downweighted,
        "design_condition": condition,
        "irls_iterations": iterations,
        "fit_valid": fit_valid,
    }


def per_year_observation_metrics(
    valid: np.ndarray,
    dates: list[datetime],
    model_years: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    B = valid.shape[0]
    Y = len(model_years)
    counts = np.zeros((B, Y), dtype=np.uint16)
    spans = np.zeros((B, Y), dtype=np.float32)

    for col, year in enumerate(model_years):
        epoch_ix = np.asarray([i for i, date in enumerate(dates) if date.year == year], dtype=np.int64)
        if epoch_ix.size == 0:
            continue
        subset = valid[:, epoch_ix]
        count = np.sum(subset, axis=1)
        counts[:, col] = np.minimum(count, 65535).astype(np.uint16)
        day_values = np.asarray(
            [(dates[int(i)] - datetime(year, 1, 1)).total_seconds() / 86400.0 for i in epoch_ix],
            dtype=np.float64,
        )
        first = np.min(np.where(subset, day_values[None, :], np.inf), axis=1)
        last = np.max(np.where(subset, day_values[None, :], -np.inf), axis=1)
        span = last - first
        span[~np.isfinite(span)] = 0.0
        spans[:, col] = span.astype(np.float32)
    return counts, spans


def adaptive_cap(
    values: np.ndarray,
    *,
    hard_cap: float,
    minimum: float,
    multiplier: float,
) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(hard_cap)
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    return float(max(minimum, min(hard_cap, median + multiplier * max(sigma, 0.0))))


def projected_xy(lon: np.ndarray, lat: np.ndarray, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    pyproj = require_import("pyproj", "pyproj")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, np.float64), np.asarray(y, np.float64)


def local_consistency_flags(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    recommended: np.ndarray,
    *,
    radius_m: float,
    k_neighbors: int,
    min_neighbors: int,
    sigma_multiplier: float,
    absolute_floor_mm_yr: float,
    workers: int,
    query_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scipy_spatial = require_import("scipy.spatial", "scipy")
    tree_class = scipy_spatial.cKDTree
    n_ps = velocity.size
    median_out = np.full(n_ps, np.nan, np.float32)
    sigma_out = np.full(n_ps, np.nan, np.float32)
    count_out = np.zeros(n_ps, np.uint16)
    consistent_out = np.zeros(n_ps, bool)

    good = np.flatnonzero(recommended)
    if good.size < max(2, min_neighbors):
        return median_out, sigma_out, count_out, consistent_out

    coordinates = np.column_stack((x[good], y[good]))
    tree = tree_class(coordinates)
    k = min(good.size, max(2, int(k_neighbors) + 1))

    for start in range(0, good.size, int(query_chunk)):
        stop = min(start + int(query_chunk), good.size)
        distance, index = tree.query(
            coordinates[start:stop],
            k=k,
            distance_upper_bound=float(radius_m),
            workers=max(1, int(workers)),
        )
        if distance.ndim == 1:
            distance = distance[:, None]
            index = index[:, None]
        valid = np.isfinite(distance) & (index < good.size)
        safe = np.where(valid, index, 0)
        neighbors = velocity[good[safe]].astype(np.float64)
        neighbors[~valid] = np.nan
        median = np.nanmedian(neighbors, axis=1)
        mad = np.nanmedian(np.abs(neighbors - median[:, None]), axis=1)
        sigma = 1.4826 * mad
        count = np.sum(valid, axis=1)
        target = good[start:stop]
        threshold = np.maximum(float(absolute_floor_mm_yr), float(sigma_multiplier) * sigma)
        consistent = (
            (count >= int(min_neighbors))
            & np.isfinite(median)
            & np.isfinite(sigma)
            & (np.abs(velocity[target] - median) <= threshold)
        )
        median_out[target] = median.astype(np.float32)
        sigma_out[target] = sigma.astype(np.float32)
        count_out[target] = np.minimum(count, 65535).astype(np.uint16)
        consistent_out[target] = consistent
        print(
            f"[JOINT][LOCAL_QC] {stop}/{good.size} ({100.0 * stop / good.size:.1f}%)",
            flush=True,
        )
    return median_out, sigma_out, count_out, consistent_out


def input_signature(dataset: Path, args: argparse.Namespace, design: Design) -> tuple[dict[str, Any], str]:
    files = []
    for name in (
        "ps2.mat",
        "uw_space_time.mat",
        "scla2.mat",
        "ifgstd2.mat",
        "parms.mat",
        "gacos_correction_debug.json",
        "stage7_sbas_debug.json",
        "stage8_sbas_debug.json",
    ):
        path = dataset / name
        if path.exists():
            stat = path.stat()
            files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "version": 1,
        "files": files,
        "parameter_names": design.parameter_names,
        "model_years": design.model_years,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"overwrite", "resume"}
        },
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def save_chunk(path: Path, signature: str, start: int, stop: int, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = {"signature": np.asarray(signature), "start": np.asarray(start), "stop": np.asarray(stop)}
    payload.update(arrays)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_chunk(path: Path, signature: str, start: int, stop: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        stored = str(np.asarray(archive["signature"]).reshape(-1)[0])
        if stored != signature:
            return None
        if int(archive["start"]) != start or int(archive["stop"]) != stop:
            return None
        return {key: np.asarray(archive[key]) for key in archive.files if key not in {"signature", "start", "stop"}}
    except Exception:
        return None


def write_hdf5(
    path: Path,
    inputs: Any,
    design: Design,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    h5py = require_import("h5py", "h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "pySTAMPS_joint_piecewise_seasonal_robust_GLS"
        h5.attrs["los_sign"] = "positive_toward_satellite"
        h5.attrs["model"] = "continuous calendar-year slopes + common seasonal harmonics"
        h5.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        h5.create_dataset("lon", data=inputs.lon, compression="gzip")
        h5.create_dataset("lat", data=inputs.lat, compression="gzip")
        h5.create_dataset("day", data=inputs.day, compression="gzip")
        h5.create_dataset("date", data=np.asarray(inputs.labels, dtype="S8"), compression="gzip")
        h5.create_dataset("year", data=np.asarray(design.model_years, dtype=np.int16))
        h5.create_dataset("parameter_name", data=np.asarray(design.parameter_names, dtype="S40"))
        for key, value in arrays.items():
            value = np.asarray(value)
            chunks = None
            if value.ndim == 1 and value.size == inputs.n_ps:
                chunks = (min(8192, inputs.n_ps),)
            elif value.ndim == 2 and value.shape[0] == inputs.n_ps:
                chunks = (min(8192, inputs.n_ps), min(8, value.shape[1]))
            kwargs: dict[str, Any] = {"data": value}
            if chunks is not None:
                kwargs["chunks"] = chunks
            if value.size > 1024:
                kwargs.update(
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            h5.create_dataset(key, **kwargs)


def write_gpkg(
    path: Path,
    inputs: Any,
    design: Design,
    arrays: dict[str, np.ndarray],
    target_epsg: int,
) -> bool:
    geopandas = optional_import("geopandas")
    if geopandas is None:
        warnings.warn("GeoPackage skipped because geopandas is unavailable")
        return False
    pandas = require_import("pandas", "pandas")
    columns: dict[str, Any] = {
        "ps_id": np.arange(1, inputs.n_ps + 1, dtype=np.int64),
        "lon": inputs.lon,
        "lat": inputs.lat,
        "model_rms": arrays["model_rmse_mm"],
        "ann_amp": arrays["annual_amplitude_mm"],
        "ann_peak": arrays["annual_peak_day"],
        "fit_ok": arrays["fit_valid"],
    }
    if "semiannual_amplitude_mm" in arrays:
        columns["semi_amp"] = arrays["semiannual_amplitude_mm"]
    for col, year in enumerate(design.model_years):
        columns[f"v{year}"] = arrays["velocity_mm_yr"][:, col]
        columns[f"se{year}"] = arrays["velocity_std_mm_yr"][:, col]
        columns[f"q{year}"] = arrays["recommended"][:, col]
        columns[f"s{year}"] = arrays["strict"][:, col]
        columns[f"sg{year}"] = arrays["significant"][:, col]
        columns[f"n{year}"] = arrays["n_obs_year"][:, col]
        columns[f"sp{year}"] = arrays["span_days_year"][:, col]
    frame = pandas.DataFrame(columns)
    geometry = geopandas.points_from_xy(frame["lon"], frame["lat"])
    gdf = geopandas.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326").to_crs(f"EPSG:{target_epsg}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer="joint_annual_velocity", driver="GPKG")
    return True


def write_year_csvs(
    out_dir: Path,
    inputs: Any,
    design: Design,
    arrays: dict[str, np.ndarray],
) -> None:
    pandas = require_import("pandas", "pandas")
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, year in enumerate(design.model_years):
        recommended = arrays["recommended"][:, col].astype(bool)
        frame = pandas.DataFrame(
            {
                "ps_id": np.arange(1, inputs.n_ps + 1, dtype=np.int64)[recommended],
                "lon": inputs.lon[recommended],
                "lat": inputs.lat[recommended],
                "year": np.full(np.count_nonzero(recommended), year, dtype=np.int16),
                "velocity_mm_yr": arrays["velocity_mm_yr"][recommended, col],
                "velocity_std_mm_yr": arrays["velocity_std_mm_yr"][recommended, col],
                "ci95_low_mm_yr": arrays["ci95_low_mm_yr"][recommended, col],
                "ci95_high_mm_yr": arrays["ci95_high_mm_yr"][recommended, col],
                "significant": arrays["significant"][recommended, col],
                "strict": arrays["strict"][recommended, col],
                "n_obs": arrays["n_obs_year"][recommended, col],
                "span_days": arrays["span_days_year"][recommended, col],
                "model_rmse_mm": arrays["model_rmse_mm"][recommended],
                "annual_amplitude_mm": arrays["annual_amplitude_mm"][recommended],
            }
        )
        frame.to_csv(out_dir / f"joint_velocity_{year}_recommended.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve() if args.out else dataset / "joint_piecewise_seasonal_velocity"
    work_root = out_root / "_work"
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    support = load_support(repo_root)
    inputs = support.load_inputs(dataset, repo_root, args.sigma_floor_deg)
    design = build_design(
        inputs.dates,
        seasonal_harmonics=args.seasonal_harmonics,
        seasonal_period_days=args.seasonal_period_days,
    )
    p = design.matrix.shape[1]
    if inputs.n_epoch < p + args.minimum_dof:
        raise JointModelError(
            f"Only {inputs.n_epoch} acquisitions for {p} parameters; "
            f"need at least {p + args.minimum_dof}"
        )

    signature_payload, signature = input_signature(dataset, args, design)
    n_ps = inputs.n_ps
    Y = len(design.model_years)

    arrays: dict[str, np.ndarray] = {
        "velocity_mm_yr": np.full((n_ps, Y), np.nan, np.float32),
        "velocity_gls_mm_yr": np.full((n_ps, Y), np.nan, np.float32),
        "velocity_std_mm_yr": np.full((n_ps, Y), np.nan, np.float32),
        "ci95_low_mm_yr": np.full((n_ps, Y), np.nan, np.float32),
        "ci95_high_mm_yr": np.full((n_ps, Y), np.nan, np.float32),
        "n_obs_year": np.zeros((n_ps, Y), np.uint16),
        "span_days_year": np.zeros((n_ps, Y), np.float32),
        "model_rmse_mm": np.full(n_ps, np.nan, np.float32),
        "whitened_rmse": np.full(n_ps, np.nan, np.float32),
        "effective_n": np.zeros(n_ps, np.float32),
        "downweighted_mode_count": np.zeros(n_ps, np.uint16),
        "design_condition": np.full(n_ps, np.nan, np.float32),
        "irls_iterations": np.zeros(n_ps, np.uint8),
        "fit_valid": np.zeros(n_ps, np.uint8),
        "annual_sin_rad": np.full(n_ps, np.nan, np.float32),
        "annual_cos_rad": np.full(n_ps, np.nan, np.float32),
        "annual_amplitude_mm": np.full(n_ps, np.nan, np.float32),
        "annual_peak_day": np.full(n_ps, np.nan, np.float32),
    }
    if design.semiannual_sin_index is not None:
        arrays.update(
            {
                "semiannual_sin_rad": np.full(n_ps, np.nan, np.float32),
                "semiannual_cos_rad": np.full(n_ps, np.nan, np.float32),
                "semiannual_amplitude_mm": np.full(n_ps, np.nan, np.float32),
            }
        )

    phase_to_velocity = -inputs.wavelength_m / (4.0 * math.pi) * 1000.0
    phase_to_mm_abs = abs(inputs.wavelength_m / (4.0 * math.pi) * 1000.0)
    context_cache: OrderedDict[bytes, WhitenedContext] = OrderedDict()
    covariance_meta_example: dict[str, Any] = {}

    def context_for(mask: np.ndarray) -> WhitenedContext | None:
        nonlocal covariance_meta_example
        key = np.packbits(mask, bitorder="little").tobytes()
        if key in context_cache:
            context = context_cache.pop(key)
            context_cache[key] = context
            return context
        X = design.matrix[mask, :]
        if X.shape[0] < p + args.minimum_dof or np.linalg.matrix_rank(X) < p:
            return None
        C = inputs.covariance[np.ix_(mask, mask)]
        whitener, meta = support.covariance_whitener(
            C,
            covariance_mode=args.covariance_mode,
            eigen_floor_rel=args.eigen_floor_rel,
        )
        context = WhitenedContext(
            mask=mask.copy(),
            X=X,
            Xw=whitener @ X,
            whitener=whitener,
            covariance_meta=meta,
            rank=int(np.linalg.matrix_rank(X)),
        )
        context_cache[key] = context
        if len(context_cache) > args.context_cache_size:
            context_cache.popitem(last=False)
        if not covariance_meta_example:
            covariance_meta_example = dict(meta)
        return context

    for start in range(0, n_ps, args.chunk_ps):
        stop = min(start + args.chunk_ps, n_ps)
        checkpoint = work_root / f"chunk_{start:07d}_{stop:07d}.npz"
        cached = load_chunk(checkpoint, signature, start, stop) if args.resume else None
        if cached is not None:
            for key, value in cached.items():
                if key in arrays:
                    arrays[key][start:stop] = value
            print(f"[JOINT][RESUME] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)", flush=True)
            continue

        y_chunk = np.asarray(inputs.ph_ts[start:stop, :], dtype=np.float64)
        valid = np.isfinite(y_chunk)
        counts_year, spans_year = per_year_observation_metrics(valid, inputs.dates, design.model_years)
        chunk: dict[str, np.ndarray] = {
            key: np.asarray(value[start:stop]).copy()
            for key, value in arrays.items()
        }
        chunk["n_obs_year"] = counts_year
        chunk["span_days_year"] = spans_year

        for rows, mask in pattern_groups(valid):
            context = context_for(mask)
            if context is None:
                continue
            result = robust_joint_gls_batch(
                y_chunk[rows, :][:, mask],
                context,
                robust=True,
                huber_c=args.huber_c,
                weight_floor=args.weight_floor,
                max_iterations=args.irls_iterations,
                convergence=args.convergence,
            )
            beta = result["beta"]
            beta_gls = result["beta_gls"]
            beta_se = result["beta_se"]
            year_beta = beta[:, design.year_column_indices]
            year_gls = beta_gls[:, design.year_column_indices]
            year_se = beta_se[:, design.year_column_indices]
            velocity = year_beta * phase_to_velocity
            velocity_gls = year_gls * phase_to_velocity
            velocity_se = year_se * abs(phase_to_velocity)

            chunk["velocity_mm_yr"][rows, :] = velocity.astype(np.float32)
            chunk["velocity_gls_mm_yr"][rows, :] = velocity_gls.astype(np.float32)
            chunk["velocity_std_mm_yr"][rows, :] = velocity_se.astype(np.float32)
            chunk["ci95_low_mm_yr"][rows, :] = (velocity - 1.96 * velocity_se).astype(np.float32)
            chunk["ci95_high_mm_yr"][rows, :] = (velocity + 1.96 * velocity_se).astype(np.float32)
            chunk["model_rmse_mm"][rows] = (result["rmse_rad"] * phase_to_mm_abs).astype(np.float32)
            chunk["whitened_rmse"][rows] = result["whitened_rmse"].astype(np.float32)
            chunk["effective_n"][rows] = result["effective_n"].astype(np.float32)
            chunk["downweighted_mode_count"][rows] = result["downweighted_mode_count"].astype(np.uint16)
            chunk["design_condition"][rows] = result["design_condition"].astype(np.float32)
            chunk["irls_iterations"][rows] = result["irls_iterations"].astype(np.uint8)
            chunk["fit_valid"][rows] = result["fit_valid"].astype(np.uint8)

            if design.annual_sin_index >= 0:
                annual_sin = beta[:, design.annual_sin_index]
                annual_cos = beta[:, design.annual_cos_index]
                amplitude = np.hypot(annual_sin, annual_cos) * phase_to_mm_abs
                phase = np.mod(np.arctan2(annual_sin, annual_cos), 2.0 * math.pi)
                peak_day = phase / (2.0 * math.pi) * args.seasonal_period_days
                chunk["annual_sin_rad"][rows] = annual_sin.astype(np.float32)
                chunk["annual_cos_rad"][rows] = annual_cos.astype(np.float32)
                chunk["annual_amplitude_mm"][rows] = amplitude.astype(np.float32)
                chunk["annual_peak_day"][rows] = peak_day.astype(np.float32)

            if design.semiannual_sin_index is not None:
                semi_sin = beta[:, design.semiannual_sin_index]
                semi_cos = beta[:, design.semiannual_cos_index]
                chunk["semiannual_sin_rad"][rows] = semi_sin.astype(np.float32)
                chunk["semiannual_cos_rad"][rows] = semi_cos.astype(np.float32)
                chunk["semiannual_amplitude_mm"][rows] = (
                    np.hypot(semi_sin, semi_cos) * phase_to_mm_abs
                ).astype(np.float32)

        for key in arrays:
            arrays[key][start:stop] = chunk[key]
        save_chunk(checkpoint, signature, start, stop, chunk)
        print(f"[JOINT][FIT] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)", flush=True)

    fit_valid = arrays["fit_valid"].astype(bool)
    rmse_cap = adaptive_cap(
        arrays["model_rmse_mm"][fit_valid],
        hard_cap=args.max_model_rmse_mm,
        minimum=args.min_model_rmse_cap_mm,
        multiplier=args.threshold_mad_multiplier,
    )
    formal_year = np.asarray(
        [
            coverage["acquisition_count"] >= args.min_year_epochs
            and coverage["span_days"] >= args.min_year_span_days
            for coverage in design.year_coverage
        ],
        dtype=bool,
    )

    recommended = np.zeros((n_ps, Y), dtype=np.uint8)
    significant = np.zeros((n_ps, Y), dtype=np.uint8)
    rate_se_caps = np.full(Y, np.nan, dtype=np.float64)
    for col, year in enumerate(design.model_years):
        eligible = (
            fit_valid
            & formal_year[col]
            & (arrays["n_obs_year"][:, col] >= args.min_year_epochs)
            & (arrays["span_days_year"][:, col] >= args.min_year_span_days)
            & np.isfinite(arrays["velocity_mm_yr"][:, col])
            & np.isfinite(arrays["velocity_std_mm_yr"][:, col])
            & (arrays["model_rmse_mm"] <= rmse_cap)
        )
        cap = adaptive_cap(
            arrays["velocity_std_mm_yr"][:, col][eligible],
            hard_cap=args.max_rate_se_mm_yr,
            minimum=args.min_rate_se_cap_mm_yr,
            multiplier=args.threshold_mad_multiplier,
        )
        rate_se_caps[col] = cap
        keep = (
            eligible
            & (arrays["velocity_std_mm_yr"][:, col] <= cap)
            & (np.abs(arrays["velocity_mm_yr"][:, col]) <= args.absolute_rate_cap_mm_yr)
        )
        recommended[:, col] = keep.astype(np.uint8)
        significant[:, col] = (
            keep
            & (
                (arrays["ci95_low_mm_yr"][:, col] > 0)
                | (arrays["ci95_high_mm_yr"][:, col] < 0)
            )
        ).astype(np.uint8)

    strict = recommended.copy()
    local_consistent = np.zeros((n_ps, Y), dtype=np.uint8)
    local_neighbor_count = np.zeros((n_ps, Y), dtype=np.uint16)
    if not args.disable_local_qc:
        epsg = int(args.target_epsg or support.auto_utm_epsg(inputs.lon, inputs.lat))
        x, y = projected_xy(inputs.lon, inputs.lat, epsg)
        for col, year in enumerate(design.model_years):
            rec = recommended[:, col].astype(bool)
            if not np.any(rec):
                strict[:, col] = 0
                continue
            _, _, count, consistent = local_consistency_flags(
                x,
                y,
                arrays["velocity_mm_yr"][:, col],
                rec,
                radius_m=args.local_radius_m,
                k_neighbors=args.local_k,
                min_neighbors=args.local_min_neighbors,
                sigma_multiplier=args.local_sigma_multiplier,
                absolute_floor_mm_yr=args.local_floor_mm_yr,
                workers=args.local_workers,
                query_chunk=args.local_query_chunk,
            )
            local_neighbor_count[:, col] = count
            local_consistent[:, col] = consistent.astype(np.uint8)
            strict[:, col] = (rec & consistent).astype(np.uint8)
    else:
        local_consistent = recommended.copy()
        local_neighbor_count[recommended.astype(bool)] = np.uint16(args.local_min_neighbors)

    arrays["recommended"] = recommended
    arrays["strict"] = strict
    arrays["significant"] = significant
    arrays["local_consistent"] = local_consistent
    arrays["local_neighbor_count"] = local_neighbor_count

    summary = []
    for col, coverage in enumerate(design.year_coverage):
        rec = recommended[:, col].astype(bool)
        strict_mask = strict[:, col].astype(bool)
        values = arrays["velocity_mm_yr"][:, col][rec]
        summary.append(
            {
                **coverage,
                "formal_year": bool(formal_year[col]),
                "rate_se_cap_mm_yr": float(rate_se_caps[col]),
                "recommended_ps": int(np.count_nonzero(rec)),
                "recommended_fraction": float(np.mean(rec)),
                "strict_ps": int(np.count_nonzero(strict_mask)),
                "significant_ps": int(np.count_nonzero(significant[:, col])),
                "velocity_p02_mm_yr": float(np.percentile(values, 2)) if values.size else None,
                "velocity_median_mm_yr": float(np.median(values)) if values.size else None,
                "velocity_mean_mm_yr": float(np.mean(values)) if values.size else None,
                "velocity_p98_mm_yr": float(np.percentile(values, 98)) if values.size else None,
            }
        )

    metadata = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "model": "continuous calendar-year piecewise-linear trend + common seasonal harmonics",
        "estimator": "SBAS acquisition covariance + covariance whitening + Huber robust GLS",
        "covariance_source": inputs.covariance_source,
        "covariance_example": covariance_meta_example,
        "parameter_names": design.parameter_names,
        "model_years": design.model_years,
        "seasonal_harmonics": args.seasonal_harmonics,
        "seasonal_period_days": args.seasonal_period_days,
        "model_rmse_cap_mm": rmse_cap,
        "rate_se_caps_mm_yr": rate_se_caps.tolist(),
        "summary": summary,
        "signature": signature,
        "signature_payload": signature_payload,
        "limitations": [
            "Seasonal coefficients are common across years for each PS, not common across space.",
            "Calendar-year slopes are continuous at year boundaries through cumulative exposure basis functions.",
            "A linear trend is assumed within each calendar year; abrupt intra-year steps are not modeled.",
            "Network covariance assumes independent interferogram errors; Huber IRLS reduces sensitivity to covariance mismatch.",
            "Recommended points do not require local consistency; strict points do.",
        ],
        "duration_sec": time.perf_counter() - started,
    }

    h5_path = out_root / "joint_piecewise_seasonal_velocity.h5"
    write_hdf5(h5_path, inputs, design, arrays, metadata)

    summary_path = out_root / "joint_year_summary.csv"
    pandas = require_import("pandas", "pandas")
    pandas.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")

    gpkg_path = out_root / "joint_piecewise_seasonal_velocity.gpkg"
    epsg = int(args.target_epsg or support.auto_utm_epsg(inputs.lon, inputs.lat))
    gpkg_written = write_gpkg(gpkg_path, inputs, design, arrays, epsg) if args.write_gpkg else False
    if args.write_year_csv:
        write_year_csvs(out_root / "yearly", inputs, design, arrays)

    report_path = out_root / "joint_piecewise_seasonal_report.json"
    metadata["outputs"] = {
        "hdf5": str(h5_path),
        "summary_csv": str(summary_path),
        "gpkg": str(gpkg_path) if gpkg_written else None,
        "yearly_csv_directory": str(out_root / "yearly") if args.write_year_csv else None,
    }
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("Joint piecewise-seasonal robust GLS completed")
    print("============================================================")
    print(f"PS          : {n_ps:,}")
    print(f"Epochs      : {inputs.n_epoch}")
    print(f"Years       : {design.model_years}")
    print(f"Parameters  : {p}")
    print(f"Model RMS cap: {rmse_cap:.3f} mm")
    print(f"HDF5        : {h5_path}")
    print(f"Summary     : {summary_path}")
    print(f"GeoPackage  : {gpkg_path if gpkg_written else 'not generated'}")
    print(f"Report      : {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Joint continuous calendar-year piecewise-linear trend + common "
            "seasonal harmonics using SBAS covariance and Huber robust GLS"
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-root", default="/home/ubuntu/software/pystamps-main")
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunk-ps", type=int, default=2048)
    parser.add_argument("--context-cache-size", type=int, default=16)
    parser.add_argument("--minimum-dof", type=int, default=8)

    parser.add_argument("--covariance-mode", choices=("network", "diagonal", "identity"), default="network")
    parser.add_argument("--eigen-floor-rel", type=float, default=1.0e-6)
    parser.add_argument("--sigma-floor-deg", type=float, default=0.1)
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument("--weight-floor", type=float, default=0.05)
    parser.add_argument("--irls-iterations", type=int, default=8)
    parser.add_argument("--convergence", type=float, default=1.0e-6)

    parser.add_argument("--seasonal-harmonics", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--seasonal-period-days", type=float, default=365.2425)

    parser.add_argument("--min-year-epochs", type=int, default=10)
    parser.add_argument("--min-year-span-days", type=float, default=240.0)
    parser.add_argument("--max-model-rmse-mm", type=float, default=15.0)
    parser.add_argument("--min-model-rmse-cap-mm", type=float, default=8.0)
    parser.add_argument("--max-rate-se-mm-yr", type=float, default=5.0)
    parser.add_argument("--min-rate-se-cap-mm-yr", type=float, default=1.5)
    parser.add_argument("--threshold-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--absolute-rate-cap-mm-yr", type=float, default=150.0)

    parser.add_argument("--disable-local-qc", action="store_true")
    parser.add_argument("--local-radius-m", type=float, default=300.0)
    parser.add_argument("--local-k", type=int, default=12)
    parser.add_argument("--local-min-neighbors", type=int, default=4)
    parser.add_argument("--local-sigma-multiplier", type=float, default=4.5)
    parser.add_argument("--local-floor-mm-yr", type=float, default=5.0)
    parser.add_argument("--local-query-chunk", type=int, default=20000)
    parser.add_argument("--local-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))

    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument("--write-gpkg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-year-csv", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
