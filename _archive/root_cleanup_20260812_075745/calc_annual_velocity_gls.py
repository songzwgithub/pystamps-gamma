#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute calendar-year LOS velocities from pySTAMPS SBAS Stage-8 phase time series
using acquisition covariance propagated from the SBAS interferogram network and
Huber robust generalized least squares (GLS).

Primary inputs
--------------
  ps2.mat                    PS coordinates and acquisition dates
  uw_space_time.mat          Stage-8 final acquisition phase: ph_uw_ts
  scla2.mat                  acquisition covariance: ifg_vcm (preferred)
  ifgstd2.mat                IFG standard deviations (covariance fallback)
  parms.mat                  wavelength, dropped IFGs, SBAS flag
  PATCH_1/ps1.mat            ifgday_ix fallback

Model per PS and calendar year
------------------------------
  phase(t) = intercept + rate_rad_day * t + error

The observation covariance is the calendar-year submatrix of the acquisition
covariance. It is whitened by eigen decomposition. Huber IRLS is then performed
in the whitened domain. Final LOS sign follows the Stage-8 convention:
  positive = toward satellite
  velocity_mm_yr = -rate_rad_day * wavelength/(4*pi) * 1000 * 365.25

This is a scalable, covariance-aware annual trend estimator. The network
covariance assumes independent IFG errors and does not represent every residual
atmospheric/spatial correlation after Stage 8; robust IRLS and sandwich standard
errors reduce sensitivity to that model mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class AnnualGLSError(RuntimeError):
    """Fatal annual GLS processing error."""


@dataclass(slots=True)
class Inputs:
    ph_ts: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    day: np.ndarray
    dates: list[datetime]
    labels: list[str]
    covariance: np.ndarray
    covariance_source: str
    n_ps: int
    n_epoch: int
    n_ifg: int
    wavelength_m: float
    reference_image_ix_1based: int


@dataclass(slots=True)
class Grid:
    epsg: int
    resolution: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    row: np.ndarray
    col: np.ndarray
    inside: np.ndarray
    transform: Any


def _require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return __import__(name)
    except Exception as exc:
        package = pip_name or name
        raise AnnualGLSError(
            f"Missing package '{name}'. Install it without replacing the working "
            f"NumPy build, for example:\n  python -m pip install {package}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(v) for v in arr.reshape(-1)).strip()
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="ignore").strip()
    return str(item).strip()


def _matrix(value: Any, rows: int, name: str, dtype: Any) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise AnnualGLSError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise AnnualGLSError(
            f"{name} shape {arr.shape}; expected first dimension {rows}"
        )
    return np.asarray(arr, dtype=dtype)


def _normalize_ifgday_ix(value: Any, n_ifg: int, n_epoch: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(value, dtype=np.int64))
    if arr.ndim != 2:
        raise AnnualGLSError(f"ifgday_ix must be 2-D, got {arr.shape}")
    if arr.shape == (2, n_ifg):
        arr = arr.T
    if arr.shape != (n_ifg, 2):
        raise AnnualGLSError(
            f"ifgday_ix shape {arr.shape}; expected ({n_ifg}, 2)"
        )
    # Accept 0-based or 1-based input, normalize to 0-based.
    if int(np.min(arr)) >= 1 and int(np.max(arr)) <= n_epoch:
        arr = arr - 1
    if int(np.min(arr)) < 0 or int(np.max(arr)) >= n_epoch:
        raise AnnualGLSError(
            f"ifgday_ix contains indices outside [0, {n_epoch - 1}]"
        )
    if np.any(arr[:, 0] == arr[:, 1]):
        raise AnnualGLSError("ifgday_ix contains zero-length interferograms")
    return arr.astype(np.int64, copy=False)


def matlab_datenum_to_datetime(value: float) -> datetime:
    integer = int(math.floor(float(value)))
    fraction = float(value) - integer
    return (
        datetime.fromordinal(integer)
        + timedelta(days=fraction)
        - timedelta(days=366)
    )


def decode_dates(day: np.ndarray) -> tuple[list[datetime], list[str]]:
    values = np.asarray(day, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise AnnualGLSError("Acquisition-day vector is empty or non-finite")
    median = float(np.median(values))
    if median > 500000:
        dates = [matlab_datenum_to_datetime(v) for v in values]
    elif median > 10_000_000:
        dates = [datetime.strptime(str(int(round(v))), "%Y%m%d") for v in values]
    else:
        origin = datetime(1970, 1, 1)
        dates = [origin + timedelta(days=float(v)) for v in values]
    return dates, [date.strftime("%Y%m%d") for date in dates]


def _drop_set(parms: dict[str, Any]) -> set[int]:
    raw = parms.get("drop_ifg_index")
    if raw is None or np.asarray(raw).size == 0:
        return set()
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    return {
        int(round(v))
        for v in values
        if np.isfinite(v) and int(round(v)) >= 1
    }


def build_network_covariance(
    ifgday_ix_0: np.ndarray,
    ifg_std_deg: np.ndarray,
    n_epoch: int,
    reference_image_0: int,
    dropped_ifg_1based: set[int],
    sigma_floor_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_ifg = int(ifgday_ix_0.shape[0])
    sigma = np.asarray(ifg_std_deg, dtype=np.float64).reshape(-1)
    if sigma.size != n_ifg:
        raise AnnualGLSError(
            f"ifg_std has {sigma.size} values; expected {n_ifg}"
        )

    usable = np.isfinite(sigma) & (sigma > 0)
    if dropped_ifg_1based:
        for index in dropped_ifg_1based:
            if 1 <= index <= n_ifg:
                usable[index - 1] = False
    use_ix = np.flatnonzero(usable)
    if use_ix.size < n_epoch - 1:
        raise AnnualGLSError(
            f"Only {use_ix.size} usable IFGs for {n_epoch} acquisitions"
        )

    sigma_use_deg = np.maximum(sigma[use_ix], float(sigma_floor_deg))
    variance = (sigma_use_deg * math.pi / 180.0) ** 2
    weights = 1.0 / variance

    G = np.zeros((use_ix.size, n_epoch), dtype=np.float64)
    pairs = ifgday_ix_0[use_ix, :]
    row = np.arange(use_ix.size)
    G[row, pairs[:, 0]] = -1.0
    G[row, pairs[:, 1]] = 1.0

    unknown = np.asarray(
        [i for i in range(n_epoch) if i != reference_image_0],
        dtype=np.int64,
    )
    Gr = G[:, unknown]
    H = Gr.T @ (weights[:, None] * Gr)
    rank = int(np.linalg.matrix_rank(H))
    if rank < n_epoch - 1:
        raise AnnualGLSError(
            f"SBAS network is rank deficient after drops: rank={rank}, "
            f"required={n_epoch - 1}"
        )
    cov_unknown = np.linalg.pinv(H, rcond=1.0e-12, hermitian=True)
    covariance = np.zeros((n_epoch, n_epoch), dtype=np.float64)
    covariance[np.ix_(unknown, unknown)] = cov_unknown
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "usable_ifg": int(use_ix.size),
        "dropped_or_invalid_ifg": int(n_ifg - use_ix.size),
        "rank": rank,
        "reference_image_ix_1based": int(reference_image_0 + 1),
        "sigma_floor_deg": float(sigma_floor_deg),
    }


def load_inputs(dataset: Path, repo_root: Path, sigma_floor_deg: float) -> Inputs:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from pystamps.io.mat import read_mat, read_mat_variables
    except Exception as exc:
        raise AnnualGLSError(
            f"Unable to import pystamps.io.mat from {repo_root}: {exc}"
        ) from exc

    required = [
        dataset / "ps2.mat",
        dataset / "uw_space_time.mat",
        dataset / "parms.mat",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AnnualGLSError("Missing required inputs: " + ", ".join(missing))

    ps = read_mat_variables(
        dataset / "ps2.mat",
        ("n_ps", "n_ifg", "n_image", "lonlat", "day", "master_ix", "ifgday_ix"),
    )
    n_ps = int(round(_scalar(ps.get("n_ps"), 0)))
    n_ifg = int(round(_scalar(ps.get("n_ifg"), 0)))
    if n_ps <= 0 or n_ifg <= 0:
        raise AnnualGLSError("ps2.mat contains invalid n_ps or n_ifg")
    lonlat = _matrix(ps["lonlat"], n_ps, "ps2.lonlat", np.float64)
    if lonlat.shape[1] < 2:
        raise AnnualGLSError("ps2.lonlat requires at least two columns")

    uw = read_mat_variables(
        dataset / "uw_space_time.mat",
        ("ph_uw_ts", "day", "ifgday_ix", "reference_image_ix"),
    )
    ph_ts = _matrix(
        uw["ph_uw_ts"], n_ps, "uw_space_time.ph_uw_ts", np.float32
    )
    day = np.asarray(uw.get("day", ps.get("day")), dtype=np.float64).reshape(-1)
    n_epoch = int(day.size)
    if ph_ts.shape[1] != n_epoch:
        raise AnnualGLSError(
            f"ph_uw_ts has {ph_ts.shape[1]} epochs but day has {n_epoch}"
        )

    dates, labels = decode_dates(day)
    parms = read_mat(dataset / "parms.mat")
    if _text(parms.get("small_baseline_flag"), "n").lower() != "y":
        raise AnnualGLSError("This script requires small_baseline_flag='y'")
    wavelength_m = float(_scalar(parms.get("lambda"), 0.0555))
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise AnnualGLSError("Invalid radar wavelength in parms.mat")

    reference_1 = int(round(_scalar(
        uw.get("reference_image_ix", ps.get("master_ix")), 1
    )))
    if reference_1 < 1 or reference_1 > n_epoch:
        reference_1 = 1
    reference_0 = reference_1 - 1

    covariance = None
    covariance_source = ""
    scla_path = dataset / "scla2.mat"
    if scla_path.exists():
        try:
            scla = read_mat_variables(scla_path, ("ifg_vcm",))
            candidate = np.squeeze(np.asarray(scla.get("ifg_vcm"), dtype=np.float64))
            if candidate.shape == (n_epoch, n_epoch) and np.all(np.isfinite(candidate)):
                covariance = 0.5 * (candidate + candidate.T)
                covariance_source = "scla2.mat:ifg_vcm"
        except Exception as exc:
            warnings.warn(f"Unable to use scla2.ifg_vcm; rebuilding covariance: {exc}")

    if covariance is None:
        ifgday_raw = uw.get("ifgday_ix", ps.get("ifgday_ix"))
        network_source = "uw_space_time.mat/ps2.mat"
        if ifgday_raw is None or np.asarray(ifgday_raw).size == 0:
            patch_ps = dataset / "PATCH_1" / "ps1.mat"
            if not patch_ps.exists():
                raise AnnualGLSError("Cannot locate ifgday_ix for covariance reconstruction")
            network = read_mat_variables(patch_ps, ("ifgday_ix",))
            ifgday_raw = network.get("ifgday_ix")
            network_source = str(patch_ps)
        ifgday_ix = _normalize_ifgday_ix(ifgday_raw, n_ifg, n_epoch)
        ifgstd_path = dataset / "ifgstd2.mat"
        if not ifgstd_path.exists():
            raise AnnualGLSError("Missing ifgstd2.mat for covariance reconstruction")
        ifg_std = np.asarray(
            read_mat_variables(ifgstd_path, ("ifg_std",))["ifg_std"],
            dtype=np.float64,
        ).reshape(-1)
        covariance, meta = build_network_covariance(
            ifgday_ix,
            ifg_std,
            n_epoch,
            reference_0,
            _drop_set(parms),
            sigma_floor_deg,
        )
        covariance_source = f"reconstructed:{network_source};{json.dumps(meta, sort_keys=True)}"

    if covariance.shape != (n_epoch, n_epoch):
        raise AnnualGLSError(
            f"Acquisition covariance shape {covariance.shape}; expected ({n_epoch}, {n_epoch})"
        )

    return Inputs(
        ph_ts=ph_ts,
        lon=lonlat[:, 0].astype(np.float64),
        lat=lonlat[:, 1].astype(np.float64),
        day=day,
        dates=dates,
        labels=labels,
        covariance=covariance,
        covariance_source=covariance_source,
        n_ps=n_ps,
        n_epoch=n_epoch,
        n_ifg=n_ifg,
        wavelength_m=wavelength_m,
        reference_image_ix_1based=reference_1,
    )


def covariance_whitener(
    covariance: np.ndarray,
    *,
    covariance_mode: str,
    eigen_floor_rel: float,
) -> tuple[np.ndarray, dict[str, float]]:
    C = np.asarray(covariance, dtype=np.float64)
    C = 0.5 * (C + C.T)
    n = C.shape[0]
    mode = covariance_mode.lower()
    if mode == "identity":
        C = np.eye(n, dtype=np.float64)
    elif mode == "diagonal":
        diagonal = np.diag(C).copy()
        positive = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
        replacement = float(np.median(positive)) if positive.size else 1.0
        diagonal[~np.isfinite(diagonal) | (diagonal <= 0)] = replacement
        C = np.diag(diagonal)
    elif mode != "network":
        raise AnnualGLSError(f"Unsupported covariance mode: {covariance_mode}")

    eigval, eigvec = np.linalg.eigh(C)
    positive = eigval[np.isfinite(eigval) & (eigval > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    max_eig = float(np.max(positive)) if positive.size else 1.0
    floor = max(scale * float(eigen_floor_rel), max_eig * 1.0e-12, 1.0e-15)
    clipped = np.maximum(eigval, floor)
    whitener = (eigvec / np.sqrt(clipped)[None, :]).T
    condition = float(np.max(clipped) / np.min(clipped))
    return whitener, {
        "covariance_eigen_min_raw": float(np.min(eigval)),
        "covariance_eigen_max_raw": float(np.max(eigval)),
        "covariance_eigen_floor": float(floor),
        "covariance_condition_after_floor": condition,
    }


def _weighted_line_fit(
    y: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    w = np.asarray(weights, dtype=np.float64)
    S00 = np.sum(w * x0[None, :] * x0[None, :], axis=1)
    S01 = np.sum(w * x0[None, :] * x1[None, :], axis=1)
    S11 = np.sum(w * x1[None, :] * x1[None, :], axis=1)
    b0 = np.sum(w * x0[None, :] * y, axis=1)
    b1 = np.sum(w * x1[None, :] * y, axis=1)
    determinant = S00 * S11 - S01 * S01
    valid = np.isfinite(determinant) & (determinant > 1.0e-18)

    beta0 = np.full(y.shape[0], np.nan, dtype=np.float64)
    beta1 = np.full(y.shape[0], np.nan, dtype=np.float64)
    beta0[valid] = (b0[valid] * S11[valid] - b1[valid] * S01[valid]) / determinant[valid]
    beta1[valid] = (b1[valid] * S00[valid] - b0[valid] * S01[valid]) / determinant[valid]

    inv00 = np.full_like(beta0, np.nan)
    inv01 = np.full_like(beta0, np.nan)
    inv11 = np.full_like(beta0, np.nan)
    inv00[valid] = S11[valid] / determinant[valid]
    inv01[valid] = -S01[valid] / determinant[valid]
    inv11[valid] = S00[valid] / determinant[valid]
    return beta0, beta1, (S00, S01, S11, inv00, inv01, inv11, valid)


def robust_gls_batch(
    y_original: np.ndarray,
    t_days: np.ndarray,
    covariance: np.ndarray,
    *,
    covariance_mode: str,
    eigen_floor_rel: float,
    robust: bool,
    huber_c: float,
    weight_floor: float,
    max_iterations: int,
    convergence: float,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_original, dtype=np.float64)
    t = np.asarray(t_days, dtype=np.float64).reshape(-1)
    if y.ndim != 2 or y.shape[1] != t.size:
        raise AnnualGLSError("robust_gls_batch received incompatible y/t shapes")

    X = np.column_stack((np.ones(t.size, dtype=np.float64), t))
    whitener, covariance_meta = covariance_whitener(
        covariance,
        covariance_mode=covariance_mode,
        eigen_floor_rel=eigen_floor_rel,
    )
    Xw = whitener @ X
    yw = y @ whitener.T
    x0 = Xw[:, 0]
    x1 = Xw[:, 1]

    weights = np.ones_like(yw, dtype=np.float64)
    beta0, beta1, fit_meta = _weighted_line_fit(yw, x0, x1, weights)
    iterations = np.zeros(y.shape[0], dtype=np.uint8)

    if robust:
        previous = beta1.copy()
        for iteration in range(max(1, int(max_iterations))):
            residual = yw - beta0[:, None] * x0[None, :] - beta1[:, None] * x1[None, :]
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

            beta0_new, beta1_new, fit_meta_new = _weighted_line_fit(
                yw, x0, x1, new_weights
            )
            delta = np.abs(beta1_new - previous)
            tolerance = float(convergence) * np.maximum(1.0e-8, np.abs(previous))
            converged = np.isfinite(delta) & (delta <= tolerance)
            iterations[~converged] = np.uint8(min(iteration + 1, 255))

            weights = new_weights
            beta0, beta1, fit_meta = beta0_new, beta1_new, fit_meta_new
            previous = beta1_new.copy()
            if np.all(converged | ~np.isfinite(beta1_new)):
                break

    S00, S01, S11, inv00, inv01, inv11, fit_valid = fit_meta
    residual_w = yw - beta0[:, None] * x0[None, :] - beta1[:, None] * x1[None, :]
    residual_original = y - beta0[:, None] - beta1[:, None] * t[None, :]

    # Robust sandwich covariance in whitened space.
    psi = weights * residual_w
    psi2 = psi * psi
    B00 = np.sum(psi2 * x0[None, :] * x0[None, :], axis=1)
    B01 = np.sum(psi2 * x0[None, :] * x1[None, :], axis=1)
    B11 = np.sum(psi2 * x1[None, :] * x1[None, :], axis=1)
    slope_variance = (
        inv01 * inv01 * B00
        + 2.0 * inv01 * inv11 * B01
        + inv11 * inv11 * B11
    )
    n_obs = y.shape[1]
    if n_obs > 2:
        slope_variance *= n_obs / float(n_obs - 2)

    # Fallback model-based covariance when the sandwich estimate degenerates.
    reduced = np.sum(weights * residual_w * residual_w, axis=1) / np.maximum(
        np.sum(weights, axis=1) - 2.0, 1.0
    )
    fallback_variance = inv11 * reduced
    use_fallback = ~np.isfinite(slope_variance) | (slope_variance <= 0)
    slope_variance[use_fallback] = fallback_variance[use_fallback]
    slope_se = np.sqrt(np.maximum(slope_variance, 0.0))

    effective_n = np.divide(
        np.sum(weights, axis=1) ** 2,
        np.sum(weights * weights, axis=1),
        out=np.zeros(y.shape[0], dtype=np.float64),
        where=np.sum(weights * weights, axis=1) > 0,
    )
    downweighted = np.sum(weights < 0.999, axis=1).astype(np.uint16)

    trace = S00 + S11
    discriminant = np.sqrt(np.maximum((S00 - S11) ** 2 + 4.0 * S01 * S01, 0.0))
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    design_condition = np.divide(
        lambda_max,
        lambda_min,
        out=np.full_like(lambda_max, np.inf),
        where=lambda_min > 0,
    )

    return {
        "intercept_rad": beta0,
        "slope_rad_day": beta1,
        "slope_se_rad_day": slope_se,
        "rmse_rad": np.sqrt(np.mean(residual_original * residual_original, axis=1)),
        "whitened_rmse": np.sqrt(np.mean(residual_w * residual_w, axis=1)),
        "effective_n": effective_n,
        "downweighted_count": downweighted,
        "design_condition": design_condition,
        "irls_iterations": iterations,
        "fit_valid": fit_valid,
        "covariance_meta": covariance_meta,
    }


def ols_slope_batch(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    centered = t - float(np.mean(t))
    denom = float(np.sum(centered * centered))
    if denom <= 0:
        return np.full(y.shape[0], np.nan, dtype=np.float64)
    means = np.mean(y, axis=1)
    return ((y - means[:, None]) @ centered) / denom


def _pattern_groups(valid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    packed = np.packbits(valid, axis=1, bitorder="little")
    packed = np.ascontiguousarray(packed)
    key_dtype = np.dtype((np.void, packed.shape[1]))
    keys = packed.view(key_dtype).reshape(-1)
    unique, inverse = np.unique(keys, return_inverse=True)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for group_id in range(unique.size):
        rows = np.flatnonzero(inverse == group_id)
        mask = valid[rows[0], :].copy()
        groups.append((rows, mask))
    return groups


def compute_year(
    inputs: Inputs,
    epoch_ix: np.ndarray,
    *,
    min_epochs: int,
    min_span_days: float,
    chunk_ps: int,
    covariance_mode: str,
    eigen_floor_rel: float,
    robust: bool,
    huber_c: float,
    weight_floor: float,
    max_iterations: int,
    convergence: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    epoch_ix = np.asarray(epoch_ix, dtype=np.int64)
    dates = [inputs.dates[int(index)] for index in epoch_ix]
    origin = min(dates)
    t_all = np.asarray(
        [(date - origin).total_seconds() / 86400.0 for date in dates],
        dtype=np.float64,
    )
    covariance_all = inputs.covariance[np.ix_(epoch_ix, epoch_ix)]

    n_ps = inputs.n_ps
    output = {
        "velocity_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "velocity_std_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "velocity_gls_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "velocity_ols_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "ci95_low_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "ci95_high_mm_yr": np.full(n_ps, np.nan, dtype=np.float32),
        "rmse_mm": np.full(n_ps, np.nan, dtype=np.float32),
        "whitened_rmse": np.full(n_ps, np.nan, dtype=np.float32),
        "n_obs": np.zeros(n_ps, dtype=np.uint16),
        "span_days": np.zeros(n_ps, dtype=np.float32),
        "effective_n": np.zeros(n_ps, dtype=np.float32),
        "downweighted_mode_count": np.zeros(n_ps, dtype=np.uint16),
        "design_condition": np.full(n_ps, np.nan, dtype=np.float32),
        "irls_iterations": np.zeros(n_ps, dtype=np.uint8),
        "accepted": np.zeros(n_ps, dtype=np.uint8),
    }

    phase_to_velocity = (
        -inputs.wavelength_m / (4.0 * math.pi) * 1000.0 * 365.25
    )
    phase_to_mm = abs(inputs.wavelength_m / (4.0 * math.pi) * 1000.0)
    year_cov_meta: dict[str, Any] = {}

    for start in range(0, n_ps, int(chunk_ps)):
        stop = min(start + int(chunk_ps), n_ps)
        y_chunk = np.asarray(inputs.ph_ts[start:stop, :][:, epoch_ix], dtype=np.float64)
        valid = np.isfinite(y_chunk)
        groups = _pattern_groups(valid)

        for rows_local, mask in groups:
            n_obs = int(np.count_nonzero(mask))
            global_rows = start + rows_local
            output["n_obs"][global_rows] = np.uint16(min(n_obs, 65535))
            if n_obs == 0:
                continue
            t = t_all[mask]
            span = float(np.max(t) - np.min(t)) if t.size else 0.0
            output["span_days"][global_rows] = np.float32(span)
            if n_obs < int(min_epochs) or span < float(min_span_days):
                continue

            y = y_chunk[rows_local, :][:, mask]
            C = covariance_all[np.ix_(mask, mask)]

            # Non-robust GLS diagnostic.
            gls = robust_gls_batch(
                y,
                t,
                C,
                covariance_mode=covariance_mode,
                eigen_floor_rel=eigen_floor_rel,
                robust=False,
                huber_c=huber_c,
                weight_floor=weight_floor,
                max_iterations=1,
                convergence=convergence,
            )
            final = gls
            if robust:
                final = robust_gls_batch(
                    y,
                    t,
                    C,
                    covariance_mode=covariance_mode,
                    eigen_floor_rel=eigen_floor_rel,
                    robust=True,
                    huber_c=huber_c,
                    weight_floor=weight_floor,
                    max_iterations=max_iterations,
                    convergence=convergence,
                )

            robust_velocity = final["slope_rad_day"] * phase_to_velocity
            robust_std = final["slope_se_rad_day"] * abs(phase_to_velocity)
            gls_velocity = gls["slope_rad_day"] * phase_to_velocity
            ols_velocity = ols_slope_batch(y, t) * phase_to_velocity
            fit_valid = np.asarray(final["fit_valid"], dtype=bool)
            finite = (
                fit_valid
                & np.isfinite(robust_velocity)
                & np.isfinite(robust_std)
                & (final["design_condition"] < 1.0e14)
            )

            output["velocity_mm_yr"][global_rows] = robust_velocity.astype(np.float32)
            output["velocity_std_mm_yr"][global_rows] = robust_std.astype(np.float32)
            output["velocity_gls_mm_yr"][global_rows] = gls_velocity.astype(np.float32)
            output["velocity_ols_mm_yr"][global_rows] = ols_velocity.astype(np.float32)
            output["ci95_low_mm_yr"][global_rows] = (robust_velocity - 1.96 * robust_std).astype(np.float32)
            output["ci95_high_mm_yr"][global_rows] = (robust_velocity + 1.96 * robust_std).astype(np.float32)
            output["rmse_mm"][global_rows] = (final["rmse_rad"] * phase_to_mm).astype(np.float32)
            output["whitened_rmse"][global_rows] = final["whitened_rmse"].astype(np.float32)
            output["effective_n"][global_rows] = final["effective_n"].astype(np.float32)
            output["downweighted_mode_count"][global_rows] = final["downweighted_count"].astype(np.uint16)
            output["design_condition"][global_rows] = final["design_condition"].astype(np.float32)
            output["irls_iterations"][global_rows] = final["irls_iterations"].astype(np.uint8)
            output["accepted"][global_rows] = finite.astype(np.uint8)

            # Do not leave rejected estimates looking valid.
            rejected_rows = global_rows[~finite]
            if rejected_rows.size:
                for key in (
                    "velocity_mm_yr",
                    "velocity_std_mm_yr",
                    "ci95_low_mm_yr",
                    "ci95_high_mm_yr",
                ):
                    output[key][rejected_rows] = np.nan

            if not year_cov_meta:
                year_cov_meta = dict(final["covariance_meta"])

        print(
            f"[ANNUAL_GLS] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    metadata = {
        "calendar_acquisitions": int(epoch_ix.size),
        "calendar_start": min(dates).strftime("%Y-%m-%d"),
        "calendar_end": max(dates).strftime("%Y-%m-%d"),
        "calendar_span_days": float((max(dates) - min(dates)).total_seconds() / 86400.0),
        "covariance": year_cov_meta,
    }
    return output, metadata


def auto_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = max(1, min(60, int(math.floor((lon0 + 180.0) / 6.0)) + 1))
    return (32600 if lat0 >= 0 else 32700) + zone


def build_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    resolution_m: float,
    target_epsg: int | None,
    template_raster: Path | None,
) -> Grid:
    rasterio = _require_import("rasterio", "rasterio")
    pyproj = _require_import("pyproj", "pyproj")
    if template_raster is not None and template_raster.exists():
        with rasterio.open(template_raster) as src:
            epsg_value = src.crs.to_epsg() if src.crs is not None else None
            if epsg_value is None:
                raise AnnualGLSError(f"Template raster has no EPSG CRS: {template_raster}")
            epsg = int(epsg_value)
            resolution = float(abs(src.transform.a))
            xmin, ymin, xmax, ymax = map(float, src.bounds)
            width, height = int(src.width), int(src.height)
            transform = src.transform
    else:
        from rasterio.transform import from_origin
        epsg = int(target_epsg or auto_utm_epsg(lon, lat))
        resolution = float(resolution_m)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        xmin = math.floor(float(np.min(x[finite])) / resolution) * resolution - resolution
        xmax = math.ceil(float(np.max(x[finite])) / resolution) * resolution + resolution
        ymin = math.floor(float(np.min(y[finite])) / resolution) * resolution - resolution
        ymax = math.ceil(float(np.max(y[finite])) / resolution) * resolution + resolution
        width = int(round((xmax - xmin) / resolution))
        height = int(round((ymax - ymin) / resolution))
        transform = from_origin(xmin, ymax, resolution, resolution)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    col = np.floor((x - xmin) / resolution).astype(np.int64)
    row = np.floor((ymax - y) / resolution).astype(np.int64)
    inside = (
        np.isfinite(x) & np.isfinite(y)
        & (row >= 0) & (row < height)
        & (col >= 0) & (col < width)
    )
    return Grid(epsg, resolution, xmin, ymin, xmax, ymax, width, height, row, col, inside, transform)


def aggregate_mean(values: np.ndarray, grid: Grid, min_points: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = grid.inside & np.isfinite(values)
    ncell = grid.width * grid.height
    if not np.any(valid):
        return (
            np.full((grid.height, grid.width), np.nan, dtype=np.float32),
            np.zeros((grid.height, grid.width), dtype=np.uint32),
        )
    linear = grid.row[valid] * grid.width + grid.col[valid]
    counts = np.bincount(linear, minlength=ncell).astype(np.uint32)
    sums = np.bincount(linear, weights=values[valid], minlength=ncell).astype(np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.full(ncell, np.nan, dtype=np.float64),
        where=counts >= max(1, int(min_points)),
    )
    return means.reshape(grid.height, grid.width).astype(np.float32), counts.reshape(grid.height, grid.width)


def write_geotiff(
    path: Path,
    array: np.ndarray,
    grid: Grid,
    *,
    description: str,
    unit: str,
    year: int,
    resampling: str = "average",
) -> None:
    rasterio = _require_import("rasterio", "rasterio")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(array, dtype=np.float32).copy()
    out[~np.isfinite(out)] = -9999.0
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{grid.epsg}",
        "transform": grid.transform,
        "nodata": -9999.0,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            year=str(year),
            unit=unit,
            description=description,
            los_sign="positive_toward_satellite",
            estimator="network_covariance_Huber_robust_GLS",
            rasterization="mean_of_valid_PS_values_in_cell",
        )
        factors = [f for f in (2, 4, 8, 16, 32) if grid.width // f >= 1 and grid.height // f >= 1]
        if factors:
            overview_resampling = (
                rasterio.enums.Resampling.nearest
                if resampling == "nearest"
                else rasterio.enums.Resampling.average
            )
            dst.build_overviews(factors, overview_resampling)
            dst.update_tags(ns="rio_overview", resampling=resampling)


def reproject_wgs84(src_path: Path, dst_path: Path) -> None:
    rasterio = _require_import("rasterio", "rasterio")
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs="EPSG:4326", transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
                resampling=Resampling.bilinear,
            )
            dst.update_tags(**src.tags())


def select_reference_indices(
    mode: str,
    lon: np.ndarray,
    lat: np.ndarray,
    grid: Grid,
    *,
    ref_lon: float | None,
    ref_lat: float | None,
    ref_radius_m: float,
    ref_bbox: Sequence[float] | None,
) -> np.ndarray:
    mode = mode.lower()
    if mode in {"existing", "none"}:
        return np.asarray([], dtype=np.int64)
    if mode == "global-median":
        return np.flatnonzero(np.isfinite(lon) & np.isfinite(lat)).astype(np.int64)
    if mode == "bbox":
        if ref_bbox is None or len(ref_bbox) != 4:
            raise AnnualGLSError("bbox reference requires --ref-bbox xmin ymin xmax ymax")
        xmin, ymin, xmax, ymax = map(float, ref_bbox)
        ix = np.flatnonzero(
            np.isfinite(lon) & np.isfinite(lat)
            & (lon >= xmin) & (lon <= xmax)
            & (lat >= ymin) & (lat <= ymax)
        )
        if ix.size == 0:
            raise AnnualGLSError("Reference bbox contains no PS")
        return ix.astype(np.int64)
    if mode == "point":
        if ref_lon is None or ref_lat is None:
            raise AnnualGLSError("point reference requires --ref-lon and --ref-lat")
        pyproj = _require_import("pyproj", "pyproj")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        rx, ry = transformer.transform(float(ref_lon), float(ref_lat))
        distance = np.hypot(np.asarray(x) - rx, np.asarray(y) - ry)
        ix = np.flatnonzero(np.isfinite(distance) & (distance <= float(ref_radius_m)))
        if ix.size == 0:
            ix = np.asarray([int(np.nanargmin(distance))], dtype=np.int64)
            warnings.warn("No PS inside reference radius; nearest PS used")
        return ix
    raise AnnualGLSError(f"Unsupported reference mode: {mode}")


def write_year_csv(path: Path, inputs: Inputs, year: int, result: dict[str, np.ndarray]) -> None:
    pandas = _require_import("pandas", "pandas")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame({
        "ps_id": np.arange(1, inputs.n_ps + 1, dtype=np.int64),
        "lon": inputs.lon,
        "lat": inputs.lat,
        "year": np.full(inputs.n_ps, year, dtype=np.int16),
        **result,
    })
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_h5(
    path: Path,
    inputs: Inputs,
    years: list[int],
    results: dict[int, dict[str, np.ndarray]],
    reference_offsets: dict[int, float],
    metadata: dict[str, Any],
) -> None:
    h5py = _require_import("h5py", "h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "pySTAMPS_SBAS_annual_velocity_robust_GLS"
        h5.attrs["los_sign"] = "positive_toward_satellite"
        h5.attrs["covariance_source"] = inputs.covariance_source
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
                h5.attrs[str(key)] = value
        h5.create_dataset("lon", data=inputs.lon, compression="gzip")
        h5.create_dataset("lat", data=inputs.lat, compression="gzip")
        h5.create_dataset("year", data=np.asarray(years, dtype=np.int16))
        h5.create_dataset(
            "reference_velocity_offset_mm_yr",
            data=np.asarray([reference_offsets[y] for y in years], dtype=np.float32),
        )
        keys = list(results[years[0]].keys())
        for key in keys:
            sample = results[years[0]][key]
            dtype = sample.dtype
            fillvalue: float | int = np.nan if np.issubdtype(dtype, np.floating) else 0
            ds = h5.create_dataset(
                key,
                shape=(inputs.n_ps, len(years)),
                dtype=dtype,
                chunks=(min(8192, inputs.n_ps), 1),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=fillvalue,
            )
            for col, year in enumerate(years):
                ds[:, col] = results[year][key]


def plot_map(path: Path, raster: np.ndarray, grid: Grid, year: int, dpi: int) -> None:
    matplotlib = _require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = raster[np.isfinite(raster)]
    vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        raster,
        extent=(grid.xmin, grid.xmax, grid.ymin, grid.ymax),
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(f"{year} LOS velocity — robust network GLS")
    ax.set_xlabel(f"Easting (m), EPSG:{grid.epsg}")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("mm/yr; positive toward satellite")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def write_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve() if args.out else dataset / "annual_velocity_gls"
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(dataset, repo_root, args.sigma_floor_deg)
    available_years = sorted({date.year for date in inputs.dates})
    years = [year for year in (args.years or available_years) if year in available_years]
    if not years:
        raise AnnualGLSError("No requested calendar year is available")

    template = Path(args.template_raster).expanduser().resolve() if args.template_raster else dataset / "postprocess" / "rasters" / "geo_velocity.tif"
    if not template.exists():
        template = None
    grid = build_grid(
        inputs.lon,
        inputs.lat,
        resolution_m=args.resolution_m,
        target_epsg=args.target_epsg,
        template_raster=template,
    )
    reference_ix = select_reference_indices(
        args.reference_mode,
        inputs.lon,
        inputs.lat,
        grid,
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        ref_radius_m=args.ref_radius_m,
        ref_bbox=args.ref_bbox,
    )

    results: dict[int, dict[str, np.ndarray]] = {}
    reference_offsets: dict[int, float] = {}
    year_metadata: dict[int, dict[str, Any]] = {}
    statistics: list[dict[str, Any]] = []

    for year in years:
        epoch_ix = np.asarray([i for i, date in enumerate(inputs.dates) if date.year == year], dtype=np.int64)
        calendar_span = (
            max(inputs.dates[i] for i in epoch_ix) - min(inputs.dates[i] for i in epoch_ix)
        ).total_seconds() / 86400.0 if epoch_ix.size else 0.0
        if epoch_ix.size < args.min_epochs or calendar_span < args.min_span_days:
            warnings.warn(
                f"{year} skipped: acquisitions={epoch_ix.size}, span={calendar_span:.1f} d; "
                f"requirements are {args.min_epochs} and {args.min_span_days:.1f} d"
            )
            continue

        print("\n============================================================")
        print(f"Robust network GLS annual velocity: {year}")
        print("============================================================")
        result, meta = compute_year(
            inputs,
            epoch_ix,
            min_epochs=args.min_epochs,
            min_span_days=args.min_span_days,
            chunk_ps=args.chunk_ps,
            covariance_mode=args.covariance_mode,
            eigen_floor_rel=args.eigen_floor_rel,
            robust=not args.disable_robust,
            huber_c=args.huber_c,
            weight_floor=args.weight_floor,
            max_iterations=args.irls_iterations,
            convergence=args.convergence,
        )

        offset = 0.0
        valid_ref = reference_ix[
            np.isfinite(result["velocity_mm_yr"][reference_ix])
        ] if reference_ix.size else np.asarray([], dtype=np.int64)
        if valid_ref.size:
            offset = float(np.median(result["velocity_mm_yr"][valid_ref]))
            gls_ref = reference_ix[np.isfinite(result["velocity_gls_mm_yr"][reference_ix])]
            ols_ref = reference_ix[np.isfinite(result["velocity_ols_mm_yr"][reference_ix])]
            offset_gls = float(np.median(result["velocity_gls_mm_yr"][gls_ref])) if gls_ref.size else offset
            offset_ols = float(np.median(result["velocity_ols_mm_yr"][ols_ref])) if ols_ref.size else offset
            result["velocity_mm_yr"] = (result["velocity_mm_yr"].astype(np.float64) - offset).astype(np.float32)
            result["ci95_low_mm_yr"] = (result["ci95_low_mm_yr"].astype(np.float64) - offset).astype(np.float32)
            result["ci95_high_mm_yr"] = (result["ci95_high_mm_yr"].astype(np.float64) - offset).astype(np.float32)
            result["velocity_gls_mm_yr"] = (result["velocity_gls_mm_yr"].astype(np.float64) - offset_gls).astype(np.float32)
            result["velocity_ols_mm_yr"] = (result["velocity_ols_mm_yr"].astype(np.float64) - offset_ols).astype(np.float32)
        reference_offsets[year] = offset
        results[year] = result
        year_metadata[year] = meta

        accepted = result["accepted"].astype(bool) & np.isfinite(result["velocity_mm_yr"])
        velocity = result["velocity_mm_yr"][accepted]
        std = result["velocity_std_mm_yr"][accepted]
        delta_ols = (
            result["velocity_mm_yr"] - result["velocity_ols_mm_yr"]
        )[accepted]

        rasters = {
            f"geo_velocity_{year}.tif": (result["velocity_mm_yr"], "Robust network GLS LOS velocity", "mm/yr", "average"),
            f"geo_velocity_std_{year}.tif": (result["velocity_std_mm_yr"], "Annual velocity standard error", "mm/yr", "average"),
            f"geo_velocity_gls_minus_ols_{year}.tif": (result["velocity_mm_yr"] - result["velocity_ols_mm_yr"], "Robust GLS minus OLS velocity", "mm/yr", "average"),
            f"geo_velocity_nobs_{year}.tif": (result["n_obs"].astype(np.float32), "Annual valid acquisition count", "count", "nearest"),
            f"geo_velocity_downweighted_modes_{year}.tif": (result["downweighted_mode_count"].astype(np.float32), "Huber downweighted whitened residual-mode count", "count", "nearest"),
        }
        velocity_raster = None
        for filename, (values, description, unit, overview_mode) in rasters.items():
            raster, _count = aggregate_mean(values, grid, args.min_points_per_cell)
            path = out_root / "rasters" / filename
            write_geotiff(path, raster, grid, description=description, unit=unit, year=year, resampling=overview_mode)
            if filename == f"geo_velocity_{year}.tif":
                velocity_raster = raster
            if args.wgs84_copy:
                reproject_wgs84(path, out_root / "rasters" / "wgs84" / filename.replace(".tif", "_wgs84.tif"))

        if velocity_raster is not None:
            plot_map(out_root / "plots" / f"velocity_{year}.png", velocity_raster, grid, year, args.plot_dpi)
        if args.write_point_csv:
            write_year_csv(out_root / "points" / f"ps_velocity_gls_{year}.csv", inputs, year, result)

        statistics.append({
            "year": year,
            "acquisition_count": int(epoch_ix.size),
            "period_start": meta["calendar_start"],
            "period_end": meta["calendar_end"],
            "period_span_days": meta["calendar_span_days"],
            "valid_ps": int(np.count_nonzero(accepted)),
            "valid_fraction": float(np.mean(accepted)),
            "reference_ps": int(valid_ref.size),
            "reference_offset_mm_yr": offset,
            "median_velocity_mm_yr": float(np.median(velocity)) if velocity.size else np.nan,
            "mean_velocity_mm_yr": float(np.mean(velocity)) if velocity.size else np.nan,
            "p02_velocity_mm_yr": float(np.percentile(velocity, 2)) if velocity.size else np.nan,
            "p98_velocity_mm_yr": float(np.percentile(velocity, 98)) if velocity.size else np.nan,
            "median_std_mm_yr": float(np.median(std)) if std.size else np.nan,
            "median_abs_gls_minus_ols_mm_yr": float(np.median(np.abs(delta_ols))) if delta_ols.size else np.nan,
            "ps_with_downweighted_modes": int(np.count_nonzero(result["downweighted_mode_count"] > 0)),
        })

    years_done = sorted(results)
    if not years_done:
        raise AnnualGLSError("No year met the annual coverage requirements")

    write_h5(
        out_root / "annual_velocity_gls.h5",
        inputs,
        years_done,
        results,
        reference_offsets,
        {
            "covariance_mode": args.covariance_mode,
            "robust": not args.disable_robust,
            "huber_c": args.huber_c,
            "weight_floor": args.weight_floor,
            "irls_iterations": args.irls_iterations,
            "min_epochs": args.min_epochs,
            "min_span_days": args.min_span_days,
            "wavelength_m": inputs.wavelength_m,
        },
    )
    write_statistics(out_root / "annual_velocity_gls_statistics.csv", statistics)

    report = {
        "status": "completed",
        "dataset": str(dataset),
        "output": str(out_root),
        "n_ps": inputs.n_ps,
        "n_epoch": inputs.n_epoch,
        "n_ifg": inputs.n_ifg,
        "years": years_done,
        "estimator": "SBAS network covariance + Huber robust GLS",
        "covariance_source": inputs.covariance_source,
        "covariance_mode": args.covariance_mode,
        "los_sign": "positive_toward_satellite",
        "min_epochs": args.min_epochs,
        "min_span_days": args.min_span_days,
        "robust": not args.disable_robust,
        "huber_c": args.huber_c,
        "weight_floor": args.weight_floor,
        "reference_mode": args.reference_mode,
        "reference_ps": int(reference_ix.size),
        "template_raster": str(template) if template else None,
        "grid_epsg": grid.epsg,
        "resolution_m": grid.resolution,
        "year_metadata": year_metadata,
        "statistics": statistics,
        "duration_sec": time.perf_counter() - started,
        "limitations": [
            "The propagated covariance assumes independent interferogram errors.",
            "Stage-8 filtering changes the exact residual covariance; robust IRLS on covariance-whitened residual modes and sandwich SE are used as protection against covariance mismatch.",
            "The annual model estimates a within-calendar-year linear component and does not explicitly separate a deterministic seasonal harmonic.",
        ],
    }
    (out_root / "annual_velocity_gls_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n============================================================")
    print("Annual robust GLS processing completed")
    print("============================================================")
    print(f"Years      : {years_done}")
    print(f"Output     : {out_root}")
    print(f"Covariance : {inputs.covariance_source}")
    print(f"Report     : {out_root / 'annual_velocity_gls_report.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute covariance-aware Huber robust annual LOS velocities for pySTAMPS SBAS"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-root", default="/home/ubuntu/software/pystamps-main")
    parser.add_argument("--out", default=None)
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-span-days", type=float, default=240.0)
    parser.add_argument("--chunk-ps", type=int, default=4096)
    parser.add_argument("--covariance-mode", choices=("network", "diagonal", "identity"), default="network")
    parser.add_argument("--eigen-floor-rel", type=float, default=1.0e-6)
    parser.add_argument("--sigma-floor-deg", type=float, default=0.1)
    parser.add_argument("--disable-robust", action="store_true")
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument("--weight-floor", type=float, default=0.05)
    parser.add_argument("--irls-iterations", type=int, default=8)
    parser.add_argument("--convergence", type=float, default=1.0e-6)
    parser.add_argument("--resolution-m", type=float, default=50.0)
    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument("--template-raster", default=None)
    parser.add_argument("--min-points-per-cell", type=int, default=1)
    parser.add_argument("--wgs84-copy", action="store_true")
    parser.add_argument("--write-point-csv", action="store_true")
    parser.add_argument("--reference-mode", choices=("existing", "none", "global-median", "point", "bbox"), default="existing")
    parser.add_argument("--ref-lon", type=float, default=None)
    parser.add_argument("--ref-lat", type=float, default=None)
    parser.add_argument("--ref-radius-m", type=float, default=500.0)
    parser.add_argument("--ref-bbox", type=float, nargs=4, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--plot-dpi", type=int, default=200)
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
