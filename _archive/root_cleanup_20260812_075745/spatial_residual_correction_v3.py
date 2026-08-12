#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative per-acquisition spatial residual correction for GACOS-corrected
pySTAMPS SBAS acquisition time series.

Method
------
1. Read Stage-8 acquisition phase (uw_space_time.mat / ph_uw_ts).
2. Fit a preliminary per-PS temporal model consisting of a long-term linear
   component plus annual and optional semiannual harmonics.
3. Temporally high-pass the residual with a leave-one-out Gaussian smoother.
4. For each acquisition, estimate only the long-spatial-wavelength component
   of the temporal high-pass residual using robust coarse-cell medians and a
   normalized Gaussian spatial low-pass filter.
5. Accept a correction only when spatial-cell holdout validation shows a
   reduction in robust residual scatter.
6. Subtract the accepted acquisition-specific spatial field, preserve the
   original temporal reference, write a new uw_space_time.mat, and create a
   shadow dataset for downstream annual-rate inversion.

This follows the conservative principle used by mature InSAR time-series
workflows: isolate temporally incoherent residuals and estimate only their
long-spatial-wavelength component. It deliberately avoids pixel-scale spatial
smoothing, raster averaging, and overwriting the original Stage-8 product.

Important limitation
--------------------
Any spatial atmospheric correction can attenuate real deformation that is both
rapid in time and broad in space. The code therefore uses temporal high-pass
residuals, kilometre-scale smoothing, holdout validation, amplitude caps, and
per-epoch accept/reject QA. The corrected result must still be compared with
uncorrected full-period and annual products.
"""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class SpatialResidualError(RuntimeError):
    pass


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise SpatialResidualError(
            f"缺少Python包：{name}\n"
            f"请在stamps环境安装：python -m pip install {pip_name or name}\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc


def load_support(repo_root: Path):
    path = repo_root / "calc_annual_velocity_gls.py"
    if not path.exists():
        raise SpatialResidualError(f"缺少GLS支撑模块：{path}")
    spec = importlib.util.spec_from_file_location(
        "pystamps_spatial_residual_support",
        path,
    )
    if spec is None or spec.loader is None:
        raise SpatialResidualError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if not math.isfinite(sigma) or sigma <= 1.0e-8:
        sigma = float(np.sqrt(np.mean((finite - median) ** 2)))
    return median, sigma


def adaptive_cap(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    multiplier: float,
) -> float:
    median, sigma = robust_location_scale(values)
    if not math.isfinite(median):
        return float(maximum)
    return float(max(minimum, min(maximum, median + multiplier * sigma)))


def build_preliminary_design(
    dates: list[datetime],
    harmonics: int,
    period_days: float,
) -> tuple[np.ndarray, list[str]]:
    if harmonics not in {0, 1, 2}:
        raise SpatialResidualError("preliminary_harmonics只能为0、1或2")
    origin = dates[0]
    t_days = np.asarray(
        [(date - origin).total_seconds() / 86400.0 for date in dates],
        np.float64,
    )
    t_year = t_days / 365.2425
    center = float(np.mean(t_year))
    columns = [
        np.ones(t_year.size, np.float64),
        t_year - center,
    ]
    names = ["intercept_mm", "linear_mm_yr"]
    if harmonics >= 1:
        omega = 2.0 * math.pi / float(period_days)
        columns.extend(
            [np.sin(omega * t_days), np.cos(omega * t_days)]
        )
        names.extend(["annual_sin_mm", "annual_cos_mm"])
    if harmonics >= 2:
        omega2 = 4.0 * math.pi / float(period_days)
        columns.extend(
            [np.sin(omega2 * t_days), np.cos(omega2 * t_days)]
        )
        names.extend(["semiannual_sin_mm", "semiannual_cos_mm"])
    X = np.column_stack(columns).astype(np.float64)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise SpatialResidualError("初始时间模型设计矩阵秩不足")
    return X, names


def robust_temporal_fit_batch(
    y: np.ndarray,
    X: np.ndarray,
    *,
    huber_c: float,
    weight_floor: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small-parameter robust weighted least squares for one PS chunk."""
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    valid = np.isfinite(y)
    n_row, n_time = y.shape
    p = X.shape[1]
    if n_time != X.shape[0]:
        raise SpatialResidualError("时间模型y/X维度不一致")

    weights = valid.astype(np.float64)
    y0 = np.where(valid, y, 0.0)
    beta = np.full((n_row, p), np.nan, np.float64)
    fit_valid = np.sum(valid, axis=1) >= p + 4

    for iteration in range(max(1, int(iterations))):
        normal = np.einsum(
            "nt,ti,tj->nij",
            weights,
            X,
            X,
            optimize=True,
        )
        rhs = np.einsum(
            "nt,ti,nt->ni",
            weights,
            X,
            y0,
            optimize=True,
        )
        trace = np.trace(normal, axis1=1, axis2=2)
        ridge = np.maximum(trace * 1.0e-11 / max(p, 1), 1.0e-12)
        normal[:, np.arange(p), np.arange(p)] += ridge[:, None]

        try:
            candidate = np.linalg.solve(
                normal,
                rhs[..., None],
            ).squeeze(-1)
        except np.linalg.LinAlgError:
            candidate = np.full_like(beta, np.nan)
            for row in range(n_row):
                if not fit_valid[row]:
                    continue
                try:
                    candidate[row] = np.linalg.solve(normal[row], rhs[row])
                except np.linalg.LinAlgError:
                    continue
        beta = candidate
        prediction = beta @ X.T
        residual = np.where(valid, y - prediction, np.nan)
        center = np.nanmedian(residual, axis=1)
        mad = np.nanmedian(
            np.abs(residual - center[:, None]),
            axis=1,
        )
        scale = 1.4826 * mad
        rms = np.sqrt(np.nanmean(residual * residual, axis=1))
        scale = np.where(
            np.isfinite(scale) & (scale > 1.0e-6),
            scale,
            rms,
        )
        scale = np.maximum(scale, 1.0e-6)
        standardized = np.abs(residual) / (
            float(huber_c) * scale[:, None]
        )
        new_weights = np.ones_like(standardized)
        large = standardized > 1.0
        new_weights[large] = 1.0 / standardized[large]
        new_weights = np.clip(new_weights, float(weight_floor), 1.0)
        new_weights[~valid] = 0.0
        weights = new_weights

    prediction = beta @ X.T
    residual = np.where(valid, y - prediction, np.nan)
    rmse = np.sqrt(np.nanmean(residual * residual, axis=1))
    fit_valid &= np.all(np.isfinite(beta), axis=1) & np.isfinite(rmse)
    beta[~fit_valid, :] = np.nan
    rmse[~fit_valid] = np.nan
    return beta, rmse, fit_valid


def temporal_neighbor_weights(
    dates: list[datetime],
    sigma_days: float,
    truncate_sigma: float,
    min_neighbors: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    t = np.asarray(
        [(date - dates[0]).total_seconds() / 86400.0 for date in dates],
        np.float64,
    )
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(t.size):
        delta = np.abs(t - t[index])
        mask = (
            (delta > 0)
            & (delta <= float(truncate_sigma) * float(sigma_days))
        )
        candidates = np.flatnonzero(mask)
        if candidates.size < min_neighbors:
            candidates = np.argsort(delta)[1 : min_neighbors + 1]
        weights = np.exp(
            -0.5 * (delta[candidates] / float(sigma_days)) ** 2
        )
        weights /= np.sum(weights)
        output.append((candidates.astype(np.int64), weights.astype(np.float64)))
    return output


def projected_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    pyproj = require_import("pyproj", "pyproj")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{int(epsg)}",
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, np.float64), np.asarray(y, np.float64)


def auto_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = max(1, min(60, int(math.floor((lon0 + 180.0) / 6.0)) + 1))
    return (32600 if lat0 >= 0 else 32700) + zone


def build_spatial_grid(
    x: np.ndarray,
    y: np.ndarray,
    spacing_m: float,
) -> dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(y)
    spacing = float(spacing_m)
    xmin = math.floor(float(np.min(x[finite])) / spacing) * spacing
    ymin = math.floor(float(np.min(y[finite])) / spacing) * spacing
    xmax = math.ceil(float(np.max(x[finite])) / spacing) * spacing
    ymax = math.ceil(float(np.max(y[finite])) / spacing) * spacing
    nx = max(2, int(math.ceil((xmax - xmin) / spacing)))
    ny = max(2, int(math.ceil((ymax - ymin) / spacing)))
    col_float = (x - xmin) / spacing - 0.5
    row_float = (y - ymin) / spacing - 0.5
    col = np.floor((x - xmin) / spacing).astype(np.int64)
    row = np.floor((y - ymin) / spacing).astype(np.int64)
    inside = (
        finite
        & (col >= 0)
        & (col < nx)
        & (row >= 0)
        & (row < ny)
    )
    cell_id = np.full(x.size, -1, np.int64)
    cell_id[inside] = row[inside] * nx + col[inside]
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "spacing_m": spacing,
        "nx": nx,
        "ny": ny,
        "row": row,
        "col": col,
        "row_float": row_float,
        "col_float": col_float,
        "inside": inside,
        "cell_id": cell_id,
    }


def cell_medians(
    values: np.ndarray,
    point_indices: np.ndarray,
    cell_ids: np.ndarray,
    n_cells: int,
    min_cell_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust cell medians for a static point layout."""
    point_values = np.asarray(values[point_indices], np.float64)
    ids = np.asarray(cell_ids[point_indices], np.int64)
    valid = np.isfinite(point_values) & (ids >= 0)
    if not np.any(valid):
        return (
            np.full(n_cells, np.nan, np.float64),
            np.zeros(n_cells, np.uint32),
        )
    point_values = point_values[valid]
    ids = ids[valid]
    order = np.argsort(ids, kind="mergesort")
    ids = ids[order]
    point_values = point_values[order]
    unique, starts, counts = np.unique(
        ids,
        return_index=True,
        return_counts=True,
    )
    medians = np.full(n_cells, np.nan, np.float64)
    cell_count = np.zeros(n_cells, np.uint32)
    for cell, start, count in zip(unique, starts, counts, strict=False):
        cell_count[int(cell)] = int(count)
        if count >= int(min_cell_points):
            medians[int(cell)] = float(
                np.median(point_values[start : start + count])
            )
    return medians, cell_count


def smooth_cell_grid(
    cell_values: np.ndarray,
    ny: int,
    nx: int,
    sigma_cells: float,
    max_extrapolation_cells: float,
) -> np.ndarray:
    scipy_ndimage = require_import("scipy.ndimage", "scipy")
    grid = np.asarray(cell_values, np.float64).reshape(ny, nx)
    valid = np.isfinite(grid)
    if np.count_nonzero(valid) < 4:
        return np.full((ny, nx), np.nan, np.float64)

    # Nearest-cell initialization followed by a long-wavelength Gaussian
    # filter. Distances beyond the permitted extrapolation are masked.
    distance, indices = scipy_ndimage.distance_transform_edt(
        ~valid,
        return_indices=True,
    )
    filled = grid[tuple(indices)]
    smoothed = scipy_ndimage.gaussian_filter(
        filled,
        sigma=float(sigma_cells),
        mode="nearest",
        truncate=3.0,
    )
    smoothed[distance > float(max_extrapolation_cells)] = np.nan
    return smoothed


def interpolate_grid_to_points(
    grid_values: np.ndarray,
    row_float: np.ndarray,
    col_float: np.ndarray,
) -> np.ndarray:
    scipy_ndimage = require_import("scipy.ndimage", "scipy")
    values = np.asarray(grid_values, np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(row_float.size, np.nan, np.float64)
    # Fill any residual holes before bilinear interpolation.
    if not np.all(finite):
        _distance, indices = scipy_ndimage.distance_transform_edt(
            ~finite,
            return_indices=True,
        )
        values = values[tuple(indices)]
    coordinates = np.vstack((row_float, col_float))
    return scipy_ndimage.map_coordinates(
        values,
        coordinates,
        order=1,
        mode="nearest",
    )


def estimate_epoch_spatial_field(
    highpass_mm: np.ndarray,
    control_mask: np.ndarray,
    grid: dict[str, Any],
    *,
    min_cell_points: int,
    spatial_sigma_cells: float,
    max_extrapolation_cells: float,
    residual_clip_sigma: float,
    cv_modulo: int,
    cv_offset: int,
    min_cv_cells: int,
    min_cv_improvement: float,
    max_correction_mm: float,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(highpass_mm, np.float64)
    base = (
        np.asarray(control_mask, bool)
        & grid["inside"]
        & np.isfinite(values)
    )
    median, sigma = robust_location_scale(values[base])
    if not math.isfinite(sigma) or sigma <= 0:
        return np.zeros(values.size, np.float32), {
            "accepted": False,
            "reason": "invalid_control_scale",
        }
    clipped = base & (
        np.abs(values - median) <= float(residual_clip_sigma) * sigma
    )
    control_indices = np.flatnonzero(clipped)
    n_cells = int(grid["nx"] * grid["ny"])
    medians, counts = cell_medians(
        values,
        control_indices,
        grid["cell_id"],
        n_cells,
        min_cell_points,
    )
    valid_cells = np.flatnonzero(np.isfinite(medians))
    if valid_cells.size < max(10, min_cv_cells * 2):
        return np.zeros(values.size, np.float32), {
            "accepted": False,
            "reason": "too_few_valid_cells",
            "control_points": int(control_indices.size),
            "valid_cells": int(valid_cells.size),
        }

    holdout = valid_cells[
        (valid_cells + int(cv_offset)) % int(cv_modulo) == 0
    ]
    train = np.setdiff1d(valid_cells, holdout, assume_unique=True)
    if holdout.size < int(min_cv_cells) or train.size < int(min_cv_cells):
        return np.zeros(values.size, np.float32), {
            "accepted": False,
            "reason": "too_few_cv_cells",
            "control_points": int(control_indices.size),
            "valid_cells": int(valid_cells.size),
            "holdout_cells": int(holdout.size),
        }

    train_values = np.full(n_cells, np.nan, np.float64)
    train_values[train] = medians[train]
    cv_grid = smooth_cell_grid(
        train_values,
        int(grid["ny"]),
        int(grid["nx"]),
        spatial_sigma_cells,
        max_extrapolation_cells,
    ).reshape(-1)
    before_values = medians[holdout]
    predicted_values = cv_grid[holdout]
    cv_valid = np.isfinite(before_values) & np.isfinite(predicted_values)
    if np.count_nonzero(cv_valid) < int(min_cv_cells):
        return np.zeros(values.size, np.float32), {
            "accepted": False,
            "reason": "insufficient_cv_predictions",
            "holdout_cells": int(holdout.size),
        }
    _loc_before, scatter_before = robust_location_scale(
        before_values[cv_valid]
    )
    _loc_after, scatter_after = robust_location_scale(
        before_values[cv_valid] - predicted_values[cv_valid]
    )
    improvement = (
        (scatter_before - scatter_after) / scatter_before
        if scatter_before > 0
        else float("nan")
    )
    accepted = (
        math.isfinite(improvement)
        and improvement >= float(min_cv_improvement)
        and scatter_after < scatter_before
    )
    if not accepted:
        return np.zeros(values.size, np.float32), {
            "accepted": False,
            "reason": "cv_improvement_below_threshold",
            "control_points": int(control_indices.size),
            "valid_cells": int(valid_cells.size),
            "holdout_cells": int(np.count_nonzero(cv_valid)),
            "cv_scatter_before_mm": float(scatter_before),
            "cv_scatter_after_mm": float(scatter_after),
            "cv_improvement_fraction": float(improvement),
        }

    final_grid = smooth_cell_grid(
        medians,
        int(grid["ny"]),
        int(grid["nx"]),
        spatial_sigma_cells,
        max_extrapolation_cells,
    )
    correction = interpolate_grid_to_points(
        final_grid,
        grid["row_float"],
        grid["col_float"],
    )
    finite_ref = (
        np.asarray(reference_mask, bool)
        & np.isfinite(correction)
    )
    if not np.any(finite_ref):
        finite_ref = clipped & np.isfinite(correction)
    reference_value = (
        float(np.median(correction[finite_ref]))
        if np.any(finite_ref)
        else 0.0
    )
    correction -= reference_value
    correction = np.clip(
        correction,
        -float(max_correction_mm),
        float(max_correction_mm),
    )
    correction[~grid["inside"]] = 0.0

    finite_correction = correction[np.isfinite(correction)]
    return correction.astype(np.float32), {
        "accepted": True,
        "reason": "accepted",
        "control_points": int(control_indices.size),
        "valid_cells": int(valid_cells.size),
        "holdout_cells": int(np.count_nonzero(cv_valid)),
        "cv_scatter_before_mm": float(scatter_before),
        "cv_scatter_after_mm": float(scatter_after),
        "cv_improvement_fraction": float(improvement),
        "correction_p02_mm": float(np.percentile(finite_correction, 2)),
        "correction_median_mm": float(np.median(finite_correction)),
        "correction_p98_mm": float(np.percentile(finite_correction, 98)),
        "correction_max_abs_mm": float(
            np.max(np.abs(finite_correction))
        ),
        "reference_value_removed_mm": reference_value,
    }


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def processing_signature(
    dataset: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    files = []
    for name in (
        "ps2.mat",
        "uw_space_time.mat",
        "parms.mat",
        "scla2.mat",
        "ifgstd2.mat",
        "gacos_correction_debug.json",
        "stage7_sbas_debug.json",
        "stage8_sbas_debug.json",
    ):
        path = dataset / name
        if path.exists():
            files.append(file_signature(path))
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"overwrite", "resume"}
    }
    payload = {
        "version": 3,
        "files": files,
        "parameters": parameters,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def read_progress(path: Path, signature: str) -> dict[str, Any]:
    if not path.exists():
        return {"signature": signature, "prelim_rows": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            return {"signature": signature, "prelim_rows": 0}
        return payload
    except Exception:
        return {"signature": signature, "prelim_rows": 0}


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_corrected_mat(
    path: Path,
    inputs: Any,
    aps_mm: np.memmap,
    *,
    phase_to_mm: float,
    ifgday_ix: np.ndarray,
    chunk_ps: int,
    signature: str,
) -> None:
    h5py = require_import("h5py", "h5py")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    reference_ix = int(inputs.reference_image_ix_1based) - 1
    with h5py.File(temporary, "w") as h5:
        phase_dataset = h5.create_dataset(
            "ph_uw_ts",
            shape=(inputs.n_ps, inputs.n_epoch),
            dtype=np.float32,
            chunks=(min(chunk_ps, inputs.n_ps), min(32, inputs.n_epoch)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        phase_dataset.attrs["PY_STAMPS_row_major"] = np.asarray(
            1, dtype=np.uint8
        )
        for start in range(0, inputs.n_ps, chunk_ps):
            stop = min(start + chunk_ps, inputs.n_ps)
            raw_phase = np.asarray(
                inputs.ph_ts[start:stop, :],
                np.float64,
            )
            raw_mm = raw_phase * float(phase_to_mm)
            corrected_mm = raw_mm - np.asarray(
                aps_mm[start:stop, :],
                np.float64,
            )
            if 0 <= reference_ix < inputs.n_epoch:
                corrected_mm -= corrected_mm[:, reference_ix][:, None]
            corrected_phase = corrected_mm / float(phase_to_mm)
            phase_dataset[start:stop, :] = corrected_phase.astype(np.float32)
            print(
                f"[SPATIAL_V3][WRITE_MAT] {stop}/{inputs.n_ps} "
                f"({100.0 * stop / inputs.n_ps:.1f}%)",
                flush=True,
            )

        for name, data in (
            ("day", np.asarray(inputs.day, np.float64).reshape(-1, 1)),
            ("ifgday_ix", np.asarray(ifgday_ix)),
            (
                "reference_image_ix",
                np.asarray(
                    [[inputs.reference_image_ix_1based]],
                    np.int32,
                ),
            ),
        ):
            dataset = h5.create_dataset(name, data=data)
            dataset.attrs["PY_STAMPS_row_major"] = np.asarray(
                1, dtype=np.uint8
            )
        h5.attrs["spatial_residual_corrected"] = np.asarray(
            1, dtype=np.uint8
        )
        h5.attrs["processing_signature"] = signature
        h5.attrs["los_sign"] = "positive_toward_satellite"
    os.replace(temporary, path)


def create_shadow_dataset(
    source: Path,
    shadow: Path,
    corrected_mat: Path,
    debug_path: Path,
) -> None:
    shadow.mkdir(parents=True, exist_ok=True)
    names = (
        "ps2.mat",
        "parms.mat",
        "scla2.mat",
        "ifgstd2.mat",
        "gacos_correction_debug.json",
        "stage7_sbas_debug.json",
        "stage8_sbas_debug.json",
    )
    for name in names:
        source_path = source / name
        if not source_path.exists():
            continue
        target = shadow / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source_path)
    patch = source / "PATCH_1"
    if patch.exists():
        target_patch = shadow / "PATCH_1"
        if target_patch.exists() or target_patch.is_symlink():
            if target_patch.is_symlink() or target_patch.is_file():
                target_patch.unlink()
            else:
                shutil.rmtree(target_patch)
        target_patch.symlink_to(patch, target_is_directory=True)

    target_uw = shadow / "uw_space_time.mat"
    if target_uw.exists() or target_uw.is_symlink():
        target_uw.unlink()
    target_uw.symlink_to(corrected_mat)
    shutil.copy2(debug_path, shadow / "spatial_residual_correction_debug.json")


def write_qa_plot(path: Path, qa_rows: list[dict[str, Any]]) -> None:
    matplotlib = require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [datetime.fromisoformat(row["date"]) for row in qa_rows]
    improvements = np.asarray(
        [row.get("cv_improvement_fraction", np.nan) for row in qa_rows],
        np.float64,
    )
    p98 = np.asarray(
        [row.get("correction_p98_mm", np.nan) for row in qa_rows],
        np.float64,
    )
    p02 = np.asarray(
        [row.get("correction_p02_mm", np.nan) for row in qa_rows],
        np.float64,
    )
    accepted = np.asarray(
        [bool(row.get("accepted", False)) for row in qa_rows]
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, 100.0 * improvements, marker=".", linewidth=0.8)
    ax.axhline(0.0, linewidth=0.8)
    ax.scatter(
        np.asarray(dates, dtype=object)[accepted],
        100.0 * improvements[accepted],
        s=12,
        label="accepted",
    )
    ax.set_ylabel("Holdout improvement (%)")
    ax.set_title("Per-acquisition spatial residual correction QA")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, p02, linewidth=0.8, label="correction P02")
    ax.plot(dates, p98, linewidth=0.8, label="correction P98")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("LOS correction (mm)")
    ax.set_title("Accepted spatial correction amplitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path.with_name("02_spatial_correction_amplitude.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = Path(
        args.out or dataset / "spatial_residual_correction_v3"
    ).expanduser().resolve()
    shadow_dataset = Path(
        args.shadow_dataset or dataset / "_spatial_corrected_dataset_v3"
    ).expanduser().resolve()
    work = output_root / "_work"

    if args.overwrite:
        for path in (output_root, shadow_dataset):
            if path.exists():
                shutil.rmtree(path)
    output_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    support = load_support(repo_root)
    inputs = support.load_inputs(dataset, repo_root, args.sigma_floor_deg)
    signature_payload, signature = processing_signature(dataset, args)
    progress_path = work / "progress.json"
    progress = read_progress(progress_path, signature)

    phase_to_mm = (
        -inputs.wavelength_m / (4.0 * math.pi) * 1000.0
    )
    X, parameter_names = build_preliminary_design(
        inputs.dates,
        args.preliminary_harmonics,
        args.seasonal_period_days,
    )

    residual_path = work / "preliminary_residual_mm.f32"
    aps_path = work / "aps_at_ps_mm.f32"
    beta_path = work / "preliminary_beta.f32"
    rmse_path = work / "preliminary_rmse_mm.f32"
    valid_path = work / "preliminary_fit_valid.u1"
    expected_sizes = {
        residual_path: inputs.n_ps * inputs.n_epoch * 4,
        beta_path: inputs.n_ps * X.shape[1] * 4,
        rmse_path: inputs.n_ps * 4,
        valid_path: inputs.n_ps,
    }
    prelim_files_valid = all(
        path.exists() and path.stat().st_size == expected
        for path, expected in expected_sizes.items()
    )
    aps_file_valid = (
        aps_path.exists()
        and aps_path.stat().st_size == inputs.n_ps * inputs.n_epoch * 4
    )
    if not prelim_files_valid:
        progress["prelim_rows"] = 0
        progress.pop("epoch_signature", None)
    residual = np.memmap(
        residual_path,
        dtype=np.float32,
        mode="r+" if residual_path.exists() else "w+",
        shape=(inputs.n_ps, inputs.n_epoch),
    )
    aps = np.memmap(
        aps_path,
        dtype=np.float32,
        mode="r+" if aps_path.exists() else "w+",
        shape=(inputs.n_ps, inputs.n_epoch),
    )
    prelim_beta = np.memmap(
        beta_path,
        dtype=np.float32,
        mode="r+" if beta_path.exists() else "w+",
        shape=(inputs.n_ps, X.shape[1]),
    )
    prelim_rmse = np.memmap(
        rmse_path,
        dtype=np.float32,
        mode="r+" if rmse_path.exists() else "w+",
        shape=(inputs.n_ps,),
    )
    prelim_valid_u1 = np.memmap(
        valid_path,
        dtype=np.uint8,
        mode="r+" if valid_path.exists() else "w+",
        shape=(inputs.n_ps,),
    )
    prelim_valid = np.asarray(prelim_valid_u1, dtype=np.uint8).astype(bool)

    if progress.get("prelim_rows", 0) < inputs.n_ps:
        start_row = int(progress.get("prelim_rows", 0)) if args.resume else 0
        if start_row == 0:
            prelim_beta[:] = np.nan
            prelim_rmse[:] = np.nan
            prelim_valid_u1[:] = 0
            prelim_beta.flush()
            prelim_rmse.flush()
            prelim_valid_u1.flush()
        for start in range(start_row, inputs.n_ps, args.chunk_ps):
            stop = min(start + args.chunk_ps, inputs.n_ps)
            phase = np.asarray(inputs.ph_ts[start:stop, :], np.float64)
            displacement = phase * phase_to_mm
            beta, rmse, valid = robust_temporal_fit_batch(
                displacement,
                X,
                huber_c=args.huber_c,
                weight_floor=args.weight_floor,
                iterations=args.preliminary_irls_iterations,
            )
            prediction = beta @ X.T
            residual[start:stop, :] = (
                displacement - prediction
            ).astype(np.float32)
            prelim_beta[start:stop, :] = beta.astype(np.float32)
            prelim_rmse[start:stop] = rmse.astype(np.float32)
            prelim_valid_u1[start:stop] = valid.astype(np.uint8)
            residual.flush()
            prelim_beta.flush()
            prelim_rmse.flush()
            prelim_valid_u1.flush()
            progress.update(
                {
                    "signature": signature,
                    "prelim_rows": stop,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_progress(progress_path, progress)
            print(
                f"[SPATIAL_V3][PRELIM] {stop}/{inputs.n_ps} "
                f"({100.0 * stop / inputs.n_ps:.1f}%)",
                flush=True,
            )

    prelim_valid = np.asarray(prelim_valid_u1, dtype=np.uint8).astype(bool)
    metrics_path = work / "preliminary_fit_metrics.npz"
    np.savez_compressed(
        metrics_path,
        parameter_names=np.asarray(parameter_names),
        rmse_cap_pending=np.asarray(1, np.uint8),
    )

    rmse_cap = adaptive_cap(
        prelim_rmse[prelim_valid],
        minimum=args.min_preliminary_rmse_cap_mm,
        maximum=args.max_preliminary_rmse_mm,
        multiplier=args.threshold_mad_multiplier,
    )
    linear_rate = prelim_beta[:, 1]
    control_mask = (
        prelim_valid
        & np.isfinite(prelim_rmse)
        & (prelim_rmse <= rmse_cap)
        & np.isfinite(linear_rate)
        & (np.abs(linear_rate) <= args.max_abs_preliminary_rate_mm_yr)
    )

    target_epsg = int(
        args.target_epsg or auto_utm_epsg(inputs.lon, inputs.lat)
    )
    x, y = projected_xy(inputs.lon, inputs.lat, target_epsg)
    grid = build_spatial_grid(x, y, args.spatial_grid_m)

    reference_mask = control_mask.copy()
    if args.reference_lon is not None and args.reference_lat is not None:
        ref_x, ref_y = projected_xy(
            np.asarray([args.reference_lon]),
            np.asarray([args.reference_lat]),
            target_epsg,
        )
        reference_mask = control_mask & (
            (x - ref_x[0]) ** 2 + (y - ref_y[0]) ** 2
            <= float(args.reference_radius_m) ** 2
        )
        if np.count_nonzero(reference_mask) < args.min_reference_ps:
            warnings.warn(
                "指定参考区PS不足，退回全部控制点中位数参考"
            )
            reference_mask = control_mask.copy()

    temporal_weights = temporal_neighbor_weights(
        inputs.dates,
        args.temporal_sigma_days,
        args.temporal_truncate_sigma,
        args.temporal_min_neighbors,
    )

    epoch_done_path = work / "epoch_done.npy"
    qa_json_path = work / "epoch_qa.json"
    if (
        epoch_done_path.exists()
        and aps_file_valid
        and args.resume
        and progress.get("epoch_signature") == signature
    ):
        epoch_done = np.load(epoch_done_path).astype(bool)
    else:
        epoch_done = np.zeros(inputs.n_epoch, bool)
        aps[:] = 0.0
        aps.flush()
    if qa_json_path.exists() and args.resume:
        try:
            qa_map = json.loads(qa_json_path.read_text(encoding="utf-8"))
        except Exception:
            qa_map = {}
    else:
        qa_map = {}

    reference_epoch = int(inputs.reference_image_ix_1based) - 1
    for epoch in range(inputs.n_epoch):
        if epoch_done[epoch]:
            print(
                f"[SPATIAL_V3][RESUME] {epoch + 1}/{inputs.n_epoch} "
                f"{inputs.labels[epoch]}",
                flush=True,
            )
            continue
        if epoch == reference_epoch:
            aps[:, epoch] = 0.0
            qa = {
                "accepted": False,
                "reason": "reference_epoch_forced_zero",
            }
        else:
            neighbor_ix, neighbor_w = temporal_weights[epoch]
            smooth = np.asarray(
                residual[:, neighbor_ix],
                np.float64,
            ) @ neighbor_w
            highpass = (
                np.asarray(residual[:, epoch], np.float64) - smooth
            )
            correction, qa = estimate_epoch_spatial_field(
                highpass,
                control_mask,
                grid,
                min_cell_points=args.min_cell_points,
                spatial_sigma_cells=args.spatial_sigma_cells,
                max_extrapolation_cells=args.max_extrapolation_cells,
                residual_clip_sigma=args.residual_clip_sigma,
                cv_modulo=args.cv_modulo,
                cv_offset=epoch % args.cv_modulo,
                min_cv_cells=args.min_cv_cells,
                min_cv_improvement=args.min_cv_improvement,
                max_correction_mm=args.max_correction_mm,
                reference_mask=reference_mask,
            )
            aps[:, epoch] = correction
        aps.flush()
        qa.update(
            {
                "epoch_index": epoch,
                "date": inputs.dates[epoch].date().isoformat(),
                "label": inputs.labels[epoch],
            }
        )
        qa_map[str(epoch)] = qa
        epoch_done[epoch] = True
        np.save(epoch_done_path, epoch_done.astype(np.uint8))
        progress["epoch_signature"] = signature
        progress["completed_epochs"] = int(np.count_nonzero(epoch_done))
        write_progress(progress_path, progress)
        qa_json_path.write_text(
            json.dumps(qa_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[SPATIAL_V3][EPOCH] {epoch + 1}/{inputs.n_epoch} "
            f"{inputs.labels[epoch]} accepted={qa.get('accepted')} "
            f"improvement={qa.get('cv_improvement_fraction')}",
            flush=True,
        )

    qa_rows = [qa_map[str(index)] for index in range(inputs.n_epoch)]
    qa_csv = output_root / "epoch_spatial_correction_qa.csv"
    fieldnames = sorted({key for row in qa_rows for key in row})
    with qa_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_rows)

    accepted_count = sum(bool(row.get("accepted", False)) for row in qa_rows)
    accepted_fraction = accepted_count / float(inputs.n_epoch)
    write_qa_plot(output_root / "plots" / "01_holdout_improvement.png", qa_rows)

    # Read network indices from the original Stage-8 file, falling back to ps2.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pystamps.io.mat import read_mat_variables

    uw_meta = read_mat_variables(
        dataset / "uw_space_time.mat",
        ("ifgday_ix",),
    )
    ifgday_ix = uw_meta.get("ifgday_ix")
    if ifgday_ix is None or np.asarray(ifgday_ix).size == 0:
        ps_meta = read_mat_variables(dataset / "ps2.mat", ("ifgday_ix",))
        ifgday_ix = ps_meta.get("ifgday_ix")
    if ifgday_ix is None:
        raise SpatialResidualError("无法读取ifgday_ix")

    corrected_mat = output_root / "uw_space_time_spatial_corrected.mat"
    existing_debug_path = output_root / "spatial_residual_correction_debug.json"
    reuse_corrected_mat = False
    if corrected_mat.exists() and existing_debug_path.exists() and not args.overwrite:
        try:
            existing_debug = json.loads(
                existing_debug_path.read_text(encoding="utf-8")
            )
            reuse_corrected_mat = (
                existing_debug.get("signature") == signature
                and bool(np.all(epoch_done))
            )
        except Exception:
            reuse_corrected_mat = False
    if reuse_corrected_mat:
        print(
            f"[SPATIAL_V3][RESUME] reuse corrected MAT: {corrected_mat}",
            flush=True,
        )
    else:
        write_corrected_mat(
            corrected_mat,
            inputs,
            aps,
            phase_to_mm=phase_to_mm,
            ifgday_ix=np.asarray(ifgday_ix),
            chunk_ps=args.chunk_ps,
            signature=signature,
        )

    debug = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "output": str(output_root),
        "shadow_dataset": str(shadow_dataset),
        "source_time_series": str(dataset / "uw_space_time.mat"),
        "corrected_time_series": str(corrected_mat),
        "method": (
            "per-PS linear+seasonal preliminary model; leave-one-out temporal "
            "Gaussian high-pass; robust coarse-cell medians; Gaussian spatial "
            "low-pass; spatial-cell holdout acceptance"
        ),
        "n_ps": inputs.n_ps,
        "n_epoch": inputs.n_epoch,
        "target_epsg": target_epsg,
        "preliminary_parameter_names": parameter_names,
        "preliminary_rmse_cap_mm": rmse_cap,
        "control_ps": int(np.count_nonzero(control_mask)),
        "reference_ps": int(np.count_nonzero(reference_mask)),
        "accepted_epochs": accepted_count,
        "accepted_epoch_fraction": accepted_fraction,
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"overwrite", "resume"}
        },
        "signature": signature,
        "signature_payload": signature_payload,
        "epoch_qa_csv": str(qa_csv),
        "limitations": [
            (
                "Broad and rapid real deformation may be attenuated; compare "
                "corrected and uncorrected annual products."
            ),
            (
                "Only epochs that improve held-out spatial-cell residual "
                "scatter are corrected."
            ),
            (
                "The original Stage-8 time series is not overwritten."
            ),
        ],
        "duration_sec": time.perf_counter() - started,
    }
    debug_path = output_root / "spatial_residual_correction_debug.json"
    debug_path.write_text(
        json.dumps(debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    create_shadow_dataset(
        dataset,
        shadow_dataset,
        corrected_mat,
        debug_path,
    )

    print("\n============================================================")
    print("Spatial residual correction V3 completed")
    print("============================================================")
    print(f"Control PS       : {np.count_nonzero(control_mask):,}")
    print(f"Accepted epochs  : {accepted_count}/{inputs.n_epoch}")
    print(f"Corrected MAT    : {corrected_mat}")
    print(f"Shadow dataset   : {shadow_dataset}")
    print(f"QA CSV           : {qa_csv}")
    print(f"Debug            : {debug_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative per-acquisition spatial residual correction for "
            "GACOS-corrected pySTAMPS SBAS time series"
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--repo-root",
        default="/home/ubuntu/software/pystamps-main",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--shadow-dataset", default=None)
    parser.add_argument("--chunk-ps", type=int, default=4096)
    parser.add_argument("--sigma-floor-deg", type=float, default=0.1)

    parser.add_argument(
        "--preliminary-harmonics",
        type=int,
        choices=(0, 1, 2),
        default=2,
    )
    parser.add_argument(
        "--seasonal-period-days",
        type=float,
        default=365.2425,
    )
    parser.add_argument(
        "--preliminary-irls-iterations",
        type=int,
        default=4,
    )
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument("--weight-floor", type=float, default=0.05)
    parser.add_argument(
        "--min-preliminary-rmse-cap-mm",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--max-preliminary-rmse-mm",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--max-abs-preliminary-rate-mm-yr",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--threshold-mad-multiplier",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--temporal-sigma-days",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--temporal-truncate-sigma",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--temporal-min-neighbors",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--spatial-grid-m",
        type=float,
        default=4000.0,
    )
    parser.add_argument(
        "--spatial-sigma-cells",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--max-extrapolation-cells",
        type=float,
        default=4.0,
    )
    parser.add_argument("--min-cell-points", type=int, default=30)
    parser.add_argument(
        "--residual-clip-sigma",
        type=float,
        default=4.0,
    )
    parser.add_argument("--cv-modulo", type=int, default=5)
    parser.add_argument("--min-cv-cells", type=int, default=20)
    parser.add_argument(
        "--min-cv-improvement",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-correction-mm",
        type=float,
        default=30.0,
    )

    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument("--reference-lon", type=float, default=None)
    parser.add_argument("--reference-lat", type=float, default=None)
    parser.add_argument(
        "--reference-radius-m",
        type=float,
        default=1000.0,
    )
    parser.add_argument("--min-reference-ps", type=int, default=20)

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("用户中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
