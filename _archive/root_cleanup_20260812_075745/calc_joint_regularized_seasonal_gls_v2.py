#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regularized annual LOS-rate inversion for GACOS-corrected pySTAMPS SBAS output.

Primary model per PS
--------------------
    phase(t) = intercept
             + sum_y slope_y * exposure_y(t)
             + annual_sin * sin(2*pi*t/T)
             + annual_cos * cos(2*pi*t/T)
             [+ semiannual terms]
             + error

The annual slope sequence is regularized with a second-difference penalty:

    ||D2 * velocity_year||^2 / sigma_curvature^2

where velocity_year is in mm/yr. The curvature scale can be supplied directly
or selected automatically by generalized cross validation (GCV) on a
deterministic sample of PS points.

Estimator
---------
  * Stage-8 acquisition phase from uw_space_time.mat
  * SBAS acquisition covariance from scla2.mat/ifg_vcm or network fallback
  * covariance whitening
  * penalized Huber IRLS generalized least squares
  * sandwich-style conditional covariance
  * independent de-seasonalized per-year robust GLS validation

Primary GIS output
------------------
  * one final original-PS Shapefile per formal calendar year
  * one diagnostic GeoPackage containing all fit-valid PS and all QA fields

No 30 m / 50 m / 100 m grid, no spatial averaging, no interpolation and no
spatial smoothing are used.

LOS sign convention
-------------------
  positive = toward satellite
  negative = away from satellite

This is a custom pySTAMPS extension. It is not an official StaMPS routine and
is not certified as element-wise equivalent to MATLAB StaMPS.
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
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class JointRegularizedError(RuntimeError):
    """Fatal processing error."""


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


@dataclass(slots=True)
class Penalty:
    matrix: np.ndarray
    curvature_operator_velocity: np.ndarray
    curvature_sigma_mm_yr: float
    first_difference_sigma_mm_yr: float | None


# -----------------------------------------------------------------------------
# Imports and input support
# -----------------------------------------------------------------------------


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise JointRegularizedError(
            f"缺少Python包：{name}\n"
            f"请在当前stamps环境安装：python -m pip install {pip_name or name}\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def load_support(repo_root: Path):
    path = repo_root / "calc_annual_velocity_gls.py"
    if not path.exists():
        raise JointRegularizedError(f"缺少GLS支撑模块：{path}")
    spec = importlib.util.spec_from_file_location(
        "pystamps_regularized_gls_support",
        path,
    )
    if spec is None or spec.loader is None:
        raise JointRegularizedError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Design matrix and regularization
# -----------------------------------------------------------------------------


def days_from(date: datetime, origin: datetime) -> float:
    return (date - origin).total_seconds() / 86400.0


def build_design(
    dates: list[datetime],
    *,
    seasonal_harmonics: int,
    seasonal_period_days: float,
) -> Design:
    if len(dates) < 3:
        raise JointRegularizedError("至少需要3个获取日期")
    if seasonal_harmonics not in {0, 1, 2}:
        raise JointRegularizedError("seasonal_harmonics只能为0、1或2")

    origin = datetime(min(date.year for date in dates), 1, 1)
    t_days = np.asarray(
        [days_from(date, origin) for date in dates],
        dtype=np.float64,
    )
    model_years = list(
        range(
            min(date.year for date in dates),
            max(date.year for date in dates) + 1,
        )
    )

    columns: list[np.ndarray] = [
        np.ones(len(dates), dtype=np.float64)
    ]
    names = ["intercept_rad"]
    year_column_indices: list[int] = []
    year_coverage: list[dict[str, Any]] = []

    for year in model_years:
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        start_day = days_from(start, origin)
        end_day = days_from(end, origin)

        # Continuous cumulative exposure basis. The derivative inside each
        # calendar year is the corresponding annual slope coefficient.
        exposure_year = (
            np.clip(t_days - start_day, 0.0, end_day - start_day)
            / 365.25
        )
        year_column_indices.append(len(columns))
        columns.append(exposure_year)
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
                "period_start": (
                    min(year_dates).strftime("%Y-%m-%d")
                    if year_dates
                    else None
                ),
                "period_end": (
                    max(year_dates).strftime("%Y-%m-%d")
                    if year_dates
                    else None
                ),
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

    matrix = np.column_stack(columns).astype(np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise JointRegularizedError(
            f"联合设计矩阵秩不足：rank={rank}, 参数={matrix.shape[1]}"
        )

    return Design(
        matrix=matrix,
        parameter_names=names,
        model_years=model_years,
        year_column_indices=np.asarray(
            year_column_indices,
            dtype=np.int64,
        ),
        annual_sin_index=annual_sin_index,
        annual_cos_index=annual_cos_index,
        semiannual_sin_index=semiannual_sin_index,
        semiannual_cos_index=semiannual_cos_index,
        origin=origin,
        t_days=t_days,
        year_coverage=year_coverage,
    )


def difference_matrix(order: int, n: int) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be >= 1")
    if n <= order:
        return np.zeros((0, n), dtype=np.float64)
    return np.diff(np.eye(n, dtype=np.float64), n=order, axis=0)


def build_penalty(
    design: Design,
    *,
    phase_to_velocity_abs: float,
    curvature_sigma_mm_yr: float,
    first_difference_sigma_mm_yr: float | None,
) -> Penalty:
    p = design.matrix.shape[1]
    n_year = len(design.model_years)

    selector = np.zeros((n_year, p), dtype=np.float64)
    selector[
        np.arange(n_year),
        design.year_column_indices,
    ] = float(phase_to_velocity_abs)

    d2_velocity = difference_matrix(2, n_year) @ selector
    penalty = np.zeros((p, p), dtype=np.float64)

    if not np.isfinite(curvature_sigma_mm_yr) or curvature_sigma_mm_yr <= 0:
        raise JointRegularizedError(
            "curvature_sigma_mm_yr必须是有限正数"
        )
    if d2_velocity.size:
        penalty += (
            d2_velocity.T @ d2_velocity
            / float(curvature_sigma_mm_yr) ** 2
        )

    if first_difference_sigma_mm_yr is not None:
        if (
            not np.isfinite(first_difference_sigma_mm_yr)
            or first_difference_sigma_mm_yr <= 0
        ):
            raise JointRegularizedError(
                "first_difference_sigma_mm_yr必须是有限正数"
            )
        d1_velocity = difference_matrix(1, n_year) @ selector
        if d1_velocity.size:
            penalty += (
                d1_velocity.T @ d1_velocity
                / float(first_difference_sigma_mm_yr) ** 2
            )

    return Penalty(
        matrix=penalty,
        curvature_operator_velocity=d2_velocity,
        curvature_sigma_mm_yr=float(curvature_sigma_mm_yr),
        first_difference_sigma_mm_yr=(
            None
            if first_difference_sigma_mm_yr is None
            else float(first_difference_sigma_mm_yr)
        ),
    )


# -----------------------------------------------------------------------------
# Robust penalized GLS
# -----------------------------------------------------------------------------


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


def solve_penalized_batch(
    yw: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Solve batch penalized weighted least squares."""
    batch = int(yw.shape[0])
    p = int(Xw.shape[1])

    data_normal = np.einsum(
        "bn,np,nq->bpq",
        weights,
        Xw,
        Xw,
        optimize=True,
    )
    rhs = np.einsum(
        "bn,bn,np->bp",
        weights,
        yw,
        Xw,
        optimize=True,
    )

    normal = data_normal + penalty[None, :, :]
    trace = np.trace(normal, axis1=1, axis2=2)
    numerical_ridge = np.maximum(
        trace / max(p, 1) * 1.0e-12,
        1.0e-15,
    )
    normal = (
        normal
        + numerical_ridge[:, None, None]
        * np.eye(p, dtype=np.float64)[None, :, :]
    )

    beta = np.full((batch, p), np.nan, dtype=np.float64)
    inverse = np.full((batch, p, p), np.nan, dtype=np.float64)
    valid = np.ones(batch, dtype=bool)

    try:
        beta = np.linalg.solve(normal, rhs[..., None]).squeeze(-1)
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        for row in range(batch):
            try:
                beta[row] = np.linalg.solve(normal[row], rhs[row])
                inverse[row] = np.linalg.inv(normal[row])
            except np.linalg.LinAlgError:
                valid[row] = False

    eigenvalues = np.linalg.eigvalsh(normal)
    condition = np.divide(
        eigenvalues[:, -1],
        eigenvalues[:, 0],
        out=np.full(batch, np.inf, dtype=np.float64),
        where=eigenvalues[:, 0] > 0,
    )

    effective_df = np.einsum(
        "bij,bji->b",
        inverse,
        data_normal,
        optimize=True,
    )
    effective_df = np.clip(effective_df, 0.0, float(p))

    valid &= (
        np.all(np.isfinite(beta), axis=1)
        & np.isfinite(condition)
        & np.isfinite(effective_df)
    )
    return beta, inverse, condition, effective_df, valid


def robust_penalized_gls_batch(
    y_original: np.ndarray,
    context: WhitenedContext,
    penalty: Penalty,
    *,
    huber_c: float,
    weight_floor: float,
    max_iterations: int,
    convergence: float,
    max_condition: float,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_original, dtype=np.float64)
    X = context.X
    Xw = context.Xw
    yw = y @ context.whitener.T
    batch, n_obs = yw.shape
    p = Xw.shape[1]

    weights = np.ones_like(yw, dtype=np.float64)
    beta, inverse, condition, effective_df, fit_valid = (
        solve_penalized_batch(
            yw,
            Xw,
            weights,
            penalty.matrix,
        )
    )
    beta_nonrobust = beta.copy()
    iterations = np.zeros(batch, dtype=np.uint8)

    previous = beta.copy()
    for iteration in range(max(1, int(max_iterations))):
        residual = yw - beta @ Xw.T
        center = np.median(residual, axis=1)
        mad = np.median(
            np.abs(residual - center[:, None]),
            axis=1,
        )
        scale = 1.4826 * mad
        rms = np.sqrt(np.mean(residual * residual, axis=1))
        scale = np.where(
            np.isfinite(scale) & (scale > 1.0e-8),
            scale,
            rms,
        )
        scale = np.maximum(scale, 1.0e-8)

        standardized = (
            np.abs(residual)
            / (float(huber_c) * scale[:, None])
        )
        new_weights = np.ones_like(standardized)
        large = standardized > 1.0
        new_weights[large] = 1.0 / standardized[large]
        new_weights = np.clip(
            new_weights,
            float(weight_floor),
            1.0,
        )

        (
            beta_new,
            inverse_new,
            condition_new,
            effective_df_new,
            valid_new,
        ) = solve_penalized_batch(
            yw,
            Xw,
            new_weights,
            penalty.matrix,
        )

        denominator = np.maximum(1.0e-7, np.abs(previous))
        relative = np.max(
            np.abs(beta_new - previous) / denominator,
            axis=1,
        )
        converged = (
            np.isfinite(relative)
            & (relative <= float(convergence))
        )
        iterations[~converged] = np.uint8(
            min(iteration + 1, 255)
        )

        beta = beta_new
        inverse = inverse_new
        condition = condition_new
        effective_df = effective_df_new
        fit_valid &= valid_new
        weights = new_weights
        previous = beta_new.copy()

        if np.all(converged | ~fit_valid):
            break

    residual_w = yw - beta @ Xw.T
    residual_original = y - beta @ X.T

    psi2 = (weights * residual_w) ** 2
    meat = np.einsum(
        "bn,np,nq->bpq",
        psi2,
        Xw,
        Xw,
        optimize=True,
    )
    covariance_beta = inverse @ meat @ inverse

    weighted_n = np.sum(weights, axis=1)
    dof = np.maximum(weighted_n - effective_df, 1.0)
    reduced = (
        np.sum(weights * residual_w * residual_w, axis=1)
        / dof
    )

    # Penalized model-based fallback covariance. This is conditional on the
    # selected regularization strength and does not include smoothing bias.
    data_normal = np.einsum(
        "bn,np,nq->bpq",
        weights,
        Xw,
        Xw,
        optimize=True,
    )
    fallback = inverse @ data_normal @ inverse
    fallback *= reduced[:, None, None]

    diagonal = np.diagonal(
        covariance_beta,
        axis1=1,
        axis2=2,
    ).copy()
    fallback_diagonal = np.diagonal(
        fallback,
        axis1=1,
        axis2=2,
    )
    bad = ~np.isfinite(diagonal) | (diagonal <= 0)
    diagonal[bad] = fallback_diagonal[bad]
    beta_se = np.sqrt(np.maximum(diagonal, 0.0))

    effective_n = np.divide(
        weighted_n**2,
        np.sum(weights * weights, axis=1),
        out=np.zeros(batch, dtype=np.float64),
        where=np.sum(weights * weights, axis=1) > 0,
    )
    downweighted = np.sum(
        weights < 0.999,
        axis=1,
    ).astype(np.uint16)

    fit_valid &= (
        np.all(np.isfinite(beta), axis=1)
        & np.all(np.isfinite(beta_se), axis=1)
        & np.isfinite(condition)
        & (condition <= float(max_condition))
        & (effective_n > effective_df + 1.0)
    )

    return {
        "beta": beta,
        "beta_nonrobust": beta_nonrobust,
        "beta_se": beta_se,
        "rmse_rad": np.sqrt(
            np.mean(residual_original * residual_original, axis=1)
        ),
        "whitened_rmse": np.sqrt(
            np.mean(residual_w * residual_w, axis=1)
        ),
        "effective_n": effective_n,
        "effective_df": effective_df,
        "downweighted_mode_count": downweighted,
        "design_condition": condition,
        "irls_iterations": iterations,
        "fit_valid": fit_valid,
    }


# -----------------------------------------------------------------------------
# GCV selection of regularization strength
# -----------------------------------------------------------------------------


def parse_sigma_candidates(text: str) -> list[float]:
    output: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value) or value <= 0:
            raise JointRegularizedError(
                f"非法GCV候选值：{token}"
            )
        output.append(value)
    output = sorted(set(output))
    if not output:
        raise JointRegularizedError("GCV候选值为空")
    return output


def deterministic_sample_indices(
    valid_counts: np.ndarray,
    sample_size: int,
    seed: int,
    minimum_count: int,
) -> np.ndarray:
    eligible = np.flatnonzero(valid_counts >= int(minimum_count))
    if eligible.size == 0:
        raise JointRegularizedError("没有满足GCV采样条件的PS")
    if eligible.size <= sample_size:
        return eligible
    rng = np.random.default_rng(int(seed))
    return np.sort(
        rng.choice(
            eligible,
            size=int(sample_size),
            replace=False,
        )
    )


def gcv_score_for_context(
    y: np.ndarray,
    context: WhitenedContext,
    penalty_matrix: np.ndarray,
) -> np.ndarray:
    """Non-robust GCV score in the whitened domain for multiple PS."""
    yw = y @ context.whitener.T
    Xw = context.Xw
    n_obs = Xw.shape[0]

    data_normal = Xw.T @ Xw
    normal = data_normal + penalty_matrix
    trace = float(np.trace(normal))
    ridge = max(
        trace / max(normal.shape[0], 1) * 1.0e-12,
        1.0e-15,
    )
    normal = normal + ridge * np.eye(normal.shape[0])

    inverse = np.linalg.inv(normal)
    beta = (inverse @ Xw.T @ yw.T).T
    residual = yw - beta @ Xw.T
    rss = np.sum(residual * residual, axis=1)
    effective_df = float(np.trace(inverse @ data_normal))
    denominator = max(float(n_obs) - effective_df, 1.0)

    # Standard GCV up to a candidate-independent scale factor.
    return rss / (denominator * denominator)


def select_curvature_sigma_gcv(
    inputs: Any,
    design: Design,
    support: Any,
    *,
    phase_to_velocity_abs: float,
    candidates: list[float],
    first_difference_sigma_mm_yr: float | None,
    covariance_mode: str,
    eigen_floor_rel: float,
    sample_size: int,
    seed: int,
    minimum_dof: int,
    log_tolerance: float,
) -> tuple[float, list[dict[str, Any]]]:
    valid_counts = np.sum(
        np.isfinite(inputs.ph_ts),
        axis=1,
    )
    sample_ix = deterministic_sample_indices(
        valid_counts,
        sample_size,
        seed,
        design.matrix.shape[1] + minimum_dof,
    )
    sample = np.asarray(
        inputs.ph_ts[sample_ix, :],
        dtype=np.float64,
    )
    valid = np.isfinite(sample)

    contexts: dict[bytes, WhitenedContext | None] = {}
    score_lists: dict[float, list[np.ndarray]] = {
        value: [] for value in candidates
    }

    penalties = {
        value: build_penalty(
            design,
            phase_to_velocity_abs=phase_to_velocity_abs,
            curvature_sigma_mm_yr=value,
            first_difference_sigma_mm_yr=(
                first_difference_sigma_mm_yr
            ),
        )
        for value in candidates
    }

    for rows, mask in pattern_groups(valid):
        key = np.packbits(mask, bitorder="little").tobytes()
        if key not in contexts:
            X = design.matrix[mask, :]
            if (
                X.shape[0]
                < design.matrix.shape[1] + minimum_dof
                or np.linalg.matrix_rank(X)
                < design.matrix.shape[1]
            ):
                contexts[key] = None
            else:
                covariance = inputs.covariance[np.ix_(mask, mask)]
                whitener, meta = support.covariance_whitener(
                    covariance,
                    covariance_mode=covariance_mode,
                    eigen_floor_rel=eigen_floor_rel,
                )
                contexts[key] = WhitenedContext(
                    mask=mask.copy(),
                    X=X,
                    Xw=whitener @ X,
                    whitener=whitener,
                    covariance_meta=meta,
                )
        context = contexts[key]
        if context is None:
            continue

        y = sample[rows, :][:, mask]
        for candidate in candidates:
            scores = gcv_score_for_context(
                y,
                context,
                penalties[candidate].matrix,
            )
            scores = scores[
                np.isfinite(scores) & (scores > 0)
            ]
            if scores.size:
                score_lists[candidate].append(scores)

    table: list[dict[str, Any]] = []
    for candidate in candidates:
        values = (
            np.concatenate(score_lists[candidate])
            if score_lists[candidate]
            else np.asarray([], dtype=np.float64)
        )
        if values.size:
            log_values = np.log(values)
            median_log = float(np.median(log_values))
            mad_log = float(
                1.4826
                * np.median(
                    np.abs(log_values - median_log)
                )
            )
            median_score = float(np.exp(median_log))
        else:
            median_log = float("inf")
            mad_log = float("nan")
            median_score = float("inf")
        table.append(
            {
                "curvature_sigma_mm_yr": float(candidate),
                "sample_count": int(values.size),
                "median_gcv": median_score,
                "median_log_gcv": median_log,
                "mad_log_gcv": mad_log,
            }
        )

    finite_rows = [
        row for row in table
        if np.isfinite(row["median_log_gcv"])
    ]
    if not finite_rows:
        raise JointRegularizedError("GCV未得到有效候选结果")

    minimum_log = min(row["median_log_gcv"] for row in finite_rows)
    acceptable = [
        row
        for row in finite_rows
        if row["median_log_gcv"]
        <= minimum_log + float(log_tolerance)
    ]
    # Prefer the weakest regularization (largest curvature sigma) among
    # statistically near-equivalent GCV candidates to reduce over-smoothing.
    selected_row = max(
        acceptable,
        key=lambda row: row["curvature_sigma_mm_yr"],
    )
    selected = float(selected_row["curvature_sigma_mm_yr"])
    return selected, table


# -----------------------------------------------------------------------------
# Metrics, checkpoints and independent validation
# -----------------------------------------------------------------------------


def per_year_observation_metrics(
    valid: np.ndarray,
    dates: list[datetime],
    model_years: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    batch = valid.shape[0]
    n_year = len(model_years)
    counts = np.zeros((batch, n_year), dtype=np.uint16)
    spans = np.zeros((batch, n_year), dtype=np.float32)

    for col, year in enumerate(model_years):
        epoch_ix = np.asarray(
            [
                index
                for index, date in enumerate(dates)
                if date.year == year
            ],
            dtype=np.int64,
        )
        if epoch_ix.size == 0:
            continue
        subset = valid[:, epoch_ix]
        count = np.sum(subset, axis=1)
        counts[:, col] = np.minimum(
            count,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

        day_values = np.asarray(
            [
                (
                    dates[int(index)] - datetime(year, 1, 1)
                ).total_seconds()
                / 86400.0
                for index in epoch_ix
            ],
            dtype=np.float64,
        )
        first = np.min(
            np.where(subset, day_values[None, :], np.inf),
            axis=1,
        )
        last = np.max(
            np.where(subset, day_values[None, :], -np.inf),
            axis=1,
        )
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
    sigma = 1.4826 * float(
        np.median(np.abs(finite - median))
    )
    return float(
        max(
            minimum,
            min(
                hard_cap,
                median + multiplier * max(sigma, 0.0),
            ),
        )
    )


def input_signature(
    dataset: Path,
    args: argparse.Namespace,
    design: Design,
    selected_curvature_sigma: float,
) -> tuple[dict[str, Any], str]:
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
            files.append(
                {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    payload = {
        "version": 2,
        "files": files,
        "parameter_names": design.parameter_names,
        "model_years": design.model_years,
        "selected_curvature_sigma_mm_yr": (
            selected_curvature_sigma
        ),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"overwrite", "resume"}
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def save_chunk(
    path: Path,
    signature: str,
    start: int,
    stop: int,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload: dict[str, Any] = {
        "signature": np.asarray(signature),
        "start": np.asarray(start),
        "stop": np.asarray(stop),
    }
    payload.update(arrays)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_chunk(
    path: Path,
    signature: str,
    start: int,
    stop: int,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        stored = str(
            np.asarray(archive["signature"]).reshape(-1)[0]
        )
        if stored != signature:
            return None
        if int(archive["start"]) != start:
            return None
        if int(archive["stop"]) != stop:
            return None
        return {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"signature", "start", "stop"}
        }
    except Exception:
        return None


def save_year_validation_checkpoint(
    path: Path,
    signature: str,
    year: int,
    arrays: dict[str, np.ndarray],
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    payload: dict[str, Any] = {
        "signature": np.asarray(signature),
        "year": np.asarray(year),
    }
    payload.update(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_year_validation_checkpoint(
    path: Path,
    signature: str,
    year: int,
    n_ps: int,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        if (
            str(np.asarray(archive["signature"]).reshape(-1)[0])
            != signature
        ):
            return None
        if int(np.asarray(archive["year"]).reshape(-1)[0]) != year:
            return None
        output = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"signature", "year"}
        }
        if any(value.shape[0] != n_ps for value in output.values()):
            return None
        return output
    except Exception:
        return None


def independent_year_fit(
    inputs: Any,
    support: Any,
    design: Design,
    arrays: dict[str, np.ndarray],
    year_col: int,
    *,
    chunk_ps: int,
    min_epochs: int,
    min_span_days: float,
    covariance_mode: str,
    eigen_floor_rel: float,
    huber_c: float,
    weight_floor: float,
    max_iterations: int,
    convergence: float,
    max_condition: float,
) -> dict[str, np.ndarray]:
    year = design.model_years[year_col]
    epoch_ix = np.asarray(
        [
            index
            for index, date in enumerate(inputs.dates)
            if date.year == year
        ],
        dtype=np.int64,
    )
    dates = [inputs.dates[int(index)] for index in epoch_ix]
    origin = min(dates)
    t_all = np.asarray(
        [
            (date - origin).total_seconds() / 86400.0
            for date in dates
        ],
        dtype=np.float64,
    )
    covariance_all = inputs.covariance[np.ix_(epoch_ix, epoch_ix)]

    n_ps = inputs.n_ps
    output = {
        "ind_velocity_mm_yr": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "ind_velocity_std_mm_yr": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "ind_rmse_mm": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "ind_n_obs": np.zeros(n_ps, dtype=np.uint16),
        "ind_span_days": np.zeros(n_ps, dtype=np.float32),
        "ind_fit_valid": np.zeros(n_ps, dtype=np.uint8),
    }

    annual_sin_basis = (
        design.matrix[epoch_ix, design.annual_sin_index]
        if design.annual_sin_index >= 0
        else np.zeros(epoch_ix.size, dtype=np.float64)
    )
    annual_cos_basis = (
        design.matrix[epoch_ix, design.annual_cos_index]
        if design.annual_cos_index >= 0
        else np.zeros(epoch_ix.size, dtype=np.float64)
    )
    semi_sin_basis = (
        design.matrix[epoch_ix, design.semiannual_sin_index]
        if design.semiannual_sin_index is not None
        else np.zeros(epoch_ix.size, dtype=np.float64)
    )
    semi_cos_basis = (
        design.matrix[epoch_ix, design.semiannual_cos_index]
        if design.semiannual_cos_index is not None
        else np.zeros(epoch_ix.size, dtype=np.float64)
    )

    phase_to_velocity = (
        -inputs.wavelength_m
        / (4.0 * math.pi)
        * 1000.0
        * 365.25
    )
    phase_to_mm = abs(
        inputs.wavelength_m
        / (4.0 * math.pi)
        * 1000.0
    )

    for start in range(0, n_ps, int(chunk_ps)):
        stop = min(start + int(chunk_ps), n_ps)
        y = np.asarray(
            inputs.ph_ts[start:stop, :][:, epoch_ix],
            dtype=np.float64,
        )

        if design.annual_sin_index >= 0:
            y -= (
                arrays["annual_sin_rad"][start:stop, None]
                * annual_sin_basis[None, :]
            )
            y -= (
                arrays["annual_cos_rad"][start:stop, None]
                * annual_cos_basis[None, :]
            )
        if design.semiannual_sin_index is not None:
            y -= (
                arrays["semiannual_sin_rad"][start:stop, None]
                * semi_sin_basis[None, :]
            )
            y -= (
                arrays["semiannual_cos_rad"][start:stop, None]
                * semi_cos_basis[None, :]
            )

        valid = np.isfinite(y)
        for rows, mask in pattern_groups(valid):
            global_rows = start + rows
            n_obs = int(np.count_nonzero(mask))
            output["ind_n_obs"][global_rows] = np.uint16(
                min(n_obs, np.iinfo(np.uint16).max)
            )
            if n_obs == 0:
                continue
            t = t_all[mask]
            span = float(np.max(t) - np.min(t))
            output["ind_span_days"][global_rows] = np.float32(span)
            if n_obs < int(min_epochs) or span < float(min_span_days):
                continue

            result = support.robust_gls_batch(
                y[rows, :][:, mask],
                t,
                covariance_all[np.ix_(mask, mask)],
                covariance_mode=covariance_mode,
                eigen_floor_rel=eigen_floor_rel,
                robust=True,
                huber_c=huber_c,
                weight_floor=weight_floor,
                max_iterations=max_iterations,
                convergence=convergence,
            )
            velocity = result["slope_rad_day"] * phase_to_velocity
            velocity_se = (
                result["slope_se_rad_day"]
                * abs(phase_to_velocity)
            )
            fit = (
                np.asarray(result["fit_valid"], dtype=bool)
                & np.isfinite(velocity)
                & np.isfinite(velocity_se)
                & (
                    result["design_condition"]
                    <= float(max_condition)
                )
            )

            output["ind_velocity_mm_yr"][global_rows] = (
                velocity.astype(np.float32)
            )
            output["ind_velocity_std_mm_yr"][global_rows] = (
                velocity_se.astype(np.float32)
            )
            output["ind_rmse_mm"][global_rows] = (
                result["rmse_rad"] * phase_to_mm
            ).astype(np.float32)
            output["ind_fit_valid"][global_rows] = fit.astype(np.uint8)

            rejected = global_rows[~fit]
            if rejected.size:
                output["ind_velocity_mm_yr"][rejected] = np.nan
                output["ind_velocity_std_mm_yr"][rejected] = np.nan

        print(
            f"[VALIDATE][{year}] {stop}/{n_ps} "
            f"({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    return output


# -----------------------------------------------------------------------------
# GIS and tabular output
# -----------------------------------------------------------------------------


def auto_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = max(
        1,
        min(60, int(math.floor((lon0 + 180.0) / 6.0)) + 1),
    )
    return (32600 if lat0 >= 0 else 32700) + zone


def clean_shapefile(path: Path) -> None:
    for suffix in (
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".fix",
        ".sbn",
        ".sbx",
    ):
        component = path.with_suffix(suffix)
        if component.exists():
            component.unlink()


def write_shapefile(gdf: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_shapefile(path)
    gdf.to_file(
        path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False,
    )
    path.with_suffix(".cpg").write_text(
        "UTF-8",
        encoding="ascii",
    )


def qml_display_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 40.0
    value = float(np.percentile(np.abs(finite), 98))
    return max(10.0, math.ceil(value / 5.0) * 5.0)


def write_qml(path: Path, limit: float) -> None:
    text = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="VEL_MM_YR" graduatedMethod="GraduatedColor">
    <ranges>
      <range lower="{-limit}" upper="{-limit/2}" label="{-limit:.0f} to {-limit/2:.0f}" symbol="0"/>
      <range lower="{-limit/2}" upper="-8" label="{-limit/2:.0f} to -8" symbol="1"/>
      <range lower="-8" upper="8" label="-8 to 8" symbol="2"/>
      <range lower="8" upper="{limit/2}" label="8 to {limit/2:.0f}" symbol="3"/>
      <range lower="{limit/2}" upper="{limit}" label="{limit/2:.0f} to {limit:.0f}" symbol="4"/>
    </ranges>
    <symbols>
      <symbol type="marker" name="0"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="33,102,172,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.25"/></Option></layer></symbol>
      <symbol type="marker" name="1"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="103,169,207,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.22"/></Option></layer></symbol>
      <symbol type="marker" name="2"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="210,210,210,80"/><Option name="outline_style" value="no"/><Option name="size" value="0.12"/></Option></layer></symbol>
      <symbol type="marker" name="3"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="239,138,98,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.22"/></Option></layer></symbol>
      <symbol type="marker" name="4"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="178,24,43,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.25"/></Option></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
    path.write_text(text, encoding="utf-8")


def write_outputs(
    output_root: Path,
    inputs: Any,
    design: Design,
    arrays: dict[str, np.ndarray],
    formal_year: np.ndarray,
    summary: list[dict[str, Any]],
    target_epsg: int,
    *,
    write_diagnostic_gpkg: bool,
    write_final_shp: bool,
) -> dict[str, Any]:
    pandas = require_import("pandas", "pandas")
    geopandas = require_import("geopandas", "geopandas")

    base_columns: dict[str, Any] = {
        "ps_id": np.arange(
            1,
            inputs.n_ps + 1,
            dtype=np.int64,
        ),
        "lon": inputs.lon,
        "lat": inputs.lat,
        "model_rms": arrays["model_rmse_mm"],
        "white_rms": arrays["whitened_rmse"],
        "eff_n": arrays["effective_n"],
        "eff_df": arrays["effective_df"],
        "cond_reg": arrays["design_condition"],
        "curv_rms": arrays["curvature_rms_mm_yr"],
        "ann_amp": arrays["annual_amplitude_mm"],
        "ann_peak": arrays["annual_peak_day"],
        "fit_ok": arrays["fit_valid"],
    }

    for col, year in enumerate(design.model_years):
        base_columns[f"v{year}"] = arrays["velocity_mm_yr"][:, col]
        base_columns[f"se{year}"] = arrays["velocity_std_mm_yr"][:, col]
        base_columns[f"vi{year}"] = arrays["ind_velocity_mm_yr"][:, col]
        base_columns[f"sei{year}"] = arrays[
            "ind_velocity_std_mm_yr"
        ][:, col]
        base_columns[f"dv{year}"] = arrays["model_delta_mm_yr"][:, col]
        base_columns[f"ag{year}"] = arrays["model_agree"][:, col]
        base_columns[f"q{year}"] = arrays["final_quality"][:, col]
        base_columns[f"sg{year}"] = arrays["significant"][:, col]
        base_columns[f"n{year}"] = arrays["n_obs_year"][:, col]
        base_columns[f"sp{year}"] = arrays["span_days_year"][:, col]

    base = pandas.DataFrame(base_columns)

    geometry = geopandas.points_from_xy(base["lon"], base["lat"])
    gdf = geopandas.GeoDataFrame(
        base,
        geometry=geometry,
        crs="EPSG:4326",
    ).to_crs(f"EPSG:{int(target_epsg)}")

    outputs: dict[str, Any] = {}
    if write_diagnostic_gpkg:
        gpkg = output_root / "joint_regularized_diagnostic.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        gdf.to_file(
            gpkg,
            layer="joint_regularized_annual_velocity",
            driver="GPKG",
        )
        outputs["diagnostic_gpkg"] = str(gpkg)

    formal_dir = output_root / "annual_shapefiles" / "formal"
    partial_dir = output_root / "annual_shapefiles" / "partial"
    shp_records: list[dict[str, Any]] = []

    if write_final_shp:
        for col, year in enumerate(design.model_years):
            final = arrays["final_quality"][:, col].astype(bool)
            source = gdf.loc[final].copy()
            velocity = arrays["velocity_mm_yr"][:, col][final]
            velocity_se = arrays["velocity_std_mm_yr"][:, col][final]
            ind_velocity = arrays["ind_velocity_mm_yr"][:, col][final]
            ind_se = arrays[
                "ind_velocity_std_mm_yr"
            ][:, col][final]
            delta = arrays["model_delta_mm_yr"][:, col][final]

            extreme = np.zeros(velocity.size, dtype=np.int16)
            extreme[np.abs(velocity) > 50.0] = 1
            extreme[np.abs(velocity) > 100.0] = 2

            if not np.any(final):
                shp_records.append(
                    {
                        "year": year,
                        "formal_year": int(formal_year[col]),
                        "points": 0,
                        "shapefile": None,
                        "reason": "no_final_quality_points",
                    }
                )
                continue

            result = geopandas.GeoDataFrame(
                {
                    "PS_ID": source["ps_id"].to_numpy(np.int64),
                    "YEAR": np.full(
                        len(source),
                        year,
                        dtype=np.int16,
                    ),
                    "VEL_MM_YR": velocity.astype(np.float32),
                    "VEL_SE": velocity_se.astype(np.float32),
                    "CI95_LO": (
                        velocity - 1.96 * velocity_se
                    ).astype(np.float32),
                    "CI95_HI": (
                        velocity + 1.96 * velocity_se
                    ).astype(np.float32),
                    "IND_VEL": ind_velocity.astype(np.float32),
                    "IND_SE": ind_se.astype(np.float32),
                    "DELTA_V": delta.astype(np.float32),
                    "AGREE": np.ones(
                        len(source),
                        dtype=np.int16,
                    ),
                    "Q_FINAL": np.ones(
                        len(source),
                        dtype=np.int16,
                    ),
                    "Q_SIGNIF": arrays[
                        "significant"
                    ][:, col][final].astype(np.int16),
                    "N_OBS": arrays[
                        "n_obs_year"
                    ][:, col][final].astype(np.int16),
                    "SPAN_DAY": arrays[
                        "span_days_year"
                    ][:, col][final].astype(np.float32),
                    "MOD_RMS": arrays[
                        "model_rmse_mm"
                    ][final].astype(np.float32),
                    "CURV_RMS": arrays[
                        "curvature_rms_mm_yr"
                    ][final].astype(np.float32),
                    "ANN_AMP": arrays[
                        "annual_amplitude_mm"
                    ][final].astype(np.float32),
                    "FORMAL_YR": np.full(
                        len(source),
                        int(formal_year[col]),
                        dtype=np.int16,
                    ),
                    "EXTREME": extreme,
                },
                geometry=source.geometry.to_numpy(),
                crs=source.crs,
            )

            directory = formal_dir if formal_year[col] else partial_dir
            suffix = "final" if formal_year[col] else "partial"
            shp = directory / f"annual_velocity_{year}_{suffix}.shp"
            write_shapefile(result, shp)
            write_qml(
                shp.with_suffix(".qml"),
                qml_display_limit(velocity),
            )
            shp_records.append(
                {
                    "year": year,
                    "formal_year": int(formal_year[col]),
                    "points": int(len(result)),
                    "shapefile": str(shp),
                }
            )

    summary_path = output_root / "joint_regularized_year_summary.csv"
    pandas.DataFrame(summary).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    outputs["summary_csv"] = str(summary_path)
    outputs["annual_shapefiles"] = shp_records

    field_dictionary = [
        ("VEL_MM_YR", "正则化联合年度LOS速率，mm/yr"),
        ("VEL_SE", "联合年度速率条件标准误差，mm/yr"),
        ("IND_VEL", "去季节后独立年度稳健GLS速率，mm/yr"),
        ("IND_SE", "独立年度速率标准误差，mm/yr"),
        ("DELTA_V", "联合速率减独立速率，mm/yr"),
        ("AGREE", "两个年度模型是否一致；输出点均为1"),
        ("Q_FINAL", "最终质量标记；输出点均为1"),
        ("Q_SIGNIF", "联合速率95%置信区间是否不跨0"),
        ("MOD_RMS", "全时段联合模型时间残差RMS，mm"),
        ("CURV_RMS", "年度速率二阶差分RMS，mm/yr"),
        ("ANN_AMP", "共同年周期振幅，mm"),
        ("EXTREME", "0正常；1为|V|>50；2为|V|>100"),
    ]
    dictionary_path = output_root / "field_dictionary.csv"
    with dictionary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "description"])
        writer.writerows(field_dictionary)
    outputs["field_dictionary"] = str(dictionary_path)
    return outputs


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dataset / "joint_regularized_seasonal_velocity_v2"
    )
    work_root = output_root / "_work"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    support = load_support(repo_root)
    inputs = support.load_inputs(
        dataset,
        repo_root,
        args.sigma_floor_deg,
    )
    design = build_design(
        inputs.dates,
        seasonal_harmonics=args.seasonal_harmonics,
        seasonal_period_days=args.seasonal_period_days,
    )

    p = design.matrix.shape[1]
    if inputs.n_epoch < p + args.minimum_dof:
        raise JointRegularizedError(
            f"获取日期数{inputs.n_epoch}不足以估计{p}个参数"
        )

    phase_to_velocity = (
        -inputs.wavelength_m
        / (4.0 * math.pi)
        * 1000.0
    )
    phase_to_velocity_abs = abs(phase_to_velocity)
    phase_to_mm_abs = abs(
        inputs.wavelength_m
        / (4.0 * math.pi)
        * 1000.0
    )

    gcv_table: list[dict[str, Any]] = []
    if args.curvature_sigma_mm_yr is None:
        candidates = parse_sigma_candidates(
            args.gcv_curvature_candidates
        )
        selected_sigma, gcv_table = select_curvature_sigma_gcv(
            inputs,
            design,
            support,
            phase_to_velocity_abs=phase_to_velocity_abs,
            candidates=candidates,
            first_difference_sigma_mm_yr=(
                args.first_difference_sigma_mm_yr
            ),
            covariance_mode=args.covariance_mode,
            eigen_floor_rel=args.eigen_floor_rel,
            sample_size=args.gcv_sample_ps,
            seed=args.gcv_seed,
            minimum_dof=args.minimum_dof,
            log_tolerance=args.gcv_log_tolerance,
        )
    else:
        selected_sigma = float(args.curvature_sigma_mm_yr)

    penalty = build_penalty(
        design,
        phase_to_velocity_abs=phase_to_velocity_abs,
        curvature_sigma_mm_yr=selected_sigma,
        first_difference_sigma_mm_yr=(
            args.first_difference_sigma_mm_yr
        ),
    )

    print("============================================================")
    print("Regularized joint annual velocity V2")
    print("============================================================")
    print(f"PS                     : {inputs.n_ps:,}")
    print(f"Acquisitions           : {inputs.n_epoch}")
    print(f"Years                  : {design.model_years}")
    print(f"Parameters             : {p}")
    print(f"Curvature sigma        : {selected_sigma:.3f} mm/yr")
    print(f"Covariance             : {inputs.covariance_source}")

    if gcv_table:
        pandas = require_import("pandas", "pandas")
        gcv_path = output_root / "gcv_regularization_selection.csv"
        pandas.DataFrame(gcv_table).to_csv(
            gcv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"GCV table              : {gcv_path}")

    signature_payload, signature = input_signature(
        dataset,
        args,
        design,
        selected_sigma,
    )

    n_ps = inputs.n_ps
    n_year = len(design.model_years)
    arrays: dict[str, np.ndarray] = {
        "velocity_mm_yr": np.full(
            (n_ps, n_year),
            np.nan,
            dtype=np.float32,
        ),
        "velocity_std_mm_yr": np.full(
            (n_ps, n_year),
            np.nan,
            dtype=np.float32,
        ),
        "velocity_nonrobust_mm_yr": np.full(
            (n_ps, n_year),
            np.nan,
            dtype=np.float32,
        ),
        "n_obs_year": np.zeros(
            (n_ps, n_year),
            dtype=np.uint16,
        ),
        "span_days_year": np.zeros(
            (n_ps, n_year),
            dtype=np.float32,
        ),
        "model_rmse_mm": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "whitened_rmse": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "effective_n": np.zeros(n_ps, dtype=np.float32),
        "effective_df": np.zeros(n_ps, dtype=np.float32),
        "downweighted_mode_count": np.zeros(
            n_ps,
            dtype=np.uint16,
        ),
        "design_condition": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "irls_iterations": np.zeros(n_ps, dtype=np.uint8),
        "fit_valid": np.zeros(n_ps, dtype=np.uint8),
        "annual_sin_rad": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "annual_cos_rad": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "annual_amplitude_mm": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "annual_peak_day": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
        "curvature_rms_mm_yr": np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        ),
    }
    if design.semiannual_sin_index is not None:
        arrays["semiannual_sin_rad"] = np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        )
        arrays["semiannual_cos_rad"] = np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        )
        arrays["semiannual_amplitude_mm"] = np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        )

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
        if (
            X.shape[0] < p + args.minimum_dof
            or np.linalg.matrix_rank(X) < p
        ):
            return None
        covariance = inputs.covariance[np.ix_(mask, mask)]
        whitener, meta = support.covariance_whitener(
            covariance,
            covariance_mode=args.covariance_mode,
            eigen_floor_rel=args.eigen_floor_rel,
        )
        context = WhitenedContext(
            mask=mask.copy(),
            X=X,
            Xw=whitener @ X,
            whitener=whitener,
            covariance_meta=meta,
        )
        context_cache[key] = context
        if len(context_cache) > args.context_cache_size:
            context_cache.popitem(last=False)
        if not covariance_meta_example:
            covariance_meta_example = dict(meta)
        return context

    for start in range(0, n_ps, args.chunk_ps):
        stop = min(start + args.chunk_ps, n_ps)
        checkpoint = work_root / f"joint_{start:07d}_{stop:07d}.npz"
        cached = (
            load_chunk(checkpoint, signature, start, stop)
            if args.resume
            else None
        )
        if cached is not None:
            for key, value in cached.items():
                if key in arrays:
                    arrays[key][start:stop] = value
            print(
                f"[JOINT_V2][RESUME] {stop}/{n_ps} "
                f"({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )
            continue

        y_chunk = np.asarray(
            inputs.ph_ts[start:stop, :],
            dtype=np.float64,
        )
        valid = np.isfinite(y_chunk)
        counts_year, spans_year = per_year_observation_metrics(
            valid,
            inputs.dates,
            design.model_years,
        )
        chunk = {
            key: np.asarray(value[start:stop]).copy()
            for key, value in arrays.items()
        }
        chunk["n_obs_year"] = counts_year
        chunk["span_days_year"] = spans_year

        for rows, mask in pattern_groups(valid):
            context = context_for(mask)
            if context is None:
                continue
            result = robust_penalized_gls_batch(
                y_chunk[rows, :][:, mask],
                context,
                penalty,
                huber_c=args.huber_c,
                weight_floor=args.weight_floor,
                max_iterations=args.irls_iterations,
                convergence=args.convergence,
                max_condition=args.max_regularized_condition,
            )

            beta = result["beta"]
            beta_se = result["beta_se"]
            beta_nonrobust = result["beta_nonrobust"]
            year_beta = beta[:, design.year_column_indices]
            year_se = beta_se[:, design.year_column_indices]
            year_nonrobust = beta_nonrobust[
                :, design.year_column_indices
            ]

            velocity = year_beta * phase_to_velocity
            velocity_se = year_se * phase_to_velocity_abs
            velocity_nonrobust = year_nonrobust * phase_to_velocity

            chunk["velocity_mm_yr"][rows, :] = velocity.astype(
                np.float32
            )
            chunk["velocity_std_mm_yr"][rows, :] = (
                velocity_se.astype(np.float32)
            )
            chunk["velocity_nonrobust_mm_yr"][rows, :] = (
                velocity_nonrobust.astype(np.float32)
            )
            chunk["model_rmse_mm"][rows] = (
                result["rmse_rad"] * phase_to_mm_abs
            ).astype(np.float32)
            chunk["whitened_rmse"][rows] = result[
                "whitened_rmse"
            ].astype(np.float32)
            chunk["effective_n"][rows] = result[
                "effective_n"
            ].astype(np.float32)
            chunk["effective_df"][rows] = result[
                "effective_df"
            ].astype(np.float32)
            chunk["downweighted_mode_count"][rows] = result[
                "downweighted_mode_count"
            ].astype(np.uint16)
            chunk["design_condition"][rows] = result[
                "design_condition"
            ].astype(np.float32)
            chunk["irls_iterations"][rows] = result[
                "irls_iterations"
            ].astype(np.uint8)
            chunk["fit_valid"][rows] = result[
                "fit_valid"
            ].astype(np.uint8)

            curvature = (
                penalty.curvature_operator_velocity @ beta.T
            ).T
            curvature_rms = (
                np.sqrt(np.mean(curvature * curvature, axis=1))
                if curvature.shape[1]
                else np.zeros(beta.shape[0], dtype=np.float64)
            )
            chunk["curvature_rms_mm_yr"][rows] = (
                curvature_rms.astype(np.float32)
            )

            if design.annual_sin_index >= 0:
                annual_sin = beta[:, design.annual_sin_index]
                annual_cos = beta[:, design.annual_cos_index]
                amplitude = (
                    np.hypot(annual_sin, annual_cos)
                    * phase_to_mm_abs
                )
                phase = np.mod(
                    np.arctan2(annual_sin, annual_cos),
                    2.0 * math.pi,
                )
                peak_day = (
                    phase
                    / (2.0 * math.pi)
                    * args.seasonal_period_days
                )
                chunk["annual_sin_rad"][rows] = annual_sin.astype(
                    np.float32
                )
                chunk["annual_cos_rad"][rows] = annual_cos.astype(
                    np.float32
                )
                chunk["annual_amplitude_mm"][rows] = amplitude.astype(
                    np.float32
                )
                chunk["annual_peak_day"][rows] = peak_day.astype(
                    np.float32
                )

            if design.semiannual_sin_index is not None:
                semi_sin = beta[:, design.semiannual_sin_index]
                semi_cos = beta[:, design.semiannual_cos_index]
                chunk["semiannual_sin_rad"][rows] = semi_sin.astype(
                    np.float32
                )
                chunk["semiannual_cos_rad"][rows] = semi_cos.astype(
                    np.float32
                )
                chunk["semiannual_amplitude_mm"][rows] = (
                    np.hypot(semi_sin, semi_cos)
                    * phase_to_mm_abs
                ).astype(np.float32)

        for key in arrays:
            arrays[key][start:stop] = chunk[key]
        save_chunk(
            checkpoint,
            signature,
            start,
            stop,
            chunk,
        )
        print(
            f"[JOINT_V2][FIT] {stop}/{n_ps} "
            f"({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    # Independent annual validation after removing the common seasonal term.
    arrays["ind_velocity_mm_yr"] = np.full(
        (n_ps, n_year),
        np.nan,
        dtype=np.float32,
    )
    arrays["ind_velocity_std_mm_yr"] = np.full(
        (n_ps, n_year),
        np.nan,
        dtype=np.float32,
    )
    arrays["ind_rmse_mm"] = np.full(
        (n_ps, n_year),
        np.nan,
        dtype=np.float32,
    )
    arrays["ind_fit_valid"] = np.zeros(
        (n_ps, n_year),
        dtype=np.uint8,
    )

    if args.independent_validation:
        for col, year in enumerate(design.model_years):
            checkpoint = work_root / f"validation_{year}.npz"
            cached = (
                load_year_validation_checkpoint(
                    checkpoint,
                    signature,
                    year,
                    n_ps,
                )
                if args.resume
                else None
            )
            if cached is None:
                result = independent_year_fit(
                    inputs,
                    support,
                    design,
                    arrays,
                    col,
                    chunk_ps=args.validation_chunk_ps,
                    min_epochs=args.min_year_epochs,
                    min_span_days=args.min_year_span_days,
                    covariance_mode=args.covariance_mode,
                    eigen_floor_rel=args.eigen_floor_rel,
                    huber_c=args.huber_c,
                    weight_floor=args.weight_floor,
                    max_iterations=args.irls_iterations,
                    convergence=args.convergence,
                    max_condition=args.max_independent_condition,
                )
                save_year_validation_checkpoint(
                    checkpoint,
                    signature,
                    year,
                    result,
                )
            else:
                result = cached
                print(
                    f"[VALIDATE][RESUME] {year}",
                    flush=True,
                )

            arrays["ind_velocity_mm_yr"][:, col] = result[
                "ind_velocity_mm_yr"
            ]
            arrays["ind_velocity_std_mm_yr"][:, col] = result[
                "ind_velocity_std_mm_yr"
            ]
            arrays["ind_rmse_mm"][:, col] = result[
                "ind_rmse_mm"
            ]
            arrays["ind_fit_valid"][:, col] = result[
                "ind_fit_valid"
            ]
    else:
        # No validation means no final scientific-quality yearly output.
        warnings.warn(
            "已禁用独立年度验证；final_quality将全部为0"
        )

    fit_valid = arrays["fit_valid"].astype(bool)
    model_rmse_cap = adaptive_cap(
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

    arrays["model_delta_mm_yr"] = (
        arrays["velocity_mm_yr"]
        - arrays["ind_velocity_mm_yr"]
    ).astype(np.float32)
    arrays["model_agree"] = np.zeros(
        (n_ps, n_year),
        dtype=np.uint8,
    )
    arrays["final_quality"] = np.zeros(
        (n_ps, n_year),
        dtype=np.uint8,
    )
    arrays["significant"] = np.zeros(
        (n_ps, n_year),
        dtype=np.uint8,
    )

    rate_se_caps = np.full(n_year, np.nan, dtype=np.float64)
    ind_se_caps = np.full(n_year, np.nan, dtype=np.float64)
    summary: list[dict[str, Any]] = []

    for col, year in enumerate(design.model_years):
        joint_velocity = arrays["velocity_mm_yr"][:, col]
        joint_se = arrays["velocity_std_mm_yr"][:, col]
        ind_velocity = arrays["ind_velocity_mm_yr"][:, col]
        ind_se = arrays["ind_velocity_std_mm_yr"][:, col]
        ind_fit = arrays["ind_fit_valid"][:, col].astype(bool)

        eligible_joint = (
            fit_valid
            & formal_year[col]
            & (arrays["model_rmse_mm"] <= model_rmse_cap)
            & (
                arrays["n_obs_year"][:, col]
                >= args.min_year_epochs
            )
            & (
                arrays["span_days_year"][:, col]
                >= args.min_year_span_days
            )
            & np.isfinite(joint_velocity)
            & np.isfinite(joint_se)
        )
        joint_cap = adaptive_cap(
            joint_se[eligible_joint],
            hard_cap=args.max_rate_se_mm_yr,
            minimum=args.min_rate_se_cap_mm_yr,
            multiplier=args.threshold_mad_multiplier,
        )
        rate_se_caps[col] = joint_cap

        eligible_ind = (
            ind_fit
            & np.isfinite(ind_velocity)
            & np.isfinite(ind_se)
        )
        ind_cap = adaptive_cap(
            ind_se[eligible_ind],
            hard_cap=args.max_independent_se_mm_yr,
            minimum=args.min_independent_se_cap_mm_yr,
            multiplier=args.threshold_mad_multiplier,
        )
        ind_se_caps[col] = ind_cap

        combined_se = np.sqrt(
            np.maximum(joint_se, 0.0) ** 2
            + np.maximum(ind_se, 0.0) ** 2
        )
        tolerance = np.maximum(
            float(args.model_agreement_abs_mm_yr),
            float(args.model_agreement_sigma) * combined_se,
        )
        agree = (
            eligible_ind
            & np.isfinite(tolerance)
            & (
                np.abs(joint_velocity - ind_velocity)
                <= tolerance
            )
        )
        arrays["model_agree"][:, col] = agree.astype(np.uint8)

        final = (
            eligible_joint
            & (joint_se <= joint_cap)
            & eligible_ind
            & (ind_se <= ind_cap)
            & agree
            & (
                np.abs(joint_velocity)
                <= args.absolute_rate_cap_mm_yr
            )
        )
        arrays["final_quality"][:, col] = final.astype(np.uint8)

        significant = final & (
            ((joint_velocity - 1.96 * joint_se) > 0)
            | ((joint_velocity + 1.96 * joint_se) < 0)
        )
        arrays["significant"][:, col] = significant.astype(np.uint8)

        values = joint_velocity[final]
        deltas = arrays["model_delta_mm_yr"][:, col][final]
        summary.append(
            {
                **design.year_coverage[col],
                "formal_year": bool(formal_year[col]),
                "joint_rate_se_cap_mm_yr": float(joint_cap),
                "independent_rate_se_cap_mm_yr": float(ind_cap),
                "final_ps": int(np.count_nonzero(final)),
                "final_fraction": float(np.mean(final)),
                "significant_ps": int(np.count_nonzero(significant)),
                "model_agree_ps": int(np.count_nonzero(agree)),
                "velocity_p02_mm_yr": (
                    float(np.percentile(values, 2))
                    if values.size
                    else None
                ),
                "velocity_median_mm_yr": (
                    float(np.median(values))
                    if values.size
                    else None
                ),
                "velocity_mean_mm_yr": (
                    float(np.mean(values))
                    if values.size
                    else None
                ),
                "velocity_p98_mm_yr": (
                    float(np.percentile(values, 98))
                    if values.size
                    else None
                ),
                "model_delta_median_mm_yr": (
                    float(np.median(deltas))
                    if deltas.size
                    else None
                ),
                "model_delta_p02_mm_yr": (
                    float(np.percentile(deltas, 2))
                    if deltas.size
                    else None
                ),
                "model_delta_p98_mm_yr": (
                    float(np.percentile(deltas, 98))
                    if deltas.size
                    else None
                ),
            }
        )

    target_epsg = int(
        args.target_epsg
        or auto_utm_epsg(inputs.lon, inputs.lat)
    )
    outputs = write_outputs(
        output_root,
        inputs,
        design,
        arrays,
        formal_year,
        summary,
        target_epsg,
        write_diagnostic_gpkg=args.write_diagnostic_gpkg,
        write_final_shp=args.write_final_shp,
    )

    # NPZ is the canonical numeric archive to avoid HDF5 dimension ambiguity.
    npz_path = output_root / "joint_regularized_numeric_results.npz"
    np.savez_compressed(
        npz_path,
        year=np.asarray(design.model_years, dtype=np.int16),
        lon=np.asarray(inputs.lon, dtype=np.float64),
        lat=np.asarray(inputs.lat, dtype=np.float64),
        **arrays,
    )
    outputs["numeric_npz"] = str(npz_path)

    report = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "output": str(output_root),
        "model": (
            "continuous calendar-year slopes + common seasonal terms + "
            "second-difference annual-rate regularization"
        ),
        "estimator": (
            "SBAS acquisition covariance + covariance whitening + "
            "penalized Huber IRLS GLS"
        ),
        "independent_validation": (
            "de-seasonalized per-year robust covariance-aware GLS"
        ),
        "selected_curvature_sigma_mm_yr": selected_sigma,
        "first_difference_sigma_mm_yr": (
            args.first_difference_sigma_mm_yr
        ),
        "gcv_table": gcv_table,
        "model_years": design.model_years,
        "parameter_names": design.parameter_names,
        "model_rmse_cap_mm": model_rmse_cap,
        "joint_rate_se_caps_mm_yr": rate_se_caps.tolist(),
        "independent_rate_se_caps_mm_yr": ind_se_caps.tolist(),
        "model_agreement": {
            "absolute_floor_mm_yr": args.model_agreement_abs_mm_yr,
            "sigma_multiplier": args.model_agreement_sigma,
        },
        "summary": summary,
        "covariance_source": inputs.covariance_source,
        "covariance_example": covariance_meta_example,
        "target_epsg": target_epsg,
        "outputs": outputs,
        "signature": signature,
        "signature_payload": signature_payload,
        "limitations": [
            (
                "Regularization reduces unsupported year-to-year oscillation "
                "but may attenuate abrupt real annual-rate changes."
            ),
            (
                "Reported joint standard errors are conditional on the "
                "selected regularization and do not include smoothing bias."
            ),
            (
                "The common seasonal coefficients are shared across years "
                "for each PS, not across space."
            ),
            (
                "Final yearly points require agreement with an independent "
                "de-seasonalized annual GLS estimate."
            ),
            (
                "This custom extension is not an official StaMPS or MintPy "
                "implementation."
            ),
        ],
        "duration_sec": time.perf_counter() - started,
    }
    report_path = output_root / "joint_regularized_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n============================================================")
    print("Joint regularized annual velocity V2 completed")
    print("============================================================")
    print(f"Curvature sigma : {selected_sigma:.3f} mm/yr")
    print(f"Model RMS cap   : {model_rmse_cap:.3f} mm")
    print(f"Output          : {output_root}")
    print(f"Diagnostic GPKG : {outputs.get('diagnostic_gpkg')}")
    print(f"Numeric NPZ     : {npz_path}")
    print(f"Report          : {report_path}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regularized joint annual trend + seasonal terms with SBAS "
            "covariance, Huber GLS and independent annual validation"
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--repo-root",
        default="/home/ubuntu/software/pystamps-main",
    )
    parser.add_argument("--out", default=None)

    parser.add_argument("--chunk-ps", type=int, default=2048)
    parser.add_argument(
        "--validation-chunk-ps",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--context-cache-size",
        type=int,
        default=16,
    )
    parser.add_argument("--minimum-dof", type=int, default=8)

    parser.add_argument(
        "--covariance-mode",
        choices=("network", "diagonal", "identity"),
        default="network",
    )
    parser.add_argument(
        "--eigen-floor-rel",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--sigma-floor-deg",
        type=float,
        default=0.1,
    )
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--irls-iterations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--max-regularized-condition",
        type=float,
        default=1.0e10,
    )
    parser.add_argument(
        "--max-independent-condition",
        type=float,
        default=1.0e10,
    )

    parser.add_argument(
        "--seasonal-harmonics",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    parser.add_argument(
        "--seasonal-period-days",
        type=float,
        default=365.2425,
    )

    parser.add_argument(
        "--curvature-sigma-mm-yr",
        type=float,
        default=None,
        help=(
            "固定年度速率二阶差分先验尺度；省略时用GCV自动选择"
        ),
    )
    parser.add_argument(
        "--gcv-curvature-candidates",
        default="4,6,8,12,18,25,40,60",
    )
    parser.add_argument(
        "--gcv-sample-ps",
        type=int,
        default=2048,
    )
    parser.add_argument("--gcv-seed", type=int, default=20260731)
    parser.add_argument(
        "--gcv-log-tolerance",
        type=float,
        default=0.03,
        help=(
            "在最小median log-GCV的容差内选择更弱的正则化；"
            "0.03约对应3%的GCV差异"
        ),
    )
    parser.add_argument(
        "--first-difference-sigma-mm-yr",
        type=float,
        default=None,
        help=(
            "可选的一阶差分弱约束；默认不使用，只约束二阶差分"
        ),
    )

    parser.add_argument(
        "--independent-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model-agreement-abs-mm-yr",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--model-agreement-sigma",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--min-year-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-year-span-days",
        type=float,
        default=240.0,
    )
    parser.add_argument(
        "--max-model-rmse-mm",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--min-model-rmse-cap-mm",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--max-rate-se-mm-yr",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--min-rate-se-cap-mm-yr",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--max-independent-se-mm-yr",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--min-independent-se-cap-mm-yr",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--threshold-mad-multiplier",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--absolute-rate-cap-mm-yr",
        type=float,
        default=150.0,
    )

    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument(
        "--write-diagnostic-gpkg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-final-shp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
