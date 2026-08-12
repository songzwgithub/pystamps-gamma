#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "calc_joint_regularized_seasonal_gls_v2.py"
spec = importlib.util.spec_from_file_location("joint_v2", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def main() -> int:
    rng = np.random.default_rng(42)
    dates = []
    for year in range(2017, 2027):
        start = datetime(year, 1, 8)
        for k in range(18):
            dates.append(start + timedelta(days=20 * k))

    design = mod.build_design(
        dates,
        seasonal_harmonics=1,
        seasonal_period_days=365.2425,
    )

    wavelength = 0.05546576
    phase_to_velocity = -wavelength / (4.0 * math.pi) * 1000.0
    phase_to_velocity_abs = abs(phase_to_velocity)

    true_velocity = np.asarray(
        [-3, -5, -8, -10, -12, -13, -14, -13, -11, -9],
        dtype=np.float64,
    )
    beta = np.zeros(design.matrix.shape[1], dtype=np.float64)
    beta[design.year_column_indices] = true_velocity / phase_to_velocity
    beta[design.annual_sin_index] = 1.2
    beta[design.annual_cos_index] = -0.7

    n_ps = 256
    clean = design.matrix @ beta
    sigma_phase = 1.7
    observations = clean[None, :] + rng.normal(
        0.0,
        sigma_phase,
        size=(n_ps, len(dates)),
    )
    for row in range(0, n_ps, 17):
        observations[row, 60] += 10.0
        observations[row, 140] -= 8.0

    covariance = np.eye(len(dates), dtype=np.float64) * sigma_phase**2
    whitener = np.eye(len(dates), dtype=np.float64) / sigma_phase
    context = mod.WhitenedContext(
        mask=np.ones(len(dates), dtype=bool),
        X=design.matrix,
        Xw=whitener @ design.matrix,
        whitener=whitener,
        covariance_meta={"test": True},
    )

    weak = mod.build_penalty(
        design,
        phase_to_velocity_abs=phase_to_velocity_abs,
        curvature_sigma_mm_yr=1.0e7,
        first_difference_sigma_mm_yr=None,
    )
    regularized = mod.build_penalty(
        design,
        phase_to_velocity_abs=phase_to_velocity_abs,
        curvature_sigma_mm_yr=12.0,
        first_difference_sigma_mm_yr=None,
    )

    result_weak = mod.robust_penalized_gls_batch(
        observations,
        context,
        weak,
        huber_c=1.345,
        weight_floor=0.05,
        max_iterations=8,
        convergence=1.0e-6,
        max_condition=1.0e12,
    )
    result_reg = mod.robust_penalized_gls_batch(
        observations,
        context,
        regularized,
        huber_c=1.345,
        weight_floor=0.05,
        max_iterations=8,
        convergence=1.0e-6,
        max_condition=1.0e12,
    )

    velocity_weak = (
        result_weak["beta"][:, design.year_column_indices]
        * phase_to_velocity
    )
    velocity_reg = (
        result_reg["beta"][:, design.year_column_indices]
        * phase_to_velocity
    )

    error_weak = float(
        np.sqrt(np.mean((velocity_weak - true_velocity[None, :]) ** 2))
    )
    error_reg = float(
        np.sqrt(np.mean((velocity_reg - true_velocity[None, :]) ** 2))
    )
    curvature_weak = float(
        np.sqrt(np.mean(np.diff(velocity_weak, n=2, axis=1) ** 2))
    )
    curvature_reg = float(
        np.sqrt(np.mean(np.diff(velocity_reg, n=2, axis=1) ** 2))
    )

    if not error_reg < error_weak:
        raise AssertionError(
            f"regularization did not improve rate RMSE: {error_reg} >= {error_weak}"
        )
    if not curvature_reg < curvature_weak:
        raise AssertionError(
            "regularization did not reduce unsupported annual oscillation"
        )

    inputs = SimpleNamespace(
        ph_ts=observations,
        covariance=covariance,
    )

    class Support:
        @staticmethod
        def covariance_whitener(C, covariance_mode, eigen_floor_rel):
            eigval, eigvec = np.linalg.eigh(C)
            floor = max(float(np.max(eigval)) * eigen_floor_rel, 1.0e-12)
            eigval = np.maximum(eigval, floor)
            W = (eigvec / np.sqrt(eigval)[None, :]) @ eigvec.T
            return W, {"mode": covariance_mode}

    selected, table = mod.select_curvature_sigma_gcv(
        inputs,
        design,
        Support,
        phase_to_velocity_abs=phase_to_velocity_abs,
        candidates=[4.0, 8.0, 12.0, 20.0, 40.0],
        first_difference_sigma_mm_yr=None,
        covariance_mode="network",
        eigen_floor_rel=1.0e-6,
        sample_size=128,
        seed=7,
        minimum_dof=8,
        log_tolerance=0.03,
    )
    if selected not in {4.0, 8.0, 12.0, 20.0, 40.0}:
        raise AssertionError(f"unexpected GCV selection: {selected}")
    if len(table) != 5:
        raise AssertionError("GCV table length mismatch")

    print("Synthetic regularized joint model test passed")
    print(f"Unregularized rate RMSE: {error_weak:.3f} mm/yr")
    print(f"Regularized rate RMSE  : {error_reg:.3f} mm/yr")
    print(f"Unregularized curvature: {curvature_weak:.3f} mm/yr")
    print(f"Regularized curvature  : {curvature_reg:.3f} mm/yr")
    print(f"GCV selected sigma     : {selected:.1f} mm/yr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
