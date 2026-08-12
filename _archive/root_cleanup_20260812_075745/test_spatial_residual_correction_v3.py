#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "spatial_residual_correction_v3.py"
spec = importlib.util.spec_from_file_location("spatial_v3", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load spatial_residual_correction_v3.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> int:
    rng = np.random.default_rng(20260731)
    n = 30000
    x = rng.uniform(0.0, 100000.0, n)
    y = rng.uniform(0.0, 80000.0, n)
    lon = 117.0 + x / 90000.0
    lat = 38.0 + y / 111000.0
    grid = module.build_spatial_grid(x, y, 4000.0)

    true_field = (
        8.0 * np.sin(2.0 * np.pi * x / 90000.0)
        + 5.0 * np.cos(2.0 * np.pi * y / 70000.0)
        + 3.0 * (x / 100000.0 - 0.5)
    )
    compact_deformation = np.zeros(n)
    compact = (x - 55000.0) ** 2 + (y - 40000.0) ** 2 < 2500.0 ** 2
    compact_deformation[compact] = -18.0
    noise = rng.normal(0.0, 2.0, n)
    highpass = true_field + compact_deformation + noise
    control = np.ones(n, dtype=bool)
    reference = control & (x < 10000.0) & (y < 10000.0)

    correction, qa = module.estimate_epoch_spatial_field(
        highpass,
        control,
        grid,
        min_cell_points=20,
        spatial_sigma_cells=1.5,
        max_extrapolation_cells=4.0,
        residual_clip_sigma=4.0,
        cv_modulo=5,
        cv_offset=2,
        min_cv_cells=15,
        min_cv_improvement=0.05,
        max_correction_mm=30.0,
        reference_mask=reference,
    )
    if not qa.get("accepted", False):
        raise AssertionError(f"Synthetic field was rejected: {qa}")

    valid = ~compact
    before = np.sqrt(np.mean((highpass[valid] - np.median(highpass[valid])) ** 2))
    corrected = highpass - correction
    after = np.sqrt(np.mean((corrected[valid] - np.median(corrected[valid])) ** 2))
    if not after < 0.70 * before:
        raise AssertionError(
            f"Spatial correction insufficient: before={before}, after={after}"
        )

    # The compact anomaly must not be fully absorbed by the long-wavelength field.
    anomaly_before = float(np.median(highpass[compact]) - np.median(highpass[~compact]))
    anomaly_after = float(np.median(corrected[compact]) - np.median(corrected[~compact]))
    if abs(anomaly_after) < 0.55 * abs(anomaly_before):
        raise AssertionError(
            "Long-wavelength correction removed too much compact deformation: "
            f"before={anomaly_before}, after={anomaly_after}"
        )

    print("Synthetic spatial residual correction V3 test passed")
    print(f"QA improvement: {qa['cv_improvement_fraction']:.3f}")
    print(f"Background RMS: {before:.3f} -> {after:.3f} mm")
    print(f"Compact anomaly: {anomaly_before:.3f} -> {anomaly_after:.3f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
