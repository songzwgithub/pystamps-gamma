#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np

from pystamps.io.mat import read_mat


FAST = Path(
    "/mnt/vol-gdc28n1r/insar/cangzhou_P69/"
    "pystamps_sbas_smoke_fast"
)

DOUBLE = Path(
    "/mnt/vol-gdc28n1r/insar/cangzhou_P69/"
    "pystamps_sbas_smoke_batched_double"
)


def compare_array(
    name: str,
    fast: np.ndarray,
    reference: np.ndarray,
) -> None:
    fast = np.asarray(fast)
    reference = np.asarray(reference)

    print()
    print(name)
    print("  shape fast :", fast.shape)
    print("  shape ref  :", reference.shape)

    if fast.shape != reference.shape:
        print("  RESULT     : SHAPE MISMATCH")
        return

    if np.iscomplexobj(fast) or np.iscomplexobj(reference):
        difference = np.abs(
            fast.astype(np.complex128)
            - reference.astype(np.complex128)
        )
        reference_scale = np.abs(
            reference.astype(np.complex128)
        )
    else:
        difference = np.abs(
            fast.astype(np.float64)
            - reference.astype(np.float64)
        )
        reference_scale = np.abs(
            reference.astype(np.float64)
        )

    finite = (
        np.isfinite(difference)
        & np.isfinite(reference_scale)
    )

    if not np.any(finite):
        print("  finite count: 0")
        return

    difference = difference[finite]
    reference_scale = reference_scale[finite]

    relative = difference / np.maximum(
        reference_scale,
        1.0e-12,
    )

    print(
        "  max abs     :",
        float(np.max(difference)),
    )
    print(
        "  median abs  :",
        float(np.median(difference)),
    )
    print(
        "  p99 abs     :",
        float(np.quantile(difference, 0.99)),
    )
    print(
        "  max relative:",
        float(np.max(relative)),
    )
    print(
        "  p99 relative:",
        float(np.quantile(relative, 0.99)),
    )


fast_patches = {
    path.parent.name: path
    for path in FAST.glob("PATCH_*/pm1.mat")
}

double_patches = {
    path.parent.name: path
    for path in DOUBLE.glob("PATCH_*/pm1.mat")
}

common = sorted(
    set(fast_patches)
    & set(double_patches)
)

if not common:
    raise RuntimeError(
        "FAST和DOUBLE数据集没有共同patch"
    )

keys = [
    "K_ps",
    "C_ps",
    "coh_ps",
    "ph_patch",
    "ph_res",
    "ph_weight",
    "Nr",
    "coh_bins",
]

for patch_name in common:
    print()
    print("=" * 72)
    print("PATCH:", patch_name)
    print("=" * 72)

    fast_data = read_mat(
        fast_patches[patch_name]
    )

    double_data = read_mat(
        double_patches[patch_name]
    )

    for key in keys:
        if key not in fast_data:
            print(f"\n{key}: FAST缺失")
            continue

        if key not in double_data:
            print(f"\n{key}: DOUBLE缺失")
            continue

        compare_array(
            key,
            fast_data[key],
            double_data[key],
        )
