#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

import numpy as np

from pystamps.io.mat import read_mat
from pystamps.pipeline import ported


def max_abs_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_array = np.asarray(
        left
    )

    right_array = np.asarray(
        right
    )

    if left_array.shape != right_array.shape:
        return float(
            "inf"
        )

    if (
        left_array.size
        == 0
    ):
        return 0.0

    difference = np.abs(
        left_array
        - right_array
    )

    finite = np.isfinite(
        difference
    )

    if not np.any(
        finite
    ):
        return 0.0

    return float(
        np.max(
            difference[
                finite
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "patch",
        type=Path,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--single",
        action="store_true",
    )

    args = parser.parse_args()

    source = args.patch.expanduser().resolve()

    legacy_file = (
        source
        / "select1.mat"
    )

    if not legacy_file.exists():
        raise SystemExit(
            f"缺少原Stage 3结果：{legacy_file}"
        )

    validation_root = (
        source.parent
        / "_stage3_fast_validation"
    )

    validation_patch = (
        validation_root
        / source.name
    )

    if validation_patch.exists():
        shutil.rmtree(
            validation_patch
        )

    validation_patch.mkdir(
        parents=True
    )

    for item in source.iterdir():
        if not item.is_file():
            continue

        if item.name in {
            "select1.mat",
            "stage3_debug.json",
        }:
            continue

        os.symlink(
            item.resolve(),
            validation_patch
            / item.name,
        )

    os.environ[
        "PYSTAMPS_STAGE3_FAST"
    ] = "1"

    os.environ[
        "PYSTAMPS_STAGE3_THREADS"
    ] = str(
        max(
            1,
            args.threads,
        )
    )

    os.environ[
        "PYSTAMPS_STAGE3_SINGLE_PRECISION"
    ] = (
        "1"
        if args.single
        else "0"
    )

    os.environ[
        "PYSTAMPS_STAGE3_PROGRESS"
    ] = "1"

    started = time.perf_counter()

    result = ported.stage3_select_ps(
        validation_patch,
        backend="auto",
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        result
    )

    print(
        f"快速Stage 3耗时：{elapsed:.2f}秒"
    )

    legacy = read_mat(
        legacy_file
    )

    fast = read_mat(
        validation_patch
        / "select1.mat"
    )

    ix_legacy = np.asarray(
        legacy.get(
            "ix"
        ),
        dtype=np.int64,
    ).reshape(-1)

    ix_fast = np.asarray(
        fast.get(
            "ix"
        ),
        dtype=np.int64,
    ).reshape(-1)

    keep_legacy = np.asarray(
        legacy.get(
            "keep_ix"
        ),
        dtype=bool,
    ).reshape(-1)

    keep_fast = np.asarray(
        fast.get(
            "keep_ix"
        ),
        dtype=bool,
    ).reshape(-1)

    print()
    print(
        "================ 验证结果 ================"
    )

    print(
        f"原始ix数量：{ix_legacy.size}"
    )

    print(
        f"快速ix数量：{ix_fast.size}"
    )

    print(
        "ix完全一致：",
        bool(
            np.array_equal(
                ix_legacy,
                ix_fast,
            )
        ),
    )

    print(
        "keep_ix完全一致：",
        bool(
            np.array_equal(
                keep_legacy,
                keep_fast,
            )
        ),
    )

    for key in (
        "K_ps2",
        "C_ps2",
        "coh_ps2",
        "coh_thresh",
        "ph_patch2",
        "ph_res2",
    ):
        if (
            key not in legacy
            or key not in fast
        ):
            print(
                f"{key}: 缺失"
            )

            continue

        difference = max_abs_difference(
            np.asarray(
                legacy[
                    key
                ]
            ),
            np.asarray(
                fast[
                    key
                ]
            ),
        )

        print(
            f"{key}最大绝对差异："
            f"{difference:.12g}"
        )

    print(
        "验证目录：",
        validation_patch,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
