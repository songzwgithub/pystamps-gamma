#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat


REQUIRED_PATCH_FILES = (
    "ps1.mat",
    "ph1.mat",
    "bp1.mat",
    "da1.mat",
    "hgt1.mat",
    "la1.mat",
    "psver.mat",
)


def scalar_int(value: object, name: str) -> int:
    array = np.asarray(value).reshape(-1)

    if array.size != 1:
        raise RuntimeError(
            f"{name}应为标量，实际形状为"
            f"{np.asarray(value).shape}"
        )

    return int(round(float(array[0])))


def require_matrix(
    value: object,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)

    if array.shape != shape:
        raise RuntimeError(
            f"{name}形状错误："
            f"实际{array.shape}，期望{shape}"
        )

    return array


def check_patch(
    patch: Path,
) -> dict[str, object]:
    for filename in REQUIRED_PATCH_FILES:
        path = patch / filename

        if not path.is_file():
            raise RuntimeError(
                f"{patch.name}缺少{filename}"
            )

    ps = read_mat(patch / "ps1.mat")
    ph_payload = read_mat(
        patch / "ph1.mat"
    )
    bp_payload = read_mat(
        patch / "bp1.mat"
    )
    da_payload = read_mat(
        patch / "da1.mat"
    )
    hgt_payload = read_mat(
        patch / "hgt1.mat"
    )
    la_payload = read_mat(
        patch / "la1.mat"
    )

    n_ps = scalar_int(
        ps["n_ps"],
        "n_ps",
    )

    n_ifg = scalar_int(
        ps["n_ifg"],
        "n_ifg",
    )

    n_image = scalar_int(
        ps["n_image"],
        "n_image",
    )

    ij = require_matrix(
        ps["ij"],
        (n_ps, 3),
        "ij",
    )

    require_matrix(
        ps["lonlat"],
        (n_ps, 2),
        "lonlat",
    )

    require_matrix(
        ps["xy"],
        (n_ps, 3),
        "xy",
    )

    require_matrix(
        ps["ifgday"],
        (n_ifg, 2),
        "ifgday",
    )

    ifgday_ix = require_matrix(
        ps["ifgday_ix"],
        (n_ifg, 2),
        "ifgday_ix",
    )

    day = np.asarray(
        ps["day"]
    ).reshape(-1)

    if day.size != n_image:
        raise RuntimeError(
            f"day长度为{day.size}，"
            f"但n_image={n_image}"
        )

    if np.any(ifgday_ix < 1):
        raise RuntimeError(
            "ifgday_ix中存在小于1的索引"
        )

    if np.any(ifgday_ix > n_image):
        raise RuntimeError(
            "ifgday_ix中存在超过n_image的索引"
        )

    ph = require_matrix(
        ph_payload["ph"],
        (n_ps, n_ifg),
        "ph",
    )

    bperp_mat = require_matrix(
        bp_payload["bperp_mat"],
        (n_ps, n_ifg),
        "bperp_mat",
    )

    da = np.asarray(
        da_payload["D_A"]
    ).reshape(-1)

    hgt = np.asarray(
        hgt_payload["hgt"]
    ).reshape(-1)

    la = np.asarray(
        la_payload["la"]
    ).reshape(-1)

    for name, array in (
        ("D_A", da),
        ("hgt", hgt),
        ("la", la),
    ):
        if array.size != n_ps:
            raise RuntimeError(
                f"{name}长度为{array.size}，"
                f"但n_ps={n_ps}"
            )

    if not np.iscomplexobj(ph):
        raise RuntimeError(
            "ph不是复数矩阵"
        )

    if not np.all(
        np.isfinite(bperp_mat)
    ):
        raise RuntimeError(
            "bperp_mat包含NaN或Inf"
        )

    valid_phase = (
        np.isfinite(ph.real)
        & np.isfinite(ph.imag)
        & (np.abs(ph) > 0)
    )

    valid_fraction = float(
        np.mean(valid_phase)
    )

    if valid_fraction < 0.90:
        raise RuntimeError(
            f"有效相位比例过低："
            f"{valid_fraction:.4f}"
        )

    if np.unique(
        ij[:, 0]
    ).size != n_ps:
        raise RuntimeError(
            "候选点ID不唯一"
        )

    if not np.array_equal(
        ij[:, 0],
        np.arange(1, n_ps + 1),
    ):
        raise RuntimeError(
            "候选点ID必须连续为1..n_ps"
        )

    return {
        "patch": patch.name,
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "phase_valid_fraction": (
            valid_fraction
        ),
        "bperp_min": float(
            np.min(bperp_mat)
        ),
        "bperp_max": float(
            np.max(bperp_mat)
        ),
        "da_median": float(
            np.median(da)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=Path,
    )

    args = parser.parse_args()

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    patch_list = (
        dataset
        / "patch.list"
    )

    if not patch_list.is_file():
        raise RuntimeError(
            f"缺少patch.list：{patch_list}"
        )

    patch_names = [
        line.strip()
        for line in patch_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not patch_names:
        raise RuntimeError(
            "patch.list为空"
        )

    print(
        f"dataset: {dataset}"
    )

    print(
        f"patch count: {len(patch_names)}"
    )

    reference_n_ifg = None
    reference_n_image = None
    total_ps = 0

    for patch_name in patch_names:
        report = check_patch(
            dataset / patch_name
        )

        if reference_n_ifg is None:
            reference_n_ifg = (
                report["n_ifg"]
            )
            reference_n_image = (
                report["n_image"]
            )
        else:
            if (
                report["n_ifg"]
                != reference_n_ifg
            ):
                raise RuntimeError(
                    f"{patch_name}的n_ifg"
                    "与其他patch不一致"
                )

            if (
                report["n_image"]
                != reference_n_image
            ):
                raise RuntimeError(
                    f"{patch_name}的n_image"
                    "与其他patch不一致"
                )

        total_ps += int(
            report["n_ps"]
        )

        print(
            f"{patch_name}: "
            f"n_ps={report['n_ps']}, "
            f"phase_valid="
            f"{report['phase_valid_fraction']:.4f}, "
            f"Bperp="
            f"[{report['bperp_min']:.3f}, "
            f"{report['bperp_max']:.3f}], "
            f"DA_median="
            f"{report['da_median']:.4f}"
        )

    print()
    print("Stage-1 interface check: PASSED")
    print(f"total patch PS records: {total_ps}")
    print(f"n_image: {reference_n_image}")
    print(f"n_ifg: {reference_n_ifg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
