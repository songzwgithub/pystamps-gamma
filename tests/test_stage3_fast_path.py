from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import (
    read_mat_variables,
    write_mat,
)
from pystamps.pipeline import ported


def test_read_mat_variables_reads_only_requested(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "sample.mat"
    )

    write_mat(
        path,
        {
            "small": np.arange(
                5,
                dtype=np.float64,
            ),
            "large": np.arange(
                100,
                dtype=np.float32,
            ).reshape(
                10,
                10,
            ),
        },
    )

    payload = read_mat_variables(
        path,
        (
            "small",
        ),
    )

    assert set(
        payload
    ) == {
        "small",
    }

    np.testing.assert_allclose(
        np.asarray(
            payload[
                "small"
            ]
        ).reshape(-1),
        np.arange(
            5,
            dtype=np.float64,
        ),
    )


def test_stage3_batched_clap_matches_scalar_double() -> None:
    rng = np.random.default_rng(
        42
    )

    stack = (
        rng.normal(
            size=(
                8,
                8,
                5,
            )
        )
        + 1j
        * rng.normal(
            size=(
                8,
                8,
                5,
            )
        )
    ).astype(
        np.complex128
    )

    low_pass = np.ones(
        (
            8,
            8,
        ),
        dtype=np.float64,
    ) * 0.2

    reference = np.empty_like(
        stack,
        dtype=np.complex128,
    )

    for i_ifg in range(
        stack.shape[2]
    ):
        reference[
            :,
            :,
            i_ifg,
        ] = ported._clap_filt_patch(
            stack[
                :,
                :,
                i_ifg,
            ],
            alpha=1.0,
            beta=0.3,
            low_pass=low_pass,
        )

    actual = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=False,
    )

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_stage3_batched_clap_single_close_to_double() -> None:
    rng = np.random.default_rng(
        7
    )

    stack = (
        rng.normal(
            size=(
                8,
                8,
                4,
            )
        )
        + 1j
        * rng.normal(
            size=(
                8,
                8,
                4,
            )
        )
    ).astype(
        np.complex64
    )

    low_pass = np.ones(
        (
            8,
            8,
        ),
        dtype=np.float64,
    ) * 0.2

    double = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=False,
    )

    single = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=True,
    )

    np.testing.assert_allclose(
        single,
        double,
        rtol=5.0e-5,
        atol=5.0e-5,
    )
