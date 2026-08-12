from __future__ import annotations

import numpy as np

import pystamps.pipeline.ported as ported
from pystamps.pipeline.stage6_sbas import (
    _stage6_goldstein_filter_dense_batch,
)


def test_stage6_grid_batch_matches_legacy_wrap_filt_global() -> None:
    rng = np.random.default_rng(20260727)
    n_i, n_j, n_ifg = 72, 88, 3

    stack = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
    occupied = rng.random((n_i, n_j)) < 0.08
    phases = rng.uniform(-np.pi, np.pi, size=(n_i, n_j, n_ifg))
    stack[occupied, :] = np.exp(1j * phases[occupied, :]).astype(np.complex64)

    actual, actual_low = _stage6_goldstein_filter_dense_batch(
        stack,
        n_win=32,
        alpha=0.8,
        gold_flag=True,
        fft_workers=1,
        window_batch=5,
    )

    for index in range(n_ifg):
        expected, expected_low = ported._wrap_filt_global(
            stack[:, :, index],
            n_win=32,
            alpha=0.8,
            low_flag="y",
        )
        np.testing.assert_allclose(
            actual[:, :, index],
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            actual_low[:, :, index],
            expected_low,
            rtol=2e-6,
            atol=2e-6,
        )
