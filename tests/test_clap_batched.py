from __future__ import annotations

import numpy as np

from pystamps.pipeline import ported


def _make_test_stack(
    seed: int = 20260726,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    phase = rng.uniform(
        -np.pi,
        np.pi,
        size=(
            48,
            56,
            7,
        ),
    )

    amplitude = rng.uniform(
        0.5,
        1.5,
        size=phase.shape,
    )

    stack = (
        amplitude
        * np.exp(
            1j * phase
        )
    ).astype(
        np.complex64
    )

    # Simulate sparse PS-grid occupancy.
    occupancy = (
        rng.random(
            size=(
                48,
                56,
                1,
            )
        )
        > 0.72
    )

    stack *= occupancy

    # Confirm NaN-to-zero behaviour.
    stack[
        10,
        12,
        2,
    ] = np.nan + 1j * np.nan

    return stack


def _make_low_pass() -> np.ndarray:
    axis = np.linspace(
        -1.0,
        1.0,
        32,
        dtype=np.float64,
    )

    x_grid, y_grid = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    return np.exp(
        -(
            x_grid**2
            + y_grid**2
        )
        / 0.20
    ).astype(
        np.float64
    )


def _reference_stack(
    stack: np.ndarray,
    *,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int,
    low_pass: np.ndarray,
    preserve_precision: bool,
) -> np.ndarray:
    outputs = []

    for interferogram_index in range(
        stack.shape[2]
    ):
        outputs.append(
            ported._clap_filt_grid(
                stack[
                    :,
                    :,
                    interferogram_index,
                ],
                alpha=alpha,
                beta=beta,
                n_win=n_win,
                n_pad=n_pad,
                low_pass=low_pass,
                preserve_precision=(
                    preserve_precision
                ),
            )
        )

    return np.stack(
        outputs,
        axis=2,
    )


def test_batched_clap_matches_reference_double(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "0",
    )

    stack = _make_test_stack()
    low_pass = _make_low_pass()

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    expected = _reference_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        n_win=24,
        n_pad=8,
        low_pass=low_pass,
        preserve_precision=True,
    )

    observed = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=2,
            preserve_precision=True,
        )
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_batched_clap_single_precision_close_to_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "1",
    )

    stack = _make_test_stack()
    low_pass = _make_low_pass()

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    expected = _reference_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        n_win=24,
        n_pad=8,
        low_pass=low_pass,
        preserve_precision=False,
    )

    observed = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=2,
            preserve_precision=False,
        )
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=5.0e-4,
        atol=5.0e-4,
    )


def test_batched_clap_writes_output_buffer(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "0",
    )

    stack = _make_test_stack()
    low_pass = _make_low_pass()

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    output = np.empty(
        stack.shape,
        dtype=np.complex64,
    )

    returned = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            out=output,
            workers=2,
            preserve_precision=False,
        )
    )

    assert returned is output
    assert np.isfinite(
        output
    ).all()


def test_batched_clap_zero_stack_is_zero(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "0",
    )

    stack = np.zeros(
        (
            48,
            48,
            5,
        ),
        dtype=np.complex64,
    )

    low_pass = np.zeros(
        (
            32,
            32,
        ),
        dtype=np.float64,
    )

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    observed = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=2,
        )
    )

    np.testing.assert_array_equal(
        observed,
        np.zeros_like(
            observed
        ),
    )

def test_clap_window_batch_size_does_not_change_result(
    monkeypatch,
) -> None:
    stack = _make_test_stack()
    low_pass = _make_low_pass()

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "1",
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_FFT_WORKERS",
        "2",
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_WINDOW_BATCH",
        "1",
    )

    batch_one = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=2,
            preserve_precision=False,
        )
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_WINDOW_BATCH",
        "4",
    )

    batch_four = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=2,
            preserve_precision=False,
        )
    )

    np.testing.assert_allclose(
        batch_four,
        batch_one,
        rtol=5.0e-6,
        atol=5.0e-6,
    )


def test_clap_active_window_integral_detection() -> None:
    stack = np.zeros(
        (
            48,
            56,
            3,
        ),
        dtype=np.complex64,
    )

    stack[
        20,
        25,
        1,
    ] = 1.0 + 0.0j

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=np.zeros(
                (
                    32,
                    32,
                ),
                dtype=np.float64,
            ),
        )
    )

    active = (
        ported
        ._clap_active_window_indices(
            stack,
            prepared.windows,
        )
    )

    assert active.ndim == 1
    assert active.size > 0
    assert active.size < len(
        prepared.windows
    )

    for index in active:
        window = prepared.windows[
            int(index)
        ]

        assert np.any(
            stack[
                window.i1:
                window.i2,
                window.j1:
                window.j2,
                :,
            ]
            != 0
        )

def test_clap_ifg_parallel_matches_serial(
    monkeypatch,
) -> None:
    stack = _make_test_stack()
    low_pass = _make_low_pass()

    prepared = (
        ported
        ._prepare_clap_filt_grid_stack(
            stack.shape,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_SINGLE_PRECISION",
        "1",
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_WINDOW_BATCH",
        "4",
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_FFT_WORKERS",
        "1",
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_IFG_WORKERS",
        "1",
    )

    serial = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=1,
        )
    )

    monkeypatch.setenv(
        "PYSTAMPS_CLAP_IFG_WORKERS",
        "4",
    )

    parallel = (
        ported
        ._clap_filt_grid_stack_prepared(
            stack,
            alpha=1.0,
            beta=0.3,
            prepared=prepared,
            workers=4,
        )
    )

    np.testing.assert_allclose(
        parallel,
        serial,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
