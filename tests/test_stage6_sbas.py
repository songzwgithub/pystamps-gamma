from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import write_mat
from pystamps.pipeline.stage6_sbas import (
    _active_network,
    compute_sbas_space_time,
    load_sbas_network,
)


def test_active_network_builds_full_rank_connected_graph() -> None:
    day = np.asarray([0.0, 12.0, 24.0, 36.0], dtype=np.float64)
    ifgday_ix = np.asarray(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 3],
            [2, 4],
        ],
        dtype=np.int64,
    )

    G, day_active, ifg_active = _active_network(day, ifgday_ix)

    assert G.shape == (5, 4)
    assert day_active.shape == (4,)
    assert ifg_active.shape == (5, 2)
    assert np.linalg.matrix_rank(G) == 3
    np.testing.assert_allclose(np.sum(G, axis=1), 0.0)


def test_load_sbas_network_falls_back_to_patch_ps1(tmp_path: Path) -> None:
    root = tmp_path
    patch = root / "PATCH_1"
    patch.mkdir()

    write_mat(
        root / "ps2.mat",
        {
            "n_ps": np.asarray(2.0),
            "n_ifg": np.asarray(3.0),
            "bperp": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
        },
    )
    write_mat(
        patch / "ps1.mat",
        {
            "ifgday_ix": np.asarray([[1, 2], [2, 3], [1, 3]], dtype=np.float64),
            "day": np.asarray([[0.0], [12.0], [24.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
            "n_ifg": np.asarray(3.0),
            "n_image": np.asarray(3.0),
        },
    )

    day, ifgday_ix, bperp, source = load_sbas_network(root, 3)

    assert source == patch / "ps1.mat"
    np.testing.assert_array_equal(ifgday_ix, np.asarray([[1, 2], [2, 3], [1, 3]]))
    np.testing.assert_allclose(day, np.asarray([0.0, 12.0, 24.0]))
    np.testing.assert_allclose(bperp, np.asarray([10.0, 20.0, 30.0]))


def test_sbas_time_space_preserves_wrapped_arc_phase(tmp_path: Path) -> None:
    day = np.asarray([0.0, 12.0, 24.0, 36.0], dtype=np.float64)
    ifgday_ix = np.asarray(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 3],
            [2, 4],
        ],
        dtype=np.int64,
    )
    image_phase = np.asarray([0.0, 0.15, 0.35, 0.55], dtype=np.float64)
    true_ifg = image_phase[ifgday_ix[:, 1] - 1] - image_phase[ifgday_ix[:, 0] - 1]

    uw_ph = np.vstack(
        (
            np.ones(true_ifg.size, dtype=np.complex64),
            np.exp(1j * true_ifg).astype(np.complex64),
        )
    )
    edgs = np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64)

    _G, noise, dph_uw, meta = compute_sbas_space_time(
        uw_ph=uw_ph,
        ph_lowpass=uw_ph.copy(),
        edgs=edgs,
        day=day,
        ifgday_ix=ifgday_ix,
        bperp=np.asarray([10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64),
        time_win=36.0,
        n_trial_wraps=0.0,
        unwrap_method="3D_QUICK",
        la_flag=False,
        edge_chunk=1,
        anneal_workers=1,
        anneal_runs=1,
        strict_anneal=False,
        progress=False,
        work_dir=tmp_path / "work",
    )

    np.testing.assert_allclose(
        np.exp(1j * np.asarray(dph_uw[0], dtype=np.float64)),
        np.exp(1j * true_ifg),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(np.asarray(noise)).all()
    assert meta["rank_G"] == 3
