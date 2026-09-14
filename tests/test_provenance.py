from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.config import IFGSelectionConfig, RunConfig, RuntimeConfig
from pystamps.final_ifg_qc import (
    build_final_qc_provenance,
    final_qc_provenance_is_current,
)
from pystamps.pipeline.provenance import (
    build_stage_signature,
    stage_marker_is_current,
    write_stage_marker,
)


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_production_safety_defaults() -> None:
    runtime = RuntimeConfig()
    ifg = IFGSelectionConfig()

    assert runtime.validate_stage_provenance is True
    assert ifg.final_ifg_qc_enabled is True
    assert ifg.final_qc_require_signature is True


def test_stage_provenance_invalidates_on_input_change(tmp_path: Path) -> None:
    root = tmp_path / "work"
    patch = root / "PATCH_1"
    patch.mkdir(parents=True)

    for name in (
        "ps1.mat",
        "ph1.mat",
        "bp1.mat",
        "da1.mat",
        "hgt1.mat",
        "la1.mat",
        "inc1.mat",
    ):
        _touch(patch / name, name.encode("utf-8"))

    _touch(root / "parms.mat", b"parms")

    cfg = RunConfig()

    signature = build_stage_signature(
        dataset_root=root,
        target_dir=patch,
        stage_id=2,
        scope="patch",
        run_config=cfg,
    )

    write_stage_marker(
        patch,
        2,
        "patch",
        signature,
    )

    assert stage_marker_is_current(
        patch,
        2,
        "patch",
        signature,
    )

    _touch(patch / "ph1.mat", b"changed-phase-input")

    changed = build_stage_signature(
        dataset_root=root,
        target_dir=patch,
        stage_id=2,
        scope="patch",
        run_config=cfg,
    )

    assert not stage_marker_is_current(
        patch,
        2,
        "patch",
        changed,
    )


def test_final_qc_provenance_rejects_network_change(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()

    edges = np.asarray(
        [
            [1, 2],
            [2, 3],
            [1, 3],
        ],
        dtype=np.int64,
    )
    day = np.asarray(
        [738000.0, 738012.0, 738024.0],
        dtype=np.float64,
    )

    settings = {
        "enabled": True,
        "msd_strong_percentile": 0.975,
        "msd_extreme_percentile": 0.990,
        "network_strong_percentile": 0.975,
        "network_extreme_percentile": 0.990,
        "max_drop_fraction": 0.05,
        "preserve_network": True,
        "fail_on_cap": True,
        "chunk_ifg": 8,
        "require_signature": True,
    }

    provenance = build_final_qc_provenance(
        root,
        edges,
        settings,
        day=day,
    )
    payload = {
        "status": "ok",
        "provenance": provenance,
    }

    assert final_qc_provenance_is_current(
        root,
        payload,
        edges,
        settings,
        day=day,
    )

    changed_edges = edges.copy()
    changed_edges[0, :] = (1, 3)

    assert not final_qc_provenance_is_current(
        root,
        payload,
        changed_edges,
        settings,
        day=day,
    )


def test_final_qc_missing_signature_is_rejected_by_default(
    tmp_path: Path,
) -> None:
    settings = {
        "require_signature": True,
    }

    assert not final_qc_provenance_is_current(
        tmp_path,
        {"status": "ok"},
        np.asarray([[1, 2]], dtype=np.int64),
        settings,
        day=np.asarray([738000.0, 738012.0]),
    )
