from __future__ import annotations

from pathlib import Path

import pytest

from pystamps.prep.gamma_sbas import (
    GammaInputError,
    inspect_gamma_sbas_project,
    load_gamma_sbas_project,
    network_is_connected,
    parse_itab,
    parse_rslc_tab,
)


def _write_rslc_par(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "range_samples: 100",
                "azimuth_lines: 200",
                "image_format: FCOMPLEX",
            ]
        ),
        encoding="utf-8",
    )


def _write_off(path: Path, width: int, length: int) -> None:
    path.write_text(
        "\n".join(
            [
                f"interferogram_width: {width}",
                f"interferogram_azimuth_lines: {length}",
            ]
        ),
        encoding="utf-8",
    )


def _create_mock_project(
    tmp_path: Path,
    *,
    connected: bool = True,
) -> Path:
    root = tmp_path / "gamma_project"
    rslc_dir = root / "RSLC"
    diff_dir = root / "DIFF"
    mli_dir = root / "MLI_dir"
    dem_dir = root / "DEM_prep"

    for directory in (
        rslc_dir,
        diff_dir,
        mli_dir,
        dem_dir,
    ):
        directory.mkdir(parents=True)

    dates = [
        "20210101",
        "20210113",
        "20210125",
    ]

    rslc_lines: list[str] = []

    for date in dates:
        rslc = rslc_dir / f"{date}.rslc"
        par = rslc_dir / f"{date}.rslc.par"
        mli = mli_dir / f"{date}.mli"
        mli_par = mli_dir / f"{date}.mli.par"

        rslc.write_bytes(b"\x00")
        _write_rslc_par(par)
        mli.write_bytes(b"\x00" * (10 * 20 * 4))
        _write_off(mli_par, 10, 20)

        rslc_lines.append(f"{rslc} {par}")

    (root / "RSLC_tab").write_text(
        "\n".join(rslc_lines) + "\n",
        encoding="utf-8",
    )

    if connected:
        pairs = [(1, 2), (2, 3)]
    else:
        pairs = [(1, 2)]

    (root / "itab").write_text(
        "\n".join(
            f"{master} {slave} 1"
            for master, slave in pairs
        )
        + "\n",
        encoding="utf-8",
    )

    for master_index, slave_index in pairs:
        master_date = dates[master_index - 1]
        slave_date = dates[slave_index - 1]
        pair = f"{master_date}_{slave_date}"

        # 20 x 10 pixels x 8 bytes per FCOMPLEX pixel
        (diff_dir / f"{pair}.diff").write_bytes(
            b"\x00" * (20 * 10 * 8)
        )
        (diff_dir / f"{pair}.base").write_text(
            "\n".join(
                [
                    "initial_baseline(TCN): 0.0 10.0 20.0 m",
                    "initial_baseline_rate: 0.0 0.1 0.2 m/s",
                ]
            ),
            encoding="utf-8",
        )
        _write_off(
            diff_dir / f"{pair}.off",
            10,
            20,
        )

    return root


def test_parse_rslc_tab(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    records = parse_rslc_tab(root / "RSLC_tab")

    assert len(records) == 3
    assert records[0][2] == "20210101"
    assert records[-1][2] == "20210125"


def test_parse_itab(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    pairs = parse_itab(
        root / "itab",
        acquisition_count=3,
    )

    assert pairs == [(1, 2), (2, 3)]


def test_network_connectivity() -> None:
    assert network_is_connected(
        3,
        [(1, 2), (2, 3)],
    )

    assert not network_is_connected(
        3,
        [(1, 2)],
    )


def test_load_gamma_sbas_project(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    project = load_gamma_sbas_project(root)

    assert len(project.acquisitions) == 3
    assert len(project.interferograms) == 2
    assert project.width == 10
    assert project.length == 20
    assert project.network_connected


def test_inspection_report(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    report = inspect_gamma_sbas_project(root)

    assert report["acquisition_count"] == 3
    assert report["interferogram_count"] == 2
    assert report["network_connected"] is True
    assert report["missing_mli_count"] == 0


def test_disconnected_network_rejected(tmp_path: Path) -> None:
    root = _create_mock_project(
        tmp_path,
        connected=False,
    )

    with pytest.raises(
        GammaInputError,
        match="网络不连通",
    ):
        load_gamma_sbas_project(root)


def test_missing_base_rejected(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    base_file = (
        root
        / "DIFF"
        / "20210101_20210113.base"
    )
    base_file.unlink()

    with pytest.raises(
        GammaInputError,
        match="基线文件",
    ):
        load_gamma_sbas_project(root)


def test_wrong_diff_size_rejected(tmp_path: Path) -> None:
    root = _create_mock_project(tmp_path)

    diff_file = (
        root
        / "DIFF"
        / "20210101_20210113.diff"
    )
    diff_file.write_bytes(b"\x00" * 12)

    with pytest.raises(
        GammaInputError,
        match="FCOMPLEX尺寸",
    ):
        load_gamma_sbas_project(root)
