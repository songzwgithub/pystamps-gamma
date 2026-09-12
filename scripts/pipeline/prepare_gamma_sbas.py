#!/usr/bin/env python3
from __future__ import annotations

# === STAGE1_AUTO_PREP_V1 ===

import argparse
import json
import os
from pathlib import Path

from pystamps.config import load_config
from pystamps.prep.gamma_candidates import CandidateConfig
from pystamps.prep.gamma_stage1 import GammaStage1Config, prepare_gamma_sbas_stage1
from pystamps.project_paths import ProjectPathError, export_project_paths, resolve_project_paths

_LOCAL_CONFIG_CANDIDATES = (
    "pystamps.yaml",
    "pystamps.yml",
    "production.yaml",
    "production.yml",
)


def _find_local_config(work_dir: Path) -> Path | None:
    for name in _LOCAL_CONFIG_CANDIDATES:
        p = work_dir / name
        if p.is_file():
            return p.resolve()
    return None


def _resolve_optional_path(value: Path | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    p = value.expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a GAMMA SBAS project for pySTAMPS-GAMMA Stage 1. "
            "Default: cwd=work_dir and cwd.parent=GAMMA data_dir."
        )
    )
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--reference-date", default=None)
    ap.add_argument("--range-looks", type=int, default=None)
    ap.add_argument("--azimuth-looks", type=int, default=None)
    ap.add_argument("--da-threshold", type=float, default=0.60)
    ap.add_argument("--min-valid-fraction", type=float, default=0.90)
    ap.add_argument("--candidate-source", choices=("rslc_sbas", "mli"), default="rslc_sbas")

    g = ap.add_argument_group("optional explicit geometry inputs")
    g.add_argument("--longitude-file", type=Path, default=None)
    g.add_argument("--latitude-file", type=Path, default=None)
    g.add_argument("--height-file", type=Path, default=None)
    g.add_argument("--dem-directory", type=Path, default=None)
    g.add_argument("--dem-parameter-file", type=Path, default=None)
    g.add_argument("--radar-parameter-file", type=Path, default=None)
    g.add_argument("--lookup-table-file", type=Path, default=None)

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    work_dir = args.output.expanduser().resolve() if args.output else Path.cwd().resolve()

    if args.config:
        config_file = args.config.expanduser()
        if not config_file.is_absolute():
            config_file = Path.cwd() / config_file
        config_file = config_file.resolve()
    else:
        config_file = _find_local_config(work_dir)

    try:
        resolved = resolve_project_paths(
            config_path=config_file,
            cli_work_dir=work_dir,
            cli_data_dir=args.project,
            strict_gamma=True,
        )
    except ProjectPathError as exc:
        raise SystemExit(f"Project path error: {exc}") from exc

    export_project_paths(resolved)

    print("============================================================")
    print("pySTAMPS GAMMA STAGE-1 PREPARATION")
    print("============================================================")
    print(f"work_dir : {resolved.work_dir}")
    print(f"data_dir : {resolved.data_dir}")
    print(f"RSLC     : {resolved.rslc_dir}")
    print(f"DIFF     : {resolved.diff_dir}")
    print(f"MLI      : {resolved.mli_dir}")
    print(f"DEM      : {resolved.dem_dir}")
    print(f"RSLC_tab : {resolved.rslc_tab}")
    print(f"itab     : {resolved.itab}")
    print("============================================================")

    if args.force:
        os.environ.pop("PYSTAMPS_STAGE1_RESUME", None)
    else:
        os.environ["PYSTAMPS_STAGE1_RESUME"] = "1"

    run_cfg = load_config(config_file)
    ref = run_cfg.reference
    base = resolved.data_dir
    dem_directory = (
        _resolve_optional_path(args.dem_directory, base=base)
        if args.dem_directory is not None
        else resolved.dem_dir
    )

    cfg = GammaStage1Config(
        candidate=CandidateConfig(
            da_threshold=args.da_threshold,
            min_valid_fraction=args.min_valid_fraction,
            block_rows=2048,
            mli_is_power=True,
            normalize_per_image=False,
        ),
        candidate_source=args.candidate_source,
        reference_date=args.reference_date,
        reference_lon=ref.longitude,
        reference_lat=ref.latitude,
        reference_radius_m=(
            ref.radius_m
            if ref.longitude is not None and ref.latitude is not None
            else None
        ),
        longitude_file=_resolve_optional_path(args.longitude_file, base=base),
        latitude_file=_resolve_optional_path(args.latitude_file, base=base),
        height_file=_resolve_optional_path(args.height_file, base=base),
        dem_directory=dem_directory,
        dem_parameter_file=_resolve_optional_path(args.dem_parameter_file, base=base),
        radar_parameter_file=_resolve_optional_path(args.radar_parameter_file, base=base),
        lookup_table_file=_resolve_optional_path(args.lookup_table_file, base=base),
        range_looks=args.range_looks,
        azimuth_looks=args.azimuth_looks,
        sbas_deramp_mode="none",
        force=args.force,
    )

    result = prepare_gamma_sbas_stage1(
        resolved.data_dir,
        resolved.work_dir,
        config=cfg,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
