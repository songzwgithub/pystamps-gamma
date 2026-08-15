#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pystamps.config import load_config
from pystamps.prep.gamma_candidates import CandidateConfig
from pystamps.prep.gamma_stage1 import GammaStage1Config, prepare_gamma_sbas_stage1


def main():
    ap = argparse.ArgumentParser(
        description="Prepare a GAMMA SBAS project for pySTAMPS-GAMMA Stage 1."
    )
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=Path("config/production.yaml"))
    ap.add_argument("--reference-date", default=None)
    ap.add_argument("--range-looks", type=int, default=None)
    ap.add_argument("--azimuth-looks", type=int, default=None)
    ap.add_argument("--da-threshold", type=float, default=0.60)
    ap.add_argument("--min-valid-fraction", type=float, default=0.90)
    ap.add_argument("--candidate-source", choices=("rslc_sbas", "mli"), default="rslc_sbas")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_cfg = load_config(args.config)
    ref = run_cfg.reference

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
        range_looks=args.range_looks,
        azimuth_looks=args.azimuth_looks,
        sbas_deramp_mode="none",
        force=args.force,
    )

    result = prepare_gamma_sbas_stage1(args.project, args.output, config=cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
