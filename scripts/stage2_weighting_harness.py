from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.pipeline.ported import _stage2_psquare_weighting


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _to_vec(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise SystemExit(f"Missing required key: {key}")
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise SystemExit(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and compare the stage-2 P-square weighting path.")
    parser.add_argument("--input", help="JSON file with Nr, Na, low_coh_thresh, Nr_max_nz_ix, coh_ps")
    parser.add_argument("--snapshot", help="stage2_weighting_snapshot.json emitted by stage-2 debug capture")
    parser.add_argument("--output", required=True, help="Output JSON artifact path")
    parser.add_argument("--oracle", help="Optional oracle JSON with prand, prand_hi, prand_ps, weighting")
    args = parser.parse_args()

    if not args.input and not args.snapshot:
        raise SystemExit("One of --input or --snapshot is required")

    source_path = Path(args.snapshot or args.input)
    payload = _load_json(source_path)
    if args.snapshot:
        payload = payload.get("inputs", {})

    Nr = _to_vec(payload, "Nr")
    Na = _to_vec(payload, "Na")
    coh_ps = _to_vec(payload, "coh_ps")
    low_coh_thresh = int(payload["low_coh_thresh"])
    Nr_max_nz_ix = float(payload["Nr_max_nz_ix"])

    prand, prand_hi, prand_ps, weighting = _stage2_psquare_weighting(
        Nr,
        Na,
        low_coh_thresh,
        Nr_max_nz_ix,
        coh_ps,
    )

    result: dict[str, Any] = {
        "input": str(source_path.resolve()),
        "oracle": str(Path(args.oracle).resolve()) if args.oracle else None,
        "summary": {
            "prand_len": int(prand.size),
            "prand_hi_len": int(prand_hi.size),
            "prand_ps_len": int(prand_ps.size),
            "weighting_len": int(weighting.size),
            "weighting_min": float(np.min(weighting)) if weighting.size else 0.0,
            "weighting_mean": float(np.mean(weighting)) if weighting.size else 0.0,
            "weighting_max": float(np.max(weighting)) if weighting.size else 0.0,
        },
        "outputs": {
            "prand": prand.tolist(),
            "prand_hi": prand_hi.tolist(),
            "prand_ps": prand_ps.tolist(),
            "weighting": weighting.tolist(),
        },
    }

    if args.oracle:
        oracle = _load_json(Path(args.oracle))
        oracle_out = oracle.get("outputs", oracle)
        comparisons = {}
        ok = True
        for key, actual in (
            ("prand", prand),
            ("prand_hi", prand_hi),
            ("prand_ps", prand_ps),
            ("weighting", weighting),
        ):
            expected = np.asarray(oracle_out[key], dtype=np.float64).reshape(-1)
            max_abs = _max_abs_diff(actual, expected)
            comparisons[key] = {"max_abs": max_abs, "shape": list(actual.shape)}
            ok = ok and max_abs == 0.0
        result["oracle_compare"] = {"ok": ok, "comparisons": comparisons}

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output_path.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
