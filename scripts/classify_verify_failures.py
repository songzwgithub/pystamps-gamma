#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pystamps.config import RunConfig
from pystamps.verify import summarize_failures, verify_run_against_golden


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verify and classify residual parity failures by downstream stage.")
    parser.add_argument("--run", required=True, help="Run root to compare")
    parser.add_argument("--golden", required=True, help="Golden root to compare against")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run).expanduser().resolve()
    golden_root = Path(args.golden).expanduser().resolve()
    report = verify_run_against_golden(run_root, golden_root, RunConfig().tolerance)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "golden_root": str(golden_root),
        **summarize_failures(report),
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
