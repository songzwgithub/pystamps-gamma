#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
TOOL="$ROOT/tools/pixel_wls_sbas_v3.py"

[[ -x "$PYTHON" ]] || {
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 2
}

[[ -f "$TOOL" ]] || {
  echo "ERROR: V3 tool not installed: $TOOL" >&2
  exit 3
}

exec "$PYTHON" "$TOOL" "$@"
