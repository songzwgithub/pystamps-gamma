#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
TOOL="$ROOT/tools/ifg_network_sbas_v2.py"

[[ -x "$PYTHON" ]] || { echo "ERROR: Python not found: $PYTHON" >&2; exit 2; }
[[ -f "$TOOL" ]] || { echo "ERROR: tool not installed: $TOOL" >&2; exit 3; }

exec "$PYTHON" "$TOOL" "$@"
