#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
TOOL="$ROOT/tools/mintpy_network_inversion_v6.py"

[[ -x "$PYTHON" ]] || { echo "ERROR Python: $PYTHON" >&2; exit 2; }
[[ -f "$TOOL" ]] || { echo "ERROR tool: $TOOL" >&2; exit 3; }

exec "$PYTHON" "$TOOL" "$@"
