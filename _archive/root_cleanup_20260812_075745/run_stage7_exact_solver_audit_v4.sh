#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
TOOL="$ROOT/tools/stage7_exact_solver_audit_v4.py"

[[ -x "$PYTHON" ]] || {
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 2
}
[[ -f "$TOOL" ]] || {
  echo "ERROR: V4 tool not installed: $TOOL" >&2
  exit 3
}

exec "$PYTHON" "$TOOL" "$@"
