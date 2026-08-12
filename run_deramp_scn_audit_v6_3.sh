#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
exec "$PYTHON" "$ROOT/tools/deramp_scn_audit_v6_3.py" "$@"
