#!/usr/bin/env bash
set -euo pipefail
ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
POINTER="$ROOT/.stage6_grid_refactor_last_backup"
[[ -s "$POINTER" ]] || { echo "没有找到备份指针：$POINTER" >&2; exit 2; }
BACKUP="$(cat "$POINTER")"
MODULE="$ROOT/pystamps/pipeline/stage6_sbas.py"
[[ -f "$BACKUP/pystamps/pipeline/stage6_sbas.py.current" ]] || {
    echo "备份中没有原stage6_sbas.py：$BACKUP" >&2
    exit 3
}
cp -a "$BACKUP/pystamps/pipeline/stage6_sbas.py.current" "$MODULE"
[[ -f "$BACKUP/run_stage6_fast.sh" ]] && cp -a "$BACKUP/run_stage6_fast.sh" "$ROOT/run_stage6_fast.sh"
python -m py_compile "$MODULE"
echo "已回滚到：$BACKUP"
