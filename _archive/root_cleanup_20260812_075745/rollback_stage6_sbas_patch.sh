#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
RECORD="$ROOT/.stage6_sbas_last_backup"

[[ -f "$RECORD" ]] || {
    echo "找不到Stage 6 SBAS备份记录。" >&2
    exit 2
}

BACKUP="$(cat "$RECORD")"
[[ -d "$BACKUP" ]] || {
    echo "备份目录不存在：$BACKUP" >&2
    exit 2
}

cp -a "$BACKUP/pystamps/pipeline/ported.py" \
      "$ROOT/pystamps/pipeline/ported.py"

if [[ -f "$BACKUP/pystamps/pipeline/stage6_sbas.py" ]]; then
    cp -a "$BACKUP/pystamps/pipeline/stage6_sbas.py" \
          "$ROOT/pystamps/pipeline/stage6_sbas.py"
else
    rm -f "$ROOT/pystamps/pipeline/stage6_sbas.py"
fi

if [[ -f "$BACKUP/tests/test_stage6_sbas.py" ]]; then
    cp -a "$BACKUP/tests/test_stage6_sbas.py" \
          "$ROOT/tests/test_stage6_sbas.py"
else
    rm -f "$ROOT/tests/test_stage6_sbas.py"
fi

if [[ -f "$BACKUP/run_native_stage.sh" ]]; then
    cp -a "$BACKUP/run_native_stage.sh" \
          "$ROOT/run_native_stage.sh"
fi

rm -f "$ROOT/run_stage6_sbas.sh"

echo "已恢复：$BACKUP"
