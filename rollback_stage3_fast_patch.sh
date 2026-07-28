#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"

if [[ ! -f "$ROOT/.stage3_fast_last_backup" ]]; then
    echo "找不到最近备份记录。" >&2
    exit 2
fi

BACKUP="$(
    cat "$ROOT/.stage3_fast_last_backup"
)"

if [[ ! -d "$BACKUP" ]]; then
    echo "备份目录不存在：$BACKUP" >&2
    exit 2
fi

cp -a \
  "$BACKUP/pystamps/pipeline/ported.py" \
  "$ROOT/pystamps/pipeline/ported.py"

cp -a \
  "$BACKUP/pystamps/io/mat.py" \
  "$ROOT/pystamps/io/mat.py"

if [[ -f "$BACKUP/tests/test_stage3_fast_path.py" ]]; then
    cp -a \
      "$BACKUP/tests/test_stage3_fast_path.py" \
      "$ROOT/tests/test_stage3_fast_path.py"
else
    rm -f \
      "$ROOT/tests/test_stage3_fast_path.py"
fi

echo "已从以下目录恢复："
echo "$BACKUP"
