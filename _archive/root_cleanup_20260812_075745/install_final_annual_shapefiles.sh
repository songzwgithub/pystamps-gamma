#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYSTAMPS_PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_final_annual_shp_backup/$STAMP"

mkdir -p "$ROOT" "$BACKUP"

for name in \
  export_final_annual_shapefiles.py \
  run_export_final_annual_shapefiles.sh
do
  if [[ -f "$ROOT/$name" ]]; then
    cp -a "$ROOT/$name" "$BACKUP/"
  fi
  cp -a "$HERE/$name" "$ROOT/$name"
  chmod +x "$ROOT/$name"
done

"$PYTHON" -m py_compile \
  "$ROOT/export_final_annual_shapefiles.py"
bash -n "$ROOT/run_export_final_annual_shapefiles.sh"

echo "安装完成。备份：$BACKUP"
echo "运行："
echo "  cd '$ROOT'"
echo "  ./run_export_final_annual_shapefiles.sh"
