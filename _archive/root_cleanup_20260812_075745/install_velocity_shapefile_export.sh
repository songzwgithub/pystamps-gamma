#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYSTAMPS_PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_velocity_shp_export_backup/$STAMP"

mkdir -p "$ROOT" "$BACKUP"

for name in \
  export_velocity_shapefiles.py \
  run_export_velocity_shapefiles.sh
do
  if [[ -f "$ROOT/$name" ]]; then
    cp -a "$ROOT/$name" "$BACKUP/"
  fi
  cp -a "$HERE/$name" "$ROOT/$name"
  chmod +x "$ROOT/$name"
done

"$PYTHON" -m py_compile "$ROOT/export_velocity_shapefiles.py"
bash -n "$ROOT/run_export_velocity_shapefiles.sh"

echo "安装完成。备份目录：$BACKUP"
echo "运行："
echo "  cd '$ROOT'"
echo "  ./run_export_velocity_shapefiles.sh"
