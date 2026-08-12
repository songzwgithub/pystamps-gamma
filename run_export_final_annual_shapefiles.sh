#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"
SCRIPT="$ROOT/export_final_annual_shapefiles.py"
OUT="$DATASET/final_annual_shapefiles"

[[ -x "$PYTHON" ]] || {
  echo "错误：找不到Python：$PYTHON" >&2
  exit 2
}

[[ -f "$SCRIPT" ]] || {
  echo "错误：找不到脚本：$SCRIPT" >&2
  exit 3
}

[[ -s "$DATASET/joint_piecewise_seasonal_velocity/joint_piecewise_seasonal_velocity.gpkg" ]] || {
  echo "错误：找不到联合年度GeoPackage" >&2
  exit 4
}

"$PYTHON" "$SCRIPT" \
  --dataset "$DATASET" \
  --out "$OUT" \
  --review-se-mm-yr 4.5 \
  --overwrite
