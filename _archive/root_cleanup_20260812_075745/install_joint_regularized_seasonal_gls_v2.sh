#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYSTAMPS_PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_joint_regularized_v2_backup/$STAMP"

mkdir -p "$ROOT" "$BACKUP"

for name in \
  calc_joint_regularized_seasonal_gls_v2.py \
  calc_annual_velocity_gls.py \
  export_existing_overall_velocity_shp.py \
  run_joint_regularized_seasonal_gls_v2.sh \
  test_joint_regularized_v2.py
do
  if [[ -f "$ROOT/$name" ]]; then
    cp -a "$ROOT/$name" "$BACKUP/"
  fi
  cp -a "$HERE/$name" "$ROOT/$name"
  chmod +x "$ROOT/$name"
done

"$PYTHON" -m py_compile \
  "$ROOT/calc_joint_regularized_seasonal_gls_v2.py" \
  "$ROOT/calc_annual_velocity_gls.py" \
  "$ROOT/export_existing_overall_velocity_shp.py" \
  "$ROOT/test_joint_regularized_v2.py"

bash -n "$ROOT/run_joint_regularized_seasonal_gls_v2.sh"

cd "$ROOT"
"$PYTHON" "$ROOT/test_joint_regularized_v2.py"

echo
echo "安装和合成测试完成。"
echo "备份目录：$BACKUP"
echo "运行："
echo "  cd '$ROOT'"
echo "  ./run_joint_regularized_seasonal_gls_v2.sh"
