#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYSTAMPS_PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_spatial_residual_annual_v3_backup/$STAMP"

mkdir -p "$ROOT" "$BACKUP"

FILES=(
  spatial_residual_correction_v3.py
  calc_joint_regularized_seasonal_gls_v2.py
  calc_annual_velocity_gls.py
  run_spatial_residual_annual_v3.sh
  test_spatial_residual_correction_v3.py
  test_joint_regularized_v2.py
)

for name in "${FILES[@]}"; do
  [[ -f "$HERE/$name" ]] || {
    echo "错误：安装包缺少 $name" >&2
    exit 2
  }
  if [[ -f "$ROOT/$name" ]]; then
    cp -a "$ROOT/$name" "$BACKUP/"
  fi
  cp -a "$HERE/$name" "$ROOT/$name"
  chmod +x "$ROOT/$name"
done

"$PYTHON" -m py_compile \
  "$ROOT/spatial_residual_correction_v3.py" \
  "$ROOT/calc_joint_regularized_seasonal_gls_v2.py" \
  "$ROOT/calc_annual_velocity_gls.py" \
  "$ROOT/test_spatial_residual_correction_v3.py" \
  "$ROOT/test_joint_regularized_v2.py"

bash -n "$ROOT/run_spatial_residual_annual_v3.sh"

cd "$ROOT"
"$PYTHON" "$ROOT/test_spatial_residual_correction_v3.py"
"$PYTHON" "$ROOT/test_joint_regularized_v2.py"

echo
echo "安装与合成测试完成。"
echo "备份目录：$BACKUP"
echo "一键运行："
echo "  cd '$ROOT'"
echo "  ./run_spatial_residual_annual_v3.sh"
