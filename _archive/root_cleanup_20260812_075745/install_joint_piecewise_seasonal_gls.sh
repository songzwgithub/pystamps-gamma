#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
ENV_DIR="${PYSTAMPS_ENV_DIR:-/home/ubuntu/software/miniconda3/envs/stamps}"
PYTHON="$ENV_DIR/bin/python"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-install}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_joint_piecewise_seasonal_backup/$STAMP"

mkdir -p "$BACKUP"

for source in \
  calc_joint_piecewise_seasonal_gls.py \
  run_joint_piecewise_seasonal_gls.sh \
  test_joint_piecewise_seasonal_gls.py
do
  [[ -f "$SELF_DIR/$source" ]] || {
    echo "错误：安装包缺少：$source" >&2
    exit 2
  }
done

[[ -x "$PYTHON" ]] || {
  echo "错误：找不到Python：$PYTHON" >&2
  exit 3
}

for target in \
  "$ROOT/calc_joint_piecewise_seasonal_gls.py" \
  "$ROOT/run_joint_piecewise_seasonal_gls.sh" \
  "$ROOT/test_joint_piecewise_seasonal_gls.py"
do
  [[ -f "$target" ]] && cp -a "$target" "$BACKUP/"
done

cp "$SELF_DIR/calc_joint_piecewise_seasonal_gls.py" "$ROOT/"
cp "$SELF_DIR/run_joint_piecewise_seasonal_gls.sh" "$ROOT/"
cp "$SELF_DIR/test_joint_piecewise_seasonal_gls.py" "$ROOT/"
chmod +x \
  "$ROOT/calc_joint_piecewise_seasonal_gls.py" \
  "$ROOT/run_joint_piecewise_seasonal_gls.sh" \
  "$ROOT/test_joint_piecewise_seasonal_gls.py"

if [[ ! -f "$ROOT/calc_annual_velocity_gls.py" ]]; then
  [[ -f "$SELF_DIR/calc_annual_velocity_gls.py" ]] || {
    echo "错误：缺少支撑模块calc_annual_velocity_gls.py" >&2
    exit 4
  }
  cp "$SELF_DIR/calc_annual_velocity_gls.py" "$ROOT/"
  chmod +x "$ROOT/calc_annual_velocity_gls.py"
fi

"$PYTHON" -m py_compile \
  "$ROOT/calc_joint_piecewise_seasonal_gls.py" \
  "$ROOT/calc_annual_velocity_gls.py" \
  "$ROOT/test_joint_piecewise_seasonal_gls.py"

(
  cd "$ROOT"
  "$PYTHON" test_joint_piecewise_seasonal_gls.py
)

echo
echo "安装完成。"
echo "备份：$BACKUP"
echo "主脚本：$ROOT/calc_joint_piecewise_seasonal_gls.py"
echo "启动器：$ROOT/run_joint_piecewise_seasonal_gls.sh"

case "$MODE" in
  install)
    ;;
  install-run|run)
    cd "$ROOT"
    ./run_joint_piecewise_seasonal_gls.sh
    ;;
  *)
    echo "用法：$0 [install|install-run]" >&2
    exit 1
    ;;
esac
