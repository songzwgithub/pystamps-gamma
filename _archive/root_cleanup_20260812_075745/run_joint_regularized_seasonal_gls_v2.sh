#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"
SCRIPT="$ROOT/calc_joint_regularized_seasonal_gls_v2.py"
OVERALL_SCRIPT="$ROOT/export_existing_overall_velocity_shp.py"
OUT="$DATASET/joint_regularized_seasonal_velocity_v2"
OVERALL_OUT="$DATASET/overall_velocity_shapefiles"

LOG_DIR="$DATASET/_run_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/joint_regularized_v2_${STAMP}.log"
SOCKET="jointreg2"
SESSION="cangzhou_joint_regularized_v2"

mkdir -p "$LOG_DIR"

for file in ps2.mat parms.mat ifgstd2.mat scla2.mat uw_space_time.mat; do
  [[ -s "$DATASET/$file" ]] || {
    echo "错误：缺少输入：$DATASET/$file" >&2
    exit 2
  }
done

[[ -x "$PYTHON" ]] || {
  echo "错误：找不到Python：$PYTHON" >&2
  exit 3
}
[[ -f "$SCRIPT" ]] || {
  echo "错误：找不到脚本：$SCRIPT" >&2
  exit 4
}
[[ -f "$ROOT/calc_annual_velocity_gls.py" ]] || {
  echo "错误：缺少支撑模块：$ROOT/calc_annual_velocity_gls.py" >&2
  exit 5
}

export PATH="$ENV_DIR/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux -L "$SOCKET" new-session -d -s "$SESSION" \
  "$PYTHON '$SCRIPT' \
    --dataset '$DATASET' \
    --repo-root '$ROOT' \
    --out '$OUT' \
    --covariance-mode network \
    --chunk-ps 2048 \
    --validation-chunk-ps 4096 \
    --seasonal-harmonics 1 \
    --gcv-curvature-candidates '4,6,8,12,18,25,40,60' \
    --gcv-sample-ps 2048 \
    --gcv-log-tolerance 0.03 \
    --min-year-epochs 10 \
    --min-year-span-days 240 \
    --max-model-rmse-mm 15 \
    --max-rate-se-mm-yr 5 \
    --max-independent-se-mm-yr 8 \
    --model-agreement-abs-mm-yr 8 \
    --model-agreement-sigma 2 \
    --max-regularized-condition 1e10 \
    --max-independent-condition 1e10 \
    --write-diagnostic-gpkg \
    --write-final-shp \
    --resume \
    >> '$LOG' 2>&1 \
  && '$PYTHON' '$OVERALL_SCRIPT' \
    --dataset '$DATASET' \
    --out '$OVERALL_OUT' \
    >> '$LOG' 2>&1"

sleep 5

if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "正则化联合逐年速率V2已启动。"
  echo "日志：$LOG"
  echo
  echo "进入窗口："
  echo "  tmux -L $SOCKET attach -t $SESSION"
  echo
  echo "查看日志："
  echo "  tail -f '$LOG'"
else
  echo "任务启动后立即退出，日志如下：" >&2
  tail -n 200 "$LOG" >&2 || true
  exit 6
fi
