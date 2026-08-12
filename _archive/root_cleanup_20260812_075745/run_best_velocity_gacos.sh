#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"
SCRIPT="$ROOT/calc_best_velocity_gls.py"
OUT="$DATASET/best_velocity_gacos"

LOG_DIR="$DATASET/_run_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/best_velocity_gacos_${STAMP}.log"
SOCKET="bestvel"
SESSION="cangzhou_best_velocity"

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
    --chunk-ps 4096 \
    --min-valid-fraction 0.80 \
    --min-span-fraction 0.80 \
    --max-rmse-mm 12 \
    --max-rate-se-mm-yr 2 \
    --local-radius-m 300 \
    --local-k 12 \
    --local-min-neighbors 4 \
    --main-resolution-m 50 \
    --main-min-points 1 \
    --detail-resolution-m 100 \
    --detail-min-points 3 \
    --wgs84-copy \
    --overwrite \
    >> '$LOG' 2>&1"

sleep 5

if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "最优速率计算已启动。"
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
