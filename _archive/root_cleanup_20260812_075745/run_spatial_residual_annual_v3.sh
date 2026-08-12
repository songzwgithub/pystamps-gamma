#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"

CORRECTION_SCRIPT="$ROOT/spatial_residual_correction_v3.py"
ANNUAL_SCRIPT="$ROOT/calc_joint_regularized_seasonal_gls_v2.py"
CORRECTION_OUT="$DATASET/spatial_residual_correction_v3"
SHADOW_DATASET="$DATASET/_spatial_corrected_dataset_v3"
ANNUAL_OUT="$DATASET/spatial_corrected_annual_velocity_v3"

LOG_DIR="$DATASET/_run_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/spatial_residual_annual_v3_${STAMP}.log"
SOCKET="spatresv3"
SESSION="cangzhou_spatial_residual_annual_v3"

mkdir -p "$LOG_DIR"

for file in ps2.mat parms.mat ifgstd2.mat scla2.mat uw_space_time.mat; do
  [[ -s "$DATASET/$file" ]] || {
    echo "错误：缺少输入：$DATASET/$file" >&2
    exit 2
  }
done

for file in \
  "$CORRECTION_SCRIPT" \
  "$ANNUAL_SCRIPT" \
  "$ROOT/calc_annual_velocity_gls.py"
do
  [[ -f "$file" ]] || {
    echo "错误：缺少脚本：$file" >&2
    exit 3
  }
done

[[ -x "$PYTHON" ]] || {
  echo "错误：找不到Python：$PYTHON" >&2
  exit 4
}

export PATH="$ENV_DIR/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${PYSTAMPS_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${PYSTAMPS_BLAS_THREADS:-4}"
export MKL_NUM_THREADS="${PYSTAMPS_BLAS_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${PYSTAMPS_BLAS_THREADS:-4}"
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

OVERWRITE_FLAG=""
ANNUAL_OVERWRITE_FLAG=""
if [[ "${PYSTAMPS_V3_OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
  ANNUAL_OVERWRITE_FLAG="--overwrite"
fi

REFERENCE_ARGS=""
if [[ -n "${PYSTAMPS_REFERENCE_LON:-}" && -n "${PYSTAMPS_REFERENCE_LAT:-}" ]]; then
  REFERENCE_ARGS="--reference-lon ${PYSTAMPS_REFERENCE_LON} --reference-lat ${PYSTAMPS_REFERENCE_LAT} --reference-radius-m ${PYSTAMPS_REFERENCE_RADIUS_M:-1000}"
fi

SPATIAL_GRID_M="${PYSTAMPS_SPATIAL_GRID_M:-4000}"
SPATIAL_SIGMA_CELLS="${PYSTAMPS_SPATIAL_SIGMA_CELLS:-1.5}"
TEMPORAL_SIGMA_DAYS="${PYSTAMPS_TEMPORAL_SIGMA_DAYS:-180}"
MIN_CV_IMPROVEMENT="${PYSTAMPS_MIN_CV_IMPROVEMENT:-0.05}"
MAX_CORRECTION_MM="${PYSTAMPS_MAX_CORRECTION_MM:-30}"
CURVATURE_SIGMA="${PYSTAMPS_CURVATURE_SIGMA_MM_YR:-8}"
FIRST_DIFF_SIGMA="${PYSTAMPS_FIRST_DIFF_SIGMA_MM_YR:-15}"

cat > "$LOG_DIR/.spatial_v3_chain_${STAMP}.sh" <<CHAIN
#!/usr/bin/env bash
set -euo pipefail

"$PYTHON" "$CORRECTION_SCRIPT" \
  --dataset "$DATASET" \
  --repo-root "$ROOT" \
  --out "$CORRECTION_OUT" \
  --shadow-dataset "$SHADOW_DATASET" \
  --chunk-ps 4096 \
  --preliminary-harmonics 2 \
  --preliminary-irls-iterations 4 \
  --temporal-sigma-days "$TEMPORAL_SIGMA_DAYS" \
  --spatial-grid-m "$SPATIAL_GRID_M" \
  --spatial-sigma-cells "$SPATIAL_SIGMA_CELLS" \
  --min-cell-points 30 \
  --min-cv-improvement "$MIN_CV_IMPROVEMENT" \
  --max-correction-mm "$MAX_CORRECTION_MM" \
  --resume \
  $REFERENCE_ARGS \
  $OVERWRITE_FLAG

"$PYTHON" "$ANNUAL_SCRIPT" \
  --dataset "$SHADOW_DATASET" \
  --repo-root "$ROOT" \
  --out "$ANNUAL_OUT" \
  --covariance-mode network \
  --chunk-ps 2048 \
  --validation-chunk-ps 4096 \
  --seasonal-harmonics 1 \
  --curvature-sigma-mm-yr "$CURVATURE_SIGMA" \
  --first-difference-sigma-mm-yr "$FIRST_DIFF_SIGMA" \
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
  $ANNUAL_OVERWRITE_FLAG
CHAIN

chmod +x "$LOG_DIR/.spatial_v3_chain_${STAMP}.sh"

tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux -L "$SOCKET" new-session -d -s "$SESSION" \
  "bash '$LOG_DIR/.spatial_v3_chain_${STAMP}.sh' >> '$LOG' 2>&1"

sleep 5

if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "逐期空间残差改正与年度速率V3已启动。"
  echo "日志：$LOG"
  echo
  echo "进入窗口："
  echo "  tmux -L $SOCKET attach -t $SESSION"
  echo
  echo "查看日志："
  echo "  tail -f '$LOG'"
  echo
  echo "最终年度SHP目录："
  echo "  $ANNUAL_OUT/annual_shapefiles/formal"
else
  echo "任务启动后立即退出，日志如下：" >&2
  tail -n 240 "$LOG" >&2 || true
  exit 5
fi
