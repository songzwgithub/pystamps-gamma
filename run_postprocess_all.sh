#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
SCRIPT="$ROOT/pystamps_sbas_postprocess.py"
OUT="$DATASET/postprocess"

PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python: $PYTHON" >&2
  exit 2
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing script: $SCRIPT" >&2
  exit 3
fi

for required in \
  ps2.mat mean_v.mat scla_smooth2.mat phuw_sm2.mat uw_space_time.mat
do
  if [[ ! -s "$DATASET/$required" ]]; then
    echo "Missing Stage 7/8 output: $DATASET/$required" >&2
    exit 4
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$DATASET/_run_logs"
LOG="$LOG_DIR/postprocess_${STAMP}.log"
SOCKET="postprocess"
SESSION="cangzhou_postprocess"

mkdir -p "$LOG_DIR"

# Preserve the working NumPy environment. Do not run conda install here.
export PATH="/home/ubuntu/software/miniconda3/envs/stamps/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux -L "$SOCKET" new-session -d -s "$SESSION" \
  "$PYTHON '$SCRIPT' \
    --dataset '$DATASET' \
    --repo-root '$ROOT' \
    --out '$OUT' \
    --resolution-m 50 \
    --min-points-per-cell 1 \
    --vector-formats csv,gpkg,parquet,kml \
    --kml-max-points 20000 \
    --write-hdf5 \
    --include-scn-hdf5 \
    --plot-timeseries \
    --wgs84-copy \
    --reference-mode existing \
    --overwrite \
    >> '$LOG' 2>&1"

sleep 5

if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "Post-processing started."
  echo "Log: $LOG"
  echo
  echo "Attach:"
  echo "  tmux -L $SOCKET attach -t $SESSION"
  echo
  echo "Follow log:"
  echo "  tail -f '$LOG'"
else
  echo "Post-processing exited immediately:" >&2
  tail -n 200 "$LOG" >&2 || true
  exit 5
fi
