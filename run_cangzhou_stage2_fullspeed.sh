#!/usr/bin/env bash

set -u
set -o pipefail

cd /home/ubuntu/software/pystamps-main

REAL_DATASET=/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized

export PYSTAMPS_CLAP_SINGLE_PRECISION=1
export PYSTAMPS_CLAP_WINDOW_BATCH=8
export PYSTAMPS_CLAP_IFG_WORKERS=15
export PYSTAMPS_CLAP_FFT_WORKERS=1
export PYSTAMPS_CLAP_PROGRESS=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

mkdir -p "$REAL_DATASET/_run_logs"

LOG="$REAL_DATASET/_run_logs/stage2_fullspeed_$(date +%Y%m%d_%H%M%S).log"

echo "数据集：$REAL_DATASET"
echo "并行：2个patch × 每patch 15个IFG线程"
echo "日志：$LOG"

set +e

/usr/bin/time -v \
pystamps \
  --config /home/ubuntu/software/pystamps-main/stage2_cangzhou_fullspeed.yaml \
  run \
  --dataset "$REAL_DATASET" \
  --start-step 2 \
  --end-step 2 \
  --cpu-workers 2 \
  --io-workers 4 \
2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}

set -e

echo "Stage 2退出码：$RC"

pystamps status \
  --dataset "$REAL_DATASET" \
  || true

exit "$RC"
