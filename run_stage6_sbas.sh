#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
DATASET="${REAL_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYSTAMPS_SBAS_PROGRESS="${PYSTAMPS_SBAS_PROGRESS:-1}"
export PYSTAMPS_SBAS_EDGE_CHUNK="${PYSTAMPS_SBAS_EDGE_CHUNK:-512}"
export PYSTAMPS_STAGE6_SNAPHU_WORKERS="${PYSTAMPS_STAGE6_SNAPHU_WORKERS:-4}"
export PYSTAMPS_SBAS_ANNEAL_WORKERS="${PYSTAMPS_SBAS_ANNEAL_WORKERS:-8}"
export PYSTAMPS_SBAS_ANNEAL_RUNS="${PYSTAMPS_SBAS_ANNEAL_RUNS:-15}"
export PYSTAMPS_SBAS_STRICT_ANNEAL="${PYSTAMPS_SBAS_STRICT_ANNEAL:-1}"
export PYSTAMPS_SBAS_KEEP_WORK="${PYSTAMPS_SBAS_KEEP_WORK:-0}"

cd "$ROOT"

echo "============================================================"
echo "StaMPS-compatible SBAS Stage 6"
echo "Dataset       : $DATASET"
echo "Edge chunk    : $PYSTAMPS_SBAS_EDGE_CHUNK"
echo "SNAPHU workers: $PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo "Anneal workers: $PYSTAMPS_SBAS_ANNEAL_WORKERS"
echo "Anneal runs   : $PYSTAMPS_SBAS_ANNEAL_RUNS"
echo "Strict anneal : $PYSTAMPS_SBAS_STRICT_ANNEAL"
echo "============================================================"

python -m pystamps.pipeline.stage6_sbas \
    --dataset "$DATASET" \
    --io-workers "${PYSTAMPS_STAGE6_GRID_WORKERS:-8}" \
    "$@"
