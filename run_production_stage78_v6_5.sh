#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
DATASET="${PYSTAMPS_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"

export PYSTAMPS_ROOT="$ROOT"
export PYSTAMPS_DATASET="$DATASET"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

export PYSTAMPS_GACOS_STAGE7_ENABLE="${PYSTAMPS_GACOS_STAGE7_ENABLE:-1}"
export PYSTAMPS_GACOS_REBUILD="${PYSTAMPS_GACOS_REBUILD:-0}"
export PYSTAMPS_SBAS_STAGE7_CHUNK_PS="${PYSTAMPS_SBAS_STAGE7_CHUNK_PS:-2048}"
export PYSTAMPS_SBAS_STAGE8_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_CHUNK_PS:-1024}"
export PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS:-256}"
export PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS="${PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS:-32}"
export PYSTAMPS_SBAS_STAGE8_SPATIAL_FILTER="${PYSTAMPS_SBAS_STAGE8_SPATIAL_FILTER:-1}"

cd "$ROOT"

"$PYTHON" - <<'PY'
from pathlib import Path
import os
from pystamps.pipeline.ported import stage7_calc_scla, stage8_filter_scn

root = Path(os.environ["PYSTAMPS_DATASET"]).expanduser().resolve()

print("=" * 80, flush=True)
print("Production Stage 7: robust deramp + IFGSTD WLS", flush=True)
print("=" * 80, flush=True)
print(stage7_calc_scla(root, backend="auto", chunk_ps=0, io_workers=1), flush=True)

print("=" * 80, flush=True)
print("Production Stage 8: SCLA OFF + SCN DIRECT + median reference", flush=True)
print("=" * 80, flush=True)
print(stage8_filter_scn(root, backend="auto", chunk_ps=0, io_workers=1), flush=True)
PY
