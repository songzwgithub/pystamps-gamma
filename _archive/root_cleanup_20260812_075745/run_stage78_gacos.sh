#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
DATASET="${PYSTAMPS_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"
ENV_DIR="${PYSTAMPS_ENV_DIR:-/home/ubuntu/software/miniconda3/envs/stamps}"
PYTHON="$ENV_DIR/bin/python"

export PYSTAMPS_ROOT="$ROOT"
export PYSTAMPS_DATASET="$DATASET"
export PYSTAMPS_ENV_DIR="$ENV_DIR"
export PYTHONPATH="$ROOT"
export PATH="$ENV_DIR/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

export PYSTAMPS_GACOS_STAGE7_ENABLE="${PYSTAMPS_GACOS_STAGE7_ENABLE:-1}"
export PYSTAMPS_GACOS_FORMAT="${PYSTAMPS_GACOS_FORMAT:-auto}"
export PYSTAMPS_GACOS_UNIT="${PYSTAMPS_GACOS_UNIT:-auto}"
export PYSTAMPS_GACOS_PROJECTION="${PYSTAMPS_GACOS_PROJECTION:-zenith}"
export PYSTAMPS_GACOS_SIGN="${PYSTAMPS_GACOS_SIGN:-auto}"
export PYSTAMPS_GACOS_STRICT_DATES="${PYSTAMPS_GACOS_STRICT_DATES:-1}"
export PYSTAMPS_GACOS_REBUILD="${PYSTAMPS_GACOS_REBUILD:-0}"
export PYSTAMPS_GACOS_CHUNK_PS="${PYSTAMPS_GACOS_CHUNK_PS:-4096}"
export PYSTAMPS_GACOS_MIN_VALID_FRACTION="${PYSTAMPS_GACOS_MIN_VALID_FRACTION:-0.995}"
export PYSTAMPS_SBAS_STAGE7_CHUNK_PS="${PYSTAMPS_SBAS_STAGE7_CHUNK_PS:-2048}"
export PYSTAMPS_SBAS_STAGE8_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_CHUNK_PS:-1024}"
export PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS:-256}"
export PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS="${PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS:-32}"

cd "$ROOT"

"$PYTHON" - <<'PY_RUN'
from pathlib import Path
import os

from pystamps.pipeline.gacos_correction import ensure_gacos_corrected_phuw
from pystamps.pipeline.ported import stage7_calc_scla, stage8_filter_scn

root = Path(os.environ["PYSTAMPS_DATASET"]).expanduser().resolve()

print("============================================================", flush=True)
print("GACOS correction", flush=True)
print("============================================================", flush=True)
print(ensure_gacos_corrected_phuw(root), flush=True)

print("============================================================", flush=True)
print("Stage 7 SBAS with GACOS", flush=True)
print("============================================================", flush=True)
print(stage7_calc_scla(root, backend="auto", chunk_ps=0, io_workers=1), flush=True)

print("============================================================", flush=True)
print("Stage 8 SBAS with GACOS", flush=True)
print("============================================================", flush=True)
print(stage8_filter_scn(root, backend="auto", chunk_ps=0, io_workers=1), flush=True)
PY_RUN
