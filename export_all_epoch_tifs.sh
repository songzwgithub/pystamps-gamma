#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
SCRIPT="$ROOT/pystamps_sbas_postprocess.py"
PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"

# This regenerates the complete postprocess folder and writes all 257 epoch GeoTIFFs.
# Remove --wgs84-epoch-copy to avoid doubling disk usage.
exec "$PYTHON" "$SCRIPT" \
  --dataset "$DATASET" \
  --repo-root "$ROOT" \
  --out "$DATASET/postprocess" \
  --resolution-m 50 \
  --vector-formats csv,gpkg,parquet,kml \
  --write-hdf5 \
  --include-scn-hdf5 \
  --plot-timeseries \
  --epoch-tifs \
  --epoch-step 1 \
  --epoch-start 0 \
  --reference-mode existing \
  --wgs84-copy \
  --overwrite
