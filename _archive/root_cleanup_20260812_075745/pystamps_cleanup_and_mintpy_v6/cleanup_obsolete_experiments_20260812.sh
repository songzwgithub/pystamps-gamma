#!/usr/bin/env bash
set -euo pipefail

# Conservative cleanup for /home/ubuntu/software/pystamps-main
#
# DEFAULT = DRY RUN.
# --apply moves obsolete experiment wrappers/bundles into an archive directory.
# It does NOT delete production code, dataset outputs, Stage6 checkpoints, or
# current SBAS Stage7/8 code.
#
# Only __pycache__ directories are actually removed under --apply because they
# are Python byte-code caches and are always reproducible.

ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
MODE="dry"
if [[ "${1:-}" == "--apply" ]]; then
    MODE="apply"
elif [[ "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
    echo "Usage: $0 [--dry-run|--apply]" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ROOT/_archive/obsolete_experiments_${STAMP}"
LOG_ARCHIVE="$ROOT/_archive/old_logs_${STAMP}"

echo "ROOT    : $ROOT"
echo "MODE    : $MODE"
echo "ARCHIVE : $ARCHIVE"
echo

# Explicit, conservative list only.
# These are superseded diagnostics/experiments from the now-abandoned routes:
# hard IFG selection, pixel WLS approximations, and final-field spatial filtering.
ROOT_EXPERIMENTS=(
  "ifg_network_sbas_pipeline_v1.zip"
  "ifg_network_sbas_pipeline_v2.zip"
  "pixel_wls_sbas_pipeline_v3.zip"
  "stage7_exact_solver_audit_v4.zip"
  "finalC_local_denoise_v5.zip"
  "finalC_structure_preservation_v5_1.zip"
  "run_ifg_network_sbas_v1.sh"
  "run_ifg_network_sbas_v2.sh"
  "run_pixel_wls_sbas_v3.sh"
  "run_stage7_exact_solver_audit_v4.sh"
  "run_finalC_local_denoise_v5.sh"
  "run_finalC_structure_preservation_v5_1.sh"
)

# Matching installed experiment tools, if present.
TOOL_EXPERIMENTS=(
  "tools/ifg_network_sbas_v1.py"
  "tools/ifg_network_sbas_v2.py"
  "tools/pixel_wls_sbas_v3.py"
  "tools/stage7_exact_solver_audit_v4.py"
  "tools/finalC_local_denoise_v5.py"
  "tools/finalC_structure_preservation_v5_1.py"
)

# Logs are archived, never deleted.
OLD_LOGS=(
  "gamma_sbas_smoke.log"
  "stage2_smoke.log"
  "stage2_smoke_smallpatch.log"
  "compare_stage2_fast_double.log"
  "gamma_sbas_full_stage1.log"
  "stage2_pyspy_dump.txt"
)

move_one() {
    local rel="$1"
    local src="$ROOT/$rel"
    [[ -e "$src" ]] || return 0

    if [[ "$MODE" == "dry" ]]; then
        echo "[DRY] archive $rel"
    else
        mkdir -p "$ARCHIVE/$(dirname "$rel")"
        mv "$src" "$ARCHIVE/$rel"
        echo "[OK ] archived $rel"
    fi
}

move_log() {
    local rel="$1"
    local src="$ROOT/$rel"
    [[ -e "$src" ]] || return 0

    if [[ "$MODE" == "dry" ]]; then
        echo "[DRY] archive log $rel"
    else
        mkdir -p "$LOG_ARCHIVE/$(dirname "$rel")"
        mv "$src" "$LOG_ARCHIVE/$rel"
        echo "[OK ] archived log $rel"
    fi
}

for f in "${ROOT_EXPERIMENTS[@]}"; do move_one "$f"; done
for f in "${TOOL_EXPERIMENTS[@]}"; do move_one "$f"; done
for f in "${OLD_LOGS[@]}"; do move_log "$f"; done

echo
echo "Python byte-code caches:"
if [[ "$MODE" == "dry" ]]; then
    find "$ROOT" -type d -name '__pycache__' \
      -not -path "$ROOT/.git/*" \
      -not -path "$ROOT/_archive/*" \
      -print | sed 's#^#[DRY] remove #'
else
    find "$ROOT" -type d -name '__pycache__' \
      -not -path "$ROOT/.git/*" \
      -not -path "$ROOT/_archive/*" \
      -prune -exec rm -rf {} +
    echo "[OK ] removed __pycache__ directories"
fi

cat <<'EOF'

NOT TOUCHED:
  pystamps/
  src/
  crates/
  scripts/
  tests/
  tools/ other than the explicit obsolete experiment files above
  run_gamma_sbas_ps_optimized.py
  run_stage6_*.sh / Stage6 repair-resume scripts
  run_stage78_gacos.sh
  gacos_correction.py
  install_stage78_sbas_patch.sh
  pystamps_gacos_stage78_bundle.zip
  dataset MAT/HDF5/TIF files
  _stage6_sbas_work/grid_v2
  SNAPHU outputs
  Final-C products
  V3 coherence cache in the DATASET directory

Also NOT removed automatically:
  target/              (Rust build cache; can be expensive to rebuild)
  *.egg-info
  old post-processing scripts
  rollback/repair scripts

Run once without --apply first. If the printed list is correct:
  ./cleanup_obsolete_experiments_20260812.sh --apply
EOF
