#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
DATASET="${PYSTAMPS_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"
ENV_DIR="${PYSTAMPS_ENV_DIR:-/home/ubuntu/software/miniconda3/envs/stamps}"
PYTHON="$ENV_DIR/bin/python"
TARGET="$ROOT/pystamps/pipeline/ported.py"
STAGE7="$ROOT/pystamps/pipeline/stage7_sbas.py"
STAGE8="$ROOT/pystamps/pipeline/stage8_sbas.py"
GACOS_MODULE_SOURCE="$SELF_DIR/gacos_correction.py"
GACOS_MODULE_TARGET="$ROOT/pystamps/pipeline/gacos_correction.py"
STAGE78_INSTALLER="$SELF_DIR/install_stage78_sbas_patch.sh"
MODE="${1:-install-run}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$DATASET/_run_logs"
SOCKET="stage78gacos"
SESSION="cangzhou_stage78_gacos"

usage() {
  cat <<'EOF'
Usage:
  PYSTAMPS_GACOS_DIR=/path/to/GACOS ./install_gacos_stage78_pipeline.sh install
  PYSTAMPS_GACOS_DIR=/path/to/GACOS ./install_gacos_stage78_pipeline.sh run
  PYSTAMPS_GACOS_DIR=/path/to/GACOS ./install_gacos_stage78_pipeline.sh install-run
  PYSTAMPS_GACOS_DIR=/path/to/GACOS ./install_gacos_stage78_pipeline.sh foreground

Modes:
  install      Install/patch Stage 7/8 and GACOS support only.
  run          Run GACOS correction, Stage 7 and Stage 8 in tmux.
  install-run  Install then run in tmux (default).
  foreground   Install then run in current terminal.

Important environment variables:
  PYSTAMPS_GACOS_DIR             Directory containing YYYYMMDD*.tif or YYYYMMDD.ztd
  PYSTAMPS_GACOS_FORMAT          auto | tif | ztd                (default auto)
  PYSTAMPS_GACOS_UNIT            auto | m | cm | mm              (default auto)
  PYSTAMPS_GACOS_PROJECTION      zenith | los                    (default zenith)
  PYSTAMPS_GACOS_SIGN            auto | subtract | add           (default auto)
  PYSTAMPS_GACOS_INCIDENCE_DEG   Scalar incidence angle in degrees
  PYSTAMPS_GACOS_INCIDENCE_TIF   Per-pixel incidence GeoTIFF, degrees or radians
  PYSTAMPS_GACOS_REBUILD         1 to rebuild phuw2_gacos.mat
  PYSTAMPS_GACOS_MIN_VALID_FRACTION  Minimum PS coverage per date (default 0.995)
  PYSTAMPS_ROOT
  PYSTAMPS_DATASET
  PYSTAMPS_ENV_DIR
EOF
}

[[ "$MODE" =~ ^(install|run|install-run|foreground)$ ]] || { usage; exit 2; }

check_base() {
  [[ -d "$ROOT" ]] || { echo "错误：找不到项目目录：$ROOT" >&2; exit 3; }
  [[ -d "$DATASET" ]] || { echo "错误：找不到数据集：$DATASET" >&2; exit 4; }
  [[ -x "$PYTHON" ]] || { echo "错误：找不到Python：$PYTHON" >&2; exit 5; }
  [[ -f "$TARGET" ]] || { echo "错误：找不到：$TARGET" >&2; exit 6; }
  [[ -f "$GACOS_MODULE_SOURCE" ]] || { echo "错误：脚本包缺少：$GACOS_MODULE_SOURCE" >&2; exit 7; }
  mkdir -p "$LOG_DIR"

  local gacos_dir="${PYSTAMPS_GACOS_DIR:-}"
  if [[ -z "$gacos_dir" ]]; then
    for candidate in "$DATASET/GACOS" "$DATASET/gacos" "$(dirname "$DATASET")/GACOS" "$(dirname "$DATASET")/gacos"; do
      if [[ -d "$candidate" ]]; then
        export PYSTAMPS_GACOS_DIR="$candidate"
        gacos_dir="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$gacos_dir" || ! -d "$gacos_dir" ]]; then
    echo "错误：未找到GACOS目录。请设置：" >&2
    echo "  PYSTAMPS_GACOS_DIR=/实际/GACOS/目录" >&2
    exit 8
  fi
  export PYSTAMPS_GACOS_DIR="$(readlink -f "$gacos_dir")"
}

ensure_stage78_patch() {
  if [[ -f "$STAGE7" && -f "$STAGE8" ]] && grep -q 'STAGE78_SBAS_DISPATCH_V1' "$TARGET"; then
    echo "Stage 7/8 SBAS补丁已经存在。"
    return
  fi
  [[ -x "$STAGE78_INSTALLER" ]] || chmod +x "$STAGE78_INSTALLER"
  echo "安装Stage 7/8 SBAS基础补丁..."
  PYSTAMPS_ROOT="$ROOT" \
  PYSTAMPS_DATASET="$DATASET" \
  PYSTAMPS_ENV_DIR="$ENV_DIR" \
    "$STAGE78_INSTALLER" install
}

install_gacos_patch() {
  echo "============================================================"
  echo "安装GACOS → Stage 7/8补丁"
  echo "============================================================"

  ensure_stage78_patch

  local backup_dir="$ROOT/_gacos_patch_backup/$STAMP"
  mkdir -p "$backup_dir"
  cp -a "$STAGE7" "$backup_dir/stage7_sbas.py"
  [[ -f "$GACOS_MODULE_TARGET" ]] && cp -a "$GACOS_MODULE_TARGET" "$backup_dir/gacos_correction.py"
  echo "代码备份：$backup_dir"

  cp -a "$GACOS_MODULE_SOURCE" "$GACOS_MODULE_TARGET"

  STAGE7="$STAGE7" "$PYTHON" - <<'PY_PATCH'
from __future__ import annotations

import ast
import os
from pathlib import Path

path = Path(os.environ["STAGE7"])
source = path.read_text(encoding="utf-8")
marker = "# === GACOS_STAGE7_INPUT_V1 ==="

helper = r'''

# === GACOS_STAGE7_INPUT_V1 ===
def _stage7_phase_input(root: Path) -> Path:
    enabled = os.environ.get(
        "PYSTAMPS_GACOS_STAGE7_ENABLE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return root / "phuw2.mat"

    from pystamps.pipeline.gacos_correction import ensure_gacos_corrected_phuw

    return ensure_gacos_corrected_phuw(root)
'''

if marker not in source:
    token = "\ndef stage7_sbas_calc_scla("
    position = source.find(token)
    if position < 0:
        raise SystemExit("Cannot find stage7_sbas_calc_scla in stage7_sbas.py")
    source = source[:position] + helper + source[position:]

old = 'read_mat_variables(root / "phuw2.mat", ("ph_uw",))["ph_uw"]'
new = 'read_mat_variables(_stage7_phase_input(root), ("ph_uw",))["ph_uw"]'
if old in source:
    source = source.replace(old, new, 1)
elif new not in source:
    raise SystemExit("Cannot patch Stage 7 phase input path")

ast.parse(source, filename=str(path))
tmp = path.with_suffix(path.suffix + ".gacos_tmp")
tmp.write_text(source, encoding="utf-8")
os.replace(tmp, path)
print("Stage 7 GACOS input dispatch installed.")
PY_PATCH

  "$PYTHON" -m py_compile \
    "$GACOS_MODULE_TARGET" \
    "$STAGE7" \
    "$STAGE8" \
    "$TARGET"

  PYTHONPATH="$ROOT" "$PYTHON" - <<'PY_IMPORT'
from pystamps.pipeline.gacos_correction import ensure_gacos_corrected_phuw
from pystamps.pipeline.ported import stage7_calc_scla, stage8_filter_scn
print("GACOS function:", ensure_gacos_corrected_phuw.__name__)
print("Stage 7:", stage7_calc_scla.__name__)
print("Stage 8:", stage8_filter_scn.__name__)
print("Import check: PASSED")
PY_IMPORT

  cat > "$ROOT/run_stage78_gacos.sh" <<'RUNNER'
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
RUNNER
  chmod +x "$ROOT/run_stage78_gacos.sh"
  echo "GACOS补丁安装完成。"
}

preflight() {
  echo "============================================================"
  echo "GACOS + Stage 7/8预检查"
  echo "============================================================"
  echo "项目：$ROOT"
  echo "数据：$DATASET"
  echo "GACOS：$PYSTAMPS_GACOS_DIR"
  echo

  PYSTAMPS_DATASET="$DATASET" PYTHONPATH="$ROOT" "$PYTHON" - <<'PY_PREFLIGHT'
from pathlib import Path
import os
import shutil
import numpy as np

from pystamps.io.mat import read_mat, read_mat_variables
from pystamps.pipeline.gacos_correction import (
    _day_labels,
    _load_config,
    discover_products,
)
from pystamps.pipeline.stage6_sbas import load_sbas_network

root = Path(os.environ["PYSTAMPS_DATASET"])
required = ("ps2.mat", "phuw2.mat", "bp2.mat", "ifgstd2.mat", "parms.mat")
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("Missing inputs: " + ", ".join(missing))

ps = read_mat(root / "ps2.mat")
n_ps = int(round(float(np.asarray(ps["n_ps"]).reshape(-1)[0])))
ph = np.asarray(read_mat_variables(root / "phuw2.mat", ("ph_uw",))["ph_uw"])
if ph.shape[0] != n_ps and ph.shape[1] == n_ps:
    ph = ph.T
n_ifg = ph.shape[1]
day, ifgday_ix, _, source = load_sbas_network(root, n_ifg)
dates = _day_labels(day)
config = _load_config(root)
products = discover_products(config.gacos_dir, config.product_format)
missing_dates = [date for date in dates if date not in products]

print("phuw2.ph_uw :", ph.shape)
print("acquisitions  :", len(dates), dates[0], "to", dates[-1])
print("ifgday_ix     :", np.asarray(ifgday_ix).shape)
print("network       :", source)
print("GACOS format  :", config.product_format)
print("GACOS products:", len(products))
print("matched dates :", len(dates) - len(missing_dates), "/", len(dates))
print("tif matched   :", sum(products[d].kind == "tif" for d in dates if d in products))
print("ztd matched   :", sum(products[d].kind == "ztd" for d in dates if d in products))
if missing_dates:
    print("missing dates  :", missing_dates[:30])
    raise SystemExit("All acquisition dates require GACOS products; temporal interpolation is disabled")

parms = read_mat(root / "parms.mat")
mean_inc = ps.get("mean_incidence", parms.get("mean_incidence"))
if mean_inc is None and not os.environ.get("PYSTAMPS_GACOS_INCIDENCE_DEG") and not os.environ.get("PYSTAMPS_GACOS_INCIDENCE_TIF"):
    raise SystemExit(
        "No incidence angle found. Set PYSTAMPS_GACOS_INCIDENCE_DEG or "
        "PYSTAMPS_GACOS_INCIDENCE_TIF."
    )

free = shutil.disk_usage(root).free
print(f"free disk      : {free / 1024**3:.1f} GiB")
if free < 8 * 1024**3:
    raise SystemExit("At least 8 GiB free disk is required")
print("Preflight: PASSED")
PY_PREFLIGHT
}

backup_stage78_outputs() {
  local backup="$DATASET/_stage78_gacos_backup/$STAMP"
  mkdir -p "$backup"
  local files=(
    scla2.mat scla_smooth2.mat scla_sb2.mat
    phuw_sm2.mat bp_sm2.mat mean_v.mat uw_space_time.mat
    stage7_sbas_debug.json stage8_sbas_debug.json
  )
  local moved=0
  for file in "${files[@]}"; do
    if [[ -e "$DATASET/$file" ]]; then
      mv "$DATASET/$file" "$backup/"
      moved=1
    fi
  done
  if (( moved == 1 )); then
    echo "旧Stage 7/8输出已移动到：$backup"
  else
    rmdir "$backup" 2>/dev/null || true
    echo "未发现旧Stage 7/8输出。"
  fi
}

run_foreground() {
  preflight
  backup_stage78_outputs
  export PYSTAMPS_ROOT="$ROOT"
  export PYSTAMPS_DATASET="$DATASET"
  export PYSTAMPS_ENV_DIR="$ENV_DIR"
  "$ROOT/run_stage78_gacos.sh"
}

run_tmux() {
  preflight
  backup_stage78_outputs

  local log="$LOG_DIR/stage78_gacos_${STAMP}.log"
  touch "$log"
  tmux -L "$SOCKET" kill-server 2>/dev/null || true

  # Preserve all current GACOS configuration variables in the tmux command.
  local env_args=(
    "PYSTAMPS_ROOT=$ROOT"
    "PYSTAMPS_DATASET=$DATASET"
    "PYSTAMPS_ENV_DIR=$ENV_DIR"
    "PYSTAMPS_GACOS_DIR=$PYSTAMPS_GACOS_DIR"
    "PYSTAMPS_GACOS_FORMAT=${PYSTAMPS_GACOS_FORMAT:-auto}"
    "PYSTAMPS_GACOS_UNIT=${PYSTAMPS_GACOS_UNIT:-auto}"
    "PYSTAMPS_GACOS_PROJECTION=${PYSTAMPS_GACOS_PROJECTION:-zenith}"
    "PYSTAMPS_GACOS_SIGN=${PYSTAMPS_GACOS_SIGN:-auto}"
    "PYSTAMPS_GACOS_STRICT_DATES=${PYSTAMPS_GACOS_STRICT_DATES:-1}"
    "PYSTAMPS_GACOS_REBUILD=${PYSTAMPS_GACOS_REBUILD:-0}"
    "PYSTAMPS_GACOS_MIN_VALID_FRACTION=${PYSTAMPS_GACOS_MIN_VALID_FRACTION:-0.995}"
  )
  [[ -n "${PYSTAMPS_GACOS_INCIDENCE_DEG:-}" ]] && env_args+=("PYSTAMPS_GACOS_INCIDENCE_DEG=$PYSTAMPS_GACOS_INCIDENCE_DEG")
  [[ -n "${PYSTAMPS_GACOS_INCIDENCE_TIF:-}" ]] && env_args+=("PYSTAMPS_GACOS_INCIDENCE_TIF=$PYSTAMPS_GACOS_INCIDENCE_TIF")

  local env_command=""
  local item
  for item in "${env_args[@]}"; do
    env_command+="$(printf '%q ' "$item")"
  done

  tmux -L "$SOCKET" new-session -d -s "$SESSION" \
    "env $env_command bash '$ROOT/run_stage78_gacos.sh' >> '$log' 2>&1"

  sleep 8
  if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    echo "============================================================"
    echo "GACOS + Stage 7/8已启动"
    echo "============================================================"
    echo "日志：$log"
    echo
    echo "进入窗口："
    echo "  tmux -L $SOCKET attach -t $SESSION"
    echo
    echo "查看日志："
    echo "  tail -f '$log'"
  else
    echo "任务启动后立即退出，日志如下：" >&2
    tail -n 200 "$log" >&2 || true
    exit 9
  fi
}

check_base
case "$MODE" in
  install)
    install_gacos_patch
    preflight
    ;;
  run)
    run_tmux
    ;;
  install-run)
    install_gacos_patch
    run_tmux
    ;;
  foreground)
    install_gacos_patch
    run_foreground
    ;;
esac
