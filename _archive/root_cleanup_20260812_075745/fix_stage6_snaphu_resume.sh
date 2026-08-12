#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
TARGET="$ROOT/pystamps/pipeline/stage6_sbas.py"

DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
WORK_ROOT="$DATASET/_stage6_sbas_work"
SNAPHU_ROOT="$WORK_ROOT/snaphu"
LOG_DIR="$DATASET/_run_logs"

ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${TARGET}.bak_snaphu_resume_${STAMP}"
LAUNCHER="$ROOT/run_stage6_snaphu_resume8.sh"

MODE="${1:-}"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "修复Stage 6 SNAPHU断点续算"
echo "============================================================"
echo "源码：$TARGET"
echo "数据：$DATASET"
echo

if [[ ! -f "$TARGET" ]]; then
    echo "错误：找不到源码文件：$TARGET" >&2
    exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到stamps环境Python：$PYTHON" >&2
    exit 3
fi

# ----------------------------------------------------------------------
# 1. 备份并修改stage6_sbas.py
# ----------------------------------------------------------------------

if grep -q 'STAGE6_SNAPHU_RESUME_V1' "$TARGET"; then
    echo "断点续算补丁已经存在，跳过重复修改。"
else
    cp -a "$TARGET" "$BACKUP"
    echo "源码备份：$BACKUP"

    TARGET="$TARGET" /usr/bin/python3 <<'PY'
from __future__ import annotations

import ast
import os
from pathlib import Path

path = Path(os.environ["TARGET"])
source = path.read_text(encoding="utf-8")

start_token = "def _run_snaphu_column("
end_token = "\n\n# === STAGE6_SBAS_GRID_BATCH_V2 ==="

start = source.find(start_token)
if start < 0:
    raise SystemExit(
        "没有找到 def _run_snaphu_column(...)，"
        "当前stage6_sbas.py结构与补丁预期不一致。"
    )

end = source.find(end_token, start)
if end < 0:
    raise SystemExit(
        "没有找到 STAGE6_SBAS_GRID_BATCH_V2 标记，"
        "拒绝自动修改。"
    )

replacement = r'''# === STAGE6_SNAPHU_RESUME_V1 ===

def _stage6_read_snaphu_result(
    output_path: Path,
    *,
    ncol: int,
    nzix: np.ndarray,
    ported: Any,
) -> tuple[np.ndarray, float, int]:
    """
    Validate and read one complete SNAPHU output.

    Validation includes:
      1. exact float32 raster byte size;
      2. expected raster dimensions;
      3. finite output values;
      4. non-empty/non-constant PS-grid output.

    Returns
    -------
    values
        Unwrapped phase at non-zero grid cells.
    msd
        Mean squared spatial phase difference.
    expected_bytes
        Expected output byte size.
    """

    if not output_path.is_file():
        raise Stage6SbasError(
            f"SNAPHU output does not exist: {output_path}"
        )

    nzix_array = np.asarray(nzix)
    grid_size = int(nzix_array.size)

    if int(ncol) <= 0:
        raise Stage6SbasError(
            f"Invalid SNAPHU column count: {ncol}"
        )

    if grid_size % int(ncol) != 0:
        raise Stage6SbasError(
            "SNAPHU grid size is not divisible by ncol: "
            f"grid_size={grid_size}, ncol={ncol}"
        )

    expected_rows = grid_size // int(ncol)
    expected_bytes = (
        expected_rows
        * int(ncol)
        * np.dtype(np.float32).itemsize
    )
    actual_bytes = output_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise Stage6SbasError(
            "Incomplete SNAPHU output: "
            f"{output_path}, bytes={actual_bytes}, "
            f"expected={expected_bytes}"
        )

    ifguw = np.asarray(
        ported._load_float_grid(output_path, int(ncol)),
        dtype=np.float32,
    )

    if ifguw.shape != (expected_rows, int(ncol)):
        raise Stage6SbasError(
            "SNAPHU output shape mismatch: "
            f"{output_path}, shape={ifguw.shape}, "
            f"expected=({expected_rows}, {ncol})"
        )

    if not np.all(np.isfinite(ifguw)):
        raise Stage6SbasError(
            f"SNAPHU output contains NaN or Inf: {output_path}"
        )

    values = ported._extract_grid_values_for_ps(
        ifguw,
        nzix_array,
    ).astype(np.float32)

    if values.size == 0:
        raise Stage6SbasError(
            f"SNAPHU output contains no grid values: {output_path}"
        )

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise Stage6SbasError(
            f"SNAPHU PS-grid values are all invalid: {output_path}"
        )

    # A killed SNAPHU process may leave a preallocated all-zero file.
    # Reject such files even when the byte size appears complete.
    value_span = float(
        np.ptp(finite_values.astype(np.float64))
    )
    max_abs = float(
        np.max(np.abs(finite_values.astype(np.float64)))
    )

    if value_span <= 1.0e-7 and max_abs <= 1.0e-7:
        raise Stage6SbasError(
            "SNAPHU output is effectively all zero and is treated "
            f"as incomplete: {output_path}"
        )

    # Calculate the original StaMPS-compatible MSD, but process it
    # in row blocks to avoid allocating multiple full-grid arrays.
    sum_squares = 0.0
    denominator = 0
    row_chunk = 256

    nrow = int(ifguw.shape[0])

    for row_start in range(0, max(0, nrow - 1), row_chunk):
        row_stop = min(row_start + row_chunk, nrow - 1)

        diff = (
            ifguw[row_start:row_stop, :]
            - ifguw[row_start + 1:row_stop + 1, :]
        )
        nonzero = diff != 0

        if np.any(nonzero):
            selected = diff[nonzero].astype(
                np.float64,
                copy=False,
            )
            sum_squares += float(
                np.dot(selected, selected)
            )
            denominator += int(selected.size)

    for row_start in range(0, nrow, row_chunk):
        row_stop = min(row_start + row_chunk, nrow)

        diff = (
            ifguw[row_start:row_stop, :-1]
            - ifguw[row_start:row_stop, 1:]
        )
        nonzero = diff != 0

        if np.any(nonzero):
            selected = diff[nonzero].astype(
                np.float64,
                copy=False,
            )
            sum_squares += float(
                np.dot(selected, selected)
            )
            denominator += int(selected.size)

    msd = (
        sum_squares / float(denominator)
        if denominator > 0
        else 0.0
    )

    return values, float(msd), int(expected_bytes)


def _stage6_invalid_output_path(
    ifg_dir: Path,
    base_name: str,
) -> Path:
    """Return a non-conflicting path for an invalid partial result."""

    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = ifg_dir / f"{base_name}.invalid_{stamp}"
    sequence = 1

    while candidate.exists():
        candidate = (
            ifg_dir
            / f"{base_name}.invalid_{stamp}_{sequence:02d}"
        )
        sequence += 1

    return candidate


def _run_snaphu_column(
    *,
    index: int,
    work_root: Path,
    snaphu_exe: str,
    ncol: int,
    uw_ph: np.ndarray,
    Z: np.ndarray,
    nzix: np.ndarray,
    rowix: np.ndarray,
    colix: np.ndarray,
    nzrowix: np.ndarray,
    nzcolix: np.ndarray,
    rowcost_base: np.ndarray,
    colcost_base: np.ndarray,
    dph_noise: np.ndarray,
    dph_space_uw: np.ndarray,
    nshortcycle: float,
    ported: Any,
) -> tuple[int, np.ndarray, float]:
    """
    Run or resume one SNAPHU interferogram.

    A valid existing snaphu.out is reused. New output is first written
    to snaphu.out.partial and atomically renamed only after validation.
    """

    ifg_number = int(index) + 1
    ifg_dir = work_root / f"ifg_{ifg_number:04d}"
    ifg_dir.mkdir(parents=True, exist_ok=True)

    output_path = ifg_dir / "snaphu.out"
    partial_path = ifg_dir / "snaphu.out.partial"
    marker_path = ifg_dir / "snaphu.complete.json"

    resume_enabled = _env_bool(
        "PYSTAMPS_STAGE6_SNAPHU_RESUME",
        True,
    )

    # --------------------------------------------------------------
    # Resume a previously completed interferogram.
    # --------------------------------------------------------------
    if resume_enabled and output_path.is_file():
        try:
            values, msd, expected_bytes = (
                _stage6_read_snaphu_result(
                    output_path,
                    ncol=ncol,
                    nzix=nzix,
                    ported=ported,
                )
            )

            marker_payload = {
                "version": 1,
                "status": "completed",
                "ifg_index_1based": ifg_number,
                "output": str(output_path),
                "expected_bytes": expected_bytes,
                "actual_bytes": output_path.stat().st_size,
                "resumed": True,
                "updated_epoch_sec": time.time(),
            }
            _write_json(marker_path, marker_payload)

            return index, values, msd

        except Exception as exc:
            invalid_path = _stage6_invalid_output_path(
                ifg_dir,
                "snaphu.out",
            )
            output_path.replace(invalid_path)

            try:
                marker_path.unlink()
            except FileNotFoundError:
                pass

            print(
                "[STAGE6_SBAS][SNAPHU_RESUME] "
                f"IFG {ifg_number}: existing output rejected "
                f"and moved to {invalid_path.name}; "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )

    # Never treat a previous temporary file as complete.
    if partial_path.exists():
        invalid_partial = _stage6_invalid_output_path(
            ifg_dir,
            "snaphu.out.partial",
        )
        partial_path.replace(invalid_partial)

    conf = ifg_dir / "snaphu.conf"

    partial_conf_text = "\n".join(
        (
            "INFILE  snaphu.in",
            "OUTFILE snaphu.out.partial",
            "COSTINFILE snaphu.costinfile",
            "STATCOSTMODE  DEFO",
            "INFILEFORMAT  COMPLEX_DATA",
            "OUTFILEFORMAT FLOAT_DATA",
            "",
        )
    )
    conf.write_text(
        partial_conf_text,
        encoding="utf-8",
    )

    rowcost = rowcost_base.copy()
    colcost = colcost_base.copy()

    smooth = (
        np.asarray(
            dph_space_uw[:, index],
            dtype=np.float64,
        )
        - np.asarray(
            dph_noise[:, index],
            dtype=np.float64,
        )
    )

    wrapped = np.angle(
        np.exp(
            1j
            * np.asarray(
                dph_space_uw[:, index],
                dtype=np.float64,
            )
        )
    )
    offset_cycle = (wrapped - smooth) / TWO_PI

    offgrid = np.zeros(rowix.shape, dtype=np.int16)
    edge_index = (
        np.abs(rowix[nzrowix]).astype(np.int64) - 1
    )
    offgrid[nzrowix] = np.rint(
        offset_cycle[edge_index]
        * np.sign(rowix[nzrowix])
        * nshortcycle
    ).astype(np.int16)
    rowcost[:, 0::4] = -offgrid

    offgrid = np.zeros(colix.shape, dtype=np.int16)
    edge_index = (
        np.abs(colix[nzcolix]).astype(np.int64) - 1
    )
    offgrid[nzcolix] = np.rint(
        offset_cycle[edge_index]
        * np.sign(colix[nzcolix])
        * nshortcycle
    ).astype(np.int16)
    colcost[:, 0::4] = offgrid

    cost_path = ifg_dir / "snaphu.costinfile"
    ported._write_binary_matrix(cost_path, rowcost)

    with cost_path.open("ab") as handle:
        ported._write_binary_matrix(handle, colcost)

    ifgw = np.asarray(
        uw_ph[Z - 1, index],
        dtype=np.complex64,
    )
    ported._write_complex_raster(
        ifg_dir / "snaphu.in",
        ifgw,
    )

    try:
        ported._run_external_command(
            [
                snaphu_exe,
                "-d",
                "-f",
                "snaphu.conf",
                str(ncol),
            ],
            cwd=ifg_dir,
            log_path=ifg_dir / "snaphu.log",
        )

        values, msd, expected_bytes = (
            _stage6_read_snaphu_result(
                partial_path,
                ncol=ncol,
                nzix=nzix,
                ported=ported,
            )
        )

        # Atomic publication of the completed result.
        partial_path.replace(output_path)

        # Keep the final config consistent with the published filename.
        final_conf_text = partial_conf_text.replace(
            "OUTFILE snaphu.out.partial",
            "OUTFILE snaphu.out",
        )
        conf.write_text(
            final_conf_text,
            encoding="utf-8",
        )

        _write_json(
            marker_path,
            {
                "version": 1,
                "status": "completed",
                "ifg_index_1based": ifg_number,
                "output": str(output_path),
                "expected_bytes": expected_bytes,
                "actual_bytes": output_path.stat().st_size,
                "resumed": False,
                "updated_epoch_sec": time.time(),
            },
        )

        return index, values, msd

    except Exception:
        # Keep the partial file for diagnosis, but never expose it as
        # a completed snaphu.out.
        if partial_path.exists():
            invalid_partial = _stage6_invalid_output_path(
                ifg_dir,
                "snaphu.out.partial",
            )
            partial_path.replace(invalid_partial)
        raise
'''

new_source = (
    source[:start]
    + replacement
    + source[end:]
)

# Validate Python syntax before modifying the original file.
ast.parse(new_source, filename=str(path))

temporary = path.with_name(
    path.name + ".snaphu_resume_tmp"
)
temporary.write_text(
    new_source,
    encoding="utf-8",
)
os.replace(temporary, path)

print("已写入STAGE6_SNAPHU_RESUME_V1补丁。")
PY
fi

# ----------------------------------------------------------------------
# 2. 语法检查
# ----------------------------------------------------------------------

echo
echo "检查Python语法..."

"$PYTHON" -m py_compile "$TARGET"

grep -n \
    'STAGE6_SNAPHU_RESUME_V1\|snaphu.out.partial\|PYSTAMPS_STAGE6_SNAPHU_RESUME' \
    "$TARGET" \
    | head -n 20

echo "源码语法检查通过。"

# ----------------------------------------------------------------------
# 3. 创建专用恢复启动器
# ----------------------------------------------------------------------

cat > "$LAUNCHER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DEFAULT_DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
DATASET="${1:-$DEFAULT_DATASET}"

ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到Python：$PYTHON" >&2
    exit 2
fi

if [[ ! -x /usr/bin/snaphu ]]; then
    echo "错误：找不到/usr/bin/snaphu" >&2
    exit 3
fi

export PATH="$ENV_DIR/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"

export REAL_DATASET="$DATASET"

export PYSTAMPS_STAGE6_GRID_RESUME=1
export PYSTAMPS_STAGE6_GRID_IFG_BATCH=4
export PYSTAMPS_STAGE6_GRID_WINDOW_BATCH=32
export PYSTAMPS_STAGE6_GRID_FFT_WORKERS=16

export PYSTAMPS_SBAS_EDGE_CHUNK=8192
export PYSTAMPS_SBAS_STRICT_ANNEAL=0
export PYSTAMPS_SBAS_ANNEAL_RUNS=1
export PYSTAMPS_SBAS_ANNEAL_WORKERS=1

# SNAPHU干涉图级断点续算。
export PYSTAMPS_STAGE6_SNAPHU_RESUME=1
export PYSTAMPS_STAGE6_SNAPHU_WORKERS="${PYSTAMPS_STAGE6_SNAPHU_WORKERS:-8}"

# 完成后保留工作目录和断点文件。
export PYSTAMPS_SBAS_KEEP_WORK=1

# 避免8个SNAPHU任务再嵌套BLAS线程。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "============================================================"
echo "Stage 6 SBAS断点恢复"
echo "============================================================"
echo "Dataset       : $DATASET"
echo "Python        : $PYTHON"
echo "SNAPHU        : /usr/bin/snaphu"
echo "SNAPHU resume : $PYSTAMPS_STAGE6_SNAPHU_RESUME"
echo "SNAPHU workers: $PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo "GRID resume   : $PYSTAMPS_STAGE6_GRID_RESUME"
echo "============================================================"

exec "$PYTHON" \
    -m pystamps.pipeline.stage6_sbas \
    --dataset "$DATASET" \
    --snaphu /usr/bin/snaphu \
    --io-workers 1
RUNNER

chmod +x "$LAUNCHER"
bash -n "$LAUNCHER"

echo
echo "恢复启动器：$LAUNCHER"

# ----------------------------------------------------------------------
# 4. 当前运行状态
# ----------------------------------------------------------------------

mapfile -t STAGE6_PIDS < <(
    for pid in $(pgrep -f '[p]ython.*pystamps\.pipeline\.stage6_sbas' || true); do
        if [[ -r "/proc/$pid/cmdline" ]]; then
            cmd="$(
                tr '\0' ' ' < "/proc/$pid/cmdline"
            )"
            if [[ "$cmd" == *"$DATASET"* ]]; then
                echo "$pid"
            fi
        fi
    done
)

if [[ "$MODE" != "--restart" ]]; then
    echo
    echo "============================================================"
    echo "补丁安装完成"
    echo "============================================================"

    if (( ${#STAGE6_PIDS[@]} > 0 )); then
        echo "当前Stage 6仍在运行，未进行中断："
        ps -fp "${STAGE6_PIDS[@]}" || true
        echo
        echo "当前进程仍使用启动时加载的旧代码。"
        echo "下次启动使用："
        echo
        echo "  $LAUNCHER"
    else
        echo "当前没有检测到Stage 6进程。"
        echo "启动恢复任务："
        echo
        echo "  $LAUNCHER"
    fi

    echo
    echo "立即中断当前任务并断点恢复："
    echo
    echo "  ./fix_stage6_snaphu_resume.sh --restart"
    exit 0
fi

# ----------------------------------------------------------------------
# 5. --restart：停止当前Stage 6并恢复
# ----------------------------------------------------------------------

echo
echo "============================================================"
echo "停止当前Stage 6"
echo "============================================================"

if (( ${#STAGE6_PIDS[@]} > 0 )); then
    echo "终止Stage 6父进程：${STAGE6_PIDS[*]}"
    kill -TERM "${STAGE6_PIDS[@]}" 2>/dev/null || true
fi

# 只终止工作目录位于当前数据集SNAPHU目录内的进程。
SNAPHU_PIDS=()

for pid in $(pgrep -x snaphu || true); do
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

    if [[ "$cwd" == "$SNAPHU_ROOT"* ]]; then
        SNAPHU_PIDS+=("$pid")
    fi
done

if (( ${#SNAPHU_PIDS[@]} > 0 )); then
    echo "终止本数据集SNAPHU进程：${SNAPHU_PIDS[*]}"
    kill -TERM "${SNAPHU_PIDS[@]}" 2>/dev/null || true
fi

for _ in $(seq 1 15); do
    ACTIVE=0

    for pid in "${STAGE6_PIDS[@]}" "${SNAPHU_PIDS[@]}"; do
        [[ -n "${pid:-}" ]] || continue

        if kill -0 "$pid" 2>/dev/null; then
            ACTIVE=1
            break
        fi
    done

    (( ACTIVE == 0 )) && break
    sleep 1
done

for pid in "${STAGE6_PIDS[@]}" "${SNAPHU_PIDS[@]}"; do
    [[ -n "${pid:-}" ]] || continue

    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

echo "旧任务已停止。"

# 保存残留临时文件，不删除诊断资料。
PARTIAL_BACKUP="$DATASET/_stage6_partial_backup/snaphu_${STAMP}"
mkdir -p "$PARTIAL_BACKUP"

while IFS= read -r -d '' partial; do
    relative="${partial#"$SNAPHU_ROOT"/}"
    destination="$PARTIAL_BACKUP/$relative"

    mkdir -p "$(dirname "$destination")"
    mv "$partial" "$destination"
done < <(
    find "$SNAPHU_ROOT" \
        -type f \
        -name 'snaphu.out.partial' \
        -print0 \
        2>/dev/null
)

echo "残留partial备份：$PARTIAL_BACKUP"

# ----------------------------------------------------------------------
# 6. 独立tmux启动
# ----------------------------------------------------------------------

SOCKET="stage6resume"
SESSION="cangzhou_stage6_resume"
LOG="$LOG_DIR/stage6_snaphu_resume_${STAMP}.log"

touch "$LOG"

tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux -L "$SOCKET" \
    new-session \
    -d \
    -s "$SESSION" \
    "PYSTAMPS_STAGE6_SNAPHU_WORKERS=8 bash '$LAUNCHER' '$DATASET' >> '$LOG' 2>&1"

sleep 10

echo
if tmux -L "$SOCKET" \
    has-session \
    -t "$SESSION" \
    2>/dev/null
then
    echo "============================================================"
    echo "Stage 6断点恢复已启动"
    echo "============================================================"
    echo "日志：$LOG"
    echo
    echo "进入会话："
    echo "  tmux -L $SOCKET attach -t $SESSION"
    echo
    echo "查看日志："
    echo "  tail -f '$LOG'"
else
    echo "Stage 6启动后立即退出，日志如下：" >&2
    tail -n 200 "$LOG" >&2 || true
    exit 5
fi
