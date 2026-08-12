#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
LOCAL_BIN="$ROOT/.build-deps/bin"
LOG="$DATASET/_run_logs/stage6_3d_quick_resume.log"

mkdir -p "$LOCAL_BIN"
mkdir -p "$DATASET/_run_logs"

echo "============================================================"
echo "1. 检查SNAPHU"
echo "============================================================"

if command -v snaphu >/dev/null 2>&1; then
    SNAPHU_BIN="$(command -v snaphu)"
    echo "已安装：$SNAPHU_BIN"
else
    echo "当前PATH中没有snaphu，开始安装。"

    APT_OK=0

    if command -v apt-get >/dev/null 2>&1; then
        echo
        echo "尝试使用Ubuntu软件源安装..."

        sudo apt-get update

        if command -v add-apt-repository >/dev/null 2>&1; then
            sudo add-apt-repository -y multiverse || true
            sudo apt-get update
        fi

        if sudo apt-get install -y snaphu; then
            APT_OK=1
        fi
    fi

    if command -v snaphu >/dev/null 2>&1; then
        SNAPHU_BIN="$(command -v snaphu)"
    else
        echo
        echo "APT安装未得到可执行文件，转为编译官方SNAPHU 2.0.7。"

        if ! command -v gcc >/dev/null 2>&1 \
           || ! command -v make >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y build-essential curl
        fi

        TMPDIR_SNAPHU="$(mktemp -d)"
        trap 'rm -rf "$TMPDIR_SNAPHU"' EXIT

        ARCHIVE="$TMPDIR_SNAPHU/snaphu-v2.0.7.tar.gz"

        if command -v curl >/dev/null 2>&1; then
            curl -fL \
              "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.7.tar.gz" \
              -o "$ARCHIVE"
        elif command -v wget >/dev/null 2>&1; then
            wget \
              "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.7.tar.gz" \
              -O "$ARCHIVE"
        else
            sudo apt-get install -y curl
            curl -fL \
              "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.7.tar.gz" \
              -o "$ARCHIVE"
        fi

        tar -xzf "$ARCHIVE" -C "$TMPDIR_SNAPHU"

        SRC_DIR="$(
            find "$TMPDIR_SNAPHU" \
              -maxdepth 2 \
              -type d \
              -name 'snaphu-v*' \
              | head -n 1
        )"

        if [[ -z "$SRC_DIR" || ! -d "$SRC_DIR/src" ]]; then
            echo "错误：无法找到解压后的SNAPHU源码目录。" >&2
            exit 2
        fi

        echo "源码目录：$SRC_DIR"

        make \
          -C "$SRC_DIR/src" \
          -j"$(nproc)"

        BUILT_BIN="$(
            find "$SRC_DIR" \
              -type f \
              -name snaphu \
              -perm -111 \
              | head -n 1
        )"

        if [[ -z "$BUILT_BIN" ]]; then
            echo "错误：编译完成后未找到snaphu可执行文件。" >&2
            exit 3
        fi

        cp -f "$BUILT_BIN" "$LOCAL_BIN/snaphu"
        chmod +x "$LOCAL_BIN/snaphu"

        SNAPHU_BIN="$LOCAL_BIN/snaphu"
    fi
fi

echo
echo "============================================================"
echo "2. 验证SNAPHU"
echo "============================================================"

if [[ ! -x "$SNAPHU_BIN" ]]; then
    echo "错误：SNAPHU不可执行：$SNAPHU_BIN" >&2
    exit 4
fi

export PATH="$(dirname "$SNAPHU_BIN"):$PATH"

echo "SNAPHU路径：$(command -v snaphu)"
echo

snaphu 2>&1 | head -n 15 || true

echo
echo "验证pySTAMPS可以解析SNAPHU："

python - <<'PY'
from pystamps.pipeline import ported

resolved = ported._resolve_external_tool(
    "snaphu",
    None,
)

print("resolved snaphu:", resolved)
PY

echo
echo "============================================================"
echo "3. 检查GRID v2检查点"
echo "============================================================"

GRID_META="$DATASET/_stage6_sbas_work/grid_v2/meta.json"

if [[ -f "$GRID_META" ]]; then
    cat "$GRID_META"
else
    echo "警告：没有找到GRID v2 meta.json；Stage 6可能重新执行GRID。"
fi

echo
echo "保留以下检查点，不删除："
find "$DATASET/_stage6_sbas_work/grid_v2" \
  -maxdepth 1 \
  -type f \
  -printf '  %f  %s bytes\n' \
  2>/dev/null \
  || true

echo
echo "============================================================"
echo "4. 写入带SNAPHU路径的恢复启动脚本"
echo "============================================================"

cat > "$ROOT/run_stage6_quick_resume.sh" <<RUN
#!/usr/bin/env bash
set -euo pipefail

export PATH="$(dirname "$SNAPHU_BIN"):\$PATH"

export REAL_DATASET="$DATASET"

export PYSTAMPS_STAGE6_GRID_RESUME=1
export PYSTAMPS_STAGE6_GRID_IFG_BATCH=4
export PYSTAMPS_STAGE6_GRID_WINDOW_BATCH=32
export PYSTAMPS_STAGE6_GRID_FFT_WORKERS=16

export PYSTAMPS_SBAS_EDGE_CHUNK=8192
export PYSTAMPS_SBAS_STRICT_ANNEAL=0
export PYSTAMPS_SBAS_ANNEAL_RUNS=1
export PYSTAMPS_SBAS_ANNEAL_WORKERS=1

export PYSTAMPS_STAGE6_SNAPHU_WORKERS=4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"

exec ./run_stage6_fast.sh
RUN

chmod +x "$ROOT/run_stage6_quick_resume.sh"

echo
echo "============================================================"
echo "5. 启动Stage 6恢复任务"
echo "============================================================"

tmux kill-session \
  -t cangzhou_stage6_quick \
  2>/dev/null \
  || true

tmux new-session \
  -d \
  -s cangzhou_stage6_quick \
  "cd '$ROOT' && \
   ./run_stage6_quick_resume.sh \
   2>&1 | tee '$LOG'"

sleep 3

if tmux has-session \
    -t cangzhou_stage6_quick \
    2>/dev/null
then
    echo "Stage 6已启动。"
    echo
    echo "查看："
    echo "  tmux attach -t cangzhou_stage6_quick"
    echo
    echo "日志："
    echo "  $LOG"
else
    echo "Stage 6启动后立即退出，查看日志：" >&2
    echo "  tail -n 100 '$LOG'" >&2
    exit 5
fi
