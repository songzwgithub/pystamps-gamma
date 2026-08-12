#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"

WORKERS=20
CPUSET="0-31"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$DATASET/_run_logs"
LOG="$LOG_DIR/stage6_workers${WORKERS}_${STAMP}.log"
RUNNER="$LOG_DIR/stage6_workers${WORKERS}_${STAMP}.sh"

SOCKET="stage6max"
SESSION="cangzhou_stage6_max"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "终止当前Stage 6"
echo "============================================================"

# 1. 终止Stage 6父进程
mapfile -t PARENT_PIDS < <(
    pgrep -f \
      '[p]ython.*-m[[:space:]]+pystamps\.pipeline\.stage6_sbas' \
      || true
)

if (( ${#PARENT_PIDS[@]} > 0 )); then
    echo "Stage 6父进程：${PARENT_PIDS[*]}"
    kill -INT "${PARENT_PIDS[@]}" 2>/dev/null || true
fi

sleep 5

# 2. 终止当前数据集的SNAPHU子进程
SNAPHU_PIDS=()

while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue

    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

    if [[ "$cwd" == "$DATASET/_stage6_sbas_work/snaphu"/ifg_* ]]; then
        SNAPHU_PIDS+=("$pid")
    fi
done < <(pgrep -x snaphu || true)

if (( ${#SNAPHU_PIDS[@]} > 0 )); then
    echo "SNAPHU子进程：${SNAPHU_PIDS[*]}"
    kill -TERM "${SNAPHU_PIDS[@]}" 2>/dev/null || true
fi

sleep 5

# 3. 强制清理残留进程
mapfile -t REMAINING_PARENT < <(
    pgrep -f \
      '[p]ython.*-m[[:space:]]+pystamps\.pipeline\.stage6_sbas' \
      || true
)

if (( ${#REMAINING_PARENT[@]} > 0 )); then
    kill -KILL "${REMAINING_PARENT[@]}" 2>/dev/null || true
fi

for pid in "${SNAPHU_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

# 4. 清理旧tmux
tmux -L stage6w8 kill-server 2>/dev/null || true
tmux -L stage6w16 kill-server 2>/dev/null || true
tmux -L stage6w20 kill-server 2>/dev/null || true
tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux kill-session \
  -t cangzhou_stage6_quick \
  2>/dev/null || true

sleep 2

echo
echo "============================================================"
echo "运行环境检查"
echo "============================================================"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到Python：$PYTHON"
    exit 2
fi

if ! command -v snaphu >/dev/null 2>&1; then
    echo "错误：找不到snaphu"
    exit 3
fi

if ! command -v taskset >/dev/null 2>&1; then
    echo "错误：找不到taskset"
    exit 4
fi

"$PYTHON" - <<'PY'
import numpy as np
import scipy

print("NumPy :", np.__version__)
print("SciPy :", scipy.__version__)
print("环境检查通过")
PY

echo
lscpu | grep -E \
'^CPU\(s\):|^On-line CPU|Thread|Core|Socket' \
|| true

echo
free -h

# 5. 创建启动器
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec >> "$LOG" 2>&1

echo "============================================================"
echo "Stage 6最大安全负载运行"
echo "启动时间       ：\$(date)"
echo "SNAPHU并发     ：$WORKERS"
echo "允许CPU        ：$CPUSET"
echo "GRID FFT线程   ：32"
echo "============================================================"

export PATH="/home/ubuntu/software/miniconda3/envs/stamps/bin:/usr/bin:/usr/local/bin:/bin"
export PYTHONPATH="$ROOT"
export REAL_DATASET="$DATASET"

# GRID使用全部32个CPU
export PYSTAMPS_STAGE6_GRID_RESUME=1
export PYSTAMPS_STAGE6_GRID_IFG_BATCH=4
export PYSTAMPS_STAGE6_GRID_WINDOW_BATCH=32
export PYSTAMPS_STAGE6_GRID_FFT_WORKERS=32

# TIME采用3D_QUICK
export PYSTAMPS_SBAS_EDGE_CHUNK=8192
export PYSTAMPS_SBAS_STRICT_ANNEAL=0
export PYSTAMPS_SBAS_ANNEAL_RUNS=1
export PYSTAMPS_SBAS_ANNEAL_WORKERS=1

# SNAPHU并发
export PYSTAMPS_STAGE6_SNAPHU_WORKERS=$WORKERS

# 保存中间结果
export PYSTAMPS_SBAS_KEEP_WORK=1
export PYSTAMPS_SBAS_PROGRESS=1

# 防止每个外层进程内部嵌套BLAS线程
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export BLIS_NUM_THREADS=1

export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

ulimit -n 65535 2>/dev/null || true

cd "$ROOT"

echo "Python        ：$PYTHON"
echo "SNAPHU       ：\$(command -v snaphu)"
echo "NumPy        ：\$("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
echo "SNAPHU workers：\$PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo

exec taskset -c "$CPUSET" \
  "$PYTHON" \
  -m pystamps.pipeline.stage6_sbas \
  --dataset "$DATASET" \
  --io-workers 1
EOF

chmod +x "$RUNNER"
bash -n "$RUNNER"

touch "$LOG"

# 6. 启动tmux
tmux -L "$SOCKET" \
  new-session \
  -d \
  -s "$SESSION" \
  "bash '$RUNNER'"

sleep 10

echo
if tmux -L "$SOCKET" \
    has-session \
    -t "$SESSION" \
    2>/dev/null
then
    echo "============================================================"
    echo "Stage 6已启动"
    echo "============================================================"
    echo "SNAPHU并发：$WORKERS"
    echo "CPU范围：$CPUSET"
    echo
    echo "进入tmux："
    echo "tmux -L $SOCKET attach -t $SESSION"
    echo
    echo "查看日志："
    echo "tail -f '$LOG'"
else
    echo "启动后立即退出，日志末尾："
    tail -n 200 "$LOG" || true
    exit 5
fi
