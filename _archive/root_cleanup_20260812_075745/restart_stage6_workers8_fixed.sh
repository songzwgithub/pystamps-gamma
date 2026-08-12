#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
SNAPHU_WORK="$DATASET/_stage6_sbas_work/snaphu"
STAMP="$(date +%Y%m%d_%H%M%S)"

SOCKET="stage6w8"
SESSION="cangzhou_stage6_quick"

LOG_DIR="$DATASET/_run_logs"
LOG="$LOG_DIR/stage6_workers8_${STAMP}.log"
WRAPPER="$LOG_DIR/stage6_workers8_${STAMP}.sh"

mkdir -p "$LOG_DIR"
touch "$LOG"

cd "$ROOT"

echo "============================================================"
echo "Stage 6 SNAPHU 8并发恢复"
echo "============================================================"

# 1. 基本检查
for script in \
    "$ROOT/run_stage6_fast.sh" \
    "$ROOT/run_stage6_quick_resume.sh"
do
    if [[ ! -s "$script" ]]; then
        echo "错误：缺少启动脚本：$script" | tee -a "$LOG"
        exit 2
    fi

    bash -n "$script"
done

SNAPHU_BIN="$(command -v snaphu || true)"

if [[ -z "$SNAPHU_BIN" ]]; then
    echo "错误：当前PATH找不到snaphu。" | tee -a "$LOG"
    exit 3
fi

echo "SNAPHU：$SNAPHU_BIN" | tee -a "$LOG"

# 2. 防止重复运行
mapfile -t EXISTING < <(
    pgrep -f \
      '[p]ython.*pystamps\.pipeline\.stage6_sbas|[/]snaphu -d -f snaphu.conf' \
      || true
)

if (( ${#EXISTING[@]} > 0 )); then
    echo "检测到已有Stage 6相关进程，拒绝重复启动：" | tee -a "$LOG"
    ps -fp "${EXISTING[@]}" | tee -a "$LOG" || true
    exit 4
fi

# 3. 统计完整SNAPHU结果
# 从现有结果中取得最常见的文件大小，避免写死网格尺寸。
EXPECTED_BYTES="$(
    find "$SNAPHU_WORK" \
      -type f \
      -name snaphu.out \
      -printf '%s\n' \
      2>/dev/null \
    | sort \
    | uniq -c \
    | sort -nr \
    | awk 'NR==1 {print $2}'
)"

if [[ -z "$EXPECTED_BYTES" ]]; then
    echo "警告：没有找到现有snaphu.out，无法统计断点结果。" | tee -a "$LOG"
    COMPLETE=0
else
    COMPLETE="$(
        find "$SNAPHU_WORK" \
          -type f \
          -name snaphu.out \
          -printf '%s\n' \
          2>/dev/null \
        | awk -v expected="$EXPECTED_BYTES" '
            $1 == expected {n++}
            END {print n + 0}
          '
    )"
fi

echo "完整输出大小：${EXPECTED_BYTES:-未知}" | tee -a "$LOG"
echo "已完成SNAPHU：$COMPLETE / 763" | tee -a "$LOG"
echo "日志：$LOG" | tee -a "$LOG"

# 4. 创建真正运行的包装脚本
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec >> "$LOG" 2>&1

echo
echo "============================================================"
echo "Stage 6恢复启动时间：\$(date)"
echo "============================================================"

export PATH="$(dirname "$SNAPHU_BIN"):/usr/local/bin:/usr/bin:/bin:\$PATH"
export REAL_DATASET="$DATASET"

export PYSTAMPS_STAGE6_GRID_RESUME=1
export PYSTAMPS_STAGE6_GRID_IFG_BATCH=4
export PYSTAMPS_STAGE6_GRID_WINDOW_BATCH=32
export PYSTAMPS_STAGE6_GRID_FFT_WORKERS=16

export PYSTAMPS_SBAS_EDGE_CHUNK=8192
export PYSTAMPS_SBAS_STRICT_ANNEAL=0
export PYSTAMPS_SBAS_ANNEAL_RUNS=1
export PYSTAMPS_SBAS_ANNEAL_WORKERS=1

export PYSTAMPS_STAGE6_SNAPHU_WORKERS=8

# 每个SNAPHU进程单核运行，避免8进程再嵌套BLAS线程。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "PATH=\$PATH"
echo "snaphu=\$(command -v snaphu)"
echo "workers=\$PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo

exec bash "$ROOT/run_stage6_quick_resume.sh"
EOF

chmod +x "$WRAPPER"
bash -n "$WRAPPER"

# 5. 使用独立tmux服务器启动
tmux -L "$SOCKET" kill-server 2>/dev/null || true

tmux -L "$SOCKET" \
  new-session \
  -d \
  -s "$SESSION" \
  "bash '$WRAPPER'"

sleep 8

echo
if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    echo "Stage 6已启动。"
    echo
    echo "进入会话："
    echo "  tmux -L $SOCKET attach -t $SESSION"
    echo
    echo "查看日志："
    echo "  tail -f '$LOG'"
else
    echo "Stage 6启动后立即退出。真实错误如下："
    echo "============================================================"
    tail -n 150 "$LOG" || true
    exit 5
fi
