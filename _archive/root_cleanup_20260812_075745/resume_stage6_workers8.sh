#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"
SNAPHU="/usr/bin/snaphu"

WORK="$DATASET/_stage6_sbas_work/snaphu"
LOG_DIR="$DATASET/_run_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/stage6_workers8_resume_${STAMP}.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Stage 6 SBAS 断点恢复"
echo "时间    : $(date)"
echo "数据集  : $DATASET"
echo "Python  : $PYTHON"
echo "SNAPHU  : $SNAPHU"
echo "日志    : $LOG"
echo "============================================================"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：Python不存在：$PYTHON"
    exit 2
fi

if [[ ! -x "$SNAPHU" ]]; then
    echo "错误：SNAPHU不存在：$SNAPHU"
    exit 3
fi

# 防止重复启动
mapfile -t EXISTING < <(
    pgrep -f \
      '[p]ython.*pystamps\.pipeline\.stage6_sbas|[/]snaphu -d -f snaphu.conf' \
      || true
)

if (( ${#EXISTING[@]} > 0 )); then
    echo "检测到已有Stage 6相关进程："
    ps -fp "${EXISTING[@]}" || true
    echo "拒绝重复启动。"
    exit 4
fi

# 检查NumPy环境
"$PYTHON" - <<'PY'
import numpy as np
import scipy

print("NumPy :", np.__version__)
print("SciPy :", scipy.__version__)
print("NumPy导入正常")
PY

# 找出完整snaphu.out的标准大小
EXPECTED_BYTES="$(
    find "$WORK" \
      -type f \
      -name snaphu.out \
      -printf '%s\n' \
      2>/dev/null \
    | sort \
    | uniq -c \
    | sort -nr \
    | awk 'NR==1 {print $2}'
)"

if [[ -n "$EXPECTED_BYTES" ]]; then
    COMPLETE="$(
        find "$WORK" \
          -type f \
          -name snaphu.out \
          -printf '%s\n' \
          2>/dev/null \
        | awk -v expected="$EXPECTED_BYTES" '
            $1 == expected {n++}
            END {print n + 0}
          '
    )"

    echo "完整输出标准大小：$EXPECTED_BYTES bytes"
    echo "已有完整SNAPHU结果：$COMPLETE / 763"

    # 移走因终止产生的不完整输出
    PARTIAL_BACKUP="$DATASET/_stage6_partial_backup/snaphu_${STAMP}"
    mkdir -p "$PARTIAL_BACKUP"

    while IFS= read -r -d '' output; do
        size="$(stat -c '%s' "$output")"

        if [[ "$size" -ne "$EXPECTED_BYTES" ]]; then
            relative="${output#"$WORK"/}"
            target="$PARTIAL_BACKUP/$relative"

            mkdir -p "$(dirname "$target")"
            mv "$output" "$target"

            echo "移动不完整输出：$relative，大小=$size"
        fi
    done < <(
        find "$WORK" \
          -type f \
          -name snaphu.out \
          -print0 \
          2>/dev/null
    )
else
    echo "未发现已有完整snaphu.out。"
fi

export PATH="/usr/bin:/usr/local/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"
export REAL_DATASET="$DATASET"

# GRID断点恢复
export PYSTAMPS_STAGE6_GRID_RESUME=1
export PYSTAMPS_STAGE6_GRID_IFG_BATCH=4
export PYSTAMPS_STAGE6_GRID_WINDOW_BATCH=32
export PYSTAMPS_STAGE6_GRID_FFT_WORKERS=16

# 3D_QUICK参数
export PYSTAMPS_SBAS_EDGE_CHUNK=8192
export PYSTAMPS_SBAS_STRICT_ANNEAL=0
export PYSTAMPS_SBAS_ANNEAL_RUNS=1
export PYSTAMPS_SBAS_ANNEAL_WORKERS=1

# SNAPHU八并发
export PYSTAMPS_STAGE6_SNAPHU_WORKERS=8

# 防止8个SNAPHU进程内部再次嵌套线程
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo
echo "运行参数："
echo "  GRID resume   = $PYSTAMPS_STAGE6_GRID_RESUME"
echo "  Edge chunk    = $PYSTAMPS_SBAS_EDGE_CHUNK"
echo "  Strict anneal = $PYSTAMPS_SBAS_STRICT_ANNEAL"
echo "  SNAPHU workers= $PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo

exec "$PYTHON" \
  -m pystamps.pipeline.stage6_sbas \
  --dataset "$DATASET" \
  --io-workers 1
