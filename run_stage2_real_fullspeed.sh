#!/usr/bin/env bash

set -euo pipefail

PYSTAMPS_ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"

REAL_DATASET="${REAL_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_full_5x5}"

CONFIG_FILE="${PYSTAMPS_ROOT}/stage2_real_fullspeed.yaml"

if [[ "$REAL_DATASET" == *smoke* ]]; then
    echo "错误：REAL_DATASET仍然是Smoke目录：$REAL_DATASET" >&2
    exit 2
fi

if [[ ! -d "$REAL_DATASET" ]]; then
    echo "错误：数据集目录不存在：$REAL_DATASET" >&2
    exit 2
fi

if [[ ! -f "$REAL_DATASET/patch.list" ]]; then
    echo "错误：缺少patch.list：$REAL_DATASET" >&2
    exit 2
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误：缺少运行配置：$CONFIG_FILE" >&2
    exit 2
fi

cd "$PYSTAMPS_ROOT"

PATCH_COUNT="$(
    find "$REAL_DATASET" \
        -maxdepth 1 \
        -type d \
        -name 'PATCH_*' \
        | wc -l
)"

STAGE1_COUNT="$(
    find "$REAL_DATASET" \
        -maxdepth 2 \
        -type f \
        -name ps1.mat \
        | wc -l
)"

echo "============================================================"
echo "真实数据集：$REAL_DATASET"
echo "空间patch数：$PATCH_COUNT"
echo "Stage 1完成patch数：$STAGE1_COUNT"
echo "CPU核数：$(nproc)"
echo "============================================================"

if [[ "$PATCH_COUNT" -eq 0 ]]; then
    echo "错误：未发现PATCH_*目录" >&2
    exit 2
fi

if [[ "$STAGE1_COUNT" -ne "$PATCH_COUNT" ]]; then
    echo "错误：并非所有patch均完成Stage 1" >&2
    exit 2
fi

if (( PATCH_COUNT < 16 || PATCH_COUNT > 36 )); then
    echo "警告：建议完整场景使用16～36个patch；当前为$PATCH_COUNT个。"
fi

echo
echo "=== 内存 ==="
free -h

echo
echo "=== 数据盘 ==="
df -h "$REAL_DATASET"

# ------------------------------------------------------------
# CLAP精度
# ------------------------------------------------------------

# 已经与双精度结果验证，正式运行使用单精度。
export PYSTAMPS_CLAP_SINGLE_PRECISION=1

# 每批处理8个活动空间窗口。
export PYSTAMPS_CLAP_WINDOW_BATCH=8

# 每个patch内部，将763个IFG划分给10个线程。
export PYSTAMPS_CLAP_IFG_WORKERS=10

# 禁止每个IFG线程中的FFT再次启动多线程。
export PYSTAMPS_CLAP_FFT_WORKERS=1

# 保留CLAP内部进度。
export PYSTAMPS_CLAP_PROGRESS=1

# ------------------------------------------------------------
# 禁止NumPy/SciPy底层库再次嵌套并行
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# 避免大量线程产生过多glibc内存arena。
export MALLOC_ARENA_MAX=2

export PYTHONUNBUFFERED=1

# 临时目录放在数据盘，避免系统盘空间不足。
export TMPDIR="$REAL_DATASET/_tmp"
mkdir -p "$TMPDIR"

ulimit -n 65535 2>/dev/null || true

LOG_DIR="$REAL_DATASET/_run_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/stage2_real_fullspeed_${TIMESTAMP}.log"

echo
echo "============================================================"
echo "并行结构：3个patch进程 × 每patch 10个IFG线程"
echo "理论计算线程：30"
echo "窗口batch：8"
echo "FFT内部线程：1"
echo "日志：$LOG_FILE"
echo "============================================================"
echo

set -o pipefail

/usr/bin/time -v \
pystamps \
    --config "$CONFIG_FILE" \
    run \
    --dataset "$REAL_DATASET" \
    --start-step 2 \
    --end-step 2 \
    --cpu-workers 3 \
    --io-workers 4 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE="${PIPESTATUS[0]}"

echo
echo "============================================================"
echo "Stage 2退出码：$EXIT_CODE"
echo "日志：$LOG_FILE"
echo "============================================================"

pystamps status \
    --dataset "$REAL_DATASET" \
    || true

exit "$EXIT_CODE"
