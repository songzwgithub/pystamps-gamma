#!/usr/bin/env bash

set -Eeuo pipefail

DATASET="${1:-}"
START_STEP="${2:-2}"
END_STEP="${3:-2}"
CPU_WORKERS="${4:-16}"
IO_WORKERS="${5:-4}"
MODE="${6:-python}"

if [[ -z "${DATASET}" ]]; then
    echo "用法："
    echo "  $0 DATASET [START_STEP] [END_STEP] [CPU_WORKERS] [IO_WORKERS] [python|native]"
    exit 2
fi

DATASET="$(readlink -f "${DATASET}")"

if [[ ! -d "${DATASET}" ]]; then
    echo "错误：数据集目录不存在：${DATASET}" >&2
    exit 2
fi

if [[ ! -f "${DATASET}/patch.list" ]]; then
    echo "错误：缺少patch.list：${DATASET}" >&2
    exit 2
fi

if ! [[ "${CPU_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：CPU_WORKERS必须为正整数" >&2
    exit 2
fi

if ! [[ "${IO_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：IO_WORKERS必须为正整数" >&2
    exit 2
fi

# 禁止每个patch worker内部再次启动整机数量的BLAS线程。
# 正式多patch运行依靠cpu-workers并行。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# 防止某些OpenMP库动态增减线程。
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

LOG_DIR="${DATASET}/_run_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage_${START_STEP}_${END_STEP}_${MODE}_${STAMP}.log"
STATUS_BEFORE="${LOG_DIR}/status_before_${STAMP}.json"
STATUS_AFTER="${LOG_DIR}/status_after_${STAMP}.json"

CPU_COUNT="$(nproc)"
MEMORY_GB="$(
    awk '
        /MemTotal/ {
            printf "%.1f", $2 / 1024 / 1024
        }
    ' /proc/meminfo
)"

PATCH_COUNT="$(
    awk '
        NF > 0 {
            n += 1
        }
        END {
            print n + 0
        }
    ' "${DATASET}/patch.list"
)"

echo "======================================================"
echo "pySTAMPS CPU optimized runner"
echo "======================================================"
echo "Dataset      : ${DATASET}"
echo "Stages       : ${START_STEP}-${END_STEP}"
echo "Mode         : ${MODE}"
echo "CPU visible  : ${CPU_COUNT}"
echo "Memory       : ${MEMORY_GB} GB"
echo "Patch count  : ${PATCH_COUNT}"
echo "CPU workers  : ${CPU_WORKERS}"
echo "I/O workers  : ${IO_WORKERS}"
echo "BLAS threads : 1"
echo "Log          : ${LOG_FILE}"
echo "======================================================"

pystamps status \
    --dataset "${DATASET}" \
    | tee "${STATUS_BEFORE}"

if [[ "${MODE}" == "native" ]]; then
    REPO_ROOT="/home/ubuntu/software/pystamps-main"
    NATIVE_BIN="${REPO_ROOT}/target/release/pystamps-native"

    if [[ ! -x "${NATIVE_BIN}" ]]; then
        echo "错误：未找到原生执行器：${NATIVE_BIN}" >&2
        echo "先运行：" >&2
        echo "  cargo build --release -p pystamps-core --bin pystamps-native" >&2
        exit 2
    fi

    COMMAND=(
        "${NATIVE_BIN}"
        run
        --native-only
        --dataset
        "${DATASET}"
        --start-step
        "${START_STEP}"
        --end-step
        "${END_STEP}"
        --backend
        native
        --stage2-kernel-backend
        native
        --cpu-workers
        "${CPU_WORKERS}"
        --stage2-native-threads
        1
    )

elif [[ "${MODE}" == "python" ]]; then
    COMMAND=(
        pystamps
        run
        --dataset
        "${DATASET}"
        --start-step
        "${START_STEP}"
        --end-step
        "${END_STEP}"
        --cpu-workers
        "${CPU_WORKERS}"
        --io-workers
        "${IO_WORKERS}"
    )

else
    echo "错误：MODE必须是python或native" >&2
    exit 2
fi

echo
echo "执行命令："
printf '  %q' "${COMMAND[@]}"
echo
echo

START_EPOCH="$(date +%s)"

set +e

/usr/bin/time \
    -v \
    "${COMMAND[@]}" \
    2>&1 \
    | tee "${LOG_FILE}"

RUN_STATUS="${PIPESTATUS[0]}"

set -e

END_EPOCH="$(date +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

echo
echo "======================================================"
echo "Run completed"
echo "Exit code      : ${RUN_STATUS}"
echo "Elapsed seconds: ${ELAPSED_SECONDS}"
echo "Elapsed hours  : $(
    awk -v s="${ELAPSED_SECONDS}" \
        'BEGIN {printf "%.3f", s / 3600.0}'
)"
echo "======================================================"

pystamps status \
    --dataset "${DATASET}" \
    | tee "${STATUS_AFTER}"

echo
echo "关键错误扫描："

grep -Ein \
    'traceback|exception|memoryerror|killed|out of memory|fatal|failed' \
    "${LOG_FILE}" \
    || true

exit "${RUN_STATUS}"
