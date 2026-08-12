#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
PYFILE="$ROOT/pystamps/pipeline/stage6_sbas.py"
PRIMARY_BACKUP="${PYFILE}.bak_stage6_opt"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"

echo "============================================================"
echo "修复 Stage 6 SBAS 源文件"
echo "============================================================"

# 只终止 Stage 6，不影响其他任务
mapfile -t PIDS < <(
    pgrep -f '[p]ython.*pystamps\.pipeline\.stage6_sbas' || true
)

if (( ${#PIDS[@]} > 0 )); then
    echo "终止残留 Stage 6 进程：${PIDS[*]}"
    kill -TERM "${PIDS[@]}" || true
    sleep 5

    mapfile -t PIDS2 < <(
        pgrep -f '[p]ython.*pystamps\.pipeline\.stage6_sbas' || true
    )

    if (( ${#PIDS2[@]} > 0 )); then
        kill -KILL "${PIDS2[@]}" || true
    fi
fi

if [[ ! -f "$PYFILE" ]]; then
    echo "错误：源文件不存在：$PYFILE" >&2
    exit 2
fi

# 保留当前损坏文件
cp -a \
    "$PYFILE" \
    "${PYFILE}.broken_${STAMP}"

echo "损坏文件已保留："
echo "  ${PYFILE}.broken_${STAMP}"

# 优先使用优化脚本创建的原始备份
if [[ -s "$PRIMARY_BACKUP" ]]; then
    echo "使用备份恢复："
    echo "  $PRIMARY_BACKUP"

    cp -a \
        "$PRIMARY_BACKUP" \
        "$PYFILE"
else
    # 搜索其他可能的备份
    BACKUP="$(
        find "$ROOT" \
            -type f \
            \( \
                -name 'stage6_sbas.py.bak*' \
                -o -name 'stage6_sbas.py.before*' \
            \) \
            -size +1k \
            -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"

    if [[ -z "$BACKUP" ]]; then
        echo "错误：没有找到可用的 stage6_sbas.py 备份。" >&2
        echo "损坏文件位于：${PYFILE}.broken_${STAMP}" >&2
        exit 3
    fi

    echo "使用最近备份恢复："
    echo "  $BACKUP"

    cp -a \
        "$BACKUP" \
        "$PYFILE"
fi

echo
echo "检查文件头："
head -n 30 "$PYFILE"

echo
echo "执行Python语法检查："

python -m py_compile "$PYFILE"

echo "语法检查通过。"

# 检查模块是否可导入
python - <<'PY'
import importlib

module = importlib.import_module(
    "pystamps.pipeline.stage6_sbas"
)

print("模块导入成功：", module.__file__)
PY

# 将前一次未完成的Stage 6产物安全移动，不删除Stage 1-5成果
DATASET="${REAL_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"
PARTIAL_BACKUP="$DATASET/_stage6_partial_backup/$STAMP"

mkdir -p "$PARTIAL_BACKUP"

for name in \
    uw_grid.mat \
    uw_interp.mat \
    uw_space_time.mat \
    uw_phaseuw.mat \
    phuw2.mat \
    stage6_sbas_debug.json \
    snaphu.in \
    snaphu.out \
    snaphu.conf \
    snaphu.costinfile \
    snaphu.log
do
    if [[ -e "$DATASET/$name" ]]; then
        mv \
            "$DATASET/$name" \
            "$PARTIAL_BACKUP/"
    fi
done

echo
echo "已有Stage 6临时结果已移动到："
echo "  $PARTIAL_BACKUP"

# 重建启动脚本：
# 不再修改stage6_sbas.py，只使用原补丁已经支持的环境变量。
cat > "$ROOT/run_stage6_fast.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"

export REAL_DATASET="${REAL_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"

# 避免每个Python子进程再次启动一组BLAS/OpenMP线程
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# GRID阶段降低并发，避免8个大型进程同时持有大数组
export PYSTAMPS_STAGE6_GRID_WORKERS="${PYSTAMPS_STAGE6_GRID_WORKERS:-4}"

# 增大边块，减少Python循环和调度开销
export PYSTAMPS_SBAS_EDGE_CHUNK="${PYSTAMPS_SBAS_EDGE_CHUNK:-2048}"

# SNAPHU并发不宜过高，避免磁盘争用
export PYSTAMPS_STAGE6_SNAPHU_WORKERS="${PYSTAMPS_STAGE6_SNAPHU_WORKERS:-4}"

# 完整质量模式：保留严格退火
export PYSTAMPS_SBAS_ANNEAL_WORKERS="${PYSTAMPS_SBAS_ANNEAL_WORKERS:-8}"
export PYSTAMPS_SBAS_ANNEAL_RUNS="${PYSTAMPS_SBAS_ANNEAL_RUNS:-15}"
export PYSTAMPS_SBAS_STRICT_ANNEAL="${PYSTAMPS_SBAS_STRICT_ANNEAL:-1}"

export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "============================================================"
echo "Stage 6 SBAS运行参数"
echo "============================================================"
echo "Dataset       : $REAL_DATASET"
echo "Grid workers  : $PYSTAMPS_STAGE6_GRID_WORKERS"
echo "Edge chunk    : $PYSTAMPS_SBAS_EDGE_CHUNK"
echo "SNAPHU workers: $PYSTAMPS_STAGE6_SNAPHU_WORKERS"
echo "Anneal workers: $PYSTAMPS_SBAS_ANNEAL_WORKERS"
echo "Anneal runs   : $PYSTAMPS_SBAS_ANNEAL_RUNS"
echo "Strict anneal : $PYSTAMPS_SBAS_STRICT_ANNEAL"
echo "BLAS threads  : $OPENBLAS_NUM_THREADS"
echo "============================================================"

if [[ ! -x "$ROOT/run_stage6_sbas.sh" ]]; then
    echo "错误：缺少原始SBAS启动脚本：" >&2
    echo "  $ROOT/run_stage6_sbas.sh" >&2
    exit 4
fi

exec bash "$ROOT/run_stage6_sbas.sh"
RUN

chmod +x "$ROOT/run_stage6_fast.sh"

echo
echo "============================================================"
echo "修复完成"
echo "============================================================"
echo "启动命令："
echo "  cd $ROOT"
echo "  ./run_stage6_fast.sh"
