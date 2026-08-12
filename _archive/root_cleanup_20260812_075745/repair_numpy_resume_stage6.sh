#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"

ENV_DIR="/home/ubuntu/software/miniconda3/envs/stamps"
PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

SNAPHU_WORK="$DATASET/_stage6_sbas_work/snaphu"
LOG_DIR="$DATASET/_run_logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/stage6_workers8_numpy235_${STAMP}.log"
WRAPPER="$LOG_DIR/stage6_workers8_numpy235_${STAMP}.sh"

SOCKET="stage6w8"
SESSION="cangzhou_stage6_quick"

mkdir -p "$LOG_DIR"

cd "$ROOT"

echo "============================================================"
echo "修复NumPy并恢复Stage 6"
echo "============================================================"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到stamps环境Python：$PYTHON" >&2
    exit 2
fi

if [[ ! -x "$PIP" ]]; then
    echo "错误：找不到stamps环境pip：$PIP" >&2
    exit 3
fi

# ------------------------------------------------------------
# 1. 防止重复计算
# ------------------------------------------------------------

mapfile -t EXISTING < <(
    pgrep -f \
      '[p]ython.*pystamps\.pipeline\.stage6_sbas|[/]snaphu -d -f snaphu.conf' \
      || true
)

if (( ${#EXISTING[@]} > 0 )); then
    echo "检测到已有Stage 6相关进程："
    ps -fp "${EXISTING[@]}" || true
    echo
    echo "请先确认这些进程是否需要保留。"
    exit 4
fi

# ------------------------------------------------------------
# 2. 不导入NumPy，读取当前安装版本
# ------------------------------------------------------------

echo
echo "当前NumPy包信息："

"$PYTHON" - <<'PY'
from importlib.metadata import version, PackageNotFoundError

for name in ("numpy", "scipy", "matplotlib"):
    try:
        print(f"{name:12s}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name:12s}: not installed")
PY

echo
echo "保存当前环境清单："

"$PYTHON" -m pip freeze \
  > "$LOG_DIR/stamps_pip_freeze_before_numpy_fix_${STAMP}.txt"

if command -v conda >/dev/null 2>&1; then
    conda list -n stamps \
      > "$LOG_DIR/stamps_conda_list_before_numpy_fix_${STAMP}.txt" \
      || true
fi

# ------------------------------------------------------------
# 3. 固定到NumPy 2.3.5
# ------------------------------------------------------------

echo
echo "重新安装兼容CPU的NumPy 2.3.5..."

"$PYTHON" -m pip uninstall \
  -y numpy \
  || true

"$PYTHON" -m pip install \
  --no-cache-dir \
  --only-binary=:all: \
  "numpy==2.3.5"

# ------------------------------------------------------------
# 4. 完整导入检查
# ------------------------------------------------------------

echo
echo "验证NumPy、SciPy和pySTAMPS："

"$PYTHON" - <<'PY'
import sys

import numpy as np
import scipy

print("Python :", sys.executable)
print("NumPy  :", np.__version__)
print("SciPy  :", scipy.__version__)

a = np.arange(12, dtype=np.float64).reshape(3, 4)
b = np.fft.fft2(a)

print("FFT test:", b.shape, bool(np.all(np.isfinite(b))))

from pystamps.pipeline import ported
from pystamps.pipeline import stage6_sbas

resolved = ported._resolve_external_tool(
    "snaphu",
    "/usr/bin/snaphu",
)

print("Stage6 module:", stage6_sbas.__file__)
print("SNAPHU       :", resolved)
print("Import test  : passed")
PY

"$PYTHON" -m pip check

# ------------------------------------------------------------
# 5. 统计已有完整SNAPHU结果
# ------------------------------------------------------------

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

if [[ -n "$EXPECTED_BYTES" ]]; then
    COMPLETE="$(
        find "$SNAPHU_WORK" \
          -type f \
          -name snaphu.out \
          -printf '%s\n' \
          2>/dev/null \
        | awk -v expected="$EXPECTED_BYTES" '
            $1 == expected {count++}
            END {print count + 0}
          '
    )"
else
    EXPECTED_BYTES="unknown"
    COMPLETE=0
fi

echo
echo "完整SNAPHU输出大小：$EXPECTED_BYTES"
echo "已有完整SNAPHU结果：$COMPLETE / 763"

# ------------------------------------------------------------
# 6. 创建使用绝对Python路径的启动器
# ------------------------------------------------------------

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec >> "$LOG" 2>&1

echo "============================================================"
echo "Stage 6恢复时间：\$(date)"
echo "============================================================"

export PATH="/usr/bin:/usr/local/bin:/bin:\$PATH"
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

export PYSTAMPS_STAGE6_SNAPHU_WORKERS=8

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "Python : $PYTHON"
echo "SNAPHU: \$(command -v snaphu)"
echo "Workers: \$PYSTAMPS_STAGE6_SNAPHU_WORKERS"

"$PYTHON" - <<'PY'
import numpy as np
import scipy

print("NumPy runtime:", np.__version__)
print("SciPy runtime:", scipy.__version__)
PY

exec "$PYTHON" \
  -m pystamps.pipeline.stage6_sbas \
  --dataset "$DATASET" \
  --io-workers 1
EOF

chmod +x "$WRAPPER"
bash -n "$WRAPPER"

# ------------------------------------------------------------
# 7. 使用独立tmux服务器启动
# ------------------------------------------------------------

tmux -L "$SOCKET" \
  kill-server \
  2>/dev/null \
  || true

touch "$LOG"

tmux -L "$SOCKET" \
  new-session \
  -d \
  -s "$SESSION" \
  "bash '$WRAPPER'"

sleep 10

echo
if tmux -L "$SOCKET" \
    has-session \
    -t "$SESSION" \
    2>/dev/null
then
    echo "Stage 6恢复成功。"
    echo
    echo "进入会话："
    echo "  tmux -L $SOCKET attach -t $SESSION"
    echo
    echo "查看日志："
    echo "  tail -f '$LOG'"
else
    echo "Stage 6仍然立即退出。日志如下：" >&2
    echo "============================================================" >&2
    tail -n 200 "$LOG" >&2 || true
    exit 5
fi
