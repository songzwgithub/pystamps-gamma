#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pystamps-main"
DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
WORK="$DATASET/_stage6_sbas_work/snaphu"
EXPECTED_BYTES=$((3941 * 4757 * 4))
STAMP="$(date +%Y%m%d_%H%M%S)"
PARTIAL_BACKUP="$DATASET/_stage6_partial_backup/snaphu_${STAMP}"
LOG="$DATASET/_run_logs/stage6_3d_quick_workers8_${STAMP}.log"

cd "$ROOT"

count_complete()
{
    find "$WORK" \
      -type f \
      -name snaphu.out \
      -printf '%s\n' \
      2>/dev/null \
    | awk -v expected="$EXPECTED_BYTES" '
        $1 == expected {count++}
        END {print count + 0}
      '
}

echo "=============================================="
echo "切换前完整SNAPHU结果：$(count_complete)"
echo "完整输出期望大小：$EXPECTED_BYTES bytes"
echo "=============================================="

# 先中断Stage 6父进程
mapfile -t PARENT_PIDS < <(
    pgrep -f '[p]ython .*pystamps\.pipeline\.stage6_sbas' || true
)

if (( ${#PARENT_PIDS[@]} > 0 )); then
    echo "中断Stage 6父进程：${PARENT_PIDS[*]}"
    kill -INT "${PARENT_PIDS[@]}" || true
fi

# 给当前4个SNAPHU任务最多30秒正常结束
for _ in $(seq 1 30); do
    if ! pgrep -f '[/]usr/bin/snaphu -d -f snaphu.conf' >/dev/null; then
        break
    fi
    sleep 1
done

# 终止仍残留的SNAPHU子进程
mapfile -t CHILD_PIDS < <(
    pgrep -f '[/]usr/bin/snaphu -d -f snaphu.conf' || true
)

if (( ${#CHILD_PIDS[@]} > 0 )); then
    echo "终止未结束的SNAPHU子进程：${CHILD_PIDS[*]}"
    kill -TERM "${CHILD_PIDS[@]}" || true
    sleep 5
fi

# 安全移走尺寸不完整的snaphu.out，避免断点逻辑误判
mkdir -p "$PARTIAL_BACKUP"

while IFS= read -r -d '' output; do
    size="$(stat -c '%s' "$output")"

    if [[ "$size" -ne "$EXPECTED_BYTES" ]]; then
        relative="${output#"$WORK"/}"
        target="$PARTIAL_BACKUP/$relative"

        mkdir -p "$(dirname "$target")"
        mv "$output" "$target"

        echo "移动未完成输出：$relative，大小=$size"
    fi
done < <(
    find "$WORK" \
      -type f \
      -name snaphu.out \
      -print0 \
      2>/dev/null
)

# 把启动脚本改成允许外部覆盖并发数
python - <<'PY'
from pathlib import Path
import re

path = Path(
    "/home/ubuntu/software/pystamps-main/"
    "run_stage6_quick_resume.sh"
)

if not path.exists():
    raise SystemExit(f"缺少启动脚本：{path}")

text = path.read_text(encoding="utf-8")

replacement = (
    'export PYSTAMPS_STAGE6_SNAPHU_WORKERS='
    '"${PYSTAMPS_STAGE6_SNAPHU_WORKERS:-8}"'
)

text, count = re.subn(
    r'^export PYSTAMPS_STAGE6_SNAPHU_WORKERS=.*$',
    replacement,
    text,
    count=1,
    flags=re.MULTILINE,
)

if count == 0:
    marker = 'export PYSTAMPS_SBAS_ANNEAL_WORKERS='
    position = text.find(marker)

    if position < 0:
        raise SystemExit(
            "未找到SNAPHU workers或退火参数位置，未修改脚本。"
        )

    line_end = text.find("\n", position)

    text = (
        text[:line_end + 1]
        + replacement
        + "\n"
        + text[line_end + 1:]
    )

path.write_text(text, encoding="utf-8")
print("已设置SNAPHU并发默认值为8。")
PY

bash -n "$ROOT/run_stage6_quick_resume.sh"

echo
echo "切换后保留的完整结果：$(count_complete)"
echo "未完成文件备份：$PARTIAL_BACKUP"

tmux kill-session \
  -t cangzhou_stage6_quick \
  2>/dev/null \
  || true

tmux new-session \
  -d \
  -s cangzhou_stage6_quick \
  "cd '$ROOT' && \
   PYSTAMPS_STAGE6_SNAPHU_WORKERS=8 \
   ./run_stage6_quick_resume.sh \
   2>&1 | tee '$LOG'"

sleep 3

if tmux has-session \
    -t cangzhou_stage6_quick \
    2>/dev/null
then
    echo
    echo "8并发Stage 6已启动。"
    echo "查看：tmux attach -t cangzhou_stage6_quick"
    echo "日志：$LOG"
else
    echo "启动后立即退出，检查："
    echo "tail -n 100 '$LOG'"
    exit 1
fi
