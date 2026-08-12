#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 基础配置
# ============================================================

STAGE="${1:-}"

if [[ ! "$STAGE" =~ ^[4-8]$ ]]; then
    echo "用法："
    echo "  bash run_native_stage.sh 4"
    echo "  bash run_native_stage.sh 5"
    echo "  bash run_native_stage.sh 6"
    echo "  bash run_native_stage.sh 7"
    echo "  bash run_native_stage.sh 8"
    exit 2
fi

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"

DATASET="${REAL_DATASET:-\
/mnt/vol-gdc28n1r/insar/cangzhou_P69/\
pystamps_sbas_ps_optimized}"

BIN="$ROOT/bin/pystamps-native"

INTERVAL="${PROGRESS_INTERVAL:-5}"

FORCE_STAGE="${FORCE_STAGE:-0}"

ONLY_PATCH="${ONLY_PATCH:-}"

mkdir -p "$ROOT/bin"
mkdir -p "$DATASET/_run_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="$DATASET/_run_logs/native_stage${STAGE}_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# ============================================================
# 工具函数
# ============================================================

format_seconds()
{
    local total="${1:-0}"

    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))

    printf "%02d:%02d:%02d" \
        "$hours" \
        "$minutes" \
        "$seconds"
}


draw_bar()
{
    local done="$1"
    local total="$2"
    local width=36

    local filled=0

    if (( total > 0 )); then
        filled=$((done * width / total))
    fi

    local empty=$((width - filled))

    printf "["

    if (( filled > 0 )); then
        printf "%${filled}s" "" \
            | tr " " "#"
    fi

    if (( empty > 0 )); then
        printf "%${empty}s" "" \
            | tr " " "-"
    fi

    printf "]"
}


count_patch_output()
{
    local output_name="$1"
    local count=0

    for patch in "${PATCHES[@]}"; do
        if [[ -f "$patch/$output_name" ]]; then
            count=$((count + 1))
        fi
    done

    echo "$count"
}


native_process_stats()
{
    local stage="$1"

    local pid_list

    pid_list="$(
        pgrep -f \
        "pystamps-native.*stage.*${stage}" \
        2>/dev/null \
        | paste -sd, - \
        || true
    )"

    if [[ -z "$pid_list" ]]; then
        echo "0.0 0.0 0 0"
        return
    fi

    ps \
        -o pcpu=,rss=,nlwp= \
        -p "$pid_list" \
        2>/dev/null \
        | awk '
        {
            cpu += $1
            rss += $2
            threads += $3
            processes += 1
        }
        END {
            printf "%.1f %.3f %d %d\n",
                cpu + 0,
                rss / 1024 / 1024,
                threads + 0,
                processes + 0
        }'
}


check_required_patch_input()
{
    local input_name="$1"

    local missing=0

    for patch in "${PATCHES[@]}"; do
        if [[ ! -f "$patch/$input_name" ]]; then
            echo "缺少：$patch/$input_name"
            missing=$((missing + 1))
        fi
    done

    if (( missing > 0 )); then
        echo
        echo "共有 $missing 个 patch 缺少输入文件。"
        exit 3
    fi
}


backup_patch_outputs()
{
    local backup_root="$1"
    shift

    local output_names=("$@")

    for patch in "${PATCHES[@]}"; do
        local patch_name
        patch_name="$(basename "$patch")"

        for output_name in "${output_names[@]}"; do
            if [[ -e "$patch/$output_name" ]]; then
                mkdir -p "$backup_root/$patch_name"

                mv \
                    "$patch/$output_name" \
                    "$backup_root/$patch_name/"
            fi
        done
    done
}


backup_root_outputs()
{
    local backup_root="$1"
    shift

    local output_names=("$@")

    mkdir -p "$backup_root/root"

    for output_name in "${output_names[@]}"; do
        if [[ -e "$DATASET/$output_name" ]]; then
            mv \
                "$DATASET/$output_name" \
                "$backup_root/root/"
        fi
    done
}


run_patch_stage()
{
    local stage="$1"
    local required_input="$2"
    local primary_output="$3"
    local patch_workers="$4"
    local rayon_threads="$5"
    shift 5

    local output_bundle=("$@")

    check_required_patch_input "$required_input"

    if [[ "$FORCE_STAGE" == "1" ]]; then
        local backup_root

        backup_root="$DATASET/_native_backup/stage${stage}_${TIMESTAMP}"

        backup_patch_outputs \
            "$backup_root" \
            "${output_bundle[@]}"

        echo "旧输出已移动到："
        echo "  $backup_root"
    fi

    local pending=()

    for patch in "${PATCHES[@]}"; do
        if [[ ! -f "$patch/$primary_output" ]]; then
            pending+=("$patch")
        fi
    done

    local total="${#PATCHES[@]}"
    local initial_done=$((total - ${#pending[@]}))

    echo
    echo "============================================================"
    echo "Native Stage $stage patch计算"
    echo "============================================================"
    echo "Patch总数         : $total"
    echo "已有结果          : $initial_done"
    echo "待计算            : ${#pending[@]}"
    echo "Patch并发数       : $patch_workers"
    echo "每patch Rayon线程 : $rayon_threads"
    echo "日志目录          : $LOG_DIR"
    echo "============================================================"

    if (( ${#pending[@]} == 0 )); then
        echo "所有patch结果已经存在。"
        return
    fi

    printf '%s\0' "${pending[@]}" \
        | xargs \
            -0 \
            -r \
            -P "$patch_workers" \
            -I '{}' \
            bash -c '
                set -euo pipefail

                patch="$1"
                binary="$2"
                stage="$3"
                threads="$4"
                log_dir="$5"
                primary_output="$6"

                patch_name="$(basename "$patch")"

                log="$log_dir/${patch_name}.log"

                {
                    echo "Patch：$patch_name"
                    echo "RAYON_NUM_THREADS=$threads"
                    echo "开始时间：$(date)"
                    echo
                } > "$log"

                env \
                    RAYON_NUM_THREADS="$threads" \
                    RUST_BACKTRACE=1 \
                    MALLOC_ARENA_MAX=2 \
                    "$binary" \
                    stage "$stage" \
                    --patch "$patch" \
                    >> "$log" 2>&1

                if [[ ! -f "$patch/$primary_output" ]]; then
                    echo \
                        "命令退出但缺少输出：$patch/$primary_output" \
                        >> "$log"

                    exit 4
                fi

                echo >> "$log"
                echo "完成时间：$(date)" >> "$log"
            ' \
            _ \
            '{}' \
            "$BIN" \
            "$stage" \
            "$rayon_threads" \
            "$LOG_DIR" \
            "$primary_output" &

    local batch_pid=$!

    local started
    started="$(date +%s)"

    while kill -0 "$batch_pid" 2>/dev/null; do
        local done_count
        local now
        local elapsed
        local percent

        done_count="$(count_patch_output "$primary_output")"
        now="$(date +%s)"
        elapsed=$((now - started))

        percent=$(
            awk \
                -v d="$done_count" \
                -v t="$total" \
                'BEGIN {
                    if (t > 0) {
                        printf "%.2f", d * 100 / t
                    } else {
                        printf "0.00"
                    }
                }'
        )

        read -r \
            cpu \
            rss \
            threads \
            processes \
            <<< "$(native_process_stats "$stage")"

        printf \
            "\r[NATIVE][S%s] " \
            "$stage"

        draw_bar \
            "$done_count" \
            "$total"

        printf \
            " %d/%d %6s%% | elapsed=%s | CPU=%s%% | RSS=%sGiB | proc=%s | threads=%s" \
            "$done_count" \
            "$total" \
            "$percent" \
            "$(format_seconds "$elapsed")" \
            "$cpu" \
            "$rss" \
            "$processes" \
            "$threads"

        sleep "$INTERVAL"
    done

    echo

    if ! wait "$batch_pid"; then
        echo "Stage $stage 至少一个patch失败。"
        echo "检查日志：$LOG_DIR"
        exit 5
    fi

    local final_done

    final_done="$(count_patch_output "$primary_output")"

    echo
    echo "Stage $stage patch完成：$final_done/$total"

    if (( final_done != total )); then
        echo "仍有patch缺少结果："

        for patch in "${PATCHES[@]}"; do
            if [[ ! -f "$patch/$primary_output" ]]; then
                echo "  $(basename "$patch")"
            fi
        done

        exit 6
    fi
}


run_merged_stage()
{
    local stage="$1"
    local primary_output="$2"
    local rayon_threads="$3"
    local command_name="${4:-stage}"

    shift 4

    local output_bundle=("$@")

    if [[ "$FORCE_STAGE" == "1" ]]; then
        local backup_root

        backup_root="$DATASET/_native_backup/stage${stage}_merged_${TIMESTAMP}"

        backup_root_outputs \
            "$backup_root" \
            "${output_bundle[@]}"

        echo "旧merged输出已移动到："
        echo "  $backup_root"
    fi

    if [[ -f "$DATASET/$primary_output" ]]; then
        echo "输出已经存在，跳过："
        echo "  $DATASET/$primary_output"
        return
    fi

    local log="$LOG_DIR/merged.log"

    local command=()

    if [[ "$command_name" == "stage5-merge" ]]; then
        command=(
            "$BIN"
            stage5-merge
            --dataset "$DATASET"
        )
    else
        command=(
            "$BIN"
            stage "$stage"
            --dataset "$DATASET"
        )
    fi

    echo
    echo "============================================================"
    echo "Native Stage $stage merged计算"
    echo "============================================================"
    echo "Rayon线程 : $rayon_threads"
    echo "命令      : ${command[*]}"
    echo "日志      : $log"
    echo "============================================================"

    {
        echo "RAYON_NUM_THREADS=$rayon_threads"
        echo "开始时间：$(date)"
        echo
    } > "$log"

    env \
        RAYON_NUM_THREADS="$rayon_threads" \
        RUST_BACKTRACE=1 \
        MALLOC_ARENA_MAX=2 \
        "${command[@]}" \
        >> "$log" 2>&1 &

    local pid=$!
    local started
    started="$(date +%s)"

    while kill -0 "$pid" 2>/dev/null; do
        local now
        local elapsed
        local cpu
        local rss
        local threads

        now="$(date +%s)"
        elapsed=$((now - started))

        read -r \
            cpu \
            rss_kib \
            threads \
            <<< "$(
                ps \
                    -o pcpu=,rss=,nlwp= \
                    -p "$pid" \
                    2>/dev/null \
                    | awk '
                    {
                        printf "%.1f %.3f %d\n",
                            $1,
                            $2 / 1024 / 1024,
                            $3
                    }'
            )"

        cpu="${cpu:-0.0}"
        rss_kib="${rss_kib:-0.0}"
        threads="${threads:-0}"

        printf \
            "\r[NATIVE][S%s][merged] elapsed=%s | CPU=%s%% | RSS=%sGiB | threads=%s" \
            "$stage" \
            "$(format_seconds "$elapsed")" \
            "$cpu" \
            "$rss_kib" \
            "$threads"

        sleep "$INTERVAL"
    done

    echo

    if ! wait "$pid"; then
        echo "Stage $stage merged失败。"
        echo "日志：$log"
        tail -n 80 "$log" || true
        exit 7
    fi

    if [[ ! -f "$DATASET/$primary_output" ]]; then
        echo "命令成功，但缺少："
        echo "  $DATASET/$primary_output"
        exit 8
    fi

    echo "Stage $stage merged完成。"
}


# ============================================================
# 数据集与native binary检查
# ============================================================

if [[ ! -d "$DATASET" ]]; then
    echo "数据集不存在：$DATASET"
    exit 2
fi

cd "$ROOT"

if [[ ! -x "$BIN" || "${REBUILD_NATIVE:-0}" == "1" ]]; then
    if pgrep -af \
        'pystamps run.*--start-step[ =]3' \
        >/dev/null 2>&1
    then
        echo "检测到Stage 3仍在运行。"
        echo "不要在Stage 3计算期间编译Rust。"
        exit 9
    fi

    command -v cargo >/dev/null 2>&1 \
        || {
            echo "未找到cargo。"
            exit 10
        }

    echo "编译release版pystamps-native..."

    cargo build \
        --release \
        -p pystamps-core \
        --bin pystamps-native

    mkdir -p "$ROOT/bin"

    cp -f \
        "$ROOT/target/release/pystamps-native" \
        "$BIN"

    chmod +x "$BIN"

    if command -v strip >/dev/null 2>&1; then
        strip "$BIN" || true
    fi
fi

echo
echo "Native覆盖情况："

"$BIN" coverage \
    --start-step "$STAGE" \
    --end-step "$STAGE"

mapfile -t PATCHES < <(
    find "$DATASET" \
        -maxdepth 1 \
        -type d \
        -name 'PATCH_*' \
        -print \
        | sort -V
)

if [[ -n "$ONLY_PATCH" ]]; then
    selected=()

    for patch in "${PATCHES[@]}"; do
        if [[ "$(basename "$patch")" == "$ONLY_PATCH" ]]; then
            selected+=("$patch")
        fi
    done

    PATCHES=("${selected[@]}")

    if (( ${#PATCHES[@]} == 0 )); then
        echo "未找到：$ONLY_PATCH"
        exit 11
    fi
fi

# ============================================================
# 各Stage参数
# ============================================================

case "$STAGE" in

    4)
        PATCH_WORKERS="${STAGE4_PATCH_WORKERS:-4}"
        RAYON_THREADS="${PYSTAMPS_STAGE4_THREADS:-8}"

        run_patch_stage \
            4 \
            select1.mat \
            weed1.mat \
            "$PATCH_WORKERS" \
            "$RAYON_THREADS" \
            weed1.mat
        ;;

    5)
        PATCH_WORKERS="${STAGE5_PATCH_WORKERS:-4}"
        RAYON_THREADS="${PYSTAMPS_STAGE5_THREADS:-8}"

        run_patch_stage \
            5 \
            weed1.mat \
            ps2.mat \
            "$PATCH_WORKERS" \
            "$RAYON_THREADS" \
            ps2.mat \
            ph2.mat \
            pm2.mat \
            bp2.mat \
            hgt2.mat \
            la2.mat \
            rc2.mat \
            psver.mat

        if [[ -z "$ONLY_PATCH" ]]; then
            run_merged_stage \
                5 \
                ifgstd2.mat \
                "${PYSTAMPS_STAGE5_MERGE_THREADS:-32}" \
                stage5-merge \
                ps2.mat \
                ph2.mat \
                pm2.mat \
                bp2.mat \
                hgt2.mat \
                la2.mat \
                rc2.mat \
                psver.mat \
                ifgstd2.mat
        fi
        ;;

    6)
        # STAGE6_SBAS_DELEGATE_V1
        if python - "$DATASET" <<'PY_CHECK'
from pathlib import Path
import sys
import numpy as np
from pystamps.io.mat import read_mat_variables

root = Path(sys.argv[1])
payload = read_mat_variables(root / "parms.mat", ("small_baseline_flag",))
value = payload.get("small_baseline_flag", "n")
text = "".join(str(v) for v in np.asarray(value).reshape(-1)).strip().lower()
raise SystemExit(0 if text == "y" else 1)
PY_CHECK
        then
            exec bash "$ROOT/run_stage6_sbas.sh"
        fi
        [[ -f "$DATASET/ifgstd2.mat" ]] \
            || {
                echo "缺少Stage 5 merged结果：ifgstd2.mat"
                exit 12
            }

        run_merged_stage \
            6 \
            phuw2.mat \
            "${PYSTAMPS_STAGE6_THREADS:-32}" \
            stage \
            phuw2.mat \
            uw_phaseuw.mat \
            uw_grid.mat \
            uw_interp.mat
        ;;

    7)
        [[ -f "$DATASET/phuw2.mat" ]] \
            || {
                echo "缺少Stage 6结果：phuw2.mat"
                exit 13
            }

        run_merged_stage \
            7 \
            scla2.mat \
            "${PYSTAMPS_STAGE7_THREADS:-32}" \
            stage \
            scla2.mat \
            scla_smooth2.mat
        ;;

    8)
        [[ -f "$DATASET/phuw2.mat" ]] \
            || {
                echo "缺少Stage 6结果：phuw2.mat"
                exit 14
            }

        [[ -f "$DATASET/scla2.mat" ]] \
            || {
                echo "缺少Stage 7结果：scla2.mat"
                exit 15
            }

        run_merged_stage \
            8 \
            mean_v.mat \
            "${PYSTAMPS_STAGE8_THREADS:-32}" \
            stage \
            mean_v.mat \
            uw_space_time.mat
        ;;

esac

echo
echo "============================================================"
echo "Native Stage $STAGE 完成"
echo "日志目录：$LOG_DIR"
echo "============================================================"
