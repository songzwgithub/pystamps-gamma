#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import TextIO


STAGE_NAMES = {
    1: "Initial load",
    2: "Estimate gamma",
    3: "Select PS pixels",
    4: "Weed adjacent pixels",
    5: "Correct phase + merge",
    6: "Unwrap phase",
    7: "Calculate SCLA",
    8: "Filter SCN",
}

PATCH_ARTIFACTS = {
    1: "ps1.mat",
    2: "pm1.mat",
    3: "select1.mat",
    4: "weed1.mat",
    5: "ps2.mat",
}

MERGED_ARTIFACTS = {
    5: "ifgstd2.mat",
    6: "phuw2.mat",
    7: "scla2.mat",
    8: "mean_v.mat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pySTAMPS with stage, patch, resource and ETA progress."
    )

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=8)
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=2)

    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--bar-width", type=int, default=36)
    parser.add_argument("--log", type=Path, default=None)

    parser.add_argument(
        "--stage3-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stage3-threads", type=int, default=8)
    parser.add_argument(
        "--stage3-single-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--clap-ifg-workers", type=int, default=0)
    parser.add_argument("--clap-window-batch", type=int, default=8)
    parser.add_argument("--clap-fft-workers", type=int, default=1)

    return parser.parse_args()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not (seconds >= 0):
        return "--:--:--"

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds_i = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds_i:02d}"


def patch_directories(dataset: Path) -> list[Path]:
    def patch_number(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("PATCH_")
        try:
            return int(suffix), path.name
        except ValueError:
            return 10**9, path.name

    return sorted(
        (
            path
            for path in dataset.glob("PATCH_*")
            if path.is_dir()
        ),
        key=patch_number,
    )


def count_completed(
    dataset: Path,
    patches: list[Path],
    stage: int,
) -> tuple[int, int, bool]:
    patch_artifact = PATCH_ARTIFACTS.get(stage)

    if patch_artifact is not None:
        completed = sum(
            int((patch / patch_artifact).exists())
            for patch in patches
        )
        total = len(patches)

        merged_name = MERGED_ARTIFACTS.get(stage)
        merged_done = (
            True
            if merged_name is None
            else (dataset / merged_name).exists()
        )

        return completed, total, merged_done

    merged_name = MERGED_ARTIFACTS.get(stage)

    if merged_name is None:
        return 0, 1, False

    done = int((dataset / merged_name).exists())
    return done, 1, bool(done)


def render_bar(
    completed: int,
    total: int,
    width: int,
) -> str:
    if total <= 0:
        fraction = 0.0
    else:
        fraction = min(
            1.0,
            max(0.0, completed / total),
        )

    filled = int(round(width * fraction))

    return (
        "["
        + "#" * filled
        + "-" * (width - filled)
        + "]"
    )


def process_tree_stats(root_pid: int) -> dict[str, float | int]:
    try:
        output = subprocess.check_output(
            [
                "ps",
                "-eo",
                "pid=,ppid=,pcpu=,rss=,nlwp=,stat=",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {
            "processes": 1,
            "threads": 1,
            "cpu_percent": 0.0,
            "rss_mib": 0.0,
        }

    rows: dict[int, tuple[int, float, int, int, str]] = {}

    for line in output.splitlines():
        fields = line.split()

        if len(fields) < 6:
            continue

        try:
            pid = int(fields[0])
            ppid = int(fields[1])
            cpu = float(fields[2])
            rss_kib = int(fields[3])
            nlwp = int(fields[4])
            state = fields[5]
        except ValueError:
            continue

        rows[pid] = (
            ppid,
            cpu,
            rss_kib,
            nlwp,
            state,
        )

    descendants = {root_pid}
    changed = True

    while changed:
        changed = False

        for pid, values in rows.items():
            ppid = values[0]

            if (
                ppid in descendants
                and pid not in descendants
            ):
                descendants.add(pid)
                changed = True

    cpu_total = 0.0
    rss_total = 0
    thread_total = 0
    active_processes = 0

    for pid in descendants:
        row = rows.get(pid)

        if row is None:
            continue

        _, cpu, rss_kib, nlwp, state = row

        cpu_total += cpu
        rss_total += rss_kib
        thread_total += nlwp

        if not state.startswith("Z"):
            active_processes += 1

    return {
        "processes": active_processes,
        "threads": thread_total,
        "cpu_percent": cpu_total,
        "rss_mib": rss_total / 1024.0,
    }


def atomic_write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def output_reader(
    stream: TextIO,
    log_handle: TextIO,
    finished: threading.Event,
) -> None:
    try:
        for line in iter(stream.readline, ""):
            log_handle.write(line)
            log_handle.flush()

            print(
                line.rstrip("\n"),
                flush=True,
            )
    finally:
        finished.set()


def build_environment(
    args: argparse.Namespace,
    stage: int,
) -> dict[str, str]:
    env = os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"

    if stage == 2:
        env["PYSTAMPS_CLAP_PROGRESS"] = "1"
        env["PYSTAMPS_CLAP_WINDOW_BATCH"] = str(
            max(1, args.clap_window_batch)
        )
        env["PYSTAMPS_CLAP_FFT_WORKERS"] = str(
            max(1, args.clap_fft_workers)
        )

        if args.clap_ifg_workers > 0:
            env["PYSTAMPS_CLAP_IFG_WORKERS"] = str(
                args.clap_ifg_workers
            )

    if stage == 3:
        env["PYSTAMPS_STAGE3_FAST"] = (
            "1"
            if args.stage3_fast
            else "0"
        )

        env["PYSTAMPS_STAGE3_THREADS"] = str(
            max(1, args.stage3_threads)
        )

        env["PYSTAMPS_STAGE3_PROGRESS"] = "1"

        env["PYSTAMPS_STAGE3_SINGLE_PRECISION"] = (
            "1"
            if args.stage3_single_precision
            else "0"
        )

    return env


def build_command(
    args: argparse.Namespace,
    dataset: Path,
    stage: int,
) -> list[str]:
    command = ["pystamps"]

    if args.config is not None:
        command.extend(
            [
                "--config",
                str(args.config.resolve()),
            ]
        )

    command.extend(
        [
            "run",
            "--dataset",
            str(dataset),
            "--start-step",
            str(stage),
            "--end-step",
            str(stage),
            "--cpu-workers",
            str(args.cpu_workers),
            "--io-workers",
            str(args.io_workers),
        ]
    )

    return command


def run_stage(
    args: argparse.Namespace,
    dataset: Path,
    patches: list[Path],
    stage: int,
    log_handle: TextIO,
    progress_file: Path,
) -> int:
    name = STAGE_NAMES[stage]

    initial_completed, total, initial_merged = count_completed(
        dataset,
        patches,
        stage,
    )

    command = build_command(
        args,
        dataset,
        stage,
    )

    env = build_environment(
        args,
        stage,
    )

    print()
    print("=" * 88)
    print(f"[STAGE {stage}] {name}")
    print(f"数据集      : {dataset}")
    print(f"命令        : {' '.join(command)}")
    print(f"初始完成    : {initial_completed}/{total}")
    print(f"CPU workers : {args.cpu_workers}")
    print(f"IO workers  : {args.io_workers}")

    if stage == 3:
        print(f"Stage3 fast : {args.stage3_fast}")
        print(f"Stage3线程  : {args.stage3_threads}")
        print(
            "Stage3精度  : "
            + (
                "complex64"
                if args.stage3_single_precision
                else "complex128"
            )
        )

    print("=" * 88)
    print(flush=True)

    process = subprocess.Popen(
        command,
        env=env,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    reader_finished = threading.Event()

    reader = threading.Thread(
        target=output_reader,
        args=(
            process.stdout,
            log_handle,
            reader_finished,
        ),
        daemon=True,
    )

    reader.start()

    started = time.perf_counter()
    last_completed = initial_completed

    while process.poll() is None:
        completed, total_now, merged_done = count_completed(
            dataset,
            patches,
            stage,
        )

        stats = process_tree_stats(
            process.pid
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        newly_completed = (
            completed
            - initial_completed
        )

        if newly_completed > 0:
            rate = newly_completed / elapsed

            remaining = max(
                0,
                total_now - completed,
            )

            eta = (
                remaining / rate
                if rate > 0
                else None
            )
        else:
            eta = None

        bar = render_bar(
            completed,
            total_now,
            args.bar_width,
        )

        percent = (
            100.0 * completed / total_now
            if total_now > 0
            else 0.0
        )

        phase_text = ""

        if (
            stage == 5
            and completed == total_now
            and not merged_done
        ):
            phase_text = " | patch完成，正在合并"

        progress_line = (
            f"[PROGRESS][S{stage}] "
            f"{bar} "
            f"{completed}/{total_now} "
            f"{percent:6.2f}% "
            f"| elapsed={format_seconds(elapsed)} "
            f"| eta={format_seconds(eta)} "
            f"| CPU={stats['cpu_percent']:.1f}% "
            f"| RSS={stats['rss_mib'] / 1024.0:.2f}GiB "
            f"| proc={stats['processes']} "
            f"| threads={stats['threads']}"
            f"{phase_text}"
        )

        print(
            progress_line,
            file=sys.stderr,
            flush=True,
        )

        state = {
            "dataset": str(dataset),
            "stage": stage,
            "stage_name": name,
            "status": "running",
            "completed": completed,
            "total": total_now,
            "percent": percent,
            "merged_artifact_ready": merged_done,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "process_tree": stats,
            "command": command,
            "updated_epoch_sec": time.time(),
        }

        atomic_write_json(
            progress_file,
            state,
        )

        last_completed = completed

        time.sleep(
            max(
                0.5,
                args.interval,
            )
        )

    return_code = process.wait()

    reader_finished.wait(
        timeout=10
    )

    reader.join(
        timeout=1
    )

    completed, total_final, merged_done = count_completed(
        dataset,
        patches,
        stage,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    status = (
        "completed"
        if return_code == 0
        else "failed"
    )

    final_bar = render_bar(
        completed,
        total_final,
        args.bar_width,
    )

    print()
    print(
        f"[STAGE {stage}] {status.upper()} "
        f"{final_bar} "
        f"{completed}/{total_final} "
        f"| elapsed={format_seconds(elapsed)} "
        f"| exit={return_code}",
        flush=True,
    )

    atomic_write_json(
        progress_file,
        {
            "dataset": str(dataset),
            "stage": stage,
            "stage_name": name,
            "status": status,
            "completed": completed,
            "total": total_final,
            "merged_artifact_ready": merged_done,
            "elapsed_sec": elapsed,
            "exit_code": return_code,
            "command": command,
            "updated_epoch_sec": time.time(),
        },
    )

    return return_code


def main() -> int:
    args = parse_args()

    dataset = args.dataset.expanduser().resolve()

    if not dataset.exists():
        raise SystemExit(
            f"数据集不存在：{dataset}"
        )

    if not (
        1
        <= args.start_step
        <= args.end_step
        <= 8
    ):
        raise SystemExit(
            "Stage范围必须满足1 <= start <= end <= 8"
        )

    patches = patch_directories(
        dataset
    )

    if not patches:
        raise SystemExit(
            f"没有发现PATCH_*目录：{dataset}"
        )

    log_directory = (
        dataset
        / "_run_logs"
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        args.log.expanduser().resolve()
        if args.log is not None
        else (
            log_directory
            / (
                f"pipeline_"
                f"{args.start_step}_"
                f"{args.end_step}_"
                f"{timestamp}.log"
            )
        )
    )

    progress_file = (
        log_directory
        / "pipeline_progress.json"
    )

    print(f"数据集：{dataset}")
    print(f"Patch数量：{len(patches)}")
    print(f"日志：{log_path}")
    print(f"状态文件：{progress_file}")

    with log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_handle:
        for stage in range(
            args.start_step,
            args.end_step + 1,
        ):
            return_code = run_stage(
                args,
                dataset,
                patches,
                stage,
                log_handle,
                progress_file,
            )

            if return_code != 0:
                print(
                    f"Stage {stage}失败，终止后续流程。",
                    file=sys.stderr,
                )
                return return_code

    print()
    print("全部指定Stage运行完成。")
    print(f"日志：{log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
