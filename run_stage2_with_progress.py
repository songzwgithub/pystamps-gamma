#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pySTAMPS Stage 2 with "
            "patch-level progress and ETA."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="pySTAMPS dataset directory",
    )

    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--log-file",
        default=None,
    )

    return parser.parse_args()


def read_patch_names(
    dataset: Path,
) -> list[str]:
    patch_list = dataset / "patch.list"

    if not patch_list.is_file():
        raise FileNotFoundError(
            f"缺少patch.list：{patch_list}"
        )

    names = [
        line.strip()
        for line in patch_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not names:
        raise RuntimeError(
            "patch.list中没有有效patch"
        )

    return names


def completed_stage2_patches(
    dataset: Path,
    patch_names: list[str],
) -> set[str]:
    completed: set[str] = set()

    for patch_name in patch_names:
        path = (
            dataset
            / patch_name
            / "pm1.mat"
        )

        if (
            path.is_file()
            and path.stat().st_size > 0
        ):
            completed.add(
                patch_name
            )

    return completed


def stream_process_output(
    process: subprocess.Popen[str],
    log_handle,
) -> None:
    assert process.stdout is not None

    for line in process.stdout:
        log_handle.write(
            line
        )
        log_handle.flush()


def main() -> int:
    args = parse_args()

    if args.cpu_workers <= 0:
        raise ValueError(
            "cpu-workers必须大于0"
        )

    if args.io_workers <= 0:
        raise ValueError(
            "io-workers必须大于0"
        )

    dataset = Path(
        args.dataset
    ).expanduser().resolve()

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"数据集目录不存在：{dataset}"
        )

    patch_names = read_patch_names(
        dataset
    )

    log_directory = (
        dataset
        / "_run_logs"
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if args.log_file is None:
        log_file = (
            log_directory
            / f"stage2_{timestamp}.log"
        )
    else:
        log_file = Path(
            args.log_file
        ).expanduser().resolve()

    environment = os.environ.copy()

    # 依靠patch或CLAP任务级并行；
    # 禁止BLAS在每个任务中再次启动整机线程。
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
        }
    )

    command = [
        "/usr/bin/time",
        "-v",
        "pystamps",
        "run",
        "--dataset",
        str(
            dataset
        ),
        "--start-step",
        "2",
        "--end-step",
        "2",
        "--cpu-workers",
        str(
            args.cpu_workers
        ),
        "--io-workers",
        str(
            args.io_workers
        ),
    ]

    print(
        "执行命令："
    )
    print(
        " ".join(
            command
        )
    )
    print(
        f"日志文件：{log_file}"
    )
    print(
        f"Patch数量：{len(patch_names)}"
    )
    print()

    completed_before = (
        completed_stage2_patches(
            dataset,
            patch_names,
        )
    )

    with log_file.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )

        output_thread = threading.Thread(
            target=stream_process_output,
            args=(
                process,
                log_handle,
            ),
            daemon=True,
        )

        output_thread.start()

        bar = tqdm(
            total=len(
                patch_names
            ),
            initial=len(
                completed_before
            ),
            unit="patch",
            dynamic_ncols=True,
            desc="Stage 2",
            smoothing=0.15,
        )

        previous_completed = (
            completed_before
        )

        try:
            while process.poll() is None:
                current_completed = (
                    completed_stage2_patches(
                        dataset,
                        patch_names,
                    )
                )

                bar.n = len(
                    current_completed
                )

                new_completed = sorted(
                    current_completed
                    - previous_completed
                )

                if new_completed:
                    bar.set_postfix_str(
                        "latest="
                        + ",".join(
                            new_completed[-3:]
                        )
                    )

                else:
                    bar.set_postfix_str(
                        f"pid={process.pid}"
                    )

                bar.refresh()

                previous_completed = (
                    current_completed
                )

                time.sleep(
                    max(
                        0.2,
                        args.interval,
                    )
                )

        except KeyboardInterrupt:
            bar.write(
                "收到Ctrl+C，正在向pySTAMPS发送SIGINT..."
            )

            os.killpg(
                process.pid,
                signal.SIGINT,
            )

        return_code = (
            process.wait()
        )

        output_thread.join(
            timeout=5
        )

        final_completed = (
            completed_stage2_patches(
                dataset,
                patch_names,
            )
        )

        bar.n = len(
            final_completed
        )

        bar.refresh()
        bar.close()

    print()
    print(
        f"退出码：{return_code}"
    )
    print(
        "完成patch："
        f"{len(final_completed)}/"
        f"{len(patch_names)}"
    )
    print(
        f"日志：{log_file}"
    )

    if return_code != 0:
        print()
        print(
            "Stage 2运行失败，"
            "查看日志末尾："
        )

        subprocess.run(
            [
                "tail",
                "-n",
                "100",
                str(
                    log_file
                ),
            ],
            check=False,
        )

    return return_code


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
