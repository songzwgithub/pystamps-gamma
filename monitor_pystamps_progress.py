#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm


STAGE_OUTPUT_FILES = {
    1: "ps1.mat",
    2: "pm1.mat",
    3: "select1.mat",
    4: "weed1.mat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor pySTAMPS patch-stage progress "
            "by counting completed output MAT files."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="pySTAMPS dataset directory",
    )

    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=sorted(STAGE_OUTPUT_FILES),
        help="Stage number to monitor",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds",
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
            f"patch.list为空：{patch_list}"
        )

    return names


def completed_patches(
    dataset: Path,
    patch_names: list[str],
    output_filename: str,
) -> list[str]:
    completed: list[str] = []

    for patch_name in patch_names:
        output_file = (
            dataset
            / patch_name
            / output_filename
        )

        if (
            output_file.is_file()
            and output_file.stat().st_size > 0
        ):
            completed.append(
                patch_name
            )

    return completed


def main() -> int:
    args = parse_args()

    dataset = Path(
        args.dataset
    ).expanduser().resolve()

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"数据集目录不存在：{dataset}"
        )

    output_filename = (
        STAGE_OUTPUT_FILES[
            args.stage
        ]
    )

    patch_names = read_patch_names(
        dataset
    )

    completed_before = set(
        completed_patches(
            dataset,
            patch_names,
            output_filename,
        )
    )

    total = len(
        patch_names
    )

    print(
        f"Dataset     : {dataset}"
    )
    print(
        f"Stage       : {args.stage}"
    )
    print(
        f"Output file : {output_filename}"
    )
    print(
        f"Patch count : {total}"
    )
    print(
        f"Completed   : {len(completed_before)}"
    )
    print()

    bar = tqdm(
        total=total,
        initial=len(
            completed_before
        ),
        unit="patch",
        dynamic_ncols=True,
        desc=f"Stage {args.stage}",
        smoothing=0.15,
    )

    previous_completed = (
        completed_before
    )

    try:
        while True:
            current_completed = set(
                completed_patches(
                    dataset,
                    patch_names,
                    output_filename,
                )
            )

            current_count = len(
                current_completed
            )

            bar.n = current_count

            new_completed = sorted(
                current_completed
                - previous_completed
            )

            if new_completed:
                latest = ",".join(
                    new_completed[-3:]
                )

                bar.set_postfix_str(
                    f"latest={latest}"
                )

            else:
                bar.set_postfix_str(
                    "waiting"
                )

            bar.refresh()

            if current_count >= total:
                break

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
            "停止进度监控；"
            "不会终止正在运行的pySTAMPS任务。"
        )

        return 130

    finally:
        bar.close()

    print()
    print(
        f"Stage {args.stage}完成："
        f"{total}/{total} patches"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
