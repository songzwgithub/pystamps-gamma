from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .gamma_binary import read_gamma_raster
from .gamma_candidates import (
    CandidateConfig,
    CandidateResult,
    _to_amplitude,
    estimate_amplitude_scales,
)
from .gamma_sbas import GammaInputError, GammaSbasProject


def _fmt_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds_i = int(round(seconds))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _read_one_amplitude_block(
    path: Path,
    *,
    width: int,
    length: int,
    block_start: int,
    block_length: int,
    mli_is_power: bool,
    scale: float,
    normalize_per_image: bool,
) -> np.ndarray:
    mli_block = read_gamma_raster(
        path,
        width=width,
        length=length,
        dtype="float",
        y0=block_start,
        ny=block_length,
    )

    amplitude = _to_amplitude(
        mli_block,
        mli_is_power=mli_is_power,
    )
    del mli_block

    if normalize_per_image:
        amplitude = amplitude / scale

    return amplitude


def extract_amplitude_candidates_parallel(
    mli_files: Iterable[str | Path],
    *,
    width: int,
    length: int,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
) -> CandidateResult:
    if config is None:
        config = CandidateConfig()

    paths = [
        Path(path).expanduser().resolve()
        for path in mli_files
    ]

    if len(paths) < 3:
        raise GammaInputError(
            f"至少需要3景MLI，当前仅有{len(paths)}景"
        )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise GammaInputError(
            "以下MLI文件不存在："
            + ", ".join(str(path) for path in missing[:20])
        )

    if row_stop is None:
        row_stop = length

    if (
        row_start < 0
        or row_stop <= row_start
        or row_stop > length
    ):
        raise GammaInputError(
            f"无效行范围：{row_start}:{row_stop}，"
            f"影像总行数为{length}"
        )

    image_count = len(paths)
    minimum_valid_count = max(
        2,
        int(np.ceil(image_count * config.min_valid_fraction)),
    )

    workers = int(
        os.environ.get(
            "PYSTAMPS_DA_WORKERS",
            "8",
        )
    )
    workers = max(1, min(workers, image_count))

    progress_seconds = float(
        os.environ.get(
            "PYSTAMPS_DA_PROGRESS_SECONDS",
            "2",
        )
    )
    progress_seconds = max(0.2, progress_seconds)

    # The legacy parity baseline normally has normalization disabled,
    # in which case this avoids a complete extra pass over all MLI files.
    if config.normalize_per_image:
        print(
            "[D_A] 估计逐景幅度尺度...",
            flush=True,
        )
        scales = estimate_amplitude_scales(
            paths,
            width=width,
            length=length,
            config=config,
        )
    else:
        scales = np.ones(
            image_count,
            dtype=np.float64,
        )

    block_rows = int(config.block_rows)
    n_blocks = int(
        np.ceil(
            (row_stop - row_start)
            / block_rows
        )
    )
    total_reads = n_blocks * image_count

    print()
    print(
        "============================================================",
        flush=True,
    )
    print(
        "并行 D_A 候选点提取",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )
    print(
        f"影像尺寸             : {length} x {width}",
        flush=True,
    )
    print(
        f"MLI景数              : {image_count}",
        flush=True,
    )
    print(
        f"行范围               : {row_start}:{row_stop}",
        flush=True,
    )
    print(
        f"block_rows           : {block_rows}",
        flush=True,
    )
    print(
        f"block数量            : {n_blocks}",
        flush=True,
    )
    print(
        f"并行读取线程         : {workers}",
        flush=True,
    )
    print(
        f"D_A阈值              : {config.da_threshold}",
        flush=True,
    )
    print(
        "逐景幅度归一化       : "
        f"{config.normalize_per_image}",
        flush=True,
    )
    print(
        f"总MLI块读取任务       : {total_reads}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )

    selected_rows: list[np.ndarray] = []
    selected_cols: list[np.ndarray] = []
    selected_da: list[np.ndarray] = []
    selected_mean: list[np.ndarray] = []
    selected_valid_fraction: list[np.ndarray] = []

    run_started = time.monotonic()
    completed_reads = 0
    cumulative_candidates = 0
    last_progress = 0.0

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="pystamps-da",
    ) as executor:

        for block_index, block_start in enumerate(
            range(
                row_start,
                row_stop,
                block_rows,
            ),
            start=1,
        ):
            block_stop = min(
                block_start + block_rows,
                row_stop,
            )
            block_length = block_stop - block_start
            block_shape = (
                block_length,
                width,
            )

            print()
            print(
                f"[D_A] Block {block_index}/{n_blocks} "
                f"rows={block_start}:{block_stop} "
                f"({block_length} rows)",
                flush=True,
            )

            amplitude_sum = np.zeros(
                block_shape,
                dtype=np.float64,
            )
            amplitude_square_sum = np.zeros(
                block_shape,
                dtype=np.float64,
            )
            valid_count = np.zeros(
                block_shape,
                dtype=np.uint16,
            )

            # Submit only one worker-sized batch at a time.
            # This bounds memory while still using multiple CPU cores.
            for batch_start in range(
                0,
                image_count,
                workers,
            ):
                batch_stop = min(
                    batch_start + workers,
                    image_count,
                )

                futures = []

                for image_index in range(
                    batch_start,
                    batch_stop,
                ):
                    futures.append(
                        executor.submit(
                            _read_one_amplitude_block,
                            paths[image_index],
                            width=width,
                            length=length,
                            block_start=block_start,
                            block_length=block_length,
                            mli_is_power=config.mli_is_power,
                            scale=float(
                                scales[image_index]
                            ),
                            normalize_per_image=(
                                config.normalize_per_image
                            ),
                        )
                    )

                # Deliberately accumulate in original acquisition order.
                # Reading is parallel; floating-point reduction order remains
                # compatible with the previous serial implementation.
                for local_index, future in enumerate(
                    futures,
                ):
                    image_index = (
                        batch_start
                        + local_index
                    )

                    amplitude = future.result()

                    valid = (
                        np.isfinite(amplitude)
                        & (amplitude > 0)
                    )

                    amplitude_sum[valid] += (
                        amplitude[valid]
                    )

                    amp64 = amplitude[valid].astype(
                        np.float64,
                        copy=False,
                    )

                    amplitude_square_sum[valid] += (
                        amp64 * amp64
                    )

                    valid_count[valid] += 1

                    del amp64
                    del valid
                    del amplitude

                    completed_reads += 1

                    now = time.monotonic()

                    if (
                        now - last_progress
                        >= progress_seconds
                        or completed_reads
                        == total_reads
                        or image_index
                        == image_count - 1
                    ):
                        elapsed = (
                            now - run_started
                        )
                        fraction = (
                            completed_reads
                            / total_reads
                        )
                        rate = (
                            completed_reads
                            / elapsed
                            if elapsed > 0
                            else 0.0
                        )
                        eta = (
                            (
                                total_reads
                                - completed_reads
                            )
                            / rate
                            if rate > 0
                            else np.nan
                        )

                        current_image = (
                            image_index + 1
                        )

                        print(
                            "[D_A] "
                            f"block={block_index}/{n_blocks} | "
                            f"image={current_image}/{image_count} | "
                            f"overall={completed_reads}/{total_reads} "
                            f"({fraction * 100:6.2f}%) | "
                            f"elapsed={_fmt_seconds(elapsed)} | "
                            f"ETA={_fmt_seconds(eta)} | "
                            f"rate={rate:.2f} MLI-block/s",
                            flush=True,
                        )

                        last_progress = now

            enough = (
                valid_count
                >= minimum_valid_count
            )

            mean_amplitude = np.full(
                block_shape,
                np.nan,
                dtype=np.float64,
            )

            mean_amplitude[enough] = (
                amplitude_sum[enough]
                / valid_count[enough]
            )

            variance = np.full(
                block_shape,
                np.nan,
                dtype=np.float64,
            )

            variance[enough] = (
                amplitude_square_sum[enough]
                - (
                    amplitude_sum[enough] ** 2
                    / valid_count[enough]
                )
            ) / (
                valid_count[enough] - 1
            )

            variance[enough] = np.maximum(
                variance[enough],
                0.0,
            )

            amplitude_dispersion = np.full(
                block_shape,
                np.nan,
                dtype=np.float64,
            )

            usable = (
                enough
                & np.isfinite(mean_amplitude)
                & (mean_amplitude > 0)
                & np.isfinite(variance)
            )

            amplitude_dispersion[usable] = (
                np.sqrt(
                    variance[usable]
                )
                / mean_amplitude[usable]
            )

            candidate_mask = (
                usable
                & np.isfinite(
                    amplitude_dispersion
                )
                & (
                    amplitude_dispersion
                    <= config.da_threshold
                )
            )

            local_rows, cols = np.nonzero(
                candidate_mask
            )

            block_candidate_count = int(
                local_rows.size
            )
            cumulative_candidates += (
                block_candidate_count
            )

            print(
                "[D_A] "
                f"Block {block_index}/{n_blocks} 完成 | "
                f"本块候选={block_candidate_count:,} | "
                f"累计候选={cumulative_candidates:,}",
                flush=True,
            )

            if local_rows.size == 0:
                continue

            rows = (
                local_rows.astype(
                    np.int64
                )
                + block_start
            )

            selected_rows.append(
                rows.astype(
                    np.int32
                )
            )
            selected_cols.append(
                cols.astype(
                    np.int32
                )
            )
            selected_da.append(
                amplitude_dispersion[
                    local_rows,
                    cols,
                ].astype(
                    np.float32
                )
            )
            selected_mean.append(
                mean_amplitude[
                    local_rows,
                    cols,
                ].astype(
                    np.float32
                )
            )
            selected_valid_fraction.append(
                (
                    valid_count[
                        local_rows,
                        cols,
                    ].astype(
                        np.float32
                    )
                    / image_count
                )
            )

    elapsed_total = (
        time.monotonic()
        - run_started
    )

    print()
    print(
        "============================================================",
        flush=True,
    )
    print(
        "D_A候选提取完成",
        flush=True,
    )
    print(
        f"耗时                 : {_fmt_seconds(elapsed_total)}",
        flush=True,
    )
    print(
        f"最终候选点           : {cumulative_candidates:,}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )

    if not selected_rows:
        return CandidateResult(
            rows=np.empty(
                0,
                dtype=np.int32,
            ),
            cols=np.empty(
                0,
                dtype=np.int32,
            ),
            amplitude_dispersion=np.empty(
                0,
                dtype=np.float32,
            ),
            mean_amplitude=np.empty(
                0,
                dtype=np.float32,
            ),
            valid_fraction=np.empty(
                0,
                dtype=np.float32,
            ),
            image_count=image_count,
            config=config,
        )

    return CandidateResult(
        rows=np.concatenate(
            selected_rows
        ),
        cols=np.concatenate(
            selected_cols
        ),
        amplitude_dispersion=np.concatenate(
            selected_da
        ),
        mean_amplitude=np.concatenate(
            selected_mean
        ),
        valid_fraction=np.concatenate(
            selected_valid_fraction
        ),
        image_count=image_count,
        config=config,
    )


def extract_candidates_from_project_parallel(
    project: GammaSbasProject,
    *,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
) -> CandidateResult:
    if (
        project.width is None
        or project.length is None
    ):
        raise GammaInputError(
            "GAMMA工程尚未解析出多视影像行列数"
        )

    missing_dates = [
        acquisition.date
        for acquisition
        in project.acquisitions
        if acquisition.mli is None
    ]

    if missing_dates:
        raise GammaInputError(
            "以下日期缺少MLI："
            + ", ".join(
                missing_dates[:30]
            )
        )

    mli_files = [
        acquisition.mli
        for acquisition
        in project.acquisitions
        if acquisition.mli is not None
    ]

    return extract_amplitude_candidates_parallel(
        mli_files,
        width=int(project.width),
        length=int(project.length),
        config=config,
        row_start=row_start,
        row_stop=row_stop,
    )

