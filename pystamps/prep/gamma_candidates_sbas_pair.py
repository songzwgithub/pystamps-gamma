from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import numpy as np

from .gamma_binary import read_gamma_raster
from .gamma_candidates import (
    CandidateConfig,
    CandidateResult,
    _to_amplitude,
)
from .gamma_sbas import GammaInputError


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _read_amplitude_block(
    path: Path,
    *,
    width: int,
    length: int,
    y0: int,
    ny: int,
    mli_is_power: bool,
) -> np.ndarray:
    raster = read_gamma_raster(
        path,
        width=width,
        length=length,
        dtype="float",
        y0=y0,
        ny=ny,
    )

    return _to_amplitude(
        raster,
        mli_is_power=mli_is_power,
    )


def _calamp_one_image(
    path: Path,
    *,
    width: int,
    length: int,
    mli_is_power: bool,
    block_rows: int,
) -> float:
    total = 0.0
    count = 0

    for y0 in range(0, length, block_rows):
        ny = min(block_rows, length - y0)

        amplitude = _read_amplitude_block(
            path,
            width=width,
            length=length,
            y0=y0,
            ny=ny,
            mli_is_power=mli_is_power,
        )

        valid = (
            np.isfinite(amplitude)
            & (amplitude > 0.001)
        )

        total += float(
            np.sum(
                amplitude[valid],
                dtype=np.float64,
            )
        )

        count += int(
            np.count_nonzero(valid)
        )

    if count == 0:
        raise GammaInputError(
            f"无法计算幅度标定系数：{path}"
        )

    return total / count


def extract_candidates_from_project_sbas_pair(
    project,
    *,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
) -> CandidateResult:
    """
    StaMPS small-baseline pairwise amplitude-dispersion statistic
    on the existing 4:1 GAMMA MLI grid.

    Equivalent algebra to selsbc_patch.c:

        a_m = A_m / calibration_m
        a_s = A_s / calibration_s

        D_A =
            sqrt(sum_edges((a_m-a_s)^2) / N_edges)
            /
            (sum_edges(a_m+a_s) / (2*N_edges))

    A_i is sqrt(MLI power) in this controlled intermediate test.
    """

    if config is None:
        config = CandidateConfig()

    if project.width is None or project.length is None:
        raise GammaInputError(
            "GAMMA工程尚未解析出多视影像尺寸"
        )

    width = int(project.width)
    length = int(project.length)

    if row_stop is None:
        row_stop = length

    if (
        row_start < 0
        or row_stop <= row_start
        or row_stop > length
    ):
        raise GammaInputError(
            f"无效行范围：{row_start}:{row_stop}"
        )

    acquisitions = list(project.acquisitions)
    interferograms = list(project.interferograms)

    if not interferograms:
        raise GammaInputError(
            "SBAS网络中没有干涉对"
        )

    missing_dates = [
        acquisition.date
        for acquisition in acquisitions
        if acquisition.mli is None
    ]

    if missing_dates:
        raise GammaInputError(
            "以下日期缺少MLI："
            + ", ".join(missing_dates[:30])
        )

    mli_files = [
        Path(acquisition.mli).expanduser().resolve()
        for acquisition in acquisitions
    ]

    n_images = len(mli_files)
    n_edges = len(interferograms)

    master_indices = np.asarray(
        [
            int(ifg.master_index) - 1
            for ifg in interferograms
        ],
        dtype=np.int32,
    )

    slave_indices = np.asarray(
        [
            int(ifg.slave_index) - 1
            for ifg in interferograms
        ],
        dtype=np.int32,
    )

    if (
        np.any(master_indices < 0)
        or np.any(master_indices >= n_images)
        or np.any(slave_indices < 0)
        or np.any(slave_indices >= n_images)
    ):
        raise GammaInputError(
            "SBAS网络影像索引越界"
        )

    workers = max(
        1,
        min(
            int(
                os.environ.get(
                    "PYSTAMPS_DA_WORKERS",
                    "8",
                )
            ),
            n_images,
        ),
    )

    pair_block_rows = max(
        64,
        int(
            os.environ.get(
                "PYSTAMPS_SB_PAIR_BLOCK_ROWS",
                "256",
            )
        ),
    )

    calamp_block_rows = max(
        128,
        int(
            os.environ.get(
                "PYSTAMPS_CALAMP_BLOCK_ROWS",
                "2048",
            )
        ),
    )

    print()
    print("=" * 78, flush=True)
    print(
        "StaMPS-SB pairwise D_A（当前4:1 MLI网格）",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        f"影像尺寸              : {length} x {width}",
        flush=True,
    )
    print(
        f"获取日期数            : {n_images}",
        flush=True,
    )
    print(
        f"SB干涉对数            : {n_edges}",
        flush=True,
    )
    print(
        f"D_A阈值               : {config.da_threshold}",
        flush=True,
    )
    print(
        f"并行读取线程          : {workers}",
        flush=True,
    )
    print(
        f"pair block_rows       : {pair_block_rows}",
        flush=True,
    )
    print(
        f"calamp block_rows     : {calamp_block_rows}",
        flush=True,
    )
    print("=" * 78, flush=True)

    # ------------------------------------------------------------
    # Step 1: calamp-like per-acquisition mean amplitude.
    # ------------------------------------------------------------

    print()
    print(
        "[SB-D_A] Step 1/2：计算逐景幅度标定系数",
        flush=True,
    )

    scales = np.empty(
        n_images,
        dtype=np.float64,
    )

    calibration_started = time.monotonic()

    def calibration_job(
        image_index: int,
    ) -> tuple[int, float]:
        value = _calamp_one_image(
            mli_files[image_index],
            width=width,
            length=length,
            mli_is_power=config.mli_is_power,
            block_rows=calamp_block_rows,
        )

        return image_index, value

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="stamps-calamp",
    ) as executor:
        futures = [
            executor.submit(
                calibration_job,
                image_index,
            )
            for image_index in range(n_images)
        ]

        for completed, future in enumerate(
            futures,
            start=1,
        ):
            image_index, scale = future.result()
            scales[image_index] = scale

            if (
                completed == 1
                or completed % 8 == 0
                or completed == n_images
            ):
                elapsed = (
                    time.monotonic()
                    - calibration_started
                )

                rate = (
                    completed / elapsed
                    if elapsed > 0
                    else 0.0
                )

                eta = (
                    (n_images - completed) / rate
                    if rate > 0
                    else np.nan
                )

                print(
                    "[calamp] "
                    f"{completed}/{n_images} "
                    f"({completed / n_images * 100:6.2f}%) | "
                    f"elapsed={_fmt_time(elapsed)} | "
                    f"ETA={_fmt_time(eta)}",
                    flush=True,
                )

    if (
        np.any(~np.isfinite(scales))
        or np.any(scales <= 0)
    ):
        raise GammaInputError(
            "存在无效幅度标定系数"
        )

    print(
        "[calamp] "
        f"median={np.median(scales):.6g}, "
        f"min={np.min(scales):.6g}, "
        f"max={np.max(scales):.6g}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Step 2: selsbc pairwise D_A.
    # ------------------------------------------------------------

    print()
    print(
        "[SB-D_A] Step 2/2：按SB网络成对计算D_A",
        flush=True,
    )

    selected_rows: list[np.ndarray] = []
    selected_cols: list[np.ndarray] = []
    selected_da: list[np.ndarray] = []
    selected_mean: list[np.ndarray] = []
    selected_valid_fraction: list[np.ndarray] = []

    n_blocks = int(
        np.ceil(
            (row_stop - row_start)
            / pair_block_rows
        )
    )

    cumulative_candidates = 0
    pair_started = time.monotonic()

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="stamps-sbda",
    ) as executor:

        for block_index, y0 in enumerate(
            range(
                row_start,
                row_stop,
                pair_block_rows,
            ),
            start=1,
        ):
            y1 = min(
                y0 + pair_block_rows,
                row_stop,
            )

            ny = y1 - y0
            pixel_count = ny * width

            stack = np.empty(
                (
                    n_images,
                    pixel_count,
                ),
                dtype=np.float32,
            )

            print()
            print(
                "[SB-D_A] "
                f"Block {block_index}/{n_blocks} "
                f"rows={y0}:{y1} | "
                f"stack={stack.nbytes / 1024**3:.2f} GiB",
                flush=True,
            )

            for batch_start in range(
                0,
                n_images,
                workers,
            ):
                batch_stop = min(
                    batch_start + workers,
                    n_images,
                )

                futures = [
                    executor.submit(
                        _read_amplitude_block,
                        mli_files[image_index],
                        width=width,
                        length=length,
                        y0=y0,
                        ny=ny,
                        mli_is_power=config.mli_is_power,
                    )
                    for image_index in range(
                        batch_start,
                        batch_stop,
                    )
                ]

                for local_index, future in enumerate(
                    futures
                ):
                    image_index = (
                        batch_start
                        + local_index
                    )

                    amplitude = (
                        future.result()
                        .reshape(-1)
                        .astype(
                            np.float32,
                            copy=False,
                        )
                    )

                    amplitude /= np.float32(
                        scales[image_index]
                    )

                    amplitude = np.nan_to_num(
                        amplitude,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    stack[
                        image_index,
                        :,
                    ] = amplitude

                print(
                    "[SB-D_A] "
                    f"读取日期 "
                    f"{batch_stop}/{n_images} "
                    f"({batch_stop / n_images * 100:6.2f}%)",
                    flush=True,
                )

            sum_amplitude = np.zeros(
                pixel_count,
                dtype=np.float64,
            )

            sum_difference_sq = np.zeros(
                pixel_count,
                dtype=np.float64,
            )

            for edge_index in range(n_edges):
                master64 = (
                    stack[
                        master_indices[
                            edge_index
                        ],
                        :,
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                )

                slave64 = (
                    stack[
                        slave_indices[
                            edge_index
                        ],
                        :,
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                )

                sum_amplitude += (
                    master64
                    + slave64
                )

                difference = (
                    master64
                    - slave64
                )

                sum_difference_sq += (
                    difference
                    * difference
                )

                if (
                    (edge_index + 1) % 50 == 0
                    or edge_index + 1 == n_edges
                ):
                    print(
                        "[SB-D_A] "
                        f"pair={edge_index + 1}/{n_edges}",
                        flush=True,
                    )

            del stack

            mean_amplitude = (
                sum_amplitude
                / (
                    2.0
                    * n_edges
                )
            )

            amplitude_dispersion = np.full(
                pixel_count,
                np.nan,
                dtype=np.float64,
            )

            usable = (
                np.isfinite(mean_amplitude)
                & (mean_amplitude > 0)
            )

            amplitude_dispersion[
                usable
            ] = (
                np.sqrt(
                    sum_difference_sq[
                        usable
                    ]
                    / n_edges
                )
                / mean_amplitude[
                    usable
                ]
            )

            candidate_mask = (
                usable
                & np.isfinite(
                    amplitude_dispersion
                )
                & (
                    amplitude_dispersion
                    < config.da_threshold
                )
            )

            flat_indices = np.flatnonzero(
                candidate_mask
            )

            local_rows = (
                flat_indices
                // width
            ).astype(
                np.int32
            )

            local_cols = (
                flat_indices
                % width
            ).astype(
                np.int32
            )

            block_candidates = int(
                flat_indices.size
            )

            cumulative_candidates += (
                block_candidates
            )

            elapsed = (
                time.monotonic()
                - pair_started
            )

            print(
                "[SB-D_A] "
                f"Block {block_index}/{n_blocks} 完成 | "
                f"本块候选={block_candidates:,} | "
                f"累计候选={cumulative_candidates:,} | "
                f"elapsed={_fmt_time(elapsed)}",
                flush=True,
            )

            if flat_indices.size == 0:
                continue

            selected_rows.append(
                (
                    local_rows
                    + y0
                ).astype(
                    np.int32
                )
            )

            selected_cols.append(
                local_cols
            )

            selected_da.append(
                amplitude_dispersion[
                    flat_indices
                ].astype(
                    np.float32
                )
            )

            selected_mean.append(
                mean_amplitude[
                    flat_indices
                ].astype(
                    np.float32
                )
            )

            selected_valid_fraction.append(
                np.ones(
                    flat_indices.size,
                    dtype=np.float32,
                )
            )

    total_elapsed = (
        time.monotonic()
        - pair_started
    )

    print()
    print("=" * 78, flush=True)
    print(
        "StaMPS-SB pairwise D_A 完成",
        flush=True,
    )
    print(
        f"最终候选点            : "
        f"{cumulative_candidates:,}",
        flush=True,
    )
    print(
        f"pair阶段耗时          : "
        f"{_fmt_time(total_elapsed)}",
        flush=True,
    )
    print("=" * 78, flush=True)

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
            image_count=n_images,
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
        image_count=n_images,
        config=config,
    )
