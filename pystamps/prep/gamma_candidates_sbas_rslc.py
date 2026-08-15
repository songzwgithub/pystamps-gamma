from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import warnings

import numpy as np

from .gamma_binary import (
    read_gamma_raster,
    resolve_gamma_raster_layout,
)
from .gamma_candidates import (
    CandidateConfig,
    CandidateResult,
)
from .gamma_sbas import GammaInputError


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"

    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _read_rslc_amplitude(
    rslc: Path,
    par: Path,
    *,
    y0: int,
    ny: int,
    crop_width: int,
) -> np.ndarray:
    complex_block = read_gamma_raster(
        rslc,
        parameter_file=par,
        y0=y0,
        ny=ny,
    )

    if complex_block.shape[1] < crop_width:
        raise GammaInputError(
            f"RSLC宽度不足：{rslc}, "
            f"{complex_block.shape[1]} < {crop_width}"
        )

    complex_block = complex_block[
        :,
        :crop_width,
    ]

    return np.abs(
        complex_block
    ).astype(
        np.float32,
        copy=False,
    )


def _calamp_rslc(
    rslc: Path,
    par: Path,
    *,
    full_length: int,
    crop_width: int,
    block_rows: int,
) -> float:
    """
    StaMPS calamp.c equivalent:
    mean abs(SLC) over amplitudes > 0.001.
    """

    total = 0.0
    count = 0

    for y0 in range(
        0,
        full_length,
        block_rows,
    ):
        ny = min(
            block_rows,
            full_length - y0,
        )

        amplitude = _read_rslc_amplitude(
            rslc,
            par,
            y0=y0,
            ny=ny,
            crop_width=crop_width,
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

    if count <= 0:
        raise GammaInputError(
            f"RSLC无有效振幅：{rslc}"
        )

    return total / count


def extract_candidates_from_project_rslc_sbas(
    project,
    *,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
    range_looks: int | None = None,
    azimuth_looks: int | None = None,
) -> CandidateResult:
    """
    StaMPS SB candidate selection from original complex RSLC amplitudes,
    followed by deterministic mapping to the existing multilook IFG grid.

    Single-look statistic:
        a_m = abs(SLC_m) / calibration_m
        a_s = abs(SLC_s) / calibration_s

        D_A =
            sqrt(sum_edges((a_m-a_s)^2) / N_edges)
            /
            (sum_edges(a_m+a_s) / (2*N_edges))

    Mapping to multilook cell:
        D_A_ml = minimum single-look D_A inside the
                 range_looks x azimuth_looks cell.

    This mapping means a multilook cell is retained when at least one of its
    constituent single-look samples satisfies the StaMPS SB amplitude test.
    """

    if config is None:
        config = CandidateConfig()

    if (
        project.width is None
        or project.length is None
    ):
        raise GammaInputError(
            "GAMMA工程尚未解析出多视IFG尺寸"
        )

    ml_width = int(project.width)
    ml_length = int(project.length)

    range_looks = int(
        range_looks
        if range_looks is not None
        else os.environ.get(
            "PYSTAMPS_RANGE_LOOKS",
            "4",
        )
    )

    azimuth_looks = int(
        azimuth_looks
        if azimuth_looks is not None
        else os.environ.get(
            "PYSTAMPS_AZIMUTH_LOOKS",
            "1",
        )
    )

    if (
        range_looks <= 0
        or azimuth_looks <= 0
    ):
        raise GammaInputError(
            "range_looks/azimuth_looks必须为正整数"
        )

    acquisitions = list(
        project.acquisitions
    )

    interferograms = list(
        project.interferograms
    )

    if not acquisitions:
        raise GammaInputError(
            "没有RSLC获取日期"
        )

    if not interferograms:
        raise GammaInputError(
            "没有SBAS干涉对"
        )

    n_images = len(acquisitions)
    n_edges = len(interferograms)

    # Resolve and verify original RSLC dimensions.
    layouts = [
        resolve_gamma_raster_layout(
            acquisition.rslc,
            parameter_file=acquisition.par,
        )
        for acquisition in acquisitions
    ]

    full_widths = {
        int(layout.width)
        for layout in layouts
    }

    full_lengths = {
        int(layout.length)
        for layout in layouts
    }

    if len(full_widths) != 1:
        raise GammaInputError(
            "各RSLC宽度不一致："
            + ", ".join(
                str(v)
                for v in sorted(full_widths)
            )
        )

    if len(full_lengths) != 1:
        raise GammaInputError(
            "各RSLC行数不一致："
            + ", ".join(
                str(v)
                for v in sorted(full_lengths)
            )
        )

    full_width = next(
        iter(full_widths)
    )

    full_length = next(
        iter(full_lengths)
    )

    required_full_width = (
        ml_width
        * range_looks
    )

    required_full_length = (
        ml_length
        * azimuth_looks
    )

    if full_width < required_full_width:
        raise GammaInputError(
            "RSLC宽度小于多视网格所需宽度："
            f"{full_width} < {required_full_width}"
        )

    if full_length < required_full_length:
        raise GammaInputError(
            "RSLC行数小于多视网格所需行数："
            f"{full_length} < {required_full_length}"
        )

    # GAMMA multilooking commonly drops remainder samples at image edges.
    range_remainder = (
        full_width
        - required_full_width
    )

    azimuth_remainder = (
        full_length
        - required_full_length
    )

    if range_remainder >= range_looks:
        raise GammaInputError(
            "RSLC与4:1多视IFG宽度关系异常："
            f"full={full_width}, ml={ml_width}, "
            f"range_looks={range_looks}"
        )

    if azimuth_remainder >= azimuth_looks:
        raise GammaInputError(
            "RSLC与多视IFG行数关系异常："
            f"full={full_length}, ml={ml_length}, "
            f"azimuth_looks={azimuth_looks}"
        )

    if row_stop is None:
        row_stop = ml_length

    if (
        row_start < 0
        or row_stop <= row_start
        or row_stop > ml_length
    ):
        raise GammaInputError(
            f"无效多视行范围：{row_start}:{row_stop}"
        )

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
            "SBAS网络RSLC索引越界"
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

    ml_block_rows = max(
        16,
        int(
            os.environ.get(
                "PYSTAMPS_RSLC_ML_BLOCK_ROWS",
                "64",
            )
        ),
    )

    calamp_block_rows = max(
        64,
        int(
            os.environ.get(
                "PYSTAMPS_RSLC_CALAMP_BLOCK_ROWS",
                "256",
            )
        ),
    )

    print()
    print("=" * 82, flush=True)
    print(
        "StaMPS-SB RSLC candidate extraction + multilook-grid mapping",
        flush=True,
    )
    print("=" * 82, flush=True)
    print(
        f"RSLC尺寸                : {full_length} x {full_width}",
        flush=True,
    )
    print(
        f"4:1 IFG尺寸             : {ml_length} x {ml_width}",
        flush=True,
    )
    print(
        f"range_looks             : {range_looks}",
        flush=True,
    )
    print(
        f"azimuth_looks           : {azimuth_looks}",
        flush=True,
    )
    print(
        f"边缘舍弃                : "
        f"range={range_remainder}, "
        f"azimuth={azimuth_remainder}",
        flush=True,
    )
    print(
        f"获取日期数              : {n_images}",
        flush=True,
    )
    print(
        f"SB干涉对数              : {n_edges}",
        flush=True,
    )
    print(
        f"D_A阈值                 : {config.da_threshold}",
        flush=True,
    )
    print(
        f"并行读取线程            : {workers}",
        flush=True,
    )
    print(
        f"ML block_rows           : {ml_block_rows}",
        flush=True,
    )
    print(
        "映射方法                : "
        "min single-look D_A per multilook cell",
        flush=True,
    )
    print("=" * 82, flush=True)

    # ============================================================
    # Step 1: exact calamp-style scale from original complex RSLC.
    # ============================================================

    print()
    print(
        "[RSLC-D_A] Step 1/2：calamp逐景标定",
        flush=True,
    )

    scales = np.empty(
        n_images,
        dtype=np.float64,
    )

    scale_started = time.monotonic()

    def scale_job(
        image_index: int,
    ) -> tuple[int, float]:
        acquisition = acquisitions[
            image_index
        ]

        value = _calamp_rslc(
            acquisition.rslc,
            acquisition.par,
            full_length=(
                required_full_length
            ),
            crop_width=(
                required_full_width
            ),
            block_rows=(
                calamp_block_rows
            ),
        )

        return (
            image_index,
            value,
        )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="rslc-calamp",
    ) as executor:
        futures = [
            executor.submit(
                scale_job,
                image_index,
            )
            for image_index
            in range(n_images)
        ]

        for completed, future in enumerate(
            futures,
            start=1,
        ):
            image_index, value = (
                future.result()
            )

            scales[
                image_index
            ] = value

            if (
                completed == 1
                or completed % 8 == 0
                or completed == n_images
            ):
                elapsed = (
                    time.monotonic()
                    - scale_started
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
                    "[calamp-RSLC] "
                    f"{completed}/{n_images} "
                    f"({completed/n_images*100:6.2f}%) | "
                    f"elapsed={_fmt_time(elapsed)} | "
                    f"ETA={_fmt_time(eta)}",
                    flush=True,
                )

    if (
        np.any(~np.isfinite(scales))
        or np.any(scales <= 0)
    ):
        raise GammaInputError(
            "RSLC幅度标定系数存在无效值"
        )

    print(
        "[calamp-RSLC] "
        f"median={np.median(scales):.6g}, "
        f"min={np.min(scales):.6g}, "
        f"max={np.max(scales):.6g}",
        flush=True,
    )

    # ============================================================
    # Step 2: exact single-look pairwise D_A, then map to ML cells.
    # ============================================================

    print()
    print(
        "[RSLC-D_A] Step 2/2："
        "single-look selsbc统计并映射到4:1网格",
        flush=True,
    )

    selected_rows: list[
        np.ndarray
    ] = []
    selected_cols: list[
        np.ndarray
    ] = []
    selected_da: list[
        np.ndarray
    ] = []
    selected_mean: list[
        np.ndarray
    ] = []
    selected_valid_fraction: list[
        np.ndarray
    ] = []

    n_blocks = int(
        np.ceil(
            (row_stop - row_start)
            / ml_block_rows
        )
    )

    cumulative_single_candidates = 0
    cumulative_ml_candidates = 0
    process_started = time.monotonic()

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="rslc-sbda",
    ) as executor:

        for block_index, ml_y0 in enumerate(
            range(
                row_start,
                row_stop,
                ml_block_rows,
            ),
            start=1,
        ):
            ml_y1 = min(
                ml_y0 + ml_block_rows,
                row_stop,
            )

            ml_ny = (
                ml_y1
                - ml_y0
            )

            full_y0 = (
                ml_y0
                * azimuth_looks
            )

            full_ny = (
                ml_ny
                * azimuth_looks
            )

            full_block_width = (
                required_full_width
            )

            single_pixel_count = (
                full_ny
                * full_block_width
            )

            stack = np.empty(
                (
                    n_images,
                    single_pixel_count,
                ),
                dtype=np.float32,
            )

            print()
            print(
                "[RSLC-D_A] "
                f"Block {block_index}/{n_blocks} | "
                f"ML rows={ml_y0}:{ml_y1} | "
                f"RSLC rows={full_y0}:"
                f"{full_y0 + full_ny} | "
                f"stack={stack.nbytes/1024**3:.2f} GiB",
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

                futures = []

                for image_index in range(
                    batch_start,
                    batch_stop,
                ):
                    acquisition = acquisitions[
                        image_index
                    ]

                    futures.append(
                        executor.submit(
                            _read_rslc_amplitude,
                            acquisition.rslc,
                            acquisition.par,
                            y0=full_y0,
                            ny=full_ny,
                            crop_width=(
                                full_block_width
                            ),
                        )
                    )

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
                        scales[
                            image_index
                        ]
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
                    "[RSLC-D_A] "
                    f"读取RSLC "
                    f"{batch_stop}/{n_images} "
                    f"({batch_stop/n_images*100:6.2f}%)",
                    flush=True,
                )

            sum_amplitude = np.zeros(
                single_pixel_count,
                dtype=np.float64,
            )

            sum_difference_sq = np.zeros(
                single_pixel_count,
                dtype=np.float64,
            )

            for edge_index in range(
                n_edges
            ):
                master = (
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

                slave = (
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
                    master + slave
                )

                difference = (
                    master - slave
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
                        "[RSLC-D_A] "
                        f"pair={edge_index+1}/{n_edges}",
                        flush=True,
                    )

            del stack

            mean_single = (
                sum_amplitude
                / (
                    2.0
                    * n_edges
                )
            )

            da_single = np.full(
                single_pixel_count,
                np.inf,
                dtype=np.float64,
            )

            usable_single = (
                np.isfinite(
                    mean_single
                )
                & (
                    mean_single > 0
                )
            )

            da_single[
                usable_single
            ] = (
                np.sqrt(
                    sum_difference_sq[
                        usable_single
                    ]
                    / n_edges
                )
                / mean_single[
                    usable_single
                ]
            )

            single_candidate_count = int(
                np.count_nonzero(
                    da_single
                    < config.da_threshold
                )
            )

            cumulative_single_candidates += (
                single_candidate_count
            )

            # ----------------------------------------------------
            # Map full-resolution samples into multilook cells.
            # Array order:
            #   [ml_row, az_look, ml_col, range_look]
            # ----------------------------------------------------

            da_4d = da_single.reshape(
                ml_ny,
                azimuth_looks,
                ml_width,
                range_looks,
            )

            mean_4d = mean_single.reshape(
                ml_ny,
                azimuth_looks,
                ml_width,
                range_looks,
            )

            da_group = (
                da_4d.transpose(
                    0,
                    2,
                    1,
                    3,
                )
                .reshape(
                    ml_ny,
                    ml_width,
                    (
                        azimuth_looks
                        * range_looks
                    ),
                )
            )

            mean_group = (
                mean_4d.transpose(
                    0,
                    2,
                    1,
                    3,
                )
                .reshape(
                    ml_ny,
                    ml_width,
                    (
                        azimuth_looks
                        * range_looks
                    ),
                )
            )

            best_look = np.argmin(
                da_group,
                axis=2,
            )

            da_ml = np.take_along_axis(
                da_group,
                best_look[
                    :,
                    :,
                    None,
                ],
                axis=2,
            )[
                :,
                :,
                0,
            ]

            mean_ml = np.take_along_axis(
                mean_group,
                best_look[
                    :,
                    :,
                    None,
                ],
                axis=2,
            )[
                :,
                :,
                0,
            ]

            candidate_mask = (
                np.isfinite(da_ml)
                & (
                    da_ml
                    < config.da_threshold
                )
            )

            local_rows, local_cols = (
                np.nonzero(
                    candidate_mask
                )
            )

            ml_candidate_count = int(
                local_rows.size
            )

            cumulative_ml_candidates += (
                ml_candidate_count
            )

            elapsed = (
                time.monotonic()
                - process_started
            )

            print(
                "[RSLC-D_A] "
                f"Block {block_index}/{n_blocks} 完成 | "
                f"single-look候选="
                f"{single_candidate_count:,} | "
                f"4:1映射候选="
                f"{ml_candidate_count:,} | "
                f"累计4:1="
                f"{cumulative_ml_candidates:,} | "
                f"elapsed={_fmt_time(elapsed)}",
                flush=True,
            )

            if local_rows.size == 0:
                continue

            selected_rows.append(
                (
                    local_rows
                    + ml_y0
                ).astype(
                    np.int32
                )
            )

            selected_cols.append(
                local_cols.astype(
                    np.int32
                )
            )

            selected_da.append(
                da_ml[
                    local_rows,
                    local_cols,
                ].astype(
                    np.float32
                )
            )

            selected_mean.append(
                mean_ml[
                    local_rows,
                    local_cols,
                ].astype(
                    np.float32
                )
            )

            selected_valid_fraction.append(
                np.ones(
                    local_rows.size,
                    dtype=np.float32,
                )
            )

    elapsed_total = (
        time.monotonic()
        - process_started
    )

    print()
    print("=" * 82, flush=True)
    print(
        "RSLC StaMPS-SB candidate extraction完成",
        flush=True,
    )
    print(
        f"累计single-look候选      : "
        f"{cumulative_single_candidates:,}",
        flush=True,
    )
    print(
        f"映射后4:1候选           : "
        f"{cumulative_ml_candidates:,}",
        flush=True,
    )
    print(
        f"处理耗时                : "
        f"{_fmt_time(elapsed_total)}",
        flush=True,
    )
    print("=" * 82, flush=True)

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
