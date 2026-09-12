from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .gamma_binary import (
    iter_gamma_raster_blocks,
    read_gamma_raster,
)
from .gamma_sbas import (
    GammaInputError,
    GammaSbasProject,
)


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """Configuration for amplitude-dispersion candidate extraction."""

    da_threshold: float = 0.6
    min_valid_fraction: float = 0.90
    block_rows: int = 256

    # GAMMA MLI通常是功率/强度，而非幅度。
    mli_is_power: bool = True

    # 消除不同日期整体辐射尺度差异。
    normalize_per_image: bool = True

    def __post_init__(self) -> None:
        if self.da_threshold <= 0:
            raise ValueError(
                "da_threshold必须大于0"
            )

        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError(
                "min_valid_fraction必须位于(0, 1]"
            )

        if self.block_rows <= 0:
            raise ValueError(
                "block_rows必须大于0"
            )


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Selected zero-based multilooked candidate pixels."""

    rows: np.ndarray
    cols: np.ndarray
    amplitude_dispersion: np.ndarray
    mean_amplitude: np.ndarray
    valid_fraction: np.ndarray
    image_count: int
    config: CandidateConfig

    @property
    def count(self) -> int:
        return int(self.rows.size)

    def to_stamps_ij(self) -> np.ndarray:
        """
        Return StaMPS-style one-based:
            candidate_id, azimuth_row, range_column
        """

        candidate_id = np.arange(
            1,
            self.count + 1,
            dtype=np.int32,
        )

        return np.column_stack(
            (
                candidate_id,
                self.rows.astype(
                    np.int32,
                    copy=False,
                ) + 1,
                self.cols.astype(
                    np.int32,
                    copy=False,
                ) + 1,
            )
        ).astype(
            np.int32,
            copy=False,
        )


def _to_amplitude(
    values: np.ndarray,
    *,
    mli_is_power: bool,
) -> np.ndarray:
    image = np.asarray(
        values,
        dtype=np.float32,
    )

    amplitude = np.full(
        image.shape,
        np.nan,
        dtype=np.float32,
    )

    finite = np.isfinite(image)

    if mli_is_power:
        usable = finite & (image > 0)

        amplitude[usable] = np.sqrt(
            image[usable]
        )
    else:
        usable = finite & (image > 0)
        amplitude[usable] = image[usable]

    return amplitude


def estimate_amplitude_scales(
    mli_files: Iterable[str | Path],
    *,
    width: int,
    length: int,
    config: CandidateConfig,
) -> np.ndarray:
    """
    Estimate one global mean-amplitude scale per MLI.

    Only valid positive pixels are included.
    """

    paths = [
        Path(path).expanduser().resolve()
        for path in mli_files
    ]

    if not paths:
        raise GammaInputError(
            "没有输入MLI文件"
        )

    if not config.normalize_per_image:
        return np.ones(
            len(paths),
            dtype=np.float64,
        )

    scales = np.empty(
        len(paths),
        dtype=np.float64,
    )

    for image_index, path in enumerate(paths):
        total = 0.0
        count = 0

        for _, _, block in iter_gamma_raster_blocks(
            path,
            width=width,
            length=length,
            dtype="float",
            block_rows=config.block_rows,
        ):
            amplitude = _to_amplitude(
                block,
                mli_is_power=config.mli_is_power,
            )

            valid = (
                np.isfinite(amplitude)
                & (amplitude > 0)
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
                f"MLI没有有效正值像元：{path}"
            )

        scales[image_index] = total / count

        if (
            not np.isfinite(scales[image_index])
            or scales[image_index] <= 0
        ):
            raise GammaInputError(
                f"MLI幅度标定系数无效："
                f"{path} -> {scales[image_index]}"
            )

    return scales


def extract_amplitude_candidates(
    mli_files: Iterable[str | Path],
    *,
    width: int,
    length: int,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
) -> CandidateResult:
    """
    Parallel amplitude-dispersion candidate extraction with progress.

    The scientific calculation is unchanged:
        D_A = std(amplitude) / mean(amplitude)

    MLI reads / amplitude conversion are parallelized. Accumulation remains
    in acquisition order to minimize numerical differences from the serial
    implementation.
    """

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

    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise GammaInputError(
            "以下MLI文件不存在："
            + ", ".join(
                str(path)
                for path in missing[:20]
            )
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
        int(
            np.ceil(
                image_count
                * config.min_valid_fraction
            )
        ),
    )

    workers = int(
        os.environ.get(
            "PYSTAMPS_DA_WORKERS",
            "8",
        )
    )
    workers = max(
        1,
        min(
            workers,
            image_count,
        ),
    )

    progress_seconds = float(
        os.environ.get(
            "PYSTAMPS_DA_PROGRESS_SECONDS",
            "2",
        )
    )
    progress_seconds = max(
        0.2,
        progress_seconds,
    )

    if config.normalize_per_image:
        print(
            "[D_A] 正在估计逐景幅度尺度...",
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

    block_rows = int(
        config.block_rows
    )

    n_blocks = int(
        np.ceil(
            (
                row_stop
                - row_start
            )
            / block_rows
        )
    )

    total_tasks = (
        n_blocks
        * image_count
    )

    print()
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "并行 D_A 候选点提取",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )
    print(
        f"影像尺寸           : {length} x {width}",
        flush=True,
    )
    print(
        f"MLI景数            : {image_count}",
        flush=True,
    )
    print(
        f"block_rows         : {block_rows}",
        flush=True,
    )
    print(
        f"block数量          : {n_blocks}",
        flush=True,
    )
    print(
        f"并行线程           : {workers}",
        flush=True,
    )
    print(
        f"D_A阈值            : {config.da_threshold}",
        flush=True,
    )
    print(
        "逐景幅度归一化     : "
        f"{config.normalize_per_image}",
        flush=True,
    )
    print(
        f"总读取任务         : {total_tasks}",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )

    def _read_one(
        image_index: int,
        block_start: int,
        block_length: int,
    ) -> np.ndarray:
        mli_block = read_gamma_raster(
            paths[image_index],
            width=width,
            length=length,
            dtype="float",
            y0=block_start,
            ny=block_length,
        )

        amplitude = _to_amplitude(
            mli_block,
            mli_is_power=(
                config.mli_is_power
            ),
        )

        if config.normalize_per_image:
            amplitude = (
                amplitude
                / scales[image_index]
            )

        return amplitude

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

    started = time.monotonic()
    last_progress = 0.0
    completed_tasks = 0
    cumulative_candidates = 0

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
                block_start
                + block_rows,
                row_stop,
            )

            block_length = (
                block_stop
                - block_start
            )

            block_shape = (
                block_length,
                width,
            )

            print()
            print(
                f"[D_A] Block "
                f"{block_index}/{n_blocks} "
                f"rows={block_start}:{block_stop}",
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

            # Read one worker-sized batch at a time.
            # This keeps peak RAM controlled.
            for batch_start in range(
                0,
                image_count,
                workers,
            ):
                batch_stop = min(
                    batch_start
                    + workers,
                    image_count,
                )

                futures = [
                    executor.submit(
                        _read_one,
                        image_index,
                        block_start,
                        block_length,
                    )
                    for image_index
                    in range(
                        batch_start,
                        batch_stop,
                    )
                ]

                # Preserve chronological accumulation order.
                for local_index, future in enumerate(
                    futures
                ):
                    image_index = (
                        batch_start
                        + local_index
                    )

                    amplitude = (
                        future.result()
                    )

                    valid = (
                        np.isfinite(
                            amplitude
                        )
                        & (
                            amplitude
                            > 0
                        )
                    )

                    values = (
                        amplitude[
                            valid
                        ].astype(
                            np.float64,
                            copy=False,
                        )
                    )

                    amplitude_sum[
                        valid
                    ] += values

                    amplitude_square_sum[
                        valid
                    ] += (
                        values
                        * values
                    )

                    valid_count[
                        valid
                    ] += 1

                    completed_tasks += 1

                    now = (
                        time.monotonic()
                    )

                    if (
                        now
                        - last_progress
                        >= progress_seconds
                        or completed_tasks
                        == total_tasks
                        or image_index
                        == image_count - 1
                    ):
                        elapsed = (
                            now
                            - started
                        )

                        rate = (
                            completed_tasks
                            / elapsed
                            if elapsed > 0
                            else 0.0
                        )

                        eta_seconds = (
                            (
                                total_tasks
                                - completed_tasks
                            )
                            / rate
                            if rate > 0
                            else float("nan")
                        )

                        def _fmt_time(
                            seconds: float,
                        ) -> str:
                            if (
                                not np.isfinite(
                                    seconds
                                )
                                or seconds < 0
                            ):
                                return "--:--:--"

                            seconds = int(
                                round(
                                    seconds
                                )
                            )

                            hours, rem = divmod(
                                seconds,
                                3600,
                            )

                            minutes, secs = divmod(
                                rem,
                                60,
                            )

                            return (
                                f"{hours:02d}:"
                                f"{minutes:02d}:"
                                f"{secs:02d}"
                            )

                        print(
                            "[D_A] "
                            f"block="
                            f"{block_index}/"
                            f"{n_blocks} | "
                            f"image="
                            f"{image_index + 1}/"
                            f"{image_count} | "
                            f"overall="
                            f"{completed_tasks}/"
                            f"{total_tasks} "
                            f"("
                            f"{completed_tasks / total_tasks * 100:6.2f}"
                            f"%) | "
                            f"elapsed="
                            f"{_fmt_time(elapsed)} | "
                            f"ETA="
                            f"{_fmt_time(eta_seconds)} | "
                            f"rate="
                            f"{rate:.2f} task/s",
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

            mean_amplitude[
                enough
            ] = (
                amplitude_sum[
                    enough
                ]
                / valid_count[
                    enough
                ]
            )

            variance = np.full(
                block_shape,
                np.nan,
                dtype=np.float64,
            )

            variance[
                enough
            ] = (
                amplitude_square_sum[
                    enough
                ]
                - (
                    amplitude_sum[
                        enough
                    ]
                    ** 2
                    / valid_count[
                        enough
                    ]
                )
            ) / (
                valid_count[
                    enough
                ]
                - 1
            )

            variance[
                enough
            ] = np.maximum(
                variance[
                    enough
                ],
                0.0,
            )

            amplitude_dispersion = np.full(
                block_shape,
                np.nan,
                dtype=np.float64,
            )

            usable = (
                enough
                & np.isfinite(
                    mean_amplitude
                )
                & (
                    mean_amplitude
                    > 0
                )
                & np.isfinite(
                    variance
                )
            )

            amplitude_dispersion[
                usable
            ] = (
                np.sqrt(
                    variance[
                        usable
                    ]
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
                f"Block "
                f"{block_index}/{n_blocks} "
                f"完成 | "
                f"本块候选="
                f"{block_candidate_count:,} | "
                f"累计候选="
                f"{cumulative_candidates:,}",
                flush=True,
            )

            if (
                local_rows.size
                == 0
            ):
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

    total_elapsed = (
        time.monotonic()
        - started
    )

    print()
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "D_A候选提取完成",
        flush=True,
    )
    print(
        f"总耗时             : "
        f"{total_elapsed / 60:.2f} min",
        flush=True,
    )
    print(
        f"最终候选点         : "
        f"{cumulative_candidates:,}",
        flush=True,
    )
    print(
        "=" * 72,
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

def extract_candidates_from_project(
    project: GammaSbasProject,
    *,
    config: CandidateConfig | None = None,
    row_start: int = 0,
    row_stop: int | None = None,
) -> CandidateResult:
    """Extract candidates from all resolved project MLI files."""

    if project.width is None or project.length is None:
        raise GammaInputError(
            "GAMMA工程尚未解析出多视影像行列数"
        )

    missing_dates = [
        acquisition.date
        for acquisition in project.acquisitions
        if acquisition.mli is None
    ]

    if missing_dates:
        raise GammaInputError(
            "以下日期缺少MLI："
            + ", ".join(missing_dates[:30])
        )

    mli_files = [
        acquisition.mli
        for acquisition in project.acquisitions
        if acquisition.mli is not None
    ]

    return extract_amplitude_candidates(
        mli_files,
        width=project.width,
        length=project.length,
        config=config,
        row_start=row_start,
        row_stop=row_stop,
    )


def save_candidate_result(
    result: CandidateResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save candidate arrays and metadata for later Stage-1 assembly."""

    output_dir = Path(
        output_directory,
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    array_file = output_dir / "gamma_candidates.npz"
    metadata_file = (
        output_dir
        / "gamma_candidates_manifest.json"
    )

    np.savez_compressed(
        array_file,
        rows=result.rows,
        cols=result.cols,
        ij=result.to_stamps_ij(),
        amplitude_dispersion=(
            result.amplitude_dispersion
        ),
        mean_amplitude=result.mean_amplitude,
        valid_fraction=result.valid_fraction,
    )

    metadata = {
        "candidate_count": result.count,
        "image_count": result.image_count,
        "coordinate_convention": {
            "rows_cols": "zero-based multilooked pixels",
            "ij": (
                "one-based candidate_id, "
                "azimuth_row, range_column"
            ),
        },
        "config": {
            "da_threshold": (
                result.config.da_threshold
            ),
            "min_valid_fraction": (
                result.config.min_valid_fraction
            ),
            "block_rows": (
                result.config.block_rows
            ),
            "mli_is_power": (
                result.config.mli_is_power
            ),
            "normalize_per_image": (
                result.config.normalize_per_image
            ),
        },
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "arrays": array_file,
        "manifest": metadata_file,
    }
