from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gamma_sbas import GammaInputError


@dataclass(frozen=True, slots=True)
class PatchConfig:
    """Spatial patch layout."""

    range_patches: int = 4
    azimuth_patches: int = 4
    range_overlap: int = 50
    azimuth_overlap: int = 50

    def __post_init__(self) -> None:
        if self.range_patches <= 0:
            raise ValueError(
                "range_patches必须大于0"
            )

        if self.azimuth_patches <= 0:
            raise ValueError(
                "azimuth_patches必须大于0"
            )

        if self.range_overlap < 0:
            raise ValueError(
                "range_overlap不能小于0"
            )

        if self.azimuth_overlap < 0:
            raise ValueError(
                "azimuth_overlap不能小于0"
            )


@dataclass(frozen=True, slots=True)
class PatchDefinition:
    """One pySTAMPS patch and its candidate membership."""

    number: int
    name: str

    range_start: int
    range_stop: int
    azimuth_start: int
    azimuth_stop: int

    core_range_start: int
    core_range_stop: int
    core_azimuth_start: int
    core_azimuth_stop: int

    candidate_indices: np.ndarray

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_indices.size)

    def patch_in_values(self) -> tuple[int, int, int, int]:
        """
        Return one-based inclusive:
            range_start, range_end, azimuth_start, azimuth_end
        """

        return (
            self.range_start + 1,
            self.range_stop,
            self.azimuth_start + 1,
            self.azimuth_stop,
        )

    def patch_noover_values(
        self,
    ) -> tuple[int, int, int, int]:
        """Return one-based inclusive core bounds."""

        return (
            self.core_range_start + 1,
            self.core_range_stop,
            self.core_azimuth_start + 1,
            self.core_azimuth_stop,
        )


def _integer_boundaries(
    size: int,
    count: int,
    *,
    axis_name: str,
) -> np.ndarray:
    if count > size:
        raise GammaInputError(
            f"{axis_name}patch数量{count}超过像元数{size}"
        )

    boundaries = np.linspace(
        0,
        size,
        count + 1,
        dtype=np.int64,
    )

    if np.any(
        np.diff(boundaries) <= 0
    ):
        raise GammaInputError(
            f"{axis_name}patch边界无效：{boundaries}"
        )

    return boundaries


def build_patch_definitions(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    width: int,
    length: int,
    config: PatchConfig | None = None,
    include_empty: bool = False,
) -> tuple[PatchDefinition, ...]:
    """Split zero-based candidate coordinates into overlapping patches."""

    if config is None:
        config = PatchConfig()

    row_array = np.asarray(
        rows,
        dtype=np.int64,
    ).reshape(-1)

    col_array = np.asarray(
        cols,
        dtype=np.int64,
    ).reshape(-1)

    if row_array.shape != col_array.shape:
        raise GammaInputError(
            "rows与cols长度不一致"
        )

    if np.any(
        (row_array < 0)
        | (row_array >= length)
    ):
        raise GammaInputError(
            "候选点行坐标超出影像范围"
        )

    if np.any(
        (col_array < 0)
        | (col_array >= width)
    ):
        raise GammaInputError(
            "候选点列坐标超出影像范围"
        )

    range_boundaries = _integer_boundaries(
        width,
        config.range_patches,
        axis_name="距离向",
    )

    azimuth_boundaries = _integer_boundaries(
        length,
        config.azimuth_patches,
        axis_name="方位向",
    )

    patches: list[PatchDefinition] = []
    patch_number = 0

    # 保持StaMPS mt_prep_gamma顺序：
    # 距离向外循环，方位向内循环。
    for range_index in range(
        config.range_patches
    ):
        core_range_start = int(
            range_boundaries[range_index]
        )
        core_range_stop = int(
            range_boundaries[range_index + 1]
        )

        range_start = max(
            0,
            core_range_start
            - config.range_overlap,
        )

        range_stop = min(
            width,
            core_range_stop
            + config.range_overlap,
        )

        for azimuth_index in range(
            config.azimuth_patches
        ):
            patch_number += 1

            core_azimuth_start = int(
                azimuth_boundaries[
                    azimuth_index
                ]
            )
            core_azimuth_stop = int(
                azimuth_boundaries[
                    azimuth_index + 1
                ]
            )

            azimuth_start = max(
                0,
                core_azimuth_start
                - config.azimuth_overlap,
            )

            azimuth_stop = min(
                length,
                core_azimuth_stop
                + config.azimuth_overlap,
            )

            mask = (
                (col_array >= range_start)
                & (col_array < range_stop)
                & (row_array >= azimuth_start)
                & (row_array < azimuth_stop)
            )

            candidate_indices = np.flatnonzero(
                mask
            ).astype(np.int64)

            if (
                candidate_indices.size == 0
                and not include_empty
            ):
                continue

            patches.append(
                PatchDefinition(
                    number=patch_number,
                    name=f"PATCH_{patch_number}",
                    range_start=range_start,
                    range_stop=range_stop,
                    azimuth_start=azimuth_start,
                    azimuth_stop=azimuth_stop,
                    core_range_start=(
                        core_range_start
                    ),
                    core_range_stop=(
                        core_range_stop
                    ),
                    core_azimuth_start=(
                        core_azimuth_start
                    ),
                    core_azimuth_stop=(
                        core_azimuth_stop
                    ),
                    candidate_indices=(
                        candidate_indices
                    ),
                )
            )

    if not patches:
        raise GammaInputError(
            "patch划分后没有任何包含候选点的patch"
        )

    return tuple(patches)


def write_patch_boundaries(
    patch: PatchDefinition,
    patch_directory: str | Path,
) -> None:
    """Write patch.in and patch_noover.in."""

    directory = Path(
        patch_directory,
    ).expanduser().resolve()

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_in = "\n".join(
        str(value)
        for value in patch.patch_in_values()
    ) + "\n"

    patch_noover = "\n".join(
        str(value)
        for value in patch.patch_noover_values()
    ) + "\n"

    (
        directory
        / "patch.in"
    ).write_text(
        patch_in,
        encoding="utf-8",
    )

    (
        directory
        / "patch_noover.in"
    ).write_text(
        patch_noover,
        encoding="utf-8",
    )
