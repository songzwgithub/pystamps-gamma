from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from .gamma_patches import PatchConfig
from .gamma_sbas import GammaInputError


@dataclass(frozen=True, slots=True)
class PSOptimizationConfig:
    """
    CPU-oriented pure-PS candidate-density and patch-layout configuration.

    This does not introduce DS or phase linking. It only limits the number
    of amplitude-stable PS candidates while retaining the lowest-D_A points
    within regular radar-coordinate cells.
    """

    enabled: bool = True

    # Spatial balancing cell size in multilooked radar pixels.
    cell_rows: int = 32
    cell_cols: int = 32

    # Retain at most this many lowest-D_A PS candidates per cell.
    max_candidates_per_cell: int = 32

    # Optional global safety cap after spatial balancing.
    # The round-robin rank selection preserves spatial coverage.
    global_max_candidates: int | None = 1_000_000

    # Used to determine the automatic patch grid.
    target_candidates_per_patch: int = 25_000

    min_range_patches: int = 1
    min_azimuth_patches: int = 1

    max_range_patches: int = 40
    max_azimuth_patches: int = 40

    def __post_init__(self) -> None:
        if self.cell_rows <= 0:
            raise ValueError(
                "cell_rows必须大于0"
            )

        if self.cell_cols <= 0:
            raise ValueError(
                "cell_cols必须大于0"
            )

        if self.max_candidates_per_cell <= 0:
            raise ValueError(
                "max_candidates_per_cell必须大于0"
            )

        if (
            self.global_max_candidates is not None
            and self.global_max_candidates <= 0
        ):
            raise ValueError(
                "global_max_candidates必须大于0或为None"
            )

        if self.target_candidates_per_patch <= 0:
            raise ValueError(
                "target_candidates_per_patch必须大于0"
            )

        if self.min_range_patches <= 0:
            raise ValueError(
                "min_range_patches必须大于0"
            )

        if self.min_azimuth_patches <= 0:
            raise ValueError(
                "min_azimuth_patches必须大于0"
            )

        if (
            self.max_range_patches
            < self.min_range_patches
        ):
            raise ValueError(
                "max_range_patches不能小于min_range_patches"
            )

        if (
            self.max_azimuth_patches
            < self.min_azimuth_patches
        ):
            raise ValueError(
                "max_azimuth_patches不能小于min_azimuth_patches"
            )


@dataclass(frozen=True, slots=True)
class PSSelectionResult:
    """Indices and diagnostics for optimized pure-PS candidate selection."""

    indices: np.ndarray
    report: dict[str, object]


def _validate_candidate_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude_dispersion: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows_array = np.asarray(
        rows,
        dtype=np.int64,
    ).reshape(-1)

    cols_array = np.asarray(
        cols,
        dtype=np.int64,
    ).reshape(-1)

    da_array = np.asarray(
        amplitude_dispersion,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        rows_array.size
        == cols_array.size
        == da_array.size
    ):
        raise GammaInputError(
            "rows、cols和amplitude_dispersion长度不一致"
        )

    return (
        rows_array,
        cols_array,
        da_array,
    )


def _cell_balanced_indices(
    rows: np.ndarray,
    cols: np.ndarray,
    da: np.ndarray,
    *,
    width: int,
    cell_rows: int,
    cell_cols: int,
    max_candidates_per_cell: int,
) -> np.ndarray:
    """
    Retain the lowest-D_A candidates within each radar-coordinate cell.
    """

    n_cell_cols = int(
        math.ceil(
            width / cell_cols
        )
    )

    cell_row = (
        rows // cell_rows
    )

    cell_col = (
        cols // cell_cols
    )

    cell_key = (
        cell_row * n_cell_cols
        + cell_col
    )

    # Primary key: cell_key
    # Secondary key: D_A
    # Tertiary keys: row and column for deterministic output.
    order = np.lexsort(
        (
            cols,
            rows,
            da,
            cell_key,
        )
    )

    sorted_keys = cell_key[
        order
    ]

    starts = np.concatenate(
        (
            np.asarray(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(
                np.diff(
                    sorted_keys
                )
            ).astype(
                np.int64
            ) + 1,
        )
    )

    ends = np.concatenate(
        (
            starts[1:],
            np.asarray(
                [order.size],
                dtype=np.int64,
            ),
        )
    )

    selected_parts: list[np.ndarray] = []

    for start, end in zip(
        starts,
        ends,
        strict=True,
    ):
        selected_parts.append(
            order[
                start:
                min(
                    end,
                    start
                    + max_candidates_per_cell,
                )
            ]
        )

    if not selected_parts:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.concatenate(
        selected_parts
    ).astype(
        np.int64,
        copy=False,
    )


def _apply_global_spatial_cap(
    selected: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    da: np.ndarray,
    *,
    width: int,
    cell_rows: int,
    cell_cols: int,
    global_max_candidates: int,
) -> np.ndarray:
    """
    Apply a global cap using round-robin within-cell rank.

    All cells contribute their best candidate before second-ranked candidates
    are considered, preserving spatial coverage better than a global D_A sort.
    """

    if selected.size <= global_max_candidates:
        return selected

    n_cell_cols = int(
        math.ceil(
            width / cell_cols
        )
    )

    selected_rows = rows[
        selected
    ]

    selected_cols = cols[
        selected
    ]

    selected_da = da[
        selected
    ]

    selected_key = (
        (
            selected_rows
            // cell_rows
        )
        * n_cell_cols
        + (
            selected_cols
            // cell_cols
        )
    )

    cell_order = np.lexsort(
        (
            selected_cols,
            selected_rows,
            selected_da,
            selected_key,
        )
    )

    cell_sorted_indices = selected[
        cell_order
    ]

    cell_sorted_keys = selected_key[
        cell_order
    ]

    starts = np.concatenate(
        (
            np.asarray(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(
                np.diff(
                    cell_sorted_keys
                )
            ).astype(
                np.int64
            ) + 1,
        )
    )

    ends = np.concatenate(
        (
            starts[1:],
            np.asarray(
                [cell_sorted_indices.size],
                dtype=np.int64,
            ),
        )
    )

    within_cell_rank = np.empty(
        cell_sorted_indices.size,
        dtype=np.int32,
    )

    for start, end in zip(
        starts,
        ends,
        strict=True,
    ):
        within_cell_rank[
            start:end
        ] = np.arange(
            end - start,
            dtype=np.int32,
        )

    # Primary key: within-cell rank.
    # Secondary key: D_A.
    fair_order = np.lexsort(
        (
            da[
                cell_sorted_indices
            ],
            within_cell_rank,
        )
    )

    return cell_sorted_indices[
        fair_order[
            :global_max_candidates
        ]
    ]


def select_ps_candidates(
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude_dispersion: np.ndarray,
    *,
    width: int,
    length: int,
    config: PSOptimizationConfig,
) -> PSSelectionResult:
    """
    Select a deterministic, spatially balanced subset of pure-PS candidates.
    """

    (
        rows_array,
        cols_array,
        da_array,
    ) = _validate_candidate_arrays(
        rows,
        cols,
        amplitude_dispersion,
    )

    original_count = int(
        rows_array.size
    )

    all_indices = np.arange(
        original_count,
        dtype=np.int64,
    )

    valid = (
        np.isfinite(
            da_array
        )
        & (rows_array >= 0)
        & (rows_array < length)
        & (cols_array >= 0)
        & (cols_array < width)
    )

    valid_indices = all_indices[
        valid
    ]

    if valid_indices.size == 0:
        raise GammaInputError(
            "没有有效PS候选点可供空间优化"
        )

    if not config.enabled:
        selected = valid_indices

    else:
        local_selected = _cell_balanced_indices(
            rows_array[
                valid_indices
            ],
            cols_array[
                valid_indices
            ],
            da_array[
                valid_indices
            ],
            width=width,
            cell_rows=config.cell_rows,
            cell_cols=config.cell_cols,
            max_candidates_per_cell=(
                config
                .max_candidates_per_cell
            ),
        )

        selected = valid_indices[
            local_selected
        ]

        if (
            config.global_max_candidates
            is not None
        ):
            selected = _apply_global_spatial_cap(
                selected,
                rows_array,
                cols_array,
                da_array,
                width=width,
                cell_rows=config.cell_rows,
                cell_cols=config.cell_cols,
                global_max_candidates=(
                    config
                    .global_max_candidates
                ),
            )

    spatial_order = np.lexsort(
        (
            cols_array[
                selected
            ],
            rows_array[
                selected
            ],
        )
    )

    selected = selected[
        spatial_order
    ].astype(
        np.int64,
        copy=False,
    )

    selected_da = da_array[
        selected
    ]

    selected_rows = rows_array[
        selected
    ]

    selected_cols = cols_array[
        selected
    ]

    report = {
        "configuration": asdict(
            config
        ),
        "original_count": (
            original_count
        ),
        "valid_count": int(
            valid_indices.size
        ),
        "selected_count": int(
            selected.size
        ),
        "retained_fraction_of_valid": (
            float(
                selected.size
                / valid_indices.size
            )
        ),
        "da_min": float(
            np.min(
                selected_da
            )
        ),
        "da_median": float(
            np.median(
                selected_da
            )
        ),
        "da_max": float(
            np.max(
                selected_da
            )
        ),
        "row_min": int(
            np.min(
                selected_rows
            )
        ),
        "row_max": int(
            np.max(
                selected_rows
            )
        ),
        "col_min": int(
            np.min(
                selected_cols
            )
        ),
        "col_max": int(
            np.max(
                selected_cols
            )
        ),
    }

    return PSSelectionResult(
        indices=selected,
        report=report,
    )


def choose_automatic_patch_config(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    base_config: PatchConfig,
    optimization: PSOptimizationConfig,
) -> tuple[
    PatchConfig,
    dict[str, object],
]:
    """
    Choose a patch grid from selected candidate count and spatial aspect ratio.
    """

    rows_array = np.asarray(
        rows,
        dtype=np.int64,
    ).reshape(-1)

    cols_array = np.asarray(
        cols,
        dtype=np.int64,
    ).reshape(-1)

    if rows_array.size == 0:
        raise GammaInputError(
            "无法为零候选点计算patch布局"
        )

    desired_patch_count = max(
        1,
        int(
            math.ceil(
                rows_array.size
                / optimization
                .target_candidates_per_patch
            )
        ),
    )

    row_span = max(
        1,
        int(
            np.max(
                rows_array
            )
            - np.min(
                rows_array
            )
            + 1
        ),
    )

    col_span = max(
        1,
        int(
            np.max(
                cols_array
            )
            - np.min(
                cols_array
            )
            + 1
        ),
    )

    spatial_aspect = (
        col_span
        / row_span
    )

    range_patches = int(
        round(
            math.sqrt(
                desired_patch_count
                * spatial_aspect
            )
        )
    )

    range_patches = max(
        optimization.min_range_patches,
        min(
            optimization.max_range_patches,
            max(
                1,
                range_patches,
            ),
        ),
    )

    azimuth_patches = int(
        math.ceil(
            desired_patch_count
            / range_patches
        )
    )

    azimuth_patches = max(
        optimization.min_azimuth_patches,
        min(
            optimization.max_azimuth_patches,
            max(
                1,
                azimuth_patches,
            ),
        ),
    )

    while (
        range_patches
        * azimuth_patches
        < desired_patch_count
    ):
        can_grow_range = (
            range_patches
            < optimization.max_range_patches
        )

        can_grow_azimuth = (
            azimuth_patches
            < optimization.max_azimuth_patches
        )

        if not (
            can_grow_range
            or can_grow_azimuth
        ):
            break

        current_grid_aspect = (
            range_patches
            / azimuth_patches
        )

        if (
            can_grow_range
            and (
                not can_grow_azimuth
                or current_grid_aspect
                < spatial_aspect
            )
        ):
            range_patches += 1

        else:
            azimuth_patches += 1

    patch_config = PatchConfig(
        range_patches=(
            range_patches
        ),
        azimuth_patches=(
            azimuth_patches
        ),
        range_overlap=(
            base_config.range_overlap
        ),
        azimuth_overlap=(
            base_config.azimuth_overlap
        ),
    )

    report = {
        "selected_candidate_count": int(
            rows_array.size
        ),
        "target_candidates_per_patch": int(
            optimization
            .target_candidates_per_patch
        ),
        "desired_patch_count": int(
            desired_patch_count
        ),
        "range_patches": int(
            range_patches
        ),
        "azimuth_patches": int(
            azimuth_patches
        ),
        "actual_grid_patch_count": int(
            range_patches
            * azimuth_patches
        ),
        "estimated_candidates_per_patch": (
            float(
                rows_array.size
                / (
                    range_patches
                    * azimuth_patches
                )
            )
        ),
        "range_overlap": int(
            base_config.range_overlap
        ),
        "azimuth_overlap": int(
            base_config.azimuth_overlap
        ),
        "candidate_row_span": int(
            row_span
        ),
        "candidate_col_span": int(
            col_span
        ),
        "candidate_spatial_aspect": float(
            spatial_aspect
        ),
    }

    return (
        patch_config,
        report,
    )


def save_ps_selection(
    directory: str | Path,
    *,
    source_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude_dispersion: np.ndarray,
    report: dict[str, object],
) -> None:
    """Persist optimized candidate indices and diagnostics."""

    output = Path(
        directory
    ).expanduser().resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output
        / "gamma_ps_candidates_selected.npz",
        source_candidate_index=np.asarray(
            source_indices,
            dtype=np.int64,
        ),
        rows=np.asarray(
            rows,
            dtype=np.int32,
        ),
        cols=np.asarray(
            cols,
            dtype=np.int32,
        ),
        amplitude_dispersion=np.asarray(
            amplitude_dispersion,
            dtype=np.float32,
        ),
    )

    import json

    (
        output
        / "gamma_ps_optimization.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
