from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil

import numpy as np

from pystamps.io.mat import write_mat

from .gamma_candidates import (
    CandidateConfig,
    extract_candidates_from_project,
    save_candidate_result,
)
from .gamma_geometry import (
    build_radar_geometry,
    calculate_bperp_matrix,
    calculate_candidate_geometry,
)
from .gamma_lonlat import (
    ensure_gamma_radar_lonlat,
)
from .gamma_observations import (
    build_sbas_time_axis,
    extract_phase_stack,
    lonlat_to_local_xy,
    resolve_radar_geometry_files,
    sample_radar_geometry,
)
from .gamma_patches import (
    PatchConfig,
    build_patch_definitions,
    write_patch_boundaries,
)
from .gamma_ps_optimization import (
    PSOptimizationConfig,
    choose_automatic_patch_config,
    save_ps_selection,
    select_ps_candidates,
)
from .gamma_sbas import (
    GammaInputError,
    load_gamma_sbas_project,
)


@dataclass(frozen=True, slots=True)
class GammaStage1Config:
    """
    Complete GAMMA SBAS to pySTAMPS Stage-1 configuration.

    The workflow remains pure PS:

    1. Candidate points are initially selected using amplitude dispersion.
    2. Spatial balancing optionally retains the lowest-D_A candidates in
       regular radar-coordinate cells.
    3. No DS, SHP or phase-linking operations are introduced.
    """

    candidate: CandidateConfig = field(
        default_factory=CandidateConfig
    )

    patches: PatchConfig = field(
        default_factory=PatchConfig
    )

    ps_optimization: PSOptimizationConfig = field(
        default_factory=PSOptimizationConfig
    )

    # Automatically choose the range/azimuth patch count according to
    # selected candidate density.
    auto_patch_layout: bool = True

    reference_date: str | None = None

    # Spatial deformation reference used by GACOS, Stage 7 and Stage 8.
    reference_lon: float | None = None
    reference_lat: float | None = None
    reference_radius_m: float | None = None

    # Production SBAS Stage7/8 settings.
    sbas_deramp_mode: str = "robust_huber_balanced"
    sbas_deramp_cell_m: float = 2000.0
    sbas_deramp_anchors_per_cell: int = 8
    sbas_deramp_huber_delta: float = 1.345
    sbas_deramp_huber_iterations: int = 5
    sbas_stage8_use_scla: bool = False
    scn_time_win: float = 365.0
    scn_wavelength: float = 100.0

    # Explicit radar-coordinate geometry files.
    # Longitude and latitude must be supplied together.
    longitude_file: str | Path | None = None
    latitude_file: str | Path | None = None
    height_file: str | Path | None = None

    # Optional explicit inputs for automatic lon/lat generation.
    dem_directory: str | Path | None = None
    dem_parameter_file: str | Path | None = None
    radar_parameter_file: str | Path | None = None
    lookup_table_file: str | Path | None = None

    range_looks: int | None = None
    azimuth_looks: int | None = None

    max_invalid_interferograms: int = 1
    minimum_patch_candidates: int = 20

    candidate_row_start: int = 0
    candidate_row_stop: int | None = None

    # Force regeneration of radar-coordinate longitude/latitude.
    force_lonlat: bool = False

    # Force recreation of a non-empty Stage-1 dataset directory.
    force: bool = False

    def __post_init__(self) -> None:
        if self.max_invalid_interferograms < 0:
            raise ValueError(
                "max_invalid_interferograms不能小于0"
            )

        if self.minimum_patch_candidates <= 0:
            raise ValueError(
                "minimum_patch_candidates必须大于0"
            )

        if self.candidate_row_start < 0:
            raise ValueError(
                "candidate_row_start不能小于0"
            )

        if (
            self.candidate_row_stop is not None
            and self.candidate_row_stop
            <= self.candidate_row_start
        ):
            raise ValueError(
                "candidate_row_stop必须大于"
                "candidate_row_start"
            )

        ref_values = (
            self.reference_lon,
            self.reference_lat,
            self.reference_radius_m,
        )
        if any(v is not None for v in ref_values) and not all(
            v is not None for v in ref_values
        ):
            raise ValueError(
                "reference_lon/reference_lat/reference_radius_m必须同时设置"
            )
        if (
            self.reference_radius_m is not None
            and self.reference_radius_m <= 0
        ):
            raise ValueError("reference_radius_m必须大于0")

        if (
            self.range_looks is not None
            and self.range_looks <= 0
        ):
            raise ValueError(
                "range_looks必须大于0"
            )

        if (
            self.azimuth_looks is not None
            and self.azimuth_looks <= 0
        ):
            raise ValueError(
                "azimuth_looks必须大于0"
            )

        longitude_given = (
            self.longitude_file is not None
        )

        latitude_given = (
            self.latitude_file is not None
        )

        if longitude_given != latitude_given:
            raise ValueError(
                "longitude_file和latitude_file"
                "必须同时指定，不能只指定其中一个"
            )


def _scalar_float(
    value: float,
) -> np.ndarray:
    return np.asarray(
        [[value]],
        dtype=np.float64,
    )


def _scalar_int(
    value: int,
) -> np.ndarray:
    return np.asarray(
        [[value]],
        dtype=np.int32,
    )


def _column(
    values: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    array = (
        np.asarray(
            values,
            dtype=dtype,
        )
        if dtype is not None
        else np.asarray(values)
    )

    return array.reshape(-1, 1)


def _prepare_output_directory(
    output_directory: str | Path,
    *,
    force: bool,
) -> Path:
    output = Path(
        output_directory
    ).expanduser().resolve()

    if output.exists():
        has_content = any(
            output.iterdir()
        )

        if has_content and not force:
            raise GammaInputError(
                f"输出目录非空：{output}\n"
                "确认无误后使用force=True重新生成。"
            )

        if has_content and force:
            shutil.rmtree(
                output
            )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output


def _write_root_metadata(
    output: Path,
    *,
    project,
    radar_geometry,
    time_axis,
) -> None:
    (
        output
        / "processor.txt"
    ).write_text(
        "gamma\n",
        encoding="utf-8",
    )

    (
        output
        / "width.txt"
    ).write_text(
        f"{project.width}\n",
        encoding="utf-8",
    )

    (
        output
        / "len.txt"
    ).write_text(
        f"{project.length}\n",
        encoding="utf-8",
    )

    small_baselines = np.asarray(
        [
            [
                int(
                    interferogram.master_date
                ),
                int(
                    interferogram.slave_date
                ),
            ]
            for interferogram
            in project.interferograms
        ],
        dtype=np.int64,
    )

    np.savetxt(
        output
        / "small_baselines.list",
        small_baselines,
        fmt="%d",
    )

    parameters = {
        "small_baseline_flag": "y",
        "insar_processor": "gamma",
        "lambda": float(
            radar_geometry.wavelength
        ),
        "heading": float(
            radar_geometry.heading
        ),
        "filter_grid_size": 50.0,
        "clap_win": 32.0,
        "clap_low_pass_wavelength": 800.0,
        "clap_alpha": 1.0,
        "clap_beta": 0.3,
        "max_topo_err": 15.0,
        "select_method": "PERCENT",
        "percent_rand": 1.0,
        "density_rand": 1.0,
        "drop_ifg_index": np.empty(
            (0, 1),
            dtype=np.float64,
        ),
        "weed_standard_dev": float(
            np.pi
        ),
        "weed_max_noise": float(
            np.pi
        ),
        "weed_zero_elevation": "n",
        "weed_neighbours": "y",
        "gamma_stdev_reject": 0.0,
        "slc_osf": 1.0,
        "weed_time_win": 360.0,
        "gamma_reference_date": (
            time_axis.master_date
        ),
        "range_looks": float(config.range_looks or 1),
        "azimuth_looks": float(config.azimuth_looks or 1),
    }

    if config.reference_lon is not None:
        parameters["ref_centre_lonlat"] = np.asarray(
            [config.reference_lon, config.reference_lat],
            dtype=np.float64,
        )
        parameters["ref_radius_m"] = float(config.reference_radius_m)

    parameters["sbas_deramp_mode"] = str(config.sbas_deramp_mode)
    parameters["sbas_deramp_cell_m"] = float(config.sbas_deramp_cell_m)
    parameters["sbas_deramp_anchors_per_cell"] = int(config.sbas_deramp_anchors_per_cell)
    parameters["sbas_deramp_huber_delta"] = float(config.sbas_deramp_huber_delta)
    parameters["sbas_deramp_huber_iterations"] = int(config.sbas_deramp_huber_iterations)
    parameters["sbas_stage8_use_scla"] = ("y" if config.sbas_stage8_use_scla else "n")
    parameters["scn_time_win"] = float(config.scn_time_win)
    parameters["scn_wavelength"] = float(config.scn_wavelength)

    write_mat(
        output
        / "parms.mat",
        parameters,
    )


def _write_patch_stage1(
    patch_directory: Path,
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    height: np.ndarray,
    amplitude_dispersion: np.ndarray,
    phase: np.ndarray,
    phase_valid_fraction: np.ndarray,
    bperp_mat: np.ndarray,
    look_angle: np.ndarray,
    incidence_angle: np.ndarray,
    slant_range: np.ndarray,
    heading: float,
    time_axis,
) -> dict[str, object]:
    n_ps = int(
        rows.size
    )

    if n_ps == 0:
        raise GammaInputError(
            f"{patch_directory.name}"
            "没有有效候选点"
        )

    lonlat = np.column_stack(
        (
            longitude,
            latitude,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    local_xy, ll0 = lonlat_to_local_xy(
        longitude,
        latitude,
        heading_degrees=heading,
    )

    # 按局部y坐标、局部x坐标排序。
    sort_order = np.lexsort(
        (
            local_xy[:, 0],
            local_xy[:, 1],
        )
    )

    rows = rows[
        sort_order
    ]

    cols = cols[
        sort_order
    ]

    lonlat = lonlat[
        sort_order,
        :,
    ]

    local_xy = local_xy[
        sort_order,
        :,
    ]

    height = height[
        sort_order
    ]

    amplitude_dispersion = (
        amplitude_dispersion[
            sort_order
        ]
    )

    phase = phase[
        sort_order,
        :,
    ]

    phase_valid_fraction = (
        phase_valid_fraction[
            sort_order
        ]
    )

    bperp_mat = bperp_mat[
        sort_order,
        :,
    ]

    look_angle = look_angle[
        sort_order
    ]

    incidence_angle = incidence_angle[
        sort_order
    ]

    slant_range = slant_range[
        sort_order
    ]

    candidate_ids = np.arange(
        1,
        n_ps + 1,
        dtype=np.int32,
    )

    ij = np.column_stack(
        (
            candidate_ids,
            rows.astype(
                np.int32,
                copy=False,
            ) + 1,
            cols.astype(
                np.int32,
                copy=False,
            ) + 1,
        )
    ).astype(
        np.int32,
        copy=False,
    )

    xy = np.column_stack(
        (
            candidate_ids,
            local_xy,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    # 每列对应一个小基线干涉图。
    bperp = np.mean(
        bperp_mat.astype(
            np.float64,
            copy=False,
        ),
        axis=0,
    )

    ps_payload = {
        "ij": ij,
        "lonlat": lonlat,
        "xy": xy,
        "bperp": _column(
            bperp,
            dtype=np.float64,
        ),
        "bperp_ifg": _column(
            bperp,
            dtype=np.float64,
        ),
        "day": _column(
            time_axis.day,
            dtype=np.float64,
        ),
        "master_day": _scalar_float(
            time_axis.master_day
        ),
        "master_ix": _scalar_int(
            time_axis.master_ix
        ),
        "ifgday": np.asarray(
            time_axis.ifgday,
            dtype=np.float64,
        ),
        "ifgday_ix": np.asarray(
            time_axis.ifgday_ix,
            dtype=np.int32,
        ),
        "n_ifg": _scalar_int(
            time_axis.n_ifg
        ),
        "n_image": _scalar_int(
            time_axis.n_image
        ),
        "n_ps": _scalar_int(
            n_ps
        ),
        "sort_ix": _column(
            sort_order.astype(
                np.int32
            ) + 1,
            dtype=np.int32,
        ),
        "ll0": np.asarray(
            ll0,
            dtype=np.float64,
        ).reshape(
            1,
            2,
        ),
        # 与StaMPS内部习惯一致，使用弧度。
        "mean_incidence": _scalar_float(
            float(
                np.mean(
                    incidence_angle
                )
            )
        ),
        "mean_range": _scalar_float(
            float(
                np.mean(
                    slant_range
                )
            )
        ),
    }

    write_mat(
        patch_directory
        / "ps1.mat",
        ps_payload,
    )

    write_mat(
        patch_directory
        / "ph1.mat",
        {
            "ph": np.asarray(
                phase,
                dtype=np.complex64,
            )
        },
    )

    write_mat(
        patch_directory
        / "bp1.mat",
        {
            "bperp_mat": np.asarray(
                bperp_mat,
                dtype=np.float32,
            )
        },
    )

    write_mat(
        patch_directory
        / "da1.mat",
        {
            "D_A": _column(
                amplitude_dispersion,
                dtype=np.float32,
            )
        },
    )

    write_mat(
        patch_directory
        / "hgt1.mat",
        {
            "hgt": _column(
                height,
                dtype=np.float32,
            )
        },
    )

    write_mat(
        patch_directory
        / "la1.mat",
        {
            "la": _column(
                look_angle,
                dtype=np.float32,
            )
        },
    )

    write_mat(
        patch_directory
        / "psver.mat",
        {
            "psver": _scalar_int(1)
        },
    )

    write_mat(
        patch_directory
        / "gamma_input_quality.mat",
        {
            "phase_valid_fraction": _column(
                phase_valid_fraction,
                dtype=np.float32,
            ),
            "incidence_angle": _column(
                incidence_angle,
                dtype=np.float32,
            ),
        },
    )

    mean_incidence_radians = float(
        np.mean(
            incidence_angle
        )
    )

    return {
        "candidate_count": n_ps,
        "phase_shape": [
            int(
                phase.shape[0]
            ),
            int(
                phase.shape[1]
            ),
        ],
        "bperp_shape": [
            int(
                bperp_mat.shape[0]
            ),
            int(
                bperp_mat.shape[1]
            ),
        ],
        "da_min": float(
            np.min(
                amplitude_dispersion
            )
        ),
        "da_median": float(
            np.median(
                amplitude_dispersion
            )
        ),
        "da_max": float(
            np.max(
                amplitude_dispersion
            )
        ),
        "phase_valid_fraction_min": float(
            np.min(
                phase_valid_fraction
            )
        ),
        "phase_valid_fraction_median": float(
            np.median(
                phase_valid_fraction
            )
        ),
        "mean_incidence_radians": (
            mean_incidence_radians
        ),
        "mean_incidence_degrees": float(
            np.degrees(
                mean_incidence_radians
            )
        ),
        "mean_slant_range_m": float(
            np.mean(
                slant_range
            )
        ),
    }


def _lonlat_manifest(
    lonlat_result,
) -> dict[str, object]:
    return {
        "longitude_file": str(
            lonlat_result.longitude_file
        ),
        "latitude_file": str(
            lonlat_result.latitude_file
        ),
        "map_longitude_file": str(
            lonlat_result.map_longitude_file
        ),
        "map_latitude_file": str(
            lonlat_result.map_latitude_file
        ),
        "dem_parameter_file": str(
            lonlat_result.dem_parameter_file
        ),
        "dem_file": str(
            lonlat_result.dem_file
        ),
        "lookup_table_file": str(
            lonlat_result.lookup_table_file
        ),
        "radar_parameter_file": str(
            lonlat_result.radar_parameter_file
        ),
        "map_width": int(
            lonlat_result.map_width
        ),
        "map_length": int(
            lonlat_result.map_length
        ),
        "radar_width": int(
            lonlat_result.radar_width
        ),
        "radar_length": int(
            lonlat_result.radar_length
        ),
        "generated_map_coordinates": bool(
            lonlat_result
            .generated_map_coordinates
        ),
        "generated_radar_coordinates": bool(
            lonlat_result
            .generated_radar_coordinates
        ),
    }


def _manual_patch_layout_report(
    patch_config: PatchConfig,
    *,
    selected_candidate_count: int,
) -> dict[str, object]:
    patch_count = int(
        patch_config.range_patches
        * patch_config.azimuth_patches
    )

    return {
        "mode": "manual",
        "selected_candidate_count": int(
            selected_candidate_count
        ),
        "range_patches": int(
            patch_config.range_patches
        ),
        "azimuth_patches": int(
            patch_config.azimuth_patches
        ),
        "actual_grid_patch_count": (
            patch_count
        ),
        "estimated_candidates_per_patch": (
            float(
                selected_candidate_count
                / patch_count
            )
            if patch_count > 0
            else None
        ),
        "range_overlap": int(
            patch_config.range_overlap
        ),
        "azimuth_overlap": int(
            patch_config.azimuth_overlap
        ),
    }


def prepare_gamma_sbas_stage1(
    project_directory: str | Path,
    output_directory: str | Path,
    *,
    config: GammaStage1Config | None = None,
) -> dict[str, object]:
    """
    Convert a GAMMA SBAS project into a pySTAMPS Stage-1 dataset.

    Processing order
    ----------------
    1. Load and validate the GAMMA project.
    2. Resolve multilook radar geometry.
    3. Resolve or generate radar-coordinate lon/lat.
    4. Extract amplitude-dispersion PS candidates.
    5. Apply optional pure-PS spatial balancing.
    6. Construct automatic or manual patch definitions.
    7. Extract phase, baseline and radar geometry by patch.
    8. Write pySTAMPS Stage-1 MAT artifacts.
    """

    if config is None:
        config = GammaStage1Config()

    project_root = Path(
        project_directory
    ).expanduser().resolve()

    project = load_gamma_sbas_project(
        project_root
    )

    if (
        project.width is None
        or project.length is None
    ):
        raise GammaInputError(
            "无法确定GAMMA多视影像宽度和行数"
        )

    project_width = int(
        project.width
    )

    project_length = int(
        project.length
    )

    if project_width <= 0:
        raise GammaInputError(
            f"无效GAMMA影像宽度：{project_width}"
        )

    if project_length <= 0:
        raise GammaInputError(
            f"无效GAMMA影像行数：{project_length}"
        )

    if not project.acquisitions:
        raise GammaInputError(
            "GAMMA工程中没有有效获取日期"
        )

    if not project.interferograms:
        raise GammaInputError(
            "GAMMA工程中没有有效干涉图"
        )

    first_acquisition = (
        project.acquisitions[0]
    )

    radar_geometry = build_radar_geometry(
        first_acquisition.par,
        multilook_width=project_width,
        multilook_length=project_length,
        mli_parameter_file=(
            first_acquisition.mli_par
        ),
        range_looks=(
            config.range_looks
        ),
        azimuth_looks=(
            config.azimuth_looks
        ),
    )

    resolved_range_looks = int(
        radar_geometry.range_looks
    )

    resolved_azimuth_looks = int(
        radar_geometry.azimuth_looks
    )

    if resolved_range_looks <= 0:
        raise GammaInputError(
            "无法解析有效range_looks"
        )

    if resolved_azimuth_looks <= 0:
        raise GammaInputError(
            "无法解析有效azimuth_looks"
        )

    dem_directory = (
        Path(
            config.dem_directory
        ).expanduser().resolve()
        if config.dem_directory is not None
        else (
            project_root
            / "DEM_prep"
        )
    )

    lonlat_result = ensure_gamma_radar_lonlat(
        project_root,
        radar_width=project_width,
        radar_length=project_length,
        range_looks=(
            resolved_range_looks
        ),
        azimuth_looks=(
            resolved_azimuth_looks
        ),
        dem_directory=(
            dem_directory
        ),
        longitude_file=(
            config.longitude_file
        ),
        latitude_file=(
            config.latitude_file
        ),
        dem_parameter_file=(
            config.dem_parameter_file
        ),
        radar_parameter_file=(
            config.radar_parameter_file
        ),
        lookup_table_file=(
            config.lookup_table_file
        ),
        force=(
            config.force_lonlat
        ),
    )

    geometry_files = resolve_radar_geometry_files(
        project,
        longitude_file=(
            lonlat_result.longitude_file
        ),
        latitude_file=(
            lonlat_result.latitude_file
        ),
        height_file=(
            config.height_file
        ),
    )

    # 输入和几何检查完成后才创建输出目录。
    output = _prepare_output_directory(
        output_directory,
        force=config.force,
    )

    time_axis = build_sbas_time_axis(
        project,
        reference_date=(
            config.reference_date
        ),
    )

    print()
    print("提取振幅离散度PS候选点...")

    candidates = extract_candidates_from_project(
        project,
        config=config.candidate,
        row_start=(
            config.candidate_row_start
        ),
        row_stop=(
            config.candidate_row_stop
        ),
    )

    if candidates.count == 0:
        raise GammaInputError(
            "振幅离散度筛选后没有候选点"
        )

    print(
        "D_A阈值筛选后候选点："
        f"{candidates.count}"
    )

    # 保存D_A阈值筛选后的完整候选点。
    save_candidate_result(
        candidates,
        output / "input_cache",
    )

    print()
    print("执行纯PS空间均衡优化...")

    ps_selection = select_ps_candidates(
        candidates.rows,
        candidates.cols,
        candidates.amplitude_dispersion,
        width=project_width,
        length=project_length,
        config=(
            config.ps_optimization
        ),
    )

    selected_indices = (
        ps_selection.indices
    )

    candidate_rows = np.asarray(
        candidates.rows[
            selected_indices
        ],
        dtype=np.int32,
    )

    candidate_cols = np.asarray(
        candidates.cols[
            selected_indices
        ],
        dtype=np.int32,
    )

    candidate_da = np.asarray(
        candidates.amplitude_dispersion[
            selected_indices
        ],
        dtype=np.float32,
    )

    if candidate_rows.size == 0:
        raise GammaInputError(
            "PS空间优化后没有保留候选点"
        )

    print(
        "PS空间优化后候选点："
        f"{candidate_rows.size}"
    )

    print(
        "候选点保留比例："
        f"{candidate_rows.size / candidates.count:.4f}"
    )

    save_ps_selection(
        output / "input_cache",
        source_indices=(
            selected_indices
        ),
        rows=candidate_rows,
        cols=candidate_cols,
        amplitude_dispersion=(
            candidate_da
        ),
        report=(
            ps_selection.report
        ),
    )

    if config.auto_patch_layout:
        (
            effective_patch_config,
            patch_layout_report,
        ) = choose_automatic_patch_config(
            candidate_rows,
            candidate_cols,
            base_config=(
                config.patches
            ),
            optimization=(
                config.ps_optimization
            ),
        )

        patch_layout_report = {
            "mode": "automatic",
            **patch_layout_report,
        }

    else:
        effective_patch_config = (
            config.patches
        )

        patch_layout_report = (
            _manual_patch_layout_report(
                effective_patch_config,
                selected_candidate_count=int(
                    candidate_rows.size
                ),
            )
        )

    print()
    print("Patch布局：")
    print(
        "  range patches  : "
        f"{effective_patch_config.range_patches}"
    )
    print(
        "  azimuth patches: "
        f"{effective_patch_config.azimuth_patches}"
    )
    print(
        "  grid patch count: "
        f"{effective_patch_config.range_patches * effective_patch_config.azimuth_patches}"
    )

    patches = build_patch_definitions(
        candidate_rows,
        candidate_cols,
        width=project_width,
        length=project_length,
        config=(
            effective_patch_config
        ),
    )

    if not patches:
        raise GammaInputError(
            "优化候选点存在，但没有生成任何patch定义"
        )

    _write_root_metadata(
        output,
        project=project,
        radar_geometry=(
            radar_geometry
        ),
        time_axis=time_axis,
    )

    patch_names: list[str] = []

    patch_reports: list[
        dict[str, object]
    ] = []

    skipped_patch_reports: list[
        dict[str, object]
    ] = []

    baseline_files = [
        interferogram.base
        for interferogram
        in project.interferograms
    ]

    total_patch_definitions = len(
        patches
    )

    for patch_index, patch in enumerate(
        patches,
        start=1,
    ):
        print()
        print(
            f"[{patch_index}/{total_patch_definitions}] "
            f"处理{patch.name}"
        )

        global_indices = (
            patch.candidate_indices
        )

        rows = candidate_rows[
            global_indices
        ]

        cols = candidate_cols[
            global_indices
        ]

        amplitude_dispersion = (
            candidate_da[
                global_indices
            ]
        )

        initial_patch_count = int(
            rows.size
        )

        if initial_patch_count == 0:
            skipped_patch_reports.append(
                {
                    "patch": patch.name,
                    "reason": (
                        "no_candidate_in_patch"
                    ),
                    "initial_candidate_count": 0,
                }
            )
            continue

        geometry_samples = sample_radar_geometry(
            geometry_files,
            rows,
            cols,
            width=project_width,
            length=project_length,
        )

        geometry_keep = np.asarray(
            geometry_samples.valid,
            dtype=bool,
        )

        rows = rows[
            geometry_keep
        ]

        cols = cols[
            geometry_keep
        ]

        amplitude_dispersion = (
            amplitude_dispersion[
                geometry_keep
            ]
        )

        longitude = np.asarray(
            geometry_samples.longitude[
                geometry_keep
            ],
            dtype=np.float64,
        )

        latitude = np.asarray(
            geometry_samples.latitude[
                geometry_keep
            ],
            dtype=np.float64,
        )

        height = np.asarray(
            geometry_samples.height[
                geometry_keep
            ],
            dtype=np.float32,
        )

        geometry_valid_count = int(
            rows.size
        )

        if rows.size == 0:
            print(
                f"跳过{patch.name}："
                "没有有效经纬度和高程候选点"
            )

            skipped_patch_reports.append(
                {
                    "patch": patch.name,
                    "reason": (
                        "no_valid_geometry"
                    ),
                    "initial_candidate_count": (
                        initial_patch_count
                    ),
                    "geometry_valid_count": 0,
                }
            )

            continue

        phase_stack = extract_phase_stack(
            project,
            rows,
            cols,
            max_invalid_interferograms=(
                config
                .max_invalid_interferograms
            ),
        )

        phase_keep = np.asarray(
            phase_stack.keep_mask,
            dtype=bool,
        )

        if phase_keep.size != rows.size:
            raise GammaInputError(
                f"{patch.name}的phase keep_mask长度"
                f"{phase_keep.size}与输入候选点数"
                f"{rows.size}不一致"
            )

        rows = rows[
            phase_keep
        ]

        cols = cols[
            phase_keep
        ]

        amplitude_dispersion = (
            amplitude_dispersion[
                phase_keep
            ]
        )

        longitude = longitude[
            phase_keep
        ]

        latitude = latitude[
            phase_keep
        ]

        height = height[
            phase_keep
        ]

        phase_valid_count = int(
            rows.size
        )

        if (
            rows.size
            < config.minimum_patch_candidates
        ):
            print(
                f"跳过{patch.name}："
                f"有效候选点数={rows.size}，"
                "小于minimum_patch_candidates="
                f"{config.minimum_patch_candidates}"
            )

            skipped_patch_reports.append(
                {
                    "patch": patch.name,
                    "reason": (
                        "below_minimum_patch_candidates"
                    ),
                    "initial_candidate_count": (
                        initial_patch_count
                    ),
                    "geometry_valid_count": (
                        geometry_valid_count
                    ),
                    "phase_valid_count": (
                        phase_valid_count
                    ),
                    "minimum_patch_candidates": (
                        int(
                            config
                            .minimum_patch_candidates
                        )
                    ),
                }
            )

            continue

        phase = np.asarray(
            phase_stack.phase
        )

        valid_fraction = np.asarray(
            phase_stack.valid_fraction
        ).reshape(-1)

        # 兼容phase_stack返回完整矩阵或已筛选矩阵。
        if phase.shape[0] == phase_keep.size:
            phase = phase[
                phase_keep,
                :,
            ]

        if valid_fraction.size == phase_keep.size:
            valid_fraction = valid_fraction[
                phase_keep
            ]

        if phase.shape[0] != rows.size:
            raise GammaInputError(
                f"{patch.name}相位矩阵行数"
                f"{phase.shape[0]}与候选点数"
                f"{rows.size}不一致"
            )

        if valid_fraction.size != rows.size:
            raise GammaInputError(
                f"{patch.name}有效相位比例长度"
                f"{valid_fraction.size}与候选点数"
                f"{rows.size}不一致"
            )

        if (
            phase.shape[1]
            != len(
                project.interferograms
            )
        ):
            raise GammaInputError(
                f"{patch.name}相位矩阵干涉图数"
                f"{phase.shape[1]}与工程干涉图数"
                f"{len(project.interferograms)}"
                "不一致"
            )

        candidate_geometry = (
            calculate_candidate_geometry(
                rows,
                cols,
                radar_geometry,
            )
        )

        bperp_mat = calculate_bperp_matrix(
            baseline_files,
            rows,
            cols,
            radar_geometry,
        )

        bperp_mat = np.asarray(
            bperp_mat
        )

        if bperp_mat.shape != phase.shape:
            raise GammaInputError(
                f"{patch.name}的bperp_mat形状"
                f"{bperp_mat.shape}与phase形状"
                f"{phase.shape}不一致"
            )

        patch_directory = (
            output
            / patch.name
        )

        patch_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_patch_boundaries(
            patch,
            patch_directory,
        )

        patch_report = _write_patch_stage1(
            patch_directory,
            rows=rows,
            cols=cols,
            longitude=longitude,
            latitude=latitude,
            height=height,
            amplitude_dispersion=(
                amplitude_dispersion
            ),
            phase=phase,
            phase_valid_fraction=(
                valid_fraction
            ),
            bperp_mat=bperp_mat,
            look_angle=(
                candidate_geometry
                .look_angle
            ),
            incidence_angle=(
                candidate_geometry
                .incidence_angle
            ),
            slant_range=(
                candidate_geometry
                .slant_range
            ),
            heading=(
                radar_geometry.heading
            ),
            time_axis=time_axis,
        )

        patch_in_values = (
            patch.patch_in_values()
        )

        patch_report.update(
            {
                "patch": patch.name,
                "initial_candidate_count": (
                    initial_patch_count
                ),
                "geometry_valid_count": (
                    geometry_valid_count
                ),
                "phase_valid_count": (
                    phase_valid_count
                ),
                "range_bounds": [
                    int(
                        patch_in_values[0]
                    ),
                    int(
                        patch_in_values[1]
                    ),
                ],
                "azimuth_bounds": [
                    int(
                        patch_in_values[2]
                    ),
                    int(
                        patch_in_values[3]
                    ),
                ],
            }
        )

        (
            patch_directory
            / "gamma_patch_manifest.json"
        ).write_text(
            json.dumps(
                patch_report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        patch_names.append(
            patch.name
        )

        patch_reports.append(
            patch_report
        )

        print(
            f"{patch.name}完成："
            f"{rows.size}个PS点"
        )

    if not patch_names:
        raise GammaInputError(
            "没有生成任何有效PATCH。"
            "可降低minimum_patch_candidates，"
            "或检查经纬度、高程及干涉相位栅格。"
        )

    (
        output
        / "patch.list"
    ).write_text(
        "\n".join(
            patch_names
        ) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "input_project": str(
            project.root
        ),
        "output_dataset": str(
            output
        ),
        "processor": "gamma",
        "processing_mode": (
            "pure_ps_spatially_balanced"
        ),
        "small_baseline_flag": "y",
        "reference_date": (
            time_axis.master_date
        ),
        "acquisition_count": int(
            len(
                project.acquisitions
            )
        ),
        "interferogram_count": int(
            len(
                project.interferograms
            )
        ),

        # 保留旧字段，以兼容既有检查脚本。
        "candidate_count_before_patch": int(
            candidate_rows.size
        ),

        "candidate_count_after_da_threshold": int(
            candidates.count
        ),
        "candidate_count_after_ps_optimization": int(
            candidate_rows.size
        ),
        "candidate_retained_fraction": float(
            candidate_rows.size
            / candidates.count
        ),
        "candidate_row_start": int(
            config.candidate_row_start
        ),
        "candidate_row_stop": (
            int(
                config.candidate_row_stop
            )
            if config.candidate_row_stop
            is not None
            else None
        ),
        "candidate_configuration": {
            "max_invalid_interferograms": int(
                config
                .max_invalid_interferograms
            ),
            "minimum_patch_candidates": int(
                config
                .minimum_patch_candidates
            ),
        },
        "ps_optimization": (
            ps_selection.report
        ),
        "auto_patch_layout": bool(
            config.auto_patch_layout
        ),
        "patch_layout": (
            patch_layout_report
        ),
        "patch_definition_count": int(
            total_patch_definitions
        ),
        "patch_count": int(
            len(
                patch_names
            )
        ),
        "skipped_patch_count": int(
            len(
                skipped_patch_reports
            )
        ),
        "width": project_width,
        "length": project_length,
        "rslc_width": int(
            radar_geometry.rslc_width
        ),
        "rslc_length": int(
            radar_geometry.rslc_length
        ),
        "range_looks": int(
            radar_geometry.range_looks
        ),
        "azimuth_looks": int(
            radar_geometry.azimuth_looks
        ),
        "wavelength_m": float(
            radar_geometry.wavelength
        ),
        "heading_degrees": float(
            radar_geometry.heading
        ),
        "radar_lonlat": _lonlat_manifest(
            lonlat_result
        ),
        "geometry_files": {
            "longitude": str(
                geometry_files.longitude
            ),
            "latitude": str(
                geometry_files.latitude
            ),
            "height": str(
                geometry_files.height
            ),
        },
        "patches": (
            patch_reports
        ),
        "skipped_patches": (
            skipped_patch_reports
        ),
    }

    (
        output
        / "gamma_sbas_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("======================================================")
    print("GAMMA SBAS Stage-1准备完成")
    print("======================================================")
    print(
        "D_A筛选候选点："
        f"{candidates.count}"
    )
    print(
        "空间优化候选点："
        f"{candidate_rows.size}"
    )
    print(
        "有效patch："
        f"{len(patch_names)}"
    )
    print(
        "跳过patch："
        f"{len(skipped_patch_reports)}"
    )
    print(
        f"输出目录：{output}"
    )

    return manifest
