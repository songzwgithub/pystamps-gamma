from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .gamma_sbas import (
    GammaInputError,
    first_integer_parameter,
    parse_gamma_parameter_file,
)


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class GammaRadarGeometry:
    """Radar geometry needed by StaMPS GAMMA baseline calculations."""

    rslc_width: int
    rslc_length: int
    multilook_width: int
    multilook_length: int

    range_looks: int
    azimuth_looks: int

    range_pixel_spacing: float
    near_range_slc: float
    center_range_slc: float

    sar_to_earth_center: float
    earth_radius_below_sensor: float

    prf: float
    heading: float
    radar_frequency: float

    @property
    def wavelength(self) -> float:
        return SPEED_OF_LIGHT_M_S / self.radar_frequency


@dataclass(frozen=True, slots=True)
class GammaBaselineModel:
    """GAMMA initial TCN baseline and temporal baseline rate."""

    baseline_tcn: np.ndarray
    baseline_rate_tcn: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateRadarGeometry:
    """Geometry calculated at candidate-pixel locations."""

    range_original: np.ndarray
    azimuth_original: np.ndarray
    slant_range: np.ndarray
    look_angle: np.ndarray
    incidence_angle: np.ndarray


def _numeric_tokens(
    values: Iterable[str],
) -> list[float]:
    result: list[float] = []

    for token in values:
        cleaned = token.strip().rstrip(",")

        try:
            result.append(float(cleaned))
        except ValueError:
            continue

    return result


def _required_scalar(
    parameters: dict[str, list[str]],
    keys: tuple[str, ...],
    *,
    source: Path,
) -> float:
    for key in keys:
        values = parameters.get(key)

        if not values:
            continue

        numbers = _numeric_tokens(values)

        if numbers:
            return numbers[0]

    raise GammaInputError(
        f"{source}中缺少参数："
        + " / ".join(keys)
    )


def _optional_integer(
    parameters: dict[str, list[str]],
    keys: tuple[str, ...],
) -> int | None:
    return first_integer_parameter(
        parameters,
        keys,
    )


def _required_vector(
    parameters: dict[str, list[str]],
    keys: tuple[str, ...],
    *,
    count: int,
    source: Path,
) -> np.ndarray:
    for key in keys:
        values = parameters.get(key)

        if not values:
            continue

        numbers = _numeric_tokens(values)

        if len(numbers) >= count:
            return np.asarray(
                numbers[:count],
                dtype=np.float64,
            )

    raise GammaInputError(
        f"{source}中缺少{count}维参数："
        + " / ".join(keys)
    )


def _infer_looks(
    source_size: int,
    multilook_size: int,
    *,
    axis_name: str,
) -> int:
    if source_size <= 0 or multilook_size <= 0:
        raise GammaInputError(
            f"{axis_name}尺寸必须为正数"
        )

    approximate = source_size / multilook_size
    candidate = max(1, int(round(approximate)))

    floor_size = source_size // candidate
    ceil_size = int(np.ceil(source_size / candidate))

    if multilook_size not in {
        floor_size,
        ceil_size,
    }:
        raise GammaInputError(
            f"无法由原始{axis_name}尺寸{source_size}和"
            f"多视尺寸{multilook_size}可靠推导多视比；"
            f"近似比值为{approximate:.6f}"
        )

    return candidate


def _validate_multilook_size(
    source_size: int,
    multilook_size: int,
    looks: int,
    *,
    axis_name: str,
) -> None:
    if looks <= 0:
        raise GammaInputError(
            f"{axis_name}多视数必须大于0"
        )

    floor_size = source_size // looks
    ceil_size = int(np.ceil(source_size / looks))

    if multilook_size not in {
        floor_size,
        ceil_size,
    }:
        raise GammaInputError(
            f"{axis_name}多视尺寸不一致："
            f"原始尺寸={source_size}, "
            f"looks={looks}, "
            f"实际多视尺寸={multilook_size}, "
            f"理论尺寸约为{floor_size}或{ceil_size}"
        )


def build_radar_geometry(
    rslc_parameter_file: str | Path,
    *,
    multilook_width: int,
    multilook_length: int,
    mli_parameter_file: str | Path | None = None,
    range_looks: int | None = None,
    azimuth_looks: int | None = None,
) -> GammaRadarGeometry:
    """Build geometry from a GAMMA RSLC parameter file."""

    rslc_par = Path(
        rslc_parameter_file,
    ).expanduser().resolve()

    if not rslc_par.is_file():
        raise GammaInputError(
            f"RSLC参数文件不存在：{rslc_par}"
        )

    parameters = parse_gamma_parameter_file(
        rslc_par,
    )

    rslc_width = first_integer_parameter(
        parameters,
        ("range_samples", "width"),
    )
    rslc_length = first_integer_parameter(
        parameters,
        ("azimuth_lines", "nlines"),
    )

    if rslc_width is None or rslc_length is None:
        raise GammaInputError(
            f"无法从{rslc_par}读取RSLC行列数"
        )

    mli_parameters: dict[str, list[str]] = {}

    if mli_parameter_file is not None:
        mli_par = Path(
            mli_parameter_file,
        ).expanduser().resolve()

        if not mli_par.is_file():
            raise GammaInputError(
                f"MLI参数文件不存在：{mli_par}"
            )

        mli_parameters = parse_gamma_parameter_file(
            mli_par,
        )

    if range_looks is None:
        range_looks = _optional_integer(
            mli_parameters,
            (
                "range_looks",
                "range_looks_1",
                "range_look_factor",
            ),
        )

    if azimuth_looks is None:
        azimuth_looks = _optional_integer(
            mli_parameters,
            (
                "azimuth_looks",
                "azimuth_looks_1",
                "azimuth_look_factor",
            ),
        )

    if range_looks is None:
        range_looks = _infer_looks(
            rslc_width,
            multilook_width,
            axis_name="距离向",
        )

    if azimuth_looks is None:
        azimuth_looks = _infer_looks(
            rslc_length,
            multilook_length,
            axis_name="方位向",
        )

    _validate_multilook_size(
        rslc_width,
        multilook_width,
        range_looks,
        axis_name="距离向",
    )

    _validate_multilook_size(
        rslc_length,
        multilook_length,
        azimuth_looks,
        axis_name="方位向",
    )

    range_pixel_spacing = _required_scalar(
        parameters,
        ("range_pixel_spacing",),
        source=rslc_par,
    )

    near_range_slc = _required_scalar(
        parameters,
        (
            "near_range_slc",
            "near_range",
        ),
        source=rslc_par,
    )

    center_range_slc = _required_scalar(
        parameters,
        (
            "center_range_slc",
            "center_range",
        ),
        source=rslc_par,
    )

    sar_to_earth_center = _required_scalar(
        parameters,
        ("sar_to_earth_center",),
        source=rslc_par,
    )

    earth_radius_below_sensor = _required_scalar(
        parameters,
        ("earth_radius_below_sensor",),
        source=rslc_par,
    )

    prf = _required_scalar(
        parameters,
        ("prf",),
        source=rslc_par,
    )

    heading = _required_scalar(
        parameters,
        ("heading",),
        source=rslc_par,
    )

    radar_frequency = _required_scalar(
        parameters,
        (
            "radar_frequency",
            "center_frequency",
        ),
        source=rslc_par,
    )

    if radar_frequency <= 0:
        raise GammaInputError(
            f"{rslc_par}中的雷达频率无效："
            f"{radar_frequency}"
        )

    return GammaRadarGeometry(
        rslc_width=rslc_width,
        rslc_length=rslc_length,
        multilook_width=multilook_width,
        multilook_length=multilook_length,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        range_pixel_spacing=range_pixel_spacing,
        near_range_slc=near_range_slc,
        center_range_slc=center_range_slc,
        sar_to_earth_center=sar_to_earth_center,
        earth_radius_below_sensor=earth_radius_below_sensor,
        prf=prf,
        heading=heading,
        radar_frequency=radar_frequency,
    )


def calculate_candidate_geometry(
    rows: np.ndarray,
    cols: np.ndarray,
    geometry: GammaRadarGeometry,
) -> CandidateRadarGeometry:
    """
    Calculate geometry for zero-based multilooked pixel coordinates.

    The center of each multilook cell is mapped back to the
    corresponding original RSLC coordinate.
    """

    row_array = np.asarray(
        rows,
        dtype=np.float64,
    ).reshape(-1)

    col_array = np.asarray(
        cols,
        dtype=np.float64,
    ).reshape(-1)

    if row_array.shape != col_array.shape:
        raise GammaInputError(
            "rows和cols长度不一致"
        )

    if np.any(row_array < 0) or np.any(
        row_array >= geometry.multilook_length
    ):
        raise GammaInputError(
            "候选点行坐标超出多视影像范围"
        )

    if np.any(col_array < 0) or np.any(
        col_array >= geometry.multilook_width
    ):
        raise GammaInputError(
            "候选点列坐标超出多视影像范围"
        )

    range_original = (
        col_array * geometry.range_looks
        + (geometry.range_looks - 1) / 2.0
    )

    azimuth_original = (
        row_array * geometry.azimuth_looks
        + (geometry.azimuth_looks - 1) / 2.0
    )

    slant_range = (
        geometry.near_range_slc
        + range_original
        * geometry.range_pixel_spacing
    )

    look_argument = (
        geometry.sar_to_earth_center**2
        + slant_range**2
        - geometry.earth_radius_below_sensor**2
    ) / (
        2.0
        * geometry.sar_to_earth_center
        * slant_range
    )

    look_angle = np.arccos(
        np.clip(
            look_argument,
            -1.0,
            1.0,
        )
    )

    incidence_argument = (
        geometry.sar_to_earth_center**2
        - geometry.earth_radius_below_sensor**2
        - slant_range**2
    ) / (
        2.0
        * geometry.earth_radius_below_sensor
        * slant_range
    )

    incidence_angle = np.arccos(
        np.clip(
            incidence_argument,
            -1.0,
            1.0,
        )
    )

    return CandidateRadarGeometry(
        range_original=range_original,
        azimuth_original=azimuth_original,
        slant_range=slant_range,
        look_angle=look_angle,
        incidence_angle=incidence_angle,
    )


def read_baseline_model(
    baseline_file: str | Path,
) -> GammaBaselineModel:
    """Read initial_baseline(TCN) and initial_baseline_rate."""

    base_path = Path(
        baseline_file,
    ).expanduser().resolve()

    if not base_path.is_file():
        raise GammaInputError(
            f"基线文件不存在：{base_path}"
        )

    parameters = parse_gamma_parameter_file(
        base_path,
    )

    baseline_tcn = _required_vector(
        parameters,
        (
            "initial_baseline(TCN)",
            "initial_baseline",
        ),
        count=3,
        source=base_path,
    )

    baseline_rate_tcn = _required_vector(
        parameters,
        (
            "initial_baseline_rate",
            "baseline_rate(TCN)",
        ),
        count=3,
        source=base_path,
    )

    return GammaBaselineModel(
        baseline_tcn=baseline_tcn,
        baseline_rate_tcn=baseline_rate_tcn,
    )


def calculate_bperp_column(
    candidate_geometry: CandidateRadarGeometry,
    radar_geometry: GammaRadarGeometry,
    baseline_model: GammaBaselineModel,
) -> np.ndarray:
    """Calculate one interferogram's Bperp at all candidate pixels."""

    mean_azimuth = (
        radar_geometry.rslc_length / 2.0
        - 0.5
    )

    azimuth_time_offset = (
        candidate_geometry.azimuth_original
        - mean_azimuth
    ) / radar_geometry.prf

    # TCN顺序为T、C、N。
    bc = (
        baseline_model.baseline_tcn[1]
        + baseline_model.baseline_rate_tcn[1]
        * azimuth_time_offset
    )

    bn = (
        baseline_model.baseline_tcn[2]
        + baseline_model.baseline_rate_tcn[2]
        * azimuth_time_offset
    )

    bperp = (
        bc * np.cos(candidate_geometry.look_angle)
        - bn * np.sin(candidate_geometry.look_angle)
    )

    return np.asarray(
        bperp,
        dtype=np.float32,
    )


def calculate_bperp_matrix(
    baseline_files: Iterable[str | Path],
    rows: np.ndarray,
    cols: np.ndarray,
    radar_geometry: GammaRadarGeometry,
    *,
    output_file: str | Path | None = None,
) -> np.ndarray:
    """
    Calculate candidate-by-interferogram Bperp matrix.

    When output_file is provided, a raw float32 memmap is used to
    avoid holding the entire matrix in RAM.
    """

    base_paths = [
        Path(path).expanduser().resolve()
        for path in baseline_files
    ]

    if not base_paths:
        raise GammaInputError(
            "没有输入基线文件"
        )

    candidate_geometry = calculate_candidate_geometry(
        rows,
        cols,
        radar_geometry,
    )

    n_candidates = candidate_geometry.look_angle.size
    n_interferograms = len(base_paths)

    if output_file is None:
        matrix: np.ndarray = np.empty(
            (
                n_candidates,
                n_interferograms,
            ),
            dtype=np.float32,
        )
    else:
        output_path = Path(
            output_file,
        ).expanduser().resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        matrix = np.memmap(
            output_path,
            dtype=np.float32,
            mode="w+",
            shape=(
                n_candidates,
                n_interferograms,
            ),
            order="C",
        )

    for ifg_index, base_path in enumerate(
        base_paths,
    ):
        baseline_model = read_baseline_model(
            base_path,
        )

        matrix[:, ifg_index] = calculate_bperp_column(
            candidate_geometry,
            radar_geometry,
            baseline_model,
        )

    if isinstance(matrix, np.memmap):
        matrix.flush()

    return matrix
