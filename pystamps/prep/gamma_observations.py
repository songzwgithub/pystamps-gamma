from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .gamma_binary import (
    normalize_complex_unit,
    sample_gamma_raster,
)
from .gamma_sbas import (
    GammaInputError,
    GammaSbasProject,
)


@dataclass(frozen=True, slots=True)
class RadarGeometryFiles:
    """Radar-coordinate longitude, latitude and height rasters."""

    longitude: Path
    latitude: Path
    height: Path


@dataclass(frozen=True, slots=True)
class RadarGeometrySamples:
    """Geographic coordinates sampled at candidate pixels."""

    longitude: np.ndarray
    latitude: np.ndarray
    height: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True, slots=True)
class GammaPhaseStack:
    """Candidate phase values extracted from all interferograms."""

    phase: np.ndarray
    keep_mask: np.ndarray
    valid_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class SbasTimeAxis:
    """StaMPS small-baseline temporal-network metadata."""

    dates: tuple[str, ...]
    day: np.ndarray
    master_date: str
    master_day: float
    master_ix: int
    ifgday: np.ndarray
    ifgday_ix: np.ndarray
    n_image: int
    n_ifg: int


def matlab_datenum(date_text: str) -> float:
    """Convert YYYYMMDD to MATLAB serial date number."""

    value = datetime.strptime(
        date_text,
        "%Y%m%d",
    )

    return float(value.toordinal() + 366)


def build_sbas_time_axis(
    project: GammaSbasProject,
    *,
    reference_date: str | None = None,
) -> SbasTimeAxis:
    """Build day, ifgday and one-based ifgday_ix arrays."""

    dates = tuple(
        sorted(
            {
                acquisition.date
                for acquisition in project.acquisitions
            }
        )
    )

    if not dates:
        raise GammaInputError(
            "GAMMA工程中没有有效影像日期"
        )

    if reference_date is None:
        reference_date = dates[len(dates) // 2]

    if reference_date not in dates:
        raise GammaInputError(
            f"参考日期{reference_date}不在RSLC_tab中"
        )

    date_to_index = {
        date_text: index + 1
        for index, date_text in enumerate(dates)
    }

    day = np.asarray(
        [
            matlab_datenum(date_text)
            for date_text in dates
        ],
        dtype=np.float64,
    )

    ifgday = np.asarray(
        [
            [
                matlab_datenum(ifg.master_date),
                matlab_datenum(ifg.slave_date),
            ]
            for ifg in project.interferograms
        ],
        dtype=np.float64,
    )

    ifgday_ix = np.asarray(
        [
            [
                date_to_index[ifg.master_date],
                date_to_index[ifg.slave_date],
            ]
            for ifg in project.interferograms
        ],
        dtype=np.int32,
    )

    return SbasTimeAxis(
        dates=dates,
        day=day,
        master_date=reference_date,
        master_day=matlab_datenum(reference_date),
        master_ix=date_to_index[reference_date],
        ifgday=ifgday,
        ifgday_ix=ifgday_ix,
        n_image=len(dates),
        n_ifg=len(project.interferograms),
    )


def _resolve_explicit_geometry_file(
    value: str | Path | None,
    *,
    role: str,
) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser().resolve()

    if not path.is_file():
        raise GammaInputError(
            f"{role}栅格不存在：{path}"
        )

    return path


def _auto_find_geometry_file(
    dem_directory: Path,
    *,
    role: str,
    name_tokens: Iterable[str],
    expected_bytes: int,
) -> Path:
    candidates: list[Path] = []

    ignored_suffixes = {
        ".par",
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".json",
        ".txt",
        ".log",
    }

    for path in dem_directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in ignored_suffixes:
            continue

        name = path.name.lower()

        if not any(
            token.lower() in name
            for token in name_tokens
        ):
            continue

        if path.stat().st_size != expected_bytes:
            continue

        candidates.append(path.resolve())

    candidates = sorted(set(candidates))

    if not candidates:
        raise GammaInputError(
            f"在{dem_directory}中没有找到{role}栅格。"
            f"要求文件大小为{expected_bytes}字节，"
            f"名称包含：{', '.join(name_tokens)}"
        )

    if len(candidates) > 1:
        raise GammaInputError(
            f"在{dem_directory}中匹配到多个{role}栅格：\n"
            + "\n".join(str(path) for path in candidates)
            + f"\n请显式指定{role}文件。"
        )

    return candidates[0]


def resolve_radar_geometry_files(
    project: GammaSbasProject,
    *,
    longitude_file: str | Path | None = None,
    latitude_file: str | Path | None = None,
    height_file: str | Path | None = None,
) -> RadarGeometryFiles:
    """Resolve radar-coordinate lon/lat/height FLOAT rasters."""

    if project.width is None or project.length is None:
        raise GammaInputError(
            "工程尚未解析出雷达坐标影像行列数"
        )

    expected_bytes = (
        project.width
        * project.length
        * 4
    )

    longitude = _resolve_explicit_geometry_file(
        longitude_file,
        role="经度",
    )

    latitude = _resolve_explicit_geometry_file(
        latitude_file,
        role="纬度",
    )

    height = _resolve_explicit_geometry_file(
        height_file,
        role="高程",
    )

    if longitude is None:
        longitude = _auto_find_geometry_file(
            project.dem_dir,
            role="经度",
            name_tokens=(
                ".lon",
                "_lon",
                "longitude",
            ),
            expected_bytes=expected_bytes,
        )

    if latitude is None:
        latitude = _auto_find_geometry_file(
            project.dem_dir,
            role="纬度",
            name_tokens=(
                ".lat",
                "_lat",
                "latitude",
            ),
            expected_bytes=expected_bytes,
        )

    if height is None:
        height = _auto_find_geometry_file(
            project.dem_dir,
            role="高程",
            name_tokens=(
                ".hgt",
                "_hgt",
                "height",
                "dem.rdc",
                "dem_rdc",
            ),
            expected_bytes=expected_bytes,
        )

    return RadarGeometryFiles(
        longitude=longitude,
        latitude=latitude,
        height=height,
    )


def sample_radar_geometry(
    geometry_files: RadarGeometryFiles,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    width: int,
    length: int,
) -> RadarGeometrySamples:
    """Sample radar-coordinate lon/lat/height rasters."""

    longitude = sample_gamma_raster(
        geometry_files.longitude,
        rows,
        cols,
        width=width,
        length=length,
        dtype="float",
    ).astype(
        np.float64,
        copy=False,
    )

    latitude = sample_gamma_raster(
        geometry_files.latitude,
        rows,
        cols,
        width=width,
        length=length,
        dtype="float",
    ).astype(
        np.float64,
        copy=False,
    )

    height = sample_gamma_raster(
        geometry_files.height,
        rows,
        cols,
        width=width,
        length=length,
        dtype="float",
    ).astype(
        np.float32,
        copy=False,
    )

    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(height)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)

        # GAMMA geocode无覆盖区通常写为lon=0、lat=0。
        # 只排除成对的(0, 0)，避免对全球通用代码错误
        # 排除赤道或本初子午线上的正常点。
        & ~(
            (np.abs(longitude) <= 1.0e-6)
            & (np.abs(latitude) <= 1.0e-6)
        )
    )

    return RadarGeometrySamples(
        longitude=longitude,
        latitude=latitude,
        height=height,
        valid=valid,
    )


def extract_phase_stack(
    project: GammaSbasProject,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    max_invalid_interferograms: int = 1,
) -> GammaPhaseStack:
    """
    Extract and unit-normalize complex phase for all interferograms.

    Candidates with more than max_invalid_interferograms zero or invalid
    phase values are rejected.
    """

    if project.width is None or project.length is None:
        raise GammaInputError(
            "工程尚未解析出干涉图行列数"
        )

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

    if max_invalid_interferograms < 0:
        raise GammaInputError(
            "max_invalid_interferograms不能小于0"
        )

    n_candidates = row_array.size
    n_ifg = len(project.interferograms)

    phase = np.zeros(
        (
            n_candidates,
            n_ifg,
        ),
        dtype=np.complex64,
    )

    valid_count = np.zeros(
        n_candidates,
        dtype=np.int32,
    )

    for ifg_index, interferogram in enumerate(
        project.interferograms
    ):
        values = sample_gamma_raster(
            interferogram.diff,
            row_array,
            col_array,
            width=project.width,
            length=project.length,
            dtype="fcomplex",
        )

        unit_phase = normalize_complex_unit(
            values
        )

        phase[:, ifg_index] = unit_phase

        valid_count += (
            np.abs(unit_phase) > 0
        ).astype(np.int32)

    invalid_count = n_ifg - valid_count

    keep_mask = (
        invalid_count
        <= max_invalid_interferograms
    )

    valid_fraction = (
        valid_count.astype(np.float32)
        / max(1, n_ifg)
    )

    return GammaPhaseStack(
        phase=phase[keep_mask, :],
        keep_mask=keep_mask,
        valid_fraction=valid_fraction[keep_mask],
    )


def lonlat_to_local_xy(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    heading_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat to local metric coordinates.

    A WGS84 local tangent approximation is used. Coordinates are
    optionally rotated in the same general direction as StaMPS so that
    scene axes are approximately aligned with x and y.
    """

    lon = np.asarray(
        longitude,
        dtype=np.float64,
    ).reshape(-1)

    lat = np.asarray(
        latitude,
        dtype=np.float64,
    ).reshape(-1)

    if lon.shape != lat.shape:
        raise GammaInputError(
            "经度与纬度数组长度不一致"
        )

    if lon.size == 0:
        raise GammaInputError(
            "无法对空的经纬度数组建立局部坐标"
        )

    ll0 = np.asarray(
        [
            (
                np.nanmax(lon)
                + np.nanmin(lon)
            )
            / 2.0,
            (
                np.nanmax(lat)
                + np.nanmin(lat)
            )
            / 2.0,
        ],
        dtype=np.float64,
    )

    lon0_rad = np.deg2rad(ll0[0])
    lat0_rad = np.deg2rad(ll0[1])

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    semi_major = 6_378_137.0
    eccentricity_squared = 6.69437999014e-3

    sin_lat0 = np.sin(lat0_rad)

    prime_vertical_radius = (
        semi_major
        / np.sqrt(
            1.0
            - eccentricity_squared
            * sin_lat0**2
        )
    )

    meridian_radius = (
        semi_major
        * (
            1.0
            - eccentricity_squared
        )
        / (
            1.0
            - eccentricity_squared
            * sin_lat0**2
        )
        ** 1.5
    )

    x = (
        lon_rad - lon0_rad
    ) * prime_vertical_radius * np.cos(lat0_rad)

    y = (
        lat_rad - lat0_rad
    ) * meridian_radius

    xy = np.column_stack(
        (
            x,
            y,
        )
    )

    theta = np.deg2rad(
        180.0 - heading_degrees
    )

    rotation = np.asarray(
        [
            [
                np.cos(theta),
                np.sin(theta),
            ],
            [
                -np.sin(theta),
                np.cos(theta),
            ],
        ],
        dtype=np.float64,
    )

    rotated = xy @ rotation.T

    original_extent = np.ptp(
        xy,
        axis=0,
    )

    rotated_extent = np.ptp(
        rotated,
        axis=0,
    )

    if np.all(
        rotated_extent < original_extent
    ):
        xy = rotated

    xy = np.round(
        xy,
        decimals=3,
    )

    return (
        xy.astype(
            np.float32,
            copy=False,
        ),
        ll0,
    )
