from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


class GammaLonLatError(RuntimeError):
    """Raised when GAMMA radar-coordinate lon/lat cannot be prepared."""


@dataclass(frozen=True, slots=True)
class GammaLonLatResult:
    """Resolved or generated radar-coordinate longitude/latitude files."""

    longitude_file: Path
    latitude_file: Path

    map_longitude_file: Path
    map_latitude_file: Path

    dem_parameter_file: Path
    dem_file: Path
    lookup_table_file: Path
    radar_parameter_file: Path

    radar_width: int
    radar_length: int
    map_width: int
    map_length: int

    generated_map_coordinates: bool
    generated_radar_coordinates: bool


def _read_gamma_parameter(
    parameter_file: Path,
    key: str,
) -> str:
    """Read the first token after ``key:`` in a GAMMA parameter file."""

    with parameter_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:
        for raw_line in handle:
            left, separator, right = raw_line.partition(":")

            if not separator:
                continue

            if left.strip() != key:
                continue

            tokens = right.strip().split()

            if not tokens:
                break

            return tokens[0]

    raise GammaLonLatError(
        f"无法从参数文件读取 {key}: {parameter_file}"
    )


def _read_gamma_int(
    parameter_file: Path,
    keys: Iterable[str],
) -> int:
    """Read an integer-valued GAMMA parameter using alternative keys."""

    errors: list[str] = []

    for key in keys:
        try:
            token = _read_gamma_parameter(
                parameter_file,
                key,
            )
        except GammaLonLatError as exc:
            errors.append(str(exc))
            continue

        try:
            return int(round(float(token)))
        except ValueError:
            errors.append(
                f"{parameter_file} 中 {key} 不是数值：{token}"
            )

    key_text = ", ".join(keys)

    raise GammaLonLatError(
        f"无法从 {parameter_file} 读取任一尺寸字段："
        f"{key_text}"
    )


def _file_has_exact_size(
    path: Path,
    expected_bytes: int,
) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
    )


def _extract_date(path: Path) -> str | None:
    match = _DATE_RE.search(path.name)

    if match is None:
        return None

    return match.group(1)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    print()
    print("执行GAMMA命令：")
    print("  " + " ".join(command))

    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
        )
    except FileNotFoundError as exc:
        raise GammaLonLatError(
            f"找不到GAMMA程序：{command[0]}。"
            "请确认GAMMA环境已经加载，且命令位于PATH中。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GammaLonLatError(
            f"GAMMA命令运行失败，返回码={exc.returncode}："
            f"{' '.join(command)}"
        ) from exc


def _ensure_gamma_commands() -> None:
    missing = [
        command
        for command in ("dem_coord", "geocode")
        if shutil.which(command) is None
    ]

    if missing:
        raise GammaLonLatError(
            "以下GAMMA命令不在PATH中："
            + ", ".join(missing)
        )

def _dem_data_file_from_parameter(
    parameter_file: Path,
) -> Path:
    """
    Convert a GAMMA DEM parameter filename to its data filename.

    Examples
    --------
    N38E117.dem_par -> N38E117.dem
    dem_seg.par     -> dem_seg
    """

    name = parameter_file.name

    if name.endswith("_par"):
        return parameter_file.with_name(
            name[:-4]
        )

    return parameter_file.with_suffix("")

def _select_dem_parameter_file(
    dem_directory: Path,
    explicit_file: Path | None,
) -> tuple[Path, Path]:
    """
    Select ``*.dem_par`` and its matching DEM data file.

    For example:

        N38E117.dem_par
        N38E117.dem
    """

    if explicit_file is not None:
        parameter_file = explicit_file.expanduser().resolve()

        if not parameter_file.is_file():
            raise GammaLonLatError(
                f"指定的DEM参数文件不存在：{parameter_file}"
            )

        dem_file = _dem_data_file_from_parameter(
	    parameter_file
	)

        if not dem_file.is_file():
            raise GammaLonLatError(
                "DEM参数文件存在，但找不到同名DEM数据："
                f"{dem_file}"
            )

        return parameter_file, dem_file

    candidates: list[tuple[int, Path, Path]] = []

    for parameter_file in sorted(
        dem_directory.glob("*.dem_par")
    ):
        dem_file = _dem_data_file_from_parameter(
	    parameter_file
	)

        if not dem_file.is_file():
            continue

        score = 0
        name_lower = parameter_file.name.lower()

        # 优先标准 DEM 名称，例如 N38E117.dem_par。
        if name_lower != "dem_seg.par":
            score += 20

        if re.search(
            r"[ns]\d{1,2}[ew]\d{1,3}",
            name_lower,
        ):
            score += 10

        candidates.append(
            (
                score,
                parameter_file.resolve(),
                dem_file.resolve(),
            )
        )

    if not candidates:
        raise GammaLonLatError(
            f"{dem_directory} 中未找到同时具有DEM数据文件的 "
            "*.dem_par"
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        ),
        reverse=True,
    )

    _, parameter_file, dem_file = candidates[0]

    return parameter_file, dem_file


def _select_radar_parameter_file(
    dem_directory: Path,
    *,
    radar_width: int | None,
    radar_length: int | None,
    range_looks: int,
    azimuth_looks: int,
    explicit_file: Path | None,
) -> tuple[Path, int, int]:
    """
    Select the MLI parameter file matching the required radar grid.

    For a 4:1 project, this preferentially selects:

        20210211_4_1.vv.mli.par
    """

    if explicit_file is not None:
        candidates = [explicit_file.expanduser().resolve()]
    else:
        candidates = sorted(
            dem_directory.glob("*.mli.par")
        )

    ranked: list[tuple[int, Path, int, int]] = []

    look_token = f"_{range_looks}_{azimuth_looks}"

    for parameter_file in candidates:
        if not parameter_file.is_file():
            continue

        try:
            width = _read_gamma_int(
                parameter_file,
                (
                    "range_samples",
                    "interferogram_width",
                    "width",
                    "range_samp_1",
                ),
            )

            length = _read_gamma_int(
                parameter_file,
                (
                    "azimuth_lines",
                    "interferogram_azimuth_lines",
                    "nlines",
                    "az_samp_1",
                ),
            )
        except GammaLonLatError:
            continue

        if (
            radar_width is not None
            and width != radar_width
        ):
            continue

        if (
            radar_length is not None
            and length != radar_length
        ):
            continue

        score = 0
        name = parameter_file.name

        if look_token in name:
            score += 100

        if _extract_date(parameter_file) is not None:
            score += 20

        if name.endswith(".vv.mli.par"):
            score += 10
        elif name.endswith(".mli.par"):
            score += 5

        ranked.append(
            (
                score,
                parameter_file.resolve(),
                width,
                length,
            )
        )

    if not ranked:
        target_text = ""

        if (
            radar_width is not None
            and radar_length is not None
        ):
            target_text = (
                f"，目标尺寸为 {radar_width} × "
                f"{radar_length}"
            )

        raise GammaLonLatError(
            f"{dem_directory} 中未找到与 "
            f"{range_looks}:{azimuth_looks} 多视及雷达网格"
            f"匹配的 *.mli.par{target_text}"
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        ),
        reverse=True,
    )

    _, parameter_file, width, length = ranked[0]

    return parameter_file, width, length
    
def _select_lookup_table(
    dem_directory: Path,
    *,
    reference_date: str | None,
    map_width: int,
    map_length: int,
    explicit_file: Path | None,
) -> Path:
    """
    Select a forward geocoding lookup table.

    ``*.lt_fine`` is always preferred over ``*.lt``.
    """

    expected_bytes = map_width * map_length * 8

    if explicit_file is not None:
        lookup_file = explicit_file.expanduser().resolve()

        if not _file_has_exact_size(
            lookup_file,
            expected_bytes,
        ):
            actual = (
                lookup_file.stat().st_size
                if lookup_file.exists()
                else None
            )

            raise GammaLonLatError(
                "指定的查找表不存在或尺寸不匹配："
                f"{lookup_file}；"
                f"期望 {expected_bytes} 字节，"
                f"实际 {actual}"
            )

        return lookup_file

    patterns: list[str] = []

    if reference_date is not None:
        patterns.extend(
            (
                f"*{reference_date}*.lt_fine",
                f"*{reference_date}*.lt",
            )
        )

    patterns.extend(
        (
            "*.lt_fine",
            "*.lt",
        )
    )

    seen: set[Path] = set()
    ranked: list[tuple[int, Path]] = []

    for pattern in patterns:
        for path in sorted(
            dem_directory.glob(pattern)
        ):
            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            if not _file_has_exact_size(
                resolved,
                expected_bytes,
            ):
                continue

            score = 0

            if resolved.name.endswith(".lt_fine"):
                score += 100

            if (
                reference_date is not None
                and reference_date in resolved.name
            ):
                score += 50

            ranked.append(
                (
                    score,
                    resolved,
                )
            )

    if not ranked:
        raise GammaLonLatError(
            f"{dem_directory} 中找不到尺寸为 "
            f"{expected_bytes} 字节的 *.lt_fine 或 *.lt"
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        ),
        reverse=True,
    )

    return ranked[0][1]


def _find_existing_map_lonlat(
    dem_directory: Path,
    *,
    dem_file: Path,
    map_width: int,
    map_length: int,
) -> tuple[Path, Path] | None:
    expected_bytes = map_width * map_length * 4

    canonical_lon = dem_directory / (
        dem_file.name + ".lon"
    )
    canonical_lat = dem_directory / (
        dem_file.name + ".lat"
    )

    candidate_pairs: list[tuple[Path, Path]] = [
        (
            canonical_lon,
            canonical_lat,
        ),
        (
            dem_directory / "dem_seg.lon",
            dem_directory / "dem_seg.lat",
        ),
    ]

    lon_files = sorted(
        path
        for path in dem_directory.glob("*.lon")
        if ".rdc." not in path.name
    )

    for lon_file in lon_files:
        lat_file = lon_file.with_suffix(".lat")

        candidate_pairs.append(
            (
                lon_file,
                lat_file,
            )
        )

    seen: set[tuple[Path, Path]] = set()

    for longitude_file, latitude_file in candidate_pairs:
        pair = (
            longitude_file.resolve(),
            latitude_file.resolve(),
        )

        if pair in seen:
            continue

        seen.add(pair)

        if (
            _file_has_exact_size(
                pair[0],
                expected_bytes,
            )
            and _file_has_exact_size(
                pair[1],
                expected_bytes,
            )
        ):
            return pair

    return None


def _find_existing_radar_lonlat(
    dem_directory: Path,
    *,
    canonical_longitude_file: Path,
    canonical_latitude_file: Path,
    radar_width: int,
    radar_length: int,
) -> tuple[Path, Path] | None:
    expected_bytes = radar_width * radar_length * 4

    candidate_pairs: list[tuple[Path, Path]] = [
        (
            canonical_longitude_file,
            canonical_latitude_file,
        )
    ]

    for longitude_file in sorted(
        dem_directory.glob("*.rdc.lon")
    ):
        latitude_file = Path(
            str(longitude_file)[:-4] + ".lat"
        )

        candidate_pairs.append(
            (
                longitude_file,
                latitude_file,
            )
        )

    seen: set[tuple[Path, Path]] = set()

    for longitude_file, latitude_file in candidate_pairs:
        pair = (
            longitude_file.resolve(),
            latitude_file.resolve(),
        )

        if pair in seen:
            continue

        seen.add(pair)

        if (
            _file_has_exact_size(
                pair[0],
                expected_bytes,
            )
            and _file_has_exact_size(
                pair[1],
                expected_bytes,
            )
        ):
            return pair

    return None


def _generate_map_coordinates(
    *,
    dem_directory: Path,
    dem_parameter_file: Path,
    longitude_file: Path,
    latitude_file: Path,
    map_width: int,
    map_length: int,
) -> None:
    expected_bytes = map_width * map_length * 4

    longitude_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latitude_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    longitude_tmp = longitude_file.with_name(
        longitude_file.name + ".tmp"
    )

    latitude_tmp = latitude_file.with_name(
        latitude_file.name + ".tmp"
    )

    longitude_tmp.unlink(
        missing_ok=True,
    )

    latitude_tmp.unlink(
        missing_ok=True,
    )

    _run_command(
        [
            "dem_coord",
            str(dem_parameter_file),
            str(longitude_tmp),
            str(latitude_tmp),
        ],
        cwd=dem_directory,
    )

    if not _file_has_exact_size(
        longitude_tmp,
        expected_bytes,
    ):
        raise GammaLonLatError(
            "dem_coord生成的地图经度文件尺寸错误："
            f"{longitude_tmp}"
        )

    if not _file_has_exact_size(
        latitude_tmp,
        expected_bytes,
    ):
        raise GammaLonLatError(
            "dem_coord生成的地图纬度文件尺寸错误："
            f"{latitude_tmp}"
        )

    longitude_tmp.replace(
        longitude_file
    )

    latitude_tmp.replace(
        latitude_file
    )


def _geocode_one_coordinate(
    *,
    dem_directory: Path,
    lookup_table_file: Path,
    input_file: Path,
    map_width: int,
    output_file: Path,
    radar_width: int,
    radar_length: int,
) -> None:
    expected_bytes = radar_width * radar_length * 4

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_tmp = output_file.with_name(
        output_file.name + ".tmp"
    )

    output_tmp.unlink(
        missing_ok=True,
    )

    _run_command(
        [
            "geocode",
            str(lookup_table_file),
            str(input_file),
            str(map_width),
            str(output_tmp),
            str(radar_width),
            str(radar_length),

            # 2：1 / distance^2 插值
            "2",

            # 0：FLOAT
            "0",
        ],
        cwd=dem_directory,
    )

    if not _file_has_exact_size(
        output_tmp,
        expected_bytes,
    ):
        actual = (
            output_tmp.stat().st_size
            if output_tmp.exists()
            else None
        )

        raise GammaLonLatError(
            f"geocode输出尺寸错误：{output_tmp}；"
            f"期望 {expected_bytes} 字节，"
            f"实际 {actual}"
        )

    output_tmp.replace(
        output_file
    )


def ensure_gamma_radar_lonlat(
    project_directory: str | Path,
    *,
    radar_width: int | None = None,
    radar_length: int | None = None,
    range_looks: int = 4,
    azimuth_looks: int = 1,
    dem_directory: str | Path | None = None,
    longitude_file: str | Path | None = None,
    latitude_file: str | Path | None = None,
    dem_parameter_file: str | Path | None = None,
    radar_parameter_file: str | Path | None = None,
    lookup_table_file: str | Path | None = None,
    force: bool = False,
) -> GammaLonLatResult:
    """
    Resolve or generate radar-coordinate longitude and latitude rasters.

    Parameters
    ----------
    project_directory
        GAMMA project root containing ``DEM_prep``.
    radar_width, radar_length
        Expected interferogram/radar grid dimensions. When omitted, dimensions
        are read from the selected MLI parameter file.
    range_looks, azimuth_looks
        Used to preferentially select e.g. ``*_4_1*.mli.par`` and construct
        canonical output names.
    longitude_file, latitude_file
        Optional explicit radar-coordinate outputs. Both must be provided
        together.
    force
        Regenerate radar lon/lat even when valid files already exist.
    """

    project_root = Path(
        project_directory
    ).expanduser().resolve()

    if dem_directory is None:
        dem_dir = project_root / "DEM_prep"
    else:
        dem_dir = Path(
            dem_directory
        ).expanduser().resolve()

    if not dem_dir.is_dir():
        raise GammaLonLatError(
            f"DEM准备目录不存在：{dem_dir}"
        )

    explicit_longitude = (
        Path(longitude_file).expanduser().resolve()
        if longitude_file is not None
        else None
    )

    explicit_latitude = (
        Path(latitude_file).expanduser().resolve()
        if latitude_file is not None
        else None
    )

    if (
        explicit_longitude is None
    ) != (
        explicit_latitude is None
    ):
        raise GammaLonLatError(
            "longitude_file和latitude_file必须同时指定，"
            "不能只指定其中一个"
        )

    dem_par, dem_file = _select_dem_parameter_file(
        dem_dir,
        (
            Path(dem_parameter_file)
            if dem_parameter_file is not None
            else None
        ),
    )

    map_width = _read_gamma_int(
        dem_par,
        ("width", "range_samples"),
    )

    map_length = _read_gamma_int(
        dem_par,
        ("nlines", "azimuth_lines"),
    )

    radar_par, resolved_width, resolved_length = (
        _select_radar_parameter_file(
            dem_dir,
            radar_width=radar_width,
            radar_length=radar_length,
            range_looks=range_looks,
            azimuth_looks=azimuth_looks,
            explicit_file=(
                Path(radar_parameter_file)
                if radar_parameter_file is not None
                else None
            ),
        )
    )

    if radar_width is None:
        radar_width = resolved_width

    if radar_length is None:
        radar_length = resolved_length

    if (
        radar_width != resolved_width
        or radar_length != resolved_length
    ):
        raise GammaLonLatError(
            "选中的MLI参数文件尺寸与目标雷达网格不一致："
            f"{radar_par} -> "
            f"{resolved_width} × {resolved_length}，"
            f"目标为 {radar_width} × {radar_length}"
        )

    reference_date = _extract_date(
        radar_par
    )

    lookup_file = _select_lookup_table(
        dem_dir,
        reference_date=reference_date,
        map_width=map_width,
        map_length=map_length,
        explicit_file=(
            Path(lookup_table_file)
            if lookup_table_file is not None
            else None
        ),
    )

    if reference_date is None:
        radar_prefix = (
            f"radar_{range_looks}_{azimuth_looks}"
        )
    else:
        radar_prefix = (
            f"{reference_date}_"
            f"{range_looks}_{azimuth_looks}"
        )

    canonical_longitude = (
        explicit_longitude
        if explicit_longitude is not None
        else dem_dir / f"{radar_prefix}.rdc.lon"
    )

    canonical_latitude = (
        explicit_latitude
        if explicit_latitude is not None
        else dem_dir / f"{radar_prefix}.rdc.lat"
    )

    if not force:
        existing_radar = _find_existing_radar_lonlat(
            dem_dir,
            canonical_longitude_file=canonical_longitude,
            canonical_latitude_file=canonical_latitude,
            radar_width=radar_width,
            radar_length=radar_length,
        )

        if existing_radar is not None:
            longitude_resolved, latitude_resolved = (
                existing_radar
            )

            existing_map = _find_existing_map_lonlat(
                dem_dir,
                dem_file=dem_file,
                map_width=map_width,
                map_length=map_length,
            )

            if existing_map is None:
                map_longitude = (
                    dem_dir / (
                        dem_file.name + ".lon"
                    )
                )
                map_latitude = (
                    dem_dir / (
                        dem_file.name + ".lat"
                    )
                )
            else:
                map_longitude, map_latitude = (
                    existing_map
                )

            print(
                "复用已有雷达坐标经纬度："
            )
            print(
                f"  longitude: {longitude_resolved}"
            )
            print(
                f"  latitude : {latitude_resolved}"
            )

            return GammaLonLatResult(
                longitude_file=longitude_resolved,
                latitude_file=latitude_resolved,
                map_longitude_file=map_longitude,
                map_latitude_file=map_latitude,
                dem_parameter_file=dem_par,
                dem_file=dem_file,
                lookup_table_file=lookup_file,
                radar_parameter_file=radar_par,
                radar_width=radar_width,
                radar_length=radar_length,
                map_width=map_width,
                map_length=map_length,
                generated_map_coordinates=False,
                generated_radar_coordinates=False,
            )

    _ensure_gamma_commands()

    existing_map = _find_existing_map_lonlat(
        dem_dir,
        dem_file=dem_file,
        map_width=map_width,
        map_length=map_length,
    )

    generated_map = False

    if existing_map is None:
        map_longitude = (
            dem_dir / (
                dem_file.name + ".lon"
            )
        )

        map_latitude = (
            dem_dir / (
                dem_file.name + ".lat"
            )
        )

        print(
            "未检测到有效地图坐标lon/lat，"
            "正在运行dem_coord..."
        )

        _generate_map_coordinates(
            dem_directory=dem_dir,
            dem_parameter_file=dem_par,
            longitude_file=map_longitude,
            latitude_file=map_latitude,
            map_width=map_width,
            map_length=map_length,
        )

        generated_map = True
    else:
        map_longitude, map_latitude = (
            existing_map
        )

        print(
            "复用已有地图坐标经纬度："
        )
        print(
            f"  map longitude: {map_longitude}"
        )
        print(
            f"  map latitude : {map_latitude}"
        )

    print(
        "正在生成雷达坐标经纬度："
    )
    print(
        f"  lookup table: {lookup_file}"
    )
    print(
        f"  radar grid : {radar_width} × "
        f"{radar_length}"
    )

    _geocode_one_coordinate(
        dem_directory=dem_dir,
        lookup_table_file=lookup_file,
        input_file=map_longitude,
        map_width=map_width,
        output_file=canonical_longitude,
        radar_width=radar_width,
        radar_length=radar_length,
    )

    _geocode_one_coordinate(
        dem_directory=dem_dir,
        lookup_table_file=lookup_file,
        input_file=map_latitude,
        map_width=map_width,
        output_file=canonical_latitude,
        radar_width=radar_width,
        radar_length=radar_length,
    )

    print(
        "雷达坐标经纬度准备完成："
    )
    print(
        f"  longitude: {canonical_longitude}"
    )
    print(
        f"  latitude : {canonical_latitude}"
    )

    return GammaLonLatResult(
        longitude_file=canonical_longitude.resolve(),
        latitude_file=canonical_latitude.resolve(),
        map_longitude_file=map_longitude.resolve(),
        map_latitude_file=map_latitude.resolve(),
        dem_parameter_file=dem_par,
        dem_file=dem_file,
        lookup_table_file=lookup_file,
        radar_parameter_file=radar_par,
        radar_width=radar_width,
        radar_length=radar_length,
        map_width=map_width,
        map_length=map_length,
        generated_map_coordinates=generated_map,
        generated_radar_coordinates=True,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve or generate GAMMA radar-coordinate "
            "longitude/latitude files."
        )
    )

    parser.add_argument(
        "project_directory",
        type=Path,
    )

    parser.add_argument(
        "--dem-directory",
        type=Path,
    )

    parser.add_argument(
        "--radar-width",
        type=int,
    )

    parser.add_argument(
        "--radar-length",
        type=int,
    )

    parser.add_argument(
        "--range-looks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--azimuth-looks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--longitude-file",
        type=Path,
    )

    parser.add_argument(
        "--latitude-file",
        type=Path,
    )

    parser.add_argument(
        "--dem-parameter-file",
        type=Path,
    )

    parser.add_argument(
        "--radar-parameter-file",
        type=Path,
    )

    parser.add_argument(
        "--lookup-table-file",
        type=Path,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()

    result = ensure_gamma_radar_lonlat(
        args.project_directory,
        radar_width=args.radar_width,
        radar_length=args.radar_length,
        range_looks=args.range_looks,
        azimuth_looks=args.azimuth_looks,
        dem_directory=args.dem_directory,
        longitude_file=args.longitude_file,
        latitude_file=args.latitude_file,
        dem_parameter_file=args.dem_parameter_file,
        radar_parameter_file=args.radar_parameter_file,
        lookup_table_file=args.lookup_table_file,
        force=args.force,
    )

    print()
    print("结果：")
    print(
        f"  longitude_file = {result.longitude_file}"
    )
    print(
        f"  latitude_file  = {result.latitude_file}"
    )
    print(
        f"  generated_map  = "
        f"{result.generated_map_coordinates}"
    )
    print(
        f"  generated_radar = "
        f"{result.generated_radar_coordinates}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
