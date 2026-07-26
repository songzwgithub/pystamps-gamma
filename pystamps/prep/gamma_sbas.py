from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")

WIDTH_KEYS = (
    "interferogram_width",
    "range_samples",
    "range_samp_1",
    "width",
)

LENGTH_KEYS = (
    "interferogram_azimuth_lines",
    "azimuth_lines",
    "az_samp_1",
    "nlines",
)

MLI_SUFFIXES = (
    ".mli",
    ".rmli",
)

COMMENT_PREFIXES = ("#", "%", ";")


class GammaInputError(RuntimeError):
    """Raised when a GAMMA SBAS project is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class GammaAcquisition:
    """One acquisition listed by RSLC_tab."""

    index: int
    date: str
    rslc: Path
    par: Path
    mli: Path | None
    mli_par: Path | None


@dataclass(frozen=True, slots=True)
class GammaInterferogram:
    """One small-baseline interferogram listed by itab."""

    index: int
    master_index: int
    slave_index: int
    master_date: str
    slave_date: str
    diff: Path
    base: Path
    diff_par: Path | None
    off: Path | None
    coherence: Path | None


@dataclass(frozen=True, slots=True)
class GammaSbasProject:
    """Resolved GAMMA SBAS project inputs."""

    root: Path
    rslc_tab: Path
    itab: Path
    diff_dir: Path
    mli_dir: Path
    dem_dir: Path
    acquisitions: tuple[GammaAcquisition, ...]
    interferograms: tuple[GammaInterferogram, ...]
    width: int | None
    length: int | None
    network_connected: bool


def _strip_inline_comment(line: str) -> str:
    """Remove simple inline comments from text-list files."""

    result = line

    for marker in COMMENT_PREFIXES:
        marker_index = result.find(marker)
        if marker_index >= 0:
            result = result[:marker_index]

    return result.strip()


def _resolve_list_path(raw_path: str, list_file: Path) -> Path:
    """Resolve a path occurring inside RSLC_tab."""

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = list_file.parent / path

    return path.resolve()


def extract_date(value: str | Path) -> str:
    """Extract one YYYYMMDD date from a path or filename."""

    text = str(value)
    matches = DATE_RE.findall(text)

    if not matches:
        raise GammaInputError(
            f"无法从以下路径或文件名中提取YYYYMMDD日期：{value}"
        )

    # Path中可能有其他日期目录，文件名日期优先。
    filename_matches = DATE_RE.findall(Path(text).name)
    if filename_matches:
        return filename_matches[-1]

    return matches[-1]


def parse_gamma_parameter_file(path: Path) -> dict[str, list[str]]:
    """
    Parse a GAMMA text parameter file.

    Keys are stored without the trailing colon. Values remain as strings
    because GAMMA parameter lines may contain numerical values and units.
    """

    if not path.is_file():
        raise GammaInputError(f"GAMMA参数文件不存在：{path}")

    parameters: dict[str, list[str]] = {}

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith(COMMENT_PREFIXES):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            values = value.strip().split()
        else:
            fields = line.split()
            if len(fields) < 2:
                continue
            key = fields[0].strip()
            values = fields[1:]

        if key:
            parameters[key] = values

    return parameters


def _first_numeric_value(
    parameters: dict[str, list[str]],
    keys: Iterable[str],
) -> float | None:
    for key in keys:
        values = parameters.get(key)

        if not values:
            continue

        for token in values:
            try:
                return float(token)
            except ValueError:
                continue

    return None


def first_integer_parameter(
    parameters: dict[str, list[str]],
    keys: Iterable[str],
) -> int | None:
    value = _first_numeric_value(parameters, keys)

    if value is None:
        return None

    rounded = int(round(value))

    if rounded <= 0:
        return None

    return rounded


def parse_rslc_tab(path: Path) -> list[tuple[Path, Path, str]]:
    """
    Parse the first two columns of a GAMMA RSLC_tab.

    Returns:
        List of (rslc_path, parameter_path, YYYYMMDD).
    """

    path = path.expanduser().resolve()

    if not path.is_file():
        raise GammaInputError(f"RSLC_tab不存在：{path}")

    records: list[tuple[Path, Path, str]] = []

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(),
        start=1,
    ):
        line = _strip_inline_comment(raw_line)

        if not line:
            continue

        fields = line.split()

        if len(fields) < 2:
            raise GammaInputError(
                f"{path}:{line_number} 至少需要两列：RSLC和RSLC参数文件"
            )

        rslc = _resolve_list_path(fields[0], path)
        par = _resolve_list_path(fields[1], path)

        if not rslc.is_file():
            raise GammaInputError(
                f"{path}:{line_number} RSLC文件不存在：{rslc}"
            )

        if not par.is_file():
            raise GammaInputError(
                f"{path}:{line_number} RSLC参数文件不存在：{par}"
            )

        date = extract_date(rslc)
        records.append((rslc, par, date))

    if len(records) < 2:
        raise GammaInputError(
            f"RSLC_tab仅解析到{len(records)}景影像，无法建立干涉网络"
        )

    dates = [record[2] for record in records]
    duplicate_dates = sorted(
        {
            date
            for date in dates
            if dates.count(date) > 1
        }
    )

    if duplicate_dates:
        raise GammaInputError(
            "RSLC_tab包含重复日期："
            + ", ".join(duplicate_dates)
        )

    return records


def parse_itab(
    path: Path,
    acquisition_count: int,
) -> list[tuple[int, int]]:
    """
    Parse the first two columns of a GAMMA itab.

    GAMMA image indices are one-based.
    """

    path = path.expanduser().resolve()

    if not path.is_file():
        raise GammaInputError(f"itab不存在：{path}")

    pairs: list[tuple[int, int]] = []

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(),
        start=1,
    ):
        line = _strip_inline_comment(raw_line)

        if not line:
            continue

        fields = line.split()

        if len(fields) < 2:
            raise GammaInputError(
                f"{path}:{line_number} 至少需要两个影像索引"
            )

        try:
            master_index = int(fields[0])
            slave_index = int(fields[1])
        except ValueError as exc:
            raise GammaInputError(
                f"{path}:{line_number} 前两列必须是整数：{line}"
            ) from exc

        if not 1 <= master_index <= acquisition_count:
            raise GammaInputError(
                f"{path}:{line_number} 主影像索引越界："
                f"{master_index}，影像总数为{acquisition_count}"
            )

        if not 1 <= slave_index <= acquisition_count:
            raise GammaInputError(
                f"{path}:{line_number} 从影像索引越界："
                f"{slave_index}，影像总数为{acquisition_count}"
            )

        if master_index == slave_index:
            raise GammaInputError(
                f"{path}:{line_number} 主从影像索引相同：{master_index}"
            )

        pairs.append((master_index, slave_index))

    if not pairs:
        raise GammaInputError("itab中没有解析到有效干涉对")

    duplicate_pairs = sorted(
        {
            pair
            for pair in pairs
            if pairs.count(pair) > 1
        }
    )

    if duplicate_pairs:
        pair_text = ", ".join(
            f"{master}-{slave}"
            for master, slave in duplicate_pairs
        )
        raise GammaInputError(f"itab包含重复干涉对：{pair_text}")

    return pairs


def _find_unique_date_file(
    directory: Path,
    date: str,
    suffixes: tuple[str, ...],
) -> Path | None:
    if not directory.is_dir():
        return None

    candidates: list[Path] = []

    for path in directory.iterdir():
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        if date not in path.name:
            continue

        if any(lower_name.endswith(suffix) for suffix in suffixes):
            candidates.append(path.resolve())

    candidates = sorted(set(candidates))

    if not candidates:
        return None

    if len(candidates) > 1:
        raise GammaInputError(
            f"日期{date}在{directory}中匹配到多个MLI文件："
            + ", ".join(path.name for path in candidates)
        )

    return candidates[0]


def _find_parameter_sidecar(image: Path | None) -> Path | None:
    if image is None:
        return None

    candidates = (
        Path(f"{image}.par"),
        image.with_suffix(f"{image.suffix}.par"),
        image.with_suffix(".par"),
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def resolve_acquisitions(
    rslc_records: list[tuple[Path, Path, str]],
    mli_dir: Path,
) -> tuple[GammaAcquisition, ...]:
    acquisitions: list[GammaAcquisition] = []

    for index, (rslc, par, date) in enumerate(
        rslc_records,
        start=1,
    ):
        mli = _find_unique_date_file(
            mli_dir,
            date,
            MLI_SUFFIXES,
        )

        acquisitions.append(
            GammaAcquisition(
                index=index,
                date=date,
                rslc=rslc,
                par=par,
                mli=mli,
                mli_par=_find_parameter_sidecar(mli),
            )
        )

    return tuple(acquisitions)


def resolve_interferogram(
    *,
    pair_index: int,
    master_index: int,
    slave_index: int,
    acquisitions: tuple[GammaAcquisition, ...],
    diff_dir: Path,
) -> GammaInterferogram:
    master = acquisitions[master_index - 1]
    slave = acquisitions[slave_index - 1]

    pair_name = f"{master.date}_{slave.date}"

    diff = diff_dir / f"{pair_name}.diff"
    base = diff_dir / f"{pair_name}.base"
    diff_par = diff_dir / f"{pair_name}.diff_par"
    off = diff_dir / f"{pair_name}.off"
    coherence = diff_dir / f"{pair_name}.cc"

    if not diff.is_file():
        reverse = diff_dir / f"{slave.date}_{master.date}.diff"

        if reverse.is_file():
            raise GammaInputError(
                f"itab第{pair_index}个干涉对为{pair_name}，"
                f"但DIFF中只存在反向干涉图{reverse.name}。"
                "第一版导入模块不自动改变相位符号，请统一itab与DIFF方向。"
            )

        raise GammaInputError(
            f"缺少itab第{pair_index}个干涉对的复数干涉图：{diff}"
        )

    if not base.is_file():
        raise GammaInputError(
            f"缺少itab第{pair_index}个干涉对的基线文件：{base}"
        )

    if not diff_par.is_file() and not off.is_file():
        raise GammaInputError(
            f"{pair_name}同时缺少.diff_par和.off，"
            "无法可靠核验干涉图尺寸"
        )

    return GammaInterferogram(
        index=pair_index,
        master_index=master_index,
        slave_index=slave_index,
        master_date=master.date,
        slave_date=slave.date,
        diff=diff.resolve(),
        base=base.resolve(),
        diff_par=diff_par.resolve() if diff_par.is_file() else None,
        off=off.resolve() if off.is_file() else None,
        coherence=coherence.resolve() if coherence.is_file() else None,
    )


def resolve_interferograms(
    pairs: list[tuple[int, int]],
    acquisitions: tuple[GammaAcquisition, ...],
    diff_dir: Path,
) -> tuple[GammaInterferogram, ...]:
    return tuple(
        resolve_interferogram(
            pair_index=pair_index,
            master_index=master_index,
            slave_index=slave_index,
            acquisitions=acquisitions,
            diff_dir=diff_dir,
        )
        for pair_index, (master_index, slave_index) in enumerate(
            pairs,
            start=1,
        )
    )


def _dimensions_from_file(
    parameter_file: Path | None,
) -> tuple[int | None, int | None]:
    if parameter_file is None:
        return None, None

    parameters = parse_gamma_parameter_file(parameter_file)

    width = first_integer_parameter(
        parameters,
        WIDTH_KEYS,
    )
    length = first_integer_parameter(
        parameters,
        LENGTH_KEYS,
    )

    return width, length


def resolve_dimensions(
    acquisitions: tuple[GammaAcquisition, ...],
    interferograms: tuple[GammaInterferogram, ...],
) -> tuple[int | None, int | None]:
    """
    Resolve multilooked interferogram dimensions.

    Priority:
      1. first interferogram diff_par;
      2. first interferogram off;
      3. first MLI parameter file.
    """

    candidates: list[tuple[Path, int | None, int | None]] = []

    first_ifg = interferograms[0]

    for parameter_file in (
        first_ifg.diff_par,
        first_ifg.off,
        acquisitions[0].mli_par,
    ):
        if parameter_file is None:
            continue

        width, length = _dimensions_from_file(parameter_file)
        candidates.append((parameter_file, width, length))

    width: int | None = None
    length: int | None = None

    for _, candidate_width, candidate_length in candidates:
        if width is None and candidate_width is not None:
            width = candidate_width

        if length is None and candidate_length is not None:
            length = candidate_length

    # 对所有能解析出的尺寸进行一致性检查。
    for parameter_file, candidate_width, candidate_length in candidates:
        if (
            width is not None
            and candidate_width is not None
            and candidate_width != width
        ):
            raise GammaInputError(
                f"宽度不一致：{parameter_file}给出{candidate_width}，"
                f"此前解析结果为{width}"
            )

        if (
            length is not None
            and candidate_length is not None
            and candidate_length != length
        ):
            raise GammaInputError(
                f"行数不一致：{parameter_file}给出{candidate_length}，"
                f"此前解析结果为{length}"
            )

    return width, length


def validate_diff_file_sizes(
    interferograms: tuple[GammaInterferogram, ...],
    width: int | None,
    length: int | None,
) -> None:
    """Validate GAMMA FCOMPLEX file sizes when dimensions are known."""

    if width is None or length is None:
        return

    expected_bytes = width * length * 8

    mismatches: list[str] = []

    for interferogram in interferograms:
        actual_bytes = interferogram.diff.stat().st_size

        if actual_bytes != expected_bytes:
            mismatches.append(
                f"{interferogram.diff.name}: "
                f"{actual_bytes}字节，期望{expected_bytes}字节"
            )

    if mismatches:
        preview = "\n".join(mismatches[:20])
        suffix = (
            f"\n其余{len(mismatches) - 20}个错误未显示"
            if len(mismatches) > 20
            else ""
        )

        raise GammaInputError(
            "部分.diff文件不符合"
            f"{length}×{width}的GAMMA FCOMPLEX尺寸：\n"
            f"{preview}{suffix}"
        )


def network_is_connected(
    acquisition_count: int,
    pairs: Iterable[tuple[int, int]],
) -> bool:
    adjacency: dict[int, set[int]] = {
        index: set()
        for index in range(1, acquisition_count + 1)
    }

    used_nodes: set[int] = set()

    for master_index, slave_index in pairs:
        adjacency[master_index].add(slave_index)
        adjacency[slave_index].add(master_index)
        used_nodes.add(master_index)
        used_nodes.add(slave_index)

    if not used_nodes:
        return False

    start = min(used_nodes)
    stack = [start]
    visited: set[int] = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        stack.extend(adjacency[node] - visited)

    # 所有RSLC_tab影像都应参与并连接到网络。
    expected_nodes = set(range(1, acquisition_count + 1))
    return visited == expected_nodes


def load_gamma_sbas_project(
    project_root: str | Path,
    *,
    rslc_tab_name: str = "RSLC_tab",
    itab_name: str = "itab",
    diff_dir_name: str = "DIFF",
    mli_dir_name: str = "MLI_dir",
    dem_dir_name: str = "DEM_prep",
) -> GammaSbasProject:
    """Resolve and validate the top-level GAMMA SBAS input structure."""

    root = Path(project_root).expanduser().resolve()

    if not root.is_dir():
        raise GammaInputError(f"GAMMA工程目录不存在：{root}")

    rslc_tab = (root / rslc_tab_name).resolve()
    itab = (root / itab_name).resolve()
    diff_dir = (root / diff_dir_name).resolve()
    mli_dir = (root / mli_dir_name).resolve()
    dem_dir = (root / dem_dir_name).resolve()

    if not diff_dir.is_dir():
        raise GammaInputError(f"DIFF目录不存在：{diff_dir}")

    if not mli_dir.is_dir():
        raise GammaInputError(f"MLI_dir目录不存在：{mli_dir}")

    if not dem_dir.is_dir():
        raise GammaInputError(f"DEM_prep目录不存在：{dem_dir}")

    rslc_records = parse_rslc_tab(rslc_tab)

    acquisitions = resolve_acquisitions(
        rslc_records,
        mli_dir,
    )

    pairs = parse_itab(
        itab,
        len(acquisitions),
    )

    interferograms = resolve_interferograms(
        pairs,
        acquisitions,
        diff_dir,
    )

    width, length = resolve_dimensions(
        acquisitions,
        interferograms,
    )

    validate_diff_file_sizes(
        interferograms,
        width,
        length,
    )

    connected = network_is_connected(
        len(acquisitions),
        pairs,
    )

    if not connected:
        raise GammaInputError(
            "itab小基线网络不连通，或RSLC_tab中存在未参加任何干涉对的影像"
        )

    return GammaSbasProject(
        root=root,
        rslc_tab=rslc_tab,
        itab=itab,
        diff_dir=diff_dir,
        mli_dir=mli_dir,
        dem_dir=dem_dir,
        acquisitions=acquisitions,
        interferograms=interferograms,
        width=width,
        length=length,
        network_connected=connected,
    )


def inspect_gamma_sbas_project(
    project_root: str | Path,
    **kwargs: str,
) -> dict[str, object]:
    """Return a JSON-serializable GAMMA SBAS inspection report."""

    project = load_gamma_sbas_project(
        project_root,
        **kwargs,
    )

    missing_mli = [
        acquisition.date
        for acquisition in project.acquisitions
        if acquisition.mli is None
    ]

    missing_mli_par = [
        acquisition.date
        for acquisition in project.acquisitions
        if acquisition.mli is not None
        and acquisition.mli_par is None
    ]

    missing_coherence = [
        (
            f"{interferogram.master_date}_"
            f"{interferogram.slave_date}"
        )
        for interferogram in project.interferograms
        if interferogram.coherence is None
    ]

    return {
        "project_root": str(project.root),
        "rslc_tab": str(project.rslc_tab),
        "itab": str(project.itab),
        "diff_dir": str(project.diff_dir),
        "mli_dir": str(project.mli_dir),
        "dem_dir": str(project.dem_dir),
        "acquisition_count": len(project.acquisitions),
        "interferogram_count": len(project.interferograms),
        "first_acquisition": project.acquisitions[0].date,
        "last_acquisition": project.acquisitions[-1].date,
        "width": project.width,
        "length": project.length,
        "network_connected": project.network_connected,
        "missing_mli_count": len(missing_mli),
        "missing_mli_dates": missing_mli,
        "missing_mli_par_count": len(missing_mli_par),
        "missing_mli_par_dates": missing_mli_par,
        "missing_coherence_count": len(missing_coherence),
        "missing_coherence_pairs": missing_coherence,
        "first_pairs": [
            {
                "index": interferogram.index,
                "master": interferogram.master_date,
                "slave": interferogram.slave_date,
                "diff": interferogram.diff.name,
                "base": interferogram.base.name,
                "diff_par": (
                    interferogram.diff_par.name
                    if interferogram.diff_par is not None
                    else None
                ),
                "off": (
                    interferogram.off.name
                    if interferogram.off is not None
                    else None
                ),
                "coherence": (
                    interferogram.coherence.name
                    if interferogram.coherence is not None
                    else None
                ),
            }
            for interferogram in project.interferograms[:10]
        ],
    }
