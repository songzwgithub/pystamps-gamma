from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml


_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{6})(?!\d)"
)


class ProjectPathError(RuntimeError):
    pass


@dataclass(slots=True)
class ProjectPathsConfig:
    work_dir: str | None = None
    data_dir: str | None = None

    rslc_dir: str | None = None
    diff_dir: str | None = None
    mli_dir: str | None = None
    dem_dir: str | None = None

    rslc_tab: str | None = None
    itab: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProjectPaths:
    work_dir: Path
    data_dir: Path

    rslc_dir: Path | None
    diff_dir: Path | None
    mli_dir: Path | None
    dem_dir: Path | None

    rslc_tab: Path | None
    itab: Path | None

    sources: dict[str, str]


def _load_raw_config(
    path: str | Path | None,
) -> dict[str, Any]:

    if path is None:
        return {}

    p = Path(
        path
    ).expanduser().resolve()

    if not p.is_file():
        raise ProjectPathError(
            f"配置文件不存在：{p}"
        )

    text = p.read_text(
        encoding="utf-8"
    )

    if p.suffix.lower() in {
        ".yaml",
        ".yml",
    }:
        raw = yaml.safe_load(
            text
        ) or {}

    elif p.suffix.lower() == ".json":
        raw = json.loads(
            text
        )

    else:
        raise ProjectPathError(
            "project paths配置仅支持"
            "YAML/YML/JSON"
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise ProjectPathError(
            "主配置文件顶层必须是object"
        )

    return raw


def load_project_paths_config(
    path: str | Path | None,
) -> ProjectPathsConfig:

    raw = _load_raw_config(
        path
    )

    payload = raw.get(
        "paths",
        {},
    )

    if payload is None:
        payload = {}

    if not isinstance(
        payload,
        dict,
    ):
        raise ProjectPathError(
            "'paths'必须是object"
        )

    allowed = {
        "work_dir",
        "data_dir",
        "rslc_dir",
        "diff_dir",
        "mli_dir",
        "dem_dir",
        "rslc_tab",
        "itab",
    }

    unknown = (
        set(payload)
        - allowed
    )

    if unknown:
        raise ProjectPathError(
            "paths包含未知字段："
            + ", ".join(
                sorted(unknown)
            )
        )

    return ProjectPathsConfig(
        **payload
    )


def _resolve_path(
    raw: str | Path,
    *,
    base: Path,
) -> Path:

    p = Path(
        raw
    ).expanduser()

    if not p.is_absolute():
        p = (
            base
            / p
        )

    return p.resolve()


def _existing_directory(
    raw: str | Path | None,
    *,
    base: Path,
    name: str,
) -> Path | None:

    if raw in {
        None,
        "",
    }:
        return None

    p = _resolve_path(
        raw,
        base=base,
    )

    if not p.is_dir():
        raise ProjectPathError(
            f"{name}目录不存在：{p}"
        )

    return p


def _existing_file(
    raw: str | Path | None,
    *,
    base: Path,
    name: str,
) -> Path | None:

    if raw in {
        None,
        "",
    }:
        return None

    p = _resolve_path(
        raw,
        base=base,
    )

    if not p.is_file():
        raise ProjectPathError(
            f"{name}文件不存在：{p}"
        )

    return p


def _unique_paths(
    values: list[Path],
) -> list[Path]:

    result: list[Path] = []

    seen: set[Path] = set()

    for p in values:

        q = p.resolve()

        if q in seen:
            continue

        seen.add(q)
        result.append(q)

    return result


def _discover_rslc_tab(
    data_dir: Path,
) -> Path | None:

    direct = (
        data_dir
        / "RSLC_tab"
    )

    # Classical layout:
    # project/RSLC_tab
    if direct.is_file():
        return direct.resolve()

    candidates: list[Path] = []

    for name in (
        "RSLC_tab.txt",
        "rslc_tab",
        "rslc_tab.txt",
    ):
        p = (
            data_dir
            / name
        )

        if p.is_file():
            candidates.append(p)

    # Also support:
    # project/RSLC_tab/RSLC_tab
    if direct.is_dir():

        for name in (
            "RSLC_tab",
            "rslc_tab",
            "RSLC_tab.txt",
            "rslc_tab.txt",
        ):
            p = (
                direct
                / name
            )

            if p.is_file():
                return p.resolve()

        fuzzy = [
            p
            for p
            in direct.iterdir()
            if (
                p.is_file()
                and "rslc"
                in p.name.lower()
                and "tab"
                in p.name.lower()
            )
        ]

        if len(fuzzy) == 1:
            return fuzzy[
                0
            ].resolve()

        candidates.extend(
            fuzzy
        )

    candidates = _unique_paths(
        candidates
    )

    if len(candidates) == 1:
        return candidates[
            0
        ]

    if len(candidates) > 1:
        raise ProjectPathError(
            "发现多个RSLC_tab候选：\n  "
            + "\n  ".join(
                str(p)
                for p in candidates
            )
            + "\n请显式设置 paths.rslc_tab"
        )

    return None


def _discover_itab(
    data_dir: Path,
) -> Path | None:

    preferred = [
        data_dir
        / "itab",

        data_dir
        / "itab.txt",

        data_dir
        / "RSLC_tab"
        / "itab",

        data_dir
        / "RSLC_tab"
        / "itab.txt",
    ]

    for p in preferred:
        if p.is_file():
            return p.resolve()

    candidates: list[Path] = []

    tab_dir = (
        data_dir
        / "RSLC_tab"
    )

    if tab_dir.is_dir():

        candidates.extend(
            [
                p
                for p
                in tab_dir.iterdir()
                if (
                    p.is_file()
                    and p.name.lower().startswith(
                        "itab"
                    )
                )
            ]
        )

    candidates = _unique_paths(
        candidates
    )

    if len(candidates) == 1:
        return candidates[
            0
        ]

    if len(candidates) > 1:
        raise ProjectPathError(
            "发现多个itab候选：\n  "
            + "\n  ".join(
                str(p)
                for p in candidates
            )
            + "\n请显式设置 paths.itab"
        )

    return None


def _read_rslc_tab_tokens(
    rslc_tab: Path | None,
) -> tuple[
    list[str],
    list[str],
]:

    if (
        rslc_tab is None
        or not rslc_tab.is_file()
    ):
        return (
            [],
            [],
        )

    tokens: list[str] = []
    dates: list[str] = []

    for raw in rslc_tab.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        line = raw.strip()

        if (
            not line
            or line.startswith(
                (
                    "#",
                    "%",
                    ";",
                )
            )
        ):
            continue

        fields = line.split()

        if not fields:
            continue

        token = fields[0]

        tokens.append(
            token
        )

        matches = _DATE_RE.findall(
            Path(token).name
        )

        if not matches:
            matches = _DATE_RE.findall(
                token
            )

        if matches:
            dates.append(
                matches[-1]
            )

    return (
        tokens,
        dates,
    )


def _read_itab_pairs(
    itab: Path | None,
    dates: list[str],
) -> list[
    tuple[str, str]
]:

    if (
        itab is None
        or not itab.is_file()
        or not dates
    ):
        return []

    pairs: list[
        tuple[str, str]
    ] = []

    for raw in itab.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        line = raw.strip()

        if (
            not line
            or line.startswith(
                (
                    "#",
                    "%",
                    ";",
                )
            )
        ):
            continue

        fields = line.split()

        if len(fields) < 2:
            continue

        try:
            a = int(
                fields[0]
            )
            b = int(
                fields[1]
            )
        except ValueError:
            continue

        if (
            1 <= a <= len(dates)
            and 1 <= b <= len(dates)
        ):
            pairs.append(
                (
                    dates[a - 1],
                    dates[b - 1],
                )
            )

    return pairs


def _ancestor_named(
    path: Path,
    names: tuple[str, ...],
) -> Path | None:

    lookup = {
        name.lower()
        for name in names
    }

    for p in (
        path,
        *path.parents,
    ):
        if (
            p.name.lower()
            in lookup
        ):
            return p

    return None


def _rslc_referenced_directory(
    *,
    rslc_tab: Path | None,
    tokens: list[str],
) -> Path | None:

    if rslc_tab is None:
        return None

    referenced: list[Path] = []

    for token in tokens:

        p = Path(
            token
        ).expanduser()

        if not p.is_absolute():
            p = (
                rslc_tab.parent
                / p
            )

        p = p.resolve()

        ancestor = _ancestor_named(
            p.parent,
            (
                "RSLC",
                "RSLC_cropped",
            ),
        )

        if (
            ancestor is not None
            and ancestor.is_dir()
        ):
            referenced.append(
                ancestor.resolve()
            )

    referenced = _unique_paths(
        referenced
    )

    if len(referenced) == 1:
        return referenced[
            0
        ]

    return None


def _directory_file_names(
    directory: Path,
    *,
    recursive: bool,
) -> set[str]:

    iterator = (
        directory.rglob("*")
        if recursive
        else directory.iterdir()
    )

    return {
        p.name
        for p in iterator
        if p.is_file()
    }


def _rslc_score(
    directory: Path,
    *,
    tokens: list[str],
    dates: list[str],
) -> int:

    try:
        names = _directory_file_names(
            directory,
            recursive=True,
        )
    except OSError:
        return 0

    score = 0

    token_names = {
        Path(token).name
        for token in tokens
    }

    for name in token_names:
        if name in names:
            score += 10

        if (
            f"{name}.par"
            in names
        ):
            score += 4

    for date in dates:
        if any(
            (
                date in name
                and (
                    name.lower().endswith(
                        ".rslc"
                    )
                    or ".rslc."
                    in name.lower()
                )
            )
            for name in names
        ):
            score += 1

    return score


def _diff_score(
    directory: Path,
    *,
    pairs: list[
        tuple[str, str]
    ],
) -> int:

    try:
        names = _directory_file_names(
            directory,
            recursive=False,
        )
    except OSError:
        return 0

    score = 0

    for master, slave in pairs:

        stem = (
            f"{master}_{slave}"
        )

        if (
            f"{stem}.diff"
            in names
        ):
            score += 10

        if (
            f"{stem}.base"
            in names
        ):
            score += 4

        if (
            f"{stem}.diff_par"
            in names
            or f"{stem}.off"
            in names
        ):
            score += 2

    return score


def _mli_score(
    directory: Path,
    *,
    dates: list[str],
) -> int:

    try:
        names = _directory_file_names(
            directory,
            recursive=False,
        )
    except OSError:
        return 0

    score = 0

    for date in dates:

        if any(
            (
                date in name
                and name.lower().endswith(
                    (
                        ".mli",
                        ".rmli",
                    )
                )
            )
            for name in names
        ):
            score += 1

    return score


def _choose_scored_directory(
    *,
    candidates: list[Path],
    scorer,
    label: str,
    strict: bool,
) -> Path | None:

    candidates = [
        p.resolve()
        for p in candidates
        if p.is_dir()
    ]

    candidates = _unique_paths(
        candidates
    )

    if not candidates:

        if strict:
            raise ProjectPathError(
                f"未找到{label}目录"
            )

        return None

    if len(candidates) == 1:
        return candidates[
            0
        ]

    scored = [
        (
            int(
                scorer(p)
            ),
            p,
        )
        for p in candidates
    ]

    best_score = max(
        score
        for score, _
        in scored
    )

    best = [
        p
        for score, p
        in scored
        if score == best_score
    ]

    if len(best) == 1:
        return best[
            0
        ]

    raise ProjectPathError(
        f"{label}存在多个等价候选，"
        "无法安全自动判断：\n  "
        + "\n  ".join(
            (
                f"{p} "
                f"(score={score})"
            )
            for score, p
            in scored
        )
        + f"\n请显式设置 paths.{label.lower()}_dir"
    )


def discover_gamma_inputs(
    data_dir: str | Path,
    *,
    config: ProjectPathsConfig | None = None,
    strict: bool = False,
) -> ResolvedProjectPaths:

    data = Path(
        data_dir
    ).expanduser().resolve()

    if not data.is_dir():

        if strict:
            raise ProjectPathError(
                f"数据目录不存在：{data}"
            )

        # Preserve canonical path even before creation.
        return ResolvedProjectPaths(
            work_dir=data,
            data_dir=data,
            rslc_dir=None,
            diff_dir=None,
            mli_dir=None,
            dem_dir=None,
            rslc_tab=None,
            itab=None,
            sources={
                "data_dir":
                    "specified_missing",
            },
        )

    cfg = (
        config
        if config is not None
        else ProjectPathsConfig()
    )

    sources: dict[
        str,
        str
    ] = {}


    # ==========================================================
    # RSLC_tab / itab
    # ==========================================================

    if cfg.rslc_tab:

        rslc_tab = _existing_file(
            cfg.rslc_tab,
            base=data,
            name="RSLC_tab",
        )

        sources[
            "rslc_tab"
        ] = "config"

    else:

        rslc_tab = _discover_rslc_tab(
            data
        )

        if rslc_tab is not None:
            sources[
                "rslc_tab"
            ] = "auto"

    if cfg.itab:

        itab = _existing_file(
            cfg.itab,
            base=data,
            name="itab",
        )

        sources[
            "itab"
        ] = "config"

    else:

        itab = _discover_itab(
            data
        )

        if itab is not None:
            sources[
                "itab"
            ] = "auto"

    if (
        strict
        and rslc_tab is None
    ):
        raise ProjectPathError(
            f"在{data}中未找到RSLC_tab"
        )

    if (
        strict
        and itab is None
    ):
        raise ProjectPathError(
            f"在{data}中未找到itab"
        )


    tokens, dates = (
        _read_rslc_tab_tokens(
            rslc_tab
        )
    )

    pairs = _read_itab_pairs(
        itab,
        dates,
    )


    # ==========================================================
    # RSLC / RSLC_cropped
    # ==========================================================

    if cfg.rslc_dir:

        rslc_dir = _existing_directory(
            cfg.rslc_dir,
            base=data,
            name="RSLC",
        )

        sources[
            "rslc_dir"
        ] = "config"

    else:

        referenced = (
            _rslc_referenced_directory(
                rslc_tab=rslc_tab,
                tokens=tokens,
            )
        )

        if referenced is not None:

            rslc_dir = referenced

            sources[
                "rslc_dir"
            ] = "RSLC_tab"

        else:

            candidates = [
                data
                / "RSLC_cropped",

                data
                / "RSLC",
            ]

            rslc_dir = (
                _choose_scored_directory(
                    candidates=candidates,
                    scorer=lambda p:
                        _rslc_score(
                            p,
                            tokens=tokens,
                            dates=dates,
                        ),
                    label="RSLC",
                    strict=strict,
                )
            )

            if rslc_dir is not None:
                sources[
                    "rslc_dir"
                ] = "auto"


    # ==========================================================
    # DIFF / DIFF_dir
    # ==========================================================

    if cfg.diff_dir:

        diff_dir = _existing_directory(
            cfg.diff_dir,
            base=data,
            name="DIFF",
        )

        sources[
            "diff_dir"
        ] = "config"

    else:

        candidates = [
            data
            / "DIFF",

            data
            / "DIFF_dir",
        ]

        diff_dir = (
            _choose_scored_directory(
                candidates=candidates,
                scorer=lambda p:
                    _diff_score(
                        p,
                        pairs=pairs,
                    ),
                label="DIFF",
                strict=strict,
            )
        )

        if diff_dir is not None:
            sources[
                "diff_dir"
            ] = "auto"


    # ==========================================================
    # MLI_dir / MLI
    # ==========================================================

    if cfg.mli_dir:

        mli_dir = _existing_directory(
            cfg.mli_dir,
            base=data,
            name="MLI",
        )

        sources[
            "mli_dir"
        ] = "config"

    else:

        candidates = [
            data
            / "MLI_dir",

            data
            / "MLI",
        ]

        mli_dir = (
            _choose_scored_directory(
                candidates=candidates,
                scorer=lambda p:
                    _mli_score(
                        p,
                        dates=dates,
                    ),
                label="MLI",
                strict=strict,
            )
        )

        if mli_dir is not None:
            sources[
                "mli_dir"
            ] = "auto"


    # ==========================================================
    # DEM_prep / DEM
    # ==========================================================

    if cfg.dem_dir:

        dem_dir = _existing_directory(
            cfg.dem_dir,
            base=data,
            name="DEM",
        )

        sources[
            "dem_dir"
        ] = "config"

    else:

        dem_candidates = [
            data
            / "DEM_prep",

            data
            / "DEM",
        ]

        existing_dem = [
            p.resolve()
            for p in dem_candidates
            if p.is_dir()
        ]

        if existing_dem:

            # DEM_prep is deliberately preferred because it is
            # already in GAMMA processing geometry.
            dem_dir = existing_dem[
                0
            ]

            sources[
                "dem_dir"
            ] = "auto"

        elif strict:

            raise ProjectPathError(
                f"未找到DEM_prep或DEM目录：{data}"
            )

        else:
            dem_dir = None


    return ResolvedProjectPaths(
        work_dir=data,
        data_dir=data,
        rslc_dir=rslc_dir,
        diff_dir=diff_dir,
        mli_dir=mli_dir,
        dem_dir=dem_dir,
        rslc_tab=rslc_tab,
        itab=itab,
        sources=sources,
    )


def resolve_project_paths(
    *,
    config_path: str | Path | None = None,
    cli_work_dir: str | Path | None = None,
    cli_data_dir: str | Path | None = None,
    cwd: str | Path | None = None,
    strict_gamma: bool = False,
) -> ResolvedProjectPaths:

    cfg = load_project_paths_config(
        config_path
    )

    current = (
        Path(cwd).expanduser().resolve()
        if cwd is not None
        else Path.cwd().resolve()
    )

    sources: dict[
        str,
        str
    ] = {}


    # ==========================================================
    # work_dir
    # ==========================================================

    if cli_work_dir:

        work = _resolve_path(
            cli_work_dir,
            base=current,
        )

        sources[
            "work_dir"
        ] = "CLI"

    elif cfg.work_dir:

        work = _resolve_path(
            cfg.work_dir,
            base=current,
        )

        sources[
            "work_dir"
        ] = "config"

    else:

        work = current

        sources[
            "work_dir"
        ] = "cwd"


    # ==========================================================
    # data_dir
    # ==========================================================

    if cli_data_dir:

        data = _resolve_path(
            cli_data_dir,
            base=work,
        )

        sources[
            "data_dir"
        ] = "CLI"

    elif cfg.data_dir:

        data = _resolve_path(
            cfg.data_dir,
            base=work,
        )

        sources[
            "data_dir"
        ] = "config"

    else:

        data = work.parent.resolve()

        sources[
            "data_dir"
        ] = "work_dir.parent"


    gamma = discover_gamma_inputs(
        data,
        config=cfg,
        strict=strict_gamma,
    )

    merged_sources = dict(
        gamma.sources
    )

    merged_sources.update(
        sources
    )

    return ResolvedProjectPaths(
        work_dir=work,
        data_dir=data,
        rslc_dir=gamma.rslc_dir,
        diff_dir=gamma.diff_dir,
        mli_dir=gamma.mli_dir,
        dem_dir=gamma.dem_dir,
        rslc_tab=gamma.rslc_tab,
        itab=gamma.itab,
        sources=merged_sources,
    )


def export_project_paths(
    resolved: ResolvedProjectPaths,
) -> None:

    values = {
        "PYSTAMPS_WORK_DIR":
            resolved.work_dir,

        "PYSTAMPS_DATA_DIR":
            resolved.data_dir,

        "PYSTAMPS_RSLC_DIR":
            resolved.rslc_dir,

        "PYSTAMPS_DIFF_DIR":
            resolved.diff_dir,

        "PYSTAMPS_MLI_DIR":
            resolved.mli_dir,

        "PYSTAMPS_DEM_DIR":
            resolved.dem_dir,

        "PYSTAMPS_RSLC_TAB":
            resolved.rslc_tab,

        "PYSTAMPS_ITAB":
            resolved.itab,
    }

    for key, value in values.items():

        if value is None:
            os.environ.pop(
                key,
                None,
            )
        else:
            os.environ[
                key
            ] = str(
                value
            )


def print_project_paths(
    resolved: ResolvedProjectPaths,
) -> None:

    def fmt(
        value: Path | None,
    ) -> str:
        return (
            str(value)
            if value is not None
            else "<not found>"
        )

    print()
    print(
        "============================================================"
    )
    print(
        "pySTAMPS PROJECT PATHS"
    )
    print(
        "============================================================"
    )

    print(
        "work_dir :",
        resolved.work_dir,
        f"[{resolved.sources.get('work_dir', '?')}]",
    )

    print(
        "data_dir :",
        resolved.data_dir,
        f"[{resolved.sources.get('data_dir', '?')}]",
    )

    print()
    print(
        "GAMMA INPUTS"
    )

    print(
        "RSLC     :",
        fmt(
            resolved.rslc_dir
        ),
    )

    print(
        "DIFF     :",
        fmt(
            resolved.diff_dir
        ),
    )

    print(
        "MLI      :",
        fmt(
            resolved.mli_dir
        ),
    )

    print(
        "DEM      :",
        fmt(
            resolved.dem_dir
        ),
    )

    print(
        "RSLC_tab :",
        fmt(
            resolved.rslc_tab
        ),
    )

    print(
        "itab     :",
        fmt(
            resolved.itab
        ),
    )

    print(
        "============================================================"
    )
    print()
