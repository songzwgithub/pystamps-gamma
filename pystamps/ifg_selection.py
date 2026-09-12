from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat, read_mat_variables, write_mat


class IFGSelectionError(RuntimeError):
    """Raised when automatic IFG quality selection cannot continue."""


@dataclass(slots=True)
class IFGSelectionResult:
    mode: str
    method: str
    n_ifg: int
    drop_ifg_index: tuple[int, ...]
    changed: bool
    candidate_count: int
    protected_count: int


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _drop_indices(value: Any, n_ifg: int) -> tuple[int, ...]:
    if value is None:
        return ()

    arr = np.asarray(value).reshape(-1)

    if arr.size == 0:
        return ()

    out = np.rint(arr).astype(np.int64)

    out = np.unique(
        out[
            (out >= 1)
            & (out <= n_ifg)
        ]
    )

    return tuple(
        int(v)
        for v in out
    )


def _robust_high_z(values: np.ndarray) -> np.ndarray:
    """
    Positive robust z-score for high-side anomalies.

    median + 1.4826*MAD is used. When the MAD collapses to zero,
    a small scale based on the finite data range is used rather than
    suppressing a genuine isolated outlier.
    """
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    out = np.zeros(
        x.shape,
        dtype=np.float64,
    )

    finite = np.isfinite(x)

    if not np.any(finite):
        out[:] = np.inf
        return out

    xf = x[finite]

    med = float(
        np.median(xf)
    )

    mad = float(
        np.median(
            np.abs(
                xf - med
            )
        )
    )

    scale = (
        1.4826
        * mad
    )

    if (
        not np.isfinite(scale)
        or scale <= 1e-12
    ):
        spread = float(
            np.ptp(xf)
        )

        if spread <= 1e-12:
            scale = 1.0
        else:
            scale = max(
                spread
                / max(
                    10.0,
                    math.sqrt(
                        float(xf.size)
                    ),
                ),
                1e-12,
            )

    out[finite] = (
        x[finite]
        - med
    ) / scale

    out[finite] = np.maximum(
        out[finite],
        0.0,
    )

    out[~finite] = np.inf

    return out


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    result = np.ones(
        x.shape,
        dtype=np.float64,
    )

    finite_ix = np.flatnonzero(
        np.isfinite(x)
    )

    if finite_ix.size == 0:
        return result

    order = finite_ix[
        np.argsort(
            x[finite_ix],
            kind="stable",
        )
    ]

    if order.size == 1:
        result[order] = 0.5
        return result

    ranks = np.arange(
        1,
        order.size + 1,
        dtype=np.float64,
    )

    result[order] = (
        ranks
        / float(order.size)
    )

    return result


def _temporal_context_z(
    ifg_std: np.ndarray,
    temporal_days: np.ndarray,
    bins: int,
) -> np.ndarray:
    """
    Compare IFG noise only with IFGs having similar temporal baselines.
    """
    values = np.asarray(
        ifg_std,
        dtype=np.float64,
    )

    dt = np.asarray(
        temporal_days,
        dtype=np.float64,
    )

    global_z = _robust_high_z(
        values
    )

    out = global_z.copy()

    finite = (
        np.isfinite(values)
        & np.isfinite(dt)
    )

    if np.count_nonzero(finite) < 16:
        return out

    q = np.linspace(
        0.0,
        1.0,
        max(
            2,
            int(bins) + 1,
        ),
    )

    edges = np.unique(
        np.quantile(
            dt[finite],
            q,
        )
    )

    if edges.size < 3:
        return out

    for i in range(
        edges.size - 1
    ):
        lo = edges[i]
        hi = edges[i + 1]

        if i == edges.size - 2:
            mask = (
                finite
                & (dt >= lo)
                & (dt <= hi)
            )
        else:
            mask = (
                finite
                & (dt >= lo)
                & (dt < hi)
            )

        ix = np.flatnonzero(
            mask
        )

        if ix.size < 8:
            continue

        out[ix] = _robust_high_z(
            values[ix]
        )

    return out


def _endpoint_context_z(
    ifg_std: np.ndarray,
    ifgday_ix: np.ndarray,
) -> np.ndarray:
    """
    Compare each IFG with other IFGs sharing either acquisition endpoint.
    """
    values = np.asarray(
        ifg_std,
        dtype=np.float64,
    )

    edges = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    global_z = _robust_high_z(
        values
    )

    out = global_z.copy()

    incident: dict[int, list[int]] = {}

    for edge_ix, (
        a,
        b,
    ) in enumerate(edges):
        incident.setdefault(
            int(a),
            [],
        ).append(edge_ix)

        incident.setdefault(
            int(b),
            [],
        ).append(edge_ix)

    for j, (
        a,
        b,
    ) in enumerate(edges):
        peers = sorted(
            set(
                incident.get(
                    int(a),
                    [],
                )
                + incident.get(
                    int(b),
                    [],
                )
            )
            - {j}
        )

        if len(peers) < 6:
            continue

        peer_values = values[
            np.asarray(
                peers,
                dtype=np.int64,
            )
        ]

        finite = np.isfinite(
            peer_values
        )

        if np.count_nonzero(
            finite
        ) < 6:
            continue

        ref = peer_values[
            finite
        ]

        med = float(
            np.median(ref)
        )

        mad = float(
            np.median(
                np.abs(
                    ref - med
                )
            )
        )

        scale = (
            1.4826
            * mad
        )

        if (
            not np.isfinite(scale)
            or scale <= 1e-12
        ):
            scale = max(
                float(
                    np.std(ref)
                ),
                1e-6,
            )

        if np.isfinite(
            values[j]
        ):
            out[j] = max(
                0.0,
                (
                    values[j]
                    - med
                )
                / scale,
            )
        else:
            out[j] = np.inf

    return out


def _active_nodes(
    edges: np.ndarray,
    keep: np.ndarray,
) -> set[int]:
    if not np.any(keep):
        return set()

    selected = edges[
        keep,
        :
    ]

    return set(
        int(v)
        for v
        in selected.reshape(-1)
    )


def _network_connected(
    edges: np.ndarray,
    keep: np.ndarray,
    original_nodes: set[int],
) -> bool:
    selected = edges[
        keep,
        :
    ]

    if selected.size == 0:
        return False

    selected_nodes = _active_nodes(
        edges,
        keep,
    )

    if (
        selected_nodes
        != original_nodes
    ):
        return False

    adjacency: dict[int, set[int]] = {
        node: set()
        for node
        in original_nodes
    }

    for a, b in selected:
        a = int(a)
        b = int(b)

        adjacency[a].add(b)
        adjacency[b].add(a)

    start = next(
        iter(
            original_nodes
        )
    )

    seen = {
        start
    }

    stack = [
        start
    ]

    while stack:
        node = stack.pop()

        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue

            seen.add(
                neighbor
            )
            stack.append(
                neighbor
            )

    return (
        seen
        == original_nodes
    )


def _matlab_date_string(
    value: float,
) -> str:
    if not np.isfinite(value):
        return ""

    try:
        whole = int(
            math.floor(value)
        )

        fraction = (
            float(value)
            - whole
        )

        dt = (
            datetime.fromordinal(
                whole
            )
            + timedelta(
                days=fraction
            )
            - timedelta(
                days=366
            )
        )

        return dt.strftime(
            "%Y-%m-%d"
        )

    except Exception:
        return ""


def _find_network_source(
    root: Path,
) -> Path:
    candidates = [
        root / "ps2.mat",
    ]

    candidates.extend(
        sorted(
            root.glob(
                "PATCH_*/ps1.mat"
            )
        )
    )

    for path in candidates:
        if not path.exists():
            continue

        try:
            payload = read_mat_variables(
                path,
                (
                    "ifgday_ix",
                    "day",
                    "n_ifg",
                ),
            )
        except Exception:
            continue

        if (
            payload.get(
                "ifgday_ix"
            )
            is not None
            and np.asarray(
                payload.get(
                    "ifgday_ix"
                )
            ).size
        ):
            return path

    raise IFGSelectionError(
        "Cannot locate SBAS ifgday_ix in "
        "ps2.mat or PATCH_*/ps1.mat"
    )


def _load_network(
    root: Path,
    n_ifg: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Path,
]:
    source = _find_network_source(
        root
    )

    payload = read_mat_variables(
        source,
        (
            "ifgday_ix",
            "day",
            "n_ifg",
        ),
    )

    edges = np.asarray(
        payload.get(
            "ifgday_ix"
        ),
        dtype=np.float64,
    )

    edges = np.squeeze(
        edges
    )

    if (
        edges.ndim == 2
        and edges.shape[0] == 2
        and edges.shape[1] == n_ifg
    ):
        edges = edges.T

    if edges.shape != (
        n_ifg,
        2,
    ):
        raise IFGSelectionError(
            f"{source}: ifgday_ix shape "
            f"{edges.shape}; expected "
            f"({n_ifg}, 2)"
        )

    edges = np.rint(
        edges
    ).astype(
        np.int64
    )

    day = np.asarray(
        payload.get(
            "day"
        ),
        dtype=np.float64,
    ).reshape(-1)

    if day.size == 0:
        root_ps = read_mat_variables(
            root / "ps2.mat",
            ("day",),
        )

        day = np.asarray(
            root_ps.get(
                "day"
            ),
            dtype=np.float64,
        ).reshape(-1)

    if (
        np.min(edges) < 1
        or np.max(edges) > day.size
    ):
        raise IFGSelectionError(
            "ifgday_ix contains acquisition "
            "indices outside day vector"
        )

    return (
        edges,
        day,
        source,
    )


def _write_audit(
    root: Path,
    *,
    mode: str,
    method: str,
    ifg_std: np.ndarray,
    temporal_days: np.ndarray,
    ifgday_ix: np.ndarray,
    day: np.ndarray,
    global_z: np.ndarray,
    temporal_z: np.ndarray,
    endpoint_z: np.ndarray,
    percentile: np.ndarray,
    score: np.ndarray,
    evidence_count: np.ndarray,
    candidate: np.ndarray,
    dropped: set[int],
    protected: set[int],
    old_drop: tuple[int, ...],
    config: Any,
    network_source: Path,
) -> None:
    csv_path = (
        root
        / "ifg_quality_audit.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "ifg_index",
                "date1",
                "date2",
                "acq1_index",
                "acq2_index",
                "temporal_baseline_days",
                "ifg_std_deg",
                "z_global",
                "z_temporal_context",
                "z_endpoint_context",
                "percentile",
                "quality_score",
                "bad_metric_count",
                "candidate_bad",
                "decision",
            ]
        )

        for j in range(
            ifg_std.size
        ):
            index = j + 1

            a = int(
                ifgday_ix[j, 0]
            )

            b = int(
                ifgday_ix[j, 1]
            )

            if index in dropped:
                decision = "drop"
            elif index in protected:
                decision = (
                    "protected_network"
                )
            elif candidate[j]:
                decision = (
                    "candidate_not_dropped"
                )
            else:
                decision = "keep"

            writer.writerow(
                [
                    index,
                    _matlab_date_string(
                        day[a - 1]
                    ),
                    _matlab_date_string(
                        day[b - 1]
                    ),
                    a,
                    b,
                    float(
                        temporal_days[j]
                    ),
                    float(
                        ifg_std[j]
                    ),
                    float(
                        global_z[j]
                    ),
                    float(
                        temporal_z[j]
                    ),
                    float(
                        endpoint_z[j]
                    ),
                    float(
                        percentile[j]
                    ),
                    float(
                        score[j]
                    ),
                    int(
                        evidence_count[j]
                    ),
                    int(
                        candidate[j]
                    ),
                    decision,
                ]
            )

    payload = {
        "mode": mode,
        "method": method,
        "network_source": str(
            network_source
        ),
        "n_ifg_input": int(
            ifg_std.size
        ),
        "n_ifg_selected": int(
            ifg_std.size
            - len(dropped)
        ),
        "n_ifg_dropped": int(
            len(dropped)
        ),
        "candidate_count": int(
            np.count_nonzero(
                candidate
            )
        ),
        "protected_network_count": int(
            len(protected)
        ),
        "previous_drop_ifg_index": [
            int(v)
            for v
            in old_drop
        ],
        "effective_drop_ifg_index": sorted(
            int(v)
            for v
            in dropped
        ),
        "protected_network_ifg_index": sorted(
            int(v)
            for v
            in protected
        ),
        "settings": {
            "robust_z_threshold": float(
                getattr(
                    config,
                    "robust_z_threshold",
                    4.0,
                )
            ),
            "contextual_z_threshold": float(
                getattr(
                    config,
                    "contextual_z_threshold",
                    3.5,
                )
            ),
            "tail_quantile": float(
                getattr(
                    config,
                    "tail_quantile",
                    0.99,
                )
            ),
            "min_bad_metrics": int(
                getattr(
                    config,
                    "min_bad_metrics",
                    2,
                )
            ),
            "max_drop_fraction": float(
                getattr(
                    config,
                    "max_drop_fraction",
                    0.05,
                )
            ),
            "preserve_network": bool(
                getattr(
                    config,
                    "preserve_network",
                    True,
                )
            ),
            "temporal_bins": int(
                getattr(
                    config,
                    "temporal_bins",
                    8,
                )
            ),
        },
    }

    json_path = (
        root
        / "ifg_selection.json"
    )

    tmp = json_path.with_suffix(
        ".json.tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(
        json_path
    )


def _manual_selection(
    *,
    n_ifg: int,
    requested: tuple[int, ...],
    edges: np.ndarray,
    preserve_network: bool,
) -> tuple[
    set[int],
    set[int],
]:
    drops = set(
        int(v)
        for v
        in requested
        if 1 <= int(v) <= n_ifg
    )

    protected: set[int] = set()

    if (
        preserve_network
        and drops
    ):
        keep = np.ones(
            n_ifg,
            dtype=bool,
        )

        for index in drops:
            keep[
                index - 1
            ] = False

        original_nodes = _active_nodes(
            edges,
            np.ones(
                n_ifg,
                dtype=bool,
            ),
        )

        if not _network_connected(
            edges,
            keep,
            original_nodes,
        ):
            raise IFGSelectionError(
                "Manual drop_ifg_index would "
                "disconnect the SBAS network"
            )

    return (
        drops,
        protected,
    )


def _resolve_ifg_selection_pre_qc_internal(
    dataset_root: Path,
    config: Any,
) -> IFGSelectionResult:
    root = Path(
        dataset_root
    ).expanduser().resolve()

    mode = str(
        getattr(
            config,
            "mode",
            "auto",
        )
    ).strip().lower()

    if mode not in {
        "auto",
        "manual",
        "none",
    }:
        raise IFGSelectionError(
            "ifg_selection.mode must be "
            "auto, manual or none"
        )

    ifgstd_path = (
        root
        / "ifgstd2.mat"
    )

    if not ifgstd_path.exists():
        raise IFGSelectionError(
            f"Missing {ifgstd_path}"
        )

    ifgstd = read_mat(
        ifgstd_path
    )

    if (
        "ifg_std"
        not in ifgstd
    ):
        raise IFGSelectionError(
            "ifgstd2.mat does not contain "
            "StaMPS variable 'ifg_std'"
        )

    ifg_std = np.asarray(
        ifgstd[
            "ifg_std"
        ],
        dtype=np.float64,
    ).reshape(-1)

    n_ifg = int(
        ifg_std.size
    )

    if n_ifg < 1:
        raise IFGSelectionError(
            "ifg_std is empty"
        )

    edges, day, network_source = (
        _load_network(
            root,
            n_ifg,
        )
    )

    temporal_days = np.abs(
        day[
            edges[:, 1] - 1
        ]
        - day[
            edges[:, 0] - 1
        ]
    ).astype(
        np.float64
    )

    parms_path = (
        root
        / "parms.mat"
    )

    parms = (
        read_mat(
            parms_path
        )
        if parms_path.exists()
        else {}
    )

    old_drop = _drop_indices(
        parms.get(
            "drop_ifg_index"
        ),
        n_ifg,
    )

    robust_z_threshold = float(
        getattr(
            config,
            "robust_z_threshold",
            4.0,
        )
    )

    contextual_z_threshold = float(
        getattr(
            config,
            "contextual_z_threshold",
            3.5,
        )
    )

    tail_quantile = float(
        getattr(
            config,
            "tail_quantile",
            0.99,
        )
    )

    min_bad_metrics = int(
        getattr(
            config,
            "min_bad_metrics",
            2,
        )
    )

    max_drop_fraction = float(
        getattr(
            config,
            "max_drop_fraction",
            0.05,
        )
    )

    preserve_network = bool(
        getattr(
            config,
            "preserve_network",
            True,
        )
    )

    temporal_bins = int(
        getattr(
            config,
            "temporal_bins",
            8,
        )
    )

    if not (
        0.0
        <= max_drop_fraction
        <= 0.5
    ):
        raise IFGSelectionError(
            "max_drop_fraction must be "
            "between 0 and 0.5"
        )

    if not (
        0.5
        <= tail_quantile
        < 1.0
    ):
        raise IFGSelectionError(
            "tail_quantile must be in "
            "[0.5, 1.0)"
        )

    global_z = _robust_high_z(
        ifg_std
    )

    temporal_z = _temporal_context_z(
        ifg_std,
        temporal_days,
        temporal_bins,
    )

    endpoint_z = _endpoint_context_z(
        ifg_std,
        edges,
    )

    percentile = _percentile_rank(
        ifg_std
    )

    bad_global = (
        global_z
        >= robust_z_threshold
    )

    bad_temporal = (
        temporal_z
        >= contextual_z_threshold
    )

    bad_endpoint = (
        endpoint_z
        >= contextual_z_threshold
    )

    bad_tail = (
        percentile
        >= tail_quantile
    )

    invalid = ~np.isfinite(
        ifg_std
    )

    evidence_count = (
        bad_global.astype(
            np.int16
        )
        + bad_temporal.astype(
            np.int16
        )
        + bad_endpoint.astype(
            np.int16
        )
        + bad_tail.astype(
            np.int16
        )
    )

    evidence_count[
        invalid
    ] = 4

    candidate = (
        invalid
        | (
            (
                evidence_count
                >= min_bad_metrics
            )
            & (
                global_z
                >= min(
                    robust_z_threshold,
                    3.0,
                )
            )
        )
    )

    score = (
        0.45
        * np.minimum(
            global_z,
            20.0,
        )
        + 0.25
        * np.minimum(
            temporal_z,
            20.0,
        )
        + 0.20
        * np.minimum(
            endpoint_z,
            20.0,
        )
        + 0.10
        * np.clip(
            (
                percentile
                - tail_quantile
            )
            / max(
                1e-6,
                1.0
                - tail_quantile,
            ),
            0.0,
            1.0,
        )
        * 10.0
    )

    score[
        invalid
    ] = np.inf

    original_keep = np.ones(
        n_ifg,
        dtype=bool,
    )

    original_nodes = _active_nodes(
        edges,
        original_keep,
    )

    if (
        preserve_network
        and not _network_connected(
            edges,
            original_keep,
            original_nodes,
        )
    ):
        raise IFGSelectionError(
            "Input SBAS network is already "
            "disconnected before IFG QC"
        )

    protected: set[int] = set()

    if mode == "none":
        drops: set[int] = set()
        method = (
            "none_no_ifg_filtering"
        )

    elif mode == "manual":
        requested = tuple(
            int(v)
            for v
            in getattr(
                config,
                "drop_ifg_index",
                (),
            )
        )

        drops, protected = (
            _manual_selection(
                n_ifg=n_ifg,
                requested=requested,
                edges=edges,
                preserve_network=(
                    preserve_network
                ),
            )
        )

        method = (
            "manual_drop_ifg_index"
        )

    else:
        method = (
            "robust_ifgstd_"
            "network_preserving_v1"
        )

        max_drop_count = int(
            math.floor(
                n_ifg
                * max_drop_fraction
            )
        )

        if (
            max_drop_fraction > 0
            and max_drop_count < 1
        ):
            max_drop_count = 1

        candidate_ix = np.flatnonzero(
            candidate
        )

        candidate_ix = candidate_ix[
            np.argsort(
                -score[
                    candidate_ix
                ],
                kind="stable",
            )
        ]

        keep = np.ones(
            n_ifg,
            dtype=bool,
        )

        drops = set()

        for zero_ix in candidate_ix:
            if (
                len(drops)
                >= max_drop_count
            ):
                break

            one_ix = int(
                zero_ix
                + 1
            )

            proposed = keep.copy()

            proposed[
                zero_ix
            ] = False

            if (
                preserve_network
                and not _network_connected(
                    edges,
                    proposed,
                    original_nodes,
                )
            ):
                protected.add(
                    one_ix
                )
                continue

            keep = proposed

            drops.add(
                one_ix
            )

    new_drop = tuple(
        sorted(
            drops
        )
    )

    changed = (
        tuple(
            sorted(
                old_drop
            )
        )
        != new_drop
    )

    parms[
        "drop_ifg_index"
    ] = np.asarray(
        new_drop,
        dtype=np.float64,
    ).reshape(
        -1,
        1,
    )

    # === PYSTAMPS_GRID_QC_CONFIG_TO_PARMS_V1 ===
    # GRID QC is active only for automatic IFG selection.
    _grid_qc_enabled = (
        mode == "auto"
        and bool(
            getattr(
                config,
                "grid_qc_enabled",
                True,
            )
        )
    )

    _grid_qc_values = {
        "pystamps_grid_qc_enabled":
            1.0 if _grid_qc_enabled else 0.0,
        "pystamps_grid_qc_metric_bad_quantile":
            float(getattr(config, "grid_qc_metric_bad_quantile", 0.90)),
        "pystamps_grid_qc_score_tail_fraction":
            float(getattr(config, "grid_qc_score_tail_fraction", 0.02)),
        "pystamps_grid_qc_score_z_threshold":
            float(getattr(config, "grid_qc_score_z_threshold", 1.5)),
        "pystamps_grid_qc_extreme_z_threshold":
            float(getattr(config, "grid_qc_extreme_z_threshold", 3.5)),
        "pystamps_grid_qc_min_bad_metrics":
            float(getattr(config, "grid_qc_min_bad_metrics", 2)),
        "pystamps_grid_qc_max_drop_fraction":
            float(getattr(config, "grid_qc_max_drop_fraction", 0.05)),
        "pystamps_grid_qc_preserve_network":
            1.0 if bool(getattr(config, "grid_qc_preserve_network", True)) else 0.0,
        "pystamps_grid_qc_sample_ps":
            float(getattr(config, "grid_qc_sample_ps", 20000)),
        "pystamps_grid_qc_sample_pairs":
            float(getattr(config, "grid_qc_sample_pairs", 30000)),
        "pystamps_grid_qc_closure_sample_ps":
            float(getattr(config, "grid_qc_closure_sample_ps", 6000)),
        "pystamps_grid_qc_max_triangles":
            float(getattr(config, "grid_qc_max_triangles", 4000)),
        "pystamps_grid_qc_node_context_enabled":
            1.0 if bool(
                getattr(
                    config,
                    "grid_qc_node_context_enabled",
                    True,
                )
            ) else 0.0,
        "pystamps_grid_qc_node_min_degree":
            float(
                getattr(
                    config,
                    "grid_qc_node_min_degree",
                    3,
                )
            ),
        "pystamps_grid_qc_node_candidate_fraction":
            float(
                getattr(
                    config,
                    "grid_qc_node_candidate_fraction",
                    0.50,
                )
            ),
        "pystamps_grid_qc_edge_excess_z_threshold":
            float(
                getattr(
                    config,
                    "grid_qc_edge_excess_z_threshold",
                    2.5,
                )
            ),
        "pystamps_grid_qc_clustered_edge_excess_z_threshold":
            float(
                getattr(
                    config,
                    "grid_qc_clustered_edge_excess_z_threshold",
                    3.5,
                )
            ),
        "pystamps_grid_qc_low_degree_score_z_threshold":
            float(
                getattr(
                    config,
                    "grid_qc_low_degree_score_z_threshold",
                    4.0,
                )
            ),

        # Numeric ownership flag only.
        #
        # Do NOT persist the literal string "auto" into the
        # numeric StaMPS parms compatibility path.
        "pystamps_final_qc_owns_drop":
            1.0 if (
                bool(
                    getattr(
                        config,
                        "final_ifg_qc_enabled",
                        False,
                    )
                )
                and str(
                    getattr(
                        config,
                        "mode",
                        "auto",
                    )
                ).strip().lower()
                == "auto"
            ) else 0.0,

        "pystamps_final_ifg_qc_enabled":
            1.0 if bool(
                getattr(
                    config,
                    "final_ifg_qc_enabled",
                    False,
                )
            ) else 0.0,

        "pystamps_final_qc_msd_strong_percentile":
            float(
                getattr(
                    config,
                    "final_qc_msd_strong_percentile",
                    0.975,
                )
            ),

        "pystamps_final_qc_msd_extreme_percentile":
            float(
                getattr(
                    config,
                    "final_qc_msd_extreme_percentile",
                    0.990,
                )
            ),

        "pystamps_final_qc_network_strong_percentile":
            float(
                getattr(
                    config,
                    "final_qc_network_strong_percentile",
                    0.975,
                )
            ),

        "pystamps_final_qc_network_extreme_percentile":
            float(
                getattr(
                    config,
                    "final_qc_network_extreme_percentile",
                    0.990,
                )
            ),

        "pystamps_final_qc_max_drop_fraction":
            float(
                getattr(
                    config,
                    "final_qc_max_drop_fraction",
                    0.05,
                )
            ),

        "pystamps_final_qc_preserve_network":
            1.0 if bool(
                getattr(
                    config,
                    "final_qc_preserve_network",
                    True,
                )
            ) else 0.0,

        "pystamps_final_qc_fail_on_cap":
            1.0 if bool(
                getattr(
                    config,
                    "final_qc_fail_on_cap",
                    True,
                )
            ) else 0.0,

        "pystamps_final_qc_chunk_ifg":
            float(
                getattr(
                    config,
                    "final_qc_chunk_ifg",
                    8,
                )
            ),
    }

    for _key, _value in _grid_qc_values.items():
        parms[_key] = np.asarray(
            _value,
            dtype=np.float64,
        )

    write_mat(
        parms_path,
        parms,
    )

    _write_audit(
        root,
        mode=mode,
        method=method,
        ifg_std=ifg_std,
        temporal_days=temporal_days,
        ifgday_ix=edges,
        day=day,
        global_z=global_z,
        temporal_z=temporal_z,
        endpoint_z=endpoint_z,
        percentile=percentile,
        score=score,
        evidence_count=evidence_count,
        candidate=candidate,
        dropped=drops,
        protected=protected,
        old_drop=old_drop,
        config=config,
        network_source=network_source,
    )

    print(
        "[IFG_PRE_QC] "
        f"{mode}: "
        f"input={n_ifg}, "
        f"candidates="
        f"{int(np.count_nonzero(candidate))}, "
        f"drop={len(new_drop)}, "
        f"keep={n_ifg-len(new_drop)}, "
        f"protected_network="
        f"{len(protected)}, "
        f"changed={changed}",
        flush=True,
    )

    if new_drop:
        print(
            "[IFG_PRE_QC] "
            "effective_drop_ifg_index="
            + ",".join(
                str(v)
                for v
                in new_drop
            ),
            flush=True,
        )
    else:
        print(
            "[IFG_PRE_QC] "
            "no IFGs rejected",
            flush=True,
        )

    return IFGSelectionResult(
        mode=mode,
        method=method,
        n_ifg=n_ifg,
        drop_ifg_index=new_drop,
        changed=changed,
        candidate_count=int(
            np.count_nonzero(
                candidate
            )
        ),
        protected_count=len(
            protected
        ),
    )



# === FINAL_IFG_QC_OWNS_DROP_V1 ===
def resolve_ifg_selection(
    *args,
    **kwargs,
):
    """
    Run PRE-QC diagnostics normally.

    If a completed FINAL-QC selection already exists, restore its
    effective drop list afterwards so PRE-QC can never become the
    authoritative production drop source.
    """

    result = (
        _resolve_ifg_selection_pre_qc_internal(
            *args,
            **kwargs,
        )
    )

    from pathlib import Path
    import json
    import numpy as np

    from pystamps.io.mat import (
        read_mat,
        write_mat,
    )

    # Find dataset root generically from call arguments.
    root = None

    for value in (
        list(args)
        + list(
            kwargs.values()
        )
    ):

        if not isinstance(
            value,
            (
                str,
                Path,
            ),
        ):
            continue

        try:
            candidate = Path(
                value
            ).expanduser().resolve()
        except Exception:
            continue

        if (
            candidate.is_dir()
            and (
                candidate
                / "parms.mat"
            ).exists()
        ):
            root = candidate
            break

    if root is None:
        return result

    selection = (
        root
        / "final_ifg_qc_selection.json"
    )

    if not selection.exists():
        return result

    parms_path = (
        root
        / "parms.mat"
    )

    try:
        parms = read_mat(
            parms_path
        )

        enabled_raw = np.asarray(
            parms.get(
                "pystamps_final_ifg_qc_enabled",
                0.0,
            )
        ).reshape(-1)

        enabled = bool(
            enabled_raw.size
            and float(
                enabled_raw[0]
            ) != 0.0
        )

        if not enabled:
            return result

        payload = json.loads(
            selection.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "status"
            )
            != "ok"
        ):
            return result

        drops = [
            int(v)
            for v in payload.get(
                "effective_drop_ifg_index",
                [],
            )
        ]

        parms[
            "drop_ifg_index"
        ] = np.asarray(
            drops,
            dtype=np.float64,
        ).reshape(
            1,
            -1,
        )

        write_mat(
            parms_path,
            parms,
        )

        print(
            "[IFG_PRE_QC] "
            "FINAL-QC owns drop_ifg_index; "
            f"restored {len(drops)} "
            "final drops",
            flush=True,
        )

    except Exception as exc:

        print(
            "[IFG_PRE_QC][WARNING] "
            "could not restore "
            "FINAL-QC drop list: "
            f"{type(exc).__name__}: "
            f"{exc}",
            flush=True,
        )

    return result
