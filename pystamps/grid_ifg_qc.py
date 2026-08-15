from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHOD = "grid_multimetric_family_node_edge_v3"


@dataclass(slots=True)
class GridIFGQCResult:
    keep_local_mask: np.ndarray
    drop_original_indices: tuple[int, ...]
    protected_original_indices: tuple[int, ...]
    candidate_original_indices: tuple[int, ...]
    n_triangles: int


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def settings_from_config(config: Any) -> dict[str, Any]:
    return {
        "enabled": bool(
            getattr(config, "grid_qc_enabled", True)
        ),
        "metric_bad_quantile": float(
            getattr(config, "grid_qc_metric_bad_quantile", 0.90)
        ),
        "score_tail_fraction": float(
            getattr(config, "grid_qc_score_tail_fraction", 0.02)
        ),
        "score_z_threshold": float(
            getattr(config, "grid_qc_score_z_threshold", 1.5)
        ),
        "extreme_z_threshold": float(
            getattr(config, "grid_qc_extreme_z_threshold", 3.5)
        ),
        "min_bad_metrics": int(
            getattr(config, "grid_qc_min_bad_metrics", 2)
        ),
        "max_drop_fraction": float(
            getattr(config, "grid_qc_max_drop_fraction", 0.05)
        ),
        "preserve_network": bool(
            getattr(config, "grid_qc_preserve_network", True)
        ),
        "sample_ps": int(
            getattr(config, "grid_qc_sample_ps", 20000)
        ),
        "sample_pairs": int(
            getattr(config, "grid_qc_sample_pairs", 30000)
        ),
        "closure_sample_ps": int(
            getattr(config, "grid_qc_closure_sample_ps", 6000)
        ),
        "max_triangles": int(
            getattr(config, "grid_qc_max_triangles", 4000)
        ),
        "node_context_enabled": bool(
            getattr(config, "grid_qc_node_context_enabled", True)
        ),
        "node_min_degree": int(
            getattr(config, "grid_qc_node_min_degree", 3)
        ),
        "node_candidate_fraction": float(
            getattr(config, "grid_qc_node_candidate_fraction", 0.50)
        ),
        "edge_excess_z_threshold": float(
            getattr(config, "grid_qc_edge_excess_z_threshold", 2.5)
        ),
        "clustered_edge_excess_z_threshold": float(
            getattr(
                config,
                "grid_qc_clustered_edge_excess_z_threshold",
                3.5,
            )
        ),
        "low_degree_score_z_threshold": float(
            getattr(config, "grid_qc_low_degree_score_z_threshold", 4.0)
        ),
    }


def settings_from_parms(parms: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(round(_scalar(
            parms.get("pystamps_grid_qc_enabled"), 0.0
        ))),
        "metric_bad_quantile": _scalar(
            parms.get("pystamps_grid_qc_metric_bad_quantile"), 0.90
        ),
        "score_tail_fraction": _scalar(
            parms.get("pystamps_grid_qc_score_tail_fraction"), 0.02
        ),
        "score_z_threshold": _scalar(
            parms.get("pystamps_grid_qc_score_z_threshold"), 1.5
        ),
        "extreme_z_threshold": _scalar(
            parms.get("pystamps_grid_qc_extreme_z_threshold"), 3.5
        ),
        "min_bad_metrics": int(round(_scalar(
            parms.get("pystamps_grid_qc_min_bad_metrics"), 2
        ))),
        "max_drop_fraction": _scalar(
            parms.get("pystamps_grid_qc_max_drop_fraction"), 0.05
        ),
        "preserve_network": bool(round(_scalar(
            parms.get("pystamps_grid_qc_preserve_network"), 1.0
        ))),
        "sample_ps": int(round(_scalar(
            parms.get("pystamps_grid_qc_sample_ps"), 20000
        ))),
        "sample_pairs": int(round(_scalar(
            parms.get("pystamps_grid_qc_sample_pairs"), 30000
        ))),
        "closure_sample_ps": int(round(_scalar(
            parms.get("pystamps_grid_qc_closure_sample_ps"), 6000
        ))),
        "max_triangles": int(round(_scalar(
            parms.get("pystamps_grid_qc_max_triangles"), 4000
        ))),
        "node_context_enabled": bool(round(_scalar(
            parms.get("pystamps_grid_qc_node_context_enabled"), 1.0
        ))),
        "node_min_degree": int(round(_scalar(
            parms.get("pystamps_grid_qc_node_min_degree"), 3
        ))),
        "node_candidate_fraction": _scalar(
            parms.get("pystamps_grid_qc_node_candidate_fraction"), 0.50
        ),
        "edge_excess_z_threshold": _scalar(
            parms.get("pystamps_grid_qc_edge_excess_z_threshold"), 2.5
        ),
        "clustered_edge_excess_z_threshold": _scalar(
            parms.get(
                "pystamps_grid_qc_clustered_edge_excess_z_threshold"
            ),
            3.5,
        ),
        "low_degree_score_z_threshold": _scalar(
            parms.get("pystamps_grid_qc_low_degree_score_z_threshold"),
            4.0,
        ),
    }


def grid_qc_audit_is_current(
    dataset_root: Path,
    config: Any,
) -> bool:
    mode = str(
        getattr(config, "mode", "auto")
    ).strip().lower()

    settings = settings_from_config(config)

    if mode != "auto" or not settings["enabled"]:
        return True

    path = Path(dataset_root) / "grid_ifg_selection.json"

    if not path.exists():
        return False

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    if payload.get("method") != METHOD:
        return False

    old = payload.get("settings")

    if not isinstance(old, dict):
        return False

    for key, value in settings.items():
        if key == "enabled":
            continue

        if key not in old:
            return False

        a = old[key]
        b = value

        if isinstance(b, bool):
            if bool(a) != b:
                return False
        elif isinstance(b, int):
            if int(a) != b:
                return False
        else:
            if not math.isclose(
                float(a),
                float(b),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False

    return True


def _robust_high_z(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)

    out = np.zeros(x.shape, dtype=np.float64)

    finite = np.isfinite(x)

    if np.count_nonzero(finite) < 2:
        return out

    xf = x[finite]

    med = float(np.median(xf))
    mad = float(
        np.median(
            np.abs(xf - med)
        )
    )

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= 1e-12:
        std = float(np.std(xf))
        spread = float(np.ptp(xf))
        range_scale = (
            spread / max(1.0, math.sqrt(float(xf.size)))
        )

        scale = max(
            std,
            range_scale,
            1e-12,
        )

    out[finite] = np.maximum(
        0.0,
        (x[finite] - med) / scale,
    )

    return out


def _high_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)

    out = np.full(
        x.shape,
        0.5,
        dtype=np.float64,
    )

    finite_ix = np.flatnonzero(
        np.isfinite(x)
    )

    if finite_ix.size < 2:
        return out

    xf = x[finite_ix]

    unique, inverse, counts = np.unique(
        xf,
        return_inverse=True,
        return_counts=True,
    )

    if unique.size == 1:
        return out

    cumulative_before = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(counts[:-1]),
        )
    )

    midrank = (
        cumulative_before
        + (counts + 1.0) / 2.0
    ) / float(xf.size)

    out[finite_ix] = midrank[inverse]

    return out


def _unit_complex(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(
        values,
        dtype=np.complex64,
    )

    mag = np.abs(z)

    valid = (
        np.isfinite(z.real)
        & np.isfinite(z.imag)
        & np.isfinite(mag)
        & (mag > 0)
    )

    out = np.zeros(
        z.shape,
        dtype=np.complex64,
    )

    out[valid] = (
        z[valid]
        / mag[valid]
    )

    return out, valid


def _sample_indices(
    n: int,
    maximum: int,
) -> np.ndarray:
    if n <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    maximum = max(
        1,
        int(maximum),
    )

    if n <= maximum:
        return np.arange(
            n,
            dtype=np.int64,
        )

    return np.unique(
        np.linspace(
            0,
            n - 1,
            maximum,
            dtype=np.int64,
        )
    )


def _grid_neighbor_pairs(
    nz_i: np.ndarray,
    nz_j: np.ndarray,
    n_i: int,
    n_j: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = np.full(
        (int(n_i), int(n_j)),
        -1,
        dtype=np.int64,
    )

    rows = np.arange(
        np.asarray(nz_i).size,
        dtype=np.int64,
    )

    lookup[
        np.asarray(nz_i, dtype=np.int64) - 1,
        np.asarray(nz_j, dtype=np.int64) - 1,
    ] = rows

    a = []
    b = []

    left = lookup[:, :-1]
    right = lookup[:, 1:]

    mask = (
        (left >= 0)
        & (right >= 0)
    )

    a.append(left[mask])
    b.append(right[mask])

    top = lookup[:-1, :]
    bottom = lookup[1:, :]

    mask = (
        (top >= 0)
        & (bottom >= 0)
    )

    a.append(top[mask])
    b.append(bottom[mask])

    if not a:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    p0 = np.concatenate(a)
    p1 = np.concatenate(b)

    if p0.size > maximum:
        ix = _sample_indices(
            p0.size,
            maximum,
        )

        p0 = p0[ix]
        p1 = p1[ix]

    return p0, p1


def _build_triangles(
    edges: np.ndarray,
    maximum: int,
) -> list[
    tuple[int, int, int, int, int, int]
]:
    edge_map: dict[
        tuple[int, int],
        tuple[int, int],
    ] = {}

    neighbors: dict[
        int,
        set[int],
    ] = {}

    for j, (raw_a, raw_b) in enumerate(
        np.asarray(edges, dtype=np.int64)
    ):
        a = int(raw_a)
        b = int(raw_b)

        lo = min(a, b)
        hi = max(a, b)

        sign = 1 if (a == lo and b == hi) else -1

        # A standard SBAS network should not contain duplicated
        # acquisition pairs. Keep the first if it does.
        edge_map.setdefault(
            (lo, hi),
            (j, sign),
        )

        neighbors.setdefault(
            lo,
            set(),
        ).add(hi)

        neighbors.setdefault(
            hi,
            set(),
        ).add(lo)

    triangles = []

    for a in sorted(neighbors):
        for b in sorted(
            node
            for node
            in neighbors[a]
            if node > a
        ):
            common = (
                neighbors[a]
                & neighbors.get(b, set())
            )

            for c in sorted(
                node
                for node
                in common
                if node > b
            ):
                ab = edge_map.get((a, b))
                bc = edge_map.get((b, c))
                ac = edge_map.get((a, c))

                if (
                    ab is None
                    or bc is None
                    or ac is None
                ):
                    continue

                triangles.append(
                    (
                        ab[0],
                        ab[1],
                        bc[0],
                        bc[1],
                        ac[0],
                        ac[1],
                    )
                )

    if (
        maximum > 0
        and len(triangles) > maximum
    ):
        ix = _sample_indices(
            len(triangles),
            maximum,
        )

        triangles = [
            triangles[int(i)]
            for i in ix
        ]

    return triangles


def _network_nodes(
    edges: np.ndarray,
    keep: np.ndarray,
) -> set[int]:
    selected = np.asarray(edges)[keep]

    if selected.size == 0:
        return set()

    return set(
        int(v)
        for v
        in selected.reshape(-1)
    )


def _network_connected(
    edges: np.ndarray,
    keep: np.ndarray,
    required_nodes: set[int],
) -> bool:
    selected = np.asarray(
        edges,
        dtype=np.int64,
    )[keep]

    if selected.size == 0:
        return False

    current_nodes = _network_nodes(
        edges,
        keep,
    )

    if current_nodes != required_nodes:
        return False

    adjacency = {
        node: set()
        for node
        in required_nodes
    }

    for a, b in selected:
        a = int(a)
        b = int(b)

        adjacency[a].add(b)
        adjacency[b].add(a)

    start = next(iter(required_nodes))

    seen = {start}
    stack = [start]

    while stack:
        node = stack.pop()

        for other in adjacency[node]:
            if other not in seen:
                seen.add(other)
                stack.append(other)

    return seen == required_nodes


def _circular_mad(
    angles: np.ndarray,
) -> float:
    a = np.asarray(
        angles,
        dtype=np.float64,
    )

    if a.size == 0:
        return float("nan")

    center = float(
        np.angle(
            np.mean(
                np.exp(1j * a)
            )
        )
    )

    deviation = np.angle(
        np.exp(
            1j * (a - center)
        )
    )

    return float(
        1.4826
        * np.median(
            np.abs(deviation)
        )
    )



def _family_evidence(
    metric_values: dict[str, np.ndarray],
    settings: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Convert correlated raw metrics into independent evidence families.

    V1 counted each residual statistic separately. That caused
    residual_rms / residual_mad / residual_incoherence /
    extreme_fraction to cast several votes for essentially the same
    physical behaviour.

    V2 allows each family to contribute at most one normal bad vote
    and one extreme vote per IFG.
    """

    if not metric_values:
        raise ValueError(
            "GRID QC requires at least one quality metric"
        )

    n_ifg = int(
        np.asarray(
            next(iter(metric_values.values()))
        ).size
    )

    metric_z = {
        name: _robust_high_z(values)
        for name, values
        in metric_values.items()
    }

    metric_pct = {
        name: _high_percentile(values)
        for name, values
        in metric_values.items()
    }

    # Independent physical/statistical evidence groups.
    family_members = {
        "closure": (
            "closure_rms",
            "closure_incoherence",
        ),
        "spatial_residual": (
            "residual_rms",
            "residual_mad",
            "residual_incoherence",
            "extreme_fraction",
        ),
        "spatial_gradient": (
            "gradient_mad",
        ),
        "data_validity": (
            "invalid_fraction",
        ),
        "stage5_noise": (
            "ifg_std",
        ),
    }

    # Family weights sum to one for a fully populated run.
    family_weights = {
        "closure": 0.30,
        "spatial_residual": 0.30,
        "spatial_gradient": 0.20,
        "data_validity": 0.10,
        "stage5_noise": 0.10,
    }

    family_z: dict[str, np.ndarray] = {}
    family_pct: dict[str, np.ndarray] = {}

    bad_family_count = np.zeros(
        n_ifg,
        dtype=np.int16,
    )

    extreme_family_count = np.zeros(
        n_ifg,
        dtype=np.int16,
    )

    family_score = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    used_weight = 0.0

    bad_quantile = float(
        settings["metric_bad_quantile"]
    )

    extreme_threshold = float(
        settings["extreme_z_threshold"]
    )

    for family, members in family_members.items():

        available = [
            name
            for name in members
            if name in metric_values
        ]

        if not available:
            continue

        z_stack = np.vstack(
            [
                metric_z[name]
                for name in available
            ]
        )

        pct_stack = np.vstack(
            [
                metric_pct[name]
                for name in available
            ]
        )

        # One family = at most one vote.
        fz = np.nanmax(
            z_stack,
            axis=0,
        )

        fp = np.nanmax(
            pct_stack,
            axis=0,
        )

        family_z[family] = fz
        family_pct[family] = fp

        bad = (
            fp >= bad_quantile
        )

        extreme = (
            fz >= extreme_threshold
        )

        bad_family_count += (
            bad.astype(np.int16)
        )

        extreme_family_count += (
            extreme.astype(np.int16)
        )

        weight = float(
            family_weights[family]
        )

        family_score += (
            weight
            * np.minimum(
                fz,
                8.0,
            )
        )

        used_weight += weight

    if used_weight > 0:
        family_score /= used_weight

    return (
        family_z,
        family_pct,
        metric_z,
        bad_family_count,
        extreme_family_count,
        family_score,
    )


def _node_edge_context(
    *,
    edges: np.ndarray,
    score: np.ndarray,
    base_candidate: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Attribute common edge-quality elevation to acquisition nodes.

    If many IFGs sharing one acquisition are simultaneously poor,
    their common score elevation is treated as a node-level effect.

    The final edge test therefore uses the excess of an IFG score
    above the median quality level of its two endpoint nodes.

    This prevents one problematic acquisition from automatically
    causing every connected IFG to be deleted.
    """

    edges = np.asarray(
        edges,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    ).reshape(-1)

    base_candidate = np.asarray(
        base_candidate,
        dtype=bool,
    ).reshape(-1)

    n_ifg = score.size

    if edges.shape != (n_ifg, 2):
        raise ValueError(
            "node-context edge shape mismatch"
        )

    nodes = np.unique(
        edges.reshape(-1)
    )

    degree: dict[int, int] = {}
    score_median: dict[int, float] = {}
    candidate_fraction: dict[int, float] = {}
    clustered: dict[int, bool] = {}

    min_degree = int(
        settings.get(
            "node_min_degree",
            3,
        )
    )

    fraction_threshold = float(
        settings.get(
            "node_candidate_fraction",
            0.50,
        )
    )

    incident_by_node: dict[int, np.ndarray] = {}

    for raw_node in nodes:
        node = int(raw_node)

        ix = np.flatnonzero(
            (edges[:, 0] == node)
            | (edges[:, 1] == node)
        )

        incident_by_node[node] = ix

        degree[node] = int(
            ix.size
        )

        if ix.size:
            score_median[node] = float(
                np.nanmedian(
                    score[ix]
                )
            )

            candidate_fraction[node] = float(
                np.mean(
                    base_candidate[ix]
                )
            )
        else:
            score_median[node] = 0.0
            candidate_fraction[node] = 0.0

        clustered[node] = bool(
            degree[node] >= min_degree
            and candidate_fraction[node]
            >= fraction_threshold
        )

    node_a_degree = np.zeros(
        n_ifg,
        dtype=np.int64,
    )

    node_b_degree = np.zeros(
        n_ifg,
        dtype=np.int64,
    )

    node_a_median = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    node_b_median = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    node_a_candidate_fraction = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    node_b_candidate_fraction = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    node_clustered = np.zeros(
        n_ifg,
        dtype=bool,
    )

    for j, (raw_a, raw_b) in enumerate(edges):
        a = int(raw_a)
        b = int(raw_b)

        node_a_degree[j] = degree[a]
        node_b_degree[j] = degree[b]

        node_a_median[j] = score_median[a]
        node_b_median[j] = score_median[b]

        node_a_candidate_fraction[j] = (
            candidate_fraction[a]
        )

        node_b_candidate_fraction[j] = (
            candidate_fraction[b]
        )

        node_clustered[j] = (
            clustered[a]
            or clustered[b]
        )

    expected_score = (
        0.5
        * (
            node_a_median
            + node_b_median
        )
    )

    edge_excess = (
        score
        - expected_score
    )

    edge_excess_z = _robust_high_z(
        edge_excess
    )

    return {
        "nodes": nodes,
        "degree": degree,
        "score_median": score_median,
        "candidate_fraction": candidate_fraction,
        "clustered": clustered,
        "incident_by_node": incident_by_node,
        "node_a_degree": node_a_degree,
        "node_b_degree": node_b_degree,
        "node_a_median": node_a_median,
        "node_b_median": node_b_median,
        "node_a_candidate_fraction":
            node_a_candidate_fraction,
        "node_b_candidate_fraction":
            node_b_candidate_fraction,
        "node_clustered": node_clustered,
        "expected_score": expected_score,
        "edge_excess": edge_excess,
        "edge_excess_z": edge_excess_z,
    }


def _select_candidates(
    *,
    edges: np.ndarray,
    original_ifg_index: np.ndarray,
    metric_values: dict[str, np.ndarray],
    metric_weights: dict[str, float],
    settings: dict[str, Any],
) -> tuple[
    np.ndarray,
    set[int],
    set[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    GRID-QC V3.

    Step 1:
        build V2 independent-family candidates.

    Step 2:
        estimate acquisition-node quality baseline.

    Step 3:
        keep only IFGs whose quality score is anomalously high
        relative to their endpoint-node baselines.

    Step 4:
        preserve acquisition-network connectivity.
    """

    del metric_weights

    n_ifg = int(
        original_ifg_index.size
    )

    (
        family_z,
        family_pct,
        metric_z,
        bad_family_count,
        extreme_family_count,
        score,
    ) = _family_evidence(
        metric_values,
        settings,
    )

    score_z = _robust_high_z(
        score
    )

    score_pct = _high_percentile(
        score
    )

    tail_threshold = (
        1.0
        - float(
            settings[
                "score_tail_fraction"
            ]
        )
    )

    min_families = int(
        settings[
            "min_bad_metrics"
        ]
    )

    normal_candidate = (
        (
            score_pct
            >= tail_threshold
        )
        & (
            score_z
            >= float(
                settings[
                    "score_z_threshold"
                ]
            )
        )
        & (
            bad_family_count
            >= min_families
        )
    )

    extreme_candidate = (
        extreme_family_count
        >= 2
    )

    base_candidate = (
        normal_candidate
        | extreme_candidate
    )

    node_enabled = bool(
        settings.get(
            "node_context_enabled",
            False,
        )
    )

    if node_enabled:
        node = _node_edge_context(
            edges=edges,
            score=score,
            base_candidate=base_candidate,
            settings=settings,
        )

        edge_excess_z = np.asarray(
            node["edge_excess_z"],
            dtype=np.float64,
        )

        node_clustered = np.asarray(
            node["node_clustered"],
            dtype=bool,
        )

        normal_context_threshold = float(
            settings.get(
                "edge_excess_z_threshold",
                2.5,
            )
        )

        clustered_context_threshold = float(
            settings.get(
                "clustered_edge_excess_z_threshold",
                3.5,
            )
        )

        threshold = np.where(
            node_clustered,
            clustered_context_threshold,
            normal_context_threshold,
        )

        contextual_edge = (
            edge_excess_z
            >= threshold
        )

        min_degree = int(
            settings.get(
                "node_min_degree",
                3,
            )
        )

        low_degree = (
            np.asarray(
                node["node_a_degree"]
            )
            < min_degree
        ) | (
            np.asarray(
                node["node_b_degree"]
            )
            < min_degree
        )

        # Very sparse endpoint topology cannot support a reliable
        # node-median estimate. In that case require a much stronger
        # global anomaly plus two extreme independent families.
        low_degree_fallback = (
            low_degree
            & (
                score_z
                >= float(
                    settings.get(
                        "low_degree_score_z_threshold",
                        4.0,
                    )
                )
            )
            & (
                extreme_family_count
                >= 2
            )
        )

        candidate = (
            base_candidate
            & (
                contextual_edge
                | low_degree_fallback
            )
        )

    else:
        candidate = (
            base_candidate
        )

    candidate_ix = np.flatnonzero(
        candidate
    )

    candidate_ix = candidate_ix[
        np.argsort(
            -score[candidate_ix],
            kind="stable",
        )
    ]

    max_drop = int(
        math.floor(
            n_ifg
            * float(
                settings[
                    "max_drop_fraction"
                ]
            )
        )
    )

    if (
        candidate_ix.size
        and max_drop < 1
    ):
        max_drop = 1

    keep = np.ones(
        n_ifg,
        dtype=bool,
    )

    required_nodes = _network_nodes(
        edges,
        keep,
    )

    dropped: set[int] = set()
    protected: set[int] = set()

    for zero_ix in candidate_ix:

        if len(dropped) >= max_drop:
            break

        proposed = keep.copy()
        proposed[zero_ix] = False

        original_index = int(
            original_ifg_index[
                zero_ix
            ]
        )

        if (
            bool(
                settings[
                    "preserve_network"
                ]
            )
            and not _network_connected(
                edges,
                proposed,
                required_nodes,
            )
        ):
            protected.add(
                original_index
            )
            continue

        keep = proposed

        dropped.add(
            original_index
        )

    return (
        keep,
        dropped,
        protected,
        candidate,
        score,
        score_z,
        metric_z,
    )

def run_grid_ifg_qc(
    *,
    dataset_root: Path,
    grid_phase: np.ndarray,
    grid_lowpass: np.ndarray,
    nz_i: np.ndarray,
    nz_j: np.ndarray,
    n_i: int,
    n_j: int,
    original_ifg_index: np.ndarray,
    ifgday_ix: np.ndarray,
    ifg_std: np.ndarray | None,
    parms: dict[str, Any],
) -> GridIFGQCResult:
    root = Path(
        dataset_root
    ).expanduser().resolve()

    settings = settings_from_parms(
        parms
    )

    gp = grid_phase
    gl = grid_lowpass

    n_grid_ps, n_ifg = gp.shape

    original_ifg_index = np.asarray(
        original_ifg_index,
        dtype=np.int64,
    ).reshape(-1)

    edges = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    if original_ifg_index.size != n_ifg:
        raise RuntimeError(
            "GRID IFG QC original_ifg_index "
            "length mismatch"
        )

    if edges.shape != (n_ifg, 2):
        raise RuntimeError(
            "GRID IFG QC ifgday_ix shape "
            f"{edges.shape}; expected "
            f"({n_ifg}, 2)"
        )

    sample_ix = _sample_indices(
        n_grid_ps,
        int(settings["sample_ps"]),
    )

    pair0, pair1 = _grid_neighbor_pairs(
        nz_i,
        nz_j,
        n_i,
        n_j,
        int(settings["sample_pairs"]),
    )

    residual_rms = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    residual_mad = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    residual_incoherence = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    gradient_mad = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    extreme_fraction = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    invalid_fraction = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Filtered-vs-lowpass residual metrics
    # --------------------------------------------------------------

    for j in range(n_ifg):
        f, vf = _unit_complex(
            np.asarray(
                gp[sample_ix, j]
            )
        )

        l, vl = _unit_complex(
            np.asarray(
                gl[sample_ix, j]
            )
        )

        valid = vf & vl

        invalid_fraction[j] = (
            1.0
            - float(
                np.count_nonzero(valid)
            )
            / max(
                1,
                valid.size,
            )
        )

        if np.count_nonzero(valid) >= 8:
            q = (
                f[valid]
                * np.conj(
                    l[valid]
                )
            )

            r = np.angle(q)

            residual_rms[j] = float(
                np.sqrt(
                    np.mean(
                        r.astype(
                            np.float64
                        ) ** 2
                    )
                )
            )

            residual_mad[j] = (
                _circular_mad(r)
            )

            residual_incoherence[j] = (
                1.0
                - float(
                    np.abs(
                        np.mean(q)
                    )
                )
            )

            center = float(
                np.angle(
                    np.mean(q)
                )
            )

            dev = np.abs(
                np.angle(
                    np.exp(
                        1j
                        * (
                            r
                            - center
                        )
                    )
                )
            )

            extreme_fraction[j] = float(
                np.mean(
                    dev > (np.pi / 2.0)
                )
            )

        if pair0.size:
            f0, v0f = _unit_complex(
                np.asarray(
                    gp[pair0, j]
                )
            )

            f1, v1f = _unit_complex(
                np.asarray(
                    gp[pair1, j]
                )
            )

            l0, v0l = _unit_complex(
                np.asarray(
                    gl[pair0, j]
                )
            )

            l1, v1l = _unit_complex(
                np.asarray(
                    gl[pair1, j]
                )
            )

            valid_pair = (
                v0f
                & v1f
                & v0l
                & v1l
            )

            if (
                np.count_nonzero(
                    valid_pair
                )
                >= 8
            ):
                q0 = (
                    f0[valid_pair]
                    * np.conj(
                        l0[valid_pair]
                    )
                )

                q1 = (
                    f1[valid_pair]
                    * np.conj(
                        l1[valid_pair]
                    )
                )

                grad = np.angle(
                    q0
                    * np.conj(q1)
                )

                gradient_mad[j] = (
                    _circular_mad(
                        grad
                    )
                )

    # --------------------------------------------------------------
    # Closure-phase quality
    # --------------------------------------------------------------

    triangles = _build_triangles(
        edges,
        int(
            settings[
                "max_triangles"
            ]
        ),
    )

    closure_rms_lists = [
        []
        for _ in range(n_ifg)
    ]

    closure_inc_lists = [
        []
        for _ in range(n_ifg)
    ]

    if triangles:
        closure_ix = _sample_indices(
            n_grid_ps,
            int(
                settings[
                    "closure_sample_ps"
                ]
            ),
        )

        sampled = np.asarray(
            gp[closure_ix, :],
            dtype=np.complex64,
        )

        mag = np.abs(sampled)

        valid_sample = (
            np.isfinite(sampled.real)
            & np.isfinite(sampled.imag)
            & (mag > 0)
        )

        unit_sample = np.zeros_like(
            sampled,
            dtype=np.complex64,
        )

        unit_sample[
            valid_sample
        ] = (
            sampled[
                valid_sample
            ]
            / mag[
                valid_sample
            ]
        )

        for (
            e_ab,
            s_ab,
            e_bc,
            s_bc,
            e_ac,
            s_ac,
        ) in triangles:

            za = unit_sample[:, e_ab]
            zb = unit_sample[:, e_bc]
            zc = unit_sample[:, e_ac]

            va = valid_sample[:, e_ab]
            vb = valid_sample[:, e_bc]
            vc = valid_sample[:, e_ac]

            valid = va & vb & vc

            if np.count_nonzero(valid) < 8:
                continue

            za = za[valid]
            zb = zb[valid]
            zc = zc[valid]

            if s_ab < 0:
                za = np.conj(za)

            if s_bc < 0:
                zb = np.conj(zb)

            if s_ac < 0:
                zc = np.conj(zc)

            closure = (
                za
                * zb
                * np.conj(zc)
            )

            angle = np.angle(
                closure
            ).astype(
                np.float64
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        angle**2
                    )
                )
            )

            incoherence = (
                1.0
                - float(
                    np.abs(
                        np.mean(
                            closure
                        )
                    )
                )
            )

            for edge_ix in (
                e_ab,
                e_bc,
                e_ac,
            ):
                closure_rms_lists[
                    edge_ix
                ].append(rms)

                closure_inc_lists[
                    edge_ix
                ].append(
                    incoherence
                )

    closure_rms = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    closure_incoherence = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    triangle_count = np.zeros(
        n_ifg,
        dtype=np.int64,
    )

    for j in range(n_ifg):
        if closure_rms_lists[j]:
            closure_rms[j] = float(
                np.median(
                    closure_rms_lists[j]
                )
            )

            closure_incoherence[j] = float(
                np.median(
                    closure_inc_lists[j]
                )
            )

            triangle_count[j] = int(
                len(
                    closure_rms_lists[j]
                )
            )

    # --------------------------------------------------------------
    # Combine metrics
    # --------------------------------------------------------------

    metric_values = {
        "closure_rms": closure_rms,
        "closure_incoherence": closure_incoherence,
        "gradient_mad": gradient_mad,
        "residual_rms": residual_rms,
        "residual_mad": residual_mad,
        "residual_incoherence": residual_incoherence,
        "extreme_fraction": extreme_fraction,
        "invalid_fraction": invalid_fraction,
    }

    if (
        ifg_std is not None
        and np.asarray(ifg_std).size == n_ifg
    ):
        metric_values[
            "ifg_std"
        ] = np.asarray(
            ifg_std,
            dtype=np.float64,
        ).reshape(-1)

    weights = {
        "closure_rms": 0.22,
        "closure_incoherence": 0.13,
        "gradient_mad": 0.18,
        "residual_rms": 0.12,
        "residual_mad": 0.08,
        "residual_incoherence": 0.10,
        "extreme_fraction": 0.07,
        "invalid_fraction": 0.05,
        "ifg_std": 0.05,
    }

    (
        keep,
        dropped,
        protected,
        candidate,
        score,
        score_z,
        z_values,
    ) = _select_candidates(
        edges=edges,
        original_ifg_index=original_ifg_index,
        metric_values=metric_values,
        metric_weights=weights,
        settings=settings,
    )

    # --------------------------------------------------------------
    # Audit output
    # --------------------------------------------------------------

    candidate_original = tuple(
        int(original_ifg_index[i])
        for i
        in np.flatnonzero(candidate)
    )

    csv_path = (
        root
        / "grid_ifg_quality_audit.csv"
    )

    metric_names = list(
        metric_values
    )

    percentiles = {
        name: _high_percentile(
            metric_values[name]
        )
        for name
        in metric_names
    }

    (
        family_z,
        family_percentiles,
        _metric_z_check,
        bad_family_count,
        extreme_family_count,
        _family_score_check,
    ) = _family_evidence(
        metric_values,
        settings,
    )

    # --------------------------------------------------------------
    # V3 acquisition-node attribution audit
    # --------------------------------------------------------------

    score_percentile = _high_percentile(
        score
    )

    _normal_base_candidate = (
        (
            score_percentile
            >= (
                1.0
                - float(
                    settings[
                        "score_tail_fraction"
                    ]
                )
            )
        )
        & (
            score_z
            >= float(
                settings[
                    "score_z_threshold"
                ]
            )
        )
        & (
            bad_family_count
            >= int(
                settings[
                    "min_bad_metrics"
                ]
            )
        )
    )

    _extreme_base_candidate = (
        extreme_family_count
        >= 2
    )

    base_candidate = (
        _normal_base_candidate
        | _extreme_base_candidate
    )

    node_context = _node_edge_context(
        edges=edges,
        score=score,
        base_candidate=base_candidate,
        settings=settings,
    )

    edge_excess = np.asarray(
        node_context[
            "edge_excess"
        ],
        dtype=np.float64,
    )

    edge_excess_z = np.asarray(
        node_context[
            "edge_excess_z"
        ],
        dtype=np.float64,
    )

    node_clustered = np.asarray(
        node_context[
            "node_clustered"
        ],
        dtype=bool,
    )

    # Node-level audit file.
    node_csv_path = (
        root
        / "grid_ifg_node_audit.csv"
    )

    with node_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as node_handle:

        node_writer = csv.writer(
            node_handle
        )

        node_writer.writerow(
            [
                "acquisition_index",
                "degree",
                "median_edge_score",
                "candidate_fraction",
                "clustered_candidate_node",
            ]
        )

        for raw_node in node_context[
            "nodes"
        ]:
            _node = int(raw_node)

            node_writer.writerow(
                [
                    _node,
                    int(
                        node_context[
                            "degree"
                        ][_node]
                    ),
                    float(
                        node_context[
                            "score_median"
                        ][_node]
                    ),
                    float(
                        node_context[
                            "candidate_fraction"
                        ][_node]
                    ),
                    int(
                        node_context[
                            "clustered"
                        ][_node]
                    ),
                ]
            )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        header = [
            "ifg_index",
            "local_column",
            "acq1_index",
            "acq2_index",
            "triangle_count",
        ]

        for name in metric_names:
            header.extend(
                [
                    name,
                    f"z_{name}",
                    f"pct_{name}",
                ]
            )

        header.extend(
            [
                "bad_family_count",
                "extreme_family_count",
                "score",
                "score_z",
                "score_percentile",
                "base_candidate_bad",
                "node_a_degree",
                "node_b_degree",
                "node_a_score_median",
                "node_b_score_median",
                "node_a_candidate_fraction",
                "node_b_candidate_fraction",
                "node_clustered",
                "edge_excess",
                "edge_excess_z",
                "candidate_bad",
                "decision",
            ]
        )

        writer.writerow(header)

        for j in range(n_ifg):
            original = int(
                original_ifg_index[j]
            )

            if original in dropped:
                decision = "drop"
            elif original in protected:
                decision = (
                    "protected_network"
                )
            elif candidate[j]:
                decision = (
                    "candidate_not_dropped"
                )
            else:
                decision = "keep"

            row = [
                original,
                j + 1,
                int(edges[j, 0]),
                int(edges[j, 1]),
                int(
                    triangle_count[j]
                ),
            ]

            for name in metric_names:
                row.extend(
                    [
                        float(
                            metric_values[
                                name
                            ][j]
                        ),
                        float(
                            z_values[
                                name
                            ][j]
                        ),
                        float(
                            percentiles[
                                name
                            ][j]
                        ),
                    ]
                )

            row.extend(
                [
                    int(
                        bad_family_count[j]
                    ),
                    int(
                        extreme_family_count[j]
                    ),
                    float(score[j]),
                    float(score_z[j]),
                    float(
                        score_percentile[j]
                    ),
                    int(
                        base_candidate[j]
                    ),
                    int(
                        node_context[
                            "node_a_degree"
                        ][j]
                    ),
                    int(
                        node_context[
                            "node_b_degree"
                        ][j]
                    ),
                    float(
                        node_context[
                            "node_a_median"
                        ][j]
                    ),
                    float(
                        node_context[
                            "node_b_median"
                        ][j]
                    ),
                    float(
                        node_context[
                            "node_a_candidate_fraction"
                        ][j]
                    ),
                    float(
                        node_context[
                            "node_b_candidate_fraction"
                        ][j]
                    ),
                    int(
                        node_clustered[j]
                    ),
                    float(
                        edge_excess[j]
                    ),
                    float(
                        edge_excess_z[j]
                    ),
                    int(candidate[j]),
                    decision,
                ]
            )

            writer.writerow(row)

    max_drop_count = int(
        math.floor(
            n_ifg
            * float(
                settings[
                    "max_drop_fraction"
                ]
            )
        )
    )

    if (
        np.count_nonzero(candidate) > 0
        and max_drop_count < 1
    ):
        max_drop_count = 1

    unresolved_candidate_count = max(
        0,
        int(np.count_nonzero(candidate))
        - len(dropped)
        - len(protected),
    )

    safety_cap_hit = bool(
        max_drop_count > 0
        and len(dropped) >= max_drop_count
        and unresolved_candidate_count > 0
    )

    summary = {
        "method": METHOD,
        "n_ifg_input": int(n_ifg),
        "n_ifg_selected": int(
            np.count_nonzero(keep)
        ),
        "n_ifg_dropped": int(
            len(dropped)
        ),
        "base_candidate_count": int(
            np.count_nonzero(
                base_candidate
            )
        ),
        "candidate_count": int(
            np.count_nonzero(
                candidate
            )
        ),
        "node_filtered_out_count": int(
            np.count_nonzero(
                base_candidate
                & ~candidate
            )
        ),
        "clustered_node_count": int(
            sum(
                bool(v)
                for v
                in node_context[
                    "clustered"
                ].values()
            )
        ),
        "node_audit_file": (
            "grid_ifg_node_audit.csv"
        ),
        "protected_network_count": int(
            len(protected)
        ),
        "safety_cap_count": int(
            max_drop_count
        ),
        "safety_cap_hit": bool(
            safety_cap_hit
        ),
        "unresolved_candidate_count": int(
            unresolved_candidate_count
        ),
        "evidence_policy": {
            "families": [
                "closure",
                "spatial_residual",
                "spatial_gradient",
                "data_validity",
                "stage5_noise"
            ],
            "normal_candidate": (
                "score_tail AND score_z AND "
                "independent_bad_family_count"
            ),
            "extreme_candidate": (
                "at_least_two_independent_"
                "extreme_families"
            ),
            "node_context": (
                "edge_score_excess_above_endpoint_"
                "node_medians"
            )
        },
        "n_triangles": int(
            len(triangles)
        ),
        "effective_drop_ifg_index": sorted(
            int(v)
            for v
            in dropped
        ),
        "candidate_ifg_index": sorted(
            int(v)
            for v
            in candidate_original
        ),
        "protected_network_ifg_index": sorted(
            int(v)
            for v
            in protected
        ),
        "settings": settings,
    }

    json_path = (
        root
        / "grid_ifg_selection.json"
    )

    tmp = json_path.with_suffix(
        ".json.tmp"
    )

    tmp.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(json_path)

    print(
        "[IFG_GRID_QC] "
        f"input={n_ifg}, "
        f"triangles={len(triangles)}, "
        f"base_candidates="
        f"{int(np.count_nonzero(base_candidate))}, "
        f"candidates="
        f"{int(np.count_nonzero(candidate))}, "
        f"drop={len(dropped)}, "
        f"keep={int(np.count_nonzero(keep))}, "
        f"protected_network="
        f"{len(protected)}, "
        f"cap_hit={safety_cap_hit}",
        flush=True,
    )

    if safety_cap_hit:
        print(
            "[IFG_GRID_QC][WARNING] "
            "automatic rejection reached the "
            "safety cap; selection is unresolved",
            flush=True,
        )

    if dropped:
        print(
            "[IFG_GRID_QC] "
            "effective_drop_ifg_index="
            + ",".join(
                str(v)
                for v
                in sorted(dropped)
            ),
            flush=True,
        )

    top = np.argsort(
        -score,
        kind="stable",
    )[: min(15, n_ifg)]

    print(
        "[IFG_GRID_QC] top-score IFGs: "
        + " ".join(
            f"{int(original_ifg_index[j])}"
            f"(S={score[j]:.2f},"
            f"Z={score_z[j]:.2f},"
            f"F={int(bad_family_count[j])},"
            f"E={int(extreme_family_count[j])},"
            f"X={edge_excess_z[j]:.2f})"
            for j in top
        ),
        flush=True,
    )

    return GridIFGQCResult(
        keep_local_mask=keep,
        drop_original_indices=tuple(
            sorted(
                int(v)
                for v
                in dropped
            )
        ),
        protected_original_indices=tuple(
            sorted(
                int(v)
                for v
                in protected
            )
        ),
        candidate_original_indices=tuple(
            sorted(
                int(v)
                for v
                in candidate_original
            )
        ),
        n_triangles=len(
            triangles
        ),
    )
