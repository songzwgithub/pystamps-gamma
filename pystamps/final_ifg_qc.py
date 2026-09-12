from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat_variables

from pystamps.stage6_turbo import (
    final_qc_chunk as turbo_final_qc_chunk,
)


METHOD = "three_independent_family_final_ifg_qc_v1"


class FinalIFGQCError(RuntimeError):
    pass


def _scalar(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None:
        return float(default)

    arr = np.asarray(value)

    if arr.size == 0:
        return float(default)

    try:
        return float(
            arr.reshape(-1)[0]
        )
    except Exception:
        return float(default)


def settings_from_parms(
    parms: dict[str, Any],
) -> dict[str, Any]:

    return {
        "enabled": bool(
            round(
                _scalar(
                    parms.get(
                        "pystamps_final_ifg_qc_enabled"
                    ),
                    0.0,
                )
            )
        ),

        "msd_strong_percentile":
            _scalar(
                parms.get(
                    "pystamps_final_qc_msd_strong_percentile"
                ),
                0.975,
            ),

        "msd_extreme_percentile":
            _scalar(
                parms.get(
                    "pystamps_final_qc_msd_extreme_percentile"
                ),
                0.990,
            ),

        "network_strong_percentile":
            _scalar(
                parms.get(
                    "pystamps_final_qc_network_strong_percentile"
                ),
                0.975,
            ),

        "network_extreme_percentile":
            _scalar(
                parms.get(
                    "pystamps_final_qc_network_extreme_percentile"
                ),
                0.990,
            ),

        "max_drop_fraction":
            _scalar(
                parms.get(
                    "pystamps_final_qc_max_drop_fraction"
                ),
                0.05,
            ),

        "preserve_network": bool(
            round(
                _scalar(
                    parms.get(
                        "pystamps_final_qc_preserve_network"
                    ),
                    1.0,
                )
            )
        ),

        "fail_on_cap": bool(
            round(
                _scalar(
                    parms.get(
                        "pystamps_final_qc_fail_on_cap"
                    ),
                    1.0,
                )
            )
        ),

        "chunk_ifg": max(
            1,
            int(
                round(
                    _scalar(
                        parms.get(
                            "pystamps_final_qc_chunk_ifg"
                        ),
                        8.0,
                    )
                )
            ),
        ),
    }


def _robust_high_z(
    values: np.ndarray,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    out = np.full(
        x.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = np.isfinite(x)

    if not np.any(valid):
        return out

    xx = x[valid]

    med = float(
        np.median(xx)
    )

    mad = float(
        np.median(
            np.abs(
                xx - med
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
        scale = float(
            np.std(xx)
        )

    if (
        not np.isfinite(scale)
        or scale <= 1e-12
    ):
        scale = 1.0

    out[valid] = np.maximum(
        0.0,
        (
            x[valid]
            - med
        )
        / scale,
    )

    return out


def _percentile_fraction(
    values: np.ndarray,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    out = np.full(
        x.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = np.isfinite(x)

    xx = np.sort(
        x[valid]
    )

    if xx.size == 0:
        return out

    out[valid] = (
        np.searchsorted(
            xx,
            x[valid],
            side="right",
        )
        / float(xx.size)
    )

    return out


def _read_grid_flags(
    root: Path,
    n_ifg: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    grid_family = np.zeros(
        n_ifg,
        dtype=bool,
    )

    grid_strong = np.zeros(
        n_ifg,
        dtype=bool,
    )

    grid_bad_family_count = np.zeros(
        n_ifg,
        dtype=np.int16,
    )

    grid_extreme_family_count = np.zeros(
        n_ifg,
        dtype=np.int16,
    )

    path = (
        root
        / "grid_ifg_quality_audit.csv"
    )

    if not path.exists():
        print(
            "[IFG_FINAL_QC][WARNING] "
            "grid_ifg_quality_audit.csv missing; "
            "GRID family unavailable",
            flush=True,
        )

        return (
            grid_family,
            grid_strong,
            grid_bad_family_count,
            grid_extreme_family_count,
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = csv.DictReader(
            handle
        )

        for row in rows:

            try:
                idx = int(
                    float(
                        row[
                            "ifg_index"
                        ]
                    )
                )
            except Exception:
                continue

            if (
                idx < 1
                or idx > n_ifg
            ):
                continue

            j = idx - 1

            def _ival(
                key: str,
                default: int = 0,
            ) -> int:
                try:
                    return int(
                        float(
                            row.get(
                                key,
                                default,
                            )
                        )
                    )
                except Exception:
                    return default

            base = bool(
                _ival(
                    "base_candidate_bad"
                )
            )

            contextual = bool(
                _ival(
                    "candidate_bad"
                )
            )

            bad_count = _ival(
                "bad_family_count"
            )

            extreme_count = _ival(
                "extreme_family_count"
            )

            grid_bad_family_count[
                j
            ] = bad_count

            grid_extreme_family_count[
                j
            ] = extreme_count

            # GRID is only one independent family.
            grid_family[j] = (
                base
                or contextual
            )

            grid_strong[j] = (
                contextual
                or bad_count >= 3
                or extreme_count >= 2
            )

    return (
        grid_family,
        grid_strong,
        grid_bad_family_count,
        grid_extreme_family_count,
    )


def _candidate_flags(
    *,
    grid_family: np.ndarray,
    grid_strong: np.ndarray,
    msd_percentile: np.ndarray,
    network_percentile: np.ndarray,
    settings: dict[str, Any],
) -> tuple[
    np.ndarray,
    list[list[str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    msd_strong = (
        msd_percentile
        >= float(
            settings[
                "msd_strong_percentile"
            ]
        )
    )

    msd_extreme = (
        msd_percentile
        >= float(
            settings[
                "msd_extreme_percentile"
            ]
        )
    )

    network_strong = (
        network_percentile
        >= float(
            settings[
                "network_strong_percentile"
            ]
        )
    )

    network_extreme = (
        network_percentile
        >= float(
            settings[
                "network_extreme_percentile"
            ]
        )
    )

    n_ifg = (
        msd_percentile.size
    )

    candidate = np.zeros(
        n_ifg,
        dtype=bool,
    )

    reasons: list[list[str]] = [
        []
        for _ in range(
            n_ifg
        )
    ]

    for j in range(
        n_ifg
    ):

        rr: list[str] = []

        # A truly extreme network inconsistency can stand alone.
        if network_extreme[j]:
            rr.append(
                "network_extreme"
            )

        # A strong but non-extreme network anomaly needs
        # support from an independent family.
        if (
            network_strong[j]
            and grid_family[j]
        ):
            rr.append(
                "network_strong+grid"
            )

        if (
            network_strong[j]
            and msd_strong[j]
        ):
            rr.append(
                "network_strong+unwrap"
            )

        # Extremely rough unwrapping + strong pre-unwrapping
        # GRID evidence is independently sufficient.
        if (
            msd_extreme[j]
            and grid_strong[j]
        ):
            rr.append(
                "unwrap_extreme+grid_strong"
            )

        # De-duplicate reasons while preserving order.
        rr = list(
            dict.fromkeys(rr)
        )

        reasons[j] = rr

        candidate[j] = bool(
            rr
        )

    return (
        candidate,
        reasons,
        msd_strong,
        msd_extreme,
        network_strong,
        network_extreme,
    )


def _component_count(
    edges: np.ndarray,
    keep: np.ndarray,
) -> int:

    e = np.asarray(
        edges,
        dtype=np.int64,
    )

    keep = np.asarray(
        keep,
        dtype=bool,
    )

    nodes = np.unique(
        e.reshape(-1)
    )

    adjacency = {
        int(node): []
        for node in nodes
    }

    for j in np.flatnonzero(
        keep
    ):
        a = int(
            e[j, 0]
        )

        b = int(
            e[j, 1]
        )

        adjacency[a].append(b)
        adjacency[b].append(a)

    seen: set[int] = set()

    components = 0

    for raw_node in nodes:

        node = int(
            raw_node
        )

        if node in seen:
            continue

        components += 1

        stack = [
            node
        ]

        seen.add(
            node
        )

        while stack:

            u = stack.pop()

            for v in adjacency[
                u
            ]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

    return components


def _protect_network(
    *,
    edges: np.ndarray,
    candidate_order: np.ndarray,
    preserve_network: bool,
) -> tuple[
    np.ndarray,
    list[int],
    list[int],
]:

    edges = np.asarray(
        edges,
        dtype=np.int64,
    )

    n_ifg = edges.shape[0]

    keep = np.ones(
        n_ifg,
        dtype=bool,
    )

    base_components = (
        _component_count(
            edges,
            keep,
        )
    )

    dropped: list[int] = []

    protected: list[int] = []

    for zero_ix in np.asarray(
        candidate_order,
        dtype=np.int64,
    ):

        proposed = keep.copy()

        proposed[
            zero_ix
        ] = False

        if (
            preserve_network
            and _component_count(
                edges,
                proposed,
            )
            > base_components
        ):
            protected.append(
                int(
                    zero_ix + 1
                )
            )

            continue

        keep = proposed

        dropped.append(
            int(
                zero_ix + 1
            )
        )

    return (
        keep,
        dropped,
        protected,
    )


def run_final_ifg_qc(
    root: Path,
    *,
    ph_uw_sb: np.ndarray,
    msd: np.ndarray,
    ifgday_ix: np.ndarray,
    settings: dict[str, Any],
    progress: bool = True,
) -> dict[str, Any]:

    root = Path(
        root
    ).expanduser().resolve()

    ph_uw_sb = np.asarray(
        ph_uw_sb,
        dtype=np.float32,
    )

    if ph_uw_sb.ndim != 2:
        raise FinalIFGQCError(
            "ph_uw_sb must be 2-D"
        )

    n_ps, n_ifg = (
        ph_uw_sb.shape
    )

    msd = np.asarray(
        msd,
        dtype=np.float64,
    ).reshape(-1)

    if msd.size != n_ifg:
        raise FinalIFGQCError(
            "MSD/IFG size mismatch: "
            f"{msd.size} vs {n_ifg}"
        )

    edges = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    if (
        edges.ndim == 2
        and edges.shape == (
            2,
            n_ifg,
        )
    ):
        edges = edges.T

    if edges.shape != (
        n_ifg,
        2,
    ):
        raise FinalIFGQCError(
            "ifgday_ix shape mismatch: "
            f"{edges.shape}"
        )

    residual_path = (
        root
        / "phuw_sb_res2.mat"
    )

    if not residual_path.exists():
        raise FinalIFGQCError(
            "first-pass phuw_sb_res2.mat "
            "is required before FINAL IFG-QC"
        )

    payload = read_mat_variables(
        residual_path,
        ("ph_res",),
    )

    ph_res = np.asarray(
        payload.get(
            "ph_res"
        ),
        dtype=np.float32,
    )

    if ph_res.shape != ph_uw_sb.shape:

        if ph_res.T.shape == ph_uw_sb.shape:
            ph_res = ph_res.T
        else:
            raise FinalIFGQCError(
                "ph_res shape mismatch: "
                f"{ph_res.shape} vs "
                f"{ph_uw_sb.shape}"
            )

    print(
        "[IFG_FINAL_QC] "
        f"post-unwrapped audit: "
        f"PS={n_ps:,}, IFG={n_ifg}",
        flush=True,
    )

    net_rms = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    net_mad = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    net_p95 = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    jump_pi_fraction = np.full(
        n_ifg,
        np.nan,
        dtype=np.float64,
    )

    requested_chunk = int(
        settings[
            "chunk_ifg"
        ]
    )

    chunk = turbo_final_qc_chunk(
        n_ps=n_ps,
        requested=requested_chunk,
    )

    print(
        "[IFG_FINAL_QC][TURBO] "
        f"vector_chunk={chunk}, "
        f"PS={n_ps:,}",
        flush=True,
    )

    for start in range(
        0,
        n_ifg,
        chunk,
    ):

        stop = min(
            n_ifg,
            start + chunk,
        )

        cols = np.arange(
            start,
            stop,
        )

        # ------------------------------------------------------
        # Vectorised residual block
        # ------------------------------------------------------

        residual = (
            ph_uw_sb[
                :,
                cols,
            ].astype(
                np.float64,
                copy=False,
            )
            -
            ph_res[
                :,
                cols,
            ].astype(
                np.float64,
                copy=False,
            )
        )

        center = np.nanmedian(
            residual,
            axis=0,
        )

        residual -= center[
            None,
            :
        ]

        finite = np.isfinite(
            residual
        )

        count = np.sum(
            finite,
            axis=0,
        )

        good = (
            count >= 100
        )

        # Avoid NaN contamination in the RMS numerator.
        sq = np.where(
            finite,
            residual
            * residual,
            0.0,
        )

        rms_block = np.full(
            cols.size,
            np.nan,
            dtype=np.float64,
        )

        rms_block[good] = np.sqrt(
            np.sum(
                sq[
                    :,
                    good,
                ],
                axis=0,
            )
            / count[
                good
            ]
        )

        absolute = np.abs(
            residual
        )

        absolute[
            ~finite
        ] = np.nan

        mad_block = (
            1.4826
            * np.nanmedian(
                absolute,
                axis=0,
            )
        )

        p95_block = np.nanquantile(
            absolute,
            0.95,
            axis=0,
        )

        jump_count = np.sum(
            finite
            & (
                absolute > np.pi
            ),
            axis=0,
        )

        jump_block = np.full(
            cols.size,
            np.nan,
            dtype=np.float64,
        )

        jump_block[good] = (
            jump_count[
                good
            ]
            / count[
                good
            ]
        )

        mad_block[
            ~good
        ] = np.nan

        p95_block[
            ~good
        ] = np.nan

        net_rms[
            cols
        ] = rms_block

        net_mad[
            cols
        ] = mad_block

        net_p95[
            cols
        ] = p95_block

        jump_pi_fraction[
            cols
        ] = jump_block

        if (
            progress
            and (
                stop % max(
                    80,
                    chunk,
                )
                < chunk
                or stop == n_ifg
            )
        ):

            print(
                "[IFG_FINAL_QC] "
                f"network residual "
                f"{stop}/{n_ifg}",
                flush=True,
            )

    # ----------------------------------------------------------
    # Three independent evidence families
    # ----------------------------------------------------------

    z_msd = _robust_high_z(
        msd
    )

    msd_percentile = (
        _percentile_fraction(
            msd
        )
    )

    z_rms = _robust_high_z(
        net_rms
    )

    z_mad = _robust_high_z(
        net_mad
    )

    z_jump = _robust_high_z(
        jump_pi_fraction
    )

    # RMS / MAD / jump are correlated internal views of the
    # same network-consistency phenomenon, therefore they form
    # ONE family.
    network_family_z = np.nanmedian(
        np.column_stack(
            (
                z_rms,
                z_mad,
                z_jump,
            )
        ),
        axis=1,
    )

    network_percentile = (
        _percentile_fraction(
            network_family_z
        )
    )

    (
        grid_family,
        grid_strong,
        grid_bad_family_count,
        grid_extreme_family_count,
    ) = _read_grid_flags(
        root,
        n_ifg,
    )

    (
        candidate,
        reasons,
        msd_strong,
        msd_extreme,
        network_strong,
        network_extreme,
    ) = _candidate_flags(
        grid_family=grid_family,
        grid_strong=grid_strong,
        msd_percentile=msd_percentile,
        network_percentile=network_percentile,
        settings=settings,
    )

    candidate_ix = np.flatnonzero(
        candidate
    )

    candidate_count = int(
        candidate_ix.size
    )

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
        candidate_count > 0
        and max_drop_count < 1
    ):
        max_drop_count = 1

    cap_hit = bool(
        candidate_count
        > max_drop_count
    )

    if (
        cap_hit
        and bool(
            settings[
                "fail_on_cap"
            ]
        )
    ):
        summary = {
            "method": METHOD,
            "status": "candidate_cap_exceeded",
            "n_ifg": int(n_ifg),
            "candidate_count":
                candidate_count,
            "max_drop_count":
                max_drop_count,
            "candidate_fraction":
                candidate_count
                / float(n_ifg),
        }

        (
            root
            / "final_ifg_qc_selection.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise FinalIFGQCError(
            "FINAL IFG-QC candidate count "
            f"{candidate_count} exceeds "
            f"safety cap {max_drop_count}; "
            "automatic selection aborted"
        )

    # Candidate order only matters when graph protection is
    # required. Strongest network evidence is considered first.
    candidate_order = np.asarray(
        sorted(
            candidate_ix.tolist(),
            key=lambda j: (
                bool(
                    network_extreme[j]
                ),
                float(
                    network_percentile[
                        j
                    ]
                ),
                bool(
                    msd_extreme[j]
                ),
                float(
                    msd_percentile[
                        j
                    ]
                ),
                bool(
                    grid_strong[j]
                ),
            ),
            reverse=True,
        ),
        dtype=np.int64,
    )

    if (
        cap_hit
        and not bool(
            settings[
                "fail_on_cap"
            ]
        )
    ):
        candidate_order = (
            candidate_order[
                :max_drop_count
            ]
        )

    (
        keep,
        dropped,
        protected,
    ) = _protect_network(
        edges=edges,
        candidate_order=candidate_order,
        preserve_network=bool(
            settings[
                "preserve_network"
            ]
        ),
    )

    dropped_set = set(
        dropped
    )

    protected_set = set(
        protected
    )

    # ----------------------------------------------------------
    # Full audit CSV
    # ----------------------------------------------------------

    audit_path = (
        root
        / "final_ifg_qc_audit.csv"
    )

    with audit_path.open(
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
                "grid_family",
                "grid_strong",
                "grid_bad_family_count",
                "grid_extreme_family_count",
                "msd",
                "msd_z",
                "msd_percentile",
                "msd_strong",
                "msd_extreme",
                "network_rms",
                "network_rms_z",
                "network_mad",
                "network_mad_z",
                "network_p95",
                "jump_pi_fraction",
                "jump_pi_z",
                "network_family_z",
                "network_family_percentile",
                "network_strong",
                "network_extreme",
                "candidate",
                "decision",
                "reason",
            ]
        )

        for j in range(
            n_ifg
        ):

            idx = j + 1

            if idx in dropped_set:
                decision = "drop"
            elif idx in protected_set:
                decision = (
                    "protected_network"
                )
            else:
                decision = "keep"

            writer.writerow(
                [
                    idx,
                    int(
                        grid_family[j]
                    ),
                    int(
                        grid_strong[j]
                    ),
                    int(
                        grid_bad_family_count[
                            j
                        ]
                    ),
                    int(
                        grid_extreme_family_count[
                            j
                        ]
                    ),
                    float(
                        msd[j]
                    ),
                    float(
                        z_msd[j]
                    ),
                    float(
                        msd_percentile[j]
                    ),
                    int(
                        msd_strong[j]
                    ),
                    int(
                        msd_extreme[j]
                    ),
                    float(
                        net_rms[j]
                    ),
                    float(
                        z_rms[j]
                    ),
                    float(
                        net_mad[j]
                    ),
                    float(
                        z_mad[j]
                    ),
                    float(
                        net_p95[j]
                    ),
                    float(
                        jump_pi_fraction[j]
                    ),
                    float(
                        z_jump[j]
                    ),
                    float(
                        network_family_z[j]
                    ),
                    float(
                        network_percentile[j]
                    ),
                    int(
                        network_strong[j]
                    ),
                    int(
                        network_extreme[j]
                    ),
                    int(
                        candidate[j]
                    ),
                    decision,
                    ";".join(
                        reasons[j]
                    )
                    if reasons[j]
                    else "keep",
                ]
            )

    # ----------------------------------------------------------
    # Summary JSON
    # ----------------------------------------------------------

    selection_path = (
        root
        / "final_ifg_qc_selection.json"
    )

    summary = {
        "method":
            METHOD,

        "status":
            "ok",

        "n_ifg":
            int(n_ifg),

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_count
            / float(n_ifg),

        "max_drop_count":
            int(
                max_drop_count
            ),

        "cap_hit":
            bool(
                cap_hit
            ),

        "n_ifg_dropped":
            len(
                dropped
            ),

        "n_ifg_retained":
            int(
                np.count_nonzero(
                    keep
                )
            ),

        "protected_network_count":
            len(
                protected
            ),

        "candidate_ifg_index":
            [
                int(
                    j + 1
                )
                for j in candidate_order
            ],

        "effective_drop_ifg_index":
            [
                int(v)
                for v in dropped
            ],

        "protected_network_ifg_index":
            [
                int(v)
                for v in protected
            ],

        "policy": {
            "independent_families": [
                "GRID_input_quality",
                "MSD_unwrap_quality",
                "NETWORK_consistency",
            ],

            "network_family":
                "median robust-z of "
                "network_rms, "
                "network_mad, "
                "jump_pi_fraction",

            "candidate_rules": [
                "network_extreme",
                "network_strong+grid",
                "network_strong+unwrap",
                "unwrap_extreme+grid_strong",
            ],

            "msd_strong_percentile":
                float(
                    settings[
                        "msd_strong_percentile"
                    ]
                ),

            "msd_extreme_percentile":
                float(
                    settings[
                        "msd_extreme_percentile"
                    ]
                ),

            "network_strong_percentile":
                float(
                    settings[
                        "network_strong_percentile"
                    ]
                ),

            "network_extreme_percentile":
                float(
                    settings[
                        "network_extreme_percentile"
                    ]
                ),

            "max_drop_fraction":
                float(
                    settings[
                        "max_drop_fraction"
                    ]
                ),

            "preserve_network":
                bool(
                    settings[
                        "preserve_network"
                    ]
                ),
        },

        "grid_role":
            "preflag_only",

        "historical_ifg_indices_used":
            False,
    }

    selection_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[IFG_FINAL_QC] "
        f"input={n_ifg}, "
        f"candidates={candidate_count}, "
        f"drop={len(dropped)}, "
        f"keep={np.count_nonzero(keep)}, "
        f"protected_network={len(protected)}, "
        f"cap_hit={cap_hit}",
        flush=True,
    )

    if dropped:
        print(
            "[IFG_FINAL_QC] "
            "effective_drop_ifg_index="
            + ",".join(
                str(v)
                for v in dropped
            ),
            flush=True,
        )

    if protected:
        print(
            "[IFG_FINAL_QC] "
            "protected_network_ifg_index="
            + ",".join(
                str(v)
                for v in protected
            ),
            flush=True,
        )

    if candidate_order.size:

        top = []

        for j in candidate_order[
            :min(
                20,
                candidate_order.size,
            )
        ]:

            top.append(
                f"{j+1}"
                f"(NET={100*network_percentile[j]:.2f}%,"
                f"MSD={100*msd_percentile[j]:.2f}%,"
                f"G={int(grid_family[j])})"
            )

        print(
            "[IFG_FINAL_QC] "
            "top candidates: "
            + " ".join(top),
            flush=True,
        )

    return {
        "method":
            METHOD,

        "candidate_count":
            candidate_count,

        "drop_count":
            len(dropped),

        "keep_count":
            int(
                np.count_nonzero(
                    keep
                )
            ),

        "protected_network_count":
            len(protected),

        "candidate_ifg_index":
            [
                int(
                    j + 1
                )
                for j in candidate_order
            ],

        "effective_drop_ifg_index":
            [
                int(v)
                for v in dropped
            ],

        "protected_network_ifg_index":
            [
                int(v)
                for v in protected
            ],

        "keep_ix":
            [
                int(j)
                for j in np.flatnonzero(
                    keep
                )
            ],

        "cap_hit":
            bool(
                cap_hit
            ),
    }
