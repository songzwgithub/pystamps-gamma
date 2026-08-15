from __future__ import annotations

# === STAGE7_MATLAB_SB_PARITY_V1 ===

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from pystamps.io.mat import (
    read_mat,
    read_mat_variables,
    write_mat,
)
from pystamps.pipeline.stage6_sbas import (
    _stage6_reference_indices,
)


class Stage7SbasError(RuntimeError):
    pass


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    a = np.asarray(value)
    if a.size == 0:
        return float(default)
    return float(a.reshape(-1)[0])


def _mat_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    a = np.asarray(value)

    if a.size == 0:
        return default

    if a.dtype.kind in {"U", "S"}:
        return "".join(
            str(v)
            for v in a.reshape(-1)
        ).strip()

    if a.dtype == object:
        return str(
            a.reshape(-1)[0]
        ).strip()

    return str(
        a.reshape(-1)[0]
    ).strip()


def _as_rows(
    value: Any,
    rows: int,
    name: str,
    dtype,
) -> np.ndarray:

    a = np.asarray(value)

    if a.size == 0:
        raise Stage7SbasError(
            f"{name} is empty"
        )

    a = np.squeeze(a)

    if a.ndim == 1:
        if rows == 1:
            a = a.reshape(1, -1)
        elif a.size % rows == 0:
            a = a.reshape(rows, -1)
        else:
            raise Stage7SbasError(
                f"{name} cannot be reshaped"
            )

    if a.ndim != 2:
        raise Stage7SbasError(
            f"{name} must be 2-D"
        )

    if (
        a.shape[0] != rows
        and a.shape[1] == rows
    ):
        a = a.T

    if a.shape[0] != rows:
        raise Stage7SbasError(
            f"{name} shape={a.shape}; "
            f"expected first dimension {rows}"
        )

    return np.asarray(
        a,
        dtype=dtype,
    )


def _drop_indices(
    value: Any,
    nmax: int,
) -> np.ndarray:

    if value is None:
        return np.empty(
            0,
            dtype=np.int64,
        )

    a = np.asarray(
        value
    ).reshape(-1)

    if a.size == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    a = np.rint(
        a
    ).astype(np.int64)

    a = a[
        (a >= 1)
        & (a <= nmax)
    ]

    return np.unique(a)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def _network_matrix(
    n_image: int,
    ifgday_ix: np.ndarray,
) -> np.ndarray:

    n_ifg = ifgday_ix.shape[0]

    G = np.zeros(
        (n_ifg, n_image),
        dtype=np.float64,
    )

    rows = np.arange(
        n_ifg,
        dtype=np.int64,
    )

    G[
        rows,
        ifgday_ix[:, 0] - 1,
    ] = -1.0

    G[
        rows,
        ifgday_ix[:, 1] - 1,
    ] = +1.0

    return G


def _gls_projector(
    A: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Exact full-covariance GLS projector:

        X = (A' C^-1 A)^-1 A' C^-1 Y

    Returned matrix P satisfies:

        coeff = Y_row @ P.T
    """

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    C = np.asarray(
        covariance,
        dtype=np.float64,
    )

    if C.shape != (
        A.shape[0],
        A.shape[0],
    ):
        raise Stage7SbasError(
            "GLS covariance/design mismatch: "
            f"A={A.shape}, C={C.shape}"
        )

    CiA = np.linalg.solve(
        C,
        A,
    )

    normal = (
        A.T
        @ CiA
    )

    if np.linalg.matrix_rank(
        normal
    ) != normal.shape[0]:
        raise Stage7SbasError(
            "GLS normal matrix rank deficient"
        )

    return np.linalg.solve(
        normal,
        CiA.T,
    )


def _ols_projector(
    A: np.ndarray,
) -> np.ndarray:
    """
    MATLAB backslash / lscov(identity) equivalent
    for a full-column-rank shared design.
    """

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    if np.linalg.matrix_rank(A) != A.shape[1]:
        raise Stage7SbasError(
            f"OLS design rank deficient: "
            f"{np.linalg.matrix_rank(A)}/{A.shape[1]}"
        )

    return np.linalg.pinv(A)


def _resolve_triangle(
    explicit: str | None,
) -> str | None:

    if explicit:
        p = Path(explicit).expanduser()

        if p.exists():
            return str(p.resolve())

        q = shutil.which(explicit)

        if q:
            return q

    return shutil.which("triangle")


def _triangle_edges_external(
    xy: np.ndarray,
    work_dir: Path,
    triangle_exe: str,
) -> np.ndarray:
    """
    Reproduce ps_smooth_scla Linux path:
      triangle -e scla.1.node
      read scla.2.edge

    MATLAB writes coordinates using %f -> six decimals,
    so do the same here.
    """

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    node = work_dir / "scla.1.node"
    edge = work_dir / "scla.2.edge"
    log = work_dir / "triangle_scla.log"

    for p in work_dir.glob("scla.2.*"):
        p.unlink()

    n_ps = xy.shape[0]

    with node.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{n_ps} 2 0 0\n"
        )

        for i in range(n_ps):

            f.write(
                f"{i+1:d} "
                f"{xy[i,0]:.6f} "
                f"{xy[i,1]:.6f}\n"
            )

    with log.open(
        "w",
        encoding="utf-8",
    ) as lf:

        subprocess.run(
            [
                triangle_exe,
                "-e",
                node.name,
            ],
            cwd=work_dir,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not edge.exists():
        raise Stage7SbasError(
            f"triangle did not create {edge}"
        )

    with edge.open(
        "r",
        encoding="utf-8",
    ) as f:

        header = f.readline().split()

    if not header:
        raise Stage7SbasError(
            "empty Triangle edge file"
        )

    n_edge = int(header[0])

    edgs = np.loadtxt(
        edge,
        dtype=np.int64,
        skiprows=1,
        max_rows=n_edge,
        usecols=(1, 2),
    )

    if edgs.ndim == 1:
        edgs = edgs.reshape(1, 2)

    if edgs.shape != (
        n_edge,
        2,
    ):
        raise Stage7SbasError(
            f"Triangle edge shape "
            f"{edgs.shape}; expected "
            f"({n_edge},2)"
        )

    return edgs


def _triangle_edges_scipy(
    xy: np.ndarray,
) -> np.ndarray:
    """
    Fallback corresponding to MATLAB delaunay +
    triangulation(...); edges(...).
    """

    tri = Delaunay(
        np.asarray(
            xy,
            dtype=np.float64,
        )
    )

    simp = np.asarray(
        tri.simplices,
        dtype=np.int64,
    )

    e = np.vstack(
        (
            simp[:, [0, 1]],
            simp[:, [1, 2]],
            simp[:, [2, 0]],
        )
    )

    e.sort(
        axis=1
    )

    e = np.unique(
        e,
        axis=0,
    )

    return e + 1


def _build_edges(
    xy: np.ndarray,
    root: Path,
    triangle_path: str | None,
) -> tuple[np.ndarray, str]:

    exe = _resolve_triangle(
        triangle_path
    )

    if exe:

        work = (
            root
            / "_stage7_triangle_work"
        )

        edgs = _triangle_edges_external(
            xy,
            work,
            exe,
        )

        return (
            edgs,
            f"triangle:{exe}",
        )

    return (
        _triangle_edges_scipy(xy),
        "scipy_delaunay",
    )


def _smooth_neighbor_envelope(
    K: np.ndarray,
    C: np.ndarray,
    edgs_1based: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact vectorised equivalent of ps_smooth_scla:

    For each PS:
      find min/max K and C among Delaunay neighbours;
      clip only values lying outside neighbour envelope.
    """

    K = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(-1)

    C = np.asarray(
        C,
        dtype=np.float64,
    ).reshape(-1)

    n_ps = K.size

    edgs = np.asarray(
        edgs_1based,
        dtype=np.int64,
    )

    u = edgs[:, 0] - 1
    v = edgs[:, 1] - 1

    if (
        np.any(u < 0)
        or np.any(v < 0)
        or np.any(u >= n_ps)
        or np.any(v >= n_ps)
    ):
        raise Stage7SbasError(
            "Delaunay edge index outside PS range"
        )

    kmin = np.full(
        n_ps,
        np.inf,
        dtype=np.float64,
    )
    kmax = np.full(
        n_ps,
        -np.inf,
        dtype=np.float64,
    )

    cmin = np.full(
        n_ps,
        np.inf,
        dtype=np.float64,
    )
    cmax = np.full(
        n_ps,
        -np.inf,
        dtype=np.float64,
    )

    np.minimum.at(
        kmin,
        u,
        K[v],
    )
    np.minimum.at(
        kmin,
        v,
        K[u],
    )

    np.maximum.at(
        kmax,
        u,
        K[v],
    )
    np.maximum.at(
        kmax,
        v,
        K[u],
    )

    np.minimum.at(
        cmin,
        u,
        C[v],
    )
    np.minimum.at(
        cmin,
        v,
        C[u],
    )

    np.maximum.at(
        cmax,
        u,
        C[v],
    )
    np.maximum.at(
        cmax,
        v,
        C[u],
    )

    if (
        np.any(~np.isfinite(kmin))
        or np.any(~np.isfinite(kmax))
    ):
        raise Stage7SbasError(
            "At least one PS has no Delaunay neighbour"
        )

    Ks = K.copy()
    Cs = C.copy()

    hi = Ks > kmax
    lo = Ks < kmin

    Ks[hi] = kmax[hi]
    Ks[lo] = kmin[lo]

    hi = Cs > cmax
    lo = Cs < cmin

    Cs[hi] = cmax[hi]
    Cs[lo] = cmin[lo]

    return Ks, Cs


def _make_phase_model(
    coefficient: np.ndarray,
    baseline: np.ndarray,
    chunk: int,
    label: str,
) -> np.ndarray:

    n_ps, n_col = baseline.shape

    out = np.empty(
        (n_ps, n_col),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_ps,
        chunk,
    ):

        stop = min(
            start + chunk,
            n_ps,
        )

        out[start:stop, :] = (
            coefficient[start:stop, None]
            * baseline[start:stop, :]
        ).astype(np.float32)

        print(
            f"[STAGE7_SBAS][{label}] "
            f"{stop:,}/{n_ps:,} "
            f"({100*stop/n_ps:.1f}%)",
            flush=True,
        )

    return out


def _preflight(
    root: Path,
    triangle_path: str | None = None,
) -> None:

    root = root.expanduser().resolve()

    ps = read_mat(
        root / "ps2.mat"
    )

    parms = read_mat(
        root / "parms.mat"
    )

    n_ps = int(
        round(
            _scalar(
                ps.get("n_ps"),
                0,
            )
        )
    )

    n_ifg = int(
        round(
            _scalar(
                ps.get("n_ifg"),
                0,
            )
        )
    )

    n_image = int(
        round(
            _scalar(
                ps.get("n_image"),
                0,
            )
        )
    )

    master_ix = int(
        round(
            _scalar(
                ps.get("master_ix"),
                1,
            )
        )
    )

    ifgday_ix = np.asarray(
        ps["ifgday_ix"],
        dtype=np.int64,
    )

    if ifgday_ix.shape == (
        2,
        n_ifg,
    ):
        ifgday_ix = ifgday_ix.T

    sb = read_mat_variables(
        root / "phuw_sb2.mat",
        ("ph_uw",),
    )

    sm = read_mat(
        root / "phuw2.mat"
    )

    bp = read_mat_variables(
        root / "bp2.mat",
        ("bperp_mat",),
    )

    cov = read_mat_variables(
        root / "phuw_sb_res2.mat",
        ("sb_cov", "sm_cov"),
    )

    ph_sb = _as_rows(
        sb["ph_uw"],
        n_ps,
        "phuw_sb2.ph_uw",
        np.float32,
    )

    ph_sm = _as_rows(
        sm["ph_uw"],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )

    bperp = _as_rows(
        bp["bperp_mat"],
        n_ps,
        "bp2.bperp_mat",
        np.float32,
    )

    sb_cov = np.asarray(
        cov["sb_cov"],
        dtype=np.float64,
    )

    sm_cov = np.asarray(
        cov["sm_cov"],
        dtype=np.float64,
    )

    drop = _drop_indices(
        parms.get(
            "drop_ifg_index"
        ),
        n_ifg,
    )

    sb_scla_drop = _drop_indices(
        parms.get(
            "sb_scla_drop_index"
        ),
        n_ifg,
    )

    scla_drop = _drop_indices(
        parms.get(
            "scla_drop_index"
        ),
        n_image,
    )

    use_sb = np.setdiff1d(
        np.arange(
            1,
            n_ifg + 1,
            dtype=np.int64,
        ),
        np.union1d(
            drop,
            sb_scla_drop,
        ),
    )

    ref = _stage6_reference_indices(
        ps,
        parms,
        n_ps,
    )

    G = _network_matrix(
        n_image,
        ifgday_ix,
    )

    sm_ix_raw = sm.get(
        "unwrap_ifg_index_sm"
    )

    if (
        sm_ix_raw is None
        or np.asarray(sm_ix_raw).size == 0
    ):
        sm_ix = np.arange(
            1,
            n_image + 1,
            dtype=np.int64,
        )
        sm_source = (
            "missing -> all images; "
            "equivalent for this connected network"
        )
    else:
        sm_ix = _drop_indices(
            sm_ix_raw,
            n_image,
        )
        sm_source = (
            "phuw2.unwrap_ifg_index_sm"
        )

    sm_ix = np.setdiff1d(
        sm_ix,
        scla_drop,
    )

    sm_no_master = np.setdiff1d(
        sm_ix,
        np.asarray(
            [master_ix],
            dtype=np.int64,
        ),
    )

    rank_base = int(
        np.linalg.matrix_rank(
            G[:, sm_no_master - 1]
        )
    )

    print("=" * 84)
    print("STAGE 7 MATLAB SB PARITY PREFLIGHT")
    print("=" * 84)

    print("n_ps                     :", n_ps)
    print("n_ifg                    :", n_ifg)
    print("n_image                  :", n_image)
    print("master_ix                :", master_ix)

    print()
    print("phuw_sb2                 :", ph_sb.shape)
    print("phuw2                    :", ph_sm.shape)
    print("bp2                      :", bperp.shape)
    print("sb_cov                   :", sb_cov.shape)
    print("sm_cov                   :", sm_cov.shape)

    print()
    print("drop_ifg_index           :", drop.tolist())
    print("sb_scla_drop_index       :", sb_scla_drop.tolist())
    print("scla_drop_index          :", scla_drop.tolist())

    print()
    print("SB SCLA IFGs             :", use_sb.size)
    print("reference PS             :", ref.size)
    print("SM index source          :", sm_source)
    print("SM images excluding master:", sm_no_master.size)
    print("SM baseline G rank       :", rank_base)

    print()
    print(
        "scla_method              :",
        _mat_text(
            parms.get("scla_method"),
            "L2",
        ),
    )
    print(
        "scla_deramp              :",
        _mat_text(
            parms.get("scla_deramp"),
            "n",
        ),
    )
    print(
        "subtr_tropo              :",
        _mat_text(
            parms.get("subtr_tropo"),
            "n",
        ),
    )

    exe = _resolve_triangle(
        triangle_path
    )

    print(
        "smooth backend           :",
        exe if exe else "scipy_delaunay",
    )

    if ph_sb.shape != (
        n_ps,
        n_ifg,
    ):
        raise Stage7SbasError(
            "phuw_sb2 shape mismatch"
        )

    if ph_sm.shape != (
        n_ps,
        n_image,
    ):
        raise Stage7SbasError(
            "phuw2 shape mismatch"
        )

    if bperp.shape != (
        n_ps,
        n_ifg,
    ):
        raise Stage7SbasError(
            "bp2 shape mismatch"
        )

    if sb_cov.shape != (
        n_ifg,
        n_ifg,
    ):
        raise Stage7SbasError(
            "sb_cov shape mismatch"
        )

    if sm_cov.shape != (
        n_image,
        n_image,
    ):
        raise Stage7SbasError(
            "sm_cov shape mismatch"
        )

    if use_sb.size < 4:
        raise Stage7SbasError(
            "fewer than four SB IFGs"
        )

    if rank_base != sm_no_master.size:
        raise Stage7SbasError(
            "single-master baseline "
            "reconstruction rank deficient"
        )

    if (
        _mat_text(
            parms.get("scla_method"),
            "L2",
        ).upper()
        != "L2"
    ):
        raise Stage7SbasError(
            "Parity path currently requires "
            "scla_method='L2'"
        )

    if (
        _mat_text(
            parms.get("scla_deramp"),
            "n",
        ).lower()
        != "n"
    ):
        raise Stage7SbasError(
            "Parity path requires "
            "scla_deramp='n'"
        )

    if (
        _mat_text(
            parms.get("subtr_tropo"),
            "n",
        ).lower()
        != "n"
    ):
        raise Stage7SbasError(
            "Parity baseline requires "
            "subtr_tropo='n'"
        )

    dropped_max = np.max(
        np.abs(
            ph_sb[:, drop - 1]
        ),
        axis=0,
    )

    if np.any(
        dropped_max != 0
    ):
        raise Stage7SbasError(
            "Dropped Stage6 IFG columns "
            "are not zero"
        )

    print()
    print(
        "Expected outputs:"
    )
    print(
        "  scla_sb2.mat"
    )
    print(
        "  scla_smooth_sb2.mat"
    )
    print(
        "  scla2.mat"
    )

    print()
    print(
        "STAGE 7 PREFLIGHT: PASS"
    )


def stage7_sbas_calc_scla(
    dataset_root: Path,
    backend: str = "auto",
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]]
    | None = None,
    triangle_path: str | None = None,
) -> str:

    del (
        backend,
        enable_mat_cache,
        io_workers,
        mat_cache,
    )

    root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
    )

    started = time.perf_counter()

    debug_path = (
        root
        / "stage7_sbas_debug.json"
    )

    chunk = (
        int(chunk_ps)
        if int(chunk_ps) > 0
        else int(
            os.environ.get(
                "PYSTAMPS_STAGE7_CHUNK_PS",
                "16384",
            )
        )
    )

    chunk = max(
        256,
        chunk,
    )

    try:

        # ==============================================================
        # LOAD / GUARDS
        # ==============================================================

        ps = read_mat(
            root / "ps2.mat"
        )

        parms = read_mat(
            root / "parms.mat"
        )

        n_ps = int(
            round(
                _scalar(
                    ps.get("n_ps"),
                    0,
                )
            )
        )

        n_ifg = int(
            round(
                _scalar(
                    ps.get("n_ifg"),
                    0,
                )
            )
        )

        n_image = int(
            round(
                _scalar(
                    ps.get("n_image"),
                    0,
                )
            )
        )

        master_ix = int(
            round(
                _scalar(
                    ps.get("master_ix"),
                    1,
                )
            )
        )

        if (
            _mat_text(
                parms.get(
                    "small_baseline_flag"
                ),
                "n",
            ).lower()
            != "y"
        ):
            raise Stage7SbasError(
                "Stage7 SB path requires "
                "small_baseline_flag='y'"
            )

        if (
            _mat_text(
                parms.get("scla_method"),
                "L2",
            ).upper()
            != "L2"
        ):
            raise Stage7SbasError(
                "MATLAB parity path requires "
                "scla_method='L2'"
            )

        if (
            _mat_text(
                parms.get("scla_deramp"),
                "n",
            ).lower()
            != "n"
        ):
            raise Stage7SbasError(
                "MATLAB parity path requires "
                "scla_deramp='n'"
            )

        if (
            _mat_text(
                parms.get("subtr_tropo"),
                "n",
            ).lower()
            != "n"
        ):
            raise Stage7SbasError(
                "MATLAB parity baseline requires "
                "subtr_tropo='n'"
            )

        # Old aps*.mat files are treated specially by MATLAB.
        # Do not silently ignore them in parity mode.
        for old_aps in (
            "aps_sb2.mat",
            "aps2.mat",
        ):
            if (
                root / old_aps
            ).exists():
                raise Stage7SbasError(
                    f"{old_aps} exists; "
                    "remove/archive it before "
                    "MATLAB parity Stage7"
                )

        ifgday_ix = np.asarray(
            ps["ifgday_ix"],
            dtype=np.int64,
        )

        if ifgday_ix.shape == (
            2,
            n_ifg,
        ):
            ifgday_ix = (
                ifgday_ix.T
            )

        if ifgday_ix.shape != (
            n_ifg,
            2,
        ):
            raise Stage7SbasError(
                "ifgday_ix shape mismatch"
            )

        day = np.asarray(
            ps["day"],
            dtype=np.float64,
        ).reshape(-1)

        if day.size != n_image:
            raise Stage7SbasError(
                "ps.day length mismatch"
            )

        xy = _as_rows(
            ps["xy"],
            n_ps,
            "ps2.xy",
            np.float64,
        )

        xy2 = xy[:, 1:3]

        ref_ps = (
            _stage6_reference_indices(
                ps,
                parms,
                n_ps,
            )
        )

        if ref_ps.size == 0:
            raise Stage7SbasError(
                "No reference PS selected"
            )

        drop = _drop_indices(
            parms.get(
                "drop_ifg_index"
            ),
            n_ifg,
        )

        sb_scla_drop = (
            _drop_indices(
                parms.get(
                    "sb_scla_drop_index"
                ),
                n_ifg,
            )
        )

        scla_drop = (
            _drop_indices(
                parms.get(
                    "scla_drop_index"
                ),
                n_image,
            )
        )

        G_ifg = _network_matrix(
            n_image,
            ifgday_ix,
        )

        bp = read_mat_variables(
            root / "bp2.mat",
            ("bperp_mat",),
        )

        bperp_ifg = _as_rows(
            bp["bperp_mat"],
            n_ps,
            "bp2.bperp_mat",
            np.float32,
        )

        if bperp_ifg.shape != (
            n_ps,
            n_ifg,
        ):
            raise Stage7SbasError(
                "bp2.bperp_mat shape mismatch"
            )

        cov_payload = (
            read_mat_variables(
                root
                / "phuw_sb_res2.mat",
                (
                    "sb_cov",
                    "sm_cov",
                ),
            )
        )

        sb_cov = np.asarray(
            cov_payload["sb_cov"],
            dtype=np.float64,
        )

        sm_cov = np.asarray(
            cov_payload["sm_cov"],
            dtype=np.float64,
        )

        if sb_cov.shape != (
            n_ifg,
            n_ifg,
        ):
            raise Stage7SbasError(
                "sb_cov shape mismatch"
            )

        if sm_cov.shape != (
            n_image,
            n_image,
        ):
            raise Stage7SbasError(
                "sm_cov shape mismatch"
            )

        ph_ramp = np.empty(
            (0, 0),
            dtype=np.float64,
        )

        # ==============================================================
        # PASS 1
        #
        # Official:
        #   ps_calc_scla(1,1)
        #
        # Inputs:
        #   phuw_sb2
        #   bp2
        #   full sb_cov
        #
        # Design:
        #   [1, mean(bperp), delta_t]
        #
        # C_ps_uw = 0 for SB pass
        # ==============================================================

        print()
        print("=" * 78)
        print(
            "STAGE7 PASS 1/3: "
            "ps_calc_scla(1,1)"
        )
        print("=" * 78)

        sb_payload = (
            read_mat_variables(
                root / "phuw_sb2.mat",
                ("ph_uw",),
            )
        )

        ph_sb = _as_rows(
            sb_payload["ph_uw"],
            n_ps,
            "phuw_sb2.ph_uw",
            np.float32,
        )

        if ph_sb.shape != (
            n_ps,
            n_ifg,
        ):
            raise Stage7SbasError(
                "phuw_sb2 shape mismatch"
            )

        unwrap_sb = np.setdiff1d(
            np.arange(
                1,
                n_ifg + 1,
                dtype=np.int64,
            ),
            np.union1d(
                drop,
                sb_scla_drop,
            ),
        )

        use0 = unwrap_sb - 1

        if use0.size < 4:
            raise Stage7SbasError(
                "fewer than four IFGs "
                "for SB SCLA"
            )

        ref_mean_sb = np.nanmean(
            ph_sb[
                ref_ps,
                :,
            ].astype(np.float64),
            axis=0,
        )

        # mean baseline for each retained IFG
        bsum = np.zeros(
            use0.size,
            dtype=np.float64,
        )

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            bsum += np.sum(
                bperp_ifg[
                    start:stop,
                    :
                ][:, use0],
                axis=0,
                dtype=np.float64,
            )

        bmean = (
            bsum
            / float(n_ps)
        )

        dt_sb = (
            day[
                ifgday_ix[
                    use0,
                    1,
                ] - 1
            ]
            -
            day[
                ifgday_ix[
                    use0,
                    0,
                ] - 1
            ]
        )

        A1 = np.column_stack(
            (
                np.ones(
                    use0.size,
                    dtype=np.float64,
                ),
                bmean,
                dt_sb,
            )
        )

        C1 = sb_cov[
            np.ix_(
                use0,
                use0,
            )
        ]

        P1 = _gls_projector(
            A1,
            C1,
        )

        K1 = np.empty(
            n_ps,
            dtype=np.float64,
        )

        v1 = np.empty(
            n_ps,
            dtype=np.float64,
        )

        t_pass = time.perf_counter()

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            Y = (
                ph_sb[
                    start:stop,
                    :
                ][:, use0]
                .astype(np.float64)
                -
                ref_mean_sb[
                    use0
                ][None, :]
            )

            coeff = (
                Y
                @ P1.T
            )

            K1[
                start:stop
            ] = coeff[:, 1]

            v1[
                start:stop
            ] = coeff[:, 2]

            print(
                "[STAGE7_SBAS][SB_GLS] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        C1_ps = np.zeros(
            n_ps,
            dtype=np.float64,
        )

        ph_scla1 = (
            _make_phase_model(
                K1,
                bperp_ifg,
                chunk,
                "SB_MODEL",
            )
        )

        write_mat(
            root / "scla_sb2.mat",
            {
                "ph_scla":
                    ph_scla1,
                "K_ps_uw":
                    K1.reshape(-1, 1),
                "C_ps_uw":
                    C1_ps.reshape(-1, 1),
                "ph_ramp":
                    ph_ramp,
                "ifg_vcm":
                    sb_cov,
            },
        )

        del ph_scla1
        del ph_sb

        pass1_sec = (
            time.perf_counter()
            - t_pass
        )

        print(
            "[STAGE7_SBAS] "
            f"scla_sb2.mat written; "
            f"IFGs={use0.size}"
        )

        # ==============================================================
        # PASS 2
        #
        # Official:
        #   ps_smooth_scla(1)
        #
        # NOT a spatial low-pass filter.
        # Delaunay-neighbour min/max envelope clipping only.
        # ==============================================================

        print()
        print("=" * 78)
        print(
            "STAGE7 PASS 2/3: "
            "ps_smooth_scla(1)"
        )
        print("=" * 78)

        t_smooth = time.perf_counter()

        edgs, tri_backend = (
            _build_edges(
                xy2,
                root,
                triangle_path,
            )
        )

        print(
            "[STAGE7_SBAS][SMOOTH] "
            f"edges={edgs.shape[0]:,}, "
            f"backend={tri_backend}",
            flush=True,
        )

        K1s, C1s = (
            _smooth_neighbor_envelope(
                K1,
                C1_ps,
                edgs,
            )
        )

        changed_K = int(
            np.count_nonzero(
                K1s != K1
            )
        )

        changed_C = int(
            np.count_nonzero(
                C1s != C1_ps
            )
        )

        ph_scla1s = (
            _make_phase_model(
                K1s,
                bperp_ifg,
                chunk,
                "SB_SMOOTH_MODEL",
            )
        )

        write_mat(
            root
            / "scla_smooth_sb2.mat",
            {
                "K_ps_uw":
                    K1s.reshape(-1, 1),
                "C_ps_uw":
                    C1s.reshape(-1, 1),
                "ph_scla":
                    ph_scla1s,
                "ph_ramp":
                    ph_ramp,
            },
        )

        del ph_scla1s

        smooth_sec = (
            time.perf_counter()
            - t_smooth
        )

        print(
            "[STAGE7_SBAS][SMOOTH] "
            f"K changed={changed_K:,}/{n_ps:,}, "
            f"C changed={changed_C:,}/{n_ps:,}",
            flush=True,
        )

        # ==============================================================
        # PASS 3
        #
        # Official:
        #   ps_calc_scla(0,1)
        #
        # Inputs:
        #   phuw2 (single-master 257 acquisitions)
        #
        # For SB dataset:
        #   reconstruct single-master bperp:
        #
        #       bperp_some = G \ bp.bperp_mat'
        #
        #   K fit:
        #       sequential diff(phuw2)
        #       sequential diff(bperp_sm)
        #       OLS design [1, mean(dbperp), dt]
        #
        #   C fit:
        #       full single-master residual,
        #       full sm_cov
        # ==============================================================

        print()
        print("=" * 78)
        print(
            "STAGE7 PASS 3/3: "
            "ps_calc_scla(0,1)"
        )
        print("=" * 78)

        t_final = time.perf_counter()

        sm_payload = read_mat(
            root / "phuw2.mat"
        )

        ph_sm = _as_rows(
            sm_payload["ph_uw"],
            n_ps,
            "phuw2.ph_uw",
            np.float32,
        )

        if ph_sm.shape != (
            n_ps,
            n_image,
        ):
            raise Stage7SbasError(
                "phuw2 shape mismatch"
            )

        sm_raw = sm_payload.get(
            "unwrap_ifg_index_sm"
        )

        if (
            sm_raw is None
            or np.asarray(
                sm_raw
            ).size == 0
        ):
            unwrap_sm = np.arange(
                1,
                n_image + 1,
                dtype=np.int64,
            )
        else:
            unwrap_sm = (
                _drop_indices(
                    sm_raw,
                    n_image,
                )
            )

        unwrap_sm = np.setdiff1d(
            unwrap_sm,
            scla_drop,
        )

        unwrap_sm = np.setdiff1d(
            unwrap_sm,
            np.asarray(
                [master_ix],
                dtype=np.int64,
            ),
        )

        img0 = unwrap_sm - 1

        if img0.size < 3:
            raise Stage7SbasError(
                "too few single-master "
                "images for final SCLA"
            )

        # --------------------------------------------------------------
        # MATLAB:
        #
        # G = G(:,unwrap_ifg_index);
        # bperp_some = [G\double(bp.bperp_mat')]';
        #
        # Note: this uses the full original SB network rows.
        # --------------------------------------------------------------

        Gbase = G_ifg[
            :,
            img0,
        ]

        rank_base = int(
            np.linalg.matrix_rank(
                Gbase
            )
        )

        if rank_base != img0.size:
            raise Stage7SbasError(
                "single-master baseline "
                "reconstruction rank deficient: "
                f"{rank_base}/{img0.size}"
            )

        Pbase = np.linalg.pinv(
            Gbase
        )

        bperp_sm = np.zeros(
            (n_ps, n_image),
            dtype=np.float32,
        )

        bdiff_sum = np.zeros(
            img0.size - 1,
            dtype=np.float64,
        )

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            bsome = (
                bperp_ifg[
                    start:stop,
                    :
                ].astype(np.float64)
                @ Pbase.T
            )

            bperp_sm[
                start:stop,
                :
            ][:, img0] = (
                bsome.astype(
                    np.float32
                )
            )

            bdiff_sum += np.sum(
                np.diff(
                    bsome,
                    axis=1,
                ),
                axis=0,
                dtype=np.float64,
            )

            print(
                "[STAGE7_SBAS][BPERP_TO_SM] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        mean_bdiff = (
            bdiff_sum
            / float(n_ps)
        )

        ref_mean_sm = np.nanmean(
            ph_sm[
                ref_ps,
                :,
            ].astype(np.float64),
            axis=0,
        )

        day_seq = np.diff(
            day[img0]
        )

        A2 = np.column_stack(
            (
                np.ones(
                    img0.size - 1,
                    dtype=np.float64,
                ),
                mean_bdiff,
                day_seq,
            )
        )

        # Official use_small_baselines==0:
        # ifg_vcm_use = eye(size(ph,2))
        P2 = _ols_projector(
            A2
        )

        K2 = np.empty(
            n_ps,
            dtype=np.float64,
        )

        v2 = np.empty(
            n_ps,
            dtype=np.float64,
        )

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            ph_some = (
                ph_sm[
                    start:stop,
                    :
                ][:, img0]
                .astype(np.float64)
                -
                ref_mean_sm[
                    img0
                ][None, :]
            )

            dph = np.diff(
                ph_some,
                axis=1,
            )

            coeff = (
                dph
                @ P2.T
            )

            K2[
                start:stop
            ] = coeff[:, 1]

            v2[
                start:stop
            ] = coeff[:, 2]

            print(
                "[STAGE7_SBAS][SM_OLS_K] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        # --------------------------------------------------------------
        # Build full 257-column ph_scla.
        # --------------------------------------------------------------

        ph_scla2 = np.empty(
            (n_ps, n_image),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # MATLAB coest_mean_vel=1:
        #
        # Gc = [
        #   ones,
        #   day - master_day
        # ]
        #
        # C_ps_uw = intercept from lscov with sm_cov.
        # --------------------------------------------------------------

        Ac = np.column_stack(
            (
                np.ones(
                    img0.size,
                    dtype=np.float64,
                ),
                day[img0]
                - day[
                    master_ix - 1
                ],
            )
        )

        Csm = sm_cov[
            np.ix_(
                img0,
                img0,
            )
        ]

        Pc = _gls_projector(
            Ac,
            Csm,
        )

        C2 = np.empty(
            n_ps,
            dtype=np.float64,
        )

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            ph_scla_chunk = (
                K2[
                    start:stop,
                    None
                ]
                *
                bperp_sm[
                    start:stop,
                    :
                ].astype(np.float64)
            )

            ph_scla2[
                start:stop,
                :
            ] = (
                ph_scla_chunk
                .astype(np.float32)
            )

            y = (
                ph_sm[
                    start:stop,
                    :
                ][:, img0]
                .astype(np.float64)
                -
                ref_mean_sm[
                    img0
                ][None, :]
                -
                ph_scla_chunk[
                    :,
                    img0,
                ]
            )

            mc = (
                y
                @ Pc.T
            )

            C2[
                start:stop
            ] = mc[:, 0]

            print(
                "[STAGE7_SBAS][SM_GLS_C] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        write_mat(
            root / "scla2.mat",
            {
                "ph_scla":
                    ph_scla2,
                "K_ps_uw":
                    K2.reshape(-1, 1),
                "C_ps_uw":
                    C2.reshape(-1, 1),
                "ph_ramp":
                    ph_ramp,
                "ifg_vcm":
                    sm_cov,
            },
        )

        final_sec = (
            time.perf_counter()
            - t_final
        )

        duration = (
            time.perf_counter()
            - started
        )

        # ==============================================================
        # DEBUG
        # ==============================================================

        debug = {
            "status":
                "completed",

            "implementation":
                "MATLAB_STAMPS_SB_STAGE7_PARITY_V1",

            "duration_sec":
                duration,

            "n_ps":
                n_ps,

            "n_ifg":
                n_ifg,

            "n_image":
                n_image,

            "master_ix":
                master_ix,

            "reference_ps":
                int(ref_ps.size),

            "drop_ifg_index":
                drop.tolist(),

            "sb_scla_drop_index":
                sb_scla_drop.tolist(),

            "scla_drop_index":
                scla_drop.tolist(),

            "pass1": {
                "input":
                    "phuw_sb2.mat",
                "used_ifgs":
                    int(use0.size),
                "design_shape":
                    list(A1.shape),
                "covariance_shape":
                    list(C1.shape),
                "seconds":
                    pass1_sec,
            },

            "smooth": {
                "backend":
                    tri_backend,
                "n_edge":
                    int(edgs.shape[0]),
                "K_changed":
                    changed_K,
                "C_changed":
                    changed_C,
                "seconds":
                    smooth_sec,
            },

            "pass3": {
                "input":
                    "phuw2.mat",
                "images_without_master":
                    int(img0.size),
                "baseline_rank":
                    rank_base,
                "K_design_shape":
                    list(A2.shape),
                "C_design_shape":
                    list(Ac.shape),
                "C_covariance_shape":
                    list(Csm.shape),
                "seconds":
                    final_sec,
            },

            "outputs": [
                "scla_sb2.mat",
                "scla_smooth_sb2.mat",
                "scla2.mat",
            ],
        }

        _write_json(
            debug_path,
            debug,
        )

        if (
            root
            / "_stage7_triangle_work"
        ).exists():

            if os.environ.get(
                "PYSTAMPS_STAGE7_KEEP_WORK",
                "0",
            ).strip().lower() not in {
                "1",
                "true",
                "yes",
                "y",
            }:

                shutil.rmtree(
                    root
                    / "_stage7_triangle_work",
                    ignore_errors=True,
                )

        print()
        print("=" * 78)
        print(
            "STAGE 7 MATLAB SB PARITY COMPLETE"
        )
        print("=" * 78)

        print(
            "scla_sb2.mat        : "
            f"{n_ps} PS × {n_ifg} IFG model"
        )

        print(
            "scla_smooth_sb2.mat : "
            f"K changed {changed_K:,}/{n_ps:,}"
        )

        print(
            "scla2.mat           : "
            f"{n_ps} PS × {n_image} image model"
        )

        print(
            f"duration            : "
            f"{duration:.2f} s"
        )

        return (
            "Stage 7 MATLAB SB parity SCLA "
            f"completed for {n_ps} PS; "
            f"SB IFGs used={use0.size}"
        )

    except Exception as exc:

        _write_json(
            debug_path,
            {
                "status":
                    "failed",
                "implementation":
                    "MATLAB_STAMPS_SB_STAGE7_PARITY_V1",
                "duration_sec":
                    time.perf_counter()
                    - started,
                "exception":
                    f"{type(exc).__name__}: {exc}",
            },
        )

        raise


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "MATLAB StaMPS-compatible "
            "SBAS Stage 7"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--chunk-ps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--triangle",
        default=None,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight:

        _preflight(
            args.dataset,
            triangle_path=args.triangle,
        )

        return 0

    result = (
        stage7_sbas_calc_scla(
            args.dataset,
            chunk_ps=args.chunk_ps,
            triangle_path=args.triangle,
        )
    )

    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
