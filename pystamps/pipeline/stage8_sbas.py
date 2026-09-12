from __future__ import annotations

# === STAGE8_MATLAB_PS_SCN_FILT_PARITY_V1 ===

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, spatial

from pystamps.io.mat import (
    read_mat,
    read_mat_variables,
    write_mat,
)


class Stage8SbasError(RuntimeError):
    pass


def _scalar(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    a = np.asarray(x)
    if a.size == 0:
        return float(default)
    return float(a.reshape(-1)[0])


def _mat_text(x: Any, default: str = "") -> str:
    if x is None:
        return default

    a = np.asarray(x)

    if a.size == 0:
        return default

    if a.dtype.kind in {"U", "S"}:
        return "".join(
            str(v) for v in a.reshape(-1)
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

    a = np.squeeze(
        np.asarray(value)
    )

    if a.ndim == 1:
        if rows == 1:
            a = a.reshape(1, -1)
        elif a.size % rows == 0:
            a = a.reshape(rows, -1)

    if a.ndim != 2:
        raise Stage8SbasError(
            f"{name} must be 2-D; got {a.shape}"
        )

    if (
        a.shape[0] != rows
        and a.shape[1] == rows
    ):
        a = a.T

    if a.shape[0] != rows:
        raise Stage8SbasError(
            f"{name}: shape={a.shape}; "
            f"expected first dimension {rows}"
        )

    return np.asarray(a, dtype=dtype)


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
    work: Path,
    exe: str,
) -> np.ndarray:
    """
    Official ps_scn_filt:
        triangle -e scnfilt.1.node
    """

    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    node = work / "scnfilt.1.node"
    edge = work / "scnfilt.2.edge"
    log = work / "triangle_scn.log"

    for p in work.glob("scnfilt.2.*"):
        p.unlink()

    n_ps = xy.shape[0]

    with node.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{n_ps} 2 0 0\n"
        )

        # MATLAB fprintf('%d %f %f\n',...)
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
            [exe, "-e", node.name],
            cwd=work,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not edge.exists():
        raise Stage8SbasError(
            f"Triangle did not create {edge}"
        )

    with edge.open(
        "r",
        encoding="utf-8",
    ) as f:
        h = f.readline().split()

    if not h:
        raise Stage8SbasError(
            "Empty Triangle edge file"
        )

    n_edge = int(h[0])

    e = np.loadtxt(
        edge,
        dtype=np.int64,
        skiprows=1,
        max_rows=n_edge,
        usecols=(1, 2),
    )

    if e.ndim == 1:
        e = e.reshape(1, 2)

    if e.shape != (n_edge, 2):
        raise Stage8SbasError(
            f"Triangle edge shape {e.shape}; "
            f"expected {(n_edge,2)}"
        )

    return e


def _triangle_edges_scipy(
    xy: np.ndarray,
) -> np.ndarray:

    tri = spatial.Delaunay(
        np.asarray(
            xy,
            dtype=np.float64,
        )
    )

    s = np.asarray(
        tri.simplices,
        dtype=np.int64,
    )

    e = np.vstack(
        (
            s[:, [0, 1]],
            s[:, [1, 2]],
            s[:, [2, 0]],
        )
    )

    e.sort(axis=1)

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

        edges = _triangle_edges_external(
            xy,
            root / "_stage8_triangle_work",
            exe,
        )

        backend = f"triangle:{exe}"

    else:

        # Algebraically valid fallback.
        # Official edge solve result is graph-independent
        # for any connected graph because dph_hpt is an
        # exact node-gradient.
        edges = _triangle_edges_scipy(
            xy
        )

        backend = "scipy_delaunay_algebraic_fallback"

    # Connectivity is essential for the equivalence.
    u = edges[:, 0] - 1
    v = edges[:, 1] - 1

    n_ps = xy.shape[0]

    adj = sparse.coo_matrix(
        (
            np.ones(
                2 * edges.shape[0],
                dtype=np.uint8,
            ),
            (
                np.r_[u, v],
                np.r_[v, u],
            ),
        ),
        shape=(n_ps, n_ps),
    ).tocsr()

    ncomp = sparse.csgraph.connected_components(
        adj,
        directed=False,
        return_labels=False,
    )

    if ncomp != 1:
        raise Stage8SbasError(
            f"Delaunay graph has {ncomp} components"
        )

    return edges, backend


def _temporal_weight_matrix(
    day: np.ndarray,
    master0: int,
    time_win: float,
) -> np.ndarray:

    dt = (
        day[:, None]
        - day[None, :]
    )

    W = np.exp(
        -(dt * dt)
        / (
            2.0
            * time_win
            * time_win
        )
    )

    # Official:
    # weight_factor(master_ix)=0
    W[:, master0] = 0.0

    den = np.sum(
        W,
        axis=1,
    )

    if np.any(den <= 0):
        raise Stage8SbasError(
            "Temporal Gaussian has zero weight sum"
        )

    W /= den[:, None]

    return W


def _preflight(
    root: Path,
    triangle_path: str | None = None,
    phase_file: str = "phuw2.mat",
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
            _scalar(ps.get("n_ps"))
        )
    )

    n_ifg = int(
        round(
            _scalar(ps.get("n_ifg"))
        )
    )

    n_image = int(
        round(
            _scalar(ps.get("n_image"))
        )
    )

    master_ix = int(
        round(
            _scalar(ps.get("master_ix"))
        )
    )

    phuw = read_mat_variables(
        root / phase_file,
        ("ph_uw",),
    )

    scla = read_mat(
        root / "scla2.mat"
    )

    ph = _as_rows(
        phuw["ph_uw"],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )

    ph_scla = _as_rows(
        scla["ph_scla"],
        n_ps,
        "scla2.ph_scla",
        np.float32,
    )

    C = np.asarray(
        scla["C_ps_uw"]
    ).reshape(-1)

    xy = _as_rows(
        ps["xy"],
        n_ps,
        "ps.xy",
        np.float64,
    )

    krig = _mat_text(
        parms.get("scn_kriging_flag"),
        "n",
    ).lower()

    wavelength = _scalar(
        parms.get("scn_wavelength"),
        100.0,
    )

    time_win = _scalar(
        parms.get("scn_time_win"),
        365.0,
    )

    deramp = parms.get(
        "scn_deramp_ifg"
    )

    deramp_empty = (
        deramp is None
        or np.asarray(deramp).size == 0
    )

    exe = _resolve_triangle(
        triangle_path
    )

    print("=" * 88)
    print("STAGE 8 MATLAB ps_scn_filt PARITY PREFLIGHT")
    print("=" * 88)

    print("n_ps                 :", n_ps)
    print("n_ifg                :", n_ifg)
    print("n_image              :", n_image)
    print("master_ix            :", master_ix)

    print()
    print("phuw2.ph_uw          :", ph.shape)
    print("scla2.ph_scla        :", ph_scla.shape)
    print("scla2.C_ps_uw        :", C.shape)

    print()
    print("small_baseline_flag  :", _mat_text(
        parms.get("small_baseline_flag"), "n"
    ))
    print("scn_kriging_flag     :", krig)
    print("scn_wavelength       :", wavelength)
    print("scn_time_win         :", time_win)
    print("scn_deramp_ifg empty :", deramp_empty)

    print()
    print(
        "triangle executable   :",
        exe if exe else "NOT FOUND",
    )
    print(
        "edge backend          :",
        (
            "official Triangle"
            if exe
            else
            "SciPy Delaunay algebraic fallback"
        ),
    )

    print()
    print("Official output:")
    print("  scn2.mat")
    print("    ph_scn_slave")
    print("    ph_hpt")
    print("    ph_ramp")

    if (
        _mat_text(
            parms.get("small_baseline_flag"),
            "n",
        ).lower()
        != "y"
    ):
        raise Stage8SbasError(
            "Expected small_baseline_flag='y'"
        )

    if krig != "n":
        raise Stage8SbasError(
            "Parity implementation is ps_scn_filt, "
            "not ps_scn_filt_krig"
        )

    if not deramp_empty:
        raise Stage8SbasError(
            "Current parity baseline requires "
            "official default scn_deramp_ifg=[]"
        )

    if ph.shape != (
        n_ps,
        n_image,
    ):
        raise Stage8SbasError(
            "phuw2 shape mismatch"
        )

    if ph_scla.shape != (
        n_ps,
        n_image,
    ):
        raise Stage8SbasError(
            "scla2 shape mismatch"
        )

    if C.size != n_ps:
        raise Stage8SbasError(
            "C_ps_uw length mismatch"
        )

    if xy.shape[1] < 3:
        raise Stage8SbasError(
            "ps.xy requires id,x,y"
        )

    print()
    print("STAGE 8 PREFLIGHT: PASS")


def stage8_sbas_filter_scn(
    dataset_root: Path,
    backend: str = "auto",
    chunk_edges: int = 0,
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]]
    | None = None,
    triangle_path: str | None = None,
    snaphu_path: str | None = None,
    phase_file: str = "phuw2.mat",
) -> str:

    del (
        backend,
        chunk_edges,
        enable_mat_cache,
        io_workers,
        mat_cache,
        snaphu_path,
    )

    root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
    )

    started = time.perf_counter()

    work = (
        root
        / "_stage8_sbas_work"
    )

    if work.exists():
        shutil.rmtree(work)

    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path = (
        root
        / "stage8_sbas_debug.json"
    )

    try:

        # ========================================================
        # LOAD
        # ========================================================

        ps = read_mat(
            root / "ps2.mat"
        )

        parms = read_mat(
            root / "parms.mat"
        )

        n_ps = int(
            round(
                _scalar(ps.get("n_ps"))
            )
        )

        n_ifg = int(
            round(
                _scalar(ps.get("n_ifg"))
            )
        )

        n_image = int(
            round(
                _scalar(ps.get("n_image"))
            )
        )

        master_ix_ps = int(
            round(
                _scalar(ps.get("master_ix"))
            )
        )

        day = np.asarray(
            ps["day"],
            dtype=np.float64,
        ).reshape(-1)

        master_day = _scalar(
            ps.get("master_day"),
            day[master_ix_ps - 1],
        )

        # Official:
        # master_ix=sum(ps.master_day>ps.day)+1
        master_ix = (
            int(
                np.sum(
                    master_day > day
                )
            )
            + 1
        )

        if master_ix != master_ix_ps:
            raise Stage8SbasError(
                f"master index mismatch: "
                f"official={master_ix}, ps={master_ix_ps}"
            )

        master0 = master_ix - 1

        if day.size != n_image:
            raise Stage8SbasError(
                "ps.day length mismatch"
            )

        if (
            _mat_text(
                parms.get("small_baseline_flag"),
                "n",
            ).lower()
            != "y"
        ):
            raise Stage8SbasError(
                "Expected small_baseline_flag='y'"
            )

        if (
            _mat_text(
                parms.get("scn_kriging_flag"),
                "n",
            ).lower()
            != "n"
        ):
            raise Stage8SbasError(
                "This implements ps_scn_filt only; "
                "scn_kriging_flag must be 'n'"
            )

        deramp = parms.get(
            "scn_deramp_ifg"
        )

        if (
            deramp is not None
            and np.asarray(deramp).size
        ):
            raise Stage8SbasError(
                "Current parity baseline requires "
                "scn_deramp_ifg=[]"
            )

        time_win = float(
            _scalar(
                parms.get("scn_time_win"),
                365.0,
            )
        )

        wavelength = float(
            _scalar(
                parms.get("scn_wavelength"),
                100.0,
            )
        )

        if time_win <= 0:
            raise Stage8SbasError(
                "scn_time_win must be >0"
            )

        if wavelength <= 0:
            raise Stage8SbasError(
                "scn_wavelength must be >0"
            )

        chunk = (
            int(chunk_ps)
            if int(chunk_ps) > 0
            else int(
                os.environ.get(
                    "PYSTAMPS_STAGE8_CHUNK_PS",
                    "16384",
                )
            )
        )

        chunk = max(
            256,
            chunk,
        )

        spatial_chunk = int(
            os.environ.get(
                "PYSTAMPS_STAGE8_SPATIAL_CHUNK_PS",
                "4096",
            )
        )

        spatial_chunk = max(
            128,
            spatial_chunk,
        )

        ph_sm = _as_rows(
            read_mat_variables(
                root / phase_file,
                ("ph_uw",),
            )["ph_uw"],
            n_ps,
            "phuw2.ph_uw",
            np.float32,
        )

        if ph_sm.shape != (
            n_ps,
            n_image,
        ):
            raise Stage8SbasError(
                "phuw2 shape mismatch"
            )

        scla = read_mat(
            root / "scla2.mat"
        )

        ph_scla = _as_rows(
            scla["ph_scla"],
            n_ps,
            "scla2.ph_scla",
            np.float32,
        )

        C_ps = np.asarray(
            scla["C_ps_uw"],
            dtype=np.float32,
        ).reshape(-1)

        if C_ps.size != n_ps:
            raise Stage8SbasError(
                "C_ps_uw length mismatch"
            )

        ph_ramp_raw = scla.get(
            "ph_ramp"
        )

        has_scla_ramp = (
            ph_ramp_raw is not None
            and np.asarray(
                ph_ramp_raw
            ).size
            == n_ps * n_image
        )

        if has_scla_ramp:

            ph_scla_ramp = _as_rows(
                ph_ramp_raw,
                n_ps,
                "scla2.ph_ramp",
                np.float32,
            )

        else:

            ph_scla_ramp = None

        xy = _as_rows(
            ps["xy"],
            n_ps,
            "ps.xy",
            np.float64,
        )

        coords = np.asarray(
            xy[:, 1:3],
            dtype=np.float64,
        )

        # ========================================================
        # OFFICIAL TRIANGLE / CONNECTIVITY
        # ========================================================

        print()
        print("=" * 80)
        print("STAGE8 1/4: TRIANGLE / EDGE GRAPH")
        print("=" * 80)

        edges, edge_backend = (
            _build_edges(
                coords,
                root,
                triangle_path,
            )
        )

        print(
            "edges     :",
            f"{edges.shape[0]:,}",
        )

        print(
            "backend   :",
            edge_backend,
        )

        # ========================================================
        # PH_ALL
        # ========================================================

        print()
        print("=" * 80)
        print("STAGE8 2/4: SCLA-CORRECTED PHASE + TEMPORAL HIGHPASS")
        print("=" * 80)

        ph_all = np.memmap(
            work / "ph_all.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
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

            # Official:
            # ph_all=single(uw.ph_uw)
            # ph_all -= single(scla.ph_scla)
            # ph_all -= single(C_ps_uw)
            y = (
                ph_sm[
                    start:stop,
                    :
                ].astype(np.float32)
                -
                ph_scla[
                    start:stop,
                    :
                ].astype(np.float32)
                -
                C_ps[
                    start:stop,
                    None
                ]
            )

            if ph_scla_ramp is not None:
                y -= (
                    ph_scla_ramp[
                        start:stop,
                        :
                    ]
                )

            y[
                ~np.isfinite(y)
            ] = 0.0

            ph_all[
                start:stop,
                :
            ] = y

            print(
                "[STAGE8][PH_ALL] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        ph_all.flush()

        temporal_W = (
            _temporal_weight_matrix(
                day,
                master0,
                time_win,
            )
        )

        # --------------------------------------------------------
        # Official source:
        #
        # dph = B*ph_all
        # dph_lpt = dph*W'
        # dph_hpt = dph-dph_lpt
        # A\dph_hpt, ref_ix=1
        #
        # Algebraically:
        #
        # dph_hpt = B*(ph_all-ph_all*W')
        #
        # For a connected graph the unique solution referenced
        # to PS1 is therefore:
        #
        # ph_hpt = H-H(PS1)
        #
        # This removes no information and is NOT an approximation.
        # --------------------------------------------------------

        first = np.asarray(
            ph_all[0:1, :],
            dtype=np.float64,
        )

        h0 = (
            first
            -
            first @ temporal_W.T
        )[0]

        ph_hpt = np.memmap(
            work / "ph_hpt.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
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

            y = np.asarray(
                ph_all[
                    start:stop,
                    :
                ],
                dtype=np.float64,
            )

            h = (
                y
                -
                y @ temporal_W.T
                -
                h0[None, :]
            )

            ph_hpt[
                start:stop,
                :
            ] = h.astype(
                np.float32
            )

            print(
                "[STAGE8][TEMPORAL] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%)",
                flush=True,
            )

        ph_hpt.flush()

        # ========================================================
        # EDGE-DOMAIN MANDATORY EQUIVALENCE CHECK
        # ========================================================

        ncheck = min(
            4096,
            edges.shape[0],
        )

        eix = np.linspace(
            0,
            edges.shape[0] - 1,
            ncheck,
            dtype=np.int64,
        )

        u = (
            edges[eix, 0]
            - 1
        )

        v = (
            edges[eix, 1]
            - 1
        )

        dph = (
            np.asarray(
                ph_all[v, :],
                dtype=np.float64,
            )
            -
            np.asarray(
                ph_all[u, :],
                dtype=np.float64,
            )
        )

        # Official edge-domain temporal filter.
        dph_hpt = (
            dph
            -
            dph @ temporal_W.T
        )

        node_edge = (
            np.asarray(
                ph_hpt[v, :],
                dtype=np.float64,
            )
            -
            np.asarray(
                ph_hpt[u, :],
                dtype=np.float64,
            )
        )

        edge_equiv_max = float(
            np.max(
                np.abs(
                    dph_hpt
                    -
                    node_edge
                )
            )
        )

        print()
        print(
            "edge-domain equivalence max:",
            edge_equiv_max,
            "rad",
        )

        if edge_equiv_max > 1.0e-5:
            raise Stage8SbasError(
                "Edge-domain temporal-filter "
                "equivalence FAILED"
            )

        # ========================================================
        # EXACT SPATIAL GAUSSIAN
        # ========================================================

        print()
        print("=" * 80)
        print("STAGE8 3/4: FULL-RADIUS SPATIAL GAUSSIAN")
        print("=" * 80)

        radius = (
            4.0 * wavelength
        )

        radius_sq = (
            radius * radius
        )

        sigma2x2 = (
            2.0
            * wavelength
            * wavelength
        )

        print(
            "wavelength :",
            wavelength,
            "m",
        )

        print(
            "radius     :",
            radius,
            "m",
        )

        tree = spatial.cKDTree(
            coords
        )

        # Official ph_scn is double.
        ph_scn = np.memmap(
            work / "ph_scn.f64",
            mode="w+",
            dtype=np.float64,
            shape=(n_ps, n_image),
        )

        total_neigh = 0
        min_neigh = None
        max_neigh = 0

        for start in range(
            0,
            n_ps,
            spatial_chunk,
        ):

            stop = min(
                start + spatial_chunk,
                n_ps,
            )

            m = stop - start

            neighbour_lists = (
                tree.query_ball_point(
                    coords[
                        start:stop,
                        :
                    ],
                    r=radius,
                    workers=-1,
                )
            )

            counts = np.fromiter(
                (
                    len(v)
                    for v in neighbour_lists
                ),
                dtype=np.int64,
                count=m,
            )

            rows = np.repeat(
                np.arange(
                    m,
                    dtype=np.int64,
                ),
                counts,
            )

            if rows.size == 0:
                raise Stage8SbasError(
                    "Spatial filter produced "
                    "empty neighbour block"
                )

            cols = np.concatenate(
                [
                    np.asarray(
                        v,
                        dtype=np.int64,
                    )
                    for v
                    in neighbour_lists
                ]
            )

            global_rows = (
                start + rows
            )

            dxy = (
                coords[cols, :]
                -
                coords[
                    global_rows,
                    :
                ]
            )

            dist_sq = np.sum(
                dxy * dxy,
                axis=1,
            )

            # Official strict:
            # dist_sq < patch_dist_sq
            valid = (
                dist_sq
                < radius_sq
            )

            rows = rows[valid]
            cols = cols[valid]
            dist_sq = dist_sq[valid]

            w = np.exp(
                -dist_sq
                / sigma2x2
            )

            denom = np.bincount(
                rows,
                weights=w,
                minlength=m,
            )

            if np.any(
                denom <= 0
            ):
                raise Stage8SbasError(
                    "Zero spatial Gaussian denominator"
                )

            w /= denom[rows]

            Wsp = sparse.csr_matrix(
                (
                    w,
                    (
                        rows,
                        cols,
                    ),
                ),
                shape=(m, n_ps),
            )

            # Full set of neighbours inside 4 sigma.
            smooth = (
                Wsp
                @ ph_hpt
            )

            ph_scn[
                start:stop,
                :
            ] = np.asarray(
                smooth,
                dtype=np.float64,
            )

            nrow = np.bincount(
                rows,
                minlength=m,
            )

            total_neigh += int(
                np.sum(nrow)
            )

            block_min = int(
                np.min(nrow)
            )

            block_max = int(
                np.max(nrow)
            )

            min_neigh = (
                block_min
                if min_neigh is None
                else min(
                    min_neigh,
                    block_min,
                )
            )

            max_neigh = max(
                max_neigh,
                block_max,
            )

            print(
                "[STAGE8][SPATIAL] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:.1f}%) "
                f"nnz={rows.size:,}",
                flush=True,
            )

        ph_scn.flush()

        mean_neigh = (
            total_neigh
            / float(n_ps)
        )

        # ========================================================
        # OFFICIAL FIRST-PS REFERENCE + MASTER ZERO
        # ========================================================

        print()
        print("=" * 80)
        print("STAGE8 4/4: REFERENCE + SAVE scn2.mat")
        print("=" * 80)

        ref_scn = np.asarray(
            ph_scn[0, :],
            dtype=np.float64,
        ).copy()

        for start in range(
            0,
            n_ps,
            chunk,
        ):

            stop = min(
                start + chunk,
                n_ps,
            )

            ph_scn[
                start:stop,
                :
            ] -= ref_scn[
                None,
                :
            ]

        # Official:
        # ph_scn_slave(:,master_ix)=0
        ph_scn[
            :,
            master0
        ] = 0.0

        ph_scn.flush()

        ph_ramp_out = np.empty(
            (n_ps, 0),
            dtype=np.float64,
        )

        first_hpt_max = float(
            np.max(
                np.abs(
                    ph_hpt[0, :]
                )
            )
        )

        first_scn_max = float(
            np.max(
                np.abs(
                    ph_scn[0, :]
                )
            )
        )

        master_scn_max = float(
            np.max(
                np.abs(
                    ph_scn[
                        :,
                        master0
                    ]
                )
            )
        )

        print(
            "max |ph_hpt PS1|       :",
            first_hpt_max,
        )

        print(
            "max |ph_scn PS1|       :",
            first_scn_max,
        )

        print(
            "max |ph_scn master|    :",
            master_scn_max,
        )

        write_mat(
            root / "scn2.mat",
            {
                "ph_scn_slave":
                    ph_scn,
                "ph_hpt":
                    ph_hpt,
                "ph_ramp":
                    ph_ramp_out,
            },
        )

        duration = (
            time.perf_counter()
            - started
        )

        debug = {
            "status":
                "completed",

            "implementation":
                "MATLAB_PS_SCN_FILT_PARITY_V1",

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

            "inputs": [
                "ps2.mat",
                "phuw2.mat",
                "scla2.mat",
            ],

            "output":
                "scn2.mat",

            "scn_time_win_days":
                time_win,

            "scn_wavelength_m":
                wavelength,

            "spatial_radius_m":
                radius,

            "edge_backend":
                edge_backend,

            "n_edge":
                int(
                    edges.shape[0]
                ),

            "edge_equivalence_sample":
                int(ncheck),

            "edge_equivalence_max_rad":
                edge_equiv_max,

            "spatial_all_neighbors":
                True,

            "spatial_min_neighbors":
                int(min_neigh),

            "spatial_mean_neighbors":
                float(mean_neigh),

            "spatial_max_neighbors":
                int(max_neigh),

            "scn_reference":
                "first_PS",

            "first_ps_ph_hpt_max":
                first_hpt_max,

            "first_ps_ph_scn_max":
                first_scn_max,

            "master_ph_scn_max":
                master_scn_max,

            "note":
                (
                    "Official ps_scn_filt algebra. "
                    "Edge temporal filtering plus "
                    "incidence inversion is evaluated "
                    "through its exact connected-graph "
                    "node-domain identity; mandatory "
                    "edge-domain self-check is applied."
                ),
        }

        _write_json(
            debug_path,
            debug,
        )

        keep_work = (
            os.environ.get(
                "PYSTAMPS_STAGE8_KEEP_WORK",
                "0",
            ).strip().lower()
            in {
                "1",
                "true",
                "yes",
                "y",
            }
        )

        if not keep_work:
            shutil.rmtree(
                work,
                ignore_errors=True,
            )

            shutil.rmtree(
                root
                / "_stage8_triangle_work",
                ignore_errors=True,
            )

        print()
        print("=" * 80)
        print("STAGE 8 MATLAB ps_scn_filt PARITY COMPLETE")
        print("=" * 80)

        print(
            "scn2.mat            : written"
        )

        print(
            "ph_scn_slave        :",
            (n_ps, n_image),
            "float64",
        )

        print(
            "ph_hpt              :",
            (n_ps, n_image),
            "float32",
        )

        print(
            "ph_ramp             :",
            (n_ps, 0),
        )

        print(
            "duration             :",
            f"{duration:.2f} s",
        )

        return (
            "Stage 8 MATLAB ps_scn_filt "
            f"parity completed for {n_ps} PS "
            f"across {n_image} acquisitions"
        )

    except Exception as exc:

        _write_json(
            debug_path,
            {
                "status":
                    "failed",
                "implementation":
                    "MATLAB_PS_SCN_FILT_PARITY_V1",
                "duration_sec":
                    time.perf_counter()
                    - started,
                "exception":
                    f"{type(exc).__name__}: {exc}",
            },
        )

        raise


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--triangle",
        default=None,
    )

    parser.add_argument(
        "--chunk-ps",
        type=int,
        default=0,
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

    print(
        stage8_sbas_filter_scn(
            args.dataset,
            chunk_ps=args.chunk_ps,
            triangle_path=args.triangle,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
