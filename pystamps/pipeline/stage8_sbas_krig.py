from __future__ import annotations

"""
StaMPS ps_scn_filt_krig.m compatible SBAS Stage-8 implementation.

Scientific reference
--------------------
The scientific sequence follows StaMPS 4.0b6 matlab/ps_scn_filt_krig.m:

1. subtract SCLA / master atmosphere-orbit term / Stage-7 ramp;
2. deramp every acquisition with [1, x, y];
3. form Delaunay arcs;
4. on each arc, estimate quadratic + linear + annual + constant deformation;
5. fit temporal variograms on a random arc subset;
6. temporally krige the deformation-removed arc residual;
7. solve the arc field back to PS phase with PS1 fixed;
8. per epoch, remove a second-degree deterministic spatial field;
9. fit a spatial variogram and spatially krige APS;
10. add the per-epoch ramp back, reference to PS1, and set master APS to zero.

Large-scene engineering
-----------------------
StaMPS' 2011 MATLAB implementation performs one ordinary-kriging solve per
arc/date and per PS/date. That is not computationally viable for multi-million
point stacks. This implementation keeps the StaMPS model and ordinary-kriging
kernel, but uses two controlled accelerations for large scenes:

* temporal kriging: edges in a small spatial tile share one kriging operator;
  the variogram parameters are still evaluated from the StaMPS quadratic
  spatial model at the tile centre. A random-edge QA compares this with exact
  per-edge operators; tile size is reduced automatically until tolerance.
* spatial kriging: APS is ordinary-kriged on an adaptive regular grid and
  bilinearly interpolated to PS. Exact per-PS kriging is checked on a random
  PS QA subset; grid spacing is reduced automatically until tolerance.

Both approximations are deterministic, audited, and recorded in
stage8_sbas_krig_debug.json. The deformation model, variogram model, ordinary
kriging equations, Nmax logic, ramp handling, reference convention and master
handling follow StaMPS.

Unwrapping-error cycles are DIAGNOSTIC ONLY, matching the official source:
StaMPS computes corrected_dph but leaves the assignment dph=corrected_dph
commented out. This module likewise never applies those cycle corrections.
"""

import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import optimize, sparse, spatial
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, cg

from pystamps.io.mat import read_mat, read_mat_variables, write_mat
from pystamps.pipeline.stage8_sbas import (
    Stage8SbasError,
    _as_rows,
    _build_edges,
    _mat_text,
    _scalar,
)


class Stage8SbasKrigError(Stage8SbasError):
    pass


MATLAB_DATENUM_2000_01_01 = 730486.0
TWO_PI = 2.0 * math.pi
STAMPS_EXACT_SPATIAL_FULL_PS_V1 = True

STAMPS_SPATIAL_EXACT_REF_V2 = True


@dataclass(slots=True)
class KrigRuntime:
    seed: int
    edge_chunk: int
    ps_chunk: int
    max_vario_edges: int
    max_vario_ps: int
    spatial_vario_pairs: int
    temporal_tile_m: float
    temporal_min_tile_m: float
    temporal_qa_edges: int
    temporal_qa_tol_rad: float
    spatial_grid_m: float
    spatial_min_grid_m: float
    spatial_qa_ps: int
    spatial_qa_tol_rad: float
    spatial_qa_p95_tol_rad: float
    spatial_qa_p99_tol_rad: float
    spatial_qa_epochs: int
    cg_rtol: float
    cg_maxiter: int
    write_edge_cycles: bool


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        v = int(os.environ.get(name, str(default)))
    except Exception:
        v = default
    return max(minimum, v)


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except Exception:
        v = default
    if minimum is not None:
        v = max(minimum, v)
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", "n"}


def _runtime(chunk_edges: int, chunk_ps: int) -> KrigRuntime:
    return KrigRuntime(
        seed=_env_int("PYSTAMPS_SCN_KRIG_SEED", 0, 0),
        edge_chunk=(
            int(chunk_edges)
            if int(chunk_edges) > 0
            else _env_int("PYSTAMPS_SCN_KRIG_EDGE_CHUNK", 200_000, 1024)
        ),
        ps_chunk=(
            int(chunk_ps)
            if int(chunk_ps) > 0
            else _env_int("PYSTAMPS_SCN_KRIG_PS_CHUNK", 32_768, 256)
        ),
        max_vario_edges=_env_int("PYSTAMPS_SCN_KRIG_VARIO_EDGES", 15_000, 100),
        max_vario_ps=_env_int("PYSTAMPS_SCN_KRIG_VARIO_PS", 15_000, 100),
        spatial_vario_pairs=_env_int(
            "PYSTAMPS_SCN_KRIG_SPATIAL_VARIO_PAIRS", 2_000_000, 10_000
        ),
        temporal_tile_m=_env_float("PYSTAMPS_SCN_KRIG_TEMP_TILE_M", 5000.0, 100.0),
        temporal_min_tile_m=_env_float(
            "PYSTAMPS_SCN_KRIG_TEMP_MIN_TILE_M", 625.0, 50.0
        ),
        temporal_qa_edges=_env_int("PYSTAMPS_SCN_KRIG_TEMP_QA_EDGES", 256, 16),
        temporal_qa_tol_rad=_env_float(
            "PYSTAMPS_SCN_KRIG_TEMP_QA_TOL_RAD", 0.03, 1.0e-6
        ),
        spatial_grid_m=_env_float("PYSTAMPS_SCN_KRIG_GRID_M", 2000.0, 50.0),
        spatial_min_grid_m=_env_float(
            "PYSTAMPS_SCN_KRIG_MIN_GRID_M", 250.0, 25.0
        ),
        spatial_qa_ps=_env_int(
            "PYSTAMPS_SCN_KRIG_SPATIAL_QA_PS",
            5000,
            64,
        ),
        spatial_qa_tol_rad=_env_float(
            "PYSTAMPS_SCN_KRIG_SPATIAL_QA_TOL_RAD",
            0.05,
            1.0e-6,
        ),
        spatial_qa_p95_tol_rad=_env_float(
            "PYSTAMPS_SCN_KRIG_SPATIAL_QA_P95_TOL_RAD",
            0.10,
            1.0e-6,
        ),
        spatial_qa_p99_tol_rad=_env_float(
            "PYSTAMPS_SCN_KRIG_SPATIAL_QA_P99_TOL_RAD",
            0.20,
            1.0e-6,
        ),
        spatial_qa_epochs=_env_int(
            "PYSTAMPS_SCN_KRIG_SPATIAL_QA_EPOCHS",
            7,
            1,
        ),
        cg_rtol=_env_float("PYSTAMPS_SCN_KRIG_CG_RTOL", 1.0e-6, 1.0e-12),
        cg_maxiter=_env_int("PYSTAMPS_SCN_KRIG_CG_MAXITER", 400, 20),
        write_edge_cycles=_env_bool("PYSTAMPS_SCN_KRIG_WRITE_EDGE_CYCLES", True),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _official_time_design(day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    years = (np.asarray(day, dtype=np.float64) - MATLAB_DATENUM_2000_01_01) / 365.25
    A = np.column_stack(
        (
            years * years,
            years,
            np.sin(TWO_PI * years),
            np.cos(TWO_PI * years) - 1.0,
            np.ones_like(years),
        )
    )
    return years, A


def _ols_projector(A: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(A, dtype=np.float64))


def _spatial_plane_coefficients(
    phase: np.memmap,
    coords: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """Exact chunked OLS of [1,x,y] for every epoch."""
    n_ps, n_img = phase.shape
    xtx = np.zeros((3, 3), dtype=np.float64)
    xty = np.zeros((3, n_img), dtype=np.float64)
    for start in range(0, n_ps, chunk):
        stop = min(n_ps, start + chunk)
        G = np.column_stack(
            (
                np.ones(stop - start, dtype=np.float64),
                coords[start:stop, 0],
                coords[start:stop, 1],
            )
        )
        y = np.asarray(phase[start:stop, :], dtype=np.float64)
        xtx += G.T @ G
        xty += G.T @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def _fit_exp_variogram(
    h: np.ndarray,
    gamma: np.ndarray,
    a0: float,
    c10: float,
    c20: float,
) -> tuple[float, float, float]:
    h = np.asarray(h, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    good = np.isfinite(h) & np.isfinite(gamma)
    h = h[good]
    gamma = gamma[good]
    if h.size < 4:
        return float("nan"), float("nan"), float("nan")

    def model(x: np.ndarray, a: float, c1: float, c2: float) -> np.ndarray:
        aa = max(float(a), 1.0e-12)
        return c1 * (1.0 - np.exp(-x / aa)) + c2

    # StaMPS ps_fit_vario.m parity:
    #
    # lsqcurvefit(..., [0 0 0], [])
    #
    # Range, sill and nugget are constrained only to be
    # non-negative. There is NO artificial upper bound.
    try:
        popt, _ = optimize.curve_fit(
            model,
            h,
            gamma,
            p0=(
                max(a0, 1.0e-6),
                max(c10, 0.0),
                max(c20, 0.0),
            ),
            bounds=(
                (0.0, 0.0, 0.0),
                (np.inf, np.inf, np.inf),
            ),
            maxfev=20000,
        )
    except Exception:
        return float("nan"), float("nan"), float("nan")
    return float(popt[0]), float(popt[1]), float(popt[2])


def _empirical_variogram_from_pairs(
    distances: np.ndarray,
    semivariance: np.ndarray,
    decor_dist: float,
    nlags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """StaMPS ps_fit_vario binning, without constructing an NxN matrix."""
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    g = np.asarray(semivariance, dtype=np.float64).reshape(-1)
    good = np.isfinite(d) & np.isfinite(g) & (d >= 0.0) & (d < decor_dist)
    d = d[good]
    g = g[good]
    if d.size == 0:
        return np.empty(0), np.empty(0)

    lags = np.linspace(0.0, decor_dist, int(nlags), dtype=np.float64)
    nb = len(lags) - 1
    which = np.searchsorted(lags, d, side="right") - 1
    valid = (which >= 0) & (which < nb)
    which = which[valid]
    g = g[valid]

    sums = np.bincount(which, weights=g, minlength=nb)
    counts = np.bincount(which, minlength=nb)
    mids = 0.5 * (lags[:-1] + lags[1:])
    keep = counts > 0
    mids = mids[keep]
    vals = sums[keep] / counts[keep]

    original_intervals = nb
    last_lag = min(len(vals), int(round(original_intervals / 1.75)))
    return mids[:last_lag], vals[:last_lag]


def _temporal_pair_geometry(years: np.ndarray, decor_dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.triu_indices(len(years), 1)
    d = np.abs(years[j] - years[i])
    keep = d < decor_dist
    return i[keep], j[keep], d[keep]


def _fit_temporal_variograms(
    sample_residual: np.ndarray,
    years: np.ndarray,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(sample_residual.shape[0], max_edges)
    z = np.asarray(sample_residual[:n, :], dtype=np.float64)
    ii, jj, dist = _temporal_pair_geometry(years, 0.25 * 6.0)
    a = np.full(n, np.nan, dtype=np.float64)
    c1 = np.full(n, np.nan, dtype=np.float64)
    c2 = np.full(n, np.nan, dtype=np.float64)

    for k in range(n):
        semivar = 0.5 * (z[k, jj] - z[k, ii]) ** 2
        h, g = _empirical_variogram_from_pairs(dist, semivar, 0.25 * 6.0, 25)
        a[k], c1[k], c2[k] = _fit_exp_variogram(h, g, 0.25, 1.0, 2.0)
        if (k + 1) % 1000 == 0 or k + 1 == n:
            print(f"[STAGE8_KRIG][TEMP_VARIO] {k+1:,}/{n:,}", flush=True)
    return a, c1, c2


def _vario_spatial_design(x: np.ndarray, y: np.ndarray, mean_x: float, mean_y: float) -> np.ndarray:
    mx = mean_x if abs(mean_x) > 1.0e-9 else 1.0
    my = mean_y if abs(mean_y) > 1.0e-9 else 1.0
    xn = x / mx
    yn = y / my
    return np.column_stack((xn * xn, yn * yn, xn, yn, np.ones_like(xn)))


def _eval_vario_field(
    x: np.ndarray,
    y: np.ndarray,
    mean_x: float,
    mean_y: float,
    ahat: np.ndarray,
    c1hat: np.ndarray,
    c2hat: np.ndarray,
    fallback_a: float,
    fallback_c1: float,
    fallback_c2: float,
    mean_std: float,
    mean_years: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = _vario_spatial_design(x, y, mean_x, mean_y)
    a = D @ ahat
    c1 = D @ c1hat
    c2 = D @ c2hat
    amin = mean_years / 50.0
    cmin = mean_std / 50.0
    a = np.where(a <= amin, fallback_a, a)
    c1 = np.where(c1 <= cmin, fallback_c1, c1)
    c2 = np.where(c2 <= cmin, fallback_c2, c2)
    return a, c1, c2


def _ordinary_kriging_operator_1d(
    years: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
    nmax: int,
    exclude_self: bool,
) -> np.ndarray:
    years = np.asarray(years, dtype=np.float64)
    n = years.size
    W = np.zeros((n, n), dtype=np.float64)
    aa = max(float(a), 1.0e-9)
    nugget = max(float(nugget), 0.0)
    sill = max(float(sill), 1.0e-12)
    dx_max = aa * 4.0

    for target in range(n):
        d = np.abs(years - years[target])
        if exclude_self:
            d[target] = np.inf
        order = np.argsort(d)
        order = order[: min(int(nmax), n)]
        order = order[d[order] < dx_max]
        if order.size == 0:
            order = np.asarray([int(np.argmin(d))], dtype=np.int64)

        obs_t = years[order]
        H = np.abs(obs_t[:, None] - obs_t[None, :])
        h0 = np.abs(obs_t - years[target])
        K = sill * np.exp(-H / aa)
        K.flat[:: K.shape[0] + 1] += nugget
        k0 = sill * np.exp(-h0 / aa)  # nugget intentionally excluded: StaMPS ps_kriging
        ones = np.ones(order.size, dtype=np.float64)
        aug = np.empty((order.size + 1, order.size + 1), dtype=np.float64)
        aug[:-1, :-1] = K
        aug[:-1, -1] = ones
        aug[-1, :-1] = ones
        aug[-1, -1] = 0.0
        rhs = np.r_[k0, 1.0]
        try:
            sol = np.linalg.solve(aug, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        W[target, order] = sol[:-1]
    return W


def _gaussian_temporal_operator(day: np.ndarray, master0: int, time_win: float) -> np.ndarray:
    dt = np.asarray(day, dtype=np.float64)[:, None] - np.asarray(day, dtype=np.float64)[None, :]
    W = np.exp(-(dt * dt) / (2.0 * time_win * time_win))
    W[:, master0] = 0.0
    den = W.sum(axis=1)
    if np.any(den <= 0):
        raise Stage8SbasKrigError("Temporal Gaussian fallback has zero denominator")
    return W / den[:, None]


def _tile_key(x: np.ndarray, y: np.ndarray, xmin: float, ymin: float, tile: float) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor((x - xmin) / tile).astype(np.int64)
    iy = np.floor((y - ymin) / tile).astype(np.int64)
    return ix, iy


def _tile_center(ix: int, iy: int, xmin: float, ymin: float, tile: float) -> tuple[float, float]:
    return xmin + (ix + 0.5) * tile, ymin + (iy + 0.5) * tile


def _make_temporal_operator_cache(
    keys: Iterable[tuple[int, int]],
    *,
    xmin: float,
    ymin: float,
    tile: float,
    years: np.ndarray,
    nmax: int,
    mean_x: float,
    mean_y: float,
    ahat: np.ndarray,
    c1hat: np.ndarray,
    c2hat: np.ndarray,
    fallback_a: float,
    fallback_c1: float,
    fallback_c2: float,
    mean_std: float,
    mean_years: float,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        cx, cy = _tile_center(key[0], key[1], xmin, ymin, tile)
        a, c1, c2 = _eval_vario_field(
            np.asarray([cx]), np.asarray([cy]), mean_x, mean_y,
            ahat, c1hat, c2hat,
            fallback_a, fallback_c1, fallback_c2,
            mean_std, mean_years,
        )
        # StaMPS cv_model_all = [nugget=c2; exponential sill=c1].
        inc = _ordinary_kriging_operator_1d(years, a[0], c2[0], c1[0], nmax, False)
        loo = _ordinary_kriging_operator_1d(years, a[0], c2[0], c1[0], nmax, True)
        out[key] = (inc, loo)
    return out


def _temporal_tile_qa(
    residual: np.memmap,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    sample_ix: np.ndarray,
    tile: float,
    xmin: float,
    ymin: float,
    years: np.ndarray,
    nmax: int,
    mean_x: float,
    mean_y: float,
    ahat: np.ndarray,
    c1hat: np.ndarray,
    c2hat: np.ndarray,
    fallback_a: float,
    fallback_c1: float,
    fallback_c2: float,
    mean_std: float,
    mean_years: float,
) -> dict[str, float]:
    errs = []
    for idx in sample_ix:
        x = float(edge_x[idx]); y = float(edge_y[idx])
        a, c1, c2 = _eval_vario_field(
            np.asarray([x]), np.asarray([y]), mean_x, mean_y,
            ahat, c1hat, c2hat,
            fallback_a, fallback_c1, fallback_c2,
            mean_std, mean_years,
        )
        exact = _ordinary_kriging_operator_1d(years, a[0], c2[0], c1[0], nmax, False)
        ix, iy = _tile_key(np.asarray([x]), np.asarray([y]), xmin, ymin, tile)
        cx, cy = _tile_center(int(ix[0]), int(iy[0]), xmin, ymin, tile)
        ta, tc1, tc2 = _eval_vario_field(
            np.asarray([cx]), np.asarray([cy]), mean_x, mean_y,
            ahat, c1hat, c2hat,
            fallback_a, fallback_c1, fallback_c2,
            mean_std, mean_years,
        )
        approx = _ordinary_kriging_operator_1d(years, ta[0], tc2[0], tc1[0], nmax, False)
        z = np.asarray(residual[int(idx), :], dtype=np.float64)
        de = (approx - exact) @ z
        errs.extend(np.asarray(de, dtype=float).tolist())
    e = np.asarray(errs, dtype=np.float64)
    return {
        "rms_rad": float(np.sqrt(np.mean(e * e))) if e.size else 0.0,
        "p95_abs_rad": float(np.percentile(np.abs(e), 95)) if e.size else 0.0,
        "max_abs_rad": float(np.max(np.abs(e))) if e.size else 0.0,
    }


def _graph_laplacian(edges: np.ndarray, n_ps: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(edges[:, 0] - 1, dtype=np.int64)
    v = np.asarray(edges[:, 1] - 1, dtype=np.int64)
    adj = sparse.coo_matrix(
        (
            np.ones(2 * len(u), dtype=np.float32),
            (np.r_[u, v], np.r_[v, u]),
        ),
        shape=(n_ps, n_ps),
    ).tocsr()
    adj.sum_duplicates()
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float64)
    if np.any(degree <= 0):
        raise Stage8SbasKrigError("Delaunay graph contains isolated PS")
    L = sparse.diags(degree, format="csr") - adj.astype(np.float64)
    return L, degree, u, v


def _solve_edge_field_to_nodes(
    dph_hpt: np.memmap,
    edges: np.ndarray,
    n_ps: int,
    n_img: int,
    out: np.memmap,
    rtol: float,
    maxiter: int,
) -> list[dict[str, Any]]:
    """
    StaMPS edge -> PS least-squares reconstruction with PS1 fixed.

    Official ps_scn_filt_krig.m solves A\\dph_hpt.

    With PS1 removed this is equivalent to solving the grounded
    graph-Laplacian normal equation:

        (A.T A) x = A.T d

    Algebraic multigrid is used only as a preconditioner.
    It does NOT alter the StaMPS least-squares objective.
    """

    print(
        "[STAGE8_KRIG] building graph Laplacian",
        flush=True,
    )

    L, degree, u, v = _graph_laplacian(
        edges,
        n_ps,
    )

    # PS1 fixed to zero, exactly matching StaMPS ref_ix=1.
    Lr = L[1:, 1:].tocsr()

    # Avoid retaining two copies of the full Laplacian.
    del L

    import gc
    gc.collect()

    solver = os.environ.get(
        "PYSTAMPS_SCN_KRIG_GRAPH_SOLVER",
        "auto",
    ).strip().lower()

    if solver not in {
        "auto",
        "amg",
        "jacobi",
    }:
        raise Stage8SbasKrigError(
            "PYSTAMPS_SCN_KRIG_GRAPH_SOLVER must be "
            "auto, amg, or jacobi"
        )

    M: LinearOperator
    solver_used: str
    ml = None

    # ========================================================
    # AMG preconditioner
    # ========================================================

    if solver in {"auto", "amg"}:

        try:
            import pyamg  # type: ignore

        except ImportError as exc:

            if solver == "amg" or n_ps >= 250_000:
                raise Stage8SbasKrigError(
                    "Large-scene Edge->PS reconstruction requires "
                    "PyAMG. Install with: "
                    "`python -m pip install pyamg` "
                    "or `conda install -c conda-forge pyamg`."
                ) from exc

            pyamg = None

        if pyamg is not None:

            print(
                "[STAGE8_KRIG][EDGE_TO_PS] "
                "building AMG hierarchy...",
                flush=True,
            )

            t_amg = time.perf_counter()

            # Grounded graph Laplacian is SPD.
            # Constant vector is the near-nullspace mode of the
            # ungrounded graph Laplacian.
            B = np.ones(
                (Lr.shape[0], 1),
                dtype=np.float64,
            )

            ml = pyamg.smoothed_aggregation_solver(
                Lr,
                B=B,
                symmetry="symmetric",
                max_levels=25,
                max_coarse=500,
                keep=False,
            )

            del B
            gc.collect()

            M = ml.aspreconditioner(
                cycle="V"
            )

            solver_used = "amg_cg"

            try:
                complexity = float(
                    ml.operator_complexity()
                )
            except Exception:
                complexity = float("nan")

            print(
                "[STAGE8_KRIG][EDGE_TO_PS] AMG ready: "
                f"levels={len(ml.levels)} "
                f"operator_complexity={complexity:.3f} "
                f"setup={time.perf_counter()-t_amg:.1f}s",
                flush=True,
            )

        else:
            invdiag = 1.0 / degree[1:]

            M = LinearOperator(
                Lr.shape,
                matvec=lambda x: invdiag * x,
                dtype=np.float64,
            )

            solver_used = "jacobi_cg"

    else:

        invdiag = 1.0 / degree[1:]

        M = LinearOperator(
            Lr.shape,
            matvec=lambda x: invdiag * x,
            dtype=np.float64,
        )

        solver_used = "jacobi_cg"

    print(
        "[STAGE8_KRIG][EDGE_TO_PS] "
        f"solver={solver_used} "
        f"rtol={rtol:g} "
        f"maxiter={maxiter}",
        flush=True,
    )

    qa: list[dict[str, Any]] = []

    # Warm start from previous acquisition.
    x0 = None

    # Deterministic QA sample.
    step = max(
        1,
        len(u) // 4096,
    )

    sample_edge = np.arange(
        0,
        len(u),
        step,
        dtype=np.int64,
    )[:4096]

    # ========================================================
    # Solve all acquisitions
    # ========================================================

    for j in range(n_img):

        d = np.asarray(
            dph_hpt[:, j],
            dtype=np.float64,
        )

        # A.T @ d for incidence matrix A.
        rhs = (
            np.bincount(
                v,
                weights=d,
                minlength=n_ps,
            )
            -
            np.bincount(
                u,
                weights=d,
                minlength=n_ps,
            )
        )

        br = rhs[1:]

        bnorm = float(
            np.linalg.norm(br)
        )

        iterations = [0]

        def callback(_x):
            iterations[0] += 1

        sol, info = cg(
            Lr,
            br,
            x0=x0,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            M=M,
            callback=callback,
        )

        # Explicitly audit the normal-equation residual.
        normal_residual = (
            Lr @ sol
            - br
        )

        relative_residual = float(
            np.linalg.norm(
                normal_residual
            )
            /
            max(
                bnorm,
                np.finfo(np.float64).tiny,
            )
        )

        if (
            info != 0
            or not np.isfinite(
                relative_residual
            )
            or relative_residual
            > max(
                5.0 * rtol,
                1.0e-8,
            )
        ):
            raise Stage8SbasKrigError(
                f"Edge->PS {solver_used} failed "
                f"at epoch {j+1}: "
                f"info={info}, "
                f"iterations={iterations[0]}, "
                f"relative_residual="
                f"{relative_residual:.3e}"
            )

        out[0, j] = 0.0

        out[1:, j] = sol.astype(
            np.float32
        )

        # Warm start next acquisition.
        x0 = sol

        # ----------------------------------------------------
        # Edge-domain LS QA
        # ----------------------------------------------------

        node_d = (
            np.asarray(
                out[
                    v[sample_edge],
                    j,
                ],
                dtype=np.float64,
            )
            -
            np.asarray(
                out[
                    u[sample_edge],
                    j,
                ],
                dtype=np.float64,
            )
        )

        rr = (
            node_d
            - d[sample_edge]
        )

        qa.append(
            {
                "epoch_1based":
                    j + 1,

                "solver":
                    solver_used,

                "cg_iterations":
                    iterations[0],

                "relative_normal_residual":
                    relative_residual,

                "edge_fit_rms_rad":
                    float(
                        np.sqrt(
                            np.mean(
                                rr * rr
                            )
                        )
                    ),

                "edge_fit_max_abs_rad":
                    float(
                        np.max(
                            np.abs(rr)
                        )
                    ),
            }
        )

        print(
            "[STAGE8_KRIG][EDGE_TO_PS] "
            f"{j+1}/{n_img} "
            f"solver={solver_used} "
            f"cg={iterations[0]} "
            f"relres={relative_residual:.3e} "
            f"rms="
            f"{qa[-1]['edge_fit_rms_rad']:.4g} rad",
            flush=True,
        )

    out.flush()

    return qa


def _spatial_poly_design(x: np.ndarray, y: np.ndarray, mean_x: float, mean_y: float) -> np.ndarray:
    mx = mean_x if abs(mean_x) > 1.0e-9 else 1.0
    my = mean_y if abs(mean_y) > 1.0e-9 else 1.0
    xn = x / mx
    yn = y / my
    # Exact order from ps_scn_filt_krig.m
    return np.column_stack((xn * xn, xn, yn * yn, yn, np.ones_like(xn)))


def _sample_spatial_variogram_pairs(
    n: int,
    rng: np.random.Generator,
    npairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one fixed PS-pair geometry and reuse it for all epochs.

    Official StaMPS uses the same complete coordinate-pair geometry
    for every epoch.  For the HPC sampled-pair implementation we
    therefore keep the sampled geometry fixed across epochs.
    """
    if n < 3:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    m = int(
        min(
            npairs,
            max(
                10_000,
                n * 200,
            ),
        )
    )

    i = rng.integers(
        0,
        n,
        size=m,
        endpoint=False,
        dtype=np.int32,
    )

    j = rng.integers(
        0,
        n,
        size=m,
        endpoint=False,
        dtype=np.int32,
    )

    neq = i != j

    return (
        i[neq],
        j[neq],
    )


def _fit_spatial_variogram_sampled_pairs(
    coords: np.ndarray,
    values: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[float, float, float]:
    """
    StaMPS spatial variogram model evaluated on a fixed large
    random-pair subset.

    This is the remaining HPC approximation to the official NxN
    pair matrix; model, lagging and parameter bounds follow StaMPS.
    """
    if (
        coords.shape[0] < 3
        or pair_i.size == 0
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    i = np.asarray(
        pair_i,
        dtype=np.int64,
    )

    j = np.asarray(
        pair_j,
        dtype=np.int64,
    )

    dx = (
        coords[i, 0]
        - coords[j, 0]
    )

    dy = (
        coords[i, 1]
        - coords[j, 1]
    )

    h = np.sqrt(
        dx * dx
        + dy * dy
    )

    g = (
        0.5
        * (
            values[i]
            - values[j]
        ) ** 2
    )

    mids, vals = (
        _empirical_variogram_from_pairs(
            h,
            g,
            15_000.0 * 2.0,
            30,
        )
    )

    return _fit_exp_variogram(
        mids,
        vals,
        15_000.0,
        1.0,
        1.0,
    )


def _ordinary_kriging_batch_full50(
    neigh_xy: np.ndarray,
    neigh_z: np.ndarray,
    target_xy: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
) -> np.ndarray:
    """Vectorized ordinary kriging where every target uses the same k neighbors count."""
    B, k, _ = neigh_xy.shape
    aa = max(float(a), 1.0e-9)
    nugget = max(float(nugget), 0.0)
    sill = max(float(sill), 1.0e-12)
    diff = neigh_xy[:, :, None, :] - neigh_xy[:, None, :, :]
    H = np.sqrt(np.sum(diff * diff, axis=3))
    K = sill * np.exp(-H / aa)
    diag = np.arange(k)
    K[:, diag, diag] += nugget
    h0 = np.sqrt(np.sum((neigh_xy - target_xy[:, None, :]) ** 2, axis=2))
    k0 = sill * np.exp(-h0 / aa)
    ones = np.ones((B, k), dtype=np.float64)
    rhs = np.stack((k0, ones), axis=2)  # B,k,2
    try:
        q = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        q = np.empty_like(rhs)
        for b in range(B):
            q[b] = np.linalg.lstsq(K[b], rhs[b], rcond=None)[0]
    kinv_k = q[:, :, 0]
    kinv_1 = q[:, :, 1]
    num = 1.0 - np.sum(kinv_k, axis=1)
    den = np.sum(kinv_1, axis=1)
    mu = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1.0e-15)
    w = kinv_k + kinv_1 * mu[:, None]
    return np.sum(w * neigh_z, axis=1)


def _ordinary_kriging_one(
    obs_xy: np.ndarray,
    obs_z: np.ndarray,
    target_xy: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
) -> float:
    if len(obs_z) == 0:
        return float("nan")
    z = _ordinary_kriging_batch_full50(
        obs_xy[None, :, :], obs_z[None, :], target_xy[None, :], a, nugget, sill
    )
    return float(z[0])


def _grid_axes(coords: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    gx = np.arange(xmin, xmax + step * 0.51, step, dtype=np.float64)
    gy = np.arange(ymin, ymax + step * 0.51, step, dtype=np.float64)
    if gx[-1] < xmax:
        gx = np.r_[gx, xmax]
    if gy[-1] < ymax:
        gy = np.r_[gy, ymax]
    return gx, gy


def _spatial_krige_grid_epoch(
    *,
    tree: spatial.cKDTree,
    coords: np.ndarray,
    values_residual: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
    base_dx_max: float,
    nmax: int = 50,
    batch: int = 256,
    query_chunk: int = 20_000,
) -> np.ndarray:
    """
    HPC spatial-grid kriging.

    The kriging equations are unchanged.  Nearest-neighbour queries
    are chunked only to avoid materialising a very large
    n_grid x 50 distance/index matrix.
    """
    xx, yy = np.meshgrid(
        gx,
        gy,
    )

    targets = np.column_stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
        )
    )

    out = np.full(
        len(targets),
        np.nan,
        dtype=np.float64,
    )

    dx_max = (
        max(
            float(a),
            1.0e-9,
        )
        * 4.0
    )

    k = min(
        int(nmax),
        len(coords),
    )

    for q0 in range(
        0,
        len(targets),
        int(query_chunk),
    ):

        q1 = min(
            len(targets),
            q0 + int(query_chunk),
        )

        txy = targets[q0:q1]

        dist, ix = tree.query(
            txy,
            k=k,
            workers=-1,
        )

        if dist.ndim == 1:
            dist = dist[:, None]
            ix = ix[:, None]

        dist = np.asarray(
            dist,
            dtype=np.float64,
        )

        ix = np.asarray(
            ix,
            dtype=np.int64,
        )

        # Historical StaMPS logic:
        #
        # epoch 1 defines an outer candidate radius;
        # the current epoch 4*a can only reduce that set.
        counts = np.sum(
            (
                dist
                < float(base_dx_max)
            )
            & (
                dist
                < dx_max
            ),
            axis=1,
        ).astype(np.int32)

        counts = np.maximum(
            counts,
            1,
        )

        for c in np.unique(
            counts
        ):

            c = int(c)

            rows_all = np.flatnonzero(
                counts == c
            )

            for b0 in range(
                0,
                len(rows_all),
                int(batch),
            ):

                rows = rows_all[
                    b0:
                    b0 + int(batch)
                ]

                ni = ix[
                    rows,
                    :c,
                ]

                nxy = coords[ni]

                nz = values_residual[ni]

                out[
                    q0 + rows
                ] = (
                    _ordinary_kriging_batch_full50(
                        nxy,
                        nz,
                        txy[rows],
                        a,
                        nugget,
                        sill,
                    )
                )

    return out.reshape(
        len(gy),
        len(gx),
    )


def _interpolate_grid_to_ps(gx: np.ndarray, gy: np.ndarray, grid: np.ndarray, coords: np.ndarray, chunk: int) -> np.ndarray:
    interp = RegularGridInterpolator((gy, gx), grid, method="linear", bounds_error=False, fill_value=None)
    out = np.empty(len(coords), dtype=np.float64)
    for start in range(0, len(coords), chunk):
        stop = min(len(coords), start + chunk)
        # RegularGridInterpolator input order is y,x.
        pts = np.column_stack((coords[start:stop, 1], coords[start:stop, 0]))
        out[start:stop] = interp(pts)
    return out


def _stamps_neighbor_cache_targets(
    tree: spatial.cKDTree,
    coords: np.ndarray,
    target_ix: np.ndarray,
    base_dx_max: float,
    nmax: int = 50,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Exact neighbour-cache rule from ps_scn_filt_krig.m.

    Neighbours are determined only on epoch 1 and the candidate
    list is reused on every later epoch.
    """
    target_ix = np.asarray(
        target_ix,
        dtype=np.int64,
    ).reshape(-1)

    target_xy = np.asarray(
        coords[target_ix],
        dtype=np.float64,
    )

    k = min(
        int(nmax),
        len(coords),
    )

    dist, ids = tree.query(
        target_xy,
        k=k,
        workers=-1,
    )

    if dist.ndim == 1:
        dist = dist[:, None]
        ids = ids[:, None]

    dist = np.asarray(
        dist,
        dtype=np.float64,
    )

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    base_counts = np.sum(
        dist
        < float(base_dx_max),
        axis=1,
    ).astype(np.int32)

    # The target PS itself has distance 0.
    base_counts = np.maximum(
        base_counts,
        1,
    )

    return (
        target_xy,
        dist,
        ids,
        base_counts,
    )


def _exact_spatial_krige_targets_cached(
    coords: np.ndarray,
    values: np.ndarray,
    target_xy: np.ndarray,
    dist: np.ndarray,
    ids: np.ndarray,
    base_counts: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
    batch: int = 256,
) -> np.ndarray:
    """
    Exact local ordinary kriging corresponding to StaMPS
    ps_kriging(..., Nmax=50, dx_max=4*a, method=2).
    """
    dx_max = (
        max(
            float(a),
            1.0e-9,
        )
        * 4.0
    )

    current_counts = np.sum(
        dist < dx_max,
        axis=1,
    ).astype(np.int32)

    counts = np.minimum(
        np.asarray(
            base_counts,
            dtype=np.int32,
        ),
        current_counts,
    )

    counts = np.maximum(
        counts,
        1,
    )

    out = np.empty(
        len(target_xy),
        dtype=np.float64,
    )

    for c in np.unique(
        counts
    ):

        c = int(c)

        rows_all = np.flatnonzero(
            counts == c
        )

        for b0 in range(
            0,
            len(rows_all),
            int(batch),
        ):

            rows = rows_all[
                b0:
                b0 + int(batch)
            ]

            ni = ids[
                rows,
                :c,
            ]

            nxy = coords[ni]

            nz = values[ni]

            out[rows] = (
                _ordinary_kriging_batch_full50(
                    nxy,
                    nz,
                    target_xy[rows],
                    a,
                    nugget,
                    sill,
                )
            )

    return out


def _exact_spatial_krige_targets(
    tree: spatial.cKDTree,
    coords: np.ndarray,
    values: np.ndarray,
    target_ix: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
    base_dx_max: float,
    nmax: int = 50,
) -> np.ndarray:
    """
    Backward-compatible exact StaMPS reference wrapper.
    """
    (
        target_xy,
        dist,
        ids,
        base_counts,
    ) = _stamps_neighbor_cache_targets(
        tree,
        coords,
        target_ix,
        base_dx_max,
        nmax,
    )

    return (
        _exact_spatial_krige_targets_cached(
            coords,
            values,
            target_xy,
            dist,
            ids,
            base_counts,
            a,
            nugget,
            sill,
        )
    )



# =====================================================================
# STAMPS_EXACT_SPATIAL_FULL_PS_V1
#
# Exact StaMPS ps_scn_filt_krig spatial branch, HPC implementation.
#
# Scientific operations are unchanged:
#   * StaMPS spatial polynomial
#   * full empirical variogram within decor_dist
#   * exponential covariance
#   * first-epoch nearest-50 candidate cache
#   * later-epoch current 4*a filtering
#   * ordinary kriging, nugget excluded from target covariance
#
# Acceleration changes execution only:
#   * exact KD-tree nearest-neighbour search
#   * symmetric variogram pair de-duplication
#   * block accumulation
#   * batched linear solves
#   * shared neighbour cache
#   * thread parallelism across epochs
# =====================================================================


def _stamps_prepare_spatial_vario_pairs(
    xy: np.ndarray,
    decor_dist: float = 30_000.0,
    nlags_input: int = 30,
):
    """
    Exact pair geometry needed by StaMPS ps_fit_vario.m.

    StaMPS constructs the complete NxN pair matrix, but only pairs with
    distance < decor_dist enter the experimental variogram.

    StaMPS also includes both (i,j) and (j,i).  Their distance and
    semivariance are identical, so keeping each unordered pair once
    produces exactly the same bin mean used by lsqcurvefit.
    """
    xy = np.asarray(
        xy,
        dtype=np.float64,
    )

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"xy must be N x 2, got {xy.shape}"
        )

    tree = spatial.cKDTree(
        xy
    )

    # Strictly less than decor_dist, matching:
    # h1 >= lags(v) & h1 < lags(v+1)
    radius = np.nextafter(
        float(decor_dist),
        0.0,
    )

    pairs = tree.query_pairs(
        r=radius,
        output_type="ndarray",
    )

    pairs = np.asarray(
        pairs,
        dtype=np.int32,
    )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError(
            f"Bad query_pairs result {pairs.shape}"
        )

    nbins = int(nlags_input) - 1

    lags = np.linspace(
        0.0,
        float(decor_dist),
        int(nlags_input),
        dtype=np.float64,
    )

    width = float(
        lags[1] - lags[0]
    )

    pair_bin = np.empty(
        pairs.shape[0],
        dtype=np.uint8,
    )

    block = 2_000_000

    for start in range(
        0,
        len(pairs),
        block,
    ):
        stop = min(
            len(pairs),
            start + block,
        )

        q = pairs[start:stop]

        dx = (
            xy[q[:, 0], 0]
            - xy[q[:, 1], 0]
        )

        dy = (
            xy[q[:, 0], 1]
            - xy[q[:, 1], 1]
        )

        h = np.hypot(
            dx,
            dy,
        )

        b = np.floor(
            h / width
        ).astype(np.int64)

        if np.any(
            (b < 0)
            | (b >= nbins)
        ):
            raise RuntimeError(
                "Spatial variogram pair fell outside "
                "StaMPS lag interval"
            )

        pair_bin[start:stop] = (
            b.astype(np.uint8)
        )

    counts = np.bincount(
        pair_bin.astype(np.int64),
        minlength=nbins,
    ).astype(np.int64)

    return (
        pairs,
        pair_bin,
        counts,
        lags,
    )


def _stamps_fit_spatial_vario_allpairs(
    values: np.ndarray,
    pairs: np.ndarray,
    pair_bin: np.ndarray,
    lags: np.ndarray,
    pair_chunk: int = 2_000_000,
):
    """
    Exact empirical variogram used by ps_fit_vario.m.

    Model:
        gamma(h) = c1 * (1-exp(-h/a)) + c2

    Input StaMPS settings:
        a0  = 15000 m
        c10 = 1
        c20 = 1
        decor_dist = 30000 m
        Nlags input = 30
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    nbins = len(lags) - 1

    sums = np.zeros(
        nbins,
        dtype=np.float64,
    )

    counts = np.zeros(
        nbins,
        dtype=np.int64,
    )

    for start in range(
        0,
        len(pairs),
        int(pair_chunk),
    ):
        stop = min(
            len(pairs),
            start + int(pair_chunk),
        )

        q = pairs[start:stop]

        dz = (
            values[q[:, 0]]
            - values[q[:, 1]]
        )

        gamma = (
            0.5
            * dz
            * dz
        )

        b = pair_bin[
            start:stop
        ].astype(np.int64)

        finite = np.isfinite(
            gamma
        )

        if not np.any(finite):
            continue

        sums += np.bincount(
            b[finite],
            weights=gamma[finite],
            minlength=nbins,
        )

        counts += np.bincount(
            b[finite],
            minlength=nbins,
        ).astype(np.int64)

    good = counts > 0

    mids = (
        0.5
        * (
            lags[:-1]
            + lags[1:]
        )
    )[good]

    empirical = (
        sums[good]
        / counts[good]
    )

    # MATLAB:
    #
    # Nlags = 29 after creation of 30 lag boundaries
    # last_lag = round(Nlags/1.75)
    #
    # Positive MATLAB round -> floor(x+0.5).
    last_lag = min(
        len(mids),
        int(
            math.floor(
                nbins / 1.75
                + 0.5
            )
        ),
    )

    if last_lag < 3:
        raise RuntimeError(
            "Too few non-empty StaMPS spatial variogram bins"
        )

    mids = mids[:last_lag]
    empirical = empirical[:last_lag]

    return _fit_exp_variogram(
        mids,
        empirical,
        15_000.0,
        1.0,
        1.0,
    )


def _stamps_ordinary_kriging_from_dist_batch(
    h: np.ndarray,
    h0: np.ndarray,
    z: np.ndarray,
    a: float,
    nugget: float,
    sill: float,
):
    """
    Batched exact equivalent of StaMPS ps_kriging(..., method=2).

    h   : B x K x K observation-observation distances
    h0  : B x K target-observation distances
    z   : B x K observations

    Covariance:
        nugget on K diagonal only
        exponential covariance for K and k0
        no target nugget in k0

    Ordinary kriging:
        [K 1; 1' 0] [w;mu] = [k0;1]
    """
    h = np.asarray(
        h,
        dtype=np.float64,
    )

    h0 = np.asarray(
        h0,
        dtype=np.float64,
    )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    if z.ndim != 2:
        raise ValueError(
            f"z must be B x K, got {z.shape}"
        )

    batch, k = z.shape

    if h.shape != (
        batch,
        k,
        k,
    ):
        raise ValueError(
            f"h={h.shape}, z={z.shape}"
        )

    if h0.shape != (
        batch,
        k,
    ):
        raise ValueError(
            f"h0={h0.shape}, z={z.shape}"
        )

    a = float(a)
    nugget = float(nugget)
    sill = float(sill)

    if (
        not np.isfinite(a)
        or a <= 0.0
        or not np.isfinite(nugget)
        or nugget < 0.0
        or not np.isfinite(sill)
        or sill < 0.0
    ):
        raise RuntimeError(
            "Invalid StaMPS spatial variogram: "
            f"a={a}, c1={sill}, c2={nugget}"
        )

    cov = (
        sill
        * np.exp(
            -h / a
        )
    )

    ii = np.arange(
        k
    )

    cov[
        :,
        ii,
        ii,
    ] += nugget

    # Nugget intentionally excluded from k0,
    # exactly as ps_kriging.m.
    k0 = (
        sill
        * np.exp(
            -h0 / a
        )
    )

    system = np.empty(
        (
            batch,
            k + 1,
            k + 1,
        ),
        dtype=np.float64,
    )

    system[
        :,
        :k,
        :k,
    ] = cov

    system[
        :,
        :k,
        k,
    ] = 1.0

    system[
        :,
        k,
        :k,
    ] = 1.0

    system[
        :,
        k,
        k,
    ] = 0.0

    rhs = np.empty(
        (
            batch,
            k + 1,
        ),
        dtype=np.float64,
    )

    rhs[
        :,
        :k,
    ] = k0

    rhs[
        :,
        k,
    ] = 1.0

    try:
        # NumPy >= 2.0 requires an explicit trailing RHS dimension
        # for batched linear solves:
        #
        #   system : B x M x M
        #   rhs    : B x M x 1
        #
        # This changes only array layout, not the StaMPS
        # ordinary-kriging linear system.
        sol = np.linalg.solve(
            system,
            rhs[..., None],
        )[..., 0]

    except np.linalg.LinAlgError:
        # MATLAB backslash also has rank-deficient handling.
        # Fall back only for pathological local systems.
        sol = np.empty_like(
            rhs
        )

        for q in range(
            batch
        ):
            try:
                sol[q] = np.linalg.solve(
                    system[q],
                    rhs[q],
                )
            except np.linalg.LinAlgError:
                sol[q] = np.linalg.lstsq(
                    system[q],
                    rhs[q],
                    rcond=None,
                )[0]

    avg = np.mean(
        z,
        axis=1,
    )

    zc = (
        z
        - avg[:, None]
    )

    result = (
        np.einsum(
            "bi,bi->b",
            sol[:, :k],
            zc,
        )
        + avg
    )

    return result



def _save_cycle_h5(path: Path, shape: tuple[int, int], chunks: tuple[int, int]):
    import h5py
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    h = h5py.File(tmp, "w")
    ds = h.create_dataset(
        "unwrapping_errors", shape=shape, dtype=np.int8,
        chunks=chunks, compression="gzip", compression_opts=1, shuffle=True,
    )
    return h, ds, tmp


def stage8_sbas_filter_scn_krig(
    dataset_root: Path,
    backend: str = "auto",
    chunk_edges: int = 0,
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    triangle_path: str | None = None,
    snaphu_path: str | None = None,
    phase_file: str = "phuw2.mat",
) -> str:
    del backend, enable_mat_cache, io_workers, mat_cache, snaphu_path
    root = Path(dataset_root).expanduser().resolve()
    started = time.perf_counter()
    rt = _runtime(chunk_edges, chunk_ps)

    if phase_file != "phuw2.mat" and not _env_bool(
        "PYSTAMPS_SCN_KRIG_ALLOW_ALTERNATIVE_PHASE", False
    ):
        raise Stage8SbasKrigError(
            "StaMPS kriging mode must use the original phuw2.mat. Disable GACOS and "
            "post_unwrap_deramp. Set PYSTAMPS_SCN_KRIG_ALLOW_ALTERNATIVE_PHASE=1 "
            "only for an explicitly non-parity experiment."
        )

    ps = read_mat(root / "ps2.mat")
    parms = read_mat(root / "parms.mat")
    scla = read_mat(root / "scla2.mat")

    n_ps = int(round(_scalar(ps.get("n_ps"))))
    n_img = int(round(_scalar(ps.get("n_image"))))
    n_ifg = int(round(_scalar(ps.get("n_ifg"))))
    master_ix = int(round(_scalar(ps.get("master_ix"), 1)))
    master0 = master_ix - 1
    day = np.asarray(ps["day"], dtype=np.float64).reshape(-1)
    if len(day) != n_img:
        raise Stage8SbasKrigError("ps.day length mismatch")
    if _mat_text(parms.get("small_baseline_flag"), "n").lower() != "y":
        raise Stage8SbasKrigError("Kriging Stage-8 implementation is for SBAS small_baseline_flag='y'")
    if _mat_text(parms.get("scn_kriging_flag"), "n").lower() != "y":
        raise Stage8SbasKrigError("scn_kriging_flag must be 'y'")

    time_win = float(_scalar(parms.get("scn_time_win"), 365.0))
    ph_sm = _as_rows(
        read_mat_variables(root / phase_file, ("ph_uw",))["ph_uw"],
        n_ps, "phuw2.ph_uw", np.float32,
    )
    ph_scla = _as_rows(scla["ph_scla"], n_ps, "scla2.ph_scla", np.float32)
    C_ps = np.asarray(scla["C_ps_uw"], dtype=np.float32).reshape(-1)
    if C_ps.size != n_ps:
        raise Stage8SbasKrigError("C_ps_uw length mismatch")
    ph_scla_ramp_raw = scla.get("ph_ramp")
    ph_scla_ramp = None
    if ph_scla_ramp_raw is not None and np.asarray(ph_scla_ramp_raw).size == n_ps * n_img:
        ph_scla_ramp = _as_rows(ph_scla_ramp_raw, n_ps, "scla2.ph_ramp", np.float32)

    xy = _as_rows(ps["xy"], n_ps, "ps.xy", np.float64)
    if xy.shape[1] < 3:
        raise Stage8SbasKrigError("ps.xy requires [id,x,y]")
    coords = np.asarray(xy[:, 1:3], dtype=np.float64)

    work = root / "_stage8_sbas_krig_work"
    if work.exists() and not _env_bool("PYSTAMPS_SCN_KRIG_RESUME", False):
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    debug_path = root / "stage8_sbas_krig_debug.json"

    print("=" * 96)
    print("STAGE 8 SBAS: StaMPS ps_scn_filt_krig HPC")
    print("=" * 96)
    print("n_ps                 :", f"{n_ps:,}")
    print("n_image              :", n_img)
    print("n_ifg                :", n_ifg)
    print("master_ix            :", master_ix)
    print("phase_file           :", phase_file)
    print("temporal tile start  :", rt.temporal_tile_m, "m")
    print("spatial mode         : StaMPS exact local / all PS")

    # ------------------------------------------------------------------
    # 1. ph_all after Stage-7 corrections
    # ------------------------------------------------------------------
    ph_all = np.memmap(work / "ph_all.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_img))
    for start in range(0, n_ps, rt.ps_chunk):
        stop = min(n_ps, start + rt.ps_chunk)
        y = (
            ph_sm[start:stop, :].astype(np.float32)
            - ph_scla[start:stop, :].astype(np.float32)
            - C_ps[start:stop, None]
        )
        if ph_scla_ramp is not None:
            y -= ph_scla_ramp[start:stop, :]
        y[~np.isfinite(y)] = 0.0
        ph_all[start:stop, :] = y
        if stop == n_ps or stop % max(rt.ps_chunk * 10, 1) == 0:
            print(f"[STAGE8_KRIG][PH_ALL] {stop:,}/{n_ps:,}", flush=True)
    ph_all.flush()

    # ------------------------------------------------------------------
    # 2. Official krig_atmo=true => deramp every acquisition [1,x,y]
    # ------------------------------------------------------------------
    plane = _spatial_plane_coefficients(ph_all, coords, rt.ps_chunk)
    ph_ramp = np.memmap(work / "ph_ramp.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_img))
    for start in range(0, n_ps, rt.ps_chunk):
        stop = min(n_ps, start + rt.ps_chunk)
        G = np.column_stack((np.ones(stop-start), coords[start:stop, 0], coords[start:stop, 1]))
        ramp = G @ plane
        ph_ramp[start:stop, :] = ramp.astype(np.float32)
        ph_all[start:stop, :] = (
            np.asarray(ph_all[start:stop, :], dtype=np.float64) - ramp
        ).astype(np.float32)
    ph_all.flush(); ph_ramp.flush()

    # ------------------------------------------------------------------
    # 3. Delaunay arcs
    # ------------------------------------------------------------------
    edges, edge_backend = _build_edges(coords, root, triangle_path)
    u = np.asarray(edges[:, 0] - 1, dtype=np.int64)
    v = np.asarray(edges[:, 1] - 1, dtype=np.int64)
    n_edges = len(edges)
    edge_x = ((coords[u, 0] + coords[v, 0]) * 0.5).astype(np.float64)
    edge_y = ((coords[u, 1] + coords[v, 1]) * 0.5).astype(np.float64)
    print("edges                :", f"{n_edges:,}")
    print("edge backend         :", edge_backend)

    # ------------------------------------------------------------------
    # 4. Per-edge deformation model: quadratic + linear + annual + const
    # ------------------------------------------------------------------
    years, A_time = _official_time_design(day)
    P_time = _ols_projector(A_time)  # 5 x n_img
    residual = np.memmap(work / "edge_residual.f32", mode="w+", dtype=np.float32, shape=(n_edges, n_img))
    model_coef = np.memmap(work / "modeled_defo_edges.f32", mode="w+", dtype=np.float32, shape=(n_edges, 5))
    # StaMPS ps_scn_filt_krig.m:
    #   mean_std = nanmean(nanstd(dph));
    #
    # dph is n_edges x n_image. MATLAB nanstd(dph) therefore
    # calculates sample std (N-1) across EDGES for every epoch,
    # followed by the mean across epochs.
    #
    # Accumulate exact column statistics chunkwise.
    edge_resid_sum = np.zeros(n_img, dtype=np.float64)
    edge_resid_sumsq = np.zeros(n_img, dtype=np.float64)

    rng = np.random.default_rng(rt.seed)
    sample_count = min(rt.max_vario_edges, n_edges)
    sample_ix = np.sort(rng.choice(n_edges, size=sample_count, replace=False))
    sample_pos = {int(e): i for i, e in enumerate(sample_ix.tolist())}
    sample_resid = np.empty((sample_count, n_img), dtype=np.float32)

    for start in range(0, n_edges, rt.edge_chunk):
        stop = min(n_edges, start + rt.edge_chunk)
        dph = (
            np.asarray(ph_all[v[start:stop], :], dtype=np.float64)
            - np.asarray(ph_all[u[start:stop], :], dtype=np.float64)
        )
        coef = dph @ P_time.T
        res = dph - coef @ A_time.T
        residual[start:stop, :] = res.astype(np.float32)
        model_coef[start:stop, :] = coef.astype(np.float32)
        edge_resid_sum += np.sum(
            res,
            axis=0,
            dtype=np.float64,
        )
        edge_resid_sumsq += np.einsum(
            "ij,ij->j",
            res,
            res,
            dtype=np.float64,
            optimize=True,
        )
        local = sample_ix[(sample_ix >= start) & (sample_ix < stop)]
        for e in local:
            sample_resid[sample_pos[int(e)], :] = res[int(e)-start].astype(np.float32)
        print(f"[STAGE8_KRIG][EDGE_MODEL] {stop:,}/{n_edges:,}", flush=True)
    residual.flush()
    model_coef.flush()

    if n_edges > 1:
        # MATLAB std default normalization is N-1.
        edge_epoch_var = (
            edge_resid_sumsq
            - (
                edge_resid_sum
                * edge_resid_sum
            )
            / float(n_edges)
        ) / float(n_edges - 1)

        # Numerical roundoff can produce tiny negative values.
        edge_epoch_var = np.maximum(
            edge_epoch_var,
            0.0,
        )

        edge_epoch_std = np.sqrt(
            edge_epoch_var
        )

        mean_std = float(
            np.nanmean(
                edge_epoch_std
            )
        )
    else:
        mean_std = 0.0

    print(
        "[STAGE8_KRIG][STAMPS_MEAN_STD] "
        f"mean spatial std across epochs = "
        f"{mean_std:.6g} rad",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5. Temporal variogram on sampled edges, exactly following StaMPS model
    # ------------------------------------------------------------------
    a_s, c1_s, c2_s = _fit_temporal_variograms(sample_resid, years, sample_count)
    mean_years = float(np.nanmean(years))
    valid = (
        np.isfinite(a_s)
        & np.isfinite(c1_s)
        & np.isfinite(c2_s)
        & (a_s < mean_years * 3.0)
        & (a_s > mean_years / 50.0)
        & (c1_s > mean_std / 50.0)
        & (c1_s < 4.0 * mean_std)
        & (c2_s > mean_std / 50.0)
        & (c2_s < 4.0 * mean_std)
    )
    n_valid_vario = int(np.count_nonzero(valid))
    print("valid temporal variograms:", n_valid_vario, "/", sample_count)

    temporal_mode = "gaussian_fallback"
    ahat = c1hat = c2hat = None
    mean_x = float(np.nanmean(edge_x)); mean_y = float(np.nanmean(edge_y))
    fallback_a = float(np.nanmedian(a_s[valid])) if n_valid_vario else 0.25
    fallback_c1 = float(np.nanmedian(c1_s[valid])) if n_valid_vario else max(mean_std, 1.0e-3)
    fallback_c2 = float(np.nanmedian(c2_s[valid])) if n_valid_vario else max(mean_std, 1.0e-3)

    if n_valid_vario > 50:
        temporal_mode = "ordinary_kriging_tiled"
        sx = edge_x[sample_ix][valid]
        sy = edge_y[sample_ix][valid]
        D = _vario_spatial_design(sx, sy, mean_x, mean_y)
        ahat = np.linalg.lstsq(D, a_s[valid], rcond=None)[0]
        c1hat = np.linalg.lstsq(D, c1_s[valid], rcond=None)[0]
        c2hat = np.linalg.lstsq(D, c2_s[valid], rcond=None)[0]

    # Save temporal model diagnostics early.
    write_mat(
        root / "modeled_defo_edges.mat",
        {
            "modeled_defo_edges": model_coef,
            "A_time": A_time,
            "sample_edge_index": (sample_ix + 1).reshape(-1, 1),
            "a": a_s.reshape(-1, 1),
            "c1": c1_s.reshape(-1, 1),
            "c2": c2_s.reshape(-1, 1),
            "years": years.reshape(-1, 1),
            "mean_std": np.asarray([[mean_std]], dtype=np.float64),
        },
    )

    # ------------------------------------------------------------------
    # 6. Temporal filter / kriging in edge domain
    # ------------------------------------------------------------------
    # --------------------------------------------------------------
    # Resume support after a completed temporal-kriging stage.
    #
    # PYSTAMPS_SCN_KRIG_RESUME=1
    #     preserves the work directory.
    #
    # PYSTAMPS_SCN_KRIG_REUSE_TEMPORAL=1
    #     reuses dph_lpt.f32 and dph_hpt_edge.f32 rather than
    #     repeating the 12.9-million-edge temporal kriging.
    # --------------------------------------------------------------

    temporal_reuse = _env_bool(
        "PYSTAMPS_SCN_KRIG_REUSE_TEMPORAL",
        False,
    )

    temporal_expected_bytes = (
        int(n_edges)
        * int(n_img)
        * np.dtype(
            np.float32
        ).itemsize
    )

    temporal_paths = [
        work / "dph_lpt.f32",
        work / "dph_hpt_edge.f32",
    ]

    if temporal_reuse:

        bad = [
            str(q)
            for q in temporal_paths
            if (
                not q.is_file()
                or q.stat().st_size
                != temporal_expected_bytes
            )
        ]

        if bad:
            raise Stage8SbasKrigError(
                "PYSTAMPS_SCN_KRIG_REUSE_TEMPORAL=1 "
                "was requested, but completed temporal "
                "checkpoint files are missing or have "
                f"the wrong size: {bad}"
            )

        print(
            "[STAGE8_KRIG][CHECKPOINT] "
            "reusing completed temporal "
            "dph_lpt/dph_hpt_edge products",
            flush=True,
        )

    temporal_open_mode = (
        "r+"
        if temporal_reuse
        else "w+"
    )

    dph_lpt = np.memmap(
        work / "dph_lpt.f32",
        mode=temporal_open_mode,
        dtype=np.float32,
        shape=(n_edges, n_img),
    )

    dph_hpt_edge = np.memmap(
        work / "dph_hpt_edge.f32",
        mode=temporal_open_mode,
        dtype=np.float32,
        shape=(n_edges, n_img),
    )

    epoch_cycle_count = np.zeros(
        n_img,
        dtype=np.int64,
    )

    epoch_cycle_abs_sum = np.zeros(
        n_img,
        dtype=np.int64,
    )

    # Restore previous acquisition QA when temporal products
    # are being reused.
    epoch_qa_path = (
        root
        / "stage8_krig_epoch_qa.csv"
    )

    if (
        temporal_reuse
        and epoch_qa_path.is_file()
    ):

        try:

            with epoch_qa_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as f:

                rows = list(
                    csv.DictReader(f)
                )

            if len(rows) == n_img:

                epoch_cycle_count[:] = np.asarray(
                    [
                        int(
                            r[
                                "unwrap_error_edges"
                            ]
                        )
                        for r in rows
                    ],
                    dtype=np.int64,
                )

                epoch_cycle_abs_sum[:] = np.asarray(
                    [
                        int(
                            r[
                                "abs_cycle_sum"
                            ]
                        )
                        for r in rows
                    ],
                    dtype=np.int64,
                )

        except Exception as exc:
            raise Stage8SbasKrigError(
                "Unable to restore temporal "
                f"checkpoint epoch QA: {exc}"
            ) from exc

    cycle_h5 = (
        cycle_ds
    ) = (
        cycle_tmp
    ) = None

    cycle_path = (
        root
        / "unwrapping_errors_edges.h5"
    )

    if (
        rt.write_edge_cycles
        and not temporal_reuse
    ):

        (
            cycle_h5,
            cycle_ds,
            cycle_tmp,
        ) = _save_cycle_h5(
            cycle_path,
            (n_edges, n_img),
            (
                min(
                    rt.edge_chunk,
                    n_edges,
                ),
                n_img,
            ),
        )

    temporal_tile_used = None
    temporal_qa: dict[str, float] = {}
    nmax_time = int(math.ceil(float(np.nanmean(years)) + 2.0 * float(np.nanstd(years))))
    nmax_time = max(1, min(nmax_time, n_img))

    if temporal_mode == "gaussian_fallback":
        W = _gaussian_temporal_operator(day, master0, time_win)
        fallback_stop = (
            0
            if temporal_reuse
            else n_edges
        )

        for start in range(
            0,
            fallback_stop,
            rt.edge_chunk,
        ):
            stop = min(
                n_edges,
                start + rt.edge_chunk,
            )
            z = np.asarray(
                residual[start:stop, :],
                dtype=np.float64,
            )
            lp = z @ W.T
            hp = z - lp
            dph_lpt[start:stop, :] = lp.astype(np.float32)
            dph_hpt_edge[start:stop, :] = hp.astype(np.float32)
            if cycle_ds is not None:
                cycle_ds[start:stop, :] = 0
            print(f"[STAGE8_KRIG][TEMP_FALLBACK] {stop:,}/{n_edges:,}", flush=True)
    else:
        assert ahat is not None and c1hat is not None and c2hat is not None
        xmin = float(np.min(edge_x)); ymin = float(np.min(edge_y))
        qa_n = min(rt.temporal_qa_edges, n_edges)
        qa_ix = np.sort(rng.choice(n_edges, size=qa_n, replace=False))
        tile = rt.temporal_tile_m
        while True:
            temporal_qa = _temporal_tile_qa(
                residual, edge_x, edge_y, qa_ix, tile, xmin, ymin,
                years, nmax_time, mean_x, mean_y, ahat, c1hat, c2hat,
                fallback_a, fallback_c1, fallback_c2, mean_std, mean_years,
            )
            print(
                f"[STAGE8_KRIG][TEMP_TILE_QA] tile={tile:g}m "
                f"rms={temporal_qa['rms_rad']:.5f} rad "
                f"p95={temporal_qa['p95_abs_rad']:.5f}", flush=True,
            )
            if temporal_qa["rms_rad"] <= rt.temporal_qa_tol_rad or tile <= rt.temporal_min_tile_m:
                break
            tile = max(rt.temporal_min_tile_m, tile / 2.0)
        if temporal_qa["rms_rad"] > rt.temporal_qa_tol_rad:
            raise Stage8SbasKrigError(
                "Temporal tile approximation failed QA at minimum tile size: "
                f"RMS={temporal_qa['rms_rad']:.5f} rad > {rt.temporal_qa_tol_rad:.5f}"
            )
        temporal_tile_used = tile

        if temporal_reuse:
            print(
                "[STAGE8_KRIG][CHECKPOINT] "
                "temporal ordinary-kriging loop "
                "skipped; existing complete edge "
                "products retained",
                flush=True,
            )

        op_cache: dict[
            tuple[int, int],
            tuple[
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        krig_stop = (
            0
            if temporal_reuse
            else n_edges
        )

        for start in range(
            0,
            krig_stop,
            rt.edge_chunk,
        ):
            stop = min(
                n_edges,
                start + rt.edge_chunk,
            )
            ex = edge_x[start:stop]
            ey = edge_y[start:stop]
            tx, ty = _tile_key(ex, ey, xmin, ymin, tile)
            keys = list({(int(a), int(b)) for a, b in zip(tx.tolist(), ty.tolist())})
            missing = [k for k in keys if k not in op_cache]
            if missing:
                op_cache.update(_make_temporal_operator_cache(
                    missing, xmin=xmin, ymin=ymin, tile=tile, years=years,
                    nmax=nmax_time, mean_x=mean_x, mean_y=mean_y,
                    ahat=ahat, c1hat=c1hat, c2hat=c2hat,
                    fallback_a=fallback_a, fallback_c1=fallback_c1,
                    fallback_c2=fallback_c2, mean_std=mean_std,
                    mean_years=mean_years,
                ))
            z = np.asarray(residual[start:stop, :], dtype=np.float64)
            lp = np.empty_like(z)
            cycles = np.zeros(z.shape, dtype=np.int8)
            for key in keys:
                mask = (tx == key[0]) & (ty == key[1])
                inc, loo = op_cache[key]
                lp[mask, :] = z[mask, :] @ inc.T
                pred_loo = z[mask, :] @ loo.T
                cyc = np.rint((pred_loo - z[mask, :]) / (2.01 * math.pi))
                cyc = np.clip(cyc, -127, 127).astype(np.int8)
                cycles[mask, :] = cyc
            dph_lpt[start:stop, :] = lp.astype(np.float32)
            dph_hpt_edge[start:stop, :] = (z - lp).astype(np.float32)
            epoch_cycle_count += np.count_nonzero(cycles, axis=0)
            epoch_cycle_abs_sum += np.sum(np.abs(cycles.astype(np.int16)), axis=0, dtype=np.int64)
            if cycle_ds is not None:
                cycle_ds[start:stop, :] = cycles
            print(
                f"[STAGE8_KRIG][TEMP_KRIG] {stop:,}/{n_edges:,} "
                f"tiles_cached={len(op_cache):,}", flush=True,
            )
    dph_lpt.flush(); dph_hpt_edge.flush()
    if cycle_h5 is not None:
        cycle_h5.flush(); cycle_h5.close()
        os.replace(cycle_tmp, cycle_path)

    with (root / "stage8_krig_epoch_qa.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch_1based", "matlab_day", "unwrap_error_edges", "unwrap_error_fraction", "abs_cycle_sum"])
        for j in range(n_img):
            w.writerow([
                j + 1, float(day[j]), int(epoch_cycle_count[j]),
                float(epoch_cycle_count[j] / max(n_edges, 1)), int(epoch_cycle_abs_sum[j]),
            ])

    # ------------------------------------------------------------------
    # 7. Edge -> PS least squares with PS1 fixed
    # ------------------------------------------------------------------
    ph_hpt_path = work / "ph_hpt.f32"

    reuse_ph_hpt = _env_bool(
        "PYSTAMPS_SCN_KRIG_REUSE_PH_HPT",
        False,
    )

    ph_hpt_expected_bytes = (
        int(n_ps)
        * int(n_img)
        * np.dtype(np.float32).itemsize
    )

    if reuse_ph_hpt:

        if (
            not ph_hpt_path.is_file()
            or ph_hpt_path.stat().st_size
            != ph_hpt_expected_bytes
        ):
            raise Stage8SbasKrigError(
                "PYSTAMPS_SCN_KRIG_REUSE_PH_HPT=1 "
                "requested, but ph_hpt.f32 is missing "
                "or has the wrong size"
            )

        ph_hpt = np.memmap(
            ph_hpt_path,
            mode="r+",
            dtype=np.float32,
            shape=(n_ps, n_img),
        )

        edge_to_ps_qa = [
            {
                "checkpoint_reused": True,
            }
        ]

        print(
            "[STAGE8_KRIG][CHECKPOINT] "
            "reusing completed ph_hpt.f32; "
            "Edge->PS solve skipped",
            flush=True,
        )

    else:

        ph_hpt = np.memmap(
            ph_hpt_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_img),
        )

        edge_to_ps_qa = (
            _solve_edge_field_to_nodes(
                dph_hpt_edge,
                edges,
                n_ps,
                n_img,
                ph_hpt,
                rt.cg_rtol,
                rt.cg_maxiter,
            )
        )

    # ------------------------------------------------------------------
    # 8. Per-epoch deterministic spatial field + variogram
    # ------------------------------------------------------------------
    n_vario_ps = min(rt.max_vario_ps, n_ps)
    ind_vario = np.sort(rng.choice(n_ps, size=n_vario_ps, replace=False))
    sample_xy = coords[ind_vario]
    mean_x_ps = float(np.nanmean(sample_xy[:, 0])); mean_y_ps = float(np.nanmean(sample_xy[:, 1]))
    A_sample = _spatial_poly_design(sample_xy[:, 0], sample_xy[:, 1], mean_x_ps, mean_y_ps)
    spatial_coef = np.empty((n_img, 5), dtype=np.float64)
    spatial_a = np.empty(n_img, dtype=np.float64)
    spatial_c1 = np.empty(n_img, dtype=np.float64)
    spatial_c2 = np.empty(n_img, dtype=np.float64)

    (
        spatial_pairs,
        spatial_pair_bin,
        spatial_pair_counts,
        spatial_vario_lags,
    ) = _stamps_prepare_spatial_vario_pairs(
        sample_xy,
        decor_dist=30_000.0,
        nlags_input=30,
    )

    # Compatibility aliases used only by downstream provenance.
    spatial_pair_i = spatial_pairs[:, 0]
    spatial_pair_j = spatial_pairs[:, 1]

    print(
        "[STAGE8_KRIG][SPATIAL_VARIO_EXACT] "
        f"PS={n_vario_ps:,} "
        f"unordered pairs within 30 km="
        f"{len(spatial_pairs):,}",
        flush=True,
    )

    for j in range(n_img):
        z = np.asarray(ph_hpt[ind_vario, j], dtype=np.float64)
        xhat = np.linalg.lstsq(A_sample, z, rcond=None)[0]
        spatial_coef[j] = xhat
        detr = z - A_sample @ xhat
        a, c1, c2 = (
            _stamps_fit_spatial_vario_allpairs(
                detr,
                spatial_pairs,
                spatial_pair_bin,
                spatial_vario_lags,
                pair_chunk=_env_int(
                    "PYSTAMPS_SCN_KRIG_VARIO_PAIR_CHUNK",
                    2_000_000,
                    100_000,
                ),
            )
        )
        if not np.isfinite(a) or a <= 0:
            a = 15_000.0
        if not np.isfinite(c1) or c1 <= 0:
            c1 = max(float(np.nanvar(detr)), 1.0e-4)
        if not np.isfinite(c2) or c2 < 0:
            c2 = max(0.1 * c1, 1.0e-6)
        spatial_a[j] = a; spatial_c1[j] = c1; spatial_c2[j] = c2
        print(
            f"[STAGE8_KRIG][SPATIAL_VARIO] {j+1}/{n_img} "
            f"a={a:.1f}m c1={c1:.4g} c2={c2:.4g}", flush=True,
        )

    # ------------------------------------------------------------------
    # 9. Adaptive-grid ordinary spatial kriging + exact-target QA
    # ------------------------------------------------------------------
    # ==============================================================
    # 9. Exact StaMPS spatial ordinary kriging
    # ==============================================================

    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )

    tree = spatial.cKDTree(
        coords
    )

    spatial_mode = (
        "stamps_exact_local_all_ps"
    )

    nmax_spatial = min(
        50,
        n_ps,
    )

    if (
        not np.isfinite(spatial_a[0])
        or spatial_a[0] <= 0
    ):
        raise Stage8SbasKrigError(
            "Invalid first-epoch spatial range "
            f"a={spatial_a[0]}"
        )

    base_dx_max = (
        float(spatial_a[0])
        * 4.0
    )

    # Compatibility name used by the pre-exact Stage-8
    # provenance/debug payload. Same StaMPS quantity: 4*a1.
    spatial_base_dx_max = base_dx_max

    exact_dir = (
        work
        / "spatial_exact"
    )

    exact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_workers = _env_int(
        "PYSTAMPS_SCN_KRIG_EXACT_WORKERS",
        16,
        1,
    )

    exact_chunk = _env_int(
        "PYSTAMPS_SCN_KRIG_EXACT_CHUNK",
        1024,
        64,
    )

    neighbour_query_chunk = _env_int(
        "PYSTAMPS_SCN_KRIG_NEIGHBOR_QUERY_CHUNK",
        200_000,
        10_000,
    )

    reuse_exact = _env_bool(
        "PYSTAMPS_SCN_KRIG_REUSE_EXACT_SPATIAL",
        True,
    )

    # --------------------------------------------------------------
    # StaMPS first-epoch candidate cache.
    #
    # Official:
    #   sorted_dist,index = sort(dist)
    #   index2 = sorted_dist(1:Nmax) < 4*a_epoch1
    #   ind_nearby_all(nn,:) = ...
    #
    # KDTree query(k=50, eps=0) is the exact nearest-neighbour
    # equivalent without O(N^2) full-distance scans.
    # --------------------------------------------------------------

    neighbour_path = (
        exact_dir
        / "neighbor_ids.i32"
    )

    count_path = (
        exact_dir
        / "neighbor_count.u1"
    )

    neighbour_meta_path = (
        exact_dir
        / "neighbor_meta.json"
    )

    neighbour_bytes = (
        int(n_ps)
        * int(nmax_spatial)
        * np.dtype(np.int32).itemsize
    )

    count_bytes = (
        int(n_ps)
        * np.dtype(np.uint8).itemsize
    )

    neighbour_meta = {
        "n_ps": int(n_ps),
        "nmax": int(nmax_spatial),
        "base_a_m": float(spatial_a[0]),
        "base_dx_max_m": float(base_dx_max),
        "rule": (
            "StaMPS first epoch nearest Nmax "
            "with single(distance) < 4*a1"
        ),
    }

    neighbour_ok = (
        reuse_exact
        and neighbour_path.is_file()
        and count_path.is_file()
        and neighbour_path.stat().st_size
            == neighbour_bytes
        and count_path.stat().st_size
            == count_bytes
        and neighbour_meta_path.is_file()
    )

    if neighbour_ok:
        try:
            old_meta = json.loads(
                neighbour_meta_path.read_text(
                    encoding="utf-8"
                )
            )
            neighbour_ok = (
                old_meta
                == neighbour_meta
            )
        except Exception:
            neighbour_ok = False

    if neighbour_ok:

        neighbor_ids = np.memmap(
            neighbour_path,
            mode="r",
            dtype=np.int32,
            shape=(
                n_ps,
                nmax_spatial,
            ),
        )

        neighbor_count = np.memmap(
            count_path,
            mode="r",
            dtype=np.uint8,
            shape=(n_ps,),
        )

        print(
            "[STAGE8_KRIG][SPATIAL_NEIGHBOR] "
            "reusing exact StaMPS candidate cache",
            flush=True,
        )

    else:

        neighbor_ids = np.memmap(
            neighbour_path,
            mode="w+",
            dtype=np.int32,
            shape=(
                n_ps,
                nmax_spatial,
            ),
        )

        neighbor_count = np.memmap(
            count_path,
            mode="w+",
            dtype=np.uint8,
            shape=(n_ps,),
        )

        t_neighbor = time.perf_counter()

        for start in range(
            0,
            n_ps,
            neighbour_query_chunk,
        ):

            stop = min(
                n_ps,
                start
                + neighbour_query_chunk,
            )

            dist, ids = tree.query(
                coords[start:stop],
                k=nmax_spatial,
                eps=0.0,
                workers=-1,
            )

            if nmax_spatial == 1:
                dist = dist[:, None]
                ids = ids[:, None]

            ids = np.asarray(
                ids,
                dtype=np.int64,
            )

            if (
                np.any(ids < 0)
                or np.any(ids >= n_ps)
            ):
                raise Stage8SbasKrigError(
                    "Invalid exact spatial neighbour index"
                )

            # Official first-epoch code explicitly casts
            # all full-scene distances to MATLAB single
            # before sorting/thresholding.
            dist_single = np.asarray(
                dist,
                dtype=np.float32,
            )

            cnt = np.sum(
                dist_single
                < np.float32(
                    base_dx_max
                ),
                axis=1,
            ).astype(np.int64)

            # Target itself is always distance zero.
            cnt = np.clip(
                cnt,
                1,
                nmax_spatial,
            )

            neighbor_ids[
                start:stop
            ] = ids.astype(
                np.int32
            )

            neighbor_count[
                start:stop
            ] = cnt.astype(
                np.uint8
            )

            if (
                stop == n_ps
                or stop % 1_000_000
                < neighbour_query_chunk
            ):
                print(
                    "[STAGE8_KRIG][SPATIAL_NEIGHBOR] "
                    f"{stop:,}/{n_ps:,}",
                    flush=True,
                )

        neighbor_ids.flush()
        neighbor_count.flush()

        neighbour_meta_path.write_text(
            json.dumps(
                neighbour_meta,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "[STAGE8_KRIG][SPATIAL_NEIGHBOR] "
            f"complete in "
            f"{time.perf_counter()-t_neighbor:.1f}s; "
            f"count p05/p50/p95="
            f"{np.percentile(neighbor_count,[5,50,95]).tolist()}",
            flush=True,
        )

    # --------------------------------------------------------------
    # Exact deramped ph_hpt for every PS/epoch.
    #
    # Stored as float64 so no extra precision approximation is
    # introduced by the HPC cache.
    # --------------------------------------------------------------

    exact_state = {
        "n_ps": int(n_ps),
        "n_img": int(n_img),
        "spatial_a": [
            float(x)
            for x in spatial_a
        ],
        "spatial_c1": [
            float(x)
            for x in spatial_c1
        ],
        "spatial_c2": [
            float(x)
            for x in spatial_c2
        ],
        "spatial_coef": np.asarray(
            spatial_coef,
            dtype=np.float64,
        ).tolist(),
    }

    state_path = (
        exact_dir
        / "exact_state.json"
    )

    state_same = False

    if (
        reuse_exact
        and state_path.is_file()
    ):
        try:
            state_same = (
                json.loads(
                    state_path.read_text(
                        encoding="utf-8"
                    )
                )
                == exact_state
            )
        except Exception:
            state_same = False

    residual_path = (
        exact_dir
        / "deramped_ph_hpt.f64"
    )

    residual_bytes = (
        int(n_ps)
        * int(n_img)
        * np.dtype(np.float64).itemsize
    )

    residual_ok = (
        state_same
        and residual_path.is_file()
        and residual_path.stat().st_size
            == residual_bytes
    )

    if residual_ok:

        residual_all = np.memmap(
            residual_path,
            mode="r",
            dtype=np.float64,
            shape=(
                n_ps,
                n_img,
            ),
        )

        print(
            "[STAGE8_KRIG][SPATIAL_EXACT] "
            "reusing deramped_ph_hpt.f64",
            flush=True,
        )

    else:

        residual_all = np.memmap(
            residual_path,
            mode="w+",
            dtype=np.float64,
            shape=(
                n_ps,
                n_img,
            ),
        )

        for start in range(
            0,
            n_ps,
            rt.ps_chunk,
        ):

            stop = min(
                n_ps,
                start + rt.ps_chunk,
            )

            Aall = _spatial_poly_design(
                coords[start:stop, 0],
                coords[start:stop, 1],
                mean_x_ps,
                mean_y_ps,
            )

            trend = (
                Aall
                @ np.asarray(
                    spatial_coef,
                    dtype=np.float64,
                ).T
            )

            residual_all[
                start:stop,
                :
            ] = (
                np.asarray(
                    ph_hpt[
                        start:stop,
                        :
                    ],
                    dtype=np.float64,
                )
                - trend
            )

        residual_all.flush()

        state_path.write_text(
            json.dumps(
                exact_state,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "[STAGE8_KRIG][SPATIAL_EXACT] "
            "deramped_ph_hpt.f64 complete",
            flush=True,
        )

    # --------------------------------------------------------------
    # Exact per-PS StaMPS ordinary kriging.
    #
    # Work layout is epoch-major so each target chunk can be
    # checkpointed without strided disk writes.
    # --------------------------------------------------------------

    exact_phase_path = (
        exact_dir
        / "ph_scn_epochmajor.f32"
    )

    exact_phase_bytes = (
        int(n_img)
        * int(n_ps)
        * np.dtype(np.float32).itemsize
    )

    progress_path = (
        exact_dir
        / "next_ps.txt"
    )

    output_reusable = (
        state_same
        and reuse_exact
        and exact_phase_path.is_file()
        and exact_phase_path.stat().st_size
            == exact_phase_bytes
        and progress_path.is_file()
    )

    if output_reusable:

        ph_scn_epochmajor = np.memmap(
            exact_phase_path,
            mode="r+",
            dtype=np.float32,
            shape=(
                n_img,
                n_ps,
            ),
        )

        try:
            next_ps = int(
                progress_path.read_text(
                    encoding="utf-8"
                ).strip()
            )
        except Exception:
            next_ps = 0

        next_ps = max(
            0,
            min(
                next_ps,
                n_ps,
            ),
        )

        print(
            "[STAGE8_KRIG][SPATIAL_EXACT] "
            f"resuming at PS {next_ps:,}/{n_ps:,}",
            flush=True,
        )

    else:

        ph_scn_epochmajor = np.memmap(
            exact_phase_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                n_img,
                n_ps,
            ),
        )

        next_ps = 0

        progress_path.write_text(
            "0",
            encoding="utf-8",
        )

    # Old grid-QA files are no longer applicable.
    for stale in (
        root
        / "stage8_krig_spatial_grid_selection_qa.csv",
        root
        / "stage8_krig_spatial_exact_reference_qa.csv",
        root
        / "stage8_krig_spatial_qa_ps.npy",
    ):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    t_exact = time.perf_counter()

    last_report = next_ps

    for start in range(
        next_ps,
        n_ps,
        exact_chunk,
    ):

        stop = min(
            n_ps,
            start + exact_chunk,
        )

        target_xy = np.asarray(
            coords[start:stop],
            dtype=np.float64,
        )

        ids = np.asarray(
            neighbor_ids[
                start:stop,
                :
            ],
            dtype=np.int64,
        )

        base_count = np.asarray(
            neighbor_count[
                start:stop
            ],
            dtype=np.int64,
        )

        # Geometry is epoch-independent.
        nxy = np.asarray(
            coords[ids],
            dtype=np.float64,
        )

        nx = nxy[:, :, 0]
        ny = nxy[:, :, 1]

        h0 = np.hypot(
            nx
            - target_xy[:, 0:1],
            ny
            - target_xy[:, 1:2],
        )

        dx = (
            nx[:, :, None]
            - nx[:, None, :]
        )

        dy = (
            ny[:, :, None]
            - ny[:, None, :]
        )

        h = np.hypot(
            dx,
            dy,
        )

        del dx, dy

        # Gather the full 49-date residual cube once.
        # B x 50 x 49
        z_all = np.asarray(
            residual_all[
                ids,
                :
            ],
            dtype=np.float64,
        )

        aps_chunk = np.empty(
            (
                n_img,
                stop - start,
            ),
            dtype=np.float64,
        )

        def solve_epoch(j):
            dx_max = (
                float(spatial_a[j])
                * 4.0
            )

            current_count = np.sum(
                h0 < dx_max,
                axis=1,
            ).astype(np.int64)

            count = np.minimum(
                base_count,
                current_count,
            )

            count = np.clip(
                count,
                1,
                nmax_spatial,
            )

            out = np.empty(
                stop - start,
                dtype=np.float64,
            )

            z_epoch = z_all[
                :,
                :,
                j,
            ]

            for k in np.unique(
                count
            ):
                k = int(k)

                rows = np.flatnonzero(
                    count == k
                )

                if rows.size == 0:
                    continue

                out[rows] = (
                    _stamps_ordinary_kriging_from_dist_batch(
                        h[
                            rows,
                            :k,
                            :k,
                        ],
                        h0[
                            rows,
                            :k,
                        ],
                        z_epoch[
                            rows,
                            :k,
                        ],
                        spatial_a[j],
                        spatial_c2[j],
                        spatial_c1[j],
                    )
                )

            return (
                int(j),
                out,
            )

        with ThreadPoolExecutor(
            max_workers=min(
                exact_workers,
                n_img,
            )
        ) as pool:

            futures = [
                pool.submit(
                    solve_epoch,
                    j,
                )
                for j in range(
                    n_img
                )
            ]

            for fut in as_completed(
                futures
            ):
                j, out = fut.result()

                aps_chunk[
                    j,
                    :
                ] = out

        # Add back the exact StaMPS polynomial component and
        # all-epoch plane ramp.
        A_target = _spatial_poly_design(
            target_xy[:, 0],
            target_xy[:, 1],
            mean_x_ps,
            mean_y_ps,
        )

        poly = (
            A_target
            @ np.asarray(
                spatial_coef,
                dtype=np.float64,
            ).T
        )

        ramp = np.asarray(
            ph_ramp[
                start:stop,
                :
            ],
            dtype=np.float64,
        )

        final_chunk = (
            aps_chunk.T
            + poly
            + ramp
        )

        ph_scn_epochmajor[
            :,
            start:stop,
        ] = final_chunk.T.astype(
            np.float32
        )

        ph_scn_epochmajor.flush()

        progress_path.write_text(
            str(stop),
            encoding="utf-8",
        )

        if (
            stop == n_ps
            or stop - last_report
            >= 100_000
        ):
            elapsed = (
                time.perf_counter()
                - t_exact
            )

            processed = (
                stop - next_ps
            )

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0.0
            )

            remain = (
                (n_ps - stop) / rate
                if rate > 0
                else float("nan")
            )

            print(
                "[STAGE8_KRIG][SPATIAL_EXACT] "
                f"{stop:,}/{n_ps:,} "
                f"({100.0*stop/n_ps:.2f}%) "
                f"rate={rate:,.0f} PS/s "
                f"ETA={remain/60.0:.1f} min",
                flush=True,
            )

            last_report = stop

    # --------------------------------------------------------------
    # Convert epoch-major exact work product to the normal
    # n_ps x n_img Stage-8 layout expected by the existing
    # StaMPS-compatible downstream code.
    # --------------------------------------------------------------

    ph_scn = np.memmap(
        work / "ph_scn.f32",
        mode="w+",
        dtype=np.float32,
        shape=(
            n_ps,
            n_img,
        ),
    )

    for start in range(
        0,
        n_ps,
        rt.ps_chunk,
    ):

        stop = min(
            n_ps,
            start + rt.ps_chunk,
        )

        ph_scn[
            start:stop,
            :
        ] = np.asarray(
            ph_scn_epochmajor[
                :,
                start:stop,
            ],
            dtype=np.float32,
        ).T

    ph_scn.flush()

    print(
        "[STAGE8_KRIG][SPATIAL_EXACT] "
        "ALL PS / ALL EPOCHS COMPLETE",
        flush=True,
    )

    # Compatibility values for the old debug structure.
    # They no longer represent a grid approximation.
    grid_step = 0.0
    spatial_grid_qa = []
    spatial_final_qa = []
    qa_ps_n = 0


    with (
        root
        / "stage8_krig_spatial_exact_reference_qa.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "epoch_1based",
                "grid_m",
                "n_reference_ps",
                "rms_rad",
                "p95_abs_rad",
                "p99_abs_rad",
                "max_abs_rad",
            ]
        )

        for row in spatial_final_qa:

            w.writerow(
                [
                    row["epoch_1based"],
                    grid_step,
                    row["n_reference_ps"],
                    row["rms_rad"],
                    row["p95_abs_rad"],
                    row["p99_abs_rad"],
                    row["max_abs_rad"],
                ]
            )

    # ------------------------------------------------------------------
    # 10. StaMPS PS1 reference and master zero
    # ------------------------------------------------------------------
    ref = np.asarray(ph_scn[0, :], dtype=np.float32).copy()
    for start in range(0, n_ps, rt.ps_chunk):
        stop = min(n_ps, start + rt.ps_chunk)
        ph_scn[start:stop, :] = np.asarray(ph_scn[start:stop, :], dtype=np.float32) - ref[None, :]
    ph_scn[:, master0] = 0.0
    ph_scn.flush()

    ph_noise_res = np.memmap(work / "ph_noise_res.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_img))
    std_ph_noise = np.memmap(work / "std_ph_noise.f32", mode="w+", dtype=np.float32, shape=(n_ps,))
    for start in range(0, n_ps, rt.ps_chunk):
        stop = min(n_ps, start + rt.ps_chunk)
        nr = np.asarray(ph_hpt[start:stop, :], dtype=np.float32) - np.asarray(ph_scn[start:stop, :], dtype=np.float32)
        ph_noise_res[start:stop, :] = nr
        std_ph_noise[start:stop] = np.nanstd(nr, axis=1).astype(np.float32)
    ph_noise_res.flush(); std_ph_noise.flush()

    first_scn_max = float(np.max(np.abs(np.asarray(ph_scn[0, :], dtype=np.float64))))
    master_scn_max = float(np.max(np.abs(np.asarray(ph_scn[:, master0], dtype=np.float64))))
    if first_scn_max > 1.0e-6 or master_scn_max > 1.0e-6:
        raise Stage8SbasKrigError(
            f"Reference/master invariant failed: PS1={first_scn_max}, master={master_scn_max}"
        )

    write_mat(
        root / "scn2.mat",
        {
            "ph_scn_slave": ph_scn,
            "ph_hpt": ph_hpt,
            "ph_ramp": ph_ramp,
            "ph_noise_res": ph_noise_res,
            "std_ph_noise": std_ph_noise.reshape(-1, 1),
        },
    )

    duration = time.perf_counter() - started
    debug = {
        "status": "completed",
        "implementation": "STAMPS_PS_SCN_FILT_KRIG_HPC_V1",
        "scientific_reference": "StaMPS 4.0b6 matlab/ps_scn_filt_krig.m",
        "duration_sec": duration,
        "n_ps": n_ps,
        "n_image": n_img,
        "n_ifg": n_ifg,
        "n_edges": n_edges,
        "master_ix": master_ix,
        "phase_file": phase_file,
        "edge_backend": edge_backend,
        "deformation_model_columns": ["years^2", "years", "sin(2*pi*years)", "cos(2*pi*years)-1", "1"],
        "temporal_variogram_sample_edges": sample_count,
        "temporal_variogram_valid_edges": n_valid_vario,
        "mean_std_rad": mean_std,
        "mean_std_definition": (
            "StaMPS nanmean(nanstd(dph)): "
            "sample std across edges for each epoch, "
            "then mean across epochs"
        ),
        "temporal_mode": temporal_mode,
        "temporal_tile_m": temporal_tile_used,
        "temporal_tile_qa": temporal_qa,
        "time_nmax": nmax_time,
        "time_win_fallback_days": time_win,
        "unwrap_error_applied": False,
        "unwrap_error_reason": "Matches StaMPS: corrected_dph is diagnostic; dph=corrected_dph is commented out.",
        "edge_to_ps_qa": edge_to_ps_qa,
        "spatial_variogram_sample_ps": n_vario_ps,
        "spatial_variogram_pairs_within_30km": int(
            spatial_pair_i.size
        ),
        "spatial_variogram_bounds": (
            "StaMPS parity: non-negative, "
            "no upper bounds"
        ),
        "spatial_ranges_m": spatial_a.tolist(),
        "spatial_nugget": spatial_c2.tolist(),
        "spatial_sill": spatial_c1.tolist(),
        "spatial_grid_m": grid_step,
        "spatial_first_epoch_neighbor_cap_m": spatial_base_dx_max,
        "spatial_grid_qa": [],
        "spatial_final_exact_reference_qa": [],
        "spatial_exact_reference_ps": int(n_ps),
        "spatial_exact_reference_rule": (
            "FULL StaMPS exact-local kriging: "
            "epoch-1 nearest-50 candidate cache "
            "within 4*a1; later epochs reuse "
            "candidates and apply current 4*a"
        ),
        "spatial_qa_thresholds_rad": {
            "rms":
                rt.spatial_qa_tol_rad,
            "p95_abs":
                rt.spatial_qa_p95_tol_rad,
            "p99_abs":
                rt.spatial_qa_p99_tol_rad,
        },
        "reference": "first PS, matching StaMPS ps_scn_filt_krig.m",
        "master_scn_zero": True,
        "hpc_accelerations": {
            "temporal": "spatial tiles share ordinary-kriging operator; exact per-edge QA controls tile size",
            "spatial": (
                "FULL all-PS StaMPS exact-local ordinary "
                "kriging; KD-tree/cache/batching/threads "
                "change execution only, not the kriging equations"
            ),
            "spatial_variogram": (
                "StaMPS empirical variogram using all "
                "eligible pairs within decor_dist; symmetric "
                "duplicate pairs evaluated once because their "
                "bin means are mathematically identical"
            ),
        },
        "runtime": {
            k: getattr(rt, k) for k in rt.__dataclass_fields__
        },
    }
    _write_json(debug_path, debug)

    print("=" * 96)
    print("STAGE 8 StaMPS KRIGING COMPLETE")
    print("=" * 96)
    print("scn2.mat                 : written")
    print("ph_scn_slave             :", (n_ps, n_img), "float32")
    print("max |ph_scn PS1|         :", first_scn_max)
    print("max |ph_scn master|      :", master_scn_max)
    print("temporal mode            :", temporal_mode)
    print("temporal tile            :", temporal_tile_used)
    print(
        "spatial mode             :",
        "StaMPS exact local / all PS",
    )
    print("duration sec             :", duration)
    return (
        f"Stage 8 StaMPS ps_scn_filt_krig completed for {n_ps} PS, "
        f"{n_edges} arcs and {n_img} acquisitions"
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StaMPS ps_scn_filt_krig SBAS Stage 8")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--chunk-edges", type=int, default=0)
    ap.add_argument("--chunk-ps", type=int, default=0)
    ap.add_argument("--triangle", default=None)
    ap.add_argument("--phase-file", default="phuw2.mat")
    ns = ap.parse_args()
    print(stage8_sbas_filter_scn_krig(
        ns.dataset,
        chunk_edges=ns.chunk_edges,
        chunk_ps=ns.chunk_ps,
        triangle_path=ns.triangle,
        phase_file=ns.phase_file,
    ))
