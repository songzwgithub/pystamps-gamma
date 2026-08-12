#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6.2 — Stage7/Stage8 layer-attribution audit.

Goal
----
Identify which processing layer actually improves or degrades the deformation
result, without filtering the final velocity map and without changing 4:1 PS
sampling.

Two inversion engines
---------------------
IFGSTD
    Current Stage7 global ifg_std weighted acquisition inversion.

MINTPYNO
    MintPy-style minimum-norm interval-velocity inversion with no IFG weights.

Five layers for each engine
---------------------------
L0_RAW
    GACOS-corrected IFGs + fixed 65-PS region reference + network inversion.

L1_DERAMP
    Current global 2-D IFG ramp removal + reference + network inversion.

L2_SCLA_RAW
    L1 + branch-specific raw SCLA (K*Bperp) + raw C removal.

L3_SCLA_ENVELOPE
    L1 + branch-specific K/C neighbour-envelope clipping exactly matching the
    current Stage7 concept.

L4_STAGE8_SCN
    L3 + the actual current Stage8 residual correction:
        temporal Gaussian high-pass residual
        -> spatial Gaussian low-pass
        -> subtract the spatially correlated residual.

There is intentionally NO "temporal-only correction" layer because the actual
Stage8 temporal filter only constructs the residual that is then spatially
filtered; disabling the spatial step yields ph_scn == 0.

No production files are overwritten. No hash/authentication logic is used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, spatial
from scipy.spatial import cKDTree

# Loaded only for a real run, after --self-test handling.
ported = None
s7 = None
read_mat = None
read_mat_variables = None
load_sbas_network = None
_stage7_phase_input = None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def matlab_datenum_to_datetime(value: float) -> dt.datetime:
    integer = int(math.floor(float(value)))
    fraction = float(value) - integer
    return (
        dt.datetime.fromordinal(integer)
        + dt.timedelta(days=fraction)
        - dt.timedelta(days=366)
    )


def slope_coeff(dates: list[dt.datetime], year: int | None) -> np.ndarray:
    if year is None:
        ids = np.arange(len(dates), dtype=np.int64)
    else:
        ids = np.asarray(
            [i for i, d in enumerate(dates) if d.year == year],
            dtype=np.int64,
        )

    out = np.zeros(len(dates), dtype=np.float64)
    if ids.size < 2:
        return out

    t0 = dates[ids[0]]
    t = np.asarray(
        [(dates[i] - t0).total_seconds() / 86400.0 / 365.2425 for i in ids],
        dtype=np.float64,
    )
    tc = t - np.mean(t)
    den = float(np.sum(tc * tc))
    if den > 0:
        out[ids] = tc / den
    return out


def local_xy(lon: np.ndarray, lat: np.ndarray):
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y, lon0, lat0


def choose_vfield(gdf, preferred: str) -> str:
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if c.lower() == preferred.lower():
            return c
    raise RuntimeError(f"truth field {preferred!r} not found")


def read_truth(path: Path, preferred: str, scale: float, lon0: float, lat0: float):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)

    field = choose_vfield(gdf, preferred)
    lon = np.asarray(gdf.geometry.x, dtype=np.float64)
    lat = np.asarray(gdf.geometry.y, dtype=np.float64)
    vel = np.asarray(gdf[field], dtype=np.float64) * scale

    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y, vel


def unique_match(psx, psy, tx, ty, match_m):
    tree = cKDTree(np.column_stack([tx, ty]))
    dist, tidx = tree.query(np.column_stack([psx, psy]), k=1, workers=-1)

    good = np.isfinite(dist) & (dist <= match_m)
    p = np.flatnonzero(good)
    t = tidx[good]
    d = dist[good]

    order = np.argsort(d)
    t_sorted = t[order]
    _, first = np.unique(t_sorted, return_index=True)
    keep = np.sort(order[first])

    return p[keep], t[keep]


def metrics(pred, truth):
    good = np.isfinite(pred) & np.isfinite(truth)
    p = np.asarray(pred[good], dtype=np.float64)
    t = np.asarray(truth[good], dtype=np.float64)

    if p.size == 0:
        return {
            "n": 0,
            "rmse_mm_yr": np.nan,
            "mae_mm_yr": np.nan,
            "bias_mm_yr": np.nan,
            "correlation": np.nan,
            "pred_std_mm_yr": np.nan,
            "truth_std_mm_yr": np.nan,
        }

    e = p - t
    corr = (
        float(np.corrcoef(p, t)[0, 1])
        if p.size > 2 and np.std(p) > 0 and np.std(t) > 0
        else np.nan
    )

    return {
        "n": int(e.size),
        "rmse_mm_yr": float(np.sqrt(np.mean(e * e))),
        "mae_mm_yr": float(np.mean(np.abs(e))),
        "bias_mm_yr": float(np.mean(e)),
        "correlation": corr,
        "pred_std_mm_yr": float(np.std(p)),
        "truth_std_mm_yr": float(np.std(t)),
    }


# =============================================================================
# MintPy-NO minimum-norm interval velocity
# =============================================================================

def build_velocity_design(ifgday_ix: np.ndarray, day: np.ndarray):
    n_image = len(day)
    dt_days = np.diff(day).astype(np.float64)

    rows = []
    cols = []
    vals = []
    max_span = 0

    for e, pair in enumerate(ifgday_ix):
        a = int(pair[0]) - 1
        b = int(pair[1]) - 1
        if a < b:
            lo, hi, sign = a, b, 1.0
        else:
            lo, hi, sign = b, a, -1.0

        max_span = max(max_span, hi - lo)

        for j in range(lo, hi):
            rows.append(e)
            cols.append(j)
            vals.append(sign * dt_days[j])

    B = sparse.csr_matrix(
        (np.asarray(vals), (rows, cols)),
        shape=(len(ifgday_ix), n_image - 1),
        dtype=np.float64,
    )
    return B, dt_days, max_span


def build_normal_contributors(ifgday_ix, n_image, bandwidth):
    mats = []
    n = n_image - 1

    for d in range(bandwidth + 1):
        rr = []
        cc = []

        for e, pair in enumerate(ifgday_ix):
            a = int(pair[0]) - 1
            b = int(pair[1]) - 1
            lo, hi = min(a, b), max(a, b)

            if hi - lo <= d:
                continue

            for j in range(lo + d, hi):
                rr.append(e)
                cc.append(j - d)

        mats.append(
            sparse.csr_matrix(
                (np.ones(len(rr), dtype=np.float64), (rr, cc)),
                shape=(len(ifgday_ix), n - d),
            )
        )

    return mats


def normal_band_rhs(W, Y, B, contributors, bandwidth, dt_days):
    batch = W.shape[0]
    n = B.shape[1]
    Ab = np.zeros((batch, bandwidth + 1, n), dtype=np.float64)

    WT = W.T

    for d, C in enumerate(contributors):
        sums = (C.T @ WT).T
        if d == 0:
            coeff = dt_days * dt_days
        else:
            coeff = dt_days[d:] * dt_days[:-d]
        Ab[:, d, d:] = sums * coeff[None, :]

    rhs = (B.T @ (W * Y).T).T

    scale = np.nanmedian(Ab[:, 0, :], axis=1)
    ridge = 1.0e-12 * np.maximum(scale, 1.0e-12)
    Ab[:, 0, :] += ridge[:, None]

    return Ab, rhs


def solve_banded_batch(Ab, rhs):
    batch, bw1, n = Ab.shape
    bandwidth = bw1 - 1

    L = np.zeros_like(Ab)
    bad = np.zeros(batch, dtype=bool)
    eps = 1.0e-14

    for j in range(n):
        diag = Ab[:, 0, j].copy()

        for d in range(1, min(bandwidth, j) + 1):
            diag -= L[:, d, j] ** 2

        bad_j = (~np.isfinite(diag)) | (diag <= eps)
        bad |= bad_j
        diag = np.where(bad_j, eps, diag)
        L[:, 0, j] = np.sqrt(diag)

        for i in range(j + 1, min(n, j + bandwidth + 1)):
            d = i - j
            val = Ab[:, d, i].copy()
            k0 = max(0, j - bandwidth, i - bandwidth)

            for k in range(k0, j):
                val -= L[:, i-k, i] * L[:, j-k, j]

            L[:, d, i] = val / L[:, 0, j]

    z = np.empty_like(rhs)

    for j in range(n):
        val = rhs[:, j].copy()

        for d in range(1, min(bandwidth, j) + 1):
            val -= L[:, d, j] * z[:, j-d]

        z[:, j] = val / L[:, 0, j]

    x = np.empty_like(rhs)

    for j in range(n - 1, -1, -1):
        val = z[:, j].copy()

        for i in range(j + 1, min(n, j + bandwidth + 1)):
            val -= L[:, i-j, i] * x[:, i]

        x[:, j] = val / L[:, 0, j]

    x[bad, :] = np.nan
    return x, bad


def mintpy_no_invert_to_memmap(
    values,
    output,
    B,
    contributors,
    bandwidth,
    dt_days,
    reference_image,
    chunk_ps,
    label,
):
    n_ps = values.shape[0]
    solved_total = 0

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        Y = np.asarray(values[start:stop, :], dtype=np.float64)
        finite = np.isfinite(Y)

        W = finite.astype(np.float64)
        Y0 = np.where(finite, Y, 0.0)

        Ab, rhs = normal_band_rhs(
            W, Y0, B, contributors, bandwidth, dt_days
        )

        V, bad = solve_banded_batch(Ab, rhs)

        X = np.zeros((stop-start, len(dt_days) + 1), dtype=np.float64)
        X[:, 1:] = np.cumsum(V * dt_days[None, :], axis=1)

        # Match current Stage7 acquisition gauge.
        X -= X[:, reference_image][:, None]
        X[bad, :] = np.nan

        output[start:stop, :] = X.astype(np.float32)
        output.flush()

        solved_total += int(np.count_nonzero(np.all(np.isfinite(X), axis=1)))

        print(
            f"[V6.2][{label}] {stop}/{n_ps} solved={solved_total}",
            flush=True,
        )

    return solved_total


# =============================================================================
# Stage7 SCLA layers
# =============================================================================

def estimate_branch_scla(
    ph_proc,
    bp_sm,
    day,
    reference_image,
    ps2,
    root,
    work,
    engine,
    chunk_ps,
):
    n_ps, n_image = ph_proc.shape

    order = np.argsort(day, kind="stable")
    day_sorted = day[order]
    dt_seq = np.diff(day_sorted)

    bp_seq_mean = np.zeros(n_image - 1, dtype=np.float64)
    bp_count = np.zeros(n_image - 1, dtype=np.int64)

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        b = np.asarray(bp_sm[start:stop, :], dtype=np.float64)[:, order]
        db = np.diff(b, axis=1)

        finite = np.isfinite(db)
        bp_seq_mean += np.sum(np.where(finite, db, 0.0), axis=0)
        bp_count += np.sum(finite, axis=0)

    bp_seq_mean = np.divide(
        bp_seq_mean,
        np.maximum(bp_count, 1),
        out=np.zeros_like(bp_seq_mean),
        where=bp_count > 0,
    )

    X_seq = np.column_stack(
        (
            np.ones(n_image - 1, dtype=np.float64),
            bp_seq_mean,
            dt_seq,
        )
    )

    k = np.full(n_ps, np.nan, dtype=np.float64)
    v_seq = np.full(n_ps, np.nan, dtype=np.float64)
    valid = np.zeros(n_ps, dtype=bool)

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        phase_seq = np.diff(
            np.asarray(ph_proc[start:stop, :], dtype=np.float64)[:, order],
            axis=1,
        )

        coeff, ok = s7._fit_shared_design(
            phase_seq,
            X_seq,
            None,
            min_obs=4,
        )

        k[start:stop] = coeff[1, :]
        v_seq[start:stop] = coeff[2, :]
        valid[start:stop] = ok

    ph_scla = np.memmap(
        work / f"{engine}_ph_scla_raw.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    s7._make_phase_model(
        ph_scla,
        k,
        bp_sm,
        chunk_ps,
        f"V62_{engine}_PH_SCLA_RAW",
    )

    time_rel = np.asarray(day, dtype=np.float64) - float(day[reference_image])
    X_c = np.column_stack(
        (np.ones(n_image, dtype=np.float64), time_rel)
    )

    c = np.full(n_ps, np.nan, dtype=np.float64)
    mean_v = np.full(n_ps, np.nan, dtype=np.float64)

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        residual = (
            np.asarray(ph_proc[start:stop, :], dtype=np.float64)
            - np.asarray(ph_scla[start:stop, :], dtype=np.float64)
        )

        coeff, ok = s7._fit_shared_design(
            residual,
            X_c,
            None,
            min_obs=3,
        )

        c[start:stop] = coeff[0, :]
        mean_v[start:stop] = coeff[1, :]
        valid[start:stop] &= ok

    edges = ported._resolve_scla_smooth_edges(
        root,
        ps2,
        n_ps,
        triangle_path=None,
    )

    k_envelope, c_envelope = ported._smooth_scla_neighbor_envelope(
        k,
        c,
        edges,
    )

    ph_scla_envelope = np.memmap(
        work / f"{engine}_ph_scla_envelope.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    s7._make_phase_model(
        ph_scla_envelope,
        np.asarray(k_envelope, dtype=np.float64),
        bp_sm,
        chunk_ps,
        f"V62_{engine}_PH_SCLA_ENVELOPE",
    )

    changed_k = np.isfinite(k) & np.isfinite(k_envelope) & (np.asarray(k_envelope) != k)
    changed_c = np.isfinite(c) & np.isfinite(c_envelope) & (np.asarray(c_envelope) != c)

    return {
        "k": k,
        "c": c,
        "mean_v": mean_v,
        "valid": valid,
        "ph_scla": ph_scla,
        "k_envelope": np.asarray(k_envelope, dtype=np.float64),
        "c_envelope": np.asarray(c_envelope, dtype=np.float64),
        "ph_scla_envelope": ph_scla_envelope,
        "fraction_k_changed": float(np.mean(changed_k)),
        "fraction_c_changed": float(np.mean(changed_c)),
    }


def corrected_series(
    ph_proc,
    ph_scla,
    c,
    ref_ps,
    out_path,
    chunk_ps,
):
    n_ps, n_image = ph_proc.shape

    ref_values = (
        np.asarray(ph_proc[ref_ps, :], dtype=np.float64)
        - np.asarray(ph_scla[ref_ps, :], dtype=np.float64)
        - np.asarray(c[ref_ps], dtype=np.float64)[:, None]
    )

    reference_phase = np.nanmedian(ref_values, axis=0)
    reference_phase[~np.isfinite(reference_phase)] = 0.0

    out = np.memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = (
            np.asarray(ph_proc[start:stop, :], dtype=np.float64)
            - np.asarray(ph_scla[start:stop, :], dtype=np.float64)
            - np.asarray(c[start:stop], dtype=np.float64)[:, None]
            - reference_phase[None, :]
        )

        out[start:stop, :] = y.astype(np.float32)

    out.flush()
    return out, reference_phase


# =============================================================================
# Exact current Stage8 residual estimator
# =============================================================================

def stage8_scn_only(
    corrected,
    ps2,
    day,
    reference_image,
    ref_ps,
    parms,
    work,
    engine,
    chunk_ps,
):
    n_ps, n_image = corrected.shape

    # These are the exact current source defaults when parms are unset.
    time_win = float(
        np.asarray(parms.get("scn_time_win", 365.0)).reshape(-1)[0]
    )
    time_win = max(time_win, 1.0e-6)

    wavelength = float(
        np.asarray(parms.get("scn_wavelength", 100.0)).reshape(-1)[0]
    )
    wavelength = max(wavelength, 1.0e-6)

    radius = wavelength * 4.0
    k_neighbors = 32

    time_diff = day[:, None] - day[None, :]

    temporal_weights = np.exp(
        -(time_diff * time_diff) / (2.0 * time_win * time_win)
    )

    temporal_weights[:, reference_image] = 0.0

    row_sum = np.sum(temporal_weights, axis=1, keepdims=True)

    zero_rows = row_sum[:, 0] <= 0
    if np.any(zero_rows):
        temporal_weights[zero_rows, :] = 0.0
        temporal_weights[zero_rows, np.flatnonzero(zero_rows)] = 1.0
        row_sum = np.sum(temporal_weights, axis=1, keepdims=True)

    temporal_weights /= row_sum

    ph_hpt = np.memmap(
        work / f"{engine}_ph_hpt.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = np.asarray(corrected[start:stop, :], dtype=np.float64)
        low_time = y @ temporal_weights.T
        ph_hpt[start:stop, :] = (y - low_time).astype(np.float32)

    ph_hpt.flush()

    xy = np.asarray(ps2.get("xy"), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != n_ps or xy.shape[1] < 3:
        raise RuntimeError("ps2.xy is required for Stage8 SCN audit")

    coords = np.asarray(xy[:, 1:3], dtype=np.float64)

    k_use = min(k_neighbors, n_ps)

    tree = spatial.cKDTree(coords)

    distances, neighbours = tree.query(
        coords,
        k=k_use,
        distance_upper_bound=radius,
        workers=-1,
    )

    if k_use == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]

    invalid = neighbours >= n_ps

    neighbour_weights = np.exp(
        -(distances * distances) / (2.0 * wavelength * wavelength)
    )

    neighbour_weights[~np.isfinite(neighbour_weights)] = 0.0
    neighbour_weights[invalid] = 0.0

    neighbours_safe = neighbours.copy()
    neighbours_safe[invalid] = 0

    ph_scn = np.memmap(
        work / f"{engine}_ph_scn.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    spatial_chunk = 256

    for start in range(0, n_ps, spatial_chunk):
        stop = min(start + spatial_chunk, n_ps)

        idx = neighbours_safe[start:stop, :]
        w = neighbour_weights[start:stop, :]

        values = np.asarray(ph_hpt[idx, :], dtype=np.float64)
        finite = np.isfinite(values)

        weighted = np.where(finite, values, 0.0) * w[:, :, None]
        denom = np.sum(w[:, :, None] * finite, axis=1)

        smooth = np.divide(
            np.sum(weighted, axis=1),
            denom,
            out=np.zeros((stop-start, n_image), dtype=np.float64),
            where=denom > 0,
        )

        ph_scn[start:stop, :] = smooth.astype(np.float32)

    ph_scn.flush()

    # The current Stage8 source still uses mean here. Audit mean-vs-median before changing it.
    ref_scn_mean = np.nanmean(
        np.asarray(ph_scn[ref_ps, :], dtype=np.float64),
        axis=0,
    )
    ref_scn_median = np.nanmedian(
        np.asarray(ph_scn[ref_ps, :], dtype=np.float64),
        axis=0,
    )

    ref_scn_mean[~np.isfinite(ref_scn_mean)] = 0.0
    ref_scn_median[~np.isfinite(ref_scn_median)] = 0.0

    final = np.memmap(
        work / f"{engine}_ph_final.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        scn = (
            np.asarray(ph_scn[start:stop, :], dtype=np.float64)
            - ref_scn_mean[None, :]
        )

        ph_scn[start:stop, :] = scn.astype(np.float32)

        y = np.asarray(corrected[start:stop, :], dtype=np.float64) - scn
        final[start:stop, :] = y.astype(np.float32)

    ph_scn.flush()
    final.flush()

    return final, ph_scn, {
        "time_win_days": time_win,
        "wavelength_m": wavelength,
        "radius_m": radius,
        "k_neighbors": k_use,
        "reference_statistic_used": "mean",
        "ref_scn_mean_minus_median_rms_rad": float(
            np.sqrt(np.mean((ref_scn_mean - ref_scn_median) ** 2))
        ),
        "ref_scn_mean_minus_median_max_abs_rad": float(
            np.max(np.abs(ref_scn_mean - ref_scn_median))
        ),
    }


# =============================================================================
# Velocity / component diagnostics
# =============================================================================

def velocity_fields(
    series,
    ref_ps,
    coeffs,
    phase_to_mm,
    chunk_ps,
):
    n_ps = series.shape[0]

    reference = np.nanmedian(
        np.asarray(series[ref_ps, :], dtype=np.float64),
        axis=0,
    )
    reference[~np.isfinite(reference)] = 0.0

    out = {
        name: np.full(n_ps, np.nan, dtype=np.float32)
        for name in coeffs
    }

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = np.asarray(series[start:stop, :], dtype=np.float64)
        y -= reference[None, :]

        for name, c in coeffs.items():
            out[name][start:stop] = (
                (y @ c) * phase_to_mm
            ).astype(np.float32)

    return out


def field_stats(branch, fields, x, y):
    rows = []

    G = np.column_stack(
        (
            np.ones(len(x), dtype=np.float64),
            x / 100000.0,
            y / 100000.0,
        )
    )

    for period, value in fields.items():
        v = np.asarray(value, dtype=np.float64)
        good = np.isfinite(v)
        vv = v[good]

        beta, *_ = np.linalg.lstsq(
            G[good, :],
            vv,
            rcond=None,
        )

        p02, p50, p98 = np.percentile(vv, [2, 50, 98])

        rows.append(
            {
                "branch": branch,
                "period": period,
                "n": int(vv.size),
                "p02_mm_yr": float(p02),
                "median_mm_yr": float(p50),
                "p98_mm_yr": float(p98),
                "std_mm_yr": float(np.std(vv)),
                "plane_x_mm_yr_per_100km": float(beta[1]),
                "plane_y_mm_yr_per_100km": float(beta[2]),
            }
        )

    return rows


def component_stats(engine, component, fields):
    rows = []

    for period, value in fields.items():
        v = np.asarray(value, dtype=np.float64)
        v = v[np.isfinite(v)]

        p02, p50, p98 = np.percentile(v, [2, 50, 98])

        rows.append(
            {
                "engine": engine,
                "component": component,
                "period": period,
                "p02_mm_yr": float(p02),
                "median_mm_yr": float(p50),
                "p98_mm_yr": float(p98),
                "std_mm_yr": float(np.std(v)),
                "median_abs_mm_yr": float(np.median(np.abs(v))),
                "p95_abs_mm_yr": float(np.percentile(np.abs(v), 95)),
            }
        )

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        default="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized",
    )

    p.add_argument("--truth-dir", default="")
    p.add_argument("--truth-field", default="v")
    p.add_argument("--truth-scale", type=float, default=1.0)
    p.add_argument("--truth-match-m", type=float, default=200.0)

    p.add_argument("--chunk-ps", type=int, default=2048)

    p.add_argument("--out", default="")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--self-test", action="store_true")

    return p.parse_args()


def self_test():
    raw = np.asarray([[10.0, 12.0, 15.0]])
    proc = np.asarray([[8.0, 9.0, 11.0]])
    ramp = raw - proc
    assert np.allclose(raw - ramp, proc)

    # Temporal high-pass alone is not a Stage8 correction.
    corrected = np.asarray([[1.0, 2.0, 4.0]])
    T = np.eye(3)
    hpt = corrected - corrected @ T.T
    assert np.allclose(hpt, 0.0)

    print("SELF-TEST: PASS")


def load_project_modules():
    global ported, s7, read_mat, read_mat_variables
    global load_sbas_network, _stage7_phase_input

    from pystamps.io.mat import read_mat as _read_mat
    from pystamps.io.mat import read_mat_variables as _read_mat_variables
    from pystamps.pipeline import ported as _ported
    from pystamps.pipeline import stage7_sbas as _s7
    from pystamps.pipeline.stage6_sbas import load_sbas_network as _load_sbas_network
    from pystamps.pipeline.stage7_sbas import _stage7_phase_input as _phase_input

    read_mat = _read_mat
    read_mat_variables = _read_mat_variables
    ported = _ported
    s7 = _s7
    load_sbas_network = _load_sbas_network
    _stage7_phase_input = _phase_input


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return

    load_project_modules()

    started = time.time()

    root = Path(args.dataset).resolve()
    truth_dir = Path(args.truth_dir).resolve() if args.truth_dir else root / "cangzhou"

    out = (
        Path(args.out).resolve()
        if args.out
        else root / "_audit" / (
            "stage78_layer_attribution_v6_2_"
            + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )

    out.mkdir(parents=True, exist_ok=True)

    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    ps2 = read_mat(root / "ps2.mat")
    parms = read_mat(root / "parms.mat")

    n_ps = int(round(float(np.asarray(ps2["n_ps"]).reshape(-1)[0])))

    lonlat = np.asarray(ps2["lonlat"], dtype=np.float64)
    if lonlat.shape[0] != n_ps:
        lonlat = lonlat.T

    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    x, y, lon0, lat0 = local_xy(lon, lat)

    ref_ps = np.asarray(
        ported._select_reference_ps(ps2, parms),
        dtype=np.int64,
    ).reshape(-1)

    print("=" * 80)
    print("V6.2 Stage7/Stage8 layer attribution")
    print("=" * 80)
    print("n_ps:", n_ps)
    print("reference PS:", ref_ps.size)
    print("4:1 PS sampling unchanged")
    print("No production Stage7/Stage8 files will be overwritten")

    phase_input = _stage7_phase_input(root)

    ph_ifg_raw = s7._as_matrix(
        read_mat_variables(phase_input, ("ph_uw",))["ph_uw"],
        n_ps,
        "ph_uw",
        np.float32,
    )

    n_ifg = ph_ifg_raw.shape[1]

    bp_ifg = s7._as_matrix(
        read_mat_variables(root / "bp2.mat", ("bperp_mat",))["bperp_mat"],
        n_ps,
        "bp2.bperp_mat",
        np.float32,
    )

    ifg_std = np.asarray(
        read_mat_variables(root / "ifgstd2.mat", ("ifg_std",))["ifg_std"],
        dtype=np.float64,
    ).reshape(-1)

    day, ifgday_ix, _, network_source = load_sbas_network(root, n_ifg)
    day = np.asarray(day, dtype=np.float64).reshape(-1)
    ifgday_ix = np.asarray(ifgday_ix, dtype=np.int64)

    n_image = int(day.size)
    dates = [matlab_datenum_to_datetime(v) for v in day]

    G = s7._network_matrix(n_image, ifgday_ix)

    master_ix = int(
        round(float(np.asarray(ps2.get("master_ix", 1)).reshape(-1)[0]))
    )
    if master_ix < 1 or master_ix > n_image:
        master_ix = 1

    reference_image = master_ix - 1

    # Current Stage7 IFGSTD network.
    drop_network = s7._drop_set(parms, "drop_ifg_index")
    finite_std = np.isfinite(ifg_std) & (ifg_std > 0)

    network_mask = np.asarray(
        [i not in drop_network for i in range(1, n_ifg + 1)],
        dtype=bool,
    ) & finite_std

    use_ix = np.flatnonzero(network_mask)

    variance_ifg = (ifg_std * math.pi / 180.0) ** 2

    weights_ifg = np.zeros(n_ifg, dtype=np.float64)
    weights_ifg[network_mask] = 1.0 / variance_ifg[network_mask]

    median_weight = float(np.median(weights_ifg[network_mask]))
    if median_weight > 0:
        weights_ifg /= median_weight

    projector, unknown, rank = s7._network_projector(
        G,
        use_ix,
        weights_ifg[use_ix],
        reference_image,
    )

    # Current Stage7 deramp.
    ph_ifg_float = np.asarray(ph_ifg_raw, dtype=np.float64)

    ph_ifg_deramped, ph_ramp_ifg = ported._deramp_unwrapped_phase(
        ps2,
        ph_ifg_float,
    )

    # Same fixed reference for raw and deramped IFGs.
    raw_ref = np.nanmedian(ph_ifg_float[ref_ps, :], axis=0)
    proc_ref = np.nanmedian(ph_ifg_deramped[ref_ps, :], axis=0)

    raw_ref[~np.isfinite(raw_ref)] = 0.0
    proc_ref[~np.isfinite(proc_ref)] = 0.0

    ph_ifg_centered = ph_ifg_float - raw_ref[None, :]
    ph_ifg_proc = ph_ifg_deramped - proc_ref[None, :]

    # Bperp acquisition series: keep current Stage7 projector common to both engines.
    bp_sm = np.memmap(
        work / "bp_sm_common.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    s7._invert_network(
        np.asarray(bp_ifg, dtype=np.float64),
        G=G,
        use_ix=use_ix,
        weights=weights_ifg[use_ix],
        projector=projector,
        unknown=unknown,
        reference_image=reference_image,
        output=bp_sm,
        chunk_ps=args.chunk_ps,
        label="V62_BPERP_COMMON",
    )

    # MintPy-NO design.
    B, dt_days, max_span = build_velocity_design(ifgday_ix, day)
    bandwidth = max_span - 1

    contributors = build_normal_contributors(
        ifgday_ix,
        n_image,
        bandwidth,
    )

    wavelength = float(np.asarray(parms["lambda"]).reshape(-1)[0])
    phase_to_mm = -wavelength / (4.0 * np.pi) * 1000.0

    coeffs = {
        "FULL": slope_coeff(dates, None),
        "2021": slope_coeff(dates, 2021),
        "2022": slope_coeff(dates, 2022),
        "2023": slope_coeff(dates, 2023),
    }

    # Truth mappings, referenced to the same configured region.
    R = 6371008.8
    ref_center = np.asarray(
        parms["ref_centre_lonlat"],
        dtype=np.float64,
    ).reshape(-1)
    ref_radius = float(np.asarray(parms["ref_radius_m"]).reshape(-1)[0])

    refx = np.deg2rad(ref_center[0] - lon0) * R * np.cos(np.deg2rad(lat0))
    refy = np.deg2rad(ref_center[1] - lat0) * R

    truth = {}

    for year in (2021, 2022, 2023):
        tx, ty, tv = read_truth(
            truth_dir / f"result{year}.shp",
            args.truth_field,
            args.truth_scale,
            lon0,
            lat0,
        )

        tree = cKDTree(np.column_stack([tx, ty]))

        rid = np.asarray(
            tree.query_ball_point(
                [refx, refy],
                r=ref_radius,
            ),
            dtype=np.int64,
        )

        truth_ref = float(np.nanmedian(tv[rid]))
        tv = tv - truth_ref

        pidx, tidx = unique_match(
            x,
            y,
            tx,
            ty,
            args.truth_match_m,
        )

        truth[year] = (tv, pidx, tidx, truth_ref)

        print(
            year,
            "truth matched:",
            len(pidx),
            "truth ref:",
            truth_ref,
        )

    branch_fields: dict[str, dict[str, np.ndarray]] = {}
    branch_stats_rows = []
    component_rows = []
    scla_rows = []
    stage8_settings = {}

    engines = ("IFGSTD", "MINTPYNO")

    for engine in engines:
        print("\n" + "=" * 80)
        print("ENGINE:", engine)
        print("=" * 80)

        raw_sm = np.memmap(
            work / f"{engine}_raw_sm.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
        )

        proc_sm = np.memmap(
            work / f"{engine}_proc_sm.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
        )

        if engine == "IFGSTD":
            s7._invert_network(
                ph_ifg_centered,
                G=G,
                use_ix=use_ix,
                weights=weights_ifg[use_ix],
                projector=projector,
                unknown=unknown,
                reference_image=reference_image,
                output=raw_sm,
                chunk_ps=args.chunk_ps,
                label="V62_IFGSTD_RAW",
            )

            s7._invert_network(
                ph_ifg_proc,
                G=G,
                use_ix=use_ix,
                weights=weights_ifg[use_ix],
                projector=projector,
                unknown=unknown,
                reference_image=reference_image,
                output=proc_sm,
                chunk_ps=args.chunk_ps,
                label="V62_IFGSTD_DERAMP",
            )

        else:
            mintpy_no_invert_to_memmap(
                ph_ifg_centered,
                raw_sm,
                B,
                contributors,
                bandwidth,
                dt_days,
                reference_image,
                args.chunk_ps,
                "MINTPYNO_RAW",
            )

            mintpy_no_invert_to_memmap(
                ph_ifg_proc,
                proc_sm,
                B,
                contributors,
                bandwidth,
                dt_days,
                reference_image,
                args.chunk_ps,
                "MINTPYNO_DERAMP",
            )

        # L0 RAW and L1 DERAMP.
        name_l0 = f"{engine}_L0_RAW"
        name_l1 = f"{engine}_L1_DERAMP"

        branch_fields[name_l0] = velocity_fields(
            raw_sm,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        branch_fields[name_l1] = velocity_fields(
            proc_sm,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        branch_stats_rows += field_stats(
            name_l0,
            branch_fields[name_l0],
            x,
            y,
        )

        branch_stats_rows += field_stats(
            name_l1,
            branch_fields[name_l1],
            x,
            y,
        )

        # Ramp component = raw acquisition series - deramped acquisition series.
        ramp = np.memmap(
            work / f"{engine}_ramp_sm.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
        )

        for start in range(0, n_ps, args.chunk_ps):
            stop = min(start + args.chunk_ps, n_ps)

            ramp[start:stop, :] = (
                np.asarray(raw_sm[start:stop, :], dtype=np.float64)
                - np.asarray(proc_sm[start:stop, :], dtype=np.float64)
            ).astype(np.float32)

        ramp.flush()

        ramp_fields = velocity_fields(
            ramp,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        component_rows += component_stats(
            engine,
            "RAMP",
            ramp_fields,
        )

        # Branch-specific SCLA.
        scla = estimate_branch_scla(
            proc_sm,
            bp_sm,
            day,
            reference_image,
            ps2,
            root,
            work,
            engine,
            args.chunk_ps,
        )

        scla_rows.append(
            {
                "engine": engine,
                "valid_fraction": float(np.mean(scla["valid"])),
                "fraction_k_changed_by_envelope": scla["fraction_k_changed"],
                "fraction_c_changed_by_envelope": scla["fraction_c_changed"],
                "k_median": float(np.nanmedian(scla["k"])),
                "k_p02": float(np.nanpercentile(scla["k"], 2)),
                "k_p98": float(np.nanpercentile(scla["k"], 98)),
                "c_median": float(np.nanmedian(scla["c"])),
                "c_p02": float(np.nanpercentile(scla["c"], 2)),
                "c_p98": float(np.nanpercentile(scla["c"], 98)),
            }
        )

        corr_raw, _ = corrected_series(
            proc_sm,
            scla["ph_scla"],
            scla["c"],
            ref_ps,
            work / f"{engine}_corrected_scla_raw.f32",
            args.chunk_ps,
        )

        corr_envelope, _ = corrected_series(
            proc_sm,
            scla["ph_scla_envelope"],
            scla["c_envelope"],
            ref_ps,
            work / f"{engine}_corrected_scla_envelope.f32",
            args.chunk_ps,
        )

        name_l2 = f"{engine}_L2_SCLA_RAW"
        name_l3 = f"{engine}_L3_SCLA_ENVELOPE"

        branch_fields[name_l2] = velocity_fields(
            corr_raw,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        branch_fields[name_l3] = velocity_fields(
            corr_envelope,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        branch_stats_rows += field_stats(
            name_l2,
            branch_fields[name_l2],
            x,
            y,
        )

        branch_stats_rows += field_stats(
            name_l3,
            branch_fields[name_l3],
            x,
            y,
        )

        scla_raw_fields = velocity_fields(
            scla["ph_scla"],
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        component_rows += component_stats(
            engine,
            "SCLA_RAW",
            scla_raw_fields,
        )

        envelope_delta = np.memmap(
            work / f"{engine}_scla_envelope_delta.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
        )

        for start in range(0, n_ps, args.chunk_ps):
            stop = min(start + args.chunk_ps, n_ps)

            envelope_delta[start:stop, :] = (
                np.asarray(
                    scla["ph_scla_envelope"][start:stop, :],
                    dtype=np.float64,
                )
                - np.asarray(
                    scla["ph_scla"][start:stop, :],
                    dtype=np.float64,
                )
            ).astype(np.float32)

        envelope_delta.flush()

        envelope_fields = velocity_fields(
            envelope_delta,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        component_rows += component_stats(
            engine,
            "SCLA_ENVELOPE_MINUS_RAW",
            envelope_fields,
        )

        # L4 exact current Stage8 residual correction.
        final, ph_scn, scn_settings = stage8_scn_only(
            corr_envelope,
            ps2,
            day,
            reference_image,
            ref_ps,
            parms,
            work,
            engine,
            args.chunk_ps,
        )

        stage8_settings[engine] = scn_settings

        name_l4 = f"{engine}_L4_STAGE8_SCN"

        branch_fields[name_l4] = velocity_fields(
            final,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        branch_stats_rows += field_stats(
            name_l4,
            branch_fields[name_l4],
            x,
            y,
        )

        scn_fields = velocity_fields(
            ph_scn,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        component_rows += component_stats(
            engine,
            "STAGE8_SCN",
            scn_fields,
        )

        # Drop references to large memmaps before the next engine.
        del raw_sm, proc_sm, ramp
        del corr_raw, corr_envelope
        del envelope_delta, final, ph_scn
        del scla
        gc.collect()

    # -------------------------------------------------------------------------
    # Truth validation
    # -------------------------------------------------------------------------
    truth_rows = []
    pooled_rows = []

    for branch, fields in branch_fields.items():
        pp = []
        tt = []

        for year in (2021, 2022, 2023):
            tv, pidx, tidx, _ = truth[year]

            pred = fields[str(year)][pidx]
            obs = tv[tidx]

            m = metrics(pred, obs)

            truth_rows.append(
                {
                    "branch": branch,
                    "year": year,
                    **m,
                }
            )

            good = np.isfinite(pred) & np.isfinite(obs)
            pp.append(pred[good])
            tt.append(obs[good])

        pooled = metrics(
            np.concatenate(pp),
            np.concatenate(tt),
        )

        pooled_rows.append(
            {
                "branch": branch,
                **pooled,
            }
        )

    write_csv(out / "01_truth_by_year.csv", truth_rows)
    write_csv(out / "02_truth_pooled.csv", pooled_rows)
    write_csv(out / "03_branch_field_stats.csv", branch_stats_rows)
    write_csv(out / "04_component_velocity_stats.csv", component_rows)
    write_csv(out / "05_scla_parameter_audit.csv", scla_rows)

    # -------------------------------------------------------------------------
    # Incremental RMSE attribution
    # -------------------------------------------------------------------------
    layer_order = [
        "L0_RAW",
        "L1_DERAMP",
        "L2_SCLA_RAW",
        "L3_SCLA_ENVELOPE",
        "L4_STAGE8_SCN",
    ]

    pooled_map = {
        row["branch"]: row
        for row in pooled_rows
    }

    incremental_rows = []

    for engine in engines:
        previous = None

        for layer in layer_order:
            branch = f"{engine}_{layer}"
            current = pooled_map[branch]["rmse_mm_yr"]

            incremental_rows.append(
                {
                    "engine": engine,
                    "layer": layer,
                    "branch": branch,
                    "pooled_rmse_mm_yr": current,
                    "delta_rmse_vs_previous_mm_yr": (
                        np.nan
                        if previous is None
                        else current - previous
                    ),
                    "improvement_vs_previous_percent": (
                        np.nan
                        if previous is None
                        else 100.0 * (previous - current) / previous
                    ),
                }
            )

            previous = current

    write_csv(out / "06_incremental_rmse.csv", incremental_rows)

    # Compact velocity-only product for map comparison.
    payload = {
        "ps_index_1based": np.arange(1, n_ps + 1, dtype=np.int32),
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
    }

    for branch, fields in branch_fields.items():
        for period, value in fields.items():
            payload[f"{branch}__{period}_mm_yr"] = value

    np.savez_compressed(
        out / "branch_velocity_points.npz",
        **payload,
    )

    best = min(
        pooled_rows,
        key=lambda r: r["rmse_mm_yr"],
    )

    summary = {
        "input_phase": str(phase_input),
        "network_source": str(network_source),
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "reference_ps": int(ref_ps.size),
        "reference_image_ix_1based": int(master_ix),
        "current_ifgstd_network_rank": int(rank),
        "engines": list(engines),
        "layers": layer_order,
        "stage7_deramp": {
            "model": "one unweighted global 2-D plane per IFG using ps.xy/1000",
            "source_default_when_unset": "enabled",
        },
        "scla_envelope": (
            "clip each K/C to the neighbour min/max envelope; not a mean/Gaussian smoother"
        ),
        "stage8": stage8_settings,
        "best_truth_branch": best["branch"],
        "best_truth_pooled_rmse_mm_yr": best["rmse_mm_yr"],
        "important": (
            "No temporal-only Stage8 branch is reported because the actual temporal "
            "filter only constructs the residual used by the spatial SCN estimator."
        ),
        "runtime_seconds": time.time() - started,
    }

    (out / "07_SUMMARY.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPOOLED TRUTH")
    for row in sorted(
        pooled_rows,
        key=lambda r: r["rmse_mm_yr"],
    ):
        print(
            f"{row['branch']:34s} "
            f"RMSE={row['rmse_mm_yr']:.4f} "
            f"corr={row['correlation']:.4f}"
        )

    print(
        "\nBEST:",
        best["branch"],
        best["rmse_mm_yr"],
    )

    if not args.keep_work:
        del bp_sm
        gc.collect()
        shutil.rmtree(work, ignore_errors=True)
        print("Temporary audit work removed.")

    print("Output:", out)


if __name__ == "__main__":
    main()
