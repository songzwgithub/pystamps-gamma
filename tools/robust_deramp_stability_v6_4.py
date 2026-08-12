#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6.4 — Robust-deramp parameter stability audit.

Everything except the robust ramp sampling geometry is frozen:

    GACOS-corrected unwrapped IFGs
      -> fixed 65-PS median reference
      -> current IFGSTD WLS network inversion
      -> direct Stage8 SCN
      -> SCN median reference
      -> deformation / velocity

No SCLA is used.
No final velocity-map smoothing is used.
No IFG/PS deletion is introduced.
No production Stage7/Stage8 files are overwritten.
No hash/authentication logic is used.

Five pre-declared robust-ramp configurations are tested:

    R1 : 1500 m cells,  8 anchors/cell
    R2 : 2000 m cells,  8 anchors/cell   (pre-declared reference config)
    R3 : 2500 m cells,  8 anchors/cell
    R4 : 2000 m cells,  4 anchors/cell
    R5 : 2000 m cells, 12 anchors/cell

Huber delta and iterations are fixed for all configurations.
Truth is never used to fit the ramp or choose anchors.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat
from scipy import spatial
from scipy.spatial import cKDTree

ported = None
s7 = None
read_mat = None
read_mat_variables = None
load_sbas_network = None
_stage7_phase_input = None


CONFIGS = (
    ("R1_C1500_P8", 1500.0, 8),
    ("R2_C2000_P8", 2000.0, 8),
    ("R3_C2500_P8", 2500.0, 8),
    ("R4_C2000_P4", 2000.0, 4),
    ("R5_C2000_P12", 2000.0, 12),
)

REFERENCE_CONFIG = "R2_C2000_P8"


# -----------------------------------------------------------------------------
# I/O and generic helpers
# -----------------------------------------------------------------------------

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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MatrixReader:
    def __init__(self, path: Path, var: str, n_rows: int):
        self.path = Path(path)
        self.var = var
        self.n_rows = int(n_rows)
        self.hdf5 = h5py.is_hdf5(self.path)

        if self.hdf5:
            with h5py.File(self.path, "r") as h:
                shape = tuple(h[var].shape)
            if shape[0] == n_rows:
                self.transpose = False
                self.shape = shape
            elif shape[1] == n_rows:
                self.transpose = True
                self.shape = (shape[1], shape[0])
            else:
                raise RuntimeError(
                    f"{path}:{var} shape {shape} does not contain n_ps={n_rows}"
                )
            self.array = None
        else:
            arr = np.asarray(
                loadmat(
                    self.path,
                    variable_names=[var],
                    squeeze_me=False,
                )[var]
            )
            if arr.shape[0] == n_rows:
                self.transpose = False
                self.array = arr
            elif arr.shape[1] == n_rows:
                self.transpose = True
                self.array = arr.T
            else:
                raise RuntimeError(
                    f"{path}:{var} shape {arr.shape} does not contain n_ps={n_rows}"
                )
            self.shape = self.array.shape

    def block(self, start: int, stop: int) -> np.ndarray:
        if self.hdf5:
            with h5py.File(self.path, "r") as h:
                ds = h[self.var]
                if not self.transpose:
                    return np.asarray(ds[start:stop, :], dtype=np.float32)
                return np.asarray(ds[:, start:stop], dtype=np.float32).T
        return np.asarray(self.array[start:stop, :], dtype=np.float32)

    def rows(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if not self.hdf5:
            return np.asarray(self.array[ids, :], dtype=np.float32)

        order = np.argsort(ids)
        sorted_ids = ids[order]

        with h5py.File(self.path, "r") as h:
            ds = h[self.var]
            if not self.transpose:
                values = np.asarray(ds[sorted_ids, :], dtype=np.float32)
            else:
                values = np.asarray(ds[:, sorted_ids], dtype=np.float32).T

        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return values[inverse, :]


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


def load_project_modules():
    global ported, s7, read_mat, read_mat_variables
    global load_sbas_network, _stage7_phase_input

    from pystamps.io.mat import read_mat as _read_mat
    from pystamps.io.mat import read_mat_variables as _read_mat_variables
    from pystamps.pipeline import ported as _ported
    from pystamps.pipeline import stage7_sbas as _s7
    from pystamps.pipeline.stage6_sbas import load_sbas_network as _load_network
    from pystamps.pipeline.stage7_sbas import _stage7_phase_input as _phase_input

    read_mat = _read_mat
    read_mat_variables = _read_mat_variables
    ported = _ported
    s7 = _s7
    load_sbas_network = _load_network
    _stage7_phase_input = _phase_input


# -----------------------------------------------------------------------------
# Truth
# -----------------------------------------------------------------------------

def choose_vfield(gdf, preferred: str) -> str:
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if col.lower() == preferred.lower():
            return col
    raise RuntimeError(f"truth field {preferred!r} not found")


def read_truth(path: Path, preferred: str, scale: float, lon0: float, lat0: float):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)

    field = choose_vfield(gdf, preferred)
    lon = np.asarray(gdf.geometry.x, dtype=np.float64)
    lat = np.asarray(gdf.geometry.y, dtype=np.float64)
    velocity = np.asarray(gdf[field], dtype=np.float64) * scale

    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y, velocity


def unique_match(psx, psy, tx, ty, match_m):
    tree = cKDTree(np.column_stack([tx, ty]))
    dist, tidx = tree.query(
        np.column_stack([psx, psy]),
        k=1,
        workers=-1,
    )

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

    error = p - t
    corr = (
        float(np.corrcoef(p, t)[0, 1])
        if p.size > 2 and np.std(p) > 0 and np.std(t) > 0
        else np.nan
    )

    return {
        "n": int(error.size),
        "rmse_mm_yr": float(np.sqrt(np.mean(error * error))),
        "mae_mm_yr": float(np.mean(np.abs(error))),
        "bias_mm_yr": float(np.mean(error)),
        "correlation": corr,
        "pred_std_mm_yr": float(np.std(p)),
        "truth_std_mm_yr": float(np.std(t)),
    }


# -----------------------------------------------------------------------------
# Robust ramp
# -----------------------------------------------------------------------------

def select_balanced_anchors(
    coords_m: np.ndarray,
    quality: np.ndarray,
    cell_m: float,
    per_cell: int,
) -> np.ndarray:
    x = np.asarray(coords_m[:, 0], dtype=np.float64)
    y = np.asarray(coords_m[:, 1], dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)

    x0 = float(np.nanmin(x))
    y0 = float(np.nanmin(y))

    cx = np.floor((x - x0) / cell_m).astype(np.int64)
    cy = np.floor((y - y0) / cell_m).astype(np.int64)

    ny = int(np.max(cy)) + 1
    cell = cx * max(ny, 1) + cy

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(q)
    ids = np.flatnonzero(good)

    order = np.lexsort((-q[ids], cell[ids]))
    ids = ids[order]
    cells = cell[ids]

    keep = np.zeros(ids.size, dtype=bool)
    previous = None
    count = 0

    for i, cell_id in enumerate(cells):
        if previous is None or cell_id != previous:
            previous = cell_id
            count = 0
        if count < per_cell:
            keep[i] = True
            count += 1

    return np.sort(ids[keep])


def weighted_plane(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    good = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    Xv = X[good, :]
    yv = y[good]
    wv = w[good]

    if Xv.shape[0] < 10:
        return np.full(3, np.nan, dtype=np.float64)

    normal = Xv.T @ (wv[:, None] * Xv)
    rhs = Xv.T @ (wv * yv)
    return np.linalg.solve(normal, rhs)


def huber_plane(
    X: np.ndarray,
    y: np.ndarray,
    quality_weight: np.ndarray,
    iterations: int,
    delta: float,
):
    q = np.asarray(quality_weight, dtype=np.float64)
    beta = weighted_plane(X, y, q)

    if not np.all(np.isfinite(beta)):
        return beta, np.nan, 0

    scale = np.nan
    used_iterations = 0

    for _ in range(iterations):
        residual = y - X @ beta
        good = np.isfinite(residual) & np.isfinite(q) & (q > 0)

        if np.count_nonzero(good) < 10:
            break

        r = residual[good]
        median = float(np.median(r))
        mad = float(np.median(np.abs(r - median)))
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= 1.0e-8:
            break

        u = np.abs(residual) / (delta * scale)
        robust_weight = np.ones_like(u)

        large = u > 1.0
        robust_weight[large] = 1.0 / u[large]

        beta_new = weighted_plane(
            X,
            y,
            q * robust_weight,
        )

        if not np.all(np.isfinite(beta_new)):
            break

        used_iterations += 1

        if np.max(np.abs(beta_new - beta)) < 1.0e-8:
            beta = beta_new
            break

        beta = beta_new

    return beta, scale, used_iterations


def fit_ramp_configuration(
    phase_reader: MatrixReader,
    coords_m: np.ndarray,
    quality: np.ndarray,
    ref_center_xy: np.ndarray,
    anchors: np.ndarray,
    iterations: int,
    delta: float,
    label: str,
):
    phase_anchor = np.asarray(
        phase_reader.rows(anchors),
        dtype=np.float64,
    )

    X = np.column_stack(
        (
            (coords_m[anchors, 0] - ref_center_xy[0]) / 1000.0,
            (coords_m[anchors, 1] - ref_center_xy[1]) / 1000.0,
            np.ones(anchors.size, dtype=np.float64),
        )
    )

    q = np.clip(
        np.asarray(quality[anchors], dtype=np.float64),
        0.05,
        1.0,
    ) ** 2

    n_ifg = phase_anchor.shape[1]

    coeff = np.full((3, n_ifg), np.nan, dtype=np.float64)
    scale = np.full(n_ifg, np.nan, dtype=np.float64)
    used = np.zeros(n_ifg, dtype=np.int32)

    for j in range(n_ifg):
        beta, sc, n_used = huber_plane(
            X,
            phase_anchor[:, j],
            q,
            iterations,
            delta,
        )

        coeff[:, j] = beta
        scale[j] = sc
        used[j] = n_used

        if (j + 1) % 100 == 0 or j + 1 == n_ifg:
            print(
                f"[V6.4][{label}][RAMP] {j+1}/{n_ifg}",
                flush=True,
            )

    return coeff, scale, used


# -----------------------------------------------------------------------------
# Current IFGSTD inversion, streamed from raw IFGs
# -----------------------------------------------------------------------------

def branch_ifg_reference(
    phase_reader: MatrixReader,
    ref_ps: np.ndarray,
    coords_m: np.ndarray,
    ref_center_xy: np.ndarray,
    coeff: np.ndarray | None,
):
    ref_raw = np.asarray(
        phase_reader.rows(ref_ps),
        dtype=np.float64,
    )

    if coeff is not None:
        Xref = np.column_stack(
            (
                (coords_m[ref_ps, 0] - ref_center_xy[0]) / 1000.0,
                (coords_m[ref_ps, 1] - ref_center_xy[1]) / 1000.0,
                np.ones(ref_ps.size, dtype=np.float64),
            )
        )
        ref_raw -= Xref @ coeff

    reference = np.nanmedian(ref_raw, axis=0)
    reference[~np.isfinite(reference)] = 0.0
    return reference


def invert_streamed_ifgstd(
    phase_reader: MatrixReader,
    coords_m: np.ndarray,
    ref_center_xy: np.ndarray,
    coeff: np.ndarray | None,
    ifg_reference: np.ndarray,
    G: np.ndarray,
    use_ix: np.ndarray,
    weights: np.ndarray,
    projector: np.ndarray,
    unknown: np.ndarray,
    reference_image: int,
    output,
    chunk_ps: int,
    label: str,
):
    n_ps = phase_reader.shape[0]
    n_image = G.shape[1]

    A = G[use_ix, :][:, unknown]
    sqrt_w = np.sqrt(weights)

    solved_total = 0

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y_all = np.asarray(
            phase_reader.block(start, stop),
            dtype=np.float64,
        )

        if coeff is not None:
            X = np.column_stack(
                (
                    (coords_m[start:stop, 0] - ref_center_xy[0]) / 1000.0,
                    (coords_m[start:stop, 1] - ref_center_xy[1]) / 1000.0,
                    np.ones(stop-start, dtype=np.float64),
                )
            )
            y_all -= X @ coeff

        y_all -= ifg_reference[None, :]
        y = y_all[:, use_ix]

        current = stop - start
        out = np.full((current, n_image), np.nan, dtype=np.float64)

        complete = np.all(np.isfinite(y), axis=1)

        if np.any(complete):
            out[np.flatnonzero(complete)[:, None], unknown[None, :]] = (
                y[complete, :] @ projector.T
            )
            out[complete, reference_image] = 0.0

        for local in np.flatnonzero(~complete):
            valid = np.isfinite(y[local, :])

            if np.count_nonzero(valid) < n_image - 1:
                continue

            Av = A[valid, :]

            if np.linalg.matrix_rank(Av) != n_image - 1:
                continue

            solution, *_ = np.linalg.lstsq(
                Av * sqrt_w[valid, None],
                y[local, valid] * sqrt_w[valid],
                rcond=None,
            )

            out[local, unknown] = solution
            out[local, reference_image] = 0.0

        output[start:stop, :] = out.astype(np.float32)
        output.flush()

        solved_total += int(
            np.count_nonzero(
                np.all(np.isfinite(out), axis=1)
            )
        )

        if (
            start == 0
            or stop == n_ps
            or stop % 50000 < chunk_ps
        ):
            print(
                f"[V6.4][{label}][INVERT] "
                f"{stop}/{n_ps} solved={solved_total}",
                flush=True,
            )

    return solved_total


# -----------------------------------------------------------------------------
# Direct Stage8 SCN, fixed median reference
# -----------------------------------------------------------------------------

def build_scn_operator(ps2, day, reference_image, parms):
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
        -(time_diff * time_diff)
        / (2.0 * time_win * time_win)
    )

    temporal_weights[:, reference_image] = 0.0

    row_sum = np.sum(
        temporal_weights,
        axis=1,
        keepdims=True,
    )

    zero = row_sum[:, 0] <= 0

    if np.any(zero):
        temporal_weights[zero, :] = 0.0
        temporal_weights[zero, np.flatnonzero(zero)] = 1.0
        row_sum = np.sum(
            temporal_weights,
            axis=1,
            keepdims=True,
        )

    temporal_weights /= row_sum

    xy = np.asarray(ps2["xy"], dtype=np.float64)
    coords = np.asarray(xy[:, 1:3], dtype=np.float64)

    k_use = min(k_neighbors, coords.shape[0])

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

    invalid = neighbours >= coords.shape[0]

    neighbour_weights = np.exp(
        -(distances * distances)
        / (2.0 * wavelength * wavelength)
    )

    neighbour_weights[~np.isfinite(neighbour_weights)] = 0.0
    neighbour_weights[invalid] = 0.0

    neighbours_safe = neighbours.copy()
    neighbours_safe[invalid] = 0

    return {
        "time_win_days": time_win,
        "wavelength_m": wavelength,
        "radius_m": radius,
        "k_neighbors": k_use,
        "temporal_weights": temporal_weights,
        "neighbours": neighbours_safe,
        "neighbour_weights": neighbour_weights,
    }


def estimate_scn_direct(
    deramped_sm,
    ref_ps,
    operator,
    work: Path,
    label: str,
    chunk_ps: int,
):
    n_ps, n_image = deramped_sm.shape

    reference = np.nanmedian(
        np.asarray(deramped_sm[ref_ps, :], dtype=np.float64),
        axis=0,
    )
    reference[~np.isfinite(reference)] = 0.0

    ph_hpt = np.memmap(
        work / f"{label}_ph_hpt.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    T = operator["temporal_weights"]

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = (
            np.asarray(deramped_sm[start:stop, :], dtype=np.float64)
            - reference[None, :]
        )

        low_time = y @ T.T
        ph_hpt[start:stop, :] = (
            y - low_time
        ).astype(np.float32)

    ph_hpt.flush()

    ph_scn = np.memmap(
        work / f"{label}_ph_scn.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )

    neighbours = operator["neighbours"]
    neighbour_weights = operator["neighbour_weights"]

    spatial_chunk = 256

    for start in range(0, n_ps, spatial_chunk):
        stop = min(start + spatial_chunk, n_ps)

        idx = neighbours[start:stop, :]
        w = neighbour_weights[start:stop, :]

        values = np.asarray(
            ph_hpt[idx, :],
            dtype=np.float64,
        )

        finite = np.isfinite(values)

        weighted = (
            np.where(finite, values, 0.0)
            * w[:, :, None]
        )

        denom = np.sum(
            w[:, :, None] * finite,
            axis=1,
        )

        smooth = np.divide(
            np.sum(weighted, axis=1),
            denom,
            out=np.zeros(
                (stop-start, n_image),
                dtype=np.float64,
            ),
            where=denom > 0,
        )

        ph_scn[start:stop, :] = smooth.astype(np.float32)

    ph_scn.flush()

    # Fixed V6.3/V6.4 definition: median reference.
    ref_scn = np.nanmedian(
        np.asarray(ph_scn[ref_ps, :], dtype=np.float64),
        axis=0,
    )
    ref_scn[~np.isfinite(ref_scn)] = 0.0

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        ph_scn[start:stop, :] = (
            np.asarray(ph_scn[start:stop, :], dtype=np.float64)
            - ref_scn[None, :]
        ).astype(np.float32)

    ph_scn.flush()

    del ph_hpt
    gc.collect()

    return ph_scn


# -----------------------------------------------------------------------------
# Velocity fields
# -----------------------------------------------------------------------------

def velocity_fields(
    deramped_sm,
    ph_scn,
    ref_ps,
    coeffs,
    phase_to_mm,
    chunk_ps,
    subtract_scn: bool,
):
    n_ps = deramped_sm.shape[0]

    if subtract_scn:
        ref_values = (
            np.asarray(deramped_sm[ref_ps, :], dtype=np.float64)
            - np.asarray(ph_scn[ref_ps, :], dtype=np.float64)
        )
    else:
        ref_values = np.asarray(
            deramped_sm[ref_ps, :],
            dtype=np.float64,
        )

    reference = np.nanmedian(
        ref_values,
        axis=0,
    )
    reference[~np.isfinite(reference)] = 0.0

    result = {
        name: np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        )
        for name in coeffs
    }

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = np.asarray(
            deramped_sm[start:stop, :],
            dtype=np.float64,
        )

        if subtract_scn:
            y -= np.asarray(
                ph_scn[start:stop, :],
                dtype=np.float64,
            )

        y -= reference[None, :]

        for name, coeff in coeffs.items():
            result[name][start:stop] = (
                (y @ coeff) * phase_to_mm
            ).astype(np.float32)

    return result


def component_velocity_fields(
    ph_scn,
    ref_ps,
    coeffs,
    phase_to_mm,
    chunk_ps,
):
    n_ps = ph_scn.shape[0]

    reference = np.nanmedian(
        np.asarray(ph_scn[ref_ps, :], dtype=np.float64),
        axis=0,
    )
    reference[~np.isfinite(reference)] = 0.0

    result = {
        name: np.full(
            n_ps,
            np.nan,
            dtype=np.float32,
        )
        for name in coeffs
    }

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)

        y = (
            np.asarray(ph_scn[start:stop, :], dtype=np.float64)
            - reference[None, :]
        )

        for name, coeff in coeffs.items():
            result[name][start:stop] = (
                (y @ coeff) * phase_to_mm
            ).astype(np.float32)

    return result


# -----------------------------------------------------------------------------
# Spatial/local signal diagnostics
# -----------------------------------------------------------------------------

def build_pair_sample(coords, sample_ps, k):
    n_ps = coords.shape[0]
    rng = np.random.default_rng(20260812)

    sample = np.sort(
        rng.choice(
            n_ps,
            size=min(sample_ps, n_ps),
            replace=False,
        )
    )

    tree = cKDTree(coords)

    distances, neighbours = tree.query(
        coords[sample, :],
        k=min(k + 1, n_ps),
        workers=-1,
    )

    if distances.ndim == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]

    a = np.repeat(
        sample,
        distances.shape[1] - 1,
    )
    b = neighbours[:, 1:].reshape(-1)
    d = distances[:, 1:].reshape(-1)

    good = (
        np.isfinite(d)
        & (b >= 0)
        & (b < n_ps)
        & (a != b)
        & (d <= 1000.0)
    )

    a = a[good]
    b = b[good]
    d = d[good]

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    key = (
        lo.astype(np.int64) * np.int64(n_ps)
        + hi.astype(np.int64)
    )

    _, keep = np.unique(
        key,
        return_index=True,
    )

    return lo[keep], hi[keep], d[keep]


def pair_retention(
    config_name,
    deramp_fields,
    final_fields,
    pair_a,
    pair_b,
    pair_d,
):
    rows = []

    bins = (
        (0.0, 250.0, "0_250m"),
        (250.0, 500.0, "250_500m"),
        (500.0, 1000.0, "500_1000m"),
    )

    for period in ("2021", "2022", "2023"):
        before = np.asarray(
            deramp_fields[period],
            dtype=np.float64,
        )
        after = np.asarray(
            final_fields[period],
            dtype=np.float64,
        )

        for low, high, label in bins:
            use = (
                (pair_d >= low)
                & (pair_d < high)
            )

            d_before = np.abs(
                before[pair_a[use]]
                - before[pair_b[use]]
            )
            d_after = np.abs(
                after[pair_a[use]]
                - after[pair_b[use]]
            )

            good = (
                np.isfinite(d_before)
                & np.isfinite(d_after)
            )

            if not np.any(good):
                continue

            med_before = float(
                np.median(d_before[good])
            )
            med_after = float(
                np.median(d_after[good])
            )

            rows.append(
                {
                    "config": config_name,
                    "period": period,
                    "distance_bin": label,
                    "n_edges": int(np.count_nonzero(good)),
                    "median_pairdiff_deramp_mm_yr": med_before,
                    "median_pairdiff_final_mm_yr": med_after,
                    "pair_difference_retention": (
                        med_after / med_before
                        if med_before > 0
                        else np.nan
                    ),
                }
            )

    return rows


def field_difference_stats(
    config_name,
    reference_name,
    fields,
    reference_fields,
):
    rows = []

    for period in ("FULL", "2021", "2022", "2023"):
        a = np.asarray(fields[period], dtype=np.float64)
        b = np.asarray(reference_fields[period], dtype=np.float64)

        good = np.isfinite(a) & np.isfinite(b)
        d = a[good] - b[good]

        rows.append(
            {
                "config": config_name,
                "reference_config": reference_name,
                "period": period,
                "n": int(d.size),
                "median_diff_mm_yr": float(np.median(d)),
                "median_abs_diff_mm_yr": float(np.median(np.abs(d))),
                "p95_abs_diff_mm_yr": float(np.percentile(np.abs(d), 95)),
                "rms_diff_mm_yr": float(np.sqrt(np.mean(d*d))),
                "correlation": (
                    float(np.corrcoef(a[good], b[good])[0, 1])
                    if d.size > 2
                    and np.std(a[good]) > 0
                    and np.std(b[good]) > 0
                    else np.nan
                ),
            }
        )

    return rows


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        default=(
            "/mnt/vol-gdc28n1r/insar/"
            "cangzhou_P69/pystamps_sbas_ps_optimized"
        ),
    )

    p.add_argument("--truth-dir", default="")
    p.add_argument("--truth-field", default="v")
    p.add_argument("--truth-scale", type=float, default=1.0)
    p.add_argument("--truth-match-m", type=float, default=200.0)

    p.add_argument("--chunk-ps", type=int, default=2048)

    p.add_argument("--huber-delta", type=float, default=1.345)
    p.add_argument("--huber-iterations", type=int, default=5)

    p.add_argument("--pair-sample-ps", type=int, default=50000)
    p.add_argument("--pair-k", type=int, default=8)

    p.add_argument("--out", default="")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--self-test", action="store_true")

    return p.parse_args()


def self_test():
    rng = np.random.default_rng(1)

    x = rng.uniform(-50.0, 50.0, 4000)
    y = rng.uniform(-30.0, 30.0, 4000)

    X = np.column_stack(
        (x, y, np.ones_like(x))
    )

    beta_true = np.asarray(
        [0.02, -0.03, 1.2],
        dtype=np.float64,
    )

    z = (
        X @ beta_true
        + rng.normal(0.0, 0.05, len(x))
    )

    outlier = rng.choice(
        len(x),
        200,
        replace=False,
    )

    z[outlier] += rng.normal(
        0.0,
        5.0,
        len(outlier),
    )

    beta, _, _ = huber_plane(
        X,
        z,
        np.ones(len(x)),
        8,
        1.345,
    )

    if np.max(
        np.abs(beta[:2] - beta_true[:2])
    ) > 0.005:
        raise RuntimeError(
            "Huber plane self-test failed"
        )

    print("SELF-TEST: PASS")


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return

    load_project_modules()

    started = time.time()

    root = Path(args.dataset).resolve()

    truth_dir = (
        Path(args.truth_dir).resolve()
        if args.truth_dir
        else root / "cangzhou"
    )

    out = (
        Path(args.out).resolve()
        if args.out
        else root / "_audit" / (
            "robust_deramp_stability_v6_4_"
            + dt.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    work = out / "_work"
    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    ps2 = read_mat(root / "ps2.mat")
    parms = read_mat(root / "parms.mat")

    n_ps = int(
        round(
            float(
                np.asarray(
                    ps2["n_ps"]
                ).reshape(-1)[0]
            )
        )
    )

    lonlat = np.asarray(
        ps2["lonlat"],
        dtype=np.float64,
    )

    if lonlat.shape[0] != n_ps:
        lonlat = lonlat.T

    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    xy = np.asarray(
        ps2["xy"],
        dtype=np.float64,
    )

    coords_m = np.asarray(
        xy[:, 1:3],
        dtype=np.float64,
    )

    psx, psy, lon0, lat0 = local_xy(
        lon,
        lat,
    )

    ref_ps = np.asarray(
        ported._select_reference_ps(
            ps2,
            parms,
        ),
        dtype=np.int64,
    ).reshape(-1)

    ref_center_xy = np.nanmedian(
        coords_m[ref_ps, :],
        axis=0,
    )

    phase_input = _stage7_phase_input(root)

    phase_reader = MatrixReader(
        phase_input,
        "ph_uw",
        n_ps,
    )

    n_ifg = phase_reader.shape[1]

    ifg_std = np.asarray(
        read_mat_variables(
            root / "ifgstd2.mat",
            ("ifg_std",),
        )["ifg_std"],
        dtype=np.float64,
    ).reshape(-1)

    coh_ps = np.asarray(
        read_mat_variables(
            root / "pm2.mat",
            ("coh_ps",),
        )["coh_ps"],
        dtype=np.float64,
    ).reshape(-1)

    day, ifgday_ix, _, network_source = (
        load_sbas_network(
            root,
            n_ifg,
        )
    )

    day = np.asarray(
        day,
        dtype=np.float64,
    ).reshape(-1)

    ifgday_ix = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    n_image = day.size

    dates = [
        matlab_datenum_to_datetime(v)
        for v in day
    ]

    G = s7._network_matrix(
        n_image,
        ifgday_ix,
    )

    master_ix = int(
        round(
            float(
                np.asarray(
                    ps2.get(
                        "master_ix",
                        1,
                    )
                ).reshape(-1)[0]
            )
        )
    )

    if master_ix < 1 or master_ix > n_image:
        master_ix = 1

    reference_image = master_ix - 1

    drop_network = s7._drop_set(
        parms,
        "drop_ifg_index",
    )

    finite_std = (
        np.isfinite(ifg_std)
        & (ifg_std > 0)
    )

    network_mask = np.asarray(
        [
            i not in drop_network
            for i in range(1, n_ifg + 1)
        ],
        dtype=bool,
    ) & finite_std

    use_ix = np.flatnonzero(
        network_mask
    )

    variance_ifg = (
        ifg_std * math.pi / 180.0
    ) ** 2

    weights_ifg = np.zeros(
        n_ifg,
        dtype=np.float64,
    )

    weights_ifg[network_mask] = (
        1.0
        / variance_ifg[network_mask]
    )

    median_weight = float(
        np.median(
            weights_ifg[network_mask]
        )
    )

    if median_weight > 0:
        weights_ifg /= median_weight

    projector, unknown, rank = (
        s7._network_projector(
            G,
            use_ix,
            weights_ifg[use_ix],
            reference_image,
        )
    )

    wavelength = float(
        np.asarray(
            parms["lambda"]
        ).reshape(-1)[0]
    )

    phase_to_mm = (
        -wavelength
        / (4.0 * np.pi)
        * 1000.0
    )

    coeffs = {
        "FULL": slope_coeff(dates, None),
        "2021": slope_coeff(dates, 2021),
        "2022": slope_coeff(dates, 2022),
        "2023": slope_coeff(dates, 2023),
    }

    scn_operator = build_scn_operator(
        ps2,
        day,
        reference_image,
        parms,
    )

    pair_a, pair_b, pair_d = (
        build_pair_sample(
            coords_m,
            args.pair_sample_ps,
            args.pair_k,
        )
    )

    print("=" * 80)
    print("V6.4 robust deramp stability")
    print("=" * 80)
    print("n_ps:", n_ps)
    print("n_ifg:", n_ifg)
    print("n_image:", n_image)
    print("reference PS:", ref_ps.size)
    print("reference config:", REFERENCE_CONFIG)
    print("SCLA: OFF")
    print("network: current IFGSTD WLS")
    print("SCN: DIRECT + median reference")
    print("No production files will be overwritten")

    # -------------------------------------------------------------------------
    # Truth mappings are prepared for evaluation only.
    # -------------------------------------------------------------------------
    R = 6371008.8

    ref_center_ll = np.asarray(
        parms["ref_centre_lonlat"],
        dtype=np.float64,
    ).reshape(-1)

    ref_radius = float(
        np.asarray(
            parms["ref_radius_m"]
        ).reshape(-1)[0]
    )

    refx = (
        np.deg2rad(
            ref_center_ll[0] - lon0
        )
        * R
        * np.cos(np.deg2rad(lat0))
    )

    refy = (
        np.deg2rad(
            ref_center_ll[1] - lat0
        )
        * R
    )

    truth = {}

    for year in (2021, 2022, 2023):
        tx, ty, tv = read_truth(
            truth_dir / f"result{year}.shp",
            args.truth_field,
            args.truth_scale,
            lon0,
            lat0,
        )

        tree = cKDTree(
            np.column_stack([tx, ty])
        )

        rid = np.asarray(
            tree.query_ball_point(
                [refx, refy],
                r=ref_radius,
            ),
            dtype=np.int64,
        )

        truth_ref = float(
            np.nanmedian(tv[rid])
        )

        tv = tv - truth_ref

        pidx, tidx = unique_match(
            psx,
            psy,
            tx,
            ty,
            args.truth_match_m,
        )

        truth[year] = (
            tv,
            pidx,
            tidx,
            truth_ref,
        )

    # -------------------------------------------------------------------------
    # Fit all ramp configurations first.
    # -------------------------------------------------------------------------
    ramp_models = {}
    ramp_rows = []
    ramp_summary_rows = []

    for name, cell_m, per_cell in CONFIGS:
        print("\n" + "=" * 80)
        print("FIT:", name)
        print("=" * 80)

        anchors = select_balanced_anchors(
            coords_m,
            coh_ps,
            cell_m,
            per_cell,
        )

        coeff, scale, used = fit_ramp_configuration(
            phase_reader,
            coords_m,
            coh_ps,
            ref_center_xy,
            anchors,
            args.huber_iterations,
            args.huber_delta,
            name,
        )

        ramp_models[name] = {
            "cell_m": cell_m,
            "per_cell": per_cell,
            "anchors": anchors,
            "coeff": coeff,
            "scale": scale,
            "used": used,
        }

        ramp_summary_rows.append(
            {
                "config": name,
                "cell_m": cell_m,
                "anchors_per_cell": per_cell,
                "anchor_count": int(anchors.size),
                "median_residual_scale_rad": float(
                    np.nanmedian(scale)
                ),
                "p95_residual_scale_rad": float(
                    np.nanpercentile(scale, 95)
                ),
                "median_iterations_used": float(
                    np.nanmedian(used)
                ),
            }
        )

        for j in range(n_ifg):
            ramp_rows.append(
                {
                    "config": name,
                    "ifg_index_1based": j + 1,
                    "ax_rad_per_km": float(coeff[0, j]),
                    "ay_rad_per_km": float(coeff[1, j]),
                    "intercept_rad": float(coeff[2, j]),
                    "residual_scale_rad": float(scale[j]),
                    "iterations_used": int(used[j]),
                }
            )

    write_csv(
        out / "04_ramp_coefficients.csv",
        ramp_rows,
    )

    # Ramp coefficient stability vs pre-declared R2.
    ref_coeff = ramp_models[
        REFERENCE_CONFIG
    ]["coeff"]

    ramp_stability_rows = []

    for name, _, _ in CONFIGS:
        coeff = ramp_models[name]["coeff"]

        dax = coeff[0, :] - ref_coeff[0, :]
        day_ = coeff[1, :] - ref_coeff[1, :]

        grad_delta = np.sqrt(
            dax*dax + day_*day_
        )

        ramp_stability_rows.append(
            {
                "config": name,
                "reference_config": REFERENCE_CONFIG,
                "median_abs_delta_ax_rad_per_km": float(
                    np.nanmedian(np.abs(dax))
                ),
                "p95_abs_delta_ax_rad_per_km": float(
                    np.nanpercentile(np.abs(dax), 95)
                ),
                "median_abs_delta_ay_rad_per_km": float(
                    np.nanmedian(np.abs(day_))
                ),
                "p95_abs_delta_ay_rad_per_km": float(
                    np.nanpercentile(np.abs(day_), 95)
                ),
                "median_gradient_delta_rad_per_km": float(
                    np.nanmedian(grad_delta)
                ),
                "p95_gradient_delta_rad_per_km": float(
                    np.nanpercentile(grad_delta, 95)
                ),
            }
        )

    write_csv(
        out / "05_ramp_stability_vs_R2.csv",
        ramp_stability_rows,
    )

    # -------------------------------------------------------------------------
    # Process each configuration independently.
    # -------------------------------------------------------------------------
    final_fields = {}
    deramp_fields_all = {}
    scn_component_rows = []
    pair_rows = []

    for name, _, _ in CONFIGS:
        print("\n" + "=" * 80)
        print("PROCESS:", name)
        print("=" * 80)

        coeff = ramp_models[name]["coeff"]

        ifg_reference = branch_ifg_reference(
            phase_reader,
            ref_ps,
            coords_m,
            ref_center_xy,
            coeff,
        )

        deramped_sm = np.memmap(
            work / f"{name}_deramped_sm.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_ps, n_image),
        )

        invert_streamed_ifgstd(
            phase_reader,
            coords_m,
            ref_center_xy,
            coeff,
            ifg_reference,
            G,
            use_ix,
            weights_ifg[use_ix],
            projector,
            unknown,
            reference_image,
            deramped_sm,
            args.chunk_ps,
            name,
        )

        ph_scn = estimate_scn_direct(
            deramped_sm,
            ref_ps,
            scn_operator,
            work,
            name,
            args.chunk_ps,
        )

        deramp_fields = velocity_fields(
            deramped_sm,
            ph_scn,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
            subtract_scn=False,
        )

        final_branch_fields = velocity_fields(
            deramped_sm,
            ph_scn,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
            subtract_scn=True,
        )

        scn_fields = component_velocity_fields(
            ph_scn,
            ref_ps,
            coeffs,
            phase_to_mm,
            args.chunk_ps,
        )

        deramp_fields_all[name] = deramp_fields
        final_fields[name] = final_branch_fields

        for period, values in scn_fields.items():
            v = np.asarray(
                values,
                dtype=np.float64,
            )
            v = v[np.isfinite(v)]

            scn_component_rows.append(
                {
                    "config": name,
                    "period": period,
                    "median_mm_yr": float(
                        np.median(v)
                    ),
                    "std_mm_yr": float(
                        np.std(v)
                    ),
                    "median_abs_mm_yr": float(
                        np.median(np.abs(v))
                    ),
                    "p95_abs_mm_yr": float(
                        np.percentile(
                            np.abs(v),
                            95,
                        )
                    ),
                }
            )

        pair_rows += pair_retention(
            name,
            deramp_fields,
            final_branch_fields,
            pair_a,
            pair_b,
            pair_d,
        )

        del deramped_sm, ph_scn
        gc.collect()

        # Remove per-config full phase work immediately.
        for pattern in (
            f"{name}_deramped_sm.f32",
            f"{name}_ph_scn.f32",
            f"{name}_ph_hpt.f32",
        ):
            path = work / pattern
            if path.exists():
                path.unlink()

    write_csv(
        out / "03_scn_component_velocity_stats.csv",
        scn_component_rows,
    )

    write_csv(
        out / "06_local_pair_retention.csv",
        pair_rows,
    )

    # -------------------------------------------------------------------------
    # Truth evaluation.
    # -------------------------------------------------------------------------
    truth_rows = []
    pooled_rows = []

    for name, _, _ in CONFIGS:
        fields = final_fields[name]

        pp = []
        tt = []

        for year in (2021, 2022, 2023):
            tv, pidx, tidx, _ = truth[year]

            pred = fields[str(year)][pidx]
            obs = tv[tidx]

            result = metrics(
                pred,
                obs,
            )

            truth_rows.append(
                {
                    "config": name,
                    "year": year,
                    **result,
                }
            )

            good = (
                np.isfinite(pred)
                & np.isfinite(obs)
            )

            pp.append(pred[good])
            tt.append(obs[good])

        pooled = metrics(
            np.concatenate(pp),
            np.concatenate(tt),
        )

        pooled_rows.append(
            {
                "config": name,
                **pooled,
            }
        )

    write_csv(
        out / "01_truth_by_year.csv",
        truth_rows,
    )

    write_csv(
        out / "02_truth_pooled.csv",
        pooled_rows,
    )

    # -------------------------------------------------------------------------
    # Final-field parameter stability vs pre-declared R2.
    # -------------------------------------------------------------------------
    reference_fields = final_fields[
        REFERENCE_CONFIG
    ]

    velocity_stability_rows = []

    for name, _, _ in CONFIGS:
        velocity_stability_rows += (
            field_difference_stats(
                name,
                REFERENCE_CONFIG,
                final_fields[name],
                reference_fields,
            )
        )

    write_csv(
        out / "07_velocity_stability_vs_R2.csv",
        velocity_stability_rows,
    )

    # -------------------------------------------------------------------------
    # Compact point product.
    # -------------------------------------------------------------------------
    payload = {
        "ps_index_1based": np.arange(
            1,
            n_ps + 1,
            dtype=np.int32,
        ),
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
    }

    for name, _, _ in CONFIGS:
        for period, values in final_fields[name].items():
            payload[
                f"{name}__FINAL__{period}_mm_yr"
            ] = values

        for period, values in deramp_fields_all[name].items():
            payload[
                f"{name}__DERAMP__{period}_mm_yr"
            ] = values

    np.savez_compressed(
        out / "branch_velocity_points.npz",
        **payload,
    )

    # -------------------------------------------------------------------------
    # Stability summary — not a truth-based parameter selector.
    # -------------------------------------------------------------------------
    pooled_rmse = np.asarray(
        [
            row["rmse_mm_yr"]
            for row in pooled_rows
        ],
        dtype=np.float64,
    )

    rmse_range = float(
        np.max(pooled_rmse)
        - np.min(pooled_rmse)
    )

    nonref_velocity = [
        row
        for row in velocity_stability_rows
        if row["config"] != REFERENCE_CONFIG
    ]

    max_full_p95_diff = max(
        row["p95_abs_diff_mm_yr"]
        for row in nonref_velocity
        if row["period"] == "FULL"
    )

    max_annual_p95_diff = max(
        row["p95_abs_diff_mm_yr"]
        for row in nonref_velocity
        if row["period"] in ("2021", "2022", "2023")
    )

    local_0_250 = [
        row["pair_difference_retention"]
        for row in pair_rows
        if row["distance_bin"] == "0_250m"
    ]

    min_local_retention = float(
        np.nanmin(local_0_250)
    )

    # Heuristic used only to decide whether a production patch is justified.
    stable = (
        rmse_range <= 0.5
        and max_full_p95_diff <= 2.0
        and max_annual_p95_diff <= 3.0
        and min_local_retention >= 0.85
    )

    lowest_truth = min(
        pooled_rows,
        key=lambda row: row["rmse_mm_yr"],
    )

    summary = {
        "input_phase": str(phase_input),
        "network_source": str(network_source),
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": int(n_image),
        "reference_ps": int(ref_ps.size),
        "reference_image_ix_1based": int(master_ix),
        "network": "current IFGSTD WLS",
        "SCLA": "OFF",
        "SCN": {
            "mode": "DIRECT",
            "reference": "median",
            "time_win_days": float(
                scn_operator["time_win_days"]
            ),
            "wavelength_m": float(
                scn_operator["wavelength_m"]
            ),
            "radius_m": float(
                scn_operator["radius_m"]
            ),
            "k_neighbors": int(
                scn_operator["k_neighbors"]
            ),
        },
        "ramp": {
            "method": (
                "spatially balanced quality-guided "
                "Huber IRLS 2-D plane"
            ),
            "quality": "pm2.coh_ps",
            "huber_delta": float(
                args.huber_delta
            ),
            "huber_iterations": int(
                args.huber_iterations
            ),
            "truth_used_for_fit": False,
            "tested_configs": [
                {
                    "name": name,
                    "cell_m": cell_m,
                    "anchors_per_cell": per_cell,
                    "anchor_count": int(
                        ramp_models[name]["anchors"].size
                    ),
                }
                for name, cell_m, per_cell
                in CONFIGS
            ],
            "predeclared_reference_config": REFERENCE_CONFIG,
        },
        "stability": {
            "pooled_truth_rmse_range_mm_yr": rmse_range,
            "max_FULL_p95_abs_velocity_difference_vs_R2_mm_yr": float(
                max_full_p95_diff
            ),
            "max_annual_p95_abs_velocity_difference_vs_R2_mm_yr": float(
                max_annual_p95_diff
            ),
            "minimum_0_250m_pair_difference_retention": min_local_retention,
            "heuristic_thresholds": {
                "rmse_range_mm_yr_max": 0.5,
                "FULL_p95_diff_mm_yr_max": 2.0,
                "annual_p95_diff_mm_yr_max": 3.0,
                "0_250m_retention_min": 0.85,
            },
            "classification": (
                "ROBUST_DERAMP_STABLE_FOR_PRODUCTION_PATCH"
                if stable
                else "ROBUST_DERAMP_PARAMETER_SENSITIVE"
            ),
        },
        "truth_diagnostic_only": {
            "lowest_rmse_config": lowest_truth["config"],
            "lowest_pooled_rmse_mm_yr": lowest_truth["rmse_mm_yr"],
            "note": (
                "Lowest truth RMSE is reported for diagnosis only. "
                "V6.4 does not tune or select ramp parameters by truth."
            ),
        },
        "runtime_seconds": time.time() - started,
    }

    (out / "08_SUMMARY.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        out / "09_ramp_fit_summary.csv",
        ramp_summary_rows,
    )

    print("\nPOOLED TRUTH")
    for row in pooled_rows:
        print(
            f"{row['config']:18s} "
            f"RMSE={row['rmse_mm_yr']:.4f} "
            f"corr={row['correlation']:.4f} "
            f"bias={row['bias_mm_yr']:.4f}"
        )

    print("\nSTABILITY")
    print("RMSE range:", rmse_range)
    print("max FULL P95 |Δv| vs R2:", max_full_p95_diff)
    print("max annual P95 |Δv| vs R2:", max_annual_p95_diff)
    print("min 0-250 m retention:", min_local_retention)
    print("classification:", summary["stability"]["classification"])

    if not args.keep_work:
        shutil.rmtree(
            work,
            ignore_errors=True,
        )
        print("Temporary audit work removed.")

    print("Output:", out)


if __name__ == "__main__":
    main()
