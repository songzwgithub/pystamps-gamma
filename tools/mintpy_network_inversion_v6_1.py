#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg, sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def matlab_datenum_to_datetime(value):
    ordinal = int(value)
    frac = float(value) - ordinal
    return dt.datetime.fromordinal(ordinal) + dt.timedelta(days=frac) - dt.timedelta(days=366)


class MatrixReader:
    def __init__(self, path, var, nrow):
        self.path = Path(path)
        self.var = var
        self.nrow = int(nrow)
        self.hdf5 = h5py.is_hdf5(self.path)

        if self.hdf5:
            with h5py.File(self.path, "r") as h:
                shape = tuple(h[var].shape)
            self.transpose = shape[0] != nrow and shape[1] == nrow
        else:
            a = np.asarray(
                loadmat(
                    self.path,
                    variable_names=[var],
                    squeeze_me=False,
                )[var]
            )
            self.a = a
            self.transpose = a.shape[0] != nrow and a.shape[1] == nrow

    def rows(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if self.hdf5:
            order = np.argsort(ids)
            sids = ids[order]
            with h5py.File(self.path, "r") as h:
                ds = h[self.var]
                a = (
                    np.asarray(ds[sids, :])
                    if not self.transpose
                    else np.asarray(ds[:, sids]).T
                )
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            return a[inv]

        a = self.a.T if self.transpose else self.a
        return np.asarray(a[ids, :])

    def block(self, r0, r1):
        if self.hdf5:
            with h5py.File(self.path, "r") as h:
                ds = h[self.var]
                return (
                    np.asarray(ds[r0:r1, :])
                    if not self.transpose
                    else np.asarray(ds[:, r0:r1]).T
                )

        a = self.a.T if self.transpose else self.a
        return np.asarray(a[r0:r1, :])


def build_velocity_design(ifgday_ix, day):
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
            lo = min(a, b)
            hi = max(a, b)

            if hi - lo <= d:
                continue

            for j in range(lo + d, hi):
                rr.append(e)
                cc.append(j - d)

        mats.append(
            sparse.csr_matrix(
                (
                    np.ones(len(rr), dtype=np.float64),
                    (rr, cc),
                ),
                shape=(len(ifgday_ix), n - d),
            )
        )

    return mats


def normal_band_and_rhs(W, Y, B, contributors, bandwidth, dt_days):
    W = np.asarray(W, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    batch = W.shape[0]
    n = B.shape[1]

    Ab = np.zeros(
        (batch, bandwidth + 1, n),
        dtype=np.float64,
    )

    WT = W.T

    for d, C in enumerate(contributors):
        sums = (C.T @ WT).T
        if d == 0:
            coeff = dt_days * dt_days
        else:
            coeff = dt_days[d:] * dt_days[:-d]
        Ab[:, d, d:] = sums * coeff[None, :]

    rhs = (B.T @ (W * Y).T).T

    diag_scale = np.nanmedian(Ab[:, 0, :], axis=1)
    ridge = 1.0e-12 * np.maximum(diag_scale, 1.0e-12)
    Ab[:, 0, :] += ridge[:, None]

    return Ab, rhs


def solve_banded_batch(Ab, rhs):
    Ab = np.asarray(Ab, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)

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
                val -= (
                    L[:, i-k, i]
                    * L[:, j-k, j]
                )

            L[:, d, i] = val / L[:, 0, j]

    z = np.empty_like(rhs)

    for j in range(n):
        val = rhs[:, j].copy()

        for d in range(1, min(bandwidth, j) + 1):
            val -= L[:, d, j] * z[:, j-d]

        z[:, j] = val / L[:, 0, j]

    x = np.empty_like(rhs)

    for j in range(n-1, -1, -1):
        val = z[:, j].copy()

        for i in range(j + 1, min(n, j + bandwidth + 1)):
            val -= L[:, i-j, i] * x[:, i]

        x[:, j] = val / L[:, 0, j]

    x[bad, :] = np.nan
    return x, bad


def reconstruct_phase(V, dt_days):
    X = np.zeros(
        (V.shape[0], len(dt_days) + 1),
        dtype=np.float64,
    )
    X[:, 1:] = np.cumsum(
        V * dt_days[None, :],
        axis=1,
    )
    return X


def phase_variance_ds_lut(L, coherence_step=0.005):
    L = int(max(1, L))
    n = int(round(1.0 / coherence_step))

    coherence = (
        np.linspace(
            coherence_step,
            1.0,
            n,
            dtype=np.float64,
        )
        - coherence_step / 2.0
    )

    phi = np.linspace(
        -np.pi,
        np.pi,
        n,
        dtype=np.float64,
    )[:, None]

    dphi = 2.0 * np.pi / n

    gamma = coherence[None, :]
    beta = np.abs(gamma) * np.cos(phi)

    A = (1.0 - gamma * gamma) ** L / (2.0 * np.pi)

    Bc = (
        math.gamma(2 * L - 1)
        / (
            math.gamma(L) ** 2
            * 2.0 ** (2 * (L - 1))
        )
    )

    den = 1.0 - beta * beta

    C = (
        (2 * L - 1)
        * beta
        / den ** (L + 0.5)
        * (np.pi / 2.0 + np.arcsin(beta))
        + 1.0 / den**L
    )

    D = np.zeros_like(C)

    if L > 1:
        for r in range(L - 1):
            D += (
                math.gamma(L - 0.5)
                / math.gamma(L - 0.5 - r)
                * math.gamma(L - 1 - r)
                / math.gamma(L - 1)
                * (1.0 + (2 * r + 1) * beta * beta)
                / den ** (r + 2)
            )

        D /= 2.0 * (L - 1)

    pdf = A * (Bc * C + D)

    var = np.sum(
        phi * phi * pdf * dphi,
        axis=0,
    )

    positive = var > 0
    var[~positive] = np.min(var[positive])

    return coherence, var


def prepare_var_lut(L):
    grid, variance = phase_variance_ds_lut(L)
    return grid, 1.0 / variance


def coherence_weights(coh, mode, var_lut):
    c = np.asarray(coh, dtype=np.float64)
    c = np.where(np.isfinite(c), c, 0.05)
    c = np.clip(c, 0.05, 1.0)

    if mode == "NO":
        return np.ones_like(c)

    if mode == "COH":
        return c

    if mode == "VAR":
        grid, weight_lut = var_lut
        idx = np.searchsorted(
            grid,
            c,
            side="left",
        )
        idx = np.clip(
            idx,
            0,
            len(grid) - 1,
        )
        return weight_lut[idx]

    raise ValueError(mode)


def solve_block(
    Y,
    C,
    mode,
    B,
    contributors,
    bandwidth,
    dt_days,
    var_lut,
):
    finite = np.isfinite(Y)

    W = coherence_weights(
        C,
        mode,
        var_lut,
    )

    W = np.where(
        finite,
        W,
        0.0,
    )

    med = np.nanmedian(
        np.where(W > 0, W, np.nan),
        axis=1,
    )
    med = np.where(
        np.isfinite(med) & (med > 0),
        med,
        1.0,
    )
    W = W / med[:, None]

    Y0 = np.where(
        finite,
        Y,
        0.0,
    )

    Ab, rhs = normal_band_and_rhs(
        W,
        Y0,
        B,
        contributors,
        bandwidth,
        dt_days,
    )

    V, bad = solve_banded_batch(
        Ab,
        rhs,
    )

    X = reconstruct_phase(
        V,
        dt_days,
    )

    pred_ifg = (B @ V.T).T
    residual = np.where(
        finite,
        Y - pred_ifg,
        np.nan,
    )

    nvalid = np.sum(
        finite,
        axis=1,
    )

    tc = np.abs(
        np.nansum(
            np.exp(1j * residual),
            axis=1,
        )
        / np.maximum(nvalid, 1)
    )

    rms = np.sqrt(
        np.nansum(
            residual * residual,
            axis=1,
        )
        / np.maximum(nvalid, 1)
    )

    tc[bad] = np.nan
    rms[bad] = np.nan

    return X, tc, rms, bad


def exact_spot_audit(
    Y,
    C,
    modes,
    B,
    dt_days,
    var_lut,
    fast_results,
    rcond,
):
    rows = []

    for mode in modes:
        for i in range(len(Y)):
            y = Y[i]
            valid = np.isfinite(y)

            w = coherence_weights(
                C[i:i+1, :],
                mode,
                var_lut,
            )[0]

            valid &= (
                np.isfinite(w)
                & (w > 0)
            )

            Bv = B[valid, :]
            yv = y[valid]
            wv = w[valid]

            sw = np.sqrt(wv)

            v, _, rank, _ = linalg.lstsq(
                Bv * sw[:, None],
                yv * sw,
                cond=rcond,
            )

            if rank < B.shape[1]:
                continue

            ph = np.zeros(
                len(dt_days) + 1,
                dtype=np.float64,
            )
            ph[1:] = np.cumsum(
                v * dt_days
            )

            diff = ph - fast_results[mode][i]

            rows.append(
                {
                    "mode": mode,
                    "sample": i + 1,
                    "median_abs_diff_rad": float(
                        np.median(np.abs(diff))
                    ),
                    "max_abs_diff_rad": float(
                        np.max(np.abs(diff))
                    ),
                }
            )

    return rows


def slope_coeff(dates, year=None):
    if year is None:
        ids = np.arange(len(dates))
    else:
        ids = np.asarray(
            [
                i
                for i, d in enumerate(dates)
                if d.year == year
            ],
            dtype=np.int64,
        )

    if len(ids) < 2:
        return np.zeros(
            len(dates),
            dtype=np.float64,
        )

    t0 = dates[ids[0]]

    t = np.asarray(
        [
            (
                dates[i] - t0
            ).total_seconds()
            / 86400.0
            / 365.2425
            for i in ids
        ],
        dtype=np.float64,
    )

    tc = t - np.mean(t)

    c = np.zeros(
        len(dates),
        dtype=np.float64,
    )

    c[ids] = tc / np.sum(tc * tc)

    return c


def choose_vfield(gdf, preferred):
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if c.lower() == preferred.lower():
            return c
    raise RuntimeError(
        f"truth field {preferred} not found"
    )


def read_truth(path, preferred, scale, lon0, lat0):
    import geopandas as gpd

    gdf = gpd.read_file(path)

    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)

    field = choose_vfield(
        gdf,
        preferred,
    )

    lon = np.asarray(
        gdf.geometry.x,
        dtype=np.float64,
    )
    lat = np.asarray(
        gdf.geometry.y,
        dtype=np.float64,
    )
    v = np.asarray(
        gdf[field],
        dtype=np.float64,
    ) * scale

    R = 6371008.8

    x = (
        np.deg2rad(lon - lon0)
        * R
        * np.cos(np.deg2rad(lat0))
    )
    y = np.deg2rad(lat - lat0) * R

    return x, y, v


def match_truth(psx, psy, tx, ty, match_m):
    tree = cKDTree(
        np.column_stack([tx, ty])
    )

    dist, tidx = tree.query(
        np.column_stack([psx, psy]),
        k=1,
        workers=-1,
    )

    valid = (
        np.isfinite(dist)
        & (dist <= match_m)
    )

    p = np.flatnonzero(valid)
    t = tidx[valid]
    d = dist[valid]

    order = np.argsort(d)
    t2 = t[order]

    _, first = np.unique(
        t2,
        return_index=True,
    )

    keep = np.sort(
        order[first]
    )

    return p[keep], t[keep]


def metrics(pred, truth):
    valid = (
        np.isfinite(pred)
        & np.isfinite(truth)
    )

    p = np.asarray(
        pred[valid],
        dtype=np.float64,
    )
    t = np.asarray(
        truth[valid],
        dtype=np.float64,
    )

    e = p - t

    corr = (
        float(
            np.corrcoef(p, t)[0, 1]
        )
        if np.std(p) > 0
        and np.std(t) > 0
        else np.nan
    )

    return {
        "n": int(len(e)),
        "rmse_mm_yr": float(
            np.sqrt(np.mean(e**2))
        ),
        "mae_mm_yr": float(
            np.mean(np.abs(e))
        ),
        "correlation": corr,
        "pred_std_mm_yr": float(
            np.std(p)
        ),
        "truth_std_mm_yr": float(
            np.std(t)
        ),
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        default=(
            "/mnt/vol-gdc28n1r/insar/"
            "cangzhou_P69/"
            "pystamps_sbas_ps_optimized"
        ),
    )

    p.add_argument(
        "--truth-dir",
        default="",
    )

    p.add_argument(
        "--truth-field",
        default="v",
    )

    p.add_argument(
        "--truth-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--truth-match-m",
        type=float,
        default=200.0,
    )

    p.add_argument(
        "--block-ps",
        type=int,
        default=2048,
    )

    p.add_argument(
        "--svd-spot-count",
        type=int,
        default=24,
    )

    p.add_argument(
        "--rcond",
        type=float,
        default=1.0e-5,
    )

    p.add_argument(
        "--out",
        default="",
    )

    p.add_argument(
        "--self-test",
        action="store_true",
    )

    return p.parse_args()


def self_test():
    day = (
        np.arange(
            8,
            dtype=np.float64,
        )
        * 12.0
        + 737000.0
    )

    ifg = np.asarray(
        [
            (i+1, j+1)
            for i in range(8)
            for j in range(
                i+1,
                min(8, i+4),
            )
        ],
        dtype=np.int64,
    )

    B, dt_days, max_span = (
        build_velocity_design(
            ifg,
            day,
        )
    )

    bandwidth = max_span - 1

    contributors = (
        build_normal_contributors(
            ifg,
            8,
            bandwidth,
        )
    )

    rng = np.random.default_rng(0)

    V0 = rng.normal(
        size=(10, 7)
    )

    Y = (B @ V0.T).T
    W = np.ones_like(Y)

    Ab, rhs = normal_band_and_rhs(
        W,
        Y,
        B,
        contributors,
        bandwidth,
        dt_days,
    )

    V, bad = solve_banded_batch(
        Ab,
        rhs,
    )

    if np.any(bad):
        raise RuntimeError(
            "synthetic banded solve failed"
        )

    if np.max(
        np.abs(V - V0)
    ) > 1.0e-7:
        raise RuntimeError(
            "synthetic banded solve mismatch"
        )

    print("SELF-TEST: PASS")


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return

    started = time.time()

    # Import the installed project only for a real dataset run.
    from pystamps.io.mat import read_mat
    from pystamps.pipeline import ported
    from pystamps.pipeline.stage6_sbas import load_sbas_network
    from pystamps.pipeline.stage7_sbas import _stage7_phase_input

    root = Path(
        args.dataset
    ).resolve()

    truth_dir = (
        Path(
            args.truth_dir
        ).resolve()
        if args.truth_dir
        else root / "cangzhou"
    )

    ps2 = read_mat(
        root / "ps2.mat"
    )
    parms = read_mat(
        root / "parms.mat"
    )

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

    ref_ps = np.asarray(
        ported._select_reference_ps(
            ps2,
            parms,
        ),
        dtype=np.int64,
    )

    print(
        "reference PS:",
        ref_ps.size,
    )

    print(
        "reference:",
        np.asarray(
            parms["ref_centre_lonlat"]
        ).reshape(-1).tolist(),
        float(
            np.asarray(
                parms["ref_radius_m"]
            ).reshape(-1)[0]
        ),
        "m",
    )

    phase_input = (
        _stage7_phase_input(root)
    )

    phase_reader = MatrixReader(
        phase_input,
        "ph_uw",
        n_ps,
    )

    ref_ifg = np.nanmedian(
        np.asarray(
            phase_reader.rows(
                ref_ps
            ),
            dtype=np.float64,
        ),
        axis=0,
    )

    n_ifg = len(ref_ifg)

    (
        day,
        ifgday_ix,
        _,
        network_source,
    ) = load_sbas_network(
        root,
        n_ifg,
    )

    day = np.asarray(
        day,
        dtype=np.float64,
    ).reshape(-1)

    ifgday_ix = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    dates = [
        matlab_datenum_to_datetime(v)
        for v in day
    ]

    n_image = len(day)

    (
        B_sparse,
        dt_days,
        max_span,
    ) = build_velocity_design(
        ifgday_ix,
        day,
    )

    B = B_sparse.toarray()

    rank = int(
        np.linalg.matrix_rank(B)
    )

    if rank != n_image - 1:
        raise RuntimeError(
            f"network rank={rank}, "
            f"expected={n_image-1}"
        )

    bandwidth = max_span - 1

    contributors = (
        build_normal_contributors(
            ifgday_ix,
            n_image,
            bandwidth,
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

    rlooks = int(
        round(
            float(
                np.asarray(
                    parms.get(
                        "range_looks",
                        4,
                    )
                ).reshape(-1)[0]
            )
        )
    )

    alooks = int(
        round(
            float(
                np.asarray(
                    parms.get(
                        "azimuth_looks",
                        1,
                    )
                ).reshape(-1)[0]
            )
        )
    )

    effective_looks = max(
        1,
        int(
            round(
                rlooks
                * alooks
                / 1.94
            )
        ),
    )

    var_lut = prepare_var_lut(
        effective_looks
    )

    coh_path = (
        root
        / "_pixel_wls_cache_v3"
        / "coherence_ps_ifg_float16.dat"
    )

    if not coh_path.exists():
        raise FileNotFoundError(
            coh_path
        )

    coh = np.memmap(
        coh_path,
        mode="r",
        dtype=np.float16,
        shape=(
            n_ps,
            n_ifg,
        ),
    )

    out = (
        Path(
            args.out
        ).resolve()
        if args.out
        else root
        / "_audit"
        / (
            "mintpy_network_inversion_v6_1_"
            + dt.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    modes = [
        "NO",
        "COH",
        "VAR",
    ]

    years_all = sorted(
        set(
            d.year
            for d in dates
        )
    )

    truth_years = [
        2021,
        2022,
        2023,
    ]

    coeff_full = slope_coeff(
        dates,
        year=None,
    )

    coeff_year = {
        year: slope_coeff(
            dates,
            year=year,
        )
        for year in years_all
    }

    files = {}

    for mode in modes:
        h = h5py.File(
            out
            / f"timeseries_{mode}.h5",
            "w",
        )

        h.create_dataset(
            "phase_rad",
            shape=(
                n_ps,
                n_image,
            ),
            dtype="f4",
            chunks=(
                min(
                    args.block_ps,
                    n_ps,
                ),
                n_image,
            ),
        )

        h.create_dataset(
            "temporal_coherence",
            shape=(n_ps,),
            dtype="f4",
        )

        h.create_dataset(
            "residual_rms_rad",
            shape=(n_ps,),
            dtype="f4",
        )

        h.create_dataset(
            "velocity_mm_yr",
            shape=(n_ps,),
            dtype="f4",
        )

        for year in years_all:
            h.create_dataset(
                f"annual_velocity_{year}_mm_yr",
                shape=(n_ps,),
                dtype="f4",
            )

        h.attrs["method"] = (
            f"MINTPY_{mode}"
        )

        h.attrs["input_phase"] = (
            str(phase_input)
        )

        h.attrs["reference_ps_count"] = (
            int(ref_ps.size)
        )

        h.attrs["reference_lon"] = float(
            np.asarray(
                parms["ref_centre_lonlat"]
            ).reshape(-1)[0]
        )

        h.attrs["reference_lat"] = float(
            np.asarray(
                parms["ref_centre_lonlat"]
            ).reshape(-1)[1]
        )

        h.attrs["reference_radius_m"] = float(
            np.asarray(
                parms["ref_radius_m"]
            ).reshape(-1)[0]
        )

        h.attrs[
            "effective_independent_looks"
        ] = int(
            effective_looks
        )

        files[mode] = h

    predictions = {
        mode: {
            year: np.full(
                n_ps,
                np.nan,
                dtype=np.float32,
            )
            for year in truth_years
        }
        for mode in modes
    }

    # Exact scipy.linalg.lstsq comparison on a small subset.
    rng = np.random.default_rng(
        20260812
    )

    spot_ids = np.sort(
        rng.choice(
            n_ps,
            size=min(
                args.svd_spot_count,
                n_ps,
            ),
            replace=False,
        )
    )

    spot_y = np.asarray(
        phase_reader.rows(
            spot_ids
        ),
        dtype=np.float64,
    )

    spot_y -= ref_ifg[
        None,
        :
    ]

    spot_c = np.asarray(
        coh[
            spot_ids,
            :
        ],
        dtype=np.float64,
    )

    fast_spot = {}

    for mode in modes:
        (
            spot_x,
            _,
            _,
            _,
        ) = solve_block(
            spot_y,
            spot_c,
            mode,
            B_sparse,
            contributors,
            bandwidth,
            dt_days,
            var_lut,
        )

        fast_spot[
            mode
        ] = spot_x

    svd_rows = exact_spot_audit(
        spot_y,
        spot_c,
        modes,
        B,
        dt_days,
        var_lut,
        fast_spot,
        args.rcond,
    )

    write_csv(
        out
        / "exact_svd_spot_audit.csv",
        svd_rows,
    )

    print(
        "max exact-SVD difference:",
        max(
            row["max_abs_diff_rad"]
            for row in svd_rows
        ),
        "rad",
    )

    for r0 in range(
        0,
        n_ps,
        args.block_ps,
    ):
        r1 = min(
            n_ps,
            r0
            + args.block_ps,
        )

        Y = np.asarray(
            phase_reader.block(
                r0,
                r1,
            ),
            dtype=np.float64,
        )

        # Spatial reference is part of the observation model.
        Y -= ref_ifg[
            None,
            :
        ]

        C = np.asarray(
            coh[
                r0:r1,
                :
            ],
            dtype=np.float64,
        )

        for mode in modes:
            (
                X,
                tc,
                rms,
                bad,
            ) = solve_block(
                Y,
                C,
                mode,
                B_sparse,
                contributors,
                bandwidth,
                dt_days,
                var_lut,
            )

            vel = (
                X
                @ coeff_full
                * phase_to_mm
            )

            h = files[mode]

            h[
                "phase_rad"
            ][r0:r1, :] = (
                X.astype(
                    np.float32
                )
            )

            h[
                "temporal_coherence"
            ][r0:r1] = (
                tc.astype(
                    np.float32
                )
            )

            h[
                "residual_rms_rad"
            ][r0:r1] = (
                rms.astype(
                    np.float32
                )
            )

            h[
                "velocity_mm_yr"
            ][r0:r1] = (
                vel.astype(
                    np.float32
                )
            )

            for year in years_all:
                annual = (
                    X
                    @ coeff_year[year]
                    * phase_to_mm
                )

                h[
                    f"annual_velocity_{year}_mm_yr"
                ][r0:r1] = (
                    annual.astype(
                        np.float32
                    )
                )

                if year in truth_years:
                    predictions[
                        mode
                    ][
                        year
                    ][
                        r0:r1
                    ] = (
                        annual.astype(
                            np.float32
                        )
                    )

        if (
            r0 == 0
            or r1 == n_ps
            or r1 % 50000
            < args.block_ps
        ):
            print(
                f"PS {r1}/{n_ps}",
                flush=True,
            )

    for h in files.values():
        h.close()

    lon0 = float(
        np.median(lon)
    )
    lat0 = float(
        np.median(lat)
    )

    R = 6371008.8

    psx = (
        np.deg2rad(
            lon - lon0
        )
        * R
        * np.cos(
            np.deg2rad(
                lat0
            )
        )
    )

    psy = (
        np.deg2rad(
            lat - lat0
        )
        * R
    )

    ref_lon = float(
        np.asarray(
            parms[
                "ref_centre_lonlat"
            ]
        ).reshape(-1)[0]
    )

    ref_lat = float(
        np.asarray(
            parms[
                "ref_centre_lonlat"
            ]
        ).reshape(-1)[1]
    )

    ref_radius = float(
        np.asarray(
            parms[
                "ref_radius_m"
            ]
        ).reshape(-1)[0]
    )

    refx = (
        np.deg2rad(
            ref_lon - lon0
        )
        * R
        * np.cos(
            np.deg2rad(
                lat0
            )
        )
    )

    refy = (
        np.deg2rad(
            ref_lat - lat0
        )
        * R
    )

    truth = {}

    for year in truth_years:
        (
            tx,
            ty,
            tv,
        ) = read_truth(
            truth_dir
            / f"result{year}.shp",
            args.truth_field,
            args.truth_scale,
            lon0,
            lat0,
        )

        tr = cKDTree(
            np.column_stack(
                [
                    tx,
                    ty,
                ]
            )
        )

        rid = np.asarray(
            tr.query_ball_point(
                [
                    refx,
                    refy,
                ],
                r=ref_radius,
            ),
            dtype=np.int64,
        )

        truth_ref = float(
            np.nanmedian(
                tv[rid]
            )
        )

        tv = (
            tv
            - truth_ref
        )

        (
            pidx,
            tidx,
        ) = match_truth(
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
        )

    rows = []
    pooled_rows = []

    for mode in modes:
        pp = []
        tt = []

        for year in truth_years:
            (
                tv,
                pidx,
                tidx,
            ) = truth[
                year
            ]

            pred = predictions[
                mode
            ][
                year
            ][
                pidx
            ]

            obs = tv[
                tidx
            ]

            m = metrics(
                pred,
                obs,
            )

            rows.append(
                {
                    "method": f"MINTPY_{mode}",
                    "year": year,
                    **m,
                }
            )

            valid = (
                np.isfinite(pred)
                & np.isfinite(obs)
            )

            pp.append(
                pred[valid]
            )

            tt.append(
                obs[valid]
            )

        pm = metrics(
            np.concatenate(pp),
            np.concatenate(tt),
        )

        pooled_rows.append(
            {
                "method": f"MINTPY_{mode}",
                **pm,
            }
        )

    write_csv(
        out
        / "truth_by_year.csv",
        rows,
    )

    write_csv(
        out
        / "truth_pooled.csv",
        pooled_rows,
    )

    best = min(
        pooled_rows,
        key=lambda r: (
            r["rmse_mm_yr"]
        ),
    )

    summary = {
        "input_phase": str(
            phase_input
        ),
        "network_source": str(
            network_source
        ),
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "network_rank": rank,
        "network_bandwidth": bandwidth,
        "reference_ps": int(
            ref_ps.size
        ),
        "reference_lon": ref_lon,
        "reference_lat": ref_lat,
        "reference_radius_m": ref_radius,
        "range_looks": rlooks,
        "azimuth_looks": alooks,
        "effective_independent_looks": effective_looks,
        "methods": modes,
        "best_truth_method": (
            best["method"]
        ),
        "best_truth_pooled_rmse_mm_yr": (
            best["rmse_mm_yr"]
        ),
        "note": (
            "Raw network inversion only: "
            "no spatial velocity filtering, "
            "no IFG deletion, no Final-C, "
            "no old ph_ramp/ph_scn."
        ),
        "runtime_seconds": (
            time.time()
            - started
        ),
    }

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPOOLED TRUTH"
    )

    for row in pooled_rows:
        print(
            row["method"],
            "RMSE=",
            row["rmse_mm_yr"],
            "corr=",
            row["correlation"],
        )

    print(
        "\nBest:",
        best["method"],
    )

    print(
        "Output:",
        out,
    )


if __name__ == "__main__":
    main()
