#!/usr/bin/env python3
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

import numpy as np
from scipy.spatial import cKDTree


def load_helpers():
    path = Path(__file__).with_name("_v62_helpers.py")
    spec = importlib.util.spec_from_file_location("v62_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.load_project_modules()
    return module


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


def select_balanced_anchors(xy_m, quality, cell_m, per_cell):
    x = np.asarray(xy_m[:, 0], dtype=np.float64)
    y = np.asarray(xy_m[:, 1], dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(q)
    ids = np.flatnonzero(good)

    x0 = float(np.min(x[good]))
    y0 = float(np.min(y[good]))
    cx = np.floor((x - x0) / cell_m).astype(np.int64)
    cy = np.floor((y - y0) / cell_m).astype(np.int64)
    ny = int(np.max(cy[good])) + 1
    cell = cx * max(ny, 1) + cy

    order = np.lexsort((-q[ids], cell[ids]))
    ids = ids[order]
    cells = cell[ids]

    keep = np.zeros(ids.size, dtype=bool)
    last = None
    count = 0
    for i, c in enumerate(cells):
        if last is None or c != last:
            last = c
            count = 0
        if count < per_cell:
            keep[i] = True
            count += 1
    return np.sort(ids[keep])


def weighted_plane(X, y, w):
    good = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    Xv = X[good]
    yv = y[good]
    wv = w[good]
    normal = Xv.T @ (wv[:, None] * Xv)
    rhs = Xv.T @ (wv * yv)
    return np.linalg.solve(normal, rhs)


def huber_plane(X, y, quality_weight, iterations=5, delta=1.345):
    q = np.asarray(quality_weight, dtype=np.float64)
    beta = weighted_plane(X, y, q)
    scale = np.nan
    used = 0
    for _ in range(iterations):
        residual = y - X @ beta
        good = np.isfinite(residual) & np.isfinite(q) & (q > 0)
        r = residual[good]
        if r.size < 10:
            break
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-8:
            break
        u = np.abs(residual) / (delta * scale)
        robust = np.ones_like(u)
        high = u > 1.0
        robust[high] = 1.0 / u[high]
        beta_new = weighted_plane(X, y, q * robust)
        used += 1
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta, scale, used


def fit_robust_ramps(ph_ifg, xy_m, quality, ref_xy, anchors, iterations, delta):
    X = np.column_stack((
        (xy_m[anchors, 0] - ref_xy[0]) / 1000.0,
        (xy_m[anchors, 1] - ref_xy[1]) / 1000.0,
        np.ones(len(anchors), dtype=np.float64),
    ))
    q = np.clip(np.asarray(quality[anchors], dtype=np.float64), 0.05, 1.0) ** 2
    Y = np.asarray(ph_ifg[anchors, :], dtype=np.float64)
    n_ifg = Y.shape[1]
    coeff = np.full((3, n_ifg), np.nan, dtype=np.float64)
    scale = np.full(n_ifg, np.nan, dtype=np.float64)
    used = np.zeros(n_ifg, dtype=np.int32)
    for j in range(n_ifg):
        coeff[:, j], scale[j], used[j] = huber_plane(
            X, Y[:, j], q, iterations=iterations, delta=delta
        )
        if (j + 1) % 50 == 0 or j + 1 == n_ifg:
            print(f"[V6.3][ROBUST_RAMP] {j+1}/{n_ifg}", flush=True)
    return coeff, scale, used, X


def apply_ramp(ph_ifg, xy_m, ref_xy, coeff, out_path, chunk_ps):
    n_ps, n_ifg = ph_ifg.shape
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_ifg))
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        X = np.column_stack((
            (xy_m[start:stop, 0] - ref_xy[0]) / 1000.0,
            (xy_m[start:stop, 1] - ref_xy[1]) / 1000.0,
            np.ones(stop - start, dtype=np.float64),
        ))
        out[start:stop, :] = (
            np.asarray(ph_ifg[start:stop, :], dtype=np.float64) - X @ coeff
        ).astype(np.float32)
    out.flush()
    return out


def center_ifg(values, ref_ps, out_path, chunk_ps):
    n_ps, n_ifg = values.shape
    ref = np.nanmedian(np.asarray(values[ref_ps, :], dtype=np.float64), axis=0)
    ref[~np.isfinite(ref)] = 0.0
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_ifg))
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        out[start:stop, :] = (
            np.asarray(values[start:stop, :], dtype=np.float64) - ref[None, :]
        ).astype(np.float32)
    out.flush()
    return out


def invert_ifgstd(v62, values, G, use_ix, weights, projector, unknown,
                  reference_image, out_path, chunk_ps, label):
    n_ps = values.shape[0]
    n_image = G.shape[1]
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_image))
    v62.s7._invert_network(
        values,
        G=G,
        use_ix=use_ix,
        weights=weights,
        projector=projector,
        unknown=unknown,
        reference_image=reference_image,
        output=out,
        chunk_ps=chunk_ps,
        label=label,
    )
    return out


def direct_scn_input(ph_proc, ref_ps, out_path, chunk_ps):
    n_ps, n_image = ph_proc.shape
    ref = np.nanmedian(np.asarray(ph_proc[ref_ps, :], dtype=np.float64), axis=0)
    ref[~np.isfinite(ref)] = 0.0
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_image))
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        out[start:stop, :] = (
            np.asarray(ph_proc[start:stop, :], dtype=np.float64) - ref[None, :]
        ).astype(np.float32)
    out.flush()
    return out


def recenter_scn_to_median(scn_mean, ref_ps, out_path, chunk_ps):
    n_ps, n_image = scn_mean.shape
    ref_median = np.nanmedian(np.asarray(scn_mean[ref_ps, :], dtype=np.float64), axis=0)
    ref_median[~np.isfinite(ref_median)] = 0.0
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_image))
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        out[start:stop, :] = (
            np.asarray(scn_mean[start:stop, :], dtype=np.float64) - ref_median[None, :]
        ).astype(np.float32)
    out.flush()
    return out, ref_median


def subtract_component(base, component, out_path, chunk_ps):
    n_ps, n_image = base.shape
    out = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_ps, n_image))
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        out[start:stop, :] = (
            np.asarray(base[start:stop, :], dtype=np.float64)
            - np.asarray(component[start:stop, :], dtype=np.float64)
        ).astype(np.float32)
    out.flush()
    return out


def estimate_scn_median(v62, scn_input, ps2, day, reference_image,
                        ref_ps, parms, work, label, chunk_ps):
    final_mean, scn_mean, settings = v62.stage8_scn_only(
        scn_input, ps2, day, reference_image, ref_ps, parms,
        work, label, chunk_ps
    )
    scn_median, shift = recenter_scn_to_median(
        scn_mean, ref_ps, work / f"{label}_ph_scn_median.f32", chunk_ps
    )
    settings = dict(settings)
    settings["v63_reference_statistic"] = "median"
    settings["mean_to_median_shift_rms_rad"] = float(np.sqrt(np.mean(shift * shift)))
    settings["mean_to_median_shift_max_abs_rad"] = float(np.max(np.abs(shift)))
    del final_mean, scn_mean
    gc.collect()
    return scn_median, settings


def component_stats(name, fields):
    rows = []
    for period, value in fields.items():
        v = np.asarray(value, dtype=np.float64)
        v = v[np.isfinite(v)]
        p02, p50, p98 = np.percentile(v, [2, 50, 98])
        rows.append({
            "component": name,
            "period": period,
            "p02_mm_yr": float(p02),
            "median_mm_yr": float(p50),
            "p98_mm_yr": float(p98),
            "std_mm_yr": float(np.std(v)),
            "median_abs_mm_yr": float(np.median(np.abs(v))),
            "p95_abs_mm_yr": float(np.percentile(np.abs(v), 95)),
        })
    return rows


def build_pair_sample(coords, sample_ps=50000, k=8):
    n_ps = coords.shape[0]
    rng = np.random.default_rng(20260812)
    sample = np.sort(rng.choice(n_ps, size=min(sample_ps, n_ps), replace=False))
    tree = cKDTree(coords)
    dist, nei = tree.query(coords[sample], k=min(k + 1, n_ps), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        nei = nei[:, None]
    a = np.repeat(sample, dist.shape[1] - 1)
    b = nei[:, 1:].reshape(-1)
    d = dist[:, 1:].reshape(-1)
    good = np.isfinite(d) & (b >= 0) & (b < n_ps) & (a != b) & (d <= 1000.0)
    a, b, d = a[good], b[good], d[good]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo.astype(np.int64) * np.int64(n_ps) + hi.astype(np.int64)
    _, keep = np.unique(key, return_index=True)
    return lo[keep], hi[keep], d[keep]


def pair_retention(branch_fields, baselines, a, b, d):
    rows = []
    bins = [(0, 250, "0_250m"), (250, 500, "250_500m"), (500, 1000, "500_1000m")]
    for branch, baseline in baselines.items():
        for period in ("2021", "2022", "2023"):
            v = np.asarray(branch_fields[branch][period], dtype=np.float64)
            vb = np.asarray(branch_fields[baseline][period], dtype=np.float64)
            for lo, hi, label in bins:
                ids = (d >= lo) & (d < hi)
                dn = np.abs(v[a[ids]] - v[b[ids]])
                db = np.abs(vb[a[ids]] - vb[b[ids]])
                good = np.isfinite(dn) & np.isfinite(db)
                if not np.any(good):
                    continue
                mn = float(np.median(dn[good]))
                mb = float(np.median(db[good]))
                rows.append({
                    "branch": branch,
                    "baseline": baseline,
                    "period": period,
                    "distance_bin": label,
                    "n_edges": int(np.count_nonzero(good)),
                    "median_pairdiff_branch_mm_yr": mn,
                    "median_pairdiff_baseline_mm_yr": mb,
                    "pair_difference_retention": mn / mb if mb > 0 else np.nan,
                })
    return rows


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
    p.add_argument("--anchor-cell-m", type=float, default=2000.0)
    p.add_argument("--anchors-per-cell", type=int, default=8)
    p.add_argument("--huber-iterations", type=int, default=5)
    p.add_argument("--huber-delta", type=float, default=1.345)
    p.add_argument("--pair-sample-ps", type=int, default=50000)
    p.add_argument("--pair-k", type=int, default=8)
    p.add_argument("--out", default="")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def self_test():
    rng = np.random.default_rng(1)
    x = rng.uniform(-50, 50, 3000)
    y = rng.uniform(-30, 30, 3000)
    X = np.column_stack((x, y, np.ones_like(x)))
    beta0 = np.asarray([0.02, -0.03, 1.2])
    z = X @ beta0 + rng.normal(0, 0.05, len(x))
    ids = rng.choice(len(x), 150, replace=False)
    z[ids] += rng.normal(0, 5, len(ids))
    beta, _, _ = huber_plane(X, z, np.ones(len(x)), iterations=8, delta=1.345)
    if np.max(np.abs(beta[:2] - beta0[:2])) > 0.005:
        raise RuntimeError("Huber self-test failed")
    print("SELF-TEST: PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    v62 = load_helpers()
    started = time.time()

    root = Path(args.dataset).resolve()
    truth_dir = Path(args.truth_dir).resolve() if args.truth_dir else root / "cangzhou"
    out = (
        Path(args.out).resolve()
        if args.out
        else root / "_audit" / ("deramp_scn_audit_v6_3_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    ps2 = v62.read_mat(root / "ps2.mat")
    parms = v62.read_mat(root / "parms.mat")
    n_ps = int(round(float(np.asarray(ps2["n_ps"]).reshape(-1)[0])))

    lonlat = np.asarray(ps2["lonlat"], dtype=np.float64)
    if lonlat.shape[0] != n_ps:
        lonlat = lonlat.T
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    x_local, y_local, lon0, lat0 = v62.local_xy(lon, lat)

    xy = np.asarray(ps2["xy"], dtype=np.float64)
    coords = np.asarray(xy[:, 1:3], dtype=np.float64)

    ref_ps = np.asarray(v62.ported._select_reference_ps(ps2, parms), dtype=np.int64).reshape(-1)
    print("reference PS:", len(ref_ps))

    phase_input = v62._stage7_phase_input(root)
    ph_ifg_raw = v62.s7._as_matrix(
        v62.read_mat_variables(phase_input, ("ph_uw",))["ph_uw"],
        n_ps, "ph_uw", np.float32,
    )
    n_ifg = ph_ifg_raw.shape[1]

    bp_ifg = v62.s7._as_matrix(
        v62.read_mat_variables(root / "bp2.mat", ("bperp_mat",))["bperp_mat"],
        n_ps, "bp2.bperp_mat", np.float32,
    )
    ifg_std = np.asarray(
        v62.read_mat_variables(root / "ifgstd2.mat", ("ifg_std",))["ifg_std"],
        dtype=np.float64,
    ).reshape(-1)
    coh_ps = np.asarray(
        v62.read_mat_variables(root / "pm2.mat", ("coh_ps",))["coh_ps"],
        dtype=np.float64,
    ).reshape(-1)

    day, ifgday_ix, _, network_source = v62.load_sbas_network(root, n_ifg)
    day = np.asarray(day, dtype=np.float64).reshape(-1)
    ifgday_ix = np.asarray(ifgday_ix, dtype=np.int64)
    n_image = len(day)
    dates = [v62.matlab_datenum_to_datetime(d) for d in day]
    G = v62.s7._network_matrix(n_image, ifgday_ix)

    master_ix = int(round(float(np.asarray(ps2.get("master_ix", 1)).reshape(-1)[0])))
    if not (1 <= master_ix <= n_image):
        master_ix = 1
    reference_image = master_ix - 1

    drop = v62.s7._drop_set(parms, "drop_ifg_index")
    finite_std = np.isfinite(ifg_std) & (ifg_std > 0)
    network_mask = np.asarray([i not in drop for i in range(1, n_ifg + 1)], dtype=bool) & finite_std
    use_ix = np.flatnonzero(network_mask)
    variance_ifg = (ifg_std * math.pi / 180.0) ** 2
    weights = np.zeros(n_ifg, dtype=np.float64)
    weights[network_mask] = 1.0 / variance_ifg[network_mask]
    weights /= float(np.median(weights[network_mask]))

    projector, unknown, rank = v62.s7._network_projector(
        G, use_ix, weights[use_ix], reference_image
    )

    wavelength = float(np.asarray(parms["lambda"]).reshape(-1)[0])
    phase_to_mm = -wavelength / (4.0 * np.pi) * 1000.0
    coeffs = {
        "FULL": v62.slope_coeff(dates, None),
        "2021": v62.slope_coeff(dates, 2021),
        "2022": v62.slope_coeff(dates, 2022),
        "2023": v62.slope_coeff(dates, 2023),
    }

    # RAW reference/inversion baseline.
    raw_centered = center_ifg(ph_ifg_raw, ref_ps, work / "raw_centered.f32", args.chunk_ps)
    raw_sm = invert_ifgstd(
        v62, raw_centered, G, use_ix, weights[use_ix], projector, unknown,
        reference_image, work / "raw_sm.f32", args.chunk_ps, "V63_RAW"
    )

    # Current all-PS OLS deramp.
    current_deramped, current_ramp_ifg = v62.ported._deramp_unwrapped_phase(
        ps2, np.asarray(ph_ifg_raw, dtype=np.float64)
    )
    current_centered = center_ifg(
        current_deramped, ref_ps, work / "current_centered.f32", args.chunk_ps
    )
    current_sm = invert_ifgstd(
        v62, current_centered, G, use_ix, weights[use_ix], projector, unknown,
        reference_image, work / "current_sm.f32", args.chunk_ps, "V63_CURRENT_DERAMP"
    )

    # Balanced quality-guided Huber deramp.
    anchors = select_balanced_anchors(coords, coh_ps, args.anchor_cell_m, args.anchors_per_cell)
    ref_xy = np.nanmedian(coords[ref_ps], axis=0)
    print("robust ramp anchors:", len(anchors))
    robust_coeff, robust_scale, robust_iter, Xa = fit_robust_ramps(
        ph_ifg_raw, coords, coh_ps, ref_xy, anchors,
        args.huber_iterations, args.huber_delta
    )
    robust_deramped = apply_ramp(
        ph_ifg_raw, coords, ref_xy, robust_coeff,
        work / "robust_deramped.f32", args.chunk_ps
    )
    robust_centered = center_ifg(
        robust_deramped, ref_ps, work / "robust_centered.f32", args.chunk_ps
    )
    robust_sm = invert_ifgstd(
        v62, robust_centered, G, use_ix, weights[use_ix], projector, unknown,
        reference_image, work / "robust_sm.f32", args.chunk_ps, "V63_ROBUST_DERAMP"
    )

    # Common Bperp reconstruction.
    bp_sm = np.memmap(
        work / "bp_sm_common.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    v62.s7._invert_network(
        np.asarray(bp_ifg, dtype=np.float64),
        G=G, use_ix=use_ix, weights=weights[use_ix], projector=projector,
        unknown=unknown, reference_image=reference_image, output=bp_sm,
        chunk_ps=args.chunk_ps, label="V63_BPERP_COMMON",
    )

    branch_series = {
        "RAW": raw_sm,
        "CURRENT_DERAMP": current_sm,
        "ROBUST_DERAMP": robust_sm,
    }
    component_series = {}
    scn_settings = {}

    for prefix, ph_proc in (("CURRENT", current_sm), ("ROBUST", robust_sm)):
        print("\n===", prefix, "===")
        scla = v62.estimate_branch_scla(
            ph_proc, bp_sm, day, reference_image, ps2, root, work,
            prefix.lower(), args.chunk_ps
        )

        nuisance_input, _ = v62.corrected_series(
            ph_proc,
            scla["ph_scla_envelope"],
            scla["c_envelope"],
            ref_ps,
            work / f"{prefix.lower()}_nuisance_input.f32",
            args.chunk_ps,
        )
        scn_nuisance, settings_nuisance = estimate_scn_median(
            v62, nuisance_input, ps2, day, reference_image, ref_ps, parms,
            work, f"{prefix.lower()}_nuisance", args.chunk_ps
        )
        final_nuisance = subtract_component(
            ph_proc, scn_nuisance,
            work / f"{prefix.lower()}_final_nuisance.f32", args.chunk_ps
        )
        branch_series[f"{prefix}_SCN_NUISANCE_SCLA"] = final_nuisance
        component_series[f"{prefix}_SCN_NUISANCE_SCLA"] = scn_nuisance
        scn_settings[f"{prefix}_SCN_NUISANCE_SCLA"] = settings_nuisance

        direct_input = direct_scn_input(
            ph_proc, ref_ps, work / f"{prefix.lower()}_direct_input.f32", args.chunk_ps
        )
        scn_direct, settings_direct = estimate_scn_median(
            v62, direct_input, ps2, day, reference_image, ref_ps, parms,
            work, f"{prefix.lower()}_direct", args.chunk_ps
        )
        final_direct = subtract_component(
            ph_proc, scn_direct,
            work / f"{prefix.lower()}_final_direct.f32", args.chunk_ps
        )
        branch_series[f"{prefix}_SCN_DIRECT"] = final_direct
        component_series[f"{prefix}_SCN_DIRECT"] = scn_direct
        scn_settings[f"{prefix}_SCN_DIRECT"] = settings_direct

        del scla, nuisance_input, direct_input
        gc.collect()

    branch_fields = {
        name: v62.velocity_fields(series, ref_ps, coeffs, phase_to_mm, args.chunk_ps)
        for name, series in branch_series.items()
    }

    # Component amplitudes.
    component_rows = []
    current_ramp_sm = np.memmap(
        work / "current_ramp_sm.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    robust_ramp_sm = np.memmap(
        work / "robust_ramp_sm.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    for start in range(0, n_ps, args.chunk_ps):
        stop = min(start + args.chunk_ps, n_ps)
        r = np.asarray(raw_sm[start:stop], dtype=np.float64)
        current_ramp_sm[start:stop] = (r - np.asarray(current_sm[start:stop], dtype=np.float64)).astype(np.float32)
        robust_ramp_sm[start:stop] = (r - np.asarray(robust_sm[start:stop], dtype=np.float64)).astype(np.float32)
    current_ramp_sm.flush(); robust_ramp_sm.flush()

    for name, series in (("CURRENT_RAMP", current_ramp_sm), ("ROBUST_RAMP", robust_ramp_sm)):
        fields = v62.velocity_fields(series, ref_ps, coeffs, phase_to_mm, args.chunk_ps)
        component_rows += component_stats(name, fields)
    for name, series in component_series.items():
        fields = v62.velocity_fields(series, ref_ps, coeffs, phase_to_mm, args.chunk_ps)
        component_rows += component_stats(name, fields)

    # Truth reference and validation.
    R = 6371008.8
    ref_ll = np.asarray(parms["ref_centre_lonlat"], dtype=np.float64).reshape(-1)
    ref_radius = float(np.asarray(parms["ref_radius_m"]).reshape(-1)[0])
    refx = np.deg2rad(ref_ll[0] - lon0) * R * np.cos(np.deg2rad(lat0))
    refy = np.deg2rad(ref_ll[1] - lat0) * R

    truth = {}
    for year in (2021, 2022, 2023):
        tx, ty, tv = v62.read_truth(
            truth_dir / f"result{year}.shp", args.truth_field, args.truth_scale, lon0, lat0
        )
        tree = cKDTree(np.column_stack([tx, ty]))
        rid = np.asarray(tree.query_ball_point([refx, refy], r=ref_radius), dtype=np.int64)
        truth_ref = float(np.nanmedian(tv[rid]))
        tv = tv - truth_ref
        pidx, tidx = v62.unique_match(x_local, y_local, tx, ty, args.truth_match_m)
        truth[year] = (tv, pidx, tidx)

    truth_rows = []
    pooled_rows = []
    for branch, fields in branch_fields.items():
        pp, tt = [], []
        for year in (2021, 2022, 2023):
            tv, pidx, tidx = truth[year]
            pred = fields[str(year)][pidx]
            obs = tv[tidx]
            m = v62.metrics(pred, obs)
            truth_rows.append({"branch": branch, "year": year, **m})
            good = np.isfinite(pred) & np.isfinite(obs)
            pp.append(pred[good]); tt.append(obs[good])
        pm = v62.metrics(np.concatenate(pp), np.concatenate(tt))
        pooled_rows.append({"branch": branch, **pm})

    write_csv(out / "01_truth_by_year.csv", truth_rows)
    write_csv(out / "02_truth_pooled.csv", pooled_rows)
    write_csv(out / "03_component_velocity_stats.csv", component_rows)

    # Current vs robust ramp coefficients on the same balanced anchor geometry.
    current_coeff, *_ = np.linalg.lstsq(
        Xa, np.asarray(current_ramp_ifg[anchors, :], dtype=np.float64), rcond=None
    )
    ramp_rows = []
    for j in range(n_ifg):
        ramp_rows.append({
            "ifg_index_1based": j + 1,
            "current_ax_rad_per_km": float(current_coeff[0, j]),
            "current_ay_rad_per_km": float(current_coeff[1, j]),
            "robust_ax_rad_per_km": float(robust_coeff[0, j]),
            "robust_ay_rad_per_km": float(robust_coeff[1, j]),
            "delta_ax_rad_per_km": float(robust_coeff[0, j] - current_coeff[0, j]),
            "delta_ay_rad_per_km": float(robust_coeff[1, j] - current_coeff[1, j]),
            "robust_residual_scale_rad": float(robust_scale[j]),
            "robust_iterations": int(robust_iter[j]),
        })
    write_csv(out / "04_ramp_coefficients.csv", ramp_rows)

    # Local spatial contrast retention after SCN, diagnostic only.
    a, b, d = build_pair_sample(coords, args.pair_sample_ps, args.pair_k)
    baselines = {
        "CURRENT_SCN_NUISANCE_SCLA": "CURRENT_DERAMP",
        "CURRENT_SCN_DIRECT": "CURRENT_DERAMP",
        "ROBUST_SCN_NUISANCE_SCLA": "ROBUST_DERAMP",
        "ROBUST_SCN_DIRECT": "ROBUST_DERAMP",
    }
    write_csv(
        out / "05_local_pair_retention.csv",
        pair_retention(branch_fields, baselines, a, b, d),
    )

    payload = {
        "ps_index_1based": np.arange(1, n_ps + 1, dtype=np.int32),
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
    }
    for branch, fields in branch_fields.items():
        for period, values in fields.items():
            payload[f"{branch}__{period}_mm_yr"] = values
    np.savez_compressed(out / "branch_velocity_points.npz", **payload)

    best = min(pooled_rows, key=lambda r: r["rmse_mm_yr"])
    summary = {
        "input_phase": str(phase_input),
        "network_source": str(network_source),
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "reference_ps": int(len(ref_ps)),
        "network_inversion": "current IFGSTD WLS",
        "network_rank": int(rank),
        "robust_deramp": {
            "method": "spatially balanced quality-guided Huber IRLS plane",
            "quality": "pm2.coh_ps",
            "anchor_cell_m": float(args.anchor_cell_m),
            "anchors_per_cell": int(args.anchors_per_cell),
            "anchor_count": int(len(anchors)),
            "huber_delta": float(args.huber_delta),
            "iterations": int(args.huber_iterations),
            "truth_used_for_fit": False,
        },
        "scla_policy": "not directly subtracted; nuisance-only branch is tested for SCN estimation",
        "scn_reference": "median",
        "scn_settings": scn_settings,
        "branches": sorted(branch_fields),
        "best_truth_branch": best["branch"],
        "best_truth_pooled_rmse_mm_yr": best["rmse_mm_yr"],
        "runtime_seconds": time.time() - started,
    }
    (out / "06_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPOOLED TRUTH")
    for row in sorted(pooled_rows, key=lambda r: r["rmse_mm_yr"]):
        print(
            f"{row['branch']:30s} RMSE={row['rmse_mm_yr']:.4f} "
            f"corr={row['correlation']:.4f} bias={row['bias_mm_yr']:.4f}"
        )
    print("\nBEST:", best["branch"], best["rmse_mm_yr"])

    if not args.keep_work:
        del bp_sm
        gc.collect()
        shutil.rmtree(work, ignore_errors=True)
        print("Temporary audit work removed.")

    print("Output:", out)


if __name__ == "__main__":
    main()
