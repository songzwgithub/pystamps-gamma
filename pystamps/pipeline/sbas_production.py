from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat_variables


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else float(default)


def select_balanced_anchors(coords_m, quality, cell_m, per_cell):
    x = np.asarray(coords_m[:, 0], dtype=np.float64)
    y = np.asarray(coords_m[:, 1], dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    x0 = float(np.nanmin(x)); y0 = float(np.nanmin(y))
    cx = np.floor((x - x0) / float(cell_m)).astype(np.int64)
    cy = np.floor((y - y0) / float(cell_m)).astype(np.int64)
    ny = int(np.nanmax(cy)) + 1
    cell = cx * max(ny, 1) + cy
    ids = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & np.isfinite(q))
    order = np.lexsort((-q[ids], cell[ids]))
    ids = ids[order]; cells = cell[ids]
    keep = np.zeros(ids.size, dtype=bool)
    last = None; count = 0
    for i, c in enumerate(cells):
        if last is None or c != last:
            last = c; count = 0
        if count < int(per_cell):
            keep[i] = True; count += 1
    return np.sort(ids[keep])


def _weighted_plane(X, y, w):
    good = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    Xv = X[good]; yv = y[good]; wv = w[good]
    if Xv.shape[0] < 10:
        return np.full(3, np.nan, dtype=np.float64)
    sw = np.sqrt(wv)
    return np.linalg.lstsq(Xv * sw[:, None], yv * sw, rcond=None)[0]


def _huber_plane(X, y, q, iterations, delta):
    beta = _weighted_plane(X, y, q)
    if not np.all(np.isfinite(beta)):
        return beta, np.nan, 0
    scale = np.nan; used = 0
    for _ in range(int(iterations)):
        residual = y - X @ beta
        good = np.isfinite(residual) & np.isfinite(q) & (q > 0)
        if np.count_nonzero(good) < 10:
            break
        r = residual[good]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1.0e-8:
            break
        u = np.abs(residual) / (float(delta) * scale)
        robust = np.ones_like(u)
        hi = u > 1.0
        robust[hi] = 1.0 / u[hi]
        beta_new = _weighted_plane(X, y, q * robust)
        if not np.all(np.isfinite(beta_new)):
            break
        used += 1
        if np.max(np.abs(beta_new - beta)) < 1.0e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta, float(scale), used


def robust_deramp_unwrapped_phase(dataset_root: Path, ps: dict[str, Any], ph_all, parms):
    root = Path(dataset_root).expanduser().resolve()
    ph = np.asarray(ph_all, dtype=np.float64)
    n_ps, n_ifg = ph.shape
    xy = np.asarray(ps.get("xy"), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != n_ps or xy.shape[1] < 3:
        raise RuntimeError("ps2.xy is required for robust SBAS deramp")
    coords_m = np.asarray(xy[:, 1:3], dtype=np.float64)

    cell_m = max(_scalar(parms.get("sbas_deramp_cell_m"), 2000.0), 1.0)
    per_cell = max(int(round(_scalar(parms.get("sbas_deramp_anchors_per_cell"), 8))), 1)
    delta = max(_scalar(parms.get("sbas_deramp_huber_delta"), 1.345), 1.0e-6)
    iterations = max(int(round(_scalar(parms.get("sbas_deramp_huber_iterations"), 5))), 1)

    quality = np.asarray(
        read_mat_variables(root / "pm2.mat", ("coh_ps",))["coh_ps"],
        dtype=np.float64,
    ).reshape(-1)
    if quality.size != n_ps:
        raise RuntimeError(f"pm2.coh_ps length={quality.size}; expected {n_ps}")

    anchors = select_balanced_anchors(coords_m, quality, cell_m, per_cell)
    if anchors.size < 10:
        raise RuntimeError(f"robust deramp selected only {anchors.size} anchors")

    # Use the fixed reference-region center only for numerical conditioning of x/y.
    ref_ix = np.asarray([], dtype=np.int64)
    center_ll = np.asarray(parms.get("ref_centre_lonlat", []), dtype=np.float64).reshape(-1)
    lonlat = np.asarray(ps.get("lonlat", []), dtype=np.float64)
    if center_ll.size >= 2 and lonlat.ndim == 2:
        if lonlat.shape[0] != n_ps and lonlat.shape[1] == n_ps:
            lonlat = lonlat.T
        if lonlat.shape[0] == n_ps and lonlat.shape[1] >= 2:
            R = 6371008.8
            lon0, lat0 = float(center_ll[0]), float(center_ll[1])
            dx = np.deg2rad(lonlat[:, 0] - lon0) * R * np.cos(np.deg2rad(lat0))
            dy = np.deg2rad(lonlat[:, 1] - lat0) * R
            radius_m = _scalar(parms.get("ref_radius_m"), 500.0)
            ref_ix = np.flatnonzero(np.hypot(dx, dy) <= radius_m)
    center_xy = np.nanmedian(coords_m[ref_ix], axis=0) if ref_ix.size else np.nanmedian(coords_m, axis=0)

    Xa = np.column_stack((
        (coords_m[anchors, 0] - center_xy[0]) / 1000.0,
        (coords_m[anchors, 1] - center_xy[1]) / 1000.0,
        np.ones(anchors.size, dtype=np.float64),
    ))
    q = np.clip(quality[anchors], 0.05, 1.0) ** 2
    phase_anchor = ph[anchors]

    coeff = np.full((3, n_ifg), np.nan, dtype=np.float64)
    scale = np.full(n_ifg, np.nan, dtype=np.float64)
    used = np.zeros(n_ifg, dtype=np.int32)
    for j in range(n_ifg):
        beta, sc, it = _huber_plane(Xa, phase_anchor[:, j], q, iterations, delta)
        coeff[:, j] = beta; scale[j] = sc; used[j] = it
        if (j + 1) % 100 == 0 or j + 1 == n_ifg:
            print(f"[STAGE7_SBAS][ROBUST_DERAMP] {j+1}/{n_ifg}", flush=True)

    Xall = np.column_stack((
        (coords_m[:, 0] - center_xy[0]) / 1000.0,
        (coords_m[:, 1] - center_xy[1]) / 1000.0,
        np.ones(n_ps, dtype=np.float64),
    ))
    ph_ramp = Xall @ coeff
    ph_out = ph - ph_ramp

    debug = {
        "mode": "robust_huber_balanced",
        "quality_source": "pm2.coh_ps",
        "cell_m": float(cell_m),
        "anchors_per_cell": int(per_cell),
        "anchor_count": int(anchors.size),
        "huber_delta": float(delta),
        "huber_iterations": int(iterations),
        "median_residual_scale_rad": float(np.nanmedian(scale)),
        "p95_residual_scale_rad": float(np.nanpercentile(scale, 95)),
        "median_iterations_used": float(np.nanmedian(used)),
    }
    return ph_out, ph_ramp, debug
