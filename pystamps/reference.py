from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pystamps.config import ReferenceConfig
from pystamps.io.mat import read_mat, read_mat_variables, write_mat


class ReferenceSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceSelection:
    longitude: float
    latitude: float
    radius_m: float
    n_points: int
    method: str
    score: float | None = None
    median_coherence: float | None = None
    p10_coherence: float | None = None
    error_proxy_mad: float | None = None


def _scalar(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _as_rows(value: Any, nrow: int, name: str, dtype=None) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 1 and nrow == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ReferenceSelectionError(f"{name}: expected 2-D, got {arr.shape}")
    if arr.shape[0] != nrow and arr.shape[1] == nrow:
        arr = arr.T
    if arr.shape[0] != nrow:
        raise ReferenceSelectionError(f"{name}: shape={arr.shape}, expected rows={nrow}")
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr


def _rank01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = np.flatnonzero(np.isfinite(x))
    if good.size == 0:
        return out
    if good.size == 1:
        out[good] = 1.0
        return out
    order = good[np.argsort(x[good], kind="mergesort")]
    out[order] = np.linspace(0.0, 1.0, order.size)
    return out


def _distance_mask(lonlat, longitude, latitude, radius_m):
    earth_radius = 6371008.8
    dx = (
        np.deg2rad(lonlat[:, 0] - float(longitude))
        * earth_radius
        * np.cos(np.deg2rad(float(latitude)))
    )
    dy = np.deg2rad(lonlat[:, 1] - float(latitude)) * earth_radius
    return np.hypot(dx, dy) <= float(radius_m)


def _robust_mad(values) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _persist(root, parms, selected, top_candidates=None):
    parms["ref_lon"] = np.asarray([-np.inf, np.inf], dtype=np.float64)
    parms["ref_lat"] = np.asarray([-np.inf, np.inf], dtype=np.float64)
    parms["ref_centre_lonlat"] = np.asarray(
        [selected.longitude, selected.latitude], dtype=np.float64
    )
    parms["ref_radius"] = float(selected.radius_m)
    parms["ref_radius_m"] = float(selected.radius_m)
    parms["reference_selection_method"] = str(selected.method)
    if selected.score is not None:
        parms["reference_selection_score"] = float(selected.score)
    write_mat(root / "parms.mat", parms)

    report = {
        "selected": asdict(selected),
        "priority": "config lon/lat > existing mode > automatic quality region",
        "automatic_quality": {
            "coherence": "maximize regional median and p10 of pm2.coh_ps",
            "error_proxy": "minimize regional robust MAD of pm2.K_ps when available",
            "density": "require sufficient PS inside the configured radius",
        },
        "note": (
            "Automatic selection finds a high-quality relative reference region; "
            "it cannot prove absolute deformation stability. Prefer explicitly "
            "configured coordinates when an externally validated stable region is known."
        ),
    }
    if top_candidates is not None:
        report["top_candidates"] = top_candidates[:10]
    (root / "reference_selection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _existing(parms, lonlat):
    centre = np.asarray(parms.get("ref_centre_lonlat", []), dtype=np.float64).reshape(-1)
    if centre.size < 2:
        raise ReferenceSelectionError(
            "reference.mode='existing' requested but parms.mat has no ref_centre_lonlat"
        )
    radius = _scalar(parms.get("ref_radius"), np.nan)
    if not np.isfinite(radius):
        radius = _scalar(parms.get("ref_radius_m"), np.nan)
    if not np.isfinite(radius) or radius <= 0:
        raise ReferenceSelectionError("Existing reference has no finite positive radius")
    mask = _distance_mask(lonlat, centre[0], centre[1], radius)
    n = int(np.count_nonzero(mask))
    if n == 0:
        raise ReferenceSelectionError("Existing reference contains zero PS")
    return ReferenceSelection(
        longitude=float(centre[0]),
        latitude=float(centre[1]),
        radius_m=float(radius),
        n_points=n,
        method="existing",
    )


def resolve_reference(dataset_root: str | Path, config: ReferenceConfig) -> ReferenceSelection:
    root = Path(dataset_root).expanduser().resolve()
    ps = read_mat(root / "ps2.mat")
    parms = read_mat(root / "parms.mat")

    n_ps = int(round(_scalar(ps.get("n_ps"), 0)))
    if n_ps <= 0:
        raise ReferenceSelectionError("Unable to determine n_ps from ps2.mat")

    lonlat = _as_rows(ps.get("lonlat"), n_ps, "ps2.lonlat", np.float64)

    # 1. Config always wins.
    if config.longitude is not None and config.latitude is not None:
        mask = _distance_mask(
            lonlat, config.longitude, config.latitude, config.radius_m
        )
        n = int(np.count_nonzero(mask))
        if n == 0:
            raise ReferenceSelectionError(
                "Configured reference contains zero PS: "
                f"lon={config.longitude}, lat={config.latitude}, r={config.radius_m} m"
            )
        selected = ReferenceSelection(
            longitude=float(config.longitude),
            latitude=float(config.latitude),
            radius_m=float(config.radius_m),
            n_points=n,
            method="config",
        )
        _persist(root, parms, selected)
        print(
            f"[REFERENCE] config: lon={selected.longitude:.8f}, "
            f"lat={selected.latitude:.8f}, r={selected.radius_m:.0f} m, n={n}",
            flush=True,
        )
        return selected

    # 2. Explicit historical compatibility mode.
    if str(config.mode).strip().lower() == "existing":
        selected = _existing(parms, lonlat)
        _persist(root, parms, selected)
        print(
            f"[REFERENCE] existing: lon={selected.longitude:.8f}, "
            f"lat={selected.latitude:.8f}, r={selected.radius_m:.0f} m, "
            f"n={selected.n_points}",
            flush=True,
        )
        return selected

    # 3. Automatic Stage-5 quality-region search.
    xy = _as_rows(ps.get("xy"), n_ps, "ps2.xy", np.float64)
    if xy.shape[1] < 3:
        raise ReferenceSelectionError("ps2.xy must contain [id, x, y]")
    coords = np.asarray(xy[:, 1:3], dtype=np.float64)

    pm = read_mat_variables(root / "pm2.mat", ("coh_ps", "K_ps"))
    if "coh_ps" not in pm:
        raise ReferenceSelectionError("Automatic reference requires pm2.coh_ps")
    coherence = np.asarray(pm["coh_ps"], dtype=np.float64).reshape(-1)
    if coherence.size != n_ps:
        raise ReferenceSelectionError(
            f"pm2.coh_ps length={coherence.size}; expected {n_ps}"
        )

    if "K_ps" in pm:
        k_ps = np.asarray(pm["K_ps"], dtype=np.float64).reshape(-1)
        if k_ps.size != n_ps:
            k_ps = np.full(n_ps, np.nan, dtype=np.float64)
    else:
        k_ps = np.full(n_ps, np.nan, dtype=np.float64)

    valid = (
        np.all(np.isfinite(coords), axis=1)
        & np.all(np.isfinite(lonlat[:, :2]), axis=1)
        & np.isfinite(coherence)
    )
    valid_ix = np.flatnonzero(valid)
    if valid_ix.size < int(config.min_points):
        raise ReferenceSelectionError(
            f"Too few finite Stage-5 PS for automatic reference: {valid_ix.size}"
        )

    valid_coords = coords[valid_ix]
    tree = cKDTree(valid_coords)
    x0 = float(np.min(valid_coords[:, 0]))
    y0 = float(np.min(valid_coords[:, 1]))
    cell_size = float(config.cell_size_m)

    cx = np.floor((valid_coords[:, 0] - x0) / cell_size).astype(np.int64)
    cy = np.floor((valid_coords[:, 1] - y0) / cell_size).astype(np.int64)
    cells = np.unique(np.column_stack((cx, cy)), axis=0)

    candidates = []
    seen = set()

    for cell_x, cell_y in cells:
        local_cell = np.flatnonzero((cx == cell_x) & (cy == cell_y))
        if local_cell.size < int(config.min_points):
            continue
        centre_xy = np.nanmedian(valid_coords[local_cell], axis=0)
        local_region = np.asarray(
            tree.query_ball_point(centre_xy, r=float(config.radius_m)), dtype=np.int64
        )
        if local_region.size < int(config.min_points):
            continue
        region = np.sort(valid_ix[local_region])
        signature = tuple(region.tolist())
        if signature in seen:
            continue
        seen.add(signature)

        coh = coherence[region]
        ll = lonlat[region, :2]
        candidates.append(
            {
                "longitude": float(np.nanmedian(ll[:, 0])),
                "latitude": float(np.nanmedian(ll[:, 1])),
                "n_points": int(region.size),
                "coherence_median": float(np.nanmedian(coh)),
                "coherence_p10": float(np.nanpercentile(coh, 10)),
                "error_proxy_mad": float(_robust_mad(k_ps[region])),
            }
        )

    if not candidates:
        raise ReferenceSelectionError(
            "No automatic reference region satisfies radius/min_points constraints"
        )

    coh50 = np.asarray([r["coherence_median"] for r in candidates], dtype=np.float64)
    coh10 = np.asarray([r["coherence_p10"] for r in candidates], dtype=np.float64)
    err = np.asarray([r["error_proxy_mad"] for r in candidates], dtype=np.float64)
    counts = np.asarray([r["n_points"] for r in candidates], dtype=np.float64)

    coherence_quality = 0.70 * _rank01(coh50) + 0.30 * _rank01(coh10)
    density_quality = _rank01(np.log1p(counts))

    if np.any(np.isfinite(err)):
        error_quality = _rank01(-err)
        components = np.column_stack((coherence_quality, error_quality, density_quality))
        weights = np.asarray(
            [config.coherence_weight, config.error_proxy_weight, config.density_weight],
            dtype=np.float64,
        )
    else:
        components = np.column_stack((coherence_quality, density_quality))
        weights = np.asarray(
            [config.coherence_weight, config.density_weight], dtype=np.float64
        )

    weights /= np.sum(weights)
    scores = np.sum(components * weights[None, :], axis=1)

    for i, score in enumerate(scores):
        candidates[i]["score"] = float(score)

    def key(i):
        row = candidates[i]
        local_err = float(row["error_proxy_mad"])
        if not np.isfinite(local_err):
            local_err = np.inf
        return (
            float(row["score"]),
            float(row["coherence_median"]),
            float(row["coherence_p10"]),
            -local_err,
            float(row["n_points"]),
        )

    best = candidates[max(range(len(candidates)), key=key)]
    best_err = float(best["error_proxy_mad"])

    selected = ReferenceSelection(
        longitude=float(best["longitude"]),
        latitude=float(best["latitude"]),
        radius_m=float(config.radius_m),
        n_points=int(best["n_points"]),
        method="auto_quality_region",
        score=float(best["score"]),
        median_coherence=float(best["coherence_median"]),
        p10_coherence=float(best["coherence_p10"]),
        error_proxy_mad=best_err if np.isfinite(best_err) else None,
    )

    top = sorted(candidates, key=lambda r: r["score"], reverse=True)
    _persist(root, parms, selected, top_candidates=top)

    print(
        f"[REFERENCE] auto: lon={selected.longitude:.8f}, "
        f"lat={selected.latitude:.8f}, r={selected.radius_m:.0f} m, "
        f"n={selected.n_points}, coh50={selected.median_coherence:.4f}, "
        f"coh10={selected.p10_coherence:.4f}, score={selected.score:.4f}",
        flush=True,
    )
    return selected
