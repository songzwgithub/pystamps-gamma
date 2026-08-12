#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a strict full-period LOS velocity product from GACOS-corrected
pySTAMPS SBAS Stage-8 outputs.

Primary output:
    best_velocity_gacos/rasters/geo_velocity_best_filtered_50m.tif

Method:
    full-period SBAS acquisition covariance
    + covariance whitening
    + Huber robust GLS
    + uncertainty/RMS/local-consistency QC
    + inverse-variance weighted 50 m rasterization after point QC

Positive LOS velocity is toward the satellite.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class BestVelocityError(RuntimeError):
    pass


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise BestVelocityError(
            f"Missing package {name}. Install with:\n"
            f"  python -m pip install {pip_name or name}\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_support(repo_root: Path):
    path = repo_root / "calc_annual_velocity_gls.py"
    if not path.exists():
        raise BestVelocityError(f"Missing support module: {path}")
    spec = importlib.util.spec_from_file_location("annual_gls_support", path)
    if spec is None or spec.loader is None:
        raise BestVelocityError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def robust_cap(
    values: np.ndarray,
    hard_cap: float,
    minimum: float,
    multiplier: float = 3.0,
) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(hard_cap)
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    adaptive = median + multiplier * max(sigma, 0.0)
    return float(max(minimum, min(hard_cap, adaptive)))


def projected_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    pyproj = require_import("pyproj", "pyproj")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, float), np.asarray(y, float)


def local_qc(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    base_mask: np.ndarray,
    *,
    radius: float,
    k: int,
    min_neighbors: int,
    sigma_mult: float,
    floor: float,
    workers: int,
    chunk: int = 20000,
) -> dict[str, np.ndarray]:
    scipy_spatial = require_import("scipy.spatial", "scipy")
    tree_class = scipy_spatial.cKDTree

    good = np.flatnonzero(base_mask)
    if good.size < min_neighbors:
        raise BestVelocityError(
            f"Only {good.size} points passed basic QC"
        )

    coords = np.column_stack((x[good], y[good]))
    tree = tree_class(coords)
    query_k = min(good.size, max(2, k + 1))

    n = velocity.size
    local_median = np.full(n, np.nan, np.float32)
    local_sigma = np.full(n, np.nan, np.float32)
    local_count = np.zeros(n, np.uint16)
    consistent = np.zeros(n, bool)

    for start in range(0, good.size, chunk):
        stop = min(start + chunk, good.size)
        dist, idx = tree.query(
            coords[start:stop],
            k=query_k,
            distance_upper_bound=radius,
            workers=max(1, workers),
        )
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        valid = np.isfinite(dist) & (idx < good.size)
        safe = np.where(valid, idx, 0)
        neigh = velocity[good[safe]].astype(np.float64)
        neigh[~valid] = np.nan

        med = np.nanmedian(neigh, axis=1)
        mad = np.nanmedian(np.abs(neigh - med[:, None]), axis=1)
        sigma = 1.4826 * mad
        count = np.sum(valid, axis=1)
        point_v = velocity[good[start:stop]].astype(np.float64)
        threshold = np.maximum(floor, sigma_mult * sigma)
        keep = (
            (count >= min_neighbors)
            & np.isfinite(med)
            & np.isfinite(sigma)
            & (np.abs(point_v - med) <= threshold)
        )

        target = good[start:stop]
        local_median[target] = med.astype(np.float32)
        local_sigma[target] = sigma.astype(np.float32)
        local_count[target] = np.minimum(count, 65535).astype(np.uint16)
        consistent[target] = keep

        print(
            f"[BEST][LOCAL_QC] {stop}/{good.size} "
            f"({100.0 * stop / good.size:.1f}%)",
            flush=True,
        )

    return {
        "median": local_median,
        "sigma": local_sigma,
        "count": local_count,
        "consistent": consistent,
    }


def make_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int,
    resolution: float,
) -> dict[str, Any]:
    from rasterio.transform import from_origin

    x, y = projected_xy(lon, lat, epsg)
    finite = np.isfinite(x) & np.isfinite(y)
    res = float(resolution)
    xmin = math.floor(float(np.min(x[finite])) / res) * res - res
    xmax = math.ceil(float(np.max(x[finite])) / res) * res + res
    ymin = math.floor(float(np.min(y[finite])) / res) * res - res
    ymax = math.ceil(float(np.max(y[finite])) / res) * res + res
    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    col = np.floor((x - xmin) / res).astype(np.int64)
    row = np.floor((ymax - y) / res).astype(np.int64)
    inside = (
        finite
        & (row >= 0)
        & (row < height)
        & (col >= 0)
        & (col < width)
    )
    return {
        "epsg": epsg,
        "resolution": res,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": width,
        "height": height,
        "row": row,
        "col": col,
        "inside": inside,
        "transform": from_origin(xmin, ymax, res, res),
    }


def weighted_grid(
    values: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
    grid: dict[str, Any],
    *,
    min_points: int,
    error_floor: float,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, float)
    errors = np.asarray(errors, float)
    valid = (
        np.asarray(mask, bool)
        & grid["inside"]
        & np.isfinite(values)
        & np.isfinite(errors)
    )
    if not np.any(valid):
        raise BestVelocityError("No valid points for rasterization")

    width = int(grid["width"])
    height = int(grid["height"])
    ncell = width * height
    linear = grid["row"][valid] * width + grid["col"][valid]
    v = values[valid]
    e = np.maximum(errors[valid], error_floor)
    w = 1.0 / (e * e)

    count = np.bincount(linear, minlength=ncell).astype(np.uint32)
    sw = np.bincount(linear, weights=w, minlength=ncell)
    sw2 = np.bincount(linear, weights=w * w, minlength=ncell)
    swv = np.bincount(linear, weights=w * v, minlength=ncell)
    mean = np.divide(
        swv, sw, out=np.full(ncell, np.nan), where=sw > 0
    )

    residual = v - mean[linear]
    empirical_var = np.divide(
        np.bincount(
            linear,
            weights=w * residual * residual,
            minlength=ncell,
        ),
        sw,
        out=np.zeros(ncell),
        where=sw > 0,
    )
    neff = np.divide(
        sw * sw, sw2, out=np.zeros(ncell), where=sw2 > 0
    )
    formal_var = np.divide(
        1.0, sw, out=np.full(ncell, np.nan), where=sw > 0
    )
    empirical_mean_var = np.divide(
        empirical_var,
        neff,
        out=np.zeros(ncell),
        where=neff > 0,
    )
    standard_error = np.sqrt(
        np.maximum(formal_var + empirical_mean_var, 0.0)
    )

    cell_ok = (
        (count >= min_points)
        & np.isfinite(mean)
        & np.isfinite(standard_error)
    )
    mean[~cell_ok] = np.nan
    standard_error[~cell_ok] = np.nan
    neff[~cell_ok] = 0.0

    shape = (height, width)
    return {
        "velocity": mean.reshape(shape).astype(np.float32),
        "standard_error": standard_error.reshape(shape).astype(np.float32),
        "count": count.reshape(shape),
        "effective_n": neff.reshape(shape).astype(np.float32),
    }


def simple_grid(
    values: np.ndarray,
    mask: np.ndarray,
    grid: dict[str, Any],
    min_points: int,
) -> np.ndarray:
    values = np.asarray(values, float)
    valid = (
        np.asarray(mask, bool)
        & grid["inside"]
        & np.isfinite(values)
    )
    width = int(grid["width"])
    ncell = width * int(grid["height"])
    linear = grid["row"][valid] * width + grid["col"][valid]
    count = np.bincount(linear, minlength=ncell)
    total = np.bincount(
        linear, weights=values[valid], minlength=ncell
    )
    mean = np.divide(
        total,
        count,
        out=np.full(ncell, np.nan),
        where=count >= min_points,
    )
    return mean.reshape(
        int(grid["height"]), width
    ).astype(np.float32)


def write_tif(
    path: Path,
    array: np.ndarray,
    grid: dict[str, Any],
    *,
    description: str,
    unit: str,
    dtype: str = "float32",
    nodata: float | int = -9999.0,
    nearest: bool = False,
) -> None:
    rasterio = require_import("rasterio", "rasterio")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(array).astype(dtype, copy=True)
    if np.issubdtype(np.dtype(dtype), np.floating):
        out[~np.isfinite(out)] = nodata

    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": dtype,
        "crs": f"EPSG:{grid['epsg']}",
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 3 if dtype.startswith("float") else 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            description=description,
            unit=unit,
            los_sign="positive_toward_satellite",
            estimator="full-period SBAS covariance + Huber robust GLS",
            quality_control=(
                "uncertainty + temporal RMS + local consistency"
            ),
            interpolation="none",
        )
        factors = [
            f for f in (2, 4, 8, 16, 32)
            if int(grid["width"]) // f >= 1
            and int(grid["height"]) // f >= 1
        ]
        if factors:
            method = (
                rasterio.enums.Resampling.nearest
                if nearest
                else rasterio.enums.Resampling.average
            )
            dst.build_overviews(factors, method)


def reproject_wgs84(src: Path, dst: Path) -> None:
    rasterio = require_import("rasterio", "rasterio")
    from rasterio.warp import (
        calculate_default_transform,
        reproject,
        Resampling,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as source:
        transform, width, height = calculate_default_transform(
            source.crs,
            "EPSG:4326",
            source.width,
            source.height,
            *source.bounds,
        )
        profile = source.profile.copy()
        profile.update(
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(dst, "w", **profile) as target:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(target, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                src_nodata=source.nodata,
                dst_nodata=source.nodata,
                resampling=Resampling.bilinear,
            )


def write_qml(path: Path, limit: float) -> None:
    text = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1"
      classificationMin="{-limit}" classificationMax="{limit}">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <item alpha="255" value="{-limit}" label="{-limit:.1f}" color="#2166ac"/>
          <item alpha="255" value="{-limit/2}" label="{-limit/2:.1f}" color="#67a9cf"/>
          <item alpha="255" value="0" label="0" color="#f7f7f7"/>
          <item alpha="255" value="{limit/2}" label="{limit/2:.1f}" color="#ef8a62"/>
          <item alpha="255" value="{limit}" label="{limit:.1f}" color="#b2182b"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    path.write_text(text, encoding="utf-8")


def plot_map(
    path: Path,
    raster: np.ndarray,
    grid: dict[str, Any],
    title: str,
    label: str,
    cmap: str,
    symmetric: bool,
    dpi: int,
) -> None:
    matplotlib = require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = raster[np.isfinite(raster)]
    if finite.size == 0:
        return
    if symmetric:
        vmax = max(1.0, float(np.percentile(np.abs(finite), 98)))
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = max(0.1, float(np.percentile(finite, 98)))

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        raster,
        extent=(
            grid["xmin"], grid["xmax"],
            grid["ymin"], grid["ymax"],
        ),
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(f"Easting (m), EPSG:{grid['epsg']}")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_h5(
    path: Path,
    inputs: Any,
    result: dict[str, np.ndarray],
    best_mask: np.ndarray,
    local: dict[str, np.ndarray],
    reason: np.ndarray,
) -> None:
    h5py = require_import("h5py", "h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "pySTAMPS_best_full_period_velocity"
        h5.attrs["los_sign"] = "positive_toward_satellite"
        h5.create_dataset("lon", data=inputs.lon, compression="gzip")
        h5.create_dataset("lat", data=inputs.lat, compression="gzip")
        for name in (
            "velocity_mm_yr",
            "velocity_std_mm_yr",
            "velocity_gls_mm_yr",
            "velocity_ols_mm_yr",
            "ci95_low_mm_yr",
            "ci95_high_mm_yr",
            "rmse_mm",
            "whitened_rmse",
            "n_obs",
            "span_days",
            "effective_n",
            "downweighted_mode_count",
            "design_condition",
            "irls_iterations",
            "accepted",
        ):
            h5.create_dataset(
                name,
                data=result[name],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        h5.create_dataset(
            "local_median_mm_yr",
            data=local["median"],
            compression="gzip",
        )
        h5.create_dataset(
            "local_sigma_mm_yr",
            data=local["sigma"],
            compression="gzip",
        )
        h5.create_dataset(
            "local_neighbor_count",
            data=local["count"],
            compression="gzip",
        )
        h5.create_dataset(
            "best_quality_mask",
            data=best_mask.astype(np.uint8),
            compression="gzip",
        )
        h5.create_dataset(
            "quality_reason_code",
            data=reason,
            compression="gzip",
        )


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dataset / "best_velocity_gacos"
    )
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    support = load_support(repo_root)
    inputs = support.load_inputs(
        dataset, repo_root, args.sigma_floor_deg
    )

    total_span = float(np.max(inputs.day) - np.min(inputs.day))
    min_epochs = max(
        args.min_epochs,
        int(math.ceil(args.min_valid_fraction * inputs.n_epoch)),
    )
    min_span = max(
        args.min_span_days,
        args.min_span_fraction * total_span,
    )

    print("============================================================")
    print("Full-period robust network GLS")
    print("============================================================")
    print(f"PS             : {inputs.n_ps}")
    print(f"Acquisitions   : {inputs.n_epoch}")
    print(f"Total span     : {total_span:.1f} days")
    print(f"Min epochs     : {min_epochs}")
    print(f"Min span       : {min_span:.1f} days")
    print(f"Covariance     : {inputs.covariance_source}")

    result, fit_meta = support.compute_year(
        inputs,
        np.arange(inputs.n_epoch, dtype=np.int64),
        min_epochs=min_epochs,
        min_span_days=min_span,
        chunk_ps=args.chunk_ps,
        covariance_mode=args.covariance_mode,
        eigen_floor_rel=args.eigen_floor_rel,
        robust=True,
        huber_c=args.huber_c,
        weight_floor=args.weight_floor,
        max_iterations=args.irls_iterations,
        convergence=args.convergence,
    )

    velocity = np.asarray(result["velocity_mm_yr"], np.float32)
    rate_se = np.asarray(result["velocity_std_mm_yr"], np.float32)
    rmse = np.asarray(result["rmse_mm"], np.float32)
    fit_ok = (
        result["accepted"].astype(bool)
        & np.isfinite(velocity)
        & np.isfinite(rate_se)
        & np.isfinite(rmse)
    )

    rmse_cap = robust_cap(
        rmse[fit_ok],
        args.max_rmse_mm,
        args.min_rmse_cap_mm,
    )
    se_cap = robust_cap(
        rate_se[fit_ok],
        args.max_rate_se_mm_yr,
        args.min_rate_se_cap_mm_yr,
    )

    basic = (
        fit_ok
        & (rmse <= rmse_cap)
        & (rate_se <= se_cap)
        & (np.abs(velocity) <= args.absolute_rate_cap_mm_yr)
    )

    epsg = args.target_epsg
    template = dataset / "postprocess" / "rasters" / "geo_velocity.tif"
    if epsg is None and template.exists():
        rasterio = require_import("rasterio", "rasterio")
        with rasterio.open(template) as src:
            epsg = src.crs.to_epsg() if src.crs else None
    if epsg is None:
        epsg = support.auto_utm_epsg(inputs.lon, inputs.lat)

    x, y = projected_xy(inputs.lon, inputs.lat, int(epsg))
    local = local_qc(
        x,
        y,
        velocity,
        basic,
        radius=args.local_radius_m,
        k=args.local_k,
        min_neighbors=args.local_min_neighbors,
        sigma_mult=args.local_sigma_multiplier,
        floor=args.local_floor_mm_yr,
        workers=args.local_workers,
    )
    best = basic & local["consistent"]

    reason = np.zeros(inputs.n_ps, np.uint8)
    reason[~fit_ok] = 1
    reason[fit_ok & (rmse > rmse_cap)] = 2
    reason[fit_ok & (rmse <= rmse_cap) & (rate_se > se_cap)] = 3
    reason[
        basic & (local["count"] < args.local_min_neighbors)
    ] = 4
    reason[
        basic
        & (local["count"] >= args.local_min_neighbors)
        & ~local["consistent"]
    ] = 5
    reason[best] = 6

    main_grid = make_grid(
        inputs.lon, inputs.lat, int(epsg), args.main_resolution_m
    )
    main = weighted_grid(
        velocity,
        rate_se,
        best,
        main_grid,
        min_points=args.main_min_points,
        error_floor=args.error_floor_mm_yr,
    )
    main_rmse = simple_grid(
        rmse, best, main_grid, args.main_min_points
    )
    main_local_sigma = simple_grid(
        local["sigma"], best, main_grid, args.main_min_points
    )

    detail_grid = make_grid(
        inputs.lon, inputs.lat, int(epsg), args.detail_resolution_m
    )
    detail = weighted_grid(
        velocity,
        rate_se,
        best,
        detail_grid,
        min_points=args.detail_min_points,
        error_floor=args.error_floor_mm_yr,
    )

    rasters = out / "rasters"
    primary = rasters / "geo_velocity_best_filtered_50m.tif"
    write_tif(
        primary,
        main["velocity"],
        main_grid,
        description="Best-quality full-period LOS velocity",
        unit="mm/yr",
    )
    write_tif(
        rasters / "geo_velocity_best_filtered_std_50m.tif",
        main["standard_error"],
        main_grid,
        description="Best-velocity combined standard error",
        unit="mm/yr",
    )
    write_tif(
        rasters / "geo_velocity_best_filtered_count_50m.tif",
        main["count"],
        main_grid,
        description="Accepted PS count per 50 m cell",
        unit="count",
        dtype="uint32",
        nodata=0,
        nearest=True,
    )
    write_tif(
        rasters / "geo_velocity_best_filtered_neff_50m.tif",
        main["effective_n"],
        main_grid,
        description="Effective inverse-variance PS count",
        unit="count",
    )
    write_tif(
        rasters / "geo_velocity_best_filtered_rmse_50m.tif",
        main_rmse,
        main_grid,
        description="Mean robust temporal-fit RMS",
        unit="mm",
    )
    write_tif(
        rasters / "geo_velocity_best_filtered_local_sigma_50m.tif",
        main_local_sigma,
        main_grid,
        description="Mean local robust velocity scatter",
        unit="mm/yr",
    )
    write_tif(
        rasters / "geo_velocity_best_aggregated_100m.tif",
        detail["velocity"],
        detail_grid,
        description="Best-quality aggregated 100 m LOS velocity",
        unit="mm/yr",
    )

    if args.wgs84_copy:
        reproject_wgs84(
            primary,
            rasters / "wgs84" / "geo_velocity_best_filtered_50m_wgs84.tif",
        )

    finite_main = main["velocity"][np.isfinite(main["velocity"])]
    limit = (
        max(5.0, math.ceil(float(np.percentile(np.abs(finite_main), 98))))
        if finite_main.size
        else 20.0
    )
    write_qml(
        rasters / "geo_velocity_best_filtered_50m.qml",
        limit,
    )

    plot_map(
        out / "plots" / "01_best_velocity_map.png",
        main["velocity"],
        main_grid,
        "Best-quality full-period LOS velocity",
        "mm/yr; positive toward satellite",
        "RdBu_r",
        True,
        args.plot_dpi,
    )
    plot_map(
        out / "plots" / "02_best_velocity_uncertainty.png",
        main["standard_error"],
        main_grid,
        "Best-velocity standard error",
        "mm/yr",
        "viridis",
        False,
        args.plot_dpi,
    )

    write_h5(
        out / "points" / "best_velocity_points.h5",
        inputs,
        result,
        best,
        local,
        reason,
    )

    provenance = {}
    for name in (
        "gacos_correction_debug.json",
        "stage7_sbas_debug.json",
        "stage8_sbas_debug.json",
    ):
        path = dataset / name
        if path.exists():
            try:
                provenance[name] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except Exception:
                provenance[name] = {"path": str(path)}

    raw = velocity[np.isfinite(velocity)]
    selected = velocity[best]
    report = {
        "status": "completed",
        "dataset": str(dataset),
        "output": str(out),
        "gacos_corrected": (
            (dataset / "gacos_correction_debug.json").exists()
        ),
        "estimator": (
            "full-period SBAS covariance + Huber robust GLS"
        ),
        "covariance_source": inputs.covariance_source,
        "n_ps": inputs.n_ps,
        "n_epoch": inputs.n_epoch,
        "n_ifg": inputs.n_ifg,
        "date_start": inputs.labels[0],
        "date_end": inputs.labels[-1],
        "quality_thresholds": {
            "min_epochs": min_epochs,
            "min_span_days": min_span,
            "rmse_cap_mm": rmse_cap,
            "rate_se_cap_mm_yr": se_cap,
            "local_radius_m": args.local_radius_m,
            "local_min_neighbors": args.local_min_neighbors,
            "local_sigma_multiplier": args.local_sigma_multiplier,
            "main_resolution_m": args.main_resolution_m,
            "main_min_points": args.main_min_points,
        },
        "point_counts": {
            "fit_ok": int(np.count_nonzero(fit_ok)),
            "basic_qc": int(np.count_nonzero(basic)),
            "best_qc": int(np.count_nonzero(best)),
            "best_fraction": float(np.mean(best)),
        },
        "raw_velocity_statistics": {
            "count": int(raw.size),
            "p02": float(np.percentile(raw, 2)) if raw.size else None,
            "median": float(np.median(raw)) if raw.size else None,
            "mean": float(np.mean(raw)) if raw.size else None,
            "p98": float(np.percentile(raw, 98)) if raw.size else None,
        },
        "best_ps_statistics": {
            "count": int(selected.size),
            "p02": float(np.percentile(selected, 2)) if selected.size else None,
            "median": float(np.median(selected)) if selected.size else None,
            "mean": float(np.mean(selected)) if selected.size else None,
            "p98": float(np.percentile(selected, 98)) if selected.size else None,
        },
        "best_raster_statistics": {
            "count": int(finite_main.size),
            "p02": float(np.percentile(finite_main, 2)) if finite_main.size else None,
            "median": float(np.median(finite_main)) if finite_main.size else None,
            "mean": float(np.mean(finite_main)) if finite_main.size else None,
            "p98": float(np.percentile(finite_main, 98)) if finite_main.size else None,
            "qgis_display_range": [-limit, limit],
        },
        "primary_product": str(primary),
        "qgis_style": str(
            rasters / "geo_velocity_best_filtered_50m.qml"
        ),
        "fit_metadata": fit_meta,
        "provenance": provenance,
        "duration_sec": time.perf_counter() - started,
        "note": (
            "No spatial interpolation is used. Empty cells remain NoData."
        ),
    }
    (out / "best_velocity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n============================================================")
    print("Best velocity product completed")
    print("============================================================")
    print(f"Fit-valid PS   : {np.count_nonzero(fit_ok):,}")
    print(f"Basic-QC PS    : {np.count_nonzero(basic):,}")
    print(f"Best-QC PS     : {np.count_nonzero(best):,}")
    print(f"RMS threshold  : {rmse_cap:.3f} mm")
    print(f"SE threshold   : {se_cap:.3f} mm/yr")
    print(f"Primary raster : {primary}")
    print(
        "QGIS style    : "
        f"{rasters / 'geo_velocity_best_100m.qml'}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create the strict best-quality GACOS-corrected LOS velocity product"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--repo-root",
        default="/home/ubuntu/software/pystamps-main",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--chunk-ps", type=int, default=4096)
    p.add_argument(
        "--covariance-mode",
        choices=("network", "diagonal", "identity"),
        default="network",
    )
    p.add_argument("--eigen-floor-rel", type=float, default=1e-6)
    p.add_argument("--sigma-floor-deg", type=float, default=0.1)
    p.add_argument("--huber-c", type=float, default=1.345)
    p.add_argument("--weight-floor", type=float, default=0.05)
    p.add_argument("--irls-iterations", type=int, default=8)
    p.add_argument("--convergence", type=float, default=1e-6)

    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--min-valid-fraction", type=float, default=0.80)
    p.add_argument("--min-span-days", type=float, default=365.0)
    p.add_argument("--min-span-fraction", type=float, default=0.80)

    p.add_argument("--max-rmse-mm", type=float, default=12.0)
    p.add_argument("--min-rmse-cap-mm", type=float, default=6.0)
    p.add_argument("--max-rate-se-mm-yr", type=float, default=2.0)
    p.add_argument("--min-rate-se-cap-mm-yr", type=float, default=0.5)
    p.add_argument("--absolute-rate-cap-mm-yr", type=float, default=100.0)

    p.add_argument("--local-radius-m", type=float, default=300.0)
    p.add_argument("--local-k", type=int, default=12)
    p.add_argument("--local-min-neighbors", type=int, default=4)
    p.add_argument("--local-sigma-multiplier", type=float, default=3.5)
    p.add_argument("--local-floor-mm-yr", type=float, default=3.0)
    p.add_argument(
        "--local-workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
    )

    p.add_argument("--main-resolution-m", type=float, default=50.0)
    p.add_argument("--main-min-points", type=int, default=1)
    p.add_argument("--detail-resolution-m", type=float, default=100.0)
    p.add_argument("--detail-min-points", type=int, default=3)
    p.add_argument("--error-floor-mm-yr", type=float, default=0.25)
    p.add_argument("--target-epsg", type=int, default=None)
    p.add_argument("--wgs84-copy", action="store_true")
    p.add_argument("--plot-dpi", type=int, default=220)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    try:
        return run(parser().parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
