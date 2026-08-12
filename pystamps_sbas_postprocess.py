#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pySTAMPS SBAS Stage 7/8 geocoding, plotting and export workflow.

Designed for datasets containing:
  ps2.mat
  mean_v.mat
  scla_smooth2.mat
  phuw_sm2.mat
  uw_space_time.mat
  stage7_sbas_debug.json
  stage8_sbas_debug.json

Main outputs:
  points/ps_velocity.csv
  points/ps_velocity.gpkg              (optional)
  points/ps_velocity.parquet           (optional)
  points/ps_velocity_sample.kml        (optional)
  rasters/geo_velocity.tif
  rasters/geo_temporal_rms_mm.tif
  rasters/geo_scla_k_rad_per_m.tif
  rasters/geo_scla_c_rad.tif
  rasters/geo_ps_count.tif
  timeseries/ps_timeseries.h5           (optional)
  geo_timeseries/geo_YYYYMMDD.tif       (optional)
  plots/*.png
  postprocess_report.json
  postprocess_manifest.csv

Scientific conventions:
  * LOS displacement uses the Stage-8 convention:
        displacement_mm = -phase_rad * wavelength_m / (4*pi) * 1000
  * Positive displacement is toward the satellite.
  * Default "existing" reference mode preserves the Stage-8 reference.
  * Rasterization is bin averaging, not spatial interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape as xml_escape

import numpy as np


class PostprocessError(RuntimeError):
    """Fatal post-processing error."""


@dataclass(slots=True)
class RasterGrid:
    epsg: int
    resolution: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    x: np.ndarray
    y: np.ndarray
    row: np.ndarray
    col: np.ndarray
    inside: np.ndarray


def _require_import(name: str, install_hint: str | None = None) -> Any:
    try:
        return __import__(name)
    except Exception as exc:
        hint = install_hint or name
        raise PostprocessError(
            f"Missing required package '{name}'. Install it in a separate "
            f"post-processing environment, for example:\n"
            f"  python -m pip install {hint}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _optional_import(name: str) -> Any | None:
    try:
        return __import__(name)
    except Exception:
        return None


def _load_pystamps_io(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from pystamps.io.mat import read_mat, read_mat_variables
    except Exception as exc:
        raise PostprocessError(
            f"Unable to import pystamps.io.mat from {repo_root}: {exc}"
        ) from exc
    return read_mat, read_mat_variables


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _vector(value: Any, name: str, dtype: Any = np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size == 0:
        raise PostprocessError(f"{name} is empty")
    return arr


def _matrix(value: Any, rows: int, name: str, dtype: Any = np.float32) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise PostprocessError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise PostprocessError(
            f"{name} shape {arr.shape}; expected first dimension {rows}"
        )
    return np.asarray(arr, dtype=dtype)


def matlab_datenum_to_datetime(value: float) -> datetime:
    day_int = int(math.floor(float(value)))
    frac = float(value) - day_int
    return datetime.fromordinal(day_int) + timedelta(days=frac) - timedelta(days=366)


def day_to_date_labels(day: np.ndarray) -> tuple[list[str], list[datetime]]:
    values = np.asarray(day, dtype=np.float64).reshape(-1)
    dates: list[datetime] = []
    labels: list[str] = []

    # MATLAB datenum is normally > 500000.
    if values.size and np.nanmedian(values) > 500000:
        for value in values:
            dt = matlab_datenum_to_datetime(float(value))
            dates.append(dt)
            labels.append(dt.strftime("%Y%m%d"))
        return labels, dates

    # YYYYMMDD numeric fallback.
    if values.size and np.nanmedian(values) > 10_000_000:
        for value in values:
            dt = datetime.strptime(str(int(round(value))), "%Y%m%d")
            dates.append(dt)
            labels.append(dt.strftime("%Y%m%d"))
        return labels, dates

    # Relative-day fallback. Keep deterministic labels.
    origin = datetime(1970, 1, 1)
    for value in values:
        dt = origin + timedelta(days=float(value))
        dates.append(dt)
        labels.append(dt.strftime("%Y%m%d"))
    return labels, dates


def auto_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = int(math.floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat0 >= 0 else 32700) + zone


def build_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    resolution_m: float,
    target_epsg: int | None,
    padding_cells: int = 1,
) -> RasterGrid:
    pyproj = _require_import("pyproj", "pyproj")
    epsg = int(target_epsg or auto_utm_epsg(lon, lat))
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise PostprocessError("No finite projected point coordinates")

    res = float(resolution_m)
    if not math.isfinite(res) or res <= 0:
        raise PostprocessError("--resolution-m must be positive")

    xmin0 = float(np.nanmin(x[finite]))
    xmax0 = float(np.nanmax(x[finite]))
    ymin0 = float(np.nanmin(y[finite]))
    ymax0 = float(np.nanmax(y[finite]))

    xmin = math.floor(xmin0 / res) * res - padding_cells * res
    xmax = math.ceil(xmax0 / res) * res + padding_cells * res
    ymin = math.floor(ymin0 / res) * res - padding_cells * res
    ymax = math.ceil(ymax0 / res) * res + padding_cells * res

    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    if width <= 0 or height <= 0:
        raise PostprocessError(f"Invalid raster size {width} x {height}")
    if width * height > 400_000_000:
        raise PostprocessError(
            f"Raster would contain {width * height:,} cells. Increase --resolution-m."
        )

    col = np.floor((x - xmin) / res).astype(np.int64)
    row = np.floor((ymax - y) / res).astype(np.int64)
    inside = (
        finite
        & (col >= 0)
        & (col < width)
        & (row >= 0)
        & (row < height)
    )
    return RasterGrid(
        epsg=epsg,
        resolution=res,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        width=width,
        height=height,
        x=x,
        y=y,
        row=row,
        col=col,
        inside=inside,
    )


def aggregate_mean(
    values: np.ndarray,
    grid: RasterGrid,
    min_points: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size != grid.row.size:
        raise PostprocessError(
            f"Raster input length {vals.size}; expected {grid.row.size}"
        )

    valid = grid.inside & np.isfinite(vals)
    ncell = grid.width * grid.height
    if not np.any(valid):
        return (
            np.full((grid.height, grid.width), np.nan, dtype=np.float32),
            np.zeros((grid.height, grid.width), dtype=np.uint32),
        )

    linear = grid.row[valid] * grid.width + grid.col[valid]
    sums = np.bincount(
        linear,
        weights=vals[valid],
        minlength=ncell,
    ).astype(np.float64)
    counts = np.bincount(linear, minlength=ncell).astype(np.uint32)
    mean = np.divide(
        sums,
        counts,
        out=np.full(ncell, np.nan, dtype=np.float64),
        where=counts >= max(1, int(min_points)),
    )
    return (
        mean.reshape(grid.height, grid.width).astype(np.float32),
        counts.reshape(grid.height, grid.width),
    )


def write_geotiff(
    path: Path,
    array: np.ndarray,
    grid: RasterGrid,
    *,
    nodata: float | int,
    dtype: str,
    description: str,
    unit: str | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    rasterio = _require_import("rasterio", "rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    out = arr.astype(dtype, copy=True)

    if np.issubdtype(np.dtype(dtype), np.floating):
        out[~np.isfinite(out)] = nodata

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": f"EPSG:{grid.epsg}",
        "transform": from_origin(
            grid.xmin,
            grid.ymax,
            grid.resolution,
            grid.resolution,
        ),
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
        tag_payload = {
            "AREA_OR_POINT": "Area",
            "description": description,
            "positive_los_direction": "toward_satellite",
            "rasterization": "mean_of_PS_points_in_cell",
        }
        if unit:
            tag_payload["unit"] = unit
        if tags:
            tag_payload.update({str(k): str(v) for k, v in tags.items()})
        dst.update_tags(**tag_payload)

        factors = [
            f for f in (2, 4, 8, 16, 32)
            if grid.width // f >= 1 and grid.height // f >= 1
        ]
        if factors:
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")


def reproject_to_wgs84(src_path: Path, dst_path: Path) -> None:
    rasterio = _require_import("rasterio", "rasterio")
    from rasterio.warp import (
        calculate_default_transform,
        reproject,
        Resampling,
    )

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
            dst.update_tags(**src.tags())


def select_reference_indices(
    mode: str,
    lon: np.ndarray,
    lat: np.ndarray,
    grid: RasterGrid,
    *,
    ref_lon: float | None,
    ref_lat: float | None,
    ref_radius_m: float,
    ref_bbox: Sequence[float] | None,
) -> np.ndarray:
    mode = mode.lower()
    if mode in {"existing", "none"}:
        return np.asarray([], dtype=np.int64)
    if mode == "global-median":
        return np.flatnonzero(np.isfinite(lon) & np.isfinite(lat)).astype(np.int64)
    if mode == "point":
        if ref_lon is None or ref_lat is None:
            raise PostprocessError(
                "--reference-mode point requires --ref-lon and --ref-lat"
            )
        pyproj = _require_import("pyproj", "pyproj")
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True
        )
        rx, ry = transformer.transform(float(ref_lon), float(ref_lat))
        dist = np.hypot(grid.x - float(rx), grid.y - float(ry))
        ix = np.flatnonzero(
            np.isfinite(dist) & (dist <= float(ref_radius_m))
        )
        if ix.size == 0:
            nearest = int(np.nanargmin(dist))
            warnings.warn(
                "No PS found inside reference radius; using the nearest PS."
            )
            ix = np.asarray([nearest], dtype=np.int64)
        return ix.astype(np.int64)
    if mode == "bbox":
        if ref_bbox is None or len(ref_bbox) != 4:
            raise PostprocessError(
                "--reference-mode bbox requires --ref-bbox xmin ymin xmax ymax"
            )
        xmin, ymin, xmax, ymax = map(float, ref_bbox)
        ix = np.flatnonzero(
            np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= xmin)
            & (lon <= xmax)
            & (lat >= ymin)
            & (lat <= ymax)
        )
        if ix.size == 0:
            raise PostprocessError("No PS found inside reference bounding box")
        return ix.astype(np.int64)
    raise PostprocessError(f"Unsupported reference mode: {mode}")


def write_points_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    pandas = _require_import("pandas", "pandas")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame(columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_points_parquet(path: Path, columns: dict[str, np.ndarray]) -> bool:
    pandas = _require_import("pandas", "pandas")
    try:
        frame = pandas.DataFrame(columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return True
    except Exception as exc:
        warnings.warn(f"Parquet export skipped: {exc}")
        return False


def write_points_gpkg(path: Path, columns: dict[str, np.ndarray]) -> bool:
    geopandas = _optional_import("geopandas")
    if geopandas is None:
        warnings.warn("GeoPackage export skipped because geopandas is unavailable")
        return False
    try:
        pandas = _require_import("pandas", "pandas")
        frame = pandas.DataFrame(columns)
        geometry = geopandas.points_from_xy(frame["lon"], frame["lat"])
        gdf = geopandas.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        gdf.to_file(path, layer="ps_velocity", driver="GPKG")
        return True
    except Exception as exc:
        warnings.warn(f"GeoPackage export skipped: {exc}")
        return False


def write_points_shp(path: Path, columns: dict[str, np.ndarray]) -> bool:
    geopandas = _optional_import("geopandas")
    if geopandas is None:
        warnings.warn("Shapefile export skipped because geopandas is unavailable")
        return False
    try:
        pandas = _require_import("pandas", "pandas")
        short = {
            "ps_id": columns["ps_id"],
            "lon": columns["lon"],
            "lat": columns["lat"],
            "vel_mm_yr": columns["velocity_mm_yr"],
            "rms_mm": columns["temporal_rms_mm"],
            "scla_k": columns["scla_k_rad_per_m"],
            "scla_c": columns["scla_c_rad"],
        }
        frame = pandas.DataFrame(short)
        geometry = geopandas.points_from_xy(frame["lon"], frame["lat"])
        gdf = geopandas.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
        path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in path.parent.glob(path.stem + ".*"):
            candidate.unlink()
        gdf.to_file(path, driver="ESRI Shapefile", encoding="UTF-8")
        return True
    except Exception as exc:
        warnings.warn(f"Shapefile export skipped: {exc}")
        return False


def _kml_color(value: float, vmin: float, vmax: float) -> str:
    # KML color order is AABBGGRR. Simple blue-white-red ramp.
    if not math.isfinite(value):
        return "ff808080"
    if vmax <= vmin:
        t = 0.5
    else:
        t = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
    if t <= 0.5:
        q = t / 0.5
        r = int(round(255 * q))
        g = int(round(255 * q))
        b = 255
    else:
        q = (t - 0.5) / 0.5
        r = 255
        g = int(round(255 * (1.0 - q)))
        b = int(round(255 * (1.0 - q)))
    return f"ff{b:02x}{g:02x}{r:02x}"


def write_points_kml(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    velocity: np.ndarray,
    rms_mm: np.ndarray,
    *,
    max_points: int,
) -> None:
    finite = np.flatnonzero(
        np.isfinite(lon) & np.isfinite(lat) & np.isfinite(velocity)
    )
    if finite.size == 0:
        raise PostprocessError("No finite points for KML export")

    max_points = max(1, int(max_points))
    if finite.size > max_points:
        # Preserve extremes and add an evenly spaced background sample.
        n_extreme = min(max_points // 2, 5000)
        order = finite[np.argsort(np.abs(velocity[finite]))[::-1]]
        extremes = order[:n_extreme]
        remaining = max_points - extremes.size
        stride_sample = finite[
            np.linspace(0, finite.size - 1, remaining, dtype=np.int64)
        ]
        selected = np.unique(np.concatenate((extremes, stride_sample)))[:max_points]
    else:
        selected = finite

    clip = np.nanpercentile(velocity[finite], [2.0, 98.0])
    vmin, vmax = float(clip[0]), float(clip[1])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        handle.write("<name>pySTAMPS PS velocity sample</name>\n")
        for idx in selected:
            color = _kml_color(float(velocity[idx]), vmin, vmax)
            description = (
                f"PS ID: {idx + 1}<br/>"
                f"LOS velocity: {velocity[idx]:.3f} mm/yr<br/>"
                f"Temporal RMS: {rms_mm[idx]:.3f} mm<br/>"
                "Positive LOS is toward satellite."
            )
            handle.write("<Placemark>\n")
            handle.write(f"<name>PS_{idx + 1}</name>\n")
            handle.write(
                "<Style><IconStyle>"
                f"<color>{color}</color><scale>0.45</scale>"
                "<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>"
                "</IconStyle></Style>\n"
            )
            handle.write(f"<description><![CDATA[{description}]]></description>\n")
            handle.write(
                f"<Point><coordinates>{lon[idx]:.9f},{lat[idx]:.9f},0"
                "</coordinates></Point>\n"
            )
            handle.write("</Placemark>\n")
        handle.write("</Document></kml>\n")


def write_timeseries_h5(
    path: Path,
    phase: np.ndarray,
    day: np.ndarray,
    labels: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    wavelength_m: float,
    reference_indices: np.ndarray,
    velocity_reference_mm_yr: float,
    chunk_ps: int,
    include_scn: bool,
    scn_phase: np.ndarray | None,
) -> dict[str, Any]:
    h5py = _require_import("h5py", "h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ps, n_epoch = phase.shape
    phase_scale_mm = -float(wavelength_m) / (4.0 * math.pi) * 1000.0

    reference_displacement = np.zeros(n_epoch, dtype=np.float64)
    if reference_indices.size:
        # Reference regions are normally small. Use direct median.
        reference_displacement = np.nanmedian(
            np.asarray(phase[reference_indices, :], dtype=np.float64)
            * phase_scale_mm,
            axis=0,
        )
        reference_displacement[~np.isfinite(reference_displacement)] = 0.0

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "pySTAMPS_SBAS_geocoded_point_timeseries"
        h5.attrs["los_sign"] = "positive_toward_satellite"
        h5.attrs["wavelength_m"] = float(wavelength_m)
        h5.attrs["phase_to_displacement_mm"] = float(phase_scale_mm)
        h5.attrs["extra_velocity_reference_mm_yr"] = float(
            velocity_reference_mm_yr
        )

        h5.create_dataset("lon", data=lon.astype(np.float64), compression="gzip")
        h5.create_dataset("lat", data=lat.astype(np.float64), compression="gzip")
        h5.create_dataset(
            "day",
            data=np.asarray(day, dtype=np.float64),
            compression="gzip",
        )
        h5.create_dataset(
            "date",
            data=np.asarray(labels, dtype="S8"),
            compression="gzip",
        )
        h5.create_dataset(
            "reference_displacement_mm",
            data=reference_displacement.astype(np.float32),
        )

        dset = h5.create_dataset(
            "displacement_mm",
            shape=(n_ps, n_epoch),
            dtype=np.float32,
            chunks=(min(chunk_ps, n_ps), min(32, n_epoch)),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.nan,
        )
        scn_dset = None
        if include_scn and scn_phase is not None:
            scn_dset = h5.create_dataset(
                "spatial_noise_mm",
                shape=(n_ps, n_epoch),
                dtype=np.float32,
                chunks=(min(chunk_ps, n_ps), min(32, n_epoch)),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=np.nan,
            )

        for start in range(0, n_ps, chunk_ps):
            stop = min(start + chunk_ps, n_ps)
            block = (
                np.asarray(phase[start:stop, :], dtype=np.float64)
                * phase_scale_mm
                - reference_displacement[None, :]
            )
            dset[start:stop, :] = block.astype(np.float32)
            if scn_dset is not None and scn_phase is not None:
                scn_block = (
                    np.asarray(scn_phase[start:stop, :], dtype=np.float64)
                    * phase_scale_mm
                )
                scn_dset[start:stop, :] = scn_block.astype(np.float32)
            print(
                f"[POST][HDF5] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )

    return {
        "path": str(path),
        "n_ps": int(n_ps),
        "n_epoch": int(n_epoch),
        "reference_ps": int(reference_indices.size),
    }


def load_h5_displacement(path: Path):
    h5py = _require_import("h5py", "h5py")
    return h5py.File(path, "r")


def write_epoch_geotiffs(
    h5_path: Path,
    out_dir: Path,
    grid: RasterGrid,
    labels: list[str],
    *,
    epoch_step: int,
    epoch_start: int,
    epoch_end: int | None,
    min_points: int,
    write_wgs84: bool,
) -> list[str]:
    created: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with load_h5_displacement(h5_path) as h5:
        disp = h5["displacement_mm"]
        n_epoch = disp.shape[1]
        start = max(0, int(epoch_start))
        end = n_epoch if epoch_end is None else min(n_epoch, int(epoch_end))
        step = max(1, int(epoch_step))
        for index in range(start, end, step):
            values = np.asarray(disp[:, index], dtype=np.float32)
            raster, _counts = aggregate_mean(values, grid, min_points=min_points)
            path = out_dir / f"geo_{labels[index]}.tif"
            write_geotiff(
                path,
                raster,
                grid,
                nodata=-9999.0,
                dtype="float32",
                description=f"LOS displacement {labels[index]}",
                unit="mm",
                tags={
                    "date": labels[index],
                    "epoch_index_0based": index,
                },
            )
            created.append(str(path))
            if write_wgs84:
                wgs_path = out_dir / "wgs84" / f"geo_{labels[index]}_wgs84.tif"
                reproject_to_wgs84(path, wgs_path)
                created.append(str(wgs_path))
            print(
                f"[POST][EPOCH_TIF] {index + 1}/{n_epoch} {labels[index]}",
                flush=True,
            )
    return created


def robust_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if vmax <= vmin:
        delta = max(1.0, abs(float(vmin)) * 0.1)
        return float(vmin - delta), float(vmax + delta)
    return float(vmin), float(vmax)


def plot_raster(
    path: Path,
    array: np.ndarray,
    grid: RasterGrid,
    title: str,
    label: str,
    *,
    cmap: str,
    symmetric: bool,
    dpi: int,
) -> None:
    matplotlib = _require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(array, dtype=np.float32)
    if symmetric:
        finite = arr[np.isfinite(arr)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = max(vmax, 1.0e-9)
        vmin = -vmax
    else:
        vmin, vmax = robust_limits(arr)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        arr,
        extent=(grid.xmin, grid.xmax, grid.ymin, grid.ymax),
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(f"Easting (m), EPSG:{grid.epsg}")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_velocity_histogram(
    path: Path,
    velocity: np.ndarray,
    dpi: int,
) -> None:
    matplotlib = _require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = np.asarray(velocity, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    vmin, vmax = robust_limits(finite, 0.5, 99.5)
    shown = finite[(finite >= vmin) & (finite <= vmax)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(shown, bins=120)
    ax.axvline(float(np.nanmedian(finite)), linestyle="--", linewidth=1.5)
    ax.set_title("PS LOS velocity distribution")
    ax.set_xlabel("LOS velocity (mm/yr; positive toward satellite)")
    ax.set_ylabel("PS count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def select_representative_points(
    velocity: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    finite = np.flatnonzero(
        np.isfinite(velocity) & np.isfinite(lon) & np.isfinite(lat)
    )
    if finite.size == 0:
        return np.asarray([], dtype=np.int64), []
    values = velocity[finite]
    min_ix = int(finite[np.argmin(values)])
    max_ix = int(finite[np.argmax(values)])
    median_value = float(np.nanmedian(values))
    med_ix = int(finite[np.argmin(np.abs(values - median_value))])
    lon0 = float(np.nanmedian(lon[finite]))
    lat0 = float(np.nanmedian(lat[finite]))
    centre_ix = int(
        finite[np.argmin((lon[finite] - lon0) ** 2 + (lat[finite] - lat0) ** 2)]
    )
    indices = np.asarray([min_ix, max_ix, med_ix, centre_ix], dtype=np.int64)
    labels = ["maximum_subsidence", "maximum_uplift", "median_velocity", "scene_centre"]
    unique: list[int] = []
    unique_labels: list[str] = []
    for idx, label in zip(indices.tolist(), labels):
        if idx not in unique:
            unique.append(idx)
            unique_labels.append(label)
    return np.asarray(unique, dtype=np.int64), unique_labels


def plot_representative_timeseries(
    path: Path,
    h5_path: Path,
    velocity: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dates: list[datetime],
    dpi: int,
) -> None:
    matplotlib = _require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices, labels = select_representative_points(velocity, lon, lat)
    if indices.size == 0:
        return
    with load_h5_displacement(h5_path) as h5:
        values = np.asarray(h5["displacement_mm"][indices, :], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for row, idx, label in zip(values, indices, labels):
        ax.plot(
            dates,
            row,
            linewidth=1.2,
            label=(
                f"{label}: PS {idx + 1}, "
                f"{velocity[idx]:.1f} mm/yr"
            ),
        )
    ax.set_title("Representative LOS displacement time series")
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm; positive toward satellite)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_epoch_valid_counts(
    path: Path,
    h5_path: Path,
    dates: list[datetime],
    dpi: int,
    chunk_ps: int,
) -> np.ndarray:
    matplotlib = _require_import("matplotlib", "matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with load_h5_displacement(h5_path) as h5:
        disp = h5["displacement_mm"]
        counts = np.zeros(disp.shape[1], dtype=np.int64)
        for start in range(0, disp.shape[0], chunk_ps):
            stop = min(start + chunk_ps, disp.shape[0])
            counts += np.sum(
                np.isfinite(np.asarray(disp[start:stop, :], dtype=np.float32)),
                axis=0,
            )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(dates, counts, linewidth=1.3)
    ax.set_title("Valid PS observations by acquisition")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valid PS count")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return counts


def parse_formats(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in str(text).split(",")
        if token.strip()
    }


def file_record(path: Path, category: str) -> dict[str, Any]:
    return {
        "category": category,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "path", "exists", "size_bytes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def inspect_inputs(
    dataset: Path,
    read_mat: Any,
    read_mat_variables: Any,
) -> dict[str, Any]:
    required = [
        "ps2.mat",
        "mean_v.mat",
        "scla_smooth2.mat",
        "phuw_sm2.mat",
        "uw_space_time.mat",
    ]
    missing = [name for name in required if not (dataset / name).exists()]
    if missing:
        raise PostprocessError(
            "Missing Stage 7/8 outputs: " + ", ".join(missing)
        )

    ps = read_mat_variables(
        dataset / "ps2.mat",
        ("n_ps", "n_image", "n_ifg", "lonlat", "day", "xy"),
    )
    n_ps = int(round(_scalar(ps.get("n_ps"), 0)))
    if n_ps <= 0:
        raise PostprocessError("ps2.mat contains invalid n_ps")
    lonlat = _matrix(ps["lonlat"], n_ps, "ps2.lonlat", np.float64)
    if lonlat.shape[1] < 2:
        raise PostprocessError("ps2.lonlat needs at least two columns")

    mean = read_mat_variables(
        dataset / "mean_v.mat",
        (
            "velocity_mm_yr",
            "temporal_rms_rad",
            "phase_rate_rad_day",
            "phase_to_los_sign",
        ),
    )
    velocity = _vector(mean["velocity_mm_yr"], "mean_v.velocity_mm_yr", np.float32)
    if velocity.size != n_ps:
        raise PostprocessError(
            f"Velocity length {velocity.size}; expected {n_ps}"
        )

    scla = read_mat_variables(
        dataset / "scla_smooth2.mat",
        ("K_ps_uw", "C_ps_uw", "day"),
    )
    k = _vector(scla["K_ps_uw"], "scla_smooth2.K_ps_uw", np.float32)
    c = _vector(scla["C_ps_uw"], "scla_smooth2.C_ps_uw", np.float32)
    if k.size != n_ps or c.size != n_ps:
        raise PostprocessError("SCLA vector lengths do not match n_ps")

    day = np.asarray(
        scla.get("day", ps.get("day")),
        dtype=np.float64,
    ).reshape(-1)
    n_image = int(day.size)

    stage7_debug = {}
    stage8_debug = {}
    for name, target in (
        ("stage7_sbas_debug.json", stage7_debug),
        ("stage8_sbas_debug.json", stage8_debug),
    ):
        path = dataset / name
        if path.exists():
            target.update(json.loads(path.read_text(encoding="utf-8")))

    return {
        "n_ps": n_ps,
        "n_image": n_image,
        "n_ifg": int(round(_scalar(ps.get("n_ifg"), 0))),
        "lon": lonlat[:, 0].astype(np.float64),
        "lat": lonlat[:, 1].astype(np.float64),
        "velocity": velocity.astype(np.float32),
        "temporal_rms_rad": np.asarray(
            mean.get("temporal_rms_rad", np.full(n_ps, np.nan)),
            dtype=np.float32,
        ).reshape(-1),
        "scla_k": k.astype(np.float32),
        "scla_c": c.astype(np.float32),
        "day": day,
        "stage7_debug": stage7_debug,
        "stage8_debug": stage8_debug,
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset = Path(args.dataset).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_root = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dataset / "postprocess"
    )

    if not dataset.exists():
        raise PostprocessError(f"Dataset does not exist: {dataset}")

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    read_mat, read_mat_variables = _load_pystamps_io(repo_root)
    info = inspect_inputs(dataset, read_mat, read_mat_variables)

    n_ps = int(info["n_ps"])
    n_image = int(info["n_image"])
    lon = np.asarray(info["lon"], dtype=np.float64)
    lat = np.asarray(info["lat"], dtype=np.float64)
    velocity = np.asarray(info["velocity"], dtype=np.float32)
    temporal_rms_rad = np.asarray(info["temporal_rms_rad"], dtype=np.float32)
    scla_k = np.asarray(info["scla_k"], dtype=np.float32)
    scla_c = np.asarray(info["scla_c"], dtype=np.float32)
    day = np.asarray(info["day"], dtype=np.float64)

    parms = read_mat(dataset / "parms.mat") if (dataset / "parms.mat").exists() else {}
    wavelength_m = float(_scalar(parms.get("lambda"), args.wavelength_m))
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise PostprocessError("Invalid radar wavelength")
    phase_to_mm_abs = wavelength_m / (4.0 * math.pi) * 1000.0
    temporal_rms_mm = temporal_rms_rad.astype(np.float64) * phase_to_mm_abs

    grid = build_grid(
        lon,
        lat,
        resolution_m=args.resolution_m,
        target_epsg=args.target_epsg,
    )

    reference_ix = select_reference_indices(
        args.reference_mode,
        lon,
        lat,
        grid,
        ref_lon=args.ref_lon,
        ref_lat=args.ref_lat,
        ref_radius_m=args.ref_radius_m,
        ref_bbox=args.ref_bbox,
    )

    extra_velocity_reference = 0.0
    if reference_ix.size:
        extra_velocity_reference = float(
            np.nanmedian(velocity[reference_ix])
        )
        if not math.isfinite(extra_velocity_reference):
            extra_velocity_reference = 0.0
        velocity = (velocity.astype(np.float64) - extra_velocity_reference).astype(
            np.float32
        )

    labels, dates = day_to_date_labels(day)
    valid_epoch_count = np.full(n_ps, n_image, dtype=np.int32)

    columns = {
        "ps_id": np.arange(1, n_ps + 1, dtype=np.int64),
        "lon": lon,
        "lat": lat,
        "velocity_mm_yr": velocity,
        "temporal_rms_mm": temporal_rms_mm.astype(np.float32),
        "scla_k_rad_per_m": scla_k,
        "scla_c_rad": scla_c,
        "valid_epoch_count": valid_epoch_count,
    }

    records: list[dict[str, Any]] = []
    formats = parse_formats(args.vector_formats)
    points_dir = out_root / "points"
    rasters_dir = out_root / "rasters"
    plots_dir = out_root / "plots"
    timeseries_dir = out_root / "timeseries"
    geo_ts_dir = out_root / "geo_timeseries"

    if "csv" in formats:
        csv_path = points_dir / "ps_velocity.csv"
        write_points_csv(csv_path, columns)
        records.append(file_record(csv_path, "point_csv"))

    if "parquet" in formats:
        parquet_path = points_dir / "ps_velocity.parquet"
        if write_points_parquet(parquet_path, columns):
            records.append(file_record(parquet_path, "point_parquet"))

    if "gpkg" in formats:
        gpkg_path = points_dir / "ps_velocity.gpkg"
        if write_points_gpkg(gpkg_path, columns):
            records.append(file_record(gpkg_path, "point_gpkg"))

    if "shp" in formats:
        shp_path = points_dir / "ps_velocity.shp"
        if write_points_shp(shp_path, columns):
            records.append(file_record(shp_path, "point_shapefile"))

    if "kml" in formats:
        kml_path = points_dir / "ps_velocity_sample.kml"
        write_points_kml(
            kml_path,
            lon,
            lat,
            velocity,
            temporal_rms_mm,
            max_points=args.kml_max_points,
        )
        records.append(file_record(kml_path, "point_kml_sample"))

    metrics = [
        (
            "geo_velocity.tif",
            velocity,
            "LOS velocity",
            "mm/yr",
            "velocity",
        ),
        (
            "geo_temporal_rms_mm.tif",
            temporal_rms_mm,
            "Temporal residual RMS",
            "mm",
            "rms",
        ),
        (
            "geo_scla_k_rad_per_m.tif",
            scla_k,
            "SCLA coefficient K",
            "rad/m",
            "scla_k",
        ),
        (
            "geo_scla_c_rad.tif",
            scla_c,
            "SCLA constant C",
            "rad",
            "scla_c",
        ),
    ]

    raster_arrays: dict[str, np.ndarray] = {}
    count_array: np.ndarray | None = None
    for filename, values, description, unit, key in metrics:
        raster, counts = aggregate_mean(
            np.asarray(values),
            grid,
            min_points=args.min_points_per_cell,
        )
        raster_arrays[key] = raster
        if count_array is None:
            count_array = counts
        path = rasters_dir / filename
        write_geotiff(
            path,
            raster,
            grid,
            nodata=-9999.0,
            dtype="float32",
            description=description,
            unit=unit,
            tags={
                "reference_mode": args.reference_mode,
                "extra_velocity_reference_mm_yr": extra_velocity_reference,
            },
        )
        records.append(file_record(path, "raster"))
        if args.wgs84_copy:
            wgs_path = rasters_dir / "wgs84" / filename.replace(
                ".tif", "_wgs84.tif"
            )
            reproject_to_wgs84(path, wgs_path)
            records.append(file_record(wgs_path, "raster_wgs84"))

    if count_array is None:
        count_array = np.zeros((grid.height, grid.width), dtype=np.uint32)
    count_path = rasters_dir / "geo_ps_count.tif"
    write_geotiff(
        count_path,
        count_array,
        grid,
        nodata=0,
        dtype="uint32",
        description="PS count per raster cell",
        unit="count",
    )
    records.append(file_record(count_path, "raster"))

    plot_raster(
        plots_dir / "01_velocity_map.png",
        raster_arrays["velocity"],
        grid,
        "LOS velocity",
        "mm/yr; positive toward satellite",
        cmap="RdBu_r",
        symmetric=True,
        dpi=args.plot_dpi,
    )
    plot_raster(
        plots_dir / "02_temporal_rms_map.png",
        raster_arrays["rms"],
        grid,
        "Temporal residual RMS",
        "mm",
        cmap="viridis",
        symmetric=False,
        dpi=args.plot_dpi,
    )
    plot_raster(
        plots_dir / "03_scla_k_map.png",
        raster_arrays["scla_k"],
        grid,
        "SCLA coefficient K",
        "rad/m",
        cmap="RdBu_r",
        symmetric=True,
        dpi=args.plot_dpi,
    )
    plot_velocity_histogram(
        plots_dir / "04_velocity_histogram.png",
        velocity,
        args.plot_dpi,
    )
    for path in sorted(plots_dir.glob("*.png")):
        records.append(file_record(path, "plot"))

    h5_path = timeseries_dir / "ps_timeseries.h5"
    phase = None
    scn_phase = None
    if args.write_hdf5 or args.epoch_tifs or args.plot_timeseries:
        uw = read_mat_variables(
            dataset / "uw_space_time.mat",
            ("ph_uw_ts", "ph_scn", "day"),
        )
        phase = _matrix(
            uw["ph_uw_ts"],
            n_ps,
            "uw_space_time.ph_uw_ts",
            np.float32,
        )
        if phase.shape[1] != n_image:
            raise PostprocessError(
                f"ph_uw_ts has {phase.shape[1]} epochs; expected {n_image}"
            )
        if args.include_scn_hdf5 and "ph_scn" in uw:
            scn_phase = _matrix(
                uw["ph_scn"],
                n_ps,
                "uw_space_time.ph_scn",
                np.float32,
            )

        write_timeseries_h5(
            h5_path,
            phase,
            day,
            labels,
            lon,
            lat,
            wavelength_m=wavelength_m,
            reference_indices=reference_ix,
            velocity_reference_mm_yr=extra_velocity_reference,
            chunk_ps=args.chunk_ps,
            include_scn=args.include_scn_hdf5,
            scn_phase=scn_phase,
        )
        records.append(file_record(h5_path, "timeseries_hdf5"))

    if args.epoch_tifs:
        created = write_epoch_geotiffs(
            h5_path,
            geo_ts_dir,
            grid,
            labels,
            epoch_step=args.epoch_step,
            epoch_start=args.epoch_start,
            epoch_end=args.epoch_end,
            min_points=args.min_points_per_cell,
            write_wgs84=args.wgs84_epoch_copy,
        )
        for name in created:
            records.append(file_record(Path(name), "epoch_raster"))

    if args.plot_timeseries:
        plot_representative_timeseries(
            plots_dir / "05_representative_timeseries.png",
            h5_path,
            velocity,
            lon,
            lat,
            dates,
            args.plot_dpi,
        )
        counts = plot_epoch_valid_counts(
            plots_dir / "06_valid_ps_by_epoch.png",
            h5_path,
            dates,
            args.plot_dpi,
            args.chunk_ps,
        )
        columns["valid_epoch_count"] = np.full(
            n_ps,
            int(np.nanmedian(counts)),
            dtype=np.int32,
        )
        for path in (
            plots_dir / "05_representative_timeseries.png",
            plots_dir / "06_valid_ps_by_epoch.png",
        ):
            if path.exists():
                records.append(file_record(path, "plot"))

    velocity_finite = velocity[np.isfinite(velocity)]
    report = {
        "status": "completed",
        "dataset": str(dataset),
        "output": str(out_root),
        "n_ps": n_ps,
        "n_image": n_image,
        "n_ifg": int(info["n_ifg"]),
        "date_start": labels[0] if labels else None,
        "date_end": labels[-1] if labels else None,
        "wavelength_m": wavelength_m,
        "los_sign": "positive_toward_satellite",
        "reference_mode": args.reference_mode,
        "reference_ps": int(reference_ix.size),
        "extra_velocity_reference_mm_yr": extra_velocity_reference,
        "target_epsg": grid.epsg,
        "resolution_m": grid.resolution,
        "raster_width": grid.width,
        "raster_height": grid.height,
        "velocity_statistics_mm_yr": {
            "count": int(velocity_finite.size),
            "min": float(np.nanmin(velocity_finite)) if velocity_finite.size else None,
            "p02": float(np.nanpercentile(velocity_finite, 2)) if velocity_finite.size else None,
            "median": float(np.nanmedian(velocity_finite)) if velocity_finite.size else None,
            "mean": float(np.nanmean(velocity_finite)) if velocity_finite.size else None,
            "p98": float(np.nanpercentile(velocity_finite, 98)) if velocity_finite.size else None,
            "max": float(np.nanmax(velocity_finite)) if velocity_finite.size else None,
        },
        "stage7_debug": info["stage7_debug"],
        "stage8_debug": info["stage8_debug"],
        "outputs": records,
        "duration_sec": time.perf_counter() - started,
        "note": (
            "Point geocoding uses ps2.lonlat. Raster products are cell means "
            "of PS measurements and do not introduce interpolation."
        ),
    }
    report_path = out_root / "postprocess_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    records.append(file_record(report_path, "report"))

    manifest_path = out_root / "postprocess_manifest.csv"
    write_manifest(manifest_path, records)

    print("\n============================================================")
    print("Post-processing completed")
    print("============================================================")
    print(f"Dataset     : {dataset}")
    print(f"Output      : {out_root}")
    print(f"PS          : {n_ps}")
    print(f"Epochs      : {n_image}")
    print(f"Raster CRS  : EPSG:{grid.epsg}")
    print(f"Resolution  : {grid.resolution:.2f} m")
    print(f"Reference   : {args.reference_mode} ({reference_ix.size} PS)")
    print(f"Report      : {report_path}")
    print(f"Manifest    : {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Geocode, rasterize, plot and export pySTAMPS SBAS Stage 7/8 outputs."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="pySTAMPS dataset root",
    )
    parser.add_argument(
        "--repo-root",
        default="/home/ubuntu/software/pystamps-main",
        help="pystamps source checkout",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory; default: DATASET/postprocess",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=50.0,
        help="Projected raster cell size in metres",
    )
    parser.add_argument(
        "--target-epsg",
        type=int,
        default=None,
        help="Projected EPSG; default: automatic local UTM",
    )
    parser.add_argument(
        "--min-points-per-cell",
        type=int,
        default=1,
        help="Minimum PS count required for a valid raster cell",
    )
    parser.add_argument(
        "--vector-formats",
        default="csv,gpkg,parquet,kml",
        help="Comma-separated: csv,gpkg,parquet,shp,kml",
    )
    parser.add_argument(
        "--kml-max-points",
        type=int,
        default=20000,
        help="Maximum sampled KML points",
    )
    parser.add_argument(
        "--wgs84-copy",
        action="store_true",
        help="Also reproject summary rasters to EPSG:4326",
    )
    parser.add_argument(
        "--write-hdf5",
        action="store_true",
        help="Write compressed geocoded point time-series HDF5",
    )
    parser.add_argument(
        "--include-scn-hdf5",
        action="store_true",
        help="Include spatial-noise phase converted to mm in HDF5",
    )
    parser.add_argument(
        "--epoch-tifs",
        action="store_true",
        help="Write acquisition displacement GeoTIFFs",
    )
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Write every Nth acquisition",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=0,
        help="First 0-based acquisition index",
    )
    parser.add_argument(
        "--epoch-end",
        type=int,
        default=None,
        help="Exclusive ending acquisition index",
    )
    parser.add_argument(
        "--wgs84-epoch-copy",
        action="store_true",
        help="Also create EPSG:4326 copies of epoch rasters",
    )
    parser.add_argument(
        "--plot-timeseries",
        action="store_true",
        help="Plot representative time series and valid-observation counts",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--chunk-ps",
        type=int,
        default=4096,
        help="Point chunk size for HDF5 conversion",
    )
    parser.add_argument(
        "--wavelength-m",
        type=float,
        default=0.0555,
        help="Fallback radar wavelength",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("existing", "none", "global-median", "point", "bbox"),
        default="existing",
        help=(
            "Extra reference applied after Stage 8. 'existing' preserves the "
            "Stage-8 reference."
        ),
    )
    parser.add_argument("--ref-lon", type=float, default=None)
    parser.add_argument("--ref-lat", type=float, default=None)
    parser.add_argument("--ref-radius-m", type=float, default=500.0)
    parser.add_argument(
        "--ref-bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=None,
        help="Reference bounding box in longitude/latitude",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
