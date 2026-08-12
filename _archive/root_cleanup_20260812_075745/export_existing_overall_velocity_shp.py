#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the existing full-period robust GLS velocity as point Shapefiles."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class OverallExportError(RuntimeError):
    pass


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise OverallExportError(
            f"缺少Python包：{name}；请安装：python -m pip install {pip_name or name}"
        ) from exc


def clean_shp(path: Path) -> None:
    for suffix in (
        ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"
    ):
        item = path.with_suffix(suffix)
        if item.exists():
            item.unlink()


def write_shp(gdf: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_shp(path)
    gdf.to_file(path, driver="ESRI Shapefile", encoding="UTF-8", index=False)
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def read_vector(h5: Any, name: str, n: int, default: Any) -> np.ndarray:
    if name not in h5:
        if np.isscalar(default):
            return np.full(n, default)
        return np.asarray(default)
    values = np.asarray(h5[name][:]).reshape(-1)
    if values.size != n:
        raise OverallExportError(
            f"HDF5变量{name}长度为{values.size}，应为{n}"
        )
    return values


def find_gpkg(dataset: Path) -> Path | None:
    candidates = [
        dataset / "best_velocity_gacos" / "points_original_distribution" / "ps_velocity_all_points.gpkg",
        dataset / "best_velocity_gacos" / "points_original_distribution" / "ps_velocity_recommended.gpkg",
    ]
    return next((path for path in candidates if path.exists()), None)


def export_from_gpkg(source: Path, output: Path, epsg: int | None) -> dict[str, Any]:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")
    gdf = geopandas.read_file(source)
    if gdf.crs is None:
        raise OverallExportError(f"GeoPackage没有CRS：{source}")
    if epsg is not None:
        gdf = gdf.to_crs(f"EPSG:{epsg}")

    lookup = {name.lower(): name for name in gdf.columns}
    def col(*names: str) -> str | None:
        return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)

    v_name = col("vel_mm_yr", "velocity_mm_yr", "velocity")
    if v_name is None:
        raise OverallExportError("整体点GeoPackage中找不到速度字段")
    se_name = col("vel_std", "velocity_std_mm_yr", "vel_se")
    rmse_name = col("rmse_mm", "rmse")
    fit_name = col("fit_ok", "accepted")
    rec_name = col("recommended", "q_recom")
    strict_name = col("strict_qc", "strict", "q_strict")
    id_name = col("ps_id", "id")

    velocity = pandas.to_numeric(gdf[v_name], errors="coerce").to_numpy(float)
    se = (
        pandas.to_numeric(gdf[se_name], errors="coerce").to_numpy(float)
        if se_name else np.full(len(gdf), np.nan)
    )
    rmse = (
        pandas.to_numeric(gdf[rmse_name], errors="coerce").to_numpy(float)
        if rmse_name else np.full(len(gdf), np.nan)
    )
    fit = (
        gdf[fit_name].fillna(0).astype(int).to_numpy() == 1
        if fit_name else np.isfinite(velocity)
    )
    mask = fit & np.isfinite(velocity)
    source_rows = gdf.loc[mask].copy()
    velocity = velocity[mask]
    se = se[mask]
    rmse = rmse[mask]
    recommended = (
        source_rows[rec_name].fillna(0).astype(int).to_numpy() == 1
        if rec_name
        else (
            np.isfinite(se) & (se <= 3.0) & np.isfinite(rmse) & (rmse <= 15.0)
        )
    )
    strict = (
        source_rows[strict_name].fillna(0).astype(int).to_numpy() == 1
        if strict_name else recommended.copy()
    )
    ci_low = velocity - 1.96 * se
    ci_high = velocity + 1.96 * se

    result = geopandas.GeoDataFrame(
        {
            "PS_ID": (
                source_rows[id_name].to_numpy(np.int64)
                if id_name else np.arange(1, len(source_rows) + 1, dtype=np.int64)
            ),
            "VEL_MM_YR": velocity.astype(np.float32),
            "VEL_SE": se.astype(np.float32),
            "CI95_LO": ci_low.astype(np.float32),
            "CI95_HI": ci_high.astype(np.float32),
            "RMSE_MM": rmse.astype(np.float32),
            "Q_RECOM": recommended.astype(np.int16),
            "Q_STRICT": strict.astype(np.int16),
        },
        geometry=source_rows.geometry.to_numpy(),
        crs=source_rows.crs,
    )
    all_path = output / "overall_velocity_allfit.shp"
    rec_path = output / "overall_velocity_recommended.shp"
    write_shp(result, all_path)
    write_shp(result.loc[result["Q_RECOM"] == 1].copy(), rec_path)
    return {
        "source": str(source),
        "allfit": str(all_path),
        "recommended": str(rec_path),
        "allfit_points": int(len(result)),
        "recommended_points": int(np.count_nonzero(recommended)),
    }


def export_from_h5(source: Path, output: Path, epsg: int) -> dict[str, Any]:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")
    h5py = require_import("h5py", "h5py")
    with h5py.File(source, "r") as h5:
        lon = np.asarray(h5["lon"][:]).reshape(-1).astype(float)
        lat = np.asarray(h5["lat"][:]).reshape(-1).astype(float)
        n = lon.size
        velocity = read_vector(h5, "velocity_mm_yr", n, np.nan).astype(float)
        se = read_vector(h5, "velocity_std_mm_yr", n, np.nan).astype(float)
        rmse = read_vector(h5, "rmse_mm", n, np.nan).astype(float)
        accepted = read_vector(h5, "accepted", n, 1).astype(bool)
        strict = read_vector(h5, "best_quality_mask", n, 0).astype(bool)
    fit = accepted & np.isfinite(lon) & np.isfinite(lat) & np.isfinite(velocity)
    recommended = fit & np.isfinite(se) & (se <= 3.0) & np.isfinite(rmse) & (rmse <= 15.0)
    frame = pandas.DataFrame(
        {
            "PS_ID": np.arange(1, n + 1, dtype=np.int64)[fit],
            "VEL_MM_YR": velocity[fit].astype(np.float32),
            "VEL_SE": se[fit].astype(np.float32),
            "CI95_LO": (velocity[fit] - 1.96 * se[fit]).astype(np.float32),
            "CI95_HI": (velocity[fit] + 1.96 * se[fit]).astype(np.float32),
            "RMSE_MM": rmse[fit].astype(np.float32),
            "Q_RECOM": recommended[fit].astype(np.int16),
            "Q_STRICT": strict[fit].astype(np.int16),
            "lon": lon[fit],
            "lat": lat[fit],
        }
    )
    geometry = geopandas.points_from_xy(frame["lon"], frame["lat"])
    result = geopandas.GeoDataFrame(
        frame.drop(columns=["lon", "lat"]), geometry=geometry, crs="EPSG:4326"
    ).to_crs(f"EPSG:{epsg}")
    all_path = output / "overall_velocity_allfit.shp"
    rec_path = output / "overall_velocity_recommended.shp"
    write_shp(result, all_path)
    write_shp(result.loc[result["Q_RECOM"] == 1].copy(), rec_path)
    return {
        "source": str(source),
        "allfit": str(all_path),
        "recommended": str(rec_path),
        "allfit_points": int(len(result)),
        "recommended_points": int(np.count_nonzero(result["Q_RECOM"])),
    }


def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.out or dataset / "overall_velocity_shapefiles").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpkg = Path(args.source_gpkg).expanduser().resolve() if args.source_gpkg else find_gpkg(dataset)
    if gpkg is not None:
        report = export_from_gpkg(gpkg, output, args.target_epsg)
    else:
        h5 = Path(
            args.source_h5
            or dataset / "best_velocity_gacos" / "points" / "best_velocity_points.h5"
        ).expanduser().resolve()
        if not h5.exists():
            raise OverallExportError(f"找不到整体速率点数据：{h5}")
        report = export_from_h5(h5, output, int(args.target_epsg or 32650))
    (output / "overall_velocity_shp_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export existing full-period velocity point SHP")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--source-gpkg", default=None)
    p.add_argument("--source-h5", default=None)
    p.add_argument("--target-epsg", type=int, default=None)
    return p


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
