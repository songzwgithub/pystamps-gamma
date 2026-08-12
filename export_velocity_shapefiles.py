#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export joint annual velocities and the previous full-period velocity to ESRI
Shapefile while preserving the original irregular PS point distribution.

Annual velocities are read from:
    joint_piecewise_seasonal_velocity.gpkg

The joint HDF5 file is deliberately NOT used for annual SHP export.

Full-period velocity source priority:
    1. existing point GeoPackage from points_original_distribution
    2. best_velocity_gacos/points/best_velocity_points.h5

Annual output contains every finite, globally fit-valid PS for that year.
Quality flags are attributes; points are not discarded by recommended/strict
filters.

LOS convention:
    positive = toward satellite
    negative = away from satellite
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


class ExportError(RuntimeError):
    pass


def require_import(name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise ExportError(
            f"缺少Python包：{name}\n"
            f"请在当前stamps环境安装：python -m pip install {pip_name or name}\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc


def clean_shapefile(path: Path) -> None:
    for suffix in (
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".fix",
        ".sbn",
        ".sbx",
    ):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def write_shapefile(gdf: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_shapefile(path)

    # ESRI Shapefile field names are limited to 10 characters. All exported
    # field names are explicitly kept within this limit.
    gdf.to_file(
        path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False,
    )
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def as_int(values: Any, dtype=np.int16) -> np.ndarray:
    array = np.asarray(values)
    output = np.zeros(array.shape, dtype=dtype)
    finite = np.isfinite(array.astype(np.float64, copy=False))
    output[finite] = array[finite].astype(dtype)
    return output


def read_summary(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    pandas = require_import("pandas", "pandas")
    frame = pandas.read_csv(path)
    output: dict[int, dict[str, Any]] = {}
    if "year" not in frame.columns:
        return output
    for _, row in frame.iterrows():
        year = int(row["year"])
        output[year] = {
            key: (
                None
                if pandas.isna(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in row.to_dict().items()
        }
    return output


def normalize_bool(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"b", "i", "u", "f"}:
        return (np.nan_to_num(array, nan=0.0) != 0).astype(np.int16)
    text = np.char.lower(array.astype(str))
    return np.isin(text, ["1", "true", "yes", "y"]).astype(np.int16)


def scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if np.isfinite(value) else False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def export_annual_shapefiles(
    gpkg_path: Path,
    summary_path: Path,
    output_dir: Path,
    *,
    target_epsg: int | None,
) -> list[dict[str, Any]]:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")

    if not gpkg_path.exists():
        raise ExportError(f"找不到联合年度GeoPackage：{gpkg_path}")

    gdf = geopandas.read_file(gpkg_path)
    if gdf.empty:
        raise ExportError(f"联合年度GeoPackage为空：{gpkg_path}")
    if gdf.crs is None:
        raise ExportError(f"联合年度GeoPackage没有CRS：{gpkg_path}")

    if target_epsg is not None:
        gdf = gdf.to_crs(f"EPSG:{int(target_epsg)}")

    velocity_fields = sorted(
        [
            name
            for name in gdf.columns
            if name.startswith("v")
            and name[1:].isdigit()
            and len(name) == 5
        ],
        key=lambda name: int(name[1:]),
    )
    if not velocity_fields:
        raise ExportError(
            "GeoPackage中未发现年度速率字段，例如v2021、v2022"
        )

    summary = read_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for velocity_field in velocity_fields:
        year = int(velocity_field[1:])
        se_field = f"se{year}"
        recommended_field = f"q{year}"
        strict_field = f"s{year}"
        significant_field = f"sg{year}"
        nobs_field = f"n{year}"
        span_field = f"sp{year}"

        required = [
            velocity_field,
            se_field,
            recommended_field,
            strict_field,
            significant_field,
            nobs_field,
            span_field,
            "fit_ok",
            "model_rms",
            "ann_amp",
            "ann_peak",
            "ps_id",
        ]
        missing = [name for name in required if name not in gdf.columns]
        if missing:
            raise ExportError(
                f"{year}年缺少字段：{', '.join(missing)}"
            )

        velocity = pandas.to_numeric(
            gdf[velocity_field],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        velocity_se = pandas.to_numeric(
            gdf[se_field],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        fit_ok = normalize_bool(gdf["fit_ok"].to_numpy()).astype(bool)

        # Preserve all point-distributed annual solutions. Recommended,
        # strict and significance are fields, not deletion conditions.
        export_mask = (
            fit_ok
            & np.isfinite(velocity)
            & np.isfinite(gdf.geometry.x.to_numpy())
            & np.isfinite(gdf.geometry.y.to_numpy())
        )

        source = gdf.loc[export_mask].copy()
        v = velocity[export_mask]
        se = velocity_se[export_mask]
        q_rec = normalize_bool(
            gdf.loc[export_mask, recommended_field].to_numpy()
        )
        q_strict = normalize_bool(
            gdf.loc[export_mask, strict_field].to_numpy()
        )
        q_signif = normalize_bool(
            gdf.loc[export_mask, significant_field].to_numpy()
        )

        year_meta = summary.get(year, {})
        formal_year = int(scalar_bool(year_meta.get("formal_year", False)))
        acquisition_count = int(
            year_meta.get("acquisition_count", 0) or 0
        )

        annual = geopandas.GeoDataFrame(
            {
                "PS_ID": as_int(
                    source["ps_id"].to_numpy(),
                    dtype=np.int64,
                ),
                "YEAR": np.full(
                    len(source),
                    year,
                    dtype=np.int16,
                ),
                "VEL_MM_YR": v.astype(np.float32),
                "VEL_SE": se.astype(np.float32),
                "CI95_LO": (v - 1.96 * se).astype(np.float32),
                "CI95_HI": (v + 1.96 * se).astype(np.float32),
                "Q_RECOM": q_rec,
                "Q_STRICT": q_strict,
                "Q_SIGNIF": q_signif,
                "N_OBS": as_int(
                    source[nobs_field].to_numpy(),
                    dtype=np.int16,
                ),
                "SPAN_DAY": pandas.to_numeric(
                    source[span_field],
                    errors="coerce",
                ).to_numpy(dtype=np.float32),
                "MOD_RMS": pandas.to_numeric(
                    source["model_rms"],
                    errors="coerce",
                ).to_numpy(dtype=np.float32),
                "ANN_AMP": pandas.to_numeric(
                    source["ann_amp"],
                    errors="coerce",
                ).to_numpy(dtype=np.float32),
                "ANN_PEAK": pandas.to_numeric(
                    source["ann_peak"],
                    errors="coerce",
                ).to_numpy(dtype=np.float32),
                "FORMAL_YR": np.full(
                    len(source),
                    formal_year,
                    dtype=np.int16,
                ),
                "YR_N_ACQ": np.full(
                    len(source),
                    acquisition_count,
                    dtype=np.int16,
                ),
            },
            geometry=source.geometry.to_numpy(),
            crs=source.crs,
        )

        shp_path = output_dir / f"joint_velocity_{year}_allfit.shp"
        write_shapefile(annual, shp_path)

        recommended_count = int(np.count_nonzero(q_rec))
        strict_count = int(np.count_nonzero(q_strict))
        records.append(
            {
                "year": year,
                "formal_year": formal_year,
                "acquisition_count": acquisition_count,
                "all_fit_points": len(annual),
                "recommended_points": recommended_count,
                "strict_points": strict_count,
                "velocity_p02_mm_yr": (
                    float(np.nanpercentile(v, 2))
                    if v.size else np.nan
                ),
                "velocity_median_mm_yr": (
                    float(np.nanmedian(v))
                    if v.size else np.nan
                ),
                "velocity_mean_mm_yr": (
                    float(np.nanmean(v))
                    if v.size else np.nan
                ),
                "velocity_p98_mm_yr": (
                    float(np.nanpercentile(v, 98))
                    if v.size else np.nan
                ),
                "shapefile": str(shp_path),
            }
        )

        print(
            f"[YEAR SHP] {year}: all-fit={len(annual):,}, "
            f"recommended={recommended_count:,}, "
            f"strict={strict_count:,} -> {shp_path}",
            flush=True,
        )

    summary_out = output_dir / "annual_shapefile_summary.csv"
    pandas.DataFrame(records).to_csv(
        summary_out,
        index=False,
        encoding="utf-8-sig",
    )
    return records


def find_overall_gpkg(dataset: Path) -> Path | None:
    candidates = [
        dataset
        / "best_velocity_gacos"
        / "points_original_distribution"
        / "ps_velocity_all_points.gpkg",
        dataset
        / "best_velocity_gacos"
        / "points_original_distribution"
        / "ps_velocity_recommended.gpkg",
        dataset
        / "postprocess"
        / "points"
        / "ps_velocity.gpkg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def locate_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def export_overall_from_gpkg(
    source_path: Path,
    output_dir: Path,
    target_epsg: int | None,
) -> dict[str, Any]:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")

    gdf = geopandas.read_file(source_path)
    if gdf.empty:
        raise ExportError(f"整体速率GeoPackage为空：{source_path}")
    if gdf.crs is None:
        raise ExportError(f"整体速率GeoPackage没有CRS：{source_path}")
    if target_epsg is not None:
        gdf = gdf.to_crs(f"EPSG:{int(target_epsg)}")

    columns = list(gdf.columns)
    velocity_name = locate_column(
        columns,
        ["vel_mm_yr", "velocity_mm_yr", "velocity", "vel"],
    )
    se_name = locate_column(
        columns,
        ["vel_std", "velocity_std_mm_yr", "velocity_std", "vel_se"],
    )
    rmse_name = locate_column(
        columns,
        ["rmse_mm", "temporal_rms_mm", "rmse"],
    )
    nobs_name = locate_column(columns, ["n_obs", "nobs"])
    span_name = locate_column(columns, ["span_days", "span_day"])
    eff_name = locate_column(columns, ["effective_n", "eff_n"])
    fit_name = locate_column(columns, ["fit_ok", "accepted"])
    rec_name = locate_column(columns, ["recommended", "q_recom"])
    strict_name = locate_column(columns, ["strict_qc", "strict", "q_strict"])
    reason_name = locate_column(columns, ["reason", "quality_reason_code"])
    id_name = locate_column(columns, ["ps_id", "id"])

    if velocity_name is None:
        raise ExportError(
            f"整体速率GeoPackage中找不到速率字段：{source_path}"
        )

    velocity = pandas.to_numeric(
        gdf[velocity_name],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    velocity_se = (
        pandas.to_numeric(
            gdf[se_name],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if se_name
        else np.full(len(gdf), np.nan)
    )
    rmse = (
        pandas.to_numeric(
            gdf[rmse_name],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if rmse_name
        else np.full(len(gdf), np.nan)
    )
    fit = (
        normalize_bool(gdf[fit_name].to_numpy()).astype(bool)
        if fit_name
        else np.isfinite(velocity)
    )
    mask = fit & np.isfinite(velocity)

    selected = gdf.loc[mask].copy()
    v = velocity[mask]
    se = velocity_se[mask]
    ci_low = v - 1.96 * se
    ci_high = v + 1.96 * se

    q_recommended = (
        normalize_bool(gdf.loc[mask, rec_name].to_numpy())
        if rec_name
        else (
            np.isfinite(rmse[mask])
            & (rmse[mask] <= 15.0)
            & np.isfinite(se)
            & (se <= 3.0)
        ).astype(np.int16)
    )
    q_strict = (
        normalize_bool(gdf.loc[mask, strict_name].to_numpy())
        if strict_name
        else q_recommended.copy()
    )
    q_signif = (
        np.isfinite(ci_low)
        & np.isfinite(ci_high)
        & ((ci_low > 0) | (ci_high < 0))
    ).astype(np.int16)

    overall = geopandas.GeoDataFrame(
        {
            "PS_ID": (
                as_int(selected[id_name].to_numpy(), np.int64)
                if id_name
                else np.arange(1, len(selected) + 1, dtype=np.int64)
            ),
            "VEL_MM_YR": v.astype(np.float32),
            "VEL_SE": se.astype(np.float32),
            "CI95_LO": ci_low.astype(np.float32),
            "CI95_HI": ci_high.astype(np.float32),
            "RMSE_MM": rmse[mask].astype(np.float32),
            "N_OBS": (
                as_int(selected[nobs_name].to_numpy(), np.int16)
                if nobs_name
                else np.zeros(len(selected), np.int16)
            ),
            "SPAN_DAY": (
                pandas.to_numeric(
                    selected[span_name],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
                if span_name
                else np.full(len(selected), np.nan, np.float32)
            ),
            "EFF_N": (
                pandas.to_numeric(
                    selected[eff_name],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
                if eff_name
                else np.full(len(selected), np.nan, np.float32)
            ),
            "Q_RECOM": q_recommended,
            "Q_STRICT": q_strict,
            "Q_SIGNIF": q_signif,
            "REASON": (
                as_int(selected[reason_name].to_numpy(), np.int16)
                if reason_name
                else np.zeros(len(selected), np.int16)
            ),
        },
        geometry=selected.geometry.to_numpy(),
        crs=selected.crs,
    )

    output_path = output_dir / "overall_velocity_allfit.shp"
    write_shapefile(overall, output_path)

    recommended_path = output_dir / "overall_velocity_recommended.shp"
    write_shapefile(
        overall.loc[overall["Q_RECOM"] == 1].copy(),
        recommended_path,
    )

    return {
        "source": str(source_path),
        "all_fit_points": int(len(overall)),
        "recommended_points": int(
            np.count_nonzero(overall["Q_RECOM"].to_numpy())
        ),
        "strict_points": int(
            np.count_nonzero(overall["Q_STRICT"].to_numpy())
        ),
        "all_fit_shapefile": str(output_path),
        "recommended_shapefile": str(recommended_path),
    }


def read_h5_vector(h5: Any, name: str, n: int, default: Any) -> np.ndarray:
    if name not in h5:
        if np.isscalar(default):
            return np.full(n, default)
        return np.asarray(default)
    values = np.asarray(h5[name][:]).reshape(-1)
    if values.size != n:
        raise ExportError(
            f"HDF5变量{name}长度错误：{values.size}，应为{n}"
        )
    return values


def export_overall_from_h5(
    h5_path: Path,
    output_dir: Path,
    target_epsg: int,
) -> dict[str, Any]:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")
    h5py = require_import("h5py", "h5py")

    if not h5_path.exists():
        raise ExportError(f"找不到整体速率点结果：{h5_path}")

    with h5py.File(h5_path, "r") as h5:
        lon = np.asarray(h5["lon"][:]).reshape(-1).astype(np.float64)
        lat = np.asarray(h5["lat"][:]).reshape(-1).astype(np.float64)
        n = lon.size

        velocity = read_h5_vector(
            h5, "velocity_mm_yr", n, np.nan
        ).astype(np.float64)
        velocity_se = read_h5_vector(
            h5, "velocity_std_mm_yr", n, np.nan
        ).astype(np.float64)
        ci_low = read_h5_vector(
            h5,
            "ci95_low_mm_yr",
            n,
            velocity - 1.96 * velocity_se,
        ).astype(np.float64)
        ci_high = read_h5_vector(
            h5,
            "ci95_high_mm_yr",
            n,
            velocity + 1.96 * velocity_se,
        ).astype(np.float64)
        rmse = read_h5_vector(
            h5, "rmse_mm", n, np.nan
        ).astype(np.float64)
        n_obs = read_h5_vector(
            h5, "n_obs", n, 0
        )
        span_days = read_h5_vector(
            h5, "span_days", n, np.nan
        ).astype(np.float64)
        effective_n = read_h5_vector(
            h5, "effective_n", n, np.nan
        ).astype(np.float64)
        accepted = read_h5_vector(
            h5, "accepted", n, 1
        ).astype(bool)
        strict = read_h5_vector(
            h5, "best_quality_mask", n, 0
        ).astype(bool)
        reason = read_h5_vector(
            h5, "quality_reason_code", n, 0
        )

    fit_mask = (
        accepted
        & np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(velocity)
    )
    recommended = (
        fit_mask
        & np.isfinite(rmse)
        & (rmse <= 15.0)
        & np.isfinite(velocity_se)
        & (velocity_se <= 3.0)
    )
    significant = (
        fit_mask
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
        & ((ci_low > 0) | (ci_high < 0))
    )

    frame = pandas.DataFrame(
        {
            "PS_ID": np.arange(1, n + 1, dtype=np.int64)[fit_mask],
            "VEL_MM_YR": velocity[fit_mask].astype(np.float32),
            "VEL_SE": velocity_se[fit_mask].astype(np.float32),
            "CI95_LO": ci_low[fit_mask].astype(np.float32),
            "CI95_HI": ci_high[fit_mask].astype(np.float32),
            "RMSE_MM": rmse[fit_mask].astype(np.float32),
            "N_OBS": n_obs[fit_mask].astype(np.int16),
            "SPAN_DAY": span_days[fit_mask].astype(np.float32),
            "EFF_N": effective_n[fit_mask].astype(np.float32),
            "Q_RECOM": recommended[fit_mask].astype(np.int16),
            "Q_STRICT": strict[fit_mask].astype(np.int16),
            "Q_SIGNIF": significant[fit_mask].astype(np.int16),
            "REASON": reason[fit_mask].astype(np.int16),
            "lon": lon[fit_mask],
            "lat": lat[fit_mask],
        }
    )
    geometry = geopandas.points_from_xy(frame["lon"], frame["lat"])
    overall = geopandas.GeoDataFrame(
        frame.drop(columns=["lon", "lat"]),
        geometry=geometry,
        crs="EPSG:4326",
    ).to_crs(f"EPSG:{int(target_epsg)}")

    allfit_path = output_dir / "overall_velocity_allfit.shp"
    recommended_path = output_dir / "overall_velocity_recommended.shp"
    write_shapefile(overall, allfit_path)
    write_shapefile(
        overall.loc[overall["Q_RECOM"] == 1].copy(),
        recommended_path,
    )

    return {
        "source": str(h5_path),
        "all_fit_points": int(len(overall)),
        "recommended_points": int(
            np.count_nonzero(overall["Q_RECOM"].to_numpy())
        ),
        "strict_points": int(
            np.count_nonzero(overall["Q_STRICT"].to_numpy())
        ),
        "all_fit_shapefile": str(allfit_path),
        "recommended_shapefile": str(recommended_path),
    }


def write_field_dictionary(path: Path) -> None:
    rows = [
        ("PS_ID", "PS点编号"),
        ("YEAR", "年度"),
        ("VEL_MM_YR", "LOS速率，mm/yr；正值朝向卫星"),
        ("VEL_SE", "速率标准误差，mm/yr"),
        ("CI95_LO", "95%置信区间下限，mm/yr"),
        ("CI95_HI", "95%置信区间上限，mm/yr"),
        ("Q_RECOM", "推荐点标记；1为通过"),
        ("Q_STRICT", "严格点标记；1为通过"),
        ("Q_SIGNIF", "95%显著性标记；1为置信区间不跨0"),
        ("N_OBS", "有效获取日期数"),
        ("SPAN_DAY", "有效时间跨度，天"),
        ("MOD_RMS", "联合模型时间残差RMS，mm"),
        ("RMSE_MM", "整体速率拟合RMS，mm"),
        ("ANN_AMP", "共同年周期振幅，mm"),
        ("ANN_PEAK", "共同年周期峰值日"),
        ("FORMAL_YR", "完整年度标记；1为正式完整年度"),
        ("YR_N_ACQ", "该年度总获取日期数"),
        ("EFF_N", "有效观测数"),
        ("REASON", "质量原因代码"),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "description"])
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).expanduser().resolve()
    annual_root = dataset / "joint_piecewise_seasonal_velocity"
    annual_gpkg = Path(
        args.annual_gpkg
        or annual_root / "joint_piecewise_seasonal_velocity.gpkg"
    ).expanduser().resolve()
    annual_summary = Path(
        args.annual_summary
        or annual_root / "joint_year_summary.csv"
    ).expanduser().resolve()

    output_root = Path(
        args.out
        or dataset / "velocity_shapefiles"
    ).expanduser().resolve()
    annual_output = output_root / "annual"
    overall_output = output_root / "overall"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Preserve the annual GeoPackage CRS unless the user explicitly supplies
    # a target EPSG. The overall result is transformed to the same CRS.
    geopandas = require_import("geopandas", "geopandas")
    annual_probe = geopandas.read_file(
        annual_gpkg,
        rows=1,
    )
    annual_epsg = (
        annual_probe.crs.to_epsg()
        if annual_probe.crs is not None
        else None
    )
    target_epsg = int(
        args.target_epsg
        or annual_epsg
        or 32650
    )

    annual_records = export_annual_shapefiles(
        annual_gpkg,
        annual_summary,
        annual_output,
        target_epsg=target_epsg,
    )

    overall_source_gpkg = (
        Path(args.overall_gpkg).expanduser().resolve()
        if args.overall_gpkg
        else find_overall_gpkg(dataset)
    )
    if overall_source_gpkg is not None:
        overall_result = export_overall_from_gpkg(
            overall_source_gpkg,
            overall_output,
            target_epsg,
        )
    else:
        overall_h5 = Path(
            args.overall_h5
            or dataset
            / "best_velocity_gacos"
            / "points"
            / "best_velocity_points.h5"
        ).expanduser().resolve()
        overall_result = export_overall_from_h5(
            overall_h5,
            overall_output,
            target_epsg,
        )

    write_field_dictionary(
        output_root / "shapefile_field_dictionary.csv"
    )

    report = {
        "status": "completed",
        "dataset": str(dataset),
        "output": str(output_root),
        "target_epsg": target_epsg,
        "annual_source": str(annual_gpkg),
        "annual_hdf5_used": False,
        "annual_outputs": annual_records,
        "overall_output": overall_result,
        "los_sign": "positive_toward_satellite",
        "spatial_form": (
            "original irregular PS point distribution; no raster grid "
            "and no spatial averaging"
        ),
        "main_usage": {
            "annual": (
                "Each annual SHP contains all finite fit-valid points. "
                "Use Q_RECOM=1 for the recommended result."
            ),
            "overall": (
                "overall_velocity_allfit.shp preserves all fit-valid PS; "
                "overall_velocity_recommended.shp is the recommended subset."
            ),
        },
    }
    report_path = output_root / "shapefile_export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n============================================================")
    print("Velocity Shapefile export completed")
    print("============================================================")
    print(f"Annual SHP directory : {annual_output}")
    print(f"Overall SHP directory: {overall_output}")
    print(f"Annual HDF5 used     : No")
    print(f"Target CRS           : EPSG:{target_epsg}")
    print(f"Report               : {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export annual and full-period LOS velocities as original-PS "
            "point Shapefiles"
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--annual-gpkg", default=None)
    parser.add_argument("--annual-summary", default=None)
    parser.add_argument("--overall-gpkg", default=None)
    parser.add_argument("--overall-h5", default=None)
    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("用户中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
