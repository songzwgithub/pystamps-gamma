#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export one appropriate ESRI Shapefile for each calendar year from the completed
joint piecewise-linear + common-seasonal + SBAS-covariance + Huber-GLS result.

Source:
    joint_piecewise_seasonal_velocity.gpkg

The joint HDF5 file is NOT used.

Selection used for the main yearly product:
    finite annual velocity
    + global joint model fit valid
    + Q_STRICT == 1

Q_SIGNIF is retained as an attribute but is NOT used to remove near-zero
stable points. Extreme annual rates are retained and flagged for review rather
than silently deleted.

Output:
    final_annual_shapefiles/formal/
        annual_velocity_YYYY_final.shp
    final_annual_shapefiles/partial/
        annual_velocity_YYYY_partial.shp

Original irregular PS locations are preserved. No raster grid, averaging,
interpolation or spatial smoothing is performed.
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
            f"请在stamps环境安装：python -m pip install {pip_name or name}\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc


def bool_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"b", "i", "u", "f"}:
        return (np.nan_to_num(array, nan=0.0) != 0)
    text = np.char.lower(array.astype(str))
    return np.isin(text, ["1", "true", "yes", "y"])


def scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if np.isfinite(value) else False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def numeric(series: Any) -> np.ndarray:
    pandas = require_import("pandas", "pandas")
    return pandas.to_numeric(series, errors="coerce").to_numpy(np.float64)


def clean_shapefile(path: Path) -> None:
    for suffix in (
        ".shp", ".shx", ".dbf", ".prj", ".cpg",
        ".qix", ".fix", ".sbn", ".sbx",
    ):
        component = path.with_suffix(suffix)
        if component.exists():
            component.unlink()


def write_shapefile(gdf: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_shapefile(path)
    gdf.to_file(
        path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False,
    )
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def read_year_summary(path: Path) -> dict[int, dict[str, Any]]:
    pandas = require_import("pandas", "pandas")
    if not path.exists():
        return {}
    frame = pandas.read_csv(path)
    if "year" not in frame.columns:
        return {}
    output: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        output[int(row["year"])] = row.to_dict()
    return output


def choose_display_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 40.0
    percentile = float(np.percentile(np.abs(finite), 98))
    return max(10.0, math.ceil(percentile / 5.0) * 5.0)


def write_qml(path: Path, value_field: str, limit: float) -> None:
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="{value_field}" graduatedMethod="GraduatedColor" symbollevels="0">
    <ranges>
      <range lower="{-limit}" upper="{-limit/2}" label="{-limit:.0f} to {-limit/2:.0f}" symbol="0"/>
      <range lower="{-limit/2}" upper="-5" label="{-limit/2:.0f} to -5" symbol="1"/>
      <range lower="-5" upper="5" label="-5 to 5" symbol="2"/>
      <range lower="5" upper="{limit/2}" label="5 to {limit/2:.0f}" symbol="3"/>
      <range lower="{limit/2}" upper="{limit}" label="{limit/2:.0f} to {limit:.0f}" symbol="4"/>
    </ranges>
    <symbols>
      <symbol type="marker" name="0"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="33,102,172,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.8"/></Option></layer></symbol>
      <symbol type="marker" name="1"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="103,169,207,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.8"/></Option></layer></symbol>
      <symbol type="marker" name="2"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="247,247,247,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.8"/></Option></layer></symbol>
      <symbol type="marker" name="3"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="239,138,98,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.8"/></Option></layer></symbol>
      <symbol type="marker" name="4"><layer class="SimpleMarker"><Option type="Map"><Option name="color" value="178,24,43,255"/><Option name="outline_style" value="no"/><Option name="size" value="0.8"/></Option></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
    path.write_text(qml, encoding="utf-8")


def export(args: argparse.Namespace) -> int:
    geopandas = require_import("geopandas", "geopandas")
    pandas = require_import("pandas", "pandas")

    dataset = Path(args.dataset).expanduser().resolve()
    source_gpkg = Path(
        args.source_gpkg
        or dataset
        / "joint_piecewise_seasonal_velocity"
        / "joint_piecewise_seasonal_velocity.gpkg"
    ).expanduser().resolve()
    summary_csv = Path(
        args.summary_csv
        or dataset
        / "joint_piecewise_seasonal_velocity"
        / "joint_year_summary.csv"
    ).expanduser().resolve()
    output_root = Path(
        args.out
        or dataset / "final_annual_shapefiles"
    ).expanduser().resolve()

    if not source_gpkg.exists():
        raise ExportError(f"找不到联合年度GeoPackage：{source_gpkg}")

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    formal_dir = output_root / "formal"
    partial_dir = output_root / "partial"
    formal_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    source = geopandas.read_file(source_gpkg)
    if source.empty:
        raise ExportError(f"GeoPackage为空：{source_gpkg}")
    if source.crs is None:
        raise ExportError(f"GeoPackage没有坐标系：{source_gpkg}")

    if args.target_epsg is not None:
        source = source.to_crs(f"EPSG:{int(args.target_epsg)}")

    required_global = {
        "ps_id", "fit_ok", "model_rms", "ann_amp", "ann_peak",
    }
    missing_global = sorted(required_global - set(source.columns))
    if missing_global:
        raise ExportError(
            "GeoPackage缺少全局字段：" + ", ".join(missing_global)
        )

    velocity_fields = sorted(
        [
            name for name in source.columns
            if name.startswith("v")
            and len(name) == 5
            and name[1:].isdigit()
        ],
        key=lambda name: int(name[1:]),
    )
    if not velocity_fields:
        raise ExportError("未发现vYYYY年度速率字段")

    summary = read_year_summary(summary_csv)
    fit_ok = bool_array(source["fit_ok"].to_numpy())
    geometry_ok = (
        ~source.geometry.is_empty.to_numpy()
        & source.geometry.notna().to_numpy()
    )

    records: list[dict[str, Any]] = []

    for velocity_field in velocity_fields:
        year = int(velocity_field[1:])
        names = {
            "velocity": velocity_field,
            "se": f"se{year}",
            "recommended": f"q{year}",
            "strict": f"s{year}",
            "significant": f"sg{year}",
            "nobs": f"n{year}",
            "span": f"sp{year}",
        }
        missing = [
            name for name in names.values()
            if name not in source.columns
        ]
        if missing:
            raise ExportError(
                f"{year}年缺少字段：" + ", ".join(missing)
            )

        velocity = numeric(source[names["velocity"]])
        velocity_se = numeric(source[names["se"]])
        q_recommended = bool_array(
            source[names["recommended"]].to_numpy()
        )
        q_strict = bool_array(
            source[names["strict"]].to_numpy()
        )
        q_significant = bool_array(
            source[names["significant"]].to_numpy()
        )
        n_obs = numeric(source[names["nobs"]])
        span_days = numeric(source[names["span"]])

        # Main annual product: strict-quality points, but not significance-only.
        selected = (
            fit_ok
            & q_strict
            & np.isfinite(velocity)
            & np.isfinite(velocity_se)
            & geometry_ok
        )

        annual_meta = summary.get(year, {})
        formal = scalar_bool(annual_meta.get("formal_year", False))
        acquisition_count = int(
            float(annual_meta.get("acquisition_count", 0) or 0)
        )

        selected_source = source.loc[selected].copy()
        v = velocity[selected]
        se = velocity_se[selected]
        ci_low = v - 1.96 * se
        ci_high = v + 1.96 * se
        significant = q_significant[selected].astype(np.int16)
        recommended = q_recommended[selected].astype(np.int16)

        extreme = np.zeros(v.size, np.int16)
        extreme[np.abs(v) > 50.0] = 1
        extreme[np.abs(v) > 100.0] = 2

        review = (
            (extreme > 0)
            | (se > float(args.review_se_mm_yr))
        ).astype(np.int16)

        result = geopandas.GeoDataFrame(
            {
                "PS_ID": numeric(
                    selected_source["ps_id"]
                ).astype(np.int64),
                "YEAR": np.full(v.size, year, np.int16),
                "VEL_MM_YR": v.astype(np.float32),
                "VEL_SE": se.astype(np.float32),
                "CI95_LO": ci_low.astype(np.float32),
                "CI95_HI": ci_high.astype(np.float32),
                "Q_RECOM": recommended,
                "Q_STRICT": np.ones(v.size, np.int16),
                "Q_SIGNIF": significant,
                "N_OBS": np.nan_to_num(
                    n_obs[selected],
                    nan=0.0,
                ).astype(np.int16),
                "SPAN_DAY": span_days[selected].astype(np.float32),
                "MOD_RMS": numeric(
                    selected_source["model_rms"]
                ).astype(np.float32),
                "ANN_AMP": numeric(
                    selected_source["ann_amp"]
                ).astype(np.float32),
                "ANN_PEAK": numeric(
                    selected_source["ann_peak"]
                ).astype(np.float32),
                "FORMAL_YR": np.full(
                    v.size,
                    int(formal),
                    np.int16,
                ),
                "YR_N_ACQ": np.full(
                    v.size,
                    acquisition_count,
                    np.int16,
                ),
                "EXTREME": extreme,
                "REVIEW": review,
            },
            geometry=selected_source.geometry.to_numpy(),
            crs=selected_source.crs,
        )

        if formal:
            output_path = (
                formal_dir
                / f"annual_velocity_{year}_final.shp"
            )
        else:
            output_path = (
                partial_dir
                / f"annual_velocity_{year}_partial.shp"
            )

        write_shapefile(result, output_path)

        display_limit = choose_display_limit(v)
        write_qml(
            output_path.with_suffix(".qml"),
            "VEL_MM_YR",
            display_limit,
        )

        records.append(
            {
                "year": year,
                "formal_year": int(formal),
                "acquisition_count": acquisition_count,
                "strict_points": int(v.size),
                "significant_points": int(
                    np.count_nonzero(significant)
                ),
                "extreme_gt50_points": int(
                    np.count_nonzero(np.abs(v) > 50.0)
                ),
                "extreme_gt100_points": int(
                    np.count_nonzero(np.abs(v) > 100.0)
                ),
                "velocity_p02_mm_yr": (
                    float(np.percentile(v, 2))
                    if v.size else np.nan
                ),
                "velocity_median_mm_yr": (
                    float(np.median(v))
                    if v.size else np.nan
                ),
                "velocity_mean_mm_yr": (
                    float(np.mean(v))
                    if v.size else np.nan
                ),
                "velocity_p98_mm_yr": (
                    float(np.percentile(v, 98))
                    if v.size else np.nan
                ),
                "velocity_se_median_mm_yr": (
                    float(np.median(se))
                    if se.size else np.nan
                ),
                "velocity_se_p90_mm_yr": (
                    float(np.percentile(se, 90))
                    if se.size else np.nan
                ),
                "display_limit_mm_yr": display_limit,
                "shapefile": str(output_path),
            }
        )

        print(
            f"[FINAL ANNUAL SHP] {year}: "
            f"formal={int(formal)}, strict={v.size:,}, "
            f"significant={np.count_nonzero(significant):,}, "
            f"|V|>50={np.count_nonzero(np.abs(v)>50):,} "
            f"-> {output_path}",
            flush=True,
        )

    summary_output = output_root / "final_annual_shapefile_summary.csv"
    pandas.DataFrame(records).to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    dictionary = [
        ("PS_ID", "PS点编号"),
        ("YEAR", "年份"),
        ("VEL_MM_YR", "年度LOS速率，mm/yr；正值朝向卫星"),
        ("VEL_SE", "年度速率标准误差，mm/yr"),
        ("CI95_LO", "95%置信区间下限，mm/yr"),
        ("CI95_HI", "95%置信区间上限，mm/yr"),
        ("Q_RECOM", "推荐质量标记"),
        ("Q_STRICT", "严格质量标记；输出点均为1"),
        ("Q_SIGNIF", "95%显著性标记；不用于删点"),
        ("N_OBS", "年度有效日期数"),
        ("SPAN_DAY", "年度有效时间跨度，天"),
        ("MOD_RMS", "全时段联合模型RMS，mm"),
        ("ANN_AMP", "共同年周期振幅，mm"),
        ("ANN_PEAK", "共同年周期峰值日"),
        ("FORMAL_YR", "完整自然年标记"),
        ("YR_N_ACQ", "该年度总获取日期数"),
        ("EXTREME", "0正常；1为|V|>50；2为|V|>100 mm/yr"),
        ("REVIEW", "1表示建议复核极端速率或较高标准误差"),
    ]
    with (
        output_root / "field_dictionary.csv"
    ).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "description"])
        writer.writerows(dictionary)

    report = {
        "status": "completed",
        "source": str(source_gpkg),
        "joint_hdf5_used": False,
        "selection": (
            "fit_ok=1 and sYYYY=1 and finite annual velocity/error"
        ),
        "significance_filter_applied": False,
        "spatial_processing": (
            "original PS coordinates; no grid, no averaging, "
            "no interpolation, no spatial smoothing"
        ),
        "formal_directory": str(formal_dir),
        "partial_directory": str(partial_dir),
        "summary": str(summary_output),
        "years": records,
        "los_sign": "positive_toward_satellite",
    }
    report_path = output_root / "final_annual_shapefile_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n============================================================")
    print("Final annual Shapefiles completed")
    print("============================================================")
    print(f"Formal years : {formal_dir}")
    print(f"Partial years: {partial_dir}")
    print(f"Summary      : {summary_output}")
    print(f"Report       : {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one strict-quality original-PS Shapefile per year"
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-gpkg", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-epsg", type=int, default=None)
    parser.add_argument(
        "--review-se-mm-yr",
        type=float,
        default=4.5,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return export(build_parser().parse_args())
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
