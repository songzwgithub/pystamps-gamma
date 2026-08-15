from __future__ import annotations

# ENGINEERING_POSTPROCESS_V1

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


class EngineeringExportError(RuntimeError):
    pass


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _robust_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return -1.0, 1.0

    lo, hi = np.nanpercentile(a, [2.0, 98.0])

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(a)) if a.size else 0.0
        span = max(1.0, abs(center) * 0.1)
        lo, hi = center - span, center + span

    if symmetric:
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return -vmax, vmax

    return float(lo), float(hi)


def _save_scatter_map(
    path: Path,
    lonlat: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    symmetric: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)

    if not np.any(finite):
        return

    vmin, vmax = _robust_limits(values[finite], symmetric=symmetric)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    sc = ax.scatter(
        lonlat[finite, 0],
        lonlat[finite, 1],
        c=values[finite],
        s=7,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)

    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_prj(base: Path) -> None:
    wgs84 = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    base.with_suffix(".prj").write_text(wgs84, encoding="ascii")
    base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")


def _write_velocity_shapefile(
    base: Path,
    lonlat: np.ndarray,
    velocity: np.ndarray,
    rms: np.ndarray,
    endpoint: np.ndarray,
    cumulative: np.ndarray,
    annual_years: np.ndarray,
    annual_velocity: np.ndarray,
    annual_rms: np.ndarray,
) -> None:
    import shapefile

    writer = shapefile.Writer(
        str(base),
        shapeType=shapefile.POINT,
        encoding="utf-8",
    )

    writer.field("PS_ID", "N", decimal=0)
    writer.field("VEL_MM_YR", "F", size=18, decimal=6)
    writer.field("RMS_MM", "F", size=18, decimal=6)
    writer.field("ENDVEL", "F", size=18, decimal=6)
    writer.field("CUM_LAST", "F", size=18, decimal=6)

    for year in annual_years.tolist():
        writer.field(f"V{int(year)}", "F", size=18, decimal=6)
        writer.field(f"R{int(year)}", "F", size=18, decimal=6)

    for i in range(lonlat.shape[0]):
        writer.point(float(lonlat[i, 0]), float(lonlat[i, 1]))

        row = [
            i + 1,
            float(velocity[i]),
            float(rms[i]),
            float(endpoint[i]),
            float(cumulative[i]),
        ]

        for j in range(annual_years.size):
            row.extend(
                [
                    float(annual_velocity[i, j]),
                    float(annual_rms[i, j]),
                ]
            )

        writer.record(*row)

    writer.close()
    _write_prj(base)


def _write_timeseries_shapefile(
    base: Path,
    lonlat: np.ndarray,
    velocity: np.ndarray,
    date_int: np.ndarray,
    cumulative: np.ndarray,
) -> None:
    import shapefile

    writer = shapefile.Writer(
        str(base),
        shapeType=shapefile.POINT,
        encoding="utf-8",
    )

    writer.field("PS_ID", "N", decimal=0)
    writer.field("VEL_MM_YR", "F", size=18, decimal=6)

    for date in date_int.tolist():
        writer.field(
            f"D{int(date)}",
            "F",
            size=18,
            decimal=4,
        )

    for i in range(lonlat.shape[0]):
        writer.point(float(lonlat[i, 0]), float(lonlat[i, 1]))

        writer.record(
            i + 1,
            float(velocity[i]),
            *[float(v) for v in cumulative[i, :]],
        )

    writer.close()
    _write_prj(base)


def _write_geojson(
    path: Path,
    lonlat: np.ndarray,
    velocity: np.ndarray,
    rms: np.ndarray,
    cumulative: np.ndarray,
) -> None:
    features = []

    for i in range(lonlat.shape[0]):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(lonlat[i, 0]),
                        float(lonlat[i, 1]),
                    ],
                },
                "properties": {
                    "ps_id": i + 1,
                    "velocity_mm_yr": float(velocity[i]),
                    "residual_rms_mm": float(rms[i]),
                    "cumulative_last_mm": float(cumulative[i]),
                },
            }
        )

    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def _binned_raster(
    lonlat: np.ndarray,
    values: np.ndarray,
    *,
    resolution_m: float,
):
    from pyproj import Transformer

    median_lon = float(np.nanmedian(lonlat[:, 0]))
    median_lat = float(np.nanmedian(lonlat[:, 1]))
    epsg = _utm_epsg(median_lon, median_lat)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    x, y = transformer.transform(
        lonlat[:, 0],
        lonlat[:, 1],
    )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)

    if not np.any(finite):
        raise EngineeringExportError(
            "No finite PS available for raster export"
        )

    x = x[finite]
    y = y[finite]
    values = values[finite]

    res = float(resolution_m)
    xmin = math.floor(float(np.min(x)) / res) * res
    xmax = math.ceil(float(np.max(x)) / res) * res
    ymin = math.floor(float(np.min(y)) / res) * res
    ymax = math.ceil(float(np.max(y)) / res) * res

    width = max(1, int(math.ceil((xmax - xmin) / res)))
    height = max(1, int(math.ceil((ymax - ymin) / res)))

    col = np.floor((x - xmin) / res).astype(np.int64)
    row = np.floor((ymax - y) / res).astype(np.int64)

    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)

    linear = row * width + col
    count = np.bincount(
        linear,
        minlength=height * width,
    ).astype(np.int32)

    total = np.bincount(
        linear,
        weights=values,
        minlength=height * width,
    ).astype(np.float64)

    grid = np.full(
        height * width,
        np.nan,
        dtype=np.float32,
    )

    good = count > 0
    grid[good] = (total[good] / count[good]).astype(np.float32)

    return (
        grid.reshape(height, width),
        epsg,
        xmin,
        ymax,
        res,
        count.reshape(height, width),
    )


def _write_geotiff(
    path: Path,
    lonlat: np.ndarray,
    values: np.ndarray,
    *,
    resolution_m: float,
) -> dict[str, object]:
    import rasterio
    from rasterio.transform import from_origin

    grid, epsg, xmin, ymax, res, count = _binned_raster(
        lonlat,
        values,
        resolution_m=resolution_m,
    )

    transform = from_origin(
        xmin,
        ymax,
        res,
        res,
    )

    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid, 1)

    return {
        "epsg": epsg,
        "resolution_m": float(resolution_m),
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "occupied_cells": int(np.count_nonzero(count)),
    }


def export_engineering_products(
    output_root: Path,
    *,
    figures: bool = True,
    shapefile_enabled: bool = True,
    timeseries_shapefile: bool = True,
    geotiff: bool = True,
    grid_resolution_m: float = 100.0,
) -> dict[str, object]:
    output_root = Path(output_root).expanduser().resolve()

    data_dir = output_root / "data"
    figures_dir = _mkdir(output_root / "figures")
    gis_dir = _mkdir(output_root / "gis")
    rasters_dir = _mkdir(output_root / "rasters")

    full = np.load(data_dir / "velocity_full.npz")
    annual = np.load(data_dir / "annual_velocity.npz")

    lonlat = np.asarray(full["lonlat"], dtype=np.float64)
    velocity = np.asarray(full["velocity_mm_yr"], dtype=np.float64)
    rms = np.asarray(full["residual_rms_mm"], dtype=np.float64)
    endpoint = np.asarray(full["endpoint_velocity_mm_yr"], dtype=np.float64)
    cumulative_last = np.asarray(full["cumulative_last_mm"], dtype=np.float64)

    annual_years = np.asarray(annual["years"], dtype=np.int32)
    annual_velocity = np.asarray(annual["velocity_mm_yr"], dtype=np.float64)
    annual_rms = np.asarray(annual["residual_rms_mm"], dtype=np.float64)

    generated: list[str] = []
    raster_meta: dict[str, object] = {}

    with h5py.File(data_dir / "corrected_timeseries.h5", "r") as h5:
        date_int = np.asarray(h5["date_yyyymmdd"], dtype=np.int32)
        cumulative = np.asarray(h5["cumulative_mm"], dtype=np.float32)

    if figures:
        _save_scatter_map(
            figures_dir / "velocity_map.png",
            lonlat,
            velocity,
            title="InSAR LOS Mean Velocity",
            cbar_label="Velocity (mm/yr)",
            cmap="RdBu_r",
            symmetric=True,
        )
        generated.append("figures/velocity_map.png")

        _save_scatter_map(
            figures_dir / "cumulative_last_map.png",
            lonlat,
            cumulative_last,
            title=f"Cumulative LOS Displacement to {int(date_int[-1])}",
            cbar_label="Displacement (mm)",
            cmap="RdBu_r",
            symmetric=True,
        )
        generated.append("figures/cumulative_last_map.png")

        _save_scatter_map(
            figures_dir / "residual_rms_map.png",
            lonlat,
            rms,
            title="Velocity-fit Residual RMS",
            cbar_label="RMS (mm)",
            cmap="viridis",
            symmetric=False,
        )
        generated.append("figures/residual_rms_map.png")

        for j, year in enumerate(annual_years.tolist()):
            filename = f"annual_velocity_{int(year)}.png"
            _save_scatter_map(
                figures_dir / filename,
                lonlat,
                annual_velocity[:, j],
                title=f"InSAR LOS Velocity {int(year)}",
                cbar_label="Velocity (mm/yr)",
                cmap="RdBu_r",
                symmetric=True,
            )
            generated.append(f"figures/{filename}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        finite_vel = velocity[np.isfinite(velocity)]
        if finite_vel.size:
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            ax.hist(finite_vel, bins=60)
            ax.set_xlabel("Velocity (mm/yr)")
            ax.set_ylabel("PS count")
            ax.set_title("InSAR LOS Velocity Distribution")
            ax.grid(True, alpha=0.2)
            fig.savefig(figures_dir / "velocity_histogram.png", dpi=220)
            plt.close(fig)
            generated.append("figures/velocity_histogram.png")

        median_ts = np.nanmedian(cumulative, axis=0)
        p25 = np.nanpercentile(cumulative, 25, axis=0)
        p75 = np.nanpercentile(cumulative, 75, axis=0)

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        x = np.arange(date_int.size)
        ax.plot(x, median_ts, linewidth=1.8, label="Median")
        ax.fill_between(x, p25, p75, alpha=0.25, label="25–75%")
        tick_step = max(1, date_int.size // 8)
        ticks = np.arange(0, date_int.size, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [str(int(date_int[i])) for i in ticks],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("Cumulative LOS displacement (mm)")
        ax.set_xlabel("Acquisition date")
        ax.set_title("Project-wide PS Displacement Summary")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.savefig(figures_dir / "timeseries_summary.png", dpi=220)
        plt.close(fig)
        generated.append("figures/timeseries_summary.png")

    if shapefile_enabled:
        velocity_base = gis_dir / "insar_velocity"
        _write_velocity_shapefile(
            velocity_base,
            lonlat,
            velocity,
            rms,
            endpoint,
            cumulative_last,
            annual_years,
            annual_velocity,
            annual_rms,
        )
        generated.append("gis/insar_velocity.shp")

        _write_geojson(
            gis_dir / "insar_velocity.geojson",
            lonlat,
            velocity,
            rms,
            cumulative_last,
        )
        generated.append("gis/insar_velocity.geojson")

        if timeseries_shapefile:
            ts_base = gis_dir / "insar_timeseries"
            _write_timeseries_shapefile(
                ts_base,
                lonlat,
                velocity,
                date_int,
                cumulative,
            )
            generated.append("gis/insar_timeseries.shp")

    if geotiff:
        for filename, values in (
            ("velocity_mm_yr.tif", velocity),
            ("cumulative_last_mm.tif", cumulative_last),
            ("residual_rms_mm.tif", rms),
        ):
            meta = _write_geotiff(
                rasters_dir / filename,
                lonlat,
                values,
                resolution_m=grid_resolution_m,
            )
            generated.append(f"rasters/{filename}")
            raster_meta[filename] = meta

    summary = {
        "status": "completed",
        "n_ps": int(lonlat.shape[0]),
        "n_epochs": int(date_int.size),
        "date_start": int(date_int[0]),
        "date_end": int(date_int[-1]),
        "velocity_mm_yr": {
            "min": float(np.nanmin(velocity)),
            "median": float(np.nanmedian(velocity)),
            "max": float(np.nanmax(velocity)),
            "p02": float(np.nanpercentile(velocity, 2)),
            "p98": float(np.nanpercentile(velocity, 98)),
        },
        "cumulative_last_mm": {
            "min": float(np.nanmin(cumulative_last)),
            "median": float(np.nanmedian(cumulative_last)),
            "max": float(np.nanmax(cumulative_last)),
        },
        "annual_years": [int(v) for v in annual_years.tolist()],
        "rasterization": {
            "method": "PS-cell arithmetic mean; no spatial interpolation",
            "products": raster_meta,
        },
        "generated": generated,
    }

    (output_root / "engineering_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-shapefile", action="store_true")
    ap.add_argument("--no-timeseries-shapefile", action="store_true")
    ap.add_argument("--no-geotiff", action="store_true")
    ap.add_argument("--grid-resolution-m", type=float, default=100.0)
    args = ap.parse_args()

    summary = export_engineering_products(
        args.output_root,
        figures=not args.no_figures,
        shapefile_enabled=not args.no_shapefile,
        timeseries_shapefile=not args.no_timeseries_shapefile,
        geotiff=not args.no_geotiff,
        grid_resolution_m=args.grid_resolution_m,
    )

    print("=" * 88)
    print("ENGINEERING EXPORT COMPLETE")
    print("=" * 88)
    print("PS          :", summary["n_ps"])
    print("epochs      :", summary["n_epochs"])
    print("output_root :", Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
