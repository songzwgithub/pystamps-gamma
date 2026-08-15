from __future__ import annotations

# ENGINEERING_POSTPROCESS_V1
# VERTICAL_CONVERSION_EXPORT_V1

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np

from pystamps.io.mat import read_mat


class VerticalExportError(RuntimeError):
    pass


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_incidence_angle_rad(values, n_ps: int) -> np.ndarray:
    angle = np.asarray(values, dtype=np.float64).reshape(-1)

    if angle.size != n_ps:
        raise VerticalExportError(
            f"incidence angle count={angle.size}, expected n_ps={n_ps}"
        )

    if not np.all(np.isfinite(angle)):
        raise VerticalExportError(
            "incidence angle contains non-finite values"
        )

    q95 = float(np.nanpercentile(angle, 95.0))

    if q95 <= (np.pi / 2.0 + 0.15):
        angle_rad = angle
    elif q95 < 90.0:
        angle_rad = np.deg2rad(angle)
    else:
        raise VerticalExportError(
            f"invalid incidence-angle range; q95={q95}"
        )

    valid = (
        np.isfinite(angle_rad)
        & (angle_rad > 0.0)
        & (angle_rad < np.pi / 2.0)
    )

    if not np.all(valid):
        raise VerticalExportError(
            "invalid incidence angle outside (0, 90 deg)"
        )

    return angle_rad


def _resolve_incidence(
    dataset_root: Path,
    n_ps: int,
    source: str,
    constant_deg: float | None,
) -> tuple[np.ndarray, str]:
    source = str(source).strip().lower()
    root = Path(dataset_root).expanduser().resolve()

    if source not in {"auto", "la2", "constant"}:
        raise VerticalExportError(
            f"unsupported incidence source: {source}"
        )

    la2 = root / "la2.mat"

    if source in {"auto", "la2"} and la2.is_file():
        payload = read_mat(la2)

        if "la" in payload:
            angle = _normalize_incidence_angle_rad(
                payload["la"],
                n_ps,
            )
            return angle, "la2.mat:la"

        if source == "la2":
            raise VerticalExportError(
                "la2.mat exists but variable 'la' is missing"
            )

    if source == "la2":
        raise VerticalExportError(
            f"vertical_incidence_source=la2 but {la2} is unavailable"
        )

    if constant_deg is None:
        raise VerticalExportError(
            "No usable la2.mat was found and "
            "vertical_incidence_deg is null"
        )

    degree = float(constant_deg)

    if not 0.0 < degree < 90.0:
        raise VerticalExportError(
            "vertical_incidence_deg must be between 0 and 90"
        )

    return (
        np.full(
            n_ps,
            np.deg2rad(degree),
            dtype=np.float64,
        ),
        f"constant:{degree:.6f}deg",
    )


def _vertical_factor(
    incidence_rad: np.ndarray,
    positive: str,
) -> np.ndarray:
    cos_i = np.cos(incidence_rad)

    if np.any(cos_i <= 0.0):
        raise VerticalExportError(
            "invalid incidence angle produced cos(theta) <= 0"
        )

    factor = 1.0 / cos_i
    positive = str(positive).strip().lower()

    if positive == "up":
        return factor

    if positive == "down":
        return -factor

    raise VerticalExportError(
        "vertical_positive must be up or down"
    )


def _write_prj(base: Path) -> None:
    wgs84 = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )

    base.with_suffix(".prj").write_text(
        wgs84,
        encoding="ascii",
    )

    base.with_suffix(".cpg").write_text(
        "UTF-8\n",
        encoding="ascii",
    )


def _save_map(
    path: Path,
    lonlat: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)

    if not np.any(finite):
        return

    lo, hi = np.nanpercentile(values[finite], [2.0, 98.0])
    vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)

    fig, ax = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    scatter = ax.scatter(
        lonlat[finite, 0],
        lonlat[finite, 1],
        c=values[finite],
        s=7,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    bar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    bar.set_label(label)

    fig.savefig(path, dpi=240)
    plt.close(fig)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))

    return (32600 if lat >= 0.0 else 32700) + zone


def _write_geotiff(
    path: Path,
    lonlat: np.ndarray,
    values: np.ndarray,
    resolution_m: float,
) -> dict[str, object]:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

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
        raise VerticalExportError(
            f"no finite PS available for {path.name}"
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
    grid = grid.reshape(height, width)

    transform = from_origin(
        xmin,
        ymax,
        res,
        res,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(grid, 1)

    return {
        "epsg": epsg,
        "resolution_m": res,
        "width": width,
        "height": height,
        "occupied_cells": int(np.count_nonzero(count)),
    }


def run_vertical_export(
    dataset_root: Path,
    output_root: Path,
    *,
    incidence_source: str,
    incidence_deg: float | None,
    positive: str,
    figures: bool,
    shapefile_enabled: bool,
    timeseries_shapefile: bool,
    geotiff: bool,
    grid_resolution_m: float,
) -> dict[str, object]:
    root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    data_dir = output_root / "data"
    figures_dir = _mkdir(output_root / "figures")
    gis_dir = _mkdir(output_root / "gis")
    rasters_dir = _mkdir(output_root / "rasters")

    full = np.load(data_dir / "velocity_full.npz")
    annual = np.load(data_dir / "annual_velocity.npz")

    lonlat = np.asarray(full["lonlat"], dtype=np.float64)
    los_velocity = np.asarray(full["velocity_mm_yr"], dtype=np.float64)
    los_rms = np.asarray(full["residual_rms_mm"], dtype=np.float64)
    los_endpoint = np.asarray(
        full["endpoint_velocity_mm_yr"],
        dtype=np.float64,
    )
    los_cumulative_last = np.asarray(
        full["cumulative_last_mm"],
        dtype=np.float64,
    )

    years = np.asarray(annual["years"], dtype=np.int32)
    los_annual_velocity = np.asarray(
        annual["velocity_mm_yr"],
        dtype=np.float64,
    )
    los_annual_rms = np.asarray(
        annual["residual_rms_mm"],
        dtype=np.float64,
    )

    n_ps = int(lonlat.shape[0])

    incidence_rad, incidence_used = _resolve_incidence(
        root,
        n_ps,
        incidence_source,
        incidence_deg,
    )

    factor = _vertical_factor(
        incidence_rad,
        positive,
    )

    abs_factor = np.abs(factor)

    velocity = los_velocity * factor
    rms = los_rms * abs_factor
    endpoint = los_endpoint * factor
    cumulative_last = los_cumulative_last * factor

    annual_velocity = los_annual_velocity * factor[:, None]
    annual_rms = los_annual_rms * abs_factor[:, None]

    np.savez_compressed(
        data_dir / "vertical_velocity.npz",
        lonlat=lonlat,
        incidence_deg=np.rad2deg(incidence_rad),
        vertical_factor=factor,
        velocity_mm_yr=velocity,
        residual_rms_mm=rms,
        endpoint_velocity_mm_yr=endpoint,
        cumulative_last_mm=cumulative_last,
        years=years,
        annual_velocity_mm_yr=annual_velocity,
        annual_rms_mm=annual_rms,
    )

    with (data_dir / "vertical_velocity.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        header = [
            "lon",
            "lat",
            "incidence_deg",
            "vertical_velocity_mm_yr",
            "vertical_rms_mm",
            "vertical_endpoint_velocity_mm_yr",
            "vertical_cumulative_last_mm",
        ]

        for year in years.tolist():
            header += [
                f"vertical_velocity_{int(year)}_mm_yr",
                f"vertical_rms_{int(year)}_mm",
            ]

        writer.writerow(header)

        for i in range(n_ps):
            row = [
                f"{lonlat[i,0]:.8f}",
                f"{lonlat[i,1]:.8f}",
                f"{np.rad2deg(incidence_rad[i]):.6f}",
                f"{velocity[i]:.6f}",
                f"{rms[i]:.6f}",
                f"{endpoint[i]:.6f}",
                f"{cumulative_last[i]:.6f}",
            ]

            for j in range(years.size):
                row += [
                    f"{annual_velocity[i,j]:.6f}",
                    f"{annual_rms[i,j]:.6f}",
                ]

            writer.writerow(row)

    with h5py.File(
        data_dir / "corrected_timeseries.h5",
        "r",
    ) as src:
        dates = np.asarray(src["date_yyyymmdd"], dtype=np.int32)
        cumulative_los = np.asarray(src["cumulative_mm"], dtype=np.float32)
        los_master_ref = np.asarray(
            src["los_master_ref_mm"],
            dtype=np.float32,
        )

    cumulative = (
        cumulative_los.astype(np.float64)
        * factor[:, None]
    ).astype(np.float32)

    master_ref = (
        los_master_ref.astype(np.float64)
        * factor[:, None]
    ).astype(np.float32)

    with h5py.File(
        data_dir / "vertical_timeseries.h5",
        "w",
    ) as dst:
        dst.create_dataset(
            "lonlat",
            data=lonlat,
            compression="gzip",
            compression_opts=1,
        )
        dst.create_dataset(
            "date_yyyymmdd",
            data=dates,
        )
        dst.create_dataset(
            "incidence_deg",
            data=np.rad2deg(incidence_rad).astype(np.float32),
        )
        dst.create_dataset(
            "vertical_master_ref_mm",
            data=master_ref,
            compression="gzip",
            compression_opts=1,
        )
        dst.create_dataset(
            "vertical_cumulative_mm",
            data=cumulative,
            compression="gzip",
            compression_opts=1,
        )
        dst.attrs["formula"] = (
            "vertical_up = LOS / cos(incidence); "
            "horizontal deformation neglected"
        )
        dst.attrs["incidence_source"] = incidence_used
        dst.attrs["positive_direction"] = positive

    generated = [
        "data/vertical_velocity.npz",
        "data/vertical_velocity.csv",
        "data/vertical_timeseries.h5",
    ]

    if figures:
        _save_map(
            figures_dir / "vertical_velocity_map.png",
            lonlat,
            velocity,
            title=(
                "InSAR Vertical Mean Velocity "
                f"(positive {positive})"
            ),
            label="Vertical velocity (mm/yr)",
        )
        generated.append("figures/vertical_velocity_map.png")

        _save_map(
            figures_dir / "vertical_cumulative_last_map.png",
            lonlat,
            cumulative_last,
            title=(
                "Vertical Cumulative Displacement "
                f"to {int(dates[-1])} "
                f"(positive {positive})"
            ),
            label="Vertical displacement (mm)",
        )
        generated.append(
            "figures/vertical_cumulative_last_map.png"
        )

        for j, year in enumerate(years.tolist()):
            name = f"vertical_velocity_{int(year)}.png"

            _save_map(
                figures_dir / name,
                lonlat,
                annual_velocity[:, j],
                title=(
                    f"Vertical Velocity {int(year)} "
                    f"(positive {positive})"
                ),
                label="Vertical velocity (mm/yr)",
            )

            generated.append(f"figures/{name}")

    if shapefile_enabled:
        import shapefile

        base = gis_dir / "insar_vertical_velocity"

        writer = shapefile.Writer(
            str(base),
            shapeType=shapefile.POINT,
            encoding="utf-8",
        )

        writer.field("PS_ID", "N", decimal=0)
        writer.field("INC_DEG", "F", size=12, decimal=5)
        writer.field("VVEL_MM_YR", "F", size=18, decimal=6)
        writer.field("VRMS_MM", "F", size=18, decimal=6)
        writer.field("VCUM_LAST", "F", size=18, decimal=6)

        for year in years.tolist():
            writer.field(
                f"V{int(year)}",
                "F",
                size=18,
                decimal=6,
            )

        for i in range(n_ps):
            writer.point(
                float(lonlat[i, 0]),
                float(lonlat[i, 1]),
            )

            writer.record(
                i + 1,
                float(np.rad2deg(incidence_rad[i])),
                float(velocity[i]),
                float(rms[i]),
                float(cumulative_last[i]),
                *[
                    float(annual_velocity[i, j])
                    for j in range(years.size)
                ],
            )

        writer.close()
        _write_prj(base)

        generated.append(
            "gis/insar_vertical_velocity.shp"
        )

        if timeseries_shapefile:
            base_ts = gis_dir / "insar_vertical_timeseries"

            writer = shapefile.Writer(
                str(base_ts),
                shapeType=shapefile.POINT,
                encoding="utf-8",
            )

            writer.field("PS_ID", "N", decimal=0)
            writer.field(
                "VVEL_MM_YR",
                "F",
                size=18,
                decimal=6,
            )

            for date in dates.tolist():
                writer.field(
                    f"D{int(date)}",
                    "F",
                    size=18,
                    decimal=4,
                )

            for i in range(n_ps):
                writer.point(
                    float(lonlat[i, 0]),
                    float(lonlat[i, 1]),
                )

                writer.record(
                    i + 1,
                    float(velocity[i]),
                    *[
                        float(v)
                        for v in cumulative[i, :]
                    ],
                )

            writer.close()
            _write_prj(base_ts)

            generated.append(
                "gis/insar_vertical_timeseries.shp"
            )

    raster_meta = {}

    if geotiff:
        for filename, values in (
            (
                "vertical_velocity_mm_yr.tif",
                velocity,
            ),
            (
                "vertical_cumulative_last_mm.tif",
                cumulative_last,
            ),
        ):
            raster_meta[filename] = _write_geotiff(
                rasters_dir / filename,
                lonlat,
                values,
                grid_resolution_m,
            )

            generated.append(f"rasters/{filename}")

    summary = {
        "enabled": True,
        "assumption": (
            "horizontal deformation is negligible"
        ),
        "formula": (
            "vertical_up = LOS / cos(incidence)"
        ),
        "positive_direction": positive,
        "incidence_source": incidence_used,
        "incidence_deg": {
            "min": float(
                np.nanmin(np.rad2deg(incidence_rad))
            ),
            "median": float(
                np.nanmedian(np.rad2deg(incidence_rad))
            ),
            "max": float(
                np.nanmax(np.rad2deg(incidence_rad))
            ),
        },
        "rasterization": {
            "method": (
                "PS-cell arithmetic mean; "
                "no spatial interpolation"
            ),
            "products": raster_meta,
        },
        "generated": generated,
    }

    manifest_path = output_root / "engineering_manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest = {}

    manifest["vertical_conversion"] = summary

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--incidence-source",
        choices=("auto", "la2", "constant"),
        default="auto",
    )

    ap.add_argument(
        "--incidence-deg",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--positive",
        choices=("up", "down"),
        default="up",
    )

    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-shapefile", action="store_true")
    ap.add_argument(
        "--no-timeseries-shapefile",
        action="store_true",
    )
    ap.add_argument("--no-geotiff", action="store_true")

    ap.add_argument(
        "--grid-resolution-m",
        type=float,
        default=100.0,
    )

    args = ap.parse_args()

    summary = run_vertical_export(
        args.dataset_root,
        args.output_root,
        incidence_source=args.incidence_source,
        incidence_deg=args.incidence_deg,
        positive=args.positive,
        figures=not args.no_figures,
        shapefile_enabled=not args.no_shapefile,
        timeseries_shapefile=(
            not args.no_timeseries_shapefile
        ),
        geotiff=not args.no_geotiff,
        grid_resolution_m=args.grid_resolution_m,
    )

    print("=" * 88)
    print("VERTICAL ENGINEERING EXPORT COMPLETE")
    print("=" * 88)
    print("incidence source :", summary["incidence_source"])
    print("positive         :", summary["positive_direction"])
    print("output_root      :", Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
