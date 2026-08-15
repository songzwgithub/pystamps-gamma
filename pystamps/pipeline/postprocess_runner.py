from __future__ import annotations

# ENGINEERING_POSTPROCESS_V1

import subprocess
import sys
import time
from pathlib import Path

from pystamps.config import PostprocessConfig


class EngineeringPostprocessError(RuntimeError):
    pass


def run_engineering_postprocess(
    dataset_root: Path,
    config: PostprocessConfig,
) -> str:
    root = Path(dataset_root).expanduser().resolve()

    output_root = Path(config.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()

    data_dir = output_root / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    core_cmd = [
        sys.executable,
        "-m",
        "pystamps.postprocess_core",
        "--dataset",
        str(root),
        "--out",
        str(data_dir),
        "--chunk-ps",
        str(int(config.chunk_ps)),
        "--annual-min-obs",
        str(int(config.annual_min_obs)),
        "--annual-min-span-days",
        str(float(config.annual_min_span_days)),
    ]

    print()
    print("=" * 88)
    print("AUTOMATIC ENGINEERING POSTPROCESS")
    print("=" * 88)
    print("[POSTPROCESS] scientific products")
    print("output:", output_root)

    core = subprocess.run(
        core_cmd,
        check=False,
        text=True,
    )

    if core.returncode != 0:
        raise EngineeringPostprocessError(
            f"Scientific postprocess failed with exit code {core.returncode}"
        )

    export_cmd = [
        sys.executable,
        "-m",
        "pystamps.engineering_export",
        "--output-root",
        str(output_root),
        "--grid-resolution-m",
        str(float(config.grid_resolution_m)),
    ]

    if not config.figures:
        export_cmd.append("--no-figures")
    if not config.shapefile:
        export_cmd.append("--no-shapefile")
    if not config.timeseries_shapefile:
        export_cmd.append("--no-timeseries-shapefile")
    if not config.geotiff:
        export_cmd.append("--no-geotiff")

    print()
    print("[POSTPROCESS] engineering GIS / figures")

    exported = subprocess.run(
        export_cmd,
        check=False,
        text=True,
    )

    if exported.returncode != 0:
        raise EngineeringPostprocessError(
            f"Engineering export failed with exit code {exported.returncode}"
        )

    # === VERTICAL_CONVERSION_RUNNER_V1 ===
    if config.vertical_enabled:
        vertical_cmd = [
            sys.executable,
            "-m",
            "pystamps.vertical_export",
            "--dataset-root",
            str(root),
            "--output-root",
            str(output_root),
            "--incidence-source",
            str(config.vertical_incidence_source),
            "--positive",
            str(config.vertical_positive),
            "--grid-resolution-m",
            str(float(config.grid_resolution_m)),
        ]

        if config.vertical_incidence_deg is not None:
            vertical_cmd.extend(
                [
                    "--incidence-deg",
                    str(float(config.vertical_incidence_deg)),
                ]
            )

        if not config.figures:
            vertical_cmd.append("--no-figures")

        if not config.shapefile:
            vertical_cmd.append("--no-shapefile")

        if not config.timeseries_shapefile:
            vertical_cmd.append("--no-timeseries-shapefile")

        if not config.geotiff:
            vertical_cmd.append("--no-geotiff")

        print()
        print("[POSTPROCESS] LOS -> vertical conversion")
        print(
            "[POSTPROCESS] assumption: horizontal deformation "
            "is negligible"
        )

        vertical = subprocess.run(
            vertical_cmd,
            check=False,
            text=True,
        )

        if vertical.returncode != 0:
            raise EngineeringPostprocessError(
                "Vertical engineering export failed with "
                f"exit code {vertical.returncode}"
            )

    required = [
        data_dir / "corrected_timeseries.h5",
        data_dir / "velocity_full.csv",
        output_root / "engineering_manifest.json",
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise EngineeringPostprocessError(
            "Postprocess completed but required outputs are missing: "
            + ", ".join(missing)
        )

    elapsed = time.perf_counter() - started

    print()
    print("[POSTPROCESS] COMPLETE")
    print(f"[POSTPROCESS] output : {output_root}")
    print(f"[POSTPROCESS] elapsed: {elapsed:.2f} s")
    print("=" * 88)
    print()

    return (
        f"Engineering postprocess completed: {output_root} "
        f"({elapsed:.2f} s)"
    )
