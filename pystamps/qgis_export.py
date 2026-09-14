from __future__ import annotations

# QGIS_GPKG_EXPORT_V1
#
# A vector delivery product. It reads the internal scientific HDF5 time
# series but does not create/copy HDF5 into the GeoPackage delivery.

import argparse
import datetime
import json
import sqlite3
import struct
from pathlib import Path

import h5py
import numpy as np

from pystamps.io.mat import read_mat
from pystamps.vertical_export import (
    _normalize_incidence_angle_rad,
)


class QgisExportError(RuntimeError):
    pass


def _gpkg_point_blob(
    lon: float,
    lat: float,
) -> bytes:
    header = (
        b"GP"
        + bytes(
            (
                0,
                1,
            )
        )
        + struct.pack(
            "<i",
            4326,
        )
    )
    wkb = struct.pack(
        "<BIdd",
        1,
        1,
        float(lon),
        float(lat),
    )
    return header + wkb


def _qident(
    name: str,
) -> str:
    return (
        '"'
        + str(name).replace(
            '"',
            '""',
        )
        + '"'
    )


def _create_schema(
    conn: sqlite3.Connection,
    layer: str,
    fields: list[
        tuple[
            str,
            str,
        ]
    ],
    extent: tuple[
        float,
        float,
        float,
        float,
    ],
) -> str:
    conn.execute(
        "PRAGMA application_id = "
        "1196437808"
    )
    conn.execute(
        "PRAGMA user_version = 10300"
    )

    conn.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        );

        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL,
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER
        );

        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            PRIMARY KEY (
                table_name,
                column_name
            )
        );

        CREATE TABLE gpkg_extensions (
            table_name TEXT,
            column_name TEXT,
            extension_name TEXT NOT NULL,
            definition TEXT NOT NULL,
            scope TEXT NOT NULL,
            UNIQUE (
                table_name,
                column_name,
                extension_name
            )
        );
        """
    )

    conn.executemany(
        """
        INSERT INTO gpkg_spatial_ref_sys
        (
            srs_name,
            srs_id,
            organization,
            organization_coordsys_id,
            definition,
            description
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "Undefined Cartesian SRS",
                -1,
                "NONE",
                -1,
                "undefined",
                "undefined Cartesian "
                "coordinate reference system",
            ),
            (
                "Undefined geographic SRS",
                0,
                "NONE",
                0,
                "undefined",
                "undefined geographic "
                "coordinate reference system",
            ),
            (
                "WGS 84 geodetic",
                4326,
                "EPSG",
                4326,
                (
                    'GEOGCS["WGS 84",'
                    'DATUM["WGS_1984",'
                    'SPHEROID["WGS 84",'
                    '6378137,298.257223563]],'
                    'PRIMEM["Greenwich",0],'
                    'UNIT["degree",'
                    '0.0174532925199433]]'
                ),
                (
                    "longitude/latitude "
                    "coordinates in "
                    "decimal degrees"
                ),
            ),
        ],
    )

    definitions = [
        '"fid" INTEGER PRIMARY KEY',
        '"geom" POINT NOT NULL',
    ]

    definitions.extend(
        f"{_qident(name)} {kind}"
        for name, kind
        in fields
    )

    conn.execute(
        f"CREATE TABLE "
        f"{_qident(layer)} "
        f"({', '.join(definitions)})"
    )

    now = (
        datetime.datetime.now(
            datetime.timezone.utc
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    )

    minx, miny, maxx, maxy = (
        extent
    )

    conn.execute(
        """
        INSERT INTO gpkg_contents
        (
            table_name,
            data_type,
            identifier,
            description,
            last_change,
            min_x,
            min_y,
            max_x,
            max_y,
            srs_id
        )
        VALUES (
            ?,
            'features',
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            4326
        )
        """,
        (
            layer,
            layer,
            (
                "InSAR PS: LOS and optional "
                "vertical deformation for "
                "every epoch"
            ),
            now,
            minx,
            miny,
            maxx,
            maxy,
        ),
    )

    conn.execute(
        """
        INSERT INTO gpkg_geometry_columns
        (
            table_name,
            column_name,
            geometry_type_name,
            srs_id,
            z,
            m
        )
        VALUES (
            ?,
            'geom',
            'POINT',
            4326,
            0,
            0
        )
        """,
        (
            layer,
        ),
    )

    rtree = (
        f"rtree_{layer}_geom"
    )

    conn.execute(
        f"CREATE VIRTUAL TABLE "
        f"{_qident(rtree)} "
        "USING rtree("
        "id, minx, maxx, miny, maxy)"
    )

    conn.execute(
        """
        INSERT INTO gpkg_extensions
        (
            table_name,
            column_name,
            extension_name,
            definition,
            scope
        )
        VALUES (
            ?,
            'geom',
            'gpkg_rtree_index',
            'http://www.geopackage.org/'
            'spec/#extension_rtree',
            'write-only'
        )
        """,
        (
            layer,
        ),
    )

    # Important: schema/metadata INSERTs leave an implicit sqlite3
    # transaction open. Close it before explicit chunk transactions.
    conn.commit()

    return rtree


def _read_incidence(
    root: Path,
    n_ps: int,
) -> np.ndarray:
    payload = read_mat(
        root / "inc2.mat"
    )

    raw = payload.get(
        "inc",
        payload.get(
            "incidence_angle"
        ),
    )

    if raw is None:
        raise QgisExportError(
            "inc2.mat contains no "
            "'inc' variable"
        )

    return (
        _normalize_incidence_angle_rad(
            raw,
            n_ps,
        )
    )


def export_qgis_geopackage(
    dataset_root: Path,
    output_root: Path,
    *,
    filename: str = (
        "insar_all_epochs.gpkg"
    ),
    layer: str = "insar_ps",
    vertical_enabled: bool = False,
    vertical_positive: str = "down",
    chunk_rows: int = 500,
) -> Path:
    root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
    )
    outroot = (
        Path(output_root)
        .expanduser()
        .resolve()
    )
    data_dir = (
        outroot
        / "data"
    )

    full = np.load(
        data_dir
        / "velocity_full.npz"
    )

    annual = np.load(
        data_dir
        / "annual_velocity.npz"
    )

    lonlat = np.asarray(
        full["lonlat"],
        dtype=np.float64,
    )

    los_velocity = np.asarray(
        full["velocity_mm_yr"],
        dtype=np.float64,
    ).reshape(-1)

    los_rms = np.asarray(
        full["residual_rms_mm"],
        dtype=np.float64,
    ).reshape(-1)

    los_endpoint = np.asarray(
        full[
            "endpoint_velocity_mm_yr"
        ],
        dtype=np.float64,
    ).reshape(-1)

    los_cumulative_last = np.asarray(
        full[
            "cumulative_last_mm"
        ],
        dtype=np.float64,
    ).reshape(-1)

    years = np.asarray(
        annual["years"],
        dtype=np.int32,
    ).reshape(-1)

    annual_los = np.asarray(
        annual[
            "velocity_mm_yr"
        ],
        dtype=np.float64,
    )

    annual_los_rms = np.asarray(
        annual[
            "residual_rms_mm"
        ],
        dtype=np.float64,
    )

    n_ps = int(
        lonlat.shape[0]
    )

    time_series_path = (
        data_dir
        / "corrected_timeseries.h5"
    )

    if not (
        time_series_path
        .is_file()
    ):
        raise QgisExportError(
            "Missing internal scientific "
            f"time series: {time_series_path}"
        )

    with h5py.File(
        time_series_path,
        "r",
    ) as h5:
        dates = np.asarray(
            h5[
                "date_yyyymmdd"
            ],
            dtype=np.int32,
        ).reshape(-1)

        if (
            h5[
                "cumulative_mm"
            ].shape
            != (
                n_ps,
                dates.size,
            )
        ):
            raise QgisExportError(
                "corrected_timeseries.h5 "
                "dimensions do not match "
                "velocity products"
            )

    vertical_factor = None
    incidence_deg = None
    vertical_prefix = None

    if vertical_enabled:
        incidence = _read_incidence(
            root,
            n_ps,
        )

        incidence_deg = np.rad2deg(
            incidence
        )

        base = (
            1.0
            / np.cos(
                incidence
            )
        )

        positive = (
            str(
                vertical_positive
            )
            .strip()
            .lower()
        )

        if positive == "up":
            vertical_factor = -base
            vertical_prefix = "VUP"
        elif positive == "down":
            vertical_factor = base
            vertical_prefix = "VDN"
        else:
            raise QgisExportError(
                "vertical_positive must "
                "be 'up' or 'down'"
            )

    fields: list[
        tuple[
            str,
            str,
        ]
    ] = [
        (
            "PS_ID",
            "INTEGER",
        ),
        (
            "LON",
            "REAL",
        ),
        (
            "LAT",
            "REAL",
        ),
        (
            "LOS_VEL_MM_YR",
            "REAL",
        ),
        (
            "LOS_RMS_MM",
            "REAL",
        ),
        (
            "LOS_END_MM_YR",
            "REAL",
        ),
        (
            "LOS_CUM_LAST_MM",
            "REAL",
        ),
    ]

    if vertical_enabled:
        fields.extend(
            [
                (
                    "INC_DEG",
                    "REAL",
                ),
                (
                    f"{vertical_prefix}"
                    "_VEL_MM_YR",
                    "REAL",
                ),
                (
                    f"{vertical_prefix}"
                    "_RMS_MM",
                    "REAL",
                ),
                (
                    f"{vertical_prefix}"
                    "_END_MM_YR",
                    "REAL",
                ),
                (
                    f"{vertical_prefix}"
                    "_CUM_LAST_MM",
                    "REAL",
                ),
            ]
        )

    for year in years.tolist():
        fields.extend(
            [
                (
                    f"LOS_VEL_"
                    f"{int(year)}",
                    "REAL",
                ),
                (
                    f"LOS_RMS_"
                    f"{int(year)}",
                    "REAL",
                ),
            ]
        )

        if vertical_enabled:
            fields.extend(
                [
                    (
                        f"{vertical_prefix}"
                        f"_VEL_{int(year)}",
                        "REAL",
                    ),
                    (
                        f"{vertical_prefix}"
                        f"_RMS_{int(year)}",
                        "REAL",
                    ),
                ]
            )

    for date in dates.tolist():
        fields.append(
            (
                f"LOS_{int(date)}",
                "REAL",
            )
        )

        if vertical_enabled:
            fields.append(
                (
                    f"{vertical_prefix}_"
                    f"{int(date)}",
                    "REAL",
                )
            )

    if len(fields) > 1900:
        raise QgisExportError(
            "GeoPackage field count "
            f"{len(fields)} is too large"
        )

    gpkg = (
        outroot
        / filename
    )

    if gpkg.exists():
        gpkg.unlink()

    conn = sqlite3.connect(
        str(gpkg)
    )

    conn.execute(
        "PRAGMA journal_mode=OFF"
    )
    conn.execute(
        "PRAGMA synchronous=OFF"
    )
    conn.execute(
        "PRAGMA temp_store=MEMORY"
    )
    conn.execute(
        "PRAGMA locking_mode=EXCLUSIVE"
    )
    conn.execute(
        "PRAGMA cache_size=-262144"
    )

    extent = (
        float(
            np.nanmin(
                lonlat[
                    :,
                    0,
                ]
            )
        ),
        float(
            np.nanmin(
                lonlat[
                    :,
                    1,
                ]
            )
        ),
        float(
            np.nanmax(
                lonlat[
                    :,
                    0,
                ]
            )
        ),
        float(
            np.nanmax(
                lonlat[
                    :,
                    1,
                ]
            )
        ),
    )

    rtree = _create_schema(
        conn,
        layer,
        fields,
        extent,
    )

    names = (
        [
            "geom",
        ]
        + [
            name
            for name, _
            in fields
        ]
    )

    sql = (
        f"INSERT INTO "
        f"{_qident(layer)} "
        f"({', '.join(_qident(n) for n in names)}) "
        f"VALUES "
        f"({', '.join(['?'] * len(names))})"
    )

    rtree_sql = (
        f"INSERT INTO "
        f"{_qident(rtree)} "
        "(id, minx, maxx, miny, maxy) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    chunk = max(
        1,
        int(
            chunk_rows
        ),
    )

    with h5py.File(
        time_series_path,
        "r",
    ) as h5:
        cumulative_ds = h5[
            "cumulative_mm"
        ]

        for start in range(
            0,
            n_ps,
            chunk,
        ):
            stop = min(
                n_ps,
                start + chunk,
            )

            los_ts = np.asarray(
                cumulative_ds[
                    start:stop,
                    :,
                ],
                dtype=np.float32,
            )

            if vertical_enabled:
                vertical_ts = (
                    los_ts.astype(
                        np.float64
                    )
                    * vertical_factor[
                        start:stop,
                        None,
                    ]
                )
            else:
                vertical_ts = None

            rows = []
            spatial = []

            for local_i, i in enumerate(
                range(
                    start,
                    stop,
                )
            ):
                vals = [
                    _gpkg_point_blob(
                        lonlat[
                            i,
                            0,
                        ],
                        lonlat[
                            i,
                            1,
                        ],
                    ),
                    i + 1,
                    float(
                        lonlat[
                            i,
                            0,
                        ]
                    ),
                    float(
                        lonlat[
                            i,
                            1,
                        ]
                    ),
                    float(
                        los_velocity[i]
                    ),
                    float(
                        los_rms[i]
                    ),
                    float(
                        los_endpoint[i]
                    ),
                    float(
                        los_cumulative_last[
                            i
                        ]
                    ),
                ]

                if vertical_enabled:
                    factor = float(
                        vertical_factor[
                            i
                        ]
                    )
                    abs_factor = abs(
                        factor
                    )

                    vals.extend(
                        [
                            float(
                                incidence_deg[
                                    i
                                ]
                            ),
                            float(
                                los_velocity[
                                    i
                                ]
                                * factor
                            ),
                            float(
                                los_rms[
                                    i
                                ]
                                * abs_factor
                            ),
                            float(
                                los_endpoint[
                                    i
                                ]
                                * factor
                            ),
                            float(
                                los_cumulative_last[
                                    i
                                ]
                                * factor
                            ),
                        ]
                    )

                for j in range(
                    years.size
                ):
                    vals.extend(
                        [
                            float(
                                annual_los[
                                    i,
                                    j,
                                ]
                            ),
                            float(
                                annual_los_rms[
                                    i,
                                    j,
                                ]
                            ),
                        ]
                    )

                    if vertical_enabled:
                        factor = float(
                            vertical_factor[
                                i
                            ]
                        )
                        abs_factor = abs(
                            factor
                        )

                        vals.extend(
                            [
                                float(
                                    annual_los[
                                        i,
                                        j,
                                    ]
                                    * factor
                                ),
                                float(
                                    annual_los_rms[
                                        i,
                                        j,
                                    ]
                                    * abs_factor
                                ),
                            ]
                        )

                for j in range(
                    dates.size
                ):
                    vals.append(
                        float(
                            los_ts[
                                local_i,
                                j,
                            ]
                        )
                    )

                    if vertical_enabled:
                        vals.append(
                            float(
                                vertical_ts[
                                    local_i,
                                    j,
                                ]
                            )
                        )

                rows.append(
                    tuple(
                        vals
                    )
                )

                fid = i + 1
                x = float(
                    lonlat[
                        i,
                        0,
                    ]
                )
                y = float(
                    lonlat[
                        i,
                        1,
                    ]
                )

                spatial.append(
                    (
                        fid,
                        x,
                        x,
                        y,
                        y,
                    )
                )

            if conn.in_transaction:
                conn.commit()

            conn.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                conn.executemany(
                    sql,
                    rows,
                )
                conn.executemany(
                    rtree_sql,
                    spatial,
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

            print(
                f"[GPKG] "
                f"{stop:,}/{n_ps:,} "
                f"({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )

    conn.execute(
        f"ANALYZE "
        f"{_qident(layer)}"
    )

    conn.commit()
    conn.close()

    manifest_path = (
        outroot
        / "engineering_manifest.json"
    )

    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    else:
        manifest = {}

    manifest[
        "qgis_geopackage"
    ] = {
        "file": gpkg.name,
        "layer": layer,
        "n_ps": int(n_ps),
        "n_epochs": int(
            dates.size
        ),
        "vertical_enabled": bool(
            vertical_enabled
        ),
        "vertical_positive": (
            str(
                vertical_positive
            )
            .strip()
            .lower()
            if vertical_enabled
            else None
        ),
        "los_positive": (
            "away from satellite"
        ),
        "vertical_formula": (
            "up=-LOS/cos(incidence), "
            "down=+LOS/cos(incidence)"
            if vertical_enabled
            else None
        ),
        "note": (
            "GeoPackage is a vector "
            "delivery product. Internal "
            "corrected_timeseries.h5 is "
            "not copied into the "
            "GeoPackage."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return gpkg


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
        "--filename",
        default=(
            "insar_all_epochs.gpkg"
        ),
    )

    ap.add_argument(
        "--layer",
        default="insar_ps",
    )

    ap.add_argument(
        "--vertical",
        action="store_true",
    )

    ap.add_argument(
        "--vertical-positive",
        choices=(
            "up",
            "down",
        ),
        default="down",
    )

    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=500,
    )

    args = ap.parse_args()

    path = export_qgis_geopackage(
        args.dataset_root,
        args.output_root,
        filename=args.filename,
        layer=args.layer,
        vertical_enabled=args.vertical,
        vertical_positive=(
            args.vertical_positive
        ),
        chunk_rows=args.chunk_rows,
    )

    print(
        "QGIS GeoPackage:",
        path,
    )


if __name__ == "__main__":
    main()
