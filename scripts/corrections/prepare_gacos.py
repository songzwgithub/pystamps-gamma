#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline.stage6_sbas import (
    _stage6_reference_indices,
    load_sbas_network,
)
from pystamps.pipeline import ported


class GacosError(RuntimeError):
    pass


def scalar(x, default=0.0):
    if x is None:
        return default
    a = np.asarray(x)
    if a.size == 0:
        return default
    return a.reshape(-1)[0]


def as_rows(x, nrow, name, dtype=None):
    a = np.squeeze(np.asarray(x))

    if a.ndim == 1 and a.size % nrow == 0:
        a = a.reshape(nrow, -1)

    if a.ndim != 2:
        raise GacosError(f"{name}: invalid shape {a.shape}")

    if a.shape[0] != nrow and a.shape[1] == nrow:
        a = a.T

    if a.shape[0] != nrow:
        raise GacosError(
            f"{name}: shape={a.shape}, expected rows={nrow}"
        )

    if dtype is not None:
        a = np.asarray(a, dtype=dtype)

    return a


def matlab_date(dn):
    dn = float(dn)
    n = int(math.floor(dn))
    frac = dn - n

    return (
        datetime.fromordinal(n)
        + timedelta(days=frac)
        - timedelta(days=366)
    )


def read_gamma_par(path):
    out = {}

    for line in Path(path).read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(":", 1)
        v = v.strip()

        if v:
            out[k.strip()] = v.split()[0]

    return out


def reflink_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        subprocess.run(
            [
                "cp",
                "--reflink=auto",
                "-p",
                str(src),
                str(dst),
            ],
            check=True,
        )
    except Exception:
        shutil.copy2(src, dst)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_lv_theta_ps(
    lv_file,
    dem_par,
    lonlat,
):
    """
    GAMMA look_vector output:
      lv_theta = look-vector elevation angle above horizontal.

    Input encoding for current GAMMA product:
      big-endian FLOAT32, radians.
    """

    p = read_gamma_par(dem_par)

    width = int(float(p["width"]))
    nlines = int(float(p["nlines"]))

    expected = (
        width
        * nlines
        * 4
    )

    actual = Path(lv_file).stat().st_size

    if actual != expected:
        raise GacosError(
            f"lv_theta size mismatch: "
            f"{actual} != {expected}"
        )

    lv = np.fromfile(
        lv_file,
        dtype=">f4",
    ).reshape(
        nlines,
        width,
    ).astype(np.float64)

    corner_lat = float(
        p["corner_lat"]
    )

    corner_lon = float(
        p["corner_lon"]
    )

    post_lat = float(
        p["post_lat"]
    )

    post_lon = float(
        p["post_lon"]
    )

    lats = (
        corner_lat
        + np.arange(
            nlines,
            dtype=np.float64,
        )
        * post_lat
    )

    lons = (
        corner_lon
        + np.arange(
            width,
            dtype=np.float64,
        )
        * post_lon
    )

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        lv = lv[::-1, :]

    if lons[0] > lons[-1]:
        lons = lons[::-1]
        lv = lv[:, ::-1]

    interp = RegularGridInterpolator(
        (lats, lons),
        lv,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack(
        (
            lonlat[:, 1],
            lonlat[:, 0],
        )
    )

    lv_ps = np.asarray(
        interp(pts),
        dtype=np.float64,
    )

    if np.any(~np.isfinite(lv_ps)):
        nbad = int(
            np.count_nonzero(
                ~np.isfinite(lv_ps)
            )
        )

        raise GacosError(
            f"{nbad} PS outside lv_theta coverage"
        )

    # lv_theta = elevation angle above horizontal
    incidence_rad = (
        np.pi / 2.0
        - lv_ps
    )

    # cos(incidence) = sin(lv_theta)
    cos_inc = np.sin(
        lv_ps
    )

    if np.min(cos_inc) <= 0:
        raise GacosError(
            "Invalid LOS projection factor"
        )

    return (
        lv_ps,
        incidence_rad,
        cos_inc,
    )


def sample_gacos(path, lonlat):
    """
    Bilinear sample GACOS geographic GeoTIFF at PS lon/lat.
    """

    with rasterio.open(path) as ds:

        arr = ds.read(
            1,
            masked=True,
        )

        arr = np.asarray(
            arr.filled(np.nan),
            dtype=np.float64,
        )

        if ds.crs is not None:

            if not ds.crs.is_geographic:
                raise GacosError(
                    f"{path}: GACOS CRS is not geographic: "
                    f"{ds.crs}"
                )

        else:

            b = ds.bounds

            if not (
                -180 <= b.left <= 180
                and -180 <= b.right <= 180
                and -90 <= b.bottom <= 90
                and -90 <= b.top <= 90
            ):
                raise GacosError(
                    f"{path}: no valid geographic CRS"
                )

        inv = ~ds.transform

        col, row = inv * (
            lonlat[:, 0],
            lonlat[:, 1],
        )

        # Convert from pixel-corner coordinates
        # to array-centre coordinates.
        col = np.asarray(
            col,
            dtype=np.float64,
        ) - 0.5

        row = np.asarray(
            row,
            dtype=np.float64,
        ) - 0.5

        val = map_coordinates(
            arr,
            [row, col],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    return np.asarray(
        val,
        dtype=np.float64,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--baseline",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--gacos-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--lv-theta",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--dem-par",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--out",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    started = time.perf_counter()

    baseline = (
        args.baseline
        .expanduser()
        .resolve()
    )

    gacos_dir = (
        args.gacos_dir
        .expanduser()
        .resolve()
    )

    out = (
        args.out
        .expanduser()
        .resolve()
    )

    print("=" * 96)
    print("GACOS ACQUISITION-SPACE CORRECTION")
    print("=" * 96)

    # ==========================================================
    # Read Stage6 baseline
    # ==========================================================

    ps = read_mat(
        baseline / "ps2.mat"
    )

    parms = read_mat(
        baseline / "parms.mat"
    )

    n_ps = int(
        round(
            float(
                scalar(ps["n_ps"])
            )
        )
    )

    n_image = int(
        round(
            float(
                scalar(ps["n_image"])
            )
        )
    )

    n_ifg = int(
        round(
            float(
                scalar(ps["n_ifg"])
            )
        )
    )

    master_ix = int(
        round(
            float(
                scalar(ps["master_ix"])
            )
        )
    )

    master0 = master_ix - 1

    day = np.asarray(
        ps["day"],
        dtype=np.float64,
    ).reshape(-1)

    lonlat = as_rows(
        ps["lonlat"],
        n_ps,
        "ps.lonlat",
        np.float64,
    )

    wavelength = float(
        scalar(
            parms["lambda"]
        )
    )

    dates = [
        matlab_date(x).strftime(
            "%Y%m%d"
        )
        for x in day
    ]

    ref_ix = np.asarray(
        _stage6_reference_indices(
            ps,
            parms,
            n_ps,
        ),
        dtype=np.int64,
    ).reshape(-1)

    if ref_ix.size == 0:
        raise GacosError(
            "Reference PS set is empty"
        )

    day_net, ifgday_ix, _, net_source = (
        load_sbas_network(
            baseline,
            n_ifg,
        )
    )

    ifgday_ix = np.asarray(
        ifgday_ix,
        dtype=np.int64,
    )

    if (
        ifgday_ix.shape
        != (n_ifg, 2)
    ):
        raise GacosError(
            f"ifgday_ix={ifgday_ix.shape}"
        )

    if not np.allclose(
        day_net,
        day,
    ):
        raise GacosError(
            "Network day vector mismatch"
        )

    # ==========================================================
    # Exact GACOS date mapping
    # ==========================================================

    gacos = {}

    for p in gacos_dir.glob(
        "*.ztd.tif"
    ):
        m = re.match(
            r"^(\d{8})\.ztd\.tif$",
            p.name,
        )

        if m:
            gacos[
                m.group(1)
            ] = p

    missing = sorted(
        set(dates)
        - set(gacos)
    )

    extra = sorted(
        set(gacos)
        - set(dates)
    )

    if missing:
        raise GacosError(
            f"Missing GACOS dates: {missing}"
        )

    print()
    print("DATA CONTRACT")
    print("-" * 96)
    print("n_ps              :", n_ps)
    print("n_image           :", n_image)
    print("n_ifg             :", n_ifg)
    print("master_ix         :", master_ix)
    print("master_date       :", dates[master0])
    print("reference PS      :", ref_ix.size)
    print("lambda            :", wavelength)
    print("network source    :", net_source)
    print("GACOS files       :", len(gacos))
    print("missing           :", len(missing))
    print("extra             :", len(extra))

    # ==========================================================
    # Look-vector geometry
    # ==========================================================

    (
        lv_theta,
        incidence,
        cos_inc,
    ) = load_lv_theta_ps(
        args.lv_theta,
        args.dem_par,
        lonlat,
    )

    inc_deg = np.rad2deg(
        incidence
    )

    lv_deg = np.rad2deg(
        lv_theta
    )

    print()
    print("GEOMETRY")
    print("-" * 96)

    print(
        "lv_theta deg q1/50/99 :",
        np.percentile(
            lv_deg,
            [1, 50, 99]
        ),
    )

    print(
        "incidence deg q1/50/99:",
        np.percentile(
            inc_deg,
            [1, 50, 99]
        ),
    )

    print(
        "cos incidence q1/50/99:",
        np.percentile(
            cos_inc,
            [1, 50, 99]
        ),
    )

    # ==========================================================
    # Prepare output branch
    # ==========================================================

    if out.exists():

        if not args.overwrite:
            raise GacosError(
                f"Output already exists: {out}"
            )

        shutil.rmtree(
            out
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    work = (
        out
        / "_gacos_work"
    )

    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stage6 products that are unchanged.
    copy_required = [
        "ps2.mat",
        "parms.mat",
        "bp2.mat",
        "rc2.mat",
        "ifgstd2.mat",
        "phuw_sb_res2.mat",
    ]

    copy_optional = [
        "stage6_sbas_debug.json",
    ]

    print()
    print("COPY STAGE6 BASELINE")
    print("-" * 96)

    for name in copy_required:

        src = baseline / name

        if not src.exists():
            raise GacosError(
                f"Missing baseline {name}"
            )

        reflink_copy(
            src,
            out / name,
        )

        print("copy:", name)

    for name in copy_optional:

        src = baseline / name

        if src.exists():
            reflink_copy(
                src,
                out / name,
            )

    # ==========================================================
    # Build atmospheric correction in acquisition space
    # ==========================================================

    phase_factor = (
        4.0
        * np.pi
        / wavelength
    )

    atm = np.memmap(
        work
        / "gacos_atm_phase_rel.f32",
        mode="w+",
        dtype=np.float32,
        shape=(
            n_ps,
            n_image,
        ),
    )

    # Absolute master ZTD / LOS phase
    z_master = sample_gacos(
        gacos[
            dates[master0]
        ],
        lonlat,
    )

    if np.any(
        ~np.isfinite(
            z_master
        )
    ):
        raise GacosError(
            "Invalid master GACOS coverage"
        )

    phi_master_abs = (
        phase_factor
        * z_master
        / cos_inc
    )

    epoch_stats = []

    print()
    print("=" * 96)
    print("BUILD GACOS ACQUISITION CORRECTION")
    print("=" * 96)

    for i, date in enumerate(
        dates
    ):

        ztd = sample_gacos(
            gacos[date],
            lonlat,
        )

        if np.any(
            ~np.isfinite(ztd)
        ):

            nbad = int(
                np.count_nonzero(
                    ~np.isfinite(ztd)
                )
            )

            raise GacosError(
                f"{date}: {nbad} invalid GACOS PS"
            )

        zmed = float(
            np.median(ztd)
        )

        if not (
            0.2
            < zmed
            < 5.0
        ):
            raise GacosError(
                f"{date}: suspicious ZTD median={zmed}"
            )

        phi_abs = (
            phase_factor
            * ztd
            / cos_inc
        )

        # Temporal reference:
        # atmospheric phase relative to StaMPS master.
        rel = (
            phi_abs
            - phi_master_abs
        )

        # Spatial reference:
        # identical reference region as Stage6/postprocess.
        rel -= np.mean(
            rel[
                ref_ix
            ]
        )

        if i == master0:
            rel[:] = 0.0

        atm[
            :,
            i
        ] = rel.astype(
            np.float32
        )

        # Equivalent apparent LOS displacement correction.
        los_mm = (
            -rel
            * wavelength
            * 1000.0
            / (
                4.0
                * np.pi
            )
        )

        qz = np.percentile(
            ztd,
            [1, 50, 99],
        )

        qp = np.percentile(
            rel,
            [1, 50, 99],
        )

        qmm = np.percentile(
            los_mm,
            [1, 50, 99],
        )

        epoch_stats.append(
            [
                date,
                qz[0],
                qz[1],
                qz[2],
                qp[0],
                qp[1],
                qp[2],
                qmm[0],
                qmm[1],
                qmm[2],
            ]
        )

        print(
            f"[GACOS] {i+1:3d}/{n_image} "
            f"{date} "
            f"LOSmm(q1/50/99)="
            f"{qmm[0]:8.3f}/"
            f"{qmm[1]:8.3f}/"
            f"{qmm[2]:8.3f}",
            flush=True,
        )

    atm.flush()

    master_corr_max = float(
        np.max(
            np.abs(
                atm[
                    :,
                    master0
                ]
            )
        )
    )

    ref_corr = np.mean(
        np.asarray(
            atm[
                ref_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    ref_corr_max = float(
        np.max(
            np.abs(
                ref_corr
            )
        )
    )

    print()
    print(
        "master correction max abs:",
        master_corr_max,
    )

    print(
        "reference mean max abs   :",
        ref_corr_max,
    )

    if master_corr_max != 0.0:
        raise GacosError(
            "Master GACOS correction is not zero"
        )

    if ref_corr_max > 1e-5:
        raise GacosError(
            "Reference mean GACOS correction is not zero"
        )

    # ==========================================================
    # Correct phuw2 acquisition-space phase
    # ==========================================================

    print()
    print("=" * 96)
    print("CORRECT phuw2.mat")
    print("=" * 96)

    sm_payload = read_mat(
        baseline
        / "phuw2.mat"
    )

    ph_sm = as_rows(
        sm_payload[
            "ph_uw"
        ],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )

    if ph_sm.shape[1] != n_image:
        raise GacosError(
            f"phuw2 shape={ph_sm.shape}"
        )

    ph_sm_gacos = (
        ph_sm.astype(
            np.float64
        )
        -
        np.asarray(
            atm,
            dtype=np.float64,
        )
    ).astype(
        np.float32
    )

    # Preserve exact master phase convention.
    ph_sm_gacos[
        :,
        master0
    ] = ph_sm[
        :,
        master0
    ]

    sm_payload[
        "ph_uw"
    ] = ph_sm_gacos

    write_mat(
        out / "phuw2.mat",
        sm_payload,
    )

    print(
        "written:",
        out / "phuw2.mat",
    )

    # ==========================================================
    # Correct phuw_sb2 consistently from acquisition correction
    # ==========================================================

    print()
    print("=" * 96)
    print("CORRECT phuw_sb2.mat")
    print("=" * 96)

    sb_payload = read_mat(
        baseline
        / "phuw_sb2.mat"
    )

    ph_sb = as_rows(
        sb_payload[
            "ph_uw"
        ],
        n_ps,
        "phuw_sb2.ph_uw",
        np.float32,
    )

    if ph_sb.shape[1] != n_ifg:
        raise GacosError(
            f"phuw_sb2 shape={ph_sb.shape}"
        )

    ph_sb_gacos = np.empty(
        ph_sb.shape,
        dtype=np.float32,
    )

    drop_ifg = set(
        int(v)
        for v in
        ported._normalize_drop_index(
            parms.get(
                "drop_ifg_index"
            )
        ).tolist()
    )

    for j in range(
        n_ifg
    ):

        jj = j + 1

        if jj in drop_ifg:

            ph_sb_gacos[
                :,
                j
            ] = 0.0

            continue

        i1 = int(
            ifgday_ix[
                j,
                0
            ]
        ) - 1

        i2 = int(
            ifgday_ix[
                j,
                1
            ]
        ) - 1

        atm_ifg = (
            np.asarray(
                atm[
                    :,
                    i2
                ],
                dtype=np.float64,
            )
            -
            np.asarray(
                atm[
                    :,
                    i1
                ],
                dtype=np.float64,
            )
        )

        ph_sb_gacos[
            :,
            j
        ] = (
            ph_sb[
                :,
                j
            ].astype(
                np.float64
            )
            -
            atm_ifg
        ).astype(
            np.float32
        )

        if (
            jj % 50 == 0
            or jj == n_ifg
        ):
            print(
                f"[SB] {jj}/{n_ifg}",
                flush=True,
            )

    sb_payload[
        "ph_uw"
    ] = ph_sb_gacos

    write_mat(
        out / "phuw_sb2.mat",
        sb_payload,
    )

    print(
        "written:",
        out / "phuw_sb2.mat",
    )

    # ==========================================================
    # Independent network consistency check
    #
    # GACOS correction of an IFG must equal:
    # correction(image2) - correction(image1)
    # ==========================================================

    print()
    print("=" * 96)
    print("SB / ACQUISITION NETWORK CONSISTENCY")
    print("=" * 96)

    ps_sample = np.linspace(
        0,
        n_ps - 1,
        min(
            4096,
            n_ps,
        ),
        dtype=np.int64,
    )

    retained = np.asarray(
        [
            j
            for j in range(n_ifg)
            if (j + 1)
            not in drop_ifg
        ],
        dtype=np.int64,
    )

    ifg_sample = retained[
        np.linspace(
            0,
            retained.size - 1,
            min(
                128,
                retained.size,
            ),
            dtype=np.int64,
        )
    ]

    max_error = 0.0
    ss = 0.0
    nn = 0

    for j in ifg_sample:

        i1 = int(
            ifgday_ix[
                j,
                0
            ]
        ) - 1

        i2 = int(
            ifgday_ix[
                j,
                1
            ]
        ) - 1

        delta_sb = (
            ph_sb_gacos[
                ps_sample,
                j
            ].astype(
                np.float64
            )
            -
            ph_sb[
                ps_sample,
                j
            ].astype(
                np.float64
            )
        )

        delta_sm = (
            (
                ph_sm_gacos[
                    ps_sample,
                    i2
                ].astype(
                    np.float64
                )
                -
                ph_sm[
                    ps_sample,
                    i2
                ].astype(
                    np.float64
                )
            )
            -
            (
                ph_sm_gacos[
                    ps_sample,
                    i1
                ].astype(
                    np.float64
                )
                -
                ph_sm[
                    ps_sample,
                    i1
                ].astype(
                    np.float64
                )
            )
        )

        e = (
            delta_sb
            - delta_sm
        )

        max_error = max(
            max_error,
            float(
                np.max(
                    np.abs(e)
                )
            ),
        )

        ss += float(
            np.sum(
                e * e
            )
        )

        nn += e.size

    rms_error = math.sqrt(
        ss / nn
    )

    print(
        "sample PS             :",
        ps_sample.size,
    )

    print(
        "sample retained IFGs  :",
        ifg_sample.size,
    )

    print(
        "max consistency error :",
        max_error,
        "rad",
    )

    print(
        "RMS consistency error :",
        rms_error,
        "rad",
    )

    if max_error > 1e-4:
        raise GacosError(
            "SB/acquisition GACOS consistency FAILED"
        )

    # Dropped columns must remain exactly zero.
    drop_max = 0.0

    for d in sorted(
        drop_ifg
    ):
        drop_max = max(
            drop_max,
            float(
                np.max(
                    np.abs(
                        ph_sb_gacos[
                            :,
                            d - 1
                        ]
                    )
                )
            ),
        )

    print(
        "dropped IFG max abs   :",
        drop_max,
    )

    if drop_max != 0.0:
        raise GacosError(
            "Dropped SB columns are not zero"
        )

    # ==========================================================
    # Save GACOS correction itself for audit/reproducibility
    # ==========================================================

    print()
    print("=" * 96)
    print("WRITE GACOS CORRECTION PRODUCT")
    print("=" * 96)

    with h5py.File(
        out
        / "gacos_ps_correction.h5",
        "w",
    ) as h5:

        h5.create_dataset(
            "date_yyyymmdd",
            data=np.asarray(
                dates,
                dtype="S8",
            ),
        )

        h5.create_dataset(
            "lv_theta_deg",
            data=lv_deg.astype(
                np.float32
            ),
            compression="gzip",
            compression_opts=1,
        )

        h5.create_dataset(
            "incidence_deg",
            data=inc_deg.astype(
                np.float32
            ),
            compression="gzip",
            compression_opts=1,
        )

        h5.create_dataset(
            "cos_incidence",
            data=cos_inc.astype(
                np.float32
            ),
            compression="gzip",
            compression_opts=1,
        )

        ds = h5.create_dataset(
            "atm_phase_rel_rad",
            shape=(
                n_ps,
                n_image,
            ),
            dtype="f4",
            chunks=(
                min(
                    16384,
                    n_ps,
                ),
                n_image,
            ),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        for r0 in range(
            0,
            n_ps,
            16384,
        ):

            r1 = min(
                n_ps,
                r0 + 16384,
            )

            ds[
                r0:r1,
                :
            ] = atm[
                r0:r1,
                :
            ]

        h5.create_dataset(
            "reference_ps_0based",
            data=ref_ix,
        )

        h5.attrs[
            "formula"
        ] = (
            "phi_atm = 4*pi/lambda * "
            "ZTD/sin(lv_theta); "
            "lv_theta is elevation angle above horizontal; "
            "relative to master and StaMPS spatial reference; "
            "ph_corrected = ph_observed - phi_atm_rel"
        )

        h5.attrs[
            "master_ix_1based"
        ] = master_ix

        h5.attrs[
            "master_date"
        ] = dates[
            master0
        ]

        h5.attrs[
            "lambda_m"
        ] = wavelength

    # Per-epoch correction statistics
    with (
        out
        / "gacos_epoch_stats.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "date",
                "ztd_q01_m",
                "ztd_q50_m",
                "ztd_q99_m",
                "phase_q01_rad",
                "phase_q50_rad",
                "phase_q99_rad",
                "equiv_los_q01_mm",
                "equiv_los_q50_mm",
                "equiv_los_q99_mm",
            ]
        )

        w.writerows(
            epoch_stats
        )

    # ==========================================================
    # Debug / lineage
    # ==========================================================

    duration = (
        time.perf_counter()
        - started
    )

    debug = {
        "status":
            "completed",

        "implementation":
            "GACOS_LVTHETA_STAMPS_PARITY_V2",

        "baseline":
            str(baseline),

        "output":
            str(out),

        "n_ps":
            n_ps,

        "n_image":
            n_image,

        "n_ifg":
            n_ifg,

        "master_ix":
            master_ix,

        "master_date":
            dates[
                master0
            ],

        "reference_ps":
            int(
                ref_ix.size
            ),

        "lambda_m":
            wavelength,

        "geometry": {
            "lv_theta_file":
                str(
                    args.lv_theta
                ),

            "definition":
                "LOS elevation angle above horizontal",

            "incidence_formula":
                "pi/2 - lv_theta",

            "cos_incidence_formula":
                "sin(lv_theta)",

            "incidence_deg_min":
                float(
                    np.min(
                        inc_deg
                    )
                ),

            "incidence_deg_median":
                float(
                    np.median(
                        inc_deg
                    )
                ),

            "incidence_deg_max":
                float(
                    np.max(
                        inc_deg
                    )
                ),
        },

        "drop_ifg_index":
            sorted(
                int(x)
                for x in drop_ifg
            ),

        "network_max_error_rad":
            max_error,

        "network_rms_error_rad":
            rms_error,

        "drop_column_max_abs":
            drop_max,

        "master_correction_max_abs":
            master_corr_max,

        "reference_mean_max_abs":
            ref_corr_max,

        "duration_sec":
            duration,
    }

    write_json(
        out
        / "gacos_prepare_debug.json",
        debug,
    )

    (
        out
        / "PARENT_BASELINE.txt"
    ).write_text(
        (
            "GACOS B branch derived from frozen Stage1-8 A baseline.\n"
            f"Parent: {baseline}\n\n"
            "Stage1-6 numerical solution is inherited.\n"
            "phuw2.mat and phuw_sb2.mat are GACOS-corrected.\n"
            "phuw_sb_res2 covariance/residual product is inherited because\n"
            "the GACOS correction is exactly acquisition-space/network-consistent.\n"
            "Stage7 and Stage8 must be re-estimated in this branch.\n"
        ),
        encoding="utf-8",
    )

    shutil.rmtree(
        work,
        ignore_errors=True,
    )

    print()
    print("=" * 96)
    print("GACOS B BRANCH: PREPARATION PASS")
    print("=" * 96)

    print(
        "output              :",
        out,
    )

    print(
        "master correction   :",
        master_corr_max,
    )

    print(
        "reference mean max  :",
        ref_corr_max,
    )

    print(
        "network max error   :",
        max_error,
        "rad",
    )

    print(
        "network RMS error   :",
        rms_error,
        "rad",
    )

    print(
        "drop columns max    :",
        drop_max,
    )

    print(
        "duration            :",
        f"{duration:.2f} s",
    )

    print()
    print(
        "NEXT: rerun Stage7 and Stage8 ONLY."
    )


if __name__ == "__main__":
    main()
