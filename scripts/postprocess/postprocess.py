#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from pystamps.io.mat import (
    read_mat,
    read_mat_variables,
)
from pystamps.pipeline.stage6_sbas import (
    _stage6_reference_indices,
)


class PostprocessError(RuntimeError):
    pass


def scalar(x, default=0.0):
    if x is None:
        return default
    a = np.asarray(x)
    if a.size == 0:
        return default
    return a.reshape(-1)[0]


def as_rows(
    value,
    nrow,
    name,
    dtype=None,
):
    a = np.squeeze(np.asarray(value))

    if a.ndim == 1:
        if nrow == 1:
            a = a.reshape(1, -1)
        elif a.size % nrow == 0:
            a = a.reshape(nrow, -1)

    if a.ndim != 2:
        raise PostprocessError(
            f"{name}: expected 2-D, got {a.shape}"
        )

    if (
        a.shape[0] != nrow
        and a.shape[1] == nrow
    ):
        a = a.T

    if a.shape[0] != nrow:
        raise PostprocessError(
            f"{name}: shape={a.shape}, "
            f"expected first dim={nrow}"
        )

    if dtype is not None:
        a = np.asarray(a, dtype=dtype)

    return a


def matlab_datenum_to_datetime(dn):
    dn = float(dn)

    whole = int(math.floor(dn))
    frac = dn - whole

    return (
        datetime.fromordinal(whole)
        + timedelta(days=frac)
        - timedelta(days=366)
    )


def gls_projector(A, C):
    """
    MATLAB lscov(A,Y,C) equivalent projector:

        beta = P @ Y
        P = inv(A' C^-1 A) A' C^-1

    Returned shape = n_param x n_obs.
    """
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if A.shape[0] != C.shape[0]:
        raise PostprocessError(
            f"GLS dimension mismatch: "
            f"A={A.shape}, C={C.shape}"
        )

    CiA = np.linalg.solve(
        C,
        A,
    )

    normal = (
        A.T @ CiA
    )

    return np.linalg.solve(
        normal,
        CiA.T,
    )


def one_based_indices(
    value,
    nmax,
):
    if value is None:
        return np.empty(
            0,
            dtype=np.int64,
        )

    a = np.asarray(value)

    if a.size == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    a = np.rint(
        a.reshape(-1)
    ).astype(np.int64)

    a = a[
        (a >= 1)
        & (a <= nmax)
    ]

    return np.unique(a)


def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--out",
        default=None,
        type=Path,
    )

    ap.add_argument(
        "--chunk-ps",
        type=int,
        default=16384,
    )

    ap.add_argument(
        "--annual-min-obs",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--annual-min-span-days",
        type=float,
        default=180.0,
    )

    args = ap.parse_args()

    root = (
        args.dataset
        .expanduser()
        .resolve()
    )

    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else root / "postprocess_parity"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.perf_counter()

    print("=" * 88)
    print("StaMPS PARITY POSTPROCESS")
    print("=" * 88)

    # ==========================================================
    # Load metadata
    # ==========================================================

    ps = read_mat(
        root / "ps2.mat"
    )

    parms = read_mat(
        root / "parms.mat"
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

    if day.size != n_image:
        raise PostprocessError(
            f"ps.day={day.size}, "
            f"n_image={n_image}"
        )

    lonlat = as_rows(
        ps["lonlat"],
        n_ps,
        "ps.lonlat",
        np.float64,
    )

    if lonlat.shape[1] != 2:
        raise PostprocessError(
            f"lonlat shape={lonlat.shape}"
        )

    wavelength_m = float(
        scalar(
            parms.get("lambda"),
            np.nan,
        )
    )

    if (
        not np.isfinite(wavelength_m)
        or wavelength_m <= 0
    ):
        raise PostprocessError(
            f"Invalid radar wavelength: "
            f"{wavelength_m}"
        )

    phase_to_mm = (
        -wavelength_m
        * 1000.0
        / (4.0 * np.pi)
    )

    # ==========================================================
    # Load Stage 6/7/8 matrices
    # ==========================================================

    print()
    print("Loading Stage 6/7/8 products...")

    phuw_payload = read_mat(
        root / "phuw2.mat"
    )

    ph_uw = as_rows(
        phuw_payload["ph_uw"],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )

    scla = read_mat(
        root / "scla2.mat"
    )

    ph_scla = as_rows(
        scla["ph_scla"],
        n_ps,
        "scla2.ph_scla",
        np.float32,
    )

    C_ps = np.asarray(
        scla["C_ps_uw"],
        dtype=np.float32,
    ).reshape(-1)

    if C_ps.size != n_ps:
        raise PostprocessError(
            "C_ps_uw length mismatch"
        )

    ph_ramp = scla.get("ph_ramp")

    if (
        ph_ramp is not None
        and np.asarray(ph_ramp).size > 0
    ):
        raise PostprocessError(
            "scla2.ph_ramp is non-empty. "
            "This script implements the current "
            "u-dms baseline, not u-dmos."
        )

    ph_scn = as_rows(
        read_mat_variables(
            root / "scn2.mat",
            ("ph_scn_slave",),
        )["ph_scn_slave"],
        n_ps,
        "scn2.ph_scn_slave",
        np.float64,
    )

    sm_cov = np.asarray(
        read_mat_variables(
            root / "phuw_sb_res2.mat",
            ("sm_cov",),
        )["sm_cov"],
        dtype=np.float64,
    )

    if sm_cov.shape != (
        n_image,
        n_image,
    ):
        raise PostprocessError(
            f"sm_cov={sm_cov.shape}"
        )

    # ==========================================================
    # Same reference region as Stage 6/7
    # ==========================================================

    ref_ix = _stage6_reference_indices(
        ps,
        parms,
        n_ps,
    )

    ref_ix = np.asarray(
        ref_ix,
        dtype=np.int64,
    ).reshape(-1)

    if ref_ix.size == 0:
        raise PostprocessError(
            "No reference PS"
        )

    print()
    print("n_ps                 :", n_ps)
    print("n_image              :", n_image)
    print("master_ix            :", master_ix)
    print("reference PS         :", ref_ix.size)
    print("lambda               :", wavelength_m)
    print(
        "phase->LOS mm factor :",
        phase_to_mm,
    )

    # ==========================================================
    # Dates
    # ==========================================================

    dates = [
        matlab_datenum_to_datetime(x)
        for x in day
    ]

    date_int = np.asarray(
        [
            int(d.strftime("%Y%m%d"))
            for d in dates
        ],
        dtype=np.int32,
    )

    years_all = np.asarray(
        [d.year for d in dates],
        dtype=np.int32,
    )

    # ==========================================================
    # Final corrected phase:
    #
    # phuw2 - ph_scla - C - ph_scn
    #
    # master = 0
    # then re-reference every epoch to same 62 reference PS.
    # ==========================================================

    ref_phase = (
        ph_uw[
            ref_ix,
            :
        ].astype(np.float64)
        -
        ph_scla[
            ref_ix,
            :
        ].astype(np.float64)
        -
        C_ps[
            ref_ix,
            None
        ].astype(np.float64)
        -
        ph_scn[
            ref_ix,
            :
        ].astype(np.float64)
    )

    ref_phase[
        :,
        master0
    ] = 0.0

    epoch_ref = np.nanmean(
        ref_phase,
        axis=0,
    )

    if not np.all(
        np.isfinite(epoch_ref)
    ):
        raise PostprocessError(
            "Non-finite reference epoch mean"
        )

    epoch_ref[
        master0
    ] = 0.0

    def corrected_phase_chunk(
        start,
        stop,
    ):
        y = (
            ph_uw[
                start:stop,
                :
            ].astype(np.float64)
            -
            ph_scla[
                start:stop,
                :
            ].astype(np.float64)
            -
            C_ps[
                start:stop,
                None
            ].astype(np.float64)
            -
            ph_scn[
                start:stop,
                :
            ].astype(np.float64)
        )

        y[
            :,
            master0
        ] = 0.0

        y -= epoch_ref[
            None,
            :
        ]

        y[
            :,
            master0
        ] = 0.0

        return y

    # ==========================================================
    # Images used for formal velocity inversion
    # ==========================================================

    unwrap_sm = one_based_indices(
        phuw_payload.get(
            "unwrap_ifg_index_sm"
        ),
        n_image,
    )

    if unwrap_sm.size == 0:
        unwrap_sm = np.arange(
            1,
            n_image + 1,
            dtype=np.int64,
        )

    scla_drop = one_based_indices(
        parms.get(
            "scla_drop_index"
        ),
        n_image,
    )

    if scla_drop.size:
        unwrap_sm = np.setdiff1d(
            unwrap_sm,
            scla_drop,
        )

    unwrap_sm = np.setdiff1d(
        unwrap_sm,
        np.asarray(
            [master_ix],
            dtype=np.int64,
        ),
    )

    img0 = unwrap_sm - 1

    if img0.size < 3:
        raise PostprocessError(
            "Too few acquisition epochs "
            "for velocity inversion"
        )

    C_full = sm_cov[
        np.ix_(
            img0,
            img0,
        )
    ]

    t_full_yr = (
        day[img0]
        - day[master0]
    ) / 365.25

    A_full = np.column_stack(
        (
            np.ones(
                img0.size,
                dtype=np.float64,
            ),
            t_full_yr,
        )
    )

    P_full = gls_projector(
        A_full,
        C_full,
    )

    full_cond = float(
        np.linalg.cond(
            C_full
        )
    )

    print()
    print(
        "velocity epochs       :",
        img0.size,
    )
    print(
        "sm_cov condition      :",
        full_cond,
    )

    # ==========================================================
    # Annual inversion setup
    # ==========================================================

    annual_models = []

    unique_years = sorted(
        set(
            int(years_all[i])
            for i in img0
        )
    )

    print()
    print("Annual velocity eligibility:")

    for year in unique_years:

        idx = np.asarray(
            [
                i
                for i in img0
                if dates[i].year == year
            ],
            dtype=np.int64,
        )

        nobs = idx.size

        span = (
            float(
                day[idx[-1]]
                - day[idx[0]]
            )
            if nobs >= 2
            else 0.0
        )

        valid = (
            nobs
            >= args.annual_min_obs
            and span
            >= args.annual_min_span_days
        )

        print(
            f"  {year}: "
            f"n={nobs:3d}, "
            f"span={span:7.1f} d, "
            f"{'USE' if valid else 'SKIP'}"
        )

        if not valid:
            continue

        # covariance indices are acquisition indices
        C_y = sm_cov[
            np.ix_(
                idx,
                idx,
            )
        ]

        t_y = (
            day[idx]
            - day[idx[0]]
        ) / 365.25

        A_y = np.column_stack(
            (
                np.ones(
                    nobs,
                    dtype=np.float64,
                ),
                t_y,
            )
        )

        P_y = gls_projector(
            A_y,
            C_y,
        )

        annual_models.append(
            {
                "year": year,
                "idx": idx,
                "nobs": int(nobs),
                "span_days": span,
                "t": t_y,
                "P": P_y,
            }
        )

    annual_years = np.asarray(
        [
            x["year"]
            for x in annual_models
        ],
        dtype=np.int32,
    )

    ny = len(annual_models)

    # ==========================================================
    # Allocate summary arrays
    # ==========================================================

    velocity = np.full(
        n_ps,
        np.nan,
        dtype=np.float32,
    )

    full_rms = np.full(
        n_ps,
        np.nan,
        dtype=np.float32,
    )

    endpoint_velocity = np.full(
        n_ps,
        np.nan,
        dtype=np.float32,
    )

    cumulative_last = np.full(
        n_ps,
        np.nan,
        dtype=np.float32,
    )

    annual_velocity = np.full(
        (n_ps, ny),
        np.nan,
        dtype=np.float32,
    )

    annual_rms = np.full(
        (n_ps, ny),
        np.nan,
        dtype=np.float32,
    )

    # ==========================================================
    # HDF5 time-series product
    # ==========================================================

    h5_path = (
        out
        / "corrected_timeseries.h5"
    )

    if h5_path.exists():
        h5_path.unlink()

    h5 = h5py.File(
        h5_path,
        "w",
    )

    h5.create_dataset(
        "lonlat",
        data=lonlat,
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )

    h5.create_dataset(
        "day",
        data=day,
    )

    h5.create_dataset(
        "date_yyyymmdd",
        data=date_int,
    )

    h5.create_dataset(
        "master_ix_1based",
        data=np.asarray(
            master_ix,
            dtype=np.int32,
        ),
    )

    h5.create_dataset(
        "reference_ps_0based",
        data=ref_ix.astype(
            np.int64
        ),
    )

    d_los = h5.create_dataset(
        "los_master_ref_mm",
        shape=(n_ps, n_image),
        dtype="f4",
        chunks=(
            min(
                args.chunk_ps,
                n_ps,
            ),
            n_image,
        ),
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )

    d_cum = h5.create_dataset(
        "cumulative_mm",
        shape=(n_ps, n_image),
        dtype="f4",
        chunks=(
            min(
                args.chunk_ps,
                n_ps,
            ),
            n_image,
        ),
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )

    h5.attrs[
        "formula_phase"
    ] = (
        "phuw2 - scla2.ph_scla "
        "- scla2.C_ps_uw "
        "- scn2.ph_scn_slave"
    )

    h5.attrs[
        "phase_to_los_mm"
    ] = phase_to_mm

    h5.attrs[
        "sign_convention"
    ] = (
        "positive LOS = motion toward satellite; "
        "negative LOS = motion away from satellite"
    )

    h5.attrs[
        "cumulative_reference"
    ] = (
        "first acquisition = 0 mm"
    )

    # ==========================================================
    # Chunk processing
    # ==========================================================

    total_span_year = (
        day[-1]
        - day[0]
    ) / 365.25

    if total_span_year <= 0:
        raise PostprocessError(
            "Invalid acquisition time span"
        )

    print()
    print("=" * 88)
    print("COMPUTING FINAL PRODUCTS")
    print("=" * 88)

    chunk = max(
        256,
        int(args.chunk_ps),
    )

    for start in range(
        0,
        n_ps,
        chunk,
    ):

        stop = min(
            n_ps,
            start + chunk,
        )

        ph = corrected_phase_chunk(
            start,
            stop,
        )

        # ------------------------------------------------------
        # Master-relative LOS displacement
        # ------------------------------------------------------

        los = (
            ph
            * phase_to_mm
        )

        # ------------------------------------------------------
        # First-acquisition cumulative displacement
        # ------------------------------------------------------

        cum = (
            los
            - los[:, 0:1]
        )

        d_los[
            start:stop,
            :
        ] = los.astype(
            np.float32
        )

        d_cum[
            start:stop,
            :
        ] = cum.astype(
            np.float32
        )

        cumulative_last[
            start:stop
        ] = cum[:, -1].astype(
            np.float32
        )

        endpoint_velocity[
            start:stop
        ] = (
            cum[:, -1]
            / total_span_year
        ).astype(np.float32)

        # ------------------------------------------------------
        # Full-period formal GLS velocity
        # ------------------------------------------------------

        Y = ph[
            :,
            img0
        ]

        beta = (
            Y
            @ P_full.T
        )

        slope_rad_yr = (
            beta[:, 1]
        )

        velocity[
            start:stop
        ] = (
            slope_rad_yr
            * phase_to_mm
        ).astype(np.float32)

        pred = (
            beta[:, 0, None]
            +
            beta[:, 1, None]
            * t_full_yr[
                None,
                :
            ]
        )

        resid = (
            Y - pred
        )

        full_rms[
            start:stop
        ] = (
            np.sqrt(
                np.mean(
                    resid * resid,
                    axis=1,
                )
            )
            * abs(
                phase_to_mm
            )
        ).astype(np.float32)

        # ------------------------------------------------------
        # Annual GLS velocities
        # ------------------------------------------------------

        for j, model in enumerate(
            annual_models
        ):

            idx = model["idx"]
            t_y = model["t"]
            P_y = model["P"]

            Yy = ph[
                :,
                idx
            ]

            by = (
                Yy
                @ P_y.T
            )

            annual_velocity[
                start:stop,
                j
            ] = (
                by[:, 1]
                * phase_to_mm
            ).astype(np.float32)

            pred_y = (
                by[:, 0, None]
                +
                by[:, 1, None]
                * t_y[None, :]
            )

            ry = (
                Yy - pred_y
            )

            annual_rms[
                start:stop,
                j
            ] = (
                np.sqrt(
                    np.mean(
                        ry * ry,
                        axis=1,
                    )
                )
                * abs(
                    phase_to_mm
                )
            ).astype(np.float32)

        print(
            f"[POSTPROCESS] "
            f"{stop:,}/{n_ps:,} "
            f"({100.0*stop/n_ps:.1f}%)",
            flush=True,
        )

    h5.flush()
    h5.close()

    # ==========================================================
    # Save compact NPZ products
    # ==========================================================

    np.savez_compressed(
        out / "velocity_full.npz",
        lonlat=lonlat,
        velocity_mm_yr=velocity,
        residual_rms_mm=full_rms,
        endpoint_velocity_mm_yr=endpoint_velocity,
        cumulative_last_mm=cumulative_last,
        n_obs=np.asarray(
            img0.size,
            dtype=np.int32,
        ),
        master_ix_1based=np.asarray(
            master_ix,
            dtype=np.int32,
        ),
        wavelength_m=np.asarray(
            wavelength_m,
            dtype=np.float64,
        ),
    )

    np.savez_compressed(
        out / "annual_velocity.npz",
        lonlat=lonlat,
        years=annual_years,
        velocity_mm_yr=annual_velocity,
        residual_rms_mm=annual_rms,
        n_obs=np.asarray(
            [
                x["nobs"]
                for x in annual_models
            ],
            dtype=np.int32,
        ),
        span_days=np.asarray(
            [
                x["span_days"]
                for x in annual_models
            ],
            dtype=np.float64,
        ),
    )

    # ==========================================================
    # dates.csv
    # ==========================================================

    with (
        out / "dates.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "image_index_1based",
                "date",
                "matlab_day",
                "master",
            ]
        )

        for i, d in enumerate(dates):

            w.writerow(
                [
                    i + 1,
                    d.strftime("%Y-%m-%d"),
                    f"{day[i]:.10f}",
                    1 if i == master0 else 0,
                ]
            )

    # ==========================================================
    # velocity_full.csv
    # ==========================================================

    print()
    print("Writing velocity_full.csv ...")

    with (
        out / "velocity_full.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "lon",
                "lat",
                "velocity_mm_yr",
                "residual_rms_mm",
                "endpoint_velocity_mm_yr",
                "cumulative_last_mm",
            ]
        )

        for i in range(n_ps):

            w.writerow(
                [
                    f"{lonlat[i,0]:.8f}",
                    f"{lonlat[i,1]:.8f}",
                    f"{velocity[i]:.6f}",
                    f"{full_rms[i]:.6f}",
                    f"{endpoint_velocity[i]:.6f}",
                    f"{cumulative_last[i]:.6f}",
                ]
            )

    # ==========================================================
    # annual_velocity.csv
    # ==========================================================

    print("Writing annual_velocity.csv ...")

    header = [
        "lon",
        "lat",
    ]

    for model in annual_models:
        y = model["year"]

        header += [
            f"velocity_{y}_mm_yr",
            f"rms_{y}_mm",
        ]

    with (
        out / "annual_velocity.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)
        w.writerow(header)

        for i in range(n_ps):

            row = [
                f"{lonlat[i,0]:.8f}",
                f"{lonlat[i,1]:.8f}",
            ]

            for j in range(ny):
                row += [
                    f"{annual_velocity[i,j]:.6f}",
                    f"{annual_rms[i,j]:.6f}",
                ]

            w.writerow(row)

    # ==========================================================
    # Debug / summary
    # ==========================================================

    ref_velocity_mean = float(
        np.nanmean(
            velocity[ref_ix]
        )
    )

    velocity_finite = (
        np.isfinite(velocity)
    )

    corr_endpoint = float(
        np.corrcoef(
            velocity[
                velocity_finite
            ],
            endpoint_velocity[
                velocity_finite
            ],
        )[0, 1]
    )

    duration = (
        time.perf_counter()
        - started
    )

    debug = {
        "status":
            "completed",

        "dataset":
            str(root),

        "output_dir":
            str(out),

        "n_ps":
            n_ps,

        "n_image":
            n_image,

        "master_ix":
            master_ix,

        "reference_ps":
            int(ref_ix.size),

        "lambda_m":
            wavelength_m,

        "phase_to_los_mm":
            phase_to_mm,

        "full_velocity_epochs":
            int(img0.size),

        "full_sm_cov_condition":
            full_cond,

        "total_span_year":
            float(total_span_year),

        "annual_min_obs":
            int(args.annual_min_obs),

        "annual_min_span_days":
            float(
                args.annual_min_span_days
            ),

        "annual_years":
            [
                {
                    "year":
                        int(x["year"]),
                    "n_obs":
                        int(x["nobs"]),
                    "span_days":
                        float(
                            x["span_days"]
                        ),
                }
                for x in annual_models
            ],

        "reference_mean_velocity_mm_yr":
            ref_velocity_mean,

        "gls_vs_endpoint_correlation":
            corr_endpoint,

        "outputs": [
            "corrected_timeseries.h5",
            "velocity_full.npz",
            "annual_velocity.npz",
            "velocity_full.csv",
            "annual_velocity.csv",
            "dates.csv",
        ],

        "duration_sec":
            duration,
    }

    write_json(
        out
        / "postprocess_debug.json",
        debug,
    )

    print()
    print("=" * 88)
    print("POSTPROCESS COMPLETE")
    print("=" * 88)

    print(
        "corrected_timeseries.h5 :",
        f"{n_ps} PS × {n_image} epochs",
    )

    print(
        "full velocity PS        :",
        int(
            np.count_nonzero(
                np.isfinite(velocity)
            )
        ),
    )

    print(
        "annual years            :",
        annual_years.tolist(),
    )

    print(
        "reference mean velocity :",
        f"{ref_velocity_mean:.6f} mm/yr",
    )

    print(
        "GLS/endpoint correlation:",
        f"{corr_endpoint:.6f}",
    )

    print(
        "duration                :",
        f"{duration:.2f} s",
    )


if __name__ == "__main__":
    main()
