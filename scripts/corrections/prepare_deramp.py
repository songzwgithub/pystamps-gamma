#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat, write_mat


class DerampError(RuntimeError):
    pass


def _scalar(value):
    return np.asarray(value).reshape(-1)[0]


def _as_rows(value, nrow, name):
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise DerampError(f"{name}: unexpected shape {arr.shape}")
    if arr.shape[0] != nrow and arr.shape[1] == nrow:
        arr = arr.T
    if arr.shape[0] != nrow:
        raise DerampError(f"{name}: shape={arr.shape}, expected {nrow} rows")
    return arr


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "--reflink=auto", "-p", str(src), str(dst)], check=True)
    except Exception:
        shutil.copy2(src, dst)


def _deramp(phase, design, label):
    dtype_out = phase.dtype
    n_ps, n_col = phase.shape
    corrected = np.empty(phase.shape, dtype=dtype_out)
    coeff = np.full((n_col, 3), np.nan, dtype=np.float64)

    for k in range(n_col):
        values = np.asarray(phase[:, k], dtype=np.float64)
        good = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
        if np.count_nonzero(good) <= 5:
            corrected[:, k] = phase[:, k]
            continue
        beta, *_ = np.linalg.lstsq(design[good], values[good], rcond=None)
        output = values - design @ beta
        output[~good] = np.nan
        corrected[:, k] = output.astype(dtype_out, copy=False)
        coeff[k] = beta
        if k == 0 or (k + 1) % 25 == 0 or k + 1 == n_col:
            print(
                f"[{label}] {k+1}/{n_col} ax={beta[0]:+.6e} rad/km "
                f"by={beta[1]:+.6e} rad/km c={beta[2]:+.6e} rad",
                flush=True,
            )
    return corrected, coeff


def _process(src, dst, design, n_ps, label):
    payload = read_mat(src)
    if "ph_uw" not in payload:
        raise DerampError(f"{src}: ph_uw missing")
    phase = _as_rows(payload["ph_uw"], n_ps, f"{label}.ph_uw")
    corrected, coeff = _deramp(phase, design, label)
    payload["ph_uw"] = corrected
    write_mat(dst, payload)
    return coeff


def main():
    ap = argparse.ArgumentParser(
        description="Materialize StaMPS ps_deramp degree=1 on Stage-6 phuw_sb2 and phuw2."
    )
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source = args.input.expanduser().resolve()
    target = args.output.expanduser().resolve()

    if not source.exists():
        raise DerampError(f"Stage-6 input not found: {source}")
    if target.exists():
        if not args.overwrite:
            raise DerampError(f"Output exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    ps = read_mat(source / "ps2.mat")
    n_ps = int(round(float(_scalar(ps["n_ps"]))))
    xy = _as_rows(ps["xy"], n_ps, "ps2.xy").astype(np.float64)
    if xy.shape[1] < 3:
        raise DerampError("ps2.xy must contain [id, x, y]")

    # Exact StaMPS degree-1 ps_deramp design matrix.
    design = np.column_stack(
        (xy[:, 1] / 1000.0, xy[:, 2] / 1000.0, np.ones(n_ps, dtype=np.float64))
    )

    required = (
        "ps2.mat", "parms.mat", "bp2.mat", "rc2.mat", "ifgstd2.mat",
        "phuw_sb_res2.mat", "phuw_sb2.mat", "phuw2.mat",
    )
    for name in required:
        if not (source / name).exists():
            raise DerampError(f"Required Stage-6 file missing: {source / name}")

    for name in ("ps2.mat", "bp2.mat", "rc2.mat", "ifgstd2.mat", "phuw_sb_res2.mat"):
        _copy(source / name, target / name)
    for name in ("stage6_sbas_debug.json", "reference_selection.json"):
        if (source / name).exists():
            _copy(source / name, target / name)

    parms = read_mat(source / "parms.mat")
    parms["scla_deramp"] = "n"
    parms["sbas_deramp_mode"] = "none"
    write_mat(target / "parms.mat", parms)

    started = time.perf_counter()
    coeff_sb = _process(source / "phuw_sb2.mat", target / "phuw_sb2.mat", design, n_ps, "phuw_sb2")
    coeff_sm = _process(source / "phuw2.mat", target / "phuw2.mat", design, n_ps, "phuw2")

    np.savez_compressed(
        target / "native_deramp_coefficients.npz",
        phuw_sb2_coeff_rad_per_km=coeff_sb,
        phuw2_coeff_rad_per_km=coeff_sm,
    )

    report = {
        "status": "PASS",
        "method": "StaMPS ps_deramp degree=1",
        "input": str(source),
        "output": str(target),
        "n_ps": n_ps,
        "duration_sec": time.perf_counter() - started,
        "next_step": "Run Stage 7-8 on this branch; do not apply another deramp.",
    }
    (target / "native_deramp_prepare.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
