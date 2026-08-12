from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import spatial

from pystamps.io.mat import read_mat, read_mat_variables, write_mat
from pystamps.pipeline.stage6_sbas import load_sbas_network


class Stage8SbasError(RuntimeError):
    """Raised when the SBAS-aware Stage 8 cannot continue."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value) if value is not None else np.asarray([])
    return float(arr.reshape(-1)[0]) if arr.size else float(default)


def _as_matrix(value: Any, rows: int, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise Stage8SbasError(f"{name} must be 2-D, got {arr.shape}")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise Stage8SbasError(f"{name} shape {arr.shape}; expected first dimension {rows}")
    return np.asarray(arr, dtype=dtype)


def _network_matrix(n_image: int, ifgday_ix: np.ndarray) -> np.ndarray:
    G = np.zeros((ifgday_ix.shape[0], n_image), dtype=np.float32)
    rows = np.arange(ifgday_ix.shape[0], dtype=np.int64)
    G[rows, ifgday_ix[:, 0] - 1] = -1.0
    G[rows, ifgday_ix[:, 1] - 1] = 1.0
    return G


def _corrected_chunk(
    ph_sm: np.ndarray,
    ph_scla: np.ndarray,
    c_ps: np.ndarray,
    ph_ramp: np.ndarray | None,
    start: int,
    stop: int,
    reference_phase: np.ndarray,
) -> np.ndarray:
    y = np.asarray(ph_sm[start:stop, :], dtype=np.float64)
    y -= np.asarray(ph_scla[start:stop, :], dtype=np.float64)
    y -= np.asarray(c_ps[start:stop], dtype=np.float64)[:, None]
    if ph_ramp is not None:
        y -= np.asarray(ph_ramp[start:stop, :], dtype=np.float64)
    y -= reference_phase[None, :]
    return y


def stage8_sbas_filter_scn(
    dataset_root: Path,
    backend: str = "auto",
    chunk_edges: int = 0,
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    triangle_path: str | None = None,
    snaphu_path: str | None = None,
) -> str:
    """
    SBAS-aware Stage 8.

    It uses the acquisition-space phase reconstructed by Stage 7, subtracts
    SCLA/C/ramp, applies Gaussian temporal high-pass filtering and a scalable
    k-nearest-neighbour Gaussian spatial low-pass approximation, then writes
    the final acquisition time series and mean LOS velocity.
    """

    del backend, chunk_edges, enable_mat_cache, io_workers, mat_cache, triangle_path, snaphu_path
    from pystamps.pipeline import ported

    root = Path(dataset_root).expanduser().resolve()
    started = time.perf_counter()
    required = ("ps2.mat", "phuw_sm2.mat", "scla_smooth2.mat")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise Stage8SbasError(f"Missing Stage 8 SBAS inputs: {', '.join(missing)}")

    ps2 = read_mat(root / "ps2.mat")
    n_ps = int(round(_scalar(ps2.get("n_ps"), 0)))
    if n_ps <= 0:
        raise Stage8SbasError("ps2.mat missing valid n_ps")

    ph_sm_payload = read_mat_variables(root / "phuw_sm2.mat", ("ph_uw", "day"))
    ph_sm = _as_matrix(ph_sm_payload["ph_uw"], n_ps, "phuw_sm2.ph_uw", np.float32)
    n_ps, n_image = ph_sm.shape
    day = np.asarray(ph_sm_payload.get("day", ps2.get("day")), dtype=np.float64).reshape(-1)
    if day.size != n_image:
        raise Stage8SbasError(f"Acquisition day count {day.size}; expected {n_image}")

    scla_payload = read_mat_variables(
        root / "scla_smooth2.mat",
        ("ph_scla", "C_ps_uw", "ph_ramp", "reference_image_ix"),
    )
    ph_scla = _as_matrix(
        scla_payload["ph_scla"], n_ps, "scla_smooth2.ph_scla", np.float32
    )
    if ph_scla.shape[1] != n_image:
        raise Stage8SbasError(
            f"scla_smooth2.ph_scla has {ph_scla.shape[1]} columns; expected {n_image}"
        )
    c_ps = np.asarray(scla_payload.get("C_ps_uw", np.zeros(n_ps)), dtype=np.float32).reshape(-1)
    if c_ps.size != n_ps:
        raise Stage8SbasError("scla_smooth2.C_ps_uw length does not match n_ps")
    ramp_raw = scla_payload.get("ph_ramp")
    ph_ramp = (
        _as_matrix(ramp_raw, n_ps, "scla_smooth2.ph_ramp", np.float32)
        if ramp_raw is not None and np.asarray(ramp_raw).size == n_ps * n_image
        else None
    )

    parms = read_mat(root / "parms.mat") if (root / "parms.mat").exists() else {}
    if ported._mat_text(parms.get("small_baseline_flag", "n"), "n").lower() != "y":
        raise Stage8SbasError("Stage 8 SBAS requires small_baseline_flag='y'")

    master_ix = int(round(_scalar(scla_payload.get("reference_image_ix"), _scalar(ps2.get("master_ix"), 1))))
    if master_ix < 1 or master_ix > n_image:
        master_ix = 1
    reference_image = master_ix - 1

    chunk = int(chunk_ps) if int(chunk_ps) > 0 else int(
        os.environ.get("PYSTAMPS_SBAS_STAGE8_CHUNK_PS", "1024")
    )
    chunk = max(128, chunk)
    spatial_chunk = max(
        64,
        int(os.environ.get("PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS", "256")),
    )
    k_neighbors = max(
        4,
        int(os.environ.get("PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS", "32")),
    )
    spatial_enabled = os.environ.get("PYSTAMPS_SBAS_STAGE8_SPATIAL_FILTER", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }

    ref_ps = np.asarray(ported._select_reference_ps(ps2, parms), dtype=np.int64).reshape(-1)
    if ref_ps.size:
        ref_values = np.asarray(ph_sm[ref_ps, :], dtype=np.float64)
        ref_values -= np.asarray(ph_scla[ref_ps, :], dtype=np.float64)
        ref_values -= np.asarray(c_ps[ref_ps], dtype=np.float64)[:, None]
        if ph_ramp is not None:
            ref_values -= np.asarray(ph_ramp[ref_ps, :], dtype=np.float64)
        reference_phase = np.nanmedian(ref_values, axis=0)
        reference_phase[~np.isfinite(reference_phase)] = 0.0
    else:
        reference_phase = np.zeros(n_image, dtype=np.float64)

    time_win = float(_scalar(parms.get("scn_time_win"), 365.0))
    time_win = max(time_win, 1.0e-6)
    time_diff = day[:, None] - day[None, :]
    temporal_weights = np.exp(-(time_diff * time_diff) / (2.0 * time_win * time_win))
    temporal_weights[:, reference_image] = 0.0
    row_sum = np.sum(temporal_weights, axis=1, keepdims=True)
    zero_rows = row_sum[:, 0] <= 0
    if np.any(zero_rows):
        temporal_weights[zero_rows, :] = 0.0
        temporal_weights[zero_rows, np.flatnonzero(zero_rows)] = 1.0
        row_sum = np.sum(temporal_weights, axis=1, keepdims=True)
    temporal_weights /= row_sum

    work = root / "_stage8_sbas_work"
    work.mkdir(parents=True, exist_ok=True)
    ph_hpt = np.memmap(
        work / "ph_hpt.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )

    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        corrected = _corrected_chunk(
            ph_sm, ph_scla, c_ps, ph_ramp, start, stop, reference_phase
        )
        low_time = corrected @ temporal_weights.T
        ph_hpt[start:stop, :] = (corrected - low_time).astype(np.float32)
        ph_hpt.flush()
        print(
            f"[STAGE8_SBAS][TEMPORAL] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    ph_scn = np.memmap(
        work / "ph_scn.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    if spatial_enabled:
        xy = np.asarray(ps2.get("xy"), dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] != n_ps or xy.shape[1] < 3:
            raise Stage8SbasError("ps2.xy is required for SBAS Stage 8 spatial filtering")
        coords = np.asarray(xy[:, 1:3], dtype=np.float64)
        wavelength = max(float(_scalar(parms.get("scn_wavelength"), 100.0)), 1.0e-6)
        radius = wavelength * 4.0
        k_use = min(k_neighbors, n_ps)
        tree = spatial.cKDTree(coords)
        distances, neighbours = tree.query(
            coords,
            k=k_use,
            distance_upper_bound=radius,
            workers=-1,
        )
        if k_use == 1:
            distances = distances[:, None]
            neighbours = neighbours[:, None]
        invalid_neighbour = neighbours >= n_ps
        neighbour_weights = np.exp(
            -(distances * distances) / (2.0 * wavelength * wavelength)
        )
        neighbour_weights[~np.isfinite(neighbour_weights)] = 0.0
        neighbour_weights[invalid_neighbour] = 0.0
        neighbours_safe = neighbours.copy()
        neighbours_safe[invalid_neighbour] = 0

        for start in range(0, n_ps, spatial_chunk):
            stop = min(start + spatial_chunk, n_ps)
            idx = neighbours_safe[start:stop, :]
            w = neighbour_weights[start:stop, :]
            values = np.asarray(ph_hpt[idx, :], dtype=np.float64)
            finite = np.isfinite(values)
            weighted = np.where(finite, values, 0.0) * w[:, :, None]
            denom = np.sum(w[:, :, None] * finite, axis=1)
            smooth = np.divide(
                np.sum(weighted, axis=1),
                denom,
                out=np.zeros((stop - start, n_image), dtype=np.float64),
                where=denom > 0,
            )
            ph_scn[start:stop, :] = smooth.astype(np.float32)
            ph_scn.flush()
            print(
                f"[STAGE8_SBAS][SPATIAL] {stop}/{n_ps} "
                f"({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )
    else:
        ph_scn[:] = 0.0
        ph_scn.flush()
        wavelength = float(_scalar(parms.get("scn_wavelength"), 100.0))
        radius = wavelength * 4.0
        k_use = 0

    if ref_ps.size:
        ref_scn = np.nanmean(np.asarray(ph_scn[ref_ps, :], dtype=np.float64), axis=0)
        ref_scn[~np.isfinite(ref_scn)] = 0.0
    else:
        ref_scn = np.asarray(ph_scn[0, :], dtype=np.float64)
        ref_scn[~np.isfinite(ref_scn)] = 0.0

    final_phase = np.memmap(
        work / "ph_final.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    phase_rate = np.full(n_ps, np.nan, dtype=np.float32)
    intercept = np.full(n_ps, np.nan, dtype=np.float32)
    ps_temporal_rms = np.full(n_ps, np.nan, dtype=np.float32)

    time_rel = day - float(day[reference_image])
    time_center = time_rel - float(np.mean(time_rel))
    time_denom = float(np.sum(time_center * time_center))
    if time_denom <= 0:
        raise Stage8SbasError("Acquisition times do not span a non-zero interval")

    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        corrected = _corrected_chunk(
            ph_sm, ph_scla, c_ps, ph_ramp, start, stop, reference_phase
        )
        scn = np.asarray(ph_scn[start:stop, :], dtype=np.float64) - ref_scn[None, :]
        ph_scn[start:stop, :] = scn.astype(np.float32)
        final = corrected - scn
        final_phase[start:stop, :] = final.astype(np.float32)

        valid = np.all(np.isfinite(final), axis=1)
        indices = np.arange(start, stop, dtype=np.int64)[valid]
        if indices.size:
            final_valid = final[valid, :]
            slopes = (final_valid @ time_center) / time_denom
            means = np.mean(final_valid, axis=1)
            ints = means - slopes * float(np.mean(time_rel))
            phase_rate[indices] = slopes.astype(np.float32)
            intercept[indices] = ints.astype(np.float32)
            trend = ints[:, None] + slopes[:, None] * time_rel[None, :]
            ps_temporal_rms[indices] = np.sqrt(
                np.mean((final_valid - trend) ** 2, axis=1)
            ).astype(np.float32)

        final_phase.flush()
        ph_scn.flush()
        print(
            f"[STAGE8_SBAS][FINAL] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    wavelength_radar = float(_scalar(parms.get("lambda"), 0.0555))
    velocity_rad_yr = phase_rate.astype(np.float64) * 365.25
    velocity_mm_yr = -velocity_rad_yr * wavelength_radar / (4.0 * math.pi) * 1000.0
    write_mat(
        root / "mean_v.mat",
        {
            "m": np.vstack((intercept, phase_rate)).astype(np.float32),
            "mean_v": ported._matlab_col(phase_rate, np.float32),
            "phase_rate_rad_day": ported._matlab_col(phase_rate, np.float32),
            "phase_rate_rad_yr": ported._matlab_col(
                velocity_rad_yr.astype(np.float32), np.float32
            ),
            "velocity_mm_yr": ported._matlab_col(
                velocity_mm_yr.astype(np.float32), np.float32
            ),
            "temporal_rms_rad": ported._matlab_col(ps_temporal_rms, np.float32),
            "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
            "phase_to_los_sign": np.asarray(-1.0, dtype=np.float32),
        },
    )

    _day, ifgday_ix, _bperp, network_source = load_sbas_network(
        root,
        int(np.asarray(read_mat(root / "ps2.mat").get("n_ifg", 0)).reshape(-1)[0]),
    )
    ifgday_ix = np.asarray(ifgday_ix, dtype=np.int64)
    G = _network_matrix(n_image, ifgday_ix)
    write_mat(
        root / "uw_space_time.mat",
        {
            "ph_uw_ts": final_phase,
            "ph_scn": ph_scn,
            "day": ported._matlab_col(day.astype(np.float64), np.float64),
            "G": G,
            "ifgday_ix": ifgday_ix.astype(np.int32),
            "temporal_weight": temporal_weights.astype(np.float32),
            "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
        },
    )

    _write_json(
        root / "stage8_sbas_debug.json",
        {
            "status": "completed",
            "method": "acquisition-space SCLA correction plus temporal Gaussian high-pass and kNN spatial Gaussian low-pass",
            "dataset_root": str(root),
            "network_source": str(network_source),
            "n_ps": int(n_ps),
            "n_image": int(n_image),
            "reference_image_ix_1based": int(master_ix),
            "reference_ps": int(ref_ps.size),
            "scn_time_win_days": float(time_win),
            "scn_wavelength_m": float(wavelength),
            "spatial_filter_enabled": bool(spatial_enabled),
            "spatial_radius_m": float(radius),
            "k_neighbors": int(k_use),
            "chunk_ps": int(chunk),
            "spatial_chunk_ps": int(spatial_chunk),
            "valid_velocity_ps": int(np.count_nonzero(np.isfinite(phase_rate))),
            "duration_sec": time.perf_counter() - started,
            "note": "Custom scalable SBAS extension; spatial filter uses bounded kNN approximation and is not MATLAB parity-certified.",
        },
    )
    return f"Stage 8 SBAS filtered {n_ps} PS across {n_image} acquisition epochs"
