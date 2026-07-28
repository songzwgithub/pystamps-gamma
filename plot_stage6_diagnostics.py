#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pystamps.io.mat import read_mat_variables
from pystamps.pipeline import ported
from pystamps.pipeline import stage6_sbas as sbas


TWO_PI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG diagnostic figures for each Stage-6 SBAS sub-stage: "
            "GRID, LA, TIME, and the latest completed SNAPHU interferogram."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ifg-index",
        type=int,
        default=0,
        help="1-based IFG index. Default: latest completed SNAPHU IFG.",
    )
    parser.add_argument(
        "--sample-edges",
        type=int,
        default=20000,
        help="Number of spatial arcs sampled for LA/TIME diagnostics.",
    )
    parser.add_argument(
        "--max-raster",
        type=int,
        default=1800,
        help="Maximum displayed rows/columns after diagnostic downsampling.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def mat_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(v) for v in arr.reshape(-1)).strip()
    return str(arr.reshape(-1)[0]).strip()


def normalize_drop_indices(value: Any, n_ifg: int) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    arr = np.rint(arr).astype(np.int64)
    return np.unique(arr[(arr >= 1) & (arr <= n_ifg)])


def safe_percentiles(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(finite))
        return center - 1.0, center + 1.0
    return float(lo), float(hi)


def downsample_shape(n_i: int, n_j: int, max_raster: int) -> tuple[int, int, int]:
    factor = max(1, math.ceil(max(n_i, n_j) / max(1, max_raster)))
    return math.ceil(n_i / factor), math.ceil(n_j / factor), factor


def rasterize_circular(
    rows_1b: np.ndarray,
    cols_1b: np.ndarray,
    phase: np.ndarray,
    n_i: int,
    n_j: int,
    max_raster: int,
) -> tuple[np.ndarray, int]:
    out_i, out_j, factor = downsample_shape(n_i, n_j, max_raster)
    rows = (np.asarray(rows_1b, dtype=np.int64) - 1) // factor
    cols = (np.asarray(cols_1b, dtype=np.int64) - 1) // factor
    values = np.asarray(phase, dtype=np.float64)
    valid = (
        (rows >= 0)
        & (rows < out_i)
        & (cols >= 0)
        & (cols < out_j)
        & np.isfinite(values)
    )
    linear = rows[valid] * out_j + cols[valid]
    real_sum = np.bincount(
        linear,
        weights=np.cos(values[valid]),
        minlength=out_i * out_j,
    )
    imag_sum = np.bincount(
        linear,
        weights=np.sin(values[valid]),
        minlength=out_i * out_j,
    )
    count = np.bincount(linear, minlength=out_i * out_j)
    raster = np.full(out_i * out_j, np.nan, dtype=np.float32)
    occupied = count > 0
    raster[occupied] = np.arctan2(imag_sum[occupied], real_sum[occupied]).astype(np.float32)
    return raster.reshape(out_i, out_j), factor


def rasterize_float(
    rows_1b: np.ndarray,
    cols_1b: np.ndarray,
    values: np.ndarray,
    n_i: int,
    n_j: int,
    max_raster: int,
) -> tuple[np.ndarray, int]:
    out_i, out_j, factor = downsample_shape(n_i, n_j, max_raster)
    rows = (np.asarray(rows_1b, dtype=np.int64) - 1) // factor
    cols = (np.asarray(cols_1b, dtype=np.int64) - 1) // factor
    data = np.asarray(values, dtype=np.float64)
    valid = (
        (rows >= 0)
        & (rows < out_i)
        & (cols >= 0)
        & (cols < out_j)
        & np.isfinite(data)
    )
    linear = rows[valid] * out_j + cols[valid]
    sums = np.bincount(
        linear,
        weights=data[valid],
        minlength=out_i * out_j,
    )
    count = np.bincount(linear, minlength=out_i * out_j)
    raster = np.full(out_i * out_j, np.nan, dtype=np.float32)
    occupied = count > 0
    raster[occupied] = (sums[occupied] / count[occupied]).astype(np.float32)
    return raster.reshape(out_i, out_j), factor


def save_map(
    raster: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
    dpi: int,
    *,
    limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(raster, origin="upper", interpolation="nearest")
    if limits is not None:
        image.set_clim(*limits)
    ax.set_title(title)
    ax.set_xlabel("Downsampled grid column")
    ax.set_ylabel("Downsampled grid row")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def completed_snaphu_outputs(
    snaphu_root: Path,
    n_i: int,
    n_j: int,
) -> dict[int, Path]:
    expected_bytes = int(n_i) * int(n_j) * np.dtype(np.float32).itemsize
    completed: dict[int, Path] = {}
    pattern = re.compile(r"ifg_(\d+)$")
    if not snaphu_root.exists():
        return completed
    for directory in snaphu_root.glob("ifg_*"):
        match = pattern.match(directory.name)
        if match is None:
            continue
        index_1b = int(match.group(1))
        output = directory / "snaphu.out"
        try:
            if output.is_file() and output.stat().st_size == expected_bytes:
                completed[index_1b] = output
        except OSError:
            continue
    return completed


def load_stage_geometry(dataset: Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    grid = read_mat_variables(
        dataset / "uw_grid.mat",
        ("ij", "n_i", "n_j", "n_ifg", "n_ps"),
    )
    ij = np.asarray(grid["ij"], dtype=np.float64)
    ij = np.squeeze(ij)
    if ij.ndim != 2:
        raise RuntimeError(f"uw_grid.ij must be 2-D, got {ij.shape}")
    if ij.shape[1] != 2 and ij.shape[0] == 2:
        ij = ij.T
    if ij.shape[1] != 2:
        raise RuntimeError(f"uw_grid.ij must have two columns, got {ij.shape}")
    n_i = int(round(scalar(grid.get("n_i"))))
    n_j = int(round(scalar(grid.get("n_j"))))
    n_ifg = int(round(scalar(grid.get("n_ifg"))))
    n_grid_ps = int(round(scalar(grid.get("n_ps"), ij.shape[0])))
    if ij.shape[0] != n_grid_ps:
        raise RuntimeError(
            f"uw_grid.ij rows {ij.shape[0]} do not match n_ps {n_grid_ps}"
        )
    rows = np.rint(ij[:, 0]).astype(np.int64)
    cols = np.rint(ij[:, 1]).astype(np.int64)
    return rows, cols, n_i, n_j, n_ifg


def load_edges(dataset: Path, n_grid_ps: int) -> tuple[np.ndarray, np.ndarray, int]:
    interp = read_mat_variables(
        dataset / "uw_interp.mat",
        ("edgs", "n_edge"),
    )
    edgs = np.asarray(interp["edgs"], dtype=np.float64)
    edgs = np.squeeze(edgs)
    if edgs.ndim != 2:
        raise RuntimeError(f"uw_interp.edgs must be 2-D, got {edgs.shape}")
    if edgs.shape[1] < 3 and edgs.shape[0] >= 3:
        edgs = edgs.T
    node_a, node_b = sbas._edge_nodes(edgs, n_grid_ps)
    return node_a, node_b, node_a.size


def select_edge_sample(n_edge: int, sample_edges: int) -> np.ndarray:
    count = min(max(1, int(sample_edges)), n_edge)
    if count == n_edge:
        return np.arange(n_edge, dtype=np.int64)
    rng = np.random.default_rng(20260728)
    return np.sort(rng.choice(n_edge, size=count, replace=False).astype(np.int64))


def plot_grid(
    grid_phase: np.ndarray,
    grid_lowpass: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_i: int,
    n_j: int,
    ifg0: int,
    out_dir: Path,
    max_raster: int,
    dpi: int,
) -> dict[str, Any]:
    filtered = np.asarray(grid_phase[:, ifg0], dtype=np.complex64)
    lowpass = np.asarray(grid_lowpass[:, ifg0], dtype=np.complex64)
    residual = np.angle(filtered * np.conj(lowpass)).astype(np.float32)
    raster, factor = rasterize_circular(
        rows,
        cols,
        residual,
        n_i,
        n_j,
        max_raster,
    )
    rms = float(np.sqrt(np.nanmean(residual.astype(np.float64) ** 2)))
    path = out_dir / f"01_GRID_filter_residual_ifg_{ifg0 + 1:04d}.png"
    save_map(
        raster,
        path,
        (
            f"GRID filtering residual — IFG {ifg0 + 1} "
            f"(circular RMS={rms:.3f} rad, downsample={factor}x)"
        ),
        "Filtered minus low-pass phase (rad)",
        dpi,
        limits=(-math.pi, math.pi),
    )
    return {
        "figure": path.name,
        "ifg_index_1b": ifg0 + 1,
        "residual_rms_rad": rms,
        "downsample_factor": factor,
    }


def load_network_for_selected_ifgs(
    dataset: Path,
    selected_n_ifg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ps2 = read_mat_variables(
        dataset / "ps2.mat",
        ("n_ifg", "mean_range", "mean_incidence"),
    )
    full_n_ifg = int(round(scalar(ps2.get("n_ifg"), selected_n_ifg)))
    day, ifgday_ix, bperp, _source = sbas.load_sbas_network(dataset, full_n_ifg)

    parms = read_mat_variables(
        dataset / "parms.mat",
        (
            "drop_ifg_index",
            "max_topo_err",
            "lambda",
            "unwrap_la_error_flag",
        ),
    )
    drop = set(
        normalize_drop_indices(parms.get("drop_ifg_index"), full_n_ifg).tolist()
    )
    unwrap_ix = np.asarray(
        [index for index in range(full_n_ifg) if index + 1 not in drop],
        dtype=np.int64,
    )
    if unwrap_ix.size != selected_n_ifg:
        raise RuntimeError(
            "Selected IFG count does not match drop_ifg_index: "
            f"GRID has {selected_n_ifg}, network selection gives {unwrap_ix.size}"
        )

    return (
        np.asarray(day, dtype=np.float64),
        np.asarray(ifgday_ix[unwrap_ix, :], dtype=np.int64),
        np.asarray(bperp[unwrap_ix], dtype=np.float64),
        unwrap_ix,
    )


def plot_la(
    dataset: Path,
    grid_phase: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    edge_sample: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    selected_n_ifg = grid_phase.shape[1]
    day, ifgday_ix, bperp, _unwrap_ix = load_network_for_selected_ifgs(
        dataset,
        selected_n_ifg,
    )
    G, day_active, ifg_active = sbas._active_network(day, ifgday_ix)

    ps2 = read_mat_variables(
        dataset / "ps2.mat",
        ("mean_range", "mean_incidence"),
    )
    parms = read_mat_variables(
        dataset / "parms.mat",
        ("max_topo_err", "lambda", "unwrap_la_error_flag"),
    )
    la_enabled = (
        mat_text(parms.get("unwrap_la_error_flag"), "y").lower() == "y"
    )
    max_topo_err = scalar(parms.get("max_topo_err"), 15.0)
    wavelength = scalar(parms.get("lambda"), 0.0555)
    mean_range = scalar(ps2.get("mean_range"), 830000.0)
    mean_incidence = scalar(ps2.get("mean_incidence"), math.radians(23.0))
    denominator = (
        wavelength
        * mean_range
        * math.sin(mean_incidence)
        / (4.0 * math.pi)
    )
    max_k = max_topo_err / denominator
    n_trial_wraps = float(np.ptp(bperp)) * max_k / TWO_PI

    a = node_a[edge_sample]
    b = node_b[edge_sample]
    dph = (
        np.asarray(grid_phase[b, :], dtype=np.complex64)
        * np.conj(np.asarray(grid_phase[a, :], dtype=np.complex64))
    )
    magnitude = np.abs(dph)
    np.divide(dph, magnitude, out=dph, where=magnitude != 0)

    if la_enabled:
        k_values = sbas._fit_la_error_chunk(
            dph,
            G=G,
            ifgday_ix=ifg_active,
            day=day_active,
            bperp=bperp,
            n_trial_wraps=n_trial_wraps,
        ).astype(np.float64)
    else:
        k_values = np.zeros(edge_sample.size, dtype=np.float64)

    before = np.nanstd(np.angle(dph).astype(np.float64), axis=1)
    corrected = dph * np.exp(-1j * k_values[:, None] * bperp[None, :])
    after = np.nanstd(np.angle(corrected).astype(np.float64), axis=1)

    finite = np.isfinite(before) & np.isfinite(after)
    path = out_dir / "02_LA_before_after_arc_std.png"
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(before[finite], after[finite], s=5, alpha=0.25, rasterized=True)
    limit = float(
        max(
            0.1,
            np.nanpercentile(
                np.concatenate((before[finite], after[finite])),
                99.0,
            ),
        )
    )
    ax.plot([0.0, limit], [0.0, limit], linestyle="--")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Arc phase standard deviation before LA correction (rad)")
    ax.set_ylabel("Arc phase standard deviation after LA correction (rad)")
    median_change = float(np.nanmedian(after[finite] - before[finite]))
    improved = float(np.mean(after[finite] < before[finite])) if np.any(finite) else float("nan")
    ax.set_title(
        "LA correction effect on sampled arcs "
        f"(n={int(np.count_nonzero(finite))}, "
        f"improved={100.0 * improved:.1f}%, "
        f"median Δstd={median_change:.3f} rad)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": path.name,
        "sample_edges": int(edge_sample.size),
        "la_enabled": bool(la_enabled),
        "nonzero_k": int(np.count_nonzero(k_values)),
        "median_k": float(np.nanmedian(k_values)),
        "improved_fraction": improved,
        "median_std_change_rad": median_change,
    }


def open_time_memmaps(
    work_dir: Path,
    n_edge: int,
    n_ifg: int,
) -> tuple[np.memmap, np.memmap]:
    noise_path = work_dir / "dph_noise.f32"
    uw_path = work_dir / "dph_space_uw.f32"
    expected = n_edge * n_ifg * np.dtype(np.float32).itemsize
    for path in (noise_path, uw_path):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"{path} has {path.stat().st_size} bytes; expected {expected}"
            )
    noise = np.memmap(
        noise_path,
        mode="r",
        dtype=np.float32,
        shape=(n_edge, n_ifg),
    )
    dph_uw = np.memmap(
        uw_path,
        mode="r",
        dtype=np.float32,
        shape=(n_edge, n_ifg),
    )
    return noise, dph_uw


def plot_time(
    grid_phase: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    edge_sample: np.ndarray,
    noise: np.ndarray,
    dph_uw: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    noise_sample = np.asarray(noise[edge_sample, :], dtype=np.float32)
    uw_sample = np.asarray(dph_uw[edge_sample, :], dtype=np.float32)
    valid_fraction = np.mean(
        np.isfinite(noise_sample) & np.isfinite(uw_sample),
        axis=1,
    )
    noise_std = np.nanstd(noise_sample.astype(np.float64), axis=1)

    candidates = np.flatnonzero(
        (valid_fraction >= 0.95)
        & np.isfinite(noise_std)
    )
    if candidates.size == 0:
        candidates = np.flatnonzero(
            (valid_fraction > 0.0)
            & np.isfinite(noise_std)
        )
    if candidates.size == 0:
        raise RuntimeError("No finite TIME-stage arc is available for plotting")

    target_std = float(np.nanmedian(noise_std[candidates]))
    chosen_local = int(
        candidates[
            np.nanargmin(
                np.abs(noise_std[candidates] - target_std)
            )
        ]
    )
    edge_index = int(edge_sample[chosen_local])

    wrapped = np.angle(
        np.asarray(grid_phase[node_b[edge_index], :], dtype=np.complex64)
        * np.conj(
            np.asarray(grid_phase[node_a[edge_index], :], dtype=np.complex64)
        )
    ).astype(np.float64)
    noise_row = np.asarray(noise[edge_index, :], dtype=np.float64)
    unwrapped = np.asarray(dph_uw[edge_index, :], dtype=np.float64)
    smooth = unwrapped - noise_row

    path = out_dir / f"03_TIME_representative_edge_{edge_index + 1:07d}.png"
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(1, wrapped.size + 1, dtype=np.int64)
    ax.plot(x, wrapped, linewidth=0.8, label="Wrapped arc phase")
    ax.plot(x, smooth, linewidth=1.0, label="Temporal smooth component")
    ax.plot(x, unwrapped, linewidth=0.8, label="TIME-stage arc phase")
    ax.set_xlabel("Selected interferogram index")
    ax.set_ylabel("Arc phase (rad)")
    ax.set_title(
        f"TIME smoothing — representative edge {edge_index + 1} "
        f"(noise std={float(noise_std[chosen_local]):.3f} rad, "
        f"valid={100.0 * valid_fraction[chosen_local]:.1f}%)"
    )
    ax.legend()
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": path.name,
        "edge_index_1b": edge_index + 1,
        "noise_std_rad": float(noise_std[chosen_local]),
        "valid_fraction": float(valid_fraction[chosen_local]),
        "sample_valid_median": float(np.nanmedian(valid_fraction)),
        "sample_noise_std_median_rad": float(np.nanmedian(noise_std)),
    }


def plot_snaphu(
    completed: dict[int, Path],
    requested_ifg_1b: int,
    grid_phase: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_i: int,
    n_j: int,
    out_dir: Path,
    max_raster: int,
    dpi: int,
) -> dict[str, Any]:
    if not completed:
        raise RuntimeError("No complete SNAPHU output is available yet")

    if requested_ifg_1b > 0 and requested_ifg_1b in completed:
        ifg1 = requested_ifg_1b
    else:
        ifg1 = max(completed)
    ifg0 = ifg1 - 1
    if ifg0 >= grid_phase.shape[1]:
        raise RuntimeError(
            f"SNAPHU IFG {ifg1} exceeds GRID columns {grid_phase.shape[1]}"
        )

    grid = ported._load_float_grid(completed[ifg1], n_j)
    if grid.shape != (n_i, n_j):
        raise RuntimeError(
            f"SNAPHU grid has shape {grid.shape}; expected {(n_i, n_j)}"
        )

    linear_f = (rows - 1) + (cols - 1) * n_i
    unwrapped_values = grid.reshape(-1, order="F")[linear_f]
    raster, factor = rasterize_float(
        rows,
        cols,
        unwrapped_values,
        n_i,
        n_j,
        max_raster,
    )
    limits = safe_percentiles(raster)
    path = out_dir / f"04_SNAPHU_unwrapped_ifg_{ifg1:04d}.png"
    save_map(
        raster,
        path,
        (
            f"SNAPHU unwrapped phase — IFG {ifg1} "
            f"(completed outputs={len(completed)}, downsample={factor}x)"
        ),
        "Unwrapped phase (rad)",
        dpi,
        limits=limits,
    )

    wrapped = np.angle(np.asarray(grid_phase[:, ifg0], dtype=np.complex64))
    finite = np.isfinite(unwrapped_values) & np.isfinite(wrapped)
    cycle = np.rint(
        (unwrapped_values[finite] - wrapped[finite]) / TWO_PI
    )
    return {
        "figure": path.name,
        "ifg_index_1b": ifg1,
        "completed_snaphu_outputs": len(completed),
        "display_min_rad": limits[0],
        "display_max_rad": limits[1],
        "cycle_median": float(np.nanmedian(cycle)) if cycle.size else float("nan"),
        "cycle_min": float(np.nanmin(cycle)) if cycle.size else float("nan"),
        "cycle_max": float(np.nanmax(cycle)) if cycle.size else float("nan"),
    }



def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset does not exist: {dataset}")

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset / "stage6_png"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = dataset / "_stage6_sbas_work"
    grid_dir = work_dir / "grid_v2"
    phase_path = grid_dir / "grid_phase.npy"
    low_path = grid_dir / "grid_lowpass.npy"
    done_path = grid_dir / "done.npy"

    for path in (
        dataset / "uw_grid.mat",
        dataset / "uw_interp.mat",
        phase_path,
        low_path,
        done_path,
    ):
        if not path.exists():
            raise SystemExit(f"Required Stage-6 artifact is missing: {path}")

    grid_phase = np.load(phase_path, mmap_mode="r")
    grid_lowpass = np.load(low_path, mmap_mode="r")
    done = np.load(done_path, mmap_mode="r")

    rows, cols, n_i, n_j, selected_n_ifg = load_stage_geometry(dataset)
    if grid_phase.shape != (rows.size, selected_n_ifg):
        raise SystemExit(
            f"GRID phase shape {grid_phase.shape} does not match "
            f"geometry {(rows.size, selected_n_ifg)}"
        )
    if grid_lowpass.shape != grid_phase.shape:
        raise SystemExit(
            f"GRID low-pass shape {grid_lowpass.shape} does not match {grid_phase.shape}"
        )

    node_a, node_b, n_edge = load_edges(dataset, rows.size)
    edge_sample = select_edge_sample(n_edge, args.sample_edges)

    completed = completed_snaphu_outputs(
        work_dir / "snaphu",
        n_i,
        n_j,
    )
    if args.ifg_index > 0:
        ifg1 = args.ifg_index
    elif completed:
        ifg1 = max(completed)
    else:
        completed_grid = np.flatnonzero(np.asarray(done) != 0)
        if completed_grid.size == 0:
            raise SystemExit("No completed GRID IFG is available")
        ifg1 = int(completed_grid[-1] + 1)

    if ifg1 < 1 or ifg1 > selected_n_ifg:
        raise SystemExit(
            f"IFG index must be 1..{selected_n_ifg}; got {ifg1}"
        )
    ifg0 = ifg1 - 1
    if done[ifg0] == 0:
        raise SystemExit(f"GRID IFG {ifg1} is not complete")

    summaries: dict[str, dict[str, Any]] = {}

    print(f"[PLOT] GRID IFG {ifg1}", flush=True)
    summaries["GRID"] = plot_grid(
        grid_phase,
        grid_lowpass,
        rows,
        cols,
        n_i,
        n_j,
        ifg0,
        out_dir,
        args.max_raster,
        args.dpi,
    )

    print(f"[PLOT] LA sample edges={edge_sample.size}", flush=True)
    summaries["LA"] = plot_la(
        dataset,
        grid_phase,
        node_a,
        node_b,
        edge_sample,
        out_dir,
        args.dpi,
    )

    print("[PLOT] TIME", flush=True)
    noise, dph_uw = open_time_memmaps(
        work_dir,
        n_edge,
        selected_n_ifg,
    )
    summaries["TIME"] = plot_time(
        grid_phase,
        node_a,
        node_b,
        edge_sample,
        noise,
        dph_uw,
        out_dir,
        args.dpi,
    )

    if completed:
        print(
            f"[PLOT] SNAPHU complete={len(completed)}, "
            f"selected={ifg1 if ifg1 in completed else max(completed)}",
            flush=True,
        )
        summaries["SNAPHU"] = plot_snaphu(
            completed,
            ifg1,
            grid_phase,
            rows,
            cols,
            n_i,
            n_j,
            out_dir,
            args.max_raster,
            args.dpi,
        )
    else:
        summaries["SNAPHU"] = {
            "status": "No complete snaphu.out is available yet"
        }

    print()
    print("Stage-6 PNG figures created:")
    created = 0
    for stage, summary in summaries.items():
        figure = summary.get("figure")
        if figure:
            created += 1
            print(f"  {stage:7s}: {out_dir / figure}")
    if created == 0:
        print("  No PNG figure was created.")
    print(f"  Output : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())