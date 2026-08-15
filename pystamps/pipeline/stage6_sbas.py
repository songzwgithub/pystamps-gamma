from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from scipy import fft as scipy_fft
from scipy import ndimage

from pystamps.stage6_turbo import (
    edge_chunk as turbo_edge_chunk,
    invert_chunk as turbo_invert_chunk,
    resource_summary as turbo_resource_summary,
    turbo_blas_stage,
    workers_from_env as turbo_workers_from_env,
)

from pystamps.io.mat import read_mat, read_mat_variables, write_mat


TWO_PI = 2.0 * math.pi


class Stage6SbasError(RuntimeError):
    """Raised when the StaMPS-compatible SBAS Stage 6 cannot continue."""


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise Stage6SbasError(f"{name} must be an integer") from exc
    if value < minimum:
        raise Stage6SbasError(f"{name} must be >= {minimum}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise Stage6SbasError(f"{name} must be true/false or 1/0")


def _mat_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(v) for v in arr.reshape(-1)).strip()
    if arr.dtype == object:
        return str(arr.reshape(-1)[0]).strip()
    return str(arr.reshape(-1)[0]).strip()


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _as_rows(value: Any, rows: int, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise Stage6SbasError(f"{name} is empty")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if rows == 1:
            arr = arr.reshape(1, -1)
        elif arr.size % rows == 0:
            arr = arr.reshape(rows, -1)
        else:
            raise Stage6SbasError(f"{name} cannot be reshaped to {rows} rows")
    if arr.ndim != 2:
        raise Stage6SbasError(f"{name} must be a 2-D matrix")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise Stage6SbasError(
            f"{name} has shape {arr.shape}; expected first dimension {rows}"
        )
    return np.asarray(arr, dtype=dtype)


def _as_vector(value: Any, length: int, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.size != length:
        raise Stage6SbasError(f"{name} has {arr.size} values; expected {length}")
    return np.asarray(arr, dtype=dtype)


def _normalize_complex(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex64).copy()
    magnitude = np.abs(arr)
    np.divide(arr, magnitude, out=arr, where=magnitude != 0)
    return arr


def _drop_indices(value: Any, n_ifg: int) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    arr = np.rint(arr).astype(np.int64)
    return np.unique(arr[(arr >= 1) & (arr <= n_ifg)])



def _matlab_round(values: Any) -> np.ndarray:
    """MATLAB round(): half values round away from zero."""
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _find_network_source(dataset_root: Path) -> Path:
    candidates = [dataset_root / "ps2.mat"]
    candidates.extend(sorted(dataset_root.glob("PATCH_*/ps1.mat")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = read_mat_variables(path, ("ifgday_ix",))
        except Exception:
            continue
        if "ifgday_ix" in payload and np.asarray(payload["ifgday_ix"]).size:
            return path
    raise Stage6SbasError(
        "SBAS network geometry ifgday_ix is missing. "
        "Expected it in ps2.mat or PATCH_*/ps1.mat."
    )


def load_sbas_network(
    dataset_root: Path,
    n_ifg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load StaMPS SBAS acquisition dates, IFG endpoint indices and baselines."""

    source = _find_network_source(dataset_root)
    network = read_mat_variables(
        source,
        ("ifgday_ix", "day", "n_image", "n_ifg", "bperp", "master_day"),
    )

    ifgday_ix = np.asarray(network.get("ifgday_ix"), dtype=np.float64)
    ifgday_ix = np.squeeze(ifgday_ix)
    if ifgday_ix.ndim != 2:
        raise Stage6SbasError(f"{source}: ifgday_ix must be 2-D")
    if ifgday_ix.shape[1] != 2 and ifgday_ix.shape[0] == 2:
        ifgday_ix = ifgday_ix.T
    if ifgday_ix.shape != (n_ifg, 2):
        raise Stage6SbasError(
            f"{source}: ifgday_ix shape is {ifgday_ix.shape}; expected ({n_ifg}, 2)"
        )
    ifgday_ix = np.rint(ifgday_ix).astype(np.int64)

    day_raw = network.get("day")
    if day_raw is None or np.asarray(day_raw).size == 0:
        root_ps = read_mat_variables(dataset_root / "ps2.mat", ("day",))
        day_raw = root_ps.get("day")
    day = np.asarray(day_raw, dtype=np.float64).reshape(-1)
    if day.size < 2:
        raise Stage6SbasError(f"{source}: day must contain acquisition dates")

    if np.any(ifgday_ix < 1) or np.any(ifgday_ix > day.size):
        raise Stage6SbasError(
            f"{source}: ifgday_ix contains indices outside 1..{day.size}"
        )

    root_ps = read_mat_variables(dataset_root / "ps2.mat", ("bperp",))
    bperp_raw = root_ps.get("bperp")
    if bperp_raw is None or np.asarray(bperp_raw).size != n_ifg:
        bperp_raw = network.get("bperp")
    bperp = _as_vector(bperp_raw, n_ifg, "SBAS bperp", np.float64)

    return day, ifgday_ix, bperp, source


def _active_network(
    day: np.ndarray,
    ifgday_ix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ifg = ifgday_ix.shape[0]
    n_image = day.size
    G_full = np.zeros((n_ifg, n_image), dtype=np.float64)
    rows = np.arange(n_ifg, dtype=np.int64)
    G_full[rows, ifgday_ix[:, 0] - 1] = -1.0
    G_full[rows, ifgday_ix[:, 1] - 1] = 1.0

    used = np.any(G_full != 0, axis=0)
    if np.count_nonzero(used) < 2:
        raise Stage6SbasError("SBAS network contains fewer than two active acquisitions")

    old_to_new = np.zeros(n_image + 1, dtype=np.int64)
    old_to_new[np.flatnonzero(used) + 1] = np.arange(1, np.count_nonzero(used) + 1)
    ifg_active = old_to_new[ifgday_ix]
    day_active = day[used]
    G = G_full[:, used]

    rank = int(np.linalg.matrix_rank(G))
    expected_rank = G.shape[1] - 1
    if rank != expected_rank:
        raise Stage6SbasError(
            "SBAS acquisition graph is disconnected or rank deficient: "
            f"rank(G)={rank}, expected={expected_rank}, active_images={G.shape[1]}"
        )
    return G, day_active, ifg_active


def _edge_nodes(edgs: Any, n_node: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(edgs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise Stage6SbasError("uw_interp.edgs must be a 2-D matrix")
    node_a = np.rint(arr[:, -2]).astype(np.int64) - 1
    node_b = np.rint(arr[:, -1]).astype(np.int64) - 1
    if (
        np.any(node_a < 0)
        or np.any(node_b < 0)
        or np.any(node_a >= n_node)
        or np.any(node_b >= n_node)
    ):
        raise Stage6SbasError("uw_interp.edgs contains out-of-range node indices")
    return node_a, node_b


def _arc_phasors(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    values = (
        uw_ph[node_b[start:stop], :]
        * np.conj(uw_ph[node_a[start:stop], :])
    ).astype(np.complex64)
    return _normalize_complex(values)


def _prepare_la_subset(
    dph_space: np.ndarray,
    *,
    G: np.ndarray,
    ifgday_ix: np.ndarray,
    day: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    full_range = float(np.ptp(bperp))
    if full_range <= np.finfo(np.float64).eps:
        return dph_space[:, :1], np.zeros(1, dtype=np.float64), 0.0

    adjacent = np.flatnonzero(np.abs(np.diff(ifgday_ix, axis=1).reshape(-1)) == 1)
    if adjacent.size >= day.size - 1:
        dph_sub = dph_space[:, adjacent]
        bperp_sub = bperp[adjacent].astype(np.float64)
    else:
        per_image = np.sum(np.abs(G), axis=0)
        max_ix = int(np.argmax(per_image))
        if int(per_image[max_ix]) >= day.size - 2:
            cols = np.flatnonzero(G[:, max_ix] != 0)
            gsub = G[cols, max_ix]
            orient = -np.sign(gsub).astype(np.int64)

            dph_oriented = dph_space[:, cols].copy()
            flip = orient == -1
            if np.any(flip):
                dph_oriented[:, flip] = np.conj(dph_oriented[:, flip])

            b_oriented = bperp[cols].astype(np.float64) * orient.astype(np.float64)
            slave_ix = np.sum(ifgday_ix[cols, :], axis=1) - (max_ix + 1)
            image_ix = np.concatenate((slave_ix, np.asarray([max_ix + 1], dtype=np.int64)))
            order = np.argsort(day[image_ix - 1], kind="stable")

            dph_with_master = np.concatenate(
                (
                    dph_oriented,
                    np.ones((dph_oriented.shape[0], 1), dtype=np.complex64),
                ),
                axis=1,
            )[:, order]
            b_with_master = np.concatenate((b_oriented, np.asarray([0.0])))[order]
            dph_sub = (
                dph_with_master[:, 1:]
                * np.conj(dph_with_master[:, :-1])
            )
            dph_sub = _normalize_complex(dph_sub)
            bperp_sub = np.diff(b_with_master)
        else:
            dph_sub = dph_space
            bperp_sub = bperp.astype(np.float64)

    sub_range = float(np.ptp(bperp_sub))
    if sub_range <= np.finfo(np.float64).eps:
        return dph_sub, bperp_sub, 0.0
    scaled_trials = float(n_trial_wraps) * sub_range / full_range
    return dph_sub, bperp_sub, scaled_trials


def _fit_la_error_chunk(
    dph_space: np.ndarray,
    *,
    G: np.ndarray,
    ifgday_ix: np.ndarray,
    day: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
) -> np.ndarray:
    dph_sub, b_sub, scaled_trials = _prepare_la_subset(
        dph_space,
        G=G,
        ifgday_ix=ifgday_ix,
        day=day,
        bperp=bperp,
        n_trial_wraps=n_trial_wraps,
    )
    edge_count = dph_space.shape[0]
    if scaled_trials <= 0 or b_sub.size == 0:
        return np.zeros(edge_count, dtype=np.float32)

    b_range = float(np.ptp(b_sub))
    if b_range <= np.finfo(np.float64).eps:
        return np.zeros(edge_count, dtype=np.float32)

    trial_limit = max(1, int(math.ceil(8.0 * scaled_trials)))
    trial_mult = np.arange(-trial_limit, trial_limit + 1, dtype=np.float64)
    trial_phase = b_sub / b_range * (math.pi / 4.0)
    trial_matrix = np.exp(-1j * np.outer(trial_phase, trial_mult))
    coherence = np.abs(dph_sub @ trial_matrix)
    denominator = np.sum(np.abs(dph_sub), axis=1, keepdims=True)
    coherence = np.divide(
        coherence,
        denominator,
        out=np.zeros_like(coherence, dtype=np.float64),
        where=denominator != 0,
    )

    K = np.zeros(edge_count, dtype=np.float64)
    final_coh = np.zeros(edge_count, dtype=np.float64)

    for edge in range(edge_count):
        row = coherence[edge]
        peak = int(np.argmax(row))
        peak_value = float(row[peak])

        falling = np.flatnonzero(np.diff(row[: peak + 1]) < 0)
        peak_start = int(falling[-1] + 1) if falling.size else 0

        rising = np.flatnonzero(np.diff(row[peak:]) > 0)
        peak_end = int(rising[0] + peak) if rising.size else row.size - 1

        second = row.copy()
        second[peak_start : peak_end + 1] = 0.0
        if peak_value - float(np.max(second)) <= 0.1:
            continue

        K0 = math.pi / 4.0 / b_range * trial_mult[peak]
        cpx = dph_sub[edge]
        residual = cpx * np.exp(-1j * K0 * b_sub)
        offset = np.sum(residual)
        residual_angle = np.angle(residual * np.conj(offset))
        weights = np.abs(cpx)
        denom = float(np.sum(weights * b_sub * b_sub))
        correction = (
            float(np.sum(weights * b_sub * residual_angle)) / denom
            if denom > 0
            else 0.0
        )
        k_value = K0 + correction
        fitted = cpx * np.exp(-1j * k_value * b_sub)
        coh_value = float(np.abs(np.sum(fitted)) / max(np.sum(np.abs(fitted)), 1e-12))
        if coh_value >= 0.31:
            K[edge] = k_value
            final_coh[edge] = coh_value

    return K.astype(np.float32)


def _covariance_whitener(
    uw_ph: np.ndarray,
    ph_lowpass: np.ndarray | None,
    node_a: np.ndarray,
    node_b: np.ndarray,
    edge_chunk: int,
) -> np.ndarray:
    n_ifg = uw_ph.shape[1]
    if ph_lowpass is None or ph_lowpass.shape != uw_ph.shape:
        return np.eye(n_ifg, dtype=np.float64)

    count = 0
    total = np.zeros(n_ifg, dtype=np.float64)
    outer = np.zeros((n_ifg, n_ifg), dtype=np.float64)

    ph_noise = np.angle(
        uw_ph.astype(np.complex128)
        * np.conj(ph_lowpass.astype(np.complex128))
    )

    for start in range(0, node_a.size, edge_chunk):
        stop = min(start + edge_chunk, node_a.size)
        diff = (
            ph_noise[node_b[start:stop], :]
            - ph_noise[node_a[start:stop], :]
        ).astype(np.float64)
        total += np.sum(diff, axis=0)
        outer += diff.T @ diff
        count += diff.shape[0]

    if count <= 1:
        return np.eye(n_ifg, dtype=np.float64)

    covariance = (
        outer - np.outer(total, total) / float(count)
    ) / float(count - 1)
    covariance = 0.5 * (covariance + covariance.T)

    try:
        inverse = np.linalg.pinv(covariance, hermitian=True)
        chol = np.linalg.cholesky(inverse)
        return chol.T
    except np.linalg.LinAlgError:
        diagonal = np.diag(covariance)
        scale = np.divide(
            1.0,
            np.sqrt(diagonal),
            out=np.ones_like(diagonal),
            where=diagonal > 0,
        )
        return np.diag(scale)


def _anneal_cost(
    model: np.ndarray,
    *,
    G: np.ndarray,
    W: np.ndarray,
    dph: np.ndarray,
    x: np.ndarray,
) -> float:
    n = G.shape[1]
    series = (
        model[0] * x
        + model[1] * n / 2.0 * np.sin(2.0 * math.pi / n * x - model[2])
        + model[3] * n / 2.0 * np.sin(4.0 * math.pi / n * x - model[4])
    )
    predicted = G @ series
    residual_cycles = (dph - predicted) / TWO_PI
    wrapped_cycles = residual_cycles - np.rint(residual_cycles)
    weighted = np.abs(W @ wrapped_cycles) * TWO_PI
    return float(np.sum(weighted) + np.sum(np.abs(np.diff(series))) / 5.0)


def _boltzmann_probabilities(temperature: float, values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-12)
    maximum = float(np.nanmax(scaled))
    if maximum > 708.3964185322641:
        factor = maximum / 708.3964185322641
        probability = np.exp(-scaled / factor)
        probability /= max(float(np.nanmax(probability)), 1e-300)
        probability = probability**factor
    else:
        probability = np.exp(-scaled)
        probability /= max(float(np.nanmax(probability)), 1e-300)
    probability[~np.isfinite(probability)] = 0.0
    total = float(np.sum(probability))
    if total <= 0:
        return np.full(probability.shape, 1.0 / probability.size)
    return probability / total


def _anneal_smooth_unwrap(
    *,
    G: np.ndarray,
    W: np.ndarray,
    dph: np.ndarray,
    x: np.ndarray,
    seed: int,
    runs: int,
) -> np.ndarray:
    """Port of StaMPS uw_sb_smooth_unwrap for one high-noise arc."""

    bounds = np.asarray(
        [
            [-0.5 * math.pi, 0.5 * math.pi],
            [-0.25 * math.pi, 0.25 * math.pi],
            [-math.pi, math.pi],
            [-0.25 * math.pi, 0.25 * math.pi],
            [-math.pi, math.pi],
        ],
        dtype=np.float64,
    )
    grid = 4
    values = np.concatenate(
        (
            2.0 ** (-np.arange(1, grid + 1, dtype=np.float64)),
            np.asarray([0.0]),
            -(2.0 ** (-np.arange(1, grid + 1, dtype=np.float64))),
        )
    )
    delta = 0.5 * np.abs(bounds[:, 0] - bounds[:, 1])
    rng = np.random.default_rng(seed)
    best_cost = math.inf
    best_model = np.zeros(5, dtype=np.float64)
    ts = np.linspace(2.0, 3.0, max(1, runs))

    schedule_repeats = np.asarray([1, 2, 4, 6, 10, 6, 4, 2, 1], dtype=np.int64)
    for run in range(max(1, runs)):
        candidates = (
            rng.random((100, 5))
            * (bounds[:, 1] - bounds[:, 0])[None, :]
            + bounds[:, 0][None, :]
        )
        initial_costs = np.asarray(
            [
                _anneal_cost(model, G=G, W=W, dph=dph, x=x)
                for model in candidates
            ],
            dtype=np.float64,
        )
        current = candidates[int(np.argmin(initial_costs))].copy()
        critical_log10 = math.log10(max(float(np.mean(initial_costs)), 1e-12)) - float(ts[run])
        temperatures = np.repeat(
            np.logspace(critical_log10 + 1.0, critical_log10 - 1.0, 9),
            schedule_repeats,
        )

        current_best_cost = float(np.min(initial_costs))
        current_best_model = current.copy()

        for temperature in temperatures:
            for parameter in range(5):
                if delta[parameter] == 0:
                    continue
                candidate_values = current[parameter] + values * delta[parameter]
                candidate_values = candidate_values[
                    (candidate_values >= bounds[parameter, 0])
                    & (candidate_values <= bounds[parameter, 1])
                ]
                models = np.repeat(current[None, :], candidate_values.size, axis=0)
                models[:, parameter] = candidate_values
                costs = np.asarray(
                    [
                        _anneal_cost(model, G=G, W=W, dph=dph, x=x)
                        for model in models
                    ],
                    dtype=np.float64,
                )
                probabilities = _boltzmann_probabilities(float(temperature), costs)
                selected = int(rng.choice(np.arange(costs.size), p=probabilities))
                current = models[selected].copy()
                if float(costs[selected]) < current_best_cost:
                    current_best_cost = float(costs[selected])
                    current_best_model = current.copy()

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_model = current_best_model

    n = G.shape[1]
    return (
        best_model[0] * x
        + best_model[1] * n / 2.0 * np.sin(2.0 * math.pi / n * x - best_model[2])
        + best_model[3] * n / 2.0 * np.sin(4.0 * math.pi / n * x - best_model[4])
    ).astype(np.float64)


def compute_sbas_space_time(
    *,
    uw_ph: np.ndarray,
    ph_lowpass: np.ndarray | None,
    edgs: Any,
    day: np.ndarray,
    ifgday_ix: np.ndarray,
    bperp: np.ndarray,
    time_win: float,
    n_trial_wraps: float,
    unwrap_method: str,
    la_flag: bool,
    edge_chunk: int,
    anneal_workers: int,
    anneal_runs: int,
    strict_anneal: bool,
    progress: bool,
    work_dir: Path,
) -> tuple[np.ndarray, np.memmap, np.memmap, dict[str, Any]]:
    """StaMPS multiple-master time-space smoothing for SBAS interferograms."""

    n_node, n_ifg = uw_ph.shape
    if ifgday_ix.shape != (n_ifg, 2):
        raise Stage6SbasError(
            f"Selected ifgday_ix shape {ifgday_ix.shape} does not match uw_ph {uw_ph.shape}"
        )
    G, day_active, ifg_active = _active_network(day, ifgday_ix)
    node_a, node_b = _edge_nodes(edgs, n_node)
    n_edge = node_a.size

    method = unwrap_method.strip().upper()
    supported = {"3D", "3D_NEW", "3D_SMALL_DEF", "3D_QUICK", "3D_NO_DEF", "2D"}
    if method not in supported:
        raise Stage6SbasError(
            f"SBAS Stage 6 supports {sorted(supported)}; got unwrap_method={unwrap_method!r}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    noise_path = work_dir / "dph_noise.f32"
    unwrapped_path = work_dir / "dph_space_uw.f32"
    dph_noise_out = np.memmap(
        noise_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_edge, n_ifg),
    )
    dph_uw_out = np.memmap(
        unwrapped_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_edge, n_ifg),
    )

    K_all = np.zeros(n_edge, dtype=np.float32)
    if la_flag:
        started = time.perf_counter()
        for start in range(0, n_edge, edge_chunk):
            stop = min(start + edge_chunk, n_edge)
            dph_chunk = _arc_phasors(uw_ph, node_a, node_b, start, stop)
            K_all[start:stop] = _fit_la_error_chunk(
                dph_chunk,
                G=G,
                ifgday_ix=ifg_active,
                day=day_active,
                bperp=bperp,
                n_trial_wraps=n_trial_wraps,
            )
            if progress:
                print(
                    "[STAGE6_SBAS][LA] "
                    f"{stop}/{n_edge} ({100.0 * stop / max(1, n_edge):.1f}%) "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    G_reduced = G[:, 1:]
    G_pinv = np.linalg.pinv(G_reduced)
    time_diff = day_active[:, None] - day_active[None, :]
    time_weights = np.exp(
        -(time_diff * time_diff) / (2.0 * max(float(time_win), 1e-6) ** 2)
    )
    time_weights /= np.sum(time_weights, axis=1, keepdims=True)

    if day_active[-1] == day_active[0]:
        x = np.arange(day_active.size, dtype=np.float64)
    else:
        x = (
            (day_active - day_active[0])
            * (day_active.size - 1)
            / (day_active[-1] - day_active[0])
        )

    whitener = (
        _covariance_whitener(
            uw_ph,
            ph_lowpass,
            node_a,
            node_b,
            edge_chunk,
        )
        if method in {"3D", "3D_NEW"} and strict_anneal
        else np.eye(n_ifg, dtype=np.float64)
    )

    high_noise_total = 0
    started = time.perf_counter()

    for start in range(0, n_edge, edge_chunk):
        stop = min(start + edge_chunk, n_edge)
        dph = _arc_phasors(uw_ph, node_a, node_b, start, stop)
        if la_flag:
            dph *= np.exp(-1j * K_all[start:stop, None] * bperp[None, :])

        if method == "2D":
            smooth = np.zeros((stop - start, n_ifg), dtype=np.float64)
            noise = np.angle(dph).astype(np.float64)
        elif method == "3D_NO_DEF":
            smooth = np.zeros((stop - start, n_ifg), dtype=np.float64)
            noise = np.angle(dph).astype(np.float64)
        else:
            phase = np.angle(dph).astype(np.float64)
            series = np.zeros((day_active.size, stop - start), dtype=np.float64)
            series[1:, :] = G_pinv @ phase.T
            smooth_series = time_weights @ series
            smooth = (G @ smooth_series).T
            noise = np.angle(dph * np.exp(-1j * smooth)).astype(np.float64)

            if method in {"3D_SMALL_DEF", "3D_QUICK"}:
                invalid = np.std(noise, axis=1, ddof=1) > 1.3
                noise[invalid, :] = np.nan
                smooth[invalid, :] = np.nan
            elif method in {"3D", "3D_NEW"} and strict_anneal:
                high_local = np.flatnonzero(np.std(noise, axis=1, ddof=1) > 1.0)
                high_noise_total += int(high_local.size)
                if high_local.size:
                    def _run_one(local_index: int) -> tuple[int, np.ndarray]:
                        global_index = start + int(local_index)
                        refined = _anneal_smooth_unwrap(
                            G=G,
                            W=whitener,
                            dph=phase[int(local_index), :],
                            x=x,
                            seed=104729 + global_index,
                            runs=anneal_runs,
                        )
                        return int(local_index), refined

                    with ThreadPoolExecutor(max_workers=max(1, anneal_workers)) as pool:
                        futures = [pool.submit(_run_one, int(ix)) for ix in high_local]
                        for future in as_completed(futures):
                            local_index, refined_series = future.result()
                            smooth[local_index, :] = G @ refined_series

                    noise = np.angle(dph * np.exp(-1j * smooth)).astype(np.float64)

        dph_uw = smooth + noise
        if la_flag:
            dph_uw += K_all[start:stop, None] * bperp[None, :]

        dph_noise_out[start:stop, :] = noise.astype(np.float32)
        dph_uw_out[start:stop, :] = dph_uw.astype(np.float32)

        if progress:
            print(
                "[STAGE6_SBAS][TIME] "
                f"{stop}/{n_edge} ({100.0 * stop / max(1, n_edge):.1f}%) "
                f"elapsed={time.perf_counter() - started:.1f}s "
                f"high_noise={high_noise_total}",
                flush=True,
            )

    dph_noise_out.flush()
    dph_uw_out.flush()
    metadata = {
        "n_edge": int(n_edge),
        "n_ifg": int(n_ifg),
        "n_image_active": int(day_active.size),
        "rank_G": int(np.linalg.matrix_rank(G)),
        "la_nonzero_edges": int(np.count_nonzero(K_all)),
        "annealed_edges": int(high_noise_total),
        "unwrap_method": method,
    }
    return G, dph_noise_out, dph_uw_out, metadata



# === STAGE6_MATLAB_SB_INVERT_V1 ===

def _llh2local_m(
    lonlat: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """
    Python port of StaMPS llh2local.m.
    Input lon/lat in degrees, output metres.
    """
    ll = np.asarray(lonlat, dtype=np.float64)
    org = np.asarray(origin, dtype=np.float64).reshape(-1)

    if ll.ndim != 2 or ll.shape[1] < 2:
        raise Stage6SbasError("lonlat must be N x 2")

    a = 6378137.0
    e = 0.08209443794970

    lon = np.deg2rad(ll[:, 0])
    lat = np.deg2rad(ll[:, 1])
    org_lon = math.radians(float(org[0]))
    org_lat = math.radians(float(org[1]))

    M0 = a * (
        (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * org_lat
        - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024)
        * math.sin(2*org_lat)
        + (15*e**4/256 + 45*e**6/1024)
        * math.sin(4*org_lat)
        - (35*e**6/3072)
        * math.sin(6*org_lat)
    )

    x = np.empty(lat.size, dtype=np.float64)
    y = np.empty(lat.size, dtype=np.float64)

    nonzero = lat != 0.0

    if np.any(nonzero):
        phi = lat[nonzero]
        dlambda = lon[nonzero] - org_lon

        M = a * (
            (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * phi
            - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024)
            * np.sin(2*phi)
            + (15*e**4/256 + 45*e**6/1024)
            * np.sin(4*phi)
            - (35*e**6/3072)
            * np.sin(6*phi)
        )

        N = a / np.sqrt(
            1 - e**2 * np.sin(phi)**2
        )

        E = dlambda * np.sin(phi)
        cot_phi = 1.0 / np.tan(phi)

        x[nonzero] = (
            N * cot_phi * np.sin(E)
        )
        y[nonzero] = (
            M
            - M0
            + N * cot_phi * (1 - np.cos(E))
        )

    if np.any(~nonzero):
        dlambda = lon[~nonzero] - org_lon
        x[~nonzero] = a * dlambda
        y[~nonzero] = -M0

    return np.column_stack((x, y))


def _stage6_reference_indices(
    ps2: dict[str, Any],
    parms: dict[str, Any],
    n_ps: int,
) -> np.ndarray:
    """
    Match ps_setref.m lon/lat/radius selection.
    """
    lonlat = _as_rows(
        ps2.get("lonlat"),
        n_ps,
        "ps2.lonlat",
        np.float64,
    )

    def _pair(name: str) -> np.ndarray:
        raw = parms.get(name)
        if raw is None or np.asarray(raw).size < 2:
            return np.asarray(
                [-np.inf, np.inf],
                dtype=np.float64,
            )
        return np.asarray(
            raw,
            dtype=np.float64,
        ).reshape(-1)[:2]

    ref_lon = _pair("ref_lon")
    ref_lat = _pair("ref_lat")

    radius_raw = parms.get("ref_radius")

    if radius_raw is None or np.asarray(radius_raw).size == 0:
        radius_raw = parms.get("ref_radius_m")

    if radius_raw is None or np.asarray(radius_raw).size == 0:
        ref_radius = np.inf
    else:
        ref_radius = float(
            np.asarray(radius_raw).reshape(-1)[0]
        )

    # StaMPS convention: -Inf means no reference.
    if np.isneginf(ref_radius):
        return np.empty(0, dtype=np.int64)

    mask = (
        (lonlat[:, 0] > ref_lon[0])
        & (lonlat[:, 0] < ref_lon[1])
        & (lonlat[:, 1] > ref_lat[0])
        & (lonlat[:, 1] < ref_lat[1])
    )

    idx = np.flatnonzero(mask)

    if np.isfinite(ref_radius):
        centre_raw = parms.get("ref_centre_lonlat")
        if centre_raw is None or np.asarray(centre_raw).size < 2:
            raise Stage6SbasError(
                "finite reference radius requires ref_centre_lonlat"
            )
        centre = np.asarray(
            centre_raw,
            dtype=np.float64,
        ).reshape(-1)[:2]

        ll0_raw = ps2.get("ll0")
        if ll0_raw is None or np.asarray(ll0_raw).size < 2:
            raise Stage6SbasError(
                "finite ref_radius requires ps2.ll0"
            )

        ll0 = np.asarray(
            ll0_raw,
            dtype=np.float64,
        ).reshape(-1)[:2]

        centre_xy = _llh2local_m(
            centre.reshape(1, 2),
            ll0,
        )[0]

        ps_xy = _llh2local_m(
            lonlat[idx, :2],
            ll0,
        )

        dist2 = np.sum(
            (ps_xy - centre_xy[None, :])**2,
            axis=1,
        )

        idx = idx[
            dist2 <= ref_radius**2
        ]

    if idx.size == 0:
        raise Stage6SbasError(
            "StaMPS reference selection returned zero PS"
        )

    return idx.astype(np.int64)


def _stage6_sb_covariance(
    root: Path,
    n_ps: int,
    n_ifg: int,
    *,
    chunk_ps: int = 4096,
    progress: bool = True,
) -> np.ndarray:
    """
    Equivalent target:
        ph_noise = angle(rc.ph_rc .* conj(pm.ph_patch));
        sb_cov = double(cov(ph_noise));

    Uses block-combined covariance so the full ph_noise matrix is not
    materialised in memory.
    """
    rc = _as_rows(
        read_mat_variables(
            root / "rc2.mat",
            ("ph_rc",),
        ).get("ph_rc"),
        n_ps,
        "rc2.ph_rc",
        np.complex64,
    )

    pm = _as_rows(
        read_mat_variables(
            root / "pm2.mat",
            ("ph_patch",),
        ).get("ph_patch"),
        n_ps,
        "pm2.ph_patch",
        np.complex64,
    )

    if rc.shape[1] != n_ifg or pm.shape[1] != n_ifg:
        raise Stage6SbasError(
            "rc2/ph_patch IFG count mismatch"
        )

    count = 0
    mean = np.zeros(
        n_ifg,
        dtype=np.float64,
    )
    M2 = np.zeros(
        (n_ifg, n_ifg),
        dtype=np.float64,
    )

    t0 = time.perf_counter()

    for start in range(0, n_ps, chunk_ps):
        stop = min(
            n_ps,
            start + chunk_ps,
        )

        noise = np.angle(
            rc[start:stop, :]
            * np.conj(pm[start:stop, :])
        ).astype(
            np.float64,
            copy=False,
        )

        m = noise.shape[0]
        block_mean = np.mean(
            noise,
            axis=0,
            dtype=np.float64,
        )

        centered = (
            noise
            - block_mean[None, :]
        )

        block_M2 = (
            centered.T
            @ centered
        )

        if count == 0:
            mean[:] = block_mean
            M2[:] = block_M2
            count = m
        else:
            total = count + m
            delta = block_mean - mean

            M2 += (
                block_M2
                + np.outer(delta, delta)
                * (
                    count
                    * m
                    / float(total)
                )
            )

            mean += (
                delta
                * m
                / float(total)
            )
            count = total

        if progress:
            elapsed = (
                time.perf_counter()
                - t0
            )
            frac = stop / n_ps
            rate = (
                stop / elapsed
                if elapsed > 0
                else 0.0
            )
            eta = (
                (n_ps - stop) / rate
                if rate > 0
                else 0.0
            )

            print(
                "[STAGE6_SBAS][SB_COV] "
                f"{stop:,}/{n_ps:,} "
                f"({100*frac:5.1f}%) "
                f"elapsed={elapsed/60:.1f}m "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )

    if count <= 1:
        return np.eye(
            n_ifg,
            dtype=np.float64,
        )

    cov = M2 / float(count - 1)
    cov = 0.5 * (cov + cov.T)

    return cov


@turbo_blas_stage
def _stage6_sb_invert(
    root: Path,
    *,
    ps2: dict[str, Any],
    parms: dict[str, Any],
    ph_uw_sb: np.ndarray,
    unwrap_ix: np.ndarray,
    progress: bool = True,
) -> dict[str, Any]:
    """
    Port of official StaMPS sb_invert_uw.m.

    Input
    -----
    ph_uw_sb:
        N_PS x N_IFG unwrapped SB interferometric phase.

    Outputs
    -------
    phuw2.mat:
        N_PS x N_IMAGE acquisition phase.

    phuw_sb_res2.mat:
        ph_res, sb_cov, sm_cov.
    """
    n_ps, n_ifg = ph_uw_sb.shape

    n_image = int(
        round(
            _scalar(
                ps2.get("n_image"),
                0,
            )
        )
    )

    master_ix = int(
        round(
            _scalar(
                ps2.get("master_ix"),
                0,
            )
        )
    )

    if n_image <= 0:
        day = np.asarray(
            ps2.get("day"),
        ).reshape(-1)
        n_image = int(day.size)

    if (
        master_ix < 1
        or master_ix > n_image
    ):
        raise Stage6SbasError(
            f"invalid master_ix={master_ix}"
        )

    ifgday_ix = np.asarray(
        ps2.get("ifgday_ix"),
        dtype=np.float64,
    )

    ifgday_ix = np.squeeze(
        ifgday_ix
    )

    if (
        ifgday_ix.ndim == 2
        and ifgday_ix.shape[0] == 2
        and ifgday_ix.shape[1] == n_ifg
    ):
        ifgday_ix = ifgday_ix.T

    if ifgday_ix.shape != (n_ifg, 2):
        raise Stage6SbasError(
            "ps2.ifgday_ix has wrong shape: "
            f"{ifgday_ix.shape}"
        )

    ifgday_ix = np.rint(
        ifgday_ix
    ).astype(
        np.int64
    )

    print(
        "[STAGE6_SBAS][INVERT] "
        "building full 763x763 SB covariance...",
        flush=True,
    )

    cov_chunk = _env_int(
        "PYSTAMPS_STAGE6_SB_COV_CHUNK",
        4096,
    )

    sb_cov = _stage6_sb_covariance(
        root,
        n_ps,
        n_ifg,
        chunk_ps=cov_chunk,
        progress=progress,
    )

    # --------------------------------------------------------
    # Reference exactly as sb_invert_uw / ps_setref
    # --------------------------------------------------------
    ref_ix = _stage6_reference_indices(
        ps2,
        parms,
        n_ps,
    )

    if ref_ix.size:
        if (
            ref_ix.size == n_ps
            and np.array_equal(
                ref_ix,
                np.arange(
                    n_ps,
                    dtype=np.int64,
                ),
            )
        ):
            ref_mean = np.nanmean(
                ph_uw_sb,
                axis=0,
                dtype=np.float64,
            )
        else:
            ref_mean = np.nanmean(
                ph_uw_sb[
                    ref_ix,
                    :
                ],
                axis=0,
                dtype=np.float64,
            )
    else:
        ref_mean = np.zeros(
            n_ifg,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Official SB design matrix
    # --------------------------------------------------------
    G = np.zeros(
        (n_ifg, n_image),
        dtype=np.float64,
    )

    rows = np.arange(
        n_ifg,
        dtype=np.int64,
    )

    G[
        rows,
        ifgday_ix[:, 0] - 1,
    ] = -1.0

    G[
        rows,
        ifgday_ix[:, 1] - 1,
    ] = 1.0

    if np.sum(
        np.abs(
            G[:, master_ix - 1]
        )
    ) == 0:
        raise Stage6SbasError(
            "none of the unwrapped IFGs "
            "include the original master image"
        )

    # Official:
    # G(:,master_ix)=0
    G[:, master_ix - 1] = 0.0

    G2 = G[
        unwrap_ix,
        :
    ]

    nzc = (
        np.sum(
            np.abs(G2),
            axis=0,
        )
        != 0
    )

    G2 = G2[
        :,
        nzc,
    ]

    rank = int(
        np.linalg.matrix_rank(
            G2
        )
    )

    if rank < G2.shape[1]:
        write_mat(
            root / "phuw_sb_res2.mat",
            {
                "sb_cov": sb_cov,
            },
        )

        raise Stage6SbasError(
            "isolated SB subsets: "
            f"rank(G2)={rank}, "
            f"cols={G2.shape[1]}"
        )

    C = sb_cov[
        np.ix_(
            unwrap_ix,
            unwrap_ix,
        )
    ].copy()

    # MATLAB:
    # while rcond(C)<0.001
    #     C=C+eye(size(C,1))*0.01;
    # end
    ridge_iterations = 0

    while True:
        try:
            cond1 = float(
                np.linalg.cond(
                    C,
                    p=1,
                )
            )

            rcond = (
                0.0
                if not np.isfinite(cond1)
                or cond1 <= 0
                else 1.0 / cond1
            )
        except np.linalg.LinAlgError:
            rcond = 0.0

        if rcond >= 0.001:
            break

        C += (
            np.eye(
                C.shape[0],
                dtype=np.float64,
            )
            * 0.01
        )

        ridge_iterations += 1

        if ridge_iterations > 10000:
            raise Stage6SbasError(
                "unable to condition SB covariance"
            )

    print(
        "[STAGE6_SBAS][INVERT] "
        f"ref_ps={ref_ix.size:,}, "
        f"active_images={G2.shape[1]}, "
        f"rank={rank}, "
        f"rcond={rcond:.6g}, "
        f"ridge_iterations={ridge_iterations}",
        flush=True,
    )

    # GLS:
    # x=(G' C^-1 G)^-1 G' C^-1 y
    CinvG = np.linalg.solve(
        C,
        G2,
    )

    normal = (
        G2.T
        @ CinvG
    )

    sm_active = np.linalg.inv(
        normal
    )

    transform = (
        sm_active
        @ CinvG.T
    )

    ph_uw = np.zeros(
        (n_ps, n_image),
        dtype=np.float32,
    )

    ph_res = np.zeros(
        (n_ps, n_ifg),
        dtype=np.float32,
    )

    active_cols = np.flatnonzero(
        nzc
    )

    solve_chunk = _env_int(
        "PYSTAMPS_STAGE6_SB_INVERT_CHUNK",
        2048,
    )

    t0 = time.perf_counter()

    ref_sel = ref_mean[
        unwrap_ix
    ]

    for start in range(
        0,
        n_ps,
        solve_chunk,
    ):
        stop = min(
            n_ps,
            start + solve_chunk,
        )

        Y = (
            ph_uw_sb[
                start:stop,
                :
            ][
                :,
                unwrap_ix
            ].astype(
                np.float64,
                copy=False,
            )
            - ref_sel[None, :]
        )

        X = (
            Y
            @ transform.T
        )

        ph_uw[
            start:stop,
            active_cols
        ] = X.astype(
            np.float32
        )

        ph_res[
            start:stop,
            :
        ] = (
            ph_uw[
                start:stop,
                :
            ].astype(
                np.float64
            )
            @ G.T
        ).astype(
            np.float32
        )

        if progress:
            elapsed = (
                time.perf_counter()
                - t0
            )
            rate = (
                stop / elapsed
                if elapsed > 0
                else 0.0
            )
            eta = (
                (n_ps-stop)/rate
                if rate > 0
                else 0.0
            )

            print(
                "[STAGE6_SBAS][INVERT] "
                f"{stop:,}/{n_ps:,} "
                f"({100*stop/n_ps:5.1f}%) "
                f"elapsed={elapsed/60:.1f}m "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )

    sm_cov = np.zeros(
        (n_image, n_image),
        dtype=np.float64,
    )

    sm_cov[
        np.ix_(
            active_cols,
            active_cols,
        )
    ] = sm_active

    # Official output index additionally includes master.
    nzc_output = nzc.copy()
    nzc_output[
        master_ix - 1
    ] = True

    unwrap_ifg_index_sm = (
        np.arange(
            1,
            n_image + 1,
            dtype=np.float64,
        )[
            nzc_output
        ]
    )

    write_mat(
        root / "phuw2.mat",
        {
            "ph_uw": ph_uw,
            "unwrap_ifg_index_sm":
                unwrap_ifg_index_sm.reshape(
                    1,
                    -1,
                ),
        },
    )

    write_mat(
        root / "phuw_sb_res2.mat",
        {
            "ph_res": ph_res,
            "sb_cov": sb_cov,
            "sm_cov": sm_cov,
        },
    )

    return {
        "n_ps": int(n_ps),
        "n_ifg": int(n_ifg),
        "n_image": int(n_image),
        "master_ix": int(master_ix),
        "ref_ps": int(ref_ix.size),
        "rank_G2": int(rank),
        "active_images_without_master":
            int(active_cols.size),
        "rcond_C": float(rcond),
        "ridge_iterations":
            int(ridge_iterations),
        "phuw2_shape": [
            int(n_ps),
            int(n_image),
        ],
    }


def _backup_partial_outputs(dataset_root: Path) -> Path | None:
    names = (
        "uw_grid.mat",
        "uw_interp.mat",
        "uw_phaseuw.mat",
        "phuw_sb2.mat",
        "phuw2.mat",
        "phuw_sb_res2.mat",
        "uw_space_time.mat",
        "snaphu.conf",
        "snaphu.in",
        "snaphu.out",
        "snaphu.costinfile",
        "snaphu.log",
    )
    existing = [dataset_root / name for name in names if (dataset_root / name).exists()]
    if not existing:
        return None
    backup = (
        dataset_root
        / "_stage6_sbas_backup"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    backup.mkdir(parents=True, exist_ok=True)
    for source in existing:
        shutil.move(str(source), str(backup / source.name))
    return backup


def _run_snaphu_column(
    *,
    index: int,
    work_root: Path,
    snaphu_exe: str,
    ncol: int,
    uw_ph: np.ndarray,
    Z: np.ndarray,
    nzix: np.ndarray,
    rowix: np.ndarray,
    colix: np.ndarray,
    nzrowix: np.ndarray,
    nzcolix: np.ndarray,
    rowcost_base: np.ndarray,
    colcost_base: np.ndarray,
    dph_noise: np.ndarray,
    dph_space_uw: np.ndarray,
    nshortcycle: float,
    ported: Any,
) -> tuple[int, np.ndarray, float]:
    ifg_dir = work_root / f"ifg_{index + 1:04d}"
    ifg_dir.mkdir(parents=True, exist_ok=True)

    conf = ifg_dir / "snaphu.conf"
    conf.write_text(
        "\n".join(
            (
                "INFILE  snaphu.in",
                "OUTFILE snaphu.out",
                "COSTINFILE snaphu.costinfile",
                "STATCOSTMODE  DEFO",
                "INFILEFORMAT  COMPLEX_DATA",
                "OUTFILEFORMAT FLOAT_DATA",
                "",
            )
        ),
        encoding="utf-8",
    )

    rowcost = rowcost_base.copy()
    colcost = colcost_base.copy()
    smooth = (
        np.asarray(dph_space_uw[:, index], dtype=np.float64)
        - np.asarray(dph_noise[:, index], dtype=np.float64)
    )
    wrapped = np.angle(
        np.exp(1j * np.asarray(dph_space_uw[:, index], dtype=np.float64))
    )
    offset_cycle = (wrapped - smooth) / TWO_PI

    offgrid = np.zeros(rowix.shape, dtype=np.int16)
    edge_index = np.abs(rowix[nzrowix]).astype(np.int64) - 1
    offgrid[nzrowix] = _matlab_round(
        offset_cycle[edge_index]
        * np.sign(rowix[nzrowix])
        * nshortcycle
    ).astype(np.int16)
    rowcost[:, 0::4] = -offgrid

    offgrid = np.zeros(colix.shape, dtype=np.int16)
    edge_index = np.abs(colix[nzcolix]).astype(np.int64) - 1
    offgrid[nzcolix] = _matlab_round(
        offset_cycle[edge_index]
        * np.sign(colix[nzcolix])
        * nshortcycle
    ).astype(np.int16)
    colcost[:, 0::4] = offgrid

    # === STAGE6_SNAPHU_FULL_COSTFILE_V1 ===
    #
    # Always generate a brand-new COMPLETE cost file.
    # Do not reuse, append to, or trust a pre-existing partial file.
    #
    # Write rowcost + colcost into one temporary file, fsync it,
    # then atomically replace snaphu.costinfile.
    cost_path = (
        ifg_dir
        / "snaphu.costinfile"
    )

    cost_tmp = (
        ifg_dir
        / "snaphu.costinfile.tmp"
    )

    try:
        cost_tmp.unlink()
    except FileNotFoundError:
        pass

    with cost_tmp.open(
        "wb"
    ) as handle:

        ported._write_binary_matrix(
            handle,
            rowcost,
        )

        ported._write_binary_matrix(
            handle,
            colcost,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        cost_tmp,
        cost_path,
    )

    ifgw = np.asarray(uw_ph[Z - 1, index], dtype=np.complex64)
    ported._write_complex_raster(ifg_dir / "snaphu.in", ifgw)
    ported._run_external_command(
        [snaphu_exe, "-d", "-f", "snaphu.conf", str(ncol)],
        cwd=ifg_dir,
        log_path=ifg_dir / "snaphu.log",
    )
    ifguw = ported._load_float_grid(ifg_dir / "snaphu.out", ncol)

    diff1 = (ifguw[:-1, :] - ifguw[1:, :]).reshape(-1)
    diff1 = diff1[diff1 != 0]
    diff2 = (ifguw[:, :-1] - ifguw[:, 1:]).reshape(-1)
    diff2 = diff2[diff2 != 0]
    denominator = diff1.size + diff2.size
    msd = (
        float(
            np.sum(diff1.astype(np.float64) ** 2)
            + np.sum(diff2.astype(np.float64) ** 2)
        )
        / float(denominator)
        if denominator > 0
        else 0.0
    )
    values = ported._extract_grid_values_for_ps(ifguw, nzix).astype(np.float32)
    return index, values, msd


# === STAGE6_SBAS_GRID_PARALLEL_V3 ===

def _stage6_grid_available_memory_bytes() -> int:
    """Best-effort Linux available-memory query used only for batch-size safety."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _stage6_grid_windows(
    n_i: int,
    n_j: int,
    n_win: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare exact StaMPS wrap_filt_global windows."""
    n_inc = max(1, int(n_win) // 2)
    n_win_i = max(1, math.ceil(int(n_i) / n_inc) - 1)
    n_win_j = max(1, math.ceil(int(n_j) / n_inc) - 1)

    half = int(n_win) // 2
    x = np.arange(1, half + 1, dtype=np.float32)
    X, Y = np.meshgrid(x, x)
    base = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    base = np.concatenate((base, np.flipud(base)), axis=0).astype(np.float32)

    windows = np.empty((n_win_i * n_win_j, 6), dtype=np.int32)
    position = 0
    for ix1 in range(n_win_i):
        i1 = ix1 * n_inc
        i2 = i1 + int(n_win)
        i_shift = 0
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - int(n_win)

        for ix2 in range(n_win_j):
            j1 = ix2 * n_inc
            j2 = j1 + int(n_win)
            j_shift = 0
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - int(n_win)

            windows[position] = (i1, i2, j1, j2, i_shift, j_shift)
            position += 1

    return windows, base


def _stage6_grid_active_windows(
    occupancy: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    occupied = np.asarray(occupancy, dtype=bool)
    integral = np.zeros(
        (occupied.shape[0] + 1, occupied.shape[1] + 1),
        dtype=np.int64,
    )
    integral[1:, 1:] = (
        occupied.astype(np.int64, copy=False)
        .cumsum(axis=0, dtype=np.int64)
        .cumsum(axis=1, dtype=np.int64)
    )

    i1 = windows[:, 0].astype(np.int64)
    i2 = windows[:, 1].astype(np.int64)
    j1 = windows[:, 2].astype(np.int64)
    j2 = windows[:, 3].astype(np.int64)

    counts = (
        integral[i2, j2]
        - integral[i1, j2]
        - integral[i2, j1]
        + integral[i1, j1]
    )
    return np.flatnonzero(counts > 0).astype(np.int64)


def _stage6_window_weights(
    base_weight: np.ndarray,
    windows: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    n_win = int(base_weight.shape[0])
    output = np.empty((indices.size, n_win, n_win), dtype=np.float32)

    for local, prepared_index in enumerate(indices):
        i_shift = int(windows[int(prepared_index), 4])
        j_shift = int(windows[int(prepared_index), 5])

        weight = base_weight
        if i_shift > 0:
            shifted = np.zeros_like(base_weight)
            shifted[i_shift:, :] = base_weight[: n_win - i_shift, :]
            weight = shifted

        if j_shift > 0:
            shifted = np.zeros_like(base_weight)
            shifted[:, j_shift:] = weight[:, : n_win - j_shift]
            weight = shifted

        output[local] = weight

    return output


def _stage6_goldstein_filter_dense_batch(
    grid_stack: np.ndarray,
    *,
    n_win: int,
    alpha: float,
    gold_flag: bool,
    fft_workers: int,
    window_batch: int,
    windows: np.ndarray | None = None,
    base_weight: np.ndarray | None = None,
    active_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized equivalent of ported._wrap_filt_global for several IFGs.

    Window order, overlap-add order, weighting, Gaussian spectral smoothing,
    Goldstein exponent and low-pass kernel mirror the legacy implementation.
    Empty windows are skipped because their exact contribution is zero.
    """
    source = np.asarray(grid_stack, dtype=np.complex64)
    if source.ndim != 3:
        raise Stage6SbasError("GRID batch input must be [row, col, ifg]")

    n_i, n_j, n_ifg_batch = source.shape
    n_win = int(n_win)
    n_pad = int(round(n_win * 0.25))
    n_ex = n_win + n_pad

    if windows is None or base_weight is None:
        windows, base_weight = _stage6_grid_windows(n_i, n_j, n_win)
    if active_indices is None:
        active_indices = _stage6_grid_active_windows(
            np.any(source != 0, axis=2),
            windows,
        )

    filtered_out = np.zeros(source.shape, dtype=np.complex64, order="F")
    lowpass_out = np.zeros(source.shape, dtype=np.complex64, order="F")
    if active_indices.size == 0:
        return filtered_out, lowpass_out

    import pystamps.pipeline.ported as _ported

    gaussian = np.asarray(_ported._gausswin(7), dtype=np.float64)
    g16 = np.asarray(_ported._gausswin(n_ex, alpha=16.0), dtype=np.float64)
    low_kernel = np.fft.ifftshift(np.outer(g16, g16)).astype(np.float64)

    window_batch = max(1, min(int(window_batch), int(active_indices.size)))
    fft_workers = max(1, int(fft_workers))

    for batch_start in range(0, int(active_indices.size), window_batch):
        batch_indices = active_indices[batch_start : batch_start + window_batch]
        current = int(batch_indices.size)

        phase = np.zeros(
            (current, n_ifg_batch, n_ex, n_ex),
            dtype=np.complex128,
        )
        for local, prepared_index in enumerate(batch_indices):
            i1, i2, j1, j2 = (
                int(value)
                for value in windows[int(prepared_index), :4]
            )
            phase[local, :, :n_win, :n_win] = np.moveaxis(
                source[i1:i2, j1:j2, :],
                2,
                0,
            )

        phase_fft = scipy_fft.fft2(
            phase,
            axes=(-2, -1),
            workers=fft_workers,
        )
        amplitude = np.abs(phase_fft)
        shifted = scipy_fft.fftshift(amplitude, axes=(-2, -1))
        smooth_first = ndimage.convolve1d(
            shifted,
            gaussian,
            axis=-2,
            mode="constant",
            cval=0.0,
        )
        smooth_second = ndimage.convolve1d(
            smooth_first,
            gaussian,
            axis=-1,
            mode="constant",
            cval=0.0,
        )
        spectrum = scipy_fft.ifftshift(smooth_second, axes=(-2, -1))
        median = np.median(
            spectrum,
            axis=(-2, -1),
            keepdims=True,
        )
        np.divide(
            spectrum,
            median,
            out=spectrum,
            where=median != 0,
        )
        np.power(spectrum, float(alpha), out=spectrum)

        goldstein = scipy_fft.ifft2(
            phase_fft * spectrum,
            axes=(-2, -1),
            workers=fft_workers,
        )
        lowpass = scipy_fft.ifft2(
            phase_fft * low_kernel[None, None, :, :],
            axes=(-2, -1),
            workers=fft_workers,
        )

        weights = _stage6_window_weights(
            base_weight,
            windows,
            batch_indices,
        )
        for local, prepared_index in enumerate(batch_indices):
            i1, i2, j1, j2 = (
                int(value)
                for value in windows[int(prepared_index), :4]
            )
            weight = weights[local, :, :, None]

            if gold_flag:
                contribution = (
                    np.moveaxis(
                        goldstein[local, :, :n_win, :n_win],
                        0,
                        2,
                    )
                    * weight
                )
                filtered_out[i1:i2, j1:j2, :] = (
                    filtered_out[i1:i2, j1:j2, :]
                    + contribution
                ).astype(np.complex64)

            low_contribution = (
                np.moveaxis(
                    lowpass[local, :, :n_win, :n_win],
                    0,
                    2,
                )
                * weight
            )
            lowpass_out[i1:i2, j1:j2, :] = (
                lowpass_out[i1:i2, j1:j2, :]
                + low_contribution
            ).astype(np.complex64)

    magnitude = np.abs(source).astype(np.float32, copy=False)
    if gold_flag:
        filtered_out = (
            magnitude
            * np.exp(1j * np.angle(filtered_out))
        ).astype(np.complex64)
    else:
        filtered_out = source.copy()

    lowpass_out = (
        magnitude
        * np.exp(1j * np.angle(lowpass_out))
    ).astype(np.complex64)
    return filtered_out, lowpass_out


def _stage6_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage6_build_grid_batched(
    *,
    ph_in: np.ndarray,
    lin0: np.ndarray,
    n_i: int,
    n_j: int,
    n_win: int,
    alpha: float,
    gold_flag: bool,
    work_dir: Path,
    progress: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    dict[str, Any],
]:
    """
    Build/filter the SBAS grid in resumable IFG batches.

    Full [grid_row, grid_col, all_ifg] arrays are never allocated. Filtered
    grid-point outputs are persistent NPY memmaps, so completed IFGs survive
    interruption and can be resumed.
    """
    source = np.asarray(ph_in, dtype=np.complex64)
    n_ps, n_ifg = source.shape

    # === STAGE6_GRID_V3_AUTOTUNE ===
    #
    # V2 parallelised only the FFT calls while the outer IFG loop remained
    # serial. V3 parallelises independent IFG chunks and keeps each worker's
    # FFT internally narrow to avoid nested oversubscription.
    cpu_count = max(1, os.cpu_count() or 1)

    def _grid_fraction_env(
        name: str,
        default: float,
        *,
        maximum: float,
    ) -> float:
        raw = os.environ.get(name)
        value = float(default) if raw is None else float(raw)
        if not np.isfinite(value) or value <= 0.0 or value > maximum:
            raise Stage6SbasError(
                f"{name} must be in (0, {maximum}]"
            )
        return value

    target_cpu_fraction = _grid_fraction_env(
        "PYSTAMPS_STAGE6_GRID_CPU_FRACTION",
        0.955,
        maximum=1.0,
    )
    memory_fraction = _grid_fraction_env(
        "PYSTAMPS_STAGE6_GRID_MEMORY_FRACTION",
        0.92,
        maximum=0.95,
    )

    # Small/medium IFG chunks give the outer scheduler enough independent
    # work to saturate the CPU without making individual workspaces huge.
    ifg_batch = _env_int(
        "PYSTAMPS_STAGE6_GRID_IFG_BATCH",
        4,
    )
    window_batch = _env_int(
        "PYSTAMPS_STAGE6_GRID_WINDOW_BATCH",
        128,
    )

    # Outer IFG parallelism is now the primary parallel layer.
    # Keep inner FFT single-threaded by default.
    fft_workers = _env_int(
        "PYSTAMPS_STAGE6_GRID_FFT_WORKERS",
        1,
    )

    default_outer_workers = max(
        1,
        int(
            math.ceil(
                cpu_count
                * target_cpu_fraction
                / max(1, fft_workers)
            )
        ),
    )

    outer_workers_requested = _env_int(
        "PYSTAMPS_STAGE6_GRID_OUTER_WORKERS",
        default_outer_workers,
    )

    # Grid arrays are synchronised less frequently than V2.
    checkpoint_ifgs = _env_int(
        "PYSTAMPS_STAGE6_GRID_CHECKPOINT_IFGS",
        32,
    )

    resume = _env_bool(
        "PYSTAMPS_STAGE6_GRID_RESUME",
        True,
    )

    ifg_batch = max(
        1,
        min(
            int(ifg_batch),
            int(n_ifg),
        ),
    )

    order = np.argsort(np.asarray(lin0, dtype=np.int64), kind="stable")
    lin_sorted = np.asarray(lin0, dtype=np.int64)[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(lin_sorted) != 0).astype(np.int64) + 1,
        )
    )
    group_lin = lin_sorted[starts]

    first_values = np.add.reduceat(
        np.asarray(source[order, 0], dtype=np.complex64),
        starts,
        axis=0,
    )
    first_flat = np.zeros(int(n_i) * int(n_j), dtype=np.complex64)
    first_flat[group_lin] = first_values
    nz_flat = first_flat != 0
    if not np.any(nz_flat):
        raise Stage6SbasError("uw_grid has no non-zero points")

    nz_lin = np.flatnonzero(nz_flat).astype(np.int64)
    nz_i = nz_lin % int(n_i) + 1
    nz_j = nz_lin // int(n_i) + 1
    n_grid_ps = int(nz_lin.size)

    occupancy = nz_flat.reshape((int(n_i), int(n_j)), order="F")
    windows, base_weight = _stage6_grid_windows(
        int(n_i),
        int(n_j),
        int(n_win),
    )
    active_indices = _stage6_grid_active_windows(
        occupancy,
        windows,
    )

    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [
                int(n_ps),
                int(n_ifg),
                int(n_i),
                int(n_j),
                int(n_grid_ps),
                int(n_win),
                int(bool(gold_flag)),
            ],
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(np.asarray([float(alpha)], dtype=np.float64).tobytes())
    digest.update(np.asarray(lin0, dtype=np.int64).tobytes())
    signature = digest.hexdigest()

    grid_dir = Path(work_dir) / "grid_v2"
    grid_dir.mkdir(parents=True, exist_ok=True)
    meta_path = grid_dir / "meta.json"
    phase_path = grid_dir / "grid_phase.npy"
    low_path = grid_dir / "grid_lowpass.npy"
    done_path = grid_dir / "done.npy"

    existing_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

    can_resume = (
        resume
        and existing_meta.get("signature") == signature
        and phase_path.exists()
        and low_path.exists()
        and done_path.exists()
    )

    if not can_resume:
        for path in (phase_path, low_path, done_path, meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        grid_phase = np.lib.format.open_memmap(
            phase_path,
            mode="w+",
            dtype=np.complex64,
            shape=(n_grid_ps, n_ifg),
        )
        grid_lowpass = np.lib.format.open_memmap(
            low_path,
            mode="w+",
            dtype=np.complex64,
            shape=(n_grid_ps, n_ifg),
        )
        done = np.lib.format.open_memmap(
            done_path,
            mode="w+",
            dtype=np.uint8,
            shape=(n_ifg,),
        )
        grid_phase[:] = 0
        grid_lowpass[:] = 0
        done[:] = 0
        grid_phase.flush()
        grid_lowpass.flush()
        done.flush()
    else:
        grid_phase = np.lib.format.open_memmap(phase_path, mode="r+")
        grid_lowpass = np.lib.format.open_memmap(low_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")

    # === STAGE6_SBAS_GRID_PARALLEL_V3 ===

    # Keep the persistent done memmap unchanged until a checkpoint has
    # safely flushed the corresponding grid columns. This prevents an
    # interrupted process from recording a completed IFG before its phase
    # arrays have reached persistent storage.
    done_state = np.asarray(
        done,
        dtype=np.uint8,
    ).copy()

    completed_initial = int(
        np.count_nonzero(
            done_state
        )
    )

    grid_started = time.perf_counter()

    pending_batches: list[
        tuple[int, int]
    ] = []

    for ifg_start in range(
        0,
        n_ifg,
        ifg_batch,
    ):
        ifg_stop = min(
            ifg_start + ifg_batch,
            n_ifg,
        )

        if np.all(
            done_state[
                ifg_start:ifg_stop
            ] != 0
        ):
            continue

        pending_batches.append(
            (
                int(ifg_start),
                int(ifg_stop),
            )
        )

    pending_batch_count = len(
        pending_batches
    )

    # --------------------------------------------------------------
    # Memory-aware outer worker planning.
    #
    # The estimate deliberately includes:
    #   dense grid input/output,
    #   complex FFT workspaces,
    #   spectral smoothing arrays,
    #   Goldstein + low-pass results,
    #   grouping/selection overhead.
    #
    # It is intentionally conservative.
    # --------------------------------------------------------------

    available_memory = (
        _stage6_grid_available_memory_bytes()
    )

    n_ex = int(
        n_win
        + round(
            int(n_win)
            * 0.25
        )
    )

    active_window_batch = max(
        1,
        min(
            int(window_batch),
            max(
                1,
                int(
                    active_indices.size
                ),
            ),
        ),
    )

    # Dense grid arrays:
    # grid_stack + filtered + lowpass + temporary allowance.
    dense_grid_bytes = (
        int(n_i)
        * int(n_j)
        * int(ifg_batch)
        * 40
    )

    # Conservative per-element allowance for:
    # phase, FFT, amplitudes, shifted spectra, two smooth workspaces,
    # spectrum, Goldstein and low-pass inverse transforms.
    window_elements = (
        int(active_window_batch)
        * int(ifg_batch)
        * int(n_ex)
        * int(n_ex)
    )

    window_workspace_bytes = (
        window_elements
        * 144
    )

    ps_workspace_bytes = (
        int(n_ps)
        * int(ifg_batch)
        * 24
    )

    estimated_worker_bytes = max(
        64 * 1024 * 1024,
        int(
            dense_grid_bytes
            + window_workspace_bytes
            + ps_workspace_bytes
        ),
    )

    if available_memory > 0:
        memory_budget_bytes = max(
            estimated_worker_bytes,
            int(
                available_memory
                * memory_fraction
            ),
        )

        memory_limited_workers = max(
            1,
            int(
                memory_budget_bytes
                // estimated_worker_bytes
            ),
        )
    else:
        memory_budget_bytes = 0
        memory_limited_workers = (
            outer_workers_requested
        )

    outer_workers = max(
        1,
        min(
            int(
                outer_workers_requested
            ),
            int(
                memory_limited_workers
            ),
            max(
                1,
                int(
                    pending_batch_count
                ),
            ),
        ),
    )

    metadata = {
        "signature": signature,
        "status": "running",
        "engine": "GRID_V3_parallel_ifg",
        "n_ps": int(n_ps),
        "n_ifg": int(n_ifg),
        "n_i": int(n_i),
        "n_j": int(n_j),
        "n_grid_ps": int(n_grid_ps),
        "total_windows": int(
            windows.shape[0]
        ),
        "active_windows": int(
            active_indices.size
        ),
        "ifg_batch": int(
            ifg_batch
        ),
        "window_batch": int(
            window_batch
        ),
        "fft_workers": int(
            fft_workers
        ),
        "outer_workers_requested": int(
            outer_workers_requested
        ),
        "outer_workers": int(
            outer_workers
        ),
        "cpu_count": int(
            cpu_count
        ),
        "target_cpu_fraction": float(
            target_cpu_fraction
        ),
        "memory_fraction": float(
            memory_fraction
        ),
        "available_memory_gb": (
            float(
                available_memory
                / (1024.0 ** 3)
            )
            if available_memory > 0
            else None
        ),
        "memory_budget_gb": (
            float(
                memory_budget_bytes
                / (1024.0 ** 3)
            )
            if memory_budget_bytes > 0
            else None
        ),
        "estimated_worker_gb": float(
            estimated_worker_bytes
            / (1024.0 ** 3)
        ),
        "memory_limited_workers": int(
            memory_limited_workers
        ),
        "checkpoint_ifgs": int(
            checkpoint_ifgs
        ),
        "completed": int(
            completed_initial
        ),
        "resume": bool(
            can_resume
        ),
        "updated_epoch_sec": time.time(),
    }

    _stage6_atomic_json(
        meta_path,
        metadata,
    )

    print(
        "[STAGE6_SBAS][GRID_V3] "
        f"windows={windows.shape[0]}, "
        f"active={active_indices.size}, "
        f"ifg_batch={ifg_batch}, "
        f"window_batch={window_batch}, "
        f"outer_workers={outer_workers}/"
        f"{outer_workers_requested}, "
        f"fft_workers={fft_workers}, "
        f"cpu_target="
        f"{100.0 * target_cpu_fraction:.0f}%, "
        f"mem_budget="
        f"{100.0 * memory_fraction:.0f}%, "
        f"worker_est="
        f"{estimated_worker_bytes / (1024.0 ** 3):.2f} GB, "
        f"resumed={completed_initial}/{n_ifg}",
        flush=True,
    )


    def _process_grid_ifg_batch(
        ifg_start: int,
        ifg_stop: int,
    ) -> tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute one independent IFG chunk.

        The worker only reads shared source arrays. Persistent memmaps are
        written exclusively by the main thread.
        """

        values = np.asarray(
            source[
                order,
                ifg_start:ifg_stop,
            ],
            dtype=np.complex64,
        )

        grouped = np.add.reduceat(
            values,
            starts,
            axis=0,
        )

        current_batch = int(
            ifg_stop
            - ifg_start
        )

        grid_stack = np.zeros(
            (
                int(n_i),
                int(n_j),
                current_batch,
            ),
            dtype=np.complex64,
            order="F",
        )

        flat_stack = grid_stack.reshape(
            (
                int(n_i)
                * int(n_j),
                current_batch,
            ),
            order="F",
        )

        flat_stack[
            group_lin,
            :,
        ] = grouped

        filtered, lowpass = (
            _stage6_goldstein_filter_dense_batch(
                grid_stack,
                n_win=int(n_win),
                alpha=float(alpha),
                gold_flag=bool(
                    gold_flag
                ),
                fft_workers=int(
                    fft_workers
                ),
                window_batch=int(
                    window_batch
                ),
                windows=windows,
                base_weight=base_weight,
                active_indices=(
                    active_indices
                ),
            )
        )

        selected_filtered = (
            filtered.reshape(
                (
                    int(n_i)
                    * int(n_j),
                    current_batch,
                ),
                order="F",
            )[
                nz_flat,
                :,
            ]
            .astype(
                np.complex64,
                copy=True,
            )
        )

        selected_lowpass = (
            lowpass.reshape(
                (
                    int(n_i)
                    * int(n_j),
                    current_batch,
                ),
                order="F",
            )[
                nz_flat,
                :,
            ]
            .astype(
                np.complex64,
                copy=True,
            )
        )

        return (
            int(ifg_start),
            int(ifg_stop),
            selected_filtered,
            selected_lowpass,
        )


    checkpoint_pending = 0

    def _checkpoint_grid_v3(
        *,
        force: bool = False,
    ) -> None:
        nonlocal checkpoint_pending

        if (
            not force
            and checkpoint_pending
            < checkpoint_ifgs
        ):
            return

        # Correct persistence order:
        # 1. phase arrays
        # 2. done bitmap
        grid_phase.flush()
        grid_lowpass.flush()

        done[:] = done_state
        done.flush()

        checkpoint_pending = 0

        metadata[
            "updated_epoch_sec"
        ] = time.time()

        _stage6_atomic_json(
            meta_path,
            metadata,
        )


    def _commit_grid_result(
        result: tuple[
            int,
            int,
            np.ndarray,
            np.ndarray,
        ],
    ) -> None:
        nonlocal checkpoint_pending

        (
            ifg_start,
            ifg_stop,
            selected_filtered,
            selected_lowpass,
        ) = result

        grid_phase[
            :,
            ifg_start:ifg_stop,
        ] = selected_filtered

        grid_lowpass[
            :,
            ifg_start:ifg_stop,
        ] = selected_lowpass

        done_state[
            ifg_start:ifg_stop
        ] = 1

        checkpoint_pending += int(
            ifg_stop
            - ifg_start
        )

        completed = int(
            np.count_nonzero(
                done_state
            )
        )

        elapsed = (
            time.perf_counter()
            - grid_started
        )

        newly_completed = max(
            1,
            completed
            - completed_initial,
        )

        rate = (
            newly_completed
            / elapsed
            if elapsed > 0
            else 0.0
        )

        eta = (
            (n_ifg - completed)
            / rate
            if rate > 0
            else float("nan")
        )

        metadata.update(
            {
                "completed": completed,
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                "updated_epoch_sec": (
                    time.time()
                ),
            }
        )

        _checkpoint_grid_v3()

        if progress:
            print(
                "[STAGE6_SBAS][GRID_V3] "
                f"{completed}/{n_ifg} "
                f"({100.0 * completed / n_ifg:.1f}%), "
                f"elapsed={elapsed:.1f}s, "
                f"eta={eta:.1f}s",
                flush=True,
            )


    try:
        if pending_batches:

            if outer_workers <= 1:

                for (
                    ifg_start,
                    ifg_stop,
                ) in pending_batches:

                    result = (
                        _process_grid_ifg_batch(
                            ifg_start,
                            ifg_stop,
                        )
                    )

                    _commit_grid_result(
                        result
                    )

            else:

                with ThreadPoolExecutor(
                    max_workers=int(
                        outer_workers
                    ),
                    thread_name_prefix=(
                        "stage6-grid-v3"
                    ),
                ) as executor:

                    future_map = {
                        executor.submit(
                            _process_grid_ifg_batch,
                            ifg_start,
                            ifg_stop,
                        ): (
                            ifg_start,
                            ifg_stop,
                        )
                        for (
                            ifg_start,
                            ifg_stop,
                        )
                        in pending_batches
                    }

                    for future in as_completed(
                        future_map
                    ):
                        result = future.result()

                        _commit_grid_result(
                            result
                        )

        _checkpoint_grid_v3(
            force=True
        )

    except BaseException as exc:

        # Preserve every result that was already committed before propagating
        # the error/interruption.
        grid_phase.flush()
        grid_lowpass.flush()

        done[:] = done_state
        done.flush()

        metadata.update(
            {
                "status": "failed",
                "completed": int(
                    np.count_nonzero(
                        done_state
                    )
                ),
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "elapsed_sec": (
                    time.perf_counter()
                    - grid_started
                ),
                "updated_epoch_sec": (
                    time.time()
                ),
            }
        )

        _stage6_atomic_json(
            meta_path,
            metadata,
        )

        raise

    metadata.update(
        {
            "status": "completed",
            "completed": int(n_ifg),
            "elapsed_sec": time.perf_counter() - grid_started,
            "eta_sec": 0.0,
            "updated_epoch_sec": time.time(),
        }
    )
    _stage6_atomic_json(meta_path, metadata)

    return (
        grid_phase,
        grid_lowpass,
        nz_flat,
        nz_i,
        nz_j,
        n_grid_ps,
        metadata,
    )


# === STAGE6_SNAPHU_AUTOTUNE_FUNCTIONS_V1 ===

def _stage6_snaphu_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:

    raw = os.environ.get(
        name
    )

    if raw is None:
        value = float(
            default
        )
    else:
        try:
            value = float(
                raw
            )
        except ValueError as exc:
            raise Stage6SbasError(
                f"{name} must be numeric; "
                f"got {raw!r}"
            ) from exc

    return float(
        min(
            maximum,
            max(
                minimum,
                value,
            ),
        )
    )


def _stage6_resolve_snaphu_workers(
    *,
    n_ifg: int,
    n_i: int,
    n_j: int,
    n_grid_ps: int,
    n_edge: int,
    rowcost_bytes: int,
    colcost_bytes: int,
) -> tuple[
    int,
    dict[str, Any],
]:
    """
    Resolve safe IFG-level SNAPHU concurrency from CPU and RAM.

    SNAPHU remains single-IFG / single-tile exactly as before.
    Only independent interferograms are processed concurrently.

    Resource constraints:
      CPU limit = logical CPUs * cpu_fraction
      RAM limit = MemAvailable * memory_fraction / worker_estimate
      final     = min(CPU, RAM, configured cap, number of IFGs)

    An explicit integer PYSTAMPS_STAGE6_SNAPHU_WORKERS remains
    supported but is safety-clamped to the RAM and global cap.
    """

    logical_cpu = max(
        1,
        int(
            os.cpu_count()
            or 1
        ),
    )

    cpu_fraction = (
        _stage6_snaphu_env_float(
            "PYSTAMPS_STAGE6_SNAPHU_CPU_FRACTION",
            0.95,
            minimum=0.10,
            maximum=1.00,
        )
    )

    memory_fraction = (
        _stage6_snaphu_env_float(
            "PYSTAMPS_STAGE6_SNAPHU_MEM_FRACTION",
            0.90,
            minimum=0.10,
            maximum=0.95,
        )
    )

    max_workers_cap = max(
        1,
        _env_int(
            "PYSTAMPS_STAGE6_SNAPHU_MAX_WORKERS",
            32,
        ),
    )

    minimum_worker_mb = max(
        128.0,
        _stage6_snaphu_env_float(
            "PYSTAMPS_STAGE6_SNAPHU_MIN_WORKER_MB",
            512.0,
            minimum=128.0,
            maximum=8192.0,
        ),
    )

    cpu_limit = max(
        1,
        int(
            math.floor(
                logical_cpu
                * cpu_fraction
            )
        ),
    )

    # ----------------------------------------------------------
    # Conservative per-worker RAM estimate
    # ----------------------------------------------------------
    #
    # Python-side worker:
    #   rowcost copy
    #   colcost copy
    #   offgrid / smooth / wrapped / offset-cycle temporaries
    #   complex SNAPHU input raster
    #   float output raster
    #
    # External SNAPHU:
    #   graph / cost / solver work arrays
    #
    # We intentionally keep a generous minimum because the exact
    # SNAPHU internal allocation depends on the raster topology.
    # ----------------------------------------------------------

    grid_cells = max(
        1,
        int(n_i)
        * int(n_j),
    )

    static_cost = (
        int(rowcost_bytes)
        + int(colcost_bytes)
    )

    python_temporaries = (
        # complex input + float output +
        # several float64 temporary vectors
        grid_cells
        * (
            8
            + 4
            + 8
            + 8
        )
        +
        max(
            1,
            int(n_grid_ps),
        )
        * 48
        +
        max(
            1,
            int(n_edge),
        )
        * 64
    )

    margin_bytes = (
        96
        * 1024**2
    )

    estimated_worker_bytes = max(
        int(
            minimum_worker_mb
            * 1024**2
        ),
        int(
            2
            * static_cost
            + python_temporaries
            + margin_bytes
        ),
    )

    available_memory = (
        _stage6_grid_available_memory_bytes()
    )

    if available_memory > 0:

        memory_budget = int(
            available_memory
            * memory_fraction
        )

        memory_limit = max(
            1,
            int(
                memory_budget
                // estimated_worker_bytes
            ),
        )

    else:

        memory_budget = 0

        # If /proc/meminfo is unavailable,
        # rely on CPU + conservative worker cap.
        memory_limit = max(
            1,
            min(
                cpu_limit,
                max_workers_cap,
            ),
        )

    automatic_limit = max(
        1,
        min(
            int(n_ifg),
            cpu_limit,
            memory_limit,
            max_workers_cap,
        ),
    )

    requested_raw = os.environ.get(
        "PYSTAMPS_STAGE6_SNAPHU_WORKERS",
        "auto",
    ).strip().lower()

    if requested_raw in {
        "",
        "auto",
        "0",
    }:

        workers = automatic_limit

        mode = "auto"

        requested = None

    else:

        try:
            requested = int(
                requested_raw
            )
        except ValueError as exc:
            raise Stage6SbasError(
                "PYSTAMPS_STAGE6_SNAPHU_WORKERS "
                "must be 'auto' or an integer; "
                f"got {requested_raw!r}"
            ) from exc

        if requested < 1:
            raise Stage6SbasError(
                "PYSTAMPS_STAGE6_SNAPHU_WORKERS "
                "must be >= 1"
            )

        # Explicit values are still RAM/cap protected.
        workers = max(
            1,
            min(
                requested,
                int(n_ifg),
                memory_limit,
                max_workers_cap,
            ),
        )

        mode = "explicit"

    meta = {
        "mode":
            mode,

        "requested":
            requested,

        "logical_cpu":
            logical_cpu,

        "cpu_fraction":
            float(
                cpu_fraction
            ),

        "cpu_limit":
            int(
                cpu_limit
            ),

        "available_memory_bytes":
            int(
                available_memory
            ),

        "memory_fraction":
            float(
                memory_fraction
            ),

        "memory_budget_bytes":
            int(
                memory_budget
            ),

        "estimated_worker_bytes":
            int(
                estimated_worker_bytes
            ),

        "estimated_worker_gib":
            float(
                estimated_worker_bytes
                / 1024**3
            ),

        "memory_limit":
            int(
                memory_limit
            ),

        "max_workers_cap":
            int(
                max_workers_cap
            ),

        "workers":
            int(
                workers
            ),

        "n_ifg":
            int(
                n_ifg
            ),

        "grid_shape": [
            int(n_i),
            int(n_j),
        ],
    }

    return (
        int(
            workers
        ),
        meta,
    )


def stage6_sbas_unwrap(
    dataset_root: Path,
    backend: str = "auto",
    io_workers: int = 0,
    enable_mat_cache: bool = True,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    triangle_path: str | None = None,
    snaphu_path: str | None = None,
) -> str:
    """Run the StaMPS multiple-master SBAS Stage 6 path."""

    import pystamps.pipeline.ported as ported

    root = Path(dataset_root).expanduser().resolve()
    started = time.perf_counter()
    progress = _env_bool("PYSTAMPS_SBAS_PROGRESS", True)
    # === STAGE6_TURBO_RESOURCE_SUMMARY_V1 ===
    _turbo = turbo_resource_summary()

    print(
        "[STAGE6_SBAS][TURBO] "
        f"cpu={_turbo['logical_cpu']}, "
        f"available_ram="
        f"{_turbo['available_memory_gib']:.1f}GiB, "
        f"edge_chunk={_turbo['edge_chunk']}, "
        f"blas_threads={_turbo['blas_threads']}",
        flush=True,
    )

    edge_chunk = turbo_edge_chunk()
    # === STAGE6_SNAPHU_AUTO_RESOURCE_V1 ===
    #
    # Actual worker count is resolved immediately before SNAPHU,
    # once grid dimensions and cost-array sizes are known.
    #
    # PYSTAMPS_STAGE6_SNAPHU_WORKERS:
    #   auto / unset -> CPU + RAM autotune
    #   integer      -> requested worker count, still safety-clamped
    snaphu_workers = 1
    snaphu_worker_meta: dict[str, Any] = {
        "mode": "pending",
    }
    anneal_runs = _env_int(
        "PYSTAMPS_SBAS_ANNEAL_RUNS",
        15,
    )

    anneal_workers = turbo_workers_from_env(
        "PYSTAMPS_SBAS_ANNEAL_WORKERS",
        task_count=anneal_runs,
        cpu_fraction=0.95,
        cap=anneal_runs,
    )

    strict_anneal = _env_bool("PYSTAMPS_SBAS_STRICT_ANNEAL", True)
    keep_work = _env_bool("PYSTAMPS_SBAS_KEEP_WORK", False)

    debug_path = root / "stage6_sbas_debug.json"
    work_dir = root / "_stage6_sbas_work"
    snaphu_root = work_dir / "snaphu"
    grid_resume = _env_bool("PYSTAMPS_STAGE6_GRID_RESUME", True)
    if work_dir.exists() and not grid_resume:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    backup = _backup_partial_outputs(root)

    try:
        ps2 = read_mat(root / "ps2.mat")
        n_ps = int(round(_scalar(ps2.get("n_ps"), 0)))
        if n_ps <= 0:
            raise Stage6SbasError("ps2.mat missing valid n_ps")

        ph2 = _as_rows(
            read_mat_variables(root / "ph2.mat", ("ph",)).get("ph"),
            n_ps,
            "ph2.ph",
            np.complex64,
        )
        n_ps, n_ifg = ph2.shape

        parms = {}
        parms_path = root / "parms.mat"
        if parms_path.exists():
            parms = read_mat(parms_path)

        if _mat_text(parms.get("small_baseline_flag"), "n").lower() != "y":
            raise Stage6SbasError("stage6_sbas_unwrap requires small_baseline_flag='y'")

        # Do not silently diverge from MATLAB for options that are not yet
        # implemented in this SB port.
        if (
            _mat_text(
                parms.get("unwrap_hold_good_values"),
                "n",
            ).lower()
            == "y"
        ):
            raise Stage6SbasError(
                "unwrap_hold_good_values='y' is not yet MATLAB-parity implemented"
            )

        if (
            _mat_text(
                parms.get("subtr_tropo"),
                "n",
            ).lower()
            == "y"
        ):
            raise Stage6SbasError(
                "subtr_tropo='y' is not yet MATLAB-parity implemented in Stage 6"
            )

        unwrap_method = _mat_text(parms.get("unwrap_method"), "3D_QUICK")
        scf_flag = _mat_text(
            parms.get("unwrap_spatial_cost_func_flag"),
            "n",
        ).lower() == "y"
        if scf_flag:
            raise Stage6SbasError(
                "unwrap_spatial_cost_func_flag='y' is not enabled in this patch. "
                "Set it to 'n' to use the standard StaMPS SBAS 3-D time-space path."
            )

        # === FINAL_IFG_QC_FULL_NETWORK_INPUT_V1 ===
        final_ifg_qc_enabled = bool(
            round(
                _scalar(
                    parms.get(
                        "pystamps_final_ifg_qc_enabled"
                    ),
                    0.0,
                )
            )
        )

        # In automatic production mode FINAL-QC owns the final
        # drop list.  Use a NUMERIC compatibility flag; never pass
        # the literal string "auto" through numeric parms handling.
        final_qc_owns_drop = bool(
            final_ifg_qc_enabled
            and round(
                _scalar(
                    parms.get(
                        "pystamps_final_qc_owns_drop"
                    ),
                    1.0,
                )
            )
        )

        preliminary_drop = _drop_indices(
            parms.get(
                "drop_ifg_index"
            ),
            n_ifg,
        )

        if final_qc_owns_drop:
            if preliminary_drop.size:
                print(
                    "[IFG_FINAL_QC] "
                    "ignoring preliminary "
                    "drop_ifg_index before "
                    "all-IFG Stage6 processing: "
                    + ",".join(
                        str(int(v))
                        for v
                        in preliminary_drop
                    ),
                    flush=True,
                )

            drop = np.empty(
                0,
                dtype=np.int64,
            )
        else:
            drop = preliminary_drop

        drop_set = set(
            int(v)
            for v in drop
        )
        unwrap_ifg = np.asarray(
            [index for index in range(1, n_ifg + 1) if index not in drop_set],
            dtype=np.int64,
        )
        if unwrap_ifg.size == 0:
            raise Stage6SbasError("No interferograms remain after drop_ifg_index")
        unwrap_ix = unwrap_ifg - 1

        day, ifgday_ix_all, bperp_all, network_source = load_sbas_network(root, n_ifg)
        ifgday_ix = ifgday_ix_all[unwrap_ix, :]
        bperp = bperp_all[unwrap_ix]

        unwrap_patch_phase = _mat_text(
            parms.get("unwrap_patch_phase"),
            "n",
        ).lower() == "y"
        pm2_variables = (
            ("K_ps", "ph_patch")
            if unwrap_patch_phase
            else ("K_ps",)
        )
        pm2 = read_mat_variables(root / "pm2.mat", pm2_variables)
        bp2 = read_mat_variables(root / "bp2.mat", ("bperp_mat",))
        bperp_mat = _as_rows(
            bp2.get("bperp_mat"),
            n_ps,
            "bp2.bperp_mat",
            np.float32,
        )
        if bperp_mat.shape[1] != n_ifg:
            raise Stage6SbasError(
                f"bp2.bperp_mat has {bperp_mat.shape[1]} columns; expected {n_ifg}"
            )

        phase_restore: np.ndarray | None = None
        patch_phase_for_restore: np.ndarray | None = None

        if unwrap_patch_phase:
            ph_patch = _as_rows(
                pm2.get("ph_patch"),
                n_ps,
                "pm2.ph_patch",
                np.complex64,
            )
            if ph_patch.shape[1] != n_ifg:
                raise Stage6SbasError(
                    f"pm2.ph_patch has {ph_patch.shape[1]} columns; expected {n_ifg}"
                )
            ph_w = _normalize_complex(ph_patch)
            patch_phase_for_restore = ph_w.copy()
        else:
            rc_path = root / "rc2.mat"
            if rc_path.exists():
                try:
                    ph_w = _as_rows(
                        read_mat_variables(rc_path, ("ph_rc",)).get("ph_rc"),
                        n_ps,
                        "rc2.ph_rc",
                        np.complex64,
                    )
                except Exception:
                    ph_w = ph2.astype(np.complex64, copy=True)
            else:
                ph_w = ph2.astype(np.complex64, copy=True)

            k_raw = pm2.get("K_ps")
            if k_raw is not None and np.asarray(k_raw).size == n_ps:
                K_ps = _as_vector(k_raw, n_ps, "pm2.K_ps", np.float32)
                ph_w *= np.exp(1j * K_ps[:, None] * bperp_mat)

        # StaMPS SBAS SCLA subtraction/add-back, when a previous iteration exists.
        for scla_name in ("scla_smooth_sb2.mat",):
            scla_path = root / scla_name
            if not scla_path.exists():
                continue
            scla = read_mat(scla_path)
            k_uw = scla.get("K_ps_uw")
            if k_uw is not None and np.asarray(k_uw).size == n_ps:
                K_uw = _as_vector(k_uw, n_ps, f"{scla_name}.K_ps_uw", np.float32)
                correction = K_uw[:, None] * bperp_mat
                ph_w *= np.exp(-1j * correction)
                if phase_restore is None:
                    phase_restore = np.zeros((n_ps, n_ifg), dtype=np.float32)
                phase_restore += correction
            ramp = scla.get("ph_ramp")
            if (
                _mat_text(
                    parms.get("scla_deramp"),
                    "n",
                ).lower() == "y"
                and ramp is not None
                and np.asarray(ramp).size
            ):
                ramp_arr = _as_rows(ramp, n_ps, f"{scla_name}.ph_ramp", np.float32)
                if ramp_arr.shape == ph_w.shape:
                    ph_w *= np.exp(-1j * ramp_arr)
                    if phase_restore is None:
                        phase_restore = np.zeros((n_ps, n_ifg), dtype=np.float32)
                    phase_restore += ramp_arr
            break

        ph_w = np.asarray(ph_w, dtype=np.complex64)
        ph_w_magnitude = np.abs(ph_w)
        np.divide(
            ph_w,
            ph_w_magnitude,
            out=ph_w,
            where=ph_w_magnitude != 0,
        )
        del ph_w_magnitude

        pix_size = float(_scalar(parms.get("unwrap_grid_size"), 200.0))
        prefilt_win = int(round(_scalar(parms.get("unwrap_gold_n_win"), 32.0)))
        gold_alpha = float(_scalar(parms.get("unwrap_gold_alpha"), 0.8))
        gold_flag = _mat_text(parms.get("unwrap_prefilter_flag"), "y").lower() == "y"
        if pix_size <= 0:
            pix_size = 200.0
        if prefilt_win <= 0:
            prefilt_win = 32

        xy = _as_rows(ps2.get("xy"), n_ps, "ps2.xy", np.float32)
        if xy.shape[1] < 3:
            raise Stage6SbasError("ps2.xy must contain id, x and y columns")
        x_coord = xy[:, 1]
        y_coord = xy[:, 2]
        grid_x_min = float(np.min(x_coord))
        grid_y_min = float(np.min(y_coord))
        grid_i = np.ceil((y_coord - np.float32(grid_y_min) + np.float32(1e-3)) / np.float32(pix_size)).astype(np.int64)
        grid_j = np.ceil((x_coord - np.float32(grid_x_min) + np.float32(1e-3)) / np.float32(pix_size)).astype(np.int64)
        if grid_i.size and int(np.max(grid_i)) > 1:
            maximum = int(np.max(grid_i))
            grid_i[grid_i == maximum] = maximum - 1
        if grid_j.size and int(np.max(grid_j)) > 1:
            maximum = int(np.max(grid_j))
            grid_j[grid_j == maximum] = maximum - 1
        n_i = int(np.max(grid_i)) if grid_i.size else 1
        n_j = int(np.max(grid_j)) if grid_j.size else 1
        if min(n_i, n_j) < prefilt_win:
            raise Stage6SbasError(
                f"Minimum resampled grid dimension {min(n_i, n_j)} "
                f"is smaller than prefilter window {prefilt_win}"
            )

        if (
            unwrap_ix.size == n_ifg
            and np.array_equal(
                unwrap_ix,
                np.arange(n_ifg, dtype=np.int64),
            )
        ):
            ph_in = ph_w
        else:
            ph_in = np.ascontiguousarray(
                ph_w[:, unwrap_ix],
                dtype=np.complex64,
            )

        lin0 = (
            (grid_j - 1) * n_i
            + (grid_i - 1)
        ).astype(np.int64)

        (
            grid_phase,
            grid_lowpass,
            nz_flat,
            nz_i,
            nz_j,
            n_grid_ps,
            grid_meta,
        ) = _stage6_build_grid_batched(
            ph_in=ph_in,
            lin0=lin0,
            n_i=n_i,
            n_j=n_j,
            n_win=prefilt_win,
            alpha=gold_alpha,
            gold_flag=gold_flag,
            work_dir=work_dir,
            progress=progress,
        )

        # === STAGE6_GRID_IFG_QC_V1 ===
        # Full GRID filtering is completed first. QC then decides which
        # IFGs continue into LA / time-space unwrapping.
        grid_qc_enabled = bool(
            round(
                _scalar(
                    parms.get(
                        "pystamps_grid_qc_enabled"
                    ),
                    0.0,
                )
            )
        )

        # === FINAL_IFG_QC_GRID_PREFLAG_ONLY_V1 ===
        # Preserve the complete GRID/IFG state before GRID-QC.
        # GRID-QC may compute its normal V3 candidate set and audit,
        # but FINAL-QC production mode restores this full state so
        # no IFG is rejected before SNAPHU.
        _final_qc_full_grid_state = None

        if final_qc_owns_drop:
            _final_qc_full_grid_state = (
                grid_phase,
                grid_lowpass,
                ph_in,
                unwrap_ifg.copy(),
                unwrap_ix.copy(),
                ifgday_ix.copy(),
                bperp.copy(),
            )

        if grid_qc_enabled:
            from pystamps.grid_ifg_qc import (
                run_grid_ifg_qc,
            )

            ifg_std_current = None
            ifgstd_path = root / "ifgstd2.mat"

            if ifgstd_path.exists():
                try:
                    _ifgstd_payload = read_mat_variables(
                        ifgstd_path,
                        ("ifg_std",),
                    )

                    _ifgstd_all = np.asarray(
                        _ifgstd_payload.get(
                            "ifg_std"
                        ),
                        dtype=np.float64,
                    ).reshape(-1)

                    if _ifgstd_all.size == n_ifg:
                        ifg_std_current = (
                            _ifgstd_all[
                                unwrap_ix
                            ]
                        )
                except Exception:
                    ifg_std_current = None

            _grid_qc = run_grid_ifg_qc(
                dataset_root=root,
                grid_phase=grid_phase,
                grid_lowpass=grid_lowpass,
                nz_i=nz_i,
                nz_j=nz_j,
                n_i=n_i,
                n_j=n_j,
                original_ifg_index=unwrap_ifg,
                ifgday_ix=ifgday_ix,
                ifg_std=ifg_std_current,
                parms=parms,
            )

            _grid_keep = np.asarray(
                _grid_qc.keep_local_mask,
                dtype=bool,
            )

            if _grid_keep.size != unwrap_ix.size:
                raise Stage6SbasError(
                    "GRID IFG QC keep-mask length "
                    "does not match current IFG network"
                )

            if not np.all(_grid_keep):
                # Persistent GRID memmaps remain untouched so future
                # QC/reprocessing can reuse the complete GRID result.
                # Downstream Stage6 receives compact selected arrays.
                grid_phase = np.ascontiguousarray(
                    np.asarray(
                        grid_phase[:, _grid_keep],
                        dtype=np.complex64,
                    )
                )

                grid_lowpass = np.ascontiguousarray(
                    np.asarray(
                        grid_lowpass[:, _grid_keep],
                        dtype=np.complex64,
                    )
                )

                ph_in = np.ascontiguousarray(
                    np.asarray(
                        ph_in[:, _grid_keep],
                        dtype=np.complex64,
                    )
                )

                unwrap_ifg = np.asarray(
                    unwrap_ifg[_grid_keep],
                    dtype=np.int64,
                )

                unwrap_ix = (
                    unwrap_ifg - 1
                ).astype(np.int64)

                ifgday_ix = np.asarray(
                    ifgday_ix[_grid_keep, :],
                    dtype=np.int64,
                )

                bperp = np.asarray(
                    bperp[_grid_keep],
                )

                drop_set.update(
                    int(v)
                    for v
                    in _grid_qc.drop_original_indices
                )

                parms["drop_ifg_index"] = (
                    np.asarray(
                        sorted(drop_set),
                        dtype=np.float64,
                    ).reshape(-1, 1)
                )

                write_mat(
                    parms_path,
                    parms,
                )

                grid_meta = dict(
                    grid_meta
                )

                grid_meta[
                    "grid_qc_selected_ifg"
                ] = int(
                    unwrap_ix.size
                )

                grid_meta[
                    "grid_qc_dropped_ifg"
                ] = int(
                    len(
                        _grid_qc.drop_original_indices
                    )
                )

                print(
                    "[STAGE6_SBAS][GRID_QC] "
                    f"continuing with "
                    f"{unwrap_ix.size}/{n_ifg} IFGs",
                    flush=True,
                )

        if (
            final_qc_owns_drop
            and _final_qc_full_grid_state
            is not None
        ):
            (
                grid_phase,
                grid_lowpass,
                ph_in,
                unwrap_ifg,
                unwrap_ix,
                ifgday_ix,
                bperp,
            ) = _final_qc_full_grid_state

            # GRID is now evidence only.
            parms[
                "drop_ifg_index"
            ] = np.empty(
                (1, 0),
                dtype=np.float64,
            )

            write_mat(
                parms_path,
                parms,
            )

            grid_selection_path = (
                root
                / "grid_ifg_selection.json"
            )

            if grid_selection_path.exists():
                try:
                    grid_selection = json.loads(
                        grid_selection_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    grid_selection[
                        "role"
                    ] = "preflag_only"

                    grid_selection[
                        "applied_to_stage6"
                    ] = False

                    grid_selection[
                        "suggested_preflag_ifg_index"
                    ] = grid_selection.get(
                        "effective_drop_ifg_index",
                        [],
                    )

                    _write_json(
                        grid_selection_path,
                        grid_selection,
                    )

                except Exception as exc:
                    print(
                        "[IFG_GRID_QC][WARNING] "
                        "could not mark audit as "
                        "preflag_only: "
                        f"{type(exc).__name__}: "
                        f"{exc}",
                        flush=True,
                    )

            print(
                "[IFG_GRID_QC] "
                "V3 audit retained as preflag only; "
                f"restored full {unwrap_ix.size}/{n_ifg} "
                "IFGs for LA/SNAPHU",
                flush=True,
            )

        print(
            "[STAGE6_SBAS] "
            f"n_ps={n_ps}, n_ifg={n_ifg}, selected_ifg={unwrap_ix.size}, "
            f"n_image={day.size}, grid={n_i}x{n_j}, grid_ps={n_grid_ps}, "
            f"network_source={network_source}, "
            f"grid_active_windows={grid_meta['active_windows']}",
            flush=True,
        )

        nzix = nz_flat.reshape((n_i, n_j), order="F")
        grid_ij = np.column_stack((grid_i, grid_j)).astype(np.float64)
        xy_grid = np.column_stack(
            (
                np.arange(1, n_grid_ps + 1, dtype=np.float64),
                (nz_j.astype(np.float64) - 0.5) * pix_size,
                (nz_i.astype(np.float64) - 0.5) * pix_size,
            )
        )
        ij_grid = np.column_stack((nz_i, nz_j)).astype(np.float64)
        uw_grid_payload = {
            "ph": grid_phase,
            "ph_in": ph_in,
            "ph_lowpass": grid_lowpass,
            "ph_uw_predef": np.empty((0, 0), dtype=np.float32),
            "ph_in_predef": np.empty((0, 0), dtype=np.complex64),
            "xy": xy_grid,
            "ij": ij_grid,
            "nzix": nzix,
            "grid_x_min": np.asarray(grid_x_min, dtype=np.float32),
            "grid_y_min": np.asarray(grid_y_min, dtype=np.float32),
            "n_i": np.asarray(n_i, dtype=np.float32),
            "n_j": np.asarray(n_j, dtype=np.float32),
            "n_ifg": np.asarray(unwrap_ix.size, dtype=np.float64),
            "n_ps": np.asarray(n_grid_ps, dtype=np.float64),
            "grid_ij": grid_ij,
            "pix_size": np.asarray(pix_size, dtype=np.float64),
        }
        write_mat(root / "uw_grid.mat", uw_grid_payload)

        uw_interp_payload = ported._build_uw_interp_payload(
            root,
            uw_grid_payload,
            triangle_path=triangle_path,
        )
        write_mat(root / "uw_interp.mat", uw_interp_payload)

        la_flag = _mat_text(parms.get("unwrap_la_error_flag"), "y").lower() == "y"
        time_win = float(_scalar(parms.get("unwrap_time_win"), 730.0))
        max_topo_err = float(_scalar(parms.get("max_topo_err"), 20.0))
        wavelength = float(_scalar(parms.get("lambda"), 0.0555))
        # Official ps_unwrap.m uses fixed approximate radar range rho=830000 m.
        rho = 830000.0

        mean_inc_raw = ps2.get("mean_incidence")
        if (
            mean_inc_raw is not None
            and np.asarray(mean_inc_raw).size
        ):
            mean_incidence = float(
                np.asarray(mean_inc_raw).reshape(-1)[0]
            )
        else:
            la_path = root / "la2.mat"
            if la_path.exists():
                la_payload = read_mat_variables(
                    la_path,
                    ("la",),
                )
                la_value = la_payload.get("la")
                if (
                    la_value is not None
                    and np.asarray(la_value).size
                ):
                    mean_incidence = (
                        float(
                            np.mean(
                                np.asarray(
                                    la_value,
                                    dtype=np.float64,
                                )
                            )
                        )
                        + 0.052
                    )
                else:
                    mean_incidence = math.radians(21.0)
            else:
                mean_incidence = math.radians(21.0)

        denominator = (
            wavelength
            * rho
            * math.sin(mean_incidence)
            / (4.0 * math.pi)
        )
        max_K = max_topo_err / denominator
        n_trial_wraps = float(np.ptp(bperp_all)) * max_K / TWO_PI

        G, dph_noise, dph_space_uw, network_meta = compute_sbas_space_time(
            uw_ph=grid_phase,
            ph_lowpass=grid_lowpass,
            edgs=uw_interp_payload.get("edgs"),
            day=day,
            ifgday_ix=ifgday_ix,
            bperp=bperp,
            time_win=time_win,
            n_trial_wraps=n_trial_wraps,
            unwrap_method=unwrap_method,
            la_flag=la_flag,
            edge_chunk=edge_chunk,
            anneal_workers=anneal_workers,
            anneal_runs=anneal_runs,
            strict_anneal=strict_anneal,
            progress=progress,
            work_dir=work_dir,
        )

        rowix = np.asarray(uw_interp_payload.get("rowix"), dtype=np.float64).reshape((n_i - 1, n_j), order="F").copy()
        colix = np.asarray(uw_interp_payload.get("colix"), dtype=np.float64).reshape((n_i, n_j - 1), order="F").copy()
        Z = np.asarray(uw_interp_payload.get("Z"), dtype=np.int64).reshape((n_i, n_j), order="F")
        n_edge = int(round(_scalar(uw_interp_payload.get("n_edge"), 0)))
        grid_edges = np.concatenate(
            (
                np.abs(colix[np.abs(colix) > 0]),
                np.abs(rowix[np.abs(rowix) > 0]),
            )
        ).astype(np.int64)
        n_edges = np.bincount(grid_edges, minlength=n_edge + 1)[1:]

        sigsq_noise = (
            np.nanstd(np.asarray(dph_noise), axis=1, ddof=1)
            / TWO_PI
        ) ** 2
        bad = ~np.isfinite(sigsq_noise)
        row_abs = np.abs(np.nan_to_num(rowix, nan=0.0)).astype(np.int64)
        col_abs = np.abs(np.nan_to_num(colix, nan=0.0)).astype(np.int64)
        bad_lookup = np.zeros(n_edge + 1, dtype=bool)
        bad_lookup[np.flatnonzero(bad) + 1] = True
        rowix[bad_lookup[row_abs]] = np.nan
        colix[bad_lookup[col_abs]] = np.nan

        costscale = 100.0
        nshortcycle = 200.0
        maxshort = 32000
        sigsq_raw = _matlab_round(
            sigsq_noise * (nshortcycle**2) / costscale * n_edges
        )
        sigsq = np.ones(n_edge, dtype=np.int16)
        finite = np.isfinite(sigsq_raw)
        sigsq[finite] = np.clip(
            sigsq_raw[finite],
            1,
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        nzrowix = np.abs(rowix) > 0
        nzcolix = np.abs(colix) > 0
        rowcost_base = np.zeros((n_i - 1, n_j * 4), dtype=np.int16)
        colcost_base = np.zeros((n_i, (n_j - 1) * 4), dtype=np.int16)
        rowcost_base[:, 2::4] = maxshort
        colcost_base[:, 2::4] = maxshort
        rowcost_base[:, 3::4] = (
            np.asarray(~np.isnan(rowix), dtype=np.int16)
            * (-1 - maxshort)
            + 1
        ).astype(np.int16)
        colcost_base[:, 3::4] = (
            np.asarray(~np.isnan(colix), dtype=np.int16)
            * (-1 - maxshort)
            + 1
        ).astype(np.int16)
        rowstd = np.ones(rowix.shape, dtype=np.int16)
        colstd = np.ones(colix.shape, dtype=np.int16)
        rowstd[nzrowix] = sigsq[np.abs(rowix[nzrowix]).astype(np.int64) - 1]
        colstd[nzcolix] = sigsq[np.abs(colix[nzcolix]).astype(np.int64) - 1]
        rowcost_base[:, 1::4] = rowstd
        colcost_base[:, 1::4] = colstd

        # === STAGE6_SNAPHU_AUTO_RESOLVE_V1 ===
        snaphu_workers, snaphu_worker_meta = (
            _stage6_resolve_snaphu_workers(
                n_ifg=int(
                    unwrap_ix.size
                ),
                n_i=int(
                    n_i
                ),
                n_j=int(
                    n_j
                ),
                n_grid_ps=int(
                    n_grid_ps
                ),
                n_edge=int(
                    n_edge
                ),
                rowcost_bytes=int(
                    rowcost_base.nbytes
                ),
                colcost_bytes=int(
                    colcost_base.nbytes
                ),
            )
        )

        print(
            "[STAGE6_SBAS][SNAPHU_AUTO] "
            f"mode={snaphu_worker_meta['mode']}, "
            f"workers={snaphu_workers}, "
            f"cpu={snaphu_worker_meta['logical_cpu']}, "
            f"cpu_limit={snaphu_worker_meta['cpu_limit']}, "
            f"mem_limit={snaphu_worker_meta['memory_limit']}, "
            f"mem_budget="
            f"{snaphu_worker_meta['memory_budget_bytes']/1024**3:.1f}GiB, "
            f"worker_est="
            f"{snaphu_worker_meta['estimated_worker_gib']:.2f}GiB, "
            f"cap={snaphu_worker_meta['max_workers_cap']}",
            flush=True,
        )

        snaphu_exe = ported._resolve_external_tool("snaphu", snaphu_path)
        ph_uw_some = np.memmap(
            work_dir / "ph_uw_grid.f32",
            mode="w+",
            dtype=np.float32,
            shape=(n_grid_ps, unwrap_ix.size),
        )
        msd_some = np.zeros(unwrap_ix.size, dtype=np.float64)
        snaphu_root.mkdir(parents=True, exist_ok=True)

        completed = 0
        snaphu_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=snaphu_workers) as pool:
            futures = [
                pool.submit(
                    _run_snaphu_column,
                    index=index,
                    work_root=snaphu_root,
                    snaphu_exe=snaphu_exe,
                    ncol=n_j,
                    uw_ph=grid_phase,
                    Z=Z,
                    nzix=nzix,
                    rowix=rowix,
                    colix=colix,
                    nzrowix=nzrowix,
                    nzcolix=nzcolix,
                    rowcost_base=rowcost_base,
                    colcost_base=colcost_base,
                    dph_noise=dph_noise,
                    dph_space_uw=dph_space_uw,
                    nshortcycle=nshortcycle,
                    ported=ported,
                )
                for index in range(unwrap_ix.size)
            ]
            for future in as_completed(futures):
                index, values, msd_value = future.result()
                ph_uw_some[:, index] = values
                msd_some[index] = msd_value
                completed += 1

                progress_every = max(
                    1,
                    _env_int(
                        "PYSTAMPS_STAGE6_SNAPHU_PROGRESS_EVERY",
                        8,
                    ),
                )

                if (
                    completed == 1
                    or completed
                    % progress_every
                    == 0
                    or completed
                    == unwrap_ix.size
                ):

                    elapsed = max(
                        1e-9,
                        time.perf_counter()
                        - snaphu_started,
                    )

                    rate = (
                        completed
                        / elapsed
                    )

                    remaining = (
                        unwrap_ix.size
                        - completed
                    )

                    eta = (
                        remaining
                        / rate
                        if rate > 0
                        else 0.0
                    )

                    print(
                        "[STAGE6_SBAS][SNAPHU] "
                        f"{completed}/{unwrap_ix.size} "
                        f"({100.0 * completed / unwrap_ix.size:.1f}%) "
                        f"workers={snaphu_workers} "
                        f"rate={rate:.2f} IFG/s "
                        f"elapsed={elapsed:.1f}s "
                        f"ETA={eta:.1f}s",
                        flush=True,
                    )
        ph_uw_some.flush()

        write_mat(
            root / "uw_phaseuw.mat",
            {
                "ph_uw": np.asarray(ph_uw_some),
                "msd": msd_some.reshape(-1, 1),
            },
        )

        gridix_flat = np.zeros(n_i * n_j, dtype=np.int64)
        nz_flat_f = np.flatnonzero(nzix.reshape(-1, order="F"))
        gridix_flat[nz_flat_f] = np.arange(1, n_grid_ps + 1, dtype=np.int64)
        gridix = gridix_flat.reshape((n_i, n_j), order="F")
        ps_grid_idx = gridix[grid_i - 1, grid_j - 1]

        ph_uw_selected = np.full(
            (n_ps, unwrap_ix.size),
            np.nan,
            dtype=np.float32,
        )
        valid = ps_grid_idx > 0
        if np.any(valid):
            grid_values = np.asarray(ph_uw_some[ps_grid_idx[valid] - 1, :], dtype=np.float32)
            ph_uw_selected[valid, :] = (
                grid_values
                + np.angle(
                    ph_in[valid, :]
                    * np.exp(-1j * grid_values)
                ).astype(np.float32)
            )
            if phase_restore is not None:
                ph_uw_selected[valid, :] += (
                    phase_restore[valid, :][:, unwrap_ix]
                )

        if unwrap_patch_phase and patch_phase_for_restore is not None and (root / "rc2.mat").exists():
            rc = _as_rows(
                read_mat_variables(root / "rc2.mat", ("ph_rc",)).get("ph_rc"),
                n_ps,
                "rc2.ph_rc",
                np.complex64,
            )
            residual = np.angle(
                rc[:, unwrap_ix]
                * np.conj(patch_phase_for_restore[:, unwrap_ix])
            ).astype(np.float32)
            ph_uw_selected += residual

        # --------------------------------------------------------
        # Official ps_unwrap SB output:
        # phuw_sb2 = N_PS x N_IFG
        # --------------------------------------------------------
        ph_uw_sb = np.zeros(
            (n_ps, n_ifg),
            dtype=np.float32,
        )

        msd = np.zeros(
            n_ifg,
            dtype=np.float32,
        )

        ph_uw_sb[
            :,
            unwrap_ix
        ] = ph_uw_selected

        msd[
            unwrap_ix
        ] = msd_some.astype(
            np.float32
        )

        write_mat(
            root / "phuw_sb2.mat",
            {
                "ph_uw": ph_uw_sb,
                "msd": msd.reshape(-1, 1),
            },
        )

        print(
            "[STAGE6_SBAS] "
            "phuw_sb2 written: "
            f"{ph_uw_sb.shape}",
            flush=True,
        )

        # --------------------------------------------------------
        # Official stamps.m immediately runs sb_invert_uw.
        #
        # FINAL-QC production extension:
        #
        #   1. first inversion uses ALL IFGs
        #   2. calculate post-unwrapped network consistency
        #   3. combine GRID + MSD + NETWORK families
        #   4. protect network connectivity
        #   5. only SB inversion is repeated with final selected IFGs
        #
        # GRID / LA / SNAPHU are NOT repeated.
        # --------------------------------------------------------

        # === FINAL_IFG_QC_POST_UNWRAP_V1 ===

        first_pass_inversion_meta = (
            _stage6_sb_invert(
                root,
                ps2=ps2,
                parms=parms,
                ph_uw_sb=ph_uw_sb,
                unwrap_ix=unwrap_ix,
                progress=progress,
            )
        )

        inversion_meta = (
            first_pass_inversion_meta
        )

        final_selected_ix = (
            unwrap_ix.copy()
        )

        final_qc_meta = {
            "enabled":
                bool(
                    final_qc_owns_drop
                ),

            "processed_ifg":
                int(
                    unwrap_ix.size
                ),

            "selected_ifg":
                int(
                    unwrap_ix.size
                ),

            "drop_count":
                0,
        }

        if final_qc_owns_drop:

            from pystamps.final_ifg_qc import (
                run_final_ifg_qc,
                settings_from_parms
                as final_qc_settings_from_parms,
            )

            final_qc_settings = (
                final_qc_settings_from_parms(
                    parms
                )
            )

            final_qc_result = (
                run_final_ifg_qc(
                    root,
                    ph_uw_sb=ph_uw_sb,
                    msd=msd,
                    ifgday_ix=ifgday_ix_all,
                    settings=final_qc_settings,
                    progress=progress,
                )
            )

            final_selected_ix = np.asarray(
                final_qc_result[
                    "keep_ix"
                ],
                dtype=np.int64,
            )

            final_drop = [
                int(v)
                for v
                in final_qc_result[
                    "effective_drop_ifg_index"
                ]
            ]

            if (
                final_selected_ix.size
                == 0
            ):
                raise Stage6SbasError(
                    "FINAL IFG-QC removed "
                    "all interferograms"
                )

            parms[
                "drop_ifg_index"
            ] = np.asarray(
                final_drop,
                dtype=np.float64,
            ).reshape(
                1,
                -1,
            )

            write_mat(
                parms_path,
                parms,
            )

            final_qc_meta = {
                key: value
                for key, value
                in final_qc_result.items()
                if key != "keep_ix"
            }

            final_qc_meta[
                "enabled"
            ] = True

            final_qc_meta[
                "processed_ifg"
            ] = int(
                unwrap_ix.size
            )

            final_qc_meta[
                "selected_ifg"
            ] = int(
                final_selected_ix.size
            )

            if final_drop:

                drop_zero = np.asarray(
                    [
                        idx - 1
                        for idx
                        in final_drop
                    ],
                    dtype=np.int64,
                )

                # Restore traditional phuw_sb2 semantics:
                # final dropped columns remain present in the
                # N_PS x N_IFG matrix but are zeroed.
                ph_uw_sb[
                    :,
                    drop_zero,
                ] = 0.0

                msd[
                    drop_zero
                ] = 0.0

                write_mat(
                    root
                    / "phuw_sb2.mat",
                    {
                        "ph_uw":
                            ph_uw_sb,

                        "msd":
                            msd.reshape(
                                -1,
                                1,
                            ),
                    },
                )

                print(
                    "[IFG_FINAL_QC] "
                    "re-running SB GLS inversion only: "
                    f"{final_selected_ix.size}/{n_ifg} "
                    "IFGs retained",
                    flush=True,
                )

                inversion_meta = (
                    _stage6_sb_invert(
                        root,
                        ps2=ps2,
                        parms=parms,
                        ph_uw_sb=ph_uw_sb,
                        unwrap_ix=final_selected_ix,
                        progress=progress,
                    )
                )

            else:

                print(
                    "[IFG_FINAL_QC] "
                    "no final IFGs rejected; "
                    "first-pass SB inversion retained",
                    flush=True,
                )

        debug = {
            "status": "completed",
            "dataset_root": str(root),
            "duration_sec": time.perf_counter() - started,
            "network_source": str(network_source),
            "n_ps": int(n_ps),
            "n_ifg": int(n_ifg),
            "processed_ifg": int(unwrap_ix.size),
            "selected_ifg": int(final_selected_ix.size),
            "n_image": int(day.size),
            "grid_shape": [int(n_i), int(n_j)],
            "grid_ps": int(n_grid_ps),
            "snaphu_workers": int(snaphu_workers),
            "snaphu_autotune": snaphu_worker_meta,
            "edge_chunk": int(edge_chunk),
            "anneal_workers": int(anneal_workers),
            "anneal_runs": int(anneal_runs),
            "strict_anneal": bool(strict_anneal),
            "network": network_meta,
            "sb_inversion_first_pass":
                first_pass_inversion_meta,
            "sb_inversion": inversion_meta,
            "final_ifg_qc": final_qc_meta,
            "backup": str(backup) if backup is not None else None,
        }
        _write_json(debug_path, debug)

        # === STAGE6_PERSIST_GRID_CACHE_V1 ===
        #
        # GRID filtering is one of the expensive reusable Stage-6 products.
        # Keep grid_v2 whenever GRID resume is enabled, even when the normal
        # temporary-work policy is cleanup. Remove SNAPHU/LA/etc. temporary
        # products, but preserve:
        #
        #   _stage6_sbas_work/grid_v2/meta.json
        #   _stage6_sbas_work/grid_v2/grid_phase.npy
        #   _stage6_sbas_work/grid_v2/grid_lowpass.npy
        #   _stage6_sbas_work/grid_v2/done.npy
        #
        # PYSTAMPS_SBAS_KEEP_WORK=1 still preserves the entire work tree.
        if not keep_work:
            if grid_resume:
                grid_cache_dir = work_dir / "grid_v2"

                for child in list(work_dir.iterdir()):
                    if child == grid_cache_dir:
                        continue

                    if child.is_dir():
                        shutil.rmtree(
                            child,
                            ignore_errors=True,
                        )
                    else:
                        try:
                            child.unlink()
                        except FileNotFoundError:
                            pass

                print(
                    "[STAGE6_SBAS] "
                    "temporary work cleaned; "
                    "GRID resume cache preserved",
                    flush=True,
                )
            else:
                shutil.rmtree(
                    work_dir,
                    ignore_errors=True,
                )

        return (
            f"Stage 6 SBAS unwrapped {n_ps} PS across {n_ifg} interferograms "
            f"({unwrap_ix.size} processed, "
            f"{final_selected_ix.size} retained after FINAL IFG-QC)"
        )

    except Exception as exc:
        _write_json(
            debug_path,
            {
                "status": "failed",
                "dataset_root": str(root),
                "duration_sec": time.perf_counter() - started,
                "exception": f"{type(exc).__name__}: {exc}",
                "backup": str(backup) if backup is not None else None,
            },
        )
        raise


def _preflight(dataset_root: Path) -> None:
    root = dataset_root.expanduser().resolve()
    ps2 = read_mat(root / "ps2.mat")
    n_ps = int(round(_scalar(ps2.get("n_ps"), 0)))
    ph_shape = np.asarray(
        read_mat_variables(root / "ph2.mat", ("ph",)).get("ph")
    ).shape
    if len(ph_shape) != 2:
        raise Stage6SbasError(f"ph2.ph has invalid shape {ph_shape}")
    if ph_shape[0] != n_ps and ph_shape[1] == n_ps:
        ph_shape = (ph_shape[1], ph_shape[0])
    n_ifg = int(ph_shape[1])
    day, ifgday_ix, bperp, source = load_sbas_network(root, n_ifg)
    G, day_active, _ = _active_network(day, ifgday_ix)
    print("Stage 6 SBAS preflight")
    print(f"dataset       : {root}")
    print(f"network source: {source}")
    print(f"n_ps          : {n_ps}")
    print(f"n_ifg         : {n_ifg}")
    print(f"n_image       : {day.size}")
    print(f"active images : {day_active.size}")
    print(f"rank(G)       : {np.linalg.matrix_rank(G)}")
    print(f"expected rank : {day_active.size - 1}")
    print(f"bperp range   : {np.ptp(bperp):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StaMPS-compatible multiple-master SBAS Stage 6."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--triangle", default=None)
    parser.add_argument("--snaphu", default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        _preflight(args.dataset)
        return 0

    result = stage6_sbas_unwrap(
        args.dataset,
        backend="auto",
        io_workers=args.io_workers,
        enable_mat_cache=False,
        triangle_path=args.triangle,
        snaphu_path=args.snaphu,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
