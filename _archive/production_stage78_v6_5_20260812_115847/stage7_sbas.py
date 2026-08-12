from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat, read_mat_variables, write_mat
from pystamps.pipeline.stage6_sbas import load_sbas_network


class Stage7SbasError(RuntimeError):
    """Raised when the SBAS-aware Stage 7 cannot continue."""


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
        raise Stage7SbasError(f"{name} must be 2-D, got {arr.shape}")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise Stage7SbasError(f"{name} shape {arr.shape}; expected first dimension {rows}")
    return np.asarray(arr, dtype=dtype)


def _drop_set(parms: dict[str, Any], *names: str) -> set[int]:
    from pystamps.pipeline import ported

    result: set[int] = set()
    for name in names:
        raw = parms.get(name)
        if raw is None:
            continue
        result.update(int(v) for v in ported._normalize_drop_index(raw).tolist())
    return result


def _network_matrix(n_image: int, ifgday_ix: np.ndarray) -> np.ndarray:
    G = np.zeros((ifgday_ix.shape[0], n_image), dtype=np.float64)
    rows = np.arange(ifgday_ix.shape[0], dtype=np.int64)
    G[rows, ifgday_ix[:, 0] - 1] = -1.0
    G[rows, ifgday_ix[:, 1] - 1] = 1.0
    return G


def _weighted_transform(design: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    X = np.asarray(design, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < X.shape[1]:
        raise Stage7SbasError(f"Invalid regression design shape {X.shape}")
    if weights is None:
        return np.linalg.pinv(X)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != X.shape[0]:
        raise Stage7SbasError("Regression weights do not match design rows")
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sqrt_w[:, None]
    return np.linalg.pinv(Xw) * sqrt_w[None, :]


def _fit_shared_design(
    values: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None,
    *,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one shared design to many PS rows, with a NaN-safe fallback."""

    Y = np.asarray(values, dtype=np.float64)
    X = np.asarray(design, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != X.shape[0]:
        raise Stage7SbasError(
            f"Regression values/design mismatch: values={Y.shape}, design={X.shape}"
        )
    base_w = (
        np.ones(X.shape[0], dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    finite_design = np.all(np.isfinite(X), axis=1) & np.isfinite(base_w) & (base_w > 0)
    complete = np.all(np.isfinite(Y[:, finite_design]), axis=1)
    coeff = np.full((X.shape[1], Y.shape[0]), np.nan, dtype=np.float64)
    valid = np.zeros(Y.shape[0], dtype=bool)

    if np.any(complete):
        transform = _weighted_transform(X[finite_design, :], base_w[finite_design])
        coeff[:, complete] = transform @ Y[complete, :][:, finite_design].T
        valid[complete] = True

    for row in np.flatnonzero(~complete):
        row_valid = finite_design & np.isfinite(Y[row, :])
        if np.count_nonzero(row_valid) < max(min_obs, X.shape[1]):
            continue
        Xv = X[row_valid, :]
        if np.linalg.matrix_rank(Xv) < X.shape[1]:
            continue
        transform = _weighted_transform(Xv, base_w[row_valid])
        coeff[:, row] = transform @ Y[row, row_valid]
        valid[row] = True

    return coeff, valid


def _network_projector(
    G: np.ndarray,
    use_ix: np.ndarray,
    weights: np.ndarray,
    reference_image: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_image = G.shape[1]
    unknown = np.asarray([i for i in range(n_image) if i != reference_image], dtype=np.int64)
    A = G[use_ix, :][:, unknown]
    rank = int(np.linalg.matrix_rank(A))
    if rank != n_image - 1:
        raise Stage7SbasError(
            f"SBAS network is rank deficient: rank={rank}, expected={n_image - 1}"
        )
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    normal = A.T @ (w[:, None] * A)
    ridge = max(float(np.mean(np.diag(normal))) * 1.0e-12, 1.0e-14)
    normal.flat[:: normal.shape[0] + 1] += ridge
    projector = np.linalg.solve(normal, A.T * w[None, :])
    return projector, unknown, rank


def _invert_network(
    values: np.ndarray,
    *,
    G: np.ndarray,
    use_ix: np.ndarray,
    weights: np.ndarray,
    projector: np.ndarray,
    unknown: np.ndarray,
    reference_image: int,
    output: np.ndarray,
    chunk_ps: int,
    label: str,
) -> int:
    """Invert IFG values to acquisition values with one fixed reference image."""

    n_ps = values.shape[0]
    solved_total = 0
    A = G[use_ix, :][:, unknown]
    sqrt_w = np.sqrt(weights)

    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        y = np.asarray(values[start:stop, :][:, use_ix], dtype=np.float64)
        current = stop - start
        out = np.full((current, G.shape[1]), np.nan, dtype=np.float64)
        complete = np.all(np.isfinite(y), axis=1)
        if np.any(complete):
            out[np.flatnonzero(complete)[:, None], unknown[None, :]] = (
                y[complete, :] @ projector.T
            )
            out[complete, reference_image] = 0.0

        for local in np.flatnonzero(~complete):
            valid = np.isfinite(y[local, :])
            if np.count_nonzero(valid) < G.shape[1] - 1:
                continue
            Av = A[valid, :]
            if np.linalg.matrix_rank(Av) != G.shape[1] - 1:
                continue
            solution, *_ = np.linalg.lstsq(
                Av * sqrt_w[valid, None],
                y[local, valid] * sqrt_w[valid],
                rcond=None,
            )
            out[local, unknown] = solution
            out[local, reference_image] = 0.0

        output[start:stop, :] = out.astype(np.float32)
        if hasattr(output, "flush"):
            output.flush()
        solved_total += int(np.count_nonzero(np.all(np.isfinite(out), axis=1)))
        print(
            f"[STAGE7_SBAS][{label}] {stop}/{n_ps} "
            f"({100.0 * stop / n_ps:.1f}%) solved={solved_total}",
            flush=True,
        )
    return solved_total


def _make_phase_model(
    output: np.ndarray,
    coefficient: np.ndarray,
    baseline: np.ndarray,
    chunk_ps: int,
    label: str,
) -> None:
    n_ps = baseline.shape[0]
    for start in range(0, n_ps, chunk_ps):
        stop = min(start + chunk_ps, n_ps)
        output[start:stop, :] = (
            np.asarray(coefficient[start:stop], dtype=np.float64)[:, None]
            * np.asarray(baseline[start:stop, :], dtype=np.float64)
        ).astype(np.float32)
        if hasattr(output, "flush"):
            output.flush()
        print(
            f"[STAGE7_SBAS][{label}] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )



# === GACOS_STAGE7_INPUT_V1 ===
def _stage7_phase_input(root: Path) -> Path:
    enabled = os.environ.get(
        "PYSTAMPS_GACOS_STAGE7_ENABLE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return root / "phuw2.mat"

    from pystamps.pipeline.gacos_correction import ensure_gacos_corrected_phuw

    return ensure_gacos_corrected_phuw(root)

def stage7_sbas_calc_scla(
    dataset_root: Path,
    backend: str = "auto",
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    triangle_path: str | None = None,
) -> str:
    """
    SBAS-aware Stage 7.

    It first uses the real SB IFG geometry, then converts IFG phase and
    perpendicular baseline to a referenced acquisition series through the
    incidence matrix. Final SCLA is estimated from sequential acquisition
    differences, following the StaMPS small-baseline Stage-7 structure.
    """

    del backend, enable_mat_cache, io_workers, mat_cache
    from pystamps.pipeline import ported

    root = Path(dataset_root).expanduser().resolve()
    started = time.perf_counter()
    required = ("ps2.mat", "phuw2.mat", "bp2.mat", "ifgstd2.mat")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise Stage7SbasError(f"Missing Stage 7 inputs: {', '.join(missing)}")

    ps2 = read_mat(root / "ps2.mat")
    n_ps = int(round(_scalar(ps2.get("n_ps"), 0)))
    if n_ps <= 0:
        raise Stage7SbasError("ps2.mat missing valid n_ps")

    ph_ifg_raw = _as_matrix(
        read_mat_variables(_stage7_phase_input(root), ("ph_uw",))["ph_uw"],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )
    n_ps, n_ifg = ph_ifg_raw.shape
    bp_ifg = _as_matrix(
        read_mat_variables(root / "bp2.mat", ("bperp_mat",))["bperp_mat"],
        n_ps,
        "bp2.bperp_mat",
        np.float32,
    )
    if bp_ifg.shape[1] != n_ifg:
        raise Stage7SbasError(
            f"bp2.bperp_mat has {bp_ifg.shape[1]} columns; expected {n_ifg}"
        )

    day, ifgday_ix, _bperp, network_source = load_sbas_network(root, n_ifg)
    ifgday_ix = np.asarray(ifgday_ix, dtype=np.int64)
    n_image = int(day.size)
    G = _network_matrix(n_image, ifgday_ix)
    dt_ifg = day[ifgday_ix[:, 1] - 1] - day[ifgday_ix[:, 0] - 1]

    ifg_std = np.asarray(
        read_mat_variables(root / "ifgstd2.mat", ("ifg_std",))["ifg_std"],
        dtype=np.float64,
    ).reshape(-1)
    if ifg_std.size != n_ifg:
        raise Stage7SbasError(f"ifgstd2.ifg_std has {ifg_std.size}; expected {n_ifg}")

    parms = read_mat(root / "parms.mat") if (root / "parms.mat").exists() else {}
    if ported._mat_text(parms.get("small_baseline_flag", "n"), "n").lower() != "y":
        raise Stage7SbasError("Stage 7 SBAS requires small_baseline_flag='y'")

    chunk = int(chunk_ps) if int(chunk_ps) > 0 else int(
        os.environ.get("PYSTAMPS_SBAS_STAGE7_CHUNK_PS", "2048")
    )
    chunk = max(128, chunk)

    master_ix = int(round(_scalar(ps2.get("master_ix"), 1)))
    if master_ix < 1 or master_ix > n_image:
        master_ix = 1
    reference_image = master_ix - 1

    drop_network = _drop_set(parms, "drop_ifg_index")
    finite_std = np.isfinite(ifg_std) & (ifg_std > 0)
    network_mask = np.asarray(
        [i not in drop_network for i in range(1, n_ifg + 1)], dtype=bool
    ) & finite_std
    use_ix = np.flatnonzero(network_mask)
    if use_ix.size < n_image - 1:
        raise Stage7SbasError(
            f"Only {use_ix.size} usable IFGs remain for {n_image} acquisitions"
        )

    variance_ifg = (ifg_std * math.pi / 180.0) ** 2
    weights_ifg = np.zeros(n_ifg, dtype=np.float64)
    weights_ifg[network_mask] = 1.0 / variance_ifg[network_mask]
    median_weight = float(np.median(weights_ifg[network_mask]))
    if median_weight > 0:
        weights_ifg /= median_weight

    projector, unknown, rank = _network_projector(
        G,
        use_ix,
        weights_ifg[use_ix],
        reference_image,
    )

    # Spatial ramp and reference centering are applied in IFG space first.
    ph_ifg_float = np.asarray(ph_ifg_raw, dtype=np.float64)
    if ported._mat_text(parms.get("scla_deramp", "y"), "y").lower() == "y":
        ph_ifg_deramped, ph_ramp_ifg = ported._deramp_unwrapped_phase(ps2, ph_ifg_float)
    else:
        ph_ifg_deramped = ph_ifg_float
        ph_ramp_ifg = np.zeros_like(ph_ifg_float, dtype=np.float64)

    ref_ps = np.asarray(ported._select_reference_ps(ps2, parms), dtype=np.int64).reshape(-1)
    if ref_ps.size:
        raw_ref = np.nanmedian(ph_ifg_float[ref_ps, :], axis=0)
        proc_ref = np.nanmedian(ph_ifg_deramped[ref_ps, :], axis=0)
        raw_ref[~np.isfinite(raw_ref)] = 0.0
        proc_ref[~np.isfinite(proc_ref)] = 0.0
    else:
        raw_ref = np.zeros(n_ifg, dtype=np.float64)
        proc_ref = np.zeros(n_ifg, dtype=np.float64)
    ph_ifg_centered = ph_ifg_float - raw_ref[None, :]
    ph_ifg_proc = ph_ifg_deramped - proc_ref[None, :]

    # Direct SB diagnostic SCLA, equivalent to ps_calc_scla(1,1).
    sb_drop = drop_network | _drop_set(parms, "sb_scla_drop_index", "scla_drop_index")
    sb_mask = np.asarray([i not in sb_drop for i in range(1, n_ifg + 1)], dtype=bool) & finite_std
    sb_ix = np.flatnonzero(sb_mask)
    mean_bp_ifg = np.nanmean(np.asarray(bp_ifg, dtype=np.float64), axis=0)
    X_sb = np.column_stack(
        (
            np.ones(sb_ix.size, dtype=np.float64),
            mean_bp_ifg[sb_ix],
            dt_ifg[sb_ix],
        )
    )
    coeff_sb, valid_sb = _fit_shared_design(
        ph_ifg_proc[:, sb_ix],
        X_sb,
        weights_ifg[sb_ix],
        min_obs=4,
    )
    k_sb = coeff_sb[1, :]
    v_sb = coeff_sb[2, :]
    k_sb[~valid_sb] = np.nan
    v_sb[~valid_sb] = np.nan
    write_mat(
        root / "scla_sb2.mat",
        {
            "K_ps_uw": ported._matlab_col(k_sb.astype(np.float32), np.float32),
            "C_ps_uw": ported._matlab_col(np.zeros(n_ps, dtype=np.float32), np.float32),
            "mean_v": ported._matlab_col(v_sb.astype(np.float32), np.float32),
            "ifgday_ix": ifgday_ix.astype(np.int32),
            "ifg_dt_days": ported._matlab_col(dt_ifg.astype(np.float64), np.float64),
            "used_ifg_index": ported._matlab_col((sb_ix + 1).astype(np.int32), np.int32),
        },
    )

    work = root / "_stage7_sbas_work"
    work.mkdir(parents=True, exist_ok=True)
    ph_sm_raw = np.memmap(
        work / "ph_sm_raw.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    ph_sm_proc = np.memmap(
        work / "ph_sm_proc.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    bp_sm = np.memmap(
        work / "bp_sm.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )

    solved_raw = _invert_network(
        ph_ifg_centered,
        G=G,
        use_ix=use_ix,
        weights=weights_ifg[use_ix],
        projector=projector,
        unknown=unknown,
        reference_image=reference_image,
        output=ph_sm_raw,
        chunk_ps=chunk,
        label="PH_RAW_TO_SM",
    )
    solved_proc = _invert_network(
        ph_ifg_proc,
        G=G,
        use_ix=use_ix,
        weights=weights_ifg[use_ix],
        projector=projector,
        unknown=unknown,
        reference_image=reference_image,
        output=ph_sm_proc,
        chunk_ps=chunk,
        label="PH_PROC_TO_SM",
    )
    solved_bp = _invert_network(
        np.asarray(bp_ifg, dtype=np.float64),
        G=G,
        use_ix=use_ix,
        weights=weights_ifg[use_ix],
        projector=projector,
        unknown=unknown,
        reference_image=reference_image,
        output=bp_sm,
        chunk_ps=chunk,
        label="BPERP_TO_SM",
    )

    ph_ramp_sm = np.memmap(
        work / "ph_ramp_sm.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        ph_ramp_sm[start:stop, :] = (
            np.asarray(ph_sm_raw[start:stop, :], dtype=np.float64)
            - np.asarray(ph_sm_proc[start:stop, :], dtype=np.float64)
        ).astype(np.float32)
    ph_ramp_sm.flush()

    # Final single-master SCLA, equivalent in structure to ps_calc_scla(0,1).
    order = np.argsort(day, kind="stable")
    day_sorted = day[order]
    dt_seq = np.diff(day_sorted)
    bp_seq_mean = np.zeros(n_image - 1, dtype=np.float64)
    valid_bp_count = np.zeros(n_image - 1, dtype=np.int64)
    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        b = np.asarray(bp_sm[start:stop, :], dtype=np.float64)[:, order]
        db = np.diff(b, axis=1)
        finite = np.isfinite(db)
        bp_seq_mean += np.sum(np.where(finite, db, 0.0), axis=0)
        valid_bp_count += np.sum(finite, axis=0)
    bp_seq_mean = np.divide(
        bp_seq_mean,
        np.maximum(valid_bp_count, 1),
        out=np.zeros_like(bp_seq_mean),
        where=valid_bp_count > 0,
    )
    X_seq = np.column_stack(
        (
            np.ones(n_image - 1, dtype=np.float64),
            bp_seq_mean,
            dt_seq,
        )
    )

    k_final = np.full(n_ps, np.nan, dtype=np.float64)
    v_seq = np.full(n_ps, np.nan, dtype=np.float64)
    valid_final = np.zeros(n_ps, dtype=bool)
    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        phase_seq = np.diff(
            np.asarray(ph_sm_proc[start:stop, :], dtype=np.float64)[:, order],
            axis=1,
        )
        coeff, valid = _fit_shared_design(
            phase_seq,
            X_seq,
            None,
            min_obs=4,
        )
        k_final[start:stop] = coeff[1, :]
        v_seq[start:stop] = coeff[2, :]
        valid_final[start:stop] = valid
        print(
            f"[STAGE7_SBAS][FINAL_SCLA] {stop}/{n_ps} "
            f"({100.0 * stop / n_ps:.1f}%)",
            flush=True,
        )

    ph_scla = np.memmap(
        work / "ph_scla_sm.f32", mode="w+", dtype=np.float32, shape=(n_ps, n_image)
    )
    _make_phase_model(ph_scla, k_final, bp_sm, chunk, "PH_SCLA")

    # Estimate C and mean velocity after SCLA subtraction in acquisition space.
    time_rel = np.asarray(day, dtype=np.float64) - float(day[reference_image])
    X_c = np.column_stack((np.ones(n_image, dtype=np.float64), time_rel))
    c_final = np.full(n_ps, np.nan, dtype=np.float64)
    mean_v = np.full(n_ps, np.nan, dtype=np.float64)
    for start in range(0, n_ps, chunk):
        stop = min(start + chunk, n_ps)
        residual = (
            np.asarray(ph_sm_proc[start:stop, :], dtype=np.float64)
            - np.asarray(ph_scla[start:stop, :], dtype=np.float64)
        )
        coeff, valid = _fit_shared_design(residual, X_c, None, min_obs=3)
        c_final[start:stop] = coeff[0, :]
        mean_v[start:stop] = coeff[1, :]
        valid_final[start:stop] &= valid

    # Acquisition covariance propagated from IFG covariance.
    cov_unknown = projector @ np.diag(variance_ifg[use_ix]) @ projector.T
    sm_cov = np.zeros((n_image, n_image), dtype=np.float64)
    sm_cov[np.ix_(unknown, unknown)] = cov_unknown

    scla_payload = {
        "K_ps_uw": ported._matlab_col(k_final.astype(np.float32), np.float32),
        "C_ps_uw": ported._matlab_col(c_final.astype(np.float32), np.float32),
        "mean_v": ported._matlab_col(mean_v.astype(np.float32), np.float32),
        "ph_scla": ph_scla,
        "ph_ramp": ph_ramp_sm,
        "ifg_vcm": sm_cov,
        "scla_valid": ported._matlab_col(valid_final.astype(np.uint8), np.uint8),
        "day": ported._matlab_col(day.astype(np.float64), np.float64),
        "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
    }
    write_mat(root / "scla2.mat", scla_payload)

    smooth_edges = ported._resolve_scla_smooth_edges(
        root, ps2, n_ps, triangle_path=triangle_path
    )
    k_smooth, c_smooth = ported._smooth_scla_neighbor_envelope(
        k_final, c_final, smooth_edges
    )
    ph_scla_smooth = np.memmap(
        work / "ph_scla_smooth_sm.f32",
        mode="w+",
        dtype=np.float32,
        shape=(n_ps, n_image),
    )
    _make_phase_model(ph_scla_smooth, k_smooth, bp_sm, chunk, "PH_SCLA_SMOOTH")
    write_mat(
        root / "scla_smooth2.mat",
        {
            "K_ps_uw": ported._matlab_col(np.asarray(k_smooth, dtype=np.float32), np.float32),
            "C_ps_uw": ported._matlab_col(np.asarray(c_smooth, dtype=np.float32), np.float32),
            "mean_v": ported._matlab_col(mean_v.astype(np.float32), np.float32),
            "ph_scla": ph_scla_smooth,
            "ph_ramp": ph_ramp_sm,
            "day": ported._matlab_col(day.astype(np.float64), np.float64),
            "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
        },
    )

    write_mat(
        root / "phuw_sm2.mat",
        {
            "ph_uw": ph_sm_raw,
            "ph_uw_deramped": ph_sm_proc,
            "day": ported._matlab_col(day.astype(np.float64), np.float64),
            "ifgday_ix": ifgday_ix.astype(np.int32),
            "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
        },
    )
    write_mat(
        root / "bp_sm2.mat",
        {
            "bperp_mat": bp_sm,
            "day": ported._matlab_col(day.astype(np.float64), np.float64),
            "reference_image_ix": np.asarray(master_ix, dtype=np.int32),
        },
    )

    _write_json(
        root / "stage7_sbas_debug.json",
        {
            "status": "completed",
            "method": "StaMPS-structured SB regression plus weighted IFG-to-acquisition inversion",
            "dataset_root": str(root),
            "network_source": str(network_source),
            "n_ps": int(n_ps),
            "n_ifg": int(n_ifg),
            "n_image": int(n_image),
            "network_rank": int(rank),
            "used_network_ifg": int(use_ix.size),
            "used_sb_scla_ifg": int(sb_ix.size),
            "reference_image_ix_1based": int(master_ix),
            "reference_ps": int(ref_ps.size),
            "solved_phase_raw_ps": int(solved_raw),
            "solved_phase_processed_ps": int(solved_proc),
            "solved_baseline_ps": int(solved_bp),
            "valid_final_scla_ps": int(np.count_nonzero(valid_final)),
            "chunk_ps": int(chunk),
            "duration_sec": time.perf_counter() - started,
            "note": "Custom SBAS extension; not yet MATLAB parity-certified.",
        },
    )
    return (
        f"Stage 7 SBAS estimated SCLA for {n_ps} PS and reconstructed "
        f"{n_image} acquisition epochs from {n_ifg} IFGs"
    )
