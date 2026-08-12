#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
DATASET="${PYSTAMPS_DATASET:-/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized}"
ENV_DIR="${PYSTAMPS_ENV_DIR:-/home/ubuntu/software/miniconda3/envs/stamps}"
PYTHON="$ENV_DIR/bin/python"
TARGET="$ROOT/pystamps/pipeline/ported.py"
MODE="${1:-install-run}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$DATASET/_run_logs"
SOCKET="stage78sbas"
SESSION="cangzhou_stage78_sbas"

usage() {
  cat <<'EOF'
Usage:
  ./install_stage78_sbas_patch.sh install
  ./install_stage78_sbas_patch.sh run
  ./install_stage78_sbas_patch.sh install-run
  ./install_stage78_sbas_patch.sh foreground

Environment overrides:
  PYSTAMPS_ROOT
  PYSTAMPS_DATASET
  PYSTAMPS_ENV_DIR
EOF
}

[[ "$MODE" =~ ^(install|run|install-run|foreground)$ ]] || { usage; exit 2; }

check_base() {
  [[ -d "$ROOT" ]] || { echo "错误：找不到项目目录 $ROOT" >&2; exit 3; }
  [[ -d "$DATASET" ]] || { echo "错误：找不到数据集 $DATASET" >&2; exit 4; }
  [[ -x "$PYTHON" ]] || { echo "错误：找不到 Python $PYTHON" >&2; exit 5; }
  [[ -f "$TARGET" ]] || { echo "错误：找不到 $TARGET" >&2; exit 6; }
  mkdir -p "$LOG_DIR"
}

install_patch() {
  echo "============================================================"
  echo "安装 SBAS Stage 7/8 补丁"
  echo "============================================================"

  local backup="${TARGET}.bak_stage78_sbas_${STAMP}"
  cp -a "$TARGET" "$backup"
  echo "ported.py 备份：$backup"

  cat > "$ROOT/pystamps/pipeline/stage7_sbas.py" <<'PY_STAGE7'
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
        read_mat_variables(root / "phuw2.mat", ("ph_uw",))["ph_uw"],
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
        raw_ref = np.nanmean(ph_ifg_float[ref_ps, :], axis=0)
        proc_ref = np.nanmean(ph_ifg_deramped[ref_ps, :], axis=0)
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
PY_STAGE7

  cat > "$ROOT/pystamps/pipeline/stage8_sbas.py" <<'PY_STAGE8'
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
        reference_phase = np.nanmean(ref_values, axis=0)
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
PY_STAGE8

  TARGET="$TARGET" "$PYTHON" - <<'PY_PATCH'
from __future__ import annotations

import ast
import os
from pathlib import Path

path = Path(os.environ["TARGET"])
source = path.read_text(encoding="utf-8")
marker = "# === STAGE78_SBAS_DISPATCH_V1 ==="

wrapper = r'''

# === STAGE78_SBAS_DISPATCH_V1 ===
_stage7_calc_scla_non_sbas = stage7_calc_scla
_stage8_filter_scn_non_sbas = stage8_filter_scn


def _stage78_dataset_is_sbas(dataset_root: Path) -> bool:
    parms_path = Path(dataset_root) / "parms.mat"
    if not parms_path.exists():
        return False
    try:
        payload = read_mat_variables(parms_path, ("small_baseline_flag",))
        return _mat_text(payload.get("small_baseline_flag", "n"), "n").lower() == "y"
    except Exception:
        return False


def stage7_calc_scla(
    dataset_root: Path,
    backend: str = "auto",
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    triangle_path: str | None = None,
) -> str:
    if _stage78_dataset_is_sbas(dataset_root):
        from pystamps.pipeline.stage7_sbas import stage7_sbas_calc_scla

        return stage7_sbas_calc_scla(
            dataset_root,
            backend=backend,
            chunk_ps=chunk_ps,
            enable_mat_cache=enable_mat_cache,
            io_workers=io_workers,
            mat_cache=mat_cache,
            triangle_path=triangle_path,
        )
    return _stage7_calc_scla_non_sbas(
        dataset_root,
        backend=backend,
        chunk_ps=chunk_ps,
        enable_mat_cache=enable_mat_cache,
        io_workers=io_workers,
        mat_cache=mat_cache,
        triangle_path=triangle_path,
    )


def stage8_filter_scn(
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
    if _stage78_dataset_is_sbas(dataset_root):
        from pystamps.pipeline.stage8_sbas import stage8_sbas_filter_scn

        return stage8_sbas_filter_scn(
            dataset_root,
            backend=backend,
            chunk_edges=chunk_edges,
            chunk_ps=chunk_ps,
            enable_mat_cache=enable_mat_cache,
            io_workers=io_workers,
            mat_cache=mat_cache,
            triangle_path=triangle_path,
            snaphu_path=snaphu_path,
        )
    return _stage8_filter_scn_non_sbas(
        dataset_root,
        backend=backend,
        chunk_edges=chunk_edges,
        chunk_ps=chunk_ps,
        enable_mat_cache=enable_mat_cache,
        io_workers=io_workers,
        mat_cache=mat_cache,
        triangle_path=triangle_path,
        snaphu_path=snaphu_path,
    )
'''

if marker not in source:
    new_source = source.rstrip() + wrapper + "\n"
else:
    new_source = source

ast.parse(new_source, filename=str(path))
tmp = path.with_suffix(path.suffix + ".stage78_tmp")
tmp.write_text(new_source, encoding="utf-8")
os.replace(tmp, path)
print("Stage 7/8 SBAS dispatch installed.")
PY_PATCH

  "$PYTHON" -m py_compile \
    "$ROOT/pystamps/pipeline/ported.py" \
    "$ROOT/pystamps/pipeline/stage7_sbas.py" \
    "$ROOT/pystamps/pipeline/stage8_sbas.py"

  PYTHONPATH="$ROOT" "$PYTHON" - <<'PY_IMPORT'
from pystamps.pipeline.ported import stage7_calc_scla, stage8_filter_scn
print("stage7:", stage7_calc_scla.__module__, stage7_calc_scla.__name__)
print("stage8:", stage8_filter_scn.__module__, stage8_filter_scn.__name__)
print("Import check: PASSED")
PY_IMPORT

  echo "补丁安装完成。"
}

preflight() {
  echo "============================================================"
  echo "Stage 7/8 SBAS 预检查"
  echo "============================================================"

  DATASET="$DATASET" PYTHONPATH="$ROOT" "$PYTHON" - <<'PY_PREFLIGHT'
from pathlib import Path
import json
import os
import shutil
import numpy as np

from pystamps.io.mat import read_mat_variables

root = Path(os.environ["DATASET"])
required = ["ps2.mat", "phuw2.mat", "bp2.mat", "ifgstd2.mat", "parms.mat"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit("Missing files: " + ", ".join(missing))

ps = read_mat_variables(root / "ps2.mat", ("n_ps", "n_image", "n_ifg", "day", "ifgday_ix"))
ph = np.asarray(read_mat_variables(root / "phuw2.mat", ("ph_uw",))["ph_uw"])
bp = np.asarray(read_mat_variables(root / "bp2.mat", ("bperp_mat",))["bperp_mat"])
std = np.asarray(read_mat_variables(root / "ifgstd2.mat", ("ifg_std",))["ifg_std"]).reshape(-1)
day = np.asarray(ps["day"]).reshape(-1)
ifg = ps.get("ifgday_ix")
if ifg is None:
    ifg = read_mat_variables(root / "PATCH_1" / "ps1.mat", ("ifgday_ix",))["ifgday_ix"]
ifg = np.asarray(ifg)
if ifg.shape[0] == 2 and ifg.shape[1] != 2:
    ifg = ifg.T

n_ps = int(round(float(np.asarray(ps["n_ps"]).reshape(-1)[0])))
if ph.shape[0] != n_ps and ph.shape[1] == n_ps:
    ph = ph.T
if bp.shape[0] != n_ps and bp.shape[1] == n_ps:
    bp = bp.T

print("phuw2.ph_uw :", ph.shape)
print("ps2.day      :", day.shape)
print("ifgday_ix    :", ifg.shape)
print("bp2          :", bp.shape)
print("ifg_std      :", std.shape)

assert ph.shape == (n_ps, std.size)
assert bp.shape == ph.shape
assert ifg.shape == (std.size, 2)
assert np.all(ifg >= 1) and np.all(ifg <= day.size)

stage6_debug = root / "stage6_sbas_debug.json"
if stage6_debug.exists():
    status = json.loads(stage6_debug.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise SystemExit("stage6_sbas_debug.json is not completed")

free = shutil.disk_usage(root).free
print(f"free disk      : {free / 1024**3:.1f} GiB")
if free < 15 * 1024**3:
    raise SystemExit("At least 15 GiB free disk is recommended for Stage 7/8")
print("Preflight: PASSED")
PY_PREFLIGHT
}

backup_old_outputs() {
  local backup="$DATASET/_stage78_backup/$STAMP"
  mkdir -p "$backup"
  local names=(
    scla2.mat scla_smooth2.mat scla_sb2.mat
    phuw_sm2.mat bp_sm2.mat mean_v.mat uw_space_time.mat
    stage7_sbas_debug.json stage8_sbas_debug.json
  )
  local moved=0
  for name in "${names[@]}"; do
    if [[ -e "$DATASET/$name" ]]; then
      mv "$DATASET/$name" "$backup/"
      moved=1
    fi
  done
  for name in _stage7_sbas_work _stage8_sbas_work; do
    if [[ -e "$DATASET/$name" ]]; then
      mv "$DATASET/$name" "$backup/"
      moved=1
    fi
  done
  if (( moved )); then
    echo "旧 Stage 7/8 输出已备份：$backup"
  else
    rmdir "$backup" 2>/dev/null || true
  fi
}

create_runner() {
  local runner="$LOG_DIR/stage78_sbas_${STAMP}.sh"
  local log="$LOG_DIR/stage78_sbas_${STAMP}.log"

  cat > "$runner" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail
exec >> "$log" 2>&1

export PATH="$ENV_DIR/bin:/usr/bin:/usr/local/bin:/bin:\$PATH"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

export PYSTAMPS_SBAS_STAGE7_CHUNK_PS="${PYSTAMPS_SBAS_STAGE7_CHUNK_PS:-2048}"
export PYSTAMPS_SBAS_STAGE8_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_CHUNK_PS:-1024}"
export PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS="${PYSTAMPS_SBAS_STAGE8_SPATIAL_CHUNK_PS:-256}"
export PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS="${PYSTAMPS_SBAS_STAGE8_K_NEIGHBORS:-32}"
export PYSTAMPS_SBAS_STAGE8_SPATIAL_FILTER="${PYSTAMPS_SBAS_STAGE8_SPATIAL_FILTER:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-16}"
export MALLOC_ARENA_MAX=2

cd "$ROOT"

echo "============================================================"
echo "SBAS Stage 7/8 start: \$(date)"
echo "Dataset: $DATASET"
echo "Python : $PYTHON"
echo "============================================================"

"$PYTHON" - <<'PY_RUN'
from pathlib import Path
import json

from pystamps.pipeline.ported import stage7_calc_scla, stage8_filter_scn

root = Path("$DATASET")
print(stage7_calc_scla(root, backend="python", chunk_ps=0, enable_mat_cache=False, io_workers=1))
print(stage8_filter_scn(root, backend="python", chunk_ps=0, enable_mat_cache=False, io_workers=1))

for name in ("stage7_sbas_debug.json", "stage8_sbas_debug.json"):
    payload = json.loads((root / name).read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"{name} is not completed")

print("SBAS Stage 7/8 completed successfully.")
PY_RUN

echo "Finished: \$(date)"
EOF_RUNNER

  chmod +x "$runner"
  echo "$runner|$log"
}

run_foreground() {
  preflight
  backup_old_outputs
  local pair runner log
  pair="$(create_runner)"
  runner="${pair%%|*}"
  log="${pair#*|}"
  echo "日志：$log"
  bash "$runner"
  tail -n 80 "$log"
}

run_tmux() {
  preflight

  if pgrep -af '[p]ython.*stage7_sbas|[p]ython.*stage8_sbas' >/dev/null 2>&1; then
    echo "错误：检测到 Stage 7/8 相关 Python 进程，拒绝重复启动。" >&2
    pgrep -af '[p]ython.*stage7_sbas|[p]ython.*stage8_sbas' >&2 || true
    exit 7
  fi

  backup_old_outputs
  local pair runner log
  pair="$(create_runner)"
  runner="${pair%%|*}"
  log="${pair#*|}"

  tmux -L "$SOCKET" kill-server 2>/dev/null || true
  tmux -L "$SOCKET" new-session -d -s "$SESSION" "bash '$runner'"
  sleep 10

  if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    echo "============================================================"
    echo "Stage 7/8 已启动"
    echo "============================================================"
    echo "日志：$log"
    echo "进入会话：tmux -L $SOCKET attach -t $SESSION"
    echo "查看日志：tail -f '$log'"
  else
    echo "Stage 7/8 启动后立即退出：" >&2
    tail -n 200 "$log" >&2 || true
    exit 8
  fi
}

check_base
case "$MODE" in
  install)
    install_patch
    ;;
  run)
    run_tmux
    ;;
  install-run)
    install_patch
    run_tmux
    ;;
  foreground)
    install_patch
    run_foreground
    ;;
esac
