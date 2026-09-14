from __future__ import annotations

# POST_UNWRAP_DERAMP_V1
#
# Conservative production deramp:
# - fit a robust spatial plane at each acquisition;
# - retain only the time-linear evolution of the two spatial gradients;
# - preserve per-epoch intercepts;
# - anchor the spatial model at the existing reference-region centroid;
# - preserve the master epoch exactly.

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.config import PostUnwrapDerampConfig
from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline.stage6_sbas import _stage6_reference_indices


class PostUnwrapDerampError(RuntimeError):
    pass


_ALGORITHM_REVISION = "linear-spacetime-v1"


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _as_rows(value: Any, nrow: int, name: str, dtype=None) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 1:
        if nrow == 1:
            arr = arr.reshape(1, -1)
        elif arr.size % nrow == 0:
            arr = arr.reshape(nrow, -1)
    if arr.ndim != 2:
        raise PostUnwrapDerampError(
            f"{name}: expected 2-D matrix, got {arr.shape}"
        )
    if arr.shape[0] != nrow and arr.shape[1] == nrow:
        arr = arr.T
    if arr.shape[0] != nrow:
        raise PostUnwrapDerampError(
            f"{name}: shape={arr.shape}; expected first dimension {nrow}"
        )
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr


def _robust_plane(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    max_iter: int,
    clip: float,
) -> tuple[np.ndarray, float, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    keep = valid.copy()

    X = np.column_stack(
        (
            np.ones(x.size, dtype=np.float64),
            x,
            y,
        )
    )

    beta = np.full(3, np.nan, dtype=np.float64)

    for _ in range(max(1, int(max_iter))):
        if np.count_nonzero(keep) < 10:
            break

        beta, *_ = np.linalg.lstsq(
            X[keep],
            z[keep],
            rcond=None,
        )

        residual = z - X @ beta
        r = residual[keep]
        med = float(np.nanmedian(r))
        mad = float(np.nanmedian(np.abs(r - med)))
        sigma = 1.4826 * mad

        if not np.isfinite(sigma) or sigma <= 1.0e-12:
            break

        new_keep = valid & (
            np.abs(residual - med) <= float(clip) * sigma
        )

        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    n_keep = int(np.count_nonzero(keep))
    if n_keep < 10 or not np.all(np.isfinite(beta)):
        return beta, float("nan"), n_keep

    pred = X[keep] @ beta
    zz = z[keep]
    denom = float(np.sum((zz - np.mean(zz)) ** 2))
    r2 = (
        1.0 - float(np.sum((zz - pred) ** 2)) / denom
        if denom > 0.0
        else float("nan")
    )

    return beta, r2, n_keep


def _robust_through_origin(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int,
    clip: float,
) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    keep = valid.copy()
    slope = float("nan")

    for _ in range(max(1, int(max_iter))):
        xv = x[keep]
        yv = y[keep]
        denom = float(np.dot(xv, xv))
        if xv.size < 3 or denom <= 0.0:
            break

        slope = float(np.dot(xv, yv) / denom)
        residual = y - slope * x
        r = residual[keep]
        med = float(np.nanmedian(r))
        mad = float(np.nanmedian(np.abs(r - med)))
        sigma = 1.4826 * mad

        if not np.isfinite(sigma) or sigma <= 1.0e-12:
            break

        new_keep = valid & (
            np.abs(residual - med) <= float(clip) * sigma
        )
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    return slope, keep


def _spatial_representatives(
    x: np.ndarray,
    y: np.ndarray,
    max_count: int,
    min_count: int,
) -> np.ndarray:
    """Deterministic one-per-occupied-cell spatial sampling."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))

    if valid.size < int(min_count):
        raise PostUnwrapDerampError(
            f"Only {valid.size} finite PS are available; "
            f"minimum is {int(min_count)}"
        )

    target = min(int(max_count), int(valid.size))
    if valid.size <= target:
        return valid

    xv = x[valid]
    yv = y[valid]
    sx = max(float(np.ptp(xv)), 1.0)
    sy = max(float(np.ptp(yv)), 1.0)
    aspect = sx / sy

    nx = max(1, int(round(math.sqrt(target * aspect))))
    ny = max(1, int(math.ceil(target / nx)))

    xmin = float(np.min(xv))
    ymin = float(np.min(yv))
    dx = sx / nx
    dy = sy / ny

    cx = np.clip(
        np.floor((xv - xmin) / dx).astype(np.int64),
        0,
        nx - 1,
    )
    cy = np.clip(
        np.floor((yv - ymin) / dy).astype(np.int64),
        0,
        ny - 1,
    )
    cell = cy * nx + cx

    center_x = xmin + (cx + 0.5) * dx
    center_y = ymin + (cy + 0.5) * dy
    dist2 = (
        ((xv - center_x) / dx) ** 2
        + ((yv - center_y) / dy) ** 2
    )

    order = np.lexsort((dist2, cell))
    cell_sorted = cell[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = cell_sorted[1:] != cell_sorted[:-1]
    reps = valid[order[first]]

    if reps.size > target:
        take = np.linspace(
            0,
            reps.size - 1,
            target,
        ).round().astype(np.int64)
        reps = reps[take]

    if reps.size < int(min_count):
        missing = int(min_count) - reps.size
        used = np.zeros(x.size, dtype=bool)
        used[reps] = True
        pool = valid[~used[valid]]
        if pool.size:
            take = np.linspace(
                0,
                pool.size - 1,
                min(missing, pool.size),
            ).round().astype(np.int64)
            reps = np.unique(
                np.concatenate((reps, pool[take]))
            )

    if reps.size < int(min_count):
        raise PostUnwrapDerampError(
            f"Spatial sampling produced {reps.size} PS; "
            f"minimum is {int(min_count)}"
        )

    return np.sort(reps)


def _file_sig(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _output_name(input_phase_file: str) -> str:
    p = Path(input_phase_file)
    return f"{p.stem}_deramp{p.suffix or '.mat'}"


def _cache_signature(
    root: Path,
    input_phase_file: str,
    settings: PostUnwrapDerampConfig,
) -> dict[str, object]:
    return {
        "algorithm_revision": _ALGORITHM_REVISION,
        "input_phase": _file_sig(root / input_phase_file),
        "ps2": _file_sig(root / "ps2.mat"),
        "parms": _file_sig(root / "parms.mat"),
        "settings": {
            "enabled": bool(settings.enabled),
            "mode": str(settings.mode),
            "max_representatives": int(
                settings.max_representatives
            ),
            "min_representatives": int(
                settings.min_representatives
            ),
            "robust_clip_sigma": float(
                settings.robust_clip_sigma
            ),
            "robust_iterations": int(
                settings.robust_iterations
            ),
            "chunk_ps": int(settings.chunk_ps),
        },
    }


def ensure_post_unwrap_deramped_phase(
    dataset_root: Path,
    settings: PostUnwrapDerampConfig,
    *,
    input_phase_file: str = "phuw2.mat",
) -> Path:
    root = Path(dataset_root).expanduser().resolve()

    if not bool(settings.enabled):
        return root / input_phase_file

    mode = str(settings.mode).strip().lower()
    if mode != "linear_spacetime":
        raise PostUnwrapDerampError(
            "post_unwrap_deramp.mode must be linear_spacetime"
        )

    source_path = root / input_phase_file
    if not source_path.is_file():
        raise PostUnwrapDerampError(
            f"Input phase product does not exist: {source_path}"
        )

    for required in ("ps2.mat", "parms.mat"):
        if not (root / required).is_file():
            raise PostUnwrapDerampError(
                f"Missing deramp input: {root / required}"
            )

    output_path = root / _output_name(input_phase_file)
    diag_dir = (
        root
        / "outputs"
        / "diagnostics"
        / "post_unwrap_deramp"
    )
    diag_dir.mkdir(parents=True, exist_ok=True)

    marker = diag_dir / f"{output_path.stem}_signature.json"
    report_path = diag_dir / f"{output_path.stem}_model.json"
    csv_path = diag_dir / f"{output_path.stem}_epochs.csv"

    signature = _cache_signature(
        root,
        input_phase_file,
        settings,
    )

    if (
        output_path.is_file()
        and marker.is_file()
        and not bool(settings.rebuild)
    ):
        try:
            saved = json.loads(
                marker.read_text(encoding="utf-8")
            )
        except Exception:
            saved = None
        if saved == signature:
            print(
                f"[DERAMP] Reusing current product: "
                f"{output_path.name}",
                flush=True,
            )
            return output_path

    ps = read_mat(root / "ps2.mat")
    parms = read_mat(root / "parms.mat")

    n_ps = int(round(_scalar(ps.get("n_ps"), 0)))
    n_image = int(round(_scalar(ps.get("n_image"), 0)))
    master_ix = int(
        round(_scalar(ps.get("master_ix"), 1))
    )
    master0 = master_ix - 1

    if n_ps <= 0 or n_image <= 1:
        raise PostUnwrapDerampError(
            f"Invalid ps2 dimensions: "
            f"n_ps={n_ps}, n_image={n_image}"
        )
    if not (0 <= master0 < n_image):
        raise PostUnwrapDerampError(
            f"Invalid master_ix={master_ix}"
        )

    day = np.asarray(
        ps.get("day"),
        dtype=np.float64,
    ).reshape(-1)
    if day.size != n_image:
        raise PostUnwrapDerampError(
            f"ps2.day length={day.size}; expected {n_image}"
        )

    xy = _as_rows(
        ps.get("xy"),
        n_ps,
        "ps2.xy",
        np.float64,
    )
    if xy.shape[1] < 3:
        raise PostUnwrapDerampError(
            "ps2.xy must contain [id, x, y]"
        )

    axis_x = xy[:, 1]
    axis_y = xy[:, 2]

    refs = np.asarray(
        _stage6_reference_indices(
            ps,
            parms,
            n_ps,
        ),
        dtype=np.int64,
    ).reshape(-1)
    refs = refs[
        (refs >= 0)
        & (refs < n_ps)
    ]

    if refs.size:
        x0 = float(np.nanmean(axis_x[refs]))
        y0 = float(np.nanmean(axis_y[refs]))
    else:
        x0 = float(np.nanmedian(axis_x))
        y0 = float(np.nanmedian(axis_y))

    x_scaled = (axis_x - x0) / 1000.0
    y_scaled = (axis_y - y0) / 1000.0

    reps = _spatial_representatives(
        x_scaled,
        y_scaled,
        int(settings.max_representatives),
        int(settings.min_representatives),
    )

    payload = read_mat(source_path)
    if "ph_uw" not in payload:
        raise PostUnwrapDerampError(
            f"{source_path.name} does not contain ph_uw"
        )

    ph = _as_rows(
        payload["ph_uw"],
        n_ps,
        f"{source_path.name}.ph_uw",
        np.float32,
    )

    if ph.shape != (n_ps, n_image):
        raise PostUnwrapDerampError(
            f"{source_path.name}.ph_uw shape={ph.shape}; "
            f"expected ({n_ps}, {n_image})"
        )

    phase_reps = np.asarray(
        ph[reps, :],
        dtype=np.float64,
    )

    bx = np.full(
        n_image,
        np.nan,
        dtype=np.float64,
    )
    by = np.full(
        n_image,
        np.nan,
        dtype=np.float64,
    )
    intercept = np.full(
        n_image,
        np.nan,
        dtype=np.float64,
    )
    epoch_r2 = np.full(
        n_image,
        np.nan,
        dtype=np.float64,
    )
    epoch_n = np.zeros(
        n_image,
        dtype=np.int32,
    )

    print(
        f"[DERAMP] fitting {reps.size:,} spatial "
        f"representatives across {n_image} acquisitions",
        flush=True,
    )

    for j in range(n_image):
        beta, r2, nkeep = _robust_plane(
            x_scaled[reps],
            y_scaled[reps],
            phase_reps[:, j],
            max_iter=int(
                settings.robust_iterations
            ),
            clip=float(
                settings.robust_clip_sigma
            ),
        )
        if np.all(np.isfinite(beta)):
            intercept[j] = beta[0]
            bx[j] = beta[1]
            by[j] = beta[2]
        epoch_r2[j] = r2
        epoch_n[j] = nkeep

    dt_year = (
        day - day[master0]
    ) / 365.25

    kx, keep_x = _robust_through_origin(
        dt_year,
        bx,
        max_iter=int(
            settings.robust_iterations
        ),
        clip=float(
            settings.robust_clip_sigma
        ),
    )
    ky, keep_y = _robust_through_origin(
        dt_year,
        by,
        max_iter=int(
            settings.robust_iterations
        ),
        clip=float(
            settings.robust_clip_sigma
        ),
    )

    if not np.isfinite(kx) or not np.isfinite(ky):
        raise PostUnwrapDerampError(
            "Unable to estimate time-linear "
            "spatial-gradient rates"
        )

    bx_model = kx * dt_year
    by_model = ky * dt_year

    bx_model[master0] = 0.0
    by_model[master0] = 0.0

    chunk = max(
        1,
        int(settings.chunk_ps),
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

        correction = (
            x_scaled[
                start:stop,
                None,
            ]
            * bx_model[
                None,
                :,
            ]
            + y_scaled[
                start:stop,
                None,
            ]
            * by_model[
                None,
                :,
            ]
        )

        block = np.asarray(
            ph[
                start:stop,
                :,
            ],
            dtype=np.float64,
        )

        block -= correction

        ph[
            start:stop,
            :,
        ] = block.astype(
            np.float32
        )

        if (
            stop == n_ps
            or stop
            % max(
                chunk * 50,
                1,
            )
            == 0
        ):
            print(
                f"[DERAMP] apply "
                f"{stop:,}/{n_ps:,} "
                f"({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )

    payload["ph_uw"] = ph
    write_mat(
        output_path,
        payload,
    )

    wavelength = _scalar(
        parms.get("lambda"),
        float("nan"),
    )

    phase_to_los_mm = (
        -wavelength
        * 1000.0
        / (
            4.0
            * np.pi
        )
        if (
            np.isfinite(wavelength)
            and wavelength > 0.0
        )
        else float("nan")
    )

    report = {
        "status": "completed",
        "algorithm_revision": (
            _ALGORITHM_REVISION
        ),
        "method": "linear_spacetime",
        "input_phase": input_phase_file,
        "output_phase": output_path.name,
        "n_ps": int(n_ps),
        "n_image": int(n_image),
        "master_ix_1based": int(
            master_ix
        ),
        "representative_ps": int(
            reps.size
        ),
        "reference_ps": int(
            refs.size
        ),
        "reference_axis_x_centroid": (
            x0
        ),
        "reference_axis_y_centroid": (
            y0
        ),
        "axis_x_gradient_rate_rad_per_1000coord_per_year": (
            float(kx)
        ),
        "axis_y_gradient_rate_rad_per_1000coord_per_year": (
            float(ky)
        ),
        "phase_to_los_mm": float(
            phase_to_los_mm
        ),
        "epoch_r2_median": float(
            np.nanmedian(epoch_r2)
        ),
        "epoch_r2_p05": float(
            np.nanpercentile(
                epoch_r2,
                5,
            )
        ),
        "epoch_r2_p95": float(
            np.nanpercentile(
                epoch_r2,
                95,
            )
        ),
        "fit_epochs_axis_x": int(
            np.count_nonzero(
                keep_x
            )
        ),
        "fit_epochs_axis_y": int(
            np.count_nonzero(
                keep_y
            )
        ),
        "scientific_constraint": (
            "Only time-linear spatial gradient "
            "terms are removed. Per-epoch "
            "intercepts are preserved. The ramp "
            "is anchored at the centroid of the "
            "existing reference PS. The master "
            "epoch is preserved exactly."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    marker.write_text(
        json.dumps(
            signature,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )
        writer.writerow(
            [
                "image_ix_1based",
                "day",
                "dt_year",
                "intercept_rad",
                "axis_x_gradient_rad_per_1000coord",
                "axis_y_gradient_rad_per_1000coord",
                "robust_r2",
                "n_fit",
                "axis_x_gradient_model_rad_per_1000coord",
                "axis_y_gradient_model_rad_per_1000coord",
            ]
        )

        for j in range(
            n_image
        ):
            writer.writerow(
                [
                    j + 1,
                    f"{day[j]:.8f}",
                    f"{dt_year[j]:.10f}",
                    f"{intercept[j]:.12e}",
                    f"{bx[j]:.12e}",
                    f"{by[j]:.12e}",
                    f"{epoch_r2[j]:.8f}",
                    int(
                        epoch_n[j]
                    ),
                    f"{bx_model[j]:.12e}",
                    f"{by_model[j]:.12e}",
                ]
            )

    print(
        "[DERAMP] complete: "
        f"{input_phase_file} -> "
        f"{output_path.name}; "
        f"kx={kx:+.6e}, "
        f"ky={ky:+.6e}",
        flush=True,
    )

    return output_path
