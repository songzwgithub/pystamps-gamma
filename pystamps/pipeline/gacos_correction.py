from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from pystamps.io.mat import read_mat, read_mat_variables
from pystamps.pipeline.stage6_sbas import load_sbas_network


class GacosCorrectionError(RuntimeError):
    """Raised when GACOS correction cannot be completed safely."""


_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")


@dataclass(slots=True)
class GacosProduct:
    date: str
    path: Path
    kind: str
    rsc_path: Path | None = None


@dataclass(slots=True)
class GacosConfig:
    gacos_dir: Path
    product_format: str
    product_unit: str
    projection: str
    sign: str
    strict_dates: bool
    rebuild: bool
    incidence_tif: Path | None
    incidence_deg: float | None
    qa_ps: int
    qa_ifg: int
    chunk_ps: int
    min_valid_fraction: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", "n"}


def _scalar(value: Any, default: float | None = None) -> float:
    arr = np.asarray(value) if value is not None else np.asarray([])
    if arr.size:
        return float(arr.reshape(-1)[0])
    if default is None:
        raise GacosCorrectionError("Required scalar value is missing")
    return float(default)


def _as_matrix(value: Any, rows: int, name: str, dtype: Any) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise GacosCorrectionError(f"{name} must be 2-D, got {arr.shape}")
    if arr.shape[0] != rows and arr.shape[1] == rows:
        arr = arr.T
    if arr.shape[0] != rows:
        raise GacosCorrectionError(
            f"{name} shape {arr.shape}; expected first dimension {rows}"
        )
    return np.asarray(arr, dtype=dtype)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _matlab_day_to_date(value: float) -> str:
    integer = int(math.floor(float(value)))
    fraction = float(value) - integer
    dt = datetime.fromordinal(integer) + timedelta(days=fraction) - timedelta(days=366)
    return dt.strftime("%Y%m%d")


def _day_labels(day: np.ndarray) -> list[str]:
    values = np.asarray(day, dtype=np.float64).reshape(-1)
    if not values.size:
        raise GacosCorrectionError("Acquisition day vector is empty")
    median = float(np.nanmedian(values))
    if median > 500000:
        return [_matlab_day_to_date(v) for v in values]
    if median > 10_000_000:
        return [str(int(round(v))) for v in values]
    raise GacosCorrectionError(
        "Acquisition dates are not MATLAB datenums or YYYYMMDD values"
    )


def _resolve_gacos_dir(dataset_root: Path) -> Path:
    raw = os.environ.get("PYSTAMPS_GACOS_DIR", "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            dataset_root / "GACOS",
            dataset_root / "gacos",
            dataset_root.parent / "GACOS",
            dataset_root.parent / "gacos",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise GacosCorrectionError(
        "Unable to locate GACOS directory. Set PYSTAMPS_GACOS_DIR=/path/to/GACOS"
    )


def _load_config(dataset_root: Path) -> GacosConfig:
    product_format = os.environ.get("PYSTAMPS_GACOS_FORMAT", "auto").strip().lower()
    if product_format not in {"auto", "tif", "ztd"}:
        raise GacosCorrectionError("PYSTAMPS_GACOS_FORMAT must be auto, tif, or ztd")
    product_unit = os.environ.get("PYSTAMPS_GACOS_UNIT", "auto").strip().lower()
    if product_unit not in {"auto", "m", "cm", "mm"}:
        raise GacosCorrectionError("PYSTAMPS_GACOS_UNIT must be auto, m, cm, or mm")
    projection = os.environ.get("PYSTAMPS_GACOS_PROJECTION", "zenith").strip().lower()
    if projection not in {"zenith", "los"}:
        raise GacosCorrectionError("PYSTAMPS_GACOS_PROJECTION must be zenith or los")
    sign = os.environ.get("PYSTAMPS_GACOS_SIGN", "auto").strip().lower()
    aliases = {"-": "subtract", "+": "add", "minus": "subtract", "plus": "add"}
    sign = aliases.get(sign, sign)
    if sign not in {"auto", "subtract", "add"}:
        raise GacosCorrectionError("PYSTAMPS_GACOS_SIGN must be auto, subtract, or add")

    incidence_tif_raw = os.environ.get("PYSTAMPS_GACOS_INCIDENCE_TIF", "").strip()
    incidence_tif = Path(incidence_tif_raw).expanduser().resolve() if incidence_tif_raw else None
    incidence_deg_raw = os.environ.get("PYSTAMPS_GACOS_INCIDENCE_DEG", "").strip()
    incidence_deg = float(incidence_deg_raw) if incidence_deg_raw else None

    return GacosConfig(
        gacos_dir=_resolve_gacos_dir(dataset_root),
        product_format=product_format,
        product_unit=product_unit,
        projection=projection,
        sign=sign,
        strict_dates=_env_bool("PYSTAMPS_GACOS_STRICT_DATES", True),
        rebuild=_env_bool("PYSTAMPS_GACOS_REBUILD", False),
        incidence_tif=incidence_tif,
        incidence_deg=incidence_deg,
        qa_ps=max(1000, int(os.environ.get("PYSTAMPS_GACOS_QA_PS", "30000"))),
        qa_ifg=max(10, int(os.environ.get("PYSTAMPS_GACOS_QA_IFG", "80"))),
        chunk_ps=max(256, int(os.environ.get("PYSTAMPS_GACOS_CHUNK_PS", "4096"))),
        min_valid_fraction=float(
            os.environ.get("PYSTAMPS_GACOS_MIN_VALID_FRACTION", "0.995")
        ),
    )


def _date_from_name(path: Path) -> str | None:
    match = _DATE_RE.search(path.name)
    return match.group(1) if match else None


def _find_rsc(ztd_path: Path) -> Path | None:
    candidates = [
        Path(str(ztd_path) + ".rsc"),
        ztd_path.with_suffix(ztd_path.suffix + ".rsc"),
        ztd_path.with_suffix(".rsc"),
        ztd_path.parent / f"{ztd_path.stem}.rsc",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_products(gacos_dir: Path, product_format: str) -> dict[str, GacosProduct]:
    tif_by_date: dict[str, list[Path]] = {}
    ztd_by_date: dict[str, list[tuple[Path, Path]]] = {}

    for path in gacos_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        date = _date_from_name(path)
        if date is None:
            continue
        if lower.endswith((".tif", ".tiff")):
            tif_by_date.setdefault(date, []).append(path)
        elif lower.endswith(".ztd"):
            rsc = _find_rsc(path)
            if rsc is not None:
                ztd_by_date.setdefault(date, []).append((path, rsc))

    products: dict[str, GacosProduct] = {}
    dates = sorted(set(tif_by_date) | set(ztd_by_date))
    for date in dates:
        if product_format in {"auto", "tif"} and date in tif_by_date:
            choices = sorted(tif_by_date[date], key=lambda p: (len(str(p)), str(p)))
            products[date] = GacosProduct(date=date, path=choices[0], kind="tif")
            continue
        if product_format in {"auto", "ztd"} and date in ztd_by_date:
            choices = sorted(ztd_by_date[date], key=lambda item: (len(str(item[0])), str(item[0])))
            ztd, rsc = choices[0]
            products[date] = GacosProduct(date=date, path=ztd, kind="ztd", rsc_path=rsc)
    return products


def _parse_rsc(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith(("#", ";")):
            continue
        fields = text.replace(":", " ").split()
        if len(fields) >= 2:
            result[fields[0].strip().upper()] = fields[1].strip()
    return result


def _sample_normalized(array: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(data)
    values = ndimage.map_coordinates(
        np.where(finite, data, 0.0),
        [row, col],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    weights = ndimage.map_coordinates(
        finite.astype(np.float64),
        [row, col],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled = np.divide(
        values,
        weights,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=weights > 0.25,
    )
    missing = ~np.isfinite(sampled)
    if np.any(missing):
        nearest = ndimage.map_coordinates(
            np.where(finite, data, np.nan),
            [row[missing], col[missing]],
            order=0,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        sampled[missing] = nearest
    return sampled


def _unit_scale_from_text(text: str) -> float | None:
    value = text.strip().lower().replace(" ", "")
    if "millimeter" in value or value in {"mm", "millimetre", "millimeters"}:
        return 1.0e-3
    if "centimeter" in value or value in {"cm", "centimetre", "centimeters"}:
        return 1.0e-2
    if value in {"m", "meter", "metre", "meters", "metres"}:
        return 1.0
    return None


def _forced_unit_scale(unit: str) -> float | None:
    return {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}.get(unit)


def _sample_tif(product: GacosProduct, lon: np.ndarray, lat: np.ndarray, unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import rasterio
        from pyproj import Transformer
    except Exception as exc:
        raise GacosCorrectionError(
            "GeoTIFF GACOS products require rasterio and pyproj"
        ) from exc

    with rasterio.open(product.path) as src:
        if src.crs is None:
            raise GacosCorrectionError(f"GACOS GeoTIFF has no CRS: {product.path}")
        data = src.read(1, masked=False).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        tags = {str(k).lower(): str(v) for k, v in src.tags().items()}
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        inv = ~src.transform
        col_corner, row_corner = inv * (np.asarray(x), np.asarray(y))
        row = np.asarray(row_corner, dtype=np.float64) - 0.5
        col = np.asarray(col_corner, dtype=np.float64) - 0.5
        sampled = _sample_normalized(data, row, col)

        scale = _forced_unit_scale(unit)
        detected_unit = None
        if scale is None:
            for key in ("unit", "units", "z_units", "vertical_units"):
                if key in tags:
                    scale = _unit_scale_from_text(tags[key])
                    if scale is not None:
                        detected_unit = tags[key]
                        break
        if scale is None:
            scale = 1.0
            detected_unit = "default_m"
        sampled *= scale
        return sampled, {
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "unit_scale_to_m": float(scale),
            "detected_unit": detected_unit,
        }


def _sample_ztd(product: GacosProduct, lon: np.ndarray, lat: np.ndarray, unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    if product.rsc_path is None:
        raise GacosCorrectionError(f"Missing RSC for {product.path}")
    rsc = _parse_rsc(product.rsc_path)
    required = ("WIDTH", "FILE_LENGTH", "X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP")
    missing = [key for key in required if key not in rsc]
    if missing:
        raise GacosCorrectionError(
            f"{product.rsc_path} missing keys: {', '.join(missing)}"
        )
    width = int(float(rsc["WIDTH"]))
    length = int(float(rsc["FILE_LENGTH"]))
    x_first = float(rsc["X_FIRST"])
    y_first = float(rsc["Y_FIRST"])
    x_step = float(rsc["X_STEP"])
    y_step = float(rsc["Y_STEP"])
    if x_step == 0 or y_step == 0:
        raise GacosCorrectionError(f"Invalid X_STEP/Y_STEP in {product.rsc_path}")

    byte_order = rsc.get("BYTE_ORDER", rsc.get("ENDIAN", "LSB")).upper()
    dtype = np.dtype(">f4") if byte_order in {"MSB", "BIG", "BIG_ENDIAN"} else np.dtype("<f4")
    expected = width * length * dtype.itemsize
    actual = product.path.stat().st_size
    if actual != expected:
        raise GacosCorrectionError(
            f"GACOS ZTD byte size mismatch for {product.path}: {actual}, expected {expected}"
        )
    grid = np.memmap(product.path, dtype=dtype, mode="r", shape=(length, width))
    row = (lat - y_first) / y_step
    col = (lon - x_first) / x_step
    sampled = _sample_normalized(grid, row, col)
    scale = _forced_unit_scale(unit)
    if scale is None:
        scale = 1.0
    sampled *= scale
    return sampled, {
        "width": width,
        "height": length,
        "x_first": x_first,
        "y_first": y_first,
        "x_step": x_step,
        "y_step": y_step,
        "byte_order": byte_order,
        "unit_scale_to_m": float(scale),
    }


def sample_product(product: GacosProduct, lon: np.ndarray, lat: np.ndarray, unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    if product.kind == "tif":
        return _sample_tif(product, lon, lat, unit)
    if product.kind == "ztd":
        return _sample_ztd(product, lon, lat, unit)
    raise GacosCorrectionError(f"Unsupported GACOS product kind: {product.kind}")


def _incidence_from_tif(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    product = GacosProduct(date="incidence", path=path, kind="tif")
    values, _meta = _sample_tif(product, lon, lat, "m")
    # _sample_tif applies a unit scale intended for delay. It is 1 for forced m,
    # so numeric incidence values remain unchanged.
    return values


def _resolve_incidence(
    config: GacosConfig,
    ps2: dict[str, Any],
    parms: dict[str, Any],
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, str]:
    if config.projection == "los":
        return np.zeros(lon.size, dtype=np.float64), "not_required_product_is_los"

    if config.incidence_tif is not None:
        if not config.incidence_tif.is_file():
            raise GacosCorrectionError(
                f"Incidence GeoTIFF does not exist: {config.incidence_tif}"
            )
        incidence = _incidence_from_tif(config.incidence_tif, lon, lat)
        finite = incidence[np.isfinite(incidence)]
        if finite.size == 0:
            raise GacosCorrectionError("Incidence GeoTIFF produced no finite PS values")
        if float(np.nanpercentile(np.abs(finite), 95)) <= math.pi + 0.1:
            radians = incidence
            source = f"radian_tif:{config.incidence_tif}"
        else:
            radians = np.deg2rad(incidence)
            source = f"degree_tif:{config.incidence_tif}"
        return radians.astype(np.float64), source

    if config.incidence_deg is not None:
        radians = np.full(lon.size, math.radians(config.incidence_deg), dtype=np.float64)
        return radians, "PYSTAMPS_GACOS_INCIDENCE_DEG"

    candidates = [
        (ps2.get("mean_incidence"), "ps2.mean_incidence"),
        (parms.get("mean_incidence"), "parms.mean_incidence"),
        (parms.get("incidence_angle"), "parms.incidence_angle"),
    ]
    for raw, source in candidates:
        if raw is None or np.asarray(raw).size == 0:
            continue
        value = float(np.asarray(raw).reshape(-1)[0])
        radians = value if abs(value) <= math.pi + 0.1 else math.radians(value)
        if not (0.0 < radians < math.radians(89.0)):
            continue
        return np.full(lon.size, radians, dtype=np.float64), source

    raise GacosCorrectionError(
        "No incidence angle is available. Set PYSTAMPS_GACOS_INCIDENCE_DEG "
        "or PYSTAMPS_GACOS_INCIDENCE_TIF."
    )


def _reference_indices(ps2: dict[str, Any], parms: dict[str, Any], n_ps: int) -> tuple[np.ndarray, str]:
    from pystamps.pipeline import ported

    ref = np.asarray(ported._select_reference_ps(ps2, parms), dtype=np.int64).reshape(-1)
    ref = ref[(ref >= 0) & (ref < n_ps)]
    if ref.size:
        return np.unique(ref), "pystamps_reference"
    return np.arange(n_ps, dtype=np.int64), "global_median_fallback"


def _inventory_fingerprint(products: dict[str, GacosProduct], dates: list[str]) -> str:
    digest = hashlib.sha256()
    for date in dates:
        product = products.get(date)
        if product is None:
            digest.update(f"{date}:missing\n".encode())
            continue
        stat = product.path.stat()
        digest.update(
            f"{date}:{product.kind}:{product.path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
        )
        if product.rsc_path is not None:
            rstat = product.rsc_path.stat()
            digest.update(
                f"rsc:{product.rsc_path}:{rstat.st_size}:{rstat.st_mtime_ns}\n".encode()
            )
    return digest.hexdigest()


def _robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    median = np.nanmedian(values, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - median), axis=axis)


def _choose_sign(
    ph_ifg: np.ndarray,
    los_delay: np.memmap,
    ifgday_ix: np.ndarray,
    wavelength_m: float,
    config: GacosConfig,
) -> tuple[str, dict[str, float]]:
    if config.sign in {"subtract", "add"}:
        return config.sign, {"selection": "forced"}

    n_ps, n_ifg = ph_ifg.shape
    ps_ix = np.linspace(0, n_ps - 1, min(config.qa_ps, n_ps), dtype=np.int64)
    ifg_ix = np.linspace(0, n_ifg - 1, min(config.qa_ifg, n_ifg), dtype=np.int64)
    phase_scale = 4.0 * math.pi / wavelength_m

    scores_raw: list[float] = []
    scores_subtract: list[float] = []
    scores_add: list[float] = []
    correlations: list[float] = []

    for j in ifg_ix:
        early = int(ifgday_ix[j, 0] - 1)
        late = int(ifgday_ix[j, 1] - 1)
        atmospheric = phase_scale * (
            np.asarray(los_delay[ps_ix, late], dtype=np.float64)
            - np.asarray(los_delay[ps_ix, early], dtype=np.float64)
        )
        raw = np.asarray(ph_ifg[ps_ix, j], dtype=np.float64)
        valid = np.isfinite(raw) & np.isfinite(atmospheric)
        if np.count_nonzero(valid) < 100:
            continue
        r = raw[valid]
        a = atmospheric[valid]
        r -= np.nanmedian(r)
        a -= np.nanmedian(a)
        scores_raw.append(float(_robust_scale(r, axis=0)))
        scores_subtract.append(float(_robust_scale(r - a, axis=0)))
        scores_add.append(float(_robust_scale(r + a, axis=0)))
        if np.nanstd(r) > 0 and np.nanstd(a) > 0:
            correlations.append(float(np.corrcoef(r, a)[0, 1]))

    if not scores_subtract:
        raise GacosCorrectionError("Unable to determine GACOS correction sign from valid QA samples")

    raw_score = float(np.nanmedian(scores_raw))
    subtract_score = float(np.nanmedian(scores_subtract))
    add_score = float(np.nanmedian(scores_add))
    chosen = "subtract" if subtract_score <= add_score else "add"
    best_score = min(subtract_score, add_score)
    improvement = 100.0 * (raw_score - best_score) / raw_score if raw_score > 0 else 0.0
    return chosen, {
        "selection": "auto_robust_spatial_scale",
        "raw_score_rad": raw_score,
        "subtract_score_rad": subtract_score,
        "add_score_rad": add_score,
        "chosen_improvement_percent": improvement,
        "median_raw_atmosphere_correlation": float(np.nanmedian(correlations)) if correlations else float("nan"),
        "qa_ifg_count": len(scores_subtract),
        "qa_ps_count": int(ps_ix.size),
    }


def _write_hdf5_mat(
    path: Path,
    raw_phase: np.ndarray,
    msd: np.ndarray,
    los_delay: np.memmap,
    ifgday_ix: np.ndarray,
    wavelength_m: float,
    sign: str,
    chunk_ps: int,
) -> None:
    try:
        import h5py
    except Exception as exc:
        raise GacosCorrectionError("Writing phuw2_gacos.mat requires h5py") from exc

    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    n_ps, n_ifg = raw_phase.shape
    phase_scale = 4.0 * math.pi / wavelength_m
    sign_factor = -1.0 if sign == "subtract" else 1.0

    with h5py.File(tmp, "w") as h5:
        dset = h5.create_dataset(
            "ph_uw",
            shape=(n_ps, n_ifg),
            dtype=np.float32,
            chunks=(min(chunk_ps, n_ps), min(32, n_ifg)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        dset.attrs["PY_STAMPS_row_major"] = np.asarray(1, dtype=np.uint8)
        for start in range(0, n_ps, chunk_ps):
            stop = min(start + chunk_ps, n_ps)
            early = ifgday_ix[:, 0].astype(np.int64) - 1
            late = ifgday_ix[:, 1].astype(np.int64) - 1
            atmospheric = phase_scale * (
                np.asarray(los_delay[start:stop, :][:, late], dtype=np.float64)
                - np.asarray(los_delay[start:stop, :][:, early], dtype=np.float64)
            )
            corrected = np.asarray(raw_phase[start:stop, :], dtype=np.float64)
            corrected += sign_factor * atmospheric
            dset[start:stop, :] = corrected.astype(np.float32)
            print(
                f"[GACOS][WRITE] {stop}/{n_ps} ({100.0 * stop / n_ps:.1f}%)",
                flush=True,
            )

        msd_arr = np.asarray(msd, dtype=np.float32).reshape(-1, 1)
        msd_dset = h5.create_dataset("msd", data=msd_arr)
        msd_dset.attrs["PY_STAMPS_row_major"] = np.asarray(1, dtype=np.uint8)
        h5.attrs["gacos_corrected"] = np.asarray(1, dtype=np.uint8)
        h5.attrs["gacos_sign"] = sign
        h5.attrs["wavelength_m"] = float(wavelength_m)

    os.replace(tmp, path)


def _write_inventory_csv(
    path: Path,
    dates: list[str],
    products: dict[str, GacosProduct],
    valid_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("date", "status", "kind", "path", "rsc_path", "valid_ps"),
        )
        writer.writeheader()
        for date in dates:
            product = products.get(date)
            writer.writerow(
                {
                    "date": date,
                    "status": "ok" if product is not None else "missing",
                    "kind": product.kind if product else "",
                    "path": str(product.path) if product else "",
                    "rsc_path": str(product.rsc_path) if product and product.rsc_path else "",
                    "valid_ps": valid_counts.get(date, 0),
                }
            )


def ensure_gacos_corrected_phuw(dataset_root: Path) -> Path:
    """Create or reuse phuw2_gacos.mat and return its path."""

    root = Path(dataset_root).expanduser().resolve()
    config = _load_config(root)
    output = root / "phuw2_gacos.mat"
    debug_path = root / "gacos_correction_debug.json"
    inventory_csv = root / "gacos_date_inventory.csv"
    work_dir = root / "_gacos_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    required = ("ps2.mat", "phuw2.mat", "parms.mat")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise GacosCorrectionError(f"Missing GACOS correction inputs: {', '.join(missing)}")

    ps2 = read_mat(root / "ps2.mat")
    n_ps = int(round(_scalar(ps2.get("n_ps"), 0)))
    if n_ps <= 0:
        raise GacosCorrectionError("ps2.mat contains invalid n_ps")
    lonlat = _as_matrix(ps2.get("lonlat"), n_ps, "ps2.lonlat", np.float64)
    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    phuw = read_mat_variables(root / "phuw2.mat", ("ph_uw", "msd"))
    ph_ifg = _as_matrix(phuw["ph_uw"], n_ps, "phuw2.ph_uw", np.float32)
    n_ps, n_ifg = ph_ifg.shape
    msd = np.asarray(phuw.get("msd", np.zeros(n_ifg)), dtype=np.float32).reshape(-1)
    if msd.size != n_ifg:
        msd = np.zeros(n_ifg, dtype=np.float32)

    day, ifgday_ix, _bperp, network_source = load_sbas_network(root, n_ifg)
    dates = _day_labels(day)
    ifgday_ix = np.asarray(ifgday_ix, dtype=np.int64)
    n_image = len(dates)

    products = discover_products(config.gacos_dir, config.product_format)
    missing_dates = [date for date in dates if date not in products]
    if missing_dates:
        preview = ", ".join(missing_dates[:20])
        raise GacosCorrectionError(
            f"Missing {len(missing_dates)}/{n_image} GACOS acquisition dates: {preview}. "
            "All acquisition dates are required; temporal interpolation is intentionally not used."
        )

    fingerprint = _inventory_fingerprint(products, dates)
    source_stat = (root / "phuw2.mat").stat()
    cache_signature = {
        "phuw2_size": source_stat.st_size,
        "phuw2_mtime_ns": source_stat.st_mtime_ns,
        "inventory_fingerprint": fingerprint,
        "format": config.product_format,
        "unit": config.product_unit,
        "projection": config.projection,
        "min_valid_fraction": config.min_valid_fraction,
        "sign_requested": config.sign,
        "incidence_tif": str(config.incidence_tif) if config.incidence_tif else None,
        "incidence_deg": config.incidence_deg,
    }

    if output.exists() and debug_path.exists() and not config.rebuild:
        try:
            existing = json.loads(debug_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and existing.get("cache_signature") == cache_signature:
                print(f"[GACOS] Reusing completed correction: {output}", flush=True)
                return output
        except Exception:
            pass

    started = time.perf_counter()
    parms = read_mat(root / "parms.mat")
    wavelength_m = _scalar(parms.get("lambda"), 0.0555)
    if not (0.001 < wavelength_m < 1.0):
        raise GacosCorrectionError(f"Invalid radar wavelength: {wavelength_m}")

    incidence, incidence_source = _resolve_incidence(config, ps2, parms, lon, lat)
    if config.projection == "zenith":
        cosine = np.cos(incidence)
        if np.any(~np.isfinite(cosine)) or np.nanmin(cosine) <= 0.05:
            raise GacosCorrectionError("Invalid incidence angles for zenith-to-LOS projection")
    else:
        cosine = np.ones(n_ps, dtype=np.float64)

    ref_ix, reference_source = _reference_indices(ps2, parms, n_ps)
    delay_path = work_dir / "gacos_los_ref.f32"
    los_delay = np.memmap(delay_path, dtype=np.float32, mode="w+", shape=(n_ps, n_image))
    los_delay[:] = np.nan
    los_delay.flush()

    valid_counts: dict[str, int] = {}
    metadata_by_kind: dict[str, dict[str, Any]] = {}
    for index, date in enumerate(dates):
        product = products.get(date)
        if product is None:
            continue
        sampled, meta = sample_product(product, lon, lat, config.product_unit)
        metadata_by_kind.setdefault(product.kind, meta)
        los = sampled / cosine
        finite_los = np.isfinite(los)
        valid_fraction = float(np.count_nonzero(finite_los) / n_ps)
        if valid_fraction < config.min_valid_fraction:
            raise GacosCorrectionError(
                f"GACOS coverage for {date} is only {100.0 * valid_fraction:.2f}% "
                f"(< {100.0 * config.min_valid_fraction:.2f}%)"
            )
        if not np.all(finite_los):
            from scipy.spatial import cKDTree

            valid_ix = np.flatnonzero(finite_los)
            missing_ix = np.flatnonzero(~finite_los)
            cos_lat = math.cos(math.radians(float(np.nanmedian(lat))))
            xy = np.column_stack((lon * cos_lat, lat))
            tree = cKDTree(xy[valid_ix, :])
            _distance, nearest = tree.query(xy[missing_ix, :], k=1)
            los[missing_ix] = los[valid_ix[np.asarray(nearest, dtype=np.int64)]]

        valid_ref = ref_ix[np.isfinite(los[ref_ix])]
        if valid_ref.size == 0:
            raise GacosCorrectionError(
                f"No finite GACOS values in reference region for {date}"
            )
        reference_value = float(np.nanmedian(los[valid_ref]))
        los -= reference_value
        los_delay[:, index] = los.astype(np.float32)
        los_delay.flush()
        valid_counts[date] = int(np.count_nonzero(np.isfinite(sampled)))
        print(
            f"[GACOS][SAMPLE] {index + 1}/{n_image} {date} "
            f"kind={product.kind} valid={valid_counts[date]}/{n_ps}",
            flush=True,
        )

    _write_inventory_csv(inventory_csv, dates, products, valid_counts)

    valid_date_columns = np.all(np.isfinite(los_delay), axis=0)
    if not np.all(valid_date_columns):
        bad = [dates[i] for i in np.flatnonzero(~valid_date_columns)]
        raise GacosCorrectionError(
            f"GACOS PS sampling is incomplete for {len(bad)} dates; first: {bad[:10]}"
        )

    sign, qa = _choose_sign(
        ph_ifg,
        los_delay,
        ifgday_ix,
        wavelength_m,
        config,
    )
    print(f"[GACOS] correction sign: {sign}; QA={qa}", flush=True)

    _write_hdf5_mat(
        output,
        ph_ifg,
        msd,
        los_delay,
        ifgday_ix,
        wavelength_m,
        sign,
        config.chunk_ps,
    )

    debug = {
        "status": "completed",
        "dataset_root": str(root),
        "gacos_dir": str(config.gacos_dir),
        "output": str(output),
        "source_phuw": str(root / "phuw2.mat"),
        "network_source": str(network_source),
        "n_ps": n_ps,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "dates_start": dates[0],
        "dates_end": dates[-1],
        "product_format_requested": config.product_format,
        "product_kind_counts": {
            "tif": sum(1 for date in dates if date in products and products[date].kind == "tif"),
            "ztd": sum(1 for date in dates if date in products and products[date].kind == "ztd"),
        },
        "missing_dates": missing_dates,
        "product_unit_requested": config.product_unit,
        "min_valid_fraction": config.min_valid_fraction,
        "projection": config.projection,
        "incidence_source": incidence_source,
        "incidence_deg_median": float(np.nanmedian(np.rad2deg(incidence))) if config.projection == "zenith" else None,
        "reference_source": reference_source,
        "reference_ps": int(ref_ix.size),
        "correction_sign": sign,
        "sign_qa": qa,
        "wavelength_m": wavelength_m,
        "cache_signature": cache_signature,
        "sample_metadata": metadata_by_kind,
        "work_delay_file": str(delay_path),
        "inventory_csv": str(inventory_csv),
        "duration_sec": time.perf_counter() - started,
        "phase_formula": "ph_corr = ph_raw - atm_phase for subtract; ph_raw + atm_phase for add",
        "atm_phase_formula": "4*pi/lambda * ((LOS_late-ref_late) - (LOS_early-ref_early))",
        "note": "GACOS applied after Stage 6 and before custom SBAS Stage 7/8; original phuw2.mat is preserved.",
    }
    _write_json(debug_path, debug)
    print(f"[GACOS] completed: {output}", flush=True)
    return output
