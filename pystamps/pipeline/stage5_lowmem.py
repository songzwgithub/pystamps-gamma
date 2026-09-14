
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pystamps.io.mat import read_mat, read_mat_variables


_ROW_MAJOR = "PY_STAMPS_row_major"
_C32 = np.dtype([("real", "<f4"), ("imag", "<f4")])


def _rm_attr(ds: h5py.Dataset) -> None:
    ds.attrs[_ROW_MAJOR] = np.asarray(1, dtype=np.uint8)


def _chunk_rows(n_rows: int, n_cols: int, itemsize: int, target_mb: int = 16) -> int:
    if n_rows <= 0:
        return 1
    row_bytes = max(1, int(n_cols) * int(itemsize))
    target = int(target_mb) * 1024 * 1024
    return max(1, min(int(n_rows), target // row_bytes))


def _create_2d(
    h5: h5py.File,
    name: str,
    shape: tuple[int, int],
    dtype: Any,
) -> h5py.Dataset:
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        ds = h5.create_dataset(name, shape=(rows, cols), dtype=dtype)
    else:
        cr = _chunk_rows(rows, cols, np.dtype(dtype).itemsize)
        ds = h5.create_dataset(
            name,
            shape=(rows, cols),
            dtype=dtype,
            chunks=(min(cr, rows), cols),
        )
    _rm_attr(ds)
    return ds


def _create_numeric(
    h5: h5py.File,
    name: str,
    value: Any,
    dtype: Any | None = None,
) -> h5py.Dataset:
    arr = np.asarray(value if value is not None else [])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ds = h5.create_dataset(name, data=arr)
    _rm_attr(ds)
    return ds


def _write_complex_rows(ds: h5py.Dataset, out_pos: np.ndarray, values: np.ndarray) -> None:
    z = np.asarray(values, dtype=np.complex64)
    tmp = np.empty(z.shape, dtype=_C32)
    tmp["real"] = z.real
    tmp["imag"] = z.imag
    ds[out_pos, :] = tmp


def _n_ps_only(patch: Path) -> int:
    payload = read_mat_variables(patch / "ps2.mat", ("n_ps",))
    value = np.asarray(payload.get("n_ps", 0)).reshape(-1)
    if value.size == 0:
        raise RuntimeError(f"{patch.name}/ps2.mat missing n_ps")
    return int(round(float(value[0])))


def _copy_meta_value(ps: dict[str, Any], key: str) -> np.ndarray | None:
    value = ps.get(key)
    if value is None:
        return None
    return np.asarray(value).copy()


def stage5_merge_lowmem(dataset_root: str | Path) -> str:
    from pystamps.pipeline import ported as p

    root = Path(dataset_root).expanduser().resolve()
    patch_dirs = p._discover_patch_dirs(root)
    if not patch_dirs:
        raise p.PortedStageError("No patch directories found for merged stage-5 processing")

    parms_file = p._resolve_file(root, "parms.mat")
    parms_raw = read_mat(parms_file) if parms_file is not None else {}

    sb_raw = np.asarray(parms_raw.get("small_baseline_flag", "n"))
    if sb_raw.dtype.kind in {"U", "S"}:
        small_baseline = "".join(sb_raw.astype(str).reshape(-1).tolist()).strip().lower()
    elif sb_raw.size:
        small_baseline = str(sb_raw.reshape(-1)[0]).strip().lower()
    else:
        small_baseline = "n"

    if small_baseline != "y":
        raise p.PortedStageError(
            "Low-memory Stage-5 merger currently requires small_baseline_flag='y'"
        )

    heading_raw = np.asarray(parms_raw.get("heading", 0.0)).reshape(-1)
    heading_deg = float(heading_raw[0]) if heading_raw.size else 0.0

    # === STAGE5_LOWMEM_NETWORK_META_V1 ===
    network_meta: dict[str, np.ndarray] = {}
    for _patch in patch_dirs:
        _ps1 = _patch / "ps1.mat"
        if not _ps1.is_file():
            continue
        _net = read_mat_variables(_ps1, ("ifgday_ix", "ifgday"))
        if "ifgday_ix" in _net and np.asarray(_net["ifgday_ix"]).size:
            network_meta["ifgday_ix"] = np.asarray(_net["ifgday_ix"], dtype=np.int32)
            if "ifgday" in _net and np.asarray(_net["ifgday"]).size:
                network_meta["ifgday"] = np.asarray(_net["ifgday"], dtype=np.float64)
            break

    if "ifgday_ix" not in network_meta:
        raise p.PortedStageError(
            "Stage-5 low-memory merge could not recover ifgday_ix from PATCH_*/ps1.mat"
        )

    print(
        f"[STAGE5_MERGE][LOWMEM] pass 1/2: topology/index audit for "
        f"{len(patch_dirs)} patches",
        flush=True,
    )

    merged_index_by_key: dict[bytes, int] = {}
    merged_count = 0

    lon_chunks: list[np.ndarray] = []
    coh_chunks: list[np.ndarray] = []
    owner_patch_chunks: list[np.ndarray] = []
    owner_row_chunks: list[np.ndarray] = []

    base_meta: dict[str, np.ndarray | None] | None = None
    dims: dict[str, int] | None = None
    nonempty = 0

    required_optional = (
        "bp_patch",
        "hgt_patch",
        "la_patch",
        "inc_patch",
        "rc_patch",
    )

    for patch_id, patch in enumerate(patch_dirs):
        n_ps = _n_ps_only(patch)
        if n_ps == 0:
            print(
                f"[STAGE5_MERGE][LOWMEM] {patch.name}: n_ps=0, skip",
                flush=True,
            )
            continue

        bundle = p._load_stage5_patch_bundle(patch)
        nonempty += 1

        for attr in required_optional:
            if getattr(bundle, attr) is None:
                raise p.PortedStageError(
                    f"{patch.name}: required Stage-5 artifact missing ({attr})"
                )

        if base_meta is None:
            base_meta = {
                key: _copy_meta_value(bundle.ps, key)
                for key in (
                    "bperp",
                    "day",
                    "ll0",
                    "master_day",
                    "master_ix",
                    "n_ifg",
                    "n_image",
                    "mean_range",
                )
            }
            dims = {
                "ph": int(bundle.ph_patch2.shape[1]),
                "ph_patch": int(bundle.ph_patch_patch.shape[1]),
                "ph_res": int(bundle.ph_res_patch.shape[1]),
                "bp": int(bundle.bp_patch.shape[1]),
                "rc": int(bundle.rc_patch.shape[1]),
            }
        else:
            assert dims is not None
            current = {
                "ph": int(bundle.ph_patch2.shape[1]),
                "ph_patch": int(bundle.ph_patch_patch.shape[1]),
                "ph_res": int(bundle.ph_res_patch.shape[1]),
                "bp": int(bundle.bp_patch.shape[1]),
                "rc": int(bundle.rc_patch.shape[1]),
            }
            if current != dims:
                raise p.PortedStageError(
                    f"{patch.name}: inconsistent Stage-5 matrix widths: "
                    f"{current} != {dims}"
                )

        keep_patch, _remove = p._compute_patch_keep_mask(
            bundle.ij_cols,
            bundle.ij_keys,
            bundle.patch_bounds,
            merged_index_by_key,
        )
        kept = np.flatnonzero(keep_patch).astype(np.int32, copy=False)

        if kept.size:
            lon_chunks.append(
                bundle.lonlat_patch[kept, :].astype(np.float64, copy=True)
            )
            coh_chunks.append(
                bundle.coh_patch[kept].astype(np.float64, copy=True)
            )
            owner_patch_chunks.append(
                np.full(kept.size, patch_id, dtype=np.int16)
            )
            owner_row_chunks.append(kept.copy())

            for offset, idx in enumerate(kept.tolist()):
                merged_index_by_key.setdefault(
                    bundle.ij_keys[idx],
                    merged_count + offset,
                )
            merged_count += int(kept.size)

        if (patch_id + 1) % 5 == 0 or patch_id + 1 == len(patch_dirs):
            print(
                f"[STAGE5_MERGE][LOWMEM] pass1 "
                f"{patch_id+1}/{len(patch_dirs)} "
                f"provisional={merged_count:,}",
                flush=True,
            )

        del bundle
        gc.collect()

    if (
        nonempty == 0
        or base_meta is None
        or dims is None
        or merged_count == 0
    ):
        raise p.PortedStageError(
            "No non-empty Stage-5 patch data available for merge"
        )

    lonlat = np.concatenate(lon_chunks, axis=0)
    coh = np.concatenate(coh_chunks, axis=0)
    owner_patch = np.concatenate(owner_patch_chunks, axis=0)
    owner_row = np.concatenate(owner_row_chunks, axis=0)

    del lon_chunks, coh_chunks, owner_patch_chunks, owner_row_chunks
    gc.collect()

    keep_lonlat = p._dedup_lonlat_keep_highest_coh(lonlat, coh)
    active = np.flatnonzero(keep_lonlat)

    active_lonlat = lonlat[active, :]
    xy_local, ll0_xy = p._local_xy_from_lonlat(
        active_lonlat,
        heading_deg=heading_deg,
    )

    sort_ix = np.lexsort((xy_local[:, 0], xy_local[:, 1]))
    final_indices = active[sort_ix]

    final_lonlat = lonlat[final_indices, :].astype(
        np.float64,
        copy=True,
    )
    final_xy_local = xy_local[sort_ix, :].astype(
        np.float32,
        copy=True,
    )
    final_patch = owner_patch[final_indices].copy()
    final_source_row = owner_row[final_indices].copy()

    n_final = int(final_indices.size)

    print(
        f"[STAGE5_MERGE][LOWMEM] provisional={merged_count:,}, "
        f"final unique={n_final:,}",
        flush=True,
    )

    del lonlat, coh, owner_patch, owner_row
    del keep_lonlat, active, active_lonlat, xy_local
    del sort_ix, final_indices
    gc.collect()

    tmp_paths = {
        name: root / f"{name}.lowmem.tmp"
        for name in (
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "hgt2.mat",
            "la2.mat",
            "inc2.mat",
            "rc2.mat",
            "psver.mat",
            "ifgstd2.mat",
        )
    }

    for path in tmp_paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    success = False

    try:
        fps = h5py.File(tmp_paths["ps2.mat"], "w")
        fph = h5py.File(tmp_paths["ph2.mat"], "w")
        fpm = h5py.File(tmp_paths["pm2.mat"], "w")
        fbp = h5py.File(tmp_paths["bp2.mat"], "w")
        fhg = h5py.File(tmp_paths["hgt2.mat"], "w")
        fla = h5py.File(tmp_paths["la2.mat"], "w")
        finc = h5py.File(tmp_paths["inc2.mat"], "w")
        frc = h5py.File(tmp_paths["rc2.mat"], "w")

        files = (fps, fph, fpm, fbp, fhg, fla, finc, frc)

        try:
            ds_ij = _create_2d(
                fps,
                "ij",
                (n_final, 3),
                np.float64,
            )
            ds_lon = _create_2d(
                fps,
                "lonlat",
                (n_final, 2),
                np.float64,
            )
            ds_xy = _create_2d(
                fps,
                "xy",
                (n_final, 3),
                np.float32,
            )

            for key in (
                "bperp",
                "day",
                "master_day",
                "master_ix",
                "n_ifg",
                "n_image",
                "mean_range",
            ):
                value = base_meta.get(key)
                if value is not None:
                    _create_numeric(fps, key, value)

            for _name, _value in network_meta.items():
                _create_numeric(fps, _name, _value)

            _create_numeric(
                fps,
                "ll0",
                np.asarray(
                    ll0_xy,
                    dtype=np.float64,
                ).reshape(1, -1),
            )
            _create_numeric(
                fps,
                "n_ps",
                np.asarray([[float(n_final)]], dtype=np.float64),
            )

            ds_lon[:, :] = final_lonlat

            xy3 = np.empty((n_final, 3), dtype=np.float32)
            xy3[:, 0] = np.arange(
                1,
                n_final + 1,
                dtype=np.float32,
            )
            xy3[:, 1:] = final_xy_local
            ds_xy[:, :] = xy3
            del xy3
            gc.collect()

            ds_ph = _create_2d(
                fph,
                "ph",
                (n_final, dims["ph"]),
                _C32,
            )
            ds_k = _create_2d(
                fpm,
                "K_ps",
                (n_final, 1),
                np.float64,
            )
            ds_c = _create_2d(
                fpm,
                "C_ps",
                (n_final, 1),
                np.float64,
            )
            ds_coh = _create_2d(
                fpm,
                "coh_ps",
                (n_final, 1),
                np.float64,
            )
            ds_pp = _create_2d(
                fpm,
                "ph_patch",
                (n_final, dims["ph_patch"]),
                _C32,
            )
            ds_pr = _create_2d(
                fpm,
                "ph_res",
                (n_final, dims["ph_res"]),
                np.float32,
            )
            ds_bp = _create_2d(
                fbp,
                "bperp_mat",
                (n_final, dims["bp"]),
                np.float32,
            )
            ds_hgt = _create_2d(
                fhg,
                "hgt",
                (n_final, 1),
                np.float32,
            )
            ds_la = _create_2d(
                fla,
                "la",
                (n_final, 1),
                np.float64,
            )
            ds_inc = _create_2d(
                finc,
                "inc",
                (n_final, 1),
                np.float64,
            )
            ds_rc = _create_2d(
                frc,
                "ph_rc",
                (n_final, dims["rc"]),
                _C32,
            )

            if (
                dims["ph"] != dims["ph_patch"]
                or dims["ph"] != dims["bp"]
            ):
                raise p.PortedStageError(
                    "SB low-memory merge requires ph2, ph_patch, "
                    "and bp2 widths to match; "
                    f"got ph={dims['ph']}, "
                    f"ph_patch={dims['ph_patch']}, "
                    f"bp={dims['bp']}"
                )

            sumsq = np.zeros(dims["ph"], dtype=np.float64)
            inc_sum = 0.0
            inc_count = 0

            print(
                f"[STAGE5_MERGE][LOWMEM] pass 2/2: "
                f"stream-copy {n_final:,} final PS",
                flush=True,
            )

            for patch_id, patch in enumerate(patch_dirs):
                out_pos_all = np.flatnonzero(
                    final_patch == patch_id
                )

                if out_pos_all.size == 0:
                    continue

                bundle = p._load_stage5_patch_bundle(patch)
                src_all = final_source_row[
                    out_pos_all
                ].astype(np.int64, copy=False)

                step = _chunk_rows(
                    int(out_pos_all.size),
                    max(
                        dims["ph"],
                        dims["ph_patch"],
                        dims["bp"],
                        dims["rc"],
                    ),
                    8,
                    target_mb=24,
                )
                step = max(128, min(step, 4096))

                for start in range(
                    0,
                    int(out_pos_all.size),
                    step,
                ):
                    stop = min(
                        start + step,
                        int(out_pos_all.size),
                    )
                    out_pos = out_pos_all[start:stop]
                    src = src_all[start:stop]

                    ijv = bundle.ij_patch[
                        src,
                        :,
                    ].astype(np.float64, copy=True)
                    ijv[:, 0] = (
                        out_pos.astype(np.float64) + 1.0
                    )
                    ds_ij[out_pos, :] = ijv

                    ds_k[out_pos, 0] = bundle.k_patch[src]
                    ds_c[out_pos, 0] = bundle.c_patch[src]
                    ds_coh[out_pos, 0] = bundle.coh_patch[src]
                    ds_pr[out_pos, :] = bundle.ph_res_patch[
                        src,
                        :,
                    ]
                    ds_bp[out_pos, :] = bundle.bp_patch[
                        src,
                        :,
                    ]
                    ds_hgt[out_pos, 0] = np.asarray(
                        bundle.hgt_patch
                    )[src]
                    ds_la[out_pos, 0] = np.asarray(
                        bundle.la_patch
                    )[src]

                    incv = np.asarray(
                        bundle.inc_patch
                    )[src].astype(np.float64, copy=False)
                    ds_inc[out_pos, 0] = incv

                    phv = bundle.ph_patch2[
                        src,
                        :,
                    ].astype(np.complex64, copy=False)
                    ppv = bundle.ph_patch_patch[
                        src,
                        :,
                    ].astype(np.complex64, copy=False)
                    bpv = bundle.bp_patch[
                        src,
                        :,
                    ].astype(np.float32, copy=False)
                    kv = bundle.k_patch[
                        src
                    ].astype(np.float32, copy=False)

                    _write_complex_rows(
                        ds_ph,
                        out_pos,
                        phv,
                    )
                    _write_complex_rows(
                        ds_pp,
                        out_pos,
                        ppv,
                    )
                    _write_complex_rows(
                        ds_rc,
                        out_pos,
                        bundle.rc_patch[src, :],
                    )

                    phase = kv[:, None] * bpv
                    corr = np.exp(
                        (-1j * phase).astype(np.complex64)
                    )
                    residual = np.angle(
                        phv
                        * np.conj(ppv)
                        * corr
                    ).astype(np.float32, copy=False)

                    residual64 = residual.astype(
                        np.float64,
                        copy=False,
                    )
                    sumsq += np.sum(
                        residual64 * residual64,
                        axis=0,
                    )

                    inc_sum += float(
                        np.sum(incv, dtype=np.float64)
                    )
                    inc_count += int(incv.size)

                print(
                    f"[STAGE5_MERGE][LOWMEM] pass2 "
                    f"{patch_id+1}/{len(patch_dirs)} "
                    f"{patch.name}: wrote "
                    f"{out_pos_all.size:,} rows",
                    flush=True,
                )

                del bundle
                gc.collect()

            mean_inc = inc_sum / max(1, inc_count)

            _create_numeric(
                fps,
                "mean_incidence",
                np.asarray(
                    [[mean_inc]],
                    dtype=np.float64,
                ),
            )

            for f in files:
                f.flush()

        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

        ifg_std = (
            np.sqrt(sumsq / max(1, n_final))
            * (180.0 / np.pi)
        ).astype(np.float32)

        with h5py.File(
            tmp_paths["ifgstd2.mat"],
            "w",
        ) as f:
            _create_numeric(
                f,
                "ifg_std",
                ifg_std,
            )

        with h5py.File(
            tmp_paths["psver.mat"],
            "w",
        ) as f:
            _create_numeric(
                f,
                "psver",
                np.asarray(
                    [[2.0]],
                    dtype=np.float64,
                ),
            )

        checks = {
            "ps2.mat": (
                "n_ps",
                "ij",
                "lonlat",
                "xy",
            ),
            "ph2.mat": ("ph",),
            "pm2.mat": (
                "K_ps",
                "C_ps",
                "coh_ps",
                "ph_patch",
                "ph_res",
            ),
            "bp2.mat": ("bperp_mat",),
            "inc2.mat": ("inc",),
            "ifgstd2.mat": ("ifg_std",),
        }

        for name, vars_ in checks.items():
            payload = read_mat_variables(
                tmp_paths[name],
                vars_,
            )
            if not payload:
                raise p.PortedStageError(
                    f"Low-memory Stage-5 validation failed: "
                    f"{name}"
                )

        for name, tmp in tmp_paths.items():
            os.replace(
                tmp,
                root / name,
            )

        success = True

        return (
            f"Merged {len(patch_dirs)} patches into "
            f"{n_final} PS records "
            f"(low-memory streaming)"
        )

    finally:
        if not success:
            for path in tmp_paths.values():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
