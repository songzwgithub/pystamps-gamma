from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat


class MatReadError(RuntimeError):
    """Raised for unsupported MAT formats."""


def _decode_h5_dataset(obj: Any, h5file: Any) -> Any:
    import h5py  # type: ignore

    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        arr = np.asarray(arr)
        row_major = bool(np.asarray(obj.attrs.get("PY_STAMPS_row_major", 0)).reshape(-1)[0])

        # MATLAB complex arrays in v7.3 often appear as compound datasets.
        if arr.dtype.names and {"real", "imag"}.issubset(set(arr.dtype.names)):
            arr = arr["real"] + 1j * arr["imag"]

        # Dereference cell/object datasets recursively.
        if arr.dtype.kind == "O":
            out = np.empty(arr.shape, dtype=object)
            for idx, ref in np.ndenumerate(arr):
                out[idx] = _decode_h5_dataset(h5file[ref], h5file)
            arr = out

        # MATLAB stores arrays in column-major order; h5py exposes reversed axes.
        if arr.ndim >= 2 and not row_major:
            arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
        return arr

    if isinstance(obj, h5py.Group):
        keys = set(obj.keys())
        if {"data", "ir", "jc", "shape"}.issubset(keys):
            data = np.asarray(obj["data"][()])
            ir = np.asarray(obj["ir"][()], dtype=np.int32).reshape(-1)
            jc = np.asarray(obj["jc"][()], dtype=np.int32).reshape(-1)
            shape_arr = np.asarray(obj["shape"][()], dtype=np.int64).reshape(-1)
            if shape_arr.size >= 2:
                return sparse.csc_matrix((data, ir, jc), shape=(int(shape_arr[0]), int(shape_arr[1])))
        data: dict[str, Any] = {}
        for key in obj.keys():
            data[key] = _decode_h5_dataset(obj[key], h5file)
        return data

    return obj


def read_mat(path: str | Path) -> dict[str, Any]:
    mat_path = Path(path)
    try:
        payload = loadmat(mat_path, simplify_cells=True)
    except (NotImplementedError, ValueError):
        with mat_path.open("rb") as f:
            pure_hdf5 = f.read(8) == b"\x89HDF\r\n\x1a\n"
        if not pure_hdf5:
            try:
                import mat73  # type: ignore

                payload = mat73.loadmat(str(mat_path))
                if isinstance(payload, dict) and any(value is not None for value in payload.values()):
                    return payload
            except Exception:
                pass

        try:
            import h5py  # type: ignore
        except ImportError as import_exc:
            raise MatReadError(
                f"MAT v7.3 file requires h5py: {mat_path}. Install h5py or convert file format."
            ) from import_exc

        data: dict[str, Any] = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                data[key] = _decode_h5_dataset(f[key], f)
        return data
    return {k: v for k, v in payload.items() if not k.startswith("__")}


# === LARGE_MAT_HDF5_FALLBACK_V1 ===
_MAT_V5_MAX_BYTES = (1 << 32) - 1
_PYSTAMPS_ROW_MAJOR_ATTR = "PY_STAMPS_row_major"
_H5_COMPLEX64 = np.dtype([("real", "<f4"), ("imag", "<f4")])
_H5_COMPLEX128 = np.dtype([("real", "<f8"), ("imag", "<f8")])


def _payload_nbytes(value: Any) -> int:
    if sparse.issparse(value):
        return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    try:
        return int(np.asarray(value).nbytes)
    except Exception:
        return 0


def _large_mat_requires_hdf5(payload: dict[str, Any]) -> bool:
    total = 0
    for value in payload.values():
        nbytes = _payload_nbytes(value)
        total += nbytes
        if nbytes > _MAT_V5_MAX_BYTES:
            return True
    return total > _MAT_V5_MAX_BYTES


def _h5_row_chunk(rows: int, cols: int, itemsize: int) -> int:
    if rows <= 0:
        return 1
    row_bytes = max(1, int(cols) * int(itemsize))
    return max(1, min(int(rows), (16 * 1024 * 1024) // row_bytes))


def _h5_write_numeric_dataset(h5: Any, name: str, value: Any) -> None:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if np.iscomplexobj(arr):
        compound = _H5_COMPLEX64 if arr.dtype.itemsize <= 8 else _H5_COMPLEX128
        real_dtype = np.float32 if compound == _H5_COMPLEX64 else np.float64
        shape = arr.shape
        if arr.ndim == 2 and shape[0] > 0 and shape[1] > 0:
            cr = _h5_row_chunk(shape[0], shape[1], compound.itemsize)
            ds = h5.create_dataset(
                name, shape=shape, dtype=compound,
                chunks=(min(cr, shape[0]), shape[1]),
            )
            for start in range(0, shape[0], cr):
                stop = min(shape[0], start + cr)
                z = np.asarray(arr[start:stop, :])
                tmp = np.empty(z.shape, dtype=compound)
                tmp["real"] = z.real.astype(real_dtype, copy=False)
                tmp["imag"] = z.imag.astype(real_dtype, copy=False)
                ds[start:stop, :] = tmp
        else:
            z = np.asarray(arr)
            tmp = np.empty(z.shape, dtype=compound)
            tmp["real"] = z.real.astype(real_dtype, copy=False)
            tmp["imag"] = z.imag.astype(real_dtype, copy=False)
            ds = h5.create_dataset(name, data=tmp)
    else:
        if arr.dtype.kind in {"U", "S"}:
            text = "".join(str(x) for x in arr.reshape(-1))
            ds = h5.create_dataset(name, data=np.bytes_(text))
        elif arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            cr = _h5_row_chunk(arr.shape[0], arr.shape[1], arr.dtype.itemsize)
            ds = h5.create_dataset(
                name, shape=arr.shape, dtype=arr.dtype,
                chunks=(min(cr, arr.shape[0]), arr.shape[1]),
            )
            for start in range(0, arr.shape[0], cr):
                stop = min(arr.shape[0], start + cr)
                ds[start:stop, :] = arr[start:stop, :]
        else:
            ds = h5.create_dataset(name, data=arr)

    ds.attrs[_PYSTAMPS_ROW_MAJOR_ATTR] = np.asarray(1, dtype=np.uint8)


def _write_large_hdf5_mat(path: Path, payload: dict[str, Any]) -> None:
    import h5py  # type: ignore

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass

    try:
        with h5py.File(tmp, "w") as h5:
            for name, value in payload.items():
                if sparse.issparse(value):
                    csc = value.tocsc()
                    grp = h5.create_group(name)
                    grp.create_dataset("data", data=csc.data)
                    grp.create_dataset("ir", data=csc.indices.astype(np.int32, copy=False))
                    grp.create_dataset("jc", data=csc.indptr.astype(np.int32, copy=False))
                    grp.create_dataset("shape", data=np.asarray(csc.shape, dtype=np.int64))
                    continue

                arr = np.asarray(value)
                if arr.dtype.kind == "O":
                    raise MatReadError(
                        f"Large HDF5 MAT fallback does not support object variable '{name}'"
                    )
                _h5_write_numeric_dataset(h5, name, arr)

            h5.flush()

        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def write_mat(path: str | Path, payload: dict[str, Any]) -> None:
    mat_path = Path(path)
    if _large_mat_requires_hdf5(payload):
        _write_large_hdf5_mat(mat_path, payload)
        return
    savemat(mat_path, payload)

# === STAGE3_FAST_SELECTIVE_MAT_V1 ===
def read_mat_variables(
    path: str | Path,
    variable_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """
    Read only selected variables from a MAT file.

    For classic MAT files this uses scipy.io.loadmat(variable_names=...).
    For MAT v7.3/HDF5 files it opens only the requested HDF5 datasets.

    This is important for Stage 3 because pm1.mat may contain very large
    variables such as ph_weight that Stage 3 does not need.
    """

    mat_path = Path(path)

    names = tuple(
        dict.fromkeys(
            str(name)
            for name in variable_names
            if str(name)
        )
    )

    if not names:
        return {}

    try:
        payload = loadmat(
            mat_path,
            simplify_cells=True,
            variable_names=list(names),
        )

        return {
            key: value
            for key, value in payload.items()
            if not key.startswith("__")
            and key in names
        }

    except (
        NotImplementedError,
        ValueError,
        OSError,
    ):
        pass

    try:
        import h5py  # type: ignore

    except ImportError as exc:
        raise MatReadError(
            f"Selective MAT v7.3 reading requires h5py: {mat_path}"
        ) from exc

    data: dict[str, Any] = {}

    try:
        with h5py.File(
            mat_path,
            "r",
        ) as h5_file:
            for name in names:
                if name in h5_file:
                    data[name] = _decode_h5_dataset(
                        h5_file[name],
                        h5_file,
                    )

    except OSError as exc:
        raise MatReadError(
            f"Unable to selectively read MAT file: {mat_path}"
        ) from exc

    return data
