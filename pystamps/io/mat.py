from __future__ import annotations

from pathlib import Path
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


def write_mat(path: str | Path, payload: dict[str, Any]) -> None:
    savemat(Path(path), payload)

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
