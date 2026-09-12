from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from .gamma_sbas import (
    GammaInputError,
    first_integer_parameter,
    parse_gamma_parameter_file,
)


GammaDtypeName = Literal[
    "double",
    "fcomplex",
    "float",
    "scomplex",
    "int",
    "short",
    "uchar",
]


WIDTH_KEYS = (
    "interferogram_width",
    "range_samp_1",
    "range_samples",
    "width",
)

LENGTH_KEYS = (
    "interferogram_azimuth_lines",
    "az_samp_1",
    "azimuth_lines",
    "nlines",
)

FORMAT_KEYS = (
    "image_format",
    "data_type",
    "data_format",
)


@dataclass(frozen=True, slots=True)
class GammaDtypeSpec:
    """Storage and output definition for one GAMMA raster type."""

    name: str
    storage_dtype: np.dtype
    output_dtype: np.dtype
    bytes_per_pixel: int
    short_complex: bool = False


@dataclass(frozen=True, slots=True)
class GammaRasterLayout:
    """Resolved binary layout of one GAMMA raster."""

    path: Path
    width: int
    length: int
    dtype: GammaDtypeSpec

    @property
    def pixel_count(self) -> int:
        return self.width * self.length

    @property
    def expected_bytes(self) -> int:
        return self.pixel_count * self.dtype.bytes_per_pixel


_DTYPE_SPECS: dict[str, GammaDtypeSpec] = {
    "double": GammaDtypeSpec(
        name="double",
        storage_dtype=np.dtype(">f8"),
        output_dtype=np.dtype(np.float64),
        bytes_per_pixel=8,
    ),
    "fcomplex": GammaDtypeSpec(
        name="fcomplex",
        storage_dtype=np.dtype(">c8"),
        output_dtype=np.dtype(np.complex64),
        bytes_per_pixel=8,
    ),
    "float": GammaDtypeSpec(
        name="float",
        storage_dtype=np.dtype(">f4"),
        output_dtype=np.dtype(np.float32),
        bytes_per_pixel=4,
    ),
    "scomplex": GammaDtypeSpec(
        name="scomplex",
        storage_dtype=np.dtype(">i2"),
        output_dtype=np.dtype(np.complex64),
        bytes_per_pixel=4,
        short_complex=True,
    ),
    "int": GammaDtypeSpec(
        name="int",
        storage_dtype=np.dtype(">i4"),
        output_dtype=np.dtype(np.int32),
        bytes_per_pixel=4,
    ),
    "short": GammaDtypeSpec(
        name="short",
        storage_dtype=np.dtype(">i2"),
        output_dtype=np.dtype(np.int16),
        bytes_per_pixel=2,
    ),
    "uchar": GammaDtypeSpec(
        name="uchar",
        storage_dtype=np.dtype("u1"),
        output_dtype=np.dtype(np.uint8),
        bytes_per_pixel=1,
    ),
}


_DTYPE_ALIASES = {
    "double": "double",
    "real*8": "double",
    "fcomplex": "fcomplex",
    "complex": "fcomplex",
    "complex*8": "fcomplex",
    "float": "float",
    "real*4": "float",
    "scomplex": "scomplex",
    "short_complex": "scomplex",
    "int": "int",
    "integer": "int",
    "integer*4": "int",
    "short": "short",
    "integer*2": "short",
    "uchar": "uchar",
    "byte": "uchar",
    "unsigned_char": "uchar",
}


def normalize_gamma_dtype(
    dtype: str,
) -> GammaDtypeSpec:
    """Normalize a GAMMA datatype name."""

    key = dtype.strip().lower()

    normalized = _DTYPE_ALIASES.get(key)

    if normalized is None:
        raise GammaInputError(
            f"不支持的GAMMA栅格数据类型：{dtype}"
        )

    return _DTYPE_SPECS[normalized]


def gamma_dtype_from_parameter_file(
    parameter_file: str | Path,
) -> GammaDtypeSpec | None:
    """Read image_format/data_type/data_format from a GAMMA par file."""

    parameters = parse_gamma_parameter_file(
        Path(parameter_file),
    )

    for key in FORMAT_KEYS:
        values = parameters.get(key)

        if not values:
            continue

        try:
            return normalize_gamma_dtype(values[0])
        except GammaInputError:
            continue

    return None


def _resolve_parameter_dimensions(
    parameter_file: Path | None,
) -> tuple[int | None, int | None]:
    if parameter_file is None:
        return None, None

    parameters = parse_gamma_parameter_file(
        parameter_file,
    )

    width = first_integer_parameter(
        parameters,
        WIDTH_KEYS,
    )
    length = first_integer_parameter(
        parameters,
        LENGTH_KEYS,
    )

    return width, length


def resolve_gamma_raster_layout(
    path: str | Path,
    *,
    parameter_file: str | Path | None = None,
    width: int | None = None,
    length: int | None = None,
    dtype: str | None = None,
) -> GammaRasterLayout:
    """
    Resolve and validate a GAMMA binary raster layout.

    If length is omitted, it is inferred from:
        file_size / bytes_per_pixel / width
    """

    raster_path = Path(path).expanduser().resolve()

    if not raster_path.is_file():
        raise GammaInputError(
            f"GAMMA栅格文件不存在：{raster_path}"
        )

    par_path = (
        Path(parameter_file).expanduser().resolve()
        if parameter_file is not None
        else None
    )

    par_width, par_length = _resolve_parameter_dimensions(
        par_path,
    )

    if width is None:
        width = par_width
    elif par_width is not None and width != par_width:
        raise GammaInputError(
            f"显式宽度{width}与参数文件宽度"
            f"{par_width}不一致：{par_path}"
        )

    if length is None:
        length = par_length
    elif par_length is not None and length != par_length:
        raise GammaInputError(
            f"显式行数{length}与参数文件行数"
            f"{par_length}不一致：{par_path}"
        )

    if width is None or width <= 0:
        raise GammaInputError(
            f"无法确定GAMMA栅格宽度：{raster_path}"
        )

    if dtype is not None:
        dtype_spec = normalize_gamma_dtype(dtype)
    elif par_path is not None:
        dtype_spec = gamma_dtype_from_parameter_file(
            par_path,
        )
        if dtype_spec is None:
            dtype_spec = normalize_gamma_dtype("float")
    else:
        dtype_spec = normalize_gamma_dtype("float")

    file_size = raster_path.stat().st_size
    row_bytes = width * dtype_spec.bytes_per_pixel

    if row_bytes <= 0 or file_size % row_bytes != 0:
        raise GammaInputError(
            f"{raster_path.name}文件大小{file_size}不能被"
            f"单行字节数{row_bytes}整除；"
            f"width={width}, dtype={dtype_spec.name}"
        )

    inferred_length = file_size // row_bytes

    if length is None:
        length = inferred_length
    elif length != inferred_length:
        raise GammaInputError(
            f"{raster_path.name}尺寸不一致："
            f"参数或显式行数为{length}，"
            f"根据文件大小推导为{inferred_length}；"
            f"width={width}, dtype={dtype_spec.name}"
        )

    return GammaRasterLayout(
        path=raster_path,
        width=int(width),
        length=int(length),
        dtype=dtype_spec,
    )


def _validate_window(
    layout: GammaRasterLayout,
    *,
    x0: int,
    y0: int,
    nx: int | None,
    ny: int | None,
) -> tuple[int, int]:
    if x0 < 0 or x0 >= layout.width:
        raise GammaInputError(
            f"x0={x0}超出栅格宽度{layout.width}"
        )

    if y0 < 0 or y0 >= layout.length:
        raise GammaInputError(
            f"y0={y0}超出栅格行数{layout.length}"
        )

    if nx is None:
        nx = layout.width - x0

    if ny is None:
        ny = layout.length - y0

    if nx <= 0 or x0 + nx > layout.width:
        raise GammaInputError(
            f"读取列范围{x0}:{x0 + nx}超出宽度"
            f"{layout.width}"
        )

    if ny <= 0 or y0 + ny > layout.length:
        raise GammaInputError(
            f"读取行范围{y0}:{y0 + ny}超出行数"
            f"{layout.length}"
        )

    return int(nx), int(ny)


def _open_storage_memmap(
    layout: GammaRasterLayout,
) -> np.memmap:
    if layout.dtype.short_complex:
        shape = (
            layout.length,
            layout.width,
            2,
        )
    else:
        shape = (
            layout.length,
            layout.width,
        )

    return np.memmap(
        layout.path,
        dtype=layout.dtype.storage_dtype,
        mode="r",
        shape=shape,
        order="C",
    )


def _convert_storage_array(
    values: np.ndarray,
    dtype_spec: GammaDtypeSpec,
) -> np.ndarray:
    if dtype_spec.short_complex:
        real = np.asarray(
            values[..., 0],
            dtype=np.float32,
        )
        imag = np.asarray(
            values[..., 1],
            dtype=np.float32,
        )

        return (
            real + 1j * imag
        ).astype(np.complex64, copy=False)

    return np.asarray(
        values,
    ).astype(
        dtype_spec.output_dtype,
        copy=True,
    )


def read_gamma_raster(
    path: str | Path,
    *,
    parameter_file: str | Path | None = None,
    width: int | None = None,
    length: int | None = None,
    dtype: str | None = None,
    x0: int = 0,
    y0: int = 0,
    nx: int | None = None,
    ny: int | None = None,
) -> np.ndarray:
    """Read all or part of a GAMMA binary raster."""

    layout = resolve_gamma_raster_layout(
        path,
        parameter_file=parameter_file,
        width=width,
        length=length,
        dtype=dtype,
    )

    nx, ny = _validate_window(
        layout,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
    )

    mmap = _open_storage_memmap(layout)

    if layout.dtype.short_complex:
        storage = mmap[
            y0:y0 + ny,
            x0:x0 + nx,
            :,
        ]
    else:
        storage = mmap[
            y0:y0 + ny,
            x0:x0 + nx,
        ]

    result = _convert_storage_array(
        storage,
        layout.dtype,
    )

    del mmap

    return result


def iter_gamma_raster_blocks(
    path: str | Path,
    *,
    parameter_file: str | Path | None = None,
    width: int | None = None,
    length: int | None = None,
    dtype: str | None = None,
    block_rows: int = 256,
    row_start: int = 0,
    row_stop: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Iterate over a GAMMA raster without loading the full image.

    Yields:
        row_start, row_stop, native-endian NumPy block
    """

    if block_rows <= 0:
        raise GammaInputError(
            "block_rows必须大于0"
        )

    layout = resolve_gamma_raster_layout(
        path,
        parameter_file=parameter_file,
        width=width,
        length=length,
        dtype=dtype,
    )

    if row_stop is None:
        row_stop = layout.length

    if (
        row_start < 0
        or row_stop <= row_start
        or row_stop > layout.length
    ):
        raise GammaInputError(
            f"无效行范围：{row_start}:{row_stop}，"
            f"栅格行数为{layout.length}"
        )

    mmap = _open_storage_memmap(layout)

    try:
        for block_start in range(
            row_start,
            row_stop,
            block_rows,
        ):
            block_stop = min(
                block_start + block_rows,
                row_stop,
            )

            if layout.dtype.short_complex:
                storage = mmap[
                    block_start:block_stop,
                    :,
                    :,
                ]
            else:
                storage = mmap[
                    block_start:block_stop,
                    :,
                ]

            block = _convert_storage_array(
                storage,
                layout.dtype,
            )

            yield block_start, block_stop, block
    finally:
        del mmap


def sample_gamma_raster(
    path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    parameter_file: str | Path | None = None,
    width: int | None = None,
    length: int | None = None,
    dtype: str | None = None,
) -> np.ndarray:
    """Read selected zero-based pixel coordinates from a GAMMA raster."""

    layout = resolve_gamma_raster_layout(
        path,
        parameter_file=parameter_file,
        width=width,
        length=length,
        dtype=dtype,
    )

    row_array = np.asarray(
        rows,
        dtype=np.int64,
    ).reshape(-1)

    col_array = np.asarray(
        cols,
        dtype=np.int64,
    ).reshape(-1)

    if row_array.shape != col_array.shape:
        raise GammaInputError(
            "rows和cols长度不一致"
        )

    if row_array.size == 0:
        return np.empty(
            0,
            dtype=layout.dtype.output_dtype,
        )

    if (
        np.any(row_array < 0)
        or np.any(row_array >= layout.length)
    ):
        raise GammaInputError(
            "候选点行索引超出GAMMA栅格范围"
        )

    if (
        np.any(col_array < 0)
        or np.any(col_array >= layout.width)
    ):
        raise GammaInputError(
            "候选点列索引超出GAMMA栅格范围"
        )

    mmap = _open_storage_memmap(layout)

    if layout.dtype.short_complex:
        storage = mmap[
            row_array,
            col_array,
            :,
        ]
    else:
        storage = mmap[
            row_array,
            col_array,
        ]

    result = _convert_storage_array(
        storage,
        layout.dtype,
    )

    del mmap

    return result


def normalize_complex_unit(
    values: np.ndarray,
) -> np.ndarray:
    """Convert complex values to unit magnitude and preserve invalids as zero."""

    complex_values = np.asarray(
        values,
        dtype=np.complex64,
    )

    magnitude = np.abs(complex_values)

    valid = (
        np.isfinite(complex_values.real)
        & np.isfinite(complex_values.imag)
        & np.isfinite(magnitude)
        & (magnitude > 0)
    )

    result = np.zeros(
        complex_values.shape,
        dtype=np.complex64,
    )

    result[valid] = (
        complex_values[valid]
        / magnitude[valid]
    )

    return result


def infer_coherence_dtype(
    path: str | Path,
    *,
    width: int,
    length: int,
) -> str:
    """
    Infer whether a GAMMA coherence file is FLOAT or BYTE.

    Returns:
        "float" or "uchar"
    """

    coherence_path = Path(path).expanduser().resolve()

    if not coherence_path.is_file():
        raise GammaInputError(
            f"相干性文件不存在：{coherence_path}"
        )

    pixel_count = width * length
    file_size = coherence_path.stat().st_size

    if file_size == pixel_count * 4:
        return "float"

    if file_size == pixel_count:
        return "uchar"

    raise GammaInputError(
        f"无法判断相干性文件类型：{coherence_path.name}；"
        f"文件大小={file_size}，"
        f"FLOAT应为{pixel_count * 4}，"
        f"BYTE应为{pixel_count}"
    )


def read_gamma_coherence(
    path: str | Path,
    *,
    width: int,
    length: int,
    dtype: str | None = None,
    x0: int = 0,
    y0: int = 0,
    nx: int | None = None,
    ny: int | None = None,
) -> np.ndarray:
    """Read GAMMA FLOAT or BYTE coherence and return float32 values."""

    resolved_dtype = (
        dtype
        if dtype is not None
        else infer_coherence_dtype(
            path,
            width=width,
            length=length,
        )
    )

    coherence = read_gamma_raster(
        path,
        width=width,
        length=length,
        dtype=resolved_dtype,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
    )

    coherence = np.asarray(
        coherence,
        dtype=np.float32,
    )

    if resolved_dtype.lower() in {
        "uchar",
        "byte",
    }:
        coherence /= 255.0

    return coherence
