from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
import math
import os
from typing import Any, Callable


def logical_cpu_count() -> int:
    return max(
        1,
        int(
            os.cpu_count()
            or 1
        ),
    )


def available_memory_bytes() -> int:

    try:
        text = open(
            "/proc/meminfo",
            "r",
            encoding="utf-8",
        ).read()

        for line in text.splitlines():

            if line.startswith(
                "MemAvailable:"
            ):
                return (
                    int(
                        line.split()[1]
                    )
                    * 1024
                )

    except Exception:
        pass

    return 0


def _env_float(
    name: str,
    default: float,
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
            raise RuntimeError(
                f"{name} must be numeric, "
                f"got {raw!r}"
            ) from exc

    return min(
        maximum,
        max(
            minimum,
            value,
        ),
    )


def workers_from_env(
    name: str,
    *,
    task_count: int,
    cpu_fraction: float = 0.95,
    cap: int | None = None,
) -> int:

    task_count = max(
        1,
        int(
            task_count
        ),
    )

    cpu = logical_cpu_count()

    auto_workers = max(
        1,
        int(
            math.floor(
                cpu
                * cpu_fraction
            )
        ),
    )

    auto_workers = min(
        auto_workers,
        task_count,
    )

    if (
        cap is not None
        and cap > 0
    ):
        auto_workers = min(
            auto_workers,
            int(cap),
        )

    raw = os.environ.get(
        name,
        "auto",
    ).strip().lower()

    if raw in {
        "",
        "auto",
        "0",
    }:
        return auto_workers

    try:
        requested = int(
            raw
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be 'auto' "
            f"or integer, got {raw!r}"
        ) from exc

    if requested < 1:
        raise RuntimeError(
            f"{name} must be >= 1"
        )

    return min(
        requested,
        task_count,
        int(cap)
        if cap is not None
        and cap > 0
        else requested,
    )


def edge_chunk() -> int:
    """
    Larger chunks reduce Python dispatch overhead in
    Stage6 edge-space computations without changing
    the numerical algorithm.
    """

    raw = os.environ.get(
        "PYSTAMPS_SBAS_EDGE_CHUNK"
    )

    if raw not in {
        None,
        "",
        "auto",
        "0",
    }:
        return max(
            64,
            int(raw),
        )

    mem = (
        available_memory_bytes()
    )

    if mem >= 96 * 1024**3:
        return 8192

    if mem >= 48 * 1024**3:
        return 4096

    if mem >= 24 * 1024**3:
        return 2048

    return 1024


def invert_chunk(
    *,
    n_ps: int,
    n_ifg: int,
    n_image: int,
) -> int:
    """
    Auto-select a large GLS row chunk.

    Old default = 2048 rows.

    Larger BLAS matrices give much better GEMM efficiency.
    The memory model is intentionally conservative.
    """

    raw = os.environ.get(
        "PYSTAMPS_STAGE6_SB_INVERT_CHUNK"
    )

    if raw not in {
        None,
        "",
        "auto",
        "0",
    }:
        return max(
            512,
            min(
                int(raw),
                int(n_ps),
            ),
        )

    mem = (
        available_memory_bytes()
    )

    if mem <= 0:
        return min(
            int(n_ps),
            8192,
        )

    # Temporary float64 matrices approximately:
    #
    # Y               chunk x n_ifg
    # predicted IFG   chunk x n_ifg
    # X               chunk x n_image
    #
    bytes_per_row = (
        8
        * (
            2
            * max(
                1,
                int(n_ifg),
            )
            + max(
                1,
                int(n_image),
            )
        )
    )

    # Allow about 8% of currently available RAM
    # for an inversion chunk.
    budget = max(
        512 * 1024**2,
        int(
            mem * 0.08
        ),
    )

    candidate = max(
        2048,
        int(
            budget
            // max(
                1,
                bytes_per_row,
            )
        ),
    )

    # Huge chunks provide diminishing BLAS gains.
    candidate = min(
        candidate,
        32768,
        int(n_ps),
    )

    # Round down for stable allocation sizes.
    quantum = 1024

    candidate = max(
        2048,
        (
            candidate
            // quantum
        )
        * quantum,
    )

    return min(
        candidate,
        int(n_ps),
    )


def blas_threads() -> int:

    raw = os.environ.get(
        "PYSTAMPS_STAGE6_BLAS_THREADS",
        "auto",
    ).strip().lower()

    cpu = logical_cpu_count()

    if raw in {
        "",
        "auto",
        "0",
    }:

        return max(
            1,
            min(
                cpu,
                int(
                    math.ceil(
                        cpu * 0.95
                    )
                ),
            ),
        )

    return max(
        1,
        min(
            int(raw),
            cpu,
        ),
    )


def blas_context():

    threads = blas_threads()

    try:
        from threadpoolctl import (
            threadpool_limits,
        )

        return threadpool_limits(
            limits=threads,
            user_api="blas",
        )

    except Exception:
        return nullcontext()


def turbo_blas_stage(
    func: Callable[..., Any],
):

    @wraps(func)
    def wrapper(
        *args,
        **kwargs,
    ):

        n = blas_threads()

        print(
            "[STAGE6_SBAS][TURBO] "
            f"BLAS threads={n} "
            f"for {func.__name__}",
            flush=True,
        )

        with blas_context():
            return func(
                *args,
                **kwargs,
            )

    return wrapper


def final_qc_chunk(
    *,
    n_ps: int,
    requested: int = 8,
) -> int:
    """
    Vectorised FINAL-QC chunk.

    Each column may temporarily require several
    N_PS float64 arrays. Keep enough RAM margin
    while making each NumPy operation large.
    """

    raw = os.environ.get(
        "PYSTAMPS_STAGE6_FINAL_QC_CHUNK"
    )

    if raw not in {
        None,
        "",
        "auto",
        "0",
    }:
        return max(
            1,
            min(
                128,
                int(raw),
            ),
        )

    mem = available_memory_bytes()

    # residual + abs + masks/workspace
    estimated_per_column = (
        max(
            1,
            int(n_ps),
        )
        * 8
        * 5
    )

    if mem <= 0:
        auto = 16
    else:

        budget = max(
            256 * 1024**2,
            int(
                mem
                * 0.04
            ),
        )

        auto = int(
            budget
            // max(
                1,
                estimated_per_column,
            )
        )

    auto = max(
        int(requested),
        auto,
        8,
    )

    return min(
        auto,
        64,
    )


def resource_summary() -> dict[str, Any]:

    return {
        "logical_cpu":
            logical_cpu_count(),

        "available_memory_gib":
            available_memory_bytes()
            / 1024**3,

        "edge_chunk":
            edge_chunk(),

        "blas_threads":
            blas_threads(),
    }
