cd /mnt/h/pystamps-main

cat > apply_stage3_fast_patch.sh <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/h/pystamps-main"
PORTED="$ROOT/pystamps/pipeline/ported.py"
MATIO="$ROOT/pystamps/io/mat.py"
TEST_FILE="$ROOT/tests/test_stage3_fast_path.py"
VALIDATOR="$ROOT/validate_stage3_fast.py"

cd "$ROOT"

if pgrep -af 'pystamps run.*--start-step[ =]3' >/dev/null 2>&1; then
    echo "检测到Stage 3仍在运行。"
    echo "当前运行进程："
    pgrep -af 'pystamps run.*--start-step[ =]3' || true
    echo
    echo "请先让当前Stage 3完成。"
    echo "确实需要强制修改时使用："
    echo "FORCE_STAGE3_PATCH=1 ./apply_stage3_fast_patch.sh"
    if [[ "${FORCE_STAGE3_PATCH:-0}" != "1" ]]; then
        exit 3
    fi
fi

for file in "$PORTED" "$MATIO"; do
    if [[ ! -f "$file" ]]; then
        echo "缺少文件：$file" >&2
        exit 2
    fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.stage3_fast_backup/$TIMESTAMP"

mkdir -p "$BACKUP/pystamps/pipeline"
mkdir -p "$BACKUP/pystamps/io"
mkdir -p "$BACKUP/tests"

cp -a "$PORTED" "$BACKUP/pystamps/pipeline/ported.py"
cp -a "$MATIO" "$BACKUP/pystamps/io/mat.py"

if [[ -f "$TEST_FILE" ]]; then
    cp -a "$TEST_FILE" "$BACKUP/tests/test_stage3_fast_path.py"
fi

echo "$BACKUP" > "$ROOT/.stage3_fast_last_backup"

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import textwrap


root = Path("/mnt/h/pystamps-main")
mat_path = root / "pystamps/io/mat.py"
ported_path = root / "pystamps/pipeline/ported.py"


# ----------------------------------------------------------------------
# 1. MAT选择性读取
# ----------------------------------------------------------------------

mat_text = mat_path.read_text(encoding="utf-8")

mat_marker = "# === STAGE3_FAST_SELECTIVE_MAT_V1 ==="

mat_code = r'''
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
'''

if mat_marker not in mat_text:
    mat_text = mat_text.rstrip() + "\n\n" + textwrap.dedent(mat_code).lstrip()

mat_path.write_text(
    mat_text,
    encoding="utf-8",
)


# ----------------------------------------------------------------------
# 2. ported.py导入及Stage 3快速路径
# ----------------------------------------------------------------------

ported = ported_path.read_text(
    encoding="utf-8",
)

fast_marker = "# === STAGE3_FAST_PATCH_V1 ==="

if fast_marker in ported:
    print("Stage 3快速补丁已经存在，跳过ported.py重复修改。")

else:
    if "import threading\n" not in ported:
        if "import time\n" in ported:
            ported = ported.replace(
                "import time\n",
                "import time\nimport threading\n",
                1,
            )
        else:
            ported = (
                "import threading\n"
                + ported
            )

    if "from scipy import fft as scipy_fft\n" not in ported:
        if "from scipy import sparse, spatial\n" in ported:
            ported = ported.replace(
                "from scipy import sparse, spatial\n",
                "from scipy import sparse, spatial\n"
                "from scipy import fft as scipy_fft\n",
                1,
            )
        else:
            ported = ported.replace(
                "import numpy as np\n",
                "import numpy as np\n"
                "from scipy import fft as scipy_fft\n",
                1,
            )

    old_mat_import = (
        "from pystamps.io.mat import "
        "read_mat, write_mat"
    )

    new_mat_import = (
        "from pystamps.io.mat import "
        "read_mat, read_mat_variables, write_mat"
    )

    if old_mat_import in ported:
        ported = ported.replace(
            old_mat_import,
            new_mat_import,
            1,
        )

    elif "read_mat_variables" not in ported:
        raise RuntimeError(
            "无法定位pystamps.io.mat导入语句"
        )

    original_signature = (
        "def stage3_select_ps("
        "patch_dir: Path, "
        "backend: str = \"auto\""
        ") -> str:"
    )

    legacy_signature = (
        "def _stage3_select_ps_legacy("
        "patch_dir: Path, "
        "backend: str = \"auto\""
        ") -> str:"
    )

    if original_signature not in ported:
        raise RuntimeError(
            "无法定位原stage3_select_ps函数签名"
        )

    ported = ported.replace(
        original_signature,
        legacy_signature,
        1,
    )

    insert_marker = "\ndef _stage4_checkpoint(\n"

    if insert_marker not in ported:
        raise RuntimeError(
            "无法定位Stage 3与Stage 4之间的插入位置"
        )

    fast_code = r'''
# === STAGE3_FAST_PATCH_V1 ===

def _stage3_environment_flag(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.environ.get(name)

    if raw is None:
        return bool(default)

    value = raw.strip().lower()

    if value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise PortedStageError(
        f"{name}必须是0/1、true/false、yes/no或on/off"
    )


def _stage3_environment_positive_int(
    name: str,
    default: int,
) -> int:
    raw = os.environ.get(name)

    if raw is None:
        return max(
            1,
            int(default),
        )

    try:
        value = int(raw)

    except ValueError as exc:
        raise PortedStageError(
            f"{name}必须是正整数"
        ) from exc

    if value <= 0:
        raise PortedStageError(
            f"{name}必须大于0"
        )

    return value


def _stage3_clap_patch_stack(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    *,
    single_precision: bool,
) -> np.ndarray:
    """
    Batched equivalent of calling _clap_filt_patch once per IFG.

    Input shape:
        [window_y, window_x, interferogram]

    FFT, Gaussian smoothing, median normalisation and IFFT are applied
    to the complete interferogram stack in one operation.
    """

    complex_dtype = (
        np.complex64
        if single_precision
        else np.complex128
    )

    real_dtype = (
        np.float32
        if single_precision
        else np.float64
    )

    ph = np.asarray(
        ph_stack,
        dtype=complex_dtype,
    ).copy()

    if ph.ndim != 3:
        raise PortedStageError(
            "Stage 3 batched CLAP requires a 3-D stack"
        )

    ph[
        np.isnan(ph)
    ] = 0

    low = np.asarray(
        low_pass,
        dtype=real_dtype,
    )

    if low.shape != ph.shape[:2]:
        raise PortedStageError(
            "Stage 3 low-pass shape does not match CLAP window"
        )

    phase_fft = scipy_fft.fft2(
        ph,
        axes=(
            0,
            1,
        ),
        workers=1,
    )

    amplitude = np.abs(
        phase_fft
    ).astype(
        real_dtype,
        copy=False,
    )

    shifted = scipy_fft.fftshift(
        amplitude,
        axes=(
            0,
            1,
        ),
    )

    gaussian = np.asarray(
        _gausswin(7),
        dtype=real_dtype,
    )

    smooth_first = ndimage.convolve1d(
        shifted,
        gaussian,
        axis=0,
        mode="constant",
        cval=0.0,
    )

    smooth_second = ndimage.convolve1d(
        smooth_first,
        gaussian,
        axis=1,
        mode="constant",
        cval=0.0,
    )

    response = scipy_fft.ifftshift(
        smooth_second,
        axes=(
            0,
            1,
        ),
    )

    median_response = np.median(
        response,
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )

    np.divide(
        response,
        median_response,
        out=response,
        where=median_response != 0,
    )

    if float(alpha) != 1.0:
        np.power(
            response,
            float(alpha),
            out=response,
        )

    response -= 1.0

    np.maximum(
        response,
        0.0,
        out=response,
    )

    response *= float(
        beta
    )

    response += low[
        :,
        :,
        None,
    ]

    filtered = scipy_fft.ifft2(
        phase_fft * response,
        axes=(
            0,
            1,
        ),
        workers=1,
    )

    return filtered.astype(
        complex_dtype,
        copy=False,
    )


def _stage3_selected_grid_clap(
    *,
    ph_grid: np.ndarray,
    selected_grid_ij: np.ndarray,
    n_win: int,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    slc_osf: int,
    workers: int,
    single_precision: bool,
    show_progress: bool,
) -> np.ndarray:
    """
    Re-estimate local filtered phase once per unique grid position.

    PS candidates sharing grid_ij use the same ph_grid neighbourhood and
    therefore have exactly the same local CLAP result in the legacy code.
    """

    selected_grid = np.asarray(
        selected_grid_ij,
        dtype=np.int64,
    )

    if selected_grid.ndim != 2 or selected_grid.shape[1] != 2:
        raise PortedStageError(
            "Stage 3 selected grid coordinates must be [n, 2]"
        )

    if selected_grid.shape[0] == 0:
        return np.empty(
            (
                0,
                ph_grid.shape[2],
            ),
            dtype=(
                np.complex64
                if single_precision
                else np.complex128
            ),
        )

    unique_grid, inverse = np.unique(
        selected_grid,
        axis=0,
        return_inverse=True,
    )

    n_unique = int(
        unique_grid.shape[0]
    )

    n_ifg = int(
        ph_grid.shape[2]
    )

    complex_dtype = (
        np.complex64
        if single_precision
        else np.complex128
    )

    filtered_unique = np.zeros(
        (
            n_unique,
            n_ifg,
        ),
        dtype=complex_dtype,
    )

    thread_count = min(
        max(
            1,
            int(workers),
        ),
        n_unique,
        os.cpu_count() or 1,
    )

    n_i = int(
        ph_grid.shape[0]
    )

    n_j = int(
        ph_grid.shape[1]
    )

    half_win = int(
        n_win // 2
    )

    radius = max(
        0,
        int(slc_osf) - 1,
    )

    progress_lock = threading.Lock()

    progress = {
        "done": 0,
        "next": 0.05,
    }

    started = time.perf_counter()

    def _report_one() -> None:
        if not show_progress:
            return

        with progress_lock:
            progress[
                "done"
            ] += 1

            fraction = (
                progress["done"]
                / n_unique
            )

            if (
                fraction
                < progress["next"]
                and progress["done"] < n_unique
            ):
                return

            elapsed = (
                time.perf_counter()
                - started
            )

            rate = (
                progress["done"]
                / elapsed
                if elapsed > 0
                else 0.0
            )

            eta = (
                (
                    n_unique
                    - progress["done"]
                )
                / rate
                if rate > 0
                else float("nan")
            )

            print(
                "[STAGE3_FAST] "
                f"unique_grid="
                f"{progress['done']}/"
                f"{n_unique} "
                f"({100.0 * fraction:.1f}%), "
                f"elapsed={elapsed:.1f}s, "
                f"eta={eta:.1f}s",
                flush=True,
            )

            while (
                progress["next"]
                <= fraction
            ):
                progress[
                    "next"
                ] += 0.05

    def _compute_one(
        unique_index: int,
    ) -> None:
        ps_ij_i = int(
            unique_grid[
                unique_index,
                0,
            ]
        )

        ps_ij_j = int(
            unique_grid[
                unique_index,
                1,
            ]
        )

        i_min = max(
            ps_ij_i - half_win,
            1,
        )

        i_max = (
            i_min
            + n_win
            - 1
        )

        if i_max > n_i:
            i_min = (
                i_min
                - i_max
                + n_i
            )

            i_max = n_i

        j_min = max(
            ps_ij_j - half_win,
            1,
        )

        j_max = (
            j_min
            + n_win
            - 1
        )

        if j_max > n_j:
            j_min = (
                j_min
                - j_max
                + n_j
            )

            j_max = n_j

        if i_min < 1 or j_min < 1:
            _report_one()
            return

        ps_bit_i = (
            ps_ij_i
            - i_min
            + 1
        )

        ps_bit_j = (
            ps_ij_j
            - j_min
            + 1
        )

        phase_window = ph_grid[
            i_min - 1:
            i_max,
            j_min - 1:
            j_max,
            :,
        ].astype(
            complex_dtype,
            copy=True,
        )

        if phase_window.shape[:2] != (
            n_win,
            n_win,
        ):
            _report_one()
            return

        phase_window[
            ps_bit_i - 1,
            ps_bit_j - 1,
            :,
        ] = 0

        ii = np.arange(
            ps_bit_i - radius,
            ps_bit_i + radius + 1,
            dtype=np.int64,
        )

        ii = ii[
            (
                ii > 0
            )
            & (
                ii
                <= phase_window.shape[0]
            )
        ] - 1

        jj = np.arange(
            ps_bit_j - radius,
            ps_bit_j + radius + 1,
            dtype=np.int64,
        )

        jj = jj[
            (
                jj > 0
            )
            & (
                jj
                <= phase_window.shape[1]
            )
        ] - 1

        # Preserve the existing implementation exactly: the SLC
        # oversampling neighbourhood is zeroed only in IFG index 0.
        if ii.size and jj.size:
            phase_window[
                np.ix_(
                    ii,
                    jj,
                    np.asarray(
                        [0],
                        dtype=np.int64,
                    ),
                )
            ] = 0

        filtered = _stage3_clap_patch_stack(
            phase_window,
            alpha,
            beta,
            low_pass,
            single_precision=single_precision,
        )

        filtered_unique[
            unique_index,
            :,
        ] = filtered[
            ps_bit_i - 1,
            ps_bit_j - 1,
            :,
        ]

        _report_one()

    chunks = [
        chunk
        for chunk in np.array_split(
            np.arange(
                n_unique,
                dtype=np.int64,
            ),
            thread_count,
        )
        if chunk.size
    ]

    def _compute_chunk(
        chunk: np.ndarray,
    ) -> None:
        for unique_index in chunk:
            _compute_one(
                int(unique_index)
            )

    if show_progress:
        print(
            "[STAGE3_FAST] "
            f"selected_ps="
            f"{selected_grid.shape[0]}, "
            f"unique_grid="
            f"{n_unique}, "
            f"grid_reduction="
            f"{selected_grid.shape[0] / max(1, n_unique):.2f}x, "
            f"threads="
            f"{thread_count}, "
            f"precision="
            f"{np.dtype(complex_dtype).name}",
            flush=True,
        )

    if len(chunks) == 1:
        _compute_chunk(
            chunks[0]
        )

    else:
        with ThreadPoolExecutor(
            max_workers=len(chunks)
        ) as executor:
            futures = [
                executor.submit(
                    _compute_chunk,
                    chunk,
                )
                for chunk in chunks
            ]

            for future in futures:
                future.result()

    return filtered_unique[
        inverse,
        :,
    ]


def _stage3_select_ps_fast(
    patch_dir: Path,
    backend: str = "auto",
) -> str:
    stage3_started = time.perf_counter()

    show_progress = _stage3_environment_flag(
        "PYSTAMPS_STAGE3_PROGRESS",
        default=True,
    )

    single_precision = _stage3_environment_flag(
        "PYSTAMPS_STAGE3_SINGLE_PRECISION",
        default=True,
    )

    requested_threads = _stage3_environment_positive_int(
        "PYSTAMPS_STAGE3_THREADS",
        default=max(
            1,
            min(
                8,
                os.cpu_count() or 1,
            ),
        ),
    )

    pm_path = (
        patch_dir
        / "pm1.mat"
    )

    pm_meta = read_mat_variables(
        pm_path,
        (
            "coh_ps",
            "coh_bins",
            "Nr",
            "K_ps",
            "C_ps",
            "grid_ij",
            "n_trial_wraps",
            "low_pass",
        ),
    )

    ps = read_mat(
        patch_dir
        / "ps1.mat"
    )

    parms = _load_parms(
        patch_dir
    )

    debug_payload: dict[str, Any] = {
        "patch": patch_dir.name,
        "fast_path": True,
        "reestimate_used": False,
        "reestimate_status": "not_attempted",
        "reestimate_exception": None,
        "stage3_threads": int(
            requested_threads
        ),
        "single_precision": bool(
            single_precision
        ),
    }

    n_ps = int(
        round(
            _mat_scalar(
                ps.get(
                    "n_ps",
                    0,
                ),
                0,
            )
        )
    )

    if n_ps <= 0:
        raise PortedStageError(
            "ps1.mat missing valid n_ps"
        )

    coh_ps = _as_ps_vector(
        pm_meta.get(
            "coh_ps"
        ),
        n_ps,
        "pm1.coh_ps",
    ).astype(
        np.float64,
        copy=False,
    )

    if coh_ps.size == 0:
        raise PortedStageError(
            "pm1.mat has empty coh_ps"
        )

    coh_bins = np.asarray(
        pm_meta.get(
            "coh_bins",
            np.asarray(
                [],
                dtype=np.float64,
            ),
        ),
        dtype=np.float64,
    ).reshape(-1)

    Nr_dist = np.asarray(
        pm_meta.get(
            "Nr",
            np.asarray(
                [],
                dtype=np.float64,
            ),
        ),
        dtype=np.float64,
    ).reshape(-1)

    if coh_bins.size == 0:
        coh_bins = np.arange(
            0.005,
            1.0,
            0.01,
            dtype=np.float64,
        )

    if Nr_dist.size == 0:
        Nr_dist = np.ones(
            coh_bins.size,
            dtype=np.float64,
        )

    da_file = (
        patch_dir
        / "da1.mat"
    )

    if da_file.exists():
        D_A = np.asarray(
            read_mat(
                da_file
            ).get(
                "D_A"
            ),
            dtype=np.float64,
        ).reshape(-1)

    else:
        D_A = np.ones_like(
            coh_ps,
            dtype=np.float64,
        )

    if D_A.size >= 10000:
        D_A_sort = np.sort(
            D_A
        )

        bin_size = (
            10000
            if D_A.size >= 50000
            else 2000
        )

        D_A_max = np.concatenate(
            (
                [0.0],
                D_A_sort[
                    bin_size:
                    D_A.size - bin_size:
                    bin_size
                ],
                [
                    D_A_sort[
                        -1
                    ]
                ],
            )
        )

    else:
        D_A_max = np.asarray(
            [
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        D_A = np.ones_like(
            coh_ps,
            dtype=np.float64,
        )

    low_coh_thresh = (
        15
        if parms.small_baseline_flag.lower() == "y"
        else 31
    )

    if parms.select_method.upper() == "PERCENT":
        max_percent_rand = float(
            parms.percent_rand
        )

    else:
        xy = _as_ps_dim(
            ps.get(
                "xy"
            ),
            n_ps,
            3,
            "ps1.xy",
        ).astype(
            np.float64
        )

        if xy.size == 0:
            patch_area = 1.0

        else:
            patch_area = (
                np.prod(
                    np.max(
                        xy[
                            :,
                            1:3,
                        ],
                        axis=0,
                    )
                    - np.min(
                        xy[
                            :,
                            1:3,
                        ],
                        axis=0,
                    )
                )
                / 1e6
            )

            if patch_area <= 0:
                patch_area = 1.0

        max_percent_rand = (
            float(
                parms.density_rand
            )
            * patch_area
            / max(
                1,
                D_A_max.size - 1,
            )
        )

    (
        coh_thresh_all,
        coh_thresh_coeffs,
    ) = _coh_threshold_from_dist(
        coh_values=coh_ps,
        D_A=D_A,
        D_A_max=D_A_max,
        coh_bins=coh_bins,
        Nr_dist=Nr_dist,
        low_coh_thresh=low_coh_thresh,
        max_percent_rand=max_percent_rand,
        select_method=parms.select_method,
    )

    debug_payload[
        "initial_coh_thresh_coeffs"
    ] = np.asarray(
        coh_thresh_coeffs,
        dtype=np.float64,
    ).reshape(-1).tolist()

    ix = (
        np.where(
            coh_ps
            > coh_thresh_all
        )[
            0
        ]
        + 1
    )

    ix0 = (
        ix
        - 1
    )

    ifg_index = _ifg_index_for_selection(
        ps,
        parms,
    )

    ifg_index_ix = (
        np.asarray(
            ifg_index,
            dtype=np.int64,
        ).reshape(-1)
        - 1
    )

    pm_large = read_mat_variables(
        pm_path,
        (
            "ph_patch",
            "ph_res",
            "ph_grid",
        ),
    )

    ph_patch = _as_ps_ifg_complex(
        pm_large.get(
            "ph_patch"
        ),
        n_ps,
        "pm1.ph_patch",
    ).astype(
        np.complex64,
        copy=False,
    )

    ph_res = _as_ps_matrix(
        pm_large.get(
            "ph_res"
        ),
        n_ps,
        "pm1.ph_res",
    ).astype(
        np.float32,
        copy=False,
    )

    K_ps = _as_ps_vector(
        pm_meta.get(
            "K_ps"
        ),
        n_ps,
        "pm1.K_ps",
    ).astype(
        np.float64,
        copy=False,
    )

    C_ps = _as_ps_vector(
        pm_meta.get(
            "C_ps"
        ),
        n_ps,
        "pm1.C_ps",
    ).astype(
        np.float64,
        copy=False,
    )

    if (
        parms.gamma_stdev_reject > 0
        and ix.size > 0
        and ifg_index_ix.size > 0
    ):
        ph_res_cpx = np.exp(
            1j
            * ph_res[
                :,
                ifg_index_ix,
            ]
        )

        coh_std = np.zeros(
            ix.size,
            dtype=np.float64,
        )

        rng = np.random.default_rng(
            0
        )

        for row_i, ps_i in enumerate(
            ix0
        ):
            sample = ph_res_cpx[
                ps_i,
                :,
            ]

            n_sample = sample.size

            if n_sample == 0:
                coh_std[
                    row_i
                ] = np.inf

                continue

            draw_ix = rng.integers(
                0,
                n_sample,
                size=(
                    100,
                    n_sample,
                ),
            )

            boot = sample[
                draw_ix
            ]

            coh_boot = (
                np.abs(
                    np.sum(
                        boot,
                        axis=1,
                    )
                )
                / float(
                    n_sample
                )
            )

            coh_std[
                row_i
            ] = float(
                np.std(
                    coh_boot
                )
            )

        ix_mask_reject = (
            coh_std
            < float(
                parms.gamma_stdev_reject
            )
        )

        ix = ix[
            ix_mask_reject
        ]

        ix0 = (
            ix
            - 1
        )

    ph_patch2 = ph_patch[
        ix0,
        :,
    ].astype(
        np.complex64,
        copy=True,
    )

    ph_res2 = ph_res[
        ix0,
        :,
    ].astype(
        np.float32,
        copy=True,
    )

    K_ps2 = K_ps[
        ix0
    ].astype(
        np.float64,
        copy=True,
    )

    C_ps2 = C_ps[
        ix0
    ].astype(
        np.float64,
        copy=True,
    )

    coh_ps2 = coh_ps[
        ix0
    ].astype(
        np.float64,
        copy=True,
    )

    keep_ix = np.ones(
        ix.size,
        dtype=bool,
    )

    if ix.size > 0:
        reestimate_ok = True

        ph_grid_raw = pm_large.get(
            "ph_grid"
        )

        if ph_grid_raw is None:
            reestimate_ok = False

            ph_grid = np.empty(
                (
                    0,
                    0,
                    0,
                ),
                dtype=np.complex64,
            )

        else:
            ph_grid = _coerce_complex(
                ph_grid_raw
            )

            if (
                ph_grid.ndim != 3
                or ph_grid.shape[0] < 2
                or ph_grid.shape[1] < 2
            ):
                reestimate_ok = False

        try:
            grid_ij = _as_ps_dim(
                pm_meta.get(
                    "grid_ij"
                ),
                n_ps,
                2,
                "pm1.grid_ij",
            ).astype(
                np.int64
            )

            if grid_ij.size == 0:
                reestimate_ok = False

        except Exception:
            reestimate_ok = False

            grid_ij = np.empty(
                (
                    0,
                    2,
                ),
                dtype=np.int64,
            )

        bp1_file = (
            patch_dir
            / "bp1.mat"
        )

        if not bp1_file.exists():
            reestimate_ok = False

        if reestimate_ok:
            try:
                debug_payload[
                    "reestimate_status"
                ] = "running"

                ph_all = _as_ps_ifg_complex(
                    read_mat_variables(
                        patch_dir
                        / "ph1.mat",
                        (
                            "ph",
                        ),
                    ).get(
                        "ph"
                    ),
                    n_ps,
                    "ph1.ph",
                ).astype(
                    (
                        np.complex64
                        if single_precision
                        else np.complex128
                    ),
                    copy=False,
                )

                bperp_full = np.asarray(
                    ps.get(
                        "bperp"
                    ),
                    dtype=np.float64,
                ).reshape(-1)

                if parms.small_baseline_flag.lower() == "y":
                    ph_work = ph_all
                    bperp_work = bperp_full

                else:
                    master_ix = int(
                        round(
                            _mat_scalar(
                                ps.get(
                                    "master_ix",
                                    1,
                                ),
                                1,
                            )
                        )
                    )

                    no_master_ix = (
                        np.arange(
                            ph_all.shape[1]
                        )
                        != (
                            master_ix
                            - 1
                        )
                    )

                    ph_work = ph_all[
                        :,
                        no_master_ix,
                    ]

                    bperp_work = bperp_full[
                        no_master_ix
                    ]

                n_ifg_work = int(
                    ph_work.shape[1]
                )

                ifg_index_ix = ifg_index_ix[
                    (
                        ifg_index_ix
                        >= 0
                    )
                    & (
                        ifg_index_ix
                        < n_ifg_work
                    )
                ]

                if (
                    ifg_index_ix.size == 0
                    or ph_grid.shape[2] != n_ifg_work
                ):
                    reestimate_ok = False

                else:
                    options = _build_stage_options(
                        patch_dir
                    )

                    n_win = int(
                        round(
                            options.clap_win
                        )
                    )

                    if n_win <= 0:
                        n_win = 32

                    alpha = float(
                        options.clap_alpha
                    )

                    beta = float(
                        options.clap_beta
                    )

                    low_pass = np.asarray(
                        pm_meta.get(
                            "low_pass",
                            np.asarray(
                                [],
                                dtype=np.float64,
                            ),
                        ),
                        dtype=np.float64,
                    )

                    if low_pass.shape != (
                        n_win,
                        n_win,
                    ):
                        low_pass = _build_low_pass(
                            options
                        )

                    slc_osf = max(
                        1,
                        int(
                            round(
                                float(
                                    parms.slc_osf
                                )
                            )
                        ),
                    )

                    ph_patch2 = _stage3_selected_grid_clap(
                        ph_grid=ph_grid,
                        selected_grid_ij=grid_ij[
                            ix0,
                            :,
                        ],
                        n_win=n_win,
                        alpha=alpha,
                        beta=beta,
                        low_pass=low_pass,
                        slc_osf=slc_osf,
                        workers=requested_threads,
                        single_precision=single_precision,
                        show_progress=show_progress,
                    )

                    ph_res2 = np.zeros(
                        (
                            ix.size,
                            n_ifg_work,
                        ),
                        dtype=np.float32,
                    )

                    K_ps2 = np.full(
                        ix.size,
                        np.nan,
                        dtype=np.float64,
                    )

                    C_ps2 = np.zeros(
                        ix.size,
                        dtype=np.float64,
                    )

                    coh_ps2 = np.full(
                        ix.size,
                        np.nan,
                        dtype=np.float64,
                    )

                    psdph = (
                        ph_work[
                            ix0,
                            :,
                        ]
                        * np.conj(
                            ph_patch2
                        )
                    )

                    valid_rows = np.all(
                        psdph != 0,
                        axis=1,
                    )

                    valid_index = np.where(
                        valid_rows
                    )[
                        0
                    ]

                    if valid_index.size:
                        valid_phase = psdph[
                            valid_index,
                            :,
                        ]

                        valid_magnitude = np.abs(
                            valid_phase
                        )

                        valid_phase = np.divide(
                            valid_phase,
                            valid_magnitude,
                            out=np.zeros_like(
                                valid_phase
                            ),
                            where=valid_magnitude != 0,
                        )

                        fit_phase = valid_phase[
                            :,
                            ifg_index_ix,
                        ].astype(
                            np.complex64,
                            copy=False,
                        )

                        bperp_mat = _as_ps_matrix(
                            read_mat_variables(
                                bp1_file,
                                (
                                    "bperp_mat",
                                ),
                            ).get(
                                "bperp_mat"
                            ),
                            n_ps,
                            "bp1.bperp_mat",
                        ).astype(
                            np.float64,
                            copy=False,
                        )

                        fit_bperp = bperp_mat[
                            ix0[
                                valid_index
                            ],
                            :,
                        ][
                            :,
                            ifg_index_ix,
                        ]

                        n_trial_wraps = float(
                            _mat_scalar(
                                pm_meta.get(
                                    "n_trial_wraps",
                                    0.0,
                                ),
                                0.0,
                            )
                        )

                        try:
                            (
                                k_batch,
                                c_batch,
                                coh_batch,
                                residual_batch,
                            ) = run_stage2_topofit_kernel(
                                fit_phase,
                                fit_bperp,
                                n_trial_wraps,
                                backend=backend,
                                threads=requested_threads,
                            )

                            k_batch = np.asarray(
                                k_batch,
                                dtype=np.float64,
                            ).reshape(-1)

                            c_batch = np.asarray(
                                c_batch,
                                dtype=np.float64,
                            ).reshape(-1)

                            coh_batch = np.asarray(
                                coh_batch,
                                dtype=np.float64,
                            ).reshape(-1)

                            residual_batch = np.asarray(
                                residual_batch
                            )

                            K_ps2[
                                valid_index
                            ] = k_batch

                            C_ps2[
                                valid_index
                            ] = c_batch

                            coh_ps2[
                                valid_index
                            ] = coh_batch

                            ph_res2[
                                valid_index[
                                    :,
                                    None
                                ],
                                ifg_index_ix[
                                    None,
                                    :,
                                ],
                            ] = np.angle(
                                residual_batch
                            ).astype(
                                np.float32,
                                copy=False,
                            )

                        except Exception:
                            for local_valid, row_local in enumerate(
                                valid_index
                            ):
                                (
                                    k_opt,
                                    c_opt,
                                    coh_opt,
                                    phase_residual,
                                ) = _ps_topofit_single(
                                    fit_phase[
                                        local_valid,
                                        :,
                                    ],
                                    fit_bperp[
                                        local_valid,
                                        :,
                                    ],
                                    n_trial_wraps,
                                )

                                K_ps2[
                                    row_local
                                ] = k_opt

                                C_ps2[
                                    row_local
                                ] = c_opt

                                coh_ps2[
                                    row_local
                                ] = coh_opt

                                ph_res2[
                                    row_local,
                                    ifg_index_ix,
                                ] = np.angle(
                                    phase_residual
                                ).astype(
                                    np.float32,
                                    copy=False,
                                )

                    coh_for_threshold = coh_ps.copy()

                    coh_for_threshold[
                        ix0
                    ] = coh_ps2

                    (
                        coh_thresh_re_all,
                        coh_thresh_coeffs,
                    ) = _coh_threshold_from_dist(
                        coh_values=coh_for_threshold,
                        D_A=D_A,
                        D_A_max=D_A_max,
                        coh_bins=coh_bins,
                        Nr_dist=Nr_dist,
                        low_coh_thresh=low_coh_thresh,
                        max_percent_rand=max_percent_rand,
                        select_method=parms.select_method,
                    )

                    coh_thresh_sel = coh_thresh_re_all[
                        ix0
                    ]

                    coh_thresh_sel[
                        coh_thresh_sel
                        < 0
                    ] = 0

                    coh_thresh_all[
                        ix0
                    ] = coh_thresh_sel

                    bperp_range = float(
                        np.max(
                            bperp_work
                        )
                        - np.min(
                            bperp_work
                        )
                    )

                    if bperp_range <= 0:
                        bperp_range = 1.0

                    keep_ix = (
                        coh_ps2
                        > coh_thresh_sel
                    ) & (
                        np.abs(
                            K_ps[
                                ix0
                            ]
                            - K_ps2
                        )
                        < (
                            2
                            * np.pi
                            / bperp_range
                        )
                    )

                    debug_payload[
                        "reestimate_used"
                    ] = True

                    debug_payload[
                        "reestimate_status"
                    ] = "completed"

            except Exception as exc:
                reestimate_ok = False

                debug_payload[
                    "reestimate_status"
                ] = "failed"

                debug_payload[
                    "reestimate_exception"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        if not reestimate_ok:
            ph_patch2 = ph_patch[
                ix0,
                :,
            ].astype(
                np.complex64,
                copy=True,
            )

            ph_res2 = ph_res[
                ix0,
                :,
            ].astype(
                np.float32,
                copy=True,
            )

            K_ps2 = K_ps[
                ix0
            ].astype(
                np.float64,
                copy=True,
            )

            C_ps2 = C_ps[
                ix0
            ].astype(
                np.float64,
                copy=True,
            )

            coh_ps2 = coh_ps[
                ix0
            ].astype(
                np.float64,
                copy=True,
            )

            keep_ix = np.ones(
                ix.size,
                dtype=bool,
            )

    else:
        ph_patch2 = np.empty(
            (
                0,
                ph_patch.shape[1],
            ),
            dtype=np.complex64,
        )

        ph_res2 = np.empty(
            (
                0,
                ph_res.shape[1],
            ),
            dtype=np.float32,
        )

        K_ps2 = np.empty(
            (
                0,
            ),
            dtype=np.float64,
        )

        C_ps2 = np.empty(
            (
                0,
            ),
            dtype=np.float64,
        )

        coh_ps2 = np.empty(
            (
                0,
            ),
            dtype=np.float64,
        )

        keep_ix = np.empty(
            (
                0,
            ),
            dtype=bool,
        )

    payload: dict[str, Any] = {
        "ix": _matlab_col(
            ix,
            np.float64,
        ),
        "keep_ix": _matlab_col(
            keep_ix,
            np.bool_,
        ),
        "ph_patch2": ph_patch2.astype(
            np.complex64,
            copy=False,
        ),
        "ph_res2": ph_res2,
        "K_ps2": _matlab_col(
            K_ps2,
            np.float64,
        ),
        "C_ps2": _matlab_col(
            C_ps2,
            np.float64,
        ),
        "coh_ps2": _matlab_col(
            coh_ps2,
            np.float64,
        ),
        "coh_thresh": _matlab_col(
            coh_thresh_all[
                ix0
            ],
            np.float64,
        ),
        "coh_thresh_coeffs": (
            coh_thresh_coeffs
        ),
        "clap_alpha": np.asarray(
            _build_stage_options(
                patch_dir
            ).clap_alpha,
            dtype=np.float64,
        ),
        "clap_beta": np.asarray(
            _build_stage_options(
                patch_dir
            ).clap_beta,
            dtype=np.float64,
        ),
        "n_win": np.asarray(
            _build_stage_options(
                patch_dir
            ).clap_win,
            dtype=np.float64,
        ),
        "max_percent_rand": np.asarray(
            max_percent_rand,
            dtype=np.float32,
        ),
        "gamma_stdev_reject": np.asarray(
            parms.gamma_stdev_reject,
            dtype=np.float64,
        ),
        "small_baseline_flag": (
            _matlab_char_row(
                parms.small_baseline_flag
            )
        ),
        "ifg_index": _matlab_row(
            ifg_index,
            np.float64,
        ),
    }

    write_mat(
        patch_dir
        / "select1.mat",
        payload,
    )

    debug_payload.update(
        {
            "ix_count": int(
                ix.size
            ),
            "keep_true_count": int(
                np.count_nonzero(
                    keep_ix
                )
            ),
            "coh_thresh_coeffs": np.asarray(
                coh_thresh_coeffs,
                dtype=np.float64,
            ).reshape(-1).tolist(),
            "max_percent_rand": float(
                max_percent_rand
            ),
            "gamma_stdev_reject": float(
                parms.gamma_stdev_reject
            ),
            "duration_sec": float(
                time.perf_counter()
                - stage3_started
            ),
        }
    )

    _write_stage3_debug(
        patch_dir,
        debug_payload,
    )

    return (
        f"Stage 3 selected "
        f"{ix.size} PS "
        f"(fast path)"
    )


def stage3_select_ps(
    patch_dir: Path,
    backend: str = "auto",
) -> str:
    """
    Stage 3 dispatcher.

    The original implementation remains the default. Enable the
    optimised path explicitly with:

        PYSTAMPS_STAGE3_FAST=1
    """

    if _stage3_environment_flag(
        "PYSTAMPS_STAGE3_FAST",
        default=False,
    ):
        return _stage3_select_ps_fast(
            patch_dir,
            backend=backend,
        )

    return _stage3_select_ps_legacy(
        patch_dir,
        backend=backend,
    )
'''

    ported = ported.replace(
        insert_marker,
        "\n"
        + textwrap.dedent(
            fast_code
        ).strip()
        + "\n\n"
        + "def _stage4_checkpoint(\n",
        1,
    )

ported_path.write_text(
    ported,
    encoding="utf-8",
)

print("源代码修改完成。")
PY


# ----------------------------------------------------------------------
# 3. 单元测试
# ----------------------------------------------------------------------

cat > "$TEST_FILE" <<'PY'
from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import (
    read_mat_variables,
    write_mat,
)
from pystamps.pipeline import ported


def test_read_mat_variables_reads_only_requested(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "sample.mat"
    )

    write_mat(
        path,
        {
            "small": np.arange(
                5,
                dtype=np.float64,
            ),
            "large": np.arange(
                100,
                dtype=np.float32,
            ).reshape(
                10,
                10,
            ),
        },
    )

    payload = read_mat_variables(
        path,
        (
            "small",
        ),
    )

    assert set(
        payload
    ) == {
        "small",
    }

    np.testing.assert_allclose(
        np.asarray(
            payload[
                "small"
            ]
        ).reshape(-1),
        np.arange(
            5,
            dtype=np.float64,
        ),
    )


def test_stage3_batched_clap_matches_scalar_double() -> None:
    rng = np.random.default_rng(
        42
    )

    stack = (
        rng.normal(
            size=(
                8,
                8,
                5,
            )
        )
        + 1j
        * rng.normal(
            size=(
                8,
                8,
                5,
            )
        )
    ).astype(
        np.complex128
    )

    low_pass = np.ones(
        (
            8,
            8,
        ),
        dtype=np.float64,
    ) * 0.2

    reference = np.empty_like(
        stack,
        dtype=np.complex128,
    )

    for i_ifg in range(
        stack.shape[2]
    ):
        reference[
            :,
            :,
            i_ifg,
        ] = ported._clap_filt_patch(
            stack[
                :,
                :,
                i_ifg,
            ],
            alpha=1.0,
            beta=0.3,
            low_pass=low_pass,
        )

    actual = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=False,
    )

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_stage3_batched_clap_single_close_to_double() -> None:
    rng = np.random.default_rng(
        7
    )

    stack = (
        rng.normal(
            size=(
                8,
                8,
                4,
            )
        )
        + 1j
        * rng.normal(
            size=(
                8,
                8,
                4,
            )
        )
    ).astype(
        np.complex64
    )

    low_pass = np.ones(
        (
            8,
            8,
        ),
        dtype=np.float64,
    ) * 0.2

    double = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=False,
    )

    single = ported._stage3_clap_patch_stack(
        stack,
        alpha=1.0,
        beta=0.3,
        low_pass=low_pass,
        single_precision=True,
    )

    np.testing.assert_allclose(
        single,
        double,
        rtol=5.0e-5,
        atol=5.0e-5,
    )
PY


# ----------------------------------------------------------------------
# 4. 生成真实patch结果验证工具
# ----------------------------------------------------------------------

cat > "$VALIDATOR" <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

import numpy as np

from pystamps.io.mat import read_mat
from pystamps.pipeline import ported


def max_abs_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_array = np.asarray(
        left
    )

    right_array = np.asarray(
        right
    )

    if left_array.shape != right_array.shape:
        return float(
            "inf"
        )

    if (
        left_array.size
        == 0
    ):
        return 0.0

    difference = np.abs(
        left_array
        - right_array
    )

    finite = np.isfinite(
        difference
    )

    if not np.any(
        finite
    ):
        return 0.0

    return float(
        np.max(
            difference[
                finite
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "patch",
        type=Path,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--single",
        action="store_true",
    )

    args = parser.parse_args()

    source = args.patch.expanduser().resolve()

    legacy_file = (
        source
        / "select1.mat"
    )

    if not legacy_file.exists():
        raise SystemExit(
            f"缺少原Stage 3结果：{legacy_file}"
        )

    validation_root = (
        source.parent
        / "_stage3_fast_validation"
    )

    validation_patch = (
        validation_root
        / source.name
    )

    if validation_patch.exists():
        shutil.rmtree(
            validation_patch
        )

    validation_patch.mkdir(
        parents=True
    )

    for item in source.iterdir():
        if not item.is_file():
            continue

        if item.name in {
            "select1.mat",
            "stage3_debug.json",
        }:
            continue

        os.symlink(
            item.resolve(),
            validation_patch
            / item.name,
        )

    os.environ[
        "PYSTAMPS_STAGE3_FAST"
    ] = "1"

    os.environ[
        "PYSTAMPS_STAGE3_THREADS"
    ] = str(
        max(
            1,
            args.threads,
        )
    )

    os.environ[
        "PYSTAMPS_STAGE3_SINGLE_PRECISION"
    ] = (
        "1"
        if args.single
        else "0"
    )

    os.environ[
        "PYSTAMPS_STAGE3_PROGRESS"
    ] = "1"

    started = time.perf_counter()

    result = ported.stage3_select_ps(
        validation_patch,
        backend="auto",
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        result
    )

    print(
        f"快速Stage 3耗时：{elapsed:.2f}秒"
    )

    legacy = read_mat(
        legacy_file
    )

    fast = read_mat(
        validation_patch
        / "select1.mat"
    )

    ix_legacy = np.asarray(
        legacy.get(
            "ix"
        ),
        dtype=np.int64,
    ).reshape(-1)

    ix_fast = np.asarray(
        fast.get(
            "ix"
        ),
        dtype=np.int64,
    ).reshape(-1)

    keep_legacy = np.asarray(
        legacy.get(
            "keep_ix"
        ),
        dtype=bool,
    ).reshape(-1)

    keep_fast = np.asarray(
        fast.get(
            "keep_ix"
        ),
        dtype=bool,
    ).reshape(-1)

    print()
    print(
        "================ 验证结果 ================"
    )

    print(
        f"原始ix数量：{ix_legacy.size}"
    )

    print(
        f"快速ix数量：{ix_fast.size}"
    )

    print(
        "ix完全一致：",
        bool(
            np.array_equal(
                ix_legacy,
                ix_fast,
            )
        ),
    )

    print(
        "keep_ix完全一致：",
        bool(
            np.array_equal(
                keep_legacy,
                keep_fast,
            )
        ),
    )

    for key in (
        "K_ps2",
        "C_ps2",
        "coh_ps2",
        "coh_thresh",
        "ph_patch2",
        "ph_res2",
    ):
        if (
            key not in legacy
            or key not in fast
        ):
            print(
                f"{key}: 缺失"
            )

            continue

        difference = max_abs_difference(
            np.asarray(
                legacy[
                    key
                ]
            ),
            np.asarray(
                fast[
                    key
                ]
            ),
        )

        print(
            f"{key}最大绝对差异："
            f"{difference:.12g}"
        )

    print(
        "验证目录：",
        validation_patch,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
PY

chmod +x "$VALIDATOR"


# ----------------------------------------------------------------------
# 5. 生成回滚脚本
# ----------------------------------------------------------------------

cat > "$ROOT/rollback_stage3_fast_patch.sh" <<'BASH2'
#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/h/pystamps-main"

if [[ ! -f "$ROOT/.stage3_fast_last_backup" ]]; then
    echo "找不到最近备份记录。" >&2
    exit 2
fi

BACKUP="$(
    cat "$ROOT/.stage3_fast_last_backup"
)"

if [[ ! -d "$BACKUP" ]]; then
    echo "备份目录不存在：$BACKUP" >&2
    exit 2
fi

cp -a \
  "$BACKUP/pystamps/pipeline/ported.py" \
  "$ROOT/pystamps/pipeline/ported.py"

cp -a \
  "$BACKUP/pystamps/io/mat.py" \
  "$ROOT/pystamps/io/mat.py"

if [[ -f "$BACKUP/tests/test_stage3_fast_path.py" ]]; then
    cp -a \
      "$BACKUP/tests/test_stage3_fast_path.py" \
      "$ROOT/tests/test_stage3_fast_path.py"
else
    rm -f \
      "$ROOT/tests/test_stage3_fast_path.py"
fi

echo "已从以下目录恢复："
echo "$BACKUP"
BASH2

chmod +x "$ROOT/rollback_stage3_fast_patch.sh"


# ----------------------------------------------------------------------
# 6. 语法与测试
# ----------------------------------------------------------------------

python -m compileall -q \
  "$MATIO" \
  "$PORTED" \
  "$TEST_FILE" \
  "$VALIDATOR"

python -m pytest -q \
  tests/test_stage3_fast_path.py

python -m pytest -q \
  tests/test_stage3_ported.py

echo
echo "============================================================"
echo "Stage 3快速补丁已完成"
echo "备份目录：$BACKUP"
echo
echo "快速路径默认关闭，不会改变现有行为。"
echo "启用方式："
echo "  export PYSTAMPS_STAGE3_FAST=1"
echo "============================================================"
BASH

chmod +x apply_stage3_fast_patch.sh

./apply_stage3_fast_patch.sh