from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat


STAGE_PROVENANCE_SCHEMA = 1
STAGE_ALGORITHM_REVISION = "2026-09-12-science-v1"


_PATCH_INPUTS: dict[int, tuple[str, ...]] = {
    2: (
        "ps1.mat",
        "ph1.mat",
        "bp1.mat",
        "da1.mat",
        "hgt1.mat",
        "la1.mat",
        "inc1.mat",
    ),
    3: (
        "ps1.mat",
        "pm1.mat",
    ),
    4: (
        "ps1.mat",
        "pm1.mat",
        "select1.mat",
    ),
    5: (
        "ps1.mat",
        "ph1.mat",
        "bp1.mat",
        "pm1.mat",
        "select1.mat",
        "weed1.mat",
        "hgt1.mat",
        "la1.mat",
        "inc1.mat",
    ),
}


_MERGED_INPUTS: dict[int, tuple[str, ...]] = {
    6: (
        "ps2.mat",
        "ph2.mat",
        "pm2.mat",
        "bp2.mat",
        "hgt2.mat",
        "la2.mat",
        "inc2.mat",
        "rc2.mat",
        "ifgstd2.mat",
    ),
    7: (
        "ps2.mat",
        "bp2.mat",
    ),
    8: (
        "ps2.mat",
        "bp2.mat",
        "scla2.mat",
    ),
}


_STAGE5_PATCH_OUTPUTS: tuple[str, ...] = (
    "ps2.mat",
    "ph2.mat",
    "pm2.mat",
    "bp2.mat",
    "hgt2.mat",
    "la2.mat",
    "inc2.mat",
    "rc2.mat",
    "psver.mat",
)


_STAGE_PARM_KEYS: dict[int, tuple[str, ...]] = {
    2: (
        "small_baseline_flag",
        "lambda",
        "max_topo_err",
        "filter_grid_size",
        "filter_weighting",
        "clap_win",
        "clap_low_pass_wavelength",
        "clap_alpha",
        "clap_beta",
        "gamma_change_convergence",
        "gamma_max_iterations",
        "quick_est_gamma_flag",
        "select_reest_gamma_flag",
    ),
    3: (
        "small_baseline_flag",
        "select_method",
        "density_rand",
        "percent_rand",
        "select_reest_gamma_flag",
        "gamma_stdev_reject",
        "gamma_change_convergence",
    ),
    4: (
        "small_baseline_flag",
        "weed_standard_dev",
        "weed_max_noise",
        "weed_time_win",
        "weed_zero_elevation",
        "weed_neighbours",
    ),
    5: (
        "small_baseline_flag",
        "heading",
    ),
    6: (
        "small_baseline_flag",
        "lambda",
        "max_topo_err",
        "drop_ifg_index",
        "scla_deramp",
        "subtr_tropo",
    ),
    7: (
        "small_baseline_flag",
        "lambda",
        "drop_ifg_index",
        "scla_method",
        "scla_deramp",
        "subtr_tropo",
        "sb_scla_drop_index",
    ),
    8: (
        "small_baseline_flag",
        "lambda",
        "drop_ifg_index",
    ),
}


_STAGE_PARM_PREFIXES: dict[int, tuple[str, ...]] = {
    6: ("unwrap_",),
    7: ("scla_", "sb_scla_"),
    8: ("scn_",),
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mat_value_json_safe(value: Any) -> Any:
    array = np.asarray(value)

    if array.size == 0:
        return []

    if array.dtype.kind in {"U", "S", "O"}:
        flat = array.reshape(-1)
        return [
            str(item)
            for item in flat.tolist()
        ]

    flat = array.reshape(-1)
    out: list[Any] = []
    for item in flat:
        scalar = item.item() if hasattr(item, "item") else item
        if isinstance(scalar, (np.integer, int)):
            out.append(int(scalar))
        elif isinstance(scalar, (np.floating, float)):
            out.append(float(scalar))
        elif isinstance(scalar, (np.bool_, bool)):
            out.append(bool(scalar))
        else:
            out.append(str(scalar))

    if len(out) == 1:
        return out[0]
    return out


def _stage_parameter_signature(
    dataset_root: Path,
    stage_id: int,
) -> dict[str, Any]:
    parms_path = dataset_root / "parms.mat"
    if not parms_path.is_file():
        return {
            "present": False,
        }

    try:
        parms = read_mat(parms_path)
    except Exception as exc:
        return {
            "present": True,
            "read_error": f"{type(exc).__name__}: {exc}",
        }

    names = set(
        _STAGE_PARM_KEYS.get(
            stage_id,
            (),
        )
    )

    prefixes = _STAGE_PARM_PREFIXES.get(
        stage_id,
        (),
    )

    for key in parms:
        if any(
            str(key).startswith(prefix)
            for prefix in prefixes
        ):
            names.add(str(key))

    values: dict[str, Any] = {}
    for key in sorted(names):
        if key not in parms:
            values[key] = None
            continue
        values[key] = _mat_value_json_safe(
            parms[key]
        )

    return {
        "present": True,
        "values": values,
    }


def _file_signature(
    path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    path = Path(path)
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
        label = str(rel)
    except Exception:
        label = str(path.resolve())

    if not path.is_file():
        return {
            "path": label,
            "exists": False,
        }

    stat = path.stat()
    return {
        "path": label,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stage_config(
    run_config: Any,
    stage_id: int,
    *,
    patch_name: str | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}

    if stage_id == 2:
        runtime = run_config.runtime
        backend = runtime.stage2_kernel_backend
        if patch_name:
            backend = runtime.stage2_patch_backend_overrides.get(
                patch_name,
                backend,
            )
        config["stage2"] = {
            "runtime_backend": runtime.backend,
            "kernel_backend": backend,
            "kernel_backend_overrides": runtime.kernel_backend_overrides,
            "native_threads": runtime.stage2_native_threads,
            "checkpoint_mode": runtime.stage2_checkpoint_mode,
            "checkpoint_interval": runtime.stage2_checkpoint_interval,
        }

    if stage_id == 6:
        config["ifg_selection"] = run_config.ifg_selection
        config["reference"] = run_config.reference
        config["tools"] = {
            "triangle": run_config.tools.triangle,
            "snaphu": run_config.tools.snaphu,
        }

    if stage_id in {7, 8}:
        config["tools"] = {
            "triangle": run_config.tools.triangle,
            "snaphu": run_config.tools.snaphu,
        }
        config["gacos"] = run_config.gacos
        config["post_unwrap_deramp"] = run_config.post_unwrap_deramp

    config["compat"] = run_config.compat
    return _json_safe(config)


def stage_marker_path(
    target_dir: Path,
    stage_id: int,
    scope: str,
) -> Path:
    return (
        Path(target_dir)
        / f"_pystamps_stage{stage_id}_{scope}_provenance.json"
    )


def build_stage_signature(
    *,
    dataset_root: Path,
    target_dir: Path,
    stage_id: int,
    scope: str,
    run_config: Any,
    phase_file: str | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()

    payload: dict[str, Any] = {
        "schema": STAGE_PROVENANCE_SCHEMA,
        "algorithm_revision": STAGE_ALGORITHM_REVISION,
        "stage_id": int(stage_id),
        "scope": str(scope),
        "config": _stage_config(
            run_config,
            stage_id,
            patch_name=(target.name if scope == "patch" else None),
        ),
        "parameters": _stage_parameter_signature(
            root,
            stage_id,
        ),
        "inputs": [],
    }

    inputs: list[dict[str, Any]] = []

    if scope == "patch":
        for name in _PATCH_INPUTS.get(stage_id, ()):
            inputs.append(
                _file_signature(
                    target / name,
                    root,
                )
            )

    elif scope == "merged" and stage_id == 5:
        for patch in sorted(root.glob("PATCH_*")):
            if not patch.is_dir():
                continue
            for name in _STAGE5_PATCH_OUTPUTS:
                inputs.append(
                    _file_signature(
                        patch / name,
                        root,
                    )
                )

    elif scope == "merged":
        for name in _MERGED_INPUTS.get(stage_id, ()):
            inputs.append(
                _file_signature(
                    root / name,
                    root,
                )
            )

    if phase_file is not None:
        payload["phase_file"] = str(phase_file)
        inputs.append(
            _file_signature(
                root / phase_file,
                root,
            )
        )

    payload["inputs"] = inputs
    return payload


def stage_marker_is_current(
    target_dir: Path,
    stage_id: int,
    scope: str,
    current: dict[str, Any],
) -> bool:
    marker = stage_marker_path(
        target_dir,
        stage_id,
        scope,
    )

    if not marker.is_file():
        return False

    try:
        saved = json.loads(
            marker.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return False

    return saved == current


def write_stage_marker(
    target_dir: Path,
    stage_id: int,
    scope: str,
    payload: dict[str, Any],
) -> None:
    marker = stage_marker_path(
        target_dir,
        stage_id,
        scope,
    )

    tmp = marker.with_suffix(
        marker.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(marker)
