from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import time

from pystamps.config import ConfigError, normalize_runtime_backend
from pystamps.ifg_selection import resolve_ifg_selection
from pystamps.grid_ifg_qc import grid_qc_audit_is_current
from pystamps.io.dataset import DatasetLayout, discover_dataset, expected_stage_artifact
from pystamps.pipeline.ported import (
    PortedStageError,
    stage5_merge_and_ifgstd,
    stage6_unwrap,
    stage7_calc_scla,
    stage8_filter_scn,
    stage1_load_initial,
    stage2_estimate_gamma,
    stage3_select_ps,
    stage4_weed_ps,
    stage5_correct_and_promote,
)
from pystamps.pipeline.types import PipelineContext, PipelineReport, StageResult
from pystamps.pipeline.provenance import (
    build_stage_signature,
    stage_marker_is_current,
    write_stage_marker,
)
from pystamps.runtime.executor import HybridExecutor
from pystamps.reference import resolve_reference


@dataclass(slots=True)
class StageDef:
    stage_id: int
    name: str
    scope: str


STAGE_DEFS: list[StageDef] = [
    StageDef(1, "Initial load", "patch"),
    StageDef(2, "Estimate gamma", "patch"),
    StageDef(3, "Select PS pixels", "patch"),
    StageDef(4, "Weed adjacent pixels", "patch"),
    StageDef(5, "Correct phase + merge", "patch"),
    StageDef(6, "Unwrap phase", "merged"),
    StageDef(7, "Calculate SCLA", "merged"),
    StageDef(8, "Filter SCN", "merged"),
]


class StageExecutionError(RuntimeError):
    """Raised when a stage should run but has no Python implementation yet."""


PATCH_STAGE_BUNDLES: dict[int, list[str]] = {
    1: ["ps1.mat", "ph1.mat", "bp1.mat", "da1.mat", "hgt1.mat", "la1.mat", "psver.mat"],
    2: ["pm1.mat"],
    3: ["select1.mat"],
    4: ["weed1.mat"],
    5: ["ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "hgt2.mat", "la2.mat", "rc2.mat", "psver.mat"],
}

MERGED_STAGE_BUNDLES: dict[int, list[str]] = {
    5: ["ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "hgt2.mat", "la2.mat", "rc2.mat", "psver.mat", "ifgstd2.mat"],
    6: ["ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "ifgstd2.mat", "phuw_sb2.mat", "phuw2.mat", "phuw_sb_res2.mat", "uw_phaseuw.mat", "uw_grid.mat", "uw_interp.mat"],
    7: ["scla_sb2.mat", "scla_smooth_sb2.mat", "scla2.mat"],
    8: ["scn2.mat"],
}


def _processor_is_gamma(dataset_root: Path) -> bool:
    processor_file = dataset_root / "processor.txt"
    if not processor_file.is_file():
        return False
    try:
        processor = processor_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip().lower()
    except OSError:
        return False
    return processor == "gamma"


def _required_stage_bundle(
    dataset_root: Path,
    stage_id: int,
    scope: str,
) -> list[str]:
    base = (
        PATCH_STAGE_BUNDLES.get(stage_id, [])
        if scope == "patch"
        else MERGED_STAGE_BUNDLES.get(stage_id, [])
    )
    bundle = list(base)

    if _processor_is_gamma(dataset_root):
        if scope == "patch" and stage_id == 1 and "inc1.mat" not in bundle:
            bundle.append("inc1.mat")
        if stage_id == 5 and "inc2.mat" not in bundle:
            bundle.append("inc2.mat")

    return bundle


# === STAGE1_AUTO_PREP_V1 ===

_STAGE1_ROOT_REQUIRED = (
    "processor.txt",
    "width.txt",
    "len.txt",
    "small_baselines.list",
    "parms.mat",
)


def _stage1_dataset_complete(dataset: DatasetLayout) -> bool:
    if not dataset.patches:
        return False

    if not all((dataset.root / name).is_file() for name in _STAGE1_ROOT_REQUIRED):
        return False

    required = _required_stage_bundle(
        dataset.root,
        1,
        "patch",
    )

    return all(
        all((patch / name).is_file() for name in required)
        for patch in dataset.patches
    )


def ensure_stage1_dataset(context: PipelineContext) -> DatasetLayout:
    dataset = discover_dataset(context.dataset_root)

    if context.start_step > 1:
        return dataset

    if _stage1_dataset_complete(dataset):
        print("[STAGE1] Existing Stage-1 dataset is complete; reuse.", flush=True)
        return dataset

    print()
    print("============================================================", flush=True)
    print("pySTAMPS STAGE-1 AUTO PREPARATION", flush=True)
    print("============================================================", flush=True)
    print("[STAGE1] Stage-1 dataset is missing or incomplete.", flush=True)

    if context.dry_run:
        print("[STAGE1] dry-run: GAMMA Stage-1 preparation would run automatically.", flush=True)
        print("============================================================", flush=True)
        print()
        return dataset

    from pystamps.prep.gamma_candidates import CandidateConfig
    from pystamps.prep.gamma_stage1 import GammaStage1Config, prepare_gamma_sbas_stage1

    data_dir = Path(
        os.environ.get("PYSTAMPS_DATA_DIR", str(context.dataset_root.parent))
    ).expanduser().resolve()

    dem_raw = os.environ.get("PYSTAMPS_DEM_DIR")
    dem_directory = Path(dem_raw).expanduser().resolve() if dem_raw else None

    print(f"[STAGE1] work_dir : {context.dataset_root}", flush=True)
    print(f"[STAGE1] data_dir : {data_dir}", flush=True)
    if dem_directory is not None:
        print(f"[STAGE1] DEM      : {dem_directory}", flush=True)
    print("[STAGE1] Preparing Stage 1 from GAMMA inputs automatically...", flush=True)
    print()

    ref = context.run_config.reference
    cfg = GammaStage1Config(
        candidate=CandidateConfig(
            da_threshold=0.60,
            min_valid_fraction=0.90,
            block_rows=2048,
            mli_is_power=True,
            normalize_per_image=False,
        ),
        candidate_source="rslc_sbas",
        reference_lon=ref.longitude,
        reference_lat=ref.latitude,
        reference_radius_m=(
            ref.radius_m
            if ref.longitude is not None and ref.latitude is not None
            else None
        ),
        dem_directory=dem_directory,
        range_looks=None,
        azimuth_looks=None,
        sbas_deramp_mode="none",
        force=False,
    )

    runtime = context.run_config.runtime

    stage1_env = {
        "PYSTAMPS_STAGE1_RESUME": "1",
        "PYSTAMPS_RSLC_DA_BACKEND": str(
            runtime.stage1_rslc_da_backend
        ),
        "PYSTAMPS_DA_WORKERS": str(
            int(runtime.stage1_da_workers)
        ),
        "PYSTAMPS_DA_NATIVE_THREADS": str(
            int(runtime.stage1_da_native_threads)
        ),
        "PYSTAMPS_DA_NATIVE_CHUNK_PIXELS": str(
            int(runtime.stage1_da_native_chunk_pixels)
        ),
        "PYSTAMPS_RSLC_ML_BLOCK_ROWS": str(
            int(runtime.stage1_rslc_ml_block_rows)
        ),
        "PYSTAMPS_RSLC_CALAMP_BLOCK_ROWS": str(
            int(runtime.stage1_rslc_calamp_block_rows)
        ),
    }

    old_stage1_env = {
        key: os.environ.get(key)
        for key in stage1_env
    }

    os.environ.update(stage1_env)

    print(
        "[STAGE1] RSLC-D_A backend="
        f"{runtime.stage1_rslc_da_backend}, "
        f"io_workers={runtime.stage1_da_workers or 'auto'}, "
        f"native_threads={runtime.stage1_da_native_threads or 'auto'}, "
        f"block_rows={runtime.stage1_rslc_ml_block_rows}",
        flush=True,
    )

    try:
        prepare_gamma_sbas_stage1(
            data_dir,
            context.dataset_root,
            config=cfg,
        )
    except Exception as exc:
        raise StageExecutionError(
            f"Automatic GAMMA Stage-1 preparation failed: {exc}"
        ) from exc
    finally:
        for key, old_value in old_stage1_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    dataset = discover_dataset(context.dataset_root)

    if not _stage1_dataset_complete(dataset):
        raise StageExecutionError(
            "Automatic GAMMA Stage-1 preparation returned successfully, "
            "but the resulting Stage-1 dataset is incomplete."
        )

    print("[STAGE1] Automatic Stage-1 preparation completed.", flush=True)
    print(f"[STAGE1] patch count: {len(dataset.patches)}", flush=True)
    print("============================================================", flush=True)
    print()
    return dataset


def _stage_phase_marker_path(
    dataset_root: Path,
    stage_id: int,
) -> Path:
    return (
        dataset_root
        / f"_pystamps_stage{stage_id}_phase_input.json"
    )


def _phase_input_signature(
    dataset_root: Path,
    phase_file: str,
) -> dict[str, object]:

    path = (
        dataset_root
        / phase_file
    )

    if not path.is_file():
        raise StageExecutionError(
            f"Stage phase input does not exist: {path}"
        )

    stat = path.stat()

    return {
        "phase_file": phase_file,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _phase_input_is_current(
    dataset_root: Path,
    stage_id: int,
    phase_file: str,
) -> bool:

    marker = _stage_phase_marker_path(
        dataset_root,
        stage_id,
    )

    # Backward compatibility:
    #
    # v1.0.0 Stage 7/8 outputs have no marker and were
    # produced from ordinary phuw2.mat. Treat those as
    # current while GACOS remains disabled.
    if not marker.is_file():
        return (
            phase_file
            == "phuw2.mat"
        )

    try:
        saved = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )

        current = _phase_input_signature(
            dataset_root,
            phase_file,
        )

        return (
            saved.get("phase_file")
            == current["phase_file"]
            and int(
                saved.get("size", -1)
            )
            == current["size"]
            and int(
                saved.get(
                    "mtime_ns",
                    -1,
                )
            )
            == current["mtime_ns"]
        )

    except Exception:
        return False


def _write_phase_input_marker(
    dataset_root: Path,
    stage_id: int,
    phase_file: str,
) -> None:

    payload = _phase_input_signature(
        dataset_root,
        phase_file,
    )

    marker = _stage_phase_marker_path(
        dataset_root,
        stage_id,
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


def _resolve_stage78_phase_file(
    dataset_root: Path,
    context: PipelineContext,
) -> str:

    # Stage 7 and Stage 8 in one pipeline invocation must use
    # the exact same materialized phase product.
    if context.stage78_phase_file is not None:
        return context.stage78_phase_file

    # Build one deterministic final phase product for Stage 7/8.
    #
    # Order when both corrections are enabled:
    #   phuw2.mat -> GACOS -> post-unwrapping deramp
    #
    # Atmospheric delay is corrected first. The deramp then removes only
    # the residual time-linear long-wavelength spatial gradient.
    selected = "phuw2.mat"

    gacos = context.run_config.gacos
    if bool(gacos.enabled):
        if context.dry_run:
            selected = "phuw2_gacos.mat"
        else:
            from pystamps.pipeline.gacos_correction import (
                ensure_gacos_corrected_phuw,
            )

            corrected = ensure_gacos_corrected_phuw(
                dataset_root,
                gacos,
            )

            if corrected.parent != dataset_root.resolve():
                raise StageExecutionError(
                    "GACOS corrected phase was created outside "
                    "the dataset root"
                )

            selected = corrected.name

    deramp = context.run_config.post_unwrap_deramp
    if bool(deramp.enabled):
        if context.dry_run:
            p = Path(selected)
            selected = f"{p.stem}_deramp{p.suffix or '.mat'}"
        else:
            from pystamps.pipeline.post_unwrap_deramp import (
                ensure_post_unwrap_deramped_phase,
            )

            corrected = ensure_post_unwrap_deramped_phase(
                dataset_root,
                deramp,
                input_phase_file=selected,
            )

            if corrected.parent != dataset_root.resolve():
                raise StageExecutionError(
                    "Deramped phase was created outside the dataset root"
                )

            selected = corrected.name

    context.stage78_phase_file = selected
    return selected


def _normalize_backend(name: str) -> str:
    try:
        return normalize_runtime_backend(name)
    except ConfigError as exc:
        raise StageExecutionError(str(exc)) from exc


def _task_kind_for_stage(stage: StageDef, context: PipelineContext, patch_count: int = 0) -> str:
    # Replay mode is file-copy heavy; use IO workers regardless of backend.
    if context.run_config.compat.strict_reference:
        return "io"

    backend = _normalize_backend(context.run_config.runtime.backend)
    if backend == "threads":
        return "io"
    if backend == "processes":
        return "cpu"
    if backend == "gpu":
        # Keep GPU work in-process to avoid per-process CUDA context overhead.
        return "io"
    if backend == "native":
        return "cpu"

    # Auto mode: CPU-first latency policy.
    # Stage-1 stays threaded (metadata/file heavy).
    # Patch compute stages use processes only if there is useful fan-out.
    # Merged stages remain in-process to avoid process startup/marshalling cost.
    if stage.scope == "patch" and stage.stage_id == 1:
        return "io"
    if stage.scope == "patch":
        return "cpu" if patch_count >= 2 else "io"
    return "io"


def _default_cpu_workers() -> int:
    return max(1, os.cpu_count() or 4)


def _configured_cpu_workers(context: PipelineContext) -> int:
    value = int(context.run_config.runtime.cpu_workers)
    if value > 0:
        return value
    return _default_cpu_workers()


def _stage2_uses_full_cpu_default(stage: StageDef, context: PipelineContext) -> bool:
    runtime = context.run_config.runtime
    if stage.stage_id != 2:
        return False
    if int(runtime.stage2_native_threads) > 0:
        return False
    backends = {runtime.stage2_kernel_backend, *runtime.stage2_patch_backend_overrides.values()}
    return any(str(backend).strip().lower() in {"auto", "native"} for backend in backends)


def _effective_stage2_native_threads(
    stage: StageDef,
    context: PipelineContext,
    patch_count: int,
    *,
    stage2_kernel_backend: str | None = None,
) -> int:
    runtime = context.run_config.runtime
    requested = int(runtime.stage2_native_threads)
    if requested > 0:
        return requested
    if stage.stage_id != 2:
        return 0
    selected_backend = stage2_kernel_backend or runtime.stage2_kernel_backend
    if selected_backend.strip().lower() not in {"auto", "native"}:
        return 0
    return _configured_cpu_workers(context)


def _stage_provenance_enabled(
    context: PipelineContext,
) -> bool:
    return bool(
        getattr(
            context.run_config.runtime,
            "validate_stage_provenance",
            True,
        )
    )


def _current_stage_provenance(
    context: PipelineContext,
    stage_id: int,
    scope: str,
    target_dir: Path,
    *,
    phase_file: str | None = None,
) -> dict[str, object]:
    return build_stage_signature(
        dataset_root=context.dataset_root,
        target_dir=target_dir,
        stage_id=stage_id,
        scope=scope,
        run_config=context.run_config,
        phase_file=phase_file,
    )


def _stage_provenance_is_current(
    context: PipelineContext,
    stage_id: int,
    scope: str,
    target_dir: Path,
    *,
    phase_file: str | None = None,
) -> bool:
    if not _stage_provenance_enabled(context):
        return True

    current = _current_stage_provenance(
        context,
        stage_id,
        scope,
        target_dir,
        phase_file=phase_file,
    )

    return stage_marker_is_current(
        target_dir,
        stage_id,
        scope,
        current,
    )


def _commit_stage_provenance(
    context: PipelineContext,
    stage_id: int,
    scope: str,
    target_dir: Path,
    *,
    phase_file: str | None = None,
) -> None:
    if not _stage_provenance_enabled(context):
        return

    payload = _current_stage_provenance(
        context,
        stage_id,
        scope,
        target_dir,
        phase_file=phase_file,
    )

    write_stage_marker(
        target_dir,
        stage_id,
        scope,
        payload,
    )


def _replay_from_reference(
    context: PipelineContext,
    scope: str,
    stage_id: int,
    target_dir: Path,
) -> str | None:
    compat = context.run_config.compat
    if not compat.strict_reference or not compat.reference_root:
        return None

    ref_root = Path(compat.reference_root).expanduser().resolve()
    if not ref_root.exists():
        raise StageExecutionError(f"Reference root does not exist: {ref_root}")

    rel_dir = target_dir.relative_to(context.dataset_root)
    bundle = _required_stage_bundle(
        context.dataset_root,
        stage_id,
        scope,
    )
    copied: list[str] = []
    missing: list[str] = []

    for filename in bundle:
        src = ref_root / rel_dir / filename
        dst = target_dir / filename
        if src.exists():
            try:
                if dst.exists() and os.path.samefile(src, dst):
                    copied.append(filename)
                    continue
            except FileNotFoundError:
                pass
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(filename)
        else:
            missing.append(filename)

    if missing:
        raise StageExecutionError(
            f"Strict reference replay missing files for stage {stage_id} ({scope}): {', '.join(missing)}"
        )

    return f"Replayed {len(copied)} artifacts from reference root"


def _run_ported_patch_stage(
    stage_id: int,
    patch_dir: Path,
    backend: str = "auto",
    stage2_kernel_backend: str = "auto",
    kernel_backend_overrides: dict[str, str] | None = None,
    stage2_native_threads: int = 0,
    stage2_checkpoint_mode: str = "final",
    stage2_checkpoint_interval: int = 1,
    stage2_debug: bool = False,
    stage4_debug: bool = False,
    strict_reference: bool = False,
) -> str:
    if stage_id == 1:
        return stage1_load_initial(patch_dir, backend=backend)
    if stage_id == 2:
        return stage2_estimate_gamma(
            patch_dir,
            backend=backend,
            kernel_backend=stage2_kernel_backend,
            kernel_backend_overrides=kernel_backend_overrides,
            native_threads=stage2_native_threads,
            checkpoint_mode=stage2_checkpoint_mode,
            checkpoint_interval=stage2_checkpoint_interval,
            debug=stage2_debug,
        )
    if stage_id == 3:
        return stage3_select_ps(patch_dir, backend=backend)
    if stage_id == 4:
        return stage4_weed_ps(
            patch_dir,
            backend=backend,
            debug=stage4_debug,
            strict_reference=strict_reference,
        )
    if stage_id == 5:
        return stage5_correct_and_promote(patch_dir, backend=backend)
    raise PortedStageError(f"No ported patch implementation for stage {stage_id}")


def _stage2_kernel_backend_for_patch(context: PipelineContext, patch_dir: Path) -> str:
    overrides = context.run_config.runtime.stage2_patch_backend_overrides
    if not overrides:
        return context.run_config.runtime.stage2_kernel_backend
    return overrides.get(patch_dir.name, context.run_config.runtime.stage2_kernel_backend)


def _kernel_backend_for_name(context: PipelineContext, kernel_name: str, default_backend: str) -> str:
    overrides = context.run_config.runtime.kernel_backend_overrides
    if not overrides:
        return default_backend
    return overrides.get(kernel_name, default_backend)


def _run_patch_stage(stage: StageDef, patch_dir: Path, context: PipelineContext, patch_count: int) -> StageResult:
    expected = expected_stage_artifact(stage.stage_id, "patch")
    if expected is None:
        return StageResult(stage.stage_id, "patch", patch_dir.name, "skipped", "No expected artifact mapping")

    artifact = patch_dir / expected
    bundle = _required_stage_bundle(
        context.dataset_root,
        stage.stage_id,
        "patch",
    )
    bundle_complete = bool(
        bundle
        and all(
            (patch_dir / filename).exists()
            for filename in bundle
        )
    )

    if bundle_complete:
        if (
            stage.stage_id == 1
            or _stage_provenance_is_current(
                context,
                stage.stage_id,
                "patch",
                patch_dir,
            )
        ):
            return StageResult(
                stage.stage_id,
                "patch",
                patch_dir.name,
                "skipped_existing",
                f"{expected} present; validated stage bundle present",
            )

        print(
            f"[STAGE{stage.stage_id}] "
            f"{patch_dir.name}: existing outputs have missing/stale "
            "provenance; recomputing",
            flush=True,
        )

    if context.dry_run:
        return StageResult(stage.stage_id, "patch", patch_dir.name, "planned", f"Would produce {expected}")

    replay_details = _replay_from_reference(context, "patch", stage.stage_id, patch_dir)
    if replay_details is not None:
        if stage.stage_id >= 2:
            _commit_stage_provenance(
                context,
                stage.stage_id,
                "patch",
                patch_dir,
            )
        return StageResult(stage.stage_id, "patch", patch_dir.name, "completed", replay_details)

    try:
        stage2_kernel_backend = _stage2_kernel_backend_for_patch(context, patch_dir)
        details = _run_ported_patch_stage(
            stage.stage_id,
            patch_dir,
            backend=context.run_config.runtime.backend,
            stage2_kernel_backend=stage2_kernel_backend,
            kernel_backend_overrides=context.run_config.runtime.kernel_backend_overrides,
            stage2_native_threads=_effective_stage2_native_threads(
                stage,
                context,
                patch_count,
                stage2_kernel_backend=stage2_kernel_backend,
            ),
            stage2_checkpoint_mode=context.run_config.runtime.stage2_checkpoint_mode,
            stage2_checkpoint_interval=context.run_config.runtime.stage2_checkpoint_interval,
            stage2_debug=context.run_config.runtime.stage2_debug,
            stage4_debug=context.run_config.runtime.stage4_debug,
            strict_reference=context.run_config.compat.strict_reference,
        )
    except PortedStageError as exc:
        raise StageExecutionError(
            f"Stage {stage.stage_id} ({stage.name}) for {patch_dir.name} is not yet fully ported. "
            f"Expected output: {expected}. {exc}"
        ) from exc

    if stage.stage_id >= 2:
        _commit_stage_provenance(
            context,
            stage.stage_id,
            "patch",
            patch_dir,
        )

    return StageResult(stage.stage_id, "patch", patch_dir.name, "completed", details)


def _run_patch_stage_timed(stage: StageDef, patch_dir: Path, context: PipelineContext, patch_count: int) -> StageResult:
    t0 = time.perf_counter()
    result = _run_patch_stage(stage, patch_dir, context, patch_count)
    result.duration_sec = time.perf_counter() - t0
    return result


def _run_merged_stage(
    stage: StageDef,
    dataset_root: Path,
    context: PipelineContext,
    *,
    force_run: bool = False,
) -> StageResult:
    expected = expected_stage_artifact(stage.stage_id, "merged")
    if expected is None:
        return StageResult(stage.stage_id, "merged", dataset_root.name, "skipped", "No expected artifact mapping")

    # GRID IFG QC belongs inside Stage6, after GRID filtering.
    # Force Stage6 when the audit is missing or its configuration changed.
    if (
        stage.stage_id == 6
        and not context.dry_run
        and not context.run_config.compat.strict_reference
        and not grid_qc_audit_is_current(
            dataset_root,
            context.run_config.ifg_selection,
        )
    ):
        force_run = True

    phase_file = "phuw2.mat"

    if stage.stage_id in {7, 8}:
        phase_file = _resolve_stage78_phase_file(
            dataset_root,
            context,
        )

        if (
            not context.dry_run
            and not _phase_input_is_current(
                dataset_root,
                stage.stage_id,
                phase_file,
            )
        ):
            force_run = True

        # Direct Stage-8 execution must not use an SCLA
        # generated from a different phase input.
        if (
            stage.stage_id == 8
            and not context.dry_run
            and not _phase_input_is_current(
                dataset_root,
                7,
                phase_file,
            )
        ):
            force_run = True

    artifact = dataset_root / expected
    bundle = _required_stage_bundle(
        dataset_root,
        stage.stage_id,
        "merged",
    )
    if not bundle:
        bundle = [expected]

    bundle_complete = all(
        (dataset_root / filename).exists()
        for filename in bundle
    )

    if (
        not force_run
        and bundle_complete
    ):
        if _stage_provenance_is_current(
            context,
            stage.stage_id,
            "merged",
            dataset_root,
            phase_file=(
                phase_file
                if stage.stage_id in {7, 8}
                else None
            ),
        ):
            return StageResult(
                stage.stage_id,
                "merged",
                dataset_root.name,
                "skipped_existing",
                f"{expected} present; validated stage bundle present",
            )

        print(
            f"[STAGE{stage.stage_id}] "
            "merged outputs have missing/stale provenance; recomputing",
            flush=True,
        )

    if context.dry_run:
        return StageResult(stage.stage_id, "merged", dataset_root.name, "planned", f"Would produce {expected}")

    replay_details = _replay_from_reference(context, "merged", stage.stage_id, dataset_root)
    if replay_details is not None:
        _commit_stage_provenance(
            context,
            stage.stage_id,
            "merged",
            dataset_root,
            phase_file=(
                phase_file
                if stage.stage_id in {7, 8}
                else None
            ),
        )
        return StageResult(stage.stage_id, "merged", dataset_root.name, "completed", replay_details)

    try:
        if stage.stage_id == 5:
            details = stage5_merge_and_ifgstd(
                dataset_root,
                backend=context.run_config.runtime.backend,
                io_workers=context.run_config.runtime.io_workers,
                enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
            )
        elif stage.stage_id == 6:
            # Ensure merged stage-5 artifacts exist before unwrapping.
            if not (dataset_root / "ifgstd2.mat").exists():
                stage5_merge_and_ifgstd(
                    dataset_root,
                    backend=context.run_config.runtime.backend,
                    io_workers=context.run_config.runtime.io_workers,
                    enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                )
                _commit_stage_provenance(
                    context,
                    5,
                    "merged",
                    dataset_root,
                )

            reference = resolve_reference(
                dataset_root,
                context.run_config.reference,
            )

            print(
                "[REFERENCE] "
                f"{reference.method}: "
                f"{reference.longitude:.8f}, {reference.latitude:.8f}, "
                f"r={reference.radius_m:.0f} m, n={reference.n_points}",
                flush=True,
            )

            if (
                not context.run_config.compat.strict_reference
            ):
                resolve_ifg_selection(
                    dataset_root,
                    context.run_config.ifg_selection,
                )

            details = stage6_unwrap(
                dataset_root,
                backend=context.run_config.runtime.backend,
                io_workers=context.run_config.runtime.io_workers,
                enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                triangle_path=context.run_config.tools.triangle,
                snaphu_path=context.run_config.tools.snaphu,
            )
        elif stage.stage_id == 7:
            details = stage7_calc_scla(
                dataset_root,
                backend=_kernel_backend_for_name(context, "stage7_scla", context.run_config.runtime.backend),
                chunk_ps=context.run_config.runtime.stage7_chunk_ps,
                enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                io_workers=context.run_config.runtime.io_workers,
                triangle_path=context.run_config.tools.triangle,
                phase_file=phase_file,
            )

        elif stage.stage_id == 8:

            # If Stage 8 is launched directly and Stage 7
            # belongs to another phase input, rebuild SCLA
            # first using the selected phase.
            if not _phase_input_is_current(
                dataset_root,
                7,
                phase_file,
            ):
                stage7_calc_scla(
                    dataset_root,
                    backend=_kernel_backend_for_name(
                        context,
                        "stage7_scla",
                        context.run_config.runtime.backend,
                    ),
                    chunk_ps=context.run_config.runtime.stage7_chunk_ps,
                    enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                    io_workers=context.run_config.runtime.io_workers,
                    triangle_path=context.run_config.tools.triangle,
                    phase_file=phase_file,
                )

                _write_phase_input_marker(
                    dataset_root,
                    7,
                    phase_file,
                )
                _commit_stage_provenance(
                    context,
                    7,
                    "merged",
                    dataset_root,
                    phase_file=phase_file,
                )

            details = stage8_filter_scn(
                dataset_root,
                backend=_kernel_backend_for_name(context, "stage8_edge_noise", context.run_config.runtime.backend),
                chunk_edges=context.run_config.runtime.stage8_chunk_edges,
                chunk_ps=context.run_config.runtime.stage7_chunk_ps,
                enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                io_workers=context.run_config.runtime.io_workers,
                triangle_path=context.run_config.tools.triangle,
                snaphu_path=context.run_config.tools.snaphu,
                phase_file=phase_file,
            )

        else:
            raise PortedStageError(f"No ported merged implementation for stage {stage.stage_id}")
    except PortedStageError as exc:
        raise StageExecutionError(
            f"Stage {stage.stage_id} ({stage.name}) merged execution is not yet fully ported. "
            f"Expected output: {expected}. {exc}"
        ) from exc

    # Commit Stage 7/8 phase provenance only after the
    # numerical stage has completed successfully.
    if stage.stage_id in {7, 8}:
        _write_phase_input_marker(
            dataset_root,
            stage.stage_id,
            phase_file,
        )

    _commit_stage_provenance(
        context,
        stage.stage_id,
        "merged",
        dataset_root,
        phase_file=(
            phase_file
            if stage.stage_id in {7, 8}
            else None
        ),
    )

    return StageResult(
        stage.stage_id,
        "merged",
        dataset_root.name,
        "completed",
        details,
    )


def _run_merged_stage_timed(
    stage: StageDef,
    dataset_root: Path,
    context: PipelineContext,
    *,
    force_run: bool = False,
) -> StageResult:
    t0 = time.perf_counter()
    result = _run_merged_stage(stage, dataset_root, context, force_run=force_run)
    result.duration_sec = time.perf_counter() - t0
    return result


def _selected_stages(start_step: int, end_step: int) -> list[StageDef]:
    return [s for s in STAGE_DEFS if start_step <= s.stage_id <= end_step]


def run_pipeline(context: PipelineContext) -> PipelineReport:
    # The selected Stage7/8 phase product is valid only for
    # this invocation.
    context.stage78_phase_file = None

    dataset: DatasetLayout = ensure_stage1_dataset(context)
    report = PipelineReport()
    patch_count = len(dataset.patches)
    merged_stage5 = StageDef(5, "Merge patches", "merged")

    def _fmt_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(int(round(seconds)), 60)
        if minutes < 60:
            return f"{minutes:d}m{sec:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}h{minutes:02d}m"

    with HybridExecutor(
        io_workers=context.run_config.runtime.io_workers,
        cpu_workers=context.run_config.runtime.cpu_workers,
    ) as executor:
        for stage in _selected_stages(context.start_step, context.end_step):
            task_kind = _task_kind_for_stage(stage, context, patch_count=patch_count)
            if stage.scope == "patch":
                stage_total = len(dataset.patches)
                stage_started = time.perf_counter()

                print(
                    f"\n[STAGE{stage.stage_id}] {stage.name}: "
                    f"{stage_total} patch(es), task_kind={task_kind}",
                    flush=True,
                )

                if _stage2_uses_full_cpu_default(stage, context):
                    for patch_index, patch_dir in enumerate(dataset.patches, start=1):
                        print(
                            f"[STAGE{stage.stage_id}] START "
                            f"{patch_index}/{stage_total} {patch_dir.name}",
                            flush=True,
                        )
                        try:
                            result = _run_patch_stage_timed(
                                stage, patch_dir, context, patch_count
                            )
                        except Exception as exc:  # pragma: no cover
                            result = StageResult(
                                stage_id=stage.stage_id,
                                scope="patch",
                                target=patch_dir.name,
                                status="failed",
                                details=str(exc),
                            )
                        report.add(result)
                        elapsed = time.perf_counter() - stage_started
                        mean_per_patch = elapsed / patch_index
                        eta = mean_per_patch * (stage_total - patch_index)
                        pct = 100.0 * patch_index / max(1, stage_total)
                        print(
                            f"[STAGE{stage.stage_id}] "
                            f"{patch_index}/{stage_total} ({pct:6.2f}%) "
                            f"{patch_dir.name} {result.status} | "
                            f"patch={_fmt_duration(result.duration_sec or 0.0)} | "
                            f"elapsed={_fmt_duration(elapsed)} | "
                            f"ETA={_fmt_duration(eta)}",
                            flush=True,
                        )

                else:
                    configured_active = _configured_cpu_workers(context)
                    stage_active_limit = configured_active

                    if stage.stage_id == 2:
                        stage2_threads = _effective_stage2_native_threads(
                            stage,
                            context,
                            patch_count,
                            stage2_kernel_backend=(
                                context.run_config.runtime.stage2_kernel_backend
                            ),
                        )

                        if stage2_threads > 0:
                            cpu_budget = max(1, os.cpu_count() or 1)
                            stage_active_limit = min(
                                configured_active,
                                max(1, cpu_budget // stage2_threads),
                            )

                        print(
                            "[STAGE2] CPU budget scheduler: "
                            f"cpu={os.cpu_count() or 1}, "
                            f"configured_workers={configured_active}, "
                            f"native_threads={stage2_threads}, "
                            f"active_limit={stage_active_limit}",
                            flush=True,
                        )

                    patch_iter = iter(dataset.patches)
                    future_to_patch: dict[Future, Path] = {}
                    pending: set[Future] = set()

                    def _submit_next_patch() -> bool:
                        try:
                            patch_dir = next(patch_iter)
                        except StopIteration:
                            return False

                        fut = executor.submit(
                            task_kind,
                            _run_patch_stage_timed,
                            stage,
                            patch_dir,
                            context,
                            patch_count,
                        )
                        future_to_patch[fut] = patch_dir
                        pending.add(fut)
                        return True

                    for _ in range(min(stage_active_limit, stage_total)):
                        if not _submit_next_patch():
                            break

                    completed = 0
                    results_by_name: dict[str, StageResult] = {}
                    heartbeat_interval = 60.0

                    while pending:
                        done, still_pending = wait(
                            pending,
                            timeout=heartbeat_interval,
                            return_when=FIRST_COMPLETED,
                        )
                        pending = set(still_pending)

                        if not done:
                            elapsed = time.perf_counter() - stage_started
                            active = sum(1 for fut in pending if fut.running())
                            queued = max(
                                0,
                                stage_total - completed - len(pending),
                            )
                            pct = 100.0 * completed / max(1, stage_total)
                            print(
                                f"[STAGE{stage.stage_id}] HEARTBEAT "
                                f"{completed}/{stage_total} ({pct:6.2f}%) | "
                                f"active={active} queued={queued} | "
                                f"elapsed={_fmt_duration(elapsed)}",
                                flush=True,
                            )
                            continue

                        for fut in done:
                            patch_dir = future_to_patch[fut]
                            try:
                                result = fut.result()
                            except Exception as exc:  # pragma: no cover
                                result = StageResult(
                                    stage_id=stage.stage_id,
                                    scope="patch",
                                    target=patch_dir.name,
                                    status="failed",
                                    details=str(exc),
                                )

                            results_by_name[patch_dir.name] = result
                            completed += 1
                            elapsed = time.perf_counter() - stage_started
                            mean_per_patch = elapsed / completed
                            eta = mean_per_patch * (stage_total - completed)
                            pct = 100.0 * completed / max(1, stage_total)

                            print(
                                f"[STAGE{stage.stage_id}] "
                                f"{completed}/{stage_total} ({pct:6.2f}%) "
                                f"{patch_dir.name} {result.status} | "
                                f"patch={_fmt_duration(result.duration_sec or 0.0)} | "
                                f"elapsed={_fmt_duration(elapsed)} | "
                                f"ETA={_fmt_duration(eta)}",
                                flush=True,
                            )

                            _submit_next_patch()

                    for patch_dir in dataset.patches:
                        report.add(results_by_name[patch_dir.name])
                if stage.stage_id == 5 and context.end_step >= 5:
                    try:
                        result = _run_merged_stage_timed(merged_stage5, dataset.root, context)
                        report.add(result)
                    except Exception as exc:  # pragma: no cover
                        report.add(
                            StageResult(
                                stage_id=merged_stage5.stage_id,
                                scope="merged",
                                target=dataset.root.name,
                                status="failed",
                                details=str(exc),
                            )
                        )

                        # === STAGE5_MERGE_FAILFAST_V1 ===
                        # Stage 6-8 depend on merged Stage-5 products.
                        break
            else:
                try:
                    if task_kind == "cpu":
                        result = executor.submit(
                            "cpu",
                            _run_merged_stage_timed,
                            stage,
                            dataset.root,
                            context,
                            force_run=False,
                        ).result()
                    else:
                        result = _run_merged_stage_timed(stage, dataset.root, context)
                    report.add(result)
                except Exception as exc:  # pragma: no cover
                    report.add(
                        StageResult(
                            stage_id=stage.stage_id,
                            scope="merged",
                            target=dataset.root.name,
                            status="failed",
                            details=str(exc),
                        )
                    )
    # === ENGINEERING_POSTPROCESS_AUTO_V1 ===
    if (
        context.end_step >= 8
        and not context.dry_run
        and context.run_config.postprocess.enabled
    ):
        stage8_ok = any(
            result.stage_id == 8
            and result.scope == "merged"
            and result.status in {"completed", "skipped_existing"}
            for result in report.results
        )

        upstream_failed = any(
            result.status == "failed"
            and result.stage_id <= 8
            for result in report.results
        )

        if stage8_ok and not upstream_failed:
            from pystamps.pipeline.postprocess_runner import (
                run_engineering_postprocess,
            )

            started = time.perf_counter()

            try:
                details = run_engineering_postprocess(
                    dataset.root,
                    context.run_config.postprocess,
                )

                report.add(
                    StageResult(
                        stage_id=9,
                        scope="merged",
                        target="outputs",
                        status="completed",
                        details=details,
                        duration_sec=time.perf_counter() - started,
                    )
                )

            except Exception as exc:
                report.add(
                    StageResult(
                        stage_id=9,
                        scope="merged",
                        target="outputs",
                        status="failed",
                        details=str(exc),
                        duration_sec=time.perf_counter() - started,
                    )
                )

    return report
