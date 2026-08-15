from __future__ import annotations

from concurrent.futures import Future
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

    required = PATCH_STAGE_BUNDLES[1]
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

    old_resume = os.environ.get("PYSTAMPS_STAGE1_RESUME")
    os.environ["PYSTAMPS_STAGE1_RESUME"] = "1"

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
        if old_resume is None:
            os.environ.pop("PYSTAMPS_STAGE1_RESUME", None)
        else:
            os.environ["PYSTAMPS_STAGE1_RESUME"] = old_resume

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

    gacos = context.run_config.gacos

    if not bool(gacos.enabled):
        selected = "phuw2.mat"

    elif context.dry_run:
        # Dry-run must not create atmospheric products.
        selected = "phuw2_gacos.mat"

    else:
        # Lazy import: ordinary non-GACOS runs never enter
        # the atmospheric correction module.
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
    bundle = PATCH_STAGE_BUNDLES.get(stage_id, []) if scope == "patch" else MERGED_STAGE_BUNDLES.get(stage_id, [])
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
    if artifact.exists():
        return StageResult(stage.stage_id, "patch", patch_dir.name, "skipped_existing", f"{expected} present")

    if context.dry_run:
        return StageResult(stage.stage_id, "patch", patch_dir.name, "planned", f"Would produce {expected}")

    replay_details = _replay_from_reference(context, "patch", stage.stage_id, patch_dir)
    if replay_details is not None:
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
    bundle = MERGED_STAGE_BUNDLES.get(stage.stage_id, [expected])

    if (
        not force_run
        and all(
            (dataset_root / filename).exists()
            for filename in bundle
        )
    ):
        return StageResult(
            stage.stage_id,
            "merged",
            dataset_root.name,
            "skipped_existing",
            f"{expected} present",
        )

    if context.dry_run:
        return StageResult(stage.stage_id, "merged", dataset_root.name, "planned", f"Would produce {expected}")

    replay_details = _replay_from_reference(context, "merged", stage.stage_id, dataset_root)
    if replay_details is not None:
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

    with HybridExecutor(
        io_workers=context.run_config.runtime.io_workers,
        cpu_workers=context.run_config.runtime.cpu_workers,
    ) as executor:
        for stage in _selected_stages(context.start_step, context.end_step):
            task_kind = _task_kind_for_stage(stage, context, patch_count=patch_count)
            if stage.scope == "patch":
                if _stage2_uses_full_cpu_default(stage, context):
                    for patch_dir in dataset.patches:
                        try:
                            report.add(_run_patch_stage_timed(stage, patch_dir, context, patch_count))
                        except Exception as exc:  # pragma: no cover
                            report.add(
                                StageResult(
                                    stage_id=stage.stage_id,
                                    scope="patch",
                                    target=patch_dir.name,
                                    status="failed",
                                    details=str(exc),
                                )
                            )
                else:
                    futures: list[Future] = [
                        executor.submit(task_kind, _run_patch_stage_timed, stage, patch_dir, context, patch_count)
                        for patch_dir in dataset.patches
                    ]
                    for fut in futures:
                        try:
                            report.add(fut.result())
                        except Exception as exc:  # pragma: no cover
                            report.add(
                                StageResult(
                                    stage_id=stage.stage_id,
                                    scope="patch",
                                    target="unknown",
                                    status="failed",
                                    details=str(exc),
                                )
                            )
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
    return report
