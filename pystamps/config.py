from __future__ import annotations

import json
from importlib.resources import files as resource_files
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_RUNTIME_BACKEND_ALIASES = {
    "auto": "auto",
    "threads": "threads",
    "thread": "threads",
    "io": "threads",
    "processes": "processes",
    "process": "processes",
    "cpu": "processes",
    "gpu": "gpu",
    "native": "native",
}

_KERNEL_BACKEND_ALIASES = {
    "auto": "auto",
    "python": "python",
    "cpu": "python",
    "native": "native",
    "gpu": "cuda",
    "cuda": "cuda",
}


@dataclass(slots=True)
class RuntimeConfig:
    io_workers: int = 8
    cpu_workers: int = 0
    backend: str = "auto"
    stage2_kernel_backend: str = "native"
    stage2_patch_backend_overrides: dict[str, str] = field(default_factory=dict)
    kernel_backend_overrides: dict[str, str] = field(default_factory=dict)
    stage2_native_threads: int = 0
    stage7_chunk_ps: int = 100_000
    stage8_chunk_edges: int = 200_000
    enable_mat_stage_cache: bool = True
    stage2_checkpoint_mode: str = "final"
    stage2_checkpoint_interval: int = 1
    stage2_debug: bool = False
    stage4_debug: bool = False


@dataclass(slots=True)
class ToleranceConfig:
    rtol: float = 1e-5
    atol: float = 1e-7
    wrap_equivalence: bool = True
    wrap_period: float = 2.0 * 3.141592653589793
    wrap_keys: tuple[str, ...] = ("ph_uw", "ph", "dph_noise", "dph_space_uw")


@dataclass(slots=True)
class IFGSelectionConfig:
    # auto: robust project-relative quality selection
    # manual: use drop_ifg_index below
    # none: keep every IFG
    mode: str = "auto"

    robust_z_threshold: float = 4.0
    contextual_z_threshold: float = 3.5
    tail_quantile: float = 0.99
    min_bad_metrics: int = 2
    max_drop_fraction: float = 0.05
    preserve_network: bool = True
    temporal_bins: int = 8


    # Stage6 GRID-based multi-metric QC.
    grid_qc_enabled: bool = True
    grid_qc_metric_bad_quantile: float = 0.90
    grid_qc_score_tail_fraction: float = 0.02
    grid_qc_score_z_threshold: float = 2.5
    grid_qc_extreme_z_threshold: float = 4.5
    grid_qc_min_bad_metrics: int = 2
    grid_qc_max_drop_fraction: float = 0.05
    grid_qc_preserve_network: bool = True

    # Deterministic QC subsampling only; scientific Stage6 arrays
    # themselves are never subsampled.
    grid_qc_sample_ps: int = 20000
    grid_qc_sample_pairs: int = 30000
    grid_qc_closure_sample_ps: int = 6000
    grid_qc_max_triangles: int = 4000

    # GRID-QC V3 graph/node attribution.
    grid_qc_node_context_enabled: bool = True
    grid_qc_node_min_degree: int = 3
    grid_qc_node_candidate_fraction: float = 0.50
    grid_qc_edge_excess_z_threshold: float = 2.5
    grid_qc_clustered_edge_excess_z_threshold: float = 3.5
    grid_qc_low_degree_score_z_threshold: float = 4.0

    # FINAL IFG-QC.
    # Default False preserves backward compatibility;
    # production.yaml explicitly enables it.
    final_ifg_qc_enabled: bool = True

    final_qc_msd_strong_percentile: float = 0.975
    final_qc_msd_extreme_percentile: float = 0.990

    final_qc_network_strong_percentile: float = 0.975
    final_qc_network_extreme_percentile: float = 0.990

    final_qc_max_drop_fraction: float = 0.05
    final_qc_preserve_network: bool = True
    final_qc_fail_on_cap: bool = True

    final_qc_chunk_ifg: int = 8

    drop_ifg_index: tuple[int, ...] = ()


@dataclass(slots=True)
class ExternalToolsConfig:
    triangle: str = "triangle"
    snaphu: str = "snaphu"



@dataclass(slots=True)
class GacosConfig:
    enabled: bool = False
    gacos_dir: str | None = None

    # auto | m | cm | mm
    product_unit: str = "auto"

    # zenith | los
    projection: str = "zenith"

    # auto | subtract | add
    sign: str = "auto"

    strict_dates: bool = True
    rebuild: bool = False

    incidence_tif: str | None = None
    incidence_deg: float | None = None

    qa_ps: int = 30000
    qa_ifg: int = 80
    chunk_ps: int = 4096
    min_valid_fraction: float = 0.995

    def __post_init__(self) -> None:
        self.product_unit = str(
            self.product_unit
        ).strip().lower()

        if self.product_unit not in {
            "auto", "m", "cm", "mm"
        }:
            raise ConfigError(
                "gacos.product_unit must be "
                "auto, m, cm, or mm"
            )

        self.projection = str(
            self.projection
        ).strip().lower()

        if self.projection not in {
            "zenith", "los"
        }:
            raise ConfigError(
                "gacos.projection must be "
                "zenith or los"
            )

        aliases = {
            "-": "subtract",
            "+": "add",
            "minus": "subtract",
            "plus": "add",
        }

        self.sign = aliases.get(
            str(self.sign).strip().lower(),
            str(self.sign).strip().lower(),
        )

        if self.sign not in {
            "auto", "subtract", "add"
        }:
            raise ConfigError(
                "gacos.sign must be "
                "auto, subtract, or add"
            )

        if self.incidence_deg is not None:
            value = float(self.incidence_deg)
            if not 0.0 < value < 90.0:
                raise ConfigError(
                    "gacos.incidence_deg must "
                    "be between 0 and 90"
                )

        if int(self.qa_ps) <= 0:
            raise ConfigError(
                "gacos.qa_ps must be positive"
            )

        if int(self.qa_ifg) <= 0:
            raise ConfigError(
                "gacos.qa_ifg must be positive"
            )

        if int(self.chunk_ps) <= 0:
            raise ConfigError(
                "gacos.chunk_ps must be positive"
            )

        value = float(
            self.min_valid_fraction
        )

        if not 0.0 < value <= 1.0:
            raise ConfigError(
                "gacos.min_valid_fraction must "
                "be in (0, 1]"
            )


# === ENGINEERING_POSTPROCESS_CONFIG_V1 ===
@dataclass(slots=True)
class PostprocessConfig:
    enabled: bool = True
    output_dir: str = "outputs"

    chunk_ps: int = 16384
    annual_min_obs: int = 6
    annual_min_span_days: float = 180.0

    figures: bool = True
    shapefile: bool = True
    timeseries_shapefile: bool = True
    geotiff: bool = True
    grid_resolution_m: float = 100.0

    # === VERTICAL_CONVERSION_CONFIG_V1 ===
    # LOS -> vertical conversion. Horizontal motion is assumed negligible.
    vertical_enabled: bool = False

    # auto | la2 | constant
    vertical_incidence_source: str = "auto"
    vertical_incidence_deg: float | None = None

    # up: uplift positive; down: subsidence positive
    vertical_positive: str = "up"

    def __post_init__(self) -> None:
        if int(self.chunk_ps) <= 0:
            raise ConfigError("postprocess.chunk_ps must be positive")
        if int(self.annual_min_obs) < 2:
            raise ConfigError("postprocess.annual_min_obs must be >= 2")
        if float(self.annual_min_span_days) <= 0:
            raise ConfigError(
                "postprocess.annual_min_span_days must be positive"
            )
        if float(self.grid_resolution_m) <= 0:
            raise ConfigError(
                "postprocess.grid_resolution_m must be positive"
            )

        self.vertical_incidence_source = str(
            self.vertical_incidence_source
        ).strip().lower()

        if self.vertical_incidence_source not in {
            "auto", "la2", "constant"
        }:
            raise ConfigError(
                "postprocess.vertical_incidence_source must be "
                "auto, la2, or constant"
            )

        self.vertical_positive = str(
            self.vertical_positive
        ).strip().lower()

        if self.vertical_positive not in {"up", "down"}:
            raise ConfigError(
                "postprocess.vertical_positive must be up or down"
            )

        if self.vertical_incidence_deg is not None:
            angle = float(self.vertical_incidence_deg)
            if not 0.0 < angle < 90.0:
                raise ConfigError(
                    "postprocess.vertical_incidence_deg must be "
                    "between 0 and 90 degrees"
                )

        if (
            self.vertical_enabled
            and self.vertical_incidence_source == "constant"
            and self.vertical_incidence_deg is None
        ):
            raise ConfigError(
                "postprocess.vertical_incidence_deg is required "
                "when vertical_incidence_source=constant"
            )


@dataclass(slots=True)
class ReferenceConfig:
    mode: str = "auto"
    longitude: float | None = None
    latitude: float | None = None
    radius_m: float = 500.0
    cell_size_m: float = 1000.0
    min_points: int = 20
    coherence_weight: float = 0.60
    error_proxy_weight: float = 0.25
    density_weight: float = 0.15

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"auto", "existing"}:
            raise ConfigError("reference.mode must be 'auto' or 'existing'")
        if (self.longitude is None) != (self.latitude is None):
            raise ConfigError(
                "reference.longitude and reference.latitude must both be set or both be null"
            )
        if self.longitude is not None:
            if not -180 <= float(self.longitude) <= 180:
                raise ConfigError("reference.longitude outside [-180, 180]")
            if not -90 <= float(self.latitude) <= 90:
                raise ConfigError("reference.latitude outside [-90, 90]")
        if self.radius_m <= 0 or self.cell_size_m <= 0 or self.min_points <= 0:
            raise ConfigError("reference radius/cell/min_points must be positive")
        weights = (self.coherence_weight, self.error_proxy_weight, self.density_weight)
        if any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ConfigError("reference score weights are invalid")


@dataclass(slots=True)
class CompatibilityConfig:
    reference_root: str | None = None
    strict_reference: bool = False


@dataclass(slots=True)
class RunConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    ifg_selection: IFGSelectionConfig = field(default_factory=IFGSelectionConfig)
    tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    gacos: GacosConfig = field(default_factory=GacosConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    compat: CompatibilityConfig = field(default_factory=CompatibilityConfig)


class ConfigError(ValueError):
    """Raised when configuration is malformed."""


def _normalize_backend_override_map(
    payload: Any,
    *,
    field_name: str,
    normalizer: Any,
) -> dict[str, str]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"'{field_name}' must be an object")
    return {
        str(key): normalizer(str(value))
        for key, value in payload.items()
    }


def normalize_runtime_backend(name: str) -> str:
    normalized = _RUNTIME_BACKEND_ALIASES.get((name or "auto").strip().lower())
    if normalized is None:
        raise ConfigError(
            f"Unsupported runtime backend '{name}'. Use: auto, threads, processes, gpu, or native"
        )
    return normalized


def normalize_kernel_backend(name: str) -> str:
    normalized = _KERNEL_BACKEND_ALIASES.get((name or "auto").strip().lower())
    if normalized is None:
        raise ConfigError(
            f"Unsupported kernel backend '{name}'. Use: auto, python, native, or cuda"
        )
    return normalized


def normalize_stage2_kernel_backend(name: str) -> str:
    normalized = normalize_kernel_backend(name)
    if normalized == "cuda":
        raise ConfigError(
            f"Unsupported stage-2 kernel backend '{name}'. Use: auto, python, or native"
        )
    return normalized


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise ConfigError("Config must be YAML or JSON")

    if not isinstance(payload, dict):
        raise ConfigError("Top-level config payload must be an object")
    return payload


def _as_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be an object")
    return value


def _load_packaged_production_raw() -> dict[str, Any]:
    # Single production-default source for config-less execution.
    resource = (
        resource_files("pystamps")
        .joinpath("data")
        .joinpath("production.yaml")
    )

    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            "Installed pySTAMPS package does not contain "
            "pystamps/data/production.yaml"
        ) from exc

    payload = yaml.safe_load(text) or {}

    if not isinstance(payload, dict):
        raise ConfigError(
            "Bundled production configuration must be an object"
        )

    return payload


def load_config(path: str | Path | None = None) -> RunConfig:
    raw = (
        _load_packaged_production_raw()
        if path is None
        else _load_raw(Path(path))
    )
    runtime_payload = _as_dict(raw, "runtime")
    runtime_norm = dict(runtime_payload)
    if "backend" in runtime_norm:
        runtime_norm["backend"] = normalize_runtime_backend(str(runtime_norm["backend"]))
    if "stage2_kernel_backend" in runtime_norm:
        runtime_norm["stage2_kernel_backend"] = normalize_stage2_kernel_backend(
            str(runtime_norm["stage2_kernel_backend"])
        )
    if "stage2_patch_backend_overrides" in runtime_norm:
        runtime_norm["stage2_patch_backend_overrides"] = _normalize_backend_override_map(
            runtime_norm.get("stage2_patch_backend_overrides"),
            field_name="runtime.stage2_patch_backend_overrides",
            normalizer=normalize_stage2_kernel_backend,
        )
    if "kernel_backend_overrides" in runtime_norm:
        runtime_norm["kernel_backend_overrides"] = _normalize_backend_override_map(
            runtime_norm.get("kernel_backend_overrides"),
            field_name="runtime.kernel_backend_overrides",
            normalizer=normalize_kernel_backend,
        )

    runtime = RuntimeConfig(**runtime_norm)
    tol_payload = _as_dict(raw, "tolerance")
    wrap_keys = tol_payload.get("wrap_keys")
    if isinstance(wrap_keys, list):
        tol_payload = {**tol_payload, "wrap_keys": tuple(str(v) for v in wrap_keys)}
    tolerance = ToleranceConfig(**tol_payload)
    ifg_selection_payload = _as_dict(
        raw,
        "ifg_selection",
    )
    if (
        "drop_ifg_index"
        in ifg_selection_payload
        and isinstance(
            ifg_selection_payload[
                "drop_ifg_index"
            ],
            list,
        )
    ):
        ifg_selection_payload = {
            **ifg_selection_payload,
            "drop_ifg_index": tuple(
                int(v)
                for v
                in ifg_selection_payload[
                    "drop_ifg_index"
                ]
            ),
        }

    ifg_selection = IFGSelectionConfig(
        **ifg_selection_payload
    )

    tools = ExternalToolsConfig(**_as_dict(raw, "tools"))
    gacos = GacosConfig(**_as_dict(raw, "gacos"))
    postprocess = PostprocessConfig(**_as_dict(raw, "postprocess"))
    reference = ReferenceConfig(**_as_dict(raw, "reference"))
    compat = CompatibilityConfig(**_as_dict(raw, "compat"))
    return RunConfig(ifg_selection=ifg_selection, 
        runtime=runtime,
        tolerance=tolerance,
        tools=tools,
        gacos=gacos,
        postprocess=postprocess,
        reference=reference,
        compat=compat,
    )
