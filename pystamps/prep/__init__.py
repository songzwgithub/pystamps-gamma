"""Input preparation modules for external InSAR processors."""

from .gamma_binary import (
    GammaRasterLayout,
    infer_coherence_dtype,
    iter_gamma_raster_blocks,
    normalize_complex_unit,
    read_gamma_coherence,
    read_gamma_raster,
    resolve_gamma_raster_layout,
    sample_gamma_raster,
)
from .gamma_candidates import (
    CandidateConfig,
    CandidateResult,
    extract_amplitude_candidates,
    extract_candidates_from_project,
    save_candidate_result,
)
from .gamma_geometry import (
    CandidateRadarGeometry,
    GammaBaselineModel,
    GammaRadarGeometry,
    build_radar_geometry,
    calculate_bperp_column,
    calculate_bperp_matrix,
    calculate_candidate_geometry,
    read_baseline_model,
)
from .gamma_sbas import (
    GammaAcquisition,
    GammaInputError,
    GammaInterferogram,
    GammaSbasProject,
    inspect_gamma_sbas_project,
    load_gamma_sbas_project,
)
from .gamma_lonlat import (
    GammaLonLatError,
    GammaLonLatResult,
    ensure_gamma_radar_lonlat,
)
from .gamma_ps_optimization import (
    PSOptimizationConfig,
    PSSelectionResult,
    choose_automatic_patch_config,
    save_ps_selection,
    select_ps_candidates,
)

__all__ = [
    "CandidateConfig",
    "CandidateRadarGeometry",
    "CandidateResult",
    "GammaAcquisition",
    "GammaBaselineModel",
    "GammaInputError",
    "GammaInterferogram",
    "GammaRadarGeometry",
    "GammaRasterLayout",
    "GammaSbasProject",
    "build_radar_geometry",
    "calculate_bperp_column",
    "calculate_bperp_matrix",
    "calculate_candidate_geometry",
    "extract_amplitude_candidates",
    "extract_candidates_from_project",
    "infer_coherence_dtype",
    "inspect_gamma_sbas_project",
    "iter_gamma_raster_blocks",
    "load_gamma_sbas_project",
    "normalize_complex_unit",
    "read_baseline_model",
    "read_gamma_coherence",
    "read_gamma_raster",
    "resolve_gamma_raster_layout",
    "sample_gamma_raster",
    "save_candidate_result",
    "GammaLonLatError",
    "GammaLonLatResult",
    "ensure_gamma_radar_lonlat",
    "PSOptimizationConfig",
    "PSSelectionResult",
    "choose_automatic_patch_config",
    "save_ps_selection",
    "select_ps_candidates",
]
