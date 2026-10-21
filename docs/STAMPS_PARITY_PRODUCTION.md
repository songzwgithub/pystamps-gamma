# StaMPS-parity GAMMA SBAS production path

This note documents the large-project SBAS scientific processing path used by
pySTAMPS-GAMMA, distinguishing source-level StaMPS model parity from numerical
or measured-HPC approximations. It is **not** a claim of bit-for-bit identity
with MATLAB StaMPS, and it is **not** an assertion that GNSS and InSAR trends
must agree without reference-frame alignment.

## Scientific processing chain

```text
GAMMA RSLC / SB IFGs
  -> Stage 1–6: PS selection, SB processing, unwrapping
  -> phuw_sb2.mat, phuw2.mat
  -> Stage 7: ps_calc_scla(1,1) -> ps_smooth_scla(1) -> ps_calc_scla(0,1)
  -> scla2.mat
  -> Stage 8: ps_scn_filt_krig-compatible temporal/spatial SCN estimation
  -> scn2.mat
  -> scientific postprocess (corrected LOS time series and velocities)
  -> optional engineering GIS / figures / rasters
```

### Stage 7: single-master/SB SCLA

The expected implementation tag is `MATLAB_STAMPS_SB_STAGE7_PARITY_V1`.
The three passes use:

- `ps_calc_scla(1,1)`: SB unwrapped interferogram phases (`phuw_sb2.mat`)
  with the selected SB covariance matrix.
- `ps_smooth_scla(1)`: StaMPS SCLA smoothing procedure.
- `ps_calc_scla(0,1)`: existing single-master acquisition phase `phuw2.mat`;
  reconstruct the acquisition perpendicular-baseline geometry from the SB
  network, fit K from sequential acquisitions, and estimate `C_ps_uw` with
  acquisition-space `sm_cov`.

Do **not** invert the SB IFGs a second time to construct an alternative phase
series for the final pass.

### Stage 8: StaMPS SCN kriging

Use `scn_kriging_flag='y'` in the dataset's `parms.mat`; the default
`scn_kriging_flag='n'` preserves the original (non-kriging) Stage-8 branch.
For the StaMPS kriging branch, use the original `phuw2.mat` and avoid stacking
external corrections onto its internal epoch deramp:

```yaml
gacos:
  enabled: false
post_unwrap_deramp:
  enabled: false
```

Major Stage-8 operations:

1. Subtract Stage-7 SCLA and `C_ps_uw` from the acquisition phase; apply the
   StaMPS-style all-epoch spatial plane removal.
2. Construct Delaunay PS edges and fit the edge deformation design
   `[years^2, years, sin(2π years), cos(2π years)-1, 1]`.
3. Fit StaMPS exponential temporal variograms, then ordinary-krige the edge
   residuals and retain high-pass edge phase.
4. Reconstruct node phase by solving the **same incidence-matrix least-squares
   objective** as MATLAB's `A\\dph_hpt`; the scalable solver is grounded-
   Laplacian AMG-preconditioned CG with explicit convergence diagnostics.
5. Remove the StaMPS spatial polynomial, compute the spatial exponential
   variogram using sampled PS and all eligible unordered pairs within the
   variogram radius, and fit parameters without an artificial range/sill cap.
6. **Final spatial estimate is exact local ordinary kriging at every PS**;
   there is no regular-grid interpolation in this branch. Cache the first-
   epoch nearest 50 PS candidates, apply strict `<4a` radius selection and
   StaMPS ordinary-kriging covariance/nugget semantics for each epoch.
7. Restore the spatial plane and normalize the reference PS and master epoch;
   save `scn2.mat`.

`ordinary_kriging_tiled` is a **measured HPC approximation** to per-edge
StaMPS temporal ordinary kriging, not mathematically identical per-edge
parameter fitting. In the verified 5 km-tile project, exact-reference QA
reported RMS **0.00024 rad** and P95 **0.00053 rad**. These measured errors
should not be extrapolated automatically to other datasets or tilings.

### Large-project acceptance example (2026-10-21)

The accepted run used 4,292,662 PS, 171 input SB interferograms
(168 used by Stage 7), 49 acquisitions (2025-01-02 through 2026-08-20),
12,877,943 Delaunay edges, 15,000 spatial variogram PS, 9,459,120 valid
unordered pairs within 30 km, and full-PS exact local spatial kriging.

- Stage-7 debug `status=completed`,
  `implementation=MATLAB_STAMPS_SB_STAGE7_PARITY_V1`.
- Stage-8 `temporal mode=ordinary_kriging_tiled`,
  `spatial mode=StaMPS exact local / all PS`.
- `scn2.mat` output shape `(4292662, 49)` and both first-PS SCN
  reference and master-epoch SCN exactly zero.

This is a project-specific numerical execution record, not external
geophysical validation of every PS.

## Run instructions

### New project: configure Stage 7–8

For a fresh SBAS project, generate the canonical `scn_kriging_flag='y'`
through the GAMMA Stage-1 preparation CLI option `--scn-kriging`, or set the
canonical switch in the dataset's `parms.mat`. Confirm that Stage 6 is complete
before running Stage 7/8.

```bash
pystamps --config config/production.yaml run \
  --dataset /absolute/path/to/stamps \
  --start-step 7 --end-step 8
```

The automatic engineering postprocess runs after successful/reused Stage 8
when `postprocess.enabled: true`. Existing outputs/provenance may cause
stages to be skipped or to be recalculated; review the log.

### Resuming an interrupted *same-input* Stage 8

```bash
export PYSTAMPS_SCN_KRIG_RESUME=1
export PYSTAMPS_SCN_KRIG_REUSE_TEMPORAL=1
export PYSTAMPS_SCN_KRIG_REUSE_PH_HPT=1
export PYSTAMPS_SCN_KRIG_REUSE_EXACT_SPATIAL=1
export PYSTAMPS_SCN_KRIG_EXACT_WORKERS=16
export PYSTAMPS_SCN_KRIG_EXACT_CHUNK=1024
export PYSTAMPS_SCN_KRIG_NEIGHBOR_QUERY_CHUNK=200000
export PYSTAMPS_SCN_KRIG_VARIO_PAIR_CHUNK=2000000
```

These flags only authorize resuming complete prior checkpoints for **the
same scientific input and settings**. Deactivate/remove them when starting
a different Stage-7 product. The checkpoint dependency order is:

```text
phuw2.mat -> scla2.mat -> dph_lpt/dph_hpt_edge -> ph_hpt
             -> spatial_exact -> scn2.mat
```

If `scla2.mat` or a scientific configuration changes, invalidate all derived
Stage-8 checkpoints; timestamps alone are insufficient as an immutable
scientific fingerprint.

### Engineering postprocess on an existing completed `scn2.mat`

The pipeline's selectable stages are **1–8**, not 1–9. Therefore
`--start-step 9 --end-step 9` selects no stages and prints `[]`.
To avoid triggering Stage-8 provenance checks or recomputation, run the
postprocessor directly:

```bash
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
from pathlib import Path
from pystamps.config import load_config
from pystamps.pipeline.postprocess_runner import run_engineering_postprocess

root = Path('/absolute/path/to/stamps')
cfg = load_config('config/production.yaml')
print(run_engineering_postprocess(root, cfg.postprocess))
PY
```

Output follows the actual `postprocess` configuration (HDF5/CSV,
figures, Shapefile, raster, optional GeoPackage, optional vertical
approximation). The LOS-to-vertical approximation is **not** a direct
3-D reconstruction.

## Dependencies and checks

Install pySTAMPS-GAMMA with optional kriging dependencies
(e.g. `python -m pip install '.[kriging]'` from the repository root) and
provide the external Triangle executable used to construct PS edges.

Before delivery, verify Stage-7 implementation/debug, Stage-8 convergence
and temporal QA, exact local spatial branch, SCN reference/master invariants,
file lineage, successful postprocess completion, and reasonable regional
spatial fields. GNSS comparisons are independent external observations and
must account for LOS projection, reference frame, and scale before
quantitative equality is expected.

## Scope of this integration

The release integrates the locally executed *exact-spatial* kriging module
using a minimal dispatch and configuration patch based on the latest
`origin/main`. It intentionally does **not** bundle experimental auto-SCN-
window code, delivery-only master interpolation, unrelated Stage-6 changes,
backup files, or full project datasets.
