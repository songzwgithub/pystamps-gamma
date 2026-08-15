# pySTAMPS-GAMMA

**pySTAMPS-GAMMA** is a Python/Rust StaMPS-compatible processing workflow for GAMMA SBAS/InSAR projects.

The production workflow is project-local: create a working directory inside a GAMMA project, generate `pystamps.yaml` once, and run Stage 1–8 directly with the `pystamps` command.

## Quick start

Typical project layout:

```text
PROJECT/
├── RSLC_cropped/          # or RSLC/
├── RSLC_tab
├── MLI_dir/               # or MLI/
├── DEM_prep/              # or DEM/
├── DIFF/                  # or DIFF_dir/
├── itab
└── pystamps/              # pySTAMPS working directory
```

Generate a project-local configuration and run the full workflow:

```bash
cd PROJECT/pystamps
pystamps -g

pystamps run \
  --start-step 1 \
  --end-step 8
```

For normal production use, a repository-level `--config` path and an explicit `--dataset` path are not required.

---

## 1. Repository structure

```text
pystamps-main/
├── Cargo.lock
├── Cargo.toml
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── pyproject.toml
├── setup.cfg
├── setup.py
│
├── config/
│   └── production.yaml
│
├── crates/                    # Rust crates / native components
├── src/                       # PyO3/Rust extension source
│
├── pystamps/                  # Main Python package
│   ├── compat/
│   ├── data/
│   │   ├── production.yaml    # Installed template used by `pystamps -g`
│   │   └── *.json
│   ├── io/
│   ├── kernels/
│   ├── notebooks/
│   ├── pipeline/
│   ├── prep/
│   └── runtime/
│
├── scripts/                   # Formal user-facing utilities only
│   ├── pipeline/
│   │   └── prepare_gamma_sbas.py
│   ├── corrections/
│   │   ├── prepare_deramp.py
│   │   └── prepare_gacos.py
│   ├── postprocess/
│   │   └── postprocess.py
│   └── validate_audit.py
│
└── tests/
    ├── test_*.py
    └── scripts/               # parity / development / validation utilities
```

Development, project-specific, historical and parity helpers belong under:

```text
tests/scripts/
tests/scripts/dev/
```

Generated files are not source-release content:

```text
target/
build/
dist/
__pycache__/
*.pyc
*.so
_archive/
```

---

## 2. Installation

Linux is recommended for the production GAMMA/SNAPHU workflow.

Typical external requirements:

```text
GAMMA Remote Sensing Software
SNAPHU
Python
Rust toolchain
C/C++ build environment
GDAL / PROJ for GIS workflows
```

GAMMA and SNAPHU are external software and are not distributed with this repository.

Clone and install:

```bash
git clone https://github.com/songzwgithub/pystamps-gamma.git
cd pystamps-gamma

python -m pip install --upgrade pip
pip install -e .
```

Verify:

```bash
python -c "import pystamps; print(pystamps.__version__)"
pystamps --help
```

Manual Rust build:

```bash
cargo build --release
```

Rust build products are written under `target/` and should not be committed as source.

---

## 3. Local configuration

### 3.1 Generate `pystamps.yaml`

The recommended interface is:

```bash
pystamps -g
```

This creates:

```text
./pystamps.yaml
```

Other supported forms:

```bash
# Custom filename
pystamps -g project.yaml

# Overwrite an existing generated config
pystamps -g --force-config
```

The production template is distributed inside the installed package as:

```text
pystamps/data/production.yaml
```

Users therefore do not need to locate `config/production.yaml` inside the source repository.

### 3.2 Config discovery

Configuration priority is:

```text
explicit --config
    ↓
./pystamps.yaml
./pystamps.yml
./production.yaml
./production.yml
    ↓
RunConfig defaults
```

Normal production usage is therefore:

```bash
cd PROJECT/pystamps
pystamps run --start-step 1 --end-step 8
```

---

## 4. Project paths

The generated configuration starts with:

```yaml
paths:
  work_dir: null
  data_dir: null

  rslc_dir: null
  diff_dir: null
  mli_dir: null
  dem_dir: null

  rslc_tab: null
  itab: null
```

With these defaults:

```text
work_dir = current working directory
data_dir = parent directory of work_dir
```

For example, when running from `PROJECT/pystamps`:

```text
work_dir = PROJECT/pystamps
data_dir = PROJECT
```

Supported automatic directory names:

| Input | Accepted names |
|---|---|
| RSLC | `RSLC_cropped/`, `RSLC/` |
| interferograms | `DIFF/`, `DIFF_dir/` |
| multilooked images | `MLI_dir/`, `MLI/` |
| DEM / geometry | `DEM_prep/`, `DEM/` |
| acquisition table | `RSLC_tab` |
| SB network | `itab` |

If both `RSLC/` and `RSLC_cropped/` are present, the resolver uses project information including `RSLC_tab` rather than blindly selecting by directory name.

If both `DIFF/` and `DIFF_dir/` are present, the resolver checks consistency with the current `itab` network.

If automatic discovery remains ambiguous, set the path explicitly:

```yaml
paths:
  work_dir: null
  data_dir: null

  rslc_dir: RSLC_cropped
  diff_dir: DIFF_dir
  mli_dir: MLI_dir
  dem_dir: DEM_prep

  rslc_tab: RSLC_tab
  itab: itab
```

Relative component paths are interpreted relative to `data_dir`.

---

## 5. Normal command-line usage

Full Stage 1–8:

```bash
pystamps run \
  --start-step 1 \
  --end-step 8
```

Stage 1–5:

```bash
pystamps run --start-step 1 --end-step 5
```

Stage 6 only:

```bash
pystamps run --start-step 6 --end-step 6
```

Stage 7 only:

```bash
pystamps run --start-step 7 --end-step 7
```

Stage 8 only:

```bash
pystamps run --start-step 8 --end-step 8
```

Stage 6–8:

```bash
pystamps run --start-step 6 --end-step 8
```

Completed stages may be detected automatically and reported as:

```text
status: skipped_existing
```

rather than being recomputed.

Explicit overrides remain available when needed:

```bash
pystamps \
  --config custom.yaml \
  run \
  --dataset /PROJECT/pystamps_test \
  --data-dir /PROJECT \
  --start-step 1 \
  --end-step 8
```

Priority is:

```text
CLI override
    ↓
YAML configuration
    ↓
automatic project discovery
```

---

## 6. Startup path report

A normal run reports the resolved configuration and GAMMA inputs:

```text
[CONFIG] /PROJECT/pystamps/pystamps.yaml

============================================================
pySTAMPS PROJECT PATHS
============================================================
work_dir : /PROJECT/pystamps [cwd]
data_dir : /PROJECT [work_dir.parent]

GAMMA INPUTS
RSLC     : /PROJECT/RSLC_cropped
DIFF     : /PROJECT/DIFF
MLI      : /PROJECT/MLI_dir
DEM      : /PROJECT/DEM_prep
RSLC_tab : /PROJECT/RSLC_tab
itab     : /PROJECT/itab
============================================================
```

Check this report before the first production run.

---

## 7. Stage 1–5: PS estimation and selection

Stages 1–5 perform project preparation, phase-stability estimation, probabilistic PS selection and weed/final PS selection.

The production workflow follows StaMPS-style phase-stability logic rather than selecting the final PS set from a single fixed GAMMA coherence threshold.

Key concepts:

```text
candidate preparation
phase stability
gamma estimation
random-phase probability
density control
PS selection
spatial weed
temporal weed
final PS merge
```

Run:

```bash
pystamps run --start-step 1 --end-step 5
```

Separate legacy Stage 1–5 repository shell wrappers are not required for normal production processing.

---

## 8. Stage 6: SB unwrapping and inversion

Stage 6 performs:

```text
SB interferometric phase
        │
        ▼
3-D unwrapping preparation
        │
        ▼
SNAPHU
        │
        ▼
unwrapped SB interferograms
        │
        ▼
SB network inversion
        │
        ▼
single-master acquisition phase
```

Run:

```bash
pystamps run --start-step 6 --end-step 6
```

Typical Stage 6 products include:

```text
phuw_sb2.mat
phuw2.mat
phuw_sb_res2.mat
```

Stage 6 includes automatic CPU/RAM-aware scheduling, parallel independent-IFG unwrapping, project-relative IFG quality auditing, post-unwrapped network consistency evaluation, network-connectivity protection, and final re-inversion using retained interferograms.

Production IFG selection is project-relative:

```yaml
ifg_selection:
  mode: auto
```

Project-specific interferogram indices are not intended to be hard-coded into the production configuration.

---

## 9. Reference selection

Automatic reference-region selection is supported.

Typical configuration:

```yaml
reference:
  mode: auto
  longitude: null
  latitude: null
  radius_m: 500.0
```

If longitude and latitude are provided, a fixed reference can be used.

An automatically selected high-quality reference region is a **relative InSAR reference**. It is not proof that the physical deformation at that location is exactly zero.

---

## 10. Stage 7: SCLA

Stage 7 estimates spatially correlated look-angle / topographic phase error.

```bash
pystamps run --start-step 7 --end-step 7
```

Typical products:

```text
scla_sb2.mat
scla_smooth_sb2.mat
scla2.mat
```

Typical fields include:

```text
ph_scla
K_ps_uw
C_ps_uw
```

---

## 11. Stage 8: spatially correlated noise

Stage 8 estimates spatially correlated noise after preceding correction terms.

```bash
pystamps run --start-step 8 --end-step 8
```

Typical output:

```text
scn2.mat
```

with fields such as:

```text
ph_scn_slave
ph_hpt
ph_ramp
```

Conceptually:

```text
phase_corrected
    = phase_unwrapped
    - SCLA
    - constant phase term
    - SCN
```

---

## 12. DERAMP and optional GACOS

Formal helpers:

```bash
python scripts/corrections/prepare_deramp.py --help
python scripts/corrections/prepare_gacos.py --help
```

GACOS is optional and should be treated as an atmospheric-correction branch rather than an unconditional requirement.

For GAMMA LOS-vector geometry:

```text
lv_theta = LOS elevation angle above horizontal
incidence_angle = 90° - lv_theta
cos(incidence_angle) = sin(lv_theta)
```

Do not substitute a DEM-local incidence-angle raster without verifying that its geometric definition is equivalent.

---

## 13. Post-processing

Formal helper:

```bash
python scripts/postprocess/postprocess.py --help
```

Phase-to-LOS conversion:

```text
D_LOS = -phase * wavelength / (4*pi)
```

For millimetres:

```text
D_LOS_mm = -phase * wavelength / (4*pi) * 1000
```

Sign convention:

```text
positive LOS displacement = toward satellite
negative LOS displacement = away from satellite
```

Where horizontal deformation is assumed negligible:

```text
D_vertical = D_LOS / cos(incidence_angle)
V_vertical = V_LOS / cos(incidence_angle)
```

This is a zero-horizontal-motion approximation, not a full LOS decomposition.

---

## 14. Formal user-facing scripts

The release `scripts/` directory should contain only user-facing production utilities:

```text
scripts/pipeline/prepare_gamma_sbas.py
scripts/corrections/prepare_deramp.py
scripts/corrections/prepare_gacos.py
scripts/postprocess/postprocess.py
scripts/validate_audit.py
```

Inspect options with:

```bash
python scripts/pipeline/prepare_gamma_sbas.py --help
python scripts/corrections/prepare_deramp.py --help
python scripts/corrections/prepare_gacos.py --help
python scripts/postprocess/postprocess.py --help
python scripts/validate_audit.py --help
```

Parity, benchmark, historical and project-specific helpers belong under:

```text
tests/scripts/
tests/scripts/dev/
```

They are not part of the normal end-user workflow.

---

## 15. Testing and release checks

Run the test suite:

```bash
python -m pytest -q
```

Check for generated artifacts before release:

```bash
find . \
  \( \
    -name '*.so' \
    -o -name '*.pyc' \
    -o -name '*.bak' \
    -o -name '*.old' \
    -o -name '*.tmp' \
    -o -name '*broken*' \
    -o -name '*before_*' \
    -o -name '__pycache__' \
  \) \
  -print
```

Typical package validation:

```bash
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
```

The installed package must contain:

```text
pystamps/data/production.yaml
```

because it is the template used by:

```bash
pystamps -g
```

---

## 16. Scientific scope

pySTAMPS-GAMMA is intended to reproduce the major processing logic of the StaMPS SBAS workflow while providing a Python-oriented implementation, GAMMA integration, automatic project discovery, quality auditing and native acceleration.

Important interpretation notes:

- automatic IFG rejection is project-relative quality control, not a universal fixed index list;
- automatic reference selection identifies a high-quality relative reference, not guaranteed zero deformation;
- GACOS should be validated for each project rather than assumed to improve every dataset;
- vertical displacement from one LOS geometry requires an explicit horizontal-motion assumption;
- processing parameters should be evaluated for the sensor, multilook geometry, temporal network and deformation regime.

---

## 17. Citation

If pySTAMPS-GAMMA is used in scientific work, cite this software repository together with the relevant references for StaMPS, GAMMA, SNAPHU and any atmospheric-correction method or dataset used in the processing configuration.
