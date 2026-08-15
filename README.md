# pySTAMPS-GAMMA

**pySTAMPS-GAMMA** is a Python/Rust StaMPS-compatible processing workflow for GAMMA SBAS/InSAR time-series projects.

Recommended production usage:

```bash
cd PROJECT/pystamps
pystamps -g
pystamps run --start-step 1 --end-step 8
```

<!-- STAGE1_AUTO_PREP_README_V1 -->

For normal production use, an explicit repository-level `--config` path and an explicit `--dataset` path are not required.

When Stage 1 is requested, pySTAMPS-GAMMA checks for a complete Stage-1 `PATCH_*` dataset. If Stage 1 is missing or incomplete, GAMMA inputs are discovered automatically from `data_dir` (by default `work_dir.parent`), Stage-1 preparation is executed automatically, the generated patches are re-discovered, and processing continues through the requested stages. Existing complete Stage-1 products are reused and are not regenerated.

---

## 1. Requirements

### Python

pySTAMPS-GAMMA requires:

```text
Python >= 3.12
```

A dedicated Conda environment is recommended:

```bash
conda create -n pystamps python=3.12 -y
conda activate pystamps
```

The normal package installation installs the required Python dependencies, including `rasterio` and `pyproj` used by the GACOS workflow.

### GAMMA

GAMMA Remote Sensing Software is an external dependency and is not distributed with this repository.

### Rust / Cargo

Rust is required when installing or building pySTAMPS-GAMMA from source.

```bash
cargo --version
rustc --version
```

If Rust is not installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

A compatible prebuilt wheel already contains the compiled PyO3 extension.

---

## 2. SNAPHU — required for Stage 6

**SNAPHU is required for the production Stage-6 unwrapping workflow.**

Check:

```bash
command -v snaphu
snaphu -h
```

### Conda installation

```bash
conda install -c conda-forge snaphu
```

Verify:

```bash
command -v snaphu
snaphu -h
```

### Stanford source installation

Example using SNAPHU 2.0.7:

```bash
sudo apt-get update
sudo apt-get install -y build-essential wget

cd /tmp

wget \
  https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.7.tar.gz

tar -xzf snaphu-v2.0.7.tar.gz
cd snaphu-v2.0.7/src

make
sudo install -m 0755 snaphu /usr/local/bin/snaphu
```

Verify:

```bash
command -v snaphu
snaphu -h
```

Configuration:

```yaml
tools:
  snaphu: snaphu
```

An absolute executable path can also be configured.

---

## 3. Installation

```bash
git clone https://github.com/songzwgithub/pystamps-gamma.git
cd pystamps-gamma

python -m pip install --upgrade pip
python -m pip install .
```

For normal installation, do not use `--no-deps`; dependency resolution must retain the project constraints, including `numpy>=2.2,<2.4`.

Verify:

```bash
python -c "import pystamps; print(pystamps.__version__)"
pystamps --help
```

Native extension:

```bash
python - <<'PY'
import pystamps
import pystamps.kernels._stage2_native as native
print("pySTAMPS:", pystamps.__version__)
print("native:", native.__file__)
PY
```

---

## 4. Typical GAMMA project layout

```text
PROJECT/
├── RSLC_cropped/          # or RSLC/
├── RSLC_tab
├── MLI_dir/               # or MLI/
├── DEM_prep/              # or DEM/
├── DIFF/                  # or DIFF_dir/
├── itab
└── pystamps/
```

Run:

```bash
cd PROJECT/pystamps
pystamps -g
pystamps run --start-step 1 --end-step 8
```

For a fresh project, no separate Stage-1 preparation command is required. `pystamps run --start-step 1 ...` automatically prepares Stage 1 when `PATCH_*` products are absent or incomplete.

Automatic path convention:

```text
work_dir = current pystamps directory
data_dir = work_dir.parent
```

---

## 5. Configuration discovery

```text
explicit --config
        ↓
./pystamps.yaml
./pystamps.yml
./production.yaml
./production.yml
        ↓
packaged pystamps/data/production.yaml
```

For reproducibility, keeping a project-local `pystamps.yaml` is recommended.

---

## 6. Project paths

Generated defaults:

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

Meaning:

```text
work_dir = current working directory
data_dir = parent directory of work_dir
```

Automatic directory names:

| Input | Accepted names |
|---|---|
| RSLC | `RSLC_cropped/`, `RSLC/` |
| interferograms | `DIFF/`, `DIFF_dir/` |
| multilooked images | `MLI_dir/`, `MLI/` |
| DEM / geometry | `DEM_prep/`, `DEM/` |
| acquisition table | `RSLC_tab` |
| SB network | `itab` |

Explicit paths override automatic discovery.

---

## 7. Production defaults

```yaml
runtime:
  backend: auto
  stage2_kernel_backend: native

ifg_selection:
  mode: auto
  final_ifg_qc_enabled: true
  final_qc_preserve_network: true

reference:
  mode: auto

gacos:
  enabled: false
  gacos_dir: null

tools:
  triangle: triangle
  snaphu: snaphu
```

No fixed project-specific IFG rejection list is embedded in production configuration.

---

## 8. Stage commands

```bash
pystamps run --start-step 1 --end-step 8
pystamps run --start-step 1 --end-step 5
pystamps run --start-step 6 --end-step 6
pystamps run --start-step 7 --end-step 7
pystamps run --start-step 8 --end-step 8
pystamps run --start-step 6 --end-step 8
```

Completed products may be reported as:

```text
status: skipped_existing
```

---

## 9. Stage 6

```text
SB interferometric phase
        ↓
3-D unwrapping preparation
        ↓
SNAPHU
        ↓
unwrapped SB interferograms
        ↓
network diagnostics / MSD
        ↓
automatic FINAL IFG-QC
        ↓
network-connectivity protection
        ↓
selected-network inversion
        ↓
single-master acquisition phase
```

IFG rejection is project-relative, not a hard-coded universal index list.

---

## 10. Reference selection

```yaml
reference:
  mode: auto
  longitude: null
  latitude: null
  radius_m: 500.0
```

Automatic reference selection establishes a relative InSAR datum; it does not prove zero physical deformation.

---

## 11. GACOS — optional

GACOS atmospheric correction is optional and disabled by default.

For the SBAS workflow, GACOS is applied to the Stage-6 single-master acquisition-phase product before Stage 7 and Stage 8:

```text
Stage 6
    ↓
phuw2.mat
    ↓
optional GACOS correction
    ↓
phuw2_gacos.mat
    ↓
Stage 7
    ↓
Stage 8
```

When GACOS is disabled, Stage 7 and Stage 8 use the original `phuw2.mat`.

When GACOS is enabled, pySTAMPS-GAMMA generates or reuses `phuw2_gacos.mat`, and Stage 7 and Stage 8 use the corrected phase.

### Supported products

GACOS product format is detected automatically from the files in `gacos_dir`.

Supported forms are:

- GeoTIFF: `*.tif` or `*.tiff`;
- raw ZTD: `*.ztd` with the corresponding `.rsc` metadata file.

No `product_format` setting is required.

If both GeoTIFF and raw ZTD representations are present for the same acquisition date, the GeoTIFF product is selected deterministically.

### Configuration

Default configuration:

```yaml
gacos:
  enabled: false
  gacos_dir: null
  product_unit: auto
  projection: zenith
  sign: auto
  strict_dates: true
  rebuild: false
  incidence_tif: null
  incidence_deg: null
  qa_ps: 30000
  qa_ifg: 80
  chunk_ps: 4096
  min_valid_fraction: 0.995
```

To enable GACOS:

```yaml
gacos:
  enabled: true
  gacos_dir: /path/to/GACOS
```

`product_unit: auto` automatically resolves the product unit when possible. `projection: zenith` treats the GACOS product as zenith delay and projects it to LOS using the configured or automatically resolved incidence angle.

For SBAS processing, `phuw2.mat` contains one phase column per acquisition, not one column per SB interferogram. GACOS correction therefore works in the single-master acquisition domain. The atmospheric contribution for each acquisition is referenced to the master acquisition before phase correction.

With `sign: auto`, pySTAMPS-GAMMA evaluates the candidate correction signs from project data and records the selected sign and QA statistics.

### Outputs

Corrected phase:

```text
phuw2_gacos.mat
```

GACOS QA/debug information:

```text
gacos_correction_debug.json
```

Acquisition/product inventory:

```text
gacos_date_inventory.csv
```

Intermediate sampled LOS delays are stored under:

```text
_gacos_work/
```

### Stage 7/8 input tracking

Stage 7 and Stage 8 automatically track which phase product generated their outputs:

```text
gacos.enabled: false
    → phuw2.mat

gacos.enabled: true
    → phuw2_gacos.mat
```

Changing between corrected and uncorrected phase inputs invalidates existing Stage 7/8 results and causes those stages to be recomputed.

Re-running with the same phase input can reuse existing products and may be reported as:

```text
status: skipped_existing
```

When `rebuild: true`, the corrected phase is rebuilt once and shared by Stage 7 and Stage 8 within the same pipeline invocation.

GACOS performance remains project-dependent and should be evaluated from the QA output and the resulting displacement time series.

---

## 12. Source-checkout utilities

```text
scripts/pipeline/prepare_gamma_sbas.py
scripts/corrections/prepare_deramp.py
scripts/corrections/prepare_gacos.py
scripts/postprocess/postprocess.py
scripts/validate_audit.py
```

`scripts/pipeline/prepare_gamma_sbas.py` remains available as an advanced/manual Stage-1 preparation interface. When run from a project `pystamps/` directory, its defaults are `output/work_dir = cwd`, `project/data_dir = cwd.parent`, and a local `pystamps.yaml` is used when present.

The main installed interface is:

```bash
pystamps
```

---

## 13. Build from source

```bash
make check-python
make check-rust
make check-runtime
make build
```

The wheel must contain:

```text
pystamps/kernels/_stage2_native.*.so
pystamps/data/production.yaml
```

---

## 14. Scientific scope

pySTAMPS-GAMMA aims to reproduce the major processing logic of the StaMPS SBAS workflow while providing Python/Rust implementation, GAMMA integration, automatic project discovery, project-relative quality auditing and native acceleration.

Important interpretation points:

- automatic IFG rejection is project-relative;
- no universal fixed IFG list is embedded in production;
- reference selection establishes a relative InSAR datum;
- GACOS performance should be validated per project;
- a one-LOS vertical approximation requires an explicit horizontal-motion assumption;
- processing parameters remain project-specific scientific choices.

---

## 15. Citation

If pySTAMPS-GAMMA is used in scientific work, cite this repository together with the relevant StaMPS, GAMMA, SNAPHU and atmospheric-correction references.
