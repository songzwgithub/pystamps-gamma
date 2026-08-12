# pySTAMPS Parity Report

Date: 2026-03-05
Project: `pySTAMPS`
Run dataset: `/shared/home/rdelprete/tmp/pystamps_iter14_stage3plus`
Golden dataset: `/shared/home/rdelprete/PythonProjects/AgenticWork/pySTAMPS/inputs_and_outputs/InSAR_dataset_test`
New upload: `/shared/home/rdelprete/PythonProjects/AgenticWork/pySTAMPS/inputs_and_outputs/NEW_STUFF/unwrap_step_iter1.zip`

## Current Verify Status

Command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run pystamps verify \
  --run /shared/home/rdelprete/tmp/pystamps_iter14_stage3plus \
  --golden /shared/home/rdelprete/PythonProjects/AgenticWork/pySTAMPS/inputs_and_outputs/InSAR_dataset_test
```

Latest result:
- `ok: false`
- `checked: 26`
- `failed: 13`

Current failing keys:
- `PATCH_1..4/select1.mat`: `C_ps2` (`~1e-5`)
- `PATCH_1/weed1.mat`, `PATCH_3/weed1.mat`: `ps_max`
- `pm2.mat`: `C_ps`
- `ifgstd2.mat`: `ifg_std`
- `phuw2.mat`: `msd` (max_abs `14.9361`)
- `scla2.mat`: `C_ps_uw` (max_abs `29.4306`)
- `mean_v.mat`: `m` (max_abs `8.3154`)
- `uw_grid.mat`: `ph` (max_abs `22.9754`)
- `uw_space_time.mat`: `dph_noise` (max_abs `6.26338`)

## What Was Improved

- `uw_grid.grid_ij` parity is exact (`0` differing entries).
- `uw_interp.Z` implementation is down to a single-cell discrepancy in generated mode.
- Stage8 arc count matches golden (`689516`).
- `uw_interp.mat` from uploaded bundle is exact to golden when copied in.

## Evaluation Of Uploaded `NEW_STUFF` Bundle

Extracted files included:
- `unwrap.1.node`, `unwrap.2.edge`, `unwrap.2.ele`, `unwrap.2.node`
- `uw_grid.mat`, `uw_interp.mat`, `uw_phaseuw.mat`, `uw_space_time.mat`, `phuw2.mat`
- snaphu files (`snaphu.conf`, `snaphu.in`, `snaphu.out`, etc.)

Direct comparison vs golden:
- `uw_interp.mat`: **matches**
- `uw_phaseuw.mat`: **does not match** (`ph_uw` max diff `~23.2`)
- `phuw2.mat`: **does not match** (`ph_uw` max diff `~18.85`)
- `uw_grid.mat`: **does not match** (`ph` max diff `~23.53`)
- `uw_space_time.mat`: **does not match** (`dph_noise` max diff `~6.26`)

Conclusion: upload is useful for topology (`uw_interp`) but not from the exact same numerical run as current golden for unwrapped phase/noise fields.

## Still Needed To Finish 100% Parity

Please provide from the **exact run that produced the current golden `InSAR_dataset_test`**:

1. `uw_phaseuw.mat` (exact-run version)
2. `phuw2.mat` (exact-run version)
3. `uw_space_time.mat` (exact-run version)
4. Optional but useful: exact `unwrap.2.ele` + `unwrap.2.edge` from that same run (if different)
5. Confirmation of MATLAB release and Triangle version/options used in that golden run

Without exact-run unwrapped phase intermediates, remaining Stage6/7/8 numerical parity cannot be fully closed.
