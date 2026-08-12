# Cleanup + V6 MintPy-style inversion benchmark

## 1. Cleanup first

The cleanup script is deliberately conservative and is **dry-run by default**.

```bash
cd /home/ubuntu/software/pystamps-main

bash /上传目录/pystamps_cleanup_and_mintpy_v6/cleanup_obsolete_experiments_20260812.sh
```

Inspect the output. If correct:

```bash
bash /上传目录/pystamps_cleanup_and_mintpy_v6/cleanup_obsolete_experiments_20260812.sh --apply
```

It archives, rather than deletes, the superseded V1/V2/V3/V4/V5/V5.1 root
wrappers/bundles and matching installed experimental tools. It removes only
`__pycache__` directories.

It does not touch:
- `pystamps/`
- Stage6/7/8 production code
- Stage6 checkpoint
- SNAPHU
- Final-C
- dataset outputs
- V3 **coherence cache** in the dataset (V6 reuses this input cache)
- Rust `target/`

## 2. Install V6

```bash
cd /home/ubuntu/software/pystamps-main

rm -rf /tmp/mintpy_v6
mkdir -p /tmp/mintpy_v6

unzip /上传目录/pystamps_cleanup_and_mintpy_v6.zip \
  -d /tmp/mintpy_v6

bash /tmp/mintpy_v6/pystamps_cleanup_and_mintpy_v6/install.sh
```

Run:

```bash
cd /home/ubuntu/software/pystamps-main
./run_mintpy_network_inversion_v6.sh
```

## V6 scientific test

No spatial filtering and no IFG deletion.

It compares:

```text
CURRENT_FINAL_C
STAGE7_SAVED_LEGACY_RAMP_SCN
MINTPY_NO_LEGACY_RAMP_SCN
MINTPY_COH_LEGACY_RAMP_SCN
MINTPY_VAR_LEGACY_RAMP_SCN
```

The MintPy-style branches use minimum-norm interval velocity.

Default effective looks for the existing 4:1 Sentinel-1 stack:

```text
L = round(4 * 1 / 1.94) = 2
```

You can override only if independently justified:

```bash
./run_mintpy_network_inversion_v6.sh --ncorrlooks 3
```

## Inspect

```bash
D=/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized
LATEST=$(
  find "$D/_audit" -maxdepth 1 -type d \
    -name 'mintpy_network_inversion_v6_*' \
    -printf '%T@ %p\n' |
  sort -nr | head -1 | cut -d' ' -f2-
)

column -s, -t "$LATEST/01_exact_svd_spot_audit.csv" | head -40
column -s, -t "$LATEST/03_truth_pooled.csv"
column -s, -t "$LATEST/04_temporal_coherence_truth_quartiles.csv" | less -S
cat "$LATEST/05_CONCLUSION.json"
```

V6 is an isolation benchmark only. If a MintPy-style inversion wins, the next
round rebuilds SCLA/SCN on that new time series. If it does not, stop tuning the
inversion and move upstream to SARvey-style spatial-arc consistency / unwrap-error
diagnostics and then Dolphin-style phase linking on the registered RSLC stack.
