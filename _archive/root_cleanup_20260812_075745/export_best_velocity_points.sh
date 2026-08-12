#!/usr/bin/env bash
set -euo pipefail

DATASET="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized"
PYTHON="/home/ubuntu/software/miniconda3/envs/stamps/bin/python"

INPUT="$DATASET/best_velocity_gacos/points/best_velocity_points.h5"
OUT="$DATASET/best_velocity_gacos/points_original_distribution"

[[ -s "$INPUT" ]] || {
    echo "错误：找不到点级结果：$INPUT" >&2
    exit 2
}

mkdir -p "$OUT"

"$PYTHON" - "$INPUT" "$OUT" <<'PY'
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

input_file = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(input_file, "r") as h5:
    def read(name, default=None):
        if name in h5:
            return np.asarray(h5[name][:]).reshape(-1)
        if default is None:
            raise KeyError(f"缺少HDF5变量：{name}")
        return np.asarray(default)

    lon = read("lon").astype(np.float64)
    lat = read("lat").astype(np.float64)

    n_ps = lon.size

    velocity = read("velocity_mm_yr").astype(np.float32)
    velocity_std = read("velocity_std_mm_yr").astype(np.float32)
    velocity_gls = read("velocity_gls_mm_yr").astype(np.float32)
    velocity_ols = read("velocity_ols_mm_yr").astype(np.float32)

    rmse = read("rmse_mm").astype(np.float32)
    whitened_rmse = read("whitened_rmse").astype(np.float32)

    n_obs = read("n_obs").astype(np.int32)
    span_days = read("span_days").astype(np.float32)
    effective_n = read("effective_n").astype(np.float32)

    local_median = read(
        "local_median_mm_yr",
        np.full(n_ps, np.nan),
    ).astype(np.float32)

    local_sigma = read(
        "local_sigma_mm_yr",
        np.full(n_ps, np.nan),
    ).astype(np.float32)

    local_count = read(
        "local_neighbor_count",
        np.zeros(n_ps),
    ).astype(np.int32)

    fit_accepted = read(
        "accepted",
        np.ones(n_ps),
    ).astype(bool)

    strict_quality = read(
        "best_quality_mask",
        np.zeros(n_ps),
    ).astype(bool)

    reason = read(
        "quality_reason_code",
        np.zeros(n_ps),
    ).astype(np.uint8)

finite = (
    np.isfinite(lon)
    & np.isfinite(lat)
    & np.isfinite(velocity)
    & np.isfinite(velocity_std)
    & np.isfinite(rmse)
)

# 推荐点层：不采用局部空间异常强制删除。
recommended = (
    finite
    & fit_accepted
    & (rmse <= 15.0)
    & (velocity_std <= 3.0)
)

# 高质量严格点层：沿用原脚本所有过滤条件。
strict = finite & strict_quality

# 局部一致性只作为标记，不再用于推荐点层删除。
local_flag = np.isin(reason, [4, 5])

frame = pd.DataFrame(
    {
        "ps_id": np.arange(1, n_ps + 1, dtype=np.int64),
        "lon": lon,
        "lat": lat,
        "vel_mm_yr": velocity,
        "vel_std": velocity_std,
        "vel_gls": velocity_gls,
        "vel_ols": velocity_ols,
        "rmse_mm": rmse,
        "white_rmse": whitened_rmse,
        "n_obs": n_obs,
        "span_days": span_days,
        "effective_n": effective_n,
        "local_med": local_median,
        "local_sig": local_sigma,
        "local_n": local_count,
        "fit_ok": fit_accepted.astype(np.uint8),
        "recommended": recommended.astype(np.uint8),
        "strict_qc": strict.astype(np.uint8),
        "local_flag": local_flag.astype(np.uint8),
        "reason": reason,
    }
)

# CSV始终输出，作为通用备份。
all_csv = out_dir / "ps_velocity_all_points.csv"
recommended_csv = out_dir / "ps_velocity_recommended.csv"
strict_csv = out_dir / "ps_velocity_strict.csv"

frame.to_csv(all_csv, index=False, encoding="utf-8-sig")
frame.loc[recommended].to_csv(
    recommended_csv,
    index=False,
    encoding="utf-8-sig",
)
frame.loc[strict].to_csv(
    strict_csv,
    index=False,
    encoding="utf-8-sig",
)

print("全部PS：", n_ps)
print(
    "推荐PS：",
    int(np.count_nonzero(recommended)),
    f"({100*np.mean(recommended):.2f}%)",
)
print(
    "严格PS：",
    int(np.count_nonzero(strict)),
    f"({100*np.mean(strict):.2f}%)",
)

# GeoPackage用于QGIS。
try:
    import geopandas as gpd

    geometry = gpd.points_from_xy(frame["lon"], frame["lat"])
    gdf = gpd.GeoDataFrame(
        frame,
        geometry=geometry,
        crs="EPSG:4326",
    ).to_crs("EPSG:32650")

    outputs = [
        (
            out_dir / "ps_velocity_all_points.gpkg",
            gdf,
            "all_ps",
        ),
        (
            out_dir / "ps_velocity_recommended.gpkg",
            gdf.loc[recommended].copy(),
            "recommended_ps",
        ),
        (
            out_dir / "ps_velocity_strict.gpkg",
            gdf.loc[strict].copy(),
            "strict_ps",
        ),
    ]

    for path, layer, layer_name in outputs:
        if path.exists():
            path.unlink()
        layer.to_file(
            path,
            layer=layer_name,
            driver="GPKG",
        )
        print(path)

except Exception as exc:
    print(
        "GeoPackage未生成，但CSV已正常输出：",
        type(exc).__name__,
        exc,
    )

print("输出目录：", out_dir)
PY
