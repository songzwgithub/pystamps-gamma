#!/usr/bin/env bash
set -euo pipefail

ROOT="${PYSTAMPS_ROOT:-/home/ubuntu/software/pystamps-main}"
TARGET="$ROOT/calc_best_velocity_gls.py"
RUNNER="$ROOT/run_best_velocity_gacos.sh"
PYTHON="${PYSTAMPS_PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$ROOT/_best_velocity_patch_backup/$STAMP"

mkdir -p "$BACKUP_DIR"

[[ -f "$TARGET" ]] || {
  echo "错误：找不到 $TARGET" >&2
  exit 2
}
[[ -f "$RUNNER" ]] || {
  echo "错误：找不到 $RUNNER" >&2
  exit 3
}
[[ -x "$PYTHON" ]] || {
  echo "错误：找不到Python $PYTHON" >&2
  exit 4
}

cp -a "$TARGET" "$BACKUP_DIR/"
cp -a "$RUNNER" "$BACKUP_DIR/"

TARGET="$TARGET" RUNNER="$RUNNER" "$PYTHON" - <<'PY'
from pathlib import Path
import os

target = Path(os.environ["TARGET"])
runner = Path(os.environ["RUNNER"])

text = target.read_text(encoding="utf-8")

# 1. Fix scipy.spatial import.
if "import importlib\n" not in text:
    text = text.replace(
        "import importlib.util\n",
        "import importlib\nimport importlib.util\n",
        1,
    )

text = text.replace(
    "return __import__(name)",
    "return importlib.import_module(name)",
)

# 2. Make the primary product a filtered 50 m product, not a 100 m aggregate.
text = text.replace(
    "best_velocity_gacos/rasters/geo_velocity_best_100m.tif",
    "best_velocity_gacos/rasters/geo_velocity_best_filtered_50m.tif",
)
text = text.replace(
    "+ inverse-variance weighted 100 m aggregation",
    "+ inverse-variance weighted 50 m rasterization after point QC",
)

# Output filenames.
replacements = {
    'primary = rasters / "geo_velocity_best_100m.tif"':
        'primary = rasters / "geo_velocity_best_filtered_50m.tif"',
    'rasters / "geo_velocity_best_std_100m.tif"':
        'rasters / "geo_velocity_best_filtered_std_50m.tif"',
    'rasters / "geo_velocity_best_count_100m.tif"':
        'rasters / "geo_velocity_best_filtered_count_50m.tif"',
    'rasters / "geo_velocity_best_neff_100m.tif"':
        'rasters / "geo_velocity_best_filtered_neff_50m.tif"',
    'rasters / "geo_velocity_best_rmse_100m.tif"':
        'rasters / "geo_velocity_best_filtered_rmse_50m.tif"',
    'rasters / "geo_velocity_best_local_sigma_100m.tif"':
        'rasters / "geo_velocity_best_filtered_local_sigma_50m.tif"',
    'rasters / "geo_velocity_best_detail_50m.tif"':
        'rasters / "geo_velocity_best_aggregated_100m.tif"',
    'rasters / "wgs84" / "geo_velocity_best_100m_wgs84.tif"':
        'rasters / "wgs84" / "geo_velocity_best_filtered_50m_wgs84.tif"',
    'rasters / "geo_velocity_best_100m.qml"':
        'rasters / "geo_velocity_best_filtered_50m.qml"',
    'description="Accepted PS count per 100 m cell"':
        'description="Accepted PS count per 50 m cell"',
    'description="Best-quality detailed LOS velocity"':
        'description="Best-quality aggregated 100 m LOS velocity"',
}
for old, new in replacements.items():
    text = text.replace(old, new)

# Defaults: primary 50 m / 1 PS; auxiliary 100 m / 3 PS.
text = text.replace(
    'p.add_argument("--main-resolution-m", type=float, default=100.0)',
    'p.add_argument("--main-resolution-m", type=float, default=50.0)',
)
text = text.replace(
    'p.add_argument("--main-min-points", type=int, default=3)',
    'p.add_argument("--main-min-points", type=int, default=1)',
)
text = text.replace(
    'p.add_argument("--detail-resolution-m", type=float, default=50.0)',
    'p.add_argument("--detail-resolution-m", type=float, default=100.0)',
)
text = text.replace(
    'p.add_argument("--detail-min-points", type=int, default=1)',
    'p.add_argument("--detail-min-points", type=int, default=3)',
)

# Less aggressive local QC to preserve compact deformation features.
text = text.replace(
    'p.add_argument("--local-radius-m", type=float, default=500.0)',
    'p.add_argument("--local-radius-m", type=float, default=300.0)',
)
text = text.replace(
    'p.add_argument("--local-k", type=int, default=16)',
    'p.add_argument("--local-k", type=int, default=12)',
)
text = text.replace(
    'p.add_argument("--local-min-neighbors", type=int, default=6)',
    'p.add_argument("--local-min-neighbors", type=int, default=4)',
)

target.write_text(text, encoding="utf-8")

run = runner.read_text(encoding="utf-8")
run = run.replace("--main-resolution-m 100", "--main-resolution-m 50")
run = run.replace("--main-min-points 3", "--main-min-points 1")
run = run.replace("--detail-resolution-m 50", "--detail-resolution-m 100")
run = run.replace("--detail-min-points 1", "--detail-min-points 3")
run = run.replace("--local-radius-m 500", "--local-radius-m 300")
run = run.replace("--local-k 16", "--local-k 12")
run = run.replace("--local-min-neighbors 6", "--local-min-neighbors 4")
runner.write_text(run, encoding="utf-8")

print("已修复 scipy.spatial.cKDTree 导入。")
print("主产品改为：50 m、每格至少1个过滤合格PS。")
print("辅助产品改为：100 m、每格至少3个过滤合格PS。")
PY

"$PYTHON" -m py_compile "$TARGET"
bash -n "$RUNNER"

echo
echo "补丁完成。备份：$BACKUP_DIR"
echo
echo "关键检查："
grep -n \
  "import importlib$\|import_module(name)\|main-resolution-m\|main-min-points\|detail-resolution-m\|detail-min-points\|geo_velocity_best_filtered_50m" \
  "$TARGET" "$RUNNER" \
  | head -n 40

echo
echo "重新运行："
echo "  cd '$ROOT'"
echo "  ./run_best_velocity_gacos.sh"
