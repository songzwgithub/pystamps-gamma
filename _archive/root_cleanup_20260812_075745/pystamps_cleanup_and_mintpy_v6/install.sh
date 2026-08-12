#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/ubuntu/software/pystamps-main}"
PYTHON="${PYTHON:-/home/ubuntu/software/miniconda3/envs/stamps/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/tools"
cp "$HERE/mintpy_network_inversion_v6.py" "$ROOT/tools/mintpy_network_inversion_v6.py"
cp "$HERE/run_mintpy_network_inversion_v6.sh" "$ROOT/run_mintpy_network_inversion_v6.sh"
chmod +x "$ROOT/tools/mintpy_network_inversion_v6.py"
chmod +x "$ROOT/run_mintpy_network_inversion_v6.sh"

echo "Installed V6."
echo "No environment/package change."
echo "No Stage6/7/8 or Final-C overwrite."
echo
"$ROOT/run_mintpy_network_inversion_v6.sh" --self-test
echo
echo "Run:"
echo "  cd '$ROOT'"
echo "  ./run_mintpy_network_inversion_v6.sh"
