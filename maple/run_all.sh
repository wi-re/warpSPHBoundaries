#!/usr/bin/env bash
# Regenerate all symbolic results (src/warpSPHBoundaries/data/symbolic/*.txt).
# Run from anywhere; needs Maple 2026 (override with MAPLE=/path/to/maple).
set -euo pipefail
cd "$(dirname "$0")/.."
MAPLE="${MAPLE:-$HOME/maple2026/bin/maple}"
mkdir -p src/warpSPHBoundaries/data/symbolic results/figures
for s in 00_setup 01_planar 03_sphere; do
  echo "=== maple $s.mpl ==="
  "$MAPLE" -q "maple/$s.mpl"
done
echo "all maple scripts complete"
