#!/usr/bin/env bash
# Run the HANDOFF-track Maple checks (edge machinery, 2D). Needs Maple 2026 (override with MAPLE=/path/to/maple).
# Run from anywhere; each script prints PASS/FAIL lines. Scripts 10-13 take ~1 min in total.
set -euo pipefail
cd "$(dirname "$0")/.."
MAPLE="${MAPLE:-$HOME/maple2026/bin/maple}"
for s in 10_edge_primitives 11_truncated_monomials 12_moments_recursion 13_halfplane 14_tier3_2d; do
  echo "=== maple $s.mpl ==="
  "$MAPLE" -q "maple/$s.mpl"
done
echo "edge Maple scripts complete: every check line above must end in PASS"
