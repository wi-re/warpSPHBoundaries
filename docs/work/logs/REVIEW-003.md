# REVIEW-003 — WORK-003 (branch `local-model`, `9b640a7..119583b`)

Reviewer: Claude, 2026-10-03. Verdict: **ACCEPT T3.1, T3.2, T3.3, T3.5; ACCEPT WITH CORRECTIONS T3.4** (one vacuous negative control, corrected). No rule violation, no tolerance edited, no defect in the solver changes.

## 1. Mechanical gate
* Six commits `WORK-003 T3.1 … T3.5 / REPORT-003`, one per task. `git diff --stat 9b640a7..local-model`: 12 files, all in the allowed list (`cone_area.py`, `tensile.py`, 5 new test files, `deltasph2d.py` +26/−…, `deltasph_detcmp.py`, porting notes +2 rows, LOG/REPORT). `scene.py`, `warpbc.py`, `kernels.py`, `np2d.py`, `cover.py`, `deltasph_regress.py`, existing tests, `results/`: untouched. `git status` clean.
* `git diff` of `deltasph2d.py`: exactly the import ×2, the two config fields, the two `coneExact` blocks in `_detect_surface` (polar lines re-indented under `else:`), the `tensileExact` block in `shift`. `deltasph_detcmp.py`: `--exact cover|cone|both`, default output unchanged.
* Tests (reviewer, `warp` env): `pytest tests/edge -q` → **728 passed in 368.9 s** (= the report's count).

## 2. Evidence audit
| check | result |
|---|---|
| re-ran the full suite (the report's DoD line) | 728 passed — matches |
| re-ran `deltasph_regress check --cases tank,dambreak` (default cfg; proves the three switches off change nothing) | `OVERALL: PASS` — tank 6/6 (worst margin 0.006), dam break 8/8 (worst margin 0.588 on `ke_tstar25`; arrival 2.4897, KE vs series B 2.5e-7): the three switches off change nothing |
| independent check of `cone_area` on geometry the tests do not use (throw-away `.tmp/review3_spot.py`: 12 random star-shaped non-convex polygons, half of them cavities, random point / H / axis, wedge π/6 and full; reference = 3001² Cartesian grid + matplotlib point-in-polygon, sharing nothing with the polar brute force of the tests) | worst \|closed − grid\| / H² = **2.7e-5** (the grid's own error) |
| tolerance history | each test file has one commit; tolerances were written with the code; no later edit |
| numbers in the report vs the work document | C2 tank: polar vs exact tensile 1.3885e-2 (reviewer 1.389e-2), shift effect 3.596e-2 (3.59e-2), sign control 2.014 (2.01); cone area worst vs brute 2.955e-4 H² (2.95e-4): all reproduce the reviewer's own probes |
| failure modes (self-comparison, widened tolerances, skips, swallowed exceptions, hard-coded values from own output) | one: see §3.1 |

## 3. Findings
1. **Vacuous negative control (T3.4 (c)), corrected.** `test_magnitude_and_sign_are_real` compared `1.01·T` and `−T` with `T` itself — arithmetic, cannot fail. The work document asked for the grid value of (b); the model listed it under *Deviations* ("the two agree to 3.27e-6") but the check stayed vacuous. Corrected in `tests/edge/test_tensile_scene.py`: the controls are compared with the independent grid value (`1.01·T`: 1.41e-3 = 1.0 % of scale, `−T`: 0.282 = 2×; both > 1e-3), plus an assertion that `T` agrees with the grid (≤ 5e-5). Passes.
2. **Everything else matches the work document, including the deviations the model reported** (`coneExact`-only dam-break gate: all bit-level lines PASS — the effect is below 20× the reduction-order spread; benign).
3. **Quadrature errors in the solver are now measured end to end:** wall cone count 1.0e-2 · n_w H² (dam break, step 5000), tensile term 1.4 % (C2). None flips a surface flag (detcmp: 0 disagreements at all snapshots, both cases). The shift changes by 3.6 % of its maximum with the exact tensile term; the physics gate passes with every switch combination (KE rel ≤ 2.4e-5).
4. **`cone_area_scene` is O(N·E) (all edges, non-local formula)**: fine for the analytic tank/dam-break scenes (E = 4–8) and for the porting reference; it must become a local Warp kernel (edges within H plus the tangent/sector bookkeeping) before it is used in a scene with many edges — noted for the phase-2/3 plan, not for the next packages.

## 4. Corrections made by the reviewer (not committed by the reviewer; the user commits as `REVIEW-003: ...`)
* `tests/edge/test_tensile_scene.py`: the negative control of §3.1.
* New: `docs/work/WORK-004.md`, `docs/work/refs/{stable_plan_probe,tensile_c4_probe,slosh_gate_probe}.py`, this file; `docs/work/REVIEW.md` (sequence + lessons), `docs/work/README.md` (current package).

## 5. Facts WORK-004 assumes (all measured by the reviewer; see the document)
* A Chebyshev-quadrature variant of the Warp surface-element kernel (algorithm of `np2d stable=(16, 8)`) reaches 1e-15 relative for `W⁵` of C2 and C4 (2.7e-12 at a vertex only) where the monomial plan has 6e-8 / 1.7e-3; the other eight channels agree with the monomial plan to ≤ 1.3e-10 (the monomial plan's own error near walls); resolutions (8, 4) and below are visibly worse; it is faster than the monomial kernel (2.8× on 200k pairs, contended GPU).
* C4 tensile term: floor `T_y = −0.19782376` (grid 6e-6), L-shape vs np2d stable 3.5e-15, C4 tank polar-vs-exact 1.749e-2, shift effect 5.91e-2.
* Sloshing (nx = 200, Wendland C4) T = 1.5 s: default run reproduces the stored `slosh_B_nx200` KE to 4.2e-8 (so the gate is well-posed); with the exact C4 tensile term (stable plan, monkeypatched probe) 5.8e-5 from the default run; wall 1501 s / 2346 s on a shared GPU (≈ 42 ms/step idle). The `sensorPressureProbe` series contains NaN early on (fewer than 3 particles near the sensor): use `nan*` reductions.
* Starting commit = the `REVIEW-003: ...` commit; 79 solver tests, 728 in `tests/edge`.

## 6. Next
`WORK-004.md` (stable plan in `warpbc.py`, `tensile` for C2 + C4, `tensileExact` for C4, gates incl. truncated sloshing). Then WORK-005: Q1 Laplacian derivation + operation; WORK-006: all switches on, delete `_solid_samples`, re-validate, re-record the baseline.
