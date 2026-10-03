# REVIEW-002 — WORK-002 (branch `local-model`, `54650dd..6f5a56f`)

Reviewer: Claude, 2026-10-03. Verdict: **ACCEPT T2.1, T2.2, T2.3; ACCEPT WITH CORRECTIONS T2.4** (one wrong number in `docs/q2-conditioning.md`, corrected; the study's conclusions stand). No rule violation, no tolerance edited, no defect in the solver changes.

## 1. Mechanical gate
* Five commits: `WORK-002 T2.1 / T2.2 / T2.3 / T2.4 / REPORT-002`, one per task. `git diff --stat 54650dd..local-model`: 14 files, all in the allowed list (`kernels.py` +2, `boundaryOps.py` ±7, `cover.py` +30, `deltasph2d.py` ±5, `deltasph_regress.py`, the new modules and tests, docs). `scene.py`, `warpbc.py`, `np2d.py`, any existing test, `results/`, `regress_baseline.json`: untouched. `git status` clean.
* `git diff` of `deltasph2d.py`: exactly the import, the config field and the two-line `if cfg.coverExact` branch.
* Tests (reviewer, `warp` env): `test_cover_scene.py` + `test_deltasph_cover.py` + the 72 solver tests → **79 passed in 20.6 s** (the model's 709-test full-suite count is consistent: 702 + 6 + 1).

## 2. Evidence audit
| check | result |
|---|---|
| re-ran `deltasph_regress check --physics --cases tank,dambreak` (default cfg) | exit 0, **gate 10/10 PASS** (dam break: KE rel max vs B 2.5e-7, arrival \|diff\| 2.97e-4, density 0.99944 / 1.00631, steps drift 0; tank: rmseBulk 1.4988e-3, rmseNear 1.8137e-3, keLast 7.0e-7) — **and all 14 bit-level lines PASS** (so the default config is unchanged by the switch; worst margin 0.584, `ke_tstar25`; run-to-run noise ~1e-8 relative, as before) |
| tolerance history | `git log` per file: each test file has a single commit; tolerances were written with the code; the log shows 2 debug attempts (T2.1) and 1 (T2.2), none changes a test or tolerance |
| independent re-derivation of the T2.4 numbers | reviewer probe `docs/work/refs/q2_review_probe.py`: `w4p5` at the vertex (0,0): stable(8,6) **7.53e-05**, stable(16,8) **2.69e-12** (the report: 7.53e-05 / 2.69e-12) |
| independent check of the one new physical formula (flat-floor `T_y`) | **wrong in the report/doc, see §3.1** |
| negative controls | T2.1 H·1.01 control is real (7.2e-3 > 1e-4); the sign control of `test_cover_scene.py` compares `got` with `−got`, i.e. `2·\|got\|` (tautological; harmless because the H control and the smoke values carry the sensitivity). T2.3: `alpha = 0.5` fails the gate (KE 0.342 vs 0.05) — real. `--perturb` correctly noted as invalid for this gate |
| failure modes (self-comparison, widened tolerances, skips, swallowed exceptions, hard-coded values from own output) | none found. The smoke values in `test_cover_scene.py` are the reviewer's numbers from the work document |

## 3. Findings
1. **`docs/q2-conditioning.md` flat-floor magnitude was wrong (corrected).** The doc/report give `T_y = −6.8e-8` (w2), `−4.9e-8` (w4), from the factor `(1/5)(c2/c25)⁵/π⁴`. The right conversion is `(1/5) c2⁵ / (π⁴ c25)` (`W⁵ = c2⁵/(π⁵H¹⁰) s⁵`, the registered kernel is `c25/(πH²) s⁵`): **`T_y = −0.14101158` (w2), `−0.19782376` (w4)**, each confirmed by an independent midpoint-grid integral of `W⁴ ∂_yW` over the half disk (−0.14101146 / −0.19782346) and consistent with the `~0.14` of HANDOFF A6. The sign statement and `g0_y` were correct; the doc is corrected (a dated note at the top). The classification (relative errors) is unaffected.
2. **The np2d "vertex finding" is benign for the solver.** `w4p5` stable-route error against the distance `d` from a vertex (reviewer probe): stable(8,6) 7.5e-5 for `d ≤ 1e-3`, 7e-7 at `d = 0.01 H`, **3.6e-13 at `d = 0.03 H`**; stable(16,8) ≤ 2.7e-12 everywhere; the plain float64 route is bad everywhere (1e-5…1e-3). Particles of the solver sit ≥ `dx/√2 = 0.18 H` from a wall corner (`H = 4 dx`). Closed; no action. (For WORK-004: use `stable=(16,8)`-equivalent resolution, not (8,6), when the Chebyshev plan is designed.)
3. **Gate item `steps drift` is insensitive by construction.** `_next_dt` takes the minimum of the viscous, force and acoustic limits; in practice the acoustic one (`cfl·H/c0/ks`) always wins, so `steps` is constant (drift 0 even with `alpha = 0.5`). WORK-002's stated reason ("a different vmax history") was wrong; the item still guards the dt rule. Not changed (limits are frozen); noted in WORK-003.
4. **`coverExact = True` bit-level drift (~1e-6 in two dam-break lines) is acceptable**: the physics gate is the acceptance for switched physics; the final WORK package re-records the baseline with all switches on. Answer to the model's question.
5. **detcmp**: zero flag disagreements on both cases at all snapshots; the quadrature error of the wall cover vector grows from 3.4e-3 to 1.0e-2 · n_w H² on the evolved dam break (consistent with its ~1 %). The initial-state value 3.41e-3 vs the reviewer's 4.8e-3 of the work document is unexplained but immaterial (both under the 1e-2 bound).
6. **Sloshing gate path** exists but was never run; open (long case). Not needed before the final re-validation.

## 4. Corrections made by the reviewer (not committed by the reviewer; see the end)
* `docs/q2-conditioning.md`: note + corrected `T_y` values and factor (§3.1).
* New reviewer files: `docs/work/refs/{q2_review_probe,cone_area_probe,cone_count_probe,tensile_scene_probe}.py`, `docs/work/WORK-003.md`, this file; `docs/work/REVIEW.md` sequence updated.

## 5. Facts WORK-003 assumes
* Q3b = `area(solid ∩ disk ∩ wedge)` closed form per edge (polar sectors), verified against brute force; **tangent-case pitfall** (`z == H`) fixed in the probe and made a mandatory test. Polar quadrature errors on real states: cone count 1.398e-2 · n_w H², all-neighbour area 5.69e-3 · n_w H², no flag flips.
* Q2 for Wendland C2 through the unmodified scene path with a registered `w2p5`: Warp vs np2d(16,8) 6.5e-8, vs plain grid 1.4e-6, polar quadrature vs exact 1.389e-2 of max, effect on `shift()` 3.6e-2 of max‖upd‖ (tank initial). C4 needs the Chebyshev plan (WORK-004, `warpbc.py`).
* The physics gate does not see these switches; correctness evidence = tests + detcmp; PASS = no harm.
* Starting commit = the `REVIEW-002: ...` commit; 79 solver tests (≈ 20 s – 2 min), 709 in `tests/edge`.

## 6. Next
`WORK-003.md` (Q3b closed form + scene wrapper + `coneExact`; `tensile_vector_scene` + `tensileExact`, Wendland C2). Then WORK-004: Chebyshev-basis plan in `warpbc.py` for C4 (`W⁵` degree 40) + `tensileExact` for C4; WORK-005: Q1 Laplacian; WORK-006: all switches on, delete `_solid_samples`, re-validate, re-record the baseline (see `docs/work/REVIEW.md`).
