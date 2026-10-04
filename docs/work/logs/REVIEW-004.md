# REVIEW-004 — WORK-004 (branch `local-model`, `115c16d..7ff0ebe`)

Reviewer: Claude, 2026-10-04. Verdict: **ACCEPT T4.1, T4.2, T4.3, T4.4** (no correction needed). No rule violation, no tolerance edited, no defect found in the code.

## 1. Mechanical gate
* Four commits `WORK-004 T4.1 … T4.3 / REPORT-004`, in task order. (T4.4 has no code; its log rides with the report.) `git diff --stat 115c16d..local-model`: 12 files, **all on the allowed list** (`warpbc.py` +230/−7, `tensile.py`, `deltasph2d.py` +7/−1 — exactly the comment and the guard, 3 new test files, `test_deltasph_tensile.py` −10 (the one authorised test), `test_tensile_scene.py` ±2 (the one authorised assertion), `q2-conditioning.md` +13, porting notes +1, LOG/REPORT). `scene.py`, `kernels.py`, `np2d.py`, `deltasph_regress.py`, `results/`, the baseline: untouched. `git status` clean.
* `warpbc.py`: the monomial kernel launch is textually unchanged; the 7 deleted lines are the authorised `build_plan` refactor + the new opt-in branch in `edge_channels`. Read in full: `_ChebBuilder`, `ChebPlan`, `_clenshaw`, `_cheb_integral` (dyadic panels, clip to the chord side), `_edge_channels_cheb_kernel`, the registry/`stable=` resolution — all as specified. Two cosmetic remarks, no action: an explicit `stable=(n, p)` ignores a `plan=` argument; `STABLE_KERNELS` is a module-level global mutated by `tensile._register` (idempotent, tested).
* Tests (reviewer, `warp` env): `pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q` → **72 passed in 95 s**; the five new/changed test files → **19 passed in 10 s**; `pytest tests/edge -q` → **740 passed in 368 s** (= the report: 728 + 13 − 1).

## 2. Evidence audit
| check | result |
|---|---|
| re-ran the new tests with `-s` and compared every printed number with the report | identical: `w2p5` 7.80e-16, `w4p5` 9.43e-16 generic / 2.69e-12 all, monomial 1.66e-3, `(8,4)` 4.66e-5, np2d agreement 5.55e-15, `T` vs np2d 3.52e-15, `T_y` −0.1978237590 (grid 6.13e-6), polar-vs-exact C4 1.749e-2 / sign 2.017 / shift effect 5.908e-2 |
| independent check of `tensile_vector_scene(…, "w4")` (throw-away `docs/work/refs/review4_c4_indep.py`: own Wendland C4 normalised numerically, plain-numpy `3000²` midpoint grid of `W⁴∇W` over the half plane, sharing nothing with `edgebound.tensile`) at two points the tests do not use | `(0.2, 0.45)`, H = 1: **7.3e-6** (grid-limited); `(−0.3, 0.1)`, H = 0.7: **3.4e-7**. At `(0, 0.9)` (nearly tangent, true T ≈ 5e-24) the scene returns 1e-16 of round-off: harmless |
| tolerance history | each of the three new test files has exactly one commit; tolerances were written with the code, none edited later |
| numbers vs the work document | every reviewer number of WORK-004 reproduced (see the table in REPORT-004); sloshing KE vs B default 7.32e-8 (reviewer 4.2e-8: same order; the GPU-reduction-order spread), exact 5.78e-5 (5.78e-5), between runs 5.78e-5, `maxVelocity` 0.5921/0.5919, densities `[0.99929, 1.00487]`/`[0.99929, 1.00489]` |
| gates | both `--physics` runs `PHYSICS GATE: PASS` (dam-break KE vs B ≤ 2.4e-5, arrival diff 3e-4), default `check --cases tank,dambreak` `OVERALL: PASS` after the last code edit |
| failure modes (self-comparison, widened tolerances, skips, swallowed exceptions, hard-coded values from own output) | none found. One weak assertion: `test_tensile_exact_c4_and_c2_both_work` asserts only `max‖u_q − u_e‖ > 0` — the quantitative evidence is in the sibling test (b) (5e-3 ≤ rel ≤ 0.15) and in T4.2 (a)/(b), so accepted |

## 3. Findings
1. **The deletions of pinned tests were authorised** (guard test in `test_deltasph_tensile.py`, the `match=` of `test_non_w2_and_non_surface_raise`); the model kept the rest of both files untouched and replaced the deleted guard test by a real one. Good.
2. **Deviation 3 of the report (the model committed my reviewer files as `115c16d`)** is *my own* pre-launch commit (`REVIEW-003: …`, author rene@rtx) — the tree was dirty when the model started, the operator told it to commit. Harmless; for the next package the reviewer files are committed before launch (README rule).
3. **The Chebyshev plan is also faster** (0.32× the monomial time, 200 000 `w4p5` pairs): informational, supports making it the default of `edge_channels` in WORK-006/phase 2 (decision then, not now).
4. **Measured for WORK-005 (reviewer probes, see `WORK-005.md` and `docs/work/refs/lap_*`):** the pairwise wall viscosity and the Laplacian wall viscosity are *different operators near a wall* — see §5. This is the main input of the next package.

## 4. Corrections made by the reviewer
None to the model's code or tests. New (uncommitted, the user commits as `REVIEW-004: …`): this file, `docs/work/WORK-005.md`, `docs/work/refs/{review4_c4_indep,lap_calib_probe,lap_scene_probe,lap_lshape_probe,lap_solver_probe,slosh_visc_probe}.py`, `docs/work/refs/viscosity_switch_proto.diff`, `docs/work/README.md` + `REVIEW.md` (sequence, lessons).

## 5. Facts WORK-005 assumes (all measured by the reviewer)
See `WORK-005.md` "Context". In one paragraph: `Δλ = ∫_solid ∇²W dA` has the exact edge form `Σ_e z_e ∫ W′/r ds` and is obtained **without Warp code** as `2 λ[L] − tr Cov[L]` of the ordinary kernel `L = W′/r` (the obvious "register `∇²W` as a kernel" fails: the scene's body-indicator pseudo-pair assumes `∫K = 1` over a full disk, `∫∇²W = 0` → a spurious constant `c/H²` for every particle inside a body); `ν_eff = α c0 H/(8ξ)` follows from a moment identity (not from a wall calibration) and the wall damping of the Laplacian form is **not** proportional to the pairwise one (ratio 12.5 → 0.7 over `z/H = 0.1 … 0.7`): near the wall the Laplacian form damps 3–12× less. A prototype switch passes the physics gate (tank, dam break; dam-break KE 4.0e-4 vs B, arrival 2.4740 t*); with all four switches KE 3.5e-4, same arrival; sloshing T = 1.5 s KE 2.8e-4 vs the default run, `maxVelocity` 0.5919 — all gates PASS.

## 6. Next
`WORK-005.md` (Q1: exact wall Laplacian `Δλ`, `cfg.viscosityExact`, gates). Then WORK-006: all switches on, delete `_solid_samples`, re-validate, re-record the baseline; then phase 2.
