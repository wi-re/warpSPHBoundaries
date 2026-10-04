# REVIEW-006 — WORK-006 (branch `local-model`, `c2ce13d..4673e0c`)

Reviewer: Claude, 2026-10-04. Verdict: **ACCEPT T6.1, T6.2, T6.3, T6.4**. No rule violation, no tolerance edited, no defect found in the code. Two bookkeeping corrections (reviewer's own errors in the work documents, below).

## 1. Mechanical gate
* Ten commits in task order (`T6.1`, BLOCKED report, reviewer's `REVIEW-005b` x2, `T6.1 (resume)`, `T6.2`, `T6.3`, `T6.4`, `REPORT-006`); WORK-007's BLOCKED report follows (correct: its starting state required this review). `git diff --stat`: 24 files, all on the allowed list (plus the two reviewer files); `deltasph_detcmp.py` deleted as specified; **no change to `warpbc.py`, `kernels.py`, `np2d.py`, `tensile.py`, `cover.py`, `cone_area.py`, `deltasph_profile.py`, the reference series, any other test**. `git status` clean.
* `git diff` of `deltasph2d.py` equals spec items 1-5 line by line (config fields, `_family`, `_detect_surface(x, i, j, r, lam)`, the `rhs` viscosity dispatch incl. the `ValueError`, `_surface_state` key `near`, `shift`); `scene.py` +2 lines (the guard, exact text of the spec); `viscosity.py` one adjacency passed to both operations.
* Tests (reviewer, `warp` env): `pytest tests/edge -q` → **756 passed in 378 s** (= the report).

## 2. Evidence audit
| check | result |
|---|---|
| tolerance history of the edited tests | every authorised edit is a deletion of a switch toggle or a rename of the switch; (b), (c) tolerances (3e-2, 1e-2, 5e-4, ...) unchanged in all five files; the C2/C4 "effect on shift" checks deleted exactly as authorised (T6.1 (6)) |
| re-ran the primary acceptance command: `check --physics --cases tank,dambreak` (default config, 480 s) | **PHYSICS GATE: PASS**; every number equals the report and the WORK-005 all-switches values: tank rmseBulk 1.49929753837e-3 / rmseNear 1.81340794898e-3 / keLast 7.2843795517e-7, dam break ke_tstar 0.357363651811 / 0.754228393303 / 0.922766940739 (report ...745: GPU round-off, 8e-12), maxVelocityMax 6.59792039874, densities 0.999428350557 / 1.00603117793, arrival 2.47397042889, KE vs B 3.5344e-4 |
| the same run is also the bit-level check against the **new** baseline | every line PASS, margins 0.000-0.008 (limit 1) — the re-recorded tolerances are not borderline (`ke_tstar2/25` have the 1e-9 relative floor, the old baseline had a 1e-9 spread there; this run differs by 8e-12) |
| second acceptance item: the full suite | 756, see §1 |
| report numbers vs logs | `.tmp/w006_t63_slosh.log` last line = the report's sloshing line (KE 2.8102e-5, maxVel 0.5920, rho [0.99928, 1.00490]); spot check of the equivalence and record numbers against `LOG-006.md` and the baseline json: identical |
| negative controls read | the Laplacian wall test's control (pairwise form, 4.349, must differ by > 0.5 x 4.349) is the independent value, not a scaled copy; the guard test checks both the raise and the same-kernel equality; `test_inside_calls_by_form` has both signs (0 calls default, > 0 pairwise) |
| the guard cannot silently change a call | every `sceneOperation` call that passes an adjacency (`deltasph2d._wall_op/_surface_state`, `dfsph2d._boundary`, `viscosity.lap_lambda_scene`, `scene_bench.py`) builds it with the same kernel as the operation; the whole suite passes with the guard |
| stale references to the removed switches | only history (LOG/REPORT/WORK-001..006, porting-notes rows of 2026-10-03/04, reviewer probes in `refs/` that the work document announced would stop running); one sentence in `deltasph-validation.md` (WORK-005 section: "the default stays the pairwise form") was stale — corrected below |

## 3. Corrections made (commit `REVIEW-006: ...`)
1. **Test count: 756, not 755** (reviewer's error, in WORK-006 *and* WORK-007). `test_adjacency_kernel_guard` is parametrized over `cpu` / `cuda:0` (`test_scene.py` convention), so T6.2 adds 3 cases, not 2. The model reported it correctly and asked; the answer is: **756 is accepted** (753 after T6.1, 756 after T6.2, no test skipped or restricted). `WORK-007.md` now says 756 for the full suite and 82 for the five-file solver set (measured by the WORK-007 session and not re-derived here; the 80 of the WORK-006 T6.1 acceptance plus the two `test_adjacency_kernel_guard` cases of T6.2).
2. `docs/deltasph-validation.md`: a "superseded by WORK-006" note on the WORK-005 sentence that said the default stays pairwise.

## 4. Findings the next work document must assume
* **Behaviour of the new defaults** (measured, equal to the WORK-005 all-switches run): dam-break P1 arrival 2.4897 -> **2.4740** t* (reference 2.49, gate 0.05), `maxVelocityMax` 6.681 -> 6.598, KE vs B 3.5e-4; tank metrics move by ~1e-7. `wallViscosityForm="pairwise"` + exact cover/cone/tensile reproduces the old arrival (2.4897) to the printed digits: the polar -> exact replacement of cover/cone/tensile is not what moved the arrival, the wall-viscosity operator is.
* **Sloshing T = 1.5 s with the new defaults: KE vs the old-default series 2.8e-5** (5 % gate), maxVel 0.5920, rho [0.99928, 1.00490]. This is *10x smaller* than WORK-005's viscosity-only run (2.8e-4), which I did not expect (the viscosity change is part of both). Not a defect and not explained: the series is the old default, the quantity is a max-relative KE difference on a damped, sloshing trajectory, and exact cover/tensile may partly offset the viscosity effect there. Do not quote it as "the effect is small because X"; the honest statement is "inside the gate by 3 orders".
* **Speed:** dam break 44.4 -> **57.5 ms/step**, sloshing 72.1 -> **71.3** (profile, information); `Scene.inside` / `_solid_samples` 0 calls/step. `buildAdjacency` is now ~73 % of the step (10 calls/step, 42-52 ms): the phase-2 target. Exact ops per step (cost probe): tensile 26.6 ms (1 call), cover 16.9 (3 calls), `lap_lambda_scene` 7.4 (2 calls, one adjacency each now), cone 2.9, `_wall_data` 47.6 (3 calls). The dam break is still 30 % slower than before the exact path; the sloshing is not.
* `DeltaSPH2D` requires `SurfaceRep` walls (guard test `test_deltasph_requires_surface_walls`); `VolumeRep`/`ImplicitRep` stepping of the solver is not exercised anywhere now. Scene-layer representation independence is covered by `test_scene.py` / `test_dfsph.py`.
* `sceneOperation` rejects an adjacency of a different kernel (`ValueError: ... built for kernel ...`); an adjacency is still per-kernel moments, not a neighbour list.
* The harness `check` bit-level lines are against the **re-recorded** baseline (`results/deltasph/regress_baseline.json`, new default); the old default's numbers are in `deltasph-validation.md` (WORK-006 section, table in (3)).

## 5. Weak spots (none blocking)
* `test_tensile_exact_c4_and_c2_both_work` now only checks that the shift runs and is finite (the spec said so; the name is stale). The exactness of tensile is carried by the (a) polar-vs-exact checks of the same files and by `test_tensile_scene*`.
* The wall-Laplacian solver test pins the flat-wall particle at `z = dp/2`; moving walls (rolling tank) are covered by the sloshing gate only (no absolute check), as noted in REVIEW-005.
* The 79/80/82 counts of the "solver tests" set drift with every package that adds a test there; later documents should say "passes" and quote the full-suite number only.

## 6. Next
WORK-007 (no-slip) is unblocked: `git log -1 --format=%s` now starts with `REVIEW-006`; starting state 82 / 756. Merge of `local-model` into `main` and the tag `reviewed-006` are the user's call (no `reviewed-*` tag exists yet; the base of the next review is `4673e0c`).
