# REPORT-006
Status: **DONE**          Branch: `local-model`   HEAD: `7054c2f` (T6.4, last task commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-04

(Supersedes the BLOCKED version of 2026-10-04: the blocker, one unauthorised test pinning `VolumeRep` solver stepping, was resolved by the reviewer's RESUME NOTE — `DeltaSPH2D` requires `SurfaceRep` walls, the test replaced by the guard `test_deltasph_requires_surface_walls` — and the work continued to the Definition of done.)

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T6.1 exact cover/cone/tensile only; `wallViscosityForm` | DONE | `1c23e89` + `f6d06f7` | code + every authorised test edit in `1c23e89` (kept, not redone); the RESUME NOTE's guard test replaces the `VolumeRep` solver test; acceptance: 21 changed/new tests, the 79-solver-test set (80 passed = 79 + the one new Laplacian test), full suite 753 passed, equivalence run PHYSICS GATE: PASS with every line equal to the WORK-005 T5.3(1b) all-switches numbers within 1e-6 relative (most identical to the last printed digit) |
| T6.2 one adjacency per call; the adjacency kernel guard | DONE | `f789f70` | `lap_lambda_scene` builds one adjacency and passes it to both scene operations; `sceneOperation` rejects an adjacency whose kernel does not match the operation's; 4-file acceptance 73 passed, full suite 756 passed; per-call timing (own probe, 468 near particles) w2 6.82 → 3.44 ms, w4 9.46 → 4.55 ms, 2 → 1 adjacency |
| T6.3 gates, baseline, profile | DONE | `1e08db4` | pairwise form PHYSICS GATE: PASS (arrival 2.4897 t*, the WORK-004 configuration); baseline re-recorded (spread tol/|v| ≤ 1.4e-6, `steps` spread 0), `check` OVERALL: PASS, `check --physics` PHYSICS GATE: PASS; sloshing KE 2.8102e-5 (10× below the WORK-005 viscosity-only 2.82e-4, far below the 5e-3 finding threshold); profile 57.515 / 71.307 ms/step with `_solid_samples` and `Scene.inside` at 0 calls/step; reviewer's unmodified cost probe pasted |
| T6.4 documentation | DONE | `7054c2f` | porting-notes change-log row + validation section `## Exact wall operations by default (WORK-006)`; every number is the one from the `.tmp/` log |

## Acceptance evidence

All commands run in the conda env `warp` (`/home/lu26029/miniconda3/envs/warp/bin`), branch `local-model`, 2026-10-04, RTX PRO 6000 with the resident local LLM (~69 GiB of 97887 MiB); full captures in `LOG-006.md` and the named `.tmp/` logs.

### T6.1 (resume)

1. Changed + new tests:
   `pytest tests/edge/test_deltasph_cover.py tests/edge/test_deltasph_cone.py tests/edge/test_deltasph_tensile.py tests/edge/test_deltasph_tensile_c4.py tests/edge/test_deltasph_viscosity.py tests/edge/test_deltasph.py tests/edge/test_deltasph_exact_only.py -q`
   → **`21 passed, 14 warnings in 19.52s`**. Includes the new guard `test_deltasph_requires_surface_walls` (i: `domain="volume"` → first `step()` raises `NotImplementedError: cover_vector_scene: SurfaceRep bodies only`; ii: 25 finite surface steps), `test_wall_viscous_term_is_the_wall_laplacian`, and the 3 tests of `test_deltasph_exact_only.py`.
2. The 79 solver tests:
   `pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q`
   → **`80 passed, 14 warnings in 99.45s (0:01:39)`**. 80 = the 79 of the starting state + the one new Laplacian test added to `test_deltasph.py` by the authorised T6.1(7) edit (the acceptance's "79" is the pre-edit count of this file set; all 79 originals pass).
3. Full suite (post-edit tree): `python -m pytest tests/edge -q` → **`753 passed, 15 warnings in 404.76s (0:06:44)`** (`.tmp/w006_t61_full.log`). 749 starting + 4 new, no test added beyond those.
4. Equivalence run (T6.1(8), the primary evidence): `cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak` (default config) → `.tmp/w006_t61_equiv.log`:
   ```
   tank  rmseBulk value= 0.00149929753837   rmseNear value= 0.00181340794898   keLast value= 7.2843795517e-07   rhoMin 1.00008342046   rhoMax 1.00246989449   steps 3502
   [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 3.5344e-04 (668 samples)
   [physics] dam break P1 arrival = 2.4740 t*  (reference 2.49, |diff| = 0.0160)
   dambreak  ke_tstar1 0.357363651811   ke_tstar2 0.754228393303   ke_tstar25 0.922766940745
             p0_arrival_tstar 2.47397042889   maxVelocityMax 6.59792039874   minDensityMin 0.999428350557   maxDensityMax 1.00603117793   steps 6683
   PHYSICS GATE: PASS
   ```
   Every line equals the WORK-005 T5.3(1b) all-switches numbers within 1e-6 relative (tank rmseBulk rel diff 7e-12, keLast 3e-13, rmseNear identical; dam break ke_tstar/maxVelocity/densities identical to the printed digits, arrival identical to 2.47397042889, KE 3.5344e-4 = the WORK-005 all-switches value). The bit-level lines FAIL against the OLD baseline by design (re-recorded in T6.3). All physics gates PASS.

### T6.2

1. `python -m pytest tests/edge/test_viscosity_scene.py tests/edge/test_scene.py tests/edge/test_deltasph_viscosity.py tests/edge/test_deltasph.py -q`
   → **`73 passed, 15 warnings in 36.01s`** — every previous tolerance unchanged; new: `test_one_adjacency_per_call[cuda:0]`, `test_adjacency_kernel_guard[cpu]`/`[cuda:0]`.
2. Full suite: `python -m pytest tests/edge -q` → **`756 passed, 15 warnings in 401.11s (0:06:41)`** (`.tmp/w006_t62_full.log`). The work document expected 755 = 753 + 2 new test FUNCTIONS; the actual is 753 + 3 parametrized tests because `test_scene.py` parametrizes over `DEVICES = ["cpu", "cuda:0"]` (file convention), so `test_adjacency_kernel_guard` contributes 2. Counting difference, not a code difference (see Deviations).
3. Timing of `lap_lambda_scene` (own probe `.tmp/w006_lap_timing.py`, developed dam-break state nx67, 300 steps, `near` = 468 at λ > 1e-9): BEFORE w2 2 adjacencies 6.82 ms/call, w4 2 adjacencies 9.46 ms/call; AFTER w2 1 adjacency 3.44 ms/call, w4 1 adjacency 4.55 ms/call (reviewer's reference `docs/work/refs/review5_lap_adjacency_probe.py`: 7.17 → 3.65, 9.82 → 4.78 — same scale; wall times on this shared GPU vary 2–3×).
4. The guard cannot fire on existing code: every `sceneOperation` call site that passes an adjacency builds it with the same kernel it uses it for (checked: `deltasph2d._wall_op`/`_wall_data`/`_surface_state`, `dfsph2d._boundary`/`self.adj`; `cover_vector_scene`'s `adjacency` parameter is passed `None` by every caller). Confirmed by the green full suite.

### T6.3

1. `cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg wallViscosityForm=pairwise` → `.tmp/w006_t63_pairwise.log`: **PHYSICS GATE: PASS**. Decisive lines: tank rmseBulk 0.00149879553508 (≤ 0.001873494419 = 1.25 × baseline, PASS), rmseNear 0.00181374508459 (PASS), keLast 7.00604738264e-07 (≤ 3.503023691e-06 = 5 ×, PASS), rhoMin 1.00008277812 (≥ 0.99, PASS), rhoMax 1.00246999588 (≤ 1.01, PASS), steps 3502; dam break KE vs B 2.4098e-05 (668 samples, ≤ 0.05, PASS), **P1 arrival 2.4897 t* (reference 2.49, |diff| = 0.0003, PASS)** — it equals the old baseline's 2.48970315022 to the full printed precision, i.e. with exact cover/cone/tensile + pairwise wall term this is the WORK-004 configuration; ke_tstar1/2/25 = 0.357284798075 / 0.754059943541 / 0.922531956854, maxVelocityMax 6.67866530633, minDensityMin 0.999436862861 (≥ 0.97, PASS), maxDensityMax 1.00628666806 (≤ 1.03, PASS), steps drift 0 (PASS).
2. `python -m edgebound.deltasph_regress record --cases tank,dambreak` → `.tmp/w006_t63_record.log` wrote `results/deltasph/regress_baseline.json` (the only baseline edit of the package). Spread / tolerance table: **every tol/|value| ≤ 1.373e-06 (≤ 5 %; no WEAK flags), `steps` spread 0** (tank rmseBulk 0.001499297538 spread 1.004e-14 … dambreak ke_tstar2 0.7542283933 spread 7.597e-13, arrival 2.473970429 spread 0.000e+00, steps 6683 spread 0.000e+00). Then `check --cases tank,dambreak` → `.tmp/w006_t63_check_bit.log`: **OVERALL: PASS** (bit-level, every line PASS, margins 0.000–0.013); `check --physics --cases tank,dambreak` → `.tmp/w006_t63_check_phys.log`: **PHYSICS GATE: PASS** (tank rmseBulk ≤ 1.25 ×, rmseNear ≤ 1.25 ×, keLast ≤ 5 ×, rhoMin ≥ 0.99, rhoMax ≤ 1.01; dam break KE ≤ 0.05, |arrival − 2.49| = 0.0160295711131 ≤ 0.05, minDensity ≥ 0.97, maxDensity ≤ 1.03, steps drift 0).
3. `python .tmp/slosh_default.py` (default config = `run_sloshing(nx=200, T=1.5, shifting=True, noPen="impulse", verbose=True)`, background) → `.tmp/w006_t63_slosh.log`: completed in 1085 s wall (inside the 90-min hard timeout): **`steps 15001 wall 1085 s  KE rel max vs slosh_B = 2.8102e-05 (1500 samples, gate 5 %)  maxVel 0.5920  rho [0.99928, 1.00490]`**. 2.8102e-5 is 10× below the WORK-005 `viscosityExact`-only 2.8231e-4 (0.5919, [0.99928, 1.00489]) and far below the 5e-3 finding threshold; maxVel/densities agree with the old default (0.5921, [0.99929, 1.00487]) to the last printed digit. The stored `slosh_B_nx200` series is the old default, so 2.8102e-5 (0.0028 % over 1.5 s) is the effect of the whole WORK-006 default change.
4. `cd python && python -m edgebound.deltasph_profile` → `.tmp/w006_t63_profile.log`: dam break **57.515 ms/step** (timer; wall 57.520; before WORK-006 44.383), sloshing **71.307 ms/step** (timer; wall 71.313; before 72.095); **`_solid_samples` and `Scene.inside` 0.000 calls/step in both cases**; `Scene.buildAdjacency` 10 calls/step = 41.8279 ms/step (72.73 %) dam break / 52.2886 ms/step (73.33 %) sloshing (now the single largest cost); `sceneOperation` 12 calls/step. Reviewer estimate ≈55 / ≈70 ms/step (information, not a gate) — consistent. Reviewer cost probe `docs/work/refs/review5_cost_probe.py` (**unmodified**, from `python/`, `PYTHONPATH=.`) → `.tmp/w006_t63_cost_probe.log`:
   ```
   N 3240 near 468 edges 4
   cover_vector_scene                  5.64 ms/call  adjacency builds/call 1   calls/step 3  ->  16.92 ms/step
   cone_area_scene (th/2)              0.52 ms/call  adjacency builds/call 0   calls/step 3  ->   1.55 ms/step
   cone_area_scene (pi)                0.44 ms/call  adjacency builds/call 0   calls/step 3  ->   1.33 ms/step
   tensile_vector_scene w2            26.63 ms/call  adjacency builds/call 1   calls/step 1  ->  26.63 ms/step
   lap_lambda_scene w2                 3.72 ms/call  adjacency builds/call 1   calls/step 2  ->   7.44 ms/step
   _wall_data (existing, per call)    15.88 ms/call  adjacency builds/call 1   calls/step 3  ->  47.64 ms/step
   sum of exact ops per step: 53.9 ms
   ```
   Both tables + the before/after step times + this probe table are appended to `docs/deltasph-profile.md` (`## After WORK-006 (exact wall operations, default form)`).

### T6.4

Files exist: `docs/deltasph-porting-notes.md` (one change-log row directly under the `cfg.viscosityExact` row, no blank line) and `docs/deltasph-validation.md` (`## Exact wall operations by default (WORK-006)` after the WORK-005 section, before `## 7. Next`). The numbers in them are the ones in the log: each value above was copied from the named `.tmp/` log (T6.1(8) equivalence, T6.3(1)–(3)).

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| T6.1 changed + new tests | 21 passed | = 21 | 0 | `LOG-006.md` T6.1 checks (19.52s) |
| T6.1 79-solver-test set | 80 passed (79 + 1 new Laplacian test) | = 79 originals | 0 | `LOG-006.md` (99.45s) |
| T6.1 full suite | 753 passed | = 753 | 0 | `.tmp/w006_t61_full.log` |
| equivalence tank rmseBulk | 0.00149929753837 | WORK-005 all-switches 1.49929753836e-3 (rtol 1e-6) | rel diff 7e-12 | `.tmp/w006_t61_equiv.log` |
| equivalence tank keLast | 7.2843795517e-07 | 7.28437955168e-7 (rtol 1e-6) | rel diff 3e-13 | `.tmp/w006_t61_equiv.log` |
| equivalence dam break KE vs B | 3.5344e-04 (668 samples) | gate ≤ 0.05; WORK-005 all-switches 3.5344e-4 (abs 1e-5) | gate 0.05 | `.tmp/w006_t61_equiv.log` |
| equivalence dam break arrival | 2.47397042889 t* | reference 2.49 (abs 0.05); WORK-005 same value | 0.0160 | `.tmp/w006_t61_equiv.log` |
| T6.2 4-file acceptance | 73 passed | = 73, every previous tolerance unchanged | 0 | `LOG-006.md` T6.2 checks (36.01s) |
| T6.2 full suite | 756 passed | doc: 755 functions | +1 parametrized test (counting, see Deviations) | `.tmp/w006_t62_full.log` |
| T6.2 `lap_lambda_scene` w2 / w4 | 3.44 / 4.55 ms/call (was 6.82 / 9.46) | information (reviewer 3.65 / 4.78) | 2 → 1 adjacency | `.tmp/w006_lap_timing.py` |
| T6.3(1) pairwise arrival | 2.4897 t* | reference 2.49 (abs 0.05) | 0.0003 (= old baseline to full precision) | `.tmp/w006_t63_pairwise.log` |
| T6.3(1) pairwise KE vs B | 2.4098e-05 (668 samples) | gate ≤ 0.05 | 0.04998 | `.tmp/w006_t63_pairwise.log` |
| T6.3(2) record spread | max tol/|value| = 1.373e-06 (tank keLast); `steps` spread 0 | ≤ 5 % | ~3.6e4× | `.tmp/w006_t63_record.log` |
| T6.3(2) bit-level check | OVERALL: PASS (every line, margins 0.000–0.013) | bit-level vs new baseline | — | `.tmp/w006_t63_check_bit.log` |
| T6.3(2) physics check | PHYSICS GATE: PASS (all 10 gate lines) | the 10 gate limits | see T6.3(2) lines | `.tmp/w006_t63_check_phys.log` |
| new baseline arrival | 2.473970429 t* (old 2.48970315022) | — | 0.0157 t* earlier (the expected operator change, = WORK-005 all-switches) | `results/deltasph/regress_baseline.json` |
| new baseline ke_tstar2 / ke_tstar25 | 0.7542283933 / 0.9227669407 (old 0.754074340911 / 0.922537582805) | — | rel 2.0e-4 / 2.5e-4 (expected) | `results/deltasph/regress_baseline.json` |
| T6.3(3) sloshing KE vs `slosh_B_nx200` | 2.8102e-05 (1500 samples) | gate 5 %; finding threshold 5e-3 | 1.8e-4 below finding; 10× below WORK-005's 2.8231e-4 | `.tmp/w006_t63_slosh.log` |
| T6.3(3) sloshing maxVel / rho | 0.5920 / [0.99928, 1.00490] | information (old default 0.5921 / [0.99929, 1.00487]) | last printed digit | `.tmp/w006_t63_slosh.log` |
| T6.3(4) dam break step | 57.515 ms/step (timer) | information (reviewer ≈55; before 44.383) | consistent | `.tmp/w006_t63_profile.log` |
| T6.3(4) sloshing step | 71.307 ms/step (timer) | information (reviewer ≈70; before 72.095) | consistent | `.tmp/w006_t63_profile.log` |
| T6.3(4) `_solid_samples` / `Scene.inside` | 0.000 calls/step (both cases) | = 0 | 0 | `.tmp/w006_t63_profile.log` |
| final full suite (after last code edit) | 756 passed, 15 warnings in 404.75s | doc: 755 functions | +1 parametrized test (counting) | `.tmp/w006_final_suite.log` |

## Deviations

* **One-line module-docstring update in `tests/edge/test_deltasph.py` (T6.1 resume).** The old sentence named the replaced test's claim (solver "representation independence"); it now states the SurfaceRep requirement and where the scene-layer independence lives. Disclosed in `LOG-006.md` (RESUMED); no other line of the file changes beyond the authorised test replacement.
* **Tensile test sub-check renumbering (session 1, comment-only).** After deleting the authorised effect-on-shift checks, the surviving checks were renumbered to keep the lettering sequential (C2: (a),(b),(c); C4: (a),(b),(c)). No tolerance or check logic changed; the spec did not specify the letters of the surviving checks.
* **Viscosity test docstrings (session 1, comment-only).** The spec authorised only the cfg / toggle / (a) changes to `test_deltasph_viscosity.py`; the module docstring title and the `d_ex = ...` definition (and the (d) comment / print line) were also updated because they referenced the removed `viscosityExact` switch. No tolerance or check logic changed.
* **Bug fix in my own NEW test (T6.2).** `test_one_adjacency_per_call` had a numpy/tensor type bug (`got - ref` with `ref` a tensor); the count assertion `n_adj == 1` had already passed. Fixed the new test only (wrap `ref` in `.cpu().numpy()`); no tolerance or check of any previous test touched (permitted: a new test of my own, shown wrong independently).
* **756 vs 755 (counting difference, not a code difference).** The work document's expected count (T6.2: 755; Definition of done: 755) counts test functions; `test_adjacency_kernel_guard` is parametrized over `DEVICES = ["cpu", "cuda:0"]` (file convention, as every other test in `test_scene.py`), so it contributes 2 tests. No test was skipped, removed or device-restricted to game the count.
* **Old-baseline values in the work document vs the baseline file.** The document quotes the old dam-break ke_tstar2/25 as 0.754074336958 / 0.922537627023; the pre-re-record baseline file had 0.754074340911 / 0.922537582805 (8th-digit differences). The file is the harness's reference — the file values were used.

## Failures and open questions

* **Nothing blocking.** The session-1 BLOCKED (`test_dynamics_independent_of_the_wall_representation`) was resolved by the reviewer's RESUME NOTE (the SurfaceRep requirement + guard test) and no question from it remains open.
* **756 vs 755**: the final suite runs 756 parametrized tests; the Definition of done's 755 counts test functions (see Deviations). Question: is 756 with the documented counting difference accepted as meeting the line, or should the count be re-stated in the work document?
* **8th-digit old-baseline discrepancy** (see Deviations): the baseline file values were used throughout; the document's quoted values appear to be a transcription slip.
* **GPU wall times vary 2–3×** with the resident local LLM (49–88 % util, ~69 GiB of 97887 MiB). All gates are physics/bit-level and are unaffected; the ms/step numbers (T6.2 timing, T6.3(4) profile) are information only and were made back to back where compared.
* The bit-level lines of the T6.1 equivalence run FAIL against the OLD baseline **by design** (the baseline is the old default; T6.3 re-records it, and `check` against the new baseline is OVERALL: PASS).

## Files changed

`git diff --stat c2ce13d..HEAD` (`c2ce13d` = the branch head before the first T6.1 commit `1c23e89`; the range includes the two reviewer REVIEW-005b commits `2f39ac7`/`a5e7194`, which touched `docs/work/WORK-006.md` +11 and `docs/work/REVIEW.md` +3; this report — replacing the BLOCKED version — and the final log lines are added by the last commit):

```
 docs/deltasph-porting-notes.md         |   1 +
 docs/deltasph-profile.md               |  80 +++++++++++++++++++++++
 docs/deltasph-validation.md            |  27 ++++++++
 docs/work/REVIEW.md                    |   3 +
 docs/work/WORK-006.md                  |  11 +++-
 docs/work/logs/LOG-006.md              | 181 +++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/work/logs/REPORT-006.md           | 117 ++++++++++++++++++++++++++++++++
 python/edgebound/deltasph2d.py         | 109 ++++++++------------------
 python/edgebound/deltasph_detcmp.py    |  95 ------------------- (deleted)
 python/edgebound/deltasph_regress.py   |   2 +-
 python/edgebound/scene.py              |   2 +
 python/edgebound/viscosity.py          |   8 ++-
 results/deltasph/regress_baseline.json |  79 ++++++++++----------
 tests/edge/test_deltasph.py            |  97 ++++++++++++++++++++----
 tests/edge/test_deltasph_cone.py       |  17 ++---
 tests/edge/test_deltasph_cover.py      |  12 +--
 tests/edge/test_deltasph_exact_only.py |  76 +++++++++++++++++++ (new)
 tests/edge/test_deltasph_tensile.py    |  45 ++++++--------
 tests/edge/test_deltasph_tensile_c4.py |  67 +++++++-----------
 tests/edge/test_deltasph_viscosity.py  |  27 +++++----
 tests/edge/test_scene.py               |  21 ++++++
 tests/edge/test_viscosity_scene.py     |  47 +++++++++++++
 22 files changed, 801 insertions(+), 323 deletions(-)
```

## What I did NOT verify

* `ImplicitRep` / `SdfRep` (or `VolumeRep`) solver stepping: the guards stay (out of scope); `DeltaSPH2D` now requires `SurfaceRep` walls and the guard is tested (`test_deltasph_requires_surface_walls`), but stepping a non-SurfaceRep body was never exercised (it raises before it could be).
* The pairwise form beyond the T6.3(1) physics gate: no bit-level equivalence claim against the WORK-004 run (its arrival equals the old baseline to the full printed precision; that is a gate line, not a bit-level check).
* The full-length (7 s) sloshing run (out of scope; T = 1.5 s truncation as specified).
* Performance tuning: step times are information (the Definition of done forbids it); the profile's new dominant cost, `Scene.buildAdjacency` (≈73 % of the step), is listed, not addressed.
* Anything in `~/dev/warpSPH*` (out of scope): the porting notes document the change; the warpSPH side was not touched.
* The T6.1 equivalence run compares against the WORK-005 values at rtol 1e-6 (as the work document specifies), not at bit level; bit-level reproduction is established separately by the T6.3 re-record + `check` (OVERALL: PASS against the new baseline).
