# REPORT-004

Status: **DONE**          Branch: `local-model`   HEAD: `35c9044` (T4.3, last code commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-03

WORK-004 adds an **opt-in** Chebyshev-quadrature edge plan to `warpbc.py` (the monomial plan and the polar quadrature remain the unchanged defaults), routes the delta+ shifting tensile term through it for **both** the Wendland C2 and C4 families, enables `cfg.tensileExact` for C4 in the solver, and runs the T4.4 gates. No switch is made a default. The motivation: the Warp monomial edge plan loses digits with kernel degree (Wendland C4 `W^5` is degree 40: 1.66e-3 relative error in the gradient channel); the Chebyshev series on [0,R] + Gauss panels around the foot point is machine-precise at the same (and lower) cost.

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T4.1 | DONE | `5a701e3` | Chebyshev-quadrature plan + kernel in `warpbc.py` (opt-in `stable=` / `STABLE_KERNELS`); degree-40 `w4p5` gradient channel machine-precise (9.43e-16 generic / 2.69e-12 at a vertex, vs 1.66e-3 monomial); 6 new tests; 79 solver + 734 full suite |
| T4.2 | DONE | `ba2ef6e` | `tensile_vector_scene` for families `w2` **and** `w4`, both on the stable plan; C2 re-routed (one rule); 5 new tests; 739 full suite |
| T4.3 | DONE | `35c9044` | `tensileExact` for Wendland C4 in the solver (guard + config comment); 2 new tests, the vacuous C4 guard test deleted; 740 full suite |
| T4.4 | DONE | – (no code change; the log rides with this report) | both `--physics` checks **PHYSICS GATE: PASS** (bit-level KE/velocity/density lines FAIL as expected); sloshing default vs `tensileExact` both within the 5 % KE band of `slosh_B_nx200`; default `check --cases tank,dambreak` **OVERALL:PASS** |

## Acceptance evidence

All commands run in the `warp` env (`export PATH=/home/lu26029/miniconda3/envs/warp/bin:$PATH`), `cuda:0`, one GPU job at a time. Full captures in `.tmp/w004_*.log`; the decisive lines are copied below.

### T4.1 — Chebyshev-quadrature plan in `warpbc.py` (opt-in)

```
$ python -m pytest tests/edge/test_warpbc_stable.py -q -s
(a) w2p5: generic(206) worst 7.80e-16   all(214) worst 7.80e-16   scale 3.417e+00
(a) w4p5: generic(206) worst 9.43e-16   all(214) worst 2.69e-12   scale 3.767e+00
(b) w4p5 negative controls: monomial 1.66e-03 (> 1e-4)   stable(8,4) 4.66e-05 (> 1e-6)
(c) all-9ch max|stable - mono| 1.29e-10 (<= 5e-9, 6 kernels x 2 supports)
(d) stable Warp vs np2d.gradient(stable=(16,8)) b7: max|diff| 5.55e-15 (<= 1e-13)
(e) registry semantics OK: default == explicit(16,8); stable=False == DevicePlan; unreg None == False; default != mono
(f) w4p5 z==0 / |z|=1e-9 pairs finite: OK
6 passed in 7.28s
```

Tolerances (stated before looking, in the test): 1e-13 generic / 5e-11 all-points vs the mpmath `block_grads` reference; 5e-9 channel agreement; 1e-13 for the independent `np2d` stable route. The degree-40 `w4p5` gradient channel is machine-precise through the stable route (generic 9.43e-16, worst-at-a-vertex 2.69e-12) versus 1.66e-3 for the monomial path.

Refactor is additive: `git diff --stat 115c16d -- python/edgebound/warpbc.py` = `230 insertions(+), 7 deletions(-)`; the 7 deletions are exactly the authorised `build_plan` refactor + the opt-in branch (the monomial launch is textually unchanged; `build_plan` vs `build_cheb_plan` give identical radii/nE/nV/rin_idx).

Informational timing (`.tmp/t41_timing.py`, 200000 (point, edge) pairs of `w4p5`, unit square, min of 3): monomial 260.7 ms vs chebyshev 83.3 ms (ratio 0.32x), identical `max|c|` = 3.7667e+00 -- the stable route is both more accurate (~1e-16 vs ~1e-3) and ~3x faster for the degree-40 kernel.

### T4.2 — `tensile_vector_scene` for `w2` and `w4`

```
$ python -m pytest tests/edge/test_tensile_scene.py tests/edge/test_tensile_scene_c4.py -q -s
10 passed, 14 warnings in 3.44s
- (a) w4: scene vs factor*np2d.gradient(w4p5, stable) 3.52e-15 (scale 7.510e+02); w2: 3.63e-15 (scale 2.349e+02)
- (b) T_y(H=1,(0,0.3)) = -0.1978237590 (grid 6.13e-06); T_y(H=0.5,(0,0.15)) = -101.2857645825 (grid 6.13e-06)
- (c) 1.01*T - grid = 1.00e-02, -T - grid = 2.00e+00; monomial route vs stable = 2.20e-03
```

Tolerances (fixed in the spec, written before looking): (a) <= 1e-10 max|T| vs `factor * np2d.gradient(stable=(16,8))`; (b) <= 5e-5 rel vs the 2000^2 midpoint grid + smoke rtol 1e-6 + T_x = 0 + sign T_y < 0; (c) the `1.01*T`/`-T` and the monomial-route controls must exceed 1e-3 / 1e-4. The w4 `3.52e-15` / `2.20e-3` and the `6.13e-6` grid error all match the reviewer's measured numbers. The C2 tensile term is re-routed through the stable plan (one rule for both families); the accepted WORK-003 C2 tolerances stay valid (C2 values change by <= 6e-8 relative).

### T4.3 — `tensileExact` for Wendland C4 in the solver

```
$ python -m pytest tests/edge/test_deltasph_tensile.py tests/edge/test_deltasph_tensile_c4.py -q -s
3 passed, 14 warnings in 2.89s
- (a) C4: near=304  max|T_exact| = 7.346e+07  |T_quad - T_exact|/scale = 1.749e-02  |T_quad + T_exact|/scale = 2.017
- (b) C4 shift effect max|u_q - u_e| / max||u_q|| = 5.9079e-02
- (c) C4: five steps with tensileExact=True finite
- (d) C4: exact shift finite; max||u_q - u_e|| = 1.0174e-05 (> 0)   C2: ... = 6.4279e-06 (> 0)
```

Tolerances (fixed in the spec, written before looking): (a) max|T_quad - T_exact| <= 3e-2*max|T_exact|, max|T_exact| > 1e7, sign control > 1.0*max|T_exact|; (b) 5e-3 <= max|u_q - u_e|/max||u_q|| <= 0.15; (c) five steps finite; (d) both families' exact results differ from their polar counterparts. The C4 numbers (1.749e-2 = the polar quadrature's own error, 2.017 sign, 5.91e-2 shift effect) all match the reviewer's measured values. `git diff HEAD -- python/edgebound/deltasph2d.py` is exactly the two authorised edits; the `test_deltasph_tensile.py` diff is only the deletion of the vacuous C4 guard.

### T4.4 — gates (no code change; `.tmp/` logs)

**(1) `check --physics --cases tank,dambreak --cfg tensileExact=true`** (`.tmp/w004_t44_gate1.log`):

```
tank  (103.1 s wall)
  rmseBulk           value= 0.00149879553508 baseline= 0.00149879553509 tol= 1.499e-12 margin= 0.007  PASS
  rmseNear           value= 0.0018137450846 baseline= 0.00181374508459 tol= 1.814e-12 margin= 0.003  PASS
  [gate] rmseBulk  <= 0.001873494419 (1.25 x baseline)  PASS ; rmseNear  <= 0.002267181356  PASS ; keLast/rhoMin/rhoMax  PASS
dambreak  (453.7 s wall)
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 2.0904e-05 (668 samples)
  [physics] dam break P1 arrival = 2.4897 t*  (reference 2.49, |diff| = 0.0003)
  ke_tstar1/ke_tstar2/ke_tstar25/maxVelocityMax/minDensityMin/maxDensityMax  FAIL   (bit-level, expected)
  [gate] KE rel max vs B = 2.09040175075e-05  <= 0.05  PASS ; P1 arrival |diff| = 0.0003  <= 0.05  PASS
  [gate] minDensityMin 0.999437 >= 0.97  PASS ; maxDensityMax 1.006298 <= 1.03  PASS ; steps drift 0  PASS
PHYSICS GATE: PASS   EXIT=0
```

**(2) `check --physics --cases tank,dambreak --cfg coverExact=true,coneExact=true,tensileExact=true`** (`.tmp/w004_t44_gate2.log`):

```
tank  (167.4 s wall)
  rmseBulk           value= 0.00149879553508 ... PASS ; rmseNear 0.00181374508461 ... PASS ; all [gate] PASS
dambreak  (534.2 s wall)
  [physics] dam break KE vs dambreak_B_nx67: ... = 2.4098e-05 (668 samples)
  [physics] dam break P1 arrival = 2.4897 t*  (reference 2.49, |diff| = 0.0003)
  ke_tstar1/ke_tstar2/ke_tstar25/maxVelocityMax/minDensityMin/maxDensityMax  FAIL   (bit-level, expected)
  [gate] KE rel max vs B = 2.40976600738e-05  <= 0.05  PASS ; P1 arrival |diff| = 0.0003  PASS
  [gate] minDensityMin 0.999437 >= 0.97  PASS ; maxDensityMax 1.006287 <= 1.03  PASS ; steps drift 0  PASS
PHYSICS GATE: PASS   EXIT=0
```

The still-water tank barely moves under any switch (the tensile term is negligible there; rmseBulk/rmseNear stay within ~0.7 % of the 1e-12 bit-level tolerance, so those lines still PASS). The dam break shows the **expected** bit-level FAILs on the KE / velocity / density lines (the exact term changes the trajectory by far more than the 1e-8-ish bit-level tolerance) while the PHYSICS GATE passes with a wide margin (KE vs B <= 2.4e-05 << 5 %; P1 arrival diff 0.0003 << 0.05; densities in [0.97, 1.03]; zero step drift). This is the intended behaviour of an opt-in accuracy switch: the gate (physics) is unaffected, the bit-level regression lines are not.

**(3) sloshing** (`.tmp/slosh_gate.py`, `.tmp/w004_t44_slosh.log`): `run_sloshing(nx=200, T=1.5, shifting=True, noPen="impulse")` for `cfg = {}` and `cfg = {"tensileExact": True}`, one after the other, hard timeout 90 min each.

```
$ python -u .tmp/slosh_gate.py        # run_sloshing(nx=200, T=1.5, shifting=True, noPen="impulse")  default, then tensileExact=True
RUN default DONE  wall=1245s steps=15001
RUN exact DONE    wall=1320s steps=15001
wall time:  default = 1245 s   exact = 1320 s   (steps 15001 / 15001)
KE vs slosh_B_nx200 (default): max|KE-KE_B|/max(KE_B) = 7.3239e-08 (1500 samples)
KE vs slosh_B_nx200 (exact):   max|KE-KE_B|/max(KE_B) = 5.7811e-05 (1500 samples)
KE between runs: max|KE_exact - KE_default| / max(KE_default) = 5.7812e-05
maxVelocity max:  default = 0.5921   exact = 0.5919
minDensity min :  default = 0.99929   exact = 0.99929
maxDensity max :  default = 1.00487   exact = 1.00489
sensor probe:  n(NaN) default=4 exact=5  nanmax|p_exact - p_default| = 3.4031e+00  / nanmax|p_default| = 6.1541e-03
SLOSH_EXIT=0
```

Both runs complete (no 90-min timeout). Against the reviewer's numbers: KE vs B default **7.3239e-08** (reviewer 4.24e-8 -- same order; the default run is the baseline itself, so its small offset from B is the solver's own GPU-reduction-order determinism) and exact **5.7811e-05** (reviewer 5.78e-5, match); between runs **5.7812e-05** (reviewer 5.78e-5, match); maxVelocity **0.5921 / 0.5919** (reviewer 0.5921 / 0.5919, match); densities **[0.99929, 1.00487] / [0.99929, 1.00489]** (reviewer [0.9993, 1.0049] / [0.99929, 1.00489], match). Both runs are deep inside the 5 % KE band (7.32e-08 and 5.78e-05, both < 0.05 by 3 orders); the two runs differ by 5.78e-05 (0.006 %, far below "percent level"); the sensor-pressure probe difference is 6.15e-03 (0.6 %) with 4/5 leading NaN samples (probe undefined while <3 particles are near the sensor), handled with `nanmax` -- no NaN in any comparison quantity and nothing off by an order of magnitude. The exact term does not break the run (finite, in-band, densities in [0.97, 1.03]). No finding.

**(4) default `check --cases tank,dambreak`** (no `--cfg`, after the last code edit T4.3) (`.tmp/w004_t44_gate4.log`):

```
$ cd python && python -m edgebound.deltasph_regress check --cases tank,dambreak
cfg: {} (defaults)
tank  (162.9 s wall)
  rmseBulk/rmseNear/keLast/rhoMin/rhoMax/steps  all PASS   (defaults reproduce the baseline to within the bit-level tolerance)
dambreak  (366.7 s wall)
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 2.4762e-07 (668 samples)
  [physics] dam break P1 arrival = 2.4897 t*  (reference 2.49, |diff| = 0.0003)
  ke_tstar1          value= 0.357285986624 baseline= 0.357285986624 tol= 3.573e-10 margin= 0.000  PASS
  ke_tstar2          value= 0.754074343774 baseline= 0.754074340911 tol= 2.342e-08 margin= 0.122  PASS
  ke_tstar25         value= 0.922537622852 baseline= 0.922537582805 tol= 7.268e-08 margin= 0.551  PASS
  p0_arrival_tstar   value= 2.48970315022 baseline= 2.48970315022 tol= 2.490e-09 margin= 0.000  PASS
  maxVelocityMax     value= 6.68104559165 baseline= 6.68104559177 tol= 6.681e-09 margin= 0.017  PASS
  minDensityMin      value= 0.999437856696 baseline= 0.999437856696 tol= 9.994e-10 margin= 0.000  PASS
  maxDensityMax      value= 1.00630752139 baseline= 1.00630752117 tol= 1.006e-09 margin= 0.218  PASS
  steps              value= 6683 baseline= 6683 tol= 6.683e-06 margin= 0.000  PASS
OVERALL: PASS   EXIT=0
```

With no exact-wall switch on (the default), every line is PASS and the exit code is based on OVERALL (not the physics gate) -> **`OVERALL: PASS`**. This is the Definition-of-done check that the last code edit (T4.3) did not move the default path: the default check passes at bit level, in contrast to gates 1+2 where the opt-in `tensileExact` changes the dam-break trajectory (bit-level FAIL, physics PASS).

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| full edge suite (final re-run) | 740 passed | expect 740 (728 base + 13 new − 1 deleted) | exact | `pytest tests/edge -q` → `740 passed … 380.89s` (`w004_final_full.log`) |
| solver test set (5 files) | 79 passed | expect 79 | exact | `pytest … -q` → `79 passed … 114.57s` (`w004_t43_solver.log`) |
| w4p5 gradient channel (stable, generic) | 9.43e-16 | <= 1e-13 (vs mpmath) | 8.5 orders inside | `test_warpbc_stable.py (a)` (`w004_t41_test.log`) |
| w4p5 gradient channel (stable, at a vertex) | 2.69e-12 | <= 5e-11 | 19x inside | `test_warpbc_stable.py (a)` |
| w4p5 gradient channel (monomial) | 1.66e-03 | must be > 1e-4 (negative control) | 16x over | `test_warpbc_stable.py (b)` |
| stable Warp vs independent np2d (b7) | 5.55e-15 | <= 1e-13 | 1.8 orders inside | `test_warpbc_stable.py (d)` |
| w4 tensile T vs np2d stable | 3.52e-15 | <= 1e-10 | 4.5 orders inside | `test_tensile_scene_c4.py (a)` (`w004_t42_test.log`) |
| C4 T_y(H=1,(0,0.3)) grid error | 6.13e-06 | <= 5e-5 | 8x inside | `test_tensile_scene_c4.py (b)` |
| C4 |T_quad−T_exact|/scale (polar error) | 1.749e-02 | <= 3e-2, 1.7x inside | `test_deltasph_tensile_c4.py (a)` (`w004_t43_test.log`) |
| C4 shift effect | 5.9079e-02 | in [5e-3, 0.15] | inside | `test_deltasph_tensile_c4.py (b)` |
| dam break KE vs B (tensileExact) | 2.0904e-05 | <= 0.05 (physics) | 2400x inside | gate 1 (`w004_t44_gate1.log`) |
| dam break KE vs B (all switches) | 2.4098e-05 | <= 0.05 (physics) | 2075x inside | gate 2 (`w004_t44_gate2.log`) |
| dam break P1 arrival |diff| | 0.0003 | <= 0.05, 167x inside | gates 1+2 |
| tank rmseBulk (tensileExact) | 0.00149879553508 | <= 1.25 x baseline | margin 0.007 | gate 1 |

## Deviations

1. **DONE-line convention.** KICKOFF step 4 says to append `T-n.m DONE <hash>` after the commit; rule 4 forbids rewriting history, so a task's DONE line cannot live in its own commit. I append each DONE line **after** the commit and it rides into the *next* commit (T4.1's into T4.2's, T4.2's into T4.3's, T4.3's into this report). This is a deliberate, logged deviation from WORK-003's in-commit `DONE (commit below)` placeholder.
2. **Sloshing `verbose=True`.** The spec says `run_sloshing(..., verbose=False, **cfg)`. I used `verbose=True` (in the subprocess) so the log carries a progress line every 2000 steps -> a meaningful "last log line" if a run is killed at the 90-min timeout. The simulation is identical either way; `verbose` only controls console progress.
3. **Starting-state unblock (pre-existing, not a WORK-004 deviation).** The session began blocked (the reviewer's `REVIEW-003` and 7 other files were uncommitted). On explicit operator instruction ("commit them and then keep going") I committed those 8 files as `115c16d` -- the normal pre-launch step the process requires (option (a) of the BLOCKED report). That commit was made only under that instruction, which supersedes the README's "the model never commits them" for that one commit. The documented starting state (clean tree, HEAD = `REVIEW-003`) is thereby restored; `115c16d` is the base for the `git diff --stat` below. The pre-existing `00e0e8a` (the BLOCKED LOG/REPORT-004) sits below it and is superseded by this report (history not rewritten).

## Failures and open questions

No blocking failures. Items of note (all expected, none a defect):

- **Bit-level regression lines FAIL under `tensileExact` (gates 1+2).** With the switch on, the dam break's `ke_tstar1/2/25`, `maxVelocityMax`, `minDensityMin`, `maxDensityMax` exceed their 1e-8-ish bit-level tolerances (e.g. `maxVelocityMax` margin 198801 in gate 1). This is the intended effect of changing the wall tensile integral: the trajectory moves by ~1e-5, far more than the bit-level tolerance but far inside the 5 % physics band. The PHYSICS GATE passes in both `--physics` runs, which is what `--physics` is designed to check. Not a finding -- the switch is opt-in and off by default, so the default `check` (gate 4) is unaffected.
- **Sloshing: no finding.** Both runs completed (no timeout); every comparison quantity is the same order of magnitude as the reviewer's (KE-vs-B 7.32e-08 / 5.78e-05, between-runs 5.78e-05, maxVelocity 0.5921/0.5919, densities in [0.97, 1.03]), the two runs differ by only 5.78e-05 (0.006 %), and no comparison quantity is NaN (the sensor probe's 4/5 leading NaN samples are the expected "<3 particles near the sensor" undefined region, reduced with `nanmax`). Nothing off by an order of magnitude.

## Files changed

`git diff --stat` against the starting commit `115c16d` (REVIEW-003):

```
docs/deltasph-porting-notes.md         |   1 +
docs/q2-conditioning.md                |  13 +
docs/work/logs/LOG-004.md              | 233 +
docs/work/logs/REPORT-004.md           | 227 +-   (replaces the BLOCKED report)
python/edgebound/deltasph2d.py         |   7 +-
python/edgebound/tensile.py            |  71 +-
python/edgebound/warpbc.py             | 237 +-
tests/edge/test_deltasph_tensile.py    |  10 -
tests/edge/test_deltasph_tensile_c4.py |  98 +
tests/edge/test_tensile_scene.py       |   4 +-
tests/edge/test_tensile_scene_c4.py    | 168 +
tests/edge/test_warpbc_stable.py       | 277 +
12 files changed, 1234 insertions(+), 112 deletions(-)
```

Code/test files: `warpbc.py` (T4.1, additive Chebyshev plan), `tensile.py` (T4.2, families w2+w4), `deltasph2d.py` (T4.3, two-line guard + comment), `deltasph-porting-notes.md` (T4.3 change-log row), `q2-conditioning.md` (T4.2 resolution), `test_warpbc_stable.py` (T4.1, new), `test_tensile_scene_c4.py` (T4.2, new), `test_deltasph_tensile_c4.py` (T4.3, new), `test_deltasph_tensile.py` (T4.3, one test deleted), `test_tensile_scene.py` (T4.2, one assertion). `LOG-004.md` and `REPORT-004.md` are the log and this report, committed together as the final commit (the `+ -` on `REPORT-004.md` is because it replaces the earlier BLOCKED report).

## What I did NOT verify

- **The 3D face->edge chain** and any 3D use of the stable plan: the work is 2D only (as the scene layer).
- **The stable plan made a default**: explicitly out of scope -- the monomial plan and the polar quadrature remain the defaults; `tensileExact` / `coverExact` / `coneExact` stay off by default. I did not (and did not need to) verify any default-path change, because there is none.
- **The sloshing run beyond T=1.5 s** and beyond the recorded series: the gate runs T=1.5 s (the reviewer's length); I did not extend the run or re-score it against a longer reference.
- **The C4 tensile term against an independent (non-mpmath, non-np2d) reference** beyond what the tests do: T4.2/T4.3 verify `tensile_vector_scene(w4)` against `factor * np2d.gradient(w4p5, stable)` and against the 2000^2 midpoint grid; there is no third, independent oracle for the C4 W^5 integral (the mpmath `block_grads` and the np2d stable route share the same closed-form Chebyshev machinery at the leaf).
- **Performance beyond the 200k-pair timing** (T4.1): the 0.32x monomial-vs-chebyshev ratio is one measurement on one kernel/geometry; I did not benchmark other kernels, sizes, or the full-solver impact of the stable route.
- **The reviewer's `refs/*_probe.py`** were read as construct references only, not run as acceptance.
