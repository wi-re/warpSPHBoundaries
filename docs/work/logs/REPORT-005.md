# REPORT-005
Status: **DONE**          Branch: `local-model`   HEAD: `6247401` (T5.4, last task commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-04

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T5.1 `viscosity.py: lap_lambda_scene` + `tests/edge/test_viscosity_scene.py` | DONE | `e865fff` | 7/7 new tests pass (worst (a) 3.784e-14, (b) 2.24e-4, (c) ≤ 1e-9, (f) controls above their floors, (g) −2.000000000000 / 0.250000000000); full regression 740 + 7 = 747 passed |
| T5.2 `cfg.viscosityExact` switch (default off) + `tests/edge/test_deltasph_viscosity.py` + porting-notes row | DONE | `a67566d` | 2/2 new tests pass ((b) 1.013e-4 / 1.271e-4 of max term vs tol 5e-4; (c) operator ratios 0.899 / 0.886); 79 solver tests pass; full regression 747 + 2 = 749 passed; tank ms/step 18.75 → 32.91 (w2), 21.36 → 40.24 (w4) |
| T5.3 gates (no code) | DONE (no commit — the log rides with this report) | — | `--cfg viscosityExact=true`: PHYSICS GATE: PASS (tank 1.4993e-3 / 1.8134e-3 / 7.284e-7; dam break KE 3.9892e-4, arrival 2.4740 t\*, \|diff\| 0.0160); all four switches: PHYSICS GATE: PASS (tank at the single-switch values, dam break KE 3.5344e-4); sloshing T = 1.5 s: KE 2.8231e-4 (gate 5 %), maxVel 0.5919, ρ [0.99928, 1.00489], 1388 s; default config: OVERALL: PASS (T5.3(3) below) |
| T5.4 documentation | DONE | `6247401` | `docs/derivations/laplacian-wall.md` (status [V]), one section in `docs/deltasph-validation.md`, one index line in `docs/README.md` |

## Acceptance evidence

T5.1 (full captures `.tmp/w005_t51_test.log`, `.tmp/w005_t51_regress.log`):
```
$ python -m pytest tests/edge/test_viscosity_scene.py -q -s
(a) flat wall: worst |scene - quad| / max B = 3.784e-14 over both families and H in {1, 0.7} (tol 1e-10)
(b) worst |scene - brute| / max|brute| (tol 1e-3): L w2 2.24e-04  L w4 2.12e-04  cavity w2 2.15e-04  cavity w4 2.12e-04
(c) w2: zeros <= 1e-9 (5 cases); outside (1.2, 0.5) H = 0.3: scene 7.754195  brute 7.754179 (|diff|/value 2.1e-06)
(c) w4: zeros <= 1e-9 (5 cases); outside (1.2, 0.5) H = 0.3: scene 3.688068  brute 3.688240 (|diff|/value 4.7e-05)
(d) w2: scaling |dl07 - dl1/0.49| = 0.0e+00; moved body 0.0e+00; two-body rows/sum 3.6e-15 / 8.9e-16 / 3.6e-15 (scale 7.696e+00, tol 1e-11)
(e) guards: family='w9', ImplicitRep (DiskBody) and VolumeRep bodies all raise NotImplementedError
(f) w2: 2f*lambda only 5.757 (tol 1e-1)   -Delta-lambda 2.000 (tol 1.0)   A(0.1) = -3.0525 vs B(0.1) = 1.9599, |A-B|/B = 2.557 (tol 0.3)
(g) w2: int r W' dA = -2.000000000000 (expect -2, atol 1e-10)   pairwise bulk term for v = (y^2, 0) = 0.250000000000 fac = fac/4 (rtol 1e-10)
7 passed, 15 warnings in 13.63s
$ python -m pytest tests/edge -q
747 passed, 15 warnings in 463.29s (0:07:43)
```

T5.2 (full capture `.tmp/w005_t52_test.log`):
```
$ python -m pytest tests/edge/test_deltasph_viscosity.py -q -s
(a) DeltaSPHConfig().viscosityExact is False
(b) w2: near=693  max|pred| = 0.2893  max|d_ex - pred|/max|pred| = 1.013e-04 (tol 5e-4)
(b) w4: near=690  max|pred| = 0.2925  max|d_ex - pred|/max|pred| = 1.271e-04 (tol 5e-4)
(c) w2: max|d_ex - d_pair|/max|d_pair| = 0.899 (> 0.5)   sign flip 2.000 (> 1.0)   nu_eff -> fac/12: 0.333 (> 0.2)
(c) w4: max|d_ex - d_pair|/max|d_pair| = 0.886 (> 0.5)   sign flip 2.000 (> 1.0)   nu_eff -> fac/12: 0.333 (> 0.2)
(d) w2: five steps with viscosityExact=True finite
(d) w4: five steps with viscosityExact=True finite
2 passed, 14 warnings in 15.29s
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 129.39s (0:02:09)
$ python -m pytest tests/edge -q
749 passed, 15 warnings in 396.31s (0:06:36)
```
`git diff` of `deltasph2d.py` for T5.2: exactly one import (`from .viscosity import lap_lambda_scene`), one config field (`viscosityExact: bool = False` after `tensileExact`), and the rhs wall-viscosity block (the `dl = lap_lambda_scene(...)` computation before the body loop and the `if cfg.viscosityExact: acc = acc.index_add(0, near, (-2.0 * (fac / 8.0) * cfg.wallMass * un[near] / rho[near] * dl[bi])[:, None] * nb_[near]); continue` inside it).  Nothing else.

T5.3 (full captures `.tmp/w005_t53_gate1.log`, `.tmp/w005_t53_gate2.log`, `.tmp/w005_t53_slosh.log`, `.tmp/w005_t53_default.log`; decisive lines in `LOG-005.md` T5.3):
```
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg viscosityExact=true
  [gate] rmseBulk  value= 0.00149929753837  limit: <= 0.001873494419 (1.25 x baseline)  PASS
  [gate] rmseNear  value= 0.00181340794898  limit: <= 0.002267181356 (1.25 x baseline)  PASS
  [gate] keLast    value= 7.28437955168e-07  limit: <= 3.503023691e-06 (5 x baseline)  PASS
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 3.9892e-04 (668 samples)
  [physics] dam break P1 arrival = 2.4740 t*  (reference 2.49, |diff| = 0.0160)
  [gate] KE rel max vs B      value= 0.000398917601958  limit: <= 0.05  PASS
  [gate] P1 arrival |diff|    value= 0.0160295711131  limit: <= 0.05  PASS
  [gate] minDensityMin        value= 0.999429377347  limit: >= 0.97  PASS
  [gate] maxDensityMax        value= 1.00603521646  limit: <= 1.03  PASS
  [gate] steps drift          value= 0  limit: <= 0.05 (baseline 6683)  PASS
PHYSICS GATE: PASS

$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg coverExact=true,coneExact=true,tensileExact=true,viscosityExact=true
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 3.5344e-04 (668 samples)
  [physics] dam break P1 arrival = 2.4740 t*  (reference 2.49, |diff| = 0.0160)
  ke_tstar1/2/25 = 0.357363651811 / 0.754228393305 / 0.922766940735; maxVelocityMax 6.59792039874; minDensityMin 0.999428350557; maxDensityMax 1.00603117793
  [gate] KE rel max vs B      value= 0.000353436713337  limit: <= 0.05  PASS
PHYSICS GATE: PASS

$ python .tmp/slosh_visc.py        # run_sloshing(nx=200, T=1.5, shifting=True, noPen="impulse", verbose=True, viscosityExact=True)
viscosityExact sloshing T=1.5: steps 15001 wall 1388 s  KE rel max vs slosh_B = 2.8231e-04 (1500 samples, gate 5%)  maxVel 0.5919  rho [0.99928, 1.00489]

$ cd python && python -m edgebound.deltasph_regress check --cases tank,dambreak
cfg: {} (defaults)
tank  (281.2 s wall): all six lines PASS, margin <= 0.002 (rmseBulk 0.00149879553509, rmseNear 0.0018137450846, keLast 7.00604738261e-07, rhoMin 1.00008277812, rhoMax 1.00246999588, steps 3502)
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 2.5345e-07 (668 samples)
  [physics] dam break P1 arrival = 2.4897 t*  (reference 2.49, |diff| = 0.0003)
dambreak  (436.6 s wall): all eight lines PASS, margin <= 0.220 (ke_tstar1 0.357285986624, ke_tstar2 0.754074336958, ke_tstar25 0.922537627023, p0_arrival_tstar 2.48970315022, maxVelocityMax 6.68104559168, minDensityMin 0.999437856696, maxDensityMax 1.00630752139, steps 6683)

OVERALL: PASS
```
In all `--physics` runs the bit-level lines FAIL as expected (the term changes the trajectory); the gates PASS.  One GPU job at a time (checked `nvidia-smi` before each; the GPU is shared with a resident local LLM, wall times vary).

Final full regression (after the last code edit, T5.2 — the T5.4 changes are docs only):
```
$ python -m pytest tests/edge -q
749 passed, 15 warnings in 397.85s (0:06:37)
```

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| T5.1 (a) flat wall \|scene − quad\| / max B | 3.784e-14 | ≤ 1e-10 | 2.6e3 | `.tmp/w005_t51_test.log` |
| T5.1 (b) worst \|scene − brute\| / max\|brute\| | 2.24e-4 | ≤ 1e-3 | 4.5 | `.tmp/w005_t51_test.log` |
| T5.1 (c) degenerate zeros | ≤ 1e-9 (5 cases, both families) | ≤ 1e-9 abs | — | `.tmp/w005_t51_test.log` |
| T5.1 (c) outside point | 7.754195 (w2) / 3.688068 (w4) | ≤ 1e-3 · value vs brute | 2.1e-6 / 4.7e-5 | `.tmp/w005_t51_test.log` |
| T5.1 (d) scaling / moved body / two-body | 0.0e+00 / 0.0e+00 / 3.6e-15 | rtol 1e-9 / ≤ 1e-11 · max | — | `.tmp/w005_t51_test.log` |
| T5.1 (f) 2fλ alone / −Δλ / \|A−B\|/B (w2) | 5.757 / 2.000 / 2.557 | > 1e-1 / > 1.0 / > 0.3 | above floors | `.tmp/w005_t51_test.log` |
| T5.1 (g) ∫ r W′ dA / pairwise bulk v = (y²,0) | −2.000000000000 / 0.250000000000 fac | atol / rtol 1e-10 | — | `.tmp/w005_t51_test.log` |
| T5.2 (b) max\|d_ex − pred\| / max\|pred\| | 1.013e-4 (w2) / 1.271e-4 (w4) | ≤ 5e-4 | ~4–5 (the brute grid's own error) | `.tmp/w005_t52_test.log` |
| T5.2 (b) near / max\|pred\| | 693 / 0.2893 (w2); 690 / 0.2925 (w4) | ≥ 500 / > 0.1 | — | `.tmp/w005_t52_test.log` |
| T5.2 (c) max\|d_ex − d_pair\|/max\|d_pair\| / sign flip / ν→fac/12 | 0.899 / 2.000 / 0.333 (w2); 0.886 / 2.000 / 0.333 (w4) | > 0.5 / > 1.0 / > 0.2 | above floors | `.tmp/w005_t52_test.log` |
| T5.3(1a) tank rmseBulk / rmseNear / keLast | 1.49929753837e-3 / 1.81340794898e-3 / 7.28437955168e-7 | ≤ 1.25× / 1.25× / 5× baseline | gate PASS | `.tmp/w005_t53_gate1.log` |
| T5.3(1a) dam break KE vs B / arrival / maxVel / densities | 3.9892e-4 / 2.4740 t\* (\|diff\| 0.0160) / 6.59592351289 / [0.999429377347, 1.00603521646] | ≤ 0.05 / ≤ 0.05 / ≥ 0.97 / ≤ 1.03 | gate PASS | `.tmp/w005_t53_gate1.log` |
| T5.3(1a) dam break ke_tstar 1/2/2.5 | 0.35736468689 / 0.754254694988 / 0.92280834676 | (bit-level FAIL expected) | — | `.tmp/w005_t53_gate1.log` |
| T5.3(1b) dam break KE / arrival / maxVel / densities / ke_tstar | 3.5344e-4 / 2.4740 t\* / 6.59792039874 / [0.999428350557, 1.00603117793] / 0.357363651811 / 0.754228393305 / 0.922766940735 | as (1a) | gate PASS | `.tmp/w005_t53_gate2.log` |
| T5.3(2) sloshing KE rel max / maxVel / densities / steps | 2.8231e-4 / 0.5919 / [0.99928, 1.00489] / 15001 | ≤ 5 % | factor 177 | `.tmp/w005_t53_slosh.log` |
| T5.3(3) default config | OVERALL: PASS (bit-level) | — | — | `.tmp/w005_t53_default.log` |
| tank ms/step (informational) | 18.75 → 32.91 (w2), 21.36 → 40.24 (w4) | not a gate | +75 % / +88 % | `.tmp/w005_t52_test.log` |

## Deviations

* The session started BLOCKED at the launch-time starting-state check (the reviewer's `REVIEW-004` files were uncommitted; the documented base commit did not exist).  The old BLOCKED report was committed as `fb0ba9a`; the operator then committed the reviewer files as `4b90fb0` and relaunched ("it's unblocked, go for it").  **Deviation from the documented starting state:** `4b90fb0`'s subject is `"commit review 004 work"`, not `REVIEW-004: …`; all `git diff --stat` in this report are against `4b90fb0`, the actual starting base.  The substantive precondition (reviewer files committed, clean tree, 79/740 green) was verified after the operator's commit.
* `REPORT-005.md` was rewritten from the BLOCKED version (`fb0ba9a`) into this final report.
* The reviewer's commit `4b90fb0` also contains `docs/work/refs/lap_controls_probe.py`, which `REVIEW-004.md` §4 does not list (it is one of the `lap_*` probes; no action needed).
* The scipy `quad` reference of T5.1 (a) emits an `IntegrationWarning` at `epsrel = 1e-13` (possible round-off underestimation); the reviewer's probe used the same tolerances and the same agreement level (≤ 4e-14), so the reference is at its float64 floor, not wrong.  Noted, not acted on.

## Failures and open questions

* None — no task failed, no tolerance was missed, no code outside the allowed files was touched.  Open questions (for the reviewer / user, listed, not "fixed"):
  1. The pairwise and the Laplacian wall terms are **different operators** near the wall (the Laplacian form damps the wall-normal velocity 3–12× less in the first particle rows; flat-wall ratio 8|A|/B, `derivations/laplacian-wall.md` §6).  The gates pass with the switch on (arrival 2.4740 t\* vs reference 2.49, sloshing KE moved 0.03 % over 1.5 s), but the default stays the pairwise form (`viscosityExact` off).  Whether to switch the default is a user decision after these numbers (HANDOFF Part A Q1).
  2. The first-moment variant of the mirror field (position-dependent `∫ (v_ghost − v) ∇²W dA′`) is not implemented — it would need a scene operation for moments of `∇²W` (or of L with the mirror field as payload).  Out of scope for WORK-005.
  3. The `∇²W`-direct registration trap (`kernels._finish` divides by `∫ r·shape = 0`; the body indicator gives a spurious c/H² inside a body): caught by the tests (T5.1 (c) inside / tangent cases); the L route is the fix.  If other non-normalised kernels are ever registered, `kernels._finish` needs a guard.

## Files changed

Against the starting base `4b90fb0` (the operator's review commit; `git diff --stat 4b90fb0`, taken before the report commit):

```
 docs/README.md                        |   1 +
 docs/deltasph-porting-notes.md        |   1 +
 docs/deltasph-validation.md           |  26 +++++++++++++
 docs/derivations/laplacian-wall.md    | 106 +++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/work/logs/LOG-005.md             | 238 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 python/edgebound/deltasph2d.py        |   9 +++++
 python/edgebound/viscosity.py         | 104 ++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/edge/test_deltasph_viscosity.py | 121 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/edge/test_viscosity_scene.py    | 339 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 9 files changed, 945 insertions(+)
```
The `LOG-005.md` count is the T5.1/T5.2 portion (committed with those tasks); the T5.3 plan/results entries plus this report ride with the report commit.  Scratch files (`.tmp/`) are git-ignored and not committed.  No file outside the WORK-005 allowed list was touched; `regress_baseline.json` is untouched.

## What I did NOT verify

* No independent re-derivation of the reviewer's context numbers beyond the stated tolerances: the flat-wall `B(z)` smoke table and the `A(z)`/`8|A|/B` rows of the context table are cited as context; I measured `B(z)` at rtol 1e-6 (T5.1 (a) smoke) and `A(0.1)` at z/H = 0.1 only (T5.1 (f)).  The other `8|A|/B` rows are the reviewer's.
* The physics gates are do-no-harm bands, not an exactness proof (as in WORK-001–004): a gate PASS does not prove the wall term is exact; the exactness evidence is T5.1 (a)–(g) and T5.2 (b) against independent brute forces.
* Sloshing was truncated at T = 1.5 s (per the spec); the full 7 s record was not run with the switch on.
* Moving walls: the free-slip mirror uses a static `velocityAt`; the rolling-tank (moving-body) variant is out of scope and was not tested with the switch on.
* The `ms/step` numbers are on a GPU shared with a resident local LLM (busy ~45–90 %); absolute wall times vary 2–3× run to run, so they are information only.
* `ImplicitRep` / `VolumeRep` bodies with `viscosityExact` raise `NotImplementedError` (guard tested in T5.1 (e)); the behaviour of such a body in a *default*-config run is unchanged (the guard is dead when the flag is off).
