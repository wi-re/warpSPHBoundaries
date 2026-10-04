# REPORT-007
Status: **DONE**          Branch: `local-model`   HEAD: `b19faf7` (T7.3, last task commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-04

(Supersedes the BLOCKED version of 2026-10-04 (commit `e4a1df1`): the blocker — the starting-state check required the top commit to start with `REVIEW-006` and it was then `WORK-006: REPORT-006` — was resolved by the reviewer's commit `64e74e1 REVIEW-006: accept WORK-006; … WORK-007 starting state, lessons`, which pinned the starting state to 82 / 756 and unblocked WORK-007. The work then continued to the Definition of done on the branch.)

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T7.1 `noslip` form + tests | DONE | `b3c6621` | `wallViscosityForm = "noslip"` added (Chiron-style flux term `-2 ν_eff (v − v_w)|G|/(ρ d)`, `ν_eff = fac/8`, `d = max(d, 0.25 dx)`, all-components relative velocity; the `laplacian`/`pairwise` branches untouched, default stays `"laplacian"`) + 4 tests in `tests/edge/test_deltasph_noslip.py`: (a) flat-wall absolute (own scipy quad), (b) distance floor + Galilean, (c) Couette/Poiseuille through the solver with the free-slip forms as negative controls, (d) off-switch + five steps finite on C2/C4; the numbers match the reviewer's probes to the printed digit (Couette a_flux 0.00815/−0.00855/−0.00280/0.00001, Poiseuille −0.00143/−0.00825/−0.00672/−0.00601, ν_eff 0.00314); new tests 4 passed, 5-file set 82 passed (unchanged), full suite 760 passed |
| T7.2 gates and stability | DONE (information; no commit) | — | (1) `check --physics --cases tank,dambreak --cfg wallViscosityForm=noslip`: tank all gate lines PASS (≈ the default's), dam break KE vs B 0.3165191 (FINDING > 5 %), P1 arrival nan (P* never > 0.05 in T = 0.65 s), maxVelocityMax 3.7982989 (~43 % below the default's 6.598), densities and steps drift PASS → **PHYSICS GATE: FAIL** (the expected finding — the reference is free-slip, the no-slip wall dissipates; reported with the numbers, nothing adjusted); (2) sloshing T = 1.5 s noslip: KE rel max vs `slosh_B_nx200` 1.4646e-01 (14.6 %, 1500 samples), maxVel 0.5683, rho [0.99931, 1.00483], 963 s wall (15001 steps); (3) stability: dam break max k = 16.5835 s⁻¹, k·dt = 0.0016; sloshing max k = 20.4154 s⁻¹, k·dt = 0.0020 (both ≪ 1 — stable) |
| T7.3 documentation | DONE | `b19faf7` | `docs/derivations/noslip-wall.md` (new, [V]); `docs/deltasph-validation.md` § `## No-slip wall viscosity (WORK-007)`; one `docs/deltasph-porting-notes.md` change-log row; one `docs/README.md` index line; every number is the one in the log |

## Acceptance evidence

All commands run in the conda env `warp` (`/home/lu26029/miniconda3/envs/warp/bin`), branch `local-model`, 2026-10-04, RTX PRO 6000 shared with a resident local LLM (~69 GiB of 97887 MiB; one GPU job at a time). Full captures in `LOG-007.md` and the named `.tmp/` logs. The starting state was re-checked after `REVIEW-006` (5-file set 82 passed, full suite 756 passed — matching the pinned 82 / 756) before T7.1 started.

### T7.1

1. New tests: `python -m pytest tests/edge/test_deltasph_noslip.py -q` → **`4 passed, 14 warnings in 2.97s`** (`.tmp/w007_t71_tests.log`). Decisive printed lines (tolerances stated in the test docstring before results):
   - (a) flat wall absolute, own `scipy quad` of `F = ∫_{−L}^{L} W(√(s²+z²)) ds` (Wendland C2 written out, epsabs = epsrel = 1e-13), `z = dp/2`: v=(1,0) got (−2.018640, 0.0), max|diff|/max|pred| 2.20e-16; v=(0,−1) got (0.0, 2.018640), 4.40e-16; v=(0.6,−0.8) got (−1.211184, 1.614912), 4.12e-16 (tol 1e-9). (a-smoke, the OWN prediction, rtol 1e-4): ν_eff = 2.432164e-03, F = 8.299771, pred_x = −2.018640, pred_y = +2.018640.
   - (b) distance floor, a particle at `z = 0.1 dx` uses `d_b = 0.25 dx` with the TRUE `F(z)`: got (−4.495036, 0.0), 1.19e-15 (tol 1e-9). Galilean: co-moving wall (v = v_w) |term| 0.00e+00 (< 1e-10), resting wall |term| 3.423317 (> 0.1).
   - (c) Couette/Poiseuille through the solver (`hydrostatic_tank(dp=0.04)`, `a_flux = rhs("noslip") − rhs(viscosity=False)`, first component, mean over the 15 columns `|x| < 0.3` in rows `s = (k+0.5) dp`, k = 0..3, ν_eff = 0.00314, a_bulk(row 0) = 0.06014): Couette a_flux by row +0.00815 / −0.00855 / −0.00280 / +0.00001 (|a_flux|/|a_bulk(row 0)| = 0.136 / 0.142 / 0.047 / 0.000, all ≤ 0.2); negative controls — `laplacian` |a(row 0)| = 0.06014 and `pairwise` 0.06014, both > 0.5 · 0.06014 = 0.03007 (they fail the Couette tolerance, as a free-slip form must); Poiseuille target −2 ν_eff = −0.00628: rows 2.5 / 3.5 a_flux −0.00672 / −0.00601 (|a_flux−target|/|target| = 0.070 / 0.044 ≤ 0.15), rows 0.5 / 1.5 a_flux −0.00143 / −0.00825 (|a_flux−target|/|a_bulk−target| = 0.215 / 0.411 ≤ 0.6). These match the reviewer's solver probe (`docs/work/refs/review5_noslip_solver_probe.py`) to the printed digit.
   - (d) `wallViscosity = False`: |a(noslip, off) − a(laplacian, off)| = 2.09e-13 (< 1e-12, the form is irrelevant with no wall term); five `sim.step()` finite (x, v, ρ) on the C2 and the C4 tank (kernel-agnostic: |G_b| comes from the scene).
2. 5-file solver set: `python -m pytest tests/edge/test_deltasph.py test_scene.py test_dfsph.py test_cover_scene.py test_deltasph_cover.py -q` → **`82 passed, 14 warnings in 109.27s`** (unchanged — the noslip branch is inactive for the default `laplacian` and the `pairwise` tests).
3. Full suite (post-T7.1 tree): `python -m pytest tests/edge -q` → **`760 passed, 15 warnings in 402.36s`** (`.tmp/w007_t71_full.log`). 756 (post-REVIEW-006) + 4 new noslip cases = 760; no test added beyond the 4, no regression.

### T7.2 (information; no commit)

1. `cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg wallViscosityForm=noslip` → `.tmp/w007_t72_regress_noslip.log`. Tank (200.5 s): rmseBulk 0.00150522110766 / rmseNear 0.00186050042805 / keLast 5.56674781145e-07 / rhoMin 1.00008176803 / rhoMax 1.00247035463 / steps 3502 — all five gate lines PASS (the near-rest tank is ≈ the default's; keLast is even lower than the default's 7.28e-7 because the noslip term damps the tiny spurious wall velocities). Dam break (357.7 s): `[physics]` KE vs `dambreak_B_nx67` max rel = **0.3165191** (668 samples, **FINDING > 5 %**); P1 arrival = **nan** (P* never > 0.05 in T = 0.65 s; reference 2.49); ke_tstar 1 / 2 / 2.5 = 0.3206926 / 0.5688047 / 0.6396314 (default 0.3573637 / 0.7542284 / 0.9227669); maxVelocityMax 3.7982989 (default 6.5979204, ~43 % lower); minDensityMin 0.9990133 (gate ≥ 0.97 PASS), maxDensityMax 1.0014192 (gate ≤ 1.03 PASS), steps drift 0 (PASS). **PHYSICS GATE: FAIL** (the two dambreak KE / arrival lines) — the expected finding: the reference is free-slip, the no-slip wall removes tangential momentum, so the dam break is more damped; reported with the numbers, not a blocker, nothing adjusted.
2. `python .tmp/slosh_noslip.py` (`run_sloshing(nx=200, T=1.5, shifting=True, noPen="impulse", verbose=True, wallViscosityForm="noslip")`) → `.tmp/w007_t72_slosh.log`. Completed in 963 s wall (15001 steps): **`steps 15001 wall 963 s  KE rel max vs slosh_B = 1.4646e-01 (1500 samples, gate 5 %)  maxVel 0.5683  rho [0.99931, 1.00483]`**. 14.6 % vs the free-slip `slosh_B_nx200` reference (a FINDING, expected larger than the default's because the wall now dissipates tangentially); the wall rolls (rotating gravity), so it exercises `b.velocityAt`. (The first launch failed with `ModuleNotFoundError: No module named 'edgebound'` and a bad `RES` path — three `dirname` calls for a script in `repo/.tmp/` — both fixed before this run; see LOG-007.)
3. `python .tmp/stability_noslip.py` (the noslip explicit damping rate `k = 2 ν_eff |G_b|/(ρ d_eff)`, `d_eff = max(d, 0.25 dx)`, for the near particles after 300 steps, default config otherwise) → `.tmp/w007_t72_stability.log`: **dam break max k = 16.5835 s⁻¹ (body 0), dt = 9.727e-05 s, max k·dt = 0.0016; sloshing max k = 20.4154 s⁻¹ (body 0), dt = 1.000e-04 s, max k·dt = 0.0020**. Both `k·dt ≪ 1`, far below the 0.5 finding threshold — stable. (The max k is above the reviewer's "k ~ 5 s⁻¹ at d = dx/2" only because the nearest particles sit at d ~ 0.25 dx, where 1/d is larger; the criterion is k·dt.)

### T7.3

Files exist (committed `b19faf7`): `docs/derivations/noslip-wall.md` (new, status [V], from `TEMPLATE.md`); `docs/deltasph-validation.md` § `## No-slip wall viscosity (WORK-007)` (inserted after the WORK-006 section, before `## 7. Next`); `docs/deltasph-porting-notes.md` (one change-log row directly under the WORK-006 `wallViscosityForm` row, no blank line); `docs/README.md` (one index line under `laplacian-wall.md`). Every number in them is the one in the log (verified by construction). The default in `deltasph2d.py` is unchanged — the value stays `wallViscosityForm: str = "laplacian"` (line 83; only the comment gains the `| "noslip": …` clause), and `git diff` of `deltasph2d.py` between the T7.3 commit and its parent is empty (T7.3 is docs-only).

Final full suite (Definition of done, run once more at the end, after the T7.3 commit): `python -m pytest tests/edge -q` → **`760 passed, 15 warnings in 377.27s (0:06:17)`** (`.tmp/w007_final_full.log`). 756 + 4 new noslip cases = 760.

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| T7.1 new noslip tests | 4 passed | pass (4) | 0 | `.tmp/w007_t71_tests.log` |
| T7.1 (a) flat wall, v=(1,0) | got (−2.018640, 0.0) | ≤ 1e-9 · \|pred\| | 2.20e-16 | `.tmp/w007_t71_tests.log` |
| T7.1 (a) flat wall, v=(0,−1) | got (0.0, 2.018640) | ≤ 1e-9 · \|pred\| | 4.40e-16 | `.tmp/w007_t71_tests.log` |
| T7.1 (a-smoke) ν_eff / F / pred_x / pred_y | 2.432164e-3 / 8.299771 / −2.018640 / +2.018640 | rtol 1e-4 | pass | `.tmp/w007_t71_tests.log` |
| T7.1 (b) distance floor (z = 0.1 dx) | got (−4.495036, 0.0) | ≤ 1e-9 · \|pred\| | 1.19e-15 | `.tmp/w007_t71_tests.log` |
| T7.1 (b) Galilean co-moving / resting | 0.00e+00 / 3.423317 | < 1e-10 / > 0.1 | pass | `.tmp/w007_t71_tests.log` |
| T7.1 (c) Couette a_flux (rows 0.5–3.5) | +0.00815 / −0.00855 / −0.00280 / +0.00001 | ≤ 0.2 · \|a_bulk(row 0)\| = 0.06014 | 0.136 / 0.142 / 0.047 / 0.000 | `.tmp/w007_t71_tests.log` |
| T7.1 (c) negative controls (row 0) | laplacian 0.06014, pairwise 0.06014 | > 0.5 · 0.06014 = 0.03007 | both fail (as a free-slip form must) | `.tmp/w007_t71_tests.log` |
| T7.1 (c) Poiseuille a_flux (target −0.00628) | −0.00143 / −0.00825 / −0.00672 / −0.00601 (rows 0.5–3.5) | rows 2,3 ≤ 0.15 \|target\|; rows 0,1 ≤ 0.6 \|a_bulk−target\| | 0.070 / 0.044 / 0.215 / 0.411 | `.tmp/w007_t71_tests.log` |
| T7.1 (d) off-switch | \|a(noslip,off) − a(laplacian,off)\| = 2.09e-13 | < 1e-12 | pass | `.tmp/w007_t71_tests.log` |
| T7.1 5-file solver set | 82 passed | = 82 (unchanged) | 0 | LOG-007 (109.27 s) |
| T7.1 full suite | 760 passed | = 756 + 4 | 0 | `.tmp/w007_t71_full.log` |
| T7.2(1) tank rmseBulk / rmseNear / keLast | 0.0015052 / 0.0018605 / 5.5667e-7 | gate ≤ 1.25× / 1.25× / 5× baseline | PASS | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(1) dam break KE vs B | 0.3165191 (668 samples) | gate ≤ 0.05 | **FAIL (FINDING, > 5 %)** | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(1) dam break P1 arrival | nan t* (P* never > 0.05 in T = 0.65 s) | gate ≤ 0.05 (reference 2.49) | **FAIL (FINDING)** | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(1) dam break ke_tstar 1 / 2 / 2.5 | 0.3206926 / 0.5688047 / 0.6396314 | information (default 0.3573637 / 0.7542284 / 0.9227669) | 10–31 % below the default's | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(1) dam break maxVelocityMax | 3.7982989 | information (default 6.5979204) | ~43 % below | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(1) dam break minDensityMin / maxDensityMax | 0.9990133 / 1.0014192 | gate ≥ 0.97 / ≤ 1.03 | PASS | `.tmp/w007_t72_regress_noslip.log` |
| T7.2(2) sloshing KE vs `slosh_B_nx200` | 1.4646e-01 (1500 samples) | information (gate 5 %) | FINDING, expected larger (free-slip reference) | `.tmp/w007_t72_slosh.log` |
| T7.2(2) sloshing maxVel / rho / steps | 0.5683 / [0.99931, 1.00483] / 15001 | information | healthy | `.tmp/w007_t72_slosh.log` |
| T7.2(3) dam break max k / k·dt | 16.5835 s⁻¹ / 0.0016 | k·dt < 0.5 | 0.0016 ≪ 0.5 (stable) | `.tmp/w007_t72_stability.log` |
| T7.2(3) sloshing max k / k·dt | 20.4154 s⁻¹ / 0.0020 | k·dt < 0.5 | 0.0020 ≪ 0.5 (stable) | `.tmp/w007_t72_stability.log` |
| final full suite (after last task commit) | 760 passed, 15 warnings in 377.27 s | = 756 + 4 | 0 | `.tmp/w007_final_full.log` |

## Deviations

* **None from the work document.** The three T7.1 spec items (config comment, dispatch + ValueError, body-loop branch), the four T7.1 tests, the T7.2 runs, and the four T7.3 doc files are exactly as specified. The default `wallViscosityForm` stays `"laplacian"`; `d_min = 0.25 dx` and the form were not tuned; no existing test, `scene.py`, `warpbc.py`, `viscosity.py`, `results/`, the harness, or the baseline was touched (only the four T7.3 doc files, `deltasph2d.py` (T7.1), and the new test file).
* **Scratch-script launch fixes (T7.2(2), my own `.tmp/` script only).** The first launch of `.tmp/slosh_noslip.py` failed with `ModuleNotFoundError: No module named 'edgebound'` (a script in `repo/.tmp/` puts `.tmp/`, not the `python/` package dir, on `sys.path`) and a bad `RES` path (three `dirname` calls for a script one level below the repo root, needing two). Both were fixed in the scratch script and re-run; no repo code, test, or baseline is affected.

## Failures and open questions

* **The dam-break / sloshing no-slip runs deviate from the free-slip reference (expected, not a bug).** `PHYSICS GATE: FAIL` on T7.2(1) (dam break KE vs B 0.3165191 > 0.05; P1 arrival nan) and the sloshing KE difference 14.6 % > 5 % (T7.2(2)). Both are the intended physics: the reference series are free-slip warpSPH runs and the no-slip wall removes tangential momentum, so the no-slip flow is more damped (KE 10–31 % lower at t* = 1/2/2.5, peak velocity ~43 % lower, P1 impact too weak to cross 0.05 in the truncated window). The density and step gates stay healthy, and the tank (near-rest) gate is all PASS. Per the work document this is a finding reported with the numbers, not a blocker, and nothing is adjusted.
* **`pytest tests/edge -q` = 760, not 755.** 756 (post-REVIEW-006; the function-vs-case counting difference already pinned by `REVIEW-006`) + 4 new noslip cases = 760. No test was skipped, removed, or device-restricted.
* **GPU wall times vary 2–3×** with the resident local LLM (~69 GiB of 97887 MiB). All gates are physics/bit-level and unaffected; the ms/step and wall-time numbers are information only.
* The bit-level lines of the T7.2(1) noslip gate FAIL against the default baseline **by design** (the noslip term is not exactly zero at the tiny spurious wall velocities); the physics-gate lines are the meaningful verdict and are reported above.

## Files changed

`git diff --stat 64e74e1..HEAD` (`64e74e1` = `REVIEW-006`, the starting commit after the resume; the range is the whole WORK-007 work; this report — replacing the BLOCKED version `e4a1df1` — and the final log lines are added by the last commit):

```
 docs/README.md                     |   1 +
 docs/deltasph-porting-notes.md     |   1 +
 docs/deltasph-validation.md        |  46 ++++++++++++++++++++++++++++++
 docs/derivations/noslip-wall.md    |  88 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/work/logs/LOG-007.md          |  41 +++++++++++++++++++++++++++
 python/edgebound/deltasph2d.py     |  11 ++++++--
 tests/edge/test_deltasph_noslip.py | 253 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 7 files changed, 439 insertions(+), 2 deletions(-)
```

(The BLOCKED-phase commit `e4a1df1` and this report both touch `LOG-007.md` / `REPORT-007.md`; the diff above is against the post-resume starting commit, so it shows only the WORK-007 deliverables. No code, test, doc, or baseline file was modified outside the seven above.)

## What I did NOT verify

* **Making `"noslip"` the default** (out of scope): the default stays `"laplacian"`; only the opt-in `"noslip"` form was added and tested.
* **Partial slip (`κ ∈ [0, 1]`), per-body slip choice, a per-element `∫W ds / d_n` (new `warpbc` channel), the `1/γ` renormalisation, and a tangential-only variant** (all out of scope): not implemented; listed as open questions in `derivations/noslip-wall.md` §7.
* **The full-length (7 s) sloshing run** (out of scope): T = 1.5 s truncation as specified; the no-slip deviation is reported over that window.
* **Corners and curved walls with the no-slip form**: the term is the flat-wall model per particle (one normal, summed |G_b|, nearest-point distance); it is not Chiron's per-element form there (phase 2). The tests cover a flat wall (a, b), a flat-tank flow (c), and five finite steps (d); the dam break (flat walls + a corner) and sloshing (T7.2) run it in a realistic state but the per-element accuracy at the corner is not separately claimed.
* **`ImplicitRep` / `SdfRep` / `VolumeRep` stepping**: unchanged from WORK-006 — `DeltaSPH2D` requires `SurfaceRep` walls (the `hit` mask is a no-op for SurfaceRep, kept as the volume-representation guard); a non-SurfaceRep body raises before the wall term runs.
* **Performance tuning** (out of scope, forbidden by the Definition of done): the T7.2(3) `k·dt` is a stability diagnostic, not a performance claim; nothing was re-tuned.
* **Anything in `~/dev/warpSPH*`** (out of scope): the porting notes document the change; the warpSPH side was not touched.
