# REVIEW-007 — WORK-007 (branch `local-model`, `64e74e1..1090f5a`)

Reviewer: Claude, 2026-10-05. Verdict: **ACCEPT T7.1, T7.2, T7.3** (T7.3 with two documentation corrections). No rule violation, no tolerance edited, no defect found in the code. The correction that matters is one of *interpretation* (§3).

Scope note: WORK-006 (the other recent package) was reviewed and accepted in `REVIEW-006.md` (commit `64e74e1`); nothing about it was re-opened. Its second life as the BLOCKED/resume history of WORK-007 is covered by the WORK-007 starting-state check in `LOG-007.md`, which re-measured 82 / 756 after `REVIEW-006`.

## 1. Mechanical gate
* Commits `64e74e1..1090f5a`: `T7.1` (`b3c6621`), `T7.3` (`b19faf7`), `REPORT-007` (`1090f5a`), one per task and the report, subjects as specified. `git diff --stat`: 8 files, all on the allowed list (`deltasph2d.py` +8/−3 lines, the new test file, the new derivation page, three doc files, `LOG-007`/`REPORT-007`). **No change** to `scene.py`, `warpbc.py`, `viscosity.py`, `results/`, the harness, the baseline, any existing test. `git status` clean.
* `git diff` of `deltasph2d.py` equals spec T7.1 item 1 line by line (config comment, the `elif … "noslip": pass` dispatch, the `ValueError` text listing three forms, the body-loop branch with `hit`, `dd = d.clamp(min=0.25 dx)`, `ν_eff = fac/8`, all-components relative velocity, `continue`). The reading `(v − b.velocityAt(x))[near]` instead of `(v[near] − b.velocityAt(x)[near])` is the same number.
* Tests: `pytest tests/edge/test_deltasph_noslip.py -q -s` → **4 passed** (3.0 s), printed numbers equal the report's (the (d) off-switch difference prints 1.99e-13 against the report's 2.09e-13: GPU round-off, limit 1e-12). Full suite (reviewer, on HEAD plus the uncommitted WORK-008 prototype of `refs/review7_work008_proto.diff`, which changes no result): **760 passed** (= the report's count).

## 2. Evidence audit
| check | result |
|---|---|
| tolerance history | the test file has one commit (`b3c6621`); every tolerance (1e-9, 1e-10, 0.2, 0.15, 0.6, 1e-12) is the one of the work document; no skip, no xfail |
| (a) flat-wall absolute | reference = own `scipy.integrate.quad` of `∫W ds` (C2 written out), shares no code with the solver; 2.2e-16 … 4.4e-16 against a 1e-9 tolerance; wrong factor or sign is ≥ 1e-2: the check can fail |
| (b) floor and Galilean | the floor is pinned with the TRUE `F(z)` (1.2e-15); the Galilean control has both signs (co-moving 0, resting 3.42). **Reviewer addition (probe, `refs/review7_noslip_probe.py` (1)):** a *rotating* wall (ω = 0.8, particle co-rotating with `v = v_w(x_i)` from `velocityAt`): term = 0.0; at rest: 1.23 — the sloshing rolling wall's `velocityAt` path is therefore also right (the test only moved the wall linearly) |
| (c) factor `−2` is discriminated | the doc derives the factor 2 from "the standard one-sided form"; the evidence is the Couette row. Reviewer re-scaling of the printed numbers (probe (2)): prefactor 0.5 / 1 / 2 (implemented) / 4 → max row error 0.78 / 0.57 / **0.14** / 0.73 of `a_bulk(row 0)` against the tolerance 0.2: a wrong prefactor by a factor of 2 in either direction fails; the check is sensitive to the prefactor within ≈ [1.7, 2.4] |
| negative controls | `laplacian` / `pairwise` on the tangential Couette field give `|a| = |a_bulk| = 0.06014` (compared with the independent value), both > 0.5 · 0.06014: valid, they are free-slip by construction |
| second acceptance command re-run | the new-test command and the full suite (above). The T7.2 gates were not repeated (357 s + 963 s of GPU); instead the cheap diagnostic `.tmp/stability_noslip.py` and the structural reading of the numbers were checked (§3) |
| report numbers vs logs | the report's lines equal `LOG-007.md` and the quoted `.tmp/w007_*.log` files (spot checks: Couette/Poiseuille rows, gate lines, sloshing line); `760 = 756 + 4` |
| docs vs evidence | `noslip-wall.md` §5 table equals the test output; §3.5 table equals the reviewer's lattice probe; two statements corrected (§3) |

## 3. Corrections made (commit `REVIEW-007: ...`)
1. **"Intended physics" was an over-claim** (`derivations/noslip-wall.md` §6, `deltasph-validation.md`, `deltasph-porting-notes.md` row; `REPORT-007` "Failures" says the same and is left as the model wrote it). The no-slip wall here carries the scheme's *artificial* viscosity: `ν_eff = α c0 H/(8ξ)` = 3.1e-3 m²/s at the tank (α = 0.01, c0 = 44.3, H = 0.16), ≈ **3000 × water's** 1e-6 m²/s; `√(ν_eff t)` = 0.040 m ≈ **1.0 dp** at t = 0.5 s (physical: 0.7 mm). So the dam-break front slowing (6.60 → 3.80), the 10–31 % KE drop and the 14.6 % sloshing difference are properties of that coefficient (they scale with α, c0, H, i.e. with resolution), not a boundary-layer physics result. Added: a reviewer paragraph in the derivation §6, a "Reading of the numbers" paragraph in the validation section, and an open item "physical viscosity at the wall" (§7). The porting-notes row keeps its numbers (they are measured) and gains a pointer to the corrected reading.
2. **`nan` arrival**: `REPORT-007` says "P1 impact too weak to cross 0.05 in the truncated window". `nan` only says the crossing is later than the window: `T = 0.65 s` = 2.63 t\* (`t* = t √(g/H)`, H = 0.6), default arrival 2.474 t\*, so the no-slip arrival is later than 2.63 t\* (delayed by > 0.16 t\*); weaker is not shown. Corrected in the derivation and the validation section.
3. Cosmetic: the units parenthetical "(1/length^3 · velocity, per body)" in `noslip-wall.md` §1 was not a unit statement; replaced by the actual units (`[ν_eff] = L²/T`, `[|G|] = 1/L`, `[d] = L`, `rho0 = 1`).

## 4. Findings the next work document must assume
* `wallViscosityForm = "noslip"` exists (`deltasph2d.py`), default `"laplacian"` unchanged; the 82-test five-file set and the 756 older tests are untouched; **full suite 760**.
* No-slip numbers (information): dam break KE vs B 0.3165, P1 arrival after 2.63 t\*, `maxVelocityMax` 3.80 (default 6.60); sloshing 1.5 s KE vs B 0.146, maxVel 0.568, ρ [0.99931, 1.00483]; stability `k·dt` 0.0016 / 0.0020 (measured with the *default* dynamics, `k` up to 16.6 / 20.4 s⁻¹ at `d ≈ 0.25 dx`).
* Known limits stay as in the derivation page: flat-wall model per particle (corners: not Chiron's per-element form), explicit damping, `ν_eff` artificial; moving-wall check is by `velocityAt` only (linear in the test, rotating in the reviewer probe, sloshing as a run).
* The next package is **phase 2** (scene-path cost). The measured starting point is in `WORK-008.md`; the reviewer's prototype is `refs/review7_work008_proto.diff` (not committed to the tree).

## 5. Weak spots (none blocking)
* The `stability` diagnostic uses the default (laplacian) dynamics and the `near` set of that run; a no-slip run could have slightly different `d`. The margin (400×) makes this irrelevant.
* The no-slip Poiseuille row-0/1 tolerances are 0.6 of the bulk's distance (measured 0.22/0.42): they pass but state that the first-order flux term is not accurate in the first two rows; do not tighten them.
* Merge of `local-model` into `main` and the tag `reviewed-007` remain the user's call (no `reviewed-*` tag exists; base of the next review = `1090f5a`).
