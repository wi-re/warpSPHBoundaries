# REPORT-001

Status: DONE          Branch: `local-model`   HEAD: `62ddff8` (T1.1; this report is committed as the last commit on top)   Date: 2026-10-03

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T0.1 | DONE | `c4f38db` | truncated-run regression harness + baseline; `check` exits 0 (8m38 s), `--perturb` negative control fails 7/8 dambreak metrics |
| T0.2 | DONE | `9a824fc` | measured step profile (self-sum = 100 % of the step); scene layer dominates, `buildAdjacency` 4 and `sceneOperation` 12 per step — confirms the plan's claim |
| T1.1 | DONE | `62ddff8` | exact cover-vector formula verified against the independent polar oracle (mp−oracle ~1e-57, np−mp ~1e-16); 17 tests pass |

## Acceptance evidence

**T0.1** (all background, `setsid`, logs in `.tmp/`):
```
$ cd python && python -m edgebound.deltasph_regress record --cases tank,dambreak
wrote baseline: .../results/deltasph/regress_baseline.json
  [physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 9.5716e-09 (668 samples)
  [physics] dam break P1 arrival = 2.4897 t*  (reference 2.49, |diff| = 0.0003)
```
spread + tol/|value| table: every metric tol/|v| <= 7.878e-08; no metric has relative tol > 5 % (none weak).
```
$ cd python && time python -m edgebound.deltasph_regress check --cases tank,dambreak
  tank: 6/6 PASS   dambreak: 8/8 PASS (worst margin 0.057 on ke_tstar25)
OVERALL: PASS
real 8m37.985s        (exit 0: the harness exits 0 iff every metric PASS; << the 25-min budget)
```
```
$ cd python && python -m edgebound.deltasph_regress check --cases dambreak --perturb   # shifting=False
  ke_tstar1 FAIL  ke_tstar2 FAIL  ke_tstar25 FAIL  p0_arrival_tstar FAIL
  maxVelocityMax FAIL  minDensityMin FAIL  maxDensityMax FAIL  steps PASS
OVERALL: FAIL
PERTURB_EXIT=1        (7 of 8 dambreak metrics FAIL -> the harness can fail)
```

**T0.2**:
```
$ cd python && time python -m edgebound.deltasph_profile       (real 0m57.107s)
dam break: N=3240, 44.383 ms/step, self-sum = 100.0 % of the step
sloshing:  N=4200, 72.095 ms/step, self-sum = 100.0 %
```
`git diff --stat` for this task = the two new files + the one `docs/README.md` row (the README diff vs HEAD is only the `deltasph-profile.md` row). Full table + profiler lists in `docs/deltasph-profile.md`.

**T1.1**:
```
$ cd python && python -m pytest ../tests/edge/test_cover.py -v
17 passed in 54.37s
```

**Regression (must stay green), run once more at the end**:
```
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q
72 passed, 14 warnings in 124.15s (0:02:04)
```

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| T0.1 check worst margin (dambreak ke_tstar25) | 0.057 | tol 7.268e-08 | 0.057 < 1 | `.tmp/regress_check.log` |
| T0.1 check wall time | 8m38 s | < 25 min | 2.9x under | `.tmp/regress_check.log` |
| T0.1 physics KE rel max (check) | 6.6e-09 | 5 % (finding) | 7.5e-7 | `.tmp/regress_check.log` |
| T0.1 physics P1 arrival \|t*−2.49\| (check) | 0.0003 | 0.05 (finding) | 167x under | `.tmp/regress_check.log` |
| T0.1 weakest relative tol (all metrics) | 7.878e-08 | 5 % (weak flag) | none weak | `.tmp/regress_record.log` |
| T0.2 dambreak ms/step (timer) | 44.383 | self-sum within 10 % of step | 100.0 % | `.tmp/profile.log` |
| T0.2 dambreak buildAdjacency / step | 4.000 | plan claim 4 | matches | `.tmp/profile.log` |
| T0.2 dambreak sceneOperation / step | 12.000 | plan claim ~12 | matches | `.tmp/profile.log` |
| T0.2 dambreak ratio profiled/series | 0.709 | (informational) | series 62.562 ms/step | `.tmp/profile.log` + `dambreak_B_nx67_series.npz` |
| T0.2 sloshing ms/step (timer) | 72.095 | self-sum within 10 % | 100.0 % | `.tmp/profile.log` |
| T1.1 worst \|mp−oracle\|/H² (4 polygons, both H) | 1.743e-56 | 1e-12 | ~5.7e53x | `test_cover.py` + `.tmp/cover_report.py` |
| T1.1 worst \|np−mp\|/H² | 3.025e-16 | 1e-12 | ~3.3e3x | `test_cover.py` + `.tmp/cover_report.py` |
| T1.1 negative-control worst \|gap\| | 1.0 | > 1e-3 (must exceed) | 1000x | `test_cover.py` + `.tmp/cover_report.py` |
| T1.1 flat-floor worst \|mp/oracle−analytic\|/H² | 5.551e-17 | 1e-9 | ~1.8e7x | `test_cover.py` |

## Deviations

- **Negative control (T1.1):** implemented per the reviewer's **smoke reference** (continuous-K gradient minus the discontinuous-r edge-only, = `−H·block_grad(P,0,H)`), not the literal "central difference of the discontinuous-r value" reading. Reason: the literal reading gives a mismatch of ~1.5e-11 (see Failures) and can never exceed the required 1e-3; the smoke reference is the only reading that is > 1e-3 and matches the given number (1.0 at (1.3, 0.5)). Reported, not silent.
- **Sloshing (T0.1 acceptance 4):** not run (optional; ~35 min per run, `record` runs it twice). Reported here as "sloshing: not run".

## Failures and open questions

- **Negative-control wording (the one finding).** The work order says the negative control is "the edge-only formula −Σ n_e ∫ r ds vs the true gradient of r 1[r≤H], obtained from a central difference of the oracle value of r 1[r≤1]". That central difference converges to the **edge-only formula itself** (measured: at (1.3, 0.5) the central-diff true gradient = (−0.407089, 0), equal to the edge-only to 1.5e-11): the distributional gradient of the discontinuous kernel r 1[r≤H] **is** −Σ n_e ∫_chord r ds, so the edge-only formula is exact for that kernel and there is **no circle term to find on it**. The smoke reference's "circle term" (1.0 at (1.3, 0.5)) is instead the **continuous-K gradient minus the discontinuous-r edge-only** = (0.5929, 0) − (−0.4071, 0) = (1.0, 0) = −H·block_grad(P, 0, H), i.e. the −H·1[r≤H] part of K = r−H. **Question for the reviewer:** which reading was intended? The test implements the smoke-reference reading (the one matching the given number) and separately asserts the ~0 central-difference result as a recorded finding. No numbers were adjusted; no test/tolerance/baseline was edited to force a pass.
- **Sloshing: not run** (T0.1 acceptance 4, optional).
- **No weak relative tolerances** in T0.1 (every metric tol/|v| ≤ 7.9e-8, far under the 5 % weak flag).
- **Physics (secondary) checks** — all far under the finding thresholds, reported not acted on: record KE rel max 9.6e-9 / arrival |diff| 0.0003; check 6.6e-9 / 0.0003; the `--perturb` run 3.3e-3 / 0.016 (expected to differ, it is the negative control).
- **Pre-existing reviewer changes remain uncommitted** (as instructed at the start): `HANDOFF.md`, `docs/README.md` (a `work/` row + one word edit), and the untracked `docs/work/` tree (KICKOFF/WORK/REVIEW/README, logs/.gitkeep). The only `docs/README.md` change I committed is the single `deltasph-profile.md` index row (the reviewer's changes were re-applied to the working tree after that commit).

## Files changed

Against the starting commit `e58ac7b` (the three task commits; this report commit adds `REPORT-001.md`):
```
$ git diff --stat e58ac7b 62ddff8
 docs/README.md                         |   1 +
 docs/deltasph-profile.md               | 138 ++++++
 docs/work/logs/LOG-001.md              | 138 ++++++
 python/edgebound/cover.py              | 113 +++++
 python/edgebound/deltasph_profile.py   | 220 +++++++
 python/edgebound/deltasph_regress.py   | 188 ++++++
 results/deltasph/regress_baseline.json |  92 ++++
 tests/edge/test_cover.py               | 209 +++++++
 8 files changed, 1099 insertions(+)
```
No changes to `deltasph2d.py`, `scene.py`, `warpbc.py`, any existing test, or any existing file in `results/`. Scratch is in `.tmp/` (git-ignored).

## What I did NOT verify

- The **sloshing** regression case (T0.1 acceptance 4) — not run.
- **Integration of the cover vector into the solver** — T1.1 is formula-level only (no solver change, per the work order); the formula is verified against the oracle, not against a full solver run.
- The other quadrature items (**Q1** Laplacian, **Q2** W⁵/5, **Q3b** cone area) and any warpSPH port — explicitly out of scope.
- **Absolute** wall-clock portability of the T0.2 ms/step (measured on this GPU with ~28 GiB free and the local LLM resident; the ratio to the stored series is the portable quantity, 0.709).
