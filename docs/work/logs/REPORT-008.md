# REPORT-008
Status: **DONE**          Branch: `local-model`   HEAD: `e9f1b2a` (T8.4, last task commit; **this report is committed as the final commit** on the branch)   Date: 2026-10-05

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T8.1 | DONE | cfa483e | `warpbc.edge_channels` channel-restricted plans + a `DevicePlan` cache per `(kernel, device, channels)`; pruned (3,4) bit-identical to the full g0, plan 4 vs 21 terms; 2 new tests |
| T8.2 | DONE | 89cbd8d | pruned gradient adjacency for cover / tensile via `Scene.buildAdjacency(..., channels=)` + `sceneOperation` guard; cover / tensile bit-identical; 2 new tests |
| T8.3 | DONE | f8eb1a4 | `_wall_data` reuses the `no_penetration` adjacency across the step boundary; `buildAdjacency` 10 → 9 per step, dynamics change ≤ 8.66e-16; 3 new tests |
| T8.4 | DONE | e9f1b2a | harness PASS (margins ≤ 0.011, `PHYSICS GATE: PASS`), profile 36.479 / 46.490 ms/step (−37.2 % / −35.5 %, `buildAdjacency` 9 / `sceneOperation` 10 per step), sloshing KE 2.8100e-05, four docs updated |

## Acceptance evidence

### T8.1 — `python -m pytest tests/edge/test_channel_pruning.py -q -k "pruned_edge_channels or device_plan" -s` → `2 passed in 0.89s`

```
(a) cone stable=None: max|full g0| = 0.955 (> 0.5), pruned (3,4) torch.equal, (5,6) g0 = 0
(a) lw2 stable=None: max|full g0| = 1.592 (> 0.5), pruned (3,4) torch.equal, (5,6) g0 = 0
(a) w2 stable=None: max|full g0| = 1.485 (> 0.5), pruned (3,4) torch.equal, (5,6) g0 = 0
(a) w2 stable=(16, 8): max|full g0| = 1.485 (> 0.5), pruned (3,4) torch.equal, (5,6) g0 = 0
(a) plan sizes: pruned (3,4) nE+nV = 4 (cone, lw2, w2); full w2 nE+nV = 21 (E 16, V 5)
(b) cache: same key is the same object, pruned != full, two edge_channels calls construct 1 plan
```

Regression `python -m pytest tests/edge/test_warpbc*.py tests/edge/test_tensile_scene*.py tests/edge/test_cover_scene.py -q` → `37 passed, 14 warnings in 10.38s` (old tests unchanged and green).

### T8.2 — `python -m pytest tests/edge/test_channel_pruning.py -q -s` → `4 passed, 14 warnings in 2.50s`

```
(c) pruned adjacency channels = [3, 4] (full is None); Naive Gradient of a constant torch.equal
(c) guard: Density, Covariance, perQuery+a1 Gradient, returnReaction Gradient, channels=(0,) all raise ValueError
(d) cover: max|Δ| = 0.00e+00 (<= 1e-13 * scale 0.308)
(d) tensile w2: max|Δ| = 0.00e+00 (<= 1e-13 * scale 234.856)
(d) tensile w4: max|Δ| = 0.00e+00 (<= 1e-13 * scale 751.030)
```

Five-file solver set `python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q` → `82 passed, 14 warnings in 80.97s`.
Full suite `python -m pytest tests/edge -q` (`.tmp/w008_tests_t82.log`) → `764 passed, 15 warnings in 433.95s (0:07:13)` = 760 + 4.

### T8.3 — `python -m pytest tests/edge/test_wall_data_reuse.py -q -s` → `3 passed, 14 warnings in 3.50s`

```
(e) with cache: 36 buildAdjacency over 4 steps = 9.00/step (== 9)
(e) without cache: 40 buildAdjacency over 4 steps = 10.00/step (== 10)
(f) x: max|Δ| = 0.00e+00     (f) v: max|Δ| = 8.66e-16     (f) rho: max|Δ| = 0.00e+00   (all <= 1e-12)
(g) two _wall_data at the same state: builds = 0, 0 (second builds nothing)
(g)(i) x + 1e-3 in y: 1 build, |Δlam| vs stale cache = 8.25e-03 (> 1e-4)
(g)(ii) body + 1e-3 rad: 1 build, |Δlam| vs stale cache = 8.88e-03 (> 1e-4)
(g)(iii) Hvec * 1.01: 1 build
(g)(iv) 1.1 * rho: 0 builds, lam/G/A torch.equal to fresh
(g)(v) g = [0.5, -9.0]: 0 builds, A torch.equal to fresh, |ΔA| vs old g = 5.92e-01 (> 1e-2)
```

Five-file solver set → `82 passed, 14 warnings in 113.98s`.
Full suite (`.tmp/w008_tests_t83.log`) → `767 passed, 15 warnings in 375.56s (0:06:15)` = 760 + 4 + 3.

### T8.4

1. Harness `cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak > ../.tmp/w008_check.log` — **every bit-level line PASS, `PHYSICS GATE: PASS`**. Decisive lines (copied from the log):

```
rmseBulk           value= 0.00149929753837 baseline= 0.00149929753835 tol= 1.499e-12 margin= 0.011  PASS
rmseNear           value= 0.00181340794898 baseline= 0.00181340794898 tol= 1.813e-12 margin= 0.004  PASS
keLast             value= 7.28437955168e-07 baseline= 7.28437955168e-07 tol= 1.000e-12 margin= 0.000  PASS
rhoMin             value= 1.00008342046 baseline= 1.00008342046 tol= 1.000e-09 margin= 0.000  PASS
rhoMax             value= 1.00246989449 baseline= 1.00246989449 tol= 1.002e-09 margin= 0.000  PASS
steps              value= 3502 baseline= 3502 tol= 3.502e-06 margin= 0.000  PASS
ke_tstar1          value= 0.357363651811 baseline= 0.357363651811 tol= 3.574e-10 margin= 0.000  PASS
ke_tstar2          value= 0.754228393303 baseline= 0.754228393302 tol= 7.542e-10 margin= 0.001  PASS
ke_tstar25         value= 0.922766940751 baseline= 0.922766940747 tol= 9.228e-10 margin= 0.005  PASS
p0_arrival_tstar   value= 2.47397042889 baseline= 2.47397042889 tol= 2.474e-09 margin= 0.000  PASS
maxVelocityMax     value= 6.59792039874 baseline= 6.59792039874 tol= 6.598e-09 margin= 0.001  PASS
minDensityMin      value= 0.999428350557 baseline= 0.999428350557 tol= 9.994e-10 margin= 0.000  PASS
maxDensityMax      value= 1.00603117793 baseline= 1.00603117793 tol= 1.006e-09 margin= 0.000  PASS
steps              value= 6683 baseline= 6683 tol= 6.683e-06 margin= 0.000  PASS
[physics] dam break KE vs dambreak_B_nx67: max|KE-KE_B|/max(KE_B) over common range = 3.5344e-04 (668 samples)
[physics] dam break P1 arrival = 2.4740 t*  (reference 2.49, |diff| = 0.0160)
PHYSICS GATE: PASS
```

(The 10 `[gate]` lines all PASS as well.) The dambreak values match the reviewer's prototype to the digit (ke_tstar1/2/25 0.357363651811 / 0.754228393303 / 0.922766940751; P1 arrival 2.47397042889; maxVelocityMax 6.59792039874); the tank values match (rmseBulk 0.00149929753837, rmseNear 0.00181340794898, keLast 7.28437955168e-07). No tolerance or baseline was touched.

2. Profile after `cd python && python -m edgebound.deltasph_profile > ../.tmp/w008_profile_after.log`, back to back with the starting-state "before" (`.tmp/w008_profile_before.log`, 58.079 / 72.043 ms/step at clean HEAD):

```
CASE: dam break  ...   ms/step  (timer inclusive of step) = 36.479    (wall time / steps) = 36.484
CASE: sloshing  ...    ms/step  (timer inclusive of step) = 46.490    (wall time / steps) = 46.495
(a) buildAdjacency calls/step = 9.000 ; sceneOperation calls/step = 10.000   (both cases; required: 9 / 10)
```

Reductions: (58.079 − 36.479)/58.079 = **37.2 %** (reviewer's: −35 %); (72.043 − 46.490)/72.043 = **35.5 %** (reviewer's: −34 %). Both above the 15 % finding threshold, so no GPU-load finding.

3. Sloshing T = 1.5 s, `timeout 5400 python .tmp/slosh_default.py > .tmp/w008_slosh.log` (the WORK-006 script, run unchanged):

```
default (WORK-006) sloshing T=1.5: steps 15001 wall 896 s  KE rel max vs slosh_B = 2.8100e-05 (1500 samples, gate 5%)  maxVel 0.5920  rho [0.99928, 1.00490]
```

KE 2.8100e-05 < 1e-4 (WORK-006: 2.8102e-05; reviewer's prototype: 2.8099e-05). maxVel and densities match WORK-006 exactly; 15001 steps (same count). The wall time (896 s) is longer than the reviewer's 726 s only because of GPU load (70426/97887 MiB in use by the resident LLM at job start) — wall time is not a gate.

4. Documentation: `docs/deltasph-profile.md` `## After WORK-008` (two profiler tables, same layout as the WORK-006 section, before/after ms/step); `docs/scene-architecture.md` §6b one paragraph (pruning + guard, plan cache, cross-step reuse + key, measured numbers); `docs/deltasph-validation.md` `## Speed after WORK-008` (harness margins, profile numbers, sloshing line); `docs/deltasph-porting-notes.md` ONE change-log row. Every number in these files is one from LOG-008.

### Definition of done — final full suite (run once more at the end)

`python -m pytest tests/edge -q` (`.tmp/w008_tests_final.log`, run last):

```
767 passed, 15 warnings in 347.33s (0:05:47)
```

= 760 + 7 new (4 channel-pruning + 3 wall-data-reuse). No failures.

## Numbers

| quantity | value | tolerance | margin | file/command |
|---|---|---|---|---|
| pruned (3,4) g0 vs full (cone, lw2, w2, w2 Cheb (16,8)) | Δ = 0 (torch.equal) | exact | 0 | `test_channel_pruning.py` (a) |
| max\|full g0\| cone / lw2 / w2 / w2(16,8) | 0.955 / 1.592 / 1.485 / 1.485 | > 0.5 | 0.455 / 1.092 / 0.985 / 0.985 | (a) |
| pruned (3,4) plan size, cone / lw2 / w2 | 4 terms (E 4, V 0) | == 4 (reviewer) | 0 | (a) |
| full w2 plan size | 21 terms (E 16, V 5) | == 21 | 0 | (a) |
| DevicePlan builds over two same-key `edge_channels` calls | 1 | == 1 | 0 | (b) |
| cover (cone) pruned vs full adjacency | max\|Δ\| = 0.00e+00 | ≤ 1e-13 · max\|full\| (scale 0.308) | 0 | (d) |
| tensile w2 / w4 pruned vs full | 0.00e+00 / 0.00e+00 | ≤ 1e-13 · scale (234.856 / 751.030) | 0 | (d) |
| `buildAdjacency` calls/step with / without cache | 9.00 / 10.00 | == 9 / == 10 | 0 | `test_wall_data_reuse.py` (e) |
| max\|Δ\| x / v / rho after 8 steps (cache vs disabled) | 0 / 8.66e-16 / 0 | ≤ 1e-12 | 1.2e3× (worst: v) | (f) |
| \|Δlam\| vs stale cache, x+1e-3 / body +1e-3 rad | 8.25e-03 / 8.88e-03 | > 1e-4 | 82× / 89× | (g)(i), (ii) |
| \|ΔA\| vs old g (no rebuild) | 5.92e-01 | > 1e-2 | 59× | (g)(v) |
| full suite at T8.2 / T8.3 / final | 764 / 767 / 767 passed | 767 | 0 | `.tmp/w008_tests_{t82,t83,final}.log` |
| tank rmseBulk / rmseNear / keLast margin | 0.011 / 0.004 / 0.000 | ≤ 1 (stop) | 91× / 250× / 0 | `.tmp/w008_check.log` |
| tank rhoMin / rhoMax / steps margin | 0.000 / 0.000 / 0.000 | ≤ 1 | 0 | ditto |
| dambreak ke_tstar 1 / 2 / 2.5 margin | 0.000 / 0.001 / 0.005 | ≤ 1 | 0 / 1000× / 200× | ditto |
| dambreak p0_arrival / maxVelocityMax margin | 0.000 / 0.001 | ≤ 1 | 0 / 1000× | ditto |
| dam break KE vs `dambreak_B_nx67` (668 samples) | 3.5344e-04 | ≤ 0.05 | 141× | ditto |
| dam break P1 arrival | 2.4740 t* | \|diff\| ≤ 0.05 | 0.0160 | ditto |
| dam break ms/step before → after | 58.079 → 36.479 | ≥ 15 % reduction (finding threshold) | 37.2 % | `.tmp/w008_profile_{before,after}.log` |
| sloshing ms/step before → after | 72.043 → 46.490 | ≥ 15 % | 35.5 % | ditto |
| `buildAdjacency` / `sceneOperation` calls/step after | 9.000 / 10.000 | 9 / 10 | 0 | `.tmp/w008_profile_after.log` |
| sloshing T=1.5 KE max-rel vs `slosh_B` | 2.8100e-05 | < 1e-4 | 3.6× | `.tmp/w008_slosh.log` |
| sloshing maxVel / rho | 0.5920 / [0.99928, 1.00490] | equal to WORK-006 | matches | ditto |

## Deviations

None substantive. All code changes are the reviewer's tested prototype hunks (`docs/work/refs/review7_work008_proto.diff`), applied per task with `git apply --include` after `git apply --check`, byte-identical to the prototype; `git diff` showed only the allowed files at each step. Two process notes:

1. The "before" profile at the starting state had to be re-taken after a `git checkout python/edgebound/warpbc.py` because the T8.1 hunk was already applied at that point; the hunk was re-applied afterwards. The logged before numbers (58.079 / 72.043 ms/step) are at clean HEAD.
2. Three fixes inside MY NEW test files (not the code under test, not existing tests): the `geometry` fixture made function-scoped (it used the function-scoped `device` param); `lw2` registered lazily via `from edgebound.viscosity import lap_factor; lap_factor(1.0, "w2")` (exactly as `tests/edge/test_scene.py::test_adjacency_kernel_guard` does); a helper docstring in `test_wall_data_reuse.py` closed with `""` instead of `"""` (`SyntaxError`, verified fixed with `ast.parse`).

## Failures and open questions

Nothing failed. Findings:

1. **tank rmseBulk margin 0.011, slightly above the reviewer's ≤ 0.007** on the same code: last-digit GPU run-to-run spread (my value 0.00149929753837 vs the reviewer's 0.00149929753835 — identical to 13 digits). Far below the stop threshold of 1. Question: no action requested; noting it for the reviewer's record.
2. **Sloshing wall time 896 s vs the reviewer's 726 s** (same 15001 steps): GPU load from the resident local LLM (70426/97887 MiB in use at job start); wall time is not a gate.
3. **Remaining cost split after this package** (input to the next one): `buildAdjacency` is still the largest single cost — 21.076 ms (57.78 %) dam break / 27.857 ms (59.92 %) sloshing — now at 9 calls/step with ≈ 0.56× the per-build cost of the full-channel builds; then `_detect_surface` 5.100 / 5.260 ms (22.15 % / 17.73 %); then the fluid pair sums 9.197 / 12.853 ms (25.21 % / 27.65 % — the same absolute work, a larger share because the step is shorter).

## Files changed

`git diff --stat 780d150` (starting commit `REVIEW-007: accept WORK-007; ...`):

```
 docs/deltasph-porting-notes.md     |   1 +
 docs/deltasph-profile.md           |  74 +++++++++++++++++++++++++++++++++++++++
 docs/deltasph-validation.md        |  31 +++++++++++++++++
 docs/scene-architecture.md         |  11 ++++++
 docs/work/logs/LOG-008.md          | 112 +++++++++++++++++++++++++++++++++++++++++++++++++++
 python/edgebound/cover.py          |   1 +
 python/edgebound/deltasph2d.py     |  14 +++++---
 python/edgebound/scene.py          |  24 ++++++++-----
 python/edgebound/tensile.py        |   3 +-
 python/edgebound/warpbc.py         |  58 ++++++++++++++++++++++++-------
 tests/edge/test_channel_pruning.py | 184 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/edge/test_wall_data_reuse.py | 220 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 12 files changed, 701 insertions(+), 32 deletions(-)
```

plus this report, committed in the final commit together with the final-suite log entry in LOG-008.

## What I did NOT verify

- The reviewer's micro-probes (`review7_edge_cost_probe.py`, `review7_prune_probe.py`): the per-kernel `edge_channels` ms figures and the reviewer's prototype measurements cited in WORK-008.md §1 are the reviewer's; I did not re-run them.
- The CPU-only path: this machine has CUDA, so every run was on `cuda:0`; the `DEVICES` fallback to `cpu` in the new tests was not exercised.
- Pruning for the other operations (`_wall_data` / `_surface_state` family, `lap_lambda_scene`): explicitly out of scope; the `sceneOperation` guard rejects them (the (c) negative controls).
- Multi-body cache invalidation: the cache tests use the single-body tank; the pose check loops over all bodies, but I did not measure it on a two-body scene.
- CUDA-graph capture / fixed-capacity adjacency: a port concern (`deltasph-porting-notes.md` §3), not part of this package.
