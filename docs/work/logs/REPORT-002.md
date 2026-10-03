# REPORT-002

Status: DONE          Branch: `local-model`   HEAD: `00fdf21` (T2.4; this report is committed as the last commit on top)   Date: 2026-10-03

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T2.1 | DONE | `717a8d3` | `cone` kernel (Q3a) + `cover_vector_scene` in the scene layer; 6 new tests; worst scene−numpy error 4.4e-16 vs the 3.6e-13 bound |
| T2.2 | DONE | `3dd82a0` | `coverExact` switch in `_detect_surface` (default off, exactly 3 edits); max\|Cw_quad−Cw_exact\| = 3.41e-3·n_wH² (bound 1e-2); `check --cases tank,dambreak` OVERALL: PASS |
| T2.3 | DONE | `037c592` | `--cfg`/`--physics` gate in the harness; detcmp: zero detector disagreements on both cases; `coverExact=true` gate PASS (two bit-level lines ~1e-6, gate unaffected) |
| T2.4 | DONE | `00fdf21` | Q2 conditioning study: W⁵ g(0,0) channel classified at k = 5 (w2 route A ACCEPTABLE 6.2e-8; w4 route A NOT USABLE 1.7e-3); flat-floor sign correct; stable-route vertex finding |

## Acceptance evidence

All logs in `.tmp/` (git-ignored); every run in the conda env `warp`, `OMP_NUM_THREADS=1`.

**T2.1** (log `.tmp/work002_t21_*.log`):
```
$ python -m pytest tests/edge/test_cover_scene.py -q
6 passed, 14 warnings in 2.68s
$ python -m pytest tests/edge -q
708 passed, 14 warnings in 422.96s (0:07:02)        # = start 702 + 6 new, no count drift
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q
72 passed, 14 warnings in 115.83s (0:01:55)
```
Worst errors with margin (probe `.tmp/work002_t21_numbers.log`): (a) max|scene−np| = 4.440892e-16 vs bound 1e-12·H² = 3.6e-13 (≈8e2×); (b) 3.330669e-16 vs 1e-12 (≈3e6×); (f) 2.498002e-16 vs 3.6e-13 (≈1.4e3×). Negative controls exceed their bounds: 7.19e-3 > 1e-4 (H·1.01) and 6.16e-1 > 1e-2 (negated).

**T2.2** (log `.tmp/work002_t22_*.log`):
```
$ python -m pytest tests/edge/test_deltasph_cover.py -q
1 passed, 14 warnings in 2.81s
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q
72 passed, 14 warnings in 148.36s (0:02:28)
$ cd python && python -m edgebound.deltasph_regress check --cases tank,dambreak
OVERALL: PASS          (exit 0; tank 6/6 + dambreak 8/8 lines PASS, dambreak 582.8 s wall)
```
max\|Cw_quad − Cw_exact\| = 3.413222e-03 · n_wH² (bound 1e-2; ≈3× under the reviewer's 4.8e-3); max\|Cw_exact\| = 15.18234 = the analytic half-plane floor-row value at d = dp/2 (bound > 0.5·n_wH² = 8); sign control 1.899 > 0.5. `git diff python/edgebound/deltasph2d.py` = exactly the three edits (one import, one config field, one two-line branch).

**T2.3** (logs `.tmp/work002_t23a_phys.log`, `_negctl`, `_t23b_db`, `_t23b_tank`, `_t23c`):
```
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak        # default cfg
PHYSICS GATE: PASS        (exit 0; all 10 gate items PASS)
$ cd python && python -m edgebound.deltasph_regress check --physics --cases dambreak --cfg alpha=0.5
  [gate] KE rel max vs B      value= 0.342365700103  limit: <= 0.05  FAIL
  [gate] P1 arrival |diff|    value= inf  limit: <= 0.05  FAIL
  [gate] minDensityMin        value= 0.999688530355  limit: >= 0.97  PASS
  [gate] maxDensityMax        value= 1.00118391114  limit: <= 1.03  PASS
  [gate] steps drift          value= 0  limit: <= 0.05 (baseline 6683)  PASS
PHYSICS GATE: FAIL        (exit 1)
$ python -m edgebound.deltasph_detcmp              # dambreak (default), steps 0,500,1500,3000,5000
$ python -m edgebound.deltasph_detcmp --case tank  # steps 0,500,1500,3000,5000
  ... disagree = 0 at every snapshot of both cases; findings: none
  dambreak max|Cw_q-Cw_e|/(n_w H^2): 3.413e-03 -> 1.022e-02;   tank: 3.413e-03 -> 7.989e-03
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg coverExact=true
PHYSICS GATE: PASS        (exit 0)
```
KE relative deviations vs the baseline with `coverExact=true`: t* = 1: 0 (bit-identical), t* = 2: 1.44e-08, t* = 2.5: 2.40e-08; arrival 2.48970315022 t* (identical to baseline). Two dambreak bit-level lines FAIL by ~1e-6 relative (maxVelocityMax 6.6810029 vs 6.6810456, diff 4.26e-05; maxDensityMax diff 6.26e-07) — a handful of surface flags flip after step ~5000 (detcmp covered to step 5000 with zero disagreements); no gate item affected (see Failures).

**T2.4** (run 2 log `.tmp/work002_t24b.log`, total 6.0 s; vertex probe `.tmp/work002_t24_vertex_probe.log`; full table in `docs/q2-conditioning.md`):
```
w2 k=5 degree=25 scale= 3.417e+00 | A  6.18e-08  B  4.68e-08  C  1.13e-05  D  5.85e-16
w4 k=5 degree=40 scale= 3.767e+00 | A  1.66e-03  B  2.42e-03  C  7.53e-05  D  2.69e-12

classification at k = 5 (fixed: <= 1e-8 GOOD, <= 1e-5 ACCEPTABLE, > 1e-5 NOT USABLE):
w2: A warp ACCEPTABLE;  B np plain ACCEPTABLE;  C np stable(8,6) NOT USABLE;  D np stable(16,8) GOOD
w4: A warp NOT USABLE;  B np plain NOT USABLE;  C np stable(8,6) NOT USABLE;  D np stable(16,8) GOOD

flat-floor sign check: point (0, 0.3), support 1, solid below y = 0:
w2: g0_y = -1.5485900248e-01   T_y = -6.8366791491e-08   T_y < 0: True
w4: g0_y = -7.3201280751e-02   T_y = -4.8836588922e-08   T_y < 0: True
```
Reviewer reproduction: all eight route-A values within factor 3.9 (ratios 1.3–3.9); w4p5 B within 2.9; w4p5 C/D reproduce on the reviewer's own 6 points (3.246e-11 / 1.201e-16 vs 3.2e-11 / 5e-17, factors 1.02 / 2.4) but not on the 214-point set — the vertex finding below.

**Definition-of-done re-runs (last, before this report):**
```
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py -q
72 passed, 14 warnings in 63.51s (0:01:03)        # exit 0 (log .tmp/work002_dod_72.log)
$ python -m pytest tests/edge -q
709 passed, 14 warnings in 276.08s (0:04:36)      # exit 0; = start 702 + 6 (T2.1) + 1 (T2.2) (log .tmp/work002_dod_edge.log)
```
Default-config `check --cases tank,dambreak` OVERALL: PASS: achieved in T2.2 (evidence above; not re-run).

## Numbers

| quantity | value | tolerance / bound | margin | file/command |
|---|---|---|---|---|
| T2.1 worst \|scene−np\|, case (a) | 4.440892e-16 | 1e-12·H² = 3.6e-13 | ≈8e2× | `.tmp/work002_t21_numbers.log` |
| T2.1 worst \|scene−np\|, case (b) | 3.330669e-16 | 1e-12 | ≈3e6× | `.tmp/work002_t21_numbers.log` |
| T2.1 worst \|scene−np\|, case (f) | 2.498002e-16 | 3.6e-13 | ≈1.4e3× | `.tmp/work002_t21_numbers.log` |
| T2.1 negative control H·1.01 | 7.192869e-03 | > 1e-4 (must exceed) | 72× | `.tmp/work002_t21_numbers.log` |
| T2.2 max\|Cw_quad−Cw_exact\|/n_wH² (tank, initial) | 3.413222e-03 | ≤ 1e-2 | ≈2.9× under | `.tmp/work002_t22_numbers.log` |
| T2.2 max\|Cw_exact\| (tank, floor row) | 15.18234 | analytic 15.18 at d = dp/2 | matches to printed digits | `.tmp/work002_t22_numbers.log` |
| T2.3 gate items, default cfg | 10/10 PASS | exit 0 | — | `.tmp/work002_t23a_phys.log` |
| T2.3 gate, alpha=0.5 | KE 0.3424 / limit 0.05 | exit 1, KE FAIL | 6.8× over (as designed) | `.tmp/work002_t23a_negctl.log` |
| T2.3 detcmp disagreements | 0 (all snapshots, both cases) | > 2 % of N_near = finding | none | `.tmp/work002_t23b_*.log` |
| T2.3 detcmp max\|Cw_q−Cw_e\|/n_wH², dambreak step 5000 | 1.022e-02 | (informational) | grows from 3.413e-03 | `.tmp/work002_t23b_db.log` |
| T2.3 coverExact=true gate | 10/10 PASS | exit 0 | — | `.tmp/work002_t23c.log` |
| T2.3 coverExact=true dambreak ke_tstar2 | 0.7540743518 vs baseline 0.754074340911 | (informational) | rel 1.44e-08 | `.tmp/work002_t23c.log` |
| T2.3 coverExact=true bit-level maxVelocityMax | 6.6810029 vs 6.6810456 | tol 6.681e-09 (bit-level) | FAIL by rel 6.4e-06 (observation) | `.tmp/work002_t23c.log` |
| T2.4 w2 k=5 worst rel err, route A | 6.18e-08 | ≤ 1e-5 ACCEPTABLE | 6.2× above the 1e-8 GOOD line | `.tmp/work002_t24b.log` |
| T2.4 w4 k=5 worst rel err, route A | 1.66e-03 | ≤ 1e-5 | 166× over (NOT USABLE) | `.tmp/work002_t24b.log` |
| T2.4 w4 k=5 route D (stable(16,8)), 214 points | 2.69e-12 | ≤ 1e-8 GOOD | — | `.tmp/work002_t24b.log` |
| T2.4 w4 k=5 route C at the vertex (0,0) | 7.53e-05 (2.835e-04 abs) | (finding) | vs 3.2e-11 away from vertices | `.tmp/work002_t24b.log` + vertex probe |
| T2.4 flat-floor T_y | w2 −6.837e-08, w4 −4.884e-08 | < 0 (into the wall) | both < 0 | `.tmp/work002_t24b.log` |
| T2.4 reviewer reproduction (route A, 8 values) | ratios 1.3–3.9 | ≤ 10 | all OK | `.tmp/work002_t24b.log` |

## Deviations

- **Sloshing gate items (T2.3):** the spec's sloshing "rhoMin / rhoMax" were implemented as the case's `minDensityMin` / `maxDensityMax` metrics (the case has no other density metrics); documented in the log.
- **T2.3a acceptance (2):** `alpha=0.5` was sufficient as the negative control (exit 1, KE item FAIL); the spec's `alpha=2.0` fallback was not needed.
- No other deviations. `--perturb` is not a valid control for the physics gate (it moves the KE by only 3e-3, WORK-001) — logged as the spec requires; this is a note, not a deviation.

## Failures and open questions

- **T2.4 finding — the np2d stable routes lose accuracy at polygon vertices** (reported, not fixed; `np2d.py` is not in the allowed-files list). At w4 k = 5 the worst error of stable=(8,6) is 7.53e-05 on the 214-point set, localised at the vertex (0,0) (abs 2.835e-04), vs 3.2e-11 away from vertices (the reviewer's 6-point set contains no vertices or edge midpoints); stable=(16,8) is 2.69e-12 at the vertex vs 1.2e-16 away. The effect grows with degree (w2 k=5: 1.13e-05 at the vertex vs 1.0e-11 away). Question for the reviewer: should the stable basis handle vertex points (or should the Q2 plan route avoid querying exactly at vertices — in the solver the query points are particle positions, which sit a lattice offset from the walls, so the practical impact is probably small)?
- **coverExact=true flips a handful of dambreak surface flags after step ~5000** (T2.3c): two bit-level lines FAIL by ~1e-6 relative (maxVelocityMax, maxDensityMax); the physics gate PASSES and the KE/arrival/steps deviations are ≤ 2.4e-8. Consistent with the ~1e-2 cover-vector difference (detcmp) rotating the Barecasco cover direction near the cone boundary for a few surface particles. Question: is a ~1e-6 bit-level drift acceptable for the port (the gate is the acceptance and it passes)?
- **T2.4 run 1 failed** (`KeyError: unknown kernel EdgeKernel(...)`: `get_kernel` resolves by name only) — fixed in one debug attempt (pass the name string, as the reviewer probe does); logged.
- **detcmp Cw error grows** 3.4e-3 → ~1e-2 of n_wH² on the evolved dambreak state (the wave on the right wall); my dambreak initial-state value is 3.413e-3 where the reviewer measured 4.8e-3 on the same case (both under the 1e-2 bound; the difference is probably the exact initial state / packing — not pursued, out of scope).

## Files changed

Against the starting commit `54650dd` (the four task commits; this report commit adds `REPORT-002.md` and the `T2.4 DONE` line in `LOG-002.md`):

```
$ git diff --stat 54650dd 00fdf21
 docs/README.md                       |   1 +
 docs/deltasph-porting-notes.md       |   1 +
 docs/q2-conditioning.md              |  78 +++++++++++++++++++++++++++++
 docs/work/logs/LOG-002.md            | 443 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 python/edgebound/boundaryOps.py      |   7 ++-
 python/edgebound/cover.py            |  30 +++++++++++
 python/edgebound/deltasph2d.py       |   5 +-
 python/edgebound/deltasph_detcmp.py  |  81 ++++++++++++++++++++++++++++++
 python/edgebound/deltasph_regress.py | 108 +++++++++++++++++++++++++++++++++++-----
 python/edgebound/kernels.py          |   2 +
 python/edgebound/q2_conditioning.py  | 211 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/edge/test_cover_scene.py       | 101 +++++++++++++++++++++++++++++++++++++
 tests/edge/test_deltasph_cover.py    |  57 ++++++++++++++++++++++
 13 files changed, 1111 insertions(+), 14 deletions(-)
```
No changes to `scene.py`, `warpbc.py`, any existing test, `regress_baseline.json`, or anything in `results/`. Scratch in `.tmp/` (git-ignored).

## What I did NOT verify

- **Sloshing** regression (out of scope per WORK-002, "only if time allows"); the sloshing gate code path exists in the harness but was not run end-to-end.
- **Q1 (Laplacian), Q3b (cone count), and the Q2 implementation in the solver** — T2.4 is a conditioning study only; the solver still uses polar quadrature for the tensile term (and `coverExact` stays default-off; making it default is out of scope).
- **The w4 stable-basis Warp plan** — would require changing `warpbc.py` (out of scope); `docs/q2-conditioning.md` records what is needed.
- **The warpSPH port** — `~/dev/warpSPH*` untouched (read-only per KICKOFF rule 3).
- **Absolute wall-clock times** — the GPU is shared with the local LLM (~87 % util); margins and ratios, not seconds, are the portable quantities.
- **T2.4 route A at k > 5** and other point sets — only the spec'd k = 1..5 and 214-point set were run.
