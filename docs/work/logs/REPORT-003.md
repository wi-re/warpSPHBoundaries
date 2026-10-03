# REPORT-003

Status: DONE          Branch: `local-model`   HEAD: `58b5743` (T3.5; this report is committed as the last commit on top)   Date: 2026-10-03

## Tasks

| id | status | commit | one-line result |
|---|---|---|---|
| T3.1 | DONE | `8c997a2` | `cone_area.py`: closed-form area of solid ∩ disk ∩ wedge (Q3b geometry); 5 new tests; worst (b) 2.955e-04 H² (the brute grid dominates); pre-work full suite 709 |
| T3.2 | DONE | `e4b517f` | `cone_area_scene` for the SurfaceRep bodies of a `Scene`; 6 new tests (11 with T3.1); full suite 720, no regression from the torch-tensor `cone_area` extension |
| T3.3 | DONE | `25d6202` | `coneExact` switch in the free-surface detector (default off); detcmp zero disagreements (dambreak 3.413e-03→1.022e-02); `coneExact=true` all bit-level + gate PASS; `cover+cone` gate PASS (two bit-level lines FAIL) |
| T3.4 | DONE | `8095902` | `tensile_vector_scene` (Q2, Wendland C2); 5 new tests; scene vs numpy 6.47e-08, vs grid 3.27e-06; full suite 726 (the registered `w2p5` breaks nothing) |
| T3.5 | DONE | `58b5743` | `tensileExact` switch in `shift` (Wendland C2); 2 new tests (polar vs exact rel 1.389e-2, shift effect 3.60e-02); `tensileExact=true` and all-three gate PASS; default OVERALL: PASS |

## Acceptance evidence

All logs in `.tmp/` (git-ignored); every run in the conda env `warp`, on `cuda:0`, one GPU job at a time.

**T3.1** (log `.tmp/edge_full_start.log` + `test_cone_area.py`):
```
$ python -m pytest tests/edge/test_cone_area.py -q
5 passed in 6.94s
$ python -m pytest tests/edge -q --ignore=tests/edge/test_cone_area.py        # pre-work baseline
709 passed, 14 warnings in 405.72s (0:06:45)
```
Worst errors with margin: (b) vs the independent brute-force midpoint grid (10 rows + 120 random, seed 11) = 2.955e-04 H² (tol 1e-3 H², ~3.4× under; matches the reviewer's measured brute-grid error 2.95e-4 H² → the closed form is exact, the grid is the dominant term); (c) `cone_area` vs `cone_area_scalar` (300 random) = 1.611e-15 H² (tol 1e-11 H², ~6 orders); (d) the 4 degenerate cases = 4.441e-16 H² (machine precision).

**T3.2** (log `.tmp/edge_full_t32.log`):
```
$ python -m pytest tests/edge/test_cone_area_scene.py tests/edge/test_cone_area.py -q
11 passed, 14 warnings in 11.69s        # 6 new T3.2 + 5 T3.1
$ python -m pytest tests/edge -q
720 passed, 14 warnings in 415.21s (0:06:55)        # = 709 + 5 + 6
```
The `cone_area` extension to accept torch-tensor vertices/edges caused no regression (the 5 T3.1 tests still pass).

**T3.3** (logs `.tmp/t33_cone_test.log`, `.tmp/t33_79.log`, `.tmp/detcmp_*.log`, `.tmp/regress_cone_dambreak.log`, `.tmp/regress_cover_cone_dambreak.log`):
```
$ python -m pytest tests/edge/test_deltasph_cone.py -q
1 passed, 14 warnings in 2.86s
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 166.04s (0:02:46)
$ cd python && python -m edgebound.deltasph_detcmp --exact both        # dambreak, steps 0,500,1500,3000,5000
surface flags quad = exact, disagree = 0 at every snapshot;  max|Cw_q-Cw_e|/(n_w H^2): 3.413e-03 -> 1.022e-02;  findings: none
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg coneExact=true
PHYSICS GATE: PASS        (exit 0; tank all bit-level + gate PASS; dambreak all bit-level PASS, margin <= 0.091, + gate PASS)
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg coverExact=true,coneExact=true
PHYSICS GATE: PASS        (exit 0; dambreak bit-level maxVelocityMax FAIL margin 6382.5, maxDensityMax FAIL margin 621.8; every gate line PASS; KE rel max 6.9518e-07)
```
The `--exact both` detcmp on the tank gives the same zero-disagreement result (3.413e-03 → 7.989e-03).

**T3.4** (logs `.tmp/t34_test.log`, `.tmp/t34_margins.out`, `.tmp/edge_full_t34.log`):
```
$ python -m pytest tests/edge/test_tensile_scene.py -q
5 passed, 14 warnings in 3.04s
$ python -m pytest tests/edge -q --ignore=tests/edge/test_deltasph_tensile.py        # T3.5 draft excluded (cfg.tensileExact not yet present)
726 passed, 14 warnings in 433.56s (0:07:13)        # = 709 + 5 + 6 + 1 + 5
```
Worst errors with margin: (a) the rotated L-shape, scene vs the independent numpy `w2p5` gradient = 6.4723e-08 relative (tol 1e-6, ~15× under); (b) the half-plane, scene vs the plain-numpy 2000×2000 midpoint grid = 3.269e-06 relative (tol 5e-5, ~15× under), and the absolute `T_y` hits the reviewer's smoke values to rel ≤ 9.02e-10.

**T3.5** (logs `.tmp/t35_test.log`, `.tmp/t35_79.log`, `.tmp/t35_b_numbers.py`, `.tmp/t35_regress_*.log`, `.tmp/edge_full_final.log`):
```
$ python -m pytest tests/edge/test_deltasph_tensile.py -q
2 passed, 14 warnings in 2.62s
$ python -m pytest tests/edge/test_deltasph.py tests/edge/test_scene.py tests/edge/test_dfsph.py tests/edge/test_cover_scene.py tests/edge/test_deltasph_cover.py -q
79 passed, 14 warnings in 117.74s (0:01:57)
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg tensileExact=true
PHYSICS GATE: PASS        (exit 0; tank all bit-level + gate PASS; dambreak bit-level ke_tstar1/2/2.5, maxVelocityMax, minDensityMin, maxDensityMax FAIL; every gate line PASS; KE rel max 2.0882e-05)
$ cd python && python -m edgebound.deltasph_regress check --physics --cases tank,dambreak --cfg coverExact=true,coneExact=true,tensileExact=true
PHYSICS GATE: PASS        (exit 0; tank all PASS; dambreak the same bit-level FAILs; every gate line PASS; KE rel max 2.4096e-05)
$ cd python && python -m edgebound.deltasph_regress check --cases tank,dambreak        # default config, all switches off
OVERALL: PASS             (exit 0; tank margin <= 0.002, dambreak margin <= 0.078)
```
Test (b), polar quadrature vs the exact edge reduction (probe `.tmp/t35_b_numbers.py`): max|T_exact| = 2.496e+07 (bound > 1e6), max|T_quad−T_exact|/max|T_exact| = 1.3885e-02 (bound 3e-2, ~2.2× under; the reviewer measured 1.389e-2), sign control max|T_quad+T_exact|/max|T_exact| = 2.0137 (bound > 1.0; the reviewer measured 2.01). Test (c): 3.5959e-02 (bound 5e-3–0.15; the reviewer measured 3.59e-2).

**Definition of done — final full suite, no ignores** (log `.tmp/edge_full_final.log`):
```
$ python -m pytest tests/edge -q
728 passed, 14 warnings in 370.52s (0:06:10)        # = 709 + 5 + 6 + 1 + 5 + 2
```
And the default-config `check --cases tank,dambreak` above is `OVERALL: PASS` after the last solver edit (T3.5).

## Numbers

| quantity | value | tolerance / bound | margin | file/command |
|---|---|---|---|---|
| T3.1 worst \|cone_area − brute\| (b) | 2.955e-04 H² | ≤ 1e-3 H² | ~3.4× under | `test_cone_area.py` (LOG-003 T3.1) |
| T3.1 worst vec vs scalar (c) | 1.611e-15 H² | ≤ 1e-11 H² | ~6 orders | `test_cone_area.py` (LOG-003 T3.1) |
| T3.1 worst degenerate (d) | 4.441e-16 H² | ≤ 1e-3 H² | machine precision | `test_cone_area.py` (LOG-003 T3.1) |
| T3.2 full suite | 720 passed | = 709 + 5 + 6 | no count drift | `.tmp/edge_full_t32.log` |
| T3.3 detcmp dambreak max\|Cw_q−Cw_e\|/n_wH² @ step 5000 | 1.022e-02 | 0 disagreements | grows from 3.413e-03 | `.tmp/detcmp_both_dambreak.log` |
| T3.3 detcmp tank max\|Cw_q−Cw_e\|/n_wH² @ step 5000 | 7.989e-03 | 0 disagreements | grows from 3.413e-03 | `.tmp/detcmp_both_tank.log` |
| T3.3 `coneExact=true` gate | all PASS | exit 0 | dambreak bit-level margin ≤ 0.091 | `.tmp/regress_cone_dambreak.log` |
| T3.3 `cover+cone` gate | gate PASS | exit 0 | bit-level maxVelocityMax FAIL 6382.5 / maxDensityMax FAIL 621.8; KE rel 6.9518e-07 | `.tmp/regress_cover_cone_dambreak.log` |
| T3.4 scene vs numpy `w2p5` gradient (a) | 6.4723e-08 rel | ≤ 1e-6 | ~15× under | `.tmp/t34_margins.out` |
| T3.4 scene vs 2000² grid (b) | 3.269e-06 rel | ≤ 5e-5 | ~15× under | `.tmp/t34_margins.out` |
| T3.4 full suite (T3.5 draft ignored) | 726 passed | = 709+5+6+1+5 | no count drift | `.tmp/edge_full_t34.log` |
| T3.5 test (b) max\|T_quad−T_exact\|/max\|T_exact\| | 1.3885e-02 | ≤ 3e-2 | ~2.2× under (reviewer 1.389e-2) | `.tmp/t35_b_numbers.py` |
| T3.5 test (b) max\|T_exact\| | 2.496e+07 | > 1e6 | — (reviewer 2.5e7) | `.tmp/t35_b_numbers.py` |
| T3.5 test (b) sign control ratio | 2.0137 | > 1.0 | — (reviewer 2.01) | `.tmp/t35_b_numbers.py` |
| T3.5 test (c) shift effect | 3.5959e-02 | 5e-3–0.15 | — (reviewer 3.59e-2) | `.tmp/t35_test.log` |
| T3.5 `tensileExact=true` gate | gate PASS | exit 0 | dambreak bit-level KE/maxVelocityMax/density FAIL; KE rel 2.0882e-05 | `.tmp/t35_regress_tensile.log` |
| T3.5 all-three gate | gate PASS | exit 0 | KE rel 2.4096e-05; same bit-level FAIL pattern | `.tmp/t35_regress_all.log` |
| T3.5 default config | OVERALL: PASS | exit 0 | tank margin ≤ 0.002, dambreak ≤ 0.078 | `.tmp/t35_regress_default.log` |
| FINAL full suite (no ignores) | 728 passed | = 709+5+6+1+5+2 | no count drift | `.tmp/edge_full_final.log` |

## Deviations

- **T3.4 full-suite run used `--ignore=tests/edge/test_deltasph_tensile.py`** because that T3.5 draft references `cfg.tensileExact`, which did not exist until the T3.5 edit. The Definition-of-done full suite (above, `.tmp/edge_full_final.log`) has no ignore and includes it → 728.
- **T3.4 test (c)** compares `1.01·T` and `−T` against the scene value `T` (which agrees with the (b) grid to 3.27e-06); the work document says "the grid value of (b)" — the two agree to 3.27e-06, so the difference is the same to well within the 1e-3 bound.
- **T3.3 check #4** (`--cfg coneExact=true` only) expected "bit-level FAIL + gate PASS"; the actual is all bit-level PASS + gate PASS (the `coneExact`-only effect on the dambreak dynamics is below the bit-level tolerance, ~20× the GPU reduction-order spread). Benign — the gate PASSES.
- No switch is made default-on (all stay default-off per the work document). No gate FAIL occurred, so no findings require fixes.

## Failures and open questions

- **No blocking failure.** Every test and every physics gate PASSES. The bit-level FAILs in the dambreak runs with the exact switches ON (T3.3 `cover+cone`; T3.5 `tensileExact` and all-three) are expected: each switch changes a wall integral, which moves the dambreak dynamics past the bit-level tolerance (the gate's tolerance is 20× the GPU reduction-order spread; the bit-level tolerance is far tighter). Every physics-gate line PASSES in all of them (KE rel max ≤ 2.41e-05 ≤ 0.05, P1 arrival ≤ 0.0003 ≤ 0.05, density within 0.97/1.03, steps drift 0). These are observations, not failures.
- **Open (not a blocker):** `tensileExact` is Wendland C2 only — a Wendland C4 tank built with `tensileExact=True` raises `NotImplementedError` from `shift` (test (e)); the C4 W⁵ needs the Chebyshev plan (out of scope). The `w2p5` kernel is registered in `kernels.KERNELS` on first use (idempotent); no test or source iterates that dict, so it adds no parametrised cases (confirmed by the 728 count).

## Files changed

Against the starting commit `9b640a7` (REVIEW-002; the five task commits; this report commit adds only `REPORT-003.md`):
```
$ git diff --stat 9b640a7 58b5743
 docs/deltasph-porting-notes.md      |   2 +
 docs/work/logs/LOG-003.md           | 471 ++++++++++++++++++++++++++++++++++
 python/edgebound/cone_area.py       | 154 +++++++++++++++++++++++
 python/edgebound/deltasph2d.py      |  26 +++--
 python/edgebound/deltasph_detcmp.py |  36 ++++--
 python/edgebound/tensile.py         |  81 +++++++++++++
 tests/edge/test_cone_area.py        | 158 ++++++++++++++++++++++++
 tests/edge/test_cone_area_scene.py  | 166 +++++++++++++++++++++++++
 tests/edge/test_deltasph_cone.py    |  87 +++++++++++++
 tests/edge/test_deltasph_tensile.py |  89 +++++++++++++
 tests/edge/test_tensile_scene.py    | 145 +++++++++++++++++++++
 11 files changed, 1398 insertions(+), 17 deletions(-)
```
No changes to `scene.py`, `warpbc.py`, `kernels.py`, `np2d.py`, `cover.py`, `deltasph_regress.py`, any existing test, `regress_baseline.json`, or anything in `results/`. Scratch in `.tmp/` (git-ignored).

## What I did NOT verify

- **Sloshing** regression (out of scope per WORK-003; the sloshing case is not among the listed checks).
- **The Q1 (Laplacian) wall term** — WORK-003 covers Q2 (tensile) and Q3b (cone count); Q1 stays polar quadrature.
- **The C4 (Wendland) tensile exact reduction** — needs the Chebyshev plan; `tensileExact` raises for C4 (documented above).
- **Making any switch default-on** — out of scope; all three (`coverExact`, `coneExact`, `tensileExact`) stay default-off.
- **Absolute wall-clock times** — the GPU is shared with two local-LLM processes (~63 GB held); margins and ratios, not seconds, are the portable quantities (e.g. the dambreak run took 511 s here vs ~717 s in a less-contended session).
- **The warpSPH port** — `~/dev/warpSPH*` untouched (read-only per KICKOFF rule 3).
