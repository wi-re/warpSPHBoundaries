# Intermediate work item: repository layout and cleanup pass (before `plan-wall-evaluation.md` step 1)

Status: note, 2026-10-05. Not started. Do this **before** the steps of `docs/plan-wall-evaluation.md`: step 2 renames `SceneAdjacency` / `MomentPairs` / `buildAdjacency` everywhere, and doing the file moves first keeps the moves and the renames in separate commits.

## 1. Goal

`python/edgebound/` is 45 flat modules (9 830 lines) mixing the core math, the scene layer, two solvers, validation runners, benchmarks, studies, fixture generators and a video renderer; tests import some of the scripts; the package is not installable (`conftest.py` files insert `python/` into `sys.path`); paths are computed from `__file__` depth or from the current directory. Target:

```
pyproject.toml                  editable install (pip install -e .), pytest config; removes both sys.path hacks
src/edgebound/                  the library: core math, scene layer, solvers (simulators live here)
scripts/                        runners, harness, profiler, videos, studies, benchmarks, fixture generators
tests/                          one tree (the current tests/edge/ plus the curvbound tests), fixtures next to it
docs/  maple/  notebooks/  results/   unchanged
```

Rule for what is library and what is a script: **if a test or another library module imports it, it is library** (or the imported part is split out into the library); a module that is only run (`python -m …`, a `__main__` block, argparse, writes figures/videos/npz) is a script.

## 2. Inventory and proposed destination

| layer | modules | destination |
|---|---|---|
| A. edge machinery (mpmath → numpy → torch → Warp) | `core, primitives, kernels, geometry, oracle, mpq, np2d, np_fem, fem, torch2d, warp2d, warpbc, tier3, tier4, tier_select, implicitBodies` | `src/edgebound/` (sub-package `edge/` or flat; see §5) |
| B. scene layer and wall operators | `scene, boundaryOps, cover, cone_area, tensile, viscosity` | `src/edgebound/scene/` (or flat) |
| C. solvers and cases | `dfsph2d, deltasph2d`, `dfsph_cases`, the case builders inside `deltasph2d.py` (`hydrostatic_tank, english_wedge, marrone_dambreak, sloshing_tank`, probes `mls_pressure, wall_probes, sloshing_probes`) | `src/edgebound/sim/`; the cases and probes move out of `deltasph2d.py` into `sim/cases.py` / `sim/probes.py` |
| D. runners and the regression harness | `deltasph_validation, dfsph_validation, deltasph_regress, deltasph_profile, dfsph_runcase` | `scripts/` (the harness `deltasph_regress` and `deltasph_profile` stay runnable by a stable command, §4) |
| E. tools | `dfsph_video, deltasph_snap, deltasph_compare, dfsph_ref` | `scripts/` (`dfsph_ref` is imported by `test_dfsph.py`: see §3) |
| F. studies, probes, benchmarks | `np2d_study, fem_study, q2_conditioning, precision_probe, warp_bench, scene_bench, boundary_bench` | `scripts/studies/` and `scripts/bench/` |
| G. fixture generators | `fixtures, fem_fixtures` | `scripts/` next to the fixtures they write, but the golden loader used by tests stays in the library/tests (`test_fixtures.py`, `test_np2d.py`, `test_fem_np.py`, `test_warpbc.py` import them) |
| H. repo root | `edge_identity_check.py`, `monomial_edge_forms.py` (the latter has a known wrong `−4σ` block, HANDOFF), `conftest.py` | `scripts/` / `docs/derivations/` as history, or delete after checking nothing cites them; root `conftest.py` goes with the install |
| I. second package | `python/curvbound/` (PLAN.md track, tier 3 / oracle) with `tests/test_kernels.py, test_planar.py, test_sphere.py` | decide: `src/curvbound/` as a second package in the same `pyproject`, or leave outside until PLAN.md resumes. Not mixed into `edgebound` |

## 3. Library ↔ script couplings found (must be cut before the split)

* `tensile.py` imports `terms` from `q2_conditioning` (a "study" module); `test_warpbc_stable.py` imports `ref_coeffs, terms` from it. The truncated-power construction behind `w2p5` / `w4p5` is library code: move `terms` / `ref_coeffs` into the library (next to `kernels.py`), leave the study (plots, tables) as a script importing them.
* `test_warp2d.py` imports `warp_bench`; `test_dfsph.py` imports `dfsph_ref` (the live omniSPH reference) and `dfsph_cases`. Tests may depend on library only: move the shared helper into the library, or into `tests/` helpers, and leave the benchmark / reference runner as a script.
* `deltasph_regress` imports `run_tank, run_dambreak, run_sloshing` from `deltasph_validation`; `dfsph_validation` and `dfsph_runcase` import `dfsph_cases`. The "run a case and score it" functions are shared by the harness and the validation scripts: they become `edgebound/sim/runners.py` (library, importable) and the scripts are thin argparse wrappers.
* `fem_fixtures` imports `fixtures`; both write into `tests/fixtures/` via `Path(__file__).parents[2]`.

## 4. Things that break on a move, and how to keep them working

* **Commands in docs:** `python -m edgebound.deltasph_regress` appears 66 times, `deltasph_profile` 14, plus `deltasph_validation`, `dfsph_*`, `np2d_study`, … in `README`, `HANDOFF`, `docs/*`, `KICKOFF`, WORK/REPORT/LOG files. Decide one stable form (suggestion: `python scripts/regress.py check …`, or console scripts `edgebound-regress`, `edgebound-profile` declared in `pyproject`), update the live docs (`HANDOFF`, `docs/*.md` except `docs/work/logs`, `README`), and leave `docs/work/**` history untouched with one line at the top of `docs/work/README.md` giving the old→new command map. `deltasph_detcmp` in the old docs no longer exists (removed earlier); do not resurrect.
* **Paths:** `deltasph_regress.REPO_ROOT` (three `dirname`s), `fixtures` / `fem_fixtures` (`parents[2]`), `implicitBodies.TABLE_DIR` (`parents[2]/results/tables`, a *library* module writing a cache into `results/`), `deltasph_profile` (`../results/deltasph/…` relative to the current directory), `deltasph_compare` (`.tmp/delta/ref/…` relative), `dfsph_validation` (`__file__/../../results`). One `edgebound/paths.py` (repo root found by walking up to `pyproject.toml`, `RESULTS`, `TABLES`, `FIXTURES`, `TMP`) replaces them; the library must not require the repo checkout to import (tables cache falls back to a user cache dir when `results/` is absent).
* **The baseline** `results/deltasph/regress_baseline.json` and the series `results/deltasph/*.npz` stay where they are (tracked); only the code that reads them moves.
* **Imports inside the tree** are relative (`from .scene import …`); sub-packaging changes them. Keep `edgebound/__init__.py` re-exporting today's public names (`value, gradient, moment, …`) so notebooks and `tests` keep working; add the new locations to it deliberately, not by accident.
* `notebooks/` import `edgebound`: re-run the notebooks (or their `build_edge2d_demo.py`) after the move.
* Uncommitted state in other places: `.tmp/` scripts of the reviewer and local-model sessions import `edgebound.*` modules by flat name; they are scratch, do not migrate them.

## 5. Decisions to make first (cheap, but they fix every path in the move)

1. **Flat or sub-packages inside `src/edgebound/`?** Suggestion: sub-packages `edge/` (layer A), `scene/` (B), `sim/` (C), with `edgebound/__init__.py` re-exporting the current public API. The plan's names (`PairMoments`, `precompute`, `evaluate`) then live in `scene/`.
2. **Where `warpbc.py` goes**: it is the Warp kernel module of the edge machinery and the future home of the fused kernels (`plan-wall-evaluation.md`): `edgebound/edge/warpbc.py` now, a `kernels/` sub-package when step 1 adds the f32 plans.
3. **Entry-point form** for harness / profiler (console scripts vs `python scripts/…`), see §4.
4. **`curvbound`**: second package or parked (§2 I).
5. Whether `tests/` keeps the `edge/` subdirectory (suggestion: flatten into `tests/` with sub-folders by layer: `tests/edge`, `tests/scene`, `tests/sim` mirroring `src`).

## 6. Order of work (each step is one commit and ends green)

0. Baseline: record `pytest tests/edge -q` = **767 passed** and the five-file solver set = 82. This item changes **no result**, so the gate is tiered (user, 2026-10-05; the simulation set is not a per-step gate for renames and moves):
   * **every step:** the full test suite (it imports every module and runs both solvers; ~5 min). A failure is a defect of the move, never a physics matter.
   * **only where a step touches what the suite does not exercise:** the harness `deltasph_regress check --physics --cases tank,dambreak` (all bit-level PASS, margins as in REPORT-008) — after step 3 (`paths.py`: the harness reads the baseline JSON and the series `.npz` through the changed paths; the tests do not) and once at the end. The profiler (`deltasph_profile`, reads `results/deltasph/…`) and the fixture generators (`fixtures`, `fem_fixtures`, run once and `git diff` the JSON: must be empty) get a smoke run after the step that moves them.
   * **not run:** sloshing, wedge, the long validation scripts, videos. They are scripts around the same library calls.
1. `pyproject.toml` + editable install, remove the `sys.path` hacks (both `conftest.py`), nothing moves yet. Suite.
2. Cut the couplings of §3 in place (still flat): split `q2_conditioning` → library + study; move the case runners into one importable module; the `warp_bench` / `dfsph_ref` helpers out of the tests' import path. Suite.
3. `paths.py`; replace every `__file__`-depth and cwd-relative path (§4). Suite + harness.
4. `git mv` into `src/edgebound/…` and `scripts/…` **without editing content** except imports (one commit per layer so `git log --follow` and blame survive; no formatting changes in a move commit). Suite after each; smoke runs of the moved scripts (`--help` or a 10-step case).
5. Update the live docs and the command map; re-run the notebooks; delete or archive the root scripts (§2 H).
6. Cleanup pass (see §7), one theme per commit.

## 7. General cleanup pass (behaviour-preserving, after the moves; nothing here changes a result)

Seen while surveying; to be verified when the file is open, not assumed:
* `scene.py` (939 lines) holds poses, cell lists, four representation classes, `Body`, the adjacency, the operation dispatch and `Scene`. **Do not split it before `plan-wall-evaluation.md` step 2**: that step rewrites exactly the adjacency / precompute / operation part. After it: `scene/{pose,bodies,representations,adjacency,operations}.py`.
* `deltasph2d.py` (669 lines): the config dataclass, the solver, the Wendland helpers (`wendland4`, `dwendland4`: duplicate of `kernels.py`?), the case builders, the probes. After §2 C it is the solver and the config only.
* Duplicated pair-sum and kernel helpers between `dfsph2d.py` and `deltasph2d.py` (`neighbor_pairs` lives in `dfsph2d` and is imported by the δ solver): move to `sim/neighbors.py`; these are what phase 3 replaces with warpSPH modules, so keep them isolated, do not polish them.
* Private-name imports across modules (`from .scene import _segment_distance`, tests importing `_c2`, `_w_hat_mp`): either make them public or move them next to their users.
* Docstring headers that give run commands with `cd python && …` or `cd .tmp/omni`: replaced by the new command form; `.tmp/` paths in scripts (`deltasph_compare`, `dfsph_validation`) become arguments with defaults from `paths.py`.
* `tests/`: the flat `tests/edge/test_deltasph_*` files (cone, cover, exact_only, noslip, tensile, tensile_c4, viscosity) are per-WORK files; group by topic only if it can be done without editing a test body.
* Dead code: grep for functions with no caller (`git grep -n` per public name) before deleting; the polar quadrature path was already removed in WORK-006, so remaining candidates are in the studies, not the solver.
* Type hints / formatting: no mass reformat in the same commit as anything else; if wanted, one `ruff format` commit last, listed in `.git-blame-ignore-revs`.

## 8. Definition of done

`pip install -e .` in the `warp` env, `pytest` from the repo root = 767 passed (+ the `curvbound` tests if that package is kept), the harness command from §4 reproduces all bit-level PASS lines with the recorded margins (final run), `python -c "import edgebound"` works from any directory, no module in `src/` imports from `scripts/`, no test imports from `scripts/`, the live docs contain no stale command, `docs/work/README.md` has the command map, and `plan-wall-evaluation.md` step 1 can start from `src/edgebound/edge/warpbc.py`.
