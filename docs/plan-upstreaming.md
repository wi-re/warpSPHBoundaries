# Upstreaming the analytic-boundary series into warpSPH (2026-10-09)

Goal: the eight scratch patches in `patches/warpsph/` become commits on warpSPH `dev` (the repo's working branch, merged into `main` periodically; `CLAUDE.md` rule 5) that its maintainers can review and that the suite, `check_imports.py` and the sweep accept.  Nothing here edits `~/dev/warpSPH` (live working tree, env rule): all trials run in `.tmp/warpSPH-clone` with `PYTHONPATH`.

Written for: the repo owner deciding the packaging question and the order of work.

## 1. Facts established today

| item | result |
|---|---|
| upstream drift | `dev` is 5 commits past the patch base `75c6254` (27e32f3); all Godunov / AV plan work (`gsph`, `diffusionParameters`, `enumTypes`, `schemes/builder.py`, `schemes/__init__.py`, docs); none touches the files of the series |
| rebase | all 8 patches rebase onto 27e32f3 **without conflicts** (`.tmp/warpSPH-clone`, branch `trial-rebase`); 39 files, +2650 / -13 |
| suite on the rebased tree | see s.6 (run `.tmp/rebase_suite.log`) |
| size of the series | 0001 148 lines, 0002 90, 0003 351, 0004 1261 (four generated `wp_*` kernel modules are 1023 of them), 0005 142, 0006 147, 0007 3, 0008 553 (18 files; smaller than the README's wording suggested) |
| who reviews | the last 200 warpSPH commits are all by the owner (checked for warpSPH only; the other stack repos were not looked at); "maintainers" = the owner plus the live work of other sessions, so the practical risk is drift and the repo's own process rules, not an outside review |

## 2. What upstream requires of a change (CONTRIBUTING.md, CLAUDE.md, PLANS.md)

1. `scripts/run_tests.sh`, `scripts/check_imports.py` (runtime pass imports every module; the static pass AST-scans every `warpSPH*` import including lazy ones and checks module **and symbol**), `scripts/run_sweep.py --cases dambreak ...` for a bigger change.
2. A gradcheck script per kernel family: `@wp.kernel` / `@wp.func` changes need `scripts/gradcheck_*.py` (`tests/test_gradcheck_scripts.py` globs them).  The series adds four kernel modules (`wp_barecascoCone`, `wp_barecascoCover`, `wp_deltaShiftRaw`, `wp_minNeighbourNormalDot`); the existing `gradcheck_surfaceDetection.py` / `gradcheck_deltaShift.py` are the templates.
3. `PLANS.md`: one row per plan, *Last worked* / *Where it stands* updated in the same change; new problems go to an existing plan or `OPEN_PROBLEMS.md` first.  So the series needs `ANALYTIC_BOUNDARIES_PLAN.md` (+ row) and `OPEN_PROBLEMS.md` entries.  The *Focus* line is the owner's: suggest, don't set.
4. Validation runs: video on, progress streaming, one GPU run at a time (CLAUDE.md "Running simulations"): applies to the dam-break comparisons we cite.
5. Notebooks need `nbstripout` (none in the series).

## 3. The packaging question (the one real decision)

What the patched warpSPH takes from `warpSPHBoundaries`, measured on the rebased tree:

* **Protocol + adapter** (`boundary/provider.py`): defined in warpSPH, `warpSPHBoundaries` imported lazily in `buildBoundaryProvider`.  As designed.
* **Region / case builders** (`caseUtils/weaklyCompressible.py`, `cases/dambreak.py`): construct `Body`, `BoxRep`, `DiskBody`, `ImplicitRep`, `SurfaceRep`, `Scene` directly (lazy imports, `wallRepresentation='analytic'` only).  Acceptable, but it makes the dam-break options hard dependants of the package.
* **Package helper functions used by the wall terms**:
  * `scene.tensile.tensile_factor`, `scene.viscosity.lap_factor`: the package's scene layer, legitimate API;
  * **`sim.pairs.wendland2` and `sim.bcclosures.wallVelocity`**: from the package's *solver* layer (`sim/`), which itself imports warpSPH.  This contradicts the recorded design ("warpSPH never needs to import the package for the particle flavour; the package depends on warpSPHCore only") in spirit and creates a dependency cycle at the distribution level (warpSPH -> package `sim` -> warpSPH).  The two helpers are 3 and 15 lines.
* **Tests** `import warpSPHBoundaries.sim.deltasph2d` as the oracle (`test_analyticWallTerms`, `test_analyticShifting`): an oracle living in the package, run from the warpSPH suite.

### Options

| | A. separate package repo, optional extra (**recommended**) | B. vendor the package into warpSPH | C. keep the patch series out of tree |
|---|---|---|---|
| shape | `warpSPHBoundaries` becomes the 5th repo of the stack next to Core / Integrators / Plotting; warpSPH gets `[project.optional-dependencies] boundaries = ["warpSPHBoundaries>=0.1"]`; hooks stay guarded and lazy | `edge/`, `scene/`, tier tables, Maple / derivations move into `warpSPH/src` | patches stay here, applied by hand |
| fits the 2026-10-06 decision | yes | no (re-opens it; drags Maple, paper and 2D derivations into warpSPH) | no (rots against a live dev branch) |
| cost | package needs a public home and a release; two repos to keep in step | huge diff, review burden, build-time of tier tables | none now, grows with every upstream commit |
| warpSPH without the package | unchanged (guards, `importorskip`) | n/a | unchanged |

**Recommendation: A**, with these preconditions (the work of this plan, s.5):

1. warpSPH does not import `warpSPHBoundaries.sim.*`.  `wendland2` is replaced by warpSPHCore's kernel; `wallVelocity` becomes warpSPH's own BC-policy closure (it belongs to the scheme, "scheme owns BC physics").  After that, warpSPH imports only `warpSPHBoundaries.scene` (+ the `tensile` / `viscosity` helpers there).
2. The oracle tests that need `DeltaSPH2D` move to this repo (they compare *two implementations of one scheme*, which is the package's validation job); warpSPH keeps tests that need only the package's scene layer and compare against warpSPH's own boundary-particle path (dam break analytic vs particle walls, the regression we already have).  Net: warpSPH's suite has no dependency on `sim/`.
3. The package is installable on its own: `pyproject.toml` already lists it with a `solvers` extra for warpSPH; check that a clean venv install works without warpSPH (plan-next-steps item 9, wheel test) and that tier tables ship as package data (done).
4. `check_imports.py` static pass needs the package importable in the dev env (it is, editable) or an allow-list entry; documented in CONTRIBUTING under "Setting up".

## 4. Patch series restructure

Target (each commit passes the suite on its own, as `git rebase --exec` will verify):

| new | from | content |
|---|---|---|
| U1 | 0001 | `RigidBody.representation`, `buildAnalyticRigidBody` |
| U2 | 0002 | analytic regions + `dambreak --wallRepresentation analytic` |
| U3 | 0003 | provider protocol + adapter, wall terms, hooks 05b / 11 / 12 / 13 |
| U4 | 0004 | split Barecasco detector + wall continuum (generated `wp_*` modules + `gradcheck_analyticBoundary.py`; **no `wp_minNeighbourNormalDot`**: dropped, s.6) |
| U5 | 0005 | shifting with walls |
| U6 | 0006 + 0007 | no-penetration impulse, `drift_rates` wall flux, `analyticWallPressure`, RHS graph refusal folded into the graph commit (0007 is superseded) |
| U7 | 0008 part | Michel 2022 and implicit / dynamic shifting with walls |
| U8 | 0008 part | wall loads, `RigidBody.dynamic` / `load` |
| U9 | 0008 part | obstacles and mixed scenes in the dam break |
| U10 | 0008 part | whole-step CUDA graph with analytic walls |
| U11 | new | `ANALYTIC_BOUNDARIES_PLAN.md`, PLANS.md row, OPEN_PROBLEMS entries, README / CONTRIBUTING lines, `boundaries` extra |

Independent of the series, as separate OPEN_PROBLEMS entries (or small fixes only if the owner wants them in the same change): README findings 2 (`_curvatureGate` over the raw Verlet list), 3 (`computeDeltaShiftWarp` `W0` at `dx/kernelScale`), 6 (implicit shifting sums ghost nodes), 7 (particle rigid bodies not graph-capturable), 8 (failed capture poisons the CUDA context), 9 (graph re-execution advances shared body tensors).  Finding 1 (`drift_rates`) is part of U6 itself.

## 5. Work order

| step | what | needs |
|---|---|---|
| 0 | this plan; decision A / B / C | owner |
| 1 | decouple: drop `sim.*` imports from the series (own `wendland2` / `wallVelocity`), move the three oracle tests here (**done in the scratch clone, 428c2c3; verification in s.6**) | A |
| 2 | gradcheck script for the new kernel modules (**done in the clone, d8a4d2d**: `scripts/gradcheck_analyticBoundary.py`; cover and shift-raw gradchecked, cone count a value check against a brute force, min-dot module dropped, see s.6) | – |
| 3 | restructure to U1-U10 on top of upstream `dev` in the scratch clone (**done: branch `upstream-series` in the scratch clone (worktree `.tmp/wt-series`); U1-U10 tree identical to the verified tip `d8a4d2d`; every commit passes its tests, s.6**) | 1, 2 |
| 4 | U11: `ANALYTIC_BOUNDARIES_PLAN.md`, PLANS.md row + decisions-log line, OPEN_PROBLEMS 24-29, CONTRIBUTING / README lines, `boundaries` extra, `check_imports.py` optional-package exception (**done: commit U11**) | 3 |
| 5 | gates on the final tree: `run_tests.sh`, `check_imports.py` (package installed and not installed), `run_sweep.py --cases dambreak` (default path) and an analytic dam break with video, gradcheck suite (**done**, except a fresh dam-break run with video; s.6) | 4 |
| 6 | regenerate `patches/warpsph/` as the new series (`git format-patch upstream-dev..`), README updated with the base commit; hand over to the owner to apply on `dev` (**done**: 11 patches, `git am` on a fresh `dev` checkout reproduces the branch exactly) | 5 |
| 7 | package side: repo renamed and public (done, owner); `curvbound` merged into the package, license MIT, `warpSPHCore>=0.6.0`, `__version__`, PyPI metadata, release scripts (done, below); still to do: commit / push, then the owner publishes (`scripts/publish_pypi.sh`) | A, owner |

Scope gate: merge with the **documented refusals** (3D, incompressible / ACSPH consumers, `exactHessian` with walls, moving bodies in the graph, mixed scenes / implicit shifting in the graph) listed in the plan file; closing them is later work, not a precondition.  Reasoning: the particle path is untouched, every hook is guarded, refusals are loud, and each refusal is a separate plan row.

## 6. Verification log

* 2026-10-09: rebase of the 8 patches onto warpSPH `dev` 27e32f3: clean.
* 2026-10-09: full warpSPH suite on the rebased tree (`PYTHONPATH` clone, package editable from the env): result below.

* First full run on the rebased tree (`-x`): 26 tests passed, then **`test_analyticGraph::test_uncapturable_configurations_run_eagerly[implicit]` failed**.  Not caused by the rebase: it fails identically (3 of 3 reruns) on the original series at its old base `c63edee`.  Cause: the test compares two eager runs of the implicit shifting with `atol=1e-6`, but two identical runs differ in the velocity by 1e-5 to 2e-5 (float32, 40 steps, atomics in the implicit solve; positions 2e-7, densities 5e-7).  The code is fine; the tolerance was below the noise.  Fixed in the clone to 5e-4 of the field's scale.
* Step 2: `scripts/gradcheck_analyticBoundary.py` (CPU, float64, picked up by `tests/test_gradcheck_scripts.py`'s glob): `computeBarecascoCoverWarp` (positions, supports) and `computeDeltaShiftRawWarp` (positions, supports, masses, densities; with and without the tensile term) pass `torch.autograd.gradcheck`; `computeBarecascoConeCountWarp` is a discrete count (waived as upstream does for barecasco's second output) and is checked by value against a brute force instead (exact).
* **Finding: `wp_minNeighbourNormalDot` is dead code with a wrong adjoint, dropped from the series.** Nothing in warpSPH calls it (`solveShifting` keeps `_curvatureGate` in torch; the audit table's "new" row was never wired in), and the gradcheck exposed the loop-carried `out = wp.min(out, ...)` shape that `gradcheck_shockCapturing.py` documents: forward exact, adjoint attributed to the first neighbour instead of the minimising one.  The kernel stays in this package (`sim/modules/shifting`).  The fix recipe, if a within-support gate is ever wanted upstream (finding 2), is upstream's own: forward-only argmin, then recompute that one pair's value.  The audit doc's table row (`docs/audit-warpsph-boundary-hooks.md` s.3) still lists it as a new module.
* Step 3: `upstream-series` = U1-U10 built by script from the verified trees (0006 + 0007 squashed; 0008 split by hunk into Michel + implicit shifting, obstacles + mixed scenes, loads + free bodies, graph; the free-body dam-break parameters after the obstacles they need).  The last commit is **identical** to the verified tip `d8a4d2d` (`git diff` empty), no commit has a `warpSPHBoundaries.sim` import, and upstream's static import check passes on it.
* **Finding: `scripts/check_imports.py` treats every `warpSPH*` module as first-party**, so with `warpSPHBoundaries` not installed the static pass would report every lazy import of it as "module not found".  U11 adds an `OPTIONAL_PACKAGES = {"warpSPHBoundaries"}` exception (verified only when installed, skipped imports counted in the summary).
* Step 1 (decoupling) verification: see the next entries.

* Step 1 (decoupling): the three moved oracle tests pass 9 of 9 in float64 and float32 against the decoupled code.
* Full warpSPH suite on the decoupled tree: **1035 passed, 1 skipped, 1 failed**.  The failure, `test_threeWayShiftComparison_allRelaxTowardUniformDensity`, passes alone (3 of 3); it also failed once inside a full-suite run of the old series (it passed alone and on the unpatched base then).  Not reproduced on the unpatched base inside a full suite, so "flaky upstream" is the working assumption, not a measurement.
* Per commit U1-U10 (`tests/test_analytic*.py`, `test_deltaSPHDiffusion.py`, `test_implicitShifting*.py`): 70, 72, 72, 72, 72, 75, 85, 102, 104, 110 passed, none failed; `gradcheck_analyticBoundary.py` exits 0 from U4 on.
* U11: `check_imports.py` runtime + static clean with the package installed; static clean with it hidden (14 optional imports skipped); the U10 version of the script fails the hidden case with exactly those 14 errors.
* Deliverable: `patches/warpsph/` (11 patches, base `27e32f3`), `git am` on a fresh checkout of `dev` gives a tree identical to the branch.
* `scripts/run_sweep.py` (smoke, 5 steps per case) on the final tree: **44 of 44 cases ok**.  This shows the default (particle) path still builds and steps in every case; it does not exercise the analytic path (that is the `tests/test_analytic*.py` set).
* Not run: `run_tests.sh` as a script (it is the same whole-suite pytest run, `-p no:warnings`), the dam break with video as a new validation run (the numbers quoted in the plan file are the earlier runs of the code before the rebase and the decoupling; quoting them anew would need a re-run).

## 7. Decisions and open questions

Decided by the owner 2026-10-09:
* **Packaging A**: `warpSPHBoundaries` stays its own repo and becomes an optional extra of warpSPH.  Once the series has landed locally the owner renames the GitHub repo (`wi-re/curvatureBoundaries` -> `warpSPHBoundaries`) and makes it public, **before** anything is pushed to warpSPH upstream (so the optional extra never points at a private repo).
* Merge granularity: the full U1-U10 series on `dev`, not a first slice.
* The six upstream findings go in as OPEN_PROBLEMS entries only; the series carries no fixes for them.
* The package already has a home: `github.com/wi-re/curvatureBoundaries` (private; the import / distribution name is `warpSPHBoundaries`).  warpSPH is `wi-re/warpSPH` (public).

Done by the owner 2026-10-09: the repo is renamed to `wi-re/warpSPHBoundaries` and public (checked: `gh repo view`; the old `origin` URL redirects), so the URLs in the README / CONTRIBUTING lines of the last patch resolve.

Open: this repo's `main` (bc85c32) does not yet contain `tests/warpsph/`, `docs/plan-upstreaming.md` or the regenerated `patches/warpsph/`; they are uncommitted here.

## 8. PyPI release of warpSPHBoundaries (2026-10-09, prepared, NOT uploaded)

* `curvbound` is now `warpSPHBoundaries.curvbound` (3 modules; no second top-level package in the wheel); its Maple exports moved from `results/symbolic/` to `src/warpSPHBoundaries/data/symbolic/` (package data; `maple/*.mpl`, `run_all.sh` and the current docs point there; `docs/work/*` keep the old names as history).
* License changed Apache-2.0 -> MIT (as the rest of the stack): sole author, no third-party code bundled; copies cloned during the public window under Apache stay Apache.
* `warpSPHCore>=0.6.0` (base dependencies and `solvers` extra): the published 0.6.0 provides all 18 names the package imports (0.5.0 lacked `warpSPHCore.profiling` and `deferVerletChecks`, used by `sim/`). The `solvers` extra's `warpSPH` is unpinned until the owner's new warpSPH release (the published 0.5.0 predates the analytic-boundary series).
* `scripts/publish_pypi.sh` / `setup_pypi_token.sh` copied from warpSPH (version check on `src/warpSPHBoundaries/__init__.py`), plus `--build-only`. `--build-only` run: wheel and sdist (2.8 MB each) pass `twine check`; wheel contains only the package and its 18 data files; `pip install --dry-run` of the wheel resolves against the live PyPI. `MANIFEST.in` keeps tests/docs out of the sdist.
* Tests: this repo's full suite 1075 passed, 1 failed; the failure was a flaw in the new wheel-test assertion (string comparison of mpmath values at different global precisions), fixed (float comparison); the packaging tests pass in the order that triggered it.
* Open: `License ::` classifiers are deprecated by setuptools in favour of an SPDX expression (the stack's other repos use the classifier too, kept for parity); name `warpSPHBoundaries` was free on PyPI on 2026-10-09; first upload is the owner's (token, optional TestPyPI first: `bash scripts/publish_pypi.sh --testpypi`).
