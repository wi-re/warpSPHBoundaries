# warpSPH patch series: analytic boundaries (not pushed)

Eleven commits on warpSPH `dev` at `27e32f3` that let the weakly compressible delta+ scheme run with ANALYTIC boundaries (this package's `warpSPHBoundaries`: exact kernel integrals over the
solid) instead of boundary particles. The particle path is untouched: every hook is guarded by `schemeConfig.boundaryProvider is not None`. This is the restructured series prepared for
upstreaming (`docs/plan-upstreaming.md`); it replaces the eight scratch patches on `75c6254` (git history: b203aa2, bc85c32). Design and audit: `docs/audit-warpsph-boundary-hooks.md`.

| patch | content |
|---|---|
| 0001 | `RigidBody.representation`, `buildAnalyticRigidBody` (mass / inertia from `Body.massProperties`; infinite for an unbounded wall) |
| 0002 | `ParticleRegion.representation`, `buildRegion(representation=)` (not sampled, no ghost layer), the initializer builds the bodies and the provider; `dambreak --wallRepresentation analytic` |
| 0003 | `boundary/` (provider protocol + adapter + body binding), `modules/analyticBoundary/wallTerms.py`: wall continuity, pressure force, viscosity; hooks at stages 05b / 11 (inside the frozen-diffusion tuple) / 12 / 13 |
| 0004 | surface detection: the Barecasco passes as separate partial-sum modules (`wp_barecascoCover`, `wp_barecascoCone`), the wall continuum added between them, normals / lMin from the fluid + wall matrix; `wp_deltaShiftRaw`; `scripts/gradcheck_analyticBoundary.py` |
| 0005 | shifting: the wall's share of the raw sum in `computeDeltaShift` (warpSPH's W0 convention); `solveShifting` runs unchanged |
| 0006 | no-penetration impulse from the provider's signed distance; the wall flux in the system's time-centred continuity closure (`drift_rates`); `analyticWallPressure='normal'`; dam break regression; the RHS CUDA graph refuses providers (until 0010) |
| 0007 | Michel 2022 and implicit / dynamic shifting with walls (wall share of grad C-tilde, `U_char` from the provider's `dir_extreme`, grad C of the implicit solve; `shiftScheme` / `shiftProjection`) |
| 0008 | obstacles (circle, box, triangle) and mixed scenes (analytic tank + particle obstacle) in the dam break |
| 0009 | wall loads (`wallLoads`, `RigidBody.load`) and free analytic bodies (`RigidBody.dynamic`; `obstacleDynamic`) |
| 0010 | the whole-step CUDA graph with analytic walls (host constants `_analyticSupport` / `_analyticMass` / `_analyticDx` fixed at initialisation; moving or dynamic bodies, implicit shifting and mixed scenes refused up front) |
| 0011 | `ANALYTIC_BOUNDARIES_PLAN.md`, `PLANS.md` row and decisions log, `OPEN_PROBLEMS.md` 24-29, optional `boundaries` extra, `check_imports.py` treats `warpSPHBoundaries` as optional, README / CONTRIBUTING lines |

Not in the series (compared with the scratch series): `wp_minNeighbourNormalDot` (nothing called it; its adjoint was wrong), and warpSPH no longer imports `warpSPHBoundaries.sim.*` (two small helpers are local to
`modules/analyticBoundary/shifting.py`). The tests that compare against the reference solver `DeltaSPH2D` live here: `tests/warpsph/`.

## Apply and test (without touching the shared environment)

```
git clone <warpSPH> /some/scratch/warpSPH && cd /some/scratch/warpSPH
git checkout -b analytic-boundaries 27e32f3
git am /path/to/this/repo/patches/warpsph/*.patch
PYTHONPATH=$PWD/src ~/miniconda3/envs/warp/bin/python -m pytest tests -q           # warpSPHBoundaries comes from the environment (editable)
cd /path/to/this/repo
PYTHONPATH=/some/scratch/warpSPH/src python -m pytest tests/warpsph -q             # the reference-solver comparisons (float64)
warpSPHCore_PRECISION=float32 PYTHONPATH=/some/scratch/warpSPH/src python -m pytest tests/warpsph -q    # and float32
```
The patched tree is used only through `PYTHONPATH`; the `warp` environment's editable warpSPH (the live working tree) is never reinstalled or modified. `tests/test_analytic*.py` skip when
`warpSPHBoundaries` is not importable.

## What the tests establish

Reference-solver comparisons (this repo, `tests/warpsph/`, 9 tests, float64 and float32):
* wall terms (continuity, pressure with the p_b >= 0 clamp, viscosity; free-slip and no-slip) = `DeltaSPH2D.rhs` wall contribution, 1e-13 in double, float32 round-off in the state's precision; wall loads = `DeltaSPH2D.loadsAt`;
* detector: raw and dilated masks equal exactly, lMin and the normals to float32 round-off, regular and jittered;
* shifting: `solveShifting` with analytic walls = `DeltaSPH2D.shift` to 1.4e-4 relative (float32) with the curvature gate off.

warpSPH's own suite (`tests/test_analytic*.py`, boundary-particle path as the reference):
* dam break 0.5 s, nx 64: analytic against particle walls: max velocity 5.340 / 5.345, kinetic energy 3.567 / 3.712 (4 %), density bounds equal, no penetration; the reference solver reproduces the analytic run exactly (max velocity 2.929, density 1.0000 to 1.0055 at 0.15 s);
* Michel: the wall share of grad C-tilde equals the boundary-particle sum to 3 % (max, float32, lattice quadrature), `U_char` is the supremum of the lattice maximum; dam break at 0.3 s against particle walls 1.9 % (fastest particle) / 4.8 % (kinetic energy);
* implicit / dynamic: the solved shift equals the particle-wall path (ghost nodes excluded) to 2.8 %; dam break against the analytic delta+ run 1.1 % / 0.04 %;
* loads: a free cylinder of density 0.5 (2.0) accelerates at 3.0 (float64) / 2.5 (float32) m/s^2 against the ideal 3.27 (added-mass limit); the float32 load of a buoyant body is 3 % lower than the float64 one;
* obstacles / mixed: kinetic energy within 3-4 % of the boundary-particle obstacle, fastest particle 6-11 % lower (a spray particle), mixed scene stable, KE within 1.1 %;
* graph: the whole step is bitwise the eager one for the tank, the tank with an obstacle and Michel shifting (about 3x faster).

## Verification of this series (2026-10-09)

* Rebased onto `dev` 27e32f3 without conflicts; the last tree is identical to the tree verified as a whole.
* Every commit 0001-0010 passes `tests/test_analytic*.py`, `test_deltaSPHDiffusion.py` and `test_implicitShifting*.py` (70, 72, 72, 72, 72, 75, 85, 102, 104, 110 tests; no failure), and `scripts/gradcheck_analyticBoundary.py` from 0004 on.
* Whole warpSPH suite on the decoupled tree: 1035 passed, 1 skipped, 1 failed: `test_threeWayShiftComparison_allRelaxTowardUniformDensity`, which passes alone (3 of 3); the same test failed once inside a full-suite run of the old series and passed alone and on the unpatched base then (not reproduced on the unpatched base inside a full suite).
* `scripts/run_sweep.py` smoke sweep on the final tree: 44 of 44 cases ok (default path only).
* `scripts/check_imports.py`: runtime and static pass clean with the package installed; the static pass is clean with it absent (14 optional imports skipped); without the 0011 exception it reports those 14 as missing.

## Findings during the port (now OPEN_PROBLEMS 24-29 in 0011, unless noted)

1. **`WeaklyCompressibleSystem.drift_rates`** (time-centred continuity, default on) re-evaluates the continuity rate for the mean velocity from the pair modules only. Any wall treatment that adds its flux in the right-hand side must add it there too, or the density at the wall is advanced without the wall's compression (the analytic dam break diverged within 0.06 s until it did). Part of 0006, documented in the plan file.
2. **`_curvatureGate`** (shifting, Sun 2019 Eq. 21) takes the minimum normal dot product over the RAW Verlet list, not the pairs inside the support: OPEN_PROBLEMS 24.
3. **`computeDeltaShiftWarp`**'s tensile reference value is the kernel at dx / kernelScale, not at dx (3.7 % of the tensile term for Wendland C2 at h = 4 dx): 25.
4. The analytic hydrostatic wall term assumes hydrostatic balance next to the wall (its tangential part pushes a free-falling wall column up by 3.15 m/s^2 at the dam break's uniform-density start); the boundary particles extrapolate along the normal only. `analyticWallPressure='normal'` removes it. Harmless dynamically here; documented in the plan file.
5. Library schemes other than the symplectic Euler read host time and cannot be CUDA-graph captured (separate repo, warpSPHIntegrators).
6. `computeImplicitShift` sums the mDBC ghost nodes into grad C: the boundary-particle dam break with implicit shifting diverges (fastest particle ~900 at 0.3 s) while the analytic one is stable: 26.
7. Particle rigid bodies are not graph-capturable (`getTransformationMatrix`, `updateBodyParticlesWCSPH` read the host): 27.
8. A failed capture of the implicit shifting leaves the CUDA context in an error state; the eager fallback then crashes (`_rhsIsGraphable` now refuses implicit / dynamic shifting): 28.
9. The graph's validation and re-capture execute the step more than once on a cloned state, but rigid-body tensors are shared: a moving body is advanced extra times; only static bodies are graphed: 29.
10. `scripts/check_imports.py` treats every `warpSPH*` module as first-party: fixed in 0011 (optional package exception).
11. Test tolerance: two identical eager runs of the implicit shifting differ in the velocity by 1e-5 to 2e-5 in float32 (atomics); the graph test compares at 5e-4 of the field's scale.

## Not done

Mixed-scene equivalence beyond the dam break; the incompressible / ACSPH consumers (audit s.6 table); 3D; the `exactHessian` implicit operator with walls; obstacle shapes other than circle, box and equilateral triangle; moving analytic bodies in a CUDA graph. These are documented refusals in `ANALYTIC_BOUNDARIES_PLAN.md`.
