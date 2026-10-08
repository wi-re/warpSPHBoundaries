# warpSPH patch series: analytic boundaries (scratch, nothing pushed)

Eight patches on warpSPH `75c6254` that let the weakly compressible delta+ scheme run with ANALYTIC boundaries (this package's
`warpSPHBoundaries`: exact kernel integrals over the solid) instead of boundary particles. The particle path is untouched: every hook
is guarded by `schemeConfig.boundaryProvider is not None`. Design and audit: `docs/audit-warpsph-boundary-hooks.md` (s.6).

| patch | content |
|---|---|
| 0001 | `RigidBody.representation`, `buildAnalyticRigidBody` (mass / inertia from `Body.massProperties`; infinite for an unbounded wall) |
| 0002 | `ParticleRegion.representation`, `buildRegion(representation=)` (not sampled, no ghost layer), the initializer builds the bodies and the provider; `dambreak --wallRepresentation analytic` |
| 0003 | `boundary/` (provider protocol + adapter + body binding), `modules/analyticBoundary/wallTerms.py`: wall continuity, pressure force, viscosity; hooks at stages 05b / 11 (inside the frozen-diffusion tuple) / 12 / 13 |
| 0004 | surface detection: the Barecasco passes as separate partial-sum modules, the wall continuum added between them, normals / lMin from the fluid + wall matrix (`detectFreeSurface` dispatches) |
| 0005 | shifting: the wall's share of the raw sum in `computeDeltaShift` (warpSPH's W0 convention); `solveShifting` runs unchanged |
| 0006 | no-penetration impulse from the provider's signed distance; the wall flux in the system's time-centred continuity closure (`drift_rates`); `analyticWallPressure='normal'`; dam break regression |
| 0007 | the right-hand-side CUDA graph refuses analytic boundaries (superseded by 0008) |
| 0008 | Michel 2022 and implicit / dynamic shifting with walls (wall share of grad C-tilde, `U_char` from the provider's `dir_extreme`, grad C of the implicit solve; `shiftScheme` / `shiftProjection` dam-break params); wall loads (`wallLoads`, `RigidBody.load`) and free analytic bodies (`RigidBody.dynamic`); obstacles (circle, box, triangle) and mixed scenes in the dam break; the whole-step CUDA graph with analytic walls (host constants `_analyticSupport` / `_analyticMass` / `_analyticDx` fixed at initialisation; moving or dynamic bodies, implicit shifting and mixed scenes refused up front) |

## Apply and test (without touching the shared environment)

```
git clone <warpSPH> /some/scratch/warpSPH && cd /some/scratch/warpSPH
git checkout -b analytic-boundaries 75c6254
git am /path/to/curvatureBoundaries/patches/warpsph/*.patch
PYTHONPATH=$PWD/src ~/miniconda3/envs/warp/bin/python -m pytest tests -q          # warpSPHBoundaries comes from the environment (editable)
```
The patched tree is used only through `PYTHONPATH`; the `warp` environment's editable warpSPH (the live working tree) is never reinstalled or modified.
The new tests (`tests/test_analytic*.py`) skip when `warpSPHBoundaries` is not importable.

## What the tests establish

* wall terms (continuity, pressure with the p_b >= 0 clamp, viscosity; free-slip and no-slip) = `DeltaSPH2D.rhs` wall contribution, 1e-13 in double, float32 round-off in the state's precision;
* detector: raw and dilated masks equal exactly, lMin and the normals to float32 round-off, regular and jittered;
* shifting: `solveShifting` with analytic walls = `DeltaSPH2D.shift` to 1.4e-4 relative (float32) with the curvature gate off;
* dam break 0.5 s, nx 64: analytic against particle walls: max velocity 5.340 / 5.345, kinetic energy 3.567 / 3.712 (4 %), density bounds equal, no penetration; the reference solver reproduces the analytic run exactly (max velocity 2.929, density 1.0000 to 1.0055 at 0.15 s).
* warpSPH's own suite: 870 passed, 1 skipped (one run had a flaky `test_threeWayShiftComparison_allRelaxTowardUniformDensity` that passes alone and on the unpatched base).

## Findings during the port (upstream-relevant)

1. **`WeaklyCompressibleSystem.drift_rates`** (time-centred continuity, default on) re-evaluates the continuity rate for the mean velocity from the pair modules only. Any wall treatment that adds its flux in the right-hand side must add it there too, or the density at the wall is advanced without the wall's compression (the analytic dam break diverged within 0.06 s until it did).
2. **`_curvatureGate`** (shifting, Sun 2019 Eq. 21) takes the minimum normal dot product over the RAW Verlet list (pairs up to `verletScale` x support), not the pairs inside the support: it gates more than the equation says.
3. **`computeDeltaShiftWarp`**'s tensile reference value is the kernel at dx / kernelScale, not at dx (3.7 % of the tensile term for Wendland C2 at h = 4 dx).
4. The analytic hydrostatic wall term assumes the fluid next to the wall is in hydrostatic balance (its tangential part pushes a free-falling wall column up by 3.15 m/s^2 at the dam break's uniform-density start); the boundary particles extrapolate along the normal only. Harmless dynamically here (the run matches), `analyticWallPressure='normal'` removes it.
5. Library schemes other than the symplectic Euler read host time and cannot be CUDA-graph captured (separate repo, warpSPHIntegrators).

## Results added by 0008

* Michel: the wall share of grad C-tilde equals the boundary-particle sum to 3 % (max, float32, lattice quadrature of the wall sum), `U_char` is the supremum of the lattice maximum (closed-form flat-wall check); dam break at 0.3 s against particle walls 1.9 % (fastest particle) / 4.8 % (kinetic energy).
* Implicit / dynamic: the solved shift equals the particle-wall path (ghost nodes excluded) to 2.8 %; the `exactHessian` operator is refused (needs the wall Hessian integral); dam break against the analytic delta+ run 1.1 % / 0.04 %.
* Loads: `wallLoads` equals `DeltaSPH2D.loadsAt`; a free cylinder of density 0.5 (2.0) accelerates at 3.0 (float64) / 2.5 (float32) m/s^2 against the ideal 3.27 (added-mass limit); the acceleration is the small difference of two large numbers, the float32 load of a buoyant body is 3 % lower than the float64 one.
* Obstacles / mixed: kinetic energy within 3-4 % of the boundary-particle obstacle, fastest particle 6-11 % lower (a spray particle), mixed scene stable, KE within 1.1 %.
* Graph: the whole step is bitwise the eager one for the tank, the tank with an obstacle and Michel shifting (about 3x faster).

## Upstream findings added by 0008

6. `computeImplicitShift` sums the mDBC ghost nodes (kind 2) into grad C (the Michel / delta sums exclude them through the operation mode): the boundary-particle dam break with implicit shifting diverges (fastest particle ~900 at 0.3 s) while the analytic one is stable.
7. `getTransformationMatrix` builds tensors from device scalars (`torch.tensor([[tensor, ...]])`) and `updateBodyParticlesWCSPH` indexes with boolean masks: both read the host, so a whole-step graph cannot be captured with particle rigid bodies.
8. A failed capture of the implicit shifting (host-synchronising Krylov solvers) leaves the CUDA context in an error state ("operation not supported on global/shared address space"), the eager fallback then crashes; `_rhsIsGraphable` now refuses implicit / dynamic shifting.
9. The graph's validation and re-capture execute the step more than once on a cloned state, but rigid-body tensors are shared: a moving body is advanced extra times. Only static bodies are graphed.

## Not done

Mixed-scene equivalence beyond the dam break; the incompressible / ACSPH consumers (audit s.6 table); 3D; the `exactHessian` implicit operator with walls; obstacle shapes other than circle, box and equilateral triangle; moving analytic bodies in a CUDA graph.
