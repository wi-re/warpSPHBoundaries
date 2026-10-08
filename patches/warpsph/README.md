# warpSPH patch series: analytic boundaries (scratch, nothing pushed)

Seven patches on warpSPH `75c6254` that let the weakly compressible delta+ scheme run with ANALYTIC boundaries (this package's
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
| 0007 | the right-hand-side CUDA graph refuses analytic boundaries |

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

## Not done

Michel 2022 shifting and the implicit / dynamic shifting schemes with walls; mixed scenes (particle and analytic bodies together are structurally supported, untested); moving analytic bodies through the exact integrator (the existing explicit-Euler `finalize` loop moves them; prescribed motion only); wall loads on analytic bodies; the CUDA graph of the RHS; 3D; the incompressible / ACSPH consumers (audit s.6 table); obstacles in the dam break (`wallRepresentation='analytic'` is the plain tank).
