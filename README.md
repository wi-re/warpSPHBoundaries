# curvatureBoundaries

Curvature-aware closed forms for the semi-analytic SPH boundary integral of
Winchenbach, Akhunov & Kolb (2020),

```
lambda(x) = Int_{D_B(x)} W(|x-x'|, h) dx',     W(r,h) = C_n/h^n W_hat(r/h)
```

extended beyond the paper's 3-D planar cubic-spline result to:

* **2-D planar** boundaries for all four kernels — new arccos/ln closed forms
  (`maple/01_planar.mpl`, §3 of `docs/derivation.md`);
* **3-D planar** Wendland kernels w2/w4/w6 — single polynomials of degree 2n
  via the monomial basis `F(n,p,x)`;
* **curved boundaries**: solid sphere for the Wendland kernels — both
  branches (`2R+d >= 1` and `2R+d < 1`) are rational `N(R,d)/(R+d)`, with
  exact planar limit `R -> infinity`.

Kernels and conventions follow warpSPHCore
(`src/warpSPHCore/kernels/`, `scripts/kernels/kernel_specs.yaml`):
`q = r/h` with `h` the support radius.

## Layout

```
maple/00_setup.mpl          kernel table, normalization, F(n,p,x) basis checks
maple/01_planar.mpl         2-D + 3-D planar, all kernels
maple/03_sphere.mpl         solid sphere (Wendland w2/w4/w6)
maple/run_all.sh            regenerate src/warpSPHBoundaries/data/symbolic/ (needs Maple 2026)
src/warpSPHBoundaries/curvbound/              PLAN track: kernel table, closed-form evaluators, oracles (curvature-aware planar / sphere integrals)
src/warpSPHBoundaries/edge/         HANDOFF track: exact 2D edge-reduction integrals (mpmath -> numpy -> torch -> Warp), FEM weights, tiers 3 / 4
src/warpSPHBoundaries/scene/        bodies and representations, boundary operations, wall operators of the delta+-SPH solver, the boundary provider (`AnalyticBoundary`, `scene/provider.py`)
src/warpSPHBoundaries/data/tables/  pre-tabulated tier-3 kernel tables (package data)
scripts/                    derivation checks, studies of the integrals (scripts/studies), benchmarks (scripts/bench)
tests/                      pytest suite: tests/{edge,scene,curvbound}
notebooks/demo.ipynb        executed demo: geometry, curves, validation
docs/derivation.md          full mathematical write-up
PLAN.md                     project status & roadmap
src/warpSPHBoundaries/data/symbolic/  Maple exports (committed, shipped with the package; consumed by warpSPHBoundaries.curvbound)
../warpSPHDevelopment/curvatureBoundaries/   development progress and generated evidence (outside this repo)
```

## Package layering

`edge` <- `scene`: the edge machinery and the scene layer (`pip install .`, needs `warpSPHCore` only) are the boundary library `warpSPHBoundaries` (renamed from `edgebound` on 2026-10-06).  A scheme consumes the analytic bodies
through `warpSPHBoundaries.scene.AnalyticBoundary` (the `BoundaryProvider` protocol: integrals at the fluid particles, signed distance, body kinematics); warpSPH (`warpSPH.boundary`, `warpSPH.modules.analyticBoundary`) is the
consumer and hosts every solver.  The layering is enforced by `tests/test_package_layering.py`.

**Retired solvers.** The reference solvers `DeltaSPH2D` / `DFSPH2D` (the former `sim/` sub-package), their cases, validation runners, tests and example notebooks were removed on 2026-10-11 once warpSPH hosted the
same schemes on analytic walls (ANALYTIC_BOUNDARIES_PLAN in the warpSPH repository, stage 2 / 3).  They are kept at the git tag `oracle-solvers-2026-10-11`; warpSPH's `scripts/probe_*Oracle*.py` need them.

## Quick start

Development progress and generated evidence live in the sibling
`warpSPHDevelopment/curvatureBoundaries/` directory. See the
[validation and reproduction guide](docs/validation.md) for output configuration,
documented limitations and the external acceptance matrix.

```bash
# 1. (re)generate the symbolic results — needs Maple 2026
maple/run_all.sh                       # MAPLE=/path/to/maple to override

# 2. install (editable) and run the validation suite; 
pip install -e . --no-deps          # or: pip install -e .[test]
python -m pytest tests/ -q -n 4     # ~3 min with pytest-xdist; GPU tests use cuda:0

# 3. run the demo notebook (jupyter + matplotlib)
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/demo.ipynb --ExecutePreprocessor.kernel_name=python3
```

Python API sketch:

```python
from fractions import Fraction
from curvbound import planar2d, planar3d, sphere

planar3d("w4", Fraction(1, 3))          # exact rational (polynomial in d)
planar2d("w4", "0.3")                   # mpf (arccos/ln closed form)
sphere("w4", Fraction(4, 10), Fraction(1, 5))   # solid ball R=0.4, d=0.2
```

## Validation

The [validation guide](docs/validation.md) describes the accuracy contracts,
independent references and commands to regenerate the evidence. Study commands
write reports, figures and raw arrays outside the checkout by default. Set
`WARPSPH_DEVELOPMENT_ROOT` or an explicit `--output` to select another location.
The development archive is backed up manually, separately from the source repo.

```bash
python scripts/studies/boundary_atlas.py
python scripts/studies/curvature_corrections.py --device cpu
python -m warpSPHBoundaries.validation.robustness --device cuda:0
```

Every tier uses the same twelve-moment order and distinguishes numerical,
SDF reconstruction and geometry/model errors. Missing channels remain explicit;
quadrature refinements are empirical estimates rather than certified bounds.
See the [surface moment contract](docs/surface-moments.md),
[SDF domain guard](docs/sdf-guard.md),
[distance-only reference](docs/sdf-distance-only.md), and
[accepted smooth-SDF limitations](docs/sdf-accepted-accuracy.md).

The 2D surface implementation exposes all twelve moments in Scene and
FusedWall, with conditioned integration for small/sliver triangles. The SDF
guard validates differential-field conditioning; it does not certify an integral
tolerance. Scene SDFs expose all twelve moments through a shared order-two
curvature/variation expansion, including tangential gradient terms. Requested
moment families determine the work performed; the default guarded SDF
backend fuses sampling, support checks and moment evaluation into one float64
Warp kernel. A Torch backend remains available for diagnostics. FusedWall
continues to lower SDF representations to polygons. Higher geometric orders
and sampled differentials still limit the local model's accuracy. Shipped
disk/box tables also have restricted accuracy regimes.
These limitations and their reproducible checks are documented in the guide.

## Status

The 2D implementation includes polygon moments, triangular FEM weights,
disk/box tables, sampled SDF models with polygon fallbacks, and scene operations.
The adversarial audit documents unresolved numerical and model-validity failures;
uniform few-eps accuracy is not established. Current priorities are remaining representation/moment coverage,
higher-order model errors, and per-moment accuracy/fallback decisions.

The experimental `warpSPHBoundaries.mesh3d` reference evaluates analytic
tetrahedral meshes, closed triangular boundaries, and standalone triangle
area integrals, with polynomial moments, kernel derivatives and P0–P3 FEM
weights. It uses arbitrary precision; conditioned float32/float64 and GPU
backends remain open. See the [tetrahedral derivation](docs/derivations/tet-face-edge-chain.md)
and [triangle surface contract](docs/derivations/tri-mesh-surfaces-3d.md).
Sphere/tetrahedron overlap comparisons and plane-slice plots are reproducible
with `scripts/studies/mesh3d_overlap.py` and `mesh3d_slices.py`.
General curved-tube solutions remain deferred.

## Interactive boundary laboratory

Compare triangle meshes, exact polygons/rectangles, distance-only SDFs, curvature
models, and circle references in linked browser viewports:

```bash
python -m warpSPHBoundaries.visualizer
# Open http://127.0.0.1:8765
```

Geometry sliders, kernel/support/grid controls, twelve moment channels, shared
colormaps, difference plots, probes, and cross-sections run directly in the browser.
The optional Python provider retains closed-form polygon evaluation; browser
polygons use split Gauss quadrature, labeled separately from geometry-model error.
The 2D tab now defaults to WebGPU fp32, with an explicit float64 worker option
and automatic fallback; all fields and models retain the same quadrature rules.
[Open the live laboratory](https://fluids.dev/warpSPHBoundaries/volume.html) or
[the 2D interface](https://fluids.dev/warpSPHBoundaries/index.html?2d=1).
See the [visualizer guide](docs/visualizer.md) for models, accuracy limits, and
static hosting. The **3D volumes** tab now computes 30 moment/derivative selections
directly in the browser, with equations and validity notes alongside the plots.
It includes sphere/plane integrals, curvature orders 0–2, small-ball orders
0/2/4, and approximate cubature of sphere/box/tetrahedron solids. WebGPU compute
and a float64 JavaScript worker enable live static-site experiments without a
local Python or warpSPH installation. The local Python voxel-FFT provider remains
available. WebGPU ray casting, transfer functions and XY/XZ/YZ slices share the
computed grids. See the [new 3D derivations](docs/derivations/radial-terms-3d.md).

### Dedicated solid-simplex Warp bundles

`warpSPHBoundaries.fluxbridge.solid_functions` supplies C2/C4 triangle/tet
`density`, `pressure`, `lean`, `wcsph` and `wcsph_extended` device/batch kernels
in native fp64 or protected all-fp32. `SolidEvaluator` adds conservative
whole-element search, periodic images and deterministic CSR assembly.
See [the solid WCSPH trial handoff](docs/handoffs/solid-simplex-wcsph.md) for
physical units, the viscosity conversion and the remaining solver adapter.
These standalone kernels do not require porous coupling; no fluid-solver
dispatch changes automatically when they are imported.
