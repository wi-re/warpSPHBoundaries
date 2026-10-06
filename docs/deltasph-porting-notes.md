# δ⁺-SPH on analytic boundaries — notes for porting into warpSPH

**Status:** living document. The solver is developed here first (`deltasph-plan.md`, decision 1: self-contained `deltasph2d.py`); the long-term goal is to move the boundary code into the core of warpSPH and add the
front-end changes. This file records *where each term lives in warpSPH*, *how the wall enters there*, *what replaces it here*, and *what had to change*, so the port can be done term by term. Update the log (§5) in the same
commit as the change it describes. Paths are relative to `~/dev/warpSPH/src/warpSPH/` unless stated.

## 1. Where a wall touches the reference solver

In warpSPH the wall is **a particle population** (`state.kinds == 1`, boundary particles in layers, ghost nodes `state.ghostOffsets`) inside the *same* adjacency and the *same* operations as the fluid. Nothing is
"a boundary term": the wall is felt because (a) `kinds == 1` particles carry density/pressure/velocity written by the mDBC modules before the RHS terms run, and (b) every pair operation runs `AllToAll` /
`SuperSymmetric` over fluid + wall. The analytic walls are the opposite: a separate adjacency per body (`Scene.buildAdjacency`) and one integral per term. A port therefore has to **split every pair operation
into a fluid–fluid part (unchanged) and a wall part (scene)** and drop the mDBC population-writing stage.

Step order of one RHS call (`schemes/deltaSPH.py::_deltaSPH_rhs`) and the stages that disappear or change:

1. `buildVerletList(..., SuperSymmetric)` (warpSPHCore) — fluid adjacency stays; the wall layers leave it; one `Scene.buildAdjacency` is added (Verlet-style reuse is not implemented there, §4).
2. `computeDensities` (`modules/density/density.py`) — fluid sum; wall adds `λ` (`Density`).
3. `computeMdbcDensityEnglish2025` (`modules/mdbc/english2025.py`) — **removed**: its job (ρ_b, P_b at the wall) becomes a per-query field `P_b(x') = P_i + ρ(g − a_w)·(x' − x_i)`, clamped at 0.
4. `computeBoundaryVelocities` (`modules/mdbc/velocity.py`, free-slip mirror) — **removed**: replaced by the per-query field `v_b = v_i − 2 (v_i·n) n`.
5. `weaklyCompressibleEOS` (`modules/eos/weaklyCompressible.py`) — unchanged.
6. `detectFreeSurface` (`modules/surfaceDetection/wrapper.py`: Maronne / Barecasco / colour field) — the wall contributes to `Σ V W`, `Σ V ∇W` and the covariance `Σ V r⊗∇W`: `Density`, `Gradient`, `Covariance`. Thresholds were tuned against particle walls.
7. `computeGradRho` / `computeGradRhoL` (`modules/density/gradRho.py`, `gradRhoL.py`) — wall adds `∫ (ρ_b − ρ_i) ∇W` and the g1 moments of `L`.
8. `computeDensityDiffusion` (`modules/deltaSPH/densityDiffusion.py`, kernel `wp_densityDelta.py`) — **fluid-to-fluid only in warpSPH** (`operationMode = FluidToFluid`, as DualSPHysics): no wall term, no new kernel. The hydrostatic part `ρ0 g·x_ij/c0²` of `fourtakas2019` keeps ψ = 0 on truncated stencils.
9. `computeVelocityDiffusion` (`modules/deltaSPH/velocityDissipation.py`, kernel `wp_viscosityDelta.py`) — wall adds a second-moment integral (`∫ y⊗y g(r)`).
10. `computeMomentum` (`modules/momentum/inconsistent.py`, continuity) — wall adds `ρ_i (v_i·∇λ − ∫ v_b·∇W)`.
11. `computePressureForceSurfaceAware` (`modules/pressure/surfaceAware.py`, `wp_surfaceAware.py`) — wall adds `−(1/ρ)(P_i ∇λ + ∫ P_b ∇W)` for the rows the switch selects.
12. `computeForcing`, `computeGravity` (`modules/boundaryConditions`, `modules/gravity`) — unchanged.
13. `computeMdbcNoPenShift` (`modules/mdbc/wp_nopenshift.py`) — **replaced** by a closest-point / normal query on the scene (does not exist yet).
14. `enforceUpdates`, zero the update of `kinds != 0` — **removed** (no wall particles).

Outside the RHS: PST `modules/shifting/delta.py::computeDeltaShift` (kernel `sample/wp_deltaShift.py`), integration in `~/dev/warpSPHIntegrators/src/warpSPHIntegrators/integration.py` (`symplecticEuler`: the RHS is
called twice per step, the second time at the half step with the stage `dt/2` and no stage index), configuration `configurations/weaklyCompressible.py::Sun2017DeltaSPHConfig`, case `cases/dambreak.py`
(Marrone 3.1, `scripts/probe_deltaSPHMarrone.py`).

## 2. State the port has to carry that the analytic wall adds

| item | reference | analytic wall |
|---|---|---|
| wall geometry | boundary particle layers + `ghostOffsets` + mean boundary normal (`systems/weaklyCompressible.py::_meanBoundaryNormal`) | `Scene` of `Body`s (poses, velocities, accelerations, representations) |
| per-step wall data per fluid particle | none (pairs) | `λ`, `∇λ` (per body), g1 moments, a1-type per-query fields, all cached per step (`DFSPH2D._prepare`) |
| wall velocity / acceleration | wall particle velocities | `Body.linearVelocity`, `angularVelocity`, `accelerationAt(x)` |
| forces on bodies | sums over wall particle pairs | `perBody` scene operations; exact bookkeeping per term where possible, plan §2 row 11 |

## 3. Constraints a port must respect (what actually limits the scene layer today)

The numerical kernels of the scene layer are **already Warp float64** launched on torch-owned memory (`wp.from_torch` in `src/warpSPHBoundaries/edge/warpbc.py`), the same memory model warpSPHCore uses at its API. Torch is not the
obstacle. The obstacles are in the orchestration around the kernels:

* **Data-dependent shapes.** The broadphase and adjacency build use `int(cnt.sum())`, `torch.nonzero` and `repeat_interleave` (`scene.py` `buildCellList`, `ParticleCells`, `buildAdjacency`): every call synchronises the host and
  produces variable-length pair lists. warpSPH captures its RHS in a CUDA graph (`schemes/deltaSPH.py::_graphedRHS`, only when `_rhsIsGraphable`), which needs fixed shapes, so the scene adjacency has to become a
  fixed-capacity, Verlet-style structure (build every few steps with a skin, reuse in between) before it can sit inside a graph.
* **Per-step rebuild.** `DFSPH2D` rebuilds the scene adjacency each step (and reuses the friction adjacency as the next step's). warpSPH reuses Verlet lists across steps (`verletScale`).
* **Python loops over bodies** (~3 ms of launch overhead per body, `scene-architecture.md`): fine for a tank + a few obstacles, not for hundreds of bodies without batching.
* **Representation switches** (`SdfRep` probe validity, tier switches, `VolumeRep` moments mode) are data-dependent branches on the host.
* **2D only.** The edge machinery is complete in 2D; the 3D face→edge chain is not done.
* **Autodiff:** the torch/warp bridge with adjoints exists for the 2D edge integrals (`torch2d.py`, `warp2d.py`), not yet for the scene layer's per-query fields.

## 4. Operations added to the scene layer for δ⁺

| operation | needed by | status |
|---|---|---|
| per-query scalar field `P_b(x')` with gradient (`BodyField(perQuery=True)`) | pressure force | have (DFSPH), clamped ≥ 0 in `DeltaSPH2D._wall_excess` |
| `Covariance` (first moments `∫ y⊗∇W`) | `L`, λ_min, shifting normals | have |
| per-body `perBody` outputs | wall force bookkeeping | have (pressure part only) |
| `Scene.inside(points, body=None)` | free-surface detector, viscous wall term, shifting tensile control (polar quadrature of the solid) | **done** |
| `Scene.signed_distance(points)` → (d, n, hit) | no-penetration impulse | **done** for surface loops, SDF, implicit; not for volume representations |
| radial kernel `W'(r)/r` | density diffusion | **not needed**: the DDT is fluid-to-fluid in warpSPH |
| kernel `W⁵` / `W⁴∇W` | shifting tensile control | **replaced by quadrature** (polar sampling of the solid, 24 × 96); an exact kernel would remove the sampling error |
| Laplacian operation (`Δλ = ∫∇²W dA`, first moments of `∇²W`) | viscous wall term with an exact boundary (replaces the pairwise `p = 2` form; user, 2026-10-02) | **replaced by quadrature of the pairwise form** (1 % of a fine half-plane integral); the exact Laplacian wall term is to do; `p = 2` weights are still needed for torque |
| per-query vector field (mirrored free-slip velocity) | continuity, viscosity at curved walls | not done: flat-wall approximation with `n = ∇λ/|∇λ|` per body |
| Wendland C4 (`KernelFunctions.Wendland4`) | sloshing | works in the scene layer (kernel `w4`); `DeltaSPH2D` carries the C2 / C4 pair kernels |

## 4b. Status of the δ⁺ term inventory (2026-10-02)

**Ported** (validated on tank, wedge, Marrone dam break, sloshing): continuity with free-slip wall (exact `∇λ`, mirror with the particle's own velocity); isothermal EOS; Antuono pressure force with exact wall integrals and a
clamped hydrostatic wall pressure, dilated surface mask for the switch; fourtakas2019 density diffusion (fluid-to-fluid, as warpSPH); α-viscosity, fluid pairs and wall term; Barecasco detector with the wall as a sampled continuum;
δ⁺ shifting (Sun 2017 Eq. 7, Sun 2019 surface treatment: λ_min normals, curvature gate, λ gate, caps); no-penetration impulse; symplectic Euler with time-centred continuity; Sun-2017 time step (or a fixed dt); Wendland C2 and C4;
time-dependent (rotating) gravity; Marrone MLS wall probes and the SPHERIC sensor probes; pressure part of the wall force on each body.

**Not ported:** density-diffusion variants with renormalised ∇ρ and the renormalised pressure force; Michel-2022 and implicit shifting (warpSPH's sloshing default is Michel-2022, our run uses Sun's), `correctdrhodt` / `correctdvdt`;
physical viscosity (Morris, ν) and the viscosity switch; no-penetration `finalize` / `derivative` placements; other EOS types (Tait is only used for the sloshing probe), RK integrators, forcing / Dirichlet hooks.
Walls: moving or rotating bodies in δ⁺ (the no-penetration law and the viscous term assume static walls), curved-wall mirror normals, volume representations in `signed_distance`, exact second-moment and `W⁵` weights.
Bookkeeping / infrastructure: exact momentum bookkeeping (viscous, shifting and no-penetration contributions are not booked on the bodies), torque, Verlet-style adjacency reuse and CUDA-graph capture (the torch solver is 8× slower
than warpSPH), 3D, periodic domains, interacting bodies.

## 5. Change log

One line per change that matters to the port: date, what changed in *this* repo, which warpSPH module it replaces or touches, and any deviation from the reference's formula (with the reason).

| date | change here | warpSPH counterpart | deviation from the reference |
|---|---|---|---|
| 2026-10-02 | (start) plan and these notes | — | — |
| 2026-10-02 | `Scene.inside` | none (wall particles are explicit) | — |
| 2026-10-02 | `deltasph2d.py`: `DeltaSPH2D`, `hydrostatic_tank`; terms 2, 5, 10, 11, 12 (EOS, continuity + wall, Antuono pressure force + wall, gravity, α-viscosity fluid–fluid), fourtakas2019 DDT fluid–fluid, symplectic Euler, Sun-2017 dt | `_deltaSPH_rhs`, `symplecticEuler` (`warpSPHIntegrators/verlet.py`), `timestep/weaklyCompressible.py` | wall continuity uses the particle's own velocity, not the Shepard velocity at the ghost; wall pressure `P_b = P_i + ρ0 (g − a_w)·(x' − x_i)` clamped ≥ 0 (mDBC clones the ghost's pressure and may be negative); no wall term in α-viscosity (zero at rest) |
| 2026-10-02 | free-surface detector (Barecasco) with the wall as a continuum of particles | `modules/surfaceDetection/barecascoDetection.py`, `wp_barecasco.py` | the wall's cover vector and cone count come from sampling the solid on an 8 × 48 polar grid (`Scene.inside`) with number density `μ/dx²`; the reference counts wall particles | 
| 2026-10-02 | `neighbor_pairs` cell = 1.01 × support (shared with DFSPH) | `buildVerletList` (warpSPHCore) | warpSPH is unaffected by the border-rounding that dropped 2 % of reverse pairs here, but any port that builds a cell list with cell = support on a lattice with dx dividing the support needs the same check |
| 2026-10-02 | `english_wedge`: smooth `SurfaceRep` triangle (exact corners) as a second body; first fluid row ≥ dp/2 from it | wedge as SDF-sampled boundary particles (`caseUtils/weaklyCompressible.py::buildObstacleSDF`) | the layout of a regular lattice is not consistent with a smooth sloped wall (first-order moment error 0.2 vs 0.01 on a flat wall at dp/2); the wall particles of warpSPH share the lattice and have no such mismatch. Port consequence: an analytic obstacle needs a body-fitted initial packing (open, docs/deltasph-validation.md §3) |
| 2026-10-02 | `DeltaSPH2D.settle` / `pack` / `residual` (experimental layout tools), `staircase` control | none | diagnostic only |


| 2026-10-02 | dam break (`marrone_dambreak`, `wall_probes`, `run_dambreak`): wall term of `computeVelocityDiffusion` (free-slip mirror, polar quadrature of the solid, 24 x 96), time-centred continuity (`symplecticEuler` drift field: `rho += dt (K(x_h, vbar) - kin(k1))`), dilated surface mask for the Antuono switch (warpSPH binds `fs, fsm` swapped in `_deltaSPH_rhs`: the switch sees the DILATED set), `Scene.signed_distance` | `modules/deltaSPH/velocityDissipation.py`, `systems/weaklyCompressible.py::drift_rates`, `schemes/deltaSPH.py` | wall viscosity by quadrature (1 % of the exact half-plane value), not exact; mirror velocity = the particle's own |
| 2026-10-02 | `DeltaSPH2D.shift` (delta+ PST: Eq. 7 with the Sun-2019 surface treatment: lambda_min normals, curvature gate 15 deg, lambda gate 0.4 fluid-only, Eq. 14 cap, 0.5 dx clamp), `no_penetration` (impulse placement) | `modules/shifting/delta.py::computeDeltaShift`, `modules/shifting/wrapper.py::solveShifting`, `systems/weaklyCompressible.py::finalize`, `modules/mdbc/wp_nopenshift.py` | wall part of the tensile control `int W^4 grad W dA` by polar quadrature (no `W^5` kernel needed); normals from lambda gradients of the fluid pairs with L = (fluid + wall covariance)^-1; no-penetration law re-derived for an analytic wall at dp/2 (`f(d)`), static walls only |
| 2026-10-02 | speed: `neighbor_pairs` uses a dense distance matrix for N <= 8000 (82 -> 4 ms at N = 3240) | `buildVerletList` | none; the remaining cost is `Scene.buildAdjacency` + 3 scene operations per RHS call (~60 ms): a port should fuse them and reuse the adjacency across the two RHS calls of a step |
\n
| 2026-10-02 | sloshing (`sloshing_tank`, `sloshing_probes`, `run_sloshing`): the roll is the tank-fixed frame with rotating gravity, `DeltaSPH2D.gravityFn` evaluated after every step (as warpSPH's `postStep`), constant dt; Wendland C4 through `KERNELS` | `cases/sloshingTank.py` (`_applyRollGravity`, `postStep`, `diagnostics`), `examples/sloshingTank/SPHERIC_TestCase10` | walls stay static (no moving-wall terms needed); shifting is Sun 2017 where warpSPH's case default is Michel-2022; sensor = the reference's Gaussian Tait probe plus a wall MLS probe instead of the nearest boundary particle's density |
| 2026-10-03 | `cfg.coverExact`: wall cover vector of the free-surface detector by the exact edge reduction (`cover.cover_vector_scene`, kernel `cone`); default off; cone count (Q3b), viscous wall term (Q1) and tensile term (Q2) still use the polar samples | `barecascoDetection.py` (wall part of the cover vector) | removes the ~1 % quadrature error of the wall cover vector |
| 2026-10-03 | `cfg.coneExact`: wall part of the Barecasco cone count (and of the all-neighbour count) of the free-surface detector by the closed-form area (`cone_area.cone_area_scene`); default off | none | removes the ~1 % polar-quadrature error of the wall cone count and of the all-neighbour count |
| 2026-10-03 | `cfg.tensileExact`: wall part of the delta+ shifting tensile term by the exact edge reduction (`tensile.tensile_vector_scene`, kernel `w2p5`); default off | none | removes the ~1.4 % polar-quadrature error of the wall tensile integral; Wendland C2 only |
| 2026-10-03 | `cfg.tensileExact` now also Wendland C4, through the Chebyshev-quadrature edge plan (`warpbc.STABLE_KERNELS`, `w2p5` / `w4p5`) | none | removes the ~1.4 % (C2) / ~1.7 % (C4) polar-quadrature error of the wall tensile integral |
| 2026-10-04 | `cfg.viscosityExact`: wall part of the artificial viscosity by the exact wall Laplacian (`viscosity.lap_lambda_scene`, Wendland C2 and C4); default off | warpSPH pairwise form with free-slip mirror (`cfg.wallViscosity`, polar quadrature) | **different operator**: naive Laplacian with `ν_eff = α c0 H/(8ξ)` instead of the pairwise form; wall damping of the normal velocity is 12× weaker at `z = 0.1 H` and 0.7× at `0.7 H` (T5.1 table); not a quadrature replacement, a change of the discretisation (user decision, HANDOFF Part A Q1) |
| 2026-10-04 | `cfg.coverExact`, `coneExact`, `tensileExact`, `viscosityExact`, the polar branches of the detector / shift removed — the exact wall operations (cover, cone area, tensile C2/C4) are the only path; `cfg.wallViscosityForm` = `"laplacian"` (default, exact wall Laplacian, Wendland C2/C4) \| `"pairwise"` (the warpSPH form, polar quadrature, kept) | warpSPH pairwise form with free-slip mirror = `"pairwise"` | deviation of the default: different operator (factor 3–12 weaker damping of the wall-normal velocity in the first rows; WORK-005 table); dam-break arrival 2.4897 → 2.4740 t*; `DeltaSPH2D` now requires `SurfaceRep` walls (the polar branches were the only path that stepped `VolumeRep`/`ImplicitRep` walls; `domain="volume"` raises `NotImplementedError`; scene-layer representation independence is covered by `test_scene.py` / `test_dfsph.py`) |
| 2026-10-04 | `cfg.wallViscosityForm = "noslip"`: a third wall-viscosity form (no-slip) — the Chiron et al. 2019 wall-Laplacian flux term `-2 nu_eff (v - v_w) |G|/(rho d)` per body, all-components relative velocity, `nu_eff = alpha c0 H/(8 xi)`, `d = max(distance to the wall, 0.25 dx)`, `v_w = b.velocityAt(x_i)`; `wallViscosityForm` is now `"laplacian" \| "pairwise" \| "noslip"` (default still `"laplacian"`) | Chiron et al. 2019 wall Laplacian flux term; warpSPH has no no-slip wall viscosity | flat-wall model per particle (`|G|`, nearest-point distance), `nu_eff = alpha c0 H/(8 xi)`, `d_min = 0.25 dx`; no `1/gamma` renormalisation (the bulk term is not renormalised); no new Warp / scene code (`G` and the distance are already computed every step); derivation `derivations/noslip-wall.md`; the no-slip form damps the dam break more than the free-slip reference (KE 31.7 %, peak velocity ~43 % lower) — a finding, not adjusted; the damping is that of the scheme's artificial `nu_eff` (≈ 3000 × water's), not a physical wall friction (`derivations/noslip-wall.md` §6) |
| 2026-10-05 | scene-path cost reduction (WORK-008 phase 2a, no result changes): `warpbc.edge_channels(..., channels=)` channel-restricted plans + a `DevicePlan` cache keyed per `(kernel, device, channels)`; `Scene.buildAdjacency(..., channels=)` / `SurfaceRep.pairs(..., channels=)` with a `sceneOperation` guard (a pruned adjacency serves only the `Naive` `Gradient` of a constant scalar field); the cover vector and tensile term build a `(3, 4)`-gradient-channel adjacency; `_wall_data` reuses the `no_penetration` adjacency across the step boundary (key: positions, body poses, supports, kinds; `A` always recomputed) | none (warpSPH has no analytic walls) | results unchanged (bit-level harness PASS, margins ≤ 0.011); `buildAdjacency` 10 → 9 per step, cover / tensile adjacency restricted to the gradient channels |
