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
8. `computeDensityDiffusion` (`modules/deltaSPH/densityDiffusion.py`, kernel `wp_densityDelta.py`) — wall adds a `W'(r)/r`-weighted integral of ψ.
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

The numerical kernels of the scene layer are **already Warp float64** launched on torch-owned memory (`wp.from_torch` in `python/edgebound/warpbc.py`), the same memory model warpSPHCore uses at its API. Torch is not the
obstacle. The obstacles are in the orchestration around the kernels:

* **Data-dependent shapes.** The broadphase and adjacency build use `int(cnt.sum())`, `torch.nonzero` and `repeat_interleave` (`scene.py` `buildCellList`, `ParticleCells`, `buildAdjacency`): every call synchronises the host and
  produces variable-length pair lists. warpSPH captures its RHS in a CUDA graph (`schemes/deltaSPH.py::_graphedRHS`, only when `_rhsIsGraphable`), which needs fixed shapes, so the scene adjacency has to become a
  fixed-capacity, Verlet-style structure (build every few steps with a skin, reuse in between) before it can sit inside a graph.
* **Per-step rebuild.** `DFSPH2D` rebuilds the scene adjacency each step (and reuses the friction adjacency as the next step's). warpSPH reuses Verlet lists across steps (`verletScale`).
* **Python loops over bodies** (~3 ms of launch overhead per body, `scene-architecture.md`): fine for a tank + a few obstacles, not for hundreds of bodies without batching.
* **Representation switches** (`SdfRep` probe validity, tier switches, `VolumeRep` moments mode) are data-dependent branches on the host.
* **2D only.** The edge machinery is complete in 2D; the 3D face→edge chain is not done.
* **Autodiff:** the torch/warp bridge with adjoints exists for the 2D edge integrals (`torch2d.py`, `warp2d.py`), not yet for the scene layer's per-query fields.

## 4. Operations added to the scene layer for δ⁺ (and what is still missing)

Filled in as they are built; each entry: what, where, how verified.

| operation | needed by | status |
|---|---|---|
| per-query scalar field `P_b(x')` with gradient (`BodyField(perQuery=True)`) | pressure force, DDT | have (DFSPH) |
| `Covariance` (g1 moments) | `L`, surface detection | have (DFSPH §6) |
| per-body `perBody` force output | wall force bookkeeping | have |
| radial kernel `W'(r)/r` | DDT | to do |
| kernels `W⁵` / `W⁴ ∇W` | PST | to do |
| p = 2 moments `∫ y_a y_b g(r)` | viscosity, torque | to do |
| `Scene.closestPoint` / normal / signed distance | no-penetration, surface normals | to do |
| per-query vector field (mirrored free-slip velocity) | continuity, viscosity | to do (check) |

## 5. Change log

One line per change that matters to the port: date, what changed in *this* repo, which warpSPH module it replaces or touches, and any deviation from the reference's formula (with the reason).

| date | change here | warpSPH counterpart | deviation from the reference |
|---|---|---|---|
| 2026-10-02 | (start) plan and these notes | — | — |
