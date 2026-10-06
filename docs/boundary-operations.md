# Boundary operations with the warpSPH call shape (2D, tiers 1/2 on the GPU)

**Status:** [V] — `src/edgebound/scene/boundaryOps.py` (interface), `warpbc.py` (Warp pair engine), `implicitBodies.py` (tiers 3/4), tests `tests/edge/test_boundary_ops.py`, `test_warpbc.py`, `test_implicit_bodies.py`, `test_kernels_extra.py`

## 1. Call shape

```python
from warpSPHCore import OperationProperties, WarpOperation, GradientScheme, KernelFunctions, OperationDirection, ParticleState
from edgebound.scene.boundaryOps import BoundaryMesh, buildBoundaryAdjacency, boundaryOperation

mesh = BoundaryMesh(vertices[V,2], elements[E,3], bodyIds[E])                 # triangles (a polyline wall = a thin strip of triangles)
props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Gradient,
                            gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.BoundaryToFluid)
adj = buildBoundaryAdjacency(queryParticles, props, mesh)                     # like the Verlet/hash adjacency, but it carries the exact P1 weights
a_b = boundaryOperation(queryParticles, props, mesh, queryValues=p, referenceValues=p_nodal, referenceDensities=rho_nodal, adjacency=adj)
```

The query side is the unchanged `ParticleState`; `OperationProperties` is the *same* dataclass as for `warpOperation`. The reference side is the boundary mesh with
**P1 nodal fields** (`referenceValues[V,...]`, continuous) or per-element constants (`referenceElement[E,...]`) instead of particles; there are no volumes/masses on the reference side — the
integral over each element is exact. `adjacency` is optional and reused exactly like the particle adjacency (rebuild when particles move; geometry-static meshes keep the grid).

| `WarpOperation` | boundary contribution (sum over elements in support) | modes |
|---|---|---|
| `Density` | `∫ ρ_b W` (`referenceDensities`, default 1 → the kernel integral) | – |
| `Interpolate` | `∫ A W` | – |
| `Gradient` | `∫ a ⊗ ∇_x W` | Naive `a = A`, Difference `A − A_i`, Summation `A + A_i`, Symmetric `ρ_b ρ_i (A_i/ρ_i² + A/ρ_b²)` (nodal substitution; same weights) |
| `Divergence` | `∫ a · ∇_x W` | same |
| `Curl` (2D) | `∫ (a_y ∂_xW − a_x ∂_yW)` | same |
| `Laplacian`, `Covariance` | **not implemented** (Laplacian needs the profile `W'/r`, negative powers for the splines; Covariance = Liu matrices from the existing moments `k ≤ 2`) | – |

* `operationMode` is honoured: boundary elements are `Boundary`-kind **sources** (`BoundaryToFluid`, `AllToFluid`, `AllToAll`, … act; `FluidToFluid` etc. give zero) and the query kinds are filtered like the particle operators.
* `supportMode`: only Gather semantics (the element has no support radius; `h = supports[i]`). Per-particle supports are supported (adaptive resolution).
* `returnReaction=True` (scalar `Gradient`): per-vertex reaction `R_k = −Σ_i m_i (contribution of vertex k to out_i)`, with `Σ_k R_k = −Σ_i m_i out_i` exactly (tested to 1e-12) — action = reaction for the pressure force, distributed to the mesh vertices (total force exact; torque from vertex forces is first-order exact only — exact torque needs the `p = 2` weights, not exposed yet).
* Kernels: cubic, quartic, quintic, B7, B8, Wendland C2/C4/C6, Poly6 (all piecewise polynomial in `r`: exact). HOCT4 and Gaussian raise `NotImplementedError` (not exact; quadrature route needed).
* 2D only; periodic domains are not handled (elements are not wrapped).

## 2. What runs where

`warpbc.pair_weights` evaluates for every (particle, element) pair the P1 weights `w_k = ∫N_k W`, `G_k = ∫N_k ∇_xW` (physical units) from nine channels (moments `m_(0,0), m_(1,0), m_(0,1)` and `g_α = ∫ y^α ∇_x W`, `|α| ≤ 1`):
the exact compiled plans of `np2d` (compact-potential recursion, **inner potentials for elements inside the support**, exact indicator/atan weights, per-radius selection) run as an interpreter in one Warp kernel per pair, plus the
**far-field Gauss branch** (8×8 nodes, `(R−r)^k` kernel basis) for far elements that do not straddle a kernel radius or are tiny — the same selection rule as the numpy FEM path. Internally everything is float64 whatever the I/O dtype: the number of boundary pairs is small
(only particles within one support of a wall), so float64 is affordable, and it removes the float32 conditioning problem altogether (stage 3's Chebyshev path remains the numpy float32 route). The operations are then batched torch reductions (`index_add_`, `einsum`) over the pairs.

Accuracy (`tests/edge/test_warpbc.py`): all 19 golden FEM fixtures × 4 kernels (x inside / on an edge / at a vertex, `z → 0`, elements `1e-6 … 1 h`, far, half-plane, rim) on CPU and CUDA: **≤ 6e-9 relative to the weight scale (most ≤ 1e-12)**;
random pairs for all supported kernels 2e-11 (weights), 2e-10 (gradient weights); every operation class and mode equals an independent per-pair numpy assembly to 1e-10; covering meshes reproduce the exact integrals (`Σ w = 1`, linear fields exactly) to 1e-12.

Throughput (`python scripts/bench/boundary_bench.py`, RTX PRO 6000 Blackwell, 1 000 000 fluid particles in a tank, 804 wall triangles with edge `h/2`, `h = 3 dx`, Wendland C2):

| stage | time |
|---|---|
| element grid (rebuild only if the mesh moves) | 15 ms |
| adjacency incl. exact P1 weights (26 730 pairs, 11 964 near-wall particles) | 36 ms |
| `Gradient`/`Symmetric` operation on the cached adjacency | 0.5 ms |

i.e. the whole boundary treatment costs about as much as one particle operator pass; it scales with the *near-wall* particles only.

## 3. Tiers 3 and 4 in this interface (implemented; hard switches, no blending)

```python
from edgebound.scene.boundaryOps import BoundaryDescription
from edgebound.scene.implicitBodies import DiskBody, HalfPlaneBody, TierPolicy
desc = BoundaryDescription(mesh=wallMesh,                                   # optional explicit triangles (tiers 1/2)
                           bodies=[DiskBody(center, R), HalfPlaneBody(point, normalIntoFluid),
                                   DiskBody(c2, R2, solid="outside")],      # solid disk, planar wall, circular cavity wall
                           policy=TierPolicy(tier4MaxRadius=0.2, tier3MinRadius=2.0))   # hard thresholds in units of the particle's own h
rho_b = boundaryOperation(ps, props, desc, bodyDensities=rho0)                         # same call as for a mesh
```

For every (particle, body) pair the model is chosen **per particle** from `R/h_i` (adaptive support honoured), thresholds from `tier-selection-2d.md`:

| `R / h_i` | model | evaluated by | measured error vs the exact disk (value / gradient), `d = 0.05…0.6 h` |
|---|---|---|---|
| `≥ 2` | tier 3: `F_0 + κF_1 + κ²F_2`, gradient `∂_d λ n`, `d < 0` and cavities via the complement identity | torch, Hermite tables `F_k(q), F_k'(q)` (512 intervals per kernel, exact mpmath moments, cached in `src/edgebound/data/tables/`) | `R = 4h`: 1.8e-5 / 3.9e-5; `R = 2h`: 1.4e-4 / 2.8e-4 |
| `≤ 0.2` (and `[D−a, D+a]` inside one kernel piece, particle outside) | tier 4: disk series `K = 2` | torch, exact polynomial coefficients of `Δ^k W` | `R = 0.1h`: 2.4e-7 / 8.8e-6 (derivative of an asymptotic series loses an order) |
| in between, or the series' validity fails | tier 2: **polygonisation of the body** (fan, edge `h/16`, polygon area = disk area) merged into the element mesh; only the pairs of particles with hard tier 2 are generated | Warp pair engine | `R = 0.7h`: 6e-8 / 6e-7; `R = 0.35h`: 2e-7 / 3e-6 |
| `HalfPlaneBody` | always tier 3 with `κ = 0`: exact planar closed form `λ_2(d)` (also `d ≤ 0`) | table | ≤ 1e-9 vs the PLAN closed form |

Findings: (i) the area-preserving polygon makes the *exact-element fallback* far more accurate than my earlier `h/8`-edge numbers (the `O(ℓ²)` bias cancels), so tier 2 alone would be the accuracy
choice everywhere — tiers 3/4 are the **cost** choice (O(1) work per particle instead of ~`2πR·16/h` pairs near the surface per body); (ii) switching errors are the errors above (≲ 1.4e-4 at the tier 3/2 switch), never
a jump from the formulas themselves; (iii) the table interpolation error (1e-12 value, 1e-8 derivative) is negligible against the model error.

Operation classes for bodies: constant fields per body only (`bodyValues[B,...]`, `bodyDensities[B]`): `Density` `ρ_b λ`, `Interpolate` `A λ`, `Gradient/Divergence/Curl` `a ⊗ ∇λ` with all four gradient modes
(identical formulas to the mesh path — tested by assembling the explicit pair formulas), reactions per body (`returnReaction` → `(out, vertexReaction, bodyReaction)`, total conserved to 1e-12). A wall moving *rigidly* (velocity linear in
position) needs first moments of the body and is therefore a tier-2 / explicit-mesh case today; the first-moment `F_k` tables are the missing piece. SDFs other than disk/half-plane need a closest-point + curvature callback
(same tables, general closest point): the hook is `body.signed(positions) -> (d, n, kappa)`; a Warp SDF function can be plugged in there.

## 4. Limits worth knowing

* Boundary closures remain solver-side (extrapolated wall pressure / velocity, contact-line behaviour): the operations transfer geometry exactly, they do not decide the wall state.
* Fluid-fluid terms that are nonlinear in the boundary field (Monaghan switch, limiters, `δ`-term gradients) are not linear in nodal values; use a contact-point value (as in omniSPH) — the weights are then the constant-field ones.
* The Warp pair kernel is a plan *interpreter* (~ 30 M value-pairs/s in float64 on this GPU, `docs/autodiff-and-gpu.md`); a generated, unrolled kernel per `(kernel, |α|)` would be faster but is not needed at these pair counts.
* No autograd through the weights w.r.t. particle/vertex positions yet (the analytic gradient `G` is the `x`-derivative of the integral; the vertex adjoint exists in `warp2d.shape_gradient` for the value/moments, not for the P1 weights).
