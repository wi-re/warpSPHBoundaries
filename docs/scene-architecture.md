# Scene architecture: bodies, representations, per-type adjacency (2D)

**Status:** [V] — `python/edgebound/scene.py`, `warpbc.edge_channels`, tests `tests/edge/test_scene.py` (38), benchmark `python -m edgebound.scene_bench`.
Supersedes the "one global element grid" structure of `boundaryOps.py` for scenes (`boundaryOps` stays the volume-element engine).

## 1. Why

A scene with complex polygons / many tetrahedra must not run every particle against every element, and must not rebuild an acceleration structure when a
rigid body moves. warpSPH already tracks a pose per rigid body (`RigidBody.centerOfMass`, `orientation`, velocities); the geometry lives in the body's local
frame and is static.

## 2. Structure

```
Scene ── Body (pose: centre, angle; linear / angular velocity; OBB in the local frame)
            └── representations (all in LOCAL coordinates, each with its own static acceleration structure)
                  SurfaceRep   closed polylines, solid on the left          edge-local terms + body-level indicator   (exact)
                  VolumeRep    triangles, P1 nodal fields                   boundaryOps pair engine                    (exact)
                  ImplicitRep  DiskBody / HalfPlaneBody                     tier 3 / tier 4 / polygon-surface (hard)   (model)
                  SdfRep       sampled signed distance + optional fallback  tier 3 / fallback surface (hard)           (model)
```

Per step (`Scene.buildAdjacency`):

1. **Broadphase.** Particles are sorted once into a uniform cell grid (`ParticleCells`, cell = largest support). A body gathers the cells under the world
   box of its OBB (inflated by the support) and tests those particles *exactly* — support sphere against the OBB in the body frame (a rigid map keeps the sphere a
   sphere). Cost: O(particles near the body), independent of the number of other bodies' particles.
2. **Narrow phase per representation**, in the body frame: SurfaceRep → static edge cell list + exact point–segment distance; VolumeRep → static element grid;
   Implicit → analytic distance, hard tier per particle; Sdf → probe test (below).
3. **One adjacency per type** (`SceneAdjacency.entries[b]`: `surface`, `implicit`, `volume`), already rotated to the world frame
   (`MomentPairs.toWorld`: λ is invariant, g0 and m1 rotate as vectors, g1 as `R g1 Rᵀ`).
4. **One operation per type** (`sceneOperation`): all boundary terms are linear in the boundary data, so every type accumulates into the same output
   (`index_add_`); tests check "scene = sum of single-body scenes" and rotation covariance (rotating bodies + particles rotates the vector results).

Moving a body (`Body.move(dt)`, same explicit Euler as `warpSPH.rigidBody.integrateRigidBody`) changes only `centre` / `angle`; the cell lists, bins and
element grids are reused (tested by identity), and the result equals a freshly constructed body at the new pose to 1e-13.

## 3. The surface representation (the new piece of maths: nothing but exact edge terms)

A closed boundary needs no triangles. Every channel of the P1 engine (`m00, m10, m01, g00, g10, g01`) is

  `channel = Σ_edges (edge-local part) + indicator(x) · u`,  `u = (1,0,0, 0,0, 1,0, 0,1)` (`warpbc.indicator_vector`)

because the compact-potential recursions make all moments of order ≥ 1 and all gradient channels purely edge-local, and the only non-local quantity is the
value's indicator (`1[x ∈ body]`, which also enters `g_(1,0)_x = g_(0,1)_y = ∂m + m00`). `warpbc._edge_channels_kernel` evaluates one (particle, edge) pair
with the truncated forms only (the inner potentials of the triangle kernel rely on the constant dropping out over a closed polygon). Verified: edge sums +
indicator reproduce the triangle weights to 1e-14 (value and gradient, 3 kernels, points inside / outside / near), and the whole scene operation equals the
triangulated `VolumeRep` of the same polygon to 2e-10 for **every** operation class and gradient mode, for a non-convex L-shaped body moved and rotated.

The indicator is the winding number of the loops (`SurfaceRep.indicator`, half-open crossing rule with the same predicate `cross(q-p, x-p)` as the edge terms'
`z`), **plus `background`**: `SurfaceRep.polygon(P, solid='outside')` is a hole in an infinite solid (clockwise loop, `background = 1`), which is what a tank wall
or a circular cavity is. Particles in cells that hold no edge within a support radius get the indicator from a static per-cell table (computed once per grid build),
everything else is ray cast against a y-binned edge list.

**First moments ⇒ linear boundary fields.** With `m1 = ∫ y W` and `g1[d,j] = ∫ y_d ∂_j W` per pair, any field `A(x') = a0 + a1 (x' − c)` is exact:

  `∫ A W = A_i λ + a1·m1`,   `∫ A ⊗ ∇W = A_i ⊗ g0 + a1·g1`,   `A_i = A(x_i)`

so a rigidly moving wall (`BodyField.rigid(body)`: `a0 = v_c`, `a1 = ω [[0,-1],[1,0]]`) and any hydrostatic wall field need no nodal data. The four gradient
modes enter as an affine map of the field per query (`s A + c`): Naive `(1, 0)`, Difference `(1, -f_i)`, Summation `(1, +f_i)`, Symmetric `(ρ_i/ρ_b, (ρ_b/ρ_i) f_i)`.
`BodyField(perQuery=True)` gives the value at each query position (a wall pressure extrapolated from the particle itself, `p_b = p_i + ρ g·(x' − x_i)`).

**Reactions.** Force on a body = `−Σ m_i (contribution)`, equal to minus the total momentum change (tested to 1e-12). The torque about the body centre,
`−Σ m_i A_i ∫ (x'−c) × ∇_x W dA'`, is **exact for a constant field** (it needs exactly `g1`); tested against a 14×14-per-triangle Gauss quadrature to 5e-4 relative
(kernel-kink limited). For a position-dependent field it needs the second moments (p = 2): reported as NaN, `torqueExact` says so. Volume bodies give per-vertex
reactions and an approximate nodal torque.

## 4. Implicit and SDF representations

* `ImplicitRep(DiskBody | HalfPlaneBody)`: the hard tier policy of `tier-selection-2d.md` per particle: R/h ≥ 2 tier 3, ≤ 0.2 tier 4, else the exact **surface polygon** of
  the shape (area preserving, edges ≤ h/16; cavities as clockwise loops with background 1) — the fallback is now a `SurfaceRep`, not a triangle fan.
  Constant fields only (no first moments for tiers 3/4 yet).
* `SdfRep(values, origin, spacing, fallback=SurfaceRep)`: the only *generic* implicit type (sampled, so it works for any Warp kernel). Bilinear `d` with node-blended
  gradient and Laplacian; surface curvature `κ = Δd / (1 − dΔd)` (2D), `λ = F0 + κF1 + κ²F2` via the tier-3 tables. **Validity is a hard per-particle switch**:
  `|κh| ≤ 0.5` and `| |∇d| − 1 | ≤ 0.1` at the particle and at 16 probes on the circles of radius h/2 and h around it. This rules out ridges / medial axes (the unit-gradient test)
  and convex corners (κ → ∞); invalid particles use the fallback surface (error if none was given). Tested: a sampled disk (R = 3h, spacing h/16) is within 5e-4 of the exact
  disk; a box SDF with a fallback box is within 2e-3 everywhere, 1e-4 on average. The approximation error of the model is the tier-3 error (`κ³` terms) plus the sampling error
  of the SDF (here h/16); both are reported by the tests, not hidden.

## 5. Cost (benchmark: 1e6 particles, h = 3Δx, tank loop of 2000 edges + 100 small bodies, RTX-class GPU, float64)

| stage | time |
|---|---|
| tank body alone (broadphase 3 ms, surface adjacency + edge channels 13 ms) | 16 ms |
| all 101 bodies, adjacency (121 520 pair terms) | ~330 ms |
| operation (all bodies) | 8 ms |
| re-adjacency after moving all bodies | ~330 ms (no structure rebuilt) |

The 100 small bodies cost ~3 ms each, which is **launch / Python overhead**, not arithmetic (a body with a handful of edges and a few hundred candidates). Batching bodies of
one representation type into one launch (per-edge body index, world → local transform inside the kernel) removes it; not done yet because rigid-body scenes are normally
a handful of bodies.

## 6. Mapping to the omniSPH-style DFSPH steps (what a solver calls)

| solver step | scene call |
|---|---|
| density summation, `ρ_i += ρ_0 λ_i` | `Density` with `BodyField(rho=ρ_0)` |
| velocity divergence with a moving wall | `Divergence`, mode Difference, `queryValues = v`, `BodyField.rigid(body)` |
| pressure gradient, Neumann wall (`p_b = p_i`) | `Gradient`, mode Naive/Symmetric, `BodyField(p_i, perQuery=True)` |
| pressure gradient with hydrostatic wall pressure | same with `a1 = ρ g` |
| viscous / drag terms from the wall velocity | `Gradient` / `Interpolate` of the rigid velocity field |
| force and torque on a dynamic body | `returnReaction=True` (`SceneReaction.force`, `.torque`) |

Not in the interface yet: Laplacian / Covariance, periodic domains, the p = 2 weights for the exact torque of position-dependent fields, batching of many small bodies,
autograd through the weights.

## 7. What carries over to 3D

The interface above is dimension-agnostic: `Pose` becomes a rotation matrix / quaternion, the OBB test and the cell lists gain a coordinate, `MomentPairs` keeps the same
fields (`g1` is 3×3), and the surface representation becomes triangle faces with the face → edge reduction (the only new mathematics); implicit and SDF types reuse the
same switch structure (tier 3 with two principal curvatures).
