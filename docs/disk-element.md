# The disk (fibre) element: any radius, no polygon, no mesh

**Status 2026-10-06:** [V] reference (`edge/disk.py`, `tests/edge/test_disk.py`), tables (`edge/disktables.py`), Warp lookup (`edge/warpdisk.py`), the batched `DiskArrayRep` on the fused / graph path with closed-form cone areas (`scene/`), tests (`tests/scene/test_disk_array.py`, `tests/sim/test_fibre_bundle_solver.py`); [P] 3D (sphere / cylinder).

## Why

Fibre bundles (particles flowing past fibres to measure permeability) are 2D disks of many radii first, cylinders and spheres in 3D.  The polygon lowering of `Body.fusedReps` (plan "Phase 5 prep 1") is exact but its cost grows
with the radius (edges <= h/16) and with the number of bodies (every body is a launch over all N queries).  Measured, 6000 queries, support H = 4 dx, one launch family of the solver's fused groups (lam, G, Cov, cover, lap, tens):

| | R/H = 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 1 disk, f64, ms | 4.3 | 7.3 | 12.1 | 20.1 | 40.5 |
| 16 disks (16 bodies), f64, ms | 62 | 105 | 184 | 270 | 403 |
| 1 disk, f32, ms | 0.9 | 1.3 | 2.0 | 1.9 | 2.2 |
| 16 disks (16 bodies), f32, ms | 13.6 | 18.9 | 30.5 | 27.5 | 29.4 |

Meshing a 3D fibre / sphere into triangles has the same problem in two dimensions higher, so the element must be the curved primitive itself and its cost must not depend on the radius.

## The identities (any dimension: M(r) = int_0^r s^{d-1} W ds)

Disk (sphere) of radius R, centre at distance D from the query, y = x' - x, outward normal n of its boundary, Phi(r) = (M(r) - M(1)) / |S^{d-1}| (zero beyond the support):

    lam  = int W            = 1[D < R] + oint (n . y) Phi(r) / r^d dl       (value; `tier4.arc_value`)
    m1   = int y W          = oint n Phi(r) dl                              (div(e_k Phi) = Phi' y_k / r = y_k W)
    g0   = int grad_x W     = - oint n W(r) dl                              (grad_x = - grad_y)
    g1_{dj} = int y_d d_j W = - oint n_j y_d W dl + delta_dj lam

so every channel of the fused path is a boundary integral of W(r) and its radial primitive along the circle (2D) / sphere (3D, two angles), computed the same way for every kernel group (`w2`, `cone`, `lw2`, `w2p5`).
By the reflection symmetry about the axis through the query and the centre only five numbers survive in 2D: lam, m1 (along the axis), g0 (along the axis), g1 normal and tangential, hence the world-frame
tensors are built from the unit vector to the centre (as `planar_moments` of the half plane does).  The circle integrand is nearly singular where the circle passes close to the query (width |D - R| / sqrt(D R) in the
angle), so the reference cuts the angle at the knot radii and refines geometrically (ratio 4) towards the closest approach; the FUNCTIONS are smooth across the surface.

## Validated reference (`edge/disk.py`)

* value = the multi-precision circle identity to 3e-14 (R/H = 0.1 ... 6, near the surface the largest), 24 against 48 nodes: 1e-14;
* all five channels, in the world frame, against the polygon of the fused path (edges <= h/16): 1e-6 - 1e-7 of the scale for R/H = 0.3, 0.5, 1, 2, 6 (the polygon's own error);
* smooth across |D - R| = 0 (fourth differences of a grid straddling the surface equal h^4 f''''); limits (centre of a large disk: lam = 1; beyond the support: 0).

## Plan for the fused element

1. **Tables** (built with the reference, shipped as package data like the tier-3 tables): the five channels of each fused group as functions of (D, R), tensor-Chebyshev on panels aligned with the tangency lines D = |1 - R|, D = 1 + R.
   For R < 1 the functions scaled by R^2 are smooth in (D, R^2) down to the point limit (no separate series needed); for R >= 1 they are smooth in (D - R, 1/R) and reduce to the planar tier-3 functions at 1/R = 0.
   Evaluation: O(1) per (query, disk) pair, no dependence on R; accuracy target 1e-8 (the polygon path is 1e-6).
2. **Batched primitive** `DiskArrayRep` (centres [M, 2], radii [M], one body = a fibre bundle moving rigidly): a cell list over the centres, fixed capacity K = the most disks within H + R_max of a cell, so the
   adjacency is sync-free like `fixedadj`; one launch over (query, slot) pairs for ALL disks, not one launch family per body.  Per-fibre kinematics (each fibre its own body) stays possible through bodies
   with a one-disk rep, at the old per-body cost.
3. **Contraction** with the existing `_wall_contract_kernel` (same 9 channels per pair and group), `cone_area` of a disk (circle-circle-wedge area, closed form), signed distance / normal of the nearest disk for the
   no-penetration law (analytic, `Scene.signed_distance`).
4. **Tests:** against the reference to 1e-8, against the polygon path, the cross-representation test against wall particles (`ParticleBoundary`), a bundle in a channel with the permeability-style drag
   (`wallLoads` per fibre), graph == eager; timings against the table above.
5. **3D:** the same identities with the sphere (two-angle Gauss on the cap inside the support, same near-singular refinement) and the projected 2D kernel for infinite cylinders; finite cylinders need the end caps.

Open decisions: the table resolution per kernel (build time vs size), whether `cone_area` of a disk is tabulated or closed-form, and whether the per-fibre-body case should also use the element (probably yes: a one-disk rep).


## Result (2026-10-06): tables + `DiskArrayRep`

**Tables** (`edge/disktables.py`, shipped as `data/tables/disk_<kernel>_*.npz`, ~100 KB per kernel, built by the reference in about a second each): the channels (lam, m1x, g0x, g1xx, g1yy) as tensor-Chebyshev panels in
delta = D - R, with panel boundaries on the three lines where the functions are not analytic (the surface delta = 0: Wendland's r^3 term gives an eps^4 ln|eps| term, which is what limited a first uniform-panel
version to 1e-4; the tangency lines D = |1 - R| and D = 1 + R).  R <= 1: two radius branches x three delta intervals, tabulated f / R^2 (regular point limit) with the interval mapped through w = (2/pi) asin(sqrt(u)); R > 1:
(v, s = 1 / R), s -> 0 the planar half plane.  Resolution (S: 3 x 3 panels of 12 x 12, L: 6 x 3 panels of 12 x 12): `w2` 5e-8, `w4` 3e-8, `lw2` 5e-7 on lam / g1 (the Laplacian), `w2p5` (tensile) 2e-6, `cone` (cover
vector) 8e-5 (the cusp of 1 - r at r = 0 limits it; the detector only thresholds it); float32 against float64: 1e-6.  Evaluated at one (query, disk) slot per thread, all kernel groups in one launch.

**`DiskArrayRep`** (centres [M, 2], radii [M], one rigid body; disjoint disks): a static cell list over the centres (cell = support + largest radius, 3 x 3 cells, fixed capacity K = the most disks of any block),
one slot launch for the whole bundle, the channels in the BODY frame summed per query row in slot order by the existing contraction kernel; `ImplicitRep(DiskBody)` of a solid disk lowers to a one-disk bundle (a cavity
keeps the polygon).  The cone areas (detector) are closed form (antiderivatives of t^2 along the rays, segments cut at the tangent rays and the rays through the circle / support-circle intersections).

**Cost** (6000 queries, H = 4 dx, all fused groups, ms): one disk 0.7 - 0.8 f64 / 0.6 f32 at EVERY radius R/H = 0.25 ... 4 (polygon: 4 -> 40 f64); 16 fibres as one bundle 0.8 f64 / 0.6 - 0.7 f32 (16 bodies with
polygons: 62 -> 403 f64, 14 -> 30 f32; 16 one-disk bodies with the element: 9 f64 / 7.5 f32, the per-body launch overhead remains); 100 fibres as one bundle 0.9 - 2.5 f64 / 0.7 - 1.1 f32.

**Accuracy against the polygon path** (edges <= h/16): lam, G, Cov, A, lap 1e-6, tens 1e-5, cover 4e-5, cone areas 1e-5; a closed-form cone area against an independent numerical angular integral 1e-7 of H^2.

**Solver** (hydrostatic tank, three fibres of radii 0.08 / 0.10 / 0.12 as one body, conforming particle layers): the bundle equals the same fibres as separate bodies to 1e-9 in continuity, pressure force and loads (the
free-slip wall VISCOSITY normalises the wall direction per body, so it differs, by design, only for particles that see two fibres at once); total buoyancy within 3 % of rho g pi sum R^2; a translating bundle replays
bit-for-bit as a CUDA graph; the bundle is ONE `FusedWall` item.

**Not covered:** overlapping disks (the integrals add), per-fibre kinematics inside one bundle (separate bodies, at the per-body launch cost), cavities of circular shape (polygon), the 3D sphere / cylinder (same identities
with M(r) = int s^2 W; a two-angle cap quadrature builds the tables, a lookup in (D, R) again; infinite cylinders: the projected 2D kernel; finite cylinders need the end caps).
