# A box primitive for domain geometries: corner tables instead of an edge loop

Status 2026-10-05 (evening): **implemented** as `BoxRep` (`src/warpSPHBoundaries/scene/scene.py`, tables in `scene/box.py`, tests `tests/scene/test_box_rep.py`, `domain_scene('box', …)`). **[V]** = checked numerically, **[D]** = derived, not run, **[P]** = plan. Prompted by: *most domains are boxes (tanks, channels, flumes, baffles, floors); is there a direct treatment?* Yes, and it is cheaper and more stable than the polygon edge loop for exactly these cases.

## 1. The idea

For a radial kernel and an axis-aligned rectangle (in the body frame, rotated by the pose like every body), the integral over the rectangle is four lookups of one 2D function. With `y = x' − x` and the quadrant integral `Φ(a, b) = ∫_{y1 ≤ a, y2 ≤ b} W(|y|) dy`:

    ∫_box W = Φ(x1, y1) − Φ(x0, y1) − Φ(x1, y0) + Φ(x0, y0)        (a_i, b_j = box limits relative to the particle, clamped to [−H, H])

`Φ` saturates exactly: `Φ = 0` if `a` or `b ≤ −H`, `Φ(a, b) = Φ(H, b)` for `a ≥ H`. So the clamp makes the far field exact (no branch, no cell list, no pair list, no `atan` / `asinh` / `log`, no indicator pseudo-pair: a particle deep inside the box gets all four corners saturated and λ = 1, deep outside 0). The rectangle may be arbitrarily thin (a baffle of thickness < H is four corners, nothing is approximated by the clamp).

**All nine channels of `warpbc.edge_channels` come from two tables per kernel**, `Φ = ∫W` and `Φ1 = ∫ y1 W` (`Φ2(a, b) = Φ1(b, a)` by symmetry), and their first derivatives at the four corners. With `W(x − x')` radial, `∇_x W = −∇_y W`, and (by parts, the boundary term is a 1D marginal, i.e. a derivative of the table):

| channel | from the corners |
|---|---|
| λ = ∫W | Φ |
| m_d = ∫ y_d W | Φ_d |
| g0_j = ∫ ∂_xj W | −∂Φ/∂a_j |
| g1_dj = ∫ y_d ∂_xj W | −[boundary − δ_dj Φ];  boundary = a_j ∂_jΦ (d = j), ∂_j Φ_d (d ≠ j) |

The registered kernels of the solver (`w2`, `w4`, `lw2` for the wall Laplacian, `cone` for the cover vector, `w2p5` / `w4p5` for the tensile term) are just different tables; `cone`, `w2p5` need only the `g0` row (Φ and its derivative). The complement (a tank: solid outside the box) is the constants of the whole plane minus the box: `λ = 1, m = 0, g0 = 0, g1 = I` (the same constants the indicator pseudo-pair adds today), so the domain box costs the same four corners.

## 2. Measured [V]

`scripts/studies/box_quadrant_table.py` builds the tables by the exact stage-1 machinery (`E.value`, `E.mom` over the polygon `[−2, a] × [−2, b]`) as Chebyshev tensors on four panels (split at `a = 0`, `b = 0`, where the kernel's `r³` term is non-analytic), `w2`, H = 1, rectangle `[0,3] × [0,2]`, 140 queries inside / near walls / near the corners / outside:

| Chebyshev degree per panel | coefficients (per table) | max error λ | max error ∇λ |
|---:|---:|---:|---:|
| 8 | 324 | 8e-6 | 7e-4 |
| 16 | 1 156 | 6e-9 | 2e-6 |
| 24 | 2 500 | 3e-10 | 2e-7 |
| 32 | 4 356 | 5e-11 | 4e-8 |

All nine channels against the scene's `SurfaceRep.box` (the library's own edge kernel; 75 queries, degree 20): λ 1e-9, G (g0) 5e-7, Cov (g1) 3e-7, scales ≈ 1. The table is a controlled-error model of an exact quantity (tag T, tabulated): error falls spectrally with the degree, derivatives lose about two orders (they are the channels that matter for forces, so size the table by the gradient error). For FP32 (the production precision, `plan-wall-evaluation.md` §1) the value is at the float epsilon (6e-8) from degree ~12, the gradient needs degree ~24 – 32 or stored derivatives (open point 2), and **there is no cancellation to amplify** (the monomial edge plans lose 1e2 – 2e4× in float32; corner sums of bounded smooth tables do not).

## 3. What it does and does not cover

* **Covers:** every axis-aligned rectangle in the body frame, so rotated boxes through the pose; fluid inside (domain) or outside (obstacle); thin plates; a union of *disjoint* boxes by summing (tank with baffles = outside of the fluid box + baffle boxes inside it, which are disjoint from the outside by construction). Static or moving (rigid) bodies: only the pose changes, the tables never do.
* **Does not cover:** non-rectangular polygons (stay `SurfaceRep`), curved walls (tier 3 / 4, `ImplicitRep`), overlapping solids (the sum double counts, exactly as overlapping `SurfaceRep` loops would).
* **Corners are exact**, not special: the corner of a tank is two walls' worth of corner lookups inside the same inclusion–exclusion (`paper_exact_wall_corner.py`, part A, is about the *ghost field* at the corner, a separate question from the geometric integrals, which are exact here).
* **3D [D]:** `Φ(a, b, c)` = octant integral, eight corners, the same clamp, tables symmetric under permutations of the axes (a fundamental domain of 1/6); the channel table count is the same small number (value, first moment) because the boundary terms are again derivatives of lower-dimensional marginals of the same function. Storage: a Chebyshev tensor of degree 12 on eight octant panels is ~17 000 coefficients per table, or a uniform 64³ tricubic Hermite grid (tens of MB in float64 before the permutation symmetry, a few in float32) when a local, branch-free lookup is preferred over a degree-12 polynomial evaluation (13³ terms). Faces of a 3D box need no face → edge chain: that is the point.

## 3b. As implemented (2026-10-05)

* `BoxRep(lo, hi, solid='inside' | 'outside')`: one `RepMoments` row per query that sees the box (rows with all channels zero are dropped), put into `ent['implicit']` of `PairMoments`; `Scene.inside`, `Scene.signed_distance`, `cone_area_scene` (through `surface()`, the polygon of the same body), `cover_vector_scene`, `lap_lambda_scene`, `tensile_vector_scene` and `restrict` all accept it; `sceneOperation` and the reaction (force / torque) work unchanged.
* **Table grid:** Chebyshev degree 20 on eight panels per axis graded toward the origin (`BOX_BREAKS`), built at first use per kernel from `SurfaceRep.pairs` (the library's own edge kernel: the quadrant `[-3,0]²`, particle at `(-a,-b)`, `lam` and `m1_x`), 2 × 64 × 441 coefficients, ~0.3 s per kernel. The study's four-panel degree-32 grid is better for the smooth kernels (w2 1e-8 vs 9e-8 on g0) but poor for `lw2` (7.8e-5): the graded grid is the compromise (scan: `scripts/studies/box_quadrant_table.py` and the scratch comparison in the commit message).
* **Measured against the polygon path** (`test_box_rep.py`, 600 random queries incl. exact wall lines and corners, rotated body, per-query supports 0.3 – 0.7, both orientations; error / scale, scale 1 – 7): w2 lam 4e-11, g0 1e-7, Cov 1.5e-8; w4 6e-13, 1.7e-9, 2.4e-10; lw2 3e-10, 8e-7, 1.2e-7; w2p5 1e-14, 4e-11, 8e-14.
* **The kinked kernel `cone`** (profile `1 − q`: slope ≠ 0 at the support edge, so Φ has the kernel's own kink along the support circle, which no tensor panel aligns with; 7e-4 on g0 at degree 20, algebraic convergence) is listed in `BOX_EXACT_KERNELS` and takes the exact polygon path of `surface()`: the cover vector of the free-surface detector stays exact. The same mechanism covers any user kernel with such a kink. Why the circle limits the table: `∂²Φ/∂a∂b = W(√(a²+b²))`, so Φ has exactly the smoothness of the kernel on the circle `a² + b² = 1`; outside it Φ is additively separable (`Φ(a,b) = Φ(a,±1) + Φ(±1,b) − Φ(±1,±1)`, 1D tables), inside it is smooth: a polar-coordinate table or this split would restore the spectral rate for `cone` as well **[D, not built]**.
* **Not a bit-identical replacement:** the harness `check` is bit-level and was recorded with the polygon domain; a box domain changes results at the 1e-7 level, so it is validated by the physics gate and by direct comparison (below), not by the bit-level lines.

## 4. Where it fits

* **Scene path (no Warp needed):** a `BoxRep` representation whose `moments` returns one `RepMoments` row per query (no pairs), exactly like the implicit/SDF representations (`ent["implicit"]`), from tables built lazily per kernel the way `tier3Table` is (`implicitBodies.py`). Precompute = the corner lookups, evaluation unchanged. Test: against `SurfaceRep.box` on random queries incl. inside / thin / corner / tangent cases (lesson of WORK-004: the indicator is where silent errors hide), kernels `w2`, `w4`, `lw2`, `cone`, `w2p5`. Stop rule: if the gradient error at the table size that fits in the shared memory of one block exceeds 1e-7 of the scale in float64, split the panels further at `±H/2` before giving up.
* **Fused evaluation (step 3 of `plan-wall-evaluation.md`):** a box is one term group per query (4 corners × 2 tables), no per-(query, edge) pair structure: it removes the pair count that makes the edge kernel latency-bound for tank walls (a dam-break tank is 4 edges but 484 (query, edge) pairs at nx = 67, the latency probe of the plan; the box has one row per near-wall query). The same spec (kernel, channels, contraction) applies.
* **Exact wall Laplacian (`plan-exact-wall-laplacian.md`):** the tensor moments `T_β` of the pairwise operator and the Green moments of `∇²W` over a rectangle are again quadrant integrals of radial-profile × monomial; they are further tables (one per moment order) behind the same four-corner sum. The ghost-order hierarchy of §2 there (free-slip mirror per wall, corner partition) is then a fixed linear map applied to table values, no edge loop.
* **Domain definition:** a rectangular domain is `BoxRep(lo, hi, solid="outside")`; the validation cases (hydrostatic tank, Marrone dam break, SPHERIC sloshing, the DFSPH tank) are boxes, so every one of them would run without a single edge pair.

## 5. Open points

1. Table resolution vs the clamp: the panels are split at the origin only; the support boundary `r = 1` crosses the square `[−1, 1]²` and the kernel has a finite-order contact there, so a split at `±H/√2` or a 5-panel layout may converge faster. Measure before choosing.
2. Whether to store derivatives (Hermite) rather than differentiate the Chebyshev interpolant: stored derivatives make the gradient channels as accurate as the value.
3. A generic rectilinear (axis-aligned, non-disjoint) union via inclusion–exclusion on intersections: not needed for the validation cases; only if overlapping baffles appear.
4. Kernels registered at run time (`lw2`, `w2p5`, user kernels) need their tables built on first use: cost = (number of coefficients) × one `E.value`-class evaluation (~1 ms each in the study code, 2 s at degree 32); a Warp / torch evaluator of the exact edge form would make it instant and is already there (`warpbc.edge_channels` on a polygon with the particle at the origin).
