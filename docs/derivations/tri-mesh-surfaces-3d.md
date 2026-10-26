# Triangular mesh surfaces in 3D

**Status:** [V] analytic reference and finite tests; production/device evaluation [P].
**Tier(s):** 2 (closed boundaries); also standalone area elements · **Dimension:** 3D
**Depends on:** `tet-face-edge-chain.md`
**Implemented in:** `mesh3d.reference::Triangle`, `ClosedSurface`; `mesh3d.fem`
**Verified by:** `tests/mesh3d/`, `maple/22_mesh3d.mpl`

## 1. Statement

A triangle surface supports two distinct operations:

\[
 S_\alpha[f]=\int_{\triangle}y^\alpha f(|y|)\,dA,
 \qquad
 M_\alpha[f]=\int_{\mathrm{solid}(\mathcal S)}y^\alpha f(|y|)\,dV.
\]

`Triangle` evaluates the first on an individual triangle, including open
surface meshes. `ClosedSurface` evaluates the second using a watertight
outward-oriented triangle mesh, without tetrahedralizing its interior.
Volume and area have different units; they are not interchangeable boundary
weights. Both use the elementary edge primitives and recursions in the
[tetrahedral derivation](tet-face-edge-chain.md).

## 2. Assumptions and validity

A triangle must have nonzero area. Reversing its winding reverses its normal
and co-normals but preserves scalar area integrals and kernel-gradient
integrals. A closed mesh must satisfy the oriented-manifold conditions in
the tetrahedral derivation. Cavities require the opposite orientation to an
outer shell. Local convexity is not required by the divergence theorem.
The reference checks edge incidence, winding consistency, and positive signed
volume, but does not find self intersections or overlapping solids.

A boundary trace does not define a general interior field. ClosedSurface
volume moments integrate known global polynomials; arbitrary elementwise
interior FEM fields still require a volume mesh. P1–P3 fields defined on
triangles are surface fields and can be integrated directly with `Triangle`.

## 3. Derivation

### 3.1 Surface radial and polynomial integrals

For plane distance `a=|z|` and in-plane radius `rho`, `r²=a²+rho²`. For each
truncated radial monomial, use

\[
 \int_\triangle r^n1_{r<R}\,dA
 =\sum_e\frac{w_e}{n+2}\int_{s_0}^{s_1}
 \frac{\min(r(s),R)^{n+2}-a^{n+2}}{s^2+w_e^2}\,ds
\]

when `a<R`, and zero otherwise. Break each straight edge at its support
crossings, retaining its exterior pieces. This formula includes full-disk
containment, an edge/vertex at the projection of the query, tangency, and a
query in the triangle plane. Its derivation is §3.2 of the tetrahedral note.
It supports integer `n>=-1`; the `-1` block is needed by the cubic spline's
assembled radial derivative. The auxiliary `n=-3` profile is used only for
volume flux and only at positive plane distance.

Expand an ambient polynomial in the plane basis `y=z n+u e1+v e2`. For
`beta=gamma+e_i`, `i` in `{u,v}`, the exact in-plane recursion is

\[
 \int_\triangle q^\beta f
 =\sum_e\mu_{e,i}\int_e q^\gamma\Phi[f],ds
 -\gamma_i\int_\triangle q^{\gamma-e_i}\Phi[f],dA.
\]

Here `Phi=-∫_r^R t f(t)dt`; since `partial_(q_i)Phi=q_i f`, plane distance
does not change this recurrence. Edge polynomials are integrated with the
existing `S_(j,m)` primitives. The scalar terminal integral is the formula
above. Circular boundary arcs are already accounted for by the nonzero
exterior straight-edge flux.

### 3.2 Surface kernel gradients

For a smooth assembled kernel, `L=W'(r)/r`,

\[
 \boxed{\int_\triangle y^\alpha\partial_{x_j}W\,dA
             =-S_{\alpha+e_j}[L].}
\]

This is the full ambient gradient, including its normal component. Applying
only the 2D boundary-gradient identity in the face plane would omit that
normal component. As for volumes,

\[
 \partial_{x_j}S_\alpha[W]
 =\int_\triangle y^\alpha\partial_{x_j}W\,dA
 -\alpha_j S_{\alpha-e_j}[W].
\]

### 3.3 Surface FEM

Use the tetrahedral lattice formula with **three** barycentric coordinates.
P0–P3 have 1, 3, 6 and 10 nodes. `mesh3d.fem.expansion` obtains exact affine
barycentric gradients from the rational Gram matrix of the embedded triangle;
the extension is constant in the face-normal direction. This makes the
expansion valid on the triangle. Exact coefficient contraction gives

\[
 w_b^{\rm area}=\sum_\alpha B_{b,\alpha}S_\alpha[W],\qquad
 G_{b,j}^{\rm area}=-\sum_\alpha B_{b,\alpha}S_{\alpha+e_j}[L].
\]

### 3.4 Volume from a closed triangular surface

Sum the regular volume flux on the outward boundary. The moment recursion
and its gradient, Hessian and Laplacian formulas require only integrals on
those boundary triangles; they do not require a hidden tetrahedralization.

For any conforming tetrahedral partition with a common global polynomial
integrand, each interior face appears twice with opposite normals and equal
scalar/area moments. Its contributions cancel in the regular value flux,
compact moment recursion and all boundary derivative formulas. This proves
the equivalence of tetrahedral volume and closed-surface representations.
For different traces across an element face, preserve the weighted jump;
interior cancellation is not valid for a general discontinuous field.

Boundary data alone also gives quantities such as
`∫_boundary A n W dA`, by contracting triangle FEM weights with each constant
face normal. Interpreting that as a volume operator requires the relevant
integration-by-parts identity and an interior extension of `A`.

## 4. Special cases and limits

- For a triangle containing the entire planar support disk,
  `S0[W]=2pi ∫_|z|^h r W(r)dr`; its gradient points normally to the plane.
- At `z=0`, the normal kernel-gradient component is zero for these kernels.
  Scalar mass remains a surface area integral, not half a volume integral.
- Under homogeneous dilation, surface moments scale as `h^(|alpha|-1)`
  for the normalized **3D** kernel, and surface kernel-gradient moments as
  `h^(|alpha|-2)`.
- Reversing triangle winding preserves area weights. Reversing all faces of
  a bounded solid reverses its volume orientation and is rejected.
- Whole-support volume moments and orthogonal vertex/edge/face fractions
  agree with the tetrahedral tests.

## 5. Checks

| Check | Method | Tolerance / scope | Where |
|---|---|---|---|
| Local recursions, primitive closure and P1–P3 basis | exact Maple and rational Python | exact | `22_mesh3d.mpl`, `test_reference.py` |
| Oblique embedded triangle moments / full ambient gradient | independent area Gauss/Duffy integration, 40→80 nodes | 2e-7 and separate refinement check, four kernels | `test_direct_simplex_quadrature` |
| Area, winding reversal, in-plane query, support clipping and scale | exact area/disk expressions and geometry transformations | 1e-35 | `test_triangle_orientation_area_scaling_and_boundary_projection` |
| Surface P0–P3 reproduction | exact nodal values, moment and gradient contractions | 1e-32 | `test_exact_lagrange_basis_and_fem_reproduction` |
| Tetrahedral subdivision vs boundary-only volume | cancel internal triangles | 1e-32 | `test_tetrahedral_subdivision_and_boundary_surface_agree` |
| Sphere/triangle overlap areas | authors' independent `overlap_area` implementation | record numerical differences, especially near degeneracy | `mesh3d_overlap.py` |

## 6. Known failure modes

The reference uses arbitrary precision and visits all faces. It is not yet
a conditioned surface GPU backend, and far-face pruning is invalid for this
regular volume flux. Near-coplanar/sliver cancellation and small-element FEM
conditioning need an explicit finite-precision policy. Open/nonmanifold meshes
do not define the closed-solid volume operation. Embeddedness is caller owned.

## 7. Open questions

Mesh acceleration, compact solid-angle bookkeeping, GPU compilation and surface
shape adjoints remain follow-up work. Inscribed sphere meshes have a finite
mass-convergence check against the three existing Wendland sphere formulas.
The present implementation establishes analytic tier-1/tier-2 evaluation and
its independent verification harness.

## API example

```python
import mpmath as mp
from warpSPHBoundaries.mesh3d import Tetrahedron, TetrahedralMesh, Triangle, ClosedSurface

with mp.workdps(50):
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tet = Tetrahedron(vertices, query=("0.2", "0.2", "-0.1"))
    mass = tet.moment("w4", h="0.5")
    overlap_volume = tet.block_moment((0, 0, 0), 0, mp.mpf("0.5"))
    second_xy = tet.moment("w4", (1, 1, 0), h="0.5")
    gradient = tet.kernel_gradient("w4", h="0.5")
    weights, grad_weights = tet.nodal_weights("w4", degree=3, h="0.5", gradient=True)
    skin = ClosedSurface(vertices, tet.indices, tet.query)
    assert mp.almosteq(skin.moment("w4", h="0.5"), mass)
    area = Triangle(vertices[:3], tet.query).moment("w4", h="0.5")
    volume_mesh = TetrahedralMesh(vertices, [(0, 1, 2, 3)], tet.query)
    boundary = volume_mesh.boundary()  # cancels shared faces for larger meshes
```
