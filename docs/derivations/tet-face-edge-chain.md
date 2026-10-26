# Analytic tetrahedral volume integrals in 3D

**Status:** [V] local identities and the finite reference checks below; device conditioning remains [P].
**Tier(s):** 1, 2 · **Dimension:** 3D
**Depends on:** `moments-recursion.md`, `edge-primitives.md`, `truncated-monomials.md`
**Implemented in:** `src/warpSPHBoundaries/mesh3d/reference.py`, `mesh3d/fem.py`, browser `visualizer/static/mesh3d-analytic.js`
**Verified by:** `maple/22_mesh3d.mpl`, `tests/mesh3d/`, `scripts/studies/mesh3d_overlap.py`

## 1. Statement

Let `y = source - query`, `r = |y|`, and `h` denote the support radius.
For an outward-oriented tetrahedron or polyhedron `T`, define

\[
 M_3(r)=\int_0^r t^2 f(t)\,dt,\qquad
 F(r)=M_3(r)/r^3,\qquad z_f=n_f\cdot(p_f-x).
\]

The **regular flux identity** is

\[
 \boxed{\int_T f(r)\,dV=\sum_f z_f\int_f F(r)\,dA.}
\]

For the unnormalized constant block `f=1[r<R]`, this is precisely
`volume(T ∩ ball(x,R))`. For a truncated monomial `f=r^n 1[r<R]`, `n>=0`,

\[
 F(r)=\frac1{n+3}
 \begin{cases}r^n,&r<R,\\R^{n+3}r^{-3},&r\ge R.\end{cases}
\]

Every face integral reduces to elementary integrals on its three straight
edges. No tetrahedron/ball decomposition, sphere quadrature, or winding-number
classification is required. This particular regular form visits **all faces**,
including faces outside support; its exterior flux is nonzero.

## 2. Assumptions and validity

- A tetrahedron is nondegenerate; either vertex ordering is accepted and its
  four faces are oriented outwards using exact rational predicates.
- A general closed surface bounds a finite solid and is watertight and
  consistently oriented. Cavities point out of the solid, into the cavity.
  Edge incidence and positive total signed volume are checked; geometric
  self intersections, intersections between components, and overlaps are not.
- Kernels are the cubic spline and Wendland C2/C4/C6, with the **3D**
  normalizations from `curvbound.kernels`. Profiles are sums of truncated
  monomials with exact rational coefficients before numerical evaluation.
- `mpmath` precision is caller controlled. Construct and evaluate an element
  inside the same `mp.workdps` context. Recreate it when increasing precision;
  element caches retain values at the construction/evaluation precision.
- Faces, edges and vertices may pass through the query. Exactly zero plane
  and edge offsets are detected before square-root normalization.

## 3. Derivation

### 3.1 Volume to faces: regular and compact forms

For `r>0`,

\[
 \nabla_y\cdot(yF)=3F+rF'=M_3'(r)/r^2=f(r).
\]

For bounded `f`, `F` remains finite at the origin, and the flux of `yF` through
an infinitesimal sphere vanishes. At `r=R` the two branches of `F` agree, so
there is no distributional shell term. The divergence theorem gives §1,
including when the query is on the boundary.

The earlier compact identity is also useful **away from the boundary**:

\[
 \int_T f=4\pi M_3(R)\,1[x\in T]
 +\sum_f z_f\int_{f\cap B_R}
 \frac{M_3(r)-M_3(R)}{r^3}\,dA.
\]

Subtracting `M_3(R)y/r³` introduces a point source. On the boundary its solid
angle must replace the ordinary indicator. The regular implementation avoids
this convention and the associated cancellation of jump terms. A future
local-support/GPU implementation must restore this solid-angle bookkeeping
before pruning far faces.

### 3.2 Faces to edges without circular arcs

On a face, let `a=|z_f|`, `q=y-z_f n_f`, `rho=|q|`, so `r²=rho²+a²`.
For any locally integrable radial face profile `g(r)`, set

\[
 H(r)=\int_a^r t\,g(t)\,dt,\qquad
 B(\rho)=H(\sqrt{\rho^2+a^2})/\rho^2.
\]

Then `div_face(q B)=g(r)`. The limit at `rho=0` is regular for all profiles
used here (including `g=r^-3` when `a>0`). On an oriented face edge,

\[
 q=w_e\mu_e+s t_e,\quad \mu_e=t_e\times n_f,\quad
 \boxed{\int_f g\,dA=\sum_e w_e\int_{s_0}^{s_1}
   \frac{H(r(s))}{s^2+w_e^2}\,ds.}
\]

For `g=r^n 1[r<R]`, `a<R`, `n != -2`,

\[
 H(r)=\frac{\min(r,R)^{n+2}-a^{n+2}}{n+2}.
\]

If `a>=R`, the face integral is zero. Split the **edge parameter** at
`s=±sqrt(R²-a²-w²)` when these roots lie on the edge. Outside the face disk,
`H` is constant; those straight-edge pieces are retained. They encode the
circle contribution even when the disk lies completely inside the triangle.
Dropping them would give zero for that configuration.

Write `S_f(n,R)=∫_f r^n 1[r<R] dA` and `S_f(-3,∞)=∫_f r^-3 dA`. The explicit
volume block is

\[
 \boxed{V_{n,R}=\frac1{n+3}\sum_{f:z_f\ne0}z_f
 \left[S_f(n,R)+R^{n+3}\{S_f(-3,\infty)-S_f(-3,R)\}\right].}
\]

The `z_f=0` flux is exactly zero and is skipped **before** evaluating the
singular `r^-3` auxiliary profile. This does not discard a finite jump: the
original regular flux on that plane is zero.

### 3.3 Elementary edge primitives

Set `b=|w|>0`, `c²=w²+a²`, `r²=s²+c²`, and define

\[
 K_m=\int\frac{r^m}{s^2+w^2}\,ds,\qquad I_m=\int r^m\,ds.
\]

A closed recurrence is

\[
\begin{aligned}
 K_0&=b^{-1}\arctan(s/b),\\
 K_{-1}&=(ab)^{-1}\arctan\!\frac{as}{br},&&a>0,\\
 K_{-1}|_{a=0}&=s/(b^2r),\\
 K_1&=\operatorname{asinh}(s/c)+a^2K_{-1},\\
 K_m&=I_{m-2}+a^2K_{m-2},&&m\ge2.
\end{aligned}
\]

The recurrence follows by dividing `r²=(s²+w²)+a²` by `s²+w²`. For a face
block, the interior antiderivative is `(K_(n+2)-a^(n+2) K_0)/(n+2)`; the
exterior antiderivative is `(R^(n+2)-a^(n+2))K_0/(n+2)`.
Exactly `w=0` edges have zero regular planar flux and are skipped. Nearly zero
offsets still need sufficient arithmetic precision; replacing this test with
a tolerance is not justified.

### 3.4 All polynomial moments

For `alpha=beta+e_i` and the continuous compact potential

\[
 \Phi[f](r)=-\int_r^R t f(t)\,dt,
 \quad \Phi[r^n1_{r<R}]=\frac{r^{n+2}-R^{n+2}}{n+2}1_{r<R},
\]

`partial_(y_i) Phi = y_i f`, hence

\[
 \boxed{M_{\beta+e_i}[f]=\sum_f n_{f,i}S_{f,\beta}[\Phi]
       -\beta_i M_{\beta-e_i}[\Phi].}
\]

Degree drops by two in the volume term. Expand the face monomial in local
coordinates `y=z n+u e1+v e2`. Apply the same recursion to `(u,v)` on the
triangle, with the in-plane outward edge normal `mu`. Its boundary terms are
polynomials in `s` times `r^m`; the existing `S_(j,m)` primitives close them.
The terminal scalar face/volume terms are §§3.1–3.3. Thus this is a finite
analytic algorithm for every polynomial degree, rather than just `M0/M1/M2`.
The reference tests moments through degree four and FEM through degree three.

### 3.5 Query derivatives and the full moment contract

Let `M_alpha=∫_T y^alpha W`, `S_(f,alpha)=∫_f y^alpha W`, and
`g_(alpha,j)=∫_T y^alpha partial_(x_j) W`. Because the solid is fixed,

\[
 \boxed{\partial_{x_j}M_\alpha=-\sum_f n_{f,j}S_{f,\alpha},\qquad
 g_{\alpha,j}=\partial_{x_j}M_\alpha+\alpha_jM_{\alpha-e_j}.}
\]

The first derivative differentiates the complete translated moment; the
second differentiates only the kernel. This distinction is essential for
FEM weights. The 3D analogue of the 2D twelve-channel contract contains
**22 stored channels**: `M0` (1), `M1` (3), symmetric `M2` (6), `g0` (3),
and the full `g1` matrix (9 stored entries). For radial kernels `g1` is
symmetric, giving 19 independent components across the 22-entry layout.
On the whole support `g1=I`, not `-I`,
with the convention `y=source-query`.

For `L(r)=W'(r)/r`, Green's identity gives

\[
 \boxed{\int_T y^\alpha\Delta W
 =\sum_f z_f S_{f,\alpha}[L]
 -\sum_{f,j}\alpha_j n_{f,j}S_{f,\alpha-e_j}[W]
 +\sum_j\alpha_j(\alpha_j-1)M_{\alpha-2e_j}[W].}
\]

The kernel Hessian moments are

\[
 \boxed{\int_T y^\alpha\partial_{x_i x_j}W
 =\sum_f n_{f,i}S_{f,\alpha+e_j}[L]
 +\alpha_i g_{\alpha-e_i,j}.}
\]

These expressions apply to the assembled smooth kernels. Individual truncated
blocks jump at their cutoffs; their distributional derivative terms cancel
only after reconstructing the continuous kernel. For the cubic spline some
`L` blocks have power `-1`; the planar recursion supports this integrable
profile, including a query on the face.

### 3.6 P0–P3 tetrahedral FEM

For degree `p>0`, each integer tuple `b=(b0,b1,b2,b3)`, `sum b_i=p`, defines
node `lambda=b/p` and

\[
 N_b(\lambda)=\prod_{i=0}^3\frac1{b_i!}
              \prod_{k=0}^{b_i-1}(p\lambda_i-k).
\]

P0 has one constant centroid basis. Nodes are in descending lexicographic
order of `b`; counts are 1, 4, 10, 20. Exact rational barycentric affine
functions give `N_b(x+y)=sum_alpha B_(b,alpha)y^alpha`. Then

\[
 w_b=\sum_\alpha B_{b,\alpha}M_\alpha,\qquad
 G_{b,j}=\sum_\alpha B_{b,\alpha}g_{\alpha,j}.
\]

These integrate a prescribed nodal field and its kernel gradient. Basis
coefficients depend on the query; simply differentiating `M_alpha` while
holding those coefficients fixed would omit terms.

## 4. Special cases and limits

- Constant block: geometric sphere/tetrahedron overlap, with no normalization.
- Whole support: `M0=1`, `M1=0`, `M2=(4pi/3)∫r⁴W dr I`, `g0=0`, `g1=I`,
  `Delta M0=0`, and the integrated Hessian vanishes.
- At an orthogonal tetrahedral vertex, edge, or face, sufficiently small
  supports give respectively `1/8`, `1/4`, and `1/2` normalized mass.
- A tetrahedron with only one face near the query agrees exactly with the
  existing 3D planar closed forms, including the inside complement.
- For a homogeneous dilation of geometry, query and support by `h`,
  volume `M_alpha` scales as `h^|alpha|`, kernel-gradient moments as
  `h^(|alpha|-1)`, Hessian/Laplacian moments as `h^(|alpha|-2)`.
- Contributions on a shared tetrahedral face cancel for a common global
  polynomial integrand. For a discontinuous element field, its traces need
  not cancel; evaluate the volume elements separately.

## 5. Checks

| Check | Independent method | Absolute tolerance / scope | Reproducer |
|---|---|---|---|
| Radial volume/face divergence, compact recursion, edge primitive derivatives | Maple symbolic algebra | exact; radial powers through 14, moment degree ≤4 | `maple/22_mesh3d.mpl` |
| P1–P3 partition and nodal values, triangles and tetrahedra | Maple and exact rational Python | exact | Maple and `test_reference.py` |
| Primitive derivatives, zero plane offset | arbitrary-precision differentiation | 1e-35 | `test_new_edge_primitives` |
| Polynomial tetrahedral moments | simplex factorial integrals, even radial polynomial expansion | 1e-35 | `test_tetra_polynomial_moments_exact` |
| Planar, whole support, tangency, face/edge/vertex queries | existing closed forms and exact angular fractions | 1e-32 to 1e-35; all four kernels | `tests/mesh3d/test_reference.py` |
| Oblique volume and area moments / kernel gradients | direct Gauss/Duffy simplex integration, factored kernel shapes, 40→80 nodes | 2e-7 agreement; separate refinement gate | `test_direct_simplex_quadrature` |
| Laplacian and Hessian moments | direct volume quadrature; finite-difference gradients; Hessian trace | 2e-5 quadrature, 1e-15 finite differences | `test_reference.py` |
| Tetrahedral subdivision / closed boundary | sum four subtetrahedra and cancellation of internal faces | 1e-32 | `test_tetrahedral_subdivision_and_boundary_surface_agree` |
| Volume-mesh boundary extraction, cavities and rigid transforms | common indexed faces, shell subtraction, exact translated/rotated geometry | 1e-32 | `test_reference.py` |
| Sphere mesh convergence | inscribed octahedral refinements vs existing exact sphere formula | monotone mass convergence, three Wendland kernels | `test_sphere_mesh_convergence` |
| Constant-kernel overlap volume | authors' independent sphere/element decomposition | ordinary regression cases: 5e-15; adversarial differences recorded, not hidden | optional `test_overlap_optional.py`, `mesh3d_overlap.py` |
| Difficult overlap discrepancies | positive clipped-simplex quadrature, independent of face reduction | record quadrature estimates and refinement separately | `mesh3d/oracle.py`, overlap study |

The overlap comparison uses Strobl, Formella and Pöschel,
[Exact calculation of the overlap volume of spheres and mesh elements](https://doi.org/10.1016/j.jcp.2016.02.003),
J. Comput. Phys. 311 (2016), 158–172, and their
[MIT-licensed implementation](https://github.com/severinstrobl/overlap).
The external implementation uses float64 and is an independent comparison,
not an exact-arithmetic oracle. Large differences on very thin elements need
another reference rather than automatic attribution to either method.

Run:

```bash
python -m pytest tests/mesh3d -q
maple -q maple/22_mesh3d.mpl
python scripts/studies/mesh3d_overlap.py --output /path/to/external/evidence
python scripts/studies/mesh3d_slices.py --output /path/to/external/slices
```

`overlap` is an optional study/test dependency. An isolated installation may
be selected with `--overlap-path`; ordinary package use does not require it.
Study output follows [validation.md](../validation.md) and stays outside the
source tree.

## 6. Known failure modes

Regular fluxes cancel between distant faces and on thin/sliver elements.
Polynomial kernel coefficients and FEM reconstruction can add cancellation.
Arbitrary precision establishes a reference path; it does **not** establish a
stable float32/float64 implementation, error bound, or device throughput.
The regular flux also prevents simply culling every face outside support.
Geometry validation does not establish embeddedness of a general mesh.

### Browser closed-form evaluation

`visualizer/static/mesh3d-analytic.js` ports the elementary reductions above
to JavaScript float64. `Solid · analytic face-to-edge` evaluates the original
tetrahedron or box, with moments through degree two and all 30 displayed
moment/gradient/Hessian selections. It does not call the support cubature or
Gauss rules. Empty and full support have analytic branches; partial support
uses the split edge antiderivatives. The constant block API also evaluates
unnormalized sphere–tetrahedron overlap volumes.

The tetrahedron/box browser presets use this model as view 1, with support
cubature and the SDF half-space kept as separate comparisons. The model runs
in compensated WebGPU fp32 or in a float64 worker. The shader compiles
combined radial profiles on the host and reuses elementary edge primitives
across coefficients. Its operations and storage require no GPU float64 feature. Downloaded
volume grids use float32 and display textures use float16; the analytic core
uses paired fp32 values on WebGPU or float64 in the worker. Refining the display grid changes sampling of
the field, not the underlying point integral, and cubature quality does not
affect the analytic model.

The algorithm is exact in real arithmetic for nondegenerate supported
polyhedra and piecewise-polynomial radial profiles. This statement does not
certify floating-point geometry predicates or cancellation. The Python
reference retains rational predicates and caller-controlled precision;
the worker uses ordinary float64 geometry and compensated sums, while the GPU
uses paired fp32 geometry, coefficients, primitives and elementary functions.
A runtime arithmetic check rejects compilers that erase compensation residuals. In the
finite regression cases, all 30 channels for all four kernels, including
boundary queries and small offsets to a face, agree with 65-digit reference
values to an absolute/relative tolerance of 1e-9. Constant-block overlap and
simplex polynomial checks use tighter tolerances. These tests do not imply
a universal float64 error bound for arbitrarily thin tetrahedra or tangencies.

Reproduce with `python -m pytest tests/visualizer/test_analytic_mesh_browser.py -q`.

## 7. Open questions

Conditioned floating-point compilation, solid-angle-based local support
pruning, vertex/shape derivatives, tolerance-driven error estimates, scene
integration and Warp execution are not implemented; browser WebGPU evaluation
is available for the displayed tetrahedron/box model family. Curved implicit
surfaces and 3D slender primitives remain separate derivations. The current
mesh work is the analytic tier-1/tier-2 reference foundation.
