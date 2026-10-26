# FluxBridge native operator baseline

Implemented in `src/warpSPHBoundaries/fluxbridge`. This is an independent
library checkpoint, **not completion of all Stages 0–4**. See
[coverage](fluxbridge-coverage.md) and the [implementation plan](plan-porous-coupling.md).

## Evaluate and compose

```python
import numpy as np
from warpSPHBoundaries.fluxbridge import (
    Geometry, Evaluator, Request, baseline, PolynomialField, FieldEvaluator,
    ResistanceMaterial, local_response,
)

geometry = Geometry([[[0, 0], [2, 0], [0, 2]]], dimension=2)
evaluator = Evaluator(geometry, baseline(2), kernel="w2", precision="float64",
                      device="cpu", compact=True)
batch = evaluator.prepare([[0.2, 0.2]], supports=1.0)
batch.launch()                     # reuses buffers; no allocation/host dispatch per query
values = batch.numpy()             # explicit readback

basis = PolynomialField.basis(geometry, degree=3)
field = FieldEvaluator(basis, kernel="w2")
weights = field.evaluate([[0.2, 0.2]], 1.0)

material = ResistanceMaterial(
    geometry, resistance_controls=[[np.diag([2.0, 5.0])]],
    skeleton_controls=np.zeros((1, 3, 2)),
    resistance_degree=0, velocity_degree=1,
)
interaction = material.prepare([[0.2, 0.2]], 1.0)
loads = interaction.apply([[1.0, 0.0]], masses=[1.0])
velocity, mobility = local_response(
    loads.resistance, [[1.0, 0.0]], loads.skeleton_product, dt=0.01,
)
```

Polygons are CCW-normalized; tetrahedra accept either orientation. Supply a
list of additive disjoint volume elements; no cancellation of different material
traces occurs. `Geometry.closed_surface(vertices, faces)` checks oriented
incidence and signed volume in static preparation. Self-intersections and
geometric overlap remain caller-owned. Open triangles use
`Geometry([triangle], 3, measure="surface")` and the **3D** kernel against area.
They cannot instantiate the bulk resistance provider.

A request is `∫ y^alpha ∂x_derivative W_h dμ`, `y=source-query`.
`baseline(2)` has the twelve stored moments plus three symmetric Hessian entries;
`baseline(3)` has twenty-two plus six. Kernel gradients use query axes.
`Request((0,0), (0,1))` is a mixed Hessian entry; summing diagonal requests is
the kernel Laplacian. Scalar interpolation is also occupancy for the unit field.
Vector/tensor interpolation contracts each component independently.
`differential(field, queries, supports, hessian=True)` additionally provides
explicit gradient, vector divergence/curl, Hessian and trace. Its derivative
batches share adjacency but are not a fused jet kernel.

Global element-local field coefficients are monomials in `source-origin`, with
`origin=first vertex`. Exact rational Lagrange and Bernstein basis construction
supports P0–P3 straight simplices. Products retain degree up to nine. The device
translates coefficients for each query, so a kernel-gradient contraction of a
fixed physical field is its complete query derivative. Raw translated moments
instead obey `∂x_j M_alpha = G_alpha,j - alpha_j M_(alpha-e_j)`.

`functions(precision).element` and `.output` in `fluxbridge/wp_func.py` are
allocation-free `wp.func` entry points. The prepared `Evaluator.plan`, facet CSR,
request scales and caller-owned disjoint scratch rows form the array contract.
Batch and fused field/material consumers use those same functions. The API has
explicit fp32 and fp64 factories independent of Core's global precision. Reverse
shape AD is not implemented; analytic kernel derivatives are requested outputs.

## Compact support without losing regular flux

For a truncated radial monomial, the regular scalar flux has exterior tail
`R^(n+d) r^-d/(n+d)`. The exhaustive route keeps it on every facet.
Let `Ω` be the oriented boundary angle (2D) or solid angle (3D), omitting exactly
coplanar facets. Then the compact scalar block is

```text
V_n,R = [R^(n+d) Ω + Σ_facets z (S_n,R - R^(n+d) S_-d,R)] / (n+d).
```

`S` is a line integral in 2D and a triangle area integral in 3D. In 3D each
facet angle uses `2 atan2(det(a,b,c), |a||b||c|+(a·b)|c|+(b·c)|a|+(c·a)|b|)`.
In 2D it uses `atan2(cross(a,b), a·b)`. Exactly query-containing facets have
zero regular flux and are omitted before singular auxiliaries. Other facets'
angles give the material wedge/solid-angle fraction at a face, edge or vertex.
In the open interior `Ω` is 2π or 4π; it is zero outside, with oriented cavity
contributions. This is a boundary-angle computation, not an inside flag.

For every weighted moment choose `alpha=beta+e_i` and recurse:

```text
V_alpha[n] = (B_i,beta[n+2] - R^(n+2) B_i,beta[0]
              - beta_i (V_(beta-e_i)[n+2] - R^(n+2) V_(beta-e_i)[0])) / (n+2),
B_i,beta[n] = Σ_facets normal_i S_beta[n].
```

The degree decreases, terminating in the **corrected scalar block for each
radial power**, not a single occupancy correction reused for every moment.
Full containment thus recovers even moments despite no support/facet hits.
The independent whole-support tests check every monomial through degree nine
and its kernel derivatives against gamma-function angular integrals.

For 3D surface moments the new ambient-coordinate recurrence uses tangential
projector `P_ij=delta_ij-normal_i normal_j` and `z=normal·y`:

```text
S_alpha[n] = z normal_i S_beta[n]
 + (Σ_edges mu_i (L_beta[n+2]-R^(n+2)L_beta[0])
    - Σ_j beta_j P_ij (S_(beta-e_j)[n+2]-R^(n+2)S_(beta-e_j)[0]))/(n+2).
```

This is independently compared with the reference's local-plane polynomial
expansion. Line integrals and scalar terminal face integrals use elementary
antiderivatives. No native analytic route contains a quadrature rule.

Compact open-triangle evaluation initializes face scratch before rejecting a
support-disjoint or tangent triangle. Poisoned-scratch tests ensure empty area
intersections produce zero rather than stale moments, including all ambient
Hessian channels.

A static median AABB tree constructs conservative **whole-element** adjacency
on the host. Selected elements retain all necessary flux; the compact route
also skips support-disjoint facet AABBs after angle accumulation. Every selected
skin still pays for angle enumeration. This is not a hierarchical solid-angle
search, nor CUDA-native adjacency. Periodic adjacency enumerates unique
`(query, element, image)` tuples for arbitrary support, including multi-cell
supports. Prepared adjacency signatures reject query/support/geometry changes
and mismatching requested periods. Rebuild on motion, shifting or revision.

## Work, reaction and mobility

For supplied masses, use the common nonlocal dissipation functional

```text
E = 1/2 Σ_i m_i ∫ W_i (u_i-us(s))^T D(s) (u_i-us(s)) ds.
A_i = ∫ W_i D,    b_i = ∫ W_i D us.
f_i = -m_i(A_i u_i-b_i).
f_a^solid = Σ_i m_i ∫ N_a W_i D (u_i-us).
```

`ResistanceMaterial` requires PSD Bernstein control tensors; convex Bernstein
weights imply pointwise PSD through P3. It does not infer positivity from cubic
Lagrange nodal values. Resistance, permeability and porosity remain different
models. `D=nu K^-1` is a constant-density example; polynomial permeability is
not inverted coefficientwise. Velocity and test bases are Lagrange P0–P3.
Cubic resistance, velocity and tests require degree-nine products.

The native fused contraction, deterministic reverse-adjacency reaction kernel
and local response share the prepared arrays. `apply_device` consumes explicit
Warp velocity/mass arrays without allocation; the convenience `apply` uploads
and reads back. Reaction is reported for a fixed skeleton. General
`FieldBatch.transpose(loads, volumes)` returns the coefficient-space weighted
transpose. Tests check common work, partition, force balance and dissipation.

**Angular momentum is a known gap:** these kernel-distributed forces are
noncentral. Equal-and-opposite force and work do not establish torque balance
or rigid-rotation objectivity. `DragResult.torque_defect` reports the defect for
nonperiodic P1–P3 test bases using physical nodal positions. No corrective couple is
invented. The point-only model is not an angular-momentum conserving coupling;
the separate supplied-affine contract below includes the missing gradient work.
Moving/rotating solid Scene applications are independent.

`response_function(precision)` exposes a reusable local mobility function;
`local_response` applies `(I+dt D)^-1(u+dt forcing)`. The pressure operator must
use the corresponding response consistently in Stage 5. This local action
alone is not an incompressible porous-flow solver.

## Affine material work and angular momentum

`AffineResistanceMaterial` is an opt-in alternative, requiring P1–P3 skeleton
and test bases. The caller supplies `L_i` in the physical reconstruction
`u_i(s)=u_i+L_i(s-x_i)`. The integration geometry stays fixed. No gradient
reconstruction, pressure operator or time integrator is installed here.

For `r_i=u_i+L_i y-us(s)`, differentiate the same quadratic work:

```text
E = 1/2 Σ_i m_i ∫ W_i r_i^T D r_i.
f_i = -m_i ∫ W_i D r_i.
M_i = -m_i ∫ W_i (D r_i) ⊗ y.
f_a^solid = Σ_i m_i ∫ N_a W_i D r_i.
```

The tensor index order of `M` is `[query, force component, gradient axis]`.
`DragResult.gradient_load` returns `M`; its power is `M:L`. Force, work and
angular-momentum balance include this term. For rigid rotation, the gradient
work supplies the couple `Σ_j e_j × M[:,j]`. This is derived from the work form,
not an invented corrective torque. The source-coordinate polynomial integrals
needed for it stay within degree nine for cubic material and skeleton fields.

If a **supplied linear** reconstruction gives `L=G u`, the physical particle
force is `f+G^T M`. `pullback_gradient(result, transpose_action)` applies this
chain rule. The supplied transpose must correspond to the same G and force/work
pairing; masses are already included in the generalized loads. Constant and
linear reproduction by G are required for the demonstrated force/torque result.
Tests use a small independent affine-reproducing operator to check virtual work,
force and torque after pullback, and check zero drag under a common rigid
translation/rotation and tensor covariance under a change of frame.

```python
from warpSPHBoundaries.fluxbridge import AffineResistanceMaterial

material = AffineResistanceMaterial(
    geometry, [[np.diag([2.0, 5.0])]], np.zeros((1, 3, 2)),
)
result = material.prepare([[0.2, 0.2]], 1.0,
                          precision="float32_compensated").apply(
    [[1.0, 0.0]], [1.0], velocity_gradient=[[[0.0, 0.0], [0.0, 0.0]]],
)
# Retain result.gradient_load and apply the supplied reconstruction's transpose.
```

For a fixed supplied L, `affine_forcing` returns `∫D us W - ∫D(L y) W` for the
local response. If L depends on velocities, the full response also depends on G
and its transpose; it is not just the local matrix `(I+dt A_i)^-1`. Solver-side
integration and reconstruction acceptance remain Stage 5. Periodic/global torque
requires compatible image bookkeeping; the torque diagnostic is nonperiodic.
The original point-only `ResistanceMaterial` remains available with its explicit
noncentral torque defect.

## Radial consumers and porous reference

The existing Scene viscosity code uses registered profiles `L=W'/r` and its
specific contraction `2 lambda[L]-trace(Cov[L])`, with dimension and normalization
factors. Shifting uses `(1/5) grad ∫ W^5` (`w2p5`/`w4p5`, degree 25/40 radial
polynomials). The cover consumer uses a cone profile. Those are **not** implied
by the new W-Hessian API. Their existing Scene implementations remain in place;
no native higher-power/rational-profile contract is advertised here.

The principal porous reference remains [Böttcher et al. (2025), v2](https://arxiv.org/html/2504.07739v2).
Its Eq. (4) adds a solid-phase-volume contribution to fluid density; Eq. (6)
uses relative-velocity kernel-gradient terms. A candidate mesh analogue is
`S_i=∫theta W_i`, with a supplied-state action `∫theta (u_i-us)·grad W_i` under
solid-volume conservation. Evolving geometry/porosity requires transport and
Jacobian terms. No equivalence to its pressure formulation or drag law is
assumed. Porosity and pressure integration remain Stage 5 work by user request.

## Precision, rejection and reproducibility

The 70-digit fixture set contains 52 cases and 11,172 channels: complete volume
monomials through degree nine with kernel derivatives of order 0–2 at a fixed
interior query, plus baseline boundary/exterior and embedded-surface cases.
These are finite checks, not a global error guarantee. Full-support analytic
moments, refined Duffy quadrature, FEM reproduction, derivatives, translations,
compact/regular parity, containment, periodicity, discontinuities and common
work supply independent checks.

An explicit `precision="float32_compensated"` route now carries paired fp32
high/low values through geometry normalization, elementary primitives, analytic
recurrences, coefficient contractions and a cached output profile. All device
arithmetic remains fp32 (including explicit single-precision FMA); host profile
and field preparation uses exact rationals/float64 and splits coefficients into
two fp32 numbers. This is **not ordinary 24-bit arithmetic** and is not a silent
fp64 fallback. Results and ordinary particle arrays remain fp32. CPU fp64 remains
the current API default; GPU production work prioritizes raw-fp32 conditioning
and GPU measurements, given the target device's much lower fp64 throughput.
CPU timings do not choose that production backend. Exact-zero monomials are
removed before choosing the field's requested degree, and fused consumers skip
zero coefficients; no tolerance drops small nonzero physical coefficients.

The paired route uses element-local facets and **per-interaction** query offsets
prepared before the launch. This avoids losing small local distances when a mesh
contains elements far from its first vertex. `batch.evaluation_queries[row]`,
`evaluator.facets` and the selected element CSR form that mode's native input
contract. Raw modes retain per-query coordinates in the original prepared frame.
Both modes still use `batch.pq[row]` for the particle's support. Fused field and
transpose consumers use the same local-coordinate convention. Scratch allocation
is given by `evaluator.width`; the paired layout includes cached high/low output
profiles. Always call the element evaluator before reading a cached profile.

Native paired functions use `wp.func_native` C++/CUDA snippets. They do not use
quadrature. Tests cover arithmetic residuals, sqrt/log/atan2/asinh, the full
fixture set in exhaustive and compact forms, P0–P3 fields, degree-nine material
work, weighted transpose identities, multiple-element overlaps, periodic images
and widely separated element origins. Current CPU worst fixture mixed error is
about **4.7e-8**, within `1e-5`, with no rejected fixtures or fallback. The CUDA
compiler accepts the evaluator/field kernels for SM80 and the PTX contains no
fp64 **instructions**; two unused fp64 constants originate in Warp's headers.
This is compilation evidence only: no GPU execution or throughput was measured.

Warp 1.17's actual contraction option is `fuse_fp=False`; the former
`enable_fma` spelling did not control contraction. It is corrected in the new
FluxBridge modules. Explicit `fmaf` used for multiplication residuals remains
part of the paired arithmetic. The legacy Scene modules were not edited.

Raw fp32 is experimental: W4/W6 profile cancellation and higher weighted
kernel derivatives fail the `1e-5` target. The test suite records four expected
family failures; fp64 passes `1e-10` in the finite fixture set. There is no
silent fp64 fallback. `Evaluator.accuracy_status` exposes this distinction.
Queries/supports outside positive finite fp32 storage range are rejected before
adjacency construction; paired field coefficients must also fit fp32 high parts.
Condition numbers above `1e8` (fp64) or `1e4` (fp32), or maximum element AABB extent/support
below `1e-4` / `1e-2`, are conservatively rejected with an arbitrary-precision
reference recommendation. These thresholds are an engineering policy, not
certified error estimates. No arbitrary thin-sliver or tangency guarantee is
claimed. Inputs must be finite, supports positive and geometry nondegenerate.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge tests/test_package_layering.py -q
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_acceptance.py \
  --device cpu --fresh-cache --output /tmp/fluxbridge-evidence
# Compile-only audit, even without a GPU (requires the installed NVRTC runtime):
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_compile_cuda.py \
  --output /tmp/fluxbridge-compensated-evidence/cuda-compile
# Repeat runtime acceptance on an NVIDIA host with --device cuda:0.
```

Evidence follows [validation.md](validation.md): the explicit `/tmp` override
avoids writing outside the sandbox. Preserve/copy it to the development archive
when wanted; `/tmp` is temporary storage. Reports include source SHA256 hashes,
hardware, actual arithmetic precision, output/interaction counts, allocations,
search, warm execution, complete evaluation, readback, transpose, material actions
and a stated quadrature baseline. CUDA runs acquire a serialization lock.

## Native 3D plane and sphere baseline

```python
from warpSPHBoundaries.fluxbridge.radial import RadialEvaluator

plane = RadialEvaluator("plane", kernel="w6", precision="float64")
values = plane.evaluate([-.2, 0., .2], [0., 0., 1.], 1.)  # (3, 28)
sphere = RadialEvaluator("sphere", precision="float32_compensated")
batch = sphere.prepare([-.5, 0., .2], [0., 0., 1.], 1., radii=.5)
batch.launch()  # frozen arrays and reusable allocation; no host dispatch per query
values = batch.numpy()
```

The normal points **outward**. Plane distance is signed outward distance to its
boundary; the solid lies on the negative side. For a sphere,
`distance = |query-centre|-radius`, and its normal points from centre to query.
At the centre any unit normal gives the same isotropic result. The API accepts
accurately prepared distances; it cannot recover distances already lost by
rounding world coordinates. Supports and radii have physical length units.

The advertised contract is exactly `baseline(3)` and its subsets: degree-two W
moments, degree-one kernel-gradient moments, and the complete symmetric kernel
Hessian. These supply 28 channels and their usual interpolation, occupancy,
gradient, divergence, curl and Laplacian contractions. Requests with
`sum(alpha)+len(derivative)>2` are explicitly rejected. This is **not** the
degree-nine mesh field contract or an implicit promise of arbitrary FEM fields.
The solid geometry is exact; no local curvature or small-ball model is used.

`radial.functions(precision)` exposes composable `element` and `select` Warp
functions, with a prepared polynomial `Plan`. `element(distance,radius,h,kind,
plan,work,row)` writes nine physical, axisymmetric invariants to a caller-owned
scratch row. `kind=0` is a plane and `kind=1` a sphere. Their ordering is
`m0,m1,m2t,m2n,g0,g1t,g1n,ht,hn`, using the outward axis. `select` rotates these
into the requested Cartesian channel. Callers must provide validated unit
normals, consistent radius/distance data and disjoint scratch rows. The batch
wrapper uses precisely those same functions, and has no allocation in launch.

For normalized radial distance `r`, centre distance `D=R+d` and inward angular
coordinate `mu`, the spherical cap satisfies
`mu >= t = (r*r+D*D-R*R)/(2*D*r)`. Directly integrating powers of this expression
is ill-conditioned near `D=0`. Instead factor

```text
U = r*(1-t) = (R+D-r)/(2D) * (r-d),
```

and integrate products in Bernstein form on each overlap interval. Both affine
factors have nonnegative endpoint values there. Full shells and the exact
centre use separate limits. Preserve exact factors `r**p*(piece_end-r)**k`
of W, L=W'/r and r*Q=L' through integration. The Wendland residual polynomial
coefficients have one sign per profile; expanding them into alternating shifted
powers previously defeated much of the angular conditioning in raw fp32.
Exact rational preparation extracts the endpoint factors before precision
conversion. For cubic's outer piece, use the factored numerators r*L and r³*Q
with their explicit denominators. `Plan.radial_powers` accompanies the shifted
powers; build the plan through `RadialEvaluator` before custom device calls.
Plane caps use `U=r-d`. All angular contributions reduce to
combinations of U, U² and U³ with explicit radial powers.

The cubic outer piece contains a **linear radial term**, hence `1/r` and
`-1/r³` derivative profiles. It must not be discarded when differentiating the
piecewise kernel. Where the resulting primitive contains negative radial
powers, evaluate it with an analytic beta series about the interval's upper
endpoint. Here `lo>=1/2`, `eta=(hi-lo)/hi<=1/2`, and the inverse power is at most
two. Summing 64 terms bounds the scalar tail by `8e-18` before the explicit
`hi^-s` factor and polynomial weights. This is a numerical primitive evaluation
with a stated tail, not a truncated geometric model or quadrature fallback.
Reference fixtures use independent 70-digit numerical shell integration.
Raw fp32 may exit that positive series once `series + term == series`.
The terms are nonincreasing (`eta<=1/2`, inverse power at most two), so every
subsequent addition would also round away. This yields the same raw floating
result as all 64 steps, with fewer operations. Tests compare invariant scratch
and Cartesian outputs bit for bit against the fixed-step variant. The 64-term
mathematical tail bound is not a bound on fp32 rounding error; its finite
accuracy budget is checked independently. fp64 and paired fp32 retain 64 steps.

Finite inputs, positive support, unit normals and valid sphere centre distances
are checked by `prepare`. Normalized distances and the internal physical
support scalings `h²` and `h^-2` must also fit the selected storage precision;
otherwise preparation rejects them before a zero contribution can multiply an
overflowed scale. The conservative radius/support policy is
`[1e-4,1e4]` for fp64 and `[1e-2,1e4]` for fp32 modes. These are engineering
limits, not certified error bounds. The factored radial-profile update passes
the finite fixture budget in fp64, raw fp32 and paired fp32; the original two
raw-W6 failures are now strict passing tests. General mesh/clipped-planar raw
recurrences remain a separate accuracy target.
CUDA execution, higher weighted products and geometry derivatives remain open.

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_radial_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge/test_radial.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --device cpu --output /tmp/fluxbridge-radial-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --compile-only float32_compensated --output /tmp/fluxbridge-radial-evidence/cuda-compile
```

## Native 2D rectangles and circles

```python
from warpSPHBoundaries.fluxbridge.box import BoxEvaluator
from warpSPHBoundaries.fluxbridge.disk import DiskQuadratureEvaluator
from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator

box = BoxEvaluator([[0., 0.]], [.7, .4], angles=.2, kernel="w6")
values = box.evaluate([[.8, .1]], 1.)  # all 15 baseline channels
circle = DiskQuadratureEvaluator(kernel="w6", precision="float32", nodes=24)
values = circle.evaluate([-.2, 0., .2], [.6, .8], 1., radii=.5)
fibres = DiskArrayEvaluator([[.25, .5], [.75, .5]], [.1, .1], kernel="w6")
batch = fibres.prepare([[.1, .2]], .4, periods=(1., 1.))
batch.launch()  # prepared data, circle functions and deterministic reduction
```

`BoxEvaluator` is an exact rectangular **geometry adapter** to `Evaluator`.
Centres are `(E,2)`, physical half extents broadcast to `(E,2)`, and rotation
angles (radians) to E. All existing analytic polygon requests through degree
nine and kernel derivative order two remain available. Cartesian polynomial
fields use each rectangle's first-vertex origin with the ordinary
`PolynomialField`/`FieldEvaluator` contract. This does not supply quadrilateral
Lagrange/Bernstein FEM bases. It does not replace or certify `scene.box` tables.

The circle route is deliberately named **DiskQuadratureEvaluator**: it performs
numerical boundary integration of an exact circle. The 15-channel baseline and
its subsets are complete; higher combined degree is rejected. Eight through
64 Gauss nodes per panel are selectable, with 24 the default. Panels split at
kernel knots and refine geometrically towards the nearest circle point; the
smallest refinement scale is about `1e-8` times the supported arc. This is a
finite quadrature policy, not a certified bound for arbitrary inputs. Reports
compare the default with 48 nodes and with independent 70-digit volume shells.
No disk table, small-disk series, curvature model or polygon is substituted.

The scalar inputs use signed outward distance `d=|query-centre|-R` and an
outward unit normal. At the exact centre any unit normal is equivalent, and
radial symmetry is evaluated directly. Full support containment and external
tangency/disjointness also have explicit exact limits. Preparation checks finite
inputs, positive radius/support, valid centre distance and unit normals. The
radius/support conditioning policy is `[1e-4,1e4]` in fp64 and `[1e-2,1e4]`
in fp32 modes. Normalized distances and `h²,h^-2` scratch scalings must fit the
storage precision. Already-lost world-coordinate precision cannot be restored.

For unit support, let `M(r)=integral_0^r t W(t) dt`,
`Phi(r)=M(r)-M(1)`, `A(r)=integral_0^r t Phi(t) dt`, and `L=W'/r`.
The circle boundary has outward normal `n_b`, and `y=source-query`:

```text
mass    = integral_boundary (n_b dot y) M/r² dl
m_i     = integral_boundary n_b_i Phi dl
m_ij    = integral_boundary n_b_i y_j Phi dl
          - delta_ij integral_boundary (n_b dot y) A/r² dl
g_i     = -integral_boundary n_b_i W dl
G_ij    = delta_ij mass - integral_boundary n_b_j y_i W dl
H_ij    = integral_boundary n_b_i y_j L dl
```

These identities provide second moments and direct ambient Hessians, rather
than differentiating the older five-invariant disk lookup. W, Phi, the regular
potentials and L are prepared with exact rational algebra and conditioned
Chebyshev coefficients before precision conversion. The cubic outer piece's
linear radial term is retained in L. The exact centre and support-containment
second moment is computed from the registered kernel, including normalization.

Although W and Phi vanish outside support, **M/r² and A/r² retain tails**.
Their whole far-circle flux is integrated analytically as an angle correction;
only the supported arc uses quadrature. Thus localization does not simply drop
the regular exterior contribution. Circle geometry uses half-angle formulas
and the accurately supplied signed distance to avoid subtracting nearly equal
centre/radius terms at the surface. Paired fp32 uses explicit paired trigonometry.
Raw fp32 uses a bounded polynomial on the known half-angle range; CUDA's generic
trig slow path otherwise emits fp64 instructions. All three modes compile;
neither fp32 PTX contains fp64 instructions.

`disk.functions(precision).element` writes the same nine physical invariants
and uses the same Cartesian `select` convention as the 3D radial family, with
a zero z component in the 2D normal. Scratch is `(rows,9)` of the selected real
type. Coefficients and quadrature data in `Plan` retain high/low parts in the
paired mode. A custom Warp consumer can call these functions directly; batch
launches contain no allocation or per-query host dispatch.

`DiskArrayEvaluator` accepts immutable centres and radii and world queries.
Host preparation constructs local distance/normal data **before** fp32 storage,
uses a conservative AABB tree, enumerates unique periodic images and allocates
the interaction/reduction arrays. Bounds are rounded outward so small radii at
large offsets cannot collapse the box and lose a support intersection. Prepared
launches reuse their buffers; changing queries/supports requires new adjacency.
There is no fixed slot-capacity truncation. Overlapping disks/boxes **add**;
these are not geometric union operators. Resolved solid fibres do not receive
an additional effective resistance for the same geometry. Their 2D two-component
flow is transverse; longitudinal permeability still requires axial velocity
or a compatible 3D reference, after the solver handoff.

The old Scene disk/box APIs remain unchanged. They are still usable with porous
coupling disabled. The new circle provider does not yet support degree-nine
material products, shape adjoints, full-contract lookup/series, or a fluid solver.
Mixed absolute/relative fixture acceptance does not promise relative accuracy
for vanishingly small intersections or exact positivity after rounding.

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_diskbox_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge/test_diskbox.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_diskbox_acceptance.py \
  --device cpu --output /tmp/fluxbridge-diskbox-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_diskbox_acceptance.py \
  --compile-only float32 --output /tmp/fluxbridge-diskbox-evidence/cuda-compile
```

## Infinite planes and finite-thickness bands

```python
from warpSPHBoundaries.fluxbridge import Request
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator

plane = PlanarEvaluator("plane", 2)  # all 15 baseline channels
values = plane.evaluate([-.2, 0., .2], [.6, .8], 1.)
band = PlanarEvaluator("strip", 3, [Request((3, 3, 3), (0, 2))],
                       kernel="w6", precision="float32_compensated")
batch = band.prepare([0., .1], [0., 0., 1.], 1., half_widths=.2)
batch.launch()  # reusable allocations; caller may also consume batch.values
```

All coordinates below are query-centred, `y=source-query`. A plane with unit
outward normal n and signed outward distance d represents `n dot y <= -d`.
A band uses signed distance d from its centre plane and positive half width a,
representing `-a-d <= n dot y <= a-d`. Normals, distances, supports and widths
broadcast across a batch. Requests use global Cartesian axes. All degree-nine
monomials and order-zero/one/two kernel derivatives are supported, including
their products; differentiating the monomial itself still requires the usual
complete-derivative correction. Units remain `h**Request.scale`.

For unit support, replace the infinite tangent directions by the interval
`[-2,2]` (2D) or square `[-2,2]^2` (3D), and clip the normal interval to
`[-2,2]`. Every point in the supported unit ball already lies in these tangent
and normal bounds. Therefore the clipped region and the infinite solid agree
everywhere the kernel or its registered derivatives is nonzero. This equality
holds for every polynomial weight, independently of degree. The resulting
rectangle or closed triangulated box uses the existing analytic flux plan.
Artificial faces retain their contribution to the compact angle correction;
they are not simply discarded from the regular flux. A band is integrated
directly, avoiding subtraction of nearly equal half-space integrals.

Two triangulations of the same planar caps avoid singular intermediate
geometry: exact query-on-plane caps use a centre vertex at the origin; other
caps use an eccentric interior vertex so no interior edge goes through the
query projection. This changes neither the region nor the requested operator.
The tests include exact and near-zero offsets down to `1e-18`, arbitrary normals,
support scaling, complete derivatives, containment and poisoned scratch reuse.
Degenerate/disjoint intervals clear the entire recurrence/output cache row.

`planar.functions(dimension, precision)` exposes `element`, `output`, `power`,
`Plan`, `real`, `vec`, `work_dtype` and `face_count`. The element function accepts
normalized lower/upper bounds, a unit normal, plan, face-index array, facet
scratch, recurrence scratch and row. Caller bounds must be finite and clipped
to `[-2,2]`; every row owns `face_count` consecutive facets (4 in 2D, 16 in 3D),
each containing three `vec3` values. Recurrence scratch is `(rows, ev.width)`
of `work_dtype`; paired fp32 includes cached output channels. The output
function returns a unit-support value; fused callers apply `power(h, scale)`
once. No host dispatch, allocation or floating-point atomic occurs in a launch.
The test's custom consumer contracts the product of three cubic fields without
materializing the full channel matrix.

Preparation validates unit normals, finite inputs and physical scaling,
including intermediate positive powers used for negative exponents. Bounds
are clipped before device conversion. A conservative conditioning policy rejects
band half-width/support below `1e-4` for fp64 or `1e-2` for either fp32 mode.
There is no fallback to quadrature or to a different precision. Raw fp32
general recurrences still miss the mixed error gate; paired fp32 is explicit.
The template geometry used for static plan packing is included in measured
preparation cost. Benchmarks expect accurate signed distances/normals supplied;
they do not include a world-geometry search. The smaller 3D axial provider is
still available and can be faster when only the 28 baseline channels are needed.

This supplies polynomial moments and composable contractions, not an FEM basis
on an unbounded domain or an automatic infinite-domain material adapter. The
finite-element/material APIs remain available for finite polygons/tetrahedra.
It does not extend the analytic sphere or circle providers' weighted contracts, certify old
strip/plane tables, or add pressure/porosity integration.

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_planar_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge/test_planar.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_planar_acceptance.py \
  --device cpu --output /tmp/fluxbridge-planar-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_planar_acceptance.py \
  --compile-only float32_compensated --output /tmp/fluxbridge-planar-evidence/cuda-compile
```

## Conditioned 2D boundary quadrature in ordinary fp32

The explicit `method="conditioned_boundary"` selects numerical integration of
straight 2D polygon geometry. It is available on `Evaluator`, `BoxEvaluator`,
2D `PlanarEvaluator`, `FieldEvaluator`, `differential`, and material `prepare`
methods. The default remains `method="analytic"`. This numerical method is not
an automatic fallback or a claim that the analytic raw-fp32 recurrences pass.

```python
from warpSPHBoundaries.fluxbridge import Geometry, Evaluator, baseline

geometry = Geometry([[[0, 0], [2, 0], [0, 2]]], 2)
evaluator = Evaluator(geometry, baseline(2), kernel="w6", precision="float32",
                      method="conditioned_boundary", nodes=16, device="cpu")
batch = evaluator.prepare([[.2, .2]], 1.)
batch.launch()
values = batch.numpy()
# FieldEvaluator(..., method="conditioned_boundary", precision="float32")
# material.prepare(..., method="conditioned_boundary", precision="float32")
```

All monomials through degree nine and kernel-only derivatives through order two
are supported for W2/W4/W6/cubic. Ordinary fp32 and fp64 are supported; paired
precision, 3D volume and surface geometry are rejected. The supported geometry
is exact; its integration is numerical. Gauss order is an integer from 8 to 64,
with 16 the tested default. Tests and reports compare 16 with 32. There is no
universal quadrature error certificate; increase order and check convergence
for new workloads. Polynomial field/basis conditioning is a separate issue.

For unit support define `L=W'/r`, `Q=L'/r`, and

```text
C[k,f](r) = integral_0^min(r,1) t^(k+1) f(t) dt.
```

Euler's divergence identity gives the volume integral of `y^beta f(r)` as the
boundary flux of `y y^beta C[k,f](r) / r^(k+2)`, where k=|beta|. For an oriented
edge with signed normal distance z and angular coordinate phi,

```text
(n dot y) ds / r² = sign(z) dphi
integral_polygon y^beta f(r) dA
    = sum_edges sign(z) integral unit_direction(phi)^beta C[k,f](r(phi)) dphi.
```

Use W with beta=alpha for moments, -L with beta=alpha+e_j for gradients, and
`delta_ij L` with beta=alpha plus Q with beta=alpha+e_i+e_j for Hessians.
These are kernel-only derivatives; complete moment/field derivatives retain
their usual coefficient corrections. Every required cumulative profile is a
piecewise polynomial, including cubic's Laurent derivatives: the weighted
powers make all antiderivative exponents positive. Exact rational preparation
combines each profile before conversion to Chebyshev coefficients on its kernel
piece. Clenshaw evaluation therefore avoids subtracting large, separately
rounded radial monomial integrals. The finite prepared-profile coefficient
amplification is at most 1.55; this is a profile-basis diagnostic, not a global
condition number or quadrature bound.

The complete exterior flux is retained: for r>=1, C is its nonzero constant
rather than zero. Every face of an intersecting element participates. Angular
panels split at kernel knots and at dyadic distances from the edge foot point;
there is no fixed panel-slot truncation. Bounded-angle polynomial sine/cosine
avoids CUDA's unused large-argument fp64 trig path. Geometry, profile evaluation
and accumulation remain ordinary fp32 in that mode; output summation uses the
same kind of fp32 Neumaier residual as other raw reductions, without paired
primitives or fp64 device arithmetic.

A native whole-element test uses winding and padded point-to-segment distance
to clear disjoint rows. The distance padding scales with machine epsilon and
coordinate magnitude. It conservatively retains shallow contacts and is not
an exact geometric predicate for arbitrary ill-conditioned input. This prevents
far-field floating residuals being amplified by large polynomial translations
in exhaustive field evaluation. It never prunes individual regular-flux faces.
Existing geometry-conditioning limits apply; normalized spans above 1e8 and
unrepresentable output/intermediate support powers are additionally rejected.

`wp_func.functions(precision, method="conditioned_boundary", nodes=16)` exposes
the common `element`, `output`, `power`, `Plan`, real/vector types and batch
signature. Facets are element-local, and queries are **per interaction** offsets
from the same element origin. Use `Evaluator`/`FieldEvaluator` to prepare those
arrays. Scratch is `(interaction_count, ev.width)`, where `ev.width=2*C` for C
requested outputs: output sums followed by their fp32 residuals. `output` adds
the two entries and returns the unit-support channel; callers apply h**scale
once. No allocation or per-query host dispatch occurs in a launch. The field
consumer handles this local-coordinate layout in both forward and transpose
kernels. Existing analytic and paired layouts are unchanged.

Independent frozen fixtures cover 120 cases / 6,880 channels per precision,
including full degree-nine requests. Additional tests check complete derivatives,
units, exact/near contacts, nonconvex polygons, containment, large translations,
empty batches, periodic/exhaustive/localized sums, P0–P3 field reproduction,
transpose work and degree-nine point/affine material work. See the main plan's
latest checkpoint for accuracy, CPU/CUDA costs and remaining gaps.

## Actual CUDA execution and deferred analytic fp32 conditioning

The current regression suite runs on an NVIDIA RTX PRO 6000 Blackwell
Workstation Edition (SM120), Warp 1.17.0, driver 610.43.02: **522 passed,
32 expected analytic-fp32 failures, no skips or unexpected failures** across
CPU/CUDA and retained solid applications. Five CUDA-only graph replay tests
pass. Independent reference budgets remain `1e-10` for fp64 and `1e-5` for
accepted raw/paired fp32 routes. The numerical 2D route's worst CUDA mixed error
is 2.985e-6 across 6,880 channels; the analytic planar fp64 route's worst is
8.286e-12 across 22,820 channels. See `/tmp/fluxbridge-gpu-evidence/` and the
plan's GPU execution checkpoint for per-family reports and source hashes.

The user accepts analytic fp64 and explicitly numerical fp32 for now. Analytic
raw-fp32 recurrence repair is deferred; its existing misses remain visible.
No numerical substitution, lower budget, automatic precision fallback or
operator arithmetic change was introduced to make the CUDA tests pass.

Correctness does not imply GPU efficiency. The full analytic 3D degree-nine
mask has 2,200 outputs; one W2 plane/band query takes about 2.60 / 5.47 seconds
warm in fp64 on this device. The numerical 2D W6 mask has 330 outputs and takes
16.48 ms for one raw-fp32 query. These synchronized, shared-machine costs
include host launch overhead; the 3D full-mask path needs substantial work
before production use. Profile batch scaling, sparse output masks and device
execution separately from host search/upload before expanding its geometry.

## Polynomial-exact spherical-shell quadrature for 3D planes/bands

`PlanarEvaluator("plane" or "strip", 3, requests,
method="shell_quadrature", precision="float64", nodes=16)` evaluates the same
weighted integrals as the analytic provider, with a different explicit
integration algorithm. It is limited to infinite 3D planes/bands and supports
the full Cartesian degree-nine / kernel-derivative-order-two contract. It is
not an arbitrary tetrahedral or material-domain replacement. The existing
analytic and 2D numerical method defaults remain unchanged.

For normal coordinates `lo <= n·y <= hi`, write
`y = r*(n*t + sqrt(1-t*t)*(u*cos(phi)+v*sin(phi)))`. The volume measure is
`r*r dr dt dphi`, with `max(-1,lo/r) <= t <= min(1,hi/r)`.
Split `0 <= r <= 1` at `abs(lo)`, `abs(hi)` and every kernel knot. On each
interval the clipping branches are fixed. Derivatives retain the exact kernel
jet identities `grad_j W=-y_j L` and `H_ij W=delta_ij L+y_i*y_j Q`, where
`L=W'/r` and `Q=L'/r`. No geometry clipping or moment degree is approximated.

For maximum combined degree `d=max(|alpha|+derivative_order) <= 11`, use `d+1`
equally spaced azimuth nodes: no nonzero Fourier harmonic of degree at most d
aliases into the constant. The resulting angular integrand is a polynomial
of degree at most d in t, integrated exactly by `ceil((d+1)/2)` Gauss nodes.
After angular integration the Wendland radial integrands are polynomials of
degree at most 22; 16 radial Gauss nodes exceed the needed degree. These are
polynomial-exact statements in exact arithmetic, using the standard
[Gauss quadrature exactness property](https://dlmf.nist.gov/3.5.v). They do not
claim exact floating-point answers. Cubic's outer Hessian additionally has
Laurent terms on `r >= 1/2`; that radial quadrature is numerical and checked
against high-precision references and a 24-node rule.

The implementation evaluates factored radial profiles, including the spherical
Jacobian before any singular factor, and uses ordinary compensated summation.
Gauss rules are polished in extended host precision before storage in fp64;
no high-precision arithmetic occurs in a device launch. CPU/CUDA factories
support raw fp32/fp64; paired fp32 and 2D shell geometry are rejected.

The batch prepares cached coordinates and weighted profiles on the device,
then assigns separate GPU threads to requested outputs. Every launch rebuilds
the query-dependent samples and reuses its allocations. `api.element` supplies
the same arithmetic to composed device consumers, but computes its requested
row serially; the parallel batch is the performance route. `api.output`
returns normalized cached moments; physical support scaling is applied once.
The existing planar `facets` workspace stores sample coordinates/profiles for
this method; its capacity is `api.face_count`, not a count of geometric faces.
Sample storage grows with query count and the output mask's angular degree;
inspect `batch.bytes`. Frozen quadrature samples are not silently reused after
inputs change. See the plan's quadrature checkpoint for measured costs and
independent near-roundoff acceptance.

## Convergence-checked tetrahedral boundary quadrature

`Evaluator(tetrahedra, requests, method="tet_quadrature", nodes=16, ...)`
supplies every 3D degree-nine moment and kernel derivative through order two.
The same method is accepted by `FieldEvaluator`, `ResistanceMaterial.prepare`
and `AffineResistanceMaterial.prepare`. Four kernels and ordinary fp32/fp64
are supported. It is explicit numerical integration over exact straight
tetrahedral geometry. Surfaces, 2D geometry, general closed skins and paired
precision are rejected. Existing method and precision defaults are unchanged.

For each radial profile `f` (W, L=W'/r, Q=L'/r) and degree k, prepare
`F[k,f](r)=int_0^min(r,1) t^(k+2)*f(t) dt` in exact rational arithmetic, then
store its piecewise Chebyshev representation. Euler's vector potential
`A(y)=y*y^beta*F[k,f](|y|)/|y|^(k+3)`, k=|beta|, has divergence `y^beta*f`.
Every advertised primitive is regular at the origin. The cumulative profile
has a nonzero exterior constant; all faces of a selected tetrahedron remain
in the regular flux integral.

For a face with signed distance d from the query and outward normal n, split
it into oriented polar sectors from the plane foot to its three edges. With
rho=|d|*tan(theta), unit direction
`u=sign(d)*n*cos(theta)+e(phi)*sin(theta)`, the flux measure is
`sign(d)*sign(edge)*sin(theta) dtheta dphi`. The integrand becomes
`u^beta*F[k,f](|d|/cos(theta))`. Split theta and phi at kernel-sphere cuts, and
use dyadic geometric panels to resolve near-face/edge limits. Gaussian angular
rules are numerical here: this is **not** the planar polynomial-exactness
argument. Face contributions with d=0 vanish for these regular primitives.

Preparation compares 16/24 rules by default on every normalized requested
moment. It promotes the order in increments of eight if the mixed difference
exceeds 1e-14 (fp64) or 1e-5 (fp32), and rejects unresolved convergence at 64.
Fields additionally compare the translated/contracted per-element outputs at
1e-12 / 1e-5, promote through order 32 if necessary, and reject unresolved field
conditioning. Two-rule agreement is an estimator, not a certified error bound;
independent high-precision fixtures establish the recorded accuracy.

Panel/sample geometry is prepared in fp64 **on the host**. Directions, radii,
weights, profiles, moment sums and field arrays use the selected device dtype;
raw fp32 has no paired primitives or device fp64. A batch owns its prepared
arrays and convergence metadata in `batch.quadrature`. Different queries,
supports, material fields or geometry require a new prepared batch. Conservative
whole-element distance rejection preserves containment and exactly clears
disjoint rows before large polynomial translations. It is a padded floating
predicate, not an exact geometry certificate. Existing geometry-conditioning,
tiny-element and representable-scaling limits apply; normalized spans above
1e8, more than two million samples per interaction or more than 1 GiB of
profile/tile storage per rule are rejected explicitly.

Repeated `launch()` allocates nothing: it evaluates the combined profiles,
integrates 256-node tiles, reduces their sums deterministically, then performs
output-parallel field contractions. Tile storage is ragged by interaction,
so small rows do not inherit the largest row's padding. Ordinary compensated
sums are used without float atomics. Material force, gradient-load, reaction
and dissipation kernels consume the same packed integrals as before. The
supplied-gradient transpose remains the caller's contract; no SPH reconstruction,
pressure or porosity solver is added.

The later shared-radius cache stores each exactly equal radius only once per
interaction, after conversion to the selected storage dtype. `plan.radii` and
`plan.profiles` now index this cache; `profile_index` maps quadrature nodes to
cached radii and `profile_offsets` keeps composed rows independent. Node counts
are `node_offsets`/`weights`, not the length of the radius cache. Weights multiply
the cached profile at integration time in the original arithmetic order. Nodes,
weights, quadrature order and moment summation order are unchanged. Metadata
reports both sample and cached-radius counts. The per-rule 1 GiB profile/tile
guard uses the actual compressed allocation; it is not removed or increased.

For a composed device call, pass **`batch.integration_plan`**, its matching
element-local query/support, row and scratch to `api.element`; `evaluator.plan`
contains only static profile data. The prepared plan binds the geometry and
quadrature arrays. `element` computes that row serially, using the same profile
and integral arithmetic as the tiled path. Changed query/support arguments
produce NaNs as a stale-binding diagnostic. Geometry arguments in this common
signature are compatibility inputs; they do not regenerate the bound samples.
`api.output` returns normalized moments, so apply h**Request.scale once.
The multi-pass batch/field path is the measured fast route.

CPU and actual CUDA checks cover 10,640 independent channels per precision,
including full degree-nine jets, slivers, containment and near/exact contacts.
Worst mixed errors are 2.634e-15 in fp64 and 7.684e-7 in fp32. The real P3
affine-material workload also passes independent 70-digit-moment contractions,
work/torque checks and graph replay. See the plan's tetrahedral checkpoint and
`/tmp/fluxbridge-tetquad-evidence/` for timing, memory and source hashes. Frozen
material action is much faster on CUDA, but host preparation and sample storage
remain scaling limits and CPU performance is a separate tradeoff.

## Reusable static material evaluator

`ResistanceMaterial.compile(**evaluation_options)` and the supplied-affine
variant return an explicit `MaterialEvaluator`. It copies polynomial material
fields, test basis and constitutive metadata, then builds one reusable
`FieldEvaluator`. Recompile after changing geometry/material or evaluation
options. Each subsequent `prepare()` independently rebuilds query-dependent
adjacency, quadrature, convergence checks and batch storage; completed integrals
are recomputed on each action. Multiple prepared query batches share static
coefficients without sharing writable quadrature caches.

```python
provider = material.compile(kernel="w6", precision="float32", device="cuda:0",
                            method="tet_quadrature", preparation="device")
batch = provider.prepare(queries, supports)  # motion/support changes rebuild here
result = batch.apply(fluid_velocity, masses, supplied_velocity_gradient)
```

The example is for an `AffineResistanceMaterial`; the point-resistance action
omits the supplied gradient. Existing `material.prepare()` retains its complete
static-plus-query construction semantics. Benchmark reports distinguish static
material compilation, reused static preparation and a complete static rebuild;
none is silently removed from the cost comparison. Source-field and offset
mutation tests verify independence of existing compiled snapshots.

## Device-side tetrahedral panel preparation

Select `preparation="device"` with `method="tet_quadrature"` to count, scan,
allocate and generate the query-dependent angular rule on the chosen Warp
CPU/CUDA device. The default `preparation="host"` is retained. Other integration
families reject device preparation. **Geometry preparation is fp64 in either
case**, including for raw-fp32 evaluation. The distinct `device_panels` module
therefore contains expected fp64 arithmetic; the selected raw-fp32 evaluation,
field, reduction and material modules remain free of fp64 instructions.

The host supplies local fp64 face/query snapshots and support radii after the
existing adjacency/validity checks. A padded device tetrahedron-distance test
prunes whole disjoint elements; uncertain Gram-matrix classification retains the
element. No individual regular-flux face is dropped. Each interaction has 24
oriented face/edge/side sectors. Device counting and int64 prefix scans determine
exact node/cache storage. Only scalar allocation/error totals return to the host
at this step. Native sectors then write angular-interval descriptors; a separate
kernel generates all nodes in parallel, rounds directions/weights/radii to the
selected evaluation dtype and constructs cache indices. No quadrature arrays
are read back for node generation or radius sorting.

Radius sharing uses full radial panels and each sector's exterior constant.
It need not deduplicate equal radii across sectors, so device rules report
`cached_radii`, while host exact-deduplication rules additionally report
`unique_profile_arguments` (radius and representation kind). The value/field convergence checks and direct device consumers
use the same profile/tile arithmetic. Repeated actions remain allocation-free
and graph-compatible after preparation.

Device preparation rejects more than two million nodes per interaction,
int32 index overflow, unresolved convergence, and more than 1 GiB per rule for
estimated node/cache/work storage or its interval-generation peak. This last
budget includes more storage than the host path's profile/tile-only cap; it is
explicitly stricter, not a precision fallback. Comparison-rule temporaries,
static material buffers and CUDA allocator reservation remain separate costs.

This is device-side **panel/sample** generation. The separate optional device
BVH below supplies adjacency; default search remains host-based. Convergence
decisions, allocation and the existing requested-moment/field
comparisons synchronize with the host. Query/material changes require new
prepared batches, or recompilation of changed static material. The API does not
claim a GPU-only moving-particle pipeline or streamed unbounded batches.


### Regular near-plane potential

A face within `1e-12` normalized distance of the query, with normalized edge
lengths at most four, uses a Duffy triangle rule for the **same exact signed
face potential**. The cumulative radial primitive is evaluated as `F(r)/r^3`:
its inner polynomial has the removable factor cancelled exactly before
Chebyshev conversion, including the finite value at zero. Other faces retain
the polar angular rule; exactly zero signed-distance faces retain their exact
zero contribution. Queries and geometry are never snapped or displaced, no
faces are discarded, and sample/memory caps are unchanged. Cache entries carry
a representation kind, so radius sharing cannot mix `F` and `F/r^3`.

This avoids the excessive polar subdivision encountered at roundoff distances
from wall faces/edges. Both rules retain the existing two-order checks and
explicit unresolved-input rejection; this is not a universal error certificate.
The frozen near-plane fixture adds 1,240 independent 70-digit channels, including
face/edge/vertex limits, both sides of the representation switch, slanted faces
and the actual physical wall-grid coordinates. CPU/CUDA accuracy reports cover
11,880 channels per precision with zero failures or fallbacks. Sixteen further
CPU/CUDA primitive tests compare the origin and tiny radii with direct
80-digit radial integration.


## Whole-element device support search

`DeviceSearch(geometry, device="cuda:0").prepare(queries, supports, periods=...)`
returns an immutable `Adjacency` compatible with existing operators. Alternatively
select `search="device"` when compiling an `Evaluator`, `FieldEvaluator`,
`ResistanceMaterial`, supplied-affine material, or `DiskArrayEvaluator`. Host
search remains the default; `exhaustive=True` retains the all-element reference.
The differential wrapper builds the selected adjacency once for all jets.

Static bounds are recentered and rounded outward for Warp's fp32 BVH. The chosen
GPU builds an LBVH; CPU validation builds a native SAH tree. Query ranges,
periodic shifts and conservative ball/AABB filtering use fp64. Traversal,
integer counting/scans and sort keys run on the selected device. Sorting restores
query/element/lexicographic-image order independently of BVH traversal. Rounded
boundary contacts may retain extra elements within the declared fp64 padding.
No regular-flux face is dropped, and different material traces remain separate.
Containment, cavities and exact/near contacts are checked against exhaustive
and independent box enumeration. Whole-disk search remains an ordinary solid
provider and adds no effective resistance to resolved periodic fibres.

The present interface reads CSR arrays back to the host, and prepared consumers
upload them again. Allocation/convergence decisions also synchronize. Only
frozen batch launches are graph-compatible; search preparation is not a
GPU-only particle pipeline. Limits are 4,096 candidate periodic images per query,
int32 output indices, signed-int64 sort keys, finite local BVH bounds, and a
1 GiB estimate for owned search arrays. Opaque BVH storage, scan/sort scratch
and CUDA allocator reservation are additional. Rebuild for changed geometry;
query/support changes produce fresh adjacency with existing stale-data checks.

## Shared prepared polynomial contractions

The tetrahedral provider translates each requested monomial integral once per
interaction, then shares it across field outputs. A 220-moment/910-output P3
material no longer repeats all 5,005 translation terms per output. The transpose
uses the same per-action integrated weights. Profiles and material coefficients
retain ordinary selected-dtype storage. Neumaier sums and explicit FMA product
residuals preserve an additional **same-dtype correction workspace** until field
projection, including in fp32. This is distinct from the analytic paired-fp32
backend, which has paired profiles and coefficients; there is no automatic
precision or method fallback. Raw-fp32 evaluation/field/reduction/material PTX
still contains zero fp64 instructions. Geometry modules intentionally use fp64.

Weights and corrections are recomputed every action and refreshed after query
changes; they are not cached completed integrals. Frozen actions retain graph
capture and deterministic reductions. Degree-nine value/mixed-Hessian jets,
physical scaling, polynomial transpose work, stale queries, real P3 products and
workspace refresh are independently checked. Unrepresentable fp32 field
coefficients are rejected before launching. Signed quadrature and polynomial
conditioning retain their stated finite-domain limitations and error gates.

`MaterialBatch.bytes` now counts field/rule/forward/reverse adjacency arrays and
owned action arrays, including the supplied-affine outputs. Static coefficients,
plans/BVH, comparison temporaries and allocator scratch remain separate. The
128-row P3 raw-fp32 batch owns about 219.27 MB in the current measured scene;
225,280 bytes are the shared weight/correction workspace. These are allocated
array capacities, not total GPU-reservation or peak-process measurements.


## Explicit bounded streaming

```python
values, execution = evaluator.evaluate_streamed(q, h, chunk_size=32, return_metrics=True)
for chunk in field_evaluator.iter_evaluate(q, h, chunk_size=32):
    consume(chunk.start, chunk.stop, chunk.values)  # no native rule retained
compiled = material.compile(method="tet_quadrature", precision="float32",
                            preparation="device", search="device", device="cuda:0")
result = compiled.apply_streamed(q, h, u, masses, supplied_gradient, chunk_size=64)
```

The value APIs are also available on ordinary `DiskArrayEvaluator`; no porous
coupling is attached to those resolved fibres. Values preserve normal output
shapes/dtypes. Query/support snapshots and periodic/global-CSR validation occur
before iteration. Compatible supplied CSR is sliced with local query IDs and
fresh signatures. A changed geometry revision or failed chunk is explicit.

The material API supports ordinary material when no gradient is supplied and
requires a matching finite supplied gradient for the affine variant. Reactions
and energy accumulate deterministically in host fp64, then reaction arrays
round once to the selected output dtype. Native moment/field/action evaluation
keeps the selected dtype. `DragResult.execution` is optional metadata; ordinary
prepared calls retain their previous result fields and default to `None` there.

This driver allocates, prepares and reads back each chunk; it is not graph
capture or a GPU-only query adapter. Prepared batch `launch`/`apply_device` remain
the graph-compatible interface. Each native batch is released before iterator
yield, including on CPU where `.numpy()` can otherwise retain native memory.
The collecting API owns a full host output; the iterator avoids that allocation.
Maximum accepted-batch bytes, host output/input snapshots and reaction workspace
are separate metrics. Static providers, comparison temporaries, opaque BVH data
and CUDA reservation remain outside that accepted-batch measure. The streaming
study additionally records actual CUDA-pool current/high-water usage in fresh
processes, not an inferred total-memory claim. Chunk size never bypasses the
existing per-rule/sample/precision limits or silently retries another method.


### Explicit lower-order tetrahedral option

For the validated numerical fp32 route, pass `nodes=8` to `Evaluator`,
`FieldEvaluator`, material `.compile()` or `.prepare()`. Default order is still
16. Every prepared rule compares with a higher order, may promote, or rejects
an unresolved/resource-limited input; requested order is not guaranteed accepted
order. Streamed execution reports `requested_nodes`, `quadrature_orders`,
`quadrature_interactions_by_order`, moment/field promotions and cumulative rule
sample counts, excluding empty chunks from order histograms.

The finite 70-digit CPU/CUDA gate has 11,880 channels per precision, worst fp32
mixed error 7.153e-7. Real degree-nine P3 contractions and actual CUDA graphs
also pass. This does not establish a global error bound. In the measured
16-element P3 scene, 128 fp32 queries need about 22.5 ms including checked
preparation/application and 59.44 MB accepted native storage; a warm action is
about 1.7 ms. Full static rebuild, comparison workspace, allocator use and
streamed scaling remain separate costs in the plan's order-8 checkpoint.


## Weighted solid-sphere shell quadrature

```python
from warpSPHBoundaries.fluxbridge import SphereShellEvaluator, Request
sphere = SphereShellEvaluator(
    [Request((0,0,0)), Request((3,3,3)), Request((2,3,4), (0,2))],
    kernel="w6", precision="float32", device="cuda:0", nodes=24,
)
batch = sphere.prepare([-.2, 0., .5], [0.,0.,1.], 1., radii=.5)
values = batch.numpy()   # prepare already evaluates both rules for convergence
batch.launch()          # reusable allocation-free action, capture-compatible
```

This explicit numerical provider integrates exact solid-sphere geometry,
Cartesian monomials through degree nine and kernel-only derivatives through
order two. Signed distance is `|query-centre|-radius`; the unit normal points
from centre to query. At the centre any unit axis is equivalent. Radii/supports
broadcast per query. Units follow `h**Request.scale` once. Requests are global
source-minus-query monomials, so fixed physical polynomial coefficients must
be translated before contraction, as for the other moment providers.

Default radial order 24 compares to 32 and promotes by eight up to 64. Both
rules must fit the explicit 1 GiB per-batch rule cap. Unresolved convergence,
nonfinite normalized values, invalid geometry, unsupported precision or
unrepresentable support scaling reject. A requested order of 64 cannot supply
a higher comparison rule and is rejected for a nonempty evaluation. Empty
queries/masks, repeated/subset outputs and frozen CUDA graph actions are
supported. The underlying `sphere_shell.functions(...).element/output` contract
is composable and caller-owned; low-level callers must enforce preparation,
geometry validity and resource/accuracy policy themselves.

The independent rational cap reference covers 12,320 channels per precision,
including a full 2,200-channel degree-nine/second-derivative mask, small/large
spheres, near-centre, containment, contact and near-tangent cases. It cross-checks
784 baseline channels against the older independent radial-shell oracle. Finite
CPU/CUDA worst mixed errors are 4.094e-6 fp32 and 5.499e-15 fp64, without order
promotion. The fp32 SM120 module has zero fp64 instructions; sphere geometry is
also explicitly selected-dtype arithmetic. This is finite acceptance, not a
universal bound or positivity guarantee.

For W2 on actual CUDA, 128 baseline queries take about 0.34 ms warm / 1.39 ms
complete with 2.69 MB accepted native arrays. A one-query 2,200-output mask takes
3.74 / 14.28 ms fp32; eight such queries take 3.30 / 15.14 ms with 2.13 MB owned
arrays. Complete calls include allocation, two-order checks and readback;
comparison arrays, static profiles, host convergence arrays and total CUDA
reservation are separate. Analytic `RadialEvaluator("sphere")` retains its
28-channel baseline API. No curved-element FEM basis, sphere-array acceleration,
surface-area integration, pressure/porosity law or shape adjoint is implied.
See the [derivation](derivations/weighted-sphere-shell.md) and plan checkpoint.


## Higher-weighted 2D disks and solid fibre arrays

```python
from warpSPHBoundaries.fluxbridge import WeightedDiskEvaluator, Request
from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
requests = [Request((0,0)), Request((4,5)), Request((3,2),(0,1))]
disk = WeightedDiskEvaluator(requests, kernel="w6", precision="float32",
                             nodes=8, device="cuda:0")
batch = disk.prepare([-.2,0,.5], [.6,.8], 1., radii=.5)
values = batch.numpy()  # preparation already evaluates/checks both rules
batch.launch()         # frozen reusable action, suitable for graph capture

fibres = DiskArrayEvaluator([[0,0],[.9,.3]], [.2,.15], requests,
    method="volume_quadrature", search="device", precision="float32",
    nodes=8, device="cuda:0")
values, execution = fibres.evaluate_streamed([[.1,.1],[2.1,.1]], .8,
    periods=(2.,2.), chunk_size=128, return_metrics=True)
```

The numerical volume provider supports every 2D Cartesian moment through degree
nine and kernel derivative through order two. Distance is `|query-centre|-radius`,
normal points outward from centre to query, and h is support radius. Normalized
moments scale by `h**Request.scale`. Polynomial field products can contract those
moments; no curved FEM basis or fluid solver is supplied. Disk arrays are additive,
including overlaps, and periodic images are unique. They remain solid geometry
without an effective porous resistance. Resolved 2D flow is transverse; an axial
or 3D formulation is required for longitudinal permeability.

Default start order 24 compares to 32; explicit order 8 is validated for fp32,
with promotion to higher orders where needed. Two-order comparison is not a
certified global accuracy or positivity bound. The radius/support conditioning
policy is [1e-2,1e4] fp32 and [1e-4,1e4] fp64; finite reference radii span
0.02..100. Invalid geometry/scaling, nonfinite physical outputs/sums, unresolved
convergence and the 1 GiB per-rule storage limit reject explicitly. A nonempty
request starting at 64 cannot provide its higher comparison rule.

Radial geometry is prepared in host fp64 and stored in selected dtype. Native
sampling/contractions use factored profiles and selected-dtype Neumaier sums;
fp32 additionally uses bounded-angle fp32 sine/cosine polynomials with explicit
FMAs, avoiding the generic trig library's unused large-angle fp64 fallback.
The SM120 fp32 evaluator has zero fp64 instructions. Composable
`weighted_disk.functions(precision).element(plan, rules, scratch, work, row)`
and `.output` share the batch arithmetic, with caller-owned prepared radial
rules/offsets/query IDs, normals and disjoint scratch. Prepared geometry is frozen;
reprepare after query, support, radius or normal changes.

Independent whole-circle flux fixtures have 2,724 channels per precision, with
540 baseline cross-checks against the separate volume-shell oracle. Exact
whole-support polynomial/integration-by-parts identities check every request.
Default-order CPU/CUDA worst errors are 1.094e-6 fp32 / 3.402e-15 fp64; order-8
fp32 reaches 1.898e-6, with three of 40 cases promoting to 16. Method and
precision never change silently. The old baseline-only circle method remains
`DiskQuadratureEvaluator` and the default disk-array method.

For the measured 16-disk, one-interaction/query scene, 128 fp32 queries with all
330 outputs take about 2.18 ms warm / 21.49 ms complete at order 8, using 5.61 MB
accepted native arrays. Explicit chunk-128 streaming at 2,048 queries costs
379 ms and holds fresh-process CUDA-pool high-water near 16.87 MB, versus
257 MB for a full batch. Comparison/static/host memory and synchronization remain
separate costs. See the plan checkpoint and
[derivation](derivations/weighted-disk-volume.md) for limits and complete evidence.
