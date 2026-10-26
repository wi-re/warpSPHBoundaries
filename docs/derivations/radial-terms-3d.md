# 3D radial moments, curvature terms and small-ball expansions

**Status:** [V] finite reference, differential and asymptotic tests; general curvature variation and curved skeletons [P].
**Tier(s):** 2, 3, 4 · **Dimension:** 3D
**Depends on:** `tet-face-edge-chain.md`, `tri-mesh-surfaces-3d.md`
**Implemented in:** `mesh3d/radial.py`, `mesh3d/slender.py`, `visualizer/terms3d.py`; browser `term-engine.js`, `terms-compute.wgsl`
**Verified by:** `tests/mesh3d/test_radial_terms.py`, browser-math cross-checks, `tests/visualizer/test_terms_gpu.py`, `maple/23_radial_terms_3d.mpl`

## 1. Statement and conventions

Use dimensionless support units, `y=source-query`, `r=|y|`, and an outward
normal `n` at the closest boundary point. Positive distance `d` is outside the
solid. Physical `W_h(r)=h^-3 W(r/h)`, with the existing **3D** normalization.
Define

\[
 L=W'/r,\quad Q=L'/r,\quad R_3=Q'/r,
\]

\[
 M_\alpha=\int_\Omega y^\alpha W,\quad
 G_{\alpha,j}=-\int_\Omega y^\alpha y_j L,\quad
 H_{ij}=\int_\Omega(\delta_{ij}L+y_i y_jQ),\quad
 \Delta\lambda=\operatorname{tr}H.
\]

The viewer exposes 30 scalar selections: the existing 22-entry moment layout,
six symmetric Hessian entries, their trace (Laplacian), and gradient magnitude.
The layout retains both off-diagonal `g1` entries for compatibility. For a
radial scalar kernel `g1_ij=-∫y_i y_j L` is symmetric, so the 22 stored moment
entries have **19 independent components**, not 22. The Hessian trace and
vector norm are also derived quantities.

The derivative of a translated moment is distinct:

\[
 \partial_{x_j}M_\alpha=G_{\alpha,j}-\alpha_jM_{\alpha-e_j}.
\]

Display/export scaling is `M_alpha/h^|alpha|`, `h G0`, `g1`, `h²H`, `h²Deltaλ`.
The evaluators differentiate the kernel integrand, not an already sampled mass
grid. The browser uses these same normalized units.

## 2. Assumptions and validity

Sphere and planar shell integrations are exact-degree polynomial integrations
in exact arithmetic, evaluated numerically with 16-node Gauss rules after
splitting every radial shell and kernel knot. Floating-point rounding remains.

A local curvature model requires a smooth, single-sheet graph and controlled
curvature over the support. The interface currently uses an umbilic sphere
(`κ1=κ2=1/R`) and requires `R/h≥2`. This is a model domain, **not** an accuracy
guarantee. Arbitrary SDF Hessian reconstruction, curvature variation, medial
ambiguities and guard/fallback dispatch are not supplied by this viewer.
Nearest-normal ties on nonsmooth/medial geometry use a deterministic face;
local-plane geometry error remains explicit.

The small-ball model requires `a/h≤0.3`, a ball excluding the radial origin,
and a single smooth kernel piece over its entire radius. Origin, support and
cubic-knot crossings return missing values, unless the ball is wholly outside
support, where the integral is exactly zero. The fixed radius cutoff is not
an error certificate.

## 3. Derivation

### 3.1 Plane/sphere angular integrals

Align the polar axis with `n`; write `u=cos(theta)`. A radial shell's solid
portion is `u≤c`, clamped to `[-1,1]`. For a half-space and sphere,

\[
 c_{\rm plane}=-d/r,\qquad
 c_{\rm sphere}=\frac{a^2-D^2-r^2}{2Dr},\quad D=|x-c_{\rm sphere\ centre}|.
\]

The nonzero angular factors are

\[
 A_0=2\pi(c+1),\quad A_1=\pi(c^2-1),\quad
 A_n=\frac{2\pi}{3}(c^3+1),\quad
 A_t=\pi(c-c^3/3+2/3).
\]

`A_t` is one transverse component. The nine axisymmetric invariants are

\[
\begin{aligned}
 m_0&=\int r^2WA_0dr,&m_1&=\int r^3WA_1dr,\\
 m_{2,t/n}&=\int r^4WA_{t/n}dr,&g_0&=-\int r^3LA_1dr,\\
 g_{1,t/n}&=-\int r^4LA_{t/n}dr,&
 H_{t/n}&=\int[r^2LA_0+r^4QA_{t/n}]dr.
\end{aligned}
\]

Recover global tensors using `m1=m1_ax n`, `G0=g0_ax n`, and
`T=T_t I+(T_n-T_t)n⊗n` for `M2`, `g1` and `H`.
At the sphere centre the complete-shell factors make tensors isotropic and
vectors zero, so the arbitrary polar axis does not affect the result.

For a half-space, complete shells occupy `0<r<max(-d,0)`; partial shells occupy
`|d|<r<1`. For a sphere of radius `a`, complete shells occupy
`0<r<max(a-D,0)`, and partial shells occupy `|a-D|<r<min(1,a+D)`.
The cubic knot at `r=1/2` is a separate split. Substituting the cap formula
cancels negative radial powers; all assembled integrands are piecewise
polynomials of degree below 32 for the four supported kernels.

### 3.2 Order-two quadratic-graph curvature model

At the closest point use principal coordinates `(u,v,z)` with normal `+z`;
the query is at the coordinate origin and the tangent plane is `z=-d`.
For positive convex outward curvature the solid is locally

\[
 z\le-d-\tfrac12 Q_2(u,v),\qquad Q_2=\kappa_1u^2+\kappa_2v^2.
\]

For **each** smooth moment/derivative integrand `F(y)`, expand the displaced
upper integration limit, with `delta=-Q2/2`:

\[
 \boxed{I[F]=I_{\rm plane}[F]-\tfrac12\int Q_2F(u,v,-d)\,dudv
                  +\tfrac18\int Q_2^2\partial_zF(u,v,-d)\,dudv+\cdots.}
\]

The circle used by these plane integrals is `u²+v²≤1-d²`. For the smooth
compact kernels used here, support-boundary terms vanish at this order;
individual discontinuous truncated blocks must not be differentiated alone.
This is a graph/curvature expansion. Omitted cubic/quartic geometric jets need
separate consideration for a varying surface; bounding only pointwise
principal curvatures does not bound those terms.

For mass, put

\[
 J(d)=\int_{|d|}^1 r(r^2-d^2)W(r)dr,\quad
 H=(\kappa_1+\kappa_2)/2,\quad K=\kappa_1\kappa_2.
\]

Using `∫Q2 dtheta=2πH rho²`,
`∫Q2² dtheta=π(3H²-K)rho⁴`, and integration by parts,

\[
 \int_{|d|}^1(r^2-d^2)^2W'(r)dr=-4J(d),
\]

\[
 \boxed{\lambda=F_0(d)-\pi HJ(d)
                 +\frac{\pi d}{2}(3H^2-K)J(d)+\cdots.}
\]

The sphere provides a particularly useful independent check. If its far side
is outside support, its exact shell formula gives

\[
 \lambda_{\rm sphere}=F_0(d)-\frac{\pi J(d)}{a+d}
 =F_0-\pi J/a+\pi dJ/a^2+O(a^{-3}).
\]

This verifies both curvature coefficients and the convex-solid sign.
The interface evaluates the **general integrand expansion**, rather than
reusing this scalar correction for other channels.

For an umbilic graph, let `F=u^t z^p f(r)` with `t=0` or `2`, angular factor
`A=2π` or `π`, `z=-d`, and `rho²=r²-d²`. The correction integrand is

\[
 A r\left[-\frac\kappa2 z^p\rho^{t+2}f
 +\frac{\kappa^2}{8}\rho^{t+4}
 \left(pz^{p-1}f+z^{p+1}\frac{f'}r\right)\right].
\]

Choose `f` from `W,L,Q`, and derivatives from `L,Q,R3`, and combine the
polynomial factors defining the selected moment or derivative. Some Hessian corrections contain negative radial powers. The reference and
browser evaluate them with split Gauss integration; the exact-degree claim
for the plane/sphere polynomials does not apply to every curvature derivative.
Their finite quadrature error must be checked separately from geometric
truncation.

### 3.3 Small-ball / thin-volume mean-value terms

For `F` smooth on a 3D ball of radius `a` centred at `c`, integrate its Cartesian
Taylor series. Odd terms vanish. Isotropic second/fourth ball moments give

\[
 \boxed{\int_{B_a(c)}F\,dV=\frac{4\pi a^3}{3}
 \left[F(c)+\frac{a^2}{10}\Delta F(c)
             +\frac{a^4}{280}\Delta^2F(c)+O(a^6)\right].}
\]

The coefficients follow directly from
`E[y_i y_j]=a²δij/5` and
`E[y_i y_j y_k y_l]=a⁴(δijδkl+δikδjl+δilδjk)/35`, including the Taylor
factorials. For each expanded polynomial-kernel term,

\[
 \boxed{\Delta(y^\alpha r^p)=p(p+1+2|\alpha|)y^\alpha r^{p-2}
 +\sum_i\alpha_i(\alpha_i-1)y^{\alpha-2e_i}r^p.}
\]

Apply this twice to the **selected integrand**, not just to `W`: this supplies
small-ball moments, kernel gradients, first gradient moments, Hessians and
Laplacian through the same order. Kernel piece coefficients are assembled
before differentiation. Outside the permitted smooth piece no Taylor error
claim is made; the output remains missing.

## 4. Special cases and limits

Whole support gives `m0=1`, `m1=G0=0`, `g1=I`, isotropic `M2`, and zero Hessian
and Laplacian. At a planar boundary mass is `1/2`; the normal gradient is finite,
and the integrated Hessian/Laplacian vanishes for the symmetric kernels.
As curvature tends to zero the order-two family becomes the complete plane
family. Small-ball order zero becomes its volume times the integrand at its
centre; the order-two/four terms describe finite thickness.

## 5. Checks

| Check | Method | Scope / tolerance |
|---|---|---|
| All planar moment/derivative channels | independent arbitrary-precision tetrahedral face/edge reduction, rational oblique plane | four kernels, both signs of d, 3e-12 |
| Sphere mass | existing independent scalar shell code / rational closed forms | float64 epsilon scale |
| Sphere gradient and Hessian | central differences of independent mass / analytic gradient | 2e-8 / 3e-7 |
| Full order-two curvature family | exact sphere invariants at radii 4,8,16 | observed third-order remainder |
| Small-ball complete family | exact sphere integration at two small radii | stated finite asymptotic tolerances; origin/knot/support guards |
| Browser JavaScript | Python field catalogue and numerical cross-check, whole-support identities | float64 finite cases |
| Actual compute WGSL | execute native software Vulkan, compare all 30 scalar selections and four kernels | max error <3e-5 max(1, channel max), plane/sphere/curvature; not hardware performance |
| Local algebra | Maple product rule, angular factors, curvature invariant, Taylor coefficients | exact |

Tests establish these finite cases, not a global few-epsilon or model-error
bound. Browser f32 results are distinct from arbitrary-precision references.

## 6. Known failure modes

Radial derivatives near zero need the assembled origin polynomial, rather
than subtracting nearly equal terms after division. Higher radial derivatives
can be individually singular for C2 kernels; the complete Cartesian integrand
has the relevant regular limits. At the exactly coincident sample the browser
uses the finite full-integrand limit. Cubature still has sharp-occupancy
aliasing and finite angular/radial sampling error; it must not be labelled an
analytic mesh evaluator. More samples improve a study, but do not certify it.

## 7. Open questions

Non-umbilic browser curvature controls, varying curvature/geometric jets,
arbitrary imported meshes, analytic f32 face/edge compilation, curved strands,
and plates/shell skeleton expansions remain separate work. The small-ball
polynomial series currently runs in the browser float64 worker; WebGPU directly
computes the sphere, plane, umbilic curvature and solid-cubature families.
