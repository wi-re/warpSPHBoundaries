# Curvature-Aware Semi-Analytic SPH Boundary Integrals

## Project status

- [x] Repository created
- [x] Maple symbolic environment working (Maple 2026, batch harness verified 2026-10-01)
- [x] Python numerical validation environment working (`warp` conda env, verified 2026-10-01)
- [x] Planar result reproduced from the reference paper (Maple symbolic = eq. 18 exactly; Python quadrature agrees to 1.4e-15; h-scaling verified — see §18)
- [x] **Focus shift (2026-10-01 addendum): Wendland kernels** — single-polynomial `(1-q)^n P(q)` form, no midway knot, so the general `q^p` monomial machinery applies (see §18b)
- [x] Wendland planar closed forms derived + validated (w2/w4/w6; Maple symbolic + 50-digit quad + independent 40-digit Python oracle)
- [x] Spherical boundary derived and validated **for Wendland kernels w2/w4/w6** (two-branch closed form, exact branch join, planar limit — see §18b)
- [x] **2-D planar closed forms derived + validated for all 4 kernels** (cubic-spline branches A/B + Wendland by-parts/arccos-ln forms; exact join & boundary values — see §18c)
- [x] **Toy harnesses promoted to repo layout** (`maple/`, `python/curvbound/`, `tests/` = 49 passing, `docs/derivation.md`, executed `notebooks/demo.ipynb` — see §18c)
- [ ] 2-D circle boundary derived and validated *(next; oracle for tier 3 in 2-D)*
- [ ] Cylindrical boundary derived and validated *(optional exact; likely Decision B — see "Relation to HANDOFF.md")*
- [x] Docs skeleton + backends/verification plan added (`docs/README.md`, `docs/notation.md`, `docs/derivations/`, `docs/backends-and-verification.md`)
- [ ] Spherical boundary for the cubic spline (paper's kernel) — lower priority now
- [ ] General quadratic surface formulation derived
- [ ] Curvature expansion derived
- [ ] Exact/general numerical formulation validated
- [ ] SPH-ready approximation selected
- [ ] Results documented

---

## Relation to HANDOFF.md (added 2026-10-01, plan paused for review)

`HANDOFF.md` is the master roadmap; it covers a different method (exact
polygon/tet integrals via divergence-theorem edge reductions, FEM fields) and
a four-tier structure (§10 there). **This plan is tier 3** (implicit surface +
curvature expansion) **and the source of exact oracles for the other tiers.**
Read `HANDOFF.md` and `docs/README.md` before continuing.

### What this plan provides to the handoff

| PLAN.md result | role in the handoff roadmap |
|---|---|
| 2-D planar closed forms (§18c) | exact half-plane limit the edge identity must reproduce (`docs/derivations/edge-value-identity.md` §4) |
| 3-D planar + sphere closed forms (§18b) | tier-1/2 mesh-convergence oracle; the sphere is also the tier-4 3-D ball primitive (shell theorem) |
| curvature series (§8) | tier 3 deliverable and the continuity target at tier switches |
| exact sphere / cylinder expansions | settles the Tube Maps `K` sign question (handoff §1) |

### Revised priorities for this plan

1. **2-D circle boundary (exact)** — chord of a disk, same by-parts pattern as
   `docs/derivation.md` §3. Oracle for the 2-D tier 3 expansion
   (`docs/derivations/tier3-curvature-2d.md`) and for tier 2 polygon
   convergence.
2. **Curvature series** (§8): 2-D `F_1, F_2` from the circle first, then 3-D
   `H, K` structure. The exact sphere gives one combination of `H², K`; separating
   them needs the cylinder (see 4).
3. **Inside-solid (`d < 0`) and gradient terms** (`∇λ`, first moment) for
   planar/circle/sphere — the handoff's operators are gradients and moments, not
   only the value.
4. **Cylinder (3-D)** — demoted to *optional exact*. The §18c note that its shell
   area is "algebraic circle-circle" is likely wrong for the 3-D infinite cylinder
   (sphere ∩ cylinder is not a circle-circle intersection); the handoff §11 expects
   an elliptic/periodic cross-section integral, i.e. Decision B (semi-analytic with
   one residual 1-D integral or high-precision quadrature). High-precision
   quadrature is sufficient to get the series coefficients that separate `H²` from `K`.
5. Cubic-spline sphere: unchanged (low priority).

### Scope boundaries (two tracks, one repo)

- Do **not** edit `HANDOFF.md`, `edge_identity_check.py`,
  `monomial_edge_forms.py` or the files in `docs/derivations/` other than
  marking their check tables and status as results arrive.
- Keep this plan's derivation in `docs/derivation.md`; new derivations go in
  `docs/derivations/<name>.md` following `docs/derivations/TEMPLATE.md`.
- Conventions live in `docs/notation.md`. The sign of `κ` is fixed there
  (convex solid ⇒ `κ > 0`); if the first derivation finds otherwise, change
  `notation.md` first.
- Verification follows `docs/backends-and-verification.md`: mpmath reference +
  independent quadrature, with golden fixtures later consumed by numpy/torch/warp.

---

## Environment and key references

### Reference paper

Winchenbach, Akhunov & Kolb 2020 (author's accepted manuscript, 17 pages):

```
/home/lu26029/dev/warpSPH/literature/winchenbach2020_semi-analytic-boundaries.pdf
```

Equations this project reproduces and builds on:

| Eq. | Content |
|-----|---------|
| (3) | `W(r,h) = C_n/h^n * W_hat(r/h)` — kernel scaling |
| (4) | Cubic spline kernel `W_hat(q) = [1-q]_+^3 - 4[1/2-q]_+^3` |
| (15)–(17) | 3-D planar integral `lambda_3(d) = Int(C3*W_hat(q)*2*Pi*q*(q-d), q=d..1)` with spherical-cap area `A(q,d) = 2*Pi*q*(q-d)` (eq. 16) |
| (18) | **3-D planar closed form — the reproduction target** |
| (56)–(64) | Appendix: piecewise integrands `f`, `g` and their definite integrals (cross-checks) |

### Kernel reference: warpSPH / warpSPHCore

The kernel used is the same one warpSPH ships, so the code is the
implementation reference for anything kernel-shaped:

- Implementation: `/home/lu26029/dev/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py`
  - `cubicSpline_k(q) = (1-q)^3 - 4*cpow_warp(0.5-q, 3)` (`cpow_warp` = clamped power, i.e. `max(0, .)^3`) — identical to paper eq. (4)
  - `cubicSpline_C_d`: C1 = 8/3, C2 = 80/(7*Pi), C3 = 16/Pi
- Evaluation & conventions: `/home/lu26029/dev/warpSPHCore/src/warpSPHCore/kernels/kernel.py` (`sphKernel_`): `q = r/h` where `h` **is the support radius** (W = 0 for q > 1), `W = C_d * W_hat(q) / h^d`. Same convention as the paper — no rescaling needed.
- Machine-readable spec (normalization, moments, support, FT): `/home/lu26029/dev/warpSPHCore/scripts/kernels/kernel_specs.yaml`, entry `cubic` (Dehnen & Aly 2012 b4; C3 = 16/Pi, sigma2 = 3/40 in 3-D).

### Maple

- Installed locally at `~/maple2026/bin/maple` (Maple 2026; **not on PATH**).
- Batch run: `~/maple2026/bin/maple -q script.mpl` (stdout of `print`/`printf` streams back).
- Gotchas found during the 2026-10-01 toy runs:
  - `simplify(Int(f(q,d), q=a..b))` does **not** evaluate when `f` is a procedure — write the integrand explicitly and use `int`.
  - `printf("%s", expr)` rejects exact (non-float) expressions — use `print(...)` or `evalf` first.
  - For numeric quadrature of a piecewise kernel use `piecewise(q <= 1/2, k1(q), k2(q))`; calling the `if` function inside a numeric `Int` fails ("cannot determine if this expression is true or false").
  - `name` is **protected** — never use it as a variable (e.g. in `for name in ...`).
  - Arrow procedures with `add`/`sum` need explicit `local j;` (implicit-local warnings otherwise).
  - `fprintf(f, "text\n")`: the `\n` is interpreted **inside the format string** (verified), but never pass a newline as an *argument* — `chr(10)` is not a string in this Maple ("string expected for string format").
  - `fprintf(f, ...);` terminated with `;` **prints the number of characters written to stdout** — terminate with `:`.
  - `type(x, polynom(v))` / `polynomial(v)` / `polynom([v])` are not reliable here (error or wrong `false`). Polynomiality test that works: `degree(x, v)` returns `false` iff `x` is not a polynomial in `v`.
  - `numer`/`denom` on a rational function extract a **constant gcd** (e.g. `denom(3*(x+1)) = 3`), and `denom` of an expanded sum returns the LCM of the coefficient denominators — do not use `denom(...) = 1` as a polynomiality test.
  - `type(x, rational)` means "rational **number**", not rational function.

### Python

- Use the existing conda env: `/home/lu26029/miniconda3/envs/warp/bin/python`
  (numpy 2.5.2, scipy 1.18.0, mpmath 1.3.0, sympy 1.14.0, matplotlib 3.11.1, pandas 3.0.5, pytest 9.1.1 — all required packages present, no venv needed).
- Shared/oversubscribed box: cap BLAS threads for numerics, e.g.
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- `pytest-timeout` is NOT installed — never pass `--timeout` to pytest.
- NumPy 2.x: use `np.ptp(x)` (function form), not the removed `x.ptp()`.

### Scratch files

Probe scripts and intermediate output go in `<repo>/.tmp/` (gitignored). The
validated toy harness from the 2026-10-01 session lives in
`.tmp/toy_planar/` and is the seed for `maple/01_planar.mpl` + `tests/test_planar.py`.

---

## 1. Objective

Extend the semi-analytic SPH boundary-integral method of Winchenbach, Akhunov and Kolb from a locally planar boundary representation to a curvature-aware local differential-geometric representation.

The reference method evaluates the boundary contribution

\[
\lambda(\mathbf{x}) =
\int_{D_B(\mathbf{x})}
W(\|\mathbf{x}-\mathbf{x}'\|,h)\,d\mathbf{x}'.
\]

For arbitrary boundaries, the paper constructs a local tangent plane from a signed-distance field (SDF). The proposed project asks whether that first-order geometric approximation can be replaced by a second-order local surface model, using principal curvatures.

The target is a dimensionless formulation of the form

\[
\boxed{
\lambda =
F\!\left(
\frac{d}{h},
h\kappa_1,
h\kappa_2
\right)
}
\]

where:

- \(d\) = signed distance from the SPH particle to the boundary,
- \(h\) = kernel support radius,
- \(\kappa_1,\kappa_2\) = local principal curvatures.

The project should determine:

1. whether an exact closed form exists for useful kernel/geometry combinations;
2. whether a controlled curvature expansion is available;
3. whether the resulting expression is sufficiently cheap and robust for SPH;
4. how much accuracy is gained over the planar approximation.

---

# 2. Reference method to reproduce first

The paper's method represents the boundary locally as a plane and evaluates the boundary integral analytically.

The paper states that the locally planar assumption works well for smooth boundaries and becomes more accurate as the support radius decreases. It also notes that high-curvature regions may require increased particle resolution.

The central boundary quantity is

\[
\lambda(\mathbf{x}) =
\int_{D_B(\mathbf{x})}
W(\|\mathbf{x}-\mathbf{x}'\|,h)\,d\mathbf{x}'.
\]

The paper introduces a normalized distance and exploits kernel scaling so that the planar solution can be expressed independently of absolute support radius.

### First milestone

Reproduce the paper's planar 3-D result in Maple.

This is a mandatory unit test. Do not begin the curvature derivation until this passes.

Reference source (see "Environment and key references" above for the full
equation map):

`/home/lu26029/dev/warpSPH/literature/winchenbach2020_semi-analytic-boundaries.pdf`

Target: eq. (18) of the paper,

\[
\lambda_3(d) =
\begin{cases}
\dfrac{1}{60}\left[192d^6 - 288d^5 + 160d^3 - 84d + 30\right], & d \in [0, 0.5],\\[6pt]
-\dfrac{8}{15}\left[2d^6 - 9d^5 + 15d^4 - 10d^3 + 3d - 1\right], & d \in (0.5, 1],\\[6pt]
1 - \lambda_3(-d), & d \in [-1, 0),
\end{cases}
\]

derived as

\[
\lambda_3(d) = \int_d^1 C_3\,\widehat W(q)\;2\pi q\,(q-d)\,dq,
\qquad C_3 = \frac{16}{\pi},
\]

splitting the kernel at `q = 1/2` (appendix eqs. 59–60 give the definite
integrals of each piece as an intermediate cross-check).

---

# 3. Mathematical hierarchy

The project should progress through increasingly general geometries.

## Level 0 — Plane

Local surface:

\[
w=0.
\]

Parameters:

\[
q=\frac{d}{h}.
\]

Expected result:

\[
\lambda_0 = F_0(q).
\]

This must reproduce the published planar solution.

---

## Level 1 — Cylinder

For a cylindrical surface:

\[
\kappa_1=\frac{1}{R},
\qquad
\kappa_2=0.
\]

Expected dimensionless form:

\[
\boxed{
\lambda =
F\left(
\frac{d}{h},
\frac{h}{R}
\right)
}
\]

This is the first genuinely curved test case.

It is valuable because it introduces curvature while retaining enough symmetry to make the integral tractable.

### Tasks

- [ ] Define exact cylinder geometry.
- [ ] Define inside/outside convention.
- [ ] Derive the intersection geometry.
- [ ] Derive the exact numerical integral.
- [ ] Attempt symbolic integration in Maple.
- [ ] Compare with the planar result as \(R\rightarrow\infty\).
- [ ] Compare against high-precision numerical quadrature.
- [ ] Expand for small \(h/R\).
- [ ] Quantify planar approximation error versus \(h/R\).

---

## Level 2 — Sphere

For a sphere:

\[
\kappa_1=\kappa_2=\frac{1}{R}.
\]

Expected form:

\[
\boxed{
\lambda =
F\left(
\frac{d}{h},
\frac{h}{R},
\frac{h}{R}
\right)
}
\]

This provides an independent validation of the curvature treatment.

### Tasks

- [ ] Derive exact sphere geometry.
- [ ] Obtain numerical reference integral.
- [ ] Attempt symbolic solution.
- [ ] Verify \(R\rightarrow\infty\) gives the planar solution.
- [ ] Compare sphere and cylinder at equal mean curvature.
- [ ] Determine which curvature invariants appear.

---

# 4. Level 3 — General quadratic surface

At the closest boundary point, introduce orthonormal coordinates:

- \(u,v\): tangent-plane coordinates;
- \(w\): normal coordinate.

In principal-curvature coordinates, the second-order local surface is

\[
\boxed{
w =
\frac{1}{2}
\left(
\kappa_1u^2+\kappa_2v^2
\right)
}
\]

up to the sign convention selected for the normal.

This is the local second-order differential-geometric model.

The plane is recovered by

\[
\kappa_1=\kappa_2=0.
\]

A cylinder is recovered by

\[
\kappa_1=1/R,\qquad \kappa_2=0.
\]

A sphere is recovered by

\[
\kappa_1=\kappa_2=1/R.
\]

### Important point

A single scalar "curvature" is not sufficient in general.

The second-order geometry requires the two principal curvatures, or equivalently the second fundamental form / shape operator.

---

# 5. Differential geometry from the SDF

The reference method already obtains:

\[
d=\Phi(\mathbf{x})
\]

and

\[
\hat{\mathbf n}
=
\frac{\nabla\Phi}{|\nabla\Phi|}.
\]

The proposed extension should investigate the next derivative:

\[
H_\Phi = \nabla^2\Phi.
\]

For a sufficiently smooth signed-distance field, the Hessian contains the local curvature information.

At the boundary, the tangent-plane restriction of the Hessian is related to the shape operator.

Therefore the conceptual pipeline becomes:

\[
\boxed{
\Phi
\rightarrow
\nabla\Phi
\rightarrow
\nabla^2\Phi
\rightarrow
(d,\mathbf n,\kappa_1,\kappa_2)
\rightarrow
\lambda
}
\]

This is one of the main ideas to test.

### Tasks

- [ ] Establish precise SDF sign convention.
- [ ] Derive relationship between SDF Hessian and principal curvatures.
- [ ] Determine whether curvature should be evaluated at the closest point or particle position.
- [ ] Quantify numerical sensitivity of curvature obtained from an SDF.
- [ ] Define behaviour near discontinuous curvature / non-smooth geometry.
- [ ] Determine what happens when the SDF is only numerically approximate.

---

# 6. Kernel formulation

Start with the cubic spline kernel used by the reference derivation, which is
identical to warpSPHCore's `cubicSpline_k` (paper eq. 4):

\[
\widehat W(q) =
\begin{cases}
(1-q)^3 - 4\left(\tfrac12 - q\right)^3, & 0 \le q \le \tfrac12,\\[2pt]
(1-q)^3, & \tfrac12 \le q \le 1,\\
0, & q > 1,
\end{cases}
\qquad C_3 = \frac{16}{\pi}.
\]

Useful exact identity for the symbolic path (the low-q piece collapses to a cubic):

\[
(1-q)^3 - 4\left(\tfrac12 - q\right)^3 = \tfrac12 - 3q^2 + 3q^3.
\]

Consequence: in 3-D every planar integrand is a plain polynomial in `q`, so no
special functions appear at Level 0 (the 2-D result instead contains `acos`
terms, appendix eqs. 56–58).

**Working focus: the Wendland kernels** (2026-10-01 addendum). Unlike the
cubic spline, each Wendland kernel is a *single* polynomial times
`(1-q)^n` on `[0,1]` — no midway knot at `q = 1/2`, so no piecewise split
and the derivation works for a general `q^p` term (see §18b for the
monomial basis and the resulting closed forms). The cubic spline remains
the paper-reproduction baseline, but the curvature work proceeds with
Wendland. Definitions (warpSPHCore, `kernelFunctions/wendland{2,4,6}.py`,
dim=3; same `q = r/h`, `h` = support convention):

| kernel | `W_hat(q) = (1-q)^n P(q)` on [0,1] | n | P(q) | C3 |
|--------|------------------------------------|---|------|----|
| w2 | `(1-q)^4 (1+4q)` | 4 | `1 + 4q` | `21/(2*Pi)` |
| w4 | `(1-q)^6 (1+6q+35/3 q^2)` | 6 | `1 + 6q + 35/3 q^2` | `495/(32*Pi)` |
| w6 | `(1-q)^8 (1+8q+25q^2+32q^3)` | 8 | `1 + 8q + 25q^2 + 32q^3` | `1365/(64*Pi)` |

Write the kernel in dimensionless form:

\[
r=hq.
\]

Then

\[
W(r,h)
=
\frac{C_3}{h^3}\widehat W(q)
\]

for the 3-D case, with the exact kernel definition as above.

The volume element contributes \(h^3\), so the explicit scale cancels.

This motivates the dimensionless variables:

\[
q=\frac{d}{h},
\qquad
\alpha_1=h\kappa_1,
\qquad
\alpha_2=h\kappa_2.
\]

The target therefore becomes

\[
\boxed{
\lambda =
F(q,\alpha_1,\alpha_2)
}
\]

rather than a function of dimensional \(d,h,\kappa_1,\kappa_2\) independently.

---

# 7. Geometric integration strategy

Do not initially ask Maple to solve the complete 3-D integral directly.

Instead separate:

1. kernel dependence;
2. geometry.

Conceptually write

\[
\lambda
=
\int
W(r,h)
A(r;d,\kappa_1,\kappa_2)\,dr
\]

where \(A\) is the area of the portion of the radius-\(r\) sphere lying inside the boundary region.

For the planar case this reduces to a spherical-cap area.

For curved geometry, the new mathematical object is

\[
A(r;d,\kappa_1,\kappa_2).
\]

### Tasks

- [ ] Reconstruct planar spherical-cap geometry.
- [ ] Derive \(A(r)\) for a cylinder.
- [ ] Derive \(A(r)\) for a sphere.
- [ ] Derive/approximate \(A(r)\) for a quadratic surface.
- [ ] Identify the admissible integration ranges.
- [ ] Handle partial-support cases \(0<d/h<1\).
- [ ] Handle particles on the boundary.
- [ ] Handle particles inside the boundary.
- [ ] Verify symmetry and normalization.

---

# 8. Curvature expansion

If a complete closed form becomes unwieldy, derive a controlled expansion in

\[
\alpha_i=h\kappa_i.
\]

The expected structure is something like

\[
\lambda =
F_0(q)
+
\alpha_1F_{1}(q)
+
\alpha_2F_{2}(q)
+
\alpha_1^2F_{11}(q)
+
\alpha_1\alpha_2F_{12}(q)
+
\alpha_2^2F_{22}(q)
+\cdots.
\]

Because the two principal directions are interchangeable, investigate whether the expression can instead be written using curvature invariants such as

\[
H=\frac{\kappa_1+\kappa_2}{2}
\]

and

\[
K=\kappa_1\kappa_2.
\]

Do not assume the final invariant structure before deriving it.

### Tasks

- [ ] Compute first-order expansion.
- [ ] Compute second-order expansion.
- [ ] Test whether only mean curvature appears at first order.
- [ ] Determine second-order invariant combinations.
- [ ] Compare expansion against exact cylinder.
- [ ] Compare expansion against exact sphere.
- [ ] Determine practical range of \(h\kappa\) for a chosen error tolerance.

---

# 9. Maple harness

## Installation

Maple 2026 is already installed at `~/maple2026/bin/maple` (not on PATH).
Batch run: `~/maple2026/bin/maple -q script.mpl` — see
"Environment and key references" for the gotchas already hit.

Create a project structure such as (this repository is `curvatureBoundaries`):

```text
curvatureBoundaries/
├── README.md
├── LICENSE
├── docs/
│   ├── derivation.md
│   └── notation.md
├── maple/
│   ├── 00_setup.mpl
│   ├── 01_planar.mpl
│   ├── 02_cylinder.mpl
│   ├── 03_sphere.mpl
│   ├── 04_quadratic_surface.mpl
│   ├── 05_curvature_series.mpl
│   └── 99_checks.mpl
├── python/
│   ├── requirements.txt
│   ├── numerical_integrals.py
│   ├── validation.py
│   ├── parameter_sweep.py
│   └── plotting.py
├── results/
│   ├── symbolic/
│   ├── numerical/
│   └── figures/
├── tests/
    ├── test_planar.py
    ├── test_cylinder.py
    └── test_sphere.py
└── .tmp/          # scratch, gitignored
```

### Maple setup

- [x] Verify symbolic integration. *(toy: both planar branches integrate to paper eq. 18 exactly)*
- [ ] Verify assumptions work correctly.
- [x] Verify arbitrary precision arithmetic. *(toy: 50-digit `evalf` quadrature matches symbolic to all printed digits)*
- [x] Verify `simplify`, `combine`, `factor`, `series`, `limit`. *(toy Wendland run: `simplify` identities, `limit(R→∞)`, exact branch join — see §18b)*
- [ ] Create standard assumptions for \(h>0\), \(R>0\), etc.
- [x] Define the kernel once. *(cubic spline in `maple_toy.mpl`; Wendland w2/w4/w6 + F(n,p,·) basis in `maple_wendland.mpl`)*
- [x] Define common dimensionless variables once. *(q, d, R, D = R+d in `maple_wendland.mpl`)*
- [x] Define reusable geometry functions. *(planar cap A(q,d) and spherical cap A(q) in `maple_wendland.mpl`)*

Use exact arithmetic wherever possible.

Avoid floating-point constants during symbolic derivation.

---

# 10. Python validation harness

Python is not the primary symbolic derivation environment.

It is the independent numerical oracle.

Suggested packages:

```text
numpy
scipy
mpmath
sympy
matplotlib
pandas
pytest
```

Optional:

```text
jupyter
```

### Setup

Use the existing conda env (all packages above are already installed —
verified 2026-10-01):

```bash
/home/lu26029/miniconda3/envs/warp/bin/python
```

An isolated venv is an alternative if the project must not depend on it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

Cap BLAS threads when running numerics on this shared box
(see "Environment and key references").

### Tasks

- [x] Create virtual environment. *(existing `warp` conda env used instead)*
- [x] Install dependencies. *(numpy, scipy, mpmath, sympy, matplotlib, pandas, pytest all present)*
- [ ] Implement reference kernel.
- [ ] Implement direct numerical volume integration.
- [ ] Implement geometry-specific quadrature.
- [ ] Add high-precision `mpmath` validation.
- [ ] Add automated comparison against Maple-generated expressions.
- [ ] Add plots of relative error.

---

# 11. Validation philosophy

Every symbolic result should be independently checked.

For each geometry calculate

\[
\lambda_{\rm analytic}
\]

and

\[
\lambda_{\rm numerical}.
\]

Then calculate

\[
\epsilon_{\rm rel}
=
\frac{
|\lambda_{\rm analytic}-\lambda_{\rm numerical}|
}{
|\lambda_{\rm numerical}|
}.
\]

Use high precision for validation so that numerical quadrature error is substantially below the expected symbolic error.

### Required checks

- [ ] Dimensional consistency.
- [ ] \(h\rightarrow\) scaling invariance.
- [ ] \(d/h\rightarrow 1\) support-limit behaviour.
- [ ] \(d/h=0\) boundary behaviour.
- [ ] \(d/h<0\) interior behaviour.
- [ ] Flat limit \(\kappa_i\rightarrow0\).
- [ ] Cylinder limit.
- [ ] Sphere limit.
- [ ] Symmetry under \(\kappa_1\leftrightarrow\kappa_2\).
- [ ] Numerical convergence with quadrature tolerance.
- [ ] Numerical convergence with arbitrary precision.

---

# 12. Critical research question: what does "exact" mean?

Keep three notions separate.

### A. Exact for the planar model

The reference paper derives an analytic result for the planar geometry.

### B. Exact for the quadratic local model

If we assume

\[
w =
\frac12(\kappa_1u^2+\kappa_2v^2)
\]

then an exact integral would be exact for that mathematical surface.

It is not automatically exact for the physical boundary.

### C. Exact for the actual deforming tow

A real deforming carbon-fibre tow generally has curvature that varies spatially and may have nontrivial cross-sectional deformation.

A constant-curvature model therefore cannot generally represent the entire tow exactly.

The correct interpretation is:

> Can the local geometry be represented accurately enough over one SPH kernel support by its second-order differential geometry?

That is the practical question.

---

# 13. Carbon-fibre tow model

For the intended application, investigate the geometry separately from the integration.

A simple idealised tow can be represented as a tube around a centreline:

\[
\mathbf X(s,\theta)
=
\mathbf c(s)
+
R(s,\theta)
\left[
\cos\theta\,\mathbf n(s)
+
\sin\theta\,\mathbf b(s)
\right].
\]

For a perfectly circular, constant-radius tube, the cross-sectional curvature and centreline curvature contribute differently to the surface geometry.

Therefore:

- constant centreline curvature does not imply constant surface curvature everywhere;
- constant principal curvatures are a local approximation;
- deformation can make both principal curvatures spatially varying.

### Tasks

- [ ] Define ideal circular tube.
- [ ] Derive its principal curvatures.
- [ ] Examine dependence on angular position.
- [ ] Introduce variable radius.
- [ ] Introduce centreline curvature.
- [ ] Determine when the quadratic local model is adequate.
- [ ] Relate tow radius and bend radius to \(h\kappa\).

---

# 14. Suggested research progression

Do not jump directly to the full tow problem.

### Stage 1

\[
\text{paper plane}
\]

- [ ] Reproduce exactly.

### Stage 2

\[
\text{cylinder}
\]

- [ ] Exact numerical reference.
- [ ] Symbolic result if possible.
- [ ] Small-curvature expansion.

### Stage 3

\[
\text{sphere}
\]

- [ ] Independent curvature validation.

### Stage 4

\[
\text{quadratic surface}
\]

- [ ] General \(\kappa_1,\kappa_2\).
- [ ] Determine invariant structure.

### Stage 5

\[
\text{SDF Hessian}
\]

- [ ] Extract local curvature numerically.
- [ ] Feed curvature into \(F(q,h\kappa_1,h\kappa_2)\).

### Stage 6

\[
\text{deforming tow}
\]

- [ ] Compare local quadratic model with actual tow geometry.
- [ ] Quantify error.

---

# 15. Decision points

At each stage, record the outcome.

## Decision A — closed form

If Maple finds a manageable closed form:

- [ ] simplify it;
- [ ] validate numerically;
- [ ] document branch conditions;
- [ ] assess runtime cost.

## Decision B — semi-analytic form

If symbolic integration stops at one manageable integral:

- [ ] retain the remaining 1-D integral;
- [ ] derive stable numerical evaluation;
- [ ] precompute/tabulate if appropriate.

## Decision C — curvature series

If the exact expression is impractical:

- [ ] derive \(O(h\kappa)\);
- [ ] derive \(O((h\kappa)^2)\);
- [ ] establish error bounds empirically;
- [ ] determine a useful operating range.

## Decision D — numerical lookup

If neither closed form nor expansion is sufficiently useful:

- [ ] generate a dimensionless lookup table;
- [ ] interpolate in \(d/h,h\kappa_1,h\kappa_2\);
- [ ] compare memory and runtime against direct quadrature.

---

# 16. Performance target

The eventual SPH method should ideally reduce to something like

```text
distance d
     ↓
SDF gradient
     ↓
normal
     ↓
SDF Hessian
     ↓
principal curvatures
     ↓
q = d/h
a1 = h*kappa1
a2 = h*kappa2
     ↓
F(q,a1,a2)
```

The final evaluation should ideally be substantially cheaper than performing a fresh 3-D numerical integral for every SPH particle/boundary interaction.

Possible final implementations:

1. closed-form expression;
2. compact polynomial/rational approximation;
3. 1-D/2-D quadrature;
4. lookup table/interpolation.

---

# 17. Initial implementation checklist

## Repository

- [x] Create Git repository.
- [x] Add `.gitignore`. *(incl. `.tmp/` scratch dir)*
- [ ] Add `README.md`.
- [x] Add `LICENSE`.
- [ ] Create directory structure above.
- [x] Add reference-paper citation. *(in "Environment and key references")*
- [x] Add this plan as `PLAN.md`.

## Maple

- [x] Install Maple. *(Maple 2026 at `~/maple2026`, pre-existing)*
- [ ] Create `00_setup.mpl`.
- [ ] Define assumptions.
- [ ] Define cubic spline kernel. *(toy prototype exists in `.tmp/toy_planar/maple_toy.mpl`)*
- [ ] Define dimensionless variables.
- [x] Reproduce planar integral. *(toy, 2026-10-01)*
- [x] Reproduce planar closed form. *(toy, 2026-10-01 — both branches exact)*
- [ ] Save symbolic output.

## Python

- [x] Create virtual environment. *(existing `warp` conda env)*
- [x] Install dependencies.
- [x] Implement kernel. *(toy prototype in `.tmp/toy_planar/check_planar.py`)*
- [x] Implement direct numerical integral.
- [x] Implement planar test.
- [x] Implement error calculation.
- [ ] Add pytest tests. *(promote `check_planar.py` into `tests/test_planar.py`)*

## First scientific milestone

- [x] Maple reproduces planar result.
- [x] Python independently reproduces the same result numerically.
- [x] Relative error is within numerical precision. *(abs err 1.4e-15; see §18)*
- [x] \(h\)-scaling is verified. *(explicit h ≠ 1 quadrature agrees to ~1e-16)*
- [x] Only then begin cylinder derivation.

---

# 18. First experiment

The first useful experiment should be deliberately small.

### Question

Can Maple independently reproduce the paper's planar 3-D boundary integral and its closed-form expression?

### Inputs

\[
q=d/h
\]

and the cubic spline kernel from the reference paper.

### Outputs

1. symbolic integral;
2. simplified closed form;
3. numerical evaluation;
4. independent Python numerical evaluation;
5. relative error;
6. plot of \(\lambda(q)\).

Once this passes, copy the same harness structure to the cylinder.

### Result (2026-10-01) — PASS

Executed with the toy harness in `.tmp/toy_planar/`
(`maple_toy.mpl` run via `~/maple2026/bin/maple -q`, `check_planar.py` run with
the `warp` conda env).

Maple (exact arithmetic):

- `int` of the split integrands reproduces **both branches of paper eq. (18)
  exactly** (`simplify(LA - PA) = 0`, `simplify(LB - PB) = 0`).
- Boundary values exact: \(\lambda_3(0) = 1/2\),
  \(\lambda_3(1/2) = 1/30\) from both branches, \(\lambda_3(1) = 0\).
- 50-digit numeric quadrature of the *unsplit* piecewise-kernel integral agrees
  with the closed form at d = 0.1, 0.2, 0.3, 0.7, 0.9 to all 15 printed
  digits (abs diff 0.0).
- Interior branch d < 0: \(1 - \lambda_3(0.3) = 66979/78125\) matches the
  50-digit quadrature.

Python (scipy `quad`, independent implementation):

- 41-point grid on [0, 1]: worst absolute error **1.4e-15**
  (relative error is ill-conditioned as \(\lambda \to 0\) near d = 1).
- d < 0 branch matches (0.3, 0.7).
- h-scaling: quadrature with explicit support radius
  \((d,h) \in \{(0.6,2), (1.5,2), (0.35,0.7), (0.9,3)\}\) gives
  \(\lambda(d,h) = \lambda_3(d/h)\) to ~1e-16.

Remaining item for this milestone: item 6 (plot of \(\lambda_3(d)\)) — deferred
to the real harness in `maple/` + `python/`; not needed to start the cylinder.

---

# 18b. Second experiment — Wendland kernels: planar + spherical closed forms (2026-10-01) — PASS

### Why Wendland simplifies the derivation

Each Wendland kernel is \(\widehat W(q) = (1-q)^n P(q)\) with a plain
polynomial \(P\) on \([0,1]\) — no knot at \(q=\tfrac12\) (that belongs to
the cubic spline only). So every integrand below is a finite linear
combination of terms \(q^p(1-q)^n\), and one general antiderivative handles
all of them:

\[
F(n,p,x) \;=\; \int_0^x q^p(1-q)^n\,dq
\;=\; \sum_{j=0}^{n}(-1)^j\binom{n}{j}\frac{x^{p+j+1}}{p+j+1}.
\]

Verified in Maple: `diff(F(n,p,x),x) = x^p(1-x)^n` and
`F(n,p,1)-F(n,p,d) = int(q^p(1-q)^n, q=d..1)` for n in {4,6,8}, p in 0..10.

### Planar (3-D, unit support, particle outside the solid, d in [0,1])

With the cap area \(A(q,d) = 2\pi q(q-d)\) (paper eq. 16) and
\(q^k A = 2\pi(q^{k+2} - d\,q^{k+1})\):

\[
\lambda_{\rm plane}(d)
= 2\pi C_3 \sum_{k} c_k
\Big[
\big(F(n,k+2,1)-F(n,k+2,d)\big)
- d\,\big(F(n,k+1,1)-F(n,k+1,d)\big)
\Big],
\qquad \widehat W = (1-q)^n \textstyle\sum_k c_k q^k.
\]

**Single polynomial in d on [0,1] — no piecewise split** (the split exists
only because of the cubic spline's knot). Degrees: w2 → 8, w4 → 11,
w6 → 14 (= n + k_max + 3). Boundary values exact for all three:
\(\lambda(0) = 1/2\), \(\lambda(1) = 0\).

### Spherical boundary (solid ball of radius R, particle outside, D = R + d)

Solid-side area of the radius-q shell, derived from the intersection plane
of the two spheres (validated against a first-principles clamp formula in
Python):

\[
A(q) = \frac{\pi q}{D}\left(R^2 - D^2 + 2Dq - q^2\right),
\qquad d \le q \le \min(1,\; 2R+d),
\]

with the **second cutoff** \(q > 2R+d\): once the kernel shell engulfs the
whole obstacle there is no solid-side area left. The planar limit
\(R\to\infty\) of this cap formula gives \(2\pi q(q-d)\) exactly.
Since \(A(q)\) is a quadratic in q,

\[
\lambda(u) = \frac{\pi C_3}{D}
\left[
(R^2-D^2)\big(W_1(u)-W_1(d)\big)
+ 2D\big(W_2(u)-W_2(d)\big)
- \big(W_3(u)-W_3(d)\big)
\right],
\qquad W_j = \sum_k c_k\, F(n, k+j, \cdot),
\]

with upper limit \(u = 1\) or \(u = 2R+d\). This gives a two-branch closed
form, both branches rational with denominator \(R+d\):

- **Branch 1** (\(2R+d \ge 1\), i.e. all \(R \ge 1/2\) for d in [0,1]):
  \(\lambda = N(d)/(R+d)\), N a polynomial in d (deg 9 / 12 / 15 for
  w2/w4/w6).
- **Branch 2** (\(2R+d < 1\), support engulfs the obstacle):
  \(\lambda = N(R,d)/(R+d)\), N a bivariate polynomial (deg in R ≤
  n + k_max + 4).

### Validation (all PASS, 2026-10-01)

Maple (`.tmp/toy_planar/maple_wendland.mpl`, exact arithmetic):

- basis F verified (derivative + definite form), n in {4,6,8}, p ≤ 10;
- planar: monomial form `simplify`-equal to direct `int` for all 3 kernels;
  \(\lambda(0)=1/2\), \(\lambda(1)=0\) exact; 50-digit quadrature agrees to
  all printed digits (abs diff 0.0) at d = 0.1, 0.3, 0.7, 0.9;
- sphere: monomial form with *symbolic* upper limit u equals direct `int`
  (all 3 kernels); `limit(R→∞, branch 1) = planar` exactly;
  branch join at \(R=(1-d)/2\) exact (b1 ≡ b2);
- 50-digit quadrature: branch 1 over R in {1/2, 1, 2, 5, 20} ×
  d in {1/10, 3/10, 7/10}, and branch 2 over R in {1/8, 1/4, 3/10, 1/3} ×
  d in {1/10, 1/5, 1/3, 1/2, 3/5} with 2R+d < 1 — all within 1e-20.

Independent Python oracle (`.tmp/toy_planar/check_wendland.py`, mpmath
40-digit, reads the coefficient export `wendland_export.txt`):

- exact Fraction checks: planar poly(0) = 1/2 and poly(1) = 0 for all
  kernels; branch-1 N(1) = 0 for all 15 (kernel, R) rows; branch join at
  d = 0, R = 1/2 exact (7/24, 61/192, 43/128 for w2/w4/w6);
- 40-digit quadrature vs closed forms: planar worst abs err 3.5e-39
  (3×7 grid); branch 1 worst 2.7e-39 (15 rows × 7 d); branch 2 worst
  1.9e-39 (60 (R,d) points in 2R+d < 1);
- planar-limit sanity at R = 20: |λ_sphere − λ_plane| in
  [1.6e-4, 4.8e-3] — consistent with the expected O(1/R²) curvature
  correction, and decreasing with R and d as expected.

### Files

```text
.tmp/toy_planar/maple_wendland.mpl     # derivation + internal checks + export
.tmp/toy_planar/wendland_export.txt    # exact coefficients (30 lines)
.tmp/toy_planar/check_wendland.py      # independent 40-digit oracle
```

Export format: `planar <k> deg=m: c_m … c_0` (descending in d);
`sph b1 <k> R=r deg=m: c_m … c_0` meaning λ = N(d)/(r+d);
`sph b2 <k>: i:j:c …` meaning N(R,d) = Σ c R^i d^j, λ = N(R,d)/(R+d).

### Notes for the real harness

- The export section of `maple_wendland.mpl` is maintained via bash heredoc:
  the editor/tool layer mangles `\n` inside Maple string literals
  (see Maple gotchas). Keep it that way or regenerate carefully.
- Branch selection in SPH: branch 1 whenever \(2R + d \ge h\) (in
  dimensionless units, h = 1); branch 2 only for small obstacles
  (R < h/2) that the support engulfs.
- Next: cylinder (κ1 = 1/R, κ2 = 0) with the same F(n,p,·) machinery —
  its shell area is also algebraic (circle-circle intersection), so the
  same monomial basis should close it; then the general quadratic surface
  where the area becomes non-polynomial and a 1-D residual integral is
  expected (Decision B territory).

---

# 18c. Third experiment — 2-D planar closed forms + repo promotion (2026-10-01) — PASS

## Motivation

2-D geometry is far easier to visualize: the support **disk** cut by the
boundary line gives a chord/arc shell measure instead of a spherical cap.
This makes the branch structure, the join values, and the kernel
dependencies all directly plottable (see `notebooks/demo.ipynb`).

## Result

**2-D planar** boundary integral (unit support, particle outside, d in
[0,1]):

\[
\lambda_2(d) = \int_d^1 C_2\,\hat W(q)\, 2q\,\arccos\!\frac{d}{q}\,dq
\]

(shell measure = arc length × dq; the **single power of q** is what
distinguishes it from the 3-D cap `2πq(q−d)`).

Method: Maple cannot close `Int poly(q) arccos(d/q) dq` directly, but by
parts with `u = arccos(d/q)`, `v0(q) = Int_0^q t W(t) dt` (one power of t):

\[
\lambda_2(d) = 2C_2\Big[v_0(1)\arccos d - d\int_d^1
\frac{v_0(q)}{q\sqrt{q^2-d^2}}dq\Big]
\]

reduces to `J_m(d) = Int_d^1 q^m/sqrt(q^2-d^2) dq`, which closes for every
m (odd m: poly + sqrt(1−d²); even m: sqrt, d²·ln d, d²·ln(1+sqrt(1−d²))
terms — see `docs/derivation.md` §3).

* **Wendland w2/w4/w6**: one closed form each (arccos d + d·(poly(d²)√(1−d²)
  + poly·ln d + poly·ln(1+√(1−d²)))).
* **Cubic spline**: branch A (d ≤ 1/2, extra arccos(2d), √(1−4d²),
  ln(1+√(1−4d²)) terms) and branch B (d ≥ 1/2); exact join at d = 1/2:
  `(64π − 294√3 + 249 ln(2+√3))/(168π) ≈ 0.03748`.
* Exact boundary values for all four kernels: λ₂(0) = 1/2, λ₂(1) = 0.

## Validation (all PASS)

* In-Maple: 50-digit quadrature of the defining integrals (worst diff
  ~1e-23), antiderivative checks (`v0' = qW`), symbolic exact boundary
  limits and joins.
* Python (`tests/test_planar.py`, 40-digit mpmath, independent oracle):
  worst |closed − oracle| = **1.5e-39** over 4 kernels × 8 d; exact
  Fraction boundary/join checks; **true 2-D Cartesian quadrature**
  (no 1-D reduction, 16-subinterval outer split needed for the sqrt
  endpoint behavior — 3-way split stalls at ~3.2e-15 for cubic);
  h-scaling identity `Int (C3/h³)W_hat(r/h)2πr(r−d)dr = λ₃(d/h)`.

## Promotion (toys → repo)

The `.tmp/toy_planar/` harnesses were promoted to the §9 layout:

```
maple/00_setup.mpl          kernel table, normalization, F(n,p,x) checks
maple/01_planar.mpl         2-D + 3-D planar, all kernels -> results/symbolic/planar_export.txt
maple/03_sphere.mpl         sphere (Wendland) -> results/symbolic/sphere_export.txt
maple/run_all.sh
python/curvbound/           kernels.py, symbolic.py (Maple->mpmath translator), oracle.py
tests/                      test_kernels.py, test_planar.py, test_sphere.py (49 tests)
notebooks/demo.ipynb        executed demo (8 figures in results/figures/)
docs/derivation.md          full write-up
```

Sphere export format changed to a single `N(R,d)` triple set per branch
(branch 1 is linear in R — the `i=1` row equals the 3-D planar Wendland
polynomial, which is why `R→∞` recovers it exactly); `python/curvbound/sphere`
now works for **any** R, not just the toy's per-R rows.

## Notes / gotchas found

* Maple: `C2t["w2"]` (string) ≠ `C2t[w2]` (symbol) table keys; `add(c*q^p,
  [p,c] in P)` does not destructure pairs; a `table` assigned inside a
  `proc` is implicitly **local** unless declared `global`; `fprintf(...);`
  echoes the char count (`:` suppresses).
* Python: mpmath here cannot `mp.mpf(Fraction)` (use num/den); rational
  literals in translated Maple strings must become `mp.mpf("a/b")` or the
  result is polluted at double precision (~1e-17); the cubic-spline shape
  has W(0) = 1/2 (it is `1/2 − 3q² + 3q³` on [0,1/2]).
* Notebook: `__file__` is undefined in notebooks (detect repo from cwd);
  `mp.mpf` doesn't support `:.3e` formatting (use `mp.nstr`).

## Status

- [x] 2-D planar derived + validated (all four kernels)
- [x] Toys promoted to repo layout; `pytest tests/` = 49 passed
- [x] Demo notebook built + executed (0 errors; worst oracle error 1.5e-39)
- [ ] Next (revised 2026-10-01, see "Relation to HANDOFF.md"): 2-D circle
      boundary (chord of a disk, same by-parts pattern as §3 of
      docs/derivation.md), then the curvature series. The 3-D cylinder is
      optional/exact-only: the earlier note that its shell area is an algebraic
      circle-circle intersection is believed wrong for the infinite 3-D cylinder
      (expect Decision B).

---

# 19. Success criteria

The project is successful if it establishes a validated mapping

\[
\boxed{
(d/h,h\kappa_1,h\kappa_2)
\longrightarrow
\lambda
}
\]

with a known accuracy/cost trade-off.

A particularly useful outcome would be:

\[
\boxed{
\lambda
\approx
F_0(d/h)
+
hH\,F_1(d/h)
+
h^2
\left[
H^2F_2(d/h)
+
KF_3(d/h)
\right]
}
\]

where

\[
H=\frac{\kappa_1+\kappa_2}{2},
\qquad
K=\kappa_1\kappa_2,
\]

provided the derivation actually demonstrates that this invariant form is valid.

That would give the original semi-analytic SPH method a principled curvature correction while retaining its scale-independent character.

---

# 20. Reference

Primary source for the baseline method:

Winchenbach, R., Akhunov, R., and Kolb, A.  
**Semi-Analytic Boundary Handling Below Particle Resolution for Smoothed Particle Hydrodynamics.**  
ACM Transactions on Graphics, 39(6), Article 173, 2020.

The paper describes the analytic planar boundary integral and its extension to arbitrary geometry through local tangent planes constructed from signed distance fields.

Use the supplied PDF as the authoritative source for reproducing the baseline equations before extending them.

