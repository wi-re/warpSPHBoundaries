# The wall Laplacian Δλ = ∫_solid ∇²W dA′ (the viscosity wall term, Q1)

**Status:** [V] (identity, conversion, limits, guards, solver switch — the tests below; the physics gates in `docs/deltasph-validation.md`, WORK-005 section)
**Tier(s):** n/a (solver wall term, not an obstacle tier) · **Dimension:** 2D
**Depends on:** the kernel table (`curvbound.kernels`), the scene operations of `src/edgebound/scene/scene.py` (Density, Covariance of a registered ordinary kernel)
**Implemented in:** `src/edgebound/scene/viscosity.py::lap_lambda_scene` / `::lap_factor`; `src/edgebound/sim/deltasph2d.py` (`cfg.viscosityExact`, the wall-viscosity block of `rhs`)
**Verified by:** `tests/scene/test_viscosity_scene.py` (T5.1), `tests/sim/test_deltasph_viscosity.py` (T5.2)

Exact wall part of the naive-Laplacian artificial viscosity (Q1, the fourth and last wall quadrature of the program; the first three — λ, G, the cover — are in `deltasph2d.py`, and this is the fourth).  Default off (`cfg.viscosityExact`).

## 1. Statement

For a particle i and a solid body, with the normalised radial kernel W (support H):

```
Delta-lambda_i = int_solid lap W(x - x') dA'
               = sum_e z_e int_chord W'/r ds      (Green; z_e = signed distance of the particle to the edge line, positive on the solid side;
                                                 for a fluid-side particle z_e < 0 and W' < 0 -> Delta-lambda > 0; the outer circle contributes nothing
                                                 because W'(H) = 0; no angle term because [r W']_0 = 0)
```

(1/length^2, per body.)  Sign: **positive** for a particle on the fluid side of the wall (e.g. above a floor); 0 when the support is fully inside the solid, 0 beyond distance H, and → 0 linearly as the particle reaches the wall (z → 0).

Scene-layer route (no Warp code, no new quadrature): with the registered **ordinary normalised** kernel K_l of shape ℓ₀ = (1−q)³ (C2) / 6u⁵−5u⁶ (C4), u = 1−q, of which L(r) = W′(r)/r is a scalar multiple (L = f K_l),

```
Delta-lambda = 2 lambda[L] - tr Cov[L] = f ( 2 lambda[K_l] - tr Cov[K_l] ),      f = P c / (C_l H^2)
```

lambda the Density operation, Cov the Covariance operation (Cov = ∫ y⊗∇_x L dA′, tr Cov = −∫ r L′ dA), f = lap_factor (P = −20 for Wendland C2, −56/3 for C4; c the family normalisation c2_pi, 7 / 9; C_l the K_l normalisation, 10 / 28/3).

Wall term of the viscosity in the solver (free-slip mirror, constant mirror normal per particle, as in the pairwise form):

```
acc_wall = -2 nu_eff wallMass u_n / rho  Delta-lambda n      (per body),      nu_eff = alpha c0 H / (8 xi)
```

with u_n = (v − v_wall)·n, n the unit wall normal into the wall (the solver's own G).  Only the normal component is damped.

## 2. Assumptions and validity

* Free-slip mirror: the ghost velocity is the mirror of v across the wall plane, so v_ghost − v_i = −2 u_n n with a **constant** mirror normal n per particle (the same approximation the pairwise form makes); the solid is locally the half-plane for each particle.
* SurfaceRep bodies only (the Covariance scene operation needs the first moments of the edge, which only SurfaceRep provides); ImplicitRep / VolumeRep raise NotImplementedError.
* Wendland C2 and C4 only (the shape L = W′/r must be a truncated power series with knot 1 for the monomial edge plan; both qualify, degree 3 and 6).
* The first moments of ∇²W (a **position-dependent** mirror field) are NOT implemented: the wall term uses the scalar Δλ with the constant normal n, not the full ∫ (v_ghost − v) ∇²W dA′.

## 3. Derivation

1. **Green / divergence form.** For a radial W: ∇²W = ∇·(L y) with L = W′/r and y = x − x′, because ∇_x W(x−x′) = L(x−x′)(x−x′)/|x−x′| = L y and ∇²W = div(L y) = 2L + rL′ (d = 2; in general dL + rL′).
2. **The trace identity.** The Covariance scene operation computes Cov = ∫_solid y ⊗ ∇_x L dA′ (g1 = pairs.g1, the gradient w.r.t. the query x).  Taking the trace: tr Cov = ∫ ∇_x L · y dA′ = ∫ r L′ dA′ (L radial: ∇_x L = L′ y/r).  Hence tr Cov = −(∇²W − 2L) integrated, i.e. **∫ ∇²W dA′ = 2λ[L] − tr Cov[L]**.
3. **Why L, not ∇²W, is registered.** The scene adds a body-indicator pseudo-pair with lam = 1, g1 = ind·I — it assumes ∫ K = 1 over a full disk.  But ∫ lap W dA = 0 (divergence theorem, W and ∇W vanish at r = H): a directly registered lap W would give a spurious constant c/H² for every particle **inside** a body (28.0 = 7/0.5² measured at the centre of a square of side 2H, while the true value is 0).  With L the indicator enters as 2·ind − tr(ind·I) = 2 − 2 = 0 and **cancels exactly**; also `kernels._finish` divides by ∫ r·shape, which is 0 for a Laplacian but non-zero for L.
4. **The conversion factor.** W = c/(πH²) s(q) (q = r/H; c = float(KERNELS[family].c2_pi) = 7 for C2, 9 for C4, from the curvbound kernel table).  L = W′/r = c/(πH⁴) l(q) with l = s′/q: w2 l = −20(1−q)³ (register shape (1−q)³, P = −20); w4 l = −(56/3)(6u⁵−5u⁶), u = 1−q (register shape 6u⁵−5u⁶, P = −56/3).  The registered kernel K_l(q;H) = C_l shape(q)/(πH²) with C_l = float(KERNELS["l"+family].c2_pi) (10 for lw2, 28/3 for lw4, the `_finish` normalisation 1/(2∫ r·shape): ∫₀¹ q(1−q)³ dq = 1/20), so L = f K_l with **f = P c/(C_l H²)** (−14/H² for w2, −18/H² for w4; `lap_factor(1, ·)` = −14.0 / −18.0 — corrected in REVIEW-005: the first version of this page said C_l = 5 / 7/4 and f = −28/H² / −96/H², which the code and the tests (a)–(d) contradict).
5. **ν_eff = fac/8 = α c0 H/(8 ξ) — derived, not calibrated.** Expanding a smooth v to second order about x_i in the solver's fluid-fluid pairwise term, the only surviving moment is ∫_disk r W′ dA = 2π∫₀^H r² W′ dr = 2π([r²W]₀^H − 2∫₀^H r W dr) = **−2**, exactly, for any normalised radial kernel in 2D (∫₀^H r W dr = 1/(2π) IS the normalisation).  With the isotropic fourth moment, the pairwise bulk term = (fac/8)(∇²v + 2∇∇·v) in 2D = fac/(2(d+2)) — the ν of the solver's own dtv time-step rule.  The wall term uses the same ν_eff, so wall and bulk share the Laplacian coefficient.  **Not one operator** (reviewer, REVIEW-005): the bulk pairwise term also carries `2∇(∇·v)`, the wall term only `∇²` — measured on the solver's own bulk term in the interior of the dp = 0.04 tank, `a_x/fac` = 0.239 (expected 0.250) for `v = (y², 0)` and 0.757 (expected 0.750) for `v = (x², 0)` (lattice error 1–4 %; `docs/work/refs/review5_nu_probe.py`).  The test pins the moment at −2.000000000000 (atol 1e-10, both families) and the pairwise bulk term for v = (y², 0) at fac/4 = (fac/8)|∇²v| (rtol 1e-10).

## 4. Special cases and limits

| case | Δλ | verified |
|---|---|---|
| z → 0 (on the wall) | → 0 linearly (the mirror field vanishes; the integral loses half its disk) | (c) on-edge case ≤ 1e-9 absolute for the degenerate geometry; the linear approach is the z behaviour of B(z) in (a) |
| support fully inside the solid | 0 | (c) ≤ 1e-9 absolute (indicator cancels: 2·1 − tr(I) = 0) |
| farther than H from the solid | 0 | (c) ≤ 1e-9 absolute |
| tangent (support touches the solid at a point) | 0 | (c) ≤ 1e-9 absolute |
| scaling | Δλ(z;H) = Δλ(z/H;1)/H² | (d) exactly (0.0e+00 measured) — the H⁻² is carried by f and the registered kernel |

## 5. Checks (T5.1, `tests/scene/test_viscosity_scene.py`)

| # | check | tolerance (stated before results) | measured | status |
|---|---|---|---|---|
| a | flat wall, both families, H ∈ {1, 0.7} vs own `scipy quad` of B(z) = ∫_z^H (W″ + W′/r) 2Θ(r) r dr | ≤ 1e-10 · max B | 3.784e-14 | pass (reviewer: ≤ 4e-14) |
| a | smoke B(z) of the context table | rtol 1e-6 | pass | pass |
| b | L body (200 pts) and cavity (40 pts), both families vs own 500×1000 polar midpoint + even-odd PIP | ≤ 1e-3 · max\|brute\|, max\|brute\| > 10, ≥ 50 pts > 1e-6 | worst 2.24e-4 | pass (reviewer: 2.2e-4) |
| c | degenerate exact zeros (tangent, fully inside, far, on edge, vertex) | ≤ 1e-9 absolute | all ≤ 1e-9 | pass |
| c | outside point (1.2, 0.5) H 0.3 vs brute | ≤ 1e-3 · value | 7.754195 (w2) / 3.688068 (w4), rel 2.1e-6 / 4.7e-5 | pass (reviewer: identical) |
| d | scaling, moved body, two-body additivity | rtol 1e-9 / ≤ 1e-11 · max | 0.0e+00 / 3.6e-15 | pass |
| e | guards (family w9, ImplicitRep, VolumeRep) | raise NotImplementedError | raised | pass |
| f | negative controls: 2fλ alone, −Δλ, pairwise A(0.1) vs B(0.1) | > 1e-1 / > 1.0 / > 0.3 · B | 5.757 / 2.000 / 2.557 (w2); 5.191 / 2.000 / 2.397 (w4) | pass |
| g | moment identity ∫ r W′ dA = −2; pairwise bulk term for v = (y²,0) = fac/4 | atol/rtol 1e-10 | −2.000000000000; 0.250000000000 fac | pass |

T5.2 (`tests/sim/test_deltasph_viscosity.py`, tank `dp = 0.04`, surface domain, both kernels):

| # | check | tolerance (stated before results) | measured | status |
|---|---|---|---|---|
| a | `DeltaSPHConfig().viscosityExact is False` (default off) | — | True | pass |
| b | `d_ex = rhs(viscosityExact=True) − rhs(wallViscosity=False)` vs `−2 ν_eff wallMass u_n/ρ Δλ_brute n` (own 600×1200 brute force over the tank exterior; n, u_n from the solver's own G, not under test) | ≤ 5e-4 · max\|pred\|, max\|pred\| > 0.1, ≥ 500 non-zero particles | w2: near = 693, max\|pred\| = 0.2893, 1.013e-4; w4: near = 690, max\|pred\| = 0.2925, 1.271e-4 | pass (reviewer: 1.01e-4 / 1.27e-4) |
| c | negative controls: vs the pairwise term (different operator); sign-flipped prediction; ν_eff → fac/12 prediction | > 0.5 · max\|d_pair\| / > 1.0 · max\|pred\| / > 0.2 · max\|pred\| | w2: 0.899 / 2.000 / 0.333; w4: 0.886 / 2.000 / 0.333 | pass (reviewer: 0.899 / 0.886) |
| d | five `sim.step()` with the flag on stay finite (x, v, ρ) | finite | finite, both tanks | pass |

## 6. Known failure modes

* **The ∇²W-direct trap** (§3.3): registering the Laplacian as a kernel gives a spurious c/H² inside a body (28.0 measured) and `kernels._finish` divides by zero.  The L route avoids it; the (f) control `2fλ alone > 1e-1` pins that the cancellation is load-bearing.
* **Covariance needs the first moments** → SurfaceRep only; other reps raise.
* **Different operators** (not a bug): the pairwise and the Laplacian wall terms disagree near the wall.  Flat wall (solid y < 0), particle at (0, z), H = 1: the pairwise coefficient is A(z) = ∫_solid (W′/r)(ŷ·n)² dA = ∫_z^H W′(r)(Θ + sinΘ cosΘ) dr, Θ = arccos(z/r) (negative), and the Laplacian one is B(z) = Δλ(z); acc_pair = fac·2u_n·A·n, acc_lap = −(fac/8)·2u_n·B·n, ratio acc_pair/acc_lap = 8|A|/B:

  | z/H | 0.02 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
  |---|---|---|---|---|---|---|
  | C2 B(z) | 0.4417774 | 1.959941 | 3.190942 | 1.943131 | 0.5164654 | 0.01570062 |
  | C2 8\|A\|/B | 62.9 | 12.46 | 3.606 | 1.642 | 0.7324 | 0.1953 |
  | C4 B(z) | 0.6086062 | 2.765873 | 4.083071 | 1.708461 | 0.2060951 | 8.409243e-4 |
  | C4 8\|A\|/B | 58.7 | 11.18 | 2.939 | 1.252 | 0.5326 | 0.1371 |

  (B(z) for H = 1; for other H: B(z;H) = B(z/H;1)/H².  The B(z) row is reproduced by the T5.1 (a) smoke check at rtol 1e-6; A(0.1) = −3.0525 (C2) / −3.8645 (C4) is measured by the T5.1 (f) control.)  B → 0 linearly as z → 0, A does not: the Laplacian form damps the wall-normal velocity of the first particle rows 3–12× **less** than the pairwise form, more weakly the closer to the wall.  That is the continuum-consistent behaviour (the mirror-extended v_n is odd, ∂²v_n is smooth at the wall), and it is **the intended change of discretisation** (user decision, HANDOFF Part A Q1); the gate numbers are in `docs/deltasph-validation.md` (WORK-005 section).

## 7. Open questions

* The first-moment variant (position-dependent mirror field, ∫ (v_ghost − v) ∇²W dA′) is not implemented; it would need a scene operation for moments of ∇²W (or of L with the mirror field as payload).
* Whether `viscosityExact` stays off by default: the gates (WORK-005 T5.3) pass with it on, but it is a different operator near the wall (section 6), so the default remains the pairwise form — a user decision, not a regression verdict.

## 8. Related work: Chiron et al. 2019 (and Ferrand 2013 / Leroy 2014)

Read in REVIEW-005 (`warpSPH/literature/chiron2019_sph-3d-complex-wall-boundaries.pdf`, §6.1, Eq. 84–92, 108, §8.2; Eq. numbers as printed).  The paper's wall Laplacian is a renormalised Morris operator,

```
L(u)_i = (2/γ_i) [ Σ_j ω_j (u_i − u_j) (x_i−x_j)·∇_iW_ij / |x_i−x_j|²  +  Σ_s Σ_q A_sq (u_i − u_sq)/d_n · W_isq ]        (Eq. 92 / 108)
```

* **Fluid part** — the Morris factor `x·∇W/|x|²` is exactly `W′/r`, the kernel `L = W′/r` registered here as `lw2`/`lw4`; the pairwise ingredient is the same.
* **Wall part** — a surface term `(u_i − u_s)/d_n ∫_cutface W dS` from a first-order Taylor argument (Eq. 89–90: the tangential gradient is neglected, `∇f·n ≈ (f(x) − f(y))/((x−y)·n)`), **no-slip** with the wall velocity `u_s`, growing like `1/d_n` at the wall.  The present term is the **free-slip mirror** of a volume integral over the solid (`Δλ`), which vanishes at the wall: different boundary condition, different operator (the open question below).
* **Their quadrature** — `∫W dS` over the cutface is a sum over `N_q` points per support radius (Algorithm 1–2; 3D: the support disc replaced by a square, clipped to the triangle).  In 2D `∫_chord W ds` is what the edge reduction gives in closed form (it is the per-edge channel of `∇λ = −Σ n ∫W ds`), so the quadrature points are not needed; the `1/d_n` is a per-segment constant for a flat segment.  Reported accuracy: Poiseuille Re = 10, convergence order ≈ 1.9 (ghost particles ≈ 2, Ferrand's variant lower).  `γ_i` itself is analytic (Eq. 80–83).
* **Open question (user)** — HANDOFF row Q1 wrote `(v_b − v_i)Δλ`.  If a no-slip / tangential wall viscosity is wanted, Chiron's `(u_i − u_s)/d_n ∫W dS` is the concrete form; it would be a *second* operation (per-edge `∫W ds`, scaled by `1/d_n`), not a change to `lap_lambda_scene`.  Not implemented.
