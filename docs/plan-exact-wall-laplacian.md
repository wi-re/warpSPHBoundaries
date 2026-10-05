# Plan: exact wall Laplacian from a reconstructed ghost field (free-slip mirror, no-slip continuation)

Status 2026-10-05. **[V]** = checked numerically by the script named; **[D]** = argued, not run; **[P]** = plan. Nothing here is in the solver yet; the current wall terms (`wallViscosityForm` = `laplacian` / `pairwise` / `noslip`, `docs/derivations/laplacian-wall.md`, `noslip-wall.md`) are unchanged. Context: `paper/` §9 calls the free-slip and no-slip wall terms *models* (tag A); this note is the route to making the wall part exact for a chosen order of the velocity field and to measuring what that order costs in noise.

## 1. Statement of the idea

The viscous wall term is the solid part of the Laplacian estimator of an **extended** field,

    nu ∇²v(x_i) ≈ nu [ Σ_fluid V_j (v_j − v_i) ∇²W_ij  +  ∫_S (v_ext(x') − v_i) ∇²W(|x_i − x'|) dx' ].

Everything is exact except two things: the particle sum, and the choice of `v_ext`. If `v_ext` is a **polynomial** `p_i` in the solid (a local reconstruction at particle i), the solid integral is a linear combination of geometric moments `M_α = ∫_S y^α ∇²W dy`, which are closed form (no quadrature, no ghost particles):

    ∫_S y^α ∇²W = Σ_e ∫_chord [ y^α W′ z_e / r − W n_e·∇(y^α) ] ds + ∫_S W ∇²(y^α)        (Green's second identity)

The edge integrals are of the kinds in `paper/` §3 (`I_m`, `∫ s^j r^m`); `∫_S W ∇²(y^α)` is a lower-order value/moment integral (`thm:value`, `thm:recursion`). The moments depend on geometry only, so with a fixed order p the wall term is a **fixed linear functional of the reconstruction coefficients**; the per-particle field enters only through the fit. This fits the "evaluation, not precompute" decision of `docs/plan-wall-evaluation.md` §1.

The model is the same at every level; the levels differ in the order p of the field reconstruction, i.e. where the Taylor series of the ghost field is cut:

| level | field model in the solid | status |
|---|---|---|
| L0 (today, free-slip) | one constant `R v_i` (so `v_ghost − v_i = −2 u_n n`) | in solver |
| L1 | Shepard / SPH interpolation at mirror points, evaluated by quadrature over the solid (classic ghost-particle scheme) | [P] needs quadrature, not closed form |
| L2 | per-particle local polynomial (MLS / constrained MLS), order p, closed-form moments | [V] the identity; [P] the fit |

## 2. What was verified (scripts in `scripts/derivation_checks/`)

**[V] `paper_exact_wall_mirror.py` — free-slip flat wall, mirror ghost.** Ghost field `v_g(x+y) = R v(x + y_m)`, `y_m = R y + 2 d n` (mirror point of `x+y`, relative to the particle), Taylor-expanded **in `y_m`** to order p before substituting (truncating in `y` is wrong: `y_m` contains the constant `2dn`). Moments by Green's identity agree with ray quadrature to 1e-12 for all |α| ≤ 3 (w4, cubic). Test field: cubic polynomial with the free-slip symmetry (v_x even, v_y odd about the wall), particle at d = 0.3, true ∇²v = (2.4, 0.66):

| ghost order p | error, w4 | error, cubic |
|---|---:|---:|
| 0 (the solver's constant mirror) | 0.19 | 0.31 |
| 1 | 0.29 | 0.31 |
| 2 | 0.020 | 0.011 |
| 3 | 1.6e-12 | 3e-14 |

Reading: (i) the current model is ~10 % off on this field; (ii) **adding only the gradient (p = 1) does not help** — the neglected term is `½ H y_m²` with `|y_m|` up to ≈ h, which contributes O(∇²v) itself; second-order information is what pays; (iii) a cubic field is reproduced exactly at p = 3 (kernel smoothing is exact for degree ≤ 3). Hence the cutoff order is **not a smooth ladder**: p = 0 and 1 are about equally good, p = 2 is the first real step.

**[V] `paper_noslip_extensions.py` — no-slip flat wall, 1D reduction (fields depending on wall distance s only, kernel w4, noise-free).** Compared with the target `v″(d)`:

* *fluid only*, *constant ghost* (`2 v_w − v_i`, `noslip-wall.md` §3.5), *odd mirror* (`v_ext(s') = 2 v_w − v(−s')` with the exact field), *Chiron-type* (renormalised Green form with exact γ and slope model `v′(0) ≈ (v_i − v_w)/d`), *ANALYTIC* (field continued analytically: the smoothing bias no extension can beat), and *BC polynomial(m)*: weighted LS fit on the fluid with basis `{s, …, s^m}` (v(0) = v_w built in), **continued** into s' < 0.
* Poiseuille `s(3−s)`, errors at d = 0.1 / 0.3 / 0.6:
  const ghost 4.5 / 0.81 / 0.88, odd mirror 1.3 / 0.40 / 0.013, Chiron-type 0.23 / 0.25 / 0.032, **BC polynomial m = 2: 6e-10 / 6e-13 / 1e-9**.
  The odd mirror fails because the odd extension of a no-slip field has a jump in `v″` (Poiseuille: +2 → −2); Chiron-type fails through the slope model (true wall slope 3, model 2.7 at d = 0.3). Both are exact only for the profile they assume.
* Non-polynomial fields (`1 − exp(−s/0.5)`, `sin(1.5 s)`, `sin(3 s)`): the BC polynomial is **comparable to the odd mirror, not clearly better**; errors are dominated by the smoothing bias (ANALYTIC column) and by the fit order; the error is not monotone in m (e.g. `sin(1.5 s)`, d = 0.3: m = 2 → −0.19, m = 3 → +0.025; boundary layer d = 0.1: m = 3 → 0.13 vs odd mirror 2.5). `sin(3 s)` (kh = 3) fails for everything: under-resolved.

Take-away for no-slip: **replace the odd mirror by the analytic continuation of a boundary-condition-constrained polynomial.** It is exact where the validation flows live (Couette, Poiseuille: polynomial of degree ≤ 2 in s) and removes the kink that makes every mirror-type no-slip extension biased near the wall; it also gives the wall shear `∂_s p(0)` for free. For general smooth fields it is no worse than the alternatives, not better.

## 3. Observations that motivate the experiments (user + review, 2026-10-05)

1. **Trade-off: reconstruction order vs noise.** A third-order reconstruction of the fluid field is noisy under particle disorder and needs a large support (or a higher-order SPH scheme); even a naive Hessian is noisy. The wall term inherits that noise in proportion to the order p used. The wall error (bias) falls with p; the noise grows with p.
2. **Same model, different cutoff.** L0 (constant, "boundary velocity = fluid velocity per interaction", error between particles) → L1 (SPH/Shepard interpolation, mirrored) → L2 (MLS or similar extrapolation using the gradient or higher orders) is one model with the field order cut at different points; moments of the geometry are common to all L2 variants.
3. **Corrections found while checking** (§2): p = 1 is not an improvement over p = 0 for a second-order operator; the *symmetry itself* removes unknowns, so the constrained fit is cheaper than the generic one:
   * free-slip, flat wall (normal y), quadratic fields: `v_t ∈ span{1, x, x², y²}` (4 unknowns, not 6), `v_n ∈ span{y, xy}` (2, not 6): 6 instead of 12;
   * no-slip, `v − v_w ∈ s·q(s,t)`: degree ≤ 2 → `{s, st, s²}` per component (3, not 6), and the wall value is not an unknown.
4. **Mirror is exact only where the equations are symmetric.** Free-slip on a flat wall: the mirror extension is the true solution (reflection symmetry), so no modelling error. No-slip: odd extension is *not* a symmetry of the equations (forcing/pressure gradient are not odd), hence the continuation of a constrained fit in §2 instead. Curved walls and corners: the mirror is not exact; per-edge images double-count at corners (not covered).
5. **Operator mismatch to resolve.** The bulk term is the pairwise form `(fac/8)(∇²v + 2∇∇·v)` (`paper/` Prop. nueff); the wall term here is a `∇²W` moment form (only the ∇² part). A fully consistent wall term would use the same operator: moments of the pairwise kernel `W′/r³` with `y⊗y` weights. The profile is singular at 0 (`~ r⁻²`, integrable against `y⊗y`) and the compact-potential recursion hits a logarithm there, so this needs its own derivation. Until then, wall and bulk share only the ∇² coefficient `ν_eff = fac/8`.
6. **Chiron-type closures** (one-sided slope + 1/γ renormalisation) are first-order consistent at best **[D]**: the renormalised one-sided average of `v″` is biased by `O(h v‴)` even with the exact slope; in the §2 test the error seen is the slope-model error (v‴ = 0 for Poiseuille).

## 4. Experiments to run when time permits (each: protocol → metric → stop rule)

**E1 — noise vs bias, free-slip, disordered particles [P].** Flat wall, jittered lattice (jitter 0, 5, 10, 20 % dx, 20 realisations each), h/dx ∈ {2, 3, 4}. Fields: the §2 cubic with free-slip symmetry, plus a sin/cos field with the same symmetry. Estimators: L0 (current), mirror with fitted p = 0, 1, 2, unconstrained MLS vs **symmetry-constrained** MLS (basis counts in §3.3). Fit by weighted LS on the fluid neighbours (weights `W_ij`); moments from `paper_exact_wall_mirror.py` machinery. Metric: bias and standard deviation of `∇²v` at the near-wall rows (z/dx = 0.5 … 3.5) against the exact value. *Stop:* if constrained p = 2 has std ≤ 2× L0 std and bias ≤ ½ L0 bias, promote to implementation; otherwise keep L0 and record the numbers.
 Where to put it: `scripts/studies/`; reuse `docs/work/refs/review5_noslip_lattice_probe.py` for the lattice and probe rows.

**E2 — no-slip on the validation flows [P].** Couette and Poiseuille on the lattice of `docs/derivations/noslip-wall.md` §3.5 (rows z/dp = 0.5 … 3.5; ghost-lattice truth values in that table), with: today's flux term, constant ghost, odd mirror, BC-constrained polynomial continuation m = 2 (basis `{s, st, s²}`; moments in the wall frame, rotated). Metric: row errors (table there: Couette flux row 0 = 0.0132 vs truth 0; Poiseuille row 0 −0.016 vs −0.150). *Stop:* continuation row-0 error below the flux term's on both flows, else keep flux.

**E3 — no-slip, non-polynomial fields, noise [P].** `paper_noslip_extensions.py` is noise-free. Repeat with jittered particle data (as E1) for `1 − exp(−s/δ)`, δ ∈ {0.5, 1}·h and `sin`. Question: does the fit order m = 2 or 3 minimise bias + noise? Add the wall shear estimate `∂_s p(0)` as a diagnostic against the exact.

**E4 — operator consistency [P].** Derive the closed-form moments of the pairwise kernel (`W′/r³`, `y⊗y`) over the solid (angular-harmonic split `ŷ⊗ŷ = ½ I + traceless`; expect logarithms at `z → 0`). Verify against polar quadrature (`pairwise` form of the solver), then compare E1 with the pairwise-consistent wall term.

**E5 — geometry [P].** Corners (two edges) and a curved wall (polygon approximation, then the curvature expansion of `paper/` §6 for the mirror). Check where per-edge mirror images fail.

**E6 — cost [P].** Per-particle: number of moments (p = 2 in 2D: 6 for ∇²W-moments up to degree 2 in the wall frame), edge-integral count per wall edge, FP32 behaviour with the stable (Chebyshev) plans (`docs/exactness-and-approximations.md`). Fused-evaluation kernel, no per-pair storage (`plan-wall-evaluation.md` §1).

**E7 — 3D [P, out of scope for now].** The moments of `∇²W` over a polyhedron: Green's identity applies unchanged but needs the 3D face/edge evaluation (`paper/` §8 is a plan).

## 5. Open questions

* Does the noise of a constrained p = 2 reconstruction at the first particle rows stay within the noise already present in the bulk Laplacian? (E1.)
* Is the no-slip continuation worth it over the flux term when particles sit at `d ≈ 0.25 dx` (the `d_min` floor)? (E2.)
* Which weights for the fit: the kernel `W_ij`, a wider window, or Shepard-normalised? (E1/E3.)
* How should the free-slip wall term combine with the bulk pairwise operator near the wall: replace the pairwise wall form entirely, or only the `∇²` part? (E4.)
