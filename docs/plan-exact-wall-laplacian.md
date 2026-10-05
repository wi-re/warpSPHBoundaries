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

**E4 — operator consistency [V, closed form; the experiment comparing it with the ∇²W form is still [P]].** The pairwise wall term for a continuum of solid with ghost field `dv(y) = v_ext(x_i + y) − v_i = Σ_α c_α y^α` is `a_k = Σ_α c_{α,l} T_{α+e_k+e_l}`, with the tensor moments `T_β = ∫_S g(r) y^β / r² dx′`, `g = −W′/r` (a polynomial for the Wendland kernels: `−W′/r` of `w2` is `20c(1−q)³/H⁴`), `|β| = d`. They are closed form: the field `Q = y^β y_m φ_d(r)` has `div Q = y^β[(d+2)φ + rφ′]`, so with the compactly supported choice

    φ_d(r) = − r^−(d+2) ∫_r^H s^(d−1) g(s) ds   (= −C_d r^−(d+2) + Σ_j g_j r^(j−2)/(d+j) for g = Σ g_j r^j,  C_d = ∫_0^H s^(d−1) g ds)

the divergence theorem gives, since `y·n_out = z` is constant along an edge,

    T_β = Σ_edges z ∫_chord y^β φ_d(r) ds  +  w(x_i) C_d ∮ ŷ^β dθ       (w = winding number of the solid around x_i; the second term is the full-disk value).

Only the chord integrals `J(i,m) = ∫ s^i (s²+z²)^(m/2) ds` with `m ≥ −(d+2)` appear: rational / `atan` / `asinh` / `log` (the logarithm of §3.5 is the case `i = 1, m = −2`, the `g_0 r^−2` term against the `s` component of `y^β`; it is finite for `z ≠ 0`, and the edge term carries the factor `z`, so `z → 0` is harmless unless the chord contains the foot point, i.e. the particle on the boundary). No new primitive family: the paper's `S_{j,m}` machinery extended to negative `m` by the reduction `J(i,m) = J(i−2,m+2) − z² J(i−2,m)` and `J(0,m) = [(m+3)J(0,m+2) − s r^(m+2)]/((m+2)z²)`.
**[V] `scripts/derivation_checks/paper_pairwise_wall_moments.py`** (mpmath primitives, each `J` checked to quadrature; ray quadrature reference with exact radial integrals `P_d(r) = ∫_0^r s^(d−1) g`): all tensor moments `d = 2, 3, 4` (constant, gradient and Hessian ghost fields) for kernels `w2`, `w4` on a triangle (x outside / inside), an L-shape (x near, an edge collinear with x), a slab edge at `z = 10⁻³`: `max |closed − ray| ≤ 2·10⁻¹⁴`; the full disk gives `T_xx = πC_2` and `T_xy = 0`; flat wall `z = 0.1, 0.4, 0.8` agrees to the printed 8 digits.
Consequences: (i) the solver's `pairwise` wall form (polar quadrature of the solid, `surfaceSamples`) has an exact counterpart for the constant mirror (`d = 2`: 3 independent components `T_xx, T_xy, T_yy`, i.e. the `M2` of `deltasph2d.py`) and for every higher ghost order, at the cost of the same edge loop as `lap_lambda_scene` with different primitives; (ii) the isotropic/traceless split is not needed (the divergence field handles `ŷ⊗ŷ` directly), the apparent obstruction was the `1/r²` weight, which `φ_d` removes by integrating from the top of the support; (iii) the bulk and wall terms can now share the pairwise operator, so open question 2 below is a modelling choice, not a derivation gap. Still to do for E4: the free-slip hierarchy of §2 for the pairwise operator (ghost Taylor series in `y_m`, as in `paper_exact_wall_mirror.py`) and E1 with both operators.

**E5 — geometry [V for a right-angle corner and a circular wall; general angles [D]].** `scripts/derivation_checks/paper_exact_wall_corner.py` (A: corner, B: circle; ~10 s). Metric: error of the total `∇²v` (A) or of the solid part (B) for the ghost `v_g(x+y) = R v(x + y_m)` Taylor-truncated at order p in the mirror displacement, as in §2; kernel `w4`, H = 1.

*A. Fluid quadrant x, y > 0, free-slip symmetric cubic field* (`v_x = x f(x²,y²)`, `v_y = y g(x²,y²)`: odd/even about both walls). The solid is three regions: behind x = 0 (mirror in x), behind y = 0 (mirror in y), the corner quadrant (point reflection `R_A R_B = −I`). Estimators of the solid part: **images** (the three regions with their own mirror), **flat** (the whole solid mirrored as one flat wall, the plane of the nearest wall), **double** (each wall mirrors its whole half-plane: the corner quadrant counted twice, as a per-edge sum of half-plane images does). Errors of the total `∇²v` (rows: particle position, p):

| particle | p | images | flat | double |
|---|---:|---:|---:|---:|
| (0.10, 0.10) | 2 | 4.9e-2 | 1.1e-1 | 9.2e-1 |
| (0.25, 0.25) | 2 | 3.4e-2 | 2.8e-1 | 6.3e-1 |
| (0.50, 0.50) | 2 | 2.9e-2 | 6.2e-1 | 4.3e-2 |
| (0.15, 0.30) | 2 | 4.7e-2 | 6.0e-2 | 7.8e-1 |
| any with the corner farther than H | 0, 2, 3 | = flat | = flat | = flat |
| every position | 3 | ≤ 1e-13 | ≤ 1e-13 | 4e-3 – 0.93 (1e-13 beyond the corner zone) |

Reading: (i) **p = 3 is exact for `images` and `flat` alike**, because the symmetric field is globally equivariant: any assignment of a mirror to a solid point returns the analytic continuation. What breaks the estimate is not the choice of mirror but the *partition*: every solid point must be counted exactly once (`double` fails by 0.4 – 0.9 inside the corner zone, `|x| < ~0.7`, and by 0 beyond it). (ii) For truncated ghosts (p = 2, what a quadratic reconstruction gives) the mirror matters: the one-wall model **ignoring the other wall errs 0.06 – 0.6** (up to 45 % of the solid part at the corner distance 0.7 H), the three-region images 0.03 – 0.05, the same truncation error as at a flat wall. (iii) **p = 0 (the solver's constant mirror) is insensitive to the double count at 90° only**: on the corner quadrant `(R_A − I)v + (R_B − I)v = (R_A + R_B − 2)v = −2v = (R_A R_B − I)v`, since `R_A + R_B = 0` for orthogonal walls; the table row p = 0 has `images = double` to all digits. For any other angle `R_A + R_B ≠ R_A R_B + I` and the per-edge sum is wrong already at p = 0. (iv) Exact images (a finite group generated by the wall reflections) exist only for wedge angles π/n (n = 2: three regions; n = 3: five, …) **[D]**; for a general corner no symmetric extension exists, the ghost is a model there, and the partition rule above (count each solid point once; the closest-point rule gives point reflection through a vertex in the corner sector) is the only guidance.
Consequences for the implementation: the corner needs the solid partitioned by the nearest-feature rule (edge strips + vertex sectors), not a per-edge half-plane sum; with the closed-form edge integrals the partition costs nothing extra (the region boundaries are straight rays from the vertex: extra edges of a polygon clipped to the sector, with zero net contribution when the mirror maps agree on them, which they do when the field has the symmetry). The symmetry-constrained basis shrinks at a corner: both walls give `v_x ∈ x·{1, x², y², …}`, `v_y ∈ y·{1, x², y², …}`, so up to p = 2 only `{x}`, `{y}` (2 unknowns, `∇²` of the ghost vanishes) and up to p = 3 six unknowns (`x, x³, xy²` and `y, y³, x²y`): the corner needs the cubic order to carry curvature, the flat wall (§3.3: 6 unknowns up to p = 2) already carries it at p = 2.

*B. Circular wall* (fluid inside r < Rc, particle at distance d = 0.3 from the wall). The field is symmetric in the wall-attached coordinates (`v_θ = F(θ, ρ²)`, `v_r = −ρ G(θ, ρ²)`, ρ = Rc − r; free-slip with zero shear to O(κ)); its continuation to ρ < 0 is the curvilinear mirror `r → 2Rc − r`, the exact ghost for this field. The flat tangent-line mirror at the closest wall point (the model of §2) against it, error of the solid part:

| Rc / H | κH | p = 0 | p = 2 | p = 3 |
|---:|---:|---:|---:|---:|
| 32 | 0.031 | 0.188 | 7.7e-3 | 4.2e-3 |
| 8 | 0.125 | 0.202 | 3.2e-2 | 1.8e-2 |
| 4 | 0.25 | 0.244 | 8.6e-2 | 4.2e-2 |
| 2 | 0.5 | 0.424 | 0.347 | 9.9e-2 |

The error is **linear in κH** (it doubles as Rc halves, ratio 0.44 – 0.54 per halving at p = 2, 3): ≈ 0.14 κH at p = 3 and ≈ 0.25 κH at p = 2, while p = 0 keeps the flat-wall floor (≈ 0.19, the same constant-mirror error as §2). So the flat-tangent mirror is a first-order-in-curvature model: at Rc = 4H (a typical rounded tank corner, `ξ`-resolved) it is 17 % of the solid part at p = 3, at Rc = 32H below 3 %. The curvature expansion of `paper/` §6 gives the closest-point image of the quadratic region at the next order; using it for the mirror (`y_m` with the κ-correction) is the route to a second-order model **[P]**, and what the polygon walls of the solver already do in the λ, G integrals (exact for the polygon). Open: the same test with the polygon approximation of the circle (the solver's representation) to separate curvature error from facet error.

**E6 — cost [P].** Per-particle: number of moments (p = 2 in 2D: 6 for ∇²W-moments up to degree 2 in the wall frame), edge-integral count per wall edge, FP32 behaviour with the stable (Chebyshev) plans (`docs/exactness-and-approximations.md`). Fused-evaluation kernel, no per-pair storage (`plan-wall-evaluation.md` §1).

**E7 — 3D [P, out of scope for now].** The moments of `∇²W` over a polyhedron: Green's identity applies unchanged but needs the 3D face/edge evaluation (`paper/` §8 is a plan).

## 5. Open questions

* Does the noise of a constrained p = 2 reconstruction at the first particle rows stay within the noise already present in the bulk Laplacian? (E1.)
* Is the no-slip continuation worth it over the flux term when particles sit at `d ≈ 0.25 dx` (the `d_min` floor)? (E2.)
* Which weights for the fit: the kernel `W_ij`, a wider window, or Shepard-normalised? (E1/E3.)
* How should the free-slip wall term combine with the bulk pairwise operator near the wall: replace the pairwise wall form entirely, or only the `∇²` part? (E4: both are now closed form; what is left is to measure which is better, E1.)
