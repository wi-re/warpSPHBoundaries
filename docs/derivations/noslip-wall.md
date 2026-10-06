# The no-slip wall viscosity: Chiron-style flux term (WORK-007, Q1)

**Status:** [V] (the flat-wall absolute, distance floor, Galilean, Couette/Poiseuille, off-switch and five-step checks below; the gate / sloshing / stability numbers in `../deltasph-validation.md`, WORK-007 section)
**Tier(s):** n/a (solver wall term, not an obstacle tier) · **Dimension:** 2D
**Depends on:** the kernel table (`curvbound.kernels`), the scene operations of `src/warpSPHBoundaries/scene/scene.py` (`signed_distance`, `Body.velocityAt`, the wall gradient `G` of `_wall_data`); `laplacian-wall.md` §3.5 (ν_eff) and §8 (Chiron et al. 2019 Eq. 91–92)
**Implemented in:** `src/warpSPHBoundaries/sim/deltasph2d.py` (the `"noslip"` branch of the wall-viscosity block of `rhs`)
**Verified by:** `tests/sim/test_deltasph_noslip.py` (T7.1)

The third wall-viscosity form, `cfg.wallViscosityForm = "noslip"` (the default stays `"laplacian"`): the existing `"laplacian"` (exact wall Laplacian, free-slip) and `"pairwise"` (warpSPH free-slip mirror, polar quadrature) both damp only the wall-**normal** component of the relative velocity (free-slip). This form damps the **all-components** relative velocity (no-slip), using a one-sided finite difference of the normal derivative in the style of Chiron et al. 2019, Eq. 91–92. It needs no new Warp or scene code: the wall gradient `G` and the signed distance are already computed in every step.

## 1. Statement

For a particle i and a solid body b, with the wall gradient `G_b` of `_wall_data` (`G = wallMass ∇λ`, `|G_b| = gm`), the signed distance `d_b` of the particle to the wall (positive in the fluid) and the body velocity `v_w = b.velocityAt(x_i)`:

```
acc_wall = -2 nu_eff (v_i - v_w) |G_b| / (rho_i d_b)      (per body b),      nu_eff = alpha c0 H / (8 xi) = fac/8,      d_b = max(d_b,signed, d_min),  d_min = 0.25 dx
```

(Units of the solver, `rho0 = 1`, `wallMass` = wall mass per area in units of `rho0`: `[ν_eff] = L²/T`, `[|G|] = 1/L`, `[d] = L`, so `[acc] = L/T²`; per body.) The term opposes `v_i − v_w` in **all** components (no-slip), for a particle on the fluid side, masked by `hit` (a no-op for SurfaceRep bodies, kept as the volume-representation guard). Sign: it pulls `v_i` toward `v_w`; for a resting wall it is a damping of the absolute velocity. It is zero when the relative velocity is zero (Galilean invariance, check (b)).

## 2. Assumptions and validity

* **Flat-wall model per particle.** One normal per particle, `n = G_b/|G_b|`; `|G_b|` the summed wall gradient; `d_b` the distance to the nearest wall point. For a flat wall `|G_b| = wallMass ∫_chord W ds` **exactly** (the per-edge channel of `∇λ`), verified to 8.6e-16 relative at `z = dp/2` in `small_tank` (`review5_noslip_smoke`). Near convex corners and on curved walls this is not Chiron's per-element form (per-element `∫W ds / d_n` would need a new per-pair channel in `warpbc` — phase 2, §7).
* **No `1/γ` renormalisation.** Chiron renormalises the whole Morris operator by `1/γ_i`; the solver's bulk term is not renormalised, so this form is not (probe: Couette row 0, pairwise bulk + flux, 0.0132 with vs 0.0195 without `1/γ` — the renormalisation would over-damp row 0).
* **Kernel-agnostic.** `|G_b|` comes from the scene (`∇λ`), not from a form-specific quadrature, so the same formula holds for Wendland C2 and C4 (check (d)).
* SurfaceRep bodies (the scene `signed_distance` needs a surface loop); the `d_min` floor keeps `1/d` finite (the no-penetration law keeps particles at `d ≳ 0.25 dx`).

## 3. Derivation

1. **Chiron et al. 2019, Eq. 91–92, specialised.** The paper's wall Laplacian is a renormalised Morris operator whose wall part is a surface term `(u_i − u_s)/d_n ∫_cutface W dS` (Eq. 92/108), from a first-order Taylor argument: the normal derivative is a one-sided finite difference, `∇f·n ≈ (f(x) − f(y))/((x−y)·n)` (Eq. 89–90, the tangential gradient neglected), `d_n` the distance of the particle to the wall, `u_s` the wall velocity. It is **no-slip** and grows like `1/d_n` at the wall.
2. **The `∫W dS` in this solver.** In 2D `∫_chord W ds` over the cutface is exactly the per-edge channel of `∇λ = −Σ n ∫W ds`, which `_wall_data` already returns as the wall gradient `G_b` (no quadrature points needed, §8 of `laplacian-wall.md`): `|G_b| = wallMass ∫_chord W ds`. So the Chiron wall part becomes `(v_i − v_w) |G_b| / d_b`, with the `1/d_n` a per-particle constant.
3. **The coefficient `−2 ν_eff / ρ`.** Chiron's `∫W dS / d_n` is a first derivative of the velocity, i.e. a viscosity acting like a one-sided `∇²`. The solver's viscous (Laplacian) coefficient is `ν_eff = fac/8 = α c0 H/(8ξ)` — **derived, not calibrated** (as in `laplacian-wall.md` §3.5: expanding a smooth v to second order in the solver's fluid–fluid pairwise term, the surviving moment is `∫_disk r W′ dA = −2`, exactly, for any normalised radial kernel in 2D, so the pairwise bulk term = `(fac/8)(∇²v + 2∇∇·v)` in 2D — the ν of the solver's own dtv time-step rule). The wall term uses the same `ν_eff` so wall and bulk share the Laplacian coefficient; the factor `−2` is the standard `2ν ∇² → −2ν (v−v_w)/d` one-sided form (the same `−2 ν_eff … /ρ` prefactor the `"laplacian"` and `"pairwise"` forms carry on their wall term). Sign: it opposes `v_i − v_w`.
4. **The `d_min` floor.** `d_b = max(d_signed, 0.25 dx)`: the `1/d` in the term is singular at the wall; the floor at `0.25 dx` matches the closest the no-penetration law lets particles get. For the measured rows (`d ≥ dp/2 > 0.25 dx`) `d_b = d`; check (b) pins the floor (a particle at `z = 0.1 dx` uses `d_b = 0.25 dx` with the **true** `F(z)`).
5. **Why not the obvious "constant ghost" no-slip.** The naive no-slip is a constant ghost field `2 v_w − v_i` over the whole solid (`acc = 2 ν_eff wallMass (v_w − v_i) Δλ/ρ`, or the pairwise analogue `2 fac wallMass M₂ (v_i − v_w)/ρ`). Measured on a lattice above a wall at rest (`review5_noslip_lattice_probe.py`; dp = 1, H = 4, C2, pairwise bulk with ν_eff = 1/8; exact `ν∇²u`: Couette `u = a y` → 0, Poiseuille `u = y(Y−y)`, Y = 12 → −0.25), the total viscous acceleration of the rows z/dp = 0.5, 1.5, 2.5, 3.5 is — Couette:

  | row (z/dp) | 0.5 | 1.5 | 2.5 | 3.5 |
  |---|---|---|---|---|
  | bulk only | 0.0962 | 0.0224 | 0.0013 | 0 |
  | + constant-ghost pairwise | −0.0692 | −0.0743 | −0.0100 | 0 |
  | + constant-ghost Laplacian | +0.0781 | −0.0451 | −0.0364 | −0.0018 |
  | ghost-lattice truth (odd extension) | 0 | 0 | 0 | 0 |

  The constant-ghost forms have errors as large as the uncorrected bulk deficit (the ghost field ignores the velocity gradient: the true extension is the odd *position-dependent* mirror, whose ghost-lattice truth is exactly 0). They are not accurate no-slip operators and are not implemented. The flux term (row below, reviewer lattice probe) is the one that recovers 0 in Couette:

  | Couette, pairwise bulk + flux | 0.0132 | −0.0137 | −0.0045 | −0.0001 |
  | Poiseuille, pairwise bulk + flux | −0.0161 | −0.3707 | −0.2823 | −0.2403 |
  | Poiseuille, bulk only (for contrast) | 0.9384 | 0.0081 | −0.2271 | −0.2398 |
  | Poiseuille, ghost-lattice truth | −0.150 | −0.233 | −0.240 | −0.240 |

## 4. Special cases and limits

* **Resting wall, v_i = 0** → 0 (no relative velocity).
* **Co-moving (v_i = v_w)** → 0 (Galilean; check (b): `|term| < 1e-10`).
* **`z → d_min`** (particle inside the floor) → the term uses `d_b = 0.25 dx` with the true `F(z)` (check (b-floor): the `1/d` is capped, `F` is not).
* **Tangential field, resting wall** → the free-slip forms give 0 (they damp only the normal component) but this form gives a non-zero tangential damping (the point of the negative controls, check (c)).
* **Farther than H from the wall** → `|G_b| → 0` (the kernel support), so the term vanishes.

## 5. Checks (T7.1, `tests/sim/test_deltasph_noslip.py`)

Tank `dp = 0.04` (C2) unless stated; every reference is the test's own code.

| # | check | tolerance (stated before results) | measured | status |
|---|---|---|---|---|
| a | flat wall absolute: a particle at `z = dp/2`, `v = (1,0) / (0,−1) / (0.6,−0.8)`, `a_on − a_off` (form noslip) vs `−2 ν_eff wallMass v F/(ρ z)`, `F = ∫_{−L}^{L} W(√(s²+z²)) ds` (own `scipy quad`, epsabs = epsrel = 1e-13, Wendland C2 written out) | ≤ 1e-9 · \|pred\| (float64 round-off; the reviewer measured \|G\| vs wallMass F at 8.6e-16; a wrong factor or sign is ≥ 1e-2) | v=(1,0) (−2.018640, 0.0) 2.20e-16; v=(0,−1) (0.0, 2.018640) 4.40e-16; v=(0.6,−0.8) (−1.211184, 1.614912) 4.12e-16 | pass |
| a | smoke of the OWN prediction at z/H = 0.125 (wallMass = 1): ν_eff, F, pred_x (v=(1,0)), pred_y (v=(0,−1)) | rtol 1e-4 | ν_eff = 2.432164e-3, F = 8.299771, pred_x = −2.018640, pred_y = +2.018640 | pass |
| b | distance floor: a particle at `z = 0.1 dx` uses `d_b = 0.25 dx` with the TRUE `F(z)` | ≤ 1e-9 · \|pred\| | (−4.495036, 0.0) 1.19e-15 | pass |
| b | Galilean: wall `linearVelocity = v0`, particle `v0` → no term; at rest → non-zero | co-moving < 1e-10, resting > 0.1 | co-moving 0.00e+00, resting 3.423317 | pass |
| c | Couette `u = s` (target 0), probe configuration (`hydrostatic_tank(dp=0.04)`, `a_flux = rhs("noslip") − rhs(viscosity=False)`, first component, mean over the 15 columns `\|x\| < 0.3` in rows `s = (k+0.5) dp`, k = 0..3, ν_eff = 0.00314): `\|a_flux(row k)\| ≤ 0.2 · \|a_bulk(row 0)\|` | ≤ 0.2 · \|a_bulk(row 0)\| = 0.06014 | a_flux by row +0.00815 / −0.00855 / −0.00280 / +0.00001 (0.136 / 0.142 / 0.047 / 0.000) — matches the reviewer solver probe to the digit | pass |
| c | negative controls: the `"laplacian"` and `"pairwise"` forms on the same tangential field give `\|a(row 0)\| = \|a_bulk(row 0)\|` (free-slip: 0 tangential damping) → must fail the Couette tolerance | > 0.5 · \|a_bulk(row 0)\| = 0.03007 | laplacian 0.06014, pairwise 0.06014 (both > 0.03007) | pass |
| c | Poiseuille `u = s(0.36 − s)` (target `−2 ν_eff = −0.00628`): rows 2,3 `\|a_flux − target\| ≤ 0.15 \|target\|`; rows 0,1 `\|a_flux − target\| ≤ 0.6 \|a_bulk − target\|` | 0.15 / 0.6 as stated | rows 2.5 / 3.5 a_flux −0.00672 / −0.00601 (0.070 / 0.044); rows 0.5 / 1.5 a_flux −0.00143 / −0.00825 (0.215 / 0.411) | pass |
| d | `wallViscosity = False` gives no wall term for `"noslip"` (the form is irrelevant): `\|a(noslip, off) − a(laplacian, off)\|` | < 1e-12 | 2.09e-13 | pass |
| d | five `sim.step()` with `"noslip"` stay finite (x, v, ρ) on the C2 and the C4 tank (the form is kernel-agnostic: \|G_b\| comes from the scene) | finite | finite, both tanks | pass |

These match the reviewer's probes to the printed digits (Couette a_flux 0.00815/−0.00855/−0.00280/0.00001; Poiseuille a_flux −0.00143/−0.00825/−0.00672/−0.00601; a_bulk 0.01625/−0.00147/−0.00577/−0.00600; ν_eff 0.00314; `docs/work/refs/review5_noslip_solver_probe.py`): GPU reduction-order non-determinism is below the 5th printed digit.

## 6. Known failure modes

* **Corners and curved walls.** The term is the flat-wall model per particle (one normal, `|G_b|` summed, `d_b` to the nearest point). Near convex corners and on curved walls it is not Chiron's per-element `∫W ds / d_n` (that would need a new per-pair channel in `warpbc` — phase 2, §7); the per-particle `|G_b|` and `d_b` smooth over the geometry.
* **The `1/d` cap.** `d_min = 0.25 dx` caps the `1/d` singularity; below it the term is `∝ 1/(0.25 dx)`, not `∝ 1/d`. The no-penetration law keeps particles at `d ≳ 0.25 dx`, so the cap is rarely the active value; check (b) pins it.
* **Explicit damping rate.** The term is an explicit damping with rate `k = 2 ν_eff |G_b|/(ρ d)`. T7.2(3) measures it after 300 steps (default config otherwise): dam break max k = 16.5835 s⁻¹, dt = 9.727e-05 s, **k·dt = 0.0016**; sloshing max k = 20.4154 s⁻¹, dt = 1.000e-04 s, **k·dt = 0.0020** — both `k·dt ≪ 1` (the reviewer's estimate k ~ 5 s⁻¹ at d = dx/2 is exceeded only because the nearest particles sit at d ~ 0.25 dx, where 1/d is larger; the stability criterion is k·dt, and it is tiny). No stability issue at the solver's acoustic time step.
* **More dissipative than free-slip (not a bug).** The no-slip form removes tangential momentum at the wall, so it damps the flow more than the free-slip forms. T7.2: the dam break KE vs the free-slip reference is 0.3165 (31.7 %) and the peak velocity is ~43 % lower (3.798 vs 6.598); the sloshing KE envelope differs by 14.6 %. The density gates stay healthy. The default stays `"laplacian"` and nothing was adjusted. **Reviewer correction (REVIEW-007): this is not "the physics" of a no-slip wall; it is the no-slip wall *at the artificial viscosity of the scheme*.** `ν_eff = α c0 H/(8ξ)` is a numerical coefficient: 3.1e-3 m²/s at the tank (dp = 0.04), i.e. ≈ 3000 × the kinematic viscosity of water (1e-6 m²/s; `docs/work/refs/review7_noslip_probe.py` (3)), and its diffusion length `√(ν_eff t)` is ≈ 1 dp after 0.5 s (physical water: 0.7 mm). The wall friction of a no-slip run therefore scales with α, c0 and H (i.e. with the resolution and the sound-speed ratio), not with the fluid; the reduction of the dam-break front speed (6.6 → 3.8) and of the KE is a property of that coefficient and has no physical reference here. A physical-viscosity wall term (`ν` independent of α) is the open item in §7. Also: `nan` for the P1 arrival means the arrival lies beyond the truncated window (T = 0.65 s ≈ 2.63 t*, default arrival 2.474 t*), i.e. delayed by more than 0.16 t*; it does not show that the impact is weaker.

## 7. Open questions

* **Per-element `∫W ds / d_n`** as a new per-pair channel in `warpbc` (the true Chiron form at corners and on curved walls) — phase 2, not implemented.
* **Partial slip** `κ ∈ [0, 1]` as a blend of the free-slip and the no-slip term (the tangential damping scaled by κ) — not implemented.
* **The free-slip / no-slip choice per body** (some walls free-slip, some no-slip) — not implemented (the form is global).
* **Physical viscosity at the wall.** The wall flux uses the scheme's artificial `ν_eff` (above); a no-slip wall for a real fluid wants its own `ν` (and the bulk term of the same `ν`, `cfg.physicalViscosity` of warpSPH is not ported, `deltasph-porting-notes.md §4b`) — not implemented.
* **3D.** The per-edge `∫W ds` becomes a per-face `∫W dA`; the `1/d` and `ν_eff` carry over, but the face→edge chain and the `warpbc` channel are not done.
