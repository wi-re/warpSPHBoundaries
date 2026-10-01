# 3D: tetrahedra via face → edge reduction

**Status:** [P] (plan from `HANDOFF.md §9`; nothing derived yet)
**Tier(s):** 1, 2 · **Dimension:** 3D
**Depends on:** `edge-value-identity.md`, `moments-recursion.md`, `edge-primitives.md`
**Implemented in:** not yet
**Verified by:** not yet (cross-checks listed below)

## 1. Statement (target)

$$\int_{\rm tet} f=\mathbb 1[x\in{\rm tet}]\,4\pi M_3(R)+\sum_f z_f\int_{\rm face}\frac{M_3(r)-M_3(R)}{r^3}\,dA,$$

`M_3(r) = ∫_0^r t² f dt`. The face integrand is compact on the face disk
`ρ <= sqrt(R² - z_f²)`; the face integral is a 2D value identity in the face plane
with radial profile `g(ρ) = z_f (M_3(r) - M_3(R))/r³`, `r = sqrt(ρ² + z_f²)`:
indicator (foot of `x` in the face) + edge terms with
`G(ρ) = ∫_0^ρ t g dt = ∫_{|z|}^{r} r' g dr'` (elementary).

New edge primitive family: `∫ r^m/(s² + w_e²) ds` with `r² = s² + w_e² + z_f²`,
elementary (solid-angle-type `arctan(s z/(w r))`).

## 2. Assumptions and validity

Same as the 2D identities; elements non-overlapping; faces consistently oriented.

## 3. Derivation (plan)

Dimension-independent moment recursion; then in-plane recursion; the in-plane
potential `-∫_ρ^{ρ_R} t φ dt = -∫_r^R r' φ dr'` has the same form.

## 4. Special cases and limits

`z_f → 0` face-level jump vs indicator; tet edges/vertices inside the support.
Half-space limit must reproduce the 3D planar closed forms in `docs/derivation.md`.

## 5. Checks

| check | how | status |
|---|---|---|
| half-space limit vs 3D planar polynomials | mpmath | [P] |
| triangle solid angle vs Van Oosterom–Strackee | exact | [P] |
| polyhedral Newtonian potentials (Waldvogel 1979; Werner & Scheeres 1997) | cross-check | [P] |
| ball approximated by tets vs exact sphere closed form | convergence | [P] |

## 6. Known failure modes / 7. Open questions

Is face → edge cheaper than direct tet ∩ ball decomposition in practice
(`HANDOFF.md §13`)?
