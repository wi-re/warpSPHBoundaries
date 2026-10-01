# Tier 3 in 2D: closest-point curvature expansion of the value integral

**Status:** [P]
**Tier(s):** 3 · **Dimension:** 2D (3D version: `PLAN.md` §8, §19; exact oracles in `docs/derivation.md`)
**Depends on:** `../notation.md`, `docs/derivation.md` (2D planar closed forms)
**Implemented in:** not yet
**Verified by:** not yet

## 1. Statement (target)

For a smooth boundary with curvature `κ` at the closest point and signed distance `d`
(`q = d/h`, `α = hκ`):

$$\lambda(q,\alpha)=F_0(q)+\alpha F_1(q)+\alpha^2F_2(q)+O(\alpha^3),$$

with `F_0 = λ_2` (the planar 2D closed form of `docs/derivation.md` §3). In 2D there
is one principal curvature; the 3D structure in `H, K` (`PLAN.md` §8) is *not*
assumed. Gradient and low-order moments of the same expansion are required as
well (`../backends-and-verification.md` golden set).

## 2. Assumptions and validity

- Reach `>= h` on the solid side, a single boundary sheet in the support, smooth
  curvature over one support (otherwise the tier fails; tier selection is a
  separate open item in `HANDOFF.md §10`).
- Sign: `κ > 0` for convex solids (`notation.md`).

## 3. Derivation (plan)

1. Tubular coordinates in 2D: `x' = c(t) + s n(t)`, area element `J(s) dt ds`
   with `J(s) = 1 - κ s` (for the normal pointing into the fluid and the sign
   convention above; **confirm in Maple**, this is where Tube Maps' 3D `K` sign
   is questioned in `HANDOFF.md §1`).
2. Expand the kernel argument `|x - x'|²` in `t` and `s` to second order in `α`.
3. Integrate over `t` analytically (polynomial-times-radial) and `s` over `[0, …]`
   on the solid side.
4. Compare coefficients `F_1, F_2` against a series expansion in `1/R` of the exact
   2D circle-boundary closed form (PLAN.md next item).

## 4. Special cases and limits

`α → 0`: planar. A disk of radius `R` must reproduce `κ = 1/R` for all `d` with
`2R + d >= h` and be replaced by "support engulfs the obstacle" otherwise.

## 5. Checks

| check | how | status |
|---|---|---|
| `F_0` equals the planar closed form | Maple | [P] |
| series of exact disk result in `1/R` equals `F_0 + αF_1 + α²F_2` | Maple series + mpmath | [P] |
| sign of `κ` consistent with the disk | exact | [P] |
| truncation error vs `α` (order of convergence) | mpmath | [P] |
| compare against tier 2 polygon of the same disk as `L_T/R → 0` | exact tier 2 | [P] |

## 6. Known failure modes

`α` not small; thin solids (tier 4 regime); curvature estimated from a noisy SDF.

## 7. Open questions

Evaluate curvature at the particle or at the closest point; behaviour at curvature
discontinuities; tier-selection metric.
