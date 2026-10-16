# Shared curvature corrections for twelve 2D moments

**Scope:** geometric order two, smooth isolated boundary; dimensionless support
radius one. Verified symbolically by `maple/21_curvature_moments.mpl` and against
independent shape-Taylor and original-domain integrals. This is an asymptotic
model, not a certified integral-error bound.

## Geometry and counting

Let n point into the fluid, t=(-n_y,n_x), and s be arclength in direction t.
The query is at distance d from its closest boundary point. In that frame the
solid is z≤f(t), where

```
f(t) = -κ t²/2 - κ_s t³/6 + higher geometric orders.
```

For physical support h use d/h, α=κh, β=κ_s h². Count α as order one and β as
order two. For any scalar or gradient-moment integrand A(t,z), evaluated at
z=-d, the common correction is

```
δI[A] = ∫ [(-α t²/2 - β t³/6) A + α² t⁴ A_z/8] dt.
```

Terms α³, αβ, h³κ_ss and higher are omitted. Small closest-point curvature
alone is insufficient: an inflection can have nonzero curvature variation.

## Line coefficients and closed forms

Define B_j(d)=∫t^j W(sqrt(t²+d²))dt, over the support chord. Odd j vanish;
even coefficients are even in d. With D=-W′/r and H=D′/r, integration by parts
gives

```
∫t²D=B0, ∫t⁴D=3B2, ∫t⁶D=5B4,
∫t⁴H=-3B0, ∫t⁶H=-15B2.
```

The existing mass coefficients satisfy F1=-B2/2 and F2=3dB2/8. For a=|d|:

```
B0 = -F0′(a)
B2 = -2 F1(a)
B4 = -6 ∫_a^1 q F1(q)dq.
```

All coefficients are closed-form capable. `edge.curvature.line_moment` expands
each exact rational kernel piece into the existing S/I primitives, including
the wall and kernel knots. Production integrates the existing cubic Hermite
splines segment by segment, including their continuation below the first node;
it does not introduce new fitted tables or boundary quadrature.
`edge.curvature.moments` evaluates the complete twelve-moment model with
precision-refinable closed forms, including the wall. It shares neither the
production splines nor their numerical interpolation error. Closed form here
describes the chosen local model; geometry truncation remains an approximation.

## Complete family

Let L be planar mass. Let T and N be planar scalar second moments in the
tangent and normal directions. For positive d,

```
L=F0(d)
T=-2 ∫_d^1 F1(q)dq
N=d²F0(d)+2 ∫_d^1 qF0(q)dq.
```

For negative d take the complement: L=1-F0(a), T=σ²-T(a), N=σ²-N(a), where
σ² is the exact whole-kernel coordinate variance. The planar normal first
moment is -B2. The corrected local components are:

| Quantity | Order-two value |
|---|---|
| λ | L−αB2/2+3dα²B2/8 |
| m_t | −βB4/6 |
| m_n | −B2+αdB2/2+α²(B4−3d²B2)/8 |
| m_tt | T−αB4/2+5dα²B4/8 |
| m_tn | βdB4/6 |
| m_nn | N−αd²B2/2+α²(−2dB4+3d³B2)/8 |
| G_t | −βB2/2 |
| G_n | −B0+αdB0/2+3α²(B2−d²B0)/8 |
| g1_tt | L−3αB2/2+15dα²B2/8 |
| g1_tn=g1_nt | βdB2/2 |
| g1_nn | L+dB0−αd²B0/2+α²(−6dB2+3d³B0)/8 |

Rotate vectors and tensors using the same (t,n) frame. Restore physical units:
m1 times h, m2 times h², G divided by h; λ and g1 are dimensionless. Empty/full
supports use exact normal-free isotropic values, including zero-gradient centres.

## Differential consistency and its limit

The Frenet identities are n_s=κt, t_s=−κn and x_s=(1+dκ)t. Together with
B2′=−dB0 and B4′=−3dB2 they establish, through geometric order two,

```
∇λ = G
∂m1/∂x = g1−λI.
```

At an inflection the leading tangential mass gradient is βF1. The exact chain
rule for a finite constant-curvature mass truncation would also contain
`(F1+2αF2)β/(1+dα)`. Its additional terms are of geometric order three and
above; inserting just that rational factor would not complete the other
moments at those orders. The present family keeps one consistent order.
Spline interpolation adds numerical discrepancies to these analytic identities.

## Sampling and guarding curvature variation

For exact SDF jets Ld=Δd and Q=1−d Ld, closest-surface curvature variation is

```
κ_s = t·∇Ld / Q³.
```

`SdfRep.curvatureVariation` blends central differences of the nodal Laplacian.
It needs a two-node distance stencil. The support guard bounds
`||∇Ld||/Q_min³` over every blended-jet cell, using convex node intervals. This
gives an additional maximum support `sqrt(maxCurvatureVariationH2 / bound)`;
the default limit is |κ_s|h²≤0.5. It is a conservative domain restriction,
not a precision target or a bound on unknown subgrid geometry.

The rapid wall `z=0.001 sin(20t)` is a documented counterexample: |κ|≤0.4,
while |κ_s|≈8. Curvature-only selection can accept it; the new variation
restriction rejects it. Higher geometry derivatives can still invalidate other
locally admissible models. Forced diagnostics retain these failures explicitly.

## API, verification and remaining limits

`Tier3.curvature_moments(q,n,α,β)` is the Torch oracle for all twelve canonical
channels in support units. Scene's default `SdfRep(backend="warp")` evaluates
the same coefficients and formulas in a single float64 Warp kernel. It samples
the distance jets once, checks coverage and all-support sparse minima, locates
the spline segment once, and computes only the requested moment families.
`backend="torch"` retains the oracle for diagnostics. Scene exposes m1, g1 and
m2 through `RepMoments`; its half-plane representation also exposes m2. Invalid
stencils, nonfinite variation and posterior nonfinite/out-of-range mass request
the supplied polygon fallback. Rejected data never enter the local expansion.

`Scene.precompute(..., channels=...)` uses the existing surface channel IDs;
the canonical array order differs as documented in [surface moments](../surface-moments.md).
Unrequested outputs are zero and unrequested second moments are `None`.
Mass is always evaluated internally for the posterior validity check. Cached
nodes, Laplacian gradients, interval minima and spline integrals assume static
sampled geometry; `SdfRep.to` clears its field caches. Policy changes rebuild the
interval cache. Warm evaluation and CUDA graph replay use the active Torch
stream without copying query positions to the host. The sparse-table level is
selected with integer widths, avoiding CUDA `log2` rounding at powers of two.
Changing static sampled geometry, device or guard policy requires rebuilding
the static caches and recapturing an existing graph; query positions and
supports can change in place between replays.

Fusion changes evaluation cost, not the geometric order or its acceptance
domain. Backend parity, nonfinite inputs, corners, cavities, varying supports,
requested channels and changed graph inputs are checked in
`tests/scene/test_sdf_warp.py`. Production keeps the existing sampled-field
domain guard; it does not gain an integral-tolerance certificate.

FusedWall continues to lower SDFs to explicit polygons. Disk/box/volume
representations retain their separately documented moment coverage. This work
does not certify simulation throughput or complete higher geometric orders.

Reproduce the original-domain audit with
`python scripts/studies/curvature_corrections.py`. Independent references
integrate the actual curved domain and direct shape-Taylor integrands, with
separate model, sampled-jet reconstruction, table and quadrature estimates.
See [validation setup](../validation.md) for external output configuration.
Regressions are in `tests/validation/test_curvature.py`; Maple checks the
twelve formulas, moment derivatives and variation recovery. Progress and
supporting images are kept in the external development archive.
