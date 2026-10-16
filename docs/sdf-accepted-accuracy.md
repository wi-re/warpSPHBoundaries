# Accuracy of accepted smooth SDF models

This document explains the baseline mixed model and its references. The
[shared curvature-moment family](derivations/curvature-moments-2d.md) supersedes
the planar moment rows and restores the leading tangential gradient. Reproduce
current behavior with `scripts/studies/curvature_corrections.py`; the old report
remains historical evidence.

The sampled-cell guard validates the interpolated distance field's domain. It
does **not** certify an integral tolerance. The
[accepted-SDF report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-accuracy-cpu/report.md) measures the
local model where production Scene actually selects it. Rejected queries remain
uncomputed in this report; they are not replaced by reference or fallback values.
The separate [rejection/fallback checkpoint](sdf-guard.md) remains valid.

## Baseline moment contract and expected accuracy

All lengths are in support units. The audit retains the common twelve-moment
order and explicitly leaves the three unavailable production second moments grey.

| Channels | Shipped local model | Expected limitation |
|---|---|---|
| m00 | F0(d) + κ F1(d) + κ² F2(d) | Constant-curvature truncation; variable curvature across support; sampled d/κ |
| Gx, Gy | (F0′ + κ F1′ + κ² F2′) n | Normal derivative only; omits tangential curvature variation, in addition to truncation/reconstruction |
| m10, m01 | Half-plane moments | Curvature bias remains even with exact SDF jets |
| g1xx, g1xy, g1yx, g1yy | Half-plane moments | Curvature bias; not a consistent derivative of the curvature-corrected mass/first-moment family |
| m20, m11, m02 | Not exposed by production SDF | Reference domain integrals exist; production coverage remains M1 |

There is no universal float32 approval from the guard's |κh|≤0.5 threshold.
An accuracy selector must account for the requested moments and variation of
geometry across the support, as well as differential-field conditioning.

## Independent references and error split

`validation/sdf_accuracy.py` integrates the actual ellipse or cubic-graph solid
using Cartesian area quadrature. Support-circle intersections and kernel knots
split the domain analytically; adaptive Gauss–Kronrod integrates the outer
direction, with 48/96-node Gauss refinement in the inner direction. Factored
radial profiles avoid expanded high-degree polynomial cancellation. All twelve
moments, including the currently missing production moments, are evaluated.

Independent one-dimensional boundary flux and integration-by-parts oracles use
mpmath at 45 and 60 digits. These are separate from production polygon primitives,
curvature tables and the area rule. One fluid-side point per geometry/kernel
provides 52 independent all-moment spot checks. Stored results use longdouble;
precision agreement is limited by storage. This supports the finite epsilon32
audit, not universal few-double-eps certification.

The same-model reference uses extended-precision planar closed forms and
independent boundary shape derivatives. For a circle graph
`z = -κ t²/2 + O(κ³)` at closest-point distance d:

```
F1(d) = -1/2 ∫ t² W(sqrt(t²+d²)) dt
F2(d) =  d/8 ∫ t⁴ [-W′(r)/r] dt
```

The derivative profiles give F0′, F1′ and F2′ independently of the shipped table.
Tests compare these shape coefficients with the existing tubular derivation for
four kernels and both signed-distance continuations. Maple verifies the shape
coefficients and offset-coordinate chain rule in
[`20_sdf_accuracy.mpl`](../maple/20_sdf_accuracy.mpl).

The saved **signed** error identity is:

```
table/evaluation = shipped model(sampled jets) - reference model(sampled jets)
reconstruction   = reference model(sampled jets) - reference model(exact jets)
local model      = reference model(exact jets) - actual geometry integral
total            = table/evaluation + reconstruction + local model
```

The completed CPU/CUDA study contains **2,808 accepted records per device**
(13 geometries × four kernels × three grids × 18 offsets). No rejected record
is silently filled. None of the 156 geometry/kernel/grid sweeps meets epsilon32
on all nine implemented channels. Maximum independent boundary/area spot-check
discrepancy is 3.65e-15 absolute. This establishes a model-accuracy limitation
on the tested smooth domain, while the separate rejection/fallback gate remains
acceptable. CPU/CUDA selection agrees throughout, and the maximum absolute
difference between actual accepted Scene outputs is 4.45e-16.

Table/evaluation includes interpolation error. Refinement differences and
adaptive quadrature estimates are empirical; they are plotted separately and
never described as certified error bounds. The precision gate includes the
original-domain reference estimate. Signal scales use the maximum scalar,
vector or tensor norm over the signed-distance sweep, avoiding division by zero
for components that vanish by symmetry. Raw absolute errors remain available.

## A smooth inflection exposes a missing derivative

The graph `z=b t³/6` has κ=0 at the closest point but κ_s=-b. Its true boundary
integral generally has a tangential gradient. In offset coordinates
`x(s,d)=c(s)+d n(s)`, `x_s=(1+dκ)t`, so the mass model itself requires

```
∇ λ_model = λ_d n + (F1+2κF2) κ_s/(1+dκ) t.
```

The baseline retained only `λ_d n`. The report compares its tangential
gradient, the true geometry integral and the derivative of its own mass model.
Adding this chain-rule term would restore that particular derivative, but it
would not by itself complete the variable-curvature integral expansion or the
other moments. The current shared order-two family implements that consistent expansion; higher orders remain separate.

## Numerical endpoint defect corrected

Exact-on-wall samples also exposed an implementation error separate from the
curved-geometry approximation. The F0 table starts at qmin=1e-6. Its value used
a linear continuation below that node, but the cumulative integral Lambda was
clamped at the node, dropping an interval of mass approximately qmin/2 from the
normal first moment. The correction integrates that same continuation:

```
gap = max(qmin - |d|, 0)
Lambda += F0(qmin)*gap - F0′(qmin)*gap²/2
```

The maximum first-moment table/evaluation discrepancy across the finest-grid
study falls from **4.996e-7 to 2.386e-12**. Actual before-fix Scene measurements
are preserved in
[the archived CPU report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/sdf-accuracy-before-wall-fix/cpu/report.md)
and its CUDA companion. The current reports show the corrected outputs and an
endpoint-error plot. Kernel tables and curvature truncation are unchanged.

Regression checks cover four kernels, CPU/CUDA, both sides of the wall and the
zero slope of the even first moment. Symbolic identities are checked in `maple/20_sdf_accuracy.mpl`;
regressions are in `tests/validation/test_sdf_accuracy.py`. Test-run logs are kept
in the external development archive.

## Reproduction and limits

```
python scripts/studies/curvature_corrections.py --device cpu
python scripts/studies/curvature_corrections.py --report-only
python scripts/studies/curvature_corrections.py --device cuda:0 \
  --output "$EVIDENCE/curvature-corrections-cuda" \
  --reference-cache "$EVIDENCE/sdf-accuracy-cpu"
```

The study covers circles R/h=3,6,12,24, a concave cavity, ellipses with changing
curvature, rotations, cubic inflections and a cubic wall with nonzero curvature.
Four kernels, 18 signed offsets (including exactly on the wall) and spacings h/16,h/32,h/64 separate discretization
from model effects. Cubic-wall distances minimize over every real stationary
point of the squared-distance polynomial; a local Newton branch is not used.

This is an offline correctness study with the numerical endpoint repair above.
It does not tune GPU throughput, supply
an accuracy-aware production fallback selector, certify unknown subgrid geometry,
or complete the production curvature-moment formulas. Existing polygon fallback accuracy
remains independently established. The report and progress matrix record the
measured acceptance and per-moment gate results.

See [output configuration and reference prerequisites](validation.md) for
`EVIDENCE` setup. Development priorities and chronological verification logs
are maintained in the external acceptance matrix.
