# Weighted disk volume rule and independent circle flux

The numerical provider integrates a 2D solid disk/cylinder cross section.
Let `R` be radius/support, `d` signed outward query-to-circle distance/support,
`D=R+d >= 0` centre-to-query distance/support, and `n` the outward unit axis.
All coordinates below are normalized by support. A source point is
`y=-r*n*cos(theta)+r*t*sin(theta)`, with tangent `t=(-n_y,n_x)`.

A radial shell is full for `r <= -d`, empty for `r < d` when `d>0`, and has
a circular angular cap for `|d| <= r <= 2R+d`. Splitting also at the kernel's
knots and support radius keeps each profile piece separate. In a partial cap,
parameterize radial position by the angle on the actual circle boundary:

```
r² = d² + 4 D R sin(phi/2)²,
dr/dphi = D R sin(phi)/r,
angle = atan2(R sin(phi), d + 2R sin(phi/2)²).
```

This resolves the square-root cap endpoints without replacing the circle by
chords. Radial Gauss nodes are prepared in host fp64 after that change of
variable. Full shells instead use ordinary radial Gauss. Additional geometric
radial panels resolve the nearest-circle boundary layer. Panel grade floors
(1e-4 fp32 / 1e-8 fp64) choose integration panels; they do not move, clamp or
snap the actual distance or circle. Angular Gauss maps to `[-angle,+angle]`.
The angular count is at least `4*(1+max(|alpha|+|DS|))` and at least the radial
order and 16. Each prepared rule compares a higher radial/angular order, may
promote, or rejects unresolved convergence/resource use.

Native `wp.func` sampling evaluates factored kernel profiles, including cubic
Laurent derivative terms, in selected fp32/fp64 with the area Jacobian r. The three profile values
are W, L=W'/r, and Q=L'/r, contracted as

```
y^alpha * W,
-y^alpha * y_i * L,
y^alpha * (delta_ij*L + y_i*y_j*Q).
```

Fused device consumers and parallel batches share those sampling/contraction
functions. Every moment reduction uses explicit selected-dtype Neumaier
summation. fp32 uses bounded-angle degree-13 sine / degree-14 cosine Taylor
polynomials after reduction to `[-pi/2,pi/2]`, with explicit same-dtype FMAs.
Their real-arithmetic remainder bounds are below 7e-10 / 7e-11 there; native
floating-point errors are covered by independent fixtures, not that bound.
This avoids libdevice's unused large-angle fp64 fallback. fp64 retains ordinary
native trigonometry. No polynomial-exact radial/angle claim is made for this
circular geometry; two-order agreement is not a certified global error bound.

The independent reference uses another integration domain. For any homogeneous
monomial A of degree m and radial profile g, define

```
F_m(r) = integral_0^r t^(m+1) g(t) dt,
V(y) = y*A(y)*F_m(|y|)/|y|^(m+2).
```

Euler's identity `y.grad(A)=m*A` gives `div(V)=A*g`. Integrating `V.n` over
**every arc** of the circle therefore gives the requested volume moment. The
oracle constructs exact rational cumulative profiles before arbitrary-precision
circle integration, including the exterior regular tail. It rotates monomials
into axial/transverse powers and uses transverse symmetry; no native volume
nodes enter the calculation. All exterior arcs must remain in this reference.
Native support localization may prune whole disks whose support balls do not
intersect, but cannot prune those regular-flux arcs.

Physical outputs multiply once by `h**(|alpha|-|DS|)`. Kernel derivatives do
not differentiate `y^alpha`; a complete query derivative additionally subtracts
`alpha_j*M_(alpha-e_j)`. Exact whole-support polynomial moments and kernel
integration-by-parts identities test every request through degree nine.

Disk arrays remain additive solid geometry, including overlaps and unique
periodic images. They do not supply an effective resistance/permeability or a
porous-pressure law. Resolved 2D fibre flow is transverse; longitudinal response
requires an axial velocity formulation or 3D flow. No curved FEM shape basis or
shape adjoint is advertised by this moment provider.
