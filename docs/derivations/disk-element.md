# The disk (fibre) element: circle identities, the surface singularity, the table layout

**Status:** 2D: [V] (identities, singular structure and branch layout checked against code-independent references)
**Tier(s):** 4 (curved element, any radius) · **Dimension:** 2D (3D sphere / cylinder: same identities with M(r) = int s^2 W, not implemented)
**Depends on:** `edge-primitives.md`, `tier4-slender-series.md` (circle edge identity `arc_value`), `../notation.md`
**Implemented in:** `src/warpSPHBoundaries/edge/disk.py::disk_channels`, `edge/disktables.py` (`DiskTables`, `PanelTable`), `edge/warpdisk.py` (Warp lookup), `scene/` (`DiskArrayRep`)
**Verified by:** `scripts/derivation_checks/disk_element_checks.py` (this file), `tests/edge/test_disk.py`, `tests/scene/test_disk_array.py`; design, costs and solver results in `../disk-element.md`

## 1. Statement

Solid disk of radius R, centre at distance D from the query x (h = 1, y = x' - x, solid = {|y - (D, 0)| < R}), kernel W(r) = P(r)/pi, radial primitive
Phi(r) = (M(r) - M(1))/pi, M(r) = int_0^r t pi W(t) dt (zero beyond the support), outward normal n = e(phi) of the circle, dl = R dphi, y = (D + R cos phi, R sin phi), r = |y|:

    lam   = int W dA          = 1[D < R] + oint (n . y) Phi(r) / r^2 dl
    m1    = int y W dA        = oint n Phi(r) dl
    g0    = int grad_x W dA   = - oint n W(r) dl
    g1_dj = int y_d d_{x_j} W = - oint n_j y_d W dl + delta_dj lam

(1[D < R] is multiplied by 2 M(1) for a kernel that is not normalised.)  Reflection symmetry about the axis through x and the centre leaves five numbers (lam, m1x, g0x, g1xx, g1yy); everything else is zero or
by rotation.  Near the surface, delta = D - R -> 0, the channels are NOT analytic: with the kernel's cubic coefficient c3 (Wendland C2: W = (7/pi)(1 - 10 q^2 + 20 q^3 - 15 q^4 + 4 q^5), c3 = 140/pi)

    g0x(delta) = analytic(delta) - (3/4) c3 delta^4 ln|delta| + O(delta^6 ln|delta|),     -(3/4) c3 = -105/pi,

and lam is one derivative smoother (delta^5 ln|delta|).  At the end of the support, D = 1 + R - eta, lam ~ eta^(11/2) and g0x ~ eta^(9/2).

## 2. Assumptions and validity

* Radially symmetric kernel, piecewise polynomial in r (the registered Wendland family and the derived `lw2`, `cone`, `w2p5`); the identities hold for the kernel as registered.
* One disk; overlapping disks add (no union).  R > 0; D = 0 is moved to 1e-12.
* The circle quadrature (`disk_channels`) is not valid at |D - R| below about 1e-9 R, see section 6.

## 3. Derivation

1. **Divergence theorem on the circle.**  In 2D, div(y f(r)) = 2 f + r f'.  With f = Phi/r^2: div(y Phi/r^2) = Phi'/r = W, because Phi' = r W (M' = t pi W).  The field is singular at the query.
   If x is outside the disk the region is regular and lam = oint n . (y Phi/r^2) dl.  If x is inside, excise a small circle around x: its outward normal (from the region) is -r_hat and it contributes
   -oint r Phi(r)/r^2 r dphi -> -2 pi Phi(0) = 1, since Phi(0) = -M(1)/pi = -1/(2 pi) for a normalised kernel.  Hence the indicator.
2. **m1, g0, g1.**  div(e_k Phi) = Phi' y_k / r = y_k W (so m1 needs no singular field); grad_x = -grad_y gives g0 = -int grad_y W = -oint n W dl; and
   div(y_d W e_j) = delta_dj W + y_d d_j W gives g1 with the extra delta_dj lam.
3. **Near-singular integrand.**  When the circle passes within |delta| of the query the integrands vary on the angular scale |delta| / sqrt(D R) around phi = pi (the closest point); the quadrature cuts the
   angle at the knot radii of the kernel's pieces and refines geometrically (ratio 4) towards phi = pi.  The channel values are smooth across delta = 0 only in the sense of being continuous with the
   singular structure below; the integrand's size (~1/|delta| in lam) sets the conditioning, section 6.
4. **The log term.**  In the planar limit (R -> infinity) the nearest circle point sees a straight surface at distance eps = |delta|: g0x = int W(sqrt(eps^2 + t^2)) dt over |t| < sqrt(1 - eps^2).  A power
   c_k r^k of the kernel contributes c_k int (eps^2 + t^2)^(k/2) dt; for ODD k the antiderivative contains eps^(k+1) asinh(t/eps), and asinh(1/eps) = ln(1 + sqrt(1 + eps^2)) - ln eps, so its log part is
   -a_k eps^(k+1) ln eps with a_3 = 3/4, a_5 = 5/8.  Wendland C2 has c3 = 140/pi (the q^3 term) and c5 = 28/pi, so the first log term of g0x is -(105/pi) eps^4 ln|eps|, the next eps^6 ln|eps|.
   Curvature of the circle changes the coefficients of higher orders only.  Kernels without an odd power below the cusp (none registered) would have no log.
5. **The support end.**  At D = 1 + R - eta the part of the disk inside the support is a sliver of area ~ eta^(3/2) (circle against circle, curvatures finite), weighted by the kernel's (1 - r)^4: lam ~ eta^(11/2)
   (the gradient channels lose one power: g0x ~ eta^(9/2)).
6. **Table layout (consequence).**  A Chebyshev interpolant converges geometrically only on an interval where the function is analytic.  Panel boundaries are therefore placed on the three lines where it is
   not: delta = 0 (step 4), D = |1 - R| (the circle's far point meets the support circle; R < 1 only) and D = 1 + R (step 5).  For R < 1 the lines delta = 0 and D = 1 - R, i.e. delta = 1 - 2R, intersect at
   R = 1/2, where their order along delta swaps: hence TWO radius branches, R in [0, 1/2] with delta intervals [-R, 0], [0, 1 - 2R], [1 - 2R, 1] and R in [1/2, 1] with [-R, 1 - 2R], [1 - 2R, 0], [0, 1].
   A singularity on a panel END still gives only algebraic convergence (~n^-10 for eps^4 ln eps), so each interval is also mapped through u = sin^2(pi w / 2): the interval end is then a high-order zero in w
   and the decay is geometric (section 5).  For R <= 1 the tables store f / R^2: every channel is ~ R^2 as R -> 0 (lam -> pi R^2 W(D) for a unit-mass kernel), and the scaled function is regular at the point
   limit, which keeps the RELATIVE accuracy for tiny disks (the Chebyshev decay in R is the same with or without the scaling: 3e-11 against 9e-12 at n = 16).  For R > 1 the table is in (delta, s = 1/R); s = 0 is the
   planar half plane (the tier-3 functions), and for D < R - 1 the support lies inside the disk (lam = 1, g1 = I, m1 = g0 = 0, no table).

## 4. Special cases and limits

* R -> 0: lam / R^2 -> pi W(D) (checked, 1e-5 relative at R = 1e-3); m1, g0, g1 -> the point-mass limits.
* R -> infinity: the tier-3 half plane (s = 0 column of the large-radius tables).
* Centre of a large disk (D < R - 1): lam = 1 exactly, no table.
* Beyond the support (D > 1 + R): all channels zero.

## 5. Checks

All in `scripts/derivation_checks/disk_element_checks.py` (run: `python scripts/derivation_checks/disk_element_checks.py`; the polar integrator and the kernel are written out in the script, independent of `kernels.py`).

| check | how | result | status |
|---|---|---|---|
| five channels, 15 (D, R) incl. R < 1, R > 1, inside, outside, 1e-5 off the surface | polar (ray) integration from the query, exact radial antiderivatives (sympy), angle by adaptive quad | <= 1.5e-14 absolute, off the surface 8e-12 at 1e-5 relative offset | [V] |
| coefficient of eps^4 ln eps | sympy log part of int (eps^2 + t^2)^(3/2) dt = -(3/4) eps^4 ln eps; numerical fit of the exact planar g0x(eps) | -33.42254 fitted, -105/pi = -33.42254 predicted; k = 5: -(5/8) eps^6 ln eps | [V] |
| decay of the Chebyshev coefficients of g0x (R = 0.7, delta in [-0.2, 0.2], n = 40) | (a) one panel across delta = 0, (b) split at delta = 0, (c) split + sin^2 map | (a) n^-5.3, (b) n^-9.6, (c) geometric: 7e-16 at n = 28 (b: 1.7e-12) | [V] |
| radius-branch crossing | delta = 0 and delta = 1 - 2R meet at R = 1/2 (sympy) | R = 1/2 | [V] |
| point limit | lam / R^2 at R = 1e-3 against pi W(D) | 5.160949 / 5.160960, 1.312502 / 1.312500, 0.047041 / 0.047040 | [V] |
| f / R^2 analytic in R | Chebyshev coefficients in R at a fixed interval fraction, R in (0, 1/2) | n = 16: 2.6e-11 | [V] |
| support end exponents | slope of log|lam|, log|g0x| against log eta, D = 1 + R - eta, R = 0.3 | 5.49 and 4.48 (predicted 5.5, 4.5) | [V] |
| tables against the reference / polygon path | `tests/edge/test_disk.py`, `tests/scene/test_disk_array.py` | see `../disk-element.md` (5e-8 `w2`, polygon 1e-6) | [V] |

## 6. Known failure modes

* **Exactly on the circle (D = R) the reference returns lam off by 1/2** (the principal value of the singular circle integrand: half the jump of the indicator), and the other channels are off likewise.
  The functions are continuous there; evaluate at |D - R| >= 1e-9 R.  The tables are unaffected (interpolants of smooth panel data, nodes strictly inside the intervals).
* **Conditioning near the surface**: the absolute error of `disk_channels` grows like 6e-17 R / |D - R| (measured 8e-12, 8e-10, 4e-8 at relative offsets 1e-5, 1e-7, 1e-9).  Build tables from nodes that
  stay away from delta = 0 by more than the table's own tolerance (the w map clusters nodes at the interval ends; at the 12-node panels the nearest node is ~1e-3 of the interval).
* Disks that overlap are summed, not united.

## 7. Open questions

* 3D: sphere / infinite cylinder tables (same identities, M(r) = int s^2 W; the cap quadrature builds the tables).  The log structure carries over (odd powers of r).
* Whether `cone` (the cover vector, kernel with a cusp at r = 0: 8e-5 floor) deserves a dedicated interpolation near the query.
