"""Reviewer's probe for REVIEW-002 (throw-away; run from the repo root with the `warp` env).
(1) physical tensile term T_y at the flat-floor point (0,0.3), H=1, by (a) the closed-form factor c2^5/(pi^4 c25) and (b) an independent
    midpoint-grid integral of W^4 dW/dy over the half disk (no warpSPHBoundaries code); the WORK-002 doc uses (c2/c25)^5/pi^4.
(2) np2d stable-route error of W^5 (w4) as a function of the distance from a vertex, exact mpmath reference."""
import sys, math
sys.path.insert(0, "python")
import numpy as np, mpmath as mp
from fractions import Fraction as F
from warpSPHBoundaries.edge import kernels, np2d, geometry as G
from warpSPHBoundaries.edge.core import GUARD, block_grad
from warpSPHBoundaries.edge.kernels import power_terms as terms, power_monomials as ref_coeffs
POLY = [(0, 0), (1, 0), (1, 1), (0, 1)]

# ---- (1)
for fam, c2 in (("w2", 7.0), ("w4", 9.0)):
    nm = fam + "p5"; kernels.KERNELS[nm] = kernels._from_terms(nm, terms(5, fam))
    c2k = float(kernels.KERNELS[fam].c2_pi); c25 = float(kernels.KERNELS[nm].c2_pi)
    sq = np.array([(-2., -2.), (2., -2.), (2., 0.), (-2., 0.)])
    g0 = np2d.gradient(sq[None], np.array([[0.0, 0.3]]), nm, h=1, dtype=np.float64, stable=(16, 8))[0]
    T_good = c2k**5 / (math.pi**4 * c25) / 5 * g0[1]
    T_doc = (c2k / c25)**5 / math.pi**4 / 5 * g0[1]
    # independent: W(q) = c2k/pi * shape(q); shape from terms(1)
    t1 = terms(1, fam)
    shape = lambda q: sum(float(c) * (1 - q)**p for c, _, p in t1)
    n = 4000; xs = (np.arange(n) + .5) / n * 2 - 1; X, Y = np.meshgrid(xs, xs - 0.0)
    Yp = -(np.arange(n) + .5) / n  # solid y' in (-1,0)
    XX, YY = np.meshgrid(xs, Yp); dx = XX - 0.0; dy = 0.3 - YY; r = np.hypot(dx, dy); q = r
    ok = q < 1; h = 2.0 / n * 1.0 / n
    W = c2k / math.pi * shape(q) * ok
    dq = 1e-6; Wp = (c2k / math.pi * (shape(q + dq) - shape(q - dq)) / (2 * dq)) * ok
    Ty_grid = (W**4 * Wp * dy / np.where(r > 0, r, 1)).sum() * h
    print(fam, "c2_pi", c2k, "c25", c25, "g0_y", g0[1], " T_y closed form (c2^5/(pi^4 c25)/5):", T_good, " doc factor:", T_doc, " grid:", Ty_grid)

# ---- (2)
fam = "w4"; nm = "w4p5"; kernels.KERNELS[nm] = kernels._from_terms(nm, terms(5, fam)); a = ref_coeffs(5, fam); Cn = float(kernels.KERNELS[nm].c2_pi) / math.pi
def ref(px, py):
    with mp.workdps(150 + GUARD):
        P = G.prepare(POLY, (px, py), h=1)
        gs = [block_grad(P, j, 1) for j in range(len(a))]
        tot = [mp.mpf(0), mp.mpf(0)]
        for j, aj in enumerate(a):
            if aj:
                f = mp.mpf(aj.numerator) / mp.mpf(aj.denominator); tot[0] += f * gs[j][0]; tot[1] += f * gs[j][1]
        return np.array([float(tot[0]), float(tot[1])]) * Cn / 1.0   # same convention as q2_conditioning.ref_for (C = c2_pi/pi applied there)
scale = 3.767
print("distance from vertex (0,0) along the outward diagonal; w4p5 worst |err|/scale  (stable 8,6 | 16,8 | plain)")
for d in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3):
    for tag, p in (("out", (-d / math.sqrt(2), -d / math.sqrt(2))), ("in", (d / math.sqrt(2), d / math.sqrt(2)))):
        R = ref(*p)
        row = []
        for st in ((8, 6), (16, 8), None):
            g = np2d.gradient(np.array(POLY, float)[None], np.array([p]), nm, h=1, dtype=np.float64, stable=st)[0]
            row.append(np.abs(g - R).max() / scale)
        print("  d=%-7g %-3s" % (d, tag), "  ".join("%.2e" % e for e in row))
