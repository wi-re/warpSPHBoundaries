"""Independent quadrature oracle: POLAR coordinates around x with radial breakpoints.

    int_T y^alpha g(r) dA = int dtheta  w^alpha  [F(rho_out(theta)) - F(rho_in(theta))],
    F(rho) = int_0^rho r^(k+1) g(r) dr           (exact antiderivative of the polynomial pieces),

so the integrand is smooth between angular breakpoints.  Breakpoints: vertex
angles and the angles of the points where an edge crosses a circle of radius R_j
(solved from the quadratic |p + u d|^2 = R^2, NOT from the chord logic of core.py).
The in/out structure of a ray is found by direct ray-segment intersection and an
exact point-in-polygon test at the piece midpoint.

It shares NO code with core.py / primitives.py (only the exact vertex list from
`geometry.locate`/`signed_area2`), so it validates formula AND clipping logic.

Profiles: list of pieces (lo, hi, coeffs) with exact Fraction coefficients,
g(r) = sum_i coeffs[i] r^i on [lo, hi]  (e.g. pi*W of an EdgeKernel).
"""

import mpmath as mp

from . import geometry as G
from .kernels import kernel as get_kernel, pderiv
from .mpq import mpq, to_frac


def _F_factory(pieces, k):
    """F(rho) = int_0^rho r^(k+1) g(r) dr for piecewise-polynomial g; returns mp function."""
    pw = []
    for lo, hi, c in pieces:
        # int_lo^x r^(k+1) sum c_i r^i = sum c_i (x^(k+2+i) - lo^(k+2+i)) / (k+2+i)
        items = c.items() if isinstance(c, dict) else enumerate(c)       # power -> coeff (powers >= -1 ok)
        pw.append((mpq(lo), mpq(hi), [(mpq(ci) / (k + 2 + i), k + 2 + i) for i, ci in items if ci != 0]))

    def F(rho):
        tot = mp.mpf(0)
        for lo, hi, terms in pw:
            if rho <= lo:
                break
            top = min(rho, hi)
            for coef, ex in terms:
                tot += coef * (top ** ex - lo ** ex)
        return tot
    return F


def _intersections(vs_mp, theta):
    """rho > 0 of the ray (cos t, sin t) with every segment; list of (rho, edge_index)."""
    w = (mp.cos(theta), mp.sin(theta))
    out = []
    n = len(vs_mp)
    for i in range(n):
        p, q = vs_mp[i], vs_mp[(i + 1) % n]
        d = (q[0] - p[0], q[1] - p[1])
        den = w[0] * d[1] - w[1] * d[0]
        if den == 0:
            continue
        # p + u d = rho w  ->  u = (p x w)/(w x d)... solve 2x2
        rho = (p[0] * d[1] - p[1] * d[0]) / den
        u = (p[0] * w[1] - p[1] * w[0]) / den
        if rho > 0 and 0 <= u <= 1:
            out.append((rho, i))
    out.sort(key=lambda t: t[0])
    return out


def _inside_exact(vs_frac, pt):
    """exact winding test of the mp point pt (converted exactly) for the polygon vs_frac (origin-centred)."""
    px, py = to_frac(pt[0]), to_frac(pt[1])
    sh = [(v[0] - px, v[1] - py) for v in vs_frac]
    where, _ = G.locate(sh)
    return where in ("inside", "edge", "vertex")


def _breaks(vs_mp, radii):
    angs = []
    n = len(vs_mp)
    for i in range(n):
        p, q = vs_mp[i], vs_mp[(i + 1) % n]
        if p[0] != 0 or p[1] != 0:
            angs.append(mp.atan2(p[1], p[0]))
        d = (q[0] - p[0], q[1] - p[1])
        A = d[0] ** 2 + d[1] ** 2
        B = 2 * (p[0] * d[0] + p[1] * d[1])
        for R in radii:
            C = p[0] ** 2 + p[1] ** 2 - mpq(R) ** 2
            disc = B * B - 4 * A * C
            if disc < 0:
                continue
            sq = mp.sqrt(disc)
            for u in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
                if 0 < u < 1:
                    pt = (p[0] + u * d[0], p[1] + u * d[1])
                    angs.append(mp.atan2(pt[1], pt[0]))
        # edge direction angles through the origin line: rho -> infinity not needed (clipped by R)
    angs += [-mp.pi, mp.pi]
    angs = sorted(set(angs))
    return angs


def polar_moment_profile(verts, x, alpha, pieces, h=1, dps=40, maxdegree=10):
    """int_T y^alpha g(|y|) dA  with the exact polygon `verts`, point `x`, support radius h,
    radial profile `pieces` given in units of h (g in units where r is r/h)."""
    k = alpha[0] + alpha[1]
    with mp.workdps(dps + 15):
        h = to_frac(h)
        xf = (to_frac(x[0]), to_frac(x[1]))
        vs = [((to_frac(v[0]) - xf[0]) / h, (to_frac(v[1]) - xf[1]) / h) for v in verts]
        if G.signed_area2(vs) < 0:
            vs = vs[::-1]
        vs_mp = [(mpq(v[0]), mpq(v[1])) for v in vs]
        radii = sorted({hi for lo, hi, c in pieces})
        F = _F_factory(pieces, k)
        angs = _breaks(vs_mp, radii)
        total = mp.mpf(0)
        for a0, a1 in zip(angs[:-1], angs[1:]):
            if a1 - a0 < mp.mpf(10) ** (-(dps + 10)):
                continue
            am = (a0 + a1) / 2
            xs = _intersections(vs_mp, am)
            # intervals of the ray inside the polygon: test midpoints between successive crossings
            w = (mp.cos(am), mp.sin(am))
            rhos = [0] + [r for r, _ in xs]
            idxs = [None] + [i for _, i in xs]
            intervals = []
            for m in range(len(rhos) - 1):
                mid = (rhos[m] + rhos[m + 1]) / 2
                if mid == 0:
                    continue
                if _inside_exact(vs, (mid * w[0], mid * w[1])):
                    intervals.append((idxs[m], idxs[m + 1]))
            if not intervals:
                continue

            def rho_of(i, th):
                p, q = vs_mp[i], vs_mp[(i + 1) % len(vs_mp)]
                d = (q[0] - p[0], q[1] - p[1])
                den = mp.cos(th) * d[1] - mp.sin(th) * d[0]
                return (p[0] * d[1] - p[1] * d[0]) / den

            def f(th):
                s = mp.mpf(0)
                ca, sa = mp.cos(th), mp.sin(th)
                om = ca ** alpha[0] * sa ** alpha[1]
                for ia, ib in intervals:
                    ra = mp.mpf(0) if ia is None else rho_of(ia, th)
                    rb = rho_of(ib, th)
                    s += F(rb) - F(ra)
                return om * s
            total += mp.quad(f, [a0, a1], maxdegree=maxdegree)
        return +total


def _W_pieces(kname):
    return get_kernel(kname).pieces            # pi*W pieces


def polar_moment(verts, x, kernel, alpha, h=1, dps=40, **kw):
    """int_T y^alpha W dA by polar quadrature (units: h^k scaling applied)."""
    pieces = _W_pieces(kernel)
    v = polar_moment_profile(verts, x, alpha, pieces, h=h, dps=dps, **kw)
    return +(v / mp.pi * mpq(to_frac(h)) ** (alpha[0] + alpha[1]))


def polar_value(verts, x, kernel, h=1, dps=40, **kw):
    return polar_moment(verts, x, kernel, (0, 0), h=h, dps=dps, **kw)


def polar_gradient(verts, x, kernel, h=1, dps=40, **kw):
    """grad_x int_T W dA = int_T (-W'(r)/r) y dA  (independent radial profile -W'/r)."""
    # pderiv gives d[j] = (j+1) c_{j+1}, the coefficient of r^j in W'; W'/r has r^(j-1)
    pieces = []
    for lo, hi, c in _W_pieces(kernel):
        dc = pderiv(list(c))
        # -W'/r = -sum_j d[j] r^(j-1); the r^-1 term (linear coefficient of the outer cubic
        # piece, r >= 1/2 there) is integrable against r^(k+1), k >= 1
        pieces.append((lo, hi, {j - 1: -d for j, d in enumerate(dc) if d != 0}))
    gx = polar_moment_profile(verts, x, (1, 0), pieces, h=h, dps=dps, **kw)
    gy = polar_moment_profile(verts, x, (0, 1), pieces, h=h, dps=dps, **kw)
    hh = mpq(to_frac(h))
    return +(gx / mp.pi / hh), +(gy / mp.pi / hh)


# ======================================================================= exact DISK oracle (curved boundary, tier 2/3/4 reference)
def polar_disk_moment_profile(center, Rd, x, alpha, pieces, h=1, dps=40, maxdegree=10):
    """int_{disk(center, Rd)} y^alpha g(|y|) dA, y = x' - x, by polar quadrature around x with exact ray/circle intersections
    (breakpoints: tangent angles and the angles where the disk boundary crosses a profile radius R_j)."""
    k = alpha[0] + alpha[1]
    with mp.workdps(dps + 15):
        hh = mpq(to_frac(h))
        u = ((mpq(to_frac(center[0])) - mpq(to_frac(x[0]))) / hh, (mpq(to_frac(center[1])) - mpq(to_frac(x[1]))) / hh)
        Rr = mpq(to_frac(Rd)) / hh
        du = mp.sqrt(u[0] ** 2 + u[1] ** 2)
        F = _F_factory(pieces, k)
        inside = du < Rr
        angs = [-mp.pi, mp.pi]
        th_u = mp.atan2(u[1], u[0]) if du > 0 else mp.mpf(0)
        if not inside:
            s = mp.asin(Rr / du)
            angs += [th_u - s, th_u + s]
        for R in sorted({hi for lo, hi, c in pieces}):
            Rq = mpq(R)
            if du > 0:
                cs = (du ** 2 + Rq ** 2 - Rr ** 2) / (2 * du * Rq)
                if abs(cs) <= 1:
                    a = mp.acos(cs)
                    angs += [th_u - a, th_u + a]
        # wrap to (-pi, pi]
        w = []
        for a in angs:
            while a > mp.pi:
                a -= 2 * mp.pi
            while a < -mp.pi:
                a += 2 * mp.pi
            w.append(a)
        w = sorted(set(w))
        total = mp.mpf(0)
        for a0, a1 in zip(w[:-1], w[1:]):
            if a1 - a0 < mp.mpf(10) ** (-(dps + 10)):
                continue

            def f(th):
                ca, sa = mp.cos(th), mp.sin(th)
                b = u[0] * ca + u[1] * sa
                disc = b * b - du ** 2 + Rr ** 2
                if disc <= 0:
                    return mp.mpf(0)
                sq = mp.sqrt(disc)
                if inside:
                    rin, rout = mp.mpf(0), b + sq
                else:
                    if b <= 0:
                        return mp.mpf(0)
                    rin, rout = b - sq, b + sq
                return ca ** alpha[0] * sa ** alpha[1] * (F(rout) - F(rin))
            total += mp.quad(f, [a0, a1], maxdegree=maxdegree)
        return +total


def polar_disk_moment(center, Rd, x, kernel, alpha, h=1, dps=40, **kw):
    v = polar_disk_moment_profile(center, Rd, x, alpha, _W_pieces(kernel), h=h, dps=dps, **kw)
    return +(v / mp.pi * mpq(to_frac(h)) ** (alpha[0] + alpha[1]))


def polar_disk_value(center, Rd, x, kernel, h=1, dps=40, **kw):
    return polar_disk_moment(center, Rd, x, kernel, (0, 0), h=h, dps=dps, **kw)


def polar_disk_gradient(center, Rd, x, kernel, h=1, dps=40, **kw):
    pieces = []
    for lo, hi, c in _W_pieces(kernel):
        dc = pderiv(list(c))
        pieces.append((lo, hi, {j - 1: -d for j, d in enumerate(dc) if d != 0}))
    gx = polar_disk_moment_profile(center, Rd, x, (1, 0), pieces, h=h, dps=dps, **kw)
    gy = polar_disk_moment_profile(center, Rd, x, (0, 1), pieces, h=h, dps=dps, **kw)
    hh = mpq(to_frac(h))
    return +(gx / mp.pi / hh), +(gy / mp.pi / hh)
