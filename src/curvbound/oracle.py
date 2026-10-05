"""Independent high-precision quadrature oracles.

These re-derive lambda directly from the defining integrals (no shared code
with symbolic.py) so they can validate the exported closed forms:

  3-D planar : lambda = Int_d^1 C3 W(q) 2 pi q (q-d) dq
  2-D planar : lambda = Int_d^1 C2 W(q) 2 q arccos(d/q) dq
  sphere     : lambda = Int_d^{qmax} C3 W(q) pi q/D (R^2-D^2+2Dq-q^2) dq,
               D = R+d, qmax = min(1, 2R+d)
  2-D Cartesian: true double integral over the support disk cut by the
               chord (no 1-D reduction) -- the most independent check.

Unit support h=1, particle outside the boundary, d in [0,1].
"""
from fractions import Fraction

import mpmath as mp

from .kernels import kernel


def _to_mpf(x) -> mp.mpf:
    """int / Fraction / str / mpf -> mp.mpf (this mpmath build rejects Fraction)."""
    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)


def _c2(kd) -> mp.mpf:
    return (mp.mpf(kd.c2_pi.numerator) / mp.mpf(kd.c2_pi.denominator)) / mp.pi


def _c3(kd) -> mp.mpf:
    return (mp.mpf(kd.c3_pi.numerator) / mp.mpf(kd.c3_pi.denominator)) / mp.pi


def _w_hat_mp(kd, q: mp.mpf) -> mp.mpf:
    if kd.kind == "cubic":
        if q <= mp.mpf(1) / 2:
            return (1 - q) ** 3 - 4 * (mp.mpf(1) / 2 - q) ** 3
        return (1 - q) ** 3
    v = mp.mpf(0)
    for p, c in kd.poly:
        v += (mp.mpf(c.numerator) / mp.mpf(c.denominator)) * q ** p
    return (1 - q) ** kd.n * v


def quad_planar3d(kernel_name: str, d, dps: int = 40) -> mp.mpf:
    kd = kernel(kernel_name)
    c3 = _c3(kd)
    d = _to_mpf(d)
    f = lambda q: c3 * _w_hat_mp(kd, q) * 2 * mp.pi * q * (q - d)
    if kd.kind == "cubic" and d < mp.mpf(1) / 2:
        return mp.quad(f, [d, mp.mpf(1) / 2, 1])
    return mp.quad(f, [d, 1])


def quad_planar2d(kernel_name: str, d, dps: int = 40) -> mp.mpf:
    kd = kernel(kernel_name)
    c2 = _c2(kd)
    d = _to_mpf(d)
    f = lambda q: c2 * _w_hat_mp(kd, q) * 2 * q * mp.acos(d / q)
    if kd.kind == "cubic" and d < mp.mpf(1) / 2:
        return mp.quad(f, [d, mp.mpf(1) / 2, 1])
    return mp.quad(f, [d, 1])


def quad_sphere(kernel_name: str, R, d, dps: int = 40) -> mp.mpf:
    kd = kernel(kernel_name)
    c3 = _c3(kd)
    R = _to_mpf(R)
    d = _to_mpf(d)
    D = R + d
    qmax = min(mp.mpf(1), 2 * R + d)
    f = (lambda q: c3 * _w_hat_mp(kd, q)
         * mp.pi * q / D * (R ** 2 - D ** 2 + 2 * D * q - q ** 2))
    return mp.quad(f, [d, qmax])


def quad_planar2d_cartesian(kernel_name: str, d, dps: int = 30) -> mp.mpf:
    """True 2-D Cartesian quadrature (no 1-D shell reduction).

    Particle at the origin, boundary line y = d, boundary side y > d:
        lambda = Int_{x} Int_{y=d}^{sqrt(1-x^2)} C2 W_hat(sqrt(x^2+y^2)) dy dx
    over |x| <= sqrt(1-d^2).  For the cubic spline the knot circle r = 1/2 is
    added as a breakpoint wherever it crosses the inner interval.
    """
    kd = kernel(kernel_name)
    c2 = _c2(kd)
    d = _to_mpf(d)
    xmax = mp.sqrt(1 - d ** 2)

    def inner(x: mp.mpf) -> mp.mpf:
        ytop = mp.sqrt(1 - x ** 2)
        pts = [d]
        if kd.kind == "cubic" and x ** 2 < mp.mpf(1) / 4:
            yk = mp.sqrt(mp.mpf(1) / 4 - x ** 2)
            if d < yk < ytop:
                pts.append(yk)
        pts.append(ytop)
        g = lambda y: c2 * _w_hat_mp(kd, mp.sqrt(x ** 2 + y ** 2))
        return mp.quad(g, pts)

    # dense outer breakpoints: inner(x) ~ sqrt(xmax - |x|) at the endpoints
    # (the chord degenerates), which a 3-way split resolves only to ~3e-15
    # for the cubic spline kernel (observed)
    n = 16
    return mp.quad(inner, [-xmax + 2 * xmax * mp.mpf(i) / n for i in range(n + 1)])


def physical_planar3d(kernel_name: str, d, h, dps: int = 30) -> mp.mpf:
    """The PHYSICAL 3-D integral with explicit support radius h:

        Int_{r=d}^{h} (C3/h^3) W_hat(r/h) 2 pi r (r-d) dr

    which by the substitution r = h q must equal lambda_3(d/h).  Used for the
    h-scaling test.
    """
    kd = kernel(kernel_name)
    c3 = _c3(kd)
    d = _to_mpf(d)
    h = _to_mpf(h)
    f = (lambda r: (c3 / h ** 3) * _w_hat_mp(kd, r / h)
         * 2 * mp.pi * r * (r - d))
    if kd.kind == "cubic" and d < h / 2:
        return mp.quad(f, [d, h / 2, h])
    return mp.quad(f, [d, h])


__all__ = [
    "quad_planar3d", "quad_planar2d", "quad_sphere",
    "quad_planar2d_cartesian", "physical_planar3d",
]
