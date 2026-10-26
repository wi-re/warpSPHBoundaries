"""Independent constant-ball checks: integrate clipped affine simplex slices.

The sphere is intersected with a straight coordinate line by quadratic roots;
only the remaining one/two simplex coordinates use adaptive quadrature. The
positive integrand avoids signed face/edge cancellation. Error estimates come
from scipy QUADPACK and are empirical, not certified geometric error bounds.
"""
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from ..edge.mpq import mpq, to_frac


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _roots(a, b, c):
    if not a:
        return [] if not b else [-c/b]
    discriminant = b*b-4*a*c
    if discriminant <= 0:
        return []
    q = -.5*(b+math.copysign(math.sqrt(discriminant), b))
    return [q/a, c/q] if q else [0.]


def _cuts(p, d, inner, length):
    """Outer-coordinate breakpoints: both endpoints on sphere, line tangent."""
    cuts = []
    for a, b in [(p, d), (p+length*inner, d-inner)]:
        cuts += _roots(float(b@b), float(2*a@b), float(a@a-1))
    # Minimum distance to the infinite inner line.
    denom = float(inner@inner)
    a = p-inner*float(p@inner)/denom
    b = d-inner*float(d@inner)/denom
    cuts += _roots(float(b@b), float(2*a@b), float(a@a-1))
    return sorted(set(t for t in cuts if 0 < t < length))


def _span(p, d, limit):
    roots = _roots(float(d@d), float(2*p@d), float(p@p-1))
    if len(roots) != 2:
        return 0.
    lo, hi = sorted(roots)
    return max(0., min(limit, hi)-max(0., lo))


def constant_overlap(vertices, query, radius, tolerance=2e-11):
    """Return (volume or area, estimated absolute error) for a 3/4-node simplex.

    Coordinates are first differenced exactly, then normalized by the radius.
    The exact-input Jacobian is evaluated at the caller's mpmath precision.
    """
    v = [tuple(to_frac(x) for x in row) for row in vertices]
    q = tuple(to_frac(x) for x in query)
    r = to_frac(radius)
    if r <= 0 or len(v) not in (3, 4):
        raise ValueError("positive radius and triangle/tetrahedron required")
    rel = [tuple((x-y)/r for x, y in zip(row, q)) for row in v]
    ds = [tuple(x-y for x, y in zip(row, rel[0])) for row in rel[1:]]
    # Integrate along the longest axis; the shortest is an outer coordinate.
    ds.sort(key=lambda a: _dot(a, a))
    if len(v) == 4:
        det = abs(_dot(ds[0], _cross(ds[1], ds[2])))
        jacobian = mpq(det*r**3)
    else:
        N = _cross(ds[0], ds[1])
        jacobian = mp.sqrt(mpq(_dot(N, N)))*mpq(r)**2
    if not jacobian:
        raise ValueError("degenerate simplex")
    p = np.array([float(mpq(x)) for x in rel[0]])
    D = [np.array([float(mpq(x)) for x in d]) for d in ds]
    if len(v) == 3:
        value, err = quad(lambda u: _span(p+u*D[0], D[1], 1-u), 0, 1,
                          points=_cuts(p, D[0], D[1], 1), epsabs=tolerance, epsrel=tolerance, limit=200)
    else:
        inner_errors = []

        def outer(u):
            base = p+u*D[0]
            value, err = quad(lambda v: _span(base+v*D[1], D[2], 1-u-v), 0, 1-u,
                              points=_cuts(base, D[1], D[2], 1-u),
                              epsabs=tolerance, epsrel=tolerance, limit=200)
            inner_errors.append(err)
            return value

        value, err = quad(outer, 0, 1, epsabs=tolerance, epsrel=tolerance, limit=200)
        err += max(inner_errors, default=0)
    return mp.mpf(value)*jacobian, mp.mpf(err)*jacobian
