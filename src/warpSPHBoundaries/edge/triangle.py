"""Conditioned CPU triangle quadrature for small/ill-conditioned elements.

The original triangle is integrated, without a geometry model. Integration is
in barycentric coordinates with analytic line/circle breakpoints, a Gauss rule
across the short edge and adaptive Gauss--Kronrod along the long direction.
Error estimates combine the outer rule estimate with the maximum sampled
inner-order difference; they are empirical, not certified bounds.
The analytic edge route remains available in np_fem.weights / np2d.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad_vec

from . import fem
from .np_fem import kernel_ubasis
from .kernels import kernel as get_kernel


@dataclass
class TriangleIntegral:
    moments: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    estimate: np.ndarray
    converged: bool
    evaluations: int


@lru_cache(None)
def _rule(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return (x.astype(np.longdouble)+1)/2, w.astype(np.longdouble)/2


def _circle_cuts(a, b, radii):
    """Segment parameters at circles and closest approach; longdouble geometry."""
    aa = np.dot(b, b)
    cuts = [0., 1.]
    if aa == 0:
        return cuts
    t = -np.dot(a, b)/aa
    if 0 < t < 1:
        cuts.append(float(t))
    closest = a+t*b
    rr = np.dot(closest, closest)
    for radius in radii:
        delta = (radius*radius-rr)/aa
        if delta >= 0:
            root = np.sqrt(delta)
            for q in (t-root, t+root):
                if 0 < q < 1:
                    cuts.append(float(q))
    return sorted(set(cuts))


def integrate(vertices, point, kernel, p=2, h=1., order=32, tolerance=2e-12):
    """All twelve atlas moments plus P_p nodal weights and kernel gradients.

    Moments: m00,m10,m01,m20,m11,m02,Gx,Gy,g1xx,g1xy,g1yx,g1yy.
    y = source - query; G = integral N_i grad_query W. Scalar moments have
    physical h**degree scaling; G has 1/h and g1 has unit scaling. Input
    float coordinates are treated as the exact intended geometry. Degenerate
    triangles return zero. ``converged`` covers the adaptive outer rule only;
    use independent/refinement checks as well as the returned estimate.
    """
    if not np.isfinite(h) or h <= 0 or order < 2 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('h, tolerance must be positive and order >= 2')
    v = np.asarray(vertices, dtype=np.longdouble)
    x = np.asarray(point, dtype=np.longdouble)
    if v.shape != (3, 2) or x.shape != (2,) or not np.isfinite(v).all() or not np.isfinite(x).all():
        raise ValueError('finite triangle (3,2) and query (2,) required')
    # Form edge differences before translating by the potentially distant query.
    # Use the shortest edge as the cross-section, preserving thin geometry.
    lengths = [np.dot(v[(i+1)%3]-v[i], v[(i+1)%3]-v[i]) for i in range(3)]
    i = int(np.argmin(lengths))
    perm = [i, (i+2)%3, (i+1)%3]
    a = (v[perm[0]]-x)/h
    b = (v[perm[1]]-v[perm[0]])/h
    c = (v[perm[2]]-v[perm[0]])/h
    area2 = abs(b[0]*c[1]-b[1]*c[0])
    _, polys = fem.basis(p)
    n = len(polys)
    if area2 == 0:
        return TriangleIntegral(np.zeros(12), np.zeros(n), np.zeros((n, 2)), np.zeros(12), True, 0)
    profiles = kernel_ubasis(kernel)
    radii = [np.longdouble(R.numerator)/R.denominator for R in profiles]
    coeffs = [(R, [(k, np.longdouble(co.numerator)/co.denominator) for k, co in d.items()])
              for R, d in zip(radii, profiles.values())]
    pi = np.arccos(np.longdouble(-1))
    _, origin_hi, origin_coeffs = get_kernel(kernel).pieces[0]
    origin = [np.longdouble(co.numerator)/co.denominator for co in origin_coeffs]
    rel = (v-x)/h
    extent = np.maximum(np.max(abs(rel), axis=0), np.finfo(float).tiny)
    sx, sy = extent
    scales = np.asarray([1, sx, sy, sx*sx, sx*sy, sy*sy, sx, sy,
                         sx*sx, sx*sy, sy*sx, sy*sy]+[1]*n+[s for _ in range(n) for s in (sx, sy)], dtype=np.longdouble)

    def line(u, count):
        roots, weights = _rule(count)
        start, direction = a+u*b, (1-u)*c
        cuts = _circle_cuts(start, direction, radii)
        ts = np.concatenate([lo+(hi-lo)*roots for lo, hi in zip(cuts[:-1], cuts[1:])])
        ws = np.concatenate([(hi-lo)*weights for lo, hi in zip(cuts[:-1], cuts[1:])])*(1-u)
        y = start+ts[:, None]*direction
        r = np.hypot(y[:, 0], y[:, 1])
        W, dw = np.zeros_like(r), np.zeros_like(r)
        for R, terms in coeffs:
            t = np.maximum(R-r, 0)
            for k, co in terms:
                W += np.where(r < R, co*t**k, 0)
                if k:
                    dw -= np.where(r < R, k*co*t**(k-1), 0)
        W /= pi
        radial = -dw/pi/np.maximum(r, np.finfo(np.longdouble).tiny)
        # Factored W' still subtracts O(1) terms near zero. The exact
        # origin polynomial has c1=0, so W'/r is a regular polynomial.
        small = r < min(.125, float(origin_hi))
        if origin[1] == 0 and small.any():
            radial_origin = np.zeros_like(r)
            for k in range(len(origin)-1, 1, -1):
                radial_origin = radial_origin*r-k*origin[k]
            radial = np.where(small, radial_origin/pi, radial)
        g = radial[:, None]*y
        xx, yy = y.T
        columns = [W, xx*W, yy*W, xx*xx*W, xx*yy*W, yy*yy*W, g[:, 0], g[:, 1],
                   xx*g[:, 0], xx*g[:, 1], yy*g[:, 0], yy*g[:, 1]]
        lam = np.zeros((len(ts), 3), dtype=np.longdouble)
        lam[:, perm[0]], lam[:, perm[1]], lam[:, perm[2]] = (1-u)*(1-ts), u, (1-u)*ts
        basis = [sum(np.longdouble(float(co))*lam[:, 0]**i*lam[:, 1]**j*lam[:, 2]**k
                     for (i, j, k), co in poly.items()) for poly in polys]
        columns += [ni*W for ni in basis]
        columns += [ni*g[:, j] for ni in basis for j in range(2)]
        return np.asarray((np.stack(columns, axis=1)*ws[:, None]).sum(0)/scales, dtype=float)

    inner_max = np.zeros(len(scales))

    def integrand(u):
        fine, coarse = line(u, order), line(u, max(2, order//2))
        # Keep the noisy absolute refinement difference out of the adaptive
        # integrand. Integrating it can trigger roundoff failure in the outer
        # rule even after the physical integral has converged.
        np.maximum(inner_max, abs(fine-coarse), out=inner_max)
        return fine

    # Outer breakpoints: both long edges crossing profile circles, and circle
    # tangencies to short-edge-parallel sections. Include the kernel origin.
    cuts = _circle_cuts(a, b, radii)+_circle_cuts(a+c, b-c, radii)
    cross = lambda q, z: q[0]*z[1]-q[1]*z[0]
    bc = cross(b, c)
    if bc:
        for R in [np.longdouble(0)]+radii:
            for sign in (-1, 1):
                u = (sign*R*np.hypot(*c)-cross(a, c))/bc
                if 0 < u < 1:
                    cuts.append(float(u))
    val, error, info = quad_vec(integrand, 0, 1, epsabs=tolerance, epsrel=tolerance,
                                norm='max', points=sorted(set(cuts)), full_output=True, limit=2000)
    values = np.asarray(val, dtype=np.longdouble)*scales*area2
    estimate = (np.asarray(inner_max[:12], dtype=np.longdouble)+error)*scales[:12]*area2
    physical = np.asarray([1, h, h, h*h, h*h, h*h, 1/h, 1/h, 1, 1, 1, 1])
    return TriangleIntegral(np.asarray(values[:12]*physical, dtype=float),
                            np.asarray(values[12:12+n], dtype=float),
                            np.asarray(values[12+n:].reshape(n, 2)/h, dtype=float),
                            np.asarray(estimate*abs(physical), dtype=float), bool(info.success), info.neval)


def difficult_mask(vertices, points, h=1.):
    """Conservative CPU fallback domain; not an accuracy certificate."""
    v = np.asarray(vertices, dtype=np.longdouble)
    e = v[:, [1, 2, 0]]-v
    longest2 = (e*e).sum(-1).max(1)
    area2 = abs(e[:, 0, 0]*(-e[:, 2, 1])-e[:, 0, 1]*(-e[:, 2, 0]))
    return np.asarray((longest2 <= (np.asarray(h)*.02)**2) | (area2 <= 1e-3*longest2))
