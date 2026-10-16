"""Accepted-SDF audit helpers, independent curved-domain area integration.

All lengths are in support units. References integrate the actual ellipse or
cubic-graph solid, never a polygonization or an SDF level-set approximation.
Refinement discrepancies are empirical estimates, not rigorous bounds.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from ..edge.kernels import kernel as get_kernel
from .atlas import ALPHAS, CHANNELS
from .distance_only import halfplane_channels
from .specialized_audit import ellipse_distance


def accepted_channel_gate(total, uncertainty, scale, accepted, tolerance):
    """Fail closed for no accepted samples, nonfinite errors or uncertainty."""
    accepted = np.asarray(accepted, dtype=bool)
    if not accepted.any():
        return np.zeros(12, dtype=bool)
    error = np.abs(np.asarray(total)[accepted])
    estimate = np.asarray(uncertainty)[accepted]
    valid = np.isfinite(error) & np.isfinite(estimate) & (estimate >= 0)
    return (valid & ((error+estimate) <= tolerance*np.asarray(scale))).all(0)


@dataclass(frozen=True)
class SmoothCase:
    name: str
    kind: str = 'ellipse'
    a: float = 6.
    b: float = 6.
    angle: float = np.pi/2
    kappa: float = 0.
    cubic: float = 0.
    cavity: bool = False
    rotation: float = 0.

    @property
    def frame(self):
        if self.kind == 'ellipse':
            t = self.angle
            point = np.array([self.a*np.cos(t), self.b*np.sin(t)])
            normal = np.array([np.cos(t)/self.a, np.sin(t)/self.b])
            normal /= np.linalg.norm(normal)
            tangent = np.array([-normal[1], normal[0]])
            q = self.a**2*np.sin(t)**2+self.b**2*np.cos(t)**2
            curvature = self.a*self.b/q**1.5
            variation = -3*self.a*self.b*(self.a**2-self.b**2)*np.sin(t)*np.cos(t)/q**3
            if self.cavity:
                normal, curvature = -normal, -curvature
                variation = -variation
            return point, normal, tangent, curvature, variation
        return np.zeros(2), np.array([0., 1.]), np.array([1., 0.]), self.kappa, -self.cubic

    @property
    def matrix(self):
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return np.array([[c, -s], [s, c]])

    def height(self, x):
        return -self.kappa*x*x/2+self.cubic*x*x*x/6

    def points(self, distances):
        p, n, _, _, _ = self.frame
        return (p+np.asarray(distances)[:, None]*n)@self.matrix.T

    def distance(self, points):
        p = np.asarray(points)@self.matrix
        if self.kind == 'ellipse':
            d = ellipse_distance(p, self.a, self.b)
            return -d if self.cavity else d
        # Every real stationary point is a candidate. A cubic graph gives a
        # coercive degree-six squared distance; its global minimum is among
        # these roots. No local Newton solve chooses a branch for the audit.
        f = np.array([0., 0., -self.kappa/2, self.cubic/6])
        if not self.kappa and not self.cubic:
            return p[:, 1].copy()
        derivative = np.polynomial.polynomial.polyder(f)
        base = np.polynomial.polynomial.polymul(f, derivative)
        base[1] += 1
        degree = np.flatnonzero(base)[-1] if np.any(base) else 1
        result = np.empty(len(p))
        for start in range(0, len(p), 4096):
            v = p[start:start+4096]
            coefficients = np.broadcast_to(base[:degree+1], (len(v), degree+1)).copy()
            coefficients[:, 0] -= v[:, 0]
            coefficients[:, :len(derivative)] -= v[:, 1, None]*derivative
            companion = np.zeros((len(v), degree, degree))
            companion[:, 0, :] = -coefficients[:, :-1][:, ::-1]/coefficients[:, -1, None]
            companion[:, np.arange(1, degree), np.arange(degree-1)] = 1
            roots = np.linalg.eigvals(companion)
            real = np.abs(roots.imag) <= 1e-8*np.maximum(1, np.abs(roots.real))
            x = roots.real
            squared = (x-v[:, 0, None])**2+(self.height(x)-v[:, 1, None])**2
            squared[~real] = np.inf
            length = np.sqrt(squared.min(1))
            result[start:start+len(v)] = np.where(v[:, 1] >= self.height(v[:, 0]), length, -length)
        return result

    def intervals(self, x, px, py):
        """Actual solid's vertical interval(s), clipped by the support disk."""
        extent = np.sqrt(max(0., 1-(x-px)**2))
        lo, hi = py-extent, py+extent
        if self.kind == 'graph':
            return [(lo, min(hi, self.height(x)))]
        if abs(x) >= self.a:
            return [(lo, hi)] if self.cavity else []
        wall = self.b*np.sqrt(max(0., 1-(x/self.a)**2))
        if self.cavity:
            return [(lo, min(hi, -wall)), (max(lo, wall), hi)]
        return [(max(lo, -wall), min(hi, wall))]

    def breaks(self, point, radii):
        """Boundary/circle intersections and kernel-knot projections."""
        px, py = point
        cuts = [px-1, px, px+1]
        for r in radii:
            cuts += [px-r, px+r]
            if self.kind == 'graph':
                f = np.array([-py, 0., -self.kappa/2, self.cubic/6])
                poly = np.pad(np.polynomial.polynomial.polymul(f, f), (0, 2))
                poly[:3] += np.array([px*px-r*r, -2*px, 1])
                cuts += [z.real for z in np.polynomial.polynomial.polyroots(poly)
                         if abs(z.imag) < 1e-8 and abs(z.real-px) < 1]
            else:
                # Smooth closed curve: bracket roots with a mesh fine enough
                # to resolve the smallest R/h used by this study. No integral
                # is evaluated on that mesh.
                t = np.linspace(0, 2*np.pi, 4097)
                def residual(t):
                    return (self.a*np.cos(t)-px)**2+(self.b*np.sin(t)-py)**2-r*r
                v = residual(t)
                for j in np.flatnonzero(v[:-1]*v[1:] < 0):
                    root = brentq(residual, t[j], t[j+1], xtol=5e-15)
                    cuts.append(self.a*np.cos(root))
        if self.kind == 'ellipse':
            cuts += [-self.a, self.a]
        return np.unique(np.clip(cuts, px-1, px+1))


def smooth_cases():
    return (
        *(SmoothCase(f'circle-R{r}', a=r, b=r) for r in (3., 6., 12., 24.)),
        SmoothCase('cavity-R6', a=6, b=6, cavity=True),
        SmoothCase('circle-R6-rotated', a=6, b=6, rotation=.37),
        SmoothCase('ellipse-top', a=8, b=4),
        SmoothCase('ellipse-oblique', a=8, b=4, angle=.63),
        SmoothCase('ellipse-oblique-rotated', a=8, b=4, angle=.63, rotation=.37),
        *(SmoothCase(f'inflection-b{b:g}', kind='graph', cubic=b) for b in (.03, .12, .3)),
        SmoothCase('variable-curvature', kind='graph', kappa=.15, cubic=.15),
    )


@lru_cache(None)
def profiles(kernel):
    """Factored exact-rational W, -W'/r and its radial derivative/r."""
    r = sp.Symbol('r')
    pieces = []
    for lo, hi, coeff in get_kernel(kernel).pieces:
        w = sum(sp.Rational(c.numerator, c.denominator)*r**k for k, c in enumerate(coeff))
        d = sp.cancel(-sp.diff(w, r)/r)
        expressions = (w, d, sp.cancel(sp.diff(d, r)/r))
        pieces.append((float(lo), float(hi), tuple(sp.lambdify(r, sp.factor(e), 'numpy') for e in expressions)))
    return pieces


def profile_values(radius, kernel):
    radius = np.asarray(radius, dtype=np.longdouble)
    out = np.zeros((3, *radius.shape), dtype=np.longdouble)
    for lo, hi, functions in profiles(kernel):
        mask = (radius >= lo) & (radius < hi)
        rr = radius[mask]
        if len(rr):
            for k, fn in enumerate(functions):
                out[k][mask] = fn(rr)/np.longdouble('3.141592653589793238462643383279502884')
    return out


@lru_cache(None)
def gauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes.astype(np.longdouble), weights.astype(np.longdouble)


def rotate_channels(values, matrix):
    out = np.array(values, copy=True)
    out[..., 1:3] = values[..., 1:3]@matrix.T
    out[..., 6:8] = values[..., 6:8]@matrix.T
    scalar = np.stack((values[..., 3], values[..., 4], values[..., 4], values[..., 5]), -1).reshape(*values.shape[:-1], 2, 2)
    scalar = matrix@scalar@matrix.T
    out[..., 3:6] = scalar.reshape(*values.shape[:-1], 4)[..., [0, 1, 3]]
    gradient = matrix@values[..., 8:].reshape(*values.shape[:-1], 2, 2)@matrix.T
    out[..., 8:] = gradient.reshape(*values.shape[:-1], 4)
    return out


def area_reference(case, point, kernel='w4', order=64, tolerance=2e-12):
    """Direct Cartesian integration, analytic vertical support/knot cuts.

    Adaptive outer Gauss-Kronrod and an inner Gauss rule share neither boundary
    flux formulas nor Tier3 tables. Returns all twelve moments plus the outer
    empirical estimate. Compare inner orders separately.
    """
    px, py = np.asarray(point)@case.matrix
    radii = sorted(set(float(b.R) for b in get_kernel(kernel).blocks))
    nodes, weights = gauss(order)

    def vertical(theta):
        x = px+np.sin(theta)
        u = np.longdouble(x-px)
        total = np.zeros(12, dtype=np.longdouble)
        for bottom, top in case.intervals(x, px, py):
            if top <= bottom:
                continue
            cuts = [bottom-py, top-py, 0.]
            for r in radii:
                if abs(u) < r:
                    v = np.sqrt(np.longdouble(r)**2-u*u)
                    cuts += [-v, v]
            cuts = np.unique(np.clip(cuts, bottom-py, top-py))
            lo, hi = cuts[:-1, None], cuts[1:, None]
            v = (lo+hi)/2+(hi-lo)/2*nodes
            w = (hi-lo)/2*weights
            W, D, _ = profile_values(np.sqrt(u*u+v*v), kernel)
            integrands = np.array([W, u*W, v*W, u*u*W, u*v*W, v*v*W,
                                   u*D, v*D, u*u*D, u*v*D, u*v*D, v*v*D])
            total += np.sum(integrands*w, axis=(1, 2))
        return total*np.cos(theta)

    xcuts = case.breaks((px, py), radii)
    cuts = np.arcsin(np.clip(xcuts-px, -1, 1))
    total = np.zeros(12, dtype=np.longdouble)
    error = 0.
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        val, err, info = quad_vec(vertical, lo, hi, epsabs=tolerance/len(cuts), epsrel=tolerance,
                                 quadrature='gk21', full_output=True)
        if not info.success:
            raise RuntimeError(f'curved reference did not converge: {case.name}, point={point}, '
                               f'interval={lo, hi}, estimate={err}: {info.message}')
        total += val
        error += err
    return rotate_channels(total, case.matrix), error


def shape_coefficients(distance, kernel, order=96):
    """Independent 1D shape derivatives for F1,F2 and F0',F1',F2'.

    At fixed closest point the circle graph is z=-k*t²/2+O(k³).
    F1=-int t² W/2; F2=d*int t⁴*(-W'/r)/8. No tubular
    moment recursion, interpolation table or numerical curvature is used.
    """
    d = np.longdouble(distance)
    if abs(d) >= 1:
        return np.zeros(5, dtype=np.longdouble)
    cuts = [np.longdouble(0)]
    for b in get_kernel(kernel).blocks:
        if float(b.R) > abs(d):
            cuts.append(np.sqrt(np.longdouble(float(b.R))**2-d*d))
    cuts = np.unique(cuts)
    nodes, weights = gauss(order)
    result = np.zeros(5, dtype=np.longdouble)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        t = (lo+hi)/2+(hi-lo)/2*nodes
        w = (hi-lo)*weights  # both tangential half axes
        W, D, derivative = profile_values(np.sqrt(t*t+d*d), kernel)
        result += np.sum(np.array([-t*t*W/2, d*t**4*D/8, -W,
                                    d*t*t*D/2, t**4*(D+d*d*derivative)/8])*w, axis=1)
    return result


def local_model_reference(distance, normal, curvature, kernel='w4', order=96):
    """Extended-precision same model as production, all nine exposed channels."""
    d, n, k = map(lambda x: np.asarray(x, dtype=np.longdouble), (distance, normal, curvature))
    out = halfplane_channels(d, n, kernel, np.longdouble)
    out[:, 3:6] = np.nan  # production SDF does not expose these
    for i in range(len(d)):
        f1, f2, df0, df1, df2 = shape_coefficients(d[i], kernel, order)
        out[i, 0] += k[i]*f1+k[i]**2*f2
        out[i, 6:8] = (df0+k[i]*df1+k[i]**2*df2)*n[i]
    return out


def boundary_reference_mp(case, point, kernel='w4', dps=45):
    """Independent high-precision boundary oracle, all twelve channels.

    Uses compact radial flux for scalar moments with exact whole-plane
    corrections; G and g1 follow boundary integration by parts. This is an
    audit oracle, not the production polygon edge primitive or an SDF model.
    Points on the boundary are deliberately excluded (no singular limits).
    """
    import mpmath as mp
    with mp.workdps(dps):
        convert = lambda x: mp.mpf(float(x))
        px, py = map(convert, np.asarray(point)@case.matrix)
        pieces = get_kernel(kernel).pieces
        coeffs = [(mp.mpf(lo.numerator)/lo.denominator, mp.mpf(hi.numerator)/hi.denominator,
                   [mp.mpf(c.numerator)/c.denominator for c in coeff]) for lo, hi, coeff in pieces]

        def W(r):
            return next((sum(c*r**j for j, c in enumerate(co)) / mp.pi
                         for lo, hi, co in coeffs if lo <= r < hi), mp.mpf(0))

        def potential(r, degree):
            integral = mp.mpf(0)
            for lo, hi, co in coeffs:
                bottom = max(lo, r)
                if bottom < hi:
                    integral -= sum(c*(hi**(j+degree+2)-bottom**(j+degree+2))/(j+degree+2)
                                    for j, c in enumerate(co))
            return integral/(mp.pi*r**(degree+2))

        if case.kind == 'ellipse':
            a, b = convert(case.a), convert(case.b)
            def curve(t):
                return a*mp.cos(t)-px, b*mp.sin(t)-py, b*mp.cos(t), a*mp.sin(t)
            # n ds=(b cos t,a sin t)dt, orientation into the fluid.
            cuts = list(np.linspace(0, 2*np.pi, 65))
            # Include true intersections so adaptive integration sees even
            # small support arcs at large R/h and every kernel knot.
            scan = np.linspace(0, 2*np.pi, 4097)
            for radius in [float(hi) for _, hi, _ in pieces]:
                fn = lambda t: (case.a*np.cos(t)-float(px))**2+(case.b*np.sin(t)-float(py))**2-radius**2
                v = fn(scan)
                cuts += [brentq(fn, scan[j], scan[j+1], xtol=5e-15) for j in np.flatnonzero(v[:-1]*v[1:] < 0)]
            sign = -1 if case.cavity else 1
            inside = (px/a)**2+(py/b)**2 < 1
            if case.cavity:
                inside = not inside
        else:
            k, b = convert(case.kappa), convert(case.cubic)
            def curve(t):
                return t-px, -k*t*t/2+b*t*t*t/6-py, k*t-b*t*t/2, mp.mpf(1)
            cuts = list(case.breaks((float(px), float(py)), [float(hi) for _, hi, _ in pieces]))
            sign = 1
            inside = py < -k*px*px/2+b*px*px*px/6
        cuts = sorted(set(map(convert, cuts)))

        def integrate(channel):
            def f(t):
                u, v, nx, ny = curve(t)
                r = mp.sqrt(u*u+v*v)
                if r >= 1:
                    return mp.mpf(0)
                if channel < 6:
                    aa, bb = ALPHAS[channel]
                    return sign*u**aa*v**bb*(u*nx+v*ny)*potential(r, aa+bb)
                if channel < 8:
                    return -sign*W(r)*(nx if channel == 6 else ny)
                aa, bb = divmod(channel-8, 2)
                return -sign*(u if aa == 0 else v)*W(r)*(nx if bb == 0 else ny)
            return sum(mp.quad(f, [lo, hi]) for lo, hi in zip(cuts[:-1], cuts[1:]))

        result = [integrate(j) for j in range(12)]
        if inside:
            result[0] += 1
            variance = sum(sum(c*(hi**(j+4)-lo**(j+4))/(j+4) for j, c in enumerate(co))
                           for lo, hi, co in coeffs)
            result[3] += variance
            result[5] += variance
        result[8] += result[0]
        result[11] += result[0]
        value = np.array([np.longdouble(str(v)) for v in result])
        return rotate_channels(value, case.matrix)
