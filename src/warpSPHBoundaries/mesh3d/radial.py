"""3D axisymmetric moment families and local curvature expansion.

All outputs use unit support, y=source-query, kernel-only query derivatives.
The radial quadrature has sufficient degree for the split sphere/plane
polynomials. The curvature model expands the solid's quadratic graph to order
2; it is a local geometric approximation, not an exact curved solid.
"""
from math import comb, factorial

import numpy as np

from ..curvbound.kernels import KERNELS
from ..edge.kernels import KERNELS as BLOCKS


# Invariants: mass, axial first, transverse/axial second, axial kernel gradient,
# transverse/axial first gradient, transverse/axial kernel Hessian.
INVARIANTS = ('m0', 'm1', 'm2t', 'm2n', 'g0', 'g1t', 'g1n', 'ht', 'hn')


def kernel_jet(q, name):
    """W, W', W'', W''' for the normalized unit-support 3D kernel."""
    q = np.asarray(q, float)
    k = KERNELS[name]
    out = []
    for order in range(4):
        if name == 'cubic':
            v = factorial(3)/factorial(3-order)*(-1)**order*(
                np.maximum(1-q, 0)**(3-order)
                - 4*np.where(q < .5, (.5-q)**(3-order), 0))
        else:
            v = np.zeros_like(q)
            for j in range(order+1):
                poly = sum(float(c)*factorial(p)/factorial(p-order+j)*q**(p-order+j)
                           for p, c in k.poly if p >= order-j)
                v += comb(order, j)*(-1)**j*factorial(k.n)/factorial(k.n-j)*np.maximum(1-q, 0)**(k.n-j)*poly
        out.append(np.where((q >= 0) & (q < 1), v*float(k.c3_pi)/np.pi, 0.))
    return out


def radial_jet(q, name):
    """W, L=W'/r, Q=L'/r, R=Q'/r, with regular origin branches."""
    w, d1, d2, d3 = kernel_jet(q, name)
    q = np.asarray(q, float)
    safe = np.where(q > 1e-12, q, 1.)
    L = np.where(q > 1e-12, d1/safe, d2)
    Q = np.where(q > 1e-12, (d2-L)/safe**2, 0.)
    R = np.where(q > 1e-12, (d3-3*d2/safe+3*d1/safe**2)/safe**3, 0.)
    # Differentiate the assembled origin polynomial algebraically before
    # division. This avoids subtracting nearly equal W'' and W'/r values.
    origin = (q >= 0) & (q < .1)
    if np.any(origin):
        shape = KERNELS[name]
        coeffs = [float(c*shape.c3_pi/shape.c2_pi)/np.pi for c in BLOCKS[name].pieces[0][2]]
        qs = np.where(q > 1e-12, q, 1.)
        l = sum(p*c*q**(p-2) for p,c in enumerate(coeffs) if p >= 2 and c)
        qq = sum(p*(p-2)*c*qs**(p-4) for p,c in enumerate(coeffs) if p >= 3 and c)
        rr = sum(p*(p-2)*(p-4)*c*qs**(p-6) for p,c in enumerate(coeffs) if p >= 3 and c)
        L = np.where(origin,l,L)
        Q = np.where(origin,np.where(q > 1e-12,qq,0.),Q)
        R = np.where(origin,np.where(q > 1e-12,rr,0.),R)
    return w, L, Q, R


def _shell(q, name, cap):
    w, L, Q, _ = radial_jet(q, name)
    c = np.clip(cap, -1, 1)
    A0 = 2*np.pi*(c+1)
    A1 = np.pi*(c*c-1)
    An = 2*np.pi*(c**3+1)/3
    At = np.pi*(c-c**3/3+2/3)
    base = q*q*L*A0
    return np.stack([q*q*w*A0, q**3*w*A1, q**4*w*At, q**4*w*An,
                     -q**3*L*A1, -q**4*L*At, -q**4*L*An,
                     base+q**4*Q*At, base+q**4*Q*An], axis=-1)


def _curvature_integrand(q, d, kappa, name, order=2):
    """Order-1/2 graph displacement for nine axisymmetric invariants (umbilic)."""
    w, L, Q, R = radial_jet(q, name)
    z = -d
    rho2 = np.maximum(q*q-d*d, 0)

    def correction(tangent_degree, normal_degree, f, fp_over_r):
        angular = 2*np.pi if tangent_degree == 0 else np.pi
        p = normal_degree
        first = -.5*kappa*z**p*rho2**((tangent_degree+2)//2)*f
        derivative = (p*z**(p-1)*f if p else 0)+z**(p+1)*fp_over_r
        second = kappa*kappa/8*rho2**((tangent_degree+4)//2)*derivative
        return angular*q*(first+(second if order >= 2 else 0))

    c0 = correction(0, 0, w, L)
    c1 = correction(0, 1, w, L)
    c2t, c2n = correction(2, 0, w, L), correction(0, 2, w, L)
    g0 = -correction(0, 1, L, Q)
    g1t, g1n = -correction(2, 0, L, Q), -correction(0, 2, L, Q)
    base = correction(0, 0, L, Q)
    ht, hn = base+correction(2, 0, Q, R), base+correction(0, 2, Q, R)
    return np.stack([c0, c1, c2t, c2n, g0, g1t, g1n, ht, hn], axis=-1)


def axisymmetric(distance, name, radius=None, curvature=None, curvature_order=2):
    """Nine invariants, shape (N,9). distance is signed outward plane distance.

    radius selects the exact sphere cap geometry; curvature selects an umbilic
    order-two graph correction to a half-space. All lengths are in support units.
    """
    d = np.asarray(distance, float).ravel()
    out = np.zeros((len(d), 9))
    nodes, weights = np.polynomial.legendre.leggauss(16)
    knots = (0., .5, 1.) if name == 'cubic' else (0., 1.)
    for start in range(0, len(d), 2048):
        end = min(start+2048, len(d))
        dd = d[start:end, None]
        whole_end = np.clip(-dd, 0, 1)
        lower = np.abs(dd)
        upper = np.ones_like(dd) if radius is None else np.minimum(1., 2*radius+dd)
        for lo, hi, full in [(np.zeros_like(dd), whole_end, True), (lower, upper, False)]:
            for a, b in zip(knots[:-1], knots[1:]):
                l, r = np.maximum(lo, a), np.minimum(hi, b)
                half = np.maximum(r-l, 0)/2
                q = (l+r)/2+half*nodes
                safeq = np.maximum(q, 1e-15)
                if full:
                    cap = np.ones_like(q)
                elif radius is None:
                    cap = -dd/safeq
                else:
                    D = radius+dd
                    safeD = np.where(D > 0, D, 1.)
                    cap = (radius*radius-D*D-q*q)/(2*safeD*safeq)
                out[start:end] += np.sum(half[..., None]*weights[None, :, None]*_shell(q, name, cap), axis=1)
        if curvature is not None and curvature_order >= 1:
            for a, b in zip(knots[:-1], knots[1:]):
                l, r = np.maximum(lower, a), np.full_like(lower, b)
                half = np.maximum(r-l, 0)/2
                q = (l+r)/2+half*nodes
                corr = _curvature_integrand(q, dd, curvature, name, curvature_order)
                out[start:end] += np.sum(half[..., None]*weights[None, :, None]*corr, axis=1)
    return out


def select(invariants, normal, field):
    """Select a normalized scalar channel from the nine axial invariants."""
    v = np.asarray(invariants)
    n = np.asarray(normal)
    if field == 'm00':
        return v[:, 0]
    if field == 'grad_norm':
        return abs(v[:, 4])
    if field == 'laplacian':
        return 2*v[:, 7]+v[:, 8]
    if field in ('Gx', 'Gy', 'Gz'):
        return v[:, 4]*n[:, 'xyz'.index(field[1])]
    if field.startswith('m'):
        alpha = tuple(int(x) for x in field[1:])
        ids = [i for i, count in enumerate(alpha) for _ in range(count)]
        if len(ids) == 1:
            return v[:, 1]*n[:, ids[0]]
        i, j = ids
        return (v[:, 2] if i == j else 0)+(v[:, 3]-v[:, 2])*n[:, i]*n[:, j]
    i, j = ['xyz'.index(x) for x in field[-2:]]
    t, axial = (5, 6) if field.startswith('g1') else (7, 8)
    return (v[:, t] if i == j else 0)+(v[:, axial]-v[:, t])*n[:, i]*n[:, j]
