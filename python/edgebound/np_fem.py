"""Stage 2/3 (numpy, batched) FEM nodal weights.  Same algorithm as `fem.py`; dtype / stable options pass through to np2d.

    w, G = weights(verts, x, kernel, p, h=1, dtype=np.float64, stable=None, grad=True)

verts (N, 3, 2), x (N, 2), h scalar or (N,).  Returns w (N, n_nodes) and G (N, n_nodes, 2) (physical units:
int_T A W = sum A_i w_i, int_T A grad_x W = sum A_i G_i, G ~ 1/h).

Work is done in units of h (rel = (v - x)/h): B' = B h^|alpha| and m' = m / h^|alpha| so the products are unchanged and
no powers of h appear.  The expansion coefficients B are generated from the EXACT barycentric basis (`fem.basis`) by
batched polynomial composition of lambda_j(x + y) = lambda_j(x) + grad(lambda_j).y, with lambda_j(x) and grad(lambda_j)
computed from the x-relative vertex coordinates (well conditioned for tiny elements).
"""
import numpy as np

from . import fem, np2d


def _lambda_polys(rel):
    """per-vertex affine functions in the y-variables: lam_j(x + y) = l_j + gx_j y1 + gy_j y2   (x at origin of rel)."""
    p0, p1, p2 = rel[:, 0], rel[:, 1], rel[:, 2]
    cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    D = cr(p1 - p0, p2 - p0)
    polys = []
    for (a, b) in [(p1, p2), (p2, p0), (p0, p1)]:
        l = cr(a, b) / D                                   # lambda_j at the origin (= x)
        gx = (a[:, 1] - b[:, 1]) / D
        gy = (b[:, 0] - a[:, 0]) / D
        polys.append({(0, 0): l, (1, 0): gx, (0, 1): gy})
    return polys


def _pmul(P, Q):
    out = {}
    for (a1, b1), c1 in P.items():
        for (a2, b2), c2 in Q.items():
            k = (a1 + a2, b1 + b2)
            out[k] = out[k] + c1 * c2 if k in out else c1 * c2
    return out


def expansion(rel, p, dtype):
    """batched B[i][alpha] arrays (N,) for the P_p element (units of h)."""
    lam = _lambda_polys(rel)
    cache = {}

    def power(j, k):
        if (j, k) not in cache:
            cache[(j, k)] = {(0, 0): np.ones(len(rel), dtype=dtype)} if k == 0 else _pmul(power(j, k - 1), lam[j])
        return cache[(j, k)]
    _, polys = fem.basis(p)
    B = []
    for P in polys:
        acc = {}
        for (a, b, c), co in P.items():
            term = _pmul(_pmul(power(0, a), power(1, b)), power(2, c))
            cf = dtype(float(co)) if dtype is not np.longdouble else dtype(co.numerator) / dtype(co.denominator)
            for k, v in term.items():
                acc[k] = acc[k] + cf * v if k in acc else cf * v
        B.append(acc)
    return B


def weights(verts, x, kernel, p, h=1, dtype=np.float64, stable=None, grad=False, quad=None):
    dt = np.dtype(dtype).type
    verts = np.asarray(verts, dtype=dt)
    x = np.asarray(x, dtype=dt)
    if verts.ndim == 2:
        verts, x = verts[None], x[None]
    hh = np.asarray(h, dtype=dt)
    hb = hh.reshape(-1, 1, 1) if hh.ndim else hh
    rel = (verts - x[:, None, :]) / hb                           # units of h, x at the origin
    zero = np.zeros((len(rel), 2), dtype=dt)
    # orientation normalisation is done inside np2d; lambda polynomials are orientation independent
    B = expansion(rel, p, dt)
    als = fem.alphas(p)
    m = {al: np2d.moment(rel, zero, kernel, al, 1, dtype=dt, stable=stable, quad=quad) for al in als}
    w = np.stack([sum((Bi[al] * m[al] for al in Bi), 0) for Bi in B], axis=1)
    if not grad:
        return w
    g = {}
    for al in als:
        gm = np2d.grad_moment(rel, zero, kernel, al, 1, dtype=dt, stable=stable, quad=quad)
        g[al] = (gm[:, 0], gm[:, 1])
    G = np.stack([np.stack([sum((Bi[al] * g[al][0] for al in Bi), 0), sum((Bi[al] * g[al][1] for al in Bi), 0)], axis=-1) for Bi in B], axis=1)
    G = G / (hh.reshape(-1, 1, 1) if hh.ndim else hh)
    return w, G


# ======================================================================= far-field hybrid (Gauss on the element)
from fractions import Fraction
from functools import lru_cache
from math import comb

from .kernels import kernel as get_kernel


@lru_cache(maxsize=None)
def kernel_ubasis(kname):
    """EXACT expansion of the kernel blocks in u = R - r:  pi W(r) = sum_blocks sum_k b_k (R - r)^k  [r <= R]
    (Wendland: k >= p only; cubic: single power k = 3 per block).  Returns {R: {k: Fraction}}."""
    out = {}
    for blk in get_kernel(kname).blocks:
        R = Fraction(blk.R)
        d = {}
        for n, c in enumerate(blk.coeffs):
            if c == 0:
                continue
            # r^n = (R - u)^n = sum_k C(n,k) R^(n-k) (-u)^k
            for k in range(n + 1):
                d[k] = d.get(k, 0) + Fraction(c) * comb(n, k) * R ** (n - k) * (-1) ** k
        out[R] = {k: v for k, v in d.items() if v != 0}
    return out


def _kernel_eval(kname, r, dt, derivative=False):
    """W(r) (or W'(r)/r) for an array r, stable u-basis evaluation, dtype dt."""
    one_over_pi = dt(1 / np.pi) if dt is not np.longdouble else np.longdouble(1) / np.longdouble(np.pi)
    tot = 0
    for R, d in kernel_ubasis(kname).items():
        Rf = dt(float(R))
        u = np.maximum(Rf - r, 0) * (r < Rf)
        acc = 0
        for k, b in d.items():
            if derivative:
                if k >= 1:
                    acc = acc - dt(float(b)) * k * u ** (k - 1)
            else:
                acc = acc + dt(float(b)) * u ** k
        tot = tot + acc
    return tot * one_over_pi / (r if derivative else 1)


def gauss_weights(verts, x, kernel, p, h=1, nodes=8, dtype=np.float64, grad=False):
    """Weights by tensor-Gauss (Duffy) quadrature of N_i W (and N_i grad_x W) on the element: no cancellation, converges
    geometrically when the element is far from x relative to its size.  Same output convention as `weights`."""
    dt = np.dtype(dtype).type
    verts = np.asarray(verts, dtype=dt)
    x = np.asarray(x, dtype=dt)
    if verts.ndim == 2:
        verts, x = verts[None], x[None]
    hh = np.asarray(h, dtype=dt)
    hb = hh.reshape(-1, 1, 1) if hh.ndim else hh
    rel = (verts - x[:, None, :]) / hb
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    u = ((xg + 1) / 2).astype(dt)
    wu = (wg / 2).astype(dt)
    p0, p1, p2 = rel[:, 0], rel[:, 1], rel[:, 2]
    area2 = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    _, polys = fem.basis(p)
    N = len(rel)
    w = np.zeros((N, len(polys)), dtype=dt)
    G = np.zeros((N, len(polys), 2), dtype=dt)
    for ui, wi in zip(u, wu):
        for vi, wj in zip(u, wu):
            l1, l2 = ui, vi * (1 - ui)
            l0 = 1 - l1 - l2
            wt = wi * wj * (1 - ui) * area2
            y = l0 * p0 + l1 * p1 + l2 * p2                       # relative to x
            r = np.hypot(y[:, 0], y[:, 1])
            Wv = _kernel_eval(kernel, r, dt)
            if grad:
                dW = _kernel_eval(kernel, r, dt, derivative=True)
            Ni = [sum((dt(float(c)) * l0 ** a * l1 ** b * l2 ** cc for (a, b, cc), c in P.items()), 0) for P in polys]
            for i, n_ in enumerate(Ni):
                w[:, i] += wt * n_ * Wv
                if grad:
                    G[:, i, 0] += -wt * n_ * dW * y[:, 0]
                    G[:, i, 1] += -wt * n_ * dW * y[:, 1]
    if not grad:
        return w
    return w, G / (hh.reshape(-1, 1, 1) if hh.ndim else hh)


def far_mask(verts, x, ratio=1.0):
    """True where the element is 'far': |centroid - x| >= ratio * (longest edge)."""
    v = np.asarray(verts, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    c = v.mean(axis=1)
    rho = np.hypot(c[:, 0] - xx[:, 0], c[:, 1] - xx[:, 1])
    e = np.stack([np.hypot(*(v[:, i] - v[:, (i + 1) % 3]).T) for i in range(3)], axis=1).max(1)
    return rho >= ratio * e


def weights_hybrid(verts, x, kernel, p, h=1, dtype=np.float64, stable=None, grad=False, far_ratio=1.0, nodes=8):
    """Edge reduction near x (exact), Gauss on the element where it is far."""
    verts_ = np.asarray(verts, dtype=dtype)
    x_ = np.asarray(x, dtype=dtype)
    if verts_.ndim == 2:
        verts_, x_ = verts_[None], x_[None]
    far = far_mask(verts_, x_, far_ratio)
    near = weights(verts_, x_, kernel, p, h, dtype, stable, grad)
    if not far.any():
        return near
    hfar = np.asarray(h)[far] if np.ndim(h) else h
    farv = gauss_weights(verts_[far], x_[far], kernel, p, hfar, nodes, dtype, grad)
    if grad:
        w, G = near
        w = w.copy(); G = G.copy()
        w[far], G[far] = farv
        return w, G
    out = near.copy()
    out[far] = farv
    return out
