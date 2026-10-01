"""Stage-1 (mpmath) reference implementation of the 2D edge reductions.

Same algorithm the production backends will use:
   per edge: exact orientation signs -> chord clipping against each support radius
   -> elementary primitives (primitives.py) -> indicator term.

Public API (stage-independent signatures; `kernel` is a name, `verts` a polygon,
`x` the evaluation point, `h` the support radius, all geometry exact rationals
or anything `mpq.to_frac` accepts):

    value(verts, x, kernel, h)                     int_T W dA
    gradient(verts, x, kernel, h)                  grad_x int_T W dA            (2-tuple)
    moment(verts, x, kernel, alpha, h, method)     m_alpha = int_T y^alpha W dA (y = x' - x)
    moment_gradient(verts, x, kernel, alpha, h)    grad_x m_alpha               (2-tuple)

Scaling with the support radius:  value h^0, gradient h^-1, m_alpha h^k, grad m_alpha h^(k-1).

Moments (a) use the compact-potential recursion; (b) is the far-field form kept as a
cross-check (it needs z != 0 on every edge with a non-empty chord).
"""
from math import comb

import mpmath as mp

from . import geometry as G
from . import primitives as pr
from .kernels import kernel as get_kernel
from .mpq import mpq

GUARD = 20          # extra decimal digits carried internally (short chords, cancellation)


def _df(n):
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def omega_int(alpha):
    """oint_{S^1} w^alpha dtheta (mp)."""
    a, b = alpha
    if a % 2 or b % 2:
        return mp.mpf(0)
    return 2 * mp.pi * _df(a - 1) * _df(b - 1) / _df(a + b)


def ypoly(e: G.Edge, beta):
    """coefficients c_j (j = 0..|beta|) of  y1^beta1 y2^beta2  on the edge line, y = z n + s t."""
    a, b = beta
    c1 = [e.z * e.n[0], e.t[0]]
    c2 = [e.z * e.n[1], e.t[1]]
    out = [mp.mpf(1)]
    for _ in range(a):
        out = _pmul(out, c1)
    for _ in range(b):
        out = _pmul(out, c2)
    return out


def _pmul(a, b):
    out = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _Sdiff(j, m, lo, hi, z):
    return pr.S_all(j, m, hi, z) - pr.S_all(j, m, lo, z)


# ------------------------------------------------------------------ blocks
def block_value(P: G.Prepared, n, R):
    """int_T r^n 1[r <= R] dA  (units of h)."""
    R = mpq(R)
    tot = P.indicator * 2 * mp.pi * R ** (n + 2) / (n + 2)
    for e in P.edges:
        ch = G.chord(e, R)
        if ch is None:
            continue
        lo, hi = ch
        Ih, Il = pr.I_all(n, hi, e.z)[n], pr.I_all(n, lo, e.z)[n]
        tot += (e.z * (Ih - Il) - R ** (n + 2) * pr.dangle(e.z, lo, hi)) / (n + 2)
    return tot


def block_grad(P: G.Prepared, n, R):
    """grad_x int_T r^n 1[r <= R] dA = -sum_e n_e [I_n]_chord."""
    R = mpq(R)
    gx = gy = mp.mpf(0)
    for e in P.edges:
        ch = G.chord(e, R)
        if ch is None:
            continue
        lo, hi = ch
        d = pr.I_all(n, hi, e.z)[n] - pr.I_all(n, lo, e.z)[n]
        gx -= e.n[0] * d
        gy -= e.n[1] * d
    return gx, gy


def block_moment(P: G.Prepared, alpha, n, R, _cache=None):
    """int_T y^alpha r^n 1[r <= R] dA via the compact-potential recursion (a)."""
    if _cache is None:
        _cache = {}
    key = (alpha, n, R)
    if key in _cache:
        return _cache[key]
    a, b = alpha
    if a + b == 0:
        val = block_value(P, n, R)
        _cache[key] = val
        return val
    Rm = mpq(R)
    i = 0 if a > 0 else 1
    beta = (a - 1, b) if i == 0 else (a, b - 1)
    n2 = n + 2
    t1 = mp.mpf(0)
    for e in P.edges:
        ni = e.n[i]
        if ni == 0:
            continue
        ch = G.chord(e, Rm)
        if ch is None:
            continue
        lo, hi = ch
        c = ypoly(e, beta)
        acc = mp.mpf(0)
        for j, cj in enumerate(c):
            if cj == 0:
                continue
            acc += cj * (_Sdiff(j, n2, lo, hi, e.z) - Rm ** n2 * _Sdiff(j, 0, lo, hi, e.z))
        t1 += ni * acc / n2
    bi = beta[i]
    t2 = mp.mpf(0)
    if bi >= 1:
        bm = (beta[0] - (1 if i == 0 else 0), beta[1] - (1 if i == 1 else 0))
        t2 = bi * (block_moment(P, bm, n2, R, _cache) - Rm ** n2 * block_moment(P, bm, 0, R, _cache)) / n2
    val = t1 - t2
    _cache[key] = val
    return val


def block_moment_b(P: G.Prepared, alpha, n, R):
    """Far-field form (b); requires z != 0 on every edge with a non-empty chord."""
    a, b = alpha
    k = a + b
    e_ = n + 2 + k
    Rm = mpq(R)
    tot = P.indicator * omega_int(alpha) * Rm ** e_ / e_
    for e in P.edges:
        ch = G.chord(e, Rm)
        if ch is None:
            continue
        if e.z == 0:
            raise ZeroDivisionError("far-field form (b) is undefined for x on an edge line (z = 0)")
        lo, hi = ch
        c = ypoly(e, alpha)
        acc = mp.mpf(0)
        for j, cj in enumerate(c):
            if cj == 0:
                continue
            f1 = pr.Sg(j, n, hi, e.z) - pr.Sg(j, n, lo, e.z)
            f2 = pr.Sg(j, -2 - k, hi, e.z) - pr.Sg(j, -2 - k, lo, e.z)
            acc += cj * (f1 - Rm ** e_ * f2)
        tot += e.z * acc / e_
    return tot


def block_moment_grad(P: G.Prepared, alpha, n, R):
    """grad_x int_T y^alpha r^n 1[r<=R] dA = -sum_e n_e int_chord y^alpha r^n ds."""
    Rm = mpq(R)
    gx = gy = mp.mpf(0)
    for e in P.edges:
        ch = G.chord(e, Rm)
        if ch is None:
            continue
        lo, hi = ch
        c = ypoly(e, alpha)
        acc = mp.mpf(0)
        for j, cj in enumerate(c):
            if cj != 0:
                acc += cj * _Sdiff(j, n, lo, hi, e.z)
        gx -= e.n[0] * acc
        gy -= e.n[1] * acc
    return gx, gy


# --------------------------------------------------------------- kernel level
def _sum_blocks(kern, fn):
    """sum_j sum_n (c_n / pi) fn(n, R_j); fn may return a number or a tuple."""
    tot = None
    for blk in kern.blocks:
        for n, c in enumerate(blk.coeffs):
            if c == 0:
                continue
            v = fn(n, blk.R)
            w = mpq(c) / mp.pi
            v = tuple(w * x for x in v) if isinstance(v, tuple) else w * v
            if tot is None:
                tot = v
            elif isinstance(v, tuple):
                tot = tuple(a_ + b_ for a_, b_ in zip(tot, v))
            else:
                tot = tot + v
    return tot


def _prep(verts, x, h, prepared):
    return prepared if prepared is not None else G.prepare(verts, x, h)


def value(verts, x, kernel, h=1, dps=40, prepared=None):
    kern = get_kernel(kernel)
    with mp.workdps(dps + GUARD):
        P = _prep(verts, x, h, prepared)
        return +_sum_blocks(kern, lambda n, R: block_value(P, n, R))


def gradient(verts, x, kernel, h=1, dps=40, prepared=None):
    kern = get_kernel(kernel)
    with mp.workdps(dps + GUARD):
        P = _prep(verts, x, h, prepared)
        g = _sum_blocks(kern, lambda n, R: block_grad(P, n, R))
        hh = mpq(P.h)
        return tuple(+(c / hh) for c in g)


def moment(verts, x, kernel, alpha, h=1, dps=40, method="a", prepared=None):
    kern = get_kernel(kernel)
    alpha = (int(alpha[0]), int(alpha[1]))
    with mp.workdps(dps + GUARD):
        P = _prep(verts, x, h, prepared)
        cache = {}
        if method == "a":
            m = _sum_blocks(kern, lambda n, R: block_moment(P, alpha, n, R, cache))
        elif method == "b":
            m = _sum_blocks(kern, lambda n, R: block_moment_b(P, alpha, n, R))
        else:
            raise ValueError(method)
        return +(m * mpq(P.h) ** (alpha[0] + alpha[1]))


def moment_gradient(verts, x, kernel, alpha, h=1, dps=40, prepared=None):
    kern = get_kernel(kernel)
    alpha = (int(alpha[0]), int(alpha[1]))
    with mp.workdps(dps + GUARD):
        P = _prep(verts, x, h, prepared)
        g = _sum_blocks(kern, lambda n, R: block_moment_grad(P, alpha, n, R))
        hh = mpq(P.h)
        return tuple(+(c * hh ** (alpha[0] + alpha[1] - 1)) for c in g)
