"""Stage 2: vectorised, branch-free numpy implementation of the 2D edge reductions.

Design = EXACT CORE + CHEAP, MEASURED APPROXIMATIONS
----------------------------------------------------
* Everything that depends only on the kernel and on alpha is compiled ONCE, EXACTLY, in
  `Fraction`s (`compile_*`): the truncated-monomial blocks, the compact-potential recursion
  (profile polynomials Phi), the indicator weights (Wendland 1; cubic 8/7 and -1/7) and
  the atan weights.  Only the final rounding to float happens at the end.  No floating
  point cancellation can enter through the kernel algebra.
* The geometry-dependent part is the closed form of `primitives.py` evaluated in numpy
  (`dtype` selectable: float64 by default, longdouble / float32 to study conditioning).
* The orientation predicate and the indicator are taken from the SAME computed z_e
  (sign tests, zero counts), so for any rounding the result is the exact value for a
  slightly perturbed point -> continuous, no spurious jumps, no exact predicate needed.
* Branch-free: chords are clipped with min/max/where, empty chords are zeroed by masks.

Approximation knobs (all optional, measured in `np2d_study.py`):
  dtype   : np.float64 (default), np.longdouble, np.float32
  quad    : None = closed-form primitives; m = Gauss-Legendre with m nodes on the clipped chord
            for the polynomial parts (the atan angle stays closed form: it is cheap and the
            only part that is not smooth when z -> 0)
  unsplit : True (default) = use  z int M(r)/r^2 ds  (no indicator, no atan) for every block whose
            whole polygon lies inside its radius R; mathematically identical, immune to the
            1-(1-eps) cancellation for elements << h.

Geometry: batched convex polygons `verts (N, K, 2)` (triangles K = 3), evaluation points `x (N, 2)`,
support radius `h` (scalar or (N,)).  Non-convex polygons: decompose (the indicator here is the sign test).
"""
from fractions import Fraction
from functools import lru_cache
from math import comb

import mpmath as mp
import numpy as np

from .kernels import kernel as get_kernel

# ============================================================== exact compilation
def _phi(P, R):
    """Phi[f](r) = -int_r^R t f dt for f = sum P_n r^n  (exact, dict n -> Fraction)."""
    out, const = {}, Fraction(0)
    for n, c in P.items():
        out[n + 2] = out.get(n + 2, 0) + c / (n + 2)
        const -= c * Fraction(R) ** (n + 2) / (n + 2)
    out[0] = out.get(0, 0) + const
    return {n: c for n, c in out.items() if c != 0}


def _compile(P, R, alpha, mult, edges, values):
    a, b = alpha
    if a + b == 0:
        d = values.setdefault(R, {})
        for n, c in P.items():
            d[n] = d.get(n, 0) + mult * c
        return
    i = 0 if a > 0 else 1
    beta = (a - 1, b) if i == 0 else (a, b - 1)
    Ph = _phi(P, R)
    d = edges.setdefault((i, beta, R), {})
    for n, c in Ph.items():
        d[n] = d.get(n, 0) + mult * c
    bi = beta[i]
    if bi >= 1:
        bm = (beta[0] - (i == 0), beta[1] - (i == 1))
        _compile(Ph, R, bm, -mult * bi, edges, values)


@lru_cache(maxsize=None)
def compile_moment(kname, alpha):
    """Exact plan for m_alpha of kernel `kname`:
         edges  : {(i, beta, R): {n: c}}   sum_e n_{e,i} int_chord y^beta (sum c_n r^n) ds
         values : {R: {n: c}}              value-type terms  (profile polynomial, radius)
       all pi-free (the common factor 1/pi is applied at evaluation); c exact Fractions."""
    k = get_kernel(kname)
    edges, values = {}, {}
    for blk in k.blocks:
        P = {n: Fraction(c) for n, c in enumerate(blk.coeffs) if c != 0}
        _compile(P, blk.R, tuple(alpha), Fraction(1), edges, values)
    edges = {key: {n: c for n, c in d.items() if c != 0} for key, d in edges.items()}
    values = {R: {n: c for n, c in d.items() if c != 0} for R, d in values.items()}
    return edges, {R: d for R, d in values.items() if d}


@lru_cache(maxsize=None)
def kernel_profile(kname):
    """{R: {n: c}} pi-free blocks of the kernel itself (for gradients / moment gradients)."""
    return {blk.R: {n: Fraction(c) for n, c in enumerate(blk.coeffs) if c != 0} for blk in get_kernel(kname).blocks}


def indicator_weight(P, R):
    """exact 2 * sum P_n R^(n+2)/(n+2)  (times ind; pi-free part is 2 pi m_R / pi)."""
    return 2 * sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())


# ============================================================== number helpers
class _Num:
    def __init__(self, dtype):
        self.dt = np.dtype(dtype)
        self.t = self.dt.type
        with mp.workdps(40):
            self.inv_pi = self._from_str(mp.nstr(1 / mp.pi, 38))
            self.two_pi = self._from_str(mp.nstr(2 * mp.pi, 38))
        self._cache = {}

    def _from_str(self, s):
        return np.array(s, dtype=self.dt)[()] if self.dt == np.dtype(np.longdouble) else self.t(float(s))

    def c(self, f):
        f = Fraction(f)
        key = (f.numerator, f.denominator)
        v = self._cache.get(key)
        if v is None:
            if self.dt == np.dtype(np.longdouble):
                v = self.t(f.numerator) / self.t(f.denominator)
            else:
                v = self.t(float(f))
            self._cache[key] = v
        return v


# ============================================================== geometry
class Geo:
    """Per-call geometry in units of h: edge arrays (N,K), indicator (N,), vertex distances."""

    def __init__(self, verts, x, h, dtype):
        self.num = _Num(dtype)
        dt = self.num.dt
        verts = np.asarray(verts, dtype=dt)
        x = np.asarray(x, dtype=dt)
        h = np.asarray(h, dtype=dt)
        hh = h.reshape(-1, 1, 1) if h.ndim else h
        rel = (verts - x[:, None, :]) / hh
        # normalise orientation to ccw (branch-free)
        a2 = (rel[:, :, 0] * np.roll(rel, -1, 1)[:, :, 1] - rel[:, :, 1] * np.roll(rel, -1, 1)[:, :, 0]).sum(1)
        flip = (a2 < 0)[:, None, None]
        rel = np.where(flip, rel[:, ::-1, :], rel)
        self.N, self.K = rel.shape[0], rel.shape[1]
        self.h = h
        p = rel
        q = np.roll(rel, -1, axis=1)
        d = q - p
        ell = np.hypot(d[..., 0], d[..., 1])
        t0, t1 = d[..., 0] / ell, d[..., 1] / ell
        self.n0, self.n1 = t1, -t0
        self.t0, self.t1 = t0, t1
        # z = cross(p, d)/ell = cross(p, q)/ell: the second form has no p.p cancellation when x is near
        # an endpoint (z, s0, s1 must be mutually consistent for the atan terms near vertices)
        self.z = (p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]) / ell
        self.s0 = (d[..., 0] * p[..., 0] + d[..., 1] * p[..., 1]) / ell
        self.s1 = (d[..., 0] * q[..., 0] + d[..., 1] * q[..., 1]) / ell
        self.vmax2 = (rel ** 2).sum(-1).max(1)               # max vertex distance^2 per element
        # indicator from the SAME z: sign test + zero counts (convex polygons)
        z = self.z
        neg = (z < 0).any(1)
        nz = (z == 0).sum(1)
        pn = np.roll(rel, -1, 1)
        pp = np.roll(rel, 1, 1)
        a_ = pn - rel
        b_ = pp - rel
        cr = a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
        dt_ = a_[..., 0] * b_[..., 0] + a_[..., 1] * b_[..., 1]
        ang = np.arctan2(cr, dt_)
        ang = np.where(ang <= 0, ang + 2 * np.pi, ang)
        zprev = np.roll(z, 1, 1)                             # edge i-1 ends at vertex i, edge i starts there
        at_vertex = ((z == 0) & (zprev == 0))
        vang = (np.where(at_vertex, ang, 0).sum(1)) / self.num.two_pi
        self.ind = np.where(neg, 0, np.where(nz == 0, 1, np.where(nz == 1, 0.5, vang))).astype(dt)
        self._chords = {}

    def chord(self, R):
        """clipped chord for radius R (Fraction): (lo, hi, mask); empty chords have lo = hi = 0."""
        v = self._chords.get(R)
        if v is None:
            Rf = self.num.c(R)
            az = np.abs(self.z)
            L = np.sqrt(np.maximum((Rf - az) * (Rf + az), 0))
            lo = np.maximum(self.s0, -L)
            hi = np.minimum(self.s1, L)
            m = (az < Rf) & (lo < hi)
            lo = np.where(m, lo, 0)
            hi = np.where(m, hi, 0)
            v = (lo, hi, m)
            self._chords[R] = v
        return v


# ============================================================== primitives (numpy)
def _I_all(M, s, z):
    zz = z * z
    r2 = s * s + zz
    r = np.sqrt(r2)
    out = [s]
    if M >= 1:
        safe = np.where(z == 0, 1, np.abs(z))
        tail = np.where(z == 0, 0, zz * np.arcsinh(s / safe))
        out.append((s * r + tail) / 2)
    pw = [np.ones_like(r2)]
    for k in range(1, M // 2 + 2):
        pw.append(pw[-1] * r2)
    for m in range(2, M + 1):
        rp = pw[m // 2] if m % 2 == 0 else r * pw[(m - 1) // 2]
        out.append((s * rp + m * zz * out[m - 2]) / (m + 1))
    return out


def _S(j, m, s, z):
    """int s^j r^m ds (m >= 0), closed form."""
    zz = z * z
    if j % 2 == 0:
        h = j // 2
        I = _I_all(m + 2 * h, s, z)
        return sum(comb(h, i) * (-zz) ** (h - i) * I[m + 2 * i] for i in range(h + 1))
    h = (j - 1) // 2
    r2 = s * s + zz
    r = np.sqrt(r2)
    tot = 0
    for i in range(h + 1):
        mm = m + 2 * i
        rp = (r2 ** ((mm + 2) // 2)) if mm % 2 == 0 else r * r2 ** ((mm + 1) // 2)
        tot = tot + comb(h, i) * (-zz) ** (h - i) * rp / (mm + 2)
    return tot


_GL = {}


def _gl(m, dt):
    key = (m, np.dtype(dt))
    if key not in _GL:
        x, w = np.polynomial.legendre.leggauss(m)
        _GL[key] = (x.astype(dt), w.astype(dt))
    return _GL[key]



def _gauss_panels(j, m, lo, hi, z, nodes, dt):
    """int_lo^hi s^j r^m ds by Gauss-Legendre on 4 panels split at -|z|, 0, |z| (the foot point and the
    complex singularities s = +-i z of r^m for odd m sit at distance |z| from it)."""
    x, w = _gl(nodes, dt)
    az = np.abs(z)
    b = [lo, np.clip(-az, lo, hi), np.clip(0 * az, lo, hi), np.clip(az, lo, hi), hi]
    tot = 0
    for a_, b_ in zip(b[:-1], b[1:]):
        half = (b_ - a_) / 2
        mid = (a_ + b_) / 2
        acc = 0
        for xk, wk in zip(x, w):
            s = mid + half * xk
            acc = acc + wk * s ** j * (s * s + z * z) ** (m / 2)
        tot = tot + acc * half
    return tot


class _Ctx:
    """per-call evaluation context: geometry + options + memo of chord integrals."""

    def __init__(self, geo, quad):
        self.g = geo
        self.quad = quad
        self.memo = {}

    def sdiff(self, j, m, R):
        """(N,K) array: int_chord s^j r^m ds over the chord clipped to radius R."""
        key = (j, m, R)
        v = self.memo.get(key)
        if v is None:
            lo, hi, msk = self.g.chord(R)
            z = self.g.z
            if self.quad is None:
                v = np.where(msk, _S(j, m, hi, z) - _S(j, m, lo, z), 0)
            else:
                v = np.where(msk, _gauss_panels(j, m, lo, hi, z, self.quad, self.g.num.dt), 0)
            self.memo[key] = v
        return v

    def sfull(self, j, m):
        """int over the FULL edge (s0..s1) of s^j r^m ds (unsplit form), key R = None."""
        key = (j, m, None)
        v = self.memo.get(key)
        if v is None:
            z = self.g.z
            if self.quad is None:
                v = _S(j, m, self.g.s1, z) - _S(j, m, self.g.s0, z)
            else:
                v = _gauss_panels(j, m, self.g.s0, self.g.s1, z, self.quad, self.g.num.dt)
            self.memo[key] = v
        return v


def _ypoly(g, beta):
    a, b = beta
    c1 = [g.z * g.n0, g.t0]
    c2 = [g.z * g.n1, g.t1]
    out = [np.ones_like(g.z)]

    def mul(p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, x in enumerate(p):
            for j, y in enumerate(q):
                r[i + j] = r[i + j] + x * y
        return r
    for _ in range(a):
        out = mul(out, c1)
    for _ in range(b):
        out = mul(out, c2)
    return out


def _edge_integral(ctx, beta, R, P):
    """(N,K): int_chord y^beta (sum_n P_n r^n) ds."""
    num = ctx.g.num
    yp = _ypoly(ctx.g, beta)
    tot = 0
    for j, cj in enumerate(yp):
        inner = 0
        for n, c in P.items():
            inner = inner + num.c(c) * ctx.sdiff(j, n, R)
        tot = tot + cj * inner
    return tot


def _dangle(z, lo, hi):
    safe = np.where(z == 0, 1, z)
    return np.where(z == 0, 0, np.arctan2(safe * (hi - lo), z * z + lo * hi))


def _value_profile(ctx, R, P, unsplit):
    """(N,) int_T (sum P_n r^n) 1[r<=R] dA, pi-free coefficients, WITHOUT the 1/pi (returned separately)."""
    g, num = ctx.g, ctx.g.num
    z = g.z
    split = None
    if not unsplit or True:
        lo, hi, msk = g.chord(R)
        poly = 0
        for n, c in P.items():
            poly = poly + num.c(c) / (n + 2) * ctx.sdiff(0, n, R)
        mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
        edge = (z * poly - num.c(mR) * _dangle(z, lo, hi)).sum(1)          # times 1/pi later
        split = (edge, num.c(2 * mR) * g.ind)                             # (edge part [x 1/pi], indicator part [exact])
    if unsplit:
        inside = g.vmax2 <= num.c(Fraction(R) ** 2)
        if inside.any():
            poly = 0
            for n, c in P.items():
                poly = poly + num.c(c) / (n + 2) * ctx.sfull(0, n)
            un = (z * poly).sum(1)
            e_part = np.where(inside, un, split[0])
            i_part = np.where(inside, 0, split[1])
            return e_part, i_part
    return split


# ============================================================== public API
def _prep(verts, x, h, dtype, quad):
    verts = np.asarray(verts)
    x = np.asarray(x)
    if verts.ndim == 2:
        verts, x = verts[None], x[None]
    g = Geo(verts, x, h, dtype)
    return _Ctx(g, quad)


def value(verts, x, kernel, h=1, dtype=np.float64, quad=None, unsplit=True):
    ctx = _prep(verts, x, h, dtype, quad)
    num = ctx.g.num
    _, vals = compile_moment(kernel, (0, 0))
    e_tot, i_tot = 0, 0
    for R, P in vals.items():
        e, i = _value_profile(ctx, R, P, unsplit)
        e_tot = e_tot + e
        i_tot = i_tot + i
    return num.inv_pi * e_tot + i_tot


def gradient(verts, x, kernel, h=1, dtype=np.float64, quad=None, unsplit=True):
    return moment_gradient(verts, x, kernel, (0, 0), h, dtype, quad)


def moment(verts, x, kernel, alpha, h=1, dtype=np.float64, quad=None, unsplit=True):
    ctx = _prep(verts, x, h, dtype, quad)
    g, num = ctx.g, ctx.g.num
    edges, vals = compile_moment(kernel, tuple(alpha))
    e_tot = 0
    i_tot = 0
    for (i, beta, R), P in edges.items():
        ni = g.n0 if i == 0 else g.n1
        e_tot = e_tot + (ni * _edge_integral(ctx, beta, R, P)).sum(1)
    for R, P in vals.items():
        e, ind = _value_profile(ctx, R, P, unsplit)
        e_tot = e_tot + e
        i_tot = i_tot + ind
    hk = np.asarray(g.h, dtype=num.dt) ** (alpha[0] + alpha[1])
    return (num.inv_pi * e_tot + i_tot) * hk


def moment_gradient(verts, x, kernel, alpha, h=1, dtype=np.float64, quad=None):
    """grad_x m_alpha = -sum_e n_e int_chord y^alpha W ds   -> (N, 2)."""
    ctx = _prep(verts, x, h, dtype, quad)
    g, num = ctx.g, ctx.g.num
    gx = gy = 0
    for R, P in kernel_profile(kernel).items():
        integ = _edge_integral(ctx, tuple(alpha), R, P)
        gx = gx - (g.n0 * integ).sum(1)
        gy = gy - (g.n1 * integ).sum(1)
    k = alpha[0] + alpha[1]
    sc = num.inv_pi * np.asarray(g.h, dtype=num.dt) ** (k - 1)
    return np.stack([gx * sc, gy * sc], axis=-1)
