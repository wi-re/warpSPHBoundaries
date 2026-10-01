"""Stage 4: torch implementation of the 2D edge reductions (differentiable; float64 / float32; CPU or CUDA).

Same algorithm and the same EXACT kernel compilation as `np2d` (plans from `np2d.compile_moment`); only the array library differs.
Autodiff notes (tested in tests/edge/test_torch2d.py):
  * the total is continuous across every clipping / indicator / atan discontinuity, and all chord end-point terms are multiplied by
    integrands that vanish there (potentials vanish at R), so autograd through min/max/where gives the analytic derivative;
  * NaN-safe: sqrt and asinh/atan2 arguments use the double-`where` trick (z = 0, r = 0, chord grazing the circle);
  * the analytic derivatives are `gradient` (d/dx) and `shape_gradient` (d/d vertex); autograd is an independent check of them.

    value(verts, x, kernel, h)  gradient(...)  moment(..., alpha)  moment_gradient(...)  shape_gradient(..., alpha)
verts (N, K, 2) tensors (K = 3 for triangles; simple polygons allowed), x (N, 2), h float.
"""
from fractions import Fraction
from math import comb

import numpy as np
import torch

from . import np2d
from .np2d import compile_moment, kernel_profile

AUTO = np2d.AUTO_CLOSED_FRAC


def _c(f, like):
    f = Fraction(f)
    return torch.tensor(f.numerator / f.denominator if abs(f.denominator) < 2 ** 52 and abs(f.numerator) < 2 ** 52 else float(f),
                        dtype=like.dtype, device=like.device)


def _safe_sqrt(a):
    pos = a > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, a, torch.ones_like(a))), torch.zeros_like(a))


def _I_all(M, s, z):
    zz = z * z
    r2 = s * s + zz
    r = _safe_sqrt(r2)
    out = [s]
    if M >= 1:
        zero = z == 0
        safe = torch.where(zero, torch.ones_like(z), torch.abs(z))
        tail = torch.where(zero, torch.zeros_like(z), zz * torch.asinh(s / safe))
        out.append((s * r + tail) / 2)
    pw = [torch.ones_like(r2)]
    for _ in range(1, M // 2 + 2):
        pw.append(pw[-1] * r2)
    for m in range(2, M + 1):
        rp = pw[m // 2] if m % 2 == 0 else r * pw[(m - 1) // 2]
        out.append((s * rp + m * zz * out[m - 2]) / (m + 1))
    return out


def _S(j, m, s, z):
    zz = z * z
    if j % 2 == 0:
        h = j // 2
        I = _I_all(m + 2 * h, s, z)
        return sum(comb(h, i) * (-zz) ** (h - i) * I[m + 2 * i] for i in range(h + 1))
    h = (j - 1) // 2
    r2 = s * s + zz
    r = _safe_sqrt(r2)
    tot = 0
    for i in range(h + 1):
        mm = m + 2 * i
        rp = (r2 ** ((mm + 2) // 2)) if mm % 2 == 0 else r * r2 ** ((mm + 1) // 2)
        tot = tot + comb(h, i) * (-zz) ** (h - i) * rp / (mm + 2)
    return tot


class _AngleDiff(torch.autograd.Function):
    """atan(hi/z) - atan(lo/z) = atan2(z (hi - lo), z^2 + hi lo)   (0 at z = 0)  with the EXACT partials of atan(u/z),
    d/du = z/(z^2+u^2), d/dz = -u/(z^2+u^2), which are valid at z = 0 too.  Plain autograd through the forced-zero z = 0 branch
    would drop the O(1) term -1/hi + 1/lo of d(total)/dz (the total is continuous and so is its derivative)."""

    @staticmethod
    def forward(ctx, z, lo, hi):
        zero = z == 0
        sz = torch.where(zero, torch.ones_like(z), z)
        y = sz * (hi - lo)
        xx = z * z + lo * hi
        bad = zero | ((y == 0) & (xx == 0))
        y = torch.where(bad, torch.ones_like(y), y)
        xx = torch.where(bad, torch.ones_like(xx), xx)
        ctx.save_for_backward(z, lo, hi)
        return torch.where(bad, torch.zeros_like(z), torch.atan2(y, xx))

    @staticmethod
    def backward(ctx, gout):
        z, lo, hi = ctx.saved_tensors
        zz = z * z

        def safe_inv(a):
            return torch.where(a > 0, 1 / torch.where(a > 0, a, torch.ones_like(a)), torch.zeros_like(a))
        dh, dl = safe_inv(zz + hi * hi), safe_inv(zz + lo * lo)
        gz = (-hi * dh + lo * dl)
        # chord not empty <=> lo < hi ; empty chords have lo = hi = 0 and contribute nothing (hi = lo => zero by construction)
        return gout * gz, gout * (-z * dl), gout * (z * dh)


def _dangle(z, lo, hi):
    return _AngleDiff.apply(z, lo, hi)


class Geo:
    def __init__(self, verts, x, h):
        dt = verts.dtype
        self.dt = dt
        self.h = torch.as_tensor(h, dtype=dt, device=verts.device)
        hb = self.h.reshape(-1, 1, 1) if self.h.ndim else self.h
        rel = (verts - x[:, None, :]) / hb
        with torch.no_grad():
            a2 = (rel[:, :, 0] * torch.roll(rel, -1, 1)[:, :, 1] - rel[:, :, 1] * torch.roll(rel, -1, 1)[:, :, 0]).sum(1)
            self.flip = a2 < 0
        rel = torch.where(self.flip[:, None, None], torch.flip(rel, [1]), rel)
        self.N, self.K = rel.shape[0], rel.shape[1]
        p, q = rel, torch.roll(rel, -1, 1)
        d = q - p
        ell = _safe_sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
        self.ell = ell
        self.t0, self.t1 = d[..., 0] / ell, d[..., 1] / ell
        self.n0, self.n1 = self.t1, -self.t0
        self.z = (p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]) / ell
        self.s0 = (d[..., 0] * p[..., 0] + d[..., 1] * p[..., 1]) / ell
        self.s1 = (d[..., 0] * q[..., 0] + d[..., 1] * q[..., 1]) / ell
        self.vmax2 = (rel ** 2).sum(-1).max(1).values
        with torch.no_grad():
            self.ind = self._indicator(rel, p, q)
        self.two_pi = torch.tensor(2 * np.pi, dtype=dt, device=verts.device)
        self.inv_pi = torch.tensor(1 / np.pi, dtype=dt, device=verts.device)
        self._chords = {}

    def _indicator(self, rel, p, q):
        z = self.z
        pn, pp = torch.roll(rel, -1, 1), torch.roll(rel, 1, 1)
        a_, b_ = pn - rel, pp - rel
        cr = a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
        dt_ = a_[..., 0] * b_[..., 0] + a_[..., 1] * b_[..., 1]
        ang = torch.atan2(cr, dt_)
        ang = torch.where(ang <= 0, ang + 2 * np.pi, ang)
        at_vertex = (rel == 0).all(-1)
        vang = torch.where(at_vertex, ang, torch.zeros_like(ang)).sum(1) / (2 * np.pi)
        on_seg = (z == 0) & (self.s0 <= 0) & (self.s1 >= 0)
        on_edge = on_seg.any(1) & ~at_vertex.any(1)
        py, qy = p[..., 1], q[..., 1]
        straddle = (py > 0) != (qy > 0)
        den = torch.where(straddle, qy - py, torch.ones_like(py))
        xc = p[..., 0] + (0 - py) * (q[..., 0] - p[..., 0]) / den
        inside = ((straddle & (xc > 0)).sum(1) % 2 == 1).to(rel.dtype)
        return torch.where(at_vertex.any(1), vang, torch.where(on_edge, torch.full_like(vang, 0.5), inside))

    def chord(self, R):
        v = self._chords.get(R)
        if v is None:
            Rf = _c(R, self.z)
            az = torch.abs(self.z)
            L = _safe_sqrt((Rf - az) * (Rf + az))
            lo = torch.maximum(self.s0, -L)
            hi = torch.minimum(self.s1, L)
            m = (az < Rf) & (lo < hi)
            zero = torch.zeros_like(lo)
            v = (torch.where(m, lo, zero), torch.where(m, hi, zero), m)
            self._chords[R] = v
        return v


class Ctx:
    def __init__(self, g):
        self.g = g
        self.memo = {}

    def sdiff(self, j, m, R):
        key = (j, m, R)
        v = self.memo.get(key)
        if v is None:
            lo, hi, msk = self.g.chord(R)
            z = self.g.z
            v = torch.where(msk, _S(j, m, hi, z) - _S(j, m, lo, z), torch.zeros_like(z))
            self.memo[key] = v
        return v

    def sfull(self, j, m):
        key = (j, m, None)
        v = self.memo.get(key)
        if v is None:
            v = _S(j, m, self.g.s1, self.g.z) - _S(j, m, self.g.s0, self.g.z)
            self.memo[key] = v
        return v


def _ypoly(g, beta):
    a, b = beta
    c1 = [g.z * g.n0, g.t0]
    c2 = [g.z * g.n1, g.t1]
    out = [torch.ones_like(g.z)]

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


def _edge_integral(ctx, beta, R, P, shift=0):
    """(N,K): int_chord s^shift y^beta (sum_n P_n r^n) ds."""
    g = ctx.g
    yp = _ypoly(g, beta)
    tot = 0
    for j, cj in enumerate(yp):
        inner = 0
        for n, c in P.items():
            inner = inner + _c(c, g.z) * ctx.sdiff(j + shift, n, R)
        tot = tot + cj * inner
    return tot


def _value_profile(ctx, R, P, unsplit):
    g = ctx.g
    z = g.z
    lo, hi, msk = g.chord(R)
    poly = 0
    for n, c in P.items():
        poly = poly + _c(c / (n + 2), z) * ctx.sdiff(0, n, R)
    mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
    edge = (z * poly - _c(mR, z) * _dangle(z, lo, hi)).sum(1)
    e_part, i_part = edge, _c(2 * mR, z) * g.ind
    if unsplit:
        inside = g.vmax2 <= _c(Fraction(R) ** 2, z)
        if bool(inside.any()):
            poly = 0
            for n, c in P.items():
                poly = poly + _c(c / (n + 2), z) * ctx.sfull(0, n)
            un = (z * poly).sum(1)
            e_part = torch.where(inside, un, e_part)
            i_part = torch.where(inside, torch.zeros_like(i_part), i_part)
    return e_part, i_part


def _plan_terms(ctx, edges, vals, unsplit):
    g = ctx.g
    per = {}
    for (i, beta, R), P in edges.items():
        ni = g.n0 if i == 0 else g.n1
        e = (ni * _edge_integral(ctx, beta, R, P)).sum(1)
        d = per.setdefault(R, [0, 0])
        d[0] = d[0] + e
    for R, P in vals.items():
        e, ind = _value_profile(ctx, R, P, unsplit)
        d = per.setdefault(R, [0, 0])
        d[0] = d[0] + e
        d[1] = d[1] + ind
    return per


def _prep(verts, x, h):
    if verts.ndim == 2:
        verts, x = verts[None], x[None]
    return Ctx(Geo(verts, x, h))


def moment(verts, x, kernel, alpha, h=1.0, unsplit=True):
    ctx = _prep(verts, x, h)
    g = ctx.g
    alpha = tuple(alpha)
    trunc = _plan_terms(ctx, *compile_moment(kernel, alpha), unsplit)
    inner = _plan_terms(ctx, *compile_moment(kernel, alpha, True), unsplit) if (unsplit and alpha != (0, 0)) else None
    e_tot, i_tot = 0, 0
    for R in trunc:
        e, ind = trunc[R]
        if inner is not None:
            inside = g.vmax2 <= _c(Fraction(R) ** 2, g.z)
            if bool(inside.any()):
                e = torch.where(inside, inner[R][0], e)
                ind = torch.where(inside, inner[R][1], ind)
        e_tot, i_tot = e_tot + e, i_tot + ind
    return (g.inv_pi * e_tot + i_tot) * g.h ** (alpha[0] + alpha[1])


def value(verts, x, kernel, h=1.0, unsplit=True):
    return moment(verts, x, kernel, (0, 0), h, unsplit)


def moment_gradient(verts, x, kernel, alpha, h=1.0):
    """analytic grad_x m_alpha = - sum_e n_e int_chord y^alpha W ds   (N, 2)."""
    ctx = _prep(verts, x, h)
    g = ctx.g
    gx = gy = 0
    for R, P in kernel_profile(kernel).items():
        integ = _edge_integral(ctx, tuple(alpha), R, P)
        gx = gx - (g.n0 * integ).sum(1)
        gy = gy - (g.n1 * integ).sum(1)
    sc = g.inv_pi * g.h ** (alpha[0] + alpha[1] - 1)
    return torch.stack([gx * sc, gy * sc], dim=-1)


def gradient(verts, x, kernel, h=1.0):
    return moment_gradient(verts, x, kernel, (0, 0), h)


def shape_gradient(verts, x, kernel, alpha=(0, 0), h=1.0):
    """analytic d m_alpha / d vertex  (N, K, 2): n_e int (1 - tau) y^alpha W ds to the start vertex, n_e int tau ... to the end vertex."""
    ctx = _prep(verts, x, h)
    g = ctx.g
    ell = g.s1 - g.s0
    out = torch.zeros(g.N, g.K, 2, dtype=g.dt, device=verts.device)
    for R, P in kernel_profile(kernel).items():
        a0 = _edge_integral(ctx, tuple(alpha), R, P)
        a1s = _edge_integral(ctx, tuple(alpha), R, P, shift=1)
        a1 = (a1s - g.s0 * a0) / ell
        for comp, nrm in enumerate((g.n0, g.n1)):
            out[:, :, comp] = out[:, :, comp] + nrm * (a0 - a1) + torch.roll(nrm * a1, 1, 1)
    sc = g.inv_pi * g.h ** (alpha[0] + alpha[1] - 1)
    out = out * sc
    return torch.where(g.flip[:, None, None], torch.flip(out, [1]), out)
