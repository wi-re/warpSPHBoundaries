"""Stage 5: Warp (CPU/CUDA) kernels for the 2D edge reductions, triangles, float64.

One generic "plan interpreter" kernel: the exact compiled plan (np2d.compile_moment: truncated and inner variants, indicator weights,
all exact rationals rounded once) is flattened into constant arrays; every thread handles one (triangle, point) pair, loops over the
3 edges and the plan terms and evaluates the closed-form primitives on the fly (parity-chain recurrence for I_m, no local arrays).
Analytic adjoints are explicit kernels (no taping through loops / clip logic, as planned in backends-and-verification.md):

    moment(verts, x, kernel, alpha, h)        value / moments
    moment_gradient(...)                      d/dx   (edge-local)
    shape_gradient(...)                       d/d vertices (edge-local Reynolds adjoint)
    TorchMoment                               torch.autograd.Function: forward = Warp, backward = the explicit adjoint kernels

float64 only here (float32 on GPU needs the Chebyshev stable path of stage 3; not ported).
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
import warp as wp

from . import np2d

wp.config.quiet = True
f64 = wp.float64


# ---------------------------------------------------------------------------------------------- device functions
@wp.func
def _isqrt(a: f64) -> f64:
    if a > f64(0.0):
        return wp.sqrt(a)
    return f64(0.0)


@wp.func
def _asinh(u: f64) -> f64:
    a = wp.abs(u)
    v = wp.log(a + wp.sqrt(a * a + f64(1.0)))
    if u < f64(0.0):
        v = -v
    return v


@wp.func
def _Im(m: int, s: f64, z: f64) -> f64:
    """odd antiderivative of r^m, m >= 0 (parity-chain recurrence; z = 0 guarded)."""
    zz = z * z
    r2 = s * s + zz
    r = _isqrt(r2)
    val = s
    idx = int(0)
    if m % 2 == 1:
        tail = f64(0.0)
        if z != f64(0.0):
            tail = zz * _asinh(s / wp.abs(z))
        val = (s * r + tail) / f64(2.0)
        idx = 1
    while idx < m:
        idx += 2
        rp = f64(0.0)
        if idx % 2 == 0:
            rp = wp.pow(r2, f64(idx / 2))
        else:
            rp = r * wp.pow(r2, f64((idx - 1) / 2))
        val = (s * rp + f64(idx) * zz * val) / f64(idx + 1)
    return val


@wp.func
def _binom(n: int, k: int) -> f64:
    v = f64(1.0)
    for i in range(k):
        v = v * f64(n - i) / f64(i + 1)
    return v


@wp.func
def _S(j: int, m: int, s: f64, z: f64) -> f64:
    """int s^j r^m ds, m >= 0."""
    zz = z * z
    tot = f64(0.0)
    if j % 2 == 0:
        h = j / 2
        for i in range(h + 1):
            sg = f64(1.0)
            if (h - i) % 2 == 1:
                sg = f64(-1.0)
            tot += _binom(h, i) * sg * wp.pow(zz, f64(h - i)) * _Im(m + 2 * i, s, z)
    else:
        h = (j - 1) / 2
        r2 = s * s + zz
        r = _isqrt(r2)
        for i in range(h + 1):
            mm = m + 2 * i
            rp = f64(0.0)
            if mm % 2 == 0:
                rp = wp.pow(r2, f64((mm + 2) / 2))
            else:
                rp = r * wp.pow(r2, f64((mm + 1) / 2))
            sg = f64(1.0)
            if (h - i) % 2 == 1:
                sg = f64(-1.0)
            tot += _binom(h, i) * sg * wp.pow(zz, f64(h - i)) * rp / f64(mm + 2)
    return tot


@wp.func
def _dangle(z: f64, lo: f64, hi: f64) -> f64:
    if z == f64(0.0):
        return f64(0.0)
    y = z * (hi - lo)
    x = z * z + lo * hi
    if y == f64(0.0) and x == f64(0.0):
        return f64(0.0)
    return wp.atan2(y, x)


@wp.func
def _sdiff(j: int, m: int, lo: f64, hi: f64, z: f64) -> f64:
    return _S(j, m, hi, z) - _S(j, m, lo, z)


@wp.func
def _ypoly_coef(a: int, b: int, j: int, z: f64, n0: f64, n1: f64, t0: f64, t1: f64) -> f64:
    """coefficient of s^j in (z n0 + s t0)^a (z n1 + s t1)^b   (j <= a + b)."""
    tot = f64(0.0)
    for ja in range(a + 1):
        jb = j - ja
        if jb >= 0 and jb <= b:
            ca = _binom(a, ja) * wp.pow(t0, f64(ja)) * wp.pow(z * n0, f64(a - ja))
            cb = _binom(b, jb) * wp.pow(t1, f64(jb)) * wp.pow(z * n1, f64(b - jb))
            tot += ca * cb
    return tot


# ---------------------------------------------------------------------------------------------- the kernels
@wp.func
def _edge_integral(a: int, b: int, shift: int, lo: f64, hi: f64, z: f64, n0: f64, n1: f64, t0: f64, t1: f64,
                   cn: wp.array(dtype=int), cc: wp.array(dtype=f64), c0: int, c1: int) -> f64:
    """int_chord s^shift y^(a,b) (sum_n c_n r^n) ds."""
    tot = f64(0.0)
    for j in range(a + b + 1):
        yc = _ypoly_coef(a, b, j, z, n0, n1, t0, t1)
        if yc != f64(0.0):
            inner = f64(0.0)
            for k in range(c0, c1):
                inner += cc[k] * _sdiff(j + shift, cn[k], lo, hi, z)
            tot += yc * inner
    return tot


@wp.kernel
def _moment_kernel(verts: wp.array2d(dtype=wp.vec2d), xs: wp.array(dtype=wp.vec2d), hs: wp.array(dtype=f64),
                   radii: wp.array(dtype=f64), inv_pi: f64,
                   # edge terms
                   e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int), e_R: wp.array(dtype=int),
                   e_var: wp.array(dtype=int), e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int),
                   # value terms
                   v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int),
                   v_mR: wp.array(dtype=f64),
                   cn: wp.array(dtype=int), cc: wp.array(dtype=f64), n_edge_terms: int, n_val_terms: int, alpha_k: int,
                   out: wp.array(dtype=f64)):
    tid = wp.tid()
    h = hs[tid]
    xv = xs[tid]
    p0 = (verts[tid, 0] - xv) / h
    p1 = (verts[tid, 1] - xv) / h
    p2 = (verts[tid, 2] - xv) / h
    # ccw normalisation
    a2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if a2 < f64(0.0):
        tmp = p0
        p0 = p2
        p2 = tmp
    # per-edge geometry in vec3d (component = edge)
    z = wp.vec3d()
    s0 = wp.vec3d()
    s1 = wp.vec3d()
    n0 = wp.vec3d()
    n1 = wp.vec3d()
    t0 = wp.vec3d()
    t1 = wp.vec3d()
    vmax2 = f64(0.0)
    neg = int(0)
    nzero = int(0)
    for e in range(3):
        p = p0
        q = p1
        if e == 1:
            p = p1
            q = p2
        if e == 2:
            p = p2
            q = p0
        d = q - p
        ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        t0[e] = d[0] / ell
        t1[e] = d[1] / ell
        n0[e] = t1[e]
        n1[e] = -t0[e]
        z[e] = (p[0] * q[1] - p[1] * q[0]) / ell
        s0[e] = (d[0] * p[0] + d[1] * p[1]) / ell
        s1[e] = (d[0] * q[0] + d[1] * q[1]) / ell
        vmax2 = wp.max(vmax2, p[0] * p[0] + p[1] * p[1])
        if z[e] < f64(0.0):
            neg = 1
        if z[e] == f64(0.0):
            nzero += 1
    ind = f64(0.0)
    if neg == 0:
        if nzero == 0:
            ind = f64(1.0)
        elif nzero == 1:
            ind = f64(0.5)
        else:
            # vertex: shared by the two zero edges (edge i-1 ends at vertex i)
            for vi in range(3):
                zprev = z[(vi + 2) % 3]
                if z[vi] == f64(0.0) and zprev == f64(0.0):
                    pv = p0
                    pn = p1
                    pp = p2
                    if vi == 1:
                        pv = p1
                        pn = p2
                        pp = p0
                    if vi == 2:
                        pv = p2
                        pn = p0
                        pp = p1
                    a_ = pn - pv
                    b_ = pp - pv
                    ang = wp.atan2(a_[0] * b_[1] - a_[1] * b_[0], a_[0] * b_[0] + a_[1] * b_[1])
                    if ang <= f64(0.0):
                        ang = ang + f64(6.283185307179586)
                    ind = ang / f64(6.283185307179586)
    total_e = f64(0.0)
    total_i = f64(0.0)
    # ---------------- edge terms
    for t in range(n_edge_terms):
        R = radii[e_R[t]]
        inside = vmax2 <= R * R
        want_inner = e_var[t] == 1
        if inside == want_inner or (e_var[t] == 0 and not inside) or (e_var[t] == 1 and inside):
            for e in range(3):
                az = wp.abs(z[e])
                if az < R:
                    L = _isqrt((R - az) * (R + az))
                    lo = wp.max(s0[e], -L)
                    hi = wp.min(s1[e], L)
                    if lo < hi:
                        ni = n0[e]
                        if e_i[t] == 1:
                            ni = n1[e]
                        total_e += ni * _edge_integral(e_a[t], e_b[t], 0, lo, hi, z[e], n0[e], n1[e], t0[e], t1[e], cn, cc, e_c0[t], e_c1[t])
    # ---------------- value terms
    for t in range(n_val_terms):
        R = radii[v_R[t]]
        inside = vmax2 <= R * R
        take = (v_var[t] == 1 and inside) or (v_var[t] == 0 and not inside)
        if v_var[t] == 0 and alpha_k == 0:
            take = True            # alpha = 0: only the truncated plan exists; unsplit handled below by `inside`
        if take:
            mR = v_mR[t]
            if inside:
                for e in range(3):
                    poly = f64(0.0)
                    for k in range(v_c0[t], v_c1[t]):
                        poly += cc[k] / f64(cn[k] + 2) * _sdiff(0, cn[k], s0[e], s1[e], z[e])
                    total_e += z[e] * poly
            else:
                total_i += f64(2.0) * mR * ind
                for e in range(3):
                    az = wp.abs(z[e])
                    if az < R:
                        L = _isqrt((R - az) * (R + az))
                        lo = wp.max(s0[e], -L)
                        hi = wp.min(s1[e], L)
                        if lo < hi:
                            poly = f64(0.0)
                            for k in range(v_c0[t], v_c1[t]):
                                poly += cc[k] / f64(cn[k] + 2) * _sdiff(0, cn[k], lo, hi, z[e])
                            total_e += z[e] * poly - mR * _dangle(z[e], lo, hi)
    out[tid] = (inv_pi * total_e + total_i) * wp.pow(h, f64(alpha_k))


@wp.kernel
def _shape_kernel(verts: wp.array2d(dtype=wp.vec2d), xs: wp.array(dtype=wp.vec2d), hs: wp.array(dtype=f64),
                  radii: wp.array(dtype=f64), inv_pi: f64,
                  g_i: wp.array(dtype=int), g_a: wp.array(dtype=int), g_b: wp.array(dtype=int), g_R: wp.array(dtype=int),
                  g_c0: wp.array(dtype=int), g_c1: wp.array(dtype=int), n_terms: int,
                  cn: wp.array(dtype=int), cc: wp.array(dtype=f64), alpha_k: int, mode: int,
                  out: wp.array2d(dtype=wp.vec2d), gx: wp.array(dtype=wp.vec2d)):
    """mode 0: shape gradient (d/d vertices) + gx (d/dx = -sum_e n_e int y^alpha W ds); one pass over the kernel blocks."""
    tid = wp.tid()
    h = hs[tid]
    xv = xs[tid]
    p0 = (verts[tid, 0] - xv) / h
    p1 = (verts[tid, 1] - xv) / h
    p2 = (verts[tid, 2] - xv) / h
    a2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    flip = a2 < f64(0.0)
    if flip:
        tmp = p0
        p0 = p2
        p2 = tmp
    acc0 = wp.vec2d(f64(0.0), f64(0.0))
    acc1 = wp.vec2d(f64(0.0), f64(0.0))
    acc2 = wp.vec2d(f64(0.0), f64(0.0))
    g = wp.vec2d(f64(0.0), f64(0.0))
    for e in range(3):
        p = p0
        q = p1
        if e == 1:
            p = p1
            q = p2
        if e == 2:
            p = p2
            q = p0
        d = q - p
        ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        t0 = d[0] / ell
        t1 = d[1] / ell
        n0 = t1
        n1 = -t0
        z = (p[0] * q[1] - p[1] * q[0]) / ell
        s0 = (d[0] * p[0] + d[1] * p[1]) / ell
        s1 = (d[0] * q[0] + d[1] * q[1]) / ell
        a0 = f64(0.0)
        a1s = f64(0.0)
        for t in range(n_terms):
            R = radii[g_R[t]]
            az = wp.abs(z)
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    a0 += _edge_integral(g_a[t], g_b[t], 0, lo, hi, z, n0, n1, t0, t1, cn, cc, g_c0[t], g_c1[t])
                    a1s += _edge_integral(g_a[t], g_b[t], 1, lo, hi, z, n0, n1, t0, t1, cn, cc, g_c0[t], g_c1[t])
        a1 = (a1s - s0 * a0) / (s1 - s0)
        nvec = wp.vec2d(n0, n1)
        start = nvec * (a0 - a1)
        end = nvec * a1
        g = g - nvec * a0
        if e == 0:
            acc0 += start
            acc1 += end
        if e == 1:
            acc1 += start
            acc2 += end
        if e == 2:
            acc2 += start
            acc0 += end
    sc = inv_pi * wp.pow(h, f64(alpha_k - 1))
    if flip:
        # vertices were reversed: accumulators refer to (v2, v1, v0)
        out[tid, 0] = acc2 * sc
        out[tid, 1] = acc1 * sc
        out[tid, 2] = acc0 * sc
    else:
        out[tid, 0] = acc0 * sc
        out[tid, 1] = acc1 * sc
        out[tid, 2] = acc2 * sc
    gx[tid] = g * sc


# ---------------------------------------------------------------------------------------------- host side: flatten plans
def _flatten(plan_edges, plan_vals, variant, state):
    for (i, beta, R), P in plan_edges.items():
        state["e"].append((i, beta[0], beta[1], state["R"](R), variant, state["coef"](P)))
    for R, P in plan_vals.items():
        mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
        state["v"].append((state["R"](R), variant, state["coef"](P), float(mR)))


@lru_cache(maxsize=None)
def _moment_plan(kernel, alpha):
    radii, cn, cc = [], [], []

    def Ridx(R):
        if float(R) not in radii:
            radii.append(float(R))
        return radii.index(float(R))

    def coef(P):
        c0 = len(cn)
        for n, c in sorted(P.items()):
            cn.append(n)
            cc.append(float(c))
        return (c0, len(cn))
    st = dict(e=[], v=[], R=Ridx, coef=coef)
    _flatten(*np2d.compile_moment(kernel, alpha), 0, st)
    if alpha != (0, 0):
        _flatten(*np2d.compile_moment(kernel, alpha, True), 1, st)
    return radii, cn, cc, st["e"], st["v"]


@lru_cache(maxsize=None)
def _grad_plan(kernel, alpha):
    radii, cn, cc, rows = [], [], [], []
    for R, P in np2d.kernel_profile(kernel).items():
        radii.append(float(R))
        c0 = len(cn)
        for n, c in sorted(P.items()):
            cn.append(n)
            cc.append(float(c))
        rows.append((0, alpha[0], alpha[1], len(radii) - 1, c0, len(cn)))
    return radii, cn, cc, rows


def _arr(a, dtype, device):
    return wp.array(np.asarray(a, dtype=np.float64 if dtype is f64 else np.int32) if len(a) else np.zeros(1, dtype=np.float64 if dtype is f64 else np.int32), dtype=dtype, device=device)


def _prep_inputs(verts, x, h, device):
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    N = verts.shape[0]
    hs = np.full(N, float(h)) if np.ndim(h) == 0 else np.asarray(h, dtype=np.float64)
    return N, wp.array(verts, dtype=wp.vec2d, device=device), wp.array(x, dtype=wp.vec2d, device=device), wp.array(hs, dtype=f64, device=device)


class MomentLauncher:
    """plan arrays resident on `device`; `run(wv, wx, wh, out)` launches the moment kernel on device arrays (no host traffic)."""

    def __init__(self, kernel, alpha=(0, 0), device="cuda:0"):
        self.alpha = tuple(alpha)
        self.device = device
        radii, cn, cc, E, V = _moment_plan(kernel, self.alpha)
        col = lambda rows, k, dt: _arr([r[k] for r in rows], dt, device)
        ec = [r[5] for r in E]
        vc = [r[2] for r in V]
        self.args = [_arr(radii, f64, device), f64(1 / np.pi),
                     col(E, 0, int), col(E, 1, int), col(E, 2, int), col(E, 3, int), col(E, 4, int),
                     _arr([c[0] for c in ec], int, device), _arr([c[1] for c in ec], int, device),
                     col(V, 0, int), col(V, 1, int), _arr([c[0] for c in vc], int, device), _arr([c[1] for c in vc], int, device), col(V, 3, f64),
                     _arr(cn, int, device), _arr(cc, f64, device), len(E), len(V), self.alpha[0] + self.alpha[1]]

    def run(self, wv, wx, wh, out):
        wp.launch(_moment_kernel, dim=out.shape[0], device=self.device, inputs=[wv, wx, wh, *self.args, out])


def moment(verts, x, kernel, alpha=(0, 0), h=1.0, device="cuda:0"):
    N, wv, wx, wh = _prep_inputs(verts, x, h, device)
    out = wp.zeros(N, dtype=f64, device=device)
    MomentLauncher(kernel, alpha, device).run(wv, wx, wh, out)
    return out.numpy()


def _shape_and_grad(verts, x, kernel, alpha, h, device):
    radii, cn, cc, rows = _grad_plan(kernel, tuple(alpha))
    N, wv, wx, wh = _prep_inputs(verts, x, h, device)
    out = wp.zeros((N, 3), dtype=wp.vec2d, device=device)
    gx = wp.zeros(N, dtype=wp.vec2d, device=device)
    wp.launch(_shape_kernel, dim=N, device=device, inputs=[
        wv, wx, wh, _arr(radii, f64, device), f64(1 / np.pi),
        _arr([r[0] for r in rows], int, device), _arr([r[1] for r in rows], int, device), _arr([r[2] for r in rows], int, device),
        _arr([r[3] for r in rows], int, device), _arr([r[4] for r in rows], int, device), _arr([r[5] for r in rows], int, device), len(rows),
        _arr(cn, int, device), _arr(cc, f64, device), alpha[0] + alpha[1], 0, out, gx])
    return out.numpy(), gx.numpy()


def shape_gradient(verts, x, kernel, alpha=(0, 0), h=1.0, device="cuda:0"):
    return _shape_and_grad(verts, x, kernel, alpha, h, device)[0]


def moment_gradient(verts, x, kernel, alpha=(0, 0), h=1.0, device="cuda:0"):
    return _shape_and_grad(verts, x, kernel, alpha, h, device)[1]


def gradient(verts, x, kernel, h=1.0, device="cuda:0"):
    return moment_gradient(verts, x, kernel, (0, 0), h, device)


# ---------------------------------------------------------------------------------------------- torch autograd bridge
try:
    import torch

    class TorchMoment(torch.autograd.Function):
        """forward = Warp kernel; backward = the explicit edge-local adjoint kernels (d/dx and d/d vertices)."""

        @staticmethod
        def forward(ctx, verts, x, kernel, alpha, h, device):
            ctx.save_for_backward(verts, x)
            ctx.cfg = (kernel, tuple(alpha), h, device)
            out = moment(verts.detach().cpu().numpy(), x.detach().cpu().numpy(), kernel, alpha, h, device)
            return torch.as_tensor(out, dtype=verts.dtype, device=verts.device)

        @staticmethod
        def backward(ctx, gout):
            verts, x = ctx.saved_tensors
            kernel, alpha, h, device = ctx.cfg
            sg, gx = _shape_and_grad(verts.detach().cpu().numpy(), x.detach().cpu().numpy(), kernel, alpha, h, device)
            go = gout.detach().cpu().numpy()
            gv = torch.as_tensor(sg * go[:, None, None], dtype=verts.dtype, device=verts.device)
            gxx = torch.as_tensor(gx * go[:, None], dtype=x.dtype, device=x.device)
            return gv, gxx, None, None, None, None
except ImportError:      # pragma: no cover
    TorchMoment = None
