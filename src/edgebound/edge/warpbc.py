"""Boundary integrals as Warp operations (2D, triangles): pair-weight engine (this file, part 1).

For every (query particle i, boundary element e) pair the P1 weights of the exact edge reduction are computed on the GPU/CPU:

    w_k  = int_e N_k W dA            (k = 0,1,2 vertex weights; sum_k w_k = int_e W)
    G_k  = int_e N_k grad_x W dA     (2-vectors; physical units, ~ 1/h)

from nine "channels" -- the moments m_(0,0), m_(1,0), m_(0,1) and g_(0,0), g_(1,0), g_(0,1) (g_alpha = int y^alpha grad_x W) -- via the exact compiled
plans of `np2d` (compact-potential recursion, inner potentials for elements inside the support, exact indicator/atan weights).
Arithmetic inside the kernels is in the precision of warpSPHCore (`precision.py`: float64 by default, float32 with the Chebyshev stable plans via warpSPHCore_PRECISION); the I/O tensors
are cast to it and the returned tensors are float64.  See docs/boundary-operations.md for the cost and the float32 discussion.

The operations (Interpolate, Gradient, Divergence, Curl, Density) are applied to these weights in `boundaryOperation` (part 2).
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
import warp as wp

from . import np2d
from .np2d import _compile_profile_moment, _inner_profile, compile_moment, kernel_profile
from .precision import IS_F32, IS_F64, np_real, require_f64, torch_real, vec2_t, vec3_t
from .warp2d import _dangle, _edge_integral, _isqrt, _sdiff, real

wp.config.quiet = True

# channel layout: 0..2 = m(0,0), m(1,0), m(0,1);  3,4 = g(0,0)_{x,y};  5,6 = g(1,0)_{x,y};  7,8 = g(0,1)_{x,y}
NCH = 9
GAUSS_N = 8                 # nodes per direction of the far-field Gauss branch
FAR_RATIO = 1.0             # |centroid - x| >= FAR_RATIO * longest edge   (element far from x relative to its size)
TINY = 0.05                 # elements <= TINY h may use Gauss even when straddling a kernel radius
ALPHAS = [(0, 0), (1, 0), (0, 1)]


# ------------------------------------------------------------------------------------------------ host: plan flattening
class _PlanBuilder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.radii, self.cn, self.cc = [], [], []
        self.E, self.V = [], []          # edge terms, value terms

    def R(self, R):
        f = float(R)
        if f not in self.radii:
            self.radii.append(f)
        return self.radii.index(f)

    def coef(self, P, scale=1.0):
        c0 = len(self.cn)
        for n, c in sorted(P.items()):
            self.cn.append(n)
            self.cc.append(float(c) * scale)
        return c0, len(self.cn)

    # edge term: ch, i, a, b, Ridx, var (0 trunc cond, 1 inner cond, 2 always), gate, coeff range
    def edge(self, ch, i, beta, R, var, gate, P, scale=1.0):
        self.E.append((ch, i, beta[0], beta[1], self.R(R), var, gate, *self.coef(P, scale)))

    # value term: ch, Ridx, var, gate, coeff range, mR (pi-free m_R of the UNSCALED profile) ; the scale enters mR and the coefficients
    def value(self, ch, R, var, gate, P, scale=1.0):
        mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
        self.V.append((ch, self.R(R), var, gate, *self.coef(P, scale), float(mR) * scale))

    def moment_plan(self, ch, alpha, scale, gate, inner_gate_only=False):
        """terms of m_alpha (the kernel moment, per-radius trunc/inner selection) into channel ch, scaled."""
        edges, vals = compile_moment(self.kernel, alpha)
        always = alpha == (0, 0)
        for (i, beta, R), P in edges.items():
            self.edge(ch, i, beta, R, 0, gate, P, scale)
        for R, P in vals.items():
            self.value(ch, R, 2 if always else 0, gate, P, scale)
        if not always:
            edges, vals = compile_moment(self.kernel, alpha, True)
            for (i, beta, R), P in edges.items():
                self.edge(ch, i, beta, R, 1, gate, P, scale)
            for R, P in vals.items():
                self.value(ch, R, 1, gate, P, scale)

    def profile_moment_plan(self, ch, Q, Rin, beta, scale, gate):
        edges, vals = _compile_profile_moment(tuple(sorted(Q.items())), Rin, beta)
        for (i, bb, R), P in edges.items():
            self.edge(ch, i, bb, R, 1, gate, P, scale)
        for R, P in vals.items():
            self.value(ch, R, 1, gate, P, scale)


class _ChebBuilder(_PlanBuilder):
    """same term list as _PlanBuilder, but every coefficient range holds the CHEBYSHEV coefficients of the profile on [0, R]
    (exact Fractions via np2d.cheb_coeffs, float64 only at the end, times the term scale).  Edge-term profile is P; the value-term
    profile is Q = P/(n+2) (what the monomial kernel integrates as cc[k]/(cn[k]+2) r^cn[k]); mR is from the UNSCALED monomial P.  No cn."""

    def _cheb(self, P, R, scale):
        c0 = len(self.cc)
        self.cc.extend(float(x) * scale for x in np2d.cheb_coeffs(P, R))
        return c0, len(self.cc)

    def edge(self, ch, i, beta, R, var, gate, P, scale=1.0):
        self.E.append((ch, i, beta[0], beta[1], self.R(R), var, gate, *self._cheb(P, R, scale)))

    def value(self, ch, R, var, gate, P, scale=1.0):
        mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
        Q = {n: c / (n + 2) for n, c in P.items()}
        self.V.append((ch, self.R(R), var, gate, *self._cheb(Q, R, scale), float(mR) * scale))


def _build_terms(pb, kernel):
    """fill the term lists of builder `pb` (moments, g_alpha channels, gates, inner forms); return the innermost-radius index."""
    # moments
    for ch, al in enumerate(ALPHAS):
        pb.moment_plan(ch, al, 1.0, 0)
    # g_alpha = int y^alpha grad_x W :  GATE 1 (element NOT inside the innermost piece): truncated form  -grad-part + alpha_j m_{alpha-e_j}
    #                                   GATE 2 (inside):  inner potential form with Q = pi W - pi W(0)
    Q, Rin = _inner_profile(kernel)
    for k, al in enumerate(ALPHAS):
        for j in (0, 1):
            ch = 3 + 2 * k + j
            for R, P in kernel_profile(kernel).items():
                pb.edge(ch, j, al, R, 2, 1, P, -1.0)
            if al[j]:
                beta = (al[0] - (j == 0), al[1] - (j == 1))
                pb.moment_plan(ch, beta, float(al[j]), 1)
            # inner form
            pb.edge(ch, j, al, Rin, 2, 2, Q, -1.0)
            if al[j]:
                beta = (al[0] - (j == 0), al[1] - (j == 1))
                pb.profile_moment_plan(ch, Q, Rin, beta, float(al[j]), 2)
    return pb.R(Rin)


@lru_cache(maxsize=None)
def build_plan(kernel):
    """flatten all nine channels of the P1 weights for `kernel` (exact rationals -> float64 constants)."""
    pb = _PlanBuilder(kernel)
    rin_idx = _build_terms(pb, kernel)
    return pb, rin_idx


@lru_cache(maxsize=None)
def build_cheb_plan(kernel):
    """same term list as build_plan, but each coefficient range holds the Chebyshev coefficients of the profile on [0, R] (exact compile -> float64)."""
    pb = _ChebBuilder(kernel)
    rin_idx = _build_terms(pb, kernel)
    return pb, rin_idx


def _channel_rows(pb, channels):
    """the edge (E) and value (V) term rows of the plan builder `pb`, restricted to the terms of `channels` (None: all nine); the channels are independent sums, so the kept ones are bit-identical to the full plan."""
    if channels is None:
        return pb.E, pb.V
    keep = set(int(c) for c in channels)
    return [r for r in pb.E if r[0] in keep], [r for r in pb.V if r[0] in keep]


_DEVICE_PLANS = {}


def _device_plan(kernel, device, channels=None):
    """DevicePlan cache per (kernel, device, channels): the plan arrays are constants of the kernel (building one costs 0.37 ms, a fifth to a third of an edge_channels call)."""
    key = (kernel, str(device), None if channels is None else tuple(sorted(set(int(c) for c in channels))))
    if key not in _DEVICE_PLANS:
        _DEVICE_PLANS[key] = DevicePlan(kernel, device, channels)
    return _DEVICE_PLANS[key]


class DevicePlan:
    """the plan arrays resident on one device."""

    def __init__(self, kernel, device, channels=None):
        pb, rin_idx = build_plan(kernel)
        self.rin_idx = rin_idx
        E, V = _channel_rows(pb, channels)
        self.nE, self.nV = len(E), len(V)
        i32 = lambda rows, k: wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=device)
        self.radii_host = list(pb.radii)
        self.radii = wp.array(np.array(pb.radii, dtype=np_real), dtype=real, device=device)
        self.cn = wp.array(np.array(pb.cn or [0], dtype=np.int32), dtype=int, device=device)
        self.cc = wp.array(np.array(pb.cc or [0.0], dtype=np_real), dtype=real, device=device)
        self.e = [i32(E, k) for k in range(9)]          # ch, i, a, b, R, var, gate, c0, c1
        self.v = [i32(V, k) for k in range(6)]          # ch, R, var, gate, c0, c1 (+ mR below)
        self.v_mR = wp.array(np.array([r[6] for r in V] or [0.0], dtype=np_real), dtype=real, device=device)
        # u-basis of the kernel (pi W = sum b_k (R - r)^k per block), for the far-field Gauss branch
        from .np_fem import kernel_ubasis
        ubR, ubK, ubB = [], [], []
        for R, d in kernel_ubasis(kernel).items():
            for k, b in d.items():
                ubR.append(float(R)); ubK.append(int(k)); ubB.append(float(b))
        self.nub = len(ubR)
        self.ubR = wp.array(np.array(ubR, dtype=np_real), dtype=real, device=device)
        self.ubK = wp.array(np.array(ubK, dtype=np.int32), dtype=int, device=device)
        self.ubB = wp.array(np.array(ubB, dtype=np_real), dtype=real, device=device)
        gx, gw = np.polynomial.legendre.leggauss(GAUSS_N)
        self.gx = wp.array((gx + 1) / 2, dtype=real, device=device)
        self.gw = wp.array(gw / 2, dtype=real, device=device)


class ChebPlan:
    """the Chebyshev-quadrature plan arrays resident on one device (the opt-in `stable=` route of edge_channels): the same term
    arrays as DevicePlan (Chebyshev coefficient ranges, no cn) plus the Gauss-Legendre nodes/weights on [-1, 1]; `panels` is a
    launch argument of the kernel, not part of the arrays."""

    def __init__(self, kernel, device, nodes=16, panels=8, channels=None):
        pb, rin_idx = build_cheb_plan(kernel)
        self.rin_idx = rin_idx
        self.nodes = nodes
        self.panels = panels
        E, V = _channel_rows(pb, channels)
        self.nE, self.nV = len(E), len(V)
        i32 = lambda rows, k: wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=device)
        self.radii = wp.array(np.array(pb.radii, dtype=np_real), dtype=real, device=device)
        self.cc = wp.array(np.array(pb.cc or [0.0], dtype=np_real), dtype=real, device=device)
        self.e = [i32(E, k) for k in range(9)]          # ch, i, a, b, R, var, gate, c0, c1
        self.v = [i32(V, k) for k in range(6)]          # ch, R, var, gate, c0, c1 (+ mR below)
        self.v_mR = wp.array(np.array([r[6] for r in V] or [0.0], dtype=np_real), dtype=real, device=device)
        gx, gw = np.polynomial.legendre.leggauss(nodes)
        self.gx = wp.array(gx, dtype=real, device=device)       # on [-1, 1]
        self.gw = wp.array(gw, dtype=real, device=device)


# ------------------------------------------------------------------------------------------------ the pair kernel
@wp.kernel
def _pair_weights_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                         pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                         verts: wp.array(dtype=vec2_t), elems: wp.array(dtype=wp.vec3i),
                         radii: wp.array(dtype=real), rin_idx: int, inv_pi: real,
                         e_ch: wp.array(dtype=int), e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                         e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                         e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                         v_ch: wp.array(dtype=int), v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                         v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=real), n_v: int,
                         cn: wp.array(dtype=int), cc: wp.array(dtype=real),
                         ub_R: wp.array(dtype=real), ub_K: wp.array(dtype=int), ub_B: wp.array(dtype=real), n_ub: int,
                         gnode: wp.array(dtype=real), gwt: wp.array(dtype=real), n_g: int, far_ratio: real, tiny: real, n_radii: int,
                         wout: wp.array2d(dtype=real), gout: wp.array2d(dtype=real)):
    tid = wp.tid()
    qi = pair_q[tid]
    el = elems[pair_e[tid]]
    h = sup[qi]
    xv = pos[qi]
    r0 = (verts[el[0]] - xv) / h
    r1 = (verts[el[1]] - xv) / h
    r2 = (verts[el[2]] - xv) / h
    # barycentric functions lambda_k(x + y) = l_k + g_k . y   (x = origin), from the original vertex order (orientation independent)
    d2 = (r1[0] - r0[0]) * (r2[1] - r0[1]) - (r1[1] - r0[1]) * (r2[0] - r0[0])
    l = vec3_t()
    gx = vec3_t()
    gy = vec3_t()
    for k in range(3):
        a = r1
        b = r2
        if k == 1:
            a = r2
            b = r0
        if k == 2:
            a = r0
            b = r1
        l[k] = (a[0] * b[1] - a[1] * b[0]) / d2
        gx[k] = (a[1] - b[1]) / d2
        gy[k] = (b[0] - a[0]) / d2
    # ---- far-field Gauss branch: element far from x relative to its size, kernel smooth on it (no straddled radius) or element tiny
    cxm = (r0[0] + r1[0] + r2[0]) / real(3.0)
    cym = (r0[1] + r1[1] + r2[1]) / real(3.0)
    rho = wp.sqrt(cxm * cxm + cym * cym)
    emax = real(0.0)
    rmin = real(1.0e300)
    rmax = real(0.0)
    for e in range(3):
        pa = r0
        pb = r1
        if e == 1:
            pa = r1
            pb = r2
        if e == 2:
            pa = r2
            pb = r0
        dd = pb - pa
        ln = wp.sqrt(dd[0] * dd[0] + dd[1] * dd[1])
        emax = wp.max(emax, ln)
        tt = wp.clamp(-(pa[0] * dd[0] + pa[1] * dd[1]) / (ln * ln), real(0.0), real(1.0))
        qx = pa[0] + tt * dd[0]
        qy = pa[1] + tt * dd[1]
        rmin = wp.min(rmin, wp.sqrt(qx * qx + qy * qy))
        rmax = wp.max(rmax, wp.sqrt(pa[0] * pa[0] + pa[1] * pa[1]))
    straddle = int(0)
    for ri in range(n_radii):
        if rmin < radii[ri] and radii[ri] < rmax:
            straddle = 1
    if rho >= far_ratio * emax and (straddle == 0 or emax <= tiny):
        w0 = real(0.0)
        w1 = real(0.0)
        w2 = real(0.0)
        gx0 = real(0.0)
        gy0 = real(0.0)
        gx1 = real(0.0)
        gy1 = real(0.0)
        gx2 = real(0.0)
        gy2 = real(0.0)
        area2 = wp.abs(d2)
        for ia in range(n_g):
            for ib in range(n_g):
                ua = gnode[ia]
                vb = gnode[ib]
                m1 = ua
                m2 = vb * (real(1.0) - ua)
                m0 = real(1.0) - m1 - m2
                wt = gwt[ia] * gwt[ib] * (real(1.0) - ua) * area2
                yx = m0 * r0[0] + m1 * r1[0] + m2 * r2[0]
                yy = m0 * r0[1] + m1 * r1[1] + m2 * r2[1]
                rr = wp.sqrt(yx * yx + yy * yy)
                Wv = real(0.0)
                dW = real(0.0)
                for t_ in range(n_ub):
                    Rb = ub_R[t_]
                    if rr < Rb:
                        uu = Rb - rr
                        kk = ub_K[t_]
                        Wv += ub_B[t_] * wp.pow(uu, real(kk))
                        if kk >= 1:
                            dW -= ub_B[t_] * real(kk) * wp.pow(uu, real(kk - 1))
                Wv = Wv * inv_pi
                dWr = dW * inv_pi / wp.max(rr, real(1.0e-300))
                w0 += wt * m0 * Wv
                w1 += wt * m1 * Wv
                w2 += wt * m2 * Wv
                # grad_x W = -(W'/r) y
                gx0 += -wt * m0 * dWr * yx
                gy0 += -wt * m0 * dWr * yy
                gx1 += -wt * m1 * dWr * yx
                gy1 += -wt * m1 * dWr * yy
                gx2 += -wt * m2 * dWr * yx
                gy2 += -wt * m2 * dWr * yy
        wout[tid, 0] = w0
        wout[tid, 1] = w1
        wout[tid, 2] = w2
        gout[tid, 0] = gx0 / h
        gout[tid, 1] = gy0 / h
        gout[tid, 2] = gx1 / h
        gout[tid, 3] = gy1 / h
        gout[tid, 4] = gx2 / h
        gout[tid, 5] = gy2 / h
        return
    # ccw copy for the edge machinery
    p0 = r0
    p1 = r1
    p2 = r2
    if d2 < real(0.0):
        tmp = p0
        p0 = p2
        p2 = tmp
    z = vec3_t()
    s0 = vec3_t()
    s1 = vec3_t()
    n0 = vec3_t()
    n1 = vec3_t()
    t0 = vec3_t()
    t1 = vec3_t()
    vmax2 = real(0.0)
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
        if z[e] < real(0.0):
            neg = 1
        if z[e] == real(0.0):
            nzero += 1
    ind = real(0.0)
    if neg == 0:
        if nzero == 0:
            ind = real(1.0)
        elif nzero == 1:
            ind = real(0.5)
        else:
            for vi in range(3):
                zprev = z[(vi + 2) % 3]
                if z[vi] == real(0.0) and zprev == real(0.0):
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
                    if ang <= real(0.0):
                        ang = ang + real(6.283185307179586)
                    ind = ang / real(6.283185307179586)
    rin = radii[rin_idx]
    inside_in = vmax2 <= rin * rin
    ch_e = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))   # edge/value parts [x 1/pi]
    ch_i = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))   # indicator parts
    for t in range(n_e):
        gate_ok = (e_gate[t] == 0) or (e_gate[t] == 1 and not inside_in) or (e_gate[t] == 2 and inside_in)
        if gate_ok:
            R = radii[e_R[t]]
            inside = vmax2 <= R * R
            var_ok = (e_var[t] == 2) or (e_var[t] == 1 and inside) or (e_var[t] == 0 and not inside)
            if var_ok:
                acc = real(0.0)
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
                            acc += ni * _edge_integral(e_a[t], e_b[t], 0, lo, hi, z[e], n0[e], n1[e], t0[e], t1[e], cn, cc, e_c0[t], e_c1[t])
                ch_e[e_ch[t]] = ch_e[e_ch[t]] + acc
    for t in range(n_v):
        gate_ok = (v_gate[t] == 0) or (v_gate[t] == 1 and not inside_in) or (v_gate[t] == 2 and inside_in)
        if gate_ok:
            R = radii[v_R[t]]
            inside = vmax2 <= R * R
            var_ok = (v_var[t] == 2) or (v_var[t] == 1 and inside) or (v_var[t] == 0 and not inside)
            if var_ok:
                mR = v_mR[t]
                acc = real(0.0)
                acc_i = real(0.0)
                if inside:
                    for e in range(3):
                        poly = real(0.0)
                        for k in range(v_c0[t], v_c1[t]):
                            poly += cc[k] / real(cn[k] + 2) * _sdiff(0, cn[k], s0[e], s1[e], z[e])
                        acc += z[e] * poly
                else:
                    acc_i = real(2.0) * mR * ind
                    for e in range(3):
                        az = wp.abs(z[e])
                        if az < R:
                            L = _isqrt((R - az) * (R + az))
                            lo = wp.max(s0[e], -L)
                            hi = wp.min(s1[e], L)
                            if lo < hi:
                                poly = real(0.0)
                                for k in range(v_c0[t], v_c1[t]):
                                    poly += cc[k] / real(cn[k] + 2) * _sdiff(0, cn[k], lo, hi, z[e])
                                acc += z[e] * poly - mR * _dangle(z[e], lo, hi)
                ch_e[v_ch[t]] = ch_e[v_ch[t]] + acc
                ch_i[v_ch[t]] = ch_i[v_ch[t]] + acc_i
    c = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))
    for k in range(9):
        c[k] = inv_pi * ch_e[k] + ch_i[k]
    # weights (units of h): m = c[0..2];  g_alpha = (c[3],c[4]), (c[5],c[6]), (c[7],c[8])
    for k in range(3):
        wout[tid, k] = l[k] * c[0] + gx[k] * c[1] + gy[k] * c[2]
        gout[tid, 2 * k] = (l[k] * c[3] + gx[k] * c[5] + gy[k] * c[7]) / h
        gout[tid, 2 * k + 1] = (l[k] * c[4] + gx[k] * c[6] + gy[k] * c[8]) / h


def _wp_from(a, dtype, device, torch_dtype):
    import torch
    t = a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
    t = t.to(device=device.replace("cuda:0", "cuda:0") if device != "cpu" else "cpu", dtype=torch_dtype).contiguous()
    return wp.from_torch(t, dtype=dtype), t


def pair_weights(pair_q, pair_e, positions, supports, vertices, elements, kernel, device="cuda:0", plan=None, as_torch=False):
    """P1 weights for every (query, element) pair.  Inputs: torch tensors (on `device`, zero copy after a dtype cast) or numpy arrays, any float dtype.
    Returns (w (P,3), G (P,3,2)) float64 in physical units: numpy, or torch tensors on the device if `as_torch`."""
    import torch
    require_f64("warpbc.pair_weights (the finite-element pair kernel)")
    plan = plan or DevicePlan(kernel, device)
    P = len(pair_q)
    wout = torch.zeros((max(P, 1), 3), dtype=torch.float64, device=device)
    gout = torch.zeros((max(P, 1), 6), dtype=torch.float64, device=device)
    if P:
        keep = []
        wq, a = _wp_from(pair_q, wp.int32, device, torch.int32); keep.append(a)
        we, a = _wp_from(pair_e, wp.int32, device, torch.int32); keep.append(a)
        wpos, a = _wp_from(positions, vec2_t, device, torch.float64); keep.append(a)
        wsup, a = _wp_from(supports, real, device, torch.float64); keep.append(a)
        wv, a = _wp_from(vertices, vec2_t, device, torch.float64); keep.append(a)
        wel, a = _wp_from(elements, wp.vec3i, device, torch.int32); keep.append(a)
        wp.launch(_pair_weights_kernel, dim=P, device=device, inputs=[
            wq, we, wpos, wsup, wv, wel, plan.radii, plan.rin_idx, real(1 / np.pi),
            *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cn, plan.cc,
            plan.ubR, plan.ubK, plan.ubB, plan.nub, plan.gx, plan.gw, GAUSS_N, real(FAR_RATIO), real(TINY), len(plan.radii_host),
            wp.from_torch(wout, dtype=real), wp.from_torch(gout, dtype=real)])
        wp.synchronize_device(device)
    w, G = wout[:P], gout[:P].reshape(P, 3, 2)
    return (w, G) if as_torch else (w.cpu().numpy(), G.cpu().numpy())


# ================================================================================================ surface elements (single edges)
# A closed boundary needs no triangles: every channel is  (edge-local part)  +  (indicator of the body) x u,  the indicator being a BODY-level quantity.
# `edge_channels` returns the edge-local parts; `indicator_vector(kernel)` is u.  Only the compact-potential (truncated) forms are used: the inner
# potentials rely on the constant dropping out over a CLOSED polygon, which a single edge does not provide.


# ---- the Chebyshev-quadrature (opt-in `stable=`) route: the algorithm of `np2d stable=(nodes, panels)`, moved to Warp.
# The edge profile is compiled EXACTLY (Fractions) into its Chebyshev series on [0, R] (amplification ~1.2 instead of 1e2..2e4 for the
# monomial basis) and integrated by Gauss-Legendre on dyadic panels around the foot point; the angle term stays closed form.
@wp.func
def _clenshaw(cc: wp.array(dtype=real), c0: int, c1: int, x: real) -> real:
    """sum_k a_k T_k(x), a_k = cc[c0 + k], k = 0 .. c1 - c0 - 1."""
    b1 = real(0.0)
    b2 = real(0.0)
    for k in range(c1 - c0 - 1):
        t = real(2.0) * x * b1 - b2 + cc[c1 - 1 - k]
        b2 = b1
        b1 = t
    return x * b1 - b2 + cc[c0]


@wp.func
def _cheb_integral(a: int, b: int, lo: real, hi: real, z: real, R: real, n0: real, n1: real, t0: real, t1: real,
                   cc: wp.array(dtype=real), c0: int, c1: int, gx: wp.array(dtype=real), gw: wp.array(dtype=real),
                   nn: int, panels: int, gofs: int) -> real:
    """int_lo^hi y0^a y1^b P(r) ds  (y = z n + s t, r = sqrt(s^2 + z^2), P = the Chebyshev series cc[c0:c1] on [0, R]) by Gauss-Legendre
    (nn nodes per panel) on the dyadic panels of the chord around the foot point s = 0: per side, breakpoints 0, |z|, 2|z|, ..., 2^(panels-1)|z|
    (clipped to the side's chord end, last panel up to the end; empty panels skipped)."""
    az = wp.abs(z)
    tot = real(0.0)
    for side in range(2):
        sg = real(1.0)
        t_lo = wp.max(lo, real(0.0))
        t_hi = wp.max(hi, real(0.0))
        if side == 1:
            sg = real(-1.0)
            t_lo = wp.max(-hi, real(0.0))
            t_hi = wp.max(-lo, real(0.0))
        for j in range(panels + 1):
            ba = real(0.0)
            if j >= 1:
                ba = az * wp.pow(real(2.0), real(j - 1))
            bb = t_hi
            if j < panels:
                bb = az * wp.pow(real(2.0), real(j))        # Bs[j + 1] = |z| 2^j  (Bs[1] = |z|)
            pa = wp.clamp(ba, t_lo, t_hi)
            pb = wp.clamp(bb, t_lo, t_hi)
            if pb > pa:
                half = (pb - pa) / real(2.0)
                mid = (pa + pb) / real(2.0)
                for k in range(nn):
                    s = sg * (mid + half * gx[gofs + k])
                    r = wp.sqrt(s * s + z * z)
                    xx = wp.clamp(real(2.0) * r / R - real(1.0), real(-1.0), real(1.0))
                    y0 = z * n0 + s * t0
                    y1 = z * n1 + s * t1
                    m = real(1.0)
                    for _ in range(a):
                        m = m * y0
                    for _ in range(b):
                        m = m * y1
                    tot += half * gw[gofs + k] * m * _clenshaw(cc, c0, c1, xx)
    return tot


@wp.kernel
def _edge_channels_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                          pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                          verts: wp.array(dtype=vec2_t), edges: wp.array(dtype=wp.vec2i),
                          radii: wp.array(dtype=real), inv_pi: real,
                          e_ch: wp.array(dtype=int), e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                          e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                          e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                          v_ch: wp.array(dtype=int), v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                          v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=real), n_v: int,
                          cn: wp.array(dtype=int), cc: wp.array(dtype=real),
                          cout: wp.array2d(dtype=real)):
    tid = wp.tid()
    qi = pair_q[tid]
    ed = edges[pair_e[tid]]
    h = sup[qi]
    xv = pos[qi]
    p = (verts[ed[0]] - xv) / h
    q = (verts[ed[1]] - xv) / h
    d = q - p
    ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if ell == real(0.0):
        return
    t0 = d[0] / ell
    t1 = d[1] / ell
    n0 = t1                                   # outward (fluid-side) normal of a counter-clockwise boundary
    n1 = -t0
    z = (p[0] * q[1] - p[1] * q[0]) / ell     # positive when x lies on the solid side
    s0 = (d[0] * p[0] + d[1] * p[1]) / ell
    s1 = (d[0] * q[0] + d[1] * q[1]) / ell
    ch = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))
    az = wp.abs(z)
    for t in range(n_e):
        if (e_var[t] != 1) and (e_gate[t] != 2):
            R = radii[e_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    ni = n0
                    if e_i[t] == 1:
                        ni = n1
                    ch[e_ch[t]] = ch[e_ch[t]] + ni * _edge_integral(e_a[t], e_b[t], 0, lo, hi, z, n0, n1, t0, t1, cn, cc, e_c0[t], e_c1[t])
    for t in range(n_v):
        if (v_var[t] != 1) and (v_gate[t] != 2):
            R = radii[v_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    mR = v_mR[t]
                    poly = real(0.0)
                    for k in range(v_c0[t], v_c1[t]):
                        poly += cc[k] / real(cn[k] + 2) * _sdiff(0, cn[k], lo, hi, z)
                    ch[v_ch[t]] = ch[v_ch[t]] + z * poly - mR * _dangle(z, lo, hi)
    for k in range(9):
        cout[tid, k] = inv_pi * ch[k]


@wp.kernel
def _edge_channels_cheb_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                               pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                               verts: wp.array(dtype=vec2_t), edges: wp.array(dtype=wp.vec2i),
                               radii: wp.array(dtype=real), inv_pi: real,
                               e_ch: wp.array(dtype=int), e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                               e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                               e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                               v_ch: wp.array(dtype=int), v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                               v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=real), n_v: int,
                               cc: wp.array(dtype=real), gx: wp.array(dtype=real), gw: wp.array(dtype=real), nn: int, panels: int,
                               cout: wp.array2d(dtype=real)):
    """a copy of _edge_channels_kernel with the monomial `_edge_integral` replaced by the Chebyshev-quadrature `_cheb_integral`
    (the value term is `z * _cheb_integral(0, 0, ...) - mR * _dangle`); the chord clip, the gates, the n_i factor and the 1/pi are unchanged."""
    tid = wp.tid()
    qi = pair_q[tid]
    ed = edges[pair_e[tid]]
    h = sup[qi]
    xv = pos[qi]
    p = (verts[ed[0]] - xv) / h
    q = (verts[ed[1]] - xv) / h
    d = q - p
    ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if ell == real(0.0):
        return
    t0 = d[0] / ell
    t1 = d[1] / ell
    n0 = t1                                   # outward (fluid-side) normal of a counter-clockwise boundary
    n1 = -t0
    z = (p[0] * q[1] - p[1] * q[0]) / ell     # positive when x lies on the solid side
    s0 = (d[0] * p[0] + d[1] * p[1]) / ell
    s1 = (d[0] * q[0] + d[1] * q[1]) / ell
    ch = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))
    az = wp.abs(z)
    for t in range(n_e):
        if (e_var[t] != 1) and (e_gate[t] != 2):
            R = radii[e_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    ni = n0
                    if e_i[t] == 1:
                        ni = n1
                    ch[e_ch[t]] = ch[e_ch[t]] + ni * _cheb_integral(e_a[t], e_b[t], lo, hi, z, R, n0, n1, t0, t1, cc, e_c0[t], e_c1[t], gx, gw, nn, panels, 0)
    for t in range(n_v):
        if (v_var[t] != 1) and (v_gate[t] != 2):
            R = radii[v_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    poly = _cheb_integral(0, 0, lo, hi, z, R, n0, n1, t0, t1, cc, v_c0[t], v_c1[t], gx, gw, nn, panels, 0)
                    ch[v_ch[t]] = ch[v_ch[t]] + z * poly - v_mR[t] * _dangle(z, lo, hi)
    for k in range(9):
        cout[tid, k] = inv_pi * ch[k]


# ---- term-parallel evaluation (docs/plan-wall-evaluation.md step 1b): the sequential kernels above walk all ~21 terms of one pair in ONE thread and are latency-bound (the time is flat
# from 484 to 4 840 pairs).  Here one thread evaluates one (pair, term) and writes its contribution to scratch; a second kernel sums the terms of every pair in the SAME order as the
# sequential kernel (edge terms, then vertex terms, `(ch + a) - b` for the vertex ones), so the result is deterministic and equal to the sequential one up to the contraction of a*b + c
# into an FMA by the compiler (a few ulp).  Inactive terms (support does not reach the edge line / the chord is empty) store 0.
@wp.kernel
def _edge_terms_cheb_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                            pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                            verts: wp.array(dtype=vec2_t), edges: wp.array(dtype=wp.vec2i),
                            radii: wp.array(dtype=real),
                            e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                            e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                            e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                            v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                            v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=real), n_v: int,
                            cc: wp.array(dtype=real), gx: wp.array(dtype=real), gw: wp.array(dtype=real), nn: int, panels: int,
                            ta: wp.array(dtype=real), tb: wp.array(dtype=real)):
    tid = wp.tid()
    nT = n_e + n_v
    pr = tid / nT
    t = tid - pr * nT
    qi = pair_q[pr]
    ed = edges[pair_e[pr]]
    h = sup[qi]
    xv = pos[qi]
    p = (verts[ed[0]] - xv) / h
    q = (verts[ed[1]] - xv) / h
    d = q - p
    ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if ell == real(0.0):
        return
    t0 = d[0] / ell
    t1 = d[1] / ell
    n0 = t1
    n1 = -t0
    z = (p[0] * q[1] - p[1] * q[0]) / ell
    s0 = (d[0] * p[0] + d[1] * p[1]) / ell
    s1 = (d[0] * q[0] + d[1] * q[1]) / ell
    az = wp.abs(z)
    if t < n_e:
        if (e_var[t] != 1) and (e_gate[t] != 2):
            R = radii[e_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    ni = n0
                    if e_i[t] == 1:
                        ni = n1
                    ta[tid] = ni * _cheb_integral(e_a[t], e_b[t], lo, hi, z, R, n0, n1, t0, t1, cc, e_c0[t], e_c1[t], gx, gw, nn, panels, 0)
    else:
        u = t - n_e
        if (v_var[u] != 1) and (v_gate[u] != 2):
            R = radii[v_R[u]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    poly = _cheb_integral(0, 0, lo, hi, z, R, n0, n1, t0, t1, cc, v_c0[u], v_c1[u], gx, gw, nn, panels, 0)
                    ta[tid] = z * poly
                    tb[tid] = v_mR[u] * _dangle(z, lo, hi)


@wp.kernel
def _edge_terms_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                       pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                       verts: wp.array(dtype=vec2_t), edges: wp.array(dtype=wp.vec2i),
                       radii: wp.array(dtype=real),
                       e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                       e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                       e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                       v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                       v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=real), n_v: int,
                       cn: wp.array(dtype=int), cc: wp.array(dtype=real),
                       ta: wp.array(dtype=real), tb: wp.array(dtype=real)):
    """term-parallel twin of `_edge_channels_kernel` (the monomial plan): one thread per (pair, term), same arithmetic per term."""
    tid = wp.tid()
    nT = n_e + n_v
    pr = tid / nT
    t = tid - pr * nT
    qi = pair_q[pr]
    ed = edges[pair_e[pr]]
    h = sup[qi]
    xv = pos[qi]
    p = (verts[ed[0]] - xv) / h
    q = (verts[ed[1]] - xv) / h
    d = q - p
    ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if ell == real(0.0):
        return
    t0 = d[0] / ell
    t1 = d[1] / ell
    n0 = t1
    n1 = -t0
    z = (p[0] * q[1] - p[1] * q[0]) / ell
    s0 = (d[0] * p[0] + d[1] * p[1]) / ell
    s1 = (d[0] * q[0] + d[1] * q[1]) / ell
    az = wp.abs(z)
    if t < n_e:
        if (e_var[t] != 1) and (e_gate[t] != 2):
            R = radii[e_R[t]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    ni = n0
                    if e_i[t] == 1:
                        ni = n1
                    ta[tid] = ni * _edge_integral(e_a[t], e_b[t], 0, lo, hi, z, n0, n1, t0, t1, cn, cc, e_c0[t], e_c1[t])
    else:
        u = t - n_e
        if (v_var[u] != 1) and (v_gate[u] != 2):
            R = radii[v_R[u]]
            if az < R:
                L = _isqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    poly = real(0.0)
                    for k in range(v_c0[u], v_c1[u]):
                        poly += cc[k] / real(cn[k] + 2) * _sdiff(0, cn[k], lo, hi, z)
                    ta[tid] = z * poly
                    tb[tid] = v_mR[u] * _dangle(z, lo, hi)


@wp.kernel
def _reduce_terms_kernel(ta: wp.array(dtype=real), tb: wp.array(dtype=real), e_ch: wp.array(dtype=int), n_e: int,
                         v_ch: wp.array(dtype=int), n_v: int, inv_pi: real, cout: wp.array2d(dtype=real)):
    pr = wp.tid()
    nT = n_e + n_v
    ch = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))
    for t in range(n_e):
        ch[e_ch[t]] = ch[e_ch[t]] + ta[pr * nT + t]
    for u in range(n_v):
        ch[v_ch[u]] = ch[v_ch[u]] + ta[pr * nT + n_e + u] - tb[pr * nT + n_e + u]
    for k in range(9):
        cout[pr, k] = inv_pi * ch[k]


def indicator_vector(kernel):
    """u[9]: channel value of a body that contains the whole support (indicator 1) -- (1,0,0, 0,0, 1,0, 0,1): lambda = 1, g_(1,0)_x = g_(0,1)_y = lambda."""
    pb, _ = build_plan(kernel)
    u = np.zeros(NCH)
    for ch, R, var, gate, c0, c1, mR in [(v[0], v[1], v[2], v[3], v[4], v[5], v[6]) for v in pb.V]:
        if var != 1 and gate != 2:
            u[ch] += 2.0 * mR
    return u


# ---- the opt-in stable (Chebyshev-quadrature) registry: kernel name -> (nodes, panels).  Empty by default (the monomial plan stays the
# default of edge_channels); the tensile term (tensile.py) registers w2p5 / w4p5.  Chebyshev plans are cached per (kernel, device, nodes).
STABLE_KERNELS = {}
_Cheb_plan_cache = {}


def _cheb_plan_for(kernel, device, nodes, channels=None):
    key = (kernel, device, nodes, None if channels is None else tuple(sorted(set(int(c) for c in channels))))
    plan = _Cheb_plan_cache.get(key)
    if plan is None:
        plan = ChebPlan(kernel, device, nodes=nodes, channels=channels)
        _Cheb_plan_cache[key] = plan
    return plan


TERM_PARALLEL = "auto"          # "auto" | True | False: one thread per (pair, term) + a fixed-order reduction (below) instead of one thread per pair
TERM_PARALLEL_MAX_PAIRS = 20000   # "auto": below this many pairs the per-pair kernel does not fill the GPU (latency-bound), above it the saturated per-pair kernel wins (measured crossing 5e3 - 5e4 on a 188-SM RTX PRO 6000, 21 terms)


def _use_term_parallel(P, nT):
    if TERM_PARALLEL == "auto":
        return nT > 1 and P <= TERM_PARALLEL_MAX_PAIRS
    return bool(TERM_PARALLEL) and nT > 1


def _edge_channels_cheb(pair_q, pair_e, positions, supports, vertices, edges, kernel, device, nodes, panels, plan=None, as_torch=True, channels=None):
    """the Chebyshev-quadrature route of edge_channels (plan from `plan` or the (kernel, device, nodes) cache, launched with `panels`)."""
    import torch
    plan = plan if plan is not None else _cheb_plan_for(kernel, device, nodes, channels)
    P = len(pair_q)
    cout = torch.zeros((max(P, 1), NCH), dtype=torch_real, device=device)
    if P:
        keep = []
        wq, a = _wp_from(pair_q, wp.int32, device, torch.int32); keep.append(a)
        we, a = _wp_from(pair_e, wp.int32, device, torch.int32); keep.append(a)
        wpos, a = _wp_from(positions, vec2_t, device, torch_real); keep.append(a)
        wsup, a = _wp_from(supports, real, device, torch_real); keep.append(a)
        wv, a = _wp_from(vertices, vec2_t, device, torch_real); keep.append(a)
        wed, a = _wp_from(edges, wp.vec2i, device, torch.int32); keep.append(a)
        nT = plan.nE + plan.nV
        if _use_term_parallel(P, nT):
            ta = torch.zeros(P * nT, dtype=torch_real, device=device)
            tb = torch.zeros(P * nT, dtype=torch_real, device=device)
            wp.launch(_edge_terms_cheb_kernel, dim=P * nT, device=device, inputs=[
                wq, we, wpos, wsup, wv, wed, plan.radii,
                plan.e[1], plan.e[2], plan.e[3], plan.e[4], plan.e[5], plan.e[6], plan.e[7], plan.e[8], plan.nE,
                plan.v[1], plan.v[2], plan.v[3], plan.v[4], plan.v[5], plan.v_mR, plan.nV,
                plan.cc, plan.gx, plan.gw, nodes, panels, wp.from_torch(ta, dtype=real), wp.from_torch(tb, dtype=real)])
            wp.launch(_reduce_terms_kernel, dim=P, device=device, inputs=[
                wp.from_torch(ta, dtype=real), wp.from_torch(tb, dtype=real), plan.e[0], plan.nE, plan.v[0], plan.nV, real(1 / np.pi), wp.from_torch(cout, dtype=real)])
            keep.extend([ta, tb])
        else:
            wp.launch(_edge_channels_cheb_kernel, dim=P, device=device, inputs=[
                wq, we, wpos, wsup, wv, wed, plan.radii, real(1 / np.pi),
                *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV,
                plan.cc, plan.gx, plan.gw, nodes, panels, wp.from_torch(cout, dtype=real)])
        wp.synchronize_device(device)
    c = cout[:P].to(torch.float64)                                   # the kernel computed in `real`; the tensors handed on are float64
    return c if as_torch else c.cpu().numpy()


def edge_channels(pair_q, pair_e, positions, supports, vertices, edges, kernel, device="cuda:0", plan=None, as_torch=True, stable=None, channels=None):
    """edge-local channels c[P,9] (dimensionless, units of h) of every (query, edge) pair; edges [E,2] counter-clockwise around the solid (solid on the left).
    The total of a closed body is  sum_edges c + indicator * indicator_vector(kernel).  `stable` selects the route: a tuple (nodes, panels)
    -> the Chebyshev-quadrature plan at that resolution; False -> the monomial plan; None -> a passed ChebPlan / a STABLE_KERNELS entry, else the
    monomial plan (the default, unchanged).  `channels` (iterable of 0..8, default all): evaluate only the terms of these channels (the others stay 0; the kept ones are bit-identical)."""
    import torch
    if stable is None and plan is None and IS_F32 and STABLE_KERNELS.get(kernel) is None:
        stable = (8, 6)                                    # float32: the monomial plan amplifies rounding 1e2 - 2e4x; the Chebyshev route is the float32 default (stable=False forces the monomial one)
    if isinstance(stable, tuple):
        return _edge_channels_cheb(pair_q, pair_e, positions, supports, vertices, edges, kernel, device, stable[0], stable[1], as_torch=as_torch, channels=channels)
    if stable is None:
        if plan is not None and isinstance(plan, ChebPlan):
            return _edge_channels_cheb(pair_q, pair_e, positions, supports, vertices, edges, kernel, device, plan.nodes, plan.panels, plan=plan, as_torch=as_torch)
        if plan is None and STABLE_KERNELS.get(kernel) is not None:
            return _edge_channels_cheb(pair_q, pair_e, positions, supports, vertices, edges, kernel, device, *STABLE_KERNELS[kernel], as_torch=as_torch, channels=channels)
    # monomial path
    plan = plan or _device_plan(kernel, device, channels)
    P = len(pair_q)
    cout = torch.zeros((max(P, 1), NCH), dtype=torch_real, device=device)
    if P:
        keep = []
        wq, a = _wp_from(pair_q, wp.int32, device, torch.int32); keep.append(a)
        we, a = _wp_from(pair_e, wp.int32, device, torch.int32); keep.append(a)
        wpos, a = _wp_from(positions, vec2_t, device, torch_real); keep.append(a)
        wsup, a = _wp_from(supports, real, device, torch_real); keep.append(a)
        wv, a = _wp_from(vertices, vec2_t, device, torch_real); keep.append(a)
        wed, a = _wp_from(edges, wp.vec2i, device, torch.int32); keep.append(a)
        nT = plan.nE + plan.nV
        if _use_term_parallel(P, nT):
            ta = torch.zeros(P * nT, dtype=torch_real, device=device)
            tb = torch.zeros(P * nT, dtype=torch_real, device=device)
            wp.launch(_edge_terms_kernel, dim=P * nT, device=device, inputs=[
                wq, we, wpos, wsup, wv, wed, plan.radii,
                plan.e[1], plan.e[2], plan.e[3], plan.e[4], plan.e[5], plan.e[6], plan.e[7], plan.e[8], plan.nE,
                plan.v[1], plan.v[2], plan.v[3], plan.v[4], plan.v[5], plan.v_mR, plan.nV, plan.cn, plan.cc,
                wp.from_torch(ta, dtype=real), wp.from_torch(tb, dtype=real)])
            wp.launch(_reduce_terms_kernel, dim=P, device=device, inputs=[
                wp.from_torch(ta, dtype=real), wp.from_torch(tb, dtype=real), plan.e[0], plan.nE, plan.v[0], plan.nV, real(1 / np.pi), wp.from_torch(cout, dtype=real)])
            keep.extend([ta, tb])
        else:
            wp.launch(_edge_channels_kernel, dim=P, device=device, inputs=[
                wq, we, wpos, wsup, wv, wed, plan.radii, real(1 / np.pi),
                *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cn, plan.cc, wp.from_torch(cout, dtype=real)])
        wp.synchronize_device(device)
    c = cout[:P].to(torch.float64)                                   # the kernel computed in `real`; the tensors handed on are float64
    return c if as_torch else c.cpu().numpy()
