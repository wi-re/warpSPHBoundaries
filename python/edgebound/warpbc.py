"""Boundary integrals as Warp operations (2D, triangles): pair-weight engine (this file, part 1).

For every (query particle i, boundary element e) pair the P1 weights of the exact edge reduction are computed on the GPU/CPU:

    w_k  = int_e N_k W dA            (k = 0,1,2 vertex weights; sum_k w_k = int_e W)
    G_k  = int_e N_k grad_x W dA     (2-vectors; physical units, ~ 1/h)

from nine "channels" -- the moments m_(0,0), m_(1,0), m_(0,1) and g_(0,0), g_(1,0), g_(0,1) (g_alpha = int y^alpha grad_x W) -- via the exact compiled
plans of `np2d` (compact-potential recursion, inner potentials for elements inside the support, exact indicator/atan weights).
Arithmetic inside the kernel is float64 whatever the I/O dtype (boundary pairs are few: only particles within one support of a wall); see
docs/boundary-operations.md for the cost and the float32 discussion.

The operations (Interpolate, Gradient, Divergence, Curl, Density) are applied to these weights in `boundaryOperation` (part 2).
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
import warp as wp

from . import np2d
from .np2d import _compile_profile_moment, _inner_profile, compile_moment, kernel_profile
from .warp2d import _asinh, _binom, _dangle, _edge_integral, _Im, _isqrt, _S, _sdiff, _ypoly_coef, f64

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


@lru_cache(maxsize=None)
def build_plan(kernel):
    """flatten all nine channels of the P1 weights for `kernel` (exact rationals -> float64 constants)."""
    pb = _PlanBuilder(kernel)
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
    rin_idx = pb.R(Rin)
    return pb, rin_idx


class DevicePlan:
    """the plan arrays resident on one device."""

    def __init__(self, kernel, device):
        pb, rin_idx = build_plan(kernel)
        self.rin_idx = rin_idx
        self.nE, self.nV = len(pb.E), len(pb.V)
        i32 = lambda rows, k: wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=device)
        self.radii_host = list(pb.radii)
        self.radii = wp.array(np.array(pb.radii, dtype=np.float64), dtype=f64, device=device)
        self.cn = wp.array(np.array(pb.cn or [0], dtype=np.int32), dtype=int, device=device)
        self.cc = wp.array(np.array(pb.cc or [0.0], dtype=np.float64), dtype=f64, device=device)
        E, V = pb.E, pb.V
        self.e = [i32(E, k) for k in range(9)]          # ch, i, a, b, R, var, gate, c0, c1
        self.v = [i32(V, k) for k in range(6)]          # ch, R, var, gate, c0, c1 (+ mR below)
        self.v_mR = wp.array(np.array([r[6] for r in V] or [0.0], dtype=np.float64), dtype=f64, device=device)
        # u-basis of the kernel (pi W = sum b_k (R - r)^k per block), for the far-field Gauss branch
        from .np_fem import kernel_ubasis
        ubR, ubK, ubB = [], [], []
        for R, d in kernel_ubasis(kernel).items():
            for k, b in d.items():
                ubR.append(float(R)); ubK.append(int(k)); ubB.append(float(b))
        self.nub = len(ubR)
        self.ubR = wp.array(np.array(ubR, dtype=np.float64), dtype=f64, device=device)
        self.ubK = wp.array(np.array(ubK, dtype=np.int32), dtype=int, device=device)
        self.ubB = wp.array(np.array(ubB, dtype=np.float64), dtype=f64, device=device)
        gx, gw = np.polynomial.legendre.leggauss(GAUSS_N)
        self.gx = wp.array((gx + 1) / 2, dtype=f64, device=device)
        self.gw = wp.array(gw / 2, dtype=f64, device=device)


# ------------------------------------------------------------------------------------------------ the pair kernel
@wp.kernel
def _pair_weights_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                         pos: wp.array(dtype=wp.vec2d), sup: wp.array(dtype=f64),
                         verts: wp.array(dtype=wp.vec2d), elems: wp.array(dtype=wp.vec3i),
                         radii: wp.array(dtype=f64), rin_idx: int, inv_pi: f64,
                         e_ch: wp.array(dtype=int), e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                         e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                         e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                         v_ch: wp.array(dtype=int), v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                         v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=f64), n_v: int,
                         cn: wp.array(dtype=int), cc: wp.array(dtype=f64),
                         ub_R: wp.array(dtype=f64), ub_K: wp.array(dtype=int), ub_B: wp.array(dtype=f64), n_ub: int,
                         gnode: wp.array(dtype=f64), gwt: wp.array(dtype=f64), n_g: int, far_ratio: f64, tiny: f64, n_radii: int,
                         wout: wp.array2d(dtype=f64), gout: wp.array2d(dtype=f64)):
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
    l = wp.vec3d()
    gx = wp.vec3d()
    gy = wp.vec3d()
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
    cxm = (r0[0] + r1[0] + r2[0]) / f64(3.0)
    cym = (r0[1] + r1[1] + r2[1]) / f64(3.0)
    rho = wp.sqrt(cxm * cxm + cym * cym)
    emax = f64(0.0)
    rmin = f64(1.0e300)
    rmax = f64(0.0)
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
        tt = wp.clamp(-(pa[0] * dd[0] + pa[1] * dd[1]) / (ln * ln), f64(0.0), f64(1.0))
        qx = pa[0] + tt * dd[0]
        qy = pa[1] + tt * dd[1]
        rmin = wp.min(rmin, wp.sqrt(qx * qx + qy * qy))
        rmax = wp.max(rmax, wp.sqrt(pa[0] * pa[0] + pa[1] * pa[1]))
    straddle = int(0)
    for ri in range(n_radii):
        if rmin < radii[ri] and radii[ri] < rmax:
            straddle = 1
    if rho >= far_ratio * emax and (straddle == 0 or emax <= tiny):
        w0 = f64(0.0)
        w1 = f64(0.0)
        w2 = f64(0.0)
        gx0 = f64(0.0)
        gy0 = f64(0.0)
        gx1 = f64(0.0)
        gy1 = f64(0.0)
        gx2 = f64(0.0)
        gy2 = f64(0.0)
        area2 = wp.abs(d2)
        for ia in range(n_g):
            for ib in range(n_g):
                ua = gnode[ia]
                vb = gnode[ib]
                m1 = ua
                m2 = vb * (f64(1.0) - ua)
                m0 = f64(1.0) - m1 - m2
                wt = gwt[ia] * gwt[ib] * (f64(1.0) - ua) * area2
                yx = m0 * r0[0] + m1 * r1[0] + m2 * r2[0]
                yy = m0 * r0[1] + m1 * r1[1] + m2 * r2[1]
                rr = wp.sqrt(yx * yx + yy * yy)
                Wv = f64(0.0)
                dW = f64(0.0)
                for t_ in range(n_ub):
                    Rb = ub_R[t_]
                    if rr < Rb:
                        uu = Rb - rr
                        kk = ub_K[t_]
                        Wv += ub_B[t_] * wp.pow(uu, f64(kk))
                        if kk >= 1:
                            dW -= ub_B[t_] * f64(kk) * wp.pow(uu, f64(kk - 1))
                Wv = Wv * inv_pi
                dWr = dW * inv_pi / wp.max(rr, f64(1.0e-300))
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
    if d2 < f64(0.0):
        tmp = p0
        p0 = p2
        p2 = tmp
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
    rin = radii[rin_idx]
    inside_in = vmax2 <= rin * rin
    ch_e = wp.vector(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0))   # edge/value parts [x 1/pi]
    ch_i = wp.vector(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0))   # indicator parts
    for t in range(n_e):
        gate_ok = (e_gate[t] == 0) or (e_gate[t] == 1 and not inside_in) or (e_gate[t] == 2 and inside_in)
        if gate_ok:
            R = radii[e_R[t]]
            inside = vmax2 <= R * R
            var_ok = (e_var[t] == 2) or (e_var[t] == 1 and inside) or (e_var[t] == 0 and not inside)
            if var_ok:
                acc = f64(0.0)
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
                acc = f64(0.0)
                acc_i = f64(0.0)
                if inside:
                    for e in range(3):
                        poly = f64(0.0)
                        for k in range(v_c0[t], v_c1[t]):
                            poly += cc[k] / f64(cn[k] + 2) * _sdiff(0, cn[k], s0[e], s1[e], z[e])
                        acc += z[e] * poly
                else:
                    acc_i = f64(2.0) * mR * ind
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
                                acc += z[e] * poly - mR * _dangle(z[e], lo, hi)
                ch_e[v_ch[t]] = ch_e[v_ch[t]] + acc
                ch_i[v_ch[t]] = ch_i[v_ch[t]] + acc_i
    c = wp.vector(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0))
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
    plan = plan or DevicePlan(kernel, device)
    P = len(pair_q)
    wout = torch.zeros((max(P, 1), 3), dtype=torch.float64, device=device)
    gout = torch.zeros((max(P, 1), 6), dtype=torch.float64, device=device)
    if P:
        keep = []
        wq, a = _wp_from(pair_q, wp.int32, device, torch.int32); keep.append(a)
        we, a = _wp_from(pair_e, wp.int32, device, torch.int32); keep.append(a)
        wpos, a = _wp_from(positions, wp.vec2d, device, torch.float64); keep.append(a)
        wsup, a = _wp_from(supports, f64, device, torch.float64); keep.append(a)
        wv, a = _wp_from(vertices, wp.vec2d, device, torch.float64); keep.append(a)
        wel, a = _wp_from(elements, wp.vec3i, device, torch.int32); keep.append(a)
        wp.launch(_pair_weights_kernel, dim=P, device=device, inputs=[
            wq, we, wpos, wsup, wv, wel, plan.radii, plan.rin_idx, f64(1 / np.pi),
            *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cn, plan.cc,
            plan.ubR, plan.ubK, plan.ubB, plan.nub, plan.gx, plan.gw, GAUSS_N, f64(FAR_RATIO), f64(TINY), len(plan.radii_host),
            wp.from_torch(wout, dtype=f64), wp.from_torch(gout, dtype=f64)])
        wp.synchronize_device(device)
    w, G = wout[:P], gout[:P].reshape(P, 3, 2)
    return (w, G) if as_torch else (w.cpu().numpy(), G.cpu().numpy())
