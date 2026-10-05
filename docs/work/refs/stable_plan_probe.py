"""Reviewer's probe for WORK-004 (throw-away, NOT the deliverable; read it, write your own code, do not import it).

A Chebyshev-quadrature variant of `warpbc._edge_channels_kernel` ("stable plan"):
  host  : the same term list as `warpbc.build_plan` (edge terms (ch, i, a, b, R, var, gate), value terms (ch, R, var, gate, mR)), but each
          profile polynomial P (edge terms) / Q = P/(n+2) (value terms) is compiled EXACTLY (Fractions) into its Chebyshev coefficients on [0, R]
          (`np2d.cheb_coeffs`) before it is converted to float64;
  device: per (query, edge) pair, per term: Gauss-Legendre nodes (NODES per panel) on dyadic panels of the chord [lo, hi] clipped to R, around the
          foot point s = 0 (breakpoints 0, |z|, 2|z|, ..., 2^(PANELS-1)|z| on each side, the last panel up to the chord end; empty panels skipped),
          Clenshaw evaluation of the profile at x = clamp(2 r / R - 1, -1, 1); edge term  n_i * sum w * y0^a * y1^b * P(r);
          value term  z * (sum w * Q(r)) - mR * _dangle(z, lo, hi)   (the angle term stays closed form, exactly as in the monomial kernel).
It is the algorithm of `np2d` with `stable=(NODES, PANELS)` = (16, 8), moved to Warp.  Everything else (the chord clip, the gates, the indicator vector,
the 1/pi) is the monomial kernel's.

usage:  python stable_plan_probe.py            (accuracy vs the mpmath reference for w2p5 / w4p5 on a unit square, 214 points, + timing)
Reviewer's result 2026-10-03: see the numbers printed by the script (copied into WORK-004.md).
"""
import sys, time, math
from fractions import Fraction
sys.path.insert(0, "python")
import numpy as np, torch, warp as wp, mpmath as mp

from edgebound.edge import kernels, warpbc, np2d, geometry as G
from edgebound.edge.core import GUARD, block_grad
from edgebound.q2_conditioning import terms
from edgebound.edge.warp2d import _dangle, f64

NODES, PANELS = 16, 8


# ------------------------------------------------------------------------------------------------ host plan
class _ChebBuilder(warpbc._PlanBuilder):
    """same term list as _PlanBuilder; the coefficient ranges hold CHEBYSHEV coefficients (exact compile, then float)."""
    def _cheb(self, P, R, scale):
        a = np2d.cheb_coeffs(P, R)
        c0 = len(self.cc)
        self.cc.extend(float(x) * scale for x in a)
        return c0, len(self.cc)

    def edge(self, ch, i, beta, R, var, gate, P, scale=1.0):
        Ri = self.R(R)
        self.E.append((ch, i, beta[0], beta[1], Ri, var, gate, *self._cheb(P, R, scale)))

    def value(self, ch, R, var, gate, P, scale=1.0):
        mR = sum(c * Fraction(R) ** (n + 2) / (n + 2) for n, c in P.items())
        Q = {n: c / (n + 2) for n, c in P.items()}
        self.V.append((ch, self.R(R), var, gate, *self._cheb(Q, R, scale), float(mR) * scale))


def build_stable_plan(kernel):
    """build_plan with the Chebyshev builder (only the terms the surface-element kernel uses are needed, but keep all: indicator_vector reads V)."""
    pb = _ChebBuilder(kernel)
    for ch, al in enumerate(warpbc.ALPHAS):
        pb.moment_plan(ch, al, 1.0, 0)
    Q, Rin = np2d._inner_profile(kernel)
    for k, al in enumerate(warpbc.ALPHAS):
        for j in (0, 1):
            ch = 3 + 2 * k + j
            for R, P in np2d.kernel_profile(kernel).items():
                pb.edge(ch, j, al, R, 2, 1, P, -1.0)
            if al[j]:
                beta = (al[0] - (j == 0), al[1] - (j == 1))
                pb.moment_plan(ch, beta, float(al[j]), 1)
            pb.edge(ch, j, al, Rin, 2, 2, Q, -1.0)
            if al[j]:
                beta = (al[0] - (j == 0), al[1] - (j == 1))
                pb.profile_moment_plan(ch, Q, Rin, beta, float(al[j]), 2)
    return pb


class StablePlan:
    def __init__(self, kernel, device):
        pb = build_stable_plan(kernel)
        self.nE, self.nV = len(pb.E), len(pb.V)
        i32 = lambda rows, k: wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=device)
        self.radii = wp.array(np.array(pb.radii, dtype=np.float64), dtype=f64, device=device)
        self.cc = wp.array(np.array(pb.cc or [0.0], dtype=np.float64), dtype=f64, device=device)
        self.e = [i32(pb.E, k) for k in range(9)]
        self.v = [i32(pb.V, k) for k in range(6)]
        self.v_mR = wp.array(np.array([r[6] for r in pb.V] or [0.0], dtype=np.float64), dtype=f64, device=device)
        gx, gw = np.polynomial.legendre.leggauss(NODES)
        self.gx = wp.array(gx, dtype=f64, device=device)       # on [-1, 1]
        self.gw = wp.array(gw, dtype=f64, device=device)


# ------------------------------------------------------------------------------------------------ device
@wp.func
def _clenshaw(cc: wp.array(dtype=f64), c0: int, c1: int, x: f64) -> f64:
    """sum_k a_k T_k(x), a_k = cc[c0 + k], k = 0 .. c1 - c0 - 1."""
    b1 = f64(0.0)
    b2 = f64(0.0)
    for k in range(c1 - c0 - 1):
        t = f64(2.0) * x * b1 - b2 + cc[c1 - 1 - k]
        b2 = b1
        b1 = t
    return x * b1 - b2 + cc[c0]


@wp.func
def _stable_integral(a: int, b: int, lo: f64, hi: f64, z: f64, R: f64, n0: f64, n1: f64, t0: f64, t1: f64,
                     cc: wp.array(dtype=f64), c0: int, c1: int, gx: wp.array(dtype=f64), gw: wp.array(dtype=f64), nn: int, panels: int) -> f64:
    """int_lo^hi y0^a y1^b P(r) ds  (y = z n + s t, r = sqrt(s^2 + z^2), P = the Chebyshev series cc[c0:c1] on [0, R]) by dyadic Gauss panels."""
    az = wp.abs(z)
    tot = f64(0.0)
    for side in range(2):
        sg = f64(1.0)
        t_lo = wp.max(lo, f64(0.0))
        t_hi = wp.max(hi, f64(0.0))
        if side == 1:
            sg = f64(-1.0)
            t_lo = wp.max(-hi, f64(0.0))
            t_hi = wp.max(-lo, f64(0.0))
        for j in range(panels + 1):
            ba = f64(0.0)
            if j >= 1:
                ba = az * wp.pow(f64(2.0), f64(j - 1))
            bb = t_hi
            if j < panels:
                bb = az * wp.pow(f64(2.0), f64(j))        # Bs[j + 1] = az 2^j  (Bs[1] = az)
            pa = wp.clamp(ba, t_lo, t_hi)
            pb = wp.clamp(bb, t_lo, t_hi)
            if pb > pa:
                half = (pb - pa) / f64(2.0)
                mid = (pa + pb) / f64(2.0)
                for k in range(nn):
                    s = sg * (mid + half * gx[k])
                    r = wp.sqrt(s * s + z * z)
                    xx = wp.clamp(f64(2.0) * r / R - f64(1.0), f64(-1.0), f64(1.0))
                    y0 = z * n0 + s * t0
                    y1 = z * n1 + s * t1
                    m = f64(1.0)
                    for _ in range(a):
                        m = m * y0
                    for _ in range(b):
                        m = m * y1
                    tot += half * gw[k] * m * _clenshaw(cc, c0, c1, xx)
    return tot


@wp.kernel
def _edge_channels_stable_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                                 pos: wp.array(dtype=wp.vec2d), sup: wp.array(dtype=f64),
                                 verts: wp.array(dtype=wp.vec2d), edges: wp.array(dtype=wp.vec2i),
                                 radii: wp.array(dtype=f64), inv_pi: f64,
                                 e_ch: wp.array(dtype=int), e_i: wp.array(dtype=int), e_a: wp.array(dtype=int), e_b: wp.array(dtype=int),
                                 e_R: wp.array(dtype=int), e_var: wp.array(dtype=int), e_gate: wp.array(dtype=int),
                                 e_c0: wp.array(dtype=int), e_c1: wp.array(dtype=int), n_e: int,
                                 v_ch: wp.array(dtype=int), v_R: wp.array(dtype=int), v_var: wp.array(dtype=int), v_gate: wp.array(dtype=int),
                                 v_c0: wp.array(dtype=int), v_c1: wp.array(dtype=int), v_mR: wp.array(dtype=f64), n_v: int,
                                 cc: wp.array(dtype=f64), gx: wp.array(dtype=f64), gw: wp.array(dtype=f64), nn: int, panels: int,
                                 cout: wp.array2d(dtype=f64)):
    tid = wp.tid()
    qi = pair_q[tid]
    ed = edges[pair_e[tid]]
    h = sup[qi]
    xv = pos[qi]
    p = (verts[ed[0]] - xv) / h
    q = (verts[ed[1]] - xv) / h
    d = q - p
    ell = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if ell == f64(0.0):
        return
    t0 = d[0] / ell
    t1 = d[1] / ell
    n0 = t1
    n1 = -t0
    z = (p[0] * q[1] - p[1] * q[0]) / ell
    s0 = (d[0] * p[0] + d[1] * p[1]) / ell
    s1 = (d[0] * q[0] + d[1] * q[1]) / ell
    ch = wp.vector(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0))
    az = wp.abs(z)
    for t in range(n_e):
        if (e_var[t] != 1) and (e_gate[t] != 2):
            R = radii[e_R[t]]
            if az < R:
                L = wp.sqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    ni = n0
                    if e_i[t] == 1:
                        ni = n1
                    ch[e_ch[t]] = ch[e_ch[t]] + ni * _stable_integral(e_a[t], e_b[t], lo, hi, z, R, n0, n1, t0, t1, cc, e_c0[t], e_c1[t], gx, gw, nn, panels)
    for t in range(n_v):
        if (v_var[t] != 1) and (v_gate[t] != 2):
            R = radii[v_R[t]]
            if az < R:
                L = wp.sqrt((R - az) * (R + az))
                lo = wp.max(s0, -L)
                hi = wp.min(s1, L)
                if lo < hi:
                    poly = _stable_integral(0, 0, lo, hi, z, R, n0, n1, t0, t1, cc, v_c0[t], v_c1[t], gx, gw, nn, panels)
                    ch[v_ch[t]] = ch[v_ch[t]] + z * poly - v_mR[t] * _dangle(z, lo, hi)
    for k in range(9):
        cout[tid, k] = inv_pi * ch[k]


def stable_edge_channels(pair_q, pair_e, positions, supports, vertices, edges, kernel, device="cuda:0", plan=None):
    plan = plan or StablePlan(kernel, device)
    P = len(pair_q)
    cout = torch.zeros((max(P, 1), warpbc.NCH), dtype=torch.float64, device=device)
    keep = []
    W = lambda a, dt, td: warpbc._wp_from(a, dt, device, td)
    wq, a = W(pair_q, wp.int32, torch.int32); keep.append(a)
    we, a = W(pair_e, wp.int32, torch.int32); keep.append(a)
    wpos, a = W(positions, wp.vec2d, torch.float64); keep.append(a)
    wsup, a = W(supports, f64, torch.float64); keep.append(a)
    wv, a = W(vertices, wp.vec2d, torch.float64); keep.append(a)
    wed, a = W(edges, wp.vec2i, torch.int32); keep.append(a)
    wp.launch(_edge_channels_stable_kernel, dim=P, device=device, inputs=[
        wq, we, wpos, wsup, wv, wed, plan.radii, f64(1 / np.pi), *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV,
        plan.cc, plan.gx, plan.gw, NODES, PANELS, wp.from_torch(cout, dtype=f64)])
    wp.synchronize_device(device)
    return cout[:P]


# ------------------------------------------------------------------------------------------------ the test bed (as q2_probe.py / q2_conditioning.py)
POLY = [(0, 0), (1, 0), (1, 1), (0, 1)]


def ref_g0(nm, fam, x, k):
    """exact mpmath reference of the gradient channel g0 of the kernel `nm` = W^k (units: C/pi * grad int shape), as q2_probe.ref_g0."""
    from math import comb
    C = kernels.KERNELS[nm].c2_pi
    dg = {"w2": 5, "w4": 8}[fam] * k
    with mp.workdps(150 + GUARD):
        P = G.prepare(POLY, x, h=1)
        gs = [block_grad(P, j, 1) for j in range(dg + 1)]
        tot = [mp.mpf(0), mp.mpf(0)]
        for c, _, p in terms(k, fam):
            for j in range(p + 1):
                co = c * comb(p, j) * (-1) ** j
                tot[0] += co * gs[j][0]
                tot[1] += co * gs[j][1]
        return np.array([float(tot[0]), float(tot[1])]) * float(C) / math.pi


def main():
    dev = "cuda:0"
    rng = np.random.default_rng(5)
    X = np.vstack([np.array([(0.3, 0.4), (1.02, 0.5), (0.5, -0.3), (0.97, 0.03), (1.3, 0.5), (0.5, -0.9)]),
                   rng.uniform(-0.5, 1.5, (200, 2)), np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)], float)])
    n = len(X)
    V = torch.tensor(POLY, dtype=torch.float64, device=dev)
    E = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=torch.int32, device=dev)
    P = torch.tensor(X, dtype=torch.float64, device=dev)
    qi = torch.arange(n).repeat_interleave(4).to(torch.int32).to(dev)
    ee = torch.arange(4).repeat(n).to(torch.int32).to(dev)
    sup = torch.ones(n, dtype=torch.float64, device=dev)
    for fam in ("w2", "w4"):
        for k in (1, 3, 5):
            nm = f"{fam}p{k}"
            kernels.KERNELS[nm] = kernels._from_terms(nm, terms(k, fam))
            t0 = time.time(); ref = np.array([ref_g0(nm, fam, x, k) for x in X]); tref = time.time() - t0
            sc = np.abs(ref).max()
            t0 = time.time(); plan = StablePlan(nm, dev); tplan = time.time() - t0
            c = stable_edge_channels(qi, ee, P, sup, V, E, nm, device=dev, plan=plan)
            g = torch.zeros((n, 2), dtype=torch.float64, device=dev).index_add_(0, qi.long(), c[:, 3:5]).cpu().numpy()
            err = np.abs(g - ref).max(axis=1)
            gm = warpbc.edge_channels(qi, ee, P, sup, V, E, nm, device=dev)
            gm = torch.zeros((n, 2), dtype=torch.float64, device=dev).index_add_(0, qi.long(), gm[:, 3:5]).cpu().numpy()
            errm = np.abs(gm - ref).max(axis=1)
            np_stable = np2d.gradient(np.repeat(np.asarray(POLY, float)[None], n, 0), X, nm, h=1, dtype=np.float64, stable=(16, 8))
            print(f"{nm:5s} scale={sc:.3e}  stable-warp worst/scale={err.max()/sc:.2e} (worst point {X[err.argmax()]})  monomial-warp {errm.max()/sc:.2e}  "
                  f"np2d stable(16,8) {np.abs(np_stable-ref).max()/sc:.2e}   plan {tplan:.2f}s nE={plan.nE} nV={plan.nV}  (ref {tref:.1f}s)", flush=True)
    # timing: 100k pairs
    nm = "w4p5"
    big = torch.tensor(np.random.default_rng(1).uniform(-0.3, 1.3, (50000, 2)), dtype=torch.float64, device=dev)
    nb = len(big)
    qi = torch.arange(nb).repeat_interleave(4).to(torch.int32).to(dev); ee = torch.arange(4).repeat(nb).to(torch.int32).to(dev)
    sb = torch.ones(nb, dtype=torch.float64, device=dev)
    plan = StablePlan(nm, dev)
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        stable_edge_channels(qi, ee, big, sb, V, E, nm, device=dev, plan=plan); torch.cuda.synchronize(); ts = time.time() - t0
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        warpbc.edge_channels(qi, ee, big, sb, V, E, nm, device=dev); torch.cuda.synchronize(); tm = time.time() - t0
    print(f"timing w4p5, {len(qi)} (query, edge) pairs (all pairs, many outside the support): stable {ts*1e3:.1f} ms   monomial {tm*1e3:.1f} ms")


if __name__ == "__main__":
    main()
