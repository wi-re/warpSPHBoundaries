"""Several kernels' edge channels in ONE launch (docs/plan-wall-evaluation.md step 3a).

The solver needs, at one position set, the channels of up to four kernels over the same (query, edge) pairs: `w2` (lam, G, Cov, A), `lw2` (wall Laplacian), `cone` (cover vector,
channels 3, 4) and `w2p5` (tensile term, channels 3, 4).  `warpbc.edge_channels` evaluates them in four launches with four sets of host-side glue.  Here the term tables of all kernels are
concatenated into one plan (`FusedPlan`; the monomial and the Chebyshev routes in one table, per term a route tag, the Gauss-Legendre table offset and the panel count), one thread
evaluates one (pair, term) and a second kernel sums, per pair and per group, in the order of the single-kernel path (edge terms, then vertex terms `(ch + a) - b`): the result of every
group equals `edge_channels(..., kernel, channels)` up to the FMA contraction, deterministic, no atomics.

    groups = (FusedGroup("w2"), FusedGroup("lw2"), FusedGroup("cone", (3, 4)), FusedGroup("w2p5", (3, 4)))
    c = fused_channels(pair_q, pair_e, lpos, lsup, vertices, edges, groups, device)          # [P, len(groups) * 9], in the active precision, units of h

The route of a group is the one `edge_channels` would pick: the Chebyshev (nodes, panels) of `STABLE_KERNELS` for registered kernels, `(8, 6)` for every kernel in float32, else monomial.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import warp as wp

from . import warpbc as W
from .precision import IS_F32, IS_F64, np_real, real, torch_real, vec2_t
from .warp2d import _dangle, _edge_integral, _isqrt, _sdiff

wp.config.quiet = True


@dataclass(frozen=True)
class FusedGroup:
    """one kernel of the fused launch: `kernel` (registered name), `channels` (None = all nine), the route and its Chebyshev resolution (None: chosen as `edge_channels` does)."""
    kernel: str
    channels: Optional[tuple] = None
    route: Optional[str] = None          # "mono" | "cheb"
    nodes: Optional[int] = None
    panels: Optional[int] = None

    def resolved(self):
        route, nodes, panels = self.route, self.nodes, self.panels
        if route is None:
            reg = W.STABLE_KERNELS.get(self.kernel)
            if reg is not None:
                route, (nodes, panels) = "cheb", reg
            elif IS_F32:
                route, nodes, panels = "cheb", 8, 6
            else:
                route = "mono"
        if route == "cheb" and nodes is None:
            nodes, panels = 8, 6
        if route == "mono" and not IS_F64:
            raise NotImplementedError("the monomial plan is float64 only (active precision %s): use the Chebyshev route" % real.__name__)
        return FusedGroup(self.kernel, None if self.channels is None else tuple(sorted(set(int(c) for c in self.channels))), route, nodes, panels)


class FusedPlan:
    """the concatenated term tables of a tuple of groups on one device."""

    def __init__(self, groups, device):
        groups = tuple(g.resolved() for g in groups)
        self.groups = groups
        radii, cn, cc, gxs, gws = [], [], [], [], []
        rows = []                                              # per term: dict
        goff = [0]
        gofs_of = {}
        for g in groups:
            pb, _ = W.build_plan(g.kernel) if g.route == "mono" else W.build_cheb_plan(g.kernel)
            E, V = W._channel_rows(pb, g.channels)
            r0, c0_ = len(radii), len(cc)
            radii.extend(pb.radii)
            cc.extend(pb.cc)
            cn.extend(pb.cn if g.route == "mono" else [0] * len(pb.cc))
            gofs = 0
            if g.route == "cheb":
                if g.nodes not in gofs_of:
                    gx, gw = np.polynomial.legendre.leggauss(g.nodes)
                    gofs_of[g.nodes] = sum(len(a) for a in gxs)
                    gxs.append(gx); gws.append(gw)
                gofs = gofs_of[g.nodes]
            tag = 0 if g.route == "mono" else 1
            for (ch, i, a, b, R, var, gate, c0, c1) in E:
                rows.append((0, ch, i, a, b, r0 + R, var, gate, c0_ + c0, c0_ + c1, 0.0, tag, g.nodes or 0, g.panels or 0, gofs))
            for (ch, R, var, gate, c0, c1, mR) in V:
                rows.append((1, ch, 0, 0, 0, r0 + R, var, gate, c0_ + c0, c0_ + c1, mR, tag, g.nodes or 0, g.panels or 0, gofs))
            goff.append(len(rows))
        self.nT = len(rows)
        self.nG = len(groups)
        i32 = lambda k: wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=device)
        self.t_kind, self.t_ch, self.t_i, self.t_a, self.t_b, self.t_R, self.t_var, self.t_gate, self.t_c0, self.t_c1 = (i32(k) for k in range(10))
        self.t_mR = wp.array(np.array([r[10] for r in rows] or [0.0], dtype=np_real), dtype=real, device=device)
        self.t_route, self.t_nn, self.t_panels, self.t_gofs = (i32(k) for k in range(11, 15))
        self.goff = wp.array(np.array(goff, dtype=np.int32), dtype=int, device=device)
        self.radii = wp.array(np.array(radii or [0.0], dtype=np_real), dtype=real, device=device)
        self.cn = wp.array(np.array(cn or [0], dtype=np.int32), dtype=int, device=device)
        self.cc = wp.array(np.array(cc or [0.0], dtype=np_real), dtype=real, device=device)
        self.gx = wp.array(np.concatenate(gxs).astype(np_real) if gxs else np.zeros(1, dtype=np_real), dtype=real, device=device)
        self.gw = wp.array(np.concatenate(gws).astype(np_real) if gws else np.zeros(1, dtype=np_real), dtype=real, device=device)


_PLANS = {}


def fused_plan(groups, device):
    key = (tuple(groups), str(device), real.__name__)
    if key not in _PLANS:
        _PLANS[key] = FusedPlan(groups, str(device))
    return _PLANS[key]


@wp.kernel
def _fused_terms_kernel(pair_q: wp.array(dtype=int), pair_e: wp.array(dtype=int),
                        pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                        verts: wp.array(dtype=vec2_t), edges: wp.array(dtype=wp.vec2i),
                        radii: wp.array(dtype=real),
                        t_kind: wp.array(dtype=int), t_i: wp.array(dtype=int), t_a: wp.array(dtype=int), t_b: wp.array(dtype=int),
                        t_R: wp.array(dtype=int), t_var: wp.array(dtype=int), t_gate: wp.array(dtype=int),
                        t_c0: wp.array(dtype=int), t_c1: wp.array(dtype=int), t_mR: wp.array(dtype=real),
                        t_route: wp.array(dtype=int), t_nn: wp.array(dtype=int), t_panels: wp.array(dtype=int), t_gofs: wp.array(dtype=int), nT: int,
                        cn: wp.array(dtype=int), cc: wp.array(dtype=real), gx: wp.array(dtype=real), gw: wp.array(dtype=real),
                        ta: wp.array(dtype=real), tb: wp.array(dtype=real)):
    tid = wp.tid()
    pr = tid / nT
    t = tid - pr * nT
    if (t_var[t] == 1) or (t_gate[t] == 2):
        return
    if pair_e[pr] < 0:                                   # an empty slot of a fixed-capacity adjacency
        return
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
    R = radii[t_R[t]]
    if az < R:
        L = _isqrt((R - az) * (R + az))
        lo = wp.max(s0, -L)
        hi = wp.min(s1, L)
        if lo < hi:
            if t_kind[t] == 0:
                ni = n0
                if t_i[t] == 1:
                    ni = n1
                val = real(0.0)
                if t_route[t] == 0:
                    val = _edge_integral(t_a[t], t_b[t], 0, lo, hi, z, n0, n1, t0, t1, cn, cc, t_c0[t], t_c1[t])
                else:
                    val = W._cheb_integral(t_a[t], t_b[t], lo, hi, z, R, n0, n1, t0, t1, cc, t_c0[t], t_c1[t], gx, gw, t_nn[t], t_panels[t], t_gofs[t])
                ta[tid] = ni * val
            else:
                poly = real(0.0)
                if t_route[t] == 0:
                    for k in range(t_c0[t], t_c1[t]):
                        poly += cc[k] / real(cn[k] + 2) * _sdiff(0, cn[k], lo, hi, z)
                else:
                    poly = W._cheb_integral(0, 0, lo, hi, z, R, n0, n1, t0, t1, cc, t_c0[t], t_c1[t], gx, gw, t_nn[t], t_panels[t], t_gofs[t])
                ta[tid] = z * poly
                tb[tid] = t_mR[t] * _dangle(z, lo, hi)


@wp.kernel
def _fused_reduce_kernel(ta: wp.array(dtype=real), tb: wp.array(dtype=real), t_kind: wp.array(dtype=int), t_ch: wp.array(dtype=int),
                         goff: wp.array(dtype=int), nG: int, nT: int, inv_pi: real, cout: wp.array2d(dtype=real)):
    pr = wp.tid()
    for g in range(nG):
        ch = wp.vector(real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0), real(0.0))
        for t in range(goff[g], goff[g + 1]):
            if t_kind[t] == 0:
                ch[t_ch[t]] = ch[t_ch[t]] + ta[pr * nT + t]
            else:
                ch[t_ch[t]] = ch[t_ch[t]] + ta[pr * nT + t] - tb[pr * nT + t]
        for k in range(9):
            cout[pr, g * 9 + k] = inv_pi * ch[k]


def fused_channels(pair_q, pair_e, positions, supports, vertices, edges, groups, device="cuda:0", as_float64=False):
    """channels [P, len(groups) * 9] (group g at columns 9 g .. 9 g + 8, dimensionless, units of h) of every (query, edge) pair for all `groups` in one launch family; the
    tensor is in the active precision (`as_float64=True` casts)."""
    plan = fused_plan(groups, device)
    P = len(pair_q)
    cout = torch.zeros((max(P, 1), plan.nG * 9), dtype=torch_real, device=device)
    if P and plan.nT:
        nT = plan.nT
        ta = torch.zeros(P * nT, dtype=torch_real, device=device)
        tb = torch.zeros(P * nT, dtype=torch_real, device=device)
        wq, _ = W._wp_from(pair_q, wp.int32, device, torch.int32)
        we, _ = W._wp_from(pair_e, wp.int32, device, torch.int32)
        wpos, _ = W._wp_from(positions, vec2_t, device, torch_real)
        wsup, _ = W._wp_from(supports, real, device, torch_real)
        wv, _ = W._wp_from(vertices, vec2_t, device, torch_real)
        wed, _ = W._wp_from(edges, wp.vec2i, device, torch.int32)
        wta, wtb = wp.from_torch(ta, dtype=real), wp.from_torch(tb, dtype=real)
        wp.launch(_fused_terms_kernel, dim=P * nT, device=device, inputs=[
            wq, we, wpos, wsup, wv, wed, plan.radii, plan.t_kind, plan.t_i, plan.t_a, plan.t_b, plan.t_R, plan.t_var, plan.t_gate, plan.t_c0, plan.t_c1, plan.t_mR,
            plan.t_route, plan.t_nn, plan.t_panels, plan.t_gofs, nT, plan.cn, plan.cc, plan.gx, plan.gw, wta, wtb])
        wp.launch(_fused_reduce_kernel, dim=P, device=device, inputs=[wta, wtb, plan.t_kind, plan.t_ch, plan.goff, plan.nG, nT, real(1 / np.pi), wp.from_torch(cout, dtype=real)])
        wp.synchronize_device(device)
    c = cout[:P]
    return c.to(torch.float64) if as_float64 else c
