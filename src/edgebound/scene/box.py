"""Corner tables for axis-aligned boxes (docs/box-domain-primitive.md): the moment channels of a radial kernel over a rectangle from four lookups of two 2D functions.

With y = x' - x (units of the support h) and the quadrant integrals

    Phi(a, b)  = int_{y1 <= a, y2 <= b} W(|y|) dy,        Phi1(a, b) = int_{y1 <= a, y2 <= b} y1 W(|y|) dy      (Phi2(a, b) = Phi1(b, a)),

a rectangle [a0, a1] x [b0, b1] (limits relative to the particle) integrates to the inclusion-exclusion of the four corners, and the nine channels of
`warpbc.edge_channels` are first derivatives of the two tables (by parts the boundary terms are 1D marginals = derivatives):

    lam = Phi,   m_d = Phi_d,   g0_j = -dPhi/da_j,   g1_dj = -[boundary_dj - delta_dj Phi],   boundary_dd = a_d dPhi/da_d,   boundary_21 = dPhi_2/da,   boundary_12 = dPhi_1/db.

The arguments are clamped to [-1, 1] (the support): Phi = 0 below, saturated above, so the far field is exact without a branch.  The tables are Chebyshev tensors on the
a geometrically graded panel grid toward the origin (the kernel's odd powers of r make Phi non-analytic at (0, 0)), built once per kernel by `BoxRep`.
"""
import numpy as np
import torch
import warp as wp
from numpy.polynomial import chebyshev as C

F64 = torch.float64


def chebyshev_nodes(N):
    return np.cos(np.pi * (np.arange(N + 1) + 0.5) / (N + 1))


def fit_panels(values, N):
    """Chebyshev coefficients [N+1, N+1] of the tensor interpolant of `values[i, j]` = f(x_i, y_j) at the first-kind nodes (x, y panel-mapped to [-1, 1])."""
    nodes = chebyshev_nodes(N)
    c = C.chebfit(nodes, np.asarray(values).T, N)
    return C.chebfit(nodes, c.T, N)


def _basis(t, N):
    """T_k(t), dT_k/dt for k = 0 .. N, t [P] in [-1, 1], in closed form (a handful of launches instead of an N-step recurrence): T_k = cos(k theta), dT_k/dt = k sin(k theta) / sin(theta)
    with the limits k^2 (-1)^(k-1)... at the ends handled by the Taylor expansion sin(k th) / sin(th) = k (1 - (k^2 - 1) th^2 / 6) for th < 1e-4 (and the mirror image at pi)."""
    k = torch.arange(N + 1, dtype=F64, device=t.device)
    th = torch.acos(t.clamp(-1.0, 1.0))[:, None]
    T = torch.cos(k * th)
    sth = torch.sin(th)
    d = torch.minimum(th, torch.pi - th)
    sgn = torch.where(th > 0.5 * torch.pi, (-1.0) ** (k + 1), torch.ones_like(k))                      # sin(k (pi - d)) / sin(d) = (-1)^(k+1) sin(k d) / sin(d)
    ratio = torch.where(d < 1e-4, k * (1.0 - (k * k - 1.0) * d * d / 6.0) * sgn, torch.sin(k * th) / sth.clamp(min=1e-300))
    return T, k * ratio


class BoxTables:
    """the two corner tables of one kernel on one device, Chebyshev tensors on the graded panel grid `breaks` (per axis): `eval(name, a, b)` -> (value, d/da, d/db) at points (a, b)
    in units of the support (clamped).  Kernels with a cusp at the origin (`cone`, `lw2`: a term linear in r) make Phi non-analytic at (0, 0); the geometric grading toward the origin
    restores the spectral rate."""

    def __init__(self, coeffs, breaks, N, device):
        self.N = N
        self.breaks = torch.as_tensor(breaks, dtype=F64, device=device)                 # [np + 1]
        self.inner = self.breaks[1:-1].contiguous()
        self.coef = {k: torch.as_tensor(c, dtype=F64, device=device) for k, c in coeffs.items()}        # name -> [np, np, N+1, N+1]

    def corner(self, a, b):
        """everything one corner needs, for points (a, b) in units of the support (clamped), the basis computed once: Phi (value, d/da, d/db), Phi1 (value, d/da, d/db) and Phi_2 = Phi1(b, a)
        (value, d/da, d/db)."""
        a, b = a.clamp(-1.0, 1.0), b.clamp(-1.0, 1.0)
        ia, ib = torch.bucketize(a, self.inner, right=True), torch.bucketize(b, self.inner, right=True)
        wa, wb = self.breaks[ia + 1] - self.breaks[ia], self.breaks[ib + 1] - self.breaks[ib]
        ta, tb = (2.0 * a - (self.breaks[ia] + self.breaks[ia + 1])) / wa, (2.0 * b - (self.breaks[ib] + self.breaks[ib + 1])) / wb
        sa, sb = 2.0 / wa, 2.0 / wb
        Ta, dTa = _basis(ta, self.N)
        Tb, dTb = _basis(tb, self.N)
        out = {}
        for name in ("Phi", "Phi1"):
            Cg = self.coef[name][ia, ib]                                                # [P, N+1, N+1], first index: a
            CTb = torch.einsum("pkl,pl->pk", Cg, Tb)
            out[name] = ((Ta * CTb).sum(1), sa * (dTa * CTb).sum(1), sb * (Ta * torch.einsum("pkl,pl->pk", Cg, dTb)).sum(1))
        Cg = self.coef["Phi1"][ib, ia]                                                  # Phi_2(a, b) = Phi1(b, a): first index is now b
        CTa = torch.einsum("pkl,pl->pk", Cg, Ta)
        v2, d_first, d_second = (Tb * CTa).sum(1), sb * (dTb * CTa).sum(1), sa * (Tb * torch.einsum("pkl,pl->pk", Cg, dTa)).sum(1)
        out["Phi2"] = (v2, d_second, d_first)                                           # (value, d/da, d/db) of Phi_2
        return out

    def channels(self, a0, a1, b0, b1):
        """(lam [P], m [P,2], g0 [P,2], g1 [P,2,2]) of the rectangle [a0, a1] x [b0, b1] (relative to the particle, units of the support), library conventions
        (m = int y W, g0 = int grad_x W, g1[d, j] = int y_d d_xj W); the four corners are one batch of 4P points."""
        P = a0.shape[0]
        a = torch.cat([a1, a0, a1, a0]); b = torch.cat([b1, b1, b0, b0])
        sg = torch.cat([torch.ones(P), -torch.ones(P), -torch.ones(P), torch.ones(P)]).to(device=a0.device, dtype=F64)
        c = self.corner(a, b)
        (ph, pa, pb), (p1, p1a, p1b), (p2, p2a, p2b) = c["Phi"], c["Phi1"], c["Phi2"]
        ya = torch.where(a.abs() <= 1.0, a, torch.zeros_like(a))                      # the boundary term carries the factor y_j = a_j only inside the support
        yb = torch.where(b.abs() <= 1.0, b, torch.zeros_like(b))
        red = lambda v: (sg * v).reshape(4, P).sum(0)
        lam = red(ph)
        m = torch.stack([red(p1), red(p2)], 1)
        g0 = -torch.stack([red(pa), red(pb)], 1)
        g1 = -torch.stack([torch.stack([red(ya * pa - ph), red(p1b)], 1),               # d = 1: j = 1 boundary a dPhi/da - Phi, j = 2 boundary dPhi_1/db
                           torch.stack([red(p2a), red(yb * pb - ph)], 1)], 1)           # d = 2: j = 1 boundary dPhi_2/da,       j = 2 boundary b dPhi/db - Phi
        return lam, m, g0, g1


# ---------------------------------------------------------------------------------------------------------------------------------- Warp lookup
NB = 21                                                                                   # BOX_DEGREE + 1 (checked in `BoxTables.block`)
Vec21 = wp.types.vector(length=NB, dtype=wp.float64)
Vec9 = wp.types.vector(length=9, dtype=wp.float64)


@wp.func
def _cheb_basis(t: wp.float64):
    """T_k(t) and dT_k/dt, k = 0 .. 20 (`_basis`: cos(k th), k sin(k th) / sin(th) with the Taylor expansion within 1e-4 of the ends and the mirror image at pi)."""
    th = wp.acos(wp.clamp(t, wp.float64(-1.0), wp.float64(1.0)))
    sth = wp.sin(th)
    d = wp.min(th, wp.float64(3.1415926535897932384626433832795) - th)
    T = Vec21()
    dT = Vec21()
    for k in range(NB):
        kf = wp.float64(k)
        T[k] = wp.cos(kf * th)
        sg = wp.float64(1.0)
        if th > wp.float64(1.5707963267948966192313216916398) and k % 2 == 0:
            sg = wp.float64(-1.0)
        ratio = wp.sin(kf * th) / wp.max(sth, wp.float64(1.0e-300))
        if d < wp.float64(1.0e-4):
            ratio = kf * (wp.float64(1.0) - (kf * kf - wp.float64(1.0)) * d * d / wp.float64(6.0)) * sg
        dT[k] = kf * ratio
    return T, dT


@wp.func
def _tensor(coef: wp.array(dtype=wp.float64), base: int, U: Vec21, dU: Vec21, V: Vec21, dV: Vec21, su: wp.float64, sv: wp.float64):
    """sum_kl C[k, l] U_k V_l and its derivatives with respect to the first / second variable (scaled by su / sv)."""
    v = wp.float64(0.0)
    vu = wp.float64(0.0)
    vv = wp.float64(0.0)
    for k in range(NB):
        r0 = wp.float64(0.0)
        r1 = wp.float64(0.0)
        for l in range(NB):
            c = coef[base + k * NB + l]
            r0 += c * V[l]
            r1 += c * dV[l]
        v += U[k] * r0
        vu += dU[k] * r0
        vv += U[k] * r1
    return v, su * vu, sv * vv


@wp.func
def _panel(x: wp.float64, breaks: wp.array(dtype=wp.float64), npan: int):
    i = int(0)
    for j in range(1, npan):
        if x >= breaks[j]:
            i = j
    return i


@wp.kernel
def _box_tensor_kernel(lpos: wp.array(dtype=wp.float64), lsup: wp.array(dtype=wp.float64), lo0: wp.float64, lo1: wp.float64, hi0: wp.float64, hi1: wp.float64,
                       phi: wp.array(dtype=wp.float64), phi1: wp.array(dtype=wp.float64), breaks: wp.array(dtype=wp.float64), npan: int, vals: wp.array(dtype=wp.float64)):
    """stage 1 of the box lookup, one thread per (row, corner, table): the table value and its two derivatives at the corner (`BoxTables.corner`: Phi, Phi1, Phi2 = Phi1(b, a)) -> vals[(row * 4 + corner) * 9 + 3 table + (value, d/da, d/db)]."""
    tid = wp.tid()
    r = tid / 12
    c = (tid / 3) % 4
    tab = tid % 3
    h = lsup[r]
    a = (hi0 - lpos[2 * r]) / h
    if c == 1 or c == 3:
        a = (lo0 - lpos[2 * r]) / h
    b = (hi1 - lpos[2 * r + 1]) / h
    if c >= 2:
        b = (lo1 - lpos[2 * r + 1]) / h
    ac = wp.clamp(a, wp.float64(-1.0), wp.float64(1.0))
    bc = wp.clamp(b, wp.float64(-1.0), wp.float64(1.0))
    ia = _panel(ac, breaks, npan)
    ib = _panel(bc, breaks, npan)
    wa = breaks[ia + 1] - breaks[ia]
    wb = breaks[ib + 1] - breaks[ib]
    ta = (wp.float64(2.0) * ac - (breaks[ia] + breaks[ia + 1])) / wa
    tb = (wp.float64(2.0) * bc - (breaks[ib] + breaks[ib + 1])) / wb
    sa = wp.float64(2.0) / wa
    sb = wp.float64(2.0) / wb
    Ta, dTa = _cheb_basis(ta)
    Tb, dTb = _cheb_basis(tb)
    n2 = NB * NB
    o = (r * 4 + c) * 9 + tab * 3
    if tab == 0:
        v, d1, d2 = _tensor(phi, (ia * npan + ib) * n2, Ta, dTa, Tb, dTb, sa, sb)
        vals[o] = v
        vals[o + 1] = d1
        vals[o + 2] = d2
    elif tab == 1:
        v, d1, d2 = _tensor(phi1, (ia * npan + ib) * n2, Ta, dTa, Tb, dTb, sa, sb)
        vals[o] = v
        vals[o + 1] = d1
        vals[o + 2] = d2
    else:
        v, d1, d2 = _tensor(phi1, (ib * npan + ia) * n2, Tb, dTb, Ta, dTa, sb, sa)           # Phi_2(a, b) = Phi1(b, a): the first variable is b
        vals[o] = v
        vals[o + 1] = d2
        vals[o + 2] = d1


@wp.kernel
def _box_block_kernel(lpos: wp.array(dtype=wp.float64), lsup: wp.array(dtype=wp.float64), lo0: wp.float64, lo1: wp.float64, hi0: wp.float64, hi1: wp.float64,
                      vals: wp.array(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    """stage 2, one thread per row: the nine channels (lam, m, g0, g1; units of the support) of the rectangle by inclusion-exclusion of the four corners (`BoxTables.channels`)."""
    r = wp.tid()
    h = lsup[r]
    lam = wp.float64(0.0)
    m0 = wp.float64(0.0)
    m1 = wp.float64(0.0)
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    g00 = wp.float64(0.0)
    g01 = wp.float64(0.0)
    g10 = wp.float64(0.0)
    g11 = wp.float64(0.0)
    for c in range(4):
        a = (hi0 - lpos[2 * r]) / h
        if c == 1 or c == 3:
            a = (lo0 - lpos[2 * r]) / h
        b = (hi1 - lpos[2 * r + 1]) / h
        if c >= 2:
            b = (lo1 - lpos[2 * r + 1]) / h
        sg = wp.float64(1.0)
        if c == 1 or c == 2:
            sg = wp.float64(-1.0)
        o = (r * 4 + c) * 9
        ya = wp.float64(0.0)
        if wp.abs(a) <= wp.float64(1.0):
            ya = a
        yb = wp.float64(0.0)
        if wp.abs(b) <= wp.float64(1.0):
            yb = b
        lam += sg * vals[o]
        m0 += sg * vals[o + 3]
        m1 += sg * vals[o + 6]
        gx -= sg * vals[o + 1]
        gy -= sg * vals[o + 2]
        g00 -= sg * (ya * vals[o + 1] - vals[o])
        g01 -= sg * vals[o + 5]
        g10 -= sg * vals[o + 7]
        g11 -= sg * (yb * vals[o + 2] - vals[o])
    out[r, 0] = lam
    out[r, 1] = m0
    out[r, 2] = m1
    out[r, 3] = gx
    out[r, 4] = gy
    out[r, 5] = g00
    out[r, 6] = g01
    out[r, 7] = g10
    out[r, 8] = g11


def _block(self, lpos, lsup, lo, hi):
    """[rows, 9] float64: the nine channels (units of the support, the `warpbc.edge_channels` convention) of the rectangle [lo, hi] for the body-frame positions `lpos` [rows, 2] and supports `lsup`; the
    Warp counterpart of `channels` (one thread per row, the two tables read from device memory; stage 1 one thread per (row, corner, table), stage 2 one per row)."""
    assert self.N + 1 == NB, "box.py Warp lookup is compiled for BOX_DEGREE = %d" % (NB - 1)
    dev = str(lpos.device)
    if getattr(self, "_wp", None) is None:
        self._wp = (wp.from_torch(self.coef["Phi"].contiguous().reshape(-1), dtype=wp.float64), wp.from_torch(self.coef["Phi1"].contiguous().reshape(-1), dtype=wp.float64),
                    wp.from_torch(self.breaks.contiguous(), dtype=wp.float64), len(self.breaks) - 1)
    wphi, wphi1, wbr, npan = self._wp
    rows = lpos.shape[0]
    out = torch.zeros((max(rows, 1), 9), dtype=F64, device=lpos.device)
    if rows:
        lp = lpos.to(F64).contiguous().reshape(-1)
        ls = lsup.to(F64).contiguous()
        vals = torch.empty(rows * 36, dtype=F64, device=lpos.device)
        wlp, wls = wp.from_torch(lp, dtype=wp.float64), wp.from_torch(ls, dtype=wp.float64)
        box = [wp.float64(float(lo[0])), wp.float64(float(lo[1])), wp.float64(float(hi[0])), wp.float64(float(hi[1]))]
        wvals = wp.from_torch(vals, dtype=wp.float64)
        wp.launch(_box_tensor_kernel, dim=rows * 12, device=dev, inputs=[wlp, wls] + box + [wphi, wphi1, wbr, npan, wvals])
        wp.launch(_box_block_kernel, dim=rows, device=dev, inputs=[wlp, wls] + box + [wvals, wp.from_torch(out, dtype=wp.float64)])
        wp.synchronize_device(dev)
    return out[:rows]


BoxTables.block = _block
