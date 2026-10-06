"""The disk element on the device (docs/disk-element.md): the channels of every (query, disk) pair of a fixed-capacity slot list as a table lookup (`edge.disktables`), one thread per slot, all kernel groups of the
fused wall evaluation in one launch.  Same output convention as `warpfused.fused_channels`: [P, 9 * len(groups)] in the active precision, per group the nine channels (lam, m1 (2), g0 (2), g1 (2 x 2), dimensionless,
units of h) in the BODY frame of the query positions; the indicator of the solid is part of lam / g1 (the element is complete: `ind0` in `FusedWall`).

    c = disk_channels_device(slot_q, slot_m, lpos, lsup, centres, radii, groups, device)       # slot_m < 0: an empty slot (all channels 0)

Lookup (h = 1: D = |centre - x| / h, R = radius / h, delta = D - R): D >= 1 + R: zero; R > 1 and D <= R - 1: the support ball lies in the disk (lam = 1, g1 = I); R > 1: the table L over (v = (delta + 1) / 2, s = 1 / R);
R <= 1: one of six tables over (w, r), w = (2 / pi) asin(sqrt(u)) with u the position inside the delta interval between the surface and the tangency lines, r the position in the radius branch, times R^2.
"""
import numpy as np
import torch
import warp as wp

from .disktables import load_disk_tables
from .precision import IS_F32, np_real, real, torch_real, vec2_t, sync as _sync

wp.config.quiet = True
vec5 = wp.types.vector(length=5, dtype=real)
vec12 = wp.types.vector(length=12, dtype=real)
CH = 5


@wp.func
def _panel_eval(coef: wp.array(dtype=real), base: int, pa: int, pb: int, na: int, nb: int, a: real, b: real):
    """sum_{p,q} coef[panel, p, q, :] T_p(ta) T_q(tb) for the panel of (a, b) in [0, 1]^2 (a vec5)."""
    a = wp.clamp(a, real(0.0), real(1.0))
    b = wp.clamp(b, real(0.0), real(1.0))
    af = a * real(pa)
    bf = b * real(pb)
    i = wp.min(int(af), pa - 1)
    j = wp.min(int(bf), pb - 1)
    ta = real(2.0) * (af - real(i)) - real(1.0)
    tb = real(2.0) * (bf - real(j)) - real(1.0)
    Ta = vec12(real(0.0))
    Tb = vec12(real(0.0))
    Ta[0] = real(1.0)
    Tb[0] = real(1.0)
    if na > 1:
        Ta[1] = ta
    if nb > 1:
        Tb[1] = tb
    for k in range(2, 12):
        if k < na:
            Ta[k] = real(2.0) * ta * Ta[k - 1] - Ta[k - 2]
        if k < nb:
            Tb[k] = real(2.0) * tb * Tb[k - 1] - Tb[k - 2]
    out = vec5(real(0.0))
    for p in range(na):
        for q in range(nb):
            w = Ta[p] * Tb[q]
            idx = base + (((i * pb + j) * na + p) * nb + q) * 5
            for c in range(5):
                out[c] = out[c] + w * coef[idx + c]
    return out


@wp.func
def _disk_group(coef: wp.array(dtype=real), off: int, D: real, R: real, pu: int, pR: int, nu: int, nR: int, Lpu: int, Lps: int, Lnu: int, Lns: int):
    """(lam, m1x, g0x, g1xx, g1yy) of the disk (centre distance D, radius R, h = 1) for the tables of one kernel at `off`."""
    res = vec5(real(0.0))
    if D >= real(1.0) + R:
        return res
    Ssz = pu * pR * nu * nR * 5
    if R > real(1.0):
        if D <= R - real(1.0):
            res[0] = real(1.0)
            res[3] = real(1.0)
            res[4] = real(1.0)
            return res
        return _panel_eval(coef, off + 6 * Ssz, Lpu, Lps, Lnu, Lns, (D - R + real(1.0)) * real(0.5), real(1.0) / R)
    delta = D - R
    branch = int(0)
    if R > real(0.5):
        branch = 1
    # interval boundaries: branch 0: [-R, 0], [0, 1 - 2R], [1 - 2R, 1];  branch 1: [-R, 1 - 2R], [1 - 2R, 0], [0, 1]
    b1 = real(0.0)
    b2 = real(1.0) - real(2.0) * R
    if branch == 1:
        b1 = real(1.0) - real(2.0) * R
        b2 = real(0.0)
    k = int(2)
    lo = b2
    hi = real(1.0)
    if delta < b1:
        k = 0
        lo = -R
        hi = b1
    elif delta < b2:
        k = 1
        lo = b1
        hi = b2
    u = real(0.0)
    if hi > lo:
        u = wp.clamp((delta - lo) / (hi - lo), real(0.0), real(1.0))
    w = real(2.0 / 3.141592653589793) * wp.asin(wp.sqrt(u))
    rr = real(2.0) * R
    if branch == 1:
        rr = real(2.0) * R - real(1.0)
    r = _panel_eval(coef, off + (branch * 3 + k) * Ssz, pu, pR, nu, nR, w, rr)
    R2 = R * R
    for c in range(5):
        res[c] = r[c] * R2
    return res


@wp.kernel
def _disk_channels_kernel(slot_q: wp.array(dtype=int), slot_m: wp.array(dtype=int), pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                          centres: wp.array(dtype=vec2_t), radii: wp.array(dtype=real), coef: wp.array(dtype=real), goff: wp.array(dtype=int), nG: int, gsize: int,
                          pu: int, pR: int, nu: int, nR: int, Lpu: int, Lps: int, Lnu: int, Lns: int, cout: wp.array2d(dtype=real)):
    tid = wp.tid()
    m = slot_m[tid]
    if m < 0:
        return
    q = slot_q[tid]
    h = sup[q]
    y = (centres[m] - pos[q]) / h
    D = wp.sqrt(y[0] * y[0] + y[1] * y[1])
    R = radii[m] / h
    cx = real(1.0)
    cy = real(0.0)
    if D > real(1.0e-12):
        cx = y[0] / D
        cy = y[1] / D
    tx = -cy
    ty = cx
    for g in range(nG):
        r = _disk_group(coef, goff[g], D, R, pu, pR, nu, nR, Lpu, Lps, Lnu, Lns)
        col = g * 9
        cout[tid, col] = r[0]
        cout[tid, col + 1] = r[1] * cx
        cout[tid, col + 2] = r[1] * cy
        cout[tid, col + 3] = r[2] * cx
        cout[tid, col + 4] = r[2] * cy
        cout[tid, col + 5] = r[3] * cx * cx + r[4] * tx * tx
        cout[tid, col + 6] = r[3] * cx * cy + r[4] * tx * ty
        cout[tid, col + 7] = r[3] * cx * cy + r[4] * tx * ty
        cout[tid, col + 8] = r[3] * cy * cy + r[4] * ty * ty


class DiskPlan:
    """the tables of the kernels of a group tuple on one device, concatenated: group g at offset g * (6 S tables + 1 L table)."""

    def __init__(self, groups, device):
        self.nG = len(groups)
        blocks, cfg = [], None
        for g in groups:
            S, L, c = load_disk_tables(g.kernel)
            if cfg is not None and c != cfg:
                raise ValueError("disk tables of different resolution in one launch")
            cfg = c
            blocks.append(np.concatenate([S.reshape(-1), L.reshape(-1)]))
        self.cfg = cfg
        pu, pR, nu, nR, Lpu, Lps, Lnu, Lns = cfg
        self.gsize = len(blocks[0])
        self.coef = wp.array(np.concatenate(blocks).astype(np_real), dtype=real, device=device)
        self.goff = wp.array(np.arange(self.nG, dtype=np.int32) * self.gsize, dtype=int, device=device)


_PLANS = {}


def disk_plan(groups, device):
    key = (tuple(g.kernel for g in groups), str(device), real.__name__)
    if key not in _PLANS:
        _PLANS[key] = DiskPlan(groups, str(device))
    return _PLANS[key]


def disk_channels_device(slot_q, slot_m, positions, supports, centres, radii, groups, device="cuda:0", as_float64=False):
    """channels [P, 9 * len(groups)] of the (query, disk) slots (see the module docstring); `positions` [N, 2] and `supports` [N] in the body frame, `centres` [M, 2], `radii` [M]."""
    plan = disk_plan(groups, device)
    P = len(slot_q)
    cout = torch.zeros((max(P, 1), plan.nG * 9), dtype=torch_real, device=device)
    if P:
        pu, pR, nu, nR, Lpu, Lps, Lnu, Lns = plan.cfg
        wq = wp.from_torch(slot_q.to(torch.int32).contiguous(), dtype=wp.int32)
        wm = wp.from_torch(slot_m.to(torch.int32).contiguous(), dtype=wp.int32)
        wpos = wp.from_torch(positions.to(torch_real).contiguous(), dtype=vec2_t)
        wsup = wp.from_torch(supports.to(torch_real).contiguous(), dtype=real)
        wc = wp.from_torch(centres.to(torch_real).contiguous(), dtype=vec2_t)
        wr = wp.from_torch(radii.to(torch_real).contiguous(), dtype=real)
        wp.launch(_disk_channels_kernel, dim=P, device=device, inputs=[wq, wm, wpos, wsup, wc, wr, plan.coef, plan.goff, plan.nG, plan.gsize, pu, pR, nu, nR, Lpu, Lps, Lnu, Lns,
                                                                        wp.from_torch(cout, dtype=real)])
        _sync(device)
    c = cout[:P]
    return c.to(torch.float64) if as_float64 else c
