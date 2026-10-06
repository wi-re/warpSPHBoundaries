"""The disk element on the device (docs/disk-element.md): the channels of every (query, disk) pair of a fixed-capacity slot list as a table lookup (`edge.disktables`), one thread per slot, all kernel groups of the
fused wall evaluation in one launch.  Same output convention as `warpfused.fused_channels`: [P, 9 * len(groups)] in the active precision, per group the nine channels (lam, m1 (2), g0 (2), g1 (2 x 2), dimensionless,
units of h) in the BODY frame of the query positions; the indicator of the solid is part of lam / g1 (the element is complete: `ind0` in `FusedWall`).

    c = disk_channels_device(slot_q, slot_m, lpos, lsup, centres, radii, groups, device)       # slot_m < 0: an empty slot (all channels 0)

Two launches: one thread per (slot, group, table channel) does the lookups (the latency of a launch is the serial work of its longest thread: one thread per slot took 200 us whatever the number of active slots),
one thread per slot assembles the nine channels.  Lookup (h = 1: D = |centre - x| / h, R = radius / h, delta = D - R): D >= 1 + R: zero; R > 1 and D <= R - 1: the support ball lies in the disk (lam = 1, g1 = I); R > 1: the table L over (v = (delta + 1) / 2, s = 1 / R);
R <= 1: one of six tables over (w, r), w = (2 / pi) asin(sqrt(u)) with u the position inside the delta interval between the surface and the tangency lines, r the position in the radius branch, times R^2.
"""
import numpy as np
import torch
import warp as wp

from .disktables import load_disk_tables
from .precision import IS_F32, np_real, real, torch_real, vec2_t, sync as _sync

wp.config.quiet = True
vec12 = wp.types.vector(length=12, dtype=real)
CH = 5


@wp.func
def _panel_eval1(coef: wp.array(dtype=real), base: int, pa: int, pb: int, na: int, nb: int, a: real, b: real, ch: int):
    """sum_{p,q} coef[panel, p, q, ch] T_p(ta) T_q(tb) for the panel of (a, b) in [0, 1]^2: ONE channel (one thread of the launch; the latency of a launch is the serial work of its longest thread)."""
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
    out = real(0.0)
    for p in range(na):
        acc = real(0.0)
        for q in range(nb):
            acc += Tb[q] * coef[base + (((i * pb + j) * na + p) * nb + q) * 5 + ch]
        out += Ta[p] * acc
    return out


@wp.func
def _disk_group1(coef: wp.array(dtype=real), off: int, D: real, R: real, ch: int, pu: int, pR: int, nu: int, nR: int, Lpu: int, Lps: int, Lnu: int, Lns: int):
    """channel `ch` (lam, m1x, g0x, g1xx, g1yy) of the disk (centre distance D, radius R, h = 1) for the tables of one kernel at `off`."""
    if D >= real(1.0) + R:
        return real(0.0)
    Ssz = pu * pR * nu * nR * 5
    if R > real(1.0):
        if D <= R - real(1.0):
            if ch == 0 or ch == 3 or ch == 4:
                return real(1.0)
            return real(0.0)
        return _panel_eval1(coef, off + 6 * Ssz, Lpu, Lps, Lnu, Lns, (D - R + real(1.0)) * real(0.5), real(1.0) / R, ch)
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
    return _panel_eval1(coef, off + (branch * 3 + k) * Ssz, pu, pR, nu, nR, w, rr, ch) * R * R


@wp.kernel
def _disk_table_kernel(slot_q: wp.array(dtype=int), slot_m: wp.array(dtype=int), pos: wp.array(dtype=vec2_t), sup: wp.array(dtype=real),
                       centres: wp.array(dtype=vec2_t), radii: wp.array(dtype=real), coef: wp.array(dtype=real), goff: wp.array(dtype=int), nG: int,
                       pu: int, pR: int, nu: int, nR: int, Lpu: int, Lps: int, Lnu: int, Lns: int, tmp: wp.array2d(dtype=real)):
    """one thread per (slot, group, table channel): the lookup of the five table channels of every group in parallel (tmp[slot, 5 g + c])."""
    tid = wp.tid()
    slot = tid / (nG * 5)
    rem = tid - slot * nG * 5
    g = rem / 5
    c = rem - g * 5
    m = slot_m[slot]
    if m < 0:
        return
    q = slot_q[slot]
    h = sup[q]
    y = (centres[m] - pos[q]) / h
    D = wp.sqrt(y[0] * y[0] + y[1] * y[1])
    tmp[slot, g * 5 + c] = _disk_group1(coef, goff[g], D, radii[m] / h, c, pu, pR, nu, nR, Lpu, Lps, Lnu, Lns)


@wp.kernel
def _disk_assemble_kernel(slot_q: wp.array(dtype=int), slot_m: wp.array(dtype=int), pos: wp.array(dtype=vec2_t), centres: wp.array(dtype=vec2_t), tmp: wp.array2d(dtype=real),
                          nG: int, cout: wp.array2d(dtype=real)):
    """the nine channels of every group in the body frame from the five table channels and the direction to the centre."""
    slot = wp.tid()
    m = slot_m[slot]
    if m < 0:
        return
    q = slot_q[slot]
    y = centres[m] - pos[q]
    D = wp.sqrt(y[0] * y[0] + y[1] * y[1])
    cx = real(1.0)
    cy = real(0.0)
    if D > real(1.0e-12):
        cx = y[0] / D
        cy = y[1] / D
    tx = -cy
    ty = cx
    for g in range(nG):
        col = g * 9
        t0 = g * 5
        cout[slot, col] = tmp[slot, t0]
        cout[slot, col + 1] = tmp[slot, t0 + 1] * cx
        cout[slot, col + 2] = tmp[slot, t0 + 1] * cy
        cout[slot, col + 3] = tmp[slot, t0 + 2] * cx
        cout[slot, col + 4] = tmp[slot, t0 + 2] * cy
        cout[slot, col + 5] = tmp[slot, t0 + 3] * cx * cx + tmp[slot, t0 + 4] * tx * tx
        cout[slot, col + 6] = tmp[slot, t0 + 3] * cx * cy + tmp[slot, t0 + 4] * tx * ty
        cout[slot, col + 7] = tmp[slot, t0 + 3] * cx * cy + tmp[slot, t0 + 4] * tx * ty
        cout[slot, col + 8] = tmp[slot, t0 + 3] * cy * cy + tmp[slot, t0 + 4] * ty * ty


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
        tmp = torch.zeros((P, plan.nG * 5), dtype=torch_real, device=device)
        wp.launch(_disk_table_kernel, dim=P * plan.nG * 5, device=device, inputs=[wq, wm, wpos, wsup, wc, wr, plan.coef, plan.goff, plan.nG, pu, pR, nu, nR, Lpu, Lps, Lnu, Lns,
                                                                                   wp.from_torch(tmp, dtype=real)])
        wp.launch(_disk_assemble_kernel, dim=P, device=device, inputs=[wq, wm, wpos, wc, wp.from_torch(tmp, dtype=real), plan.nG, wp.from_torch(cout, dtype=real)])
        _sync(device)
    c = cout[:P]
    return c.to(torch.float64) if as_float64 else c
