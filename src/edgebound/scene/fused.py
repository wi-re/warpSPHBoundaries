"""Fused wall evaluation (docs/plan-wall-evaluation.md step 3): all kernels and outputs the solver needs at ONE position set in two launches per (body, representation).

    fw = FusedWall(scene, adjacency, groups)                                  # stage 1: one launch family for the channels of every kernel group (warpfused)
    out = fw.evaluate(outputs, a1=None)                                       # stage 2: one contraction kernel per (body, rep): {name: tensor [B, N, dim]}, world frame, float64

`groups` are `warpfused.FusedGroup`s (kernel, channels); an output is `WallOutput(name, group, kind)` with `kind`
    "lam"   sum of lam (+ the indicator)                                                     [B, N]
    "g0"    sum of grad_x W integrals, world frame                                             [B, N, 2]
    "cov"   sum of g1 = int y (x) grad_x W (+ indicator * I), world frame                       [B, N, 2, 2]
    "a1g1"  sum_d a1_d g1_dj for a per-query world vector a1 (the hydrostatic field term A)    [B, N, 2]
    "lap"   sum of (2 lam - tr g1) (the wall Laplacian of the registered `lw2`; the indicator cancels)   [B, N]
exactly the arithmetic of `sceneOperation` (Density, Gradient of a constant, Covariance, Gradient with a per-query a1) on the moments of the same kernel, without the per-pair torch
operations: the pairs of a row are summed by one thread in a fixed order (deterministic), rotated to the world frame once per query.  Surface representations only; other
representations raise (callers keep the `sceneOperation` path for them).  `evaluate` can be called repeatedly on one `FusedWall` (new a1, other outputs): stage 1 is not repeated.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from ..edge import warpfused
from ..edge.precision import real, torch_real
from .fixedadj import fixed_topology
from .scene import BOX_EXACT_KERNELS, BoxRep, SurfaceRep, boxTables

F64 = torch.float64
KINDS = {"lam": (0, 1), "g0": (1, 2), "cov": (2, 4), "a1g1": (3, 2), "lap": (4, 1)}


@dataclass(frozen=True)
class WallOutput:
    name: str
    group: int
    kind: str


@wp.kernel
def _wall_contract_kernel(row_start: wp.array(dtype=int), perm: wp.array(dtype=int), c: wp.array2d(dtype=real),
                          cand: wp.array(dtype=int), lsup: wp.array(dtype=real), ind: wp.array(dtype=real),
                          R00: real, R01: real, R10: real, R11: real,
                          n_out: int, o_group: wp.array(dtype=int), o_kind: wp.array(dtype=int), o_base: wp.array(dtype=int), o_dim: wp.array(dtype=int), o_ind: wp.array(dtype=int),
                          a1: wp.array(dtype=real), a1_base: int, bidx: int, N: int, pool: wp.array(dtype=real)):
    row = wp.tid()
    q = cand[row]
    h = lsup[row]
    i0 = row_start[row]
    i1 = row_start[row + 1]
    idv = ind[row]
    for k in range(n_out):
        g = o_group[k] * 9
        kind = o_kind[k]
        base = o_base[k] + (bidx * N + q) * o_dim[k]
        use_ind = o_ind[k]
        if kind == 0:
            s = real(0.0)
            for i in range(i0, i1):
                s += c[perm[i], g]
            if use_ind != 0:
                s += idv
            pool[base] = pool[base] + s
        elif kind == 4:
            s = real(0.0)
            for i in range(i0, i1):
                p = perm[i]
                s += real(2.0) * c[p, g] - c[p, g + 5] - c[p, g + 8]
            pool[base] = pool[base] + s
        elif kind == 1:
            vx = real(0.0)
            vy = real(0.0)
            for i in range(i0, i1):
                p = perm[i]
                vx += c[p, g + 3]
                vy += c[p, g + 4]
            vx = vx / h
            vy = vy / h
            pool[base] = pool[base] + R00 * vx + R01 * vy
            pool[base + 1] = pool[base + 1] + R10 * vx + R11 * vy
        else:
            m00 = real(0.0)
            m01 = real(0.0)
            m10 = real(0.0)
            m11 = real(0.0)
            for i in range(i0, i1):
                p = perm[i]
                m00 += c[p, g + 5]
                m01 += c[p, g + 6]
                m10 += c[p, g + 7]
                m11 += c[p, g + 8]
            if use_ind != 0:
                m00 += idv
                m11 += idv
            if kind == 2:
                pool[base] = pool[base] + R00 * (m00 * R00 + m01 * R01) + R01 * (m10 * R00 + m11 * R01)
                pool[base + 1] = pool[base + 1] + R00 * (m00 * R10 + m01 * R11) + R01 * (m10 * R10 + m11 * R11)
                pool[base + 2] = pool[base + 2] + R10 * (m00 * R00 + m01 * R01) + R11 * (m10 * R00 + m11 * R01)
                pool[base + 3] = pool[base + 3] + R10 * (m00 * R10 + m01 * R11) + R11 * (m10 * R10 + m11 * R11)
            else:
                aw0 = a1[a1_base + 2 * q]
                aw1 = a1[a1_base + 2 * q + 1]
                al0 = R00 * aw0 + R10 * aw1
                al1 = R01 * aw0 + R11 * aw1
                v0 = al0 * m00 + al1 * m10
                v1 = al0 * m01 + al1 * m11
                pool[base] = pool[base] + R00 * v0 + R01 * v1
                pool[base + 1] = pool[base + 1] + R10 * v0 + R11 * v1


@wp.func
def _wrap_pi(a: wp.float64):
    two_pi = wp.float64(6.283185307179586476925286766559)
    r = a + wp.float64(3.1415926535897932384626433832795)
    r = r - two_pi * wp.floor(r / two_pi)
    return r - wp.float64(3.1415926535897932384626433832795)


@wp.func
def _chord_deficit(l: wp.float64, h: wp.float64, z: wp.float64, H: wp.float64, be: wp.float64, pn: wp.float64):
    """the part of the sector (1/2) H^2 (h - l) that a chord interval of an edge removes: (1/2) H^2 (c_hi - c_lo) - (1/2) z^2 (tan(c_hi - pn) - tan(c_lo - pn)), [c_lo, c_hi] = [l, h] cap (pn - be, pn + be) (the copy of the 2 pi periodic interval whose centre is nearest the midpoint); 0 when empty."""
    two_pi = wp.float64(6.283185307179586476925286766559)
    m = wp.float64(0.5) * (l + h)
    k = wp.floor((m - pn) / two_pi + wp.float64(0.5))
    c_lo = wp.max(l, pn + two_pi * k - be)
    c_hi = wp.min(h, pn + two_pi * k + be)
    out = wp.float64(0.0)
    if c_lo < c_hi:
        out = wp.float64(0.5) * H * H * (c_hi - c_lo) - wp.float64(0.5) * z * z * (wp.tan(c_hi - pn) - wp.tan(c_lo - pn))
    return out


@wp.kernel
def _cone_area_kernel(row_start: wp.array(dtype=int), perm: wp.array(dtype=int), pe: wp.array(dtype=int), cand: wp.array(dtype=int),
                      lpos: wp.array(dtype=wp.float64), lsup: wp.array(dtype=wp.float64), ind: wp.array(dtype=wp.float64),
                      verts: wp.array(dtype=wp.float64), edges: wp.array(dtype=int), axis: wp.array(dtype=wp.float64), dth: wp.float64, al: wp.float64,
                      N: int, out: wp.array(dtype=wp.float64)):
    """per query row: the areas area(solid cap disk(p, H) cap wedge(p, axis, al)) -> out[q] and the full disk -> out[N + q] (added), H = lsup[row].  Local form of cone_area.py: the far edges' sector parts of the
    per-edge formula sum to (1/2) H^2 W (indicator - background) along every ray (signed crossings), so  area = (1/2) H^2 W indicator - sum_{edges within H} s_e (chord deficit);  edges at segment distance >= H have an empty chord."""
    row = wp.tid()
    q = cand[row]
    H = lsup[row]
    px = lpos[2 * row]
    py = lpos[2 * row + 1]
    th = wp.atan2(axis[2 * q + 1], axis[2 * q]) - dth                     # the world axis in the body frame
    pi = wp.float64(3.1415926535897932384626433832795)
    two_pi = wp.float64(6.283185307179586476925286766559)
    wedge_on = al < pi
    a_wedge = wp.float64(0.0)
    a_full = wp.float64(0.0)
    for i in range(row_start[row], row_start[row + 1]):
        e = pe[perm[i]]
        if e < 0:                                                         # an empty slot of a fixed-capacity adjacency
            continue
        v0 = edges[2 * e]
        v1 = edges[2 * e + 1]
        Ax = verts[2 * v0] - px
        Ay = verts[2 * v0 + 1] - py
        Bx = verts[2 * v1] - px
        By = verts[2 * v1 + 1] - py
        cr = Ax * By - Ay * Bx
        L = wp.sqrt((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay))
        if L > wp.float64(0.0) and wp.abs(cr) >= wp.float64(1.0e-300):
            s = wp.float64(1.0)
            if cr < wp.float64(0.0):
                s = wp.float64(-1.0)
            z = wp.abs(cr) / L
            if z < H:
                pa = wp.atan2(Ay, Ax)
                dphi = _wrap_pi(wp.atan2(By, Bx) - pa)
                lo = pa
                hi = pa + dphi
                if dphi < wp.float64(0.0):
                    lo = pa + dphi
                    hi = pa
                dx = (Bx - Ax) / L
                dy = (By - Ay) / L
                t = Ax * dx + Ay * dy
                pn = lo + _wrap_pi(wp.atan2(Ay - t * dy, Ax - t * dx) - lo)
                be = wp.acos(z / H)
                a_full += s * _chord_deficit(lo, hi, z, H, be, pn)
                if wedge_on:
                    c = lo + _wrap_pi(th - lo)
                    for kk in range(-1, 2):
                        w0 = c + two_pi * wp.float64(kk) - al
                        w1 = c + two_pi * wp.float64(kk) + al
                        l = wp.max(lo, w0)
                        h = wp.min(hi, w1)
                        if l < h:
                            a_wedge += s * _chord_deficit(l, h, z, H, be, pn)
                else:
                    a_wedge += s * _chord_deficit(lo, hi, z, H, be, pn)
    idv = ind[row]
    w_wedge = two_pi
    if wedge_on:
        w_wedge = wp.float64(2.0) * al
    out[q] = out[q] + wp.float64(0.5) * H * H * w_wedge * idv - a_wedge
    out[N + q] = out[N + q] + wp.float64(0.5) * H * H * two_pi * idv - a_full


_SPEC_CACHE = {}


def _spec_arrays(outputs, B, N, dev, groups):
    """the output specification as device arrays, cached per (outputs, B, N): (o_group, o_kind, o_base, o_dim, o_ind, base list, pool size)."""
    key = (outputs, B, N, dev, tuple(g.channels for g in groups))
    if key not in _SPEC_CACHE:
        base, total = [], 0
        for o in outputs:
            base.append(total)
            total += B * N * KINDS[o.kind][1]
        i32 = lambda lst: wp.array(np.array(lst or [0], dtype=np.int32), dtype=int, device=dev)
        def needs_ind(o):
            ch = groups[o.group].channels
            return int(o.kind in ("lam", "cov", "a1g1") and (ch is None or bool(set(int(k) for k in ch) & {0, 5, 6, 7, 8})))
        _SPEC_CACHE[key] = (i32([o.group for o in outputs]), i32([KINDS[o.kind][0] for o in outputs]), i32(base), i32([KINDS[o.kind][1] for o in outputs]),
                            i32([needs_ind(o) for o in outputs]), base, total, any(needs_ind(o) for o in outputs))
    return _SPEC_CACHE[key]


class FusedWall:
    """stage 1 of the fused wall evaluation at one position set (see the module docstring)."""

    def __init__(self, scene, adjacency, groups):
        self.scene = scene
        self.adj = adjacency
        self.groups = tuple(groups)
        self.dev = str(scene.device)
        self.N = adjacency.numQueries
        self.items = []                                       # per (body, rep) with pairs / rows: dict(bi, rep, topo, ba, rows, keep, ind, w, R); topo is None for the table rows of a box, c is None for a polygon item that only serves cone_area
        for bi, ba in enumerate(adjacency.bodies):
            for rep, topo in zip(ba.body.reps, ba.reps):
                if not len(ba.cand):
                    continue
                if isinstance(rep, SurfaceRep):
                    if topo is not None:
                        self._add_item(bi, ba, rep, topo, range(len(self.groups)))
                elif isinstance(rep, BoxRep):
                    self._add_box(bi, ba, rep)
                else:
                    raise NotImplementedError("FusedWall: surface and box representations only (got %s)" % type(rep).__name__)

    def _item(self, bi, ba, rep, topo, c, perm, start, ind0):
        rows = len(ba.cand)
        cand32, lsup = ba.cand.to(torch.int32).contiguous(), ba.lsup.to(torch_real).contiguous()
        ang = float(ba.body.angle)
        cs, sn = float(np.cos(ang)), float(np.sin(ang))                 # Pose.R = [[c, -s], [s, c]]
        w = None if c is None else (wp.from_torch(start, dtype=wp.int32), wp.from_torch(perm, dtype=wp.int32), wp.from_torch(c, dtype=real),
                                    wp.from_torch(cand32, dtype=wp.int32), wp.from_torch(lsup, dtype=real))
        ind = None
        if ind0:                                                         # the indicator is part of the channels (box tables) or belongs to another item: add nothing in the contraction
            z = torch.zeros(rows, dtype=torch_real, device=self.dev)
            ind = (z, wp.from_torch(z, dtype=real))
        self.items.append(dict(bi=bi, rep=rep, topo=topo, ba=ba, rows=rows, keep=(c, perm, start, cand32, lsup), ind=ind, w=w, R=(cs, -sn, sn, cs)))

    def _add_item(self, bi, ba, rep, topo, gidx, ind0=False):
        """a polygon item: the pair channels of the groups `gidx` (zero-padded to the full layout), pairs of a row summed in a fixed order."""
        gidx = list(gidx)
        c = warpfused.fused_channels(topo.qi, topo.e, ba.lpos, ba.lsup, rep.vertices, rep.edges, tuple(self.groups[g] for g in gidx), device=self.dev) if gidx else None
        if c is not None and len(gidx) < len(self.groups):
            full = torch.zeros((c.shape[0], 9 * len(self.groups)), dtype=c.dtype, device=c.device)
            for k, g in enumerate(gidx):
                full[:, 9 * g:9 * g + 9] = c[:, 9 * k:9 * k + 9]
            c = full
        rows = len(ba.cand)
        perm, start = topo.csr(rows)
        self._item(bi, ba, rep, topo, c, perm, start, ind0)

    def _add_box(self, bi, ba, rep):
        """a box: the groups of tabulated kernels are four corner lookups per query row (no pairs, one row = one 'pair', the indicator is inside the channels); the kinked kernels of `BOX_EXACT_KERNELS` (`cone`)
        take the exact polygon of the same body (`rep.surface()`), which also serves `cone_area`."""
        tab = [g for g, grp in enumerate(self.groups) if grp.kernel not in BOX_EXACT_KERNELS]
        rows, dev = len(ba.cand), self.dev
        if tab:
            c = torch.zeros((rows, 9 * len(self.groups)), dtype=torch_real, device=dev)
            for kern in dict.fromkeys(self.groups[g].kernel for g in tab):
                t = boxTables(kern, dev)
                blk = t.block(ba.lpos, ba.lsup, (float(rep.lo[0]), float(rep.lo[1])), (float(rep.hi[0]), float(rep.hi[1])))      # [rows, 9] units of h, the library convention of warpbc.edge_channels
                if rep.solid == "outside":                                  # the solid is the plane minus the box
                    if getattr(t, "_plane", None) is None:
                        big = torch.full((1,), 1e3, dtype=F64, device=dev)
                        L, M, G0, G1 = t.channels(-big, big, -big, big)
                        t._plane = torch.cat([L[:, None], M, G0, G1.flatten(1)], 1)
                    blk = t._plane - blk
                if ba.valid is not None:                                    # fixed-capacity adjacency: every query is a row, only the candidates see the box
                    blk = blk * ba.valid[:, None]
                blk = blk.to(torch_real)
                for g in tab:
                    if self.groups[g].kernel == kern:
                        c[:, 9 * g:9 * g + 9] = blk
            ar = torch.arange(rows, dtype=torch.int32, device=dev)
            self._item(bi, ba, None, None, c, ar, torch.arange(rows + 1, dtype=torch.int32, device=dev), True)
        surf = rep.surface()
        topo = surf.topology(ba.lpos, ba.lsup) if ba.valid is None else fixed_topology(surf, ba.lpos, ba.lsup, ba.valid, self.adj.supportMax)
        if topo is not None:
            exact = [g for g in range(len(self.groups)) if g not in tab]
            self._add_item(bi, ba, surf, topo, exact, ind0=True)

    @staticmethod
    def supported(scene):
        return all(isinstance(r, (SurfaceRep, BoxRep)) for b in scene.bodies for r in b.reps)

    def evaluate(self, outputs, a1=None):
        """{name: tensor [B, N, ...]} float64, world frame.  `a1`: [B, N, 2] world vectors for the "a1g1" outputs."""
        outputs = tuple(outputs)
        B, N, dev = len(self.scene.bodies), self.N, self.dev
        o_group, o_kind, o_base, o_dim, o_ind, base, total, any_ind = _spec_arrays(outputs, B, N, dev, self.groups)
        pool = torch.zeros(max(total, 1), dtype=torch_real, device=dev)
        need_a1 = any(o.kind == "a1g1" for o in outputs)
        if need_a1 and a1 is None:
            raise ValueError("FusedWall.evaluate: a1 [B, N, 2] is required for 'a1g1' outputs")
        a1t = (a1 if need_a1 else torch.zeros(1, dtype=F64, device=dev)).to(torch_real).contiguous().reshape(-1)
        wa1, wpool = wp.from_torch(a1t, dtype=real), wp.from_torch(pool, dtype=real)
        for it in self.items:
            if it["w"] is None:
                continue
            if any_ind:
                if it["ind"] is None:
                    ind = it["topo"].indicator(it["rep"], it["ba"].lpos).to(torch_real).contiguous()
                    it["ind"] = (ind, wp.from_torch(ind, dtype=real))
                windv = it["ind"][1]
            else:
                if it.get("ind0") is None:
                    z = torch.zeros(it["rows"], dtype=torch_real, device=dev)
                    it["ind0"] = (z, wp.from_torch(z, dtype=real))
                windv = it["ind0"][1]
            wstart, wperm, wc, wcand, wlsup = it["w"]
            R00, R01, R10, R11 = it["R"]
            wp.launch(_wall_contract_kernel, dim=it["rows"], device=dev, inputs=[
                wstart, wperm, wc, wcand, wlsup, windv, real(R00), real(R01), real(R10), real(R11),
                len(outputs), o_group, o_kind, o_base, o_dim, o_ind, wa1, it["bi"] * N * 2, it["bi"], N, wpool])
        wp.synchronize_device(dev)
        res = {}
        for o, b0 in zip(outputs, base):
            dim = KINDS[o.kind][1]
            t = pool[b0:b0 + B * N * dim].to(F64)
            shape = {"lam": (B, N), "lap": (B, N), "g0": (B, N, 2), "a1g1": (B, N, 2), "cov": (B, N, 2, 2)}[o.kind]
            res[o.name] = t.reshape(shape)
        return res

    def cone_area(self, axes, half_angle):
        """[2, N] float64: area(solid cap disk(x, H) cap wedge(x, axis, half_angle)) (row 0, units length^2; `half_angle >= pi` = the full disk) and the full-disk area (row 1) for the world `axes` [N, 2]
        (only the direction matters), summed over the bodies (a query outside the candidate list of a background rep is deep in its solid: the full sector); one launch per (body, rep) over the pair topology of this position set (`scene/cone_area.py` is the reference).  The disk radius is the support of the
        query, so the pair filter (edges within one support) contains every edge with a chord.  Evaluated in float64 whatever the edge-kernel precision (angle differences near pi/2 amplify float32 round-off)."""
        N, dev = self.N, self.dev
        out = torch.zeros(2 * N, dtype=F64, device=dev)
        wout = wp.from_torch(out, dtype=wp.float64)
        axis = wp.from_torch(axes.to(F64).contiguous().reshape(-1), dtype=wp.float64)
        for it in self.items:
            if it["topo"] is None:                                                    # the table rows of a box (its polygon item serves)
                continue
            ba, rep = it["ba"], it["rep"]
            _, perm, start, cand32, _ = it["keep"]
            if it.get("cone") is None:
                lpos = ba.lpos.to(F64).contiguous().reshape(-1)
                lsup = ba.lsup.to(F64).contiguous()
                ind = it["topo"].indicator(rep, ba.lpos).to(F64).contiguous()
                verts = rep.vertices.to(dev, F64).contiguous().reshape(-1)
                edges = rep.edges.to(dev, torch.int32).contiguous().reshape(-1)
                pe = it["topo"].e.to(torch.int32).contiguous()
                it["cone"] = (lpos, lsup, ind, verts, edges, pe,
                              tuple(wp.from_torch(t, dtype=dt) for t, dt in ((lpos, wp.float64), (lsup, wp.float64), (ind, wp.float64), (verts, wp.float64), (edges, wp.int32), (pe, wp.int32))))
            wl, ws, wi, wv, we, wpe = it["cone"][6]
            wp.launch(_cone_area_kernel, dim=it["rows"], device=dev, inputs=[
                wp.from_torch(start, dtype=wp.int32), wp.from_torch(perm, dtype=wp.int32), wpe, wp.from_torch(cand32, dtype=wp.int32),
                wl, ws, wi, wv, we, axis, wp.float64(float(ba.body.angle)), wp.float64(float(half_angle)), N, wout])
            if rep.background:
                out_of = torch.ones(N, dtype=F64, device=dev)
                if ba.valid is None:
                    out_of[ba.cand.long()] = 0.0
                else:
                    out_of = (~ba.valid).to(F64)
                sup = self.adj.queryParticles.supports.to(dev, F64)
                wedge = 2.0 * float(half_angle) if float(half_angle) < math.pi else 2.0 * math.pi
                out[:N] += out_of * 0.5 * sup * sup * wedge
                out[N:] += out_of * math.pi * sup * sup
        wp.synchronize_device(dev)
        return out.reshape(2, N)
