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
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from ..edge import warpfused
from ..edge.precision import real, torch_real
from .scene import SurfaceRep

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
        self.items = []                                       # per (body, surface rep) with pairs: dict(bi, rep, topo, ba, c, wp arrays, rotation)
        for bi, ba in enumerate(adjacency.bodies):
            for rep, topo in zip(ba.body.reps, ba.reps):
                if not isinstance(rep, SurfaceRep):
                    raise NotImplementedError("FusedWall: surface representations only (got %s)" % type(rep).__name__)
                if topo is None or not len(ba.cand):
                    continue
                c = warpfused.fused_channels(topo.qi, topo.e, ba.lpos, ba.lsup, rep.vertices, rep.edges, self.groups, device=self.dev)
                rows = len(ba.cand)
                perm, start = topo.csr(rows)
                cand32, lsup = ba.cand.to(torch.int32).contiguous(), ba.lsup.to(torch_real).contiguous()
                ang = float(ba.body.angle)
                cs, sn = float(np.cos(ang)), float(np.sin(ang))                 # Pose.R = [[c, -s], [s, c]]
                self.items.append(dict(bi=bi, rep=rep, topo=topo, ba=ba, rows=rows, keep=(c, perm, start, cand32, lsup), ind=None,
                                       w=(wp.from_torch(start, dtype=wp.int32), wp.from_torch(perm, dtype=wp.int32), wp.from_torch(c, dtype=real),
                                          wp.from_torch(cand32, dtype=wp.int32), wp.from_torch(lsup, dtype=real)), R=(cs, -sn, sn, cs)))

    @staticmethod
    def supported(scene):
        return all(isinstance(r, SurfaceRep) for b in scene.bodies for r in b.reps)

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
