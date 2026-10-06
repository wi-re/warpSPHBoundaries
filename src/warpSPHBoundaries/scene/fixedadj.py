"""Fixed-capacity, sync-free wall adjacency (docs/plan-wall-evaluation.md step 5): the pair topology of every query against every surface representation in one Warp launch per (body, rep), no
`torch.nonzero`, no host read, shapes that depend on nothing but N and the static geometry (so the whole thing can be captured in a CUDA graph).

    adj = fixed_adjacency(scene, queryParticles, operationProperties, supportMax)       # SceneAdjacency whose bodies carry every query as a row (`valid` marks the candidates)

Rows = queries (cand = arange(N)); `valid[r]` is the broadphase test of `Scene.candidates` (support sphere against the body OBB, and the query kind is served).  A surface representation gets
`FixedTopology`: K slots per row, `e[r * K + s]` = edge of slot s (-1: empty), filled in the order of the static edge cell list (the order of the sorted pair list of `SurfaceRep.topology`, so the per-row
sums of the fused contraction are the same sums), and the indicator of the rows (winding number + background, 0 for invalid rows; the `indicatorFast` logic: cell cache, outside the grid, y-bin ray casting).
K = the largest number of edges stored in one cell of the static cell list: the slots of a row cannot overflow.  Everything is evaluated in float64.
"""
from dataclasses import dataclass

import torch
import warp as wp

from .boundaryOps import queryAllowed
from .scene import BodyAdjacency, DiskArrayRep, SceneAdjacency, SurfaceRep

F64 = torch.float64
wf = wp.float64


@dataclass
class FixedTopology:
    """duck-types `SurfaceTopology` for `FusedWall`: slots instead of a pair list.  `e` [N * K] int32 (-1 = empty), `qi` [N * K] the row of a slot, `ind` [N] float64."""
    e: torch.Tensor
    qi: torch.Tensor
    K: int
    ind: torch.Tensor
    _csr: tuple = None

    def indicator(self, rep, lpos):
        return self.ind

    def csr(self, rows):
        if self._csr is None:
            self._csr = (torch.arange(rows * self.K, dtype=torch.int32, device=self.e.device), torch.arange(rows + 1, dtype=torch.int32, device=self.e.device) * self.K)
        return self._csr


@wp.kernel
def _fixed_topology_kernel(lpos: wp.array(dtype=wf), lsup: wp.array(dtype=wf), valid: wp.array(dtype=int), verts: wp.array(dtype=wf), edges: wp.array(dtype=int),
                           clo0: wf, clo1: wf, cell: wf, nx: int, ny: int, cstart: wp.array(dtype=int), citems: wp.array(dtype=int), K: int,
                           cind: wp.array(dtype=wf), cempty: wp.array(dtype=int), bg: wf, y0: wf, dy: wf, nb: int, bstart: wp.array(dtype=int), bitems: wp.array(dtype=int),
                           e_out: wp.array(dtype=int), ind_out: wp.array(dtype=wf)):
    r = wp.tid()
    if valid[r] == 0:
        return
    px = lpos[2 * r]
    py = lpos[2 * r + 1]
    h = lsup[r]
    cx = int(wp.floor((px - clo0) / cell))
    cy = int(wp.floor((py - clo1) / cell))
    inside = cx >= 0 and cx < nx and cy >= 0 and cy < ny
    if inside:
        cid = cx * ny + cy
        n = int(0)
        for i in range(cstart[cid], cstart[cid + 1]):
            e = citems[i]
            v0 = edges[2 * e]
            v1 = edges[2 * e + 1]
            ax = verts[2 * v0]
            ay = verts[2 * v0 + 1]
            dx = verts[2 * v1] - ax
            dyy = verts[2 * v1 + 1] - ay
            t = wp.clamp(((px - ax) * dx + (py - ay) * dyy) / wp.max(dx * dx + dyy * dyy, wf(1.0e-300)), wf(0.0), wf(1.0))
            ex = px - (ax + t * dx)
            ey = py - (ay + t * dyy)
            if wp.sqrt(ex * ex + ey * ey) < h:
                e_out[r * K + n] = e
                n += 1
    ind = bg
    fast = int(0)
    if inside:
        if cempty[cx * ny + cy] != 0:
            ind = cind[cx * ny + cy]
            fast = 1
    if inside and fast == 0:                                                       # not in an edge-free cell: ray casting along the y bin (the outside of the grid is the background)
        iy = int(wp.floor((py - y0) / dy))
        if iy >= 0 and iy < nb:
            w = wf(0.0)
            for i in range(bstart[iy], bstart[iy + 1]):
                e = bitems[i]
                v0 = edges[2 * e]
                v1 = edges[2 * e + 1]
                ax = verts[2 * v0]
                ay = verts[2 * v0 + 1]
                cxx = verts[2 * v1]
                cyy = verts[2 * v1 + 1]
                isLeft = (cxx - ax) * (py - ay) - (cyy - ay) * (px - ax)
                if ay <= py and py < cyy and isLeft > wf(0.0):
                    w += wf(1.0)
                if cyy <= py and py < ay and isLeft < wf(0.0):
                    w -= wf(1.0)
            ind = w + bg
    ind_out[r] = ind


def _static(rep, supportMax, dev):
    """the device copies of the static acceleration structures of a SurfaceRep for the support `supportMax`, cached on the rep: (K, arrays, scalars)."""
    wp.init()                                                                          # idempotent; the first launch of a process may be this one
    cache = rep.__dict__.setdefault("_fixedStatic", {})
    key = (supportMax, dev)
    if key not in cache:
        cl = rep._celllist(supportMax)
        ind, empty = rep._cell_indicator()
        y0, dy, nb, bstart, bitems = rep._ybins()
        counts = cl.start[1:] - cl.start[:-1]
        i32 = lambda t: t.to(torch.int32).contiguous()
        arrs = dict(verts=rep.vertices.to(F64).contiguous().reshape(-1), edges=i32(rep.edges).reshape(-1), cstart=i32(cl.start), citems=i32(cl.items), cind=ind.to(F64).contiguous(),
                    cempty=i32(empty), bstart=i32(bstart), bitems=i32(bitems))
        cache[key] = (max(1, int(counts.max())), arrs, dict(clo0=float(cl.lo[0]), clo1=float(cl.lo[1]), cell=float(cl.cell), nx=int(cl.dims[0]), ny=int(cl.dims[1]), y0=float(y0), dy=float(dy), nb=int(nb)),
                      {k: wp.from_torch(v, dtype=wf if v.dtype == F64 else wp.int32) for k, v in arrs.items()})
    return cache[key]


def fixed_topology(rep, lpos, lsup, valid, supportMax):
    """`FixedTopology` of the rows `lpos` [N, 2] (supports `lsup`, `valid` [N] bool) against the SurfaceRep `rep`; `supportMax` is the host float the static cell list is built for (>= every support)."""
    dev = str(lpos.device)
    K, _, sc, wa = _static(rep, float(supportMax), dev)
    N = len(lpos)
    e = torch.full((N * K,), -1, dtype=torch.int32, device=lpos.device)
    ind = torch.zeros(N, dtype=F64, device=lpos.device)
    lp = lpos.to(F64).contiguous().reshape(-1)
    ls = lsup.to(F64).contiguous()
    vi = valid.to(torch.int32).contiguous()
    wp.launch(_fixed_topology_kernel, dim=N, device=dev, inputs=[
        wp.from_torch(lp, dtype=wf), wp.from_torch(ls, dtype=wf), wp.from_torch(vi, dtype=wp.int32), wa["verts"], wa["edges"], wf(sc["clo0"]), wf(sc["clo1"]), wf(sc["cell"]), sc["nx"], sc["ny"],
        wa["cstart"], wa["citems"], K, wa["cind"], wa["cempty"], wf(float(rep.background)), wf(sc["y0"]), wf(sc["dy"]), sc["nb"], wa["bstart"], wa["bitems"],
        wp.from_torch(e, dtype=wp.int32), wp.from_torch(ind, dtype=wf)])
    return FixedTopology(e, torch.arange(N * K, device=lpos.device, dtype=torch.int32) // K, K, ind)


@dataclass
class FixedDiskTopology:
    """duck-types `FixedTopology` for a `DiskArrayRep`: K slots per row, `e[r * K + s]` = the disk of slot s (-1: empty), `qi` the row of a slot; no indicator (the disk channels are complete)."""
    e: torch.Tensor
    qi: torch.Tensor
    K: int
    _csr: tuple = None

    def csr(self, rows):
        if self._csr is None:
            self._csr = (torch.arange(rows * self.K, dtype=torch.int32, device=self.e.device), torch.arange(rows + 1, dtype=torch.int32, device=self.e.device) * self.K)
        return self._csr


@wp.kernel
def _disk_slots_kernel(lpos: wp.array(dtype=wf), lsup: wp.array(dtype=wf), valid: wp.array(dtype=int), centres: wp.array(dtype=wf), radii: wp.array(dtype=wf),
                       clo0: wf, clo1: wf, cell: wf, nx: int, ny: int, cstart: wp.array(dtype=int), citems: wp.array(dtype=int), K: int, e_out: wp.array(dtype=int)):
    r = wp.tid()
    if valid[r] == 0:
        return
    px = lpos[2 * r]
    py = lpos[2 * r + 1]
    h = lsup[r]
    cx = int(wp.floor((px - clo0) / cell))
    cy = int(wp.floor((py - clo1) / cell))
    n = int(0)
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            ix = cx + ddx
            iy = cy + ddy
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                cid = ix * ny + iy
                for i in range(cstart[cid], cstart[cid + 1]):
                    m = citems[i]
                    ex = px - centres[2 * m]
                    ey = py - centres[2 * m + 1]
                    if wp.sqrt(ex * ex + ey * ey) < h + radii[m]:
                        if n < K:
                            e_out[r * K + n] = m
                            n += 1


def fixed_disk_topology(rep, lpos, lsup, valid, supportMax):
    """`FixedDiskTopology` of the rows `lpos` [N, 2] (supports `lsup`, `valid` [N] bool) against the disks of `rep`: one launch, fixed shapes, no host synchronisation (the static cell list is built once)."""
    wp.init()
    dev = str(lpos.device)
    lo, cell, nx, ny, start, items, K = rep.cells(float(supportMax))
    N = len(lpos)
    e = torch.full((N * K,), -1, dtype=torch.int32, device=lpos.device)
    cache = rep.__dict__.setdefault("_slotArrays", {})
    key = (float(supportMax), dev)
    if key not in cache:
        cen = rep.centres.to(F64).contiguous().reshape(-1)
        rad = rep.radii.to(F64).contiguous()
        cache[key] = (cen, rad, wp.from_torch(cen, dtype=wf), wp.from_torch(rad, dtype=wf), wp.from_torch(start, dtype=wp.int32), wp.from_torch(items, dtype=wp.int32), float(lo[0]), float(lo[1]))
    cen, rad, wc, wr, wst, wit, lo0, lo1 = cache[key]
    lp = lpos.to(F64).contiguous().reshape(-1)
    ls = lsup.to(F64).contiguous()
    vi = valid.to(torch.int32).contiguous()
    wp.launch(_disk_slots_kernel, dim=N, device=dev, inputs=[wp.from_torch(lp, dtype=wf), wp.from_torch(ls, dtype=wf), wp.from_torch(vi, dtype=wp.int32), wc, wr, wf(lo0), wf(lo1), wf(cell), nx, ny,
                                                              wst, wit, K, wp.from_torch(e, dtype=wp.int32)])
    return FixedDiskTopology(e, torch.arange(N * K, device=lpos.device, dtype=torch.int32) // K, K)


def fixed_adjacency(scene, queryParticles, operationProperties, supportMax):
    """the `SceneAdjacency` of every query against every body with fixed shapes (see the module docstring); `supportMax` host float >= every query support."""
    dev = scene.device
    pos, sup = queryParticles.positions.to(dev, F64), queryParticles.supports.to(dev, F64)
    if pos.shape[1] != 2:
        raise NotImplementedError("2D only")
    allowed = queryAllowed(queryParticles, operationProperties.operationMode, dev)
    N = len(pos)
    cand = torch.arange(N, device=dev)
    bodies = []
    for body in scene.bodies:
        lo, hi = body.obb()
        lpos = body.toLocal(pos)
        d = (lo - lpos).clamp(min=0) + (lpos - hi).clamp(min=0)
        valid = (d.norm(dim=1) < sup) & allowed
        replist = body.fusedReps(supportMax, dev) or body.reps                  # implicit / SDF bodies as their exact polygon
        reps = [fixed_topology(rep, lpos, sup, valid, supportMax) if isinstance(rep, SurfaceRep) else (fixed_disk_topology(rep, lpos, sup, valid, supportMax) if isinstance(rep, DiskArrayRep) else None) for rep in replist]
        bodies.append(BodyAdjacency(body, cand, lpos, sup, reps, valid, replist))
    adj = SceneAdjacency(N, bodies, {"candidates": None}, queryParticles)
    adj.supportMax = float(supportMax)
    return adj


def indicator_device(rep, lpos, supportMax):
    """winding number + background of the points `lpos` [N, 2] (body frame) against the SurfaceRep `rep`, without a host sync (the `fixed_topology` kernel with every row valid; the edge slots are not used)."""
    n = len(lpos)
    return fixed_topology(rep, lpos, torch.full((n,), float(supportMax), dtype=F64, device=lpos.device), torch.ones(n, dtype=torch.bool, device=lpos.device), supportMax).ind
