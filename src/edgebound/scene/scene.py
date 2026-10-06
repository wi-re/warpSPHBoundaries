"""Scene layer (2D): bodies with a pose and an OBB, per-representation adjacency, per-type boundary operations.

    scene = Scene([Body(center=(0, 0), angle=0.3, reps=[SurfaceRep.polygon(P)], bodyId=0), Body(..., reps=[ImplicitRep(DiskBody(...))])])
    adj = scene.adjacency(queryParticles, operationProperties)                          # who interacts with whom: broadphase + pair topology (integers, kernel-free)
    pm = scene.precompute(adj, operationProperties)                                     # the per-pair integrals of one kernel (floats; opt-in, shared by operations at one position set)
    out = sceneOperation(queryParticles, operationProperties, scene, pm, queryValues, bodyFields)       # the evaluation (scene.pairMoments = adjacency + precompute)

Structure (docs/scene-architecture.md):

* a `Body` is a rigid frame (`center`, `angle`, velocities -- the quantities warpSPH's `RigidBody` tracks) with an oriented bounding box in its LOCAL frame and any
  number of REPRESENTATIONS, all expressed in local coordinates:
      SurfaceRep   closed polylines (counter-clockwise around the solid): exact, edge-local pair terms + a body-level indicator   [no triangulation needed]
      VolumeRep    triangle elements with P1 nodal fields                                                                        [`boundaryOps` pair engine]
      ImplicitRep  analytic primitive (`DiskBody`, `HalfPlaneBody`): hard tier 3 / tier 4 / polygon-surface switch per particle
      SdfRep       sampled signed distance (generic): tier 3 where the SDF is smooth at the scale of h, fallback surface elsewhere
* broadphase: the particle's support sphere against the body OBB in the body frame (a rigid transform keeps the sphere a sphere); narrow phase: each
  representation's own static acceleration structure (built once in the local frame -- rigid bodies never rebuild anything);
* adjacency (topology) / precompute (`PairMoments`) / evaluation (`sceneOperation`), see docs/plan-wall-evaluation.md §1; one operation kernel per type (no branch divergence on the type); all boundary terms are linear in the boundary data, so the types
  just accumulate into the same output.  Vectors are rotated back to the world frame after the local-frame evaluation.
* boundary data of a body: `BodyField(a0, a1, rho)`, the field A(x') = a0 + a1 (x' - center) in world components (a constant wall, a rigidly moving wall
  `BodyField.rigid(body)`, any field linear in position); a1 needs the first moments (surface and volume representations), implicit/SDF are constant-only.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from warpSPHCore import GradientScheme, OperationProperties, WarpOperation

from ..edge import warpbc
from .boundaryOps import (BoundaryMesh, queryAllowed, tier3Table, tier4Table, boundaryOperation,
                          buildBoundaryAdjacency, buildElementGrid, kernelName)
from .box import BoxTables, chebyshev_nodes, fit_panels
from .implicitBodies import DiskBody, HalfPlaneBody, TierPolicy, evaluateBody

F64 = torch.float64


# ----------------------------------------------------------------------------------------------------------------------- frames
_R_CACHE = {}                                                      # (angle, device) -> rotation matrix


@dataclass
class Pose:
    center: torch.Tensor            # [2] world position of the body origin (centre of mass)
    angle: float = 0.0

    @property
    def R(self):
        key = (float(self.angle), str(self.center.device))
        R = _R_CACHE.get(key)
        if R is None:                                              # a static body builds its rotation once (a host-to-device copy synchronises and cannot be graph-captured)
            if len(_R_CACHE) > 4096:                               # a rotating body makes a new angle every step
                _R_CACHE.clear()
            c, s = float(np.cos(key[0])), float(np.sin(key[0]))
            R = _R_CACHE[key] = torch.tensor([[c, -s], [s, c]], dtype=F64, device=self.center.device)
        return R

    def toLocal(self, p):
        """(p - center) R with the 2 x 2 rotation written out: a [N,2] x [2,2] float64 GEMM is a 0.27 ms cutlass launch on this class of GPU, the elementwise form is four tiny kernels."""
        c, s = float(np.cos(self.angle)), float(np.sin(self.angle))
        d = p - self.center
        return torch.stack([d[:, 0] * c + d[:, 1] * s, d[:, 1] * c - d[:, 0] * s], 1)

    def toWorld(self, p):
        c, s = float(np.cos(self.angle)), float(np.sin(self.angle))
        return torch.stack([p[:, 0] * c - p[:, 1] * s + self.center[0], p[:, 0] * s + p[:, 1] * c + self.center[1]], 1)

    def vecToWorld(self, v):
        return v @ self.R.T


# ----------------------------------------------------------------------------------------------------------------------- cell lists
@dataclass
class CellList:
    lo: torch.Tensor
    cell: float
    dims: tuple
    start: torch.Tensor
    items: torch.Tensor


def buildCellList(boxLo, boxHi, cell):
    """uniform 2D cell list: every item is inserted into all cells overlapped by its box (so a point only looks at its own cell)."""
    dev = boxLo.device
    gmin, gmax = boxLo.amin(0), boxHi.amax(0)
    dims = torch.clamp(torch.ceil((gmax - gmin) / cell).long(), min=1)
    nx, ny = int(dims[0]), int(dims[1])
    i0 = torch.floor((boxLo - gmin) / cell).long().clamp(min=0)
    i1 = torch.minimum(torch.floor((boxHi - gmin) / cell).long(), dims - 1)
    cnt = (i1 - i0 + 1).clamp(min=0)
    per = cnt[:, 0] * cnt[:, 1]
    item = torch.repeat_interleave(torch.arange(len(boxLo), device=dev), per)
    start = torch.cumsum(per, 0) - per
    local = torch.arange(int(per.sum()), device=dev) - torch.repeat_interleave(start, per)
    cx = i0[item, 0] + local % cnt[item, 0]
    cy = i0[item, 1] + local // cnt[item, 0]
    cellId = cx * ny + cy
    order = torch.argsort(cellId, stable=True)
    counts = torch.bincount(cellId[order], minlength=nx * ny)
    st = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)])
    return CellList(gmin, float(cell), (nx, ny), st, item[order])


def queryCellList(cl: CellList, pos):
    """(query index, item) candidate pairs: the items stored in the cell of each query point."""
    nx, ny = cl.dims
    cx = torch.floor((pos[:, 0] - cl.lo[0]) / cl.cell).long()
    cy = torch.floor((pos[:, 1] - cl.lo[1]) / cl.cell).long()
    ok = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    cell = torch.where(ok, cx * ny + cy, torch.zeros_like(cx))
    cnt = torch.where(ok, cl.start[cell + 1] - cl.start[cell], torch.zeros_like(cell))
    q = torch.repeat_interleave(torch.arange(len(pos), device=pos.device), cnt)
    off = torch.arange(int(cnt.sum()), device=pos.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    return q, cl.items[cl.start[cell[q]] + off]


def _segment_distance(p, a, b):
    d = b - a
    t = (((p - a) * d).sum(1) / (d * d).sum(1).clamp(min=1e-300)).clamp(0, 1)
    return (p - (a + t[:, None] * d)).norm(dim=1)


# ----------------------------------------------------------------------------------------------------------------------- pair sets
@dataclass
class SurfaceTopology:
    """the topology of one SurfaceRep against a set of queries: the (query row, edge) pairs inside one support (rows index the body's candidate list) and, on
    demand, the winding-number indicator of the rows.  Integers and the kernel-independent indicator only: the per-pair integrals are `SurfaceRep.moments`."""
    qi: torch.Tensor                 # [P] int32 row into the body's candidate list
    e: torch.Tensor                  # [P] int32 edge
    supMax: float                    # largest support of the rows when the pairs were found (sizes the cell list of the indicator)
    _ind: Optional[torch.Tensor] = None        # [rows] winding number + background, computed on first use
    _csr: Optional[tuple] = None               # (pair order sorted by row, row offsets [rows + 1]): the fused contraction sums the pairs of a row in a fixed order

    def indicator(self, rep, lpos):
        if self._ind is None:
            self._ind = rep.indicatorFast(lpos, self.supMax)
        return self._ind

    def csr(self, rows):
        """(perm [P] int32, start [rows + 1] int32): the pair ids sorted by row (stable) and the first sorted position of every row; computed once per topology."""
        if self._csr is None:
            q = self.qi.long()
            order = torch.sort(q, stable=True).indices
            start = torch.zeros(rows + 1, dtype=torch.long, device=q.device)
            start[1:] = torch.cumsum(torch.bincount(q, minlength=rows), 0)
            self._csr = (order.to(torch.int32), start.to(torch.int32))
        return self._csr

    def restrict(self, rowmap, rows):
        """the topology of the rows `rows` (sorted, into the candidate list) with new row numbers `rowmap` (-1 for dropped rows)."""
        new = rowmap[self.qi.long()]
        sel = new >= 0
        out = SurfaceTopology(new[sel].to(torch.int32), self.e[sel], self.supMax)
        if self._ind is not None:
            out._ind = self._ind[rows]
        return out


@dataclass
class RepMoments:
    """per-pair moments of the body kernel integral, LOCAL frame -> (after `toWorld`) world frame, physical units:
        lam = int W,  m1 = int y W,  g0 = int grad_x W,  g1[d, j] = int y_d d_j W   (y = x' - x_i);  m1 / g1 are None for models without first moments."""
    q: torch.Tensor                  # [P] global query index
    lam: torch.Tensor                # [P]
    g0: torch.Tensor                 # [P,2]
    m1: Optional[torch.Tensor] = None   # [P,2]
    g1: Optional[torch.Tensor] = None   # [P,2,2]

    def toWorld(self, pose: Pose):
        R = pose.R
        return RepMoments(self.q, self.lam, self.g0 @ R.T, None if self.m1 is None else self.m1 @ R.T,
                           None if self.g1 is None else R @ self.g1 @ R.T)


# ----------------------------------------------------------------------------------------------------------------------- representations
class SurfaceRep:
    """closed polylines in the body frame, counter-clockwise around the SOLID (solid on the left of every edge); `background = 1` when the solid is the
    unbounded outside (a tank wall / cavity: clockwise loop around the fluid).  lambda(x) = indicator(x) + sum_edges (edge-local terms): exact, local, and
    needs no triangulation.  The indicator is the winding number (+ background), computed with the same orientation predicate as the edge terms."""
    kind = "surface"

    def __init__(self, vertices, edges, background: int = 0):
        self.vertices = torch.as_tensor(vertices, dtype=F64)
        self.edges = torch.as_tensor(edges, dtype=torch.int32)
        self.background = int(background)
        self._cl = None
        self._clSupport = -1.0
        self._cellInd = None
        self._bins = None

    @staticmethod
    def polygon(points, solid: str = "inside"):
        """one closed loop through `points` ([n,2]); `solid='inside'`: the polygon is the body (reordered counter-clockwise), 'outside': the polygon is a hole
        in an infinite solid (clockwise orientation, background 1)."""
        P = np.asarray(points, dtype=float)
        a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        if (a > 0) != (solid == "inside"):
            P = P[::-1].copy()
        n = len(P)
        E = np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)
        return SurfaceRep(P, E, background=0 if solid == "inside" else 1)

    @staticmethod
    def box(lo, hi, solid: str = "inside"):
        (x0, y0), (x1, y1) = lo, hi
        return SurfaceRep.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], solid)

    @staticmethod
    def regularPolygon(center, radius, n, solid="inside", areaPreserving=True):
        r = radius * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n))) if areaPreserving else radius
        t = 2 * np.pi * np.arange(n) / n
        return SurfaceRep.polygon(np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], 1), solid)

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.edges = self.edges.to(device)
        return self

    def bounds(self):
        return self.vertices.amin(0), self.vertices.amax(0)

    # ---- static acceleration structures
    def _celllist(self, supportMax):
        if self._cl is None or supportMax > self._clSupport:
            a, b = self.vertices[self.edges[:, 0].long()], self.vertices[self.edges[:, 1].long()]
            lo, hi = torch.minimum(a, b) - supportMax, torch.maximum(a, b) + supportMax
            self._cl, self._clSupport = buildCellList(lo, hi, supportMax), supportMax
            self._cellInd = None
        return self._cl

    def _cell_indicator(self):
        """static per-cell indicator for the cells that hold NO edge within one support radius (the indicator is constant on them): computed once per
        grid build by ray casting the cell centres -- the per-step cost for the bulk of the particles is then a lookup."""
        if self._cellInd is None:
            cl = self._cl
            nx, ny = cl.dims
            empty = torch.nonzero((cl.start[1:] - cl.start[:-1]) == 0).flatten()
            ind = torch.zeros(nx * ny, dtype=F64, device=cl.lo.device)
            if len(empty):
                cx, cy = empty // ny, empty % ny
                centres = cl.lo + (torch.stack([cx, cy], 1).to(F64) + 0.5) * cl.cell
                ind[empty] = self.indicator(centres)
            self._cellInd = (ind, (cl.start[1:] - cl.start[:-1]) == 0)
        return self._cellInd

    def indicatorFast(self, p, supportMax):
        """`indicator`, with the cell cache for points in edge-free cells (exact: no edge within one support of such a point, so nothing can cross)."""
        cl = self._celllist(supportMax)
        ind, empty = self._cell_indicator()
        nx, ny = cl.dims
        cx = torch.floor((p[:, 0] - cl.lo[0]) / cl.cell).long()
        cy = torch.floor((p[:, 1] - cl.lo[1]) / cl.cell).long()
        inside = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        cell = torch.where(inside, cx * ny + cy, torch.zeros_like(cx))
        fast = inside & empty[cell]
        out = torch.where(fast, ind[cell], torch.zeros_like(p[:, 0]))
        outside = ~inside                                           # beyond the grid: more than a support from every edge and outside their bounding box
        out = torch.where(outside, torch.full_like(out, float(self.background)), out)
        slow = torch.nonzero(~(fast | outside)).flatten()
        if len(slow):
            out[slow] = self.indicator(p[slow])
        return out

    def _ybins(self, nbins=None):
        if self._bins is None:
            a, b = self.vertices[self.edges[:, 0].long()], self.vertices[self.edges[:, 1].long()]
            ylo, yhi = torch.minimum(a[:, 1], b[:, 1]), torch.maximum(a[:, 1], b[:, 1])
            nb = nbins or max(1, int(np.sqrt(len(self.edges))) * 2)
            y0 = float(ylo.min())
            dy = max(float(yhi.max()) - y0, 1e-300) / nb
            i0 = torch.floor((ylo - y0) / dy).long().clamp(0, nb - 1)
            i1 = torch.floor((yhi - y0) / dy).long().clamp(0, nb - 1)
            cnt = i1 - i0 + 1
            item = torch.repeat_interleave(torch.arange(len(a), device=a.device), cnt)
            st = torch.cumsum(cnt, 0) - cnt
            b_ = i0[item] + torch.arange(int(cnt.sum()), device=a.device) - torch.repeat_interleave(st, cnt)
            order = torch.argsort(b_, stable=True)
            counts = torch.bincount(b_[order], minlength=nb)
            start = torch.cat([torch.zeros(1, dtype=torch.long, device=a.device), torch.cumsum(counts, 0)])
            self._bins = (y0, dy, nb, start, item[order])
        return self._bins

    def indicator(self, p):
        """winding number of the closed loops around each point (+ background): 1 inside the solid, 0 outside (loops must be simple and consistently oriented).
        Half-open crossing rule, predicate cross(q - p, x - p) = the sign of the edge term z."""
        y0, dy, nb, start, items = self._ybins()
        iy = torch.floor((p[:, 1] - y0) / dy).long()
        ok = (iy >= 0) & (iy < nb)
        b = torch.where(ok, iy, torch.zeros_like(iy))
        cnt = torch.where(ok, start[b + 1] - start[b], torch.zeros_like(b))
        q = torch.repeat_interleave(torch.arange(len(p), device=p.device), cnt)
        off = torch.arange(int(cnt.sum()), device=p.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        e = items[start[b[q]] + off]
        a, c = self.vertices[self.edges[e, 0].long()], self.vertices[self.edges[e, 1].long()]
        x = p[q]
        isLeft = (c[:, 0] - a[:, 0]) * (x[:, 1] - a[:, 1]) - (c[:, 1] - a[:, 1]) * (x[:, 0] - a[:, 0])
        up = (a[:, 1] <= x[:, 1]) & (x[:, 1] < c[:, 1]) & (isLeft > 0)
        down = (c[:, 1] <= x[:, 1]) & (x[:, 1] < a[:, 1]) & (isLeft < 0)
        w = up.to(F64) - down.to(F64)
        wn = torch.zeros(len(p), dtype=F64, device=p.device).index_add_(0, q, w)
        return wn + self.background

    # ---- adjacency (topology) and precompute (moments), local frame
    def topology(self, lpos, lsup):
        """`SurfaceTopology` of the queries `lpos` (supports `lsup`) against this surface: the (query, edge) pairs with the segment closer than the support."""
        if len(lpos) == 0:
            return None
        cl = self._celllist(float(lsup.max()))
        qi, e = queryCellList(cl, lpos)
        a, b = self.vertices[self.edges[e, 0].long()], self.vertices[self.edges[e, 1].long()]
        keep = _segment_distance(lpos[qi], a, b) < lsup[qi]
        return SurfaceTopology(qi[keep].to(torch.int32), e[keep].to(torch.int32), float(lsup.max()))

    def moments(self, topo, lpos, lsup, qglobal, kernel, device, channels=None):
        """RepMoments (local frame) of the topology `topo` (exact: edge terms + indicator pseudo-pairs).  `channels` (default all nine of warpbc.edge_channels):
        only these are evaluated, the others are 0; the indicator pseudo-pairs (lam, g1) are skipped when no channel of lam / g1 is asked for."""
        qi, e = topo.qi, topo.e
        c = warpbc.edge_channels(qi, e, lpos, lsup, self.vertices, self.edges, kernel, device=str(device), channels=channels)        # [P,9] units of h
        h = lsup[qi.long()]
        lam = c[:, 0]
        m1 = h[:, None] * c[:, 1:3]
        g0 = c[:, 3:5] / h[:, None]
        g1 = torch.stack([c[:, 5:7], c[:, 7:9]], 1)
        if channels is not None and not (set(int(k) for k in channels) & {0, 5, 6, 7, 8}):
            return RepMoments(qglobal[qi.long()], lam, g0, m1, g1)                                  # gradient-only: the indicator adds lam and g1 only
        ind = topo.indicator(self, lpos)
        nz = torch.nonzero(ind != 0).flatten()
        eye = torch.eye(2, dtype=F64, device=device)
        qs = torch.cat([qi.long(), nz])
        return RepMoments(qglobal[qs], torch.cat([lam, ind[nz]]), torch.cat([g0, torch.zeros((len(nz), 2), dtype=F64, device=device)]),
                          torch.cat([m1, torch.zeros((len(nz), 2), dtype=F64, device=device)]),
                          torch.cat([g1, ind[nz][:, None, None] * eye]))

    def pairs(self, lpos, lsup, qglobal, kernel, device, channels=None):
        """topology + moments in one call (the fallback surfaces of implicit representations, which have no shared adjacency)."""
        topo = self.topology(lpos, lsup)
        return None if topo is None else self.moments(topo, lpos, lsup, qglobal, kernel, device, channels)


_BOX_COEFFS = {}                                                              # (kernel, N, breaks) -> {'Phi': [np, np, N+1, N+1], 'Phi1': ...}: device independent
_BOX_TABLES = {}                                                              # (kernel, N, breaks, device) -> BoxTables
BOX_BREAKS = (-1.0, -0.5, -0.25, -0.0625, 0.0, 0.0625, 0.25, 0.5, 1.0)      # graded toward the origin (cusps of lw2 / odd powers of r), 8 panels per axis
BOX_DEGREE = 20
BOX_EXACT_KERNELS = frozenset({"cone"})                                      # profiles with a kink at the support edge (the table converges algebraically, 1e-3 on the gradient): exact polygon path


def boxTables(kernel: str, device, N: int = BOX_DEGREE, breaks=BOX_BREAKS) -> BoxTables:
    """the corner tables (`box.py`) of the registered kernel `kernel`, built once from the exact edge machinery: Phi(a, b) / Phi1(a, b) are the lam / m1_x channels of the quadrant
    [-3, 0]^2 for a particle at (-a, -b) (support 1), with a, b on the Chebyshev nodes of every panel of the graded grid `breaks`."""
    key = (kernel, N, tuple(breaks), str(device))
    if key not in _BOX_TABLES:
        ck = (kernel, N, tuple(breaks))
        if ck not in _BOX_COEFFS:
            nodes = chebyshev_nodes(N)
            npan = len(breaks) - 1
            rep = SurfaceRep.box((-3.0, -3.0), (0.0, 0.0)).to(device)
            xs = np.concatenate([0.5 * (breaks[i] + breaks[i + 1]) + 0.5 * (breaks[i + 1] - breaks[i]) * nodes for i in range(npan)])       # [np (N+1)] all panel nodes of one axis
            A, Bq = np.meshgrid(xs, xs, indexing="ij")
            lpos = torch.as_tensor(np.stack([-A.ravel(), -Bq.ravel()], 1), dtype=F64, device=device)
            n = len(lpos)
            mp = rep.pairs(lpos, torch.ones(n, dtype=F64, device=device), torch.arange(n, device=device), kernel, device)
            q = mp.q.long()
            phi = torch.zeros(n, dtype=F64, device=device).index_add_(0, q, mp.lam).cpu().numpy().reshape(npan, N + 1, npan, N + 1)
            phi1 = torch.zeros(n, dtype=F64, device=device).index_add_(0, q, mp.m1[:, 0]).cpu().numpy().reshape(npan, N + 1, npan, N + 1)
            coeffs = {"Phi": np.zeros((npan, npan, N + 1, N + 1)), "Phi1": np.zeros((npan, npan, N + 1, N + 1))}
            for i in range(npan):
                for j in range(npan):
                    coeffs["Phi"][i, j] = fit_panels(phi[i, :, j, :], N)
                    coeffs["Phi1"][i, j] = fit_panels(phi1[i, :, j, :], N)
            _BOX_COEFFS[ck] = coeffs
        _BOX_TABLES[key] = BoxTables(_BOX_COEFFS[ck], breaks, N, device)
    return _BOX_TABLES[key]


class BoxRep:
    """an axis-aligned rectangle in the body frame, [lo, hi]: the solid (`solid='inside'`, an obstacle / a baffle / a floor slab) or the fluid domain (`solid='outside'`: a tank,
    channel, flume).  No edges, no pairs: every moment channel of every kernel is four corner lookups of two 2D tables (`box.py`, docs/box-domain-primitive.md), one `RepMoments` row per query
    that sees the box; exact up to the table error (kernel-dependent: w2 1e-10 on lam, 1e-7 on the gradient channels; w4 1e-9; lw2 1e-6; w2p5 1e-10 at the default grid), no cancellation, no
    indicator pseudo-pair.  Kernels in `BOX_EXACT_KERNELS` (a kink at the support edge: `cone`) use the exact polygon path of `surface()` instead.
    `surface()` is the polygon of the same body (the reference, the cone-area detector, the SDF fallback)."""
    kind = "box"

    def __init__(self, lo, hi, solid: str = "inside"):
        self.lo = torch.as_tensor(lo, dtype=F64)
        self.hi = torch.as_tensor(hi, dtype=F64)
        if solid not in ("inside", "outside"):
            raise ValueError("solid must be 'inside' or 'outside'")
        self.solid = solid
        self._surface = None

    def to(self, device):
        self.lo, self.hi = self.lo.to(device), self.hi.to(device)
        self._surface = None
        return self

    def bounds(self):
        return self.lo, self.hi

    def surface(self):
        if self._surface is None:
            self._surface = SurfaceRep.box(tuple(float(v) for v in self.lo), tuple(float(v) for v in self.hi), self.solid).to(self.lo.device)
        return self._surface

    def signed(self, lpos):
        """(d, n): signed distance (positive in the fluid) and the unit normal from the wall into the fluid at the closest wall point, body frame."""
        c, hw = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        r = lpos - c
        q = r.abs() - hw                                                          # > 0: outside the box along that axis
        out = q.clamp(min=0).norm(dim=1)
        sd = out + q.max(dim=1).values.clamp(max=0)                               # signed distance to the box, positive outside it
        ax = (q[:, 1] > q[:, 0]).long()                                           # axis of the nearest face plane (the largest q)
        sgn = torch.where(r.gather(1, ax[:, None])[:, 0] >= 0, 1.0, -1.0).to(F64)
        face = torch.zeros_like(lpos).scatter_(1, ax[:, None], sgn[:, None])      # outward normal of the nearest face
        closest = torch.minimum(torch.maximum(lpos, self.lo), self.hi)
        dirv = lpos - closest
        dn = dirv.norm(dim=1, keepdim=True)
        outward = torch.where(dn > 1e-12, dirv / dn.clamp(min=1e-300), face)      # outside the box: from the closest point; inside: the nearest face
        if self.solid == "inside":
            return sd, outward
        return -sd, -face                                                          # fluid inside the box: the wall normal into the fluid is the inward normal of the nearest face

    def moments(self, lpos, lsup, qglobal, kernel, device, channels=None):
        """RepMoments (body frame) of the queries `lpos` (supports `lsup`) against the box, one row per query with a non-zero channel; `channels` as for `SurfaceRep.moments`."""
        if len(lpos) == 0:
            return None
        if kernel in BOX_EXACT_KERNELS:
            return self.surface().pairs(lpos, lsup, qglobal, kernel, device, channels)
        t = boxTables(kernel, device)
        h = lsup
        a0, a1 = (self.lo[0] - lpos[:, 0]) / h, (self.hi[0] - lpos[:, 0]) / h
        b0, b1 = (self.lo[1] - lpos[:, 1]) / h, (self.hi[1] - lpos[:, 1]) / h
        lam, m, g0, g1 = t.channels(a0, a1, b0, b1)
        if self.solid == "outside":                                               # the solid is the whole plane minus the box: the constants of the plane minus the box channels
            big = torch.full_like(a0, 1e3)
            L, M, G0, G1 = t.channels(-big, big, -big, big)
            lam, m, g0, g1 = L - lam, M - m, G0 - g0, G1 - g1
        m1 = h[:, None] * m
        g0 = g0 / h[:, None]
        if channels is not None:
            ch = set(int(k) for k in channels)
            lam = lam if 0 in ch else torch.zeros_like(lam)
            m1 = m1 if ch & {1, 2} else torch.zeros_like(m1)
            g0 = g0 if ch & {3, 4} else torch.zeros_like(g0)
            g1 = g1 if ch & {5, 6, 7, 8} else torch.zeros_like(g1)
        keep = torch.nonzero((lam != 0) | (m1 != 0).any(1) | (g0 != 0).any(1) | (g1 != 0).flatten(1).any(1)).flatten()
        return RepMoments(qglobal[keep], lam[keep], g0[keep], m1[keep], g1[keep])


class VolumeRep:
    """triangle elements in the body frame with P1 nodal fields (`boundaryOps` pair engine): needed when the field is NOT linear in position (deformable /
    measured boundary data); pair counts scale with the number of elements inside the support."""
    kind = "volume"

    def __init__(self, vertices, elements):
        self.mesh = BoundaryMesh(torch.as_tensor(vertices, dtype=F64), torch.as_tensor(elements, dtype=torch.int32))
        self._grid = None
        self._gridSupport = -1.0

    def to(self, device):
        self.mesh = BoundaryMesh(self.mesh.vertices.to(device), self.mesh.elements.to(device))
        return self

    def bounds(self):
        return self.mesh.vertices.amin(0), self.mesh.vertices.amax(0)

    def grid(self, supportMax):
        if self._grid is None or supportMax > self._gridSupport:
            self._grid, self._gridSupport = buildElementGrid(self.mesh, supportMax), supportMax
        return self._grid


class ImplicitRep:
    """analytic primitive in the body frame; per particle hard switch: tier 3 (curvature expansion) / tier 4 (small obstacle series) / tier 2 (a polygon
    SurfaceRep of the shape, area preserving, edges <= h/16).  Constant boundary fields only."""
    kind = "implicit"

    def __init__(self, shape, policy: Optional[TierPolicy] = None):
        self.shape = shape
        self.policy = policy or TierPolicy()
        self._fallback = {}

    def bounds(self):
        s = self.shape
        if isinstance(s, DiskBody):
            c = torch.as_tensor(s.center, dtype=F64)
            return c - s.radius, c + s.radius
        big = 1e8
        return torch.full((2,), -big, dtype=F64), torch.full((2,), big, dtype=F64)

    def fallbackSurface(self, hmin, device):
        s = self.shape
        if not isinstance(s, DiskBody):
            raise NotImplementedError("tier-2 fallback exists for disks only")
        n = max(24, int(np.ceil(2 * np.pi * s.radius / (self.policy.maxPolygonEdgeOverH * hmin))))
        if n not in self._fallback:
            self._fallback[n] = SurfaceRep.regularPolygon(s.center, s.radius, n, solid=s.solid).to(device)
        return self._fallback[n]


class SdfRep:
    """sampled signed distance in the body frame (positive in the fluid), bilinear d with node-blended gradient and Laplacian; surface curvature
    kappa = lap d / (1 - d lap d).  Tier 3 where |kappa h| <= `maxKappaH` and |grad d| ~ 1, else (hard switch per particle) the `fallback` SurfaceRep
    (e.g. an extracted contour); particles outside the sampled box see the extrapolated distance.  Constant boundary fields only."""
    kind = "sdf"

    def __init__(self, values, origin, spacing, fallback: Optional[SurfaceRep] = None, maxKappaH: float = 0.5, gradTol: float = 0.1):
        self.d = torch.as_tensor(values, dtype=F64)
        self.origin = torch.as_tensor(origin, dtype=F64)
        self.spacing = float(spacing)
        self.fallback = fallback
        self.maxKappaH = maxKappaH
        self.gradTol = gradTol
        self._nodes = None

    @staticmethod
    def fromFunction(fn, lo, hi, spacing, **kw):
        """sample `fn(points[M,2]) -> d` on a regular grid covering [lo, hi]."""
        nx = int(np.ceil((hi[0] - lo[0]) / spacing)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / spacing)) + 1
        X, Y = np.meshgrid(lo[0] + spacing * np.arange(nx), lo[1] + spacing * np.arange(ny), indexing="ij")
        pts = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=F64)
        return SdfRep(fn(pts).reshape(nx, ny), lo, spacing, **kw)

    def to(self, device):
        self.d, self.origin = self.d.to(device), self.origin.to(device)
        if self.fallback is not None:
            self.fallback.to(device)
        self._nodes = None
        return self

    def bounds(self):
        ext = torch.as_tensor(self.d.shape, dtype=F64, device=self.d.device) - 1
        return self.origin, self.origin + ext * self.spacing

    def _node_fields(self):
        if self._nodes is None:
            d, s = self.d, self.spacing
            p = torch.nn.functional.pad(d[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            dx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * s)
            dy = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * s)
            lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * d) / (s * s)
            self._nodes = torch.stack([d, dx, dy, lap])
        return self._nodes

    def signed(self, pos):
        """(d, n, kappa, gradNorm) at local points."""
        nodes = self._node_fields()
        nx, ny = self.d.shape
        u = (pos - self.origin) / self.spacing
        uc = torch.stack([u[:, 0].clamp(0, nx - 1), u[:, 1].clamp(0, ny - 1)], 1)
        i0 = torch.stack([uc[:, 0].floor().clamp(0, nx - 2), uc[:, 1].floor().clamp(0, ny - 2)], 1).long()
        t = uc - i0
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                wgt = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                out = out + wgt[None] * nodes[:, i0[:, 0] + dx, i0[:, 1] + dy]
        d, gx, gy, lap = out
        outside = ((u - uc) * self.spacing).norm(dim=1)             # distance beyond the sampled box: the distance keeps growing
        d = d + outside
        gn = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-300)
        kappa = lap / (1.0 - d * lap)
        return d, torch.stack([gx, gy], 1) / gn[:, None], kappa, gn


# ----------------------------------------------------------------------------------------------------------------------- bodies, scene
@dataclass
class BodyField:
    """boundary data of one body (world components): A(x') = a0 + a1 (x' - center); scalar fields: a0 [], a1 [2]; vectors: a0 [2], a1 [2,2]; `rho` the density
    of the boundary (Symmetric mode, Density).
    `perQuery = True`: a0 has a leading dimension N (the field value AT each query position, e.g. a wall pressure extrapolated from the particle itself,
    p_b = p_i + rho g (x' - x_i)) and a1 is the position gradient of that field; surface / implicit / SDF representations only."""
    a0: Optional[torch.Tensor] = None
    a1: Optional[torch.Tensor] = None
    rho: float = 1.0
    perQuery: bool = False

    @staticmethod
    def rigid(body: "Body", rho: float = 1.0):
        """velocity field v(x') = v_c + omega x (x' - c) of a rigidly moving body."""
        w = float(body.angularVelocity)
        dev = body.pose.center.device
        return BodyField(torch.as_tensor(body.linearVelocity, dtype=F64, device=dev),
                         torch.tensor([[0.0, -w], [w, 0.0]], dtype=F64, device=dev), rho)


@dataclass
class Body:
    bodyId: int = 0
    center: tuple = (0.0, 0.0)
    angle: float = 0.0
    linearVelocity: tuple = (0.0, 0.0)
    angularVelocity: float = 0.0
    reps: list = field(default_factory=list)
    linearAcceleration: tuple = (0.0, 0.0)       # prescribed motion: enter the wall pressure condition  dp/dn = rho (g - a_wall) . n
    angularAcceleration: float = 0.0

    def __post_init__(self):
        self.center = torch.as_tensor(self.center, dtype=F64)
        self.linearVelocity = torch.as_tensor(self.linearVelocity, dtype=F64)
        self.linearAcceleration = torch.as_tensor(self.linearAcceleration, dtype=F64)

    def accelerationAt(self, world):
        """acceleration of the material points of the body at world positions: a + alpha J s - omega^2 s (s = x - centre)."""
        s = world - self.center
        w, al = float(self.angularVelocity), float(self.angularAcceleration)
        return self.linearAcceleration.to(world.device) + al * torch.stack([-s[:, 1], s[:, 0]], 1) - w * w * s

    @property
    def pose(self):
        return Pose(self.center, float(self.angle))

    def move(self, dt: float):
        """explicit Euler pose update (as `warpSPH.rigidBody.integrateRigidBody`); nothing else is rebuilt."""
        self.center = self.center + dt * self.linearVelocity
        self.angle = float(self.angle) + dt * float(self.angularVelocity)
        self.linearVelocity = self.linearVelocity + dt * self.linearAcceleration.to(self.linearVelocity.device)
        self.angularVelocity = float(self.angularVelocity) + dt * float(self.angularAcceleration)

    def obb(self):
        dev = self.center.device
        los, his = zip(*[r.bounds() for r in self.reps])
        return torch.stack([a.to(dev) for a in los]).amin(0), torch.stack([a.to(dev) for a in his]).amax(0)

    def velocityAt(self, world):
        w = float(self.angularVelocity)
        r = world - self.center
        return self.linearVelocity + w * torch.stack([-r[:, 1], r[:, 0]], 1)


@dataclass
class BodyAdjacency:
    """one body against the queries: the candidate queries (global rows of the query set, sorted), their local-frame positions and supports, and the topology
    of each surface representation (`SurfaceTopology`, aligned with `body.reps`; None for the other types, whose pair structure is made by `Scene.precompute`)."""
    body: object
    cand: torch.Tensor              # [C] global query index
    lpos: torch.Tensor              # [C,2] body frame
    lsup: torch.Tensor              # [C]
    reps: list
    valid: Optional[torch.Tensor] = None     # [C] bool, fixed-capacity adjacency only (`fixedadj`): rows are all queries, `valid` marks the candidates


@dataclass
class SceneAdjacency:
    """ADJACENCY = who interacts with whom: what `Scene.adjacency` returns.  Integers (pair topology, candidate lists) and kernel-independent geometry only;
    it depends on positions and supports and on nothing else, so one adjacency serves every kernel, channel set and operation at those positions.  The
    per-pair integrals are the PRECOMPUTE (`Scene.precompute` -> `PairMoments`); `sceneOperation` is the evaluation."""
    numQueries: int
    bodies: list                    # [BodyAdjacency] per body
    stats: dict                     # candidates per body
    queryParticles: object = None   # the query state the adjacency was built from (the volume representations read masses / densities from it)

    def restrict(self, index):
        """the adjacency of the queries `index` (long tensor of distinct global rows, ascending) alone: rows renumbered 0 .. len(index) - 1, pair lists
        filtered.  No search: `index` must be a subset of this adjacency's queries at the same positions and supports (the near-wall subset of the
        full query set)."""
        dev = index.device
        newid = torch.full((self.numQueries,), -1, dtype=torch.long, device=dev)
        newid[index] = torch.arange(len(index), device=dev)
        out = []
        for ba in self.bodies:
            rows = torch.nonzero(newid[ba.cand] >= 0).flatten()
            rowmap = torch.full((len(ba.cand),), -1, dtype=torch.long, device=dev)
            rowmap[rows] = torch.arange(len(rows), device=dev)
            out.append(BodyAdjacency(ba.body, newid[ba.cand[rows]], ba.lpos[rows], ba.lsup[rows],
                                     [None if t is None else t.restrict(rowmap, rows) for t in ba.reps]))
        qp = None if self.queryParticles is None else _vsub(self.queryParticles, index, dev)
        return SceneAdjacency(len(index), out, {"candidates": [len(b.cand) for b in out]}, qp)


@dataclass
class PairMoments:
    """PRECOMPUTE = the per-pair integral values of ONE kernel (and channel set) over an adjacency: what `Scene.precompute` returns and `sceneOperation`
    evaluates.  Floats (up to 72 B per pair): explicit, opt-in; the default consumers build one, use it once or a few times and drop it."""
    numQueries: int
    kernel: str
    entries: list                   # per body: dict(body, cand, surface=[RepMoments world], implicit=[RepMoments], volume=[(rep, BoundaryAdjacency, cand)])
    stats: dict
    channels: Optional[frozenset] = None   # None: all nine moment channels; else the SurfaceRep channels that were evaluated (the others are 0): see sceneOperation


@dataclass
class ParticleCells:
    """the fluid particles sorted into a uniform grid (cell = largest support), built once per step and shared by all bodies: a body only gathers the
    particles of the cells under its inflated world bounding box (cell id = cx * ny + cy, so every x-column of a cell rectangle is one contiguous slice)."""
    lo: torch.Tensor
    cell: float
    dims: tuple
    start: torch.Tensor
    order: torch.Tensor

    @staticmethod
    def build(pos, cell):
        lo = pos.amin(0)
        dims = torch.clamp(torch.floor((pos.amax(0) - lo) / cell).long() + 1, min=1)
        nx, ny = int(dims[0]), int(dims[1])
        c = torch.floor((pos - lo) / cell).long()
        cid = c[:, 0] * ny + c[:, 1]
        order = torch.argsort(cid, stable=True)
        counts = torch.bincount(cid, minlength=nx * ny)
        return ParticleCells(lo, float(cell), (nx, ny), torch.cat([torch.zeros(1, dtype=torch.long, device=pos.device), torch.cumsum(counts, 0)]), order)

    def gather(self, wlo, whi):
        """indices of the particles in the cells overlapped by the world box [wlo, whi]."""
        nx, ny = self.dims
        i0 = int(torch.floor((wlo[0] - self.lo[0]) / self.cell).clamp(0, nx - 1)); i1 = int(torch.floor((whi[0] - self.lo[0]) / self.cell).clamp(0, nx - 1))
        j0 = int(torch.floor((wlo[1] - self.lo[1]) / self.cell).clamp(0, ny - 1)); j1 = int(torch.floor((whi[1] - self.lo[1]) / self.cell).clamp(0, ny - 1))
        rows = torch.arange(i0, i1 + 1, device=self.start.device)
        s, e = self.start[rows * ny + j0], self.start[rows * ny + j1 + 1]
        cnt = e - s
        tot = int(cnt.sum())
        off = torch.arange(tot, device=self.start.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        return self.order[torch.repeat_interleave(s, cnt) + off]


class Scene:
    def __init__(self, bodies: List[Body], device="cpu", volumeMode: str = "moments"):
        """`volumeMode`: 'moments' (volume representations as pair sets of exact moments, fields linear in position) or 'nodal' (P1 nodal data through
        `boundaryOps.boundaryOperation`, as for deformable / measured boundary fields)."""
        self.bodies = bodies
        self.device = device
        self.volumeMode = volumeMode
        for b in bodies:
            b.center = b.center.to(device)
            b.linearVelocity = b.linearVelocity.to(device)
            for r in b.reps:
                r.to(device) if hasattr(r, "to") else None

    def inside(self, points, body=None):
        """True where a world point lies inside the solid of any body (surface loops: winding number; SDF / implicit primitives: negative signed distance, positive = fluid; volume: inside a triangle).
        Not an adjacency query: no support radius, any number of points (the free-surface detector samples the solid around a particle).  `body`: index of one body only."""
        pts = points.to(self.device, torch.float64)
        out = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
        for bi, bd in enumerate(self.bodies):
            if body is not None and bi != body:
                continue
            lp = bd.pose.toLocal(pts)
            for rep in bd.reps:
                if isinstance(rep, SurfaceRep):
                    out |= rep.indicator(lp) > 0.5
                elif isinstance(rep, BoxRep):
                    out |= rep.signed(lp)[0] < 0
                elif isinstance(rep, SdfRep):
                    out |= rep.signed(lp)[0] < 0
                elif isinstance(rep, ImplicitRep):
                    out |= rep.shape.signed(lp)[0] < 0
                elif isinstance(rep, VolumeRep):
                    V, E = rep.mesh.vertices, rep.mesh.elements.long()
                    a, b, c = V[E[:, 0]], V[E[:, 1]], V[E[:, 2]]
                    for k in range(0, len(lp), 20000):                      # chunked point x triangle test (barycentric signs)
                        q = lp[k:k + 20000, None, :]
                        d1 = (q[..., 0] - b[None, :, 0]) * (a[None, :, 1] - b[None, :, 1]) - (a[None, :, 0] - b[None, :, 0]) * (q[..., 1] - b[None, :, 1])
                        d2 = (q[..., 0] - c[None, :, 0]) * (b[None, :, 1] - c[None, :, 1]) - (b[None, :, 0] - c[None, :, 0]) * (q[..., 1] - c[None, :, 1])
                        d3 = (q[..., 0] - a[None, :, 0]) * (c[None, :, 1] - a[None, :, 1]) - (c[None, :, 0] - a[None, :, 0]) * (q[..., 1] - a[None, :, 1])
                        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
                        out[k:k + 20000] |= (~(neg & pos)).any(1)
        return out

    def signed_distance(self, points, body=None):
        """(d [M], n [M,2], hit [M]) of world points to the solid of one body (or the nearest of all): d > 0 in the fluid, n the unit normal pointing from the wall into the fluid at the closest wall point.
        Surface loops: nearest edge (brute force over the edges, fine for a few dozen); SDF representations: the sampled distance and its gradient; implicit primitives: their own `signed`.  `hit` is False where
        no representation could answer (volume representations)."""
        pts = points.to(self.device, torch.float64)
        M = len(pts)
        best = torch.full((M,), float("inf"), dtype=torch.float64, device=pts.device)
        normal = torch.zeros((M, 2), dtype=torch.float64, device=pts.device)
        hit = torch.zeros(M, dtype=torch.bool, device=pts.device)
        for bi, bd in enumerate(self.bodies):
            if body is not None and bi != body:
                continue
            lp = bd.pose.toLocal(pts)
            for rep in bd.reps:
                if isinstance(rep, SurfaceRep):
                    V = rep.vertices
                    a, b = V[rep.edges[:, 0].long()], V[rep.edges[:, 1].long()]                      # [E,2]
                    e = b - a
                    t = (((lp[:, None, :] - a[None]) * e[None]).sum(2) / (e * e).sum(1)[None]).clamp(0.0, 1.0)
                    c = a[None] + t[..., None] * e[None]                                              # closest points [M,E,2]
                    dist = (lp[:, None, :] - c).norm(dim=2)
                    k = dist.argmin(1)
                    dmin = dist.gather(1, k[:, None])[:, 0]
                    cp = c[torch.arange(M, device=pts.device), k]
                    ins = rep.indicator(lp) > 0.5
                    d = torch.where(ins, -dmin, dmin)
                    # the wall normal into the fluid: the left normal of the edge points INTO the solid (solid on the left), so the fluid normal is the right normal
                    ek = e[k] / e[k].norm(dim=1, keepdim=True)
                    n_loc = torch.stack([ek[:, 1], -ek[:, 0]], 1)
                    # near a vertex the normal is the direction from the closest point to the particle (smooth across corners)
                    dirv = lp - cp
                    dn = dirv.norm(dim=1, keepdim=True)
                    n_loc = torch.where((dn > 1e-12) & (~ins[:, None]), dirv / dn.clamp(min=1e-300), n_loc)
                    n = bd.pose.vecToWorld(n_loc)
                elif isinstance(rep, BoxRep):
                    d, n_loc = rep.signed(lp)
                    n = bd.pose.vecToWorld(n_loc)
                elif isinstance(rep, SdfRep):
                    d, n_loc, _, _ = rep.signed(lp)
                    n = bd.pose.vecToWorld(n_loc)
                elif isinstance(rep, ImplicitRep):
                    d, n_loc, _ = rep.shape.signed(lp)
                    n = bd.pose.vecToWorld(n_loc)
                else:
                    continue
                better = d < best
                best = torch.where(better, d, best)
                normal = torch.where(better[:, None], n, normal)
                hit |= True
        return best, normal, hit

    def candidates(self, body: Body, pos, sup, allowed, cells: Optional[ParticleCells] = None):
        """broadphase: support sphere vs the body OBB in the body frame (exact); with `cells` only the particles under the inflated world box of the OBB are tested."""
        lo, hi = body.obb()
        pose = body.pose
        if cells is not None:
            corners = torch.stack([lo, torch.stack([hi[0], lo[1]]), hi, torch.stack([lo[0], hi[1]])])
            wc = pose.toWorld(corners)
            pad = float(sup.max())
            pre = cells.gather(wc.amin(0) - pad, wc.amax(0) + pad)
            sub = pos[pre]
            lpos = pose.toLocal(sub)
            d = (lo - lpos).clamp(min=0) + (lpos - hi).clamp(min=0)
            hit = (d.norm(dim=1) < sup[pre]) & allowed[pre]
            idx = torch.sort(pre[hit]).values
            return idx, pose.toLocal(pos[idx])
        lpos = pose.toLocal(pos)
        d = (lo - lpos).clamp(min=0) + (lpos - hi).clamp(min=0)
        hit = (d.norm(dim=1) < sup) & allowed
        idx = torch.nonzero(hit).flatten()
        return idx, lpos[idx]

    def adjacency(self, queryParticles, operationProperties: OperationProperties) -> SceneAdjacency:
        """broadphase + the pair topology of every surface representation (`SceneAdjacency`).  Only `operationProperties.operationMode` is read (which query
        kinds are served); the kernel plays no role: precompute one `PairMoments` per kernel / channel set over the same adjacency."""
        dev = self.device
        pos, sup = queryParticles.positions.to(dev, F64), queryParticles.supports.to(dev, F64)
        if pos.shape[1] != 2:
            raise NotImplementedError("2D only")
        allowed = queryAllowed(queryParticles, operationProperties.operationMode, dev)
        bodies, stats = [], {"candidates": []}
        cells = ParticleCells.build(pos, float(sup.max())) if len(self.bodies) > 1 else None
        for body in self.bodies:
            cand, lpos = self.candidates(body, pos, sup, allowed, cells)
            lsup = sup[cand]
            stats["candidates"].append(int(len(cand)))
            reps = [(rep.topology(lpos, lsup) if isinstance(rep, SurfaceRep) and len(cand) else None) for rep in body.reps]
            bodies.append(BodyAdjacency(body, cand, lpos, lsup, reps))
        return SceneAdjacency(len(pos), bodies, stats, queryParticles)

    def precompute(self, adjacency: SceneAdjacency, operationProperties: OperationProperties, channels=None) -> PairMoments:
        """the per-pair integrals of `operationProperties.kernel` over `adjacency` (`PairMoments`); `channels`: evaluate only these SurfaceRep moment channels."""
        name = kernelName(operationProperties.kernel)
        entries, stats = [], {"candidates": list(adjacency.stats["candidates"]), "pairs": 0, "tier": {}}
        for ba in adjacency.bodies:
            ent = {"body": ba.body, "cand": ba.cand, "surface": [], "implicit": [], "volume": []}
            if len(ba.cand):
                for rep, topo in zip(ba.body.reps, ba.reps):
                    self._repMoments(rep, topo, ba, ent, name, adjacency.queryParticles, operationProperties, channels)
            for key in ("surface", "implicit"):
                stats["pairs"] += sum(len(s.q) for s in ent[key])
            entries.append(ent)
        return PairMoments(adjacency.numQueries, name, entries, stats, None if channels is None else frozenset(int(c) for c in channels))

    def pairMoments(self, queryParticles, operationProperties: OperationProperties, channels=None) -> PairMoments:
        """adjacency + precompute in one call, for a consumer that needs the moments of one kernel at one position set only."""
        return self.precompute(self.adjacency(queryParticles, operationProperties), operationProperties, channels)

    def moments(self, source, queryParticles, operationProperties: OperationProperties, channels=None) -> PairMoments:
        """`source` as PairMoments: None builds adjacency + precompute, a `SceneAdjacency` is precomputed (`channels`), a `PairMoments` is returned as it is."""
        if source is None:
            return self.pairMoments(queryParticles, operationProperties, channels)
        if isinstance(source, SceneAdjacency):
            return self.precompute(source, operationProperties, channels)
        return source

    def _repMoments(self, rep, topo, ba, ent, name, queryParticles, operationProperties, channels=None):
        dev = self.device
        body, cand, lpos, lsup = ba.body, ba.cand, ba.lpos, ba.lsup
        pose = body.pose
        if isinstance(rep, SurfaceRep):
            ent["surface"].append(rep.moments(topo, lpos, lsup, cand, name, dev, channels).toWorld(pose))
        elif isinstance(rep, BoxRep):
            mom = rep.moments(lpos, lsup, cand, name, dev, channels)
            if mom is not None and len(mom.q):
                ent["implicit"].append(mom.toWorld(pose))
        elif isinstance(rep, VolumeRep):
            sub = _subState(queryParticles, cand, lpos, lsup, dev)
            adj = buildBoundaryAdjacency(sub, operationProperties, rep.mesh, grid=rep.grid(float(lsup.max())), supportMax=float(lsup.max()))
            adj.gradWeights = adj.gradWeights @ pose.R.T                                 # gradient vectors back to the world frame
            if self.volumeMode == "nodal":
                ent["volume"].append((rep, adj, cand))
            else:                                                                        # fields linear in position: moments of the P1 weights (y_k are P1 functions)
                qw = queryParticles.positions.to(dev, F64)[cand][adj.pairQuery.long()]
                Xk = pose.toWorld(rep.mesh.vertices)[rep.mesh.elements[adj.pairElement.long()].long()]            # [P,3,2]
                y = Xk - qw[:, None, :]
                w, G = adj.weights, adj.gradWeights
                ent["surface"].append(RepMoments(cand[adj.pairQuery.long()], w.sum(1), G.sum(1), torch.einsum("pk,pkd->pd", w, y),
                                                 torch.einsum("pkd,pkj->pdj", y, G)))
        elif isinstance(rep, (ImplicitRep, SdfRep)):
            if isinstance(rep, ImplicitRep):
                lam, g, tier, _ = evaluateBody(rep.shape, lpos, lsup, name, dev, rep.policy, {"t3": tier3Table(name, str(dev)), "t4": tier4Table(name, str(dev))})
                low = tier == 2
                fb = rep.fallbackSurface(float(lsup[low].min()), dev) if bool(low.any()) else None
            else:
                lam, g, low = _sdfTier3(rep, lpos, lsup, name, dev)
                fb = rep.fallback
                if bool(low.any()) and fb is None:
                    raise ValueError("SdfRep: the SDF is not smooth at the scale of h for some particles and no fallback SurfaceRep was given")
            sel = (~low) & (lam != 0)
            idx = torch.nonzero(sel).flatten()
            pairs = None
            if len(idx):
                mom = _planar_moments(rep, lpos[idx], lsup[idx], name, dev)
                pairs = (RepMoments(cand[idx], lam[idx], g[idx], *mom) if mom is not None else RepMoments(cand[idx], lam[idx], g[idx])).toWorld(pose)
            if pairs is not None:
                ent["implicit"].append(pairs)
            if bool(low.any()):
                li = torch.nonzero(low).flatten()
                ent["surface"].append(fb.pairs(lpos[li], lsup[li], cand[li], name, dev).toWorld(pose))


def _planar_moments(rep, lpos, lsup, name, dev):
    """(m1 [P,2], g1 [P,2,2]) of the half-plane model of the surface at the particle (exact for planes; O(kappa h) error for curved SDF surfaces, where the
    lambda itself keeps its tier-3 curvature terms); None for primitives without first moments (disks)."""
    if isinstance(rep, ImplicitRep):
        if not isinstance(rep.shape, HalfPlaneBody):
            return None
        d, n, _ = rep.shape.signed(lpos)
    else:
        d, n, _, _ = rep.signed(lpos)
    t3 = tier3Table(name, str(dev))
    q = d / lsup
    lam, dlam, m1n, g1nn, g1tt = t3.planar_moments(q)
    m1 = -(m1n * lsup)[:, None] * n
    tt = torch.stack([-n[:, 1], n[:, 0]], 1)
    g1 = g1nn[:, None, None] * n[:, :, None] * n[:, None, :] + g1tt[:, None, None] * tt[:, :, None] * tt[:, None, :]
    return m1, g1


def _sdfTier3(rep: SdfRep, lpos, lsup, name, dev):
    d, n, kappa, gn = rep.signed(lpos)
    t3 = tier3Table(name, str(dev))
    q = d / lsup
    lam, dl = t3.lam(q, kappa * lsup)
    valid = ((kappa * lsup).abs() <= rep.maxKappaH) & ((gn - 1).abs() <= rep.gradTol)
    # smoothness over the whole support ball: probes at 0.5 h and h in 8 directions must show a unit gradient and a small curvature too
    # (rules out ridges / medial axes and convex corners, where the local curvature expansion has nothing to say)
    ang = torch.arange(8, dtype=F64, device=dev) * (np.pi / 4)
    dirs = torch.stack([ang.cos(), ang.sin()], 1)
    for rho in (0.5, 1.0):
        probe = lpos[:, None, :] + rho * lsup[:, None, None] * dirs[None]
        pd, _, pk, pg = rep.signed(probe.reshape(-1, 2))
        ok = ((pk * lsup.repeat_interleave(8)).abs() <= rep.maxKappaH) & ((pg - 1).abs() <= rep.gradTol)
        valid &= ok.reshape(-1, 8).all(1)
    far = q >= 1.0
    low = (~valid) & (~far)
    lam = torch.where(valid, lam, torch.zeros_like(lam))
    g = (dl / lsup)[:, None] * n
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    return lam, g, low


def _subState(ps, cand, lpos, lsup, dev):
    from warpSPHCore import ParticleState
    take = lambda a: None if a is None else a.to(dev)[cand]
    return ParticleState(positions=lpos, supports=lsup, masses=take(ps.masses), kinds=take(getattr(ps, "kinds", None)),
                         densities=take(getattr(ps, "densities", None)))


# ----------------------------------------------------------------------------------------------------------------------- operations
def _canon(field_a0, field_a1):
    """-> (a0 [C], a1 [C,2], scalar flag)."""
    a0 = torch.as_tensor(field_a0, dtype=F64)
    scalar = a0.ndim == 0
    a0 = a0.reshape(1) if scalar else a0
    if field_a1 is None:
        return a0, None, scalar
    a1 = torch.as_tensor(field_a1, dtype=F64, device=a0.device)
    return a0, (a1.reshape(1, 2) if scalar else a1), scalar


def _apply(op, mode, pairs: RepMoments, body: Body, fld: BodyField, ps, queryValues, rhoI, dev):
    """contribution [P, ...] of one pair set to its queries (world frame) plus (force [2], torque or None) of the reaction on the body (Gradient of a scalar)."""
    q = pairs.q
    c = body.center
    if op == WarpOperation.Density:
        return fld.rho * pairs.lam, None, None
    if op == WarpOperation.Covariance:                                    # int y (x) grad_x W: the renormalisation (covariance) matrix of the wall
        if pairs.g1 is None:
            raise NotImplementedError("Covariance needs first moments (surface / volume / half-plane representations)")
        return fld.rho * pairs.g1, None, None
    if fld.a0 is None:
        raise ValueError("BodyField with a0 is required for this operation")
    if fld.perQuery:
        a0 = torch.as_tensor(fld.a0, dtype=F64, device=dev)
        scalar = a0.ndim == 1
        a0 = a0.reshape(-1, 1) if scalar else a0
        a1 = None if fld.a1 is None else torch.as_tensor(fld.a1, dtype=F64, device=dev)          # [N,2] (scalar) or [N,C,2]: gradient at each query
        if a1 is not None:                                              # a constant gradient (leading axis added) or one gradient per query
            a1 = a1.reshape(1, 1, 2) if (scalar and a1.ndim == 1) else (a1.reshape(-1, 1, 2) if scalar else (a1[None] if a1.ndim == 2 else a1))
    else:
        a0, a1, scalar = _canon(fld.a0, fld.a1)
        a0, a1 = a0.to(dev), (None if a1 is None else a1.to(dev))
    if a1 is not None and pairs.g1 is None:
        raise NotImplementedError("a position-dependent boundary field needs a surface or volume representation (implicit / SDF models are constant-only)")
    x = ps.positions.to(dev, F64)[q]
    if fld.perQuery:
        Ai = a0[q]                                                       # value at the query position  [P,C]
        a1p = None if a1 is None else (a1.expand(len(q), -1, -1) if a1.shape[0] == 1 else a1[q])       # [P,C,2]
    else:
        Ai = a0[None].expand(len(q), -1)
        if a1 is not None:
            Ai = Ai + (x - c) @ a1.T                                    # A at the query position  [P,C]
        a1p = None if a1 is None else a1[None].expand(len(q), -1, -1)
    if op == WarpOperation.Interpolate:
        out = Ai * pairs.lam[:, None]
        if a1p is not None:
            out = out + torch.einsum("pcd,pd->pc", a1p, pairs.m1)
        return (out[:, 0] if scalar else out), None, None
    # gradient family: effective field  s * A + cvec
    f = None
    if mode != GradientScheme.Naive:
        if queryValues is None:
            raise ValueError("queryValues required for this gradient mode")
        f = queryValues.to(dev, F64)[q]
        f = f.reshape(len(q), -1)
    if mode == GradientScheme.Symmetric:
        s = rhoI[q] / fld.rho
        cv = (fld.rho / rhoI[q])[:, None] * f
    elif mode == GradientScheme.Difference:
        s, cv = torch.ones(len(q), dtype=F64, device=dev), -f
    elif mode == GradientScheme.Summation:
        s, cv = torch.ones(len(q), dtype=F64, device=dev), f
    else:
        s, cv = torch.ones(len(q), dtype=F64, device=dev), torch.zeros_like(Ai)
    A0 = s[:, None] * Ai + cv                                           # [P,C]
    out = A0[:, :, None] * pairs.g0[:, None, :]                         # [P,C,2]
    if a1p is not None:
        out = out + s[:, None, None] * torch.einsum("pcd,pdj->pcj", a1p, pairs.g1)
    force = torque = None
    if scalar:
        m = ps.masses.to(dev, F64)[q]
        fi = -(m[:, None] * out[:, 0, :])
        force = fi.sum(0)
        if a1 is None and pairs.g1 is not None:
            r = x - c
            lever = (pairs.g1[:, 0, 1] - pairs.g1[:, 1, 0]) + (r[:, 0] * pairs.g0[:, 1] - r[:, 1] * pairs.g0[:, 0])
            torque = -(m * A0[:, 0] * lever).sum()
    if op == WarpOperation.Gradient:
        return (out[:, 0, :] if scalar else out), force, torque
    if op == WarpOperation.Divergence:
        return out[:, 0, 0] + out[:, 1, 1], force, torque
    if op == WarpOperation.Curl:
        return out[:, 1, 0] - out[:, 0, 1], force, torque
    raise NotImplementedError(f"{op}: not implemented for scene boundaries")


@dataclass
class SceneReaction:
    force: torch.Tensor                     # [B,2] force of the fluid on each body (Gradient of a scalar; = - sum m_i contribution_i)
    torque: torch.Tensor                    # [B] about the body centre, NaN where unavailable
    torqueExact: list                       # per body: True if every contribution had first moments and the field was constant


def sceneOperation(queryParticles, operationProperties: OperationProperties, scene: Scene, moments=None,
                   queryValues: Optional[torch.Tensor] = None, bodyFields: Optional[List[BodyField]] = None, returnReaction: bool = False,
                   perBody: bool = False):
    """Boundary contribution of the requested operation (Density, Interpolate, Gradient, Divergence, Curl) for every query particle, summed over all
    bodies and representations (`perBody=True`: a tensor [B, N, ...] with the contribution of each body separately).  `bodyFields[b]` is the `BodyField` of
    `scene.bodies[b]` (default: a unit density wall without field).  `moments`: the `PairMoments` to evaluate (several operations at one position set share
    one); a `SceneAdjacency` is precomputed for this call only (use it for a single operation); None builds both."""
    dev = scene.device
    adj = scene.moments(moments, queryParticles, operationProperties)
    if adj.kernel != kernelName(operationProperties.kernel):
        raise ValueError("sceneOperation: the pair moments are those of kernel %r, the operation asks for %r (a PairMoments holds the integrals of its own kernel)" % (adj.kernel, kernelName(operationProperties.kernel)))
    op, mode = operationProperties.operation, operationProperties.gradientMode
    if adj.channels is not None:                    # pruned moments (gradient channels only) serve the Naive gradient of a constant scalar field and nothing else
        if (op != WarpOperation.Gradient or mode != GradientScheme.Naive or returnReaction or not {3, 4} <= adj.channels
                or any(f.perQuery or f.a1 is not None for f in (bodyFields or []))):
            raise ValueError("sceneOperation: the pair moments were built with the channels %s only; they serve the Naive Gradient of a constant scalar field (no reaction)" % sorted(adj.channels))
    N = adj.numQueries
    nb = len(scene.bodies)
    bodyFields = bodyFields or [BodyField() for _ in range(nb)]
    rhoI = queryParticles.densities.to(dev, F64) if getattr(queryParticles, "densities", None) is not None else torch.ones(N, dtype=F64, device=dev)
    qv = None if queryValues is None else queryValues.to(dev, F64)
    out = None
    force = torch.zeros((nb, 2), dtype=F64, device=dev)
    torque = torch.zeros(nb, dtype=F64, device=dev)
    exact = [True] * nb

    outs = [None] * nb

    def accumulate(idx, contrib, bi=0):
        nonlocal out
        if perBody:
            if outs[bi] is None:
                outs[bi] = torch.zeros((N, *contrib.shape[1:]), dtype=F64, device=dev)
            outs[bi].index_add_(0, idx, contrib)
            return
        if out is None:
            out = torch.zeros((N, *contrib.shape[1:]), dtype=F64, device=dev)
        out.index_add_(0, idx, contrib)

    for bi, ent in enumerate(adj.entries):
        body, fld = ent["body"], bodyFields[bi]
        for s in [s for s in ent["surface"] + ent["implicit"] if s is not None and len(s.q)]:
            contrib, f, t = _apply(op, mode, s, body, fld, queryParticles, qv, rhoI, dev)
            accumulate(s.q, contrib, bi)
            if f is not None:
                force[bi] += f
                if t is None:
                    exact[bi] = False
                else:
                    torque[bi] += t
        for rep, vadj, cand in ent["volume"]:
            if fld.perQuery:
                raise NotImplementedError("per-query boundary fields are not available for volume representations")
            a0, a1, scalar = _canon(fld.a0, fld.a1) if fld.a0 is not None else (None, None, None)
            X = body.pose.toWorld(rep.mesh.vertices)
            nodal = None
            if a0 is not None:
                a0 = a0.to(dev)
                nodal = a0[None].expand(len(X), -1)
                if a1 is not None:
                    nodal = nodal + (X - body.center) @ a1.to(dev).T
                nodal = nodal[:, 0] if scalar else nodal
            res = boundaryOperation(_vsub(queryParticles, cand, dev), operationProperties, rep.mesh, _vq(qv, cand, nodal, dev), nodal, float(fld.rho),
                                    adjacency=vadj, returnReaction=bool(returnReaction and op == WarpOperation.Gradient and scalar))
            if isinstance(res, tuple):
                res, rv = res
                f = rv.sum(0)
                force[bi] += f
                torque[bi] += ((X - body.center)[:, 0] * rv[:, 1] - (X - body.center)[:, 1] * rv[:, 0]).sum()
                exact[bi] = False                      # nodal reactions: approximate torque (needs p = 2 weights for the exact one)
            accumulate(cand, res, bi)
    if perBody:
        ref = next((o for o in outs if o is not None), None)
        if ref is None:
            ref = _empty(op, bodyFields, qv, N, dev)
        return torch.stack([o if o is not None else torch.zeros_like(ref) for o in outs])
    if out is None:
        out = _empty(op, bodyFields, qv, N, dev)
    if returnReaction:
        tq = torque.clone()
        for bi in range(nb):
            if not exact[bi] and not adj.entries[bi]["volume"]:
                tq[bi] = float("nan")
        return out, SceneReaction(force, tq, exact)
    return out


def _vq(qv, cand, nodal, dev):
    """query values of the candidates for `boundaryOperation` (which wants an array even for the Naive mode that ignores it)."""
    if qv is not None:
        return qv[cand]
    return None if nodal is None else torch.zeros((len(cand),) + tuple(nodal.shape[1:]), dtype=F64, device=dev)


def _vsub(ps, cand, dev):
    """the query state of the candidates (world frame) that `boundaryOperation` reads masses / densities from when an adjacency is passed."""
    from warpSPHCore import ParticleState
    take = lambda a: None if a is None else a.to(dev)[cand]
    return ParticleState(positions=ps.positions.to(dev)[cand], supports=ps.supports.to(dev)[cand], masses=take(ps.masses),
                         kinds=take(getattr(ps, "kinds", None)), densities=take(getattr(ps, "densities", None)))


def _empty(op, bodyFields, qv, N, dev):
    scalar = True
    for f in bodyFields:
        if f.a0 is not None:
            scalar = torch.as_tensor(f.a0).ndim == (1 if f.perQuery else 0)
    if op == WarpOperation.Covariance:
        return torch.zeros((N, 2, 2), dtype=F64, device=dev)
    if op in (WarpOperation.Density, WarpOperation.Divergence, WarpOperation.Curl):
        return torch.zeros(N, dtype=F64, device=dev)
    if op == WarpOperation.Gradient:
        return torch.zeros((N, 2) if scalar else (N, 2, 2), dtype=F64, device=dev)
    return torch.zeros(N if scalar else (N, 2), dtype=F64, device=dev) if scalar else torch.zeros((N, 2), dtype=F64, device=dev)
