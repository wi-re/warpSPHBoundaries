"""Boundary operations with the call shape of `warpSPHCore.warpOperation` (2D triangle/polyline boundary elements, tiers 1/2).

    adjacency = buildBoundaryAdjacency(queryParticles, operationProperties, mesh)           # like the Verlet / hash-grid adjacency
    out       = boundaryOperation(queryParticles, operationProperties, mesh,
                                  queryValues, referenceValues, referenceDensities, adjacency=adjacency)

* `queryParticles` : `warpSPHCore.ParticleState` (positions [N,2], supports [N], masses, kinds, densities) -- the fluid side, unchanged.
* `operationProperties` : the SAME `warpSPHCore.OperationProperties` (kernel, operation, gradientMode, operationMode, ...).  Supported:
      operation : Interpolate, Gradient, Divergence, Curl (2D), Density     (Laplacian, Covariance: see docs/boundary-operations.md)
      gradientMode : Naive, Symmetric, Difference, Summation  (nodal-value substitution; the weights are unchanged)
      operationMode : honoured (boundary elements act as `Boundary` kind sources; query kinds filtered like the particle operators)
      supportMode : Gather semantics (the element has no support radius; h = supports[i])
* `mesh` : `BoundaryMesh` (vertices [V,2], elements [E,3], optional bodyIds [E]); fields on the boundary are P1 NODAL values `referenceValues[V,...]`
  (continuous), or per-element `[E,...]` (P0).  `referenceDensities` ([V], [E] or a float) are the boundary densities (Symmetric / Density).
* The result is EXACT for the polygonal boundary (edge reductions, `docs/derivations/`); no boundary particles, no sampling.

Hard tier switching: this module is tiers 1/2 (exact elements).  Implicit bodies (tiers 3/4) enter through `ImplicitBody` below with the same result
conventions; per (particle, body) the cheapest model within the requested tolerance is chosen (`tierPolicy`), no blending.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, WarpOperation

from . import warpbc

_KERNEL_NAMES = {KernelFunctions.CubicSpline: "cubic", KernelFunctions.QuarticSpline: "quartic", KernelFunctions.QuinticSpline: "quintic",
                 KernelFunctions.B7: "b7", KernelFunctions.B8: "b8", KernelFunctions.Poly6: "poly6",
                 KernelFunctions.Wendland2: "w2", KernelFunctions.Wendland4: "w4", KernelFunctions.Wendland6: "w6"}


def kernelName(kernel: KernelFunctions) -> str:
    try:
        return _KERNEL_NAMES[kernel]
    except KeyError:
        raise NotImplementedError(f"{kernel}: not piecewise polynomial in r (HOCT4, Gaussian) -- no exact boundary integral; use quadrature")


@dataclass
class BoundaryMesh:
    vertices: torch.Tensor                  # [V, 2]
    elements: torch.Tensor                  # [E, 3] int
    bodyIds: Optional[torch.Tensor] = None  # [E] int (rigid body / object id), default all 0

    def __post_init__(self):
        self.elements = self.elements.to(torch.int32)
        if self.bodyIds is None:
            self.bodyIds = torch.zeros(self.elements.shape[0], dtype=torch.int32, device=self.elements.device)

    @property
    def device(self):
        return self.vertices.device


# ----------------------------------------------------------------------------------------------- element grid / adjacency
@dataclass
class ElementGrid:
    cellMin: torch.Tensor
    cellSize: float
    dims: Tuple[int, int]
    cellStart: torch.Tensor     # [ncell + 1]
    cellElements: torch.Tensor  # [entries] element ids sorted by cell


def buildElementGrid(mesh: BoundaryMesh, supportMax: float) -> ElementGrid:
    """uniform grid, cell = supportMax; every element is inserted into all cells overlapped by its AABB inflated by supportMax,
    so a query particle only has to look at its OWN cell."""
    dev = mesh.device
    tri = mesh.vertices[mesh.elements.long()]                         # [E,3,2]
    lo = tri.amin(1) - supportMax
    hi = tri.amax(1) + supportMax
    gmin = lo.amin(0)
    gmax = hi.amax(0)
    cs = float(supportMax)
    dims = torch.clamp(torch.ceil((gmax - gmin) / cs).long(), min=1)
    nx, ny = int(dims[0]), int(dims[1])
    i0 = torch.floor((lo - gmin) / cs).long().clamp(min=0)
    i1 = torch.floor((hi - gmin) / cs).long()
    i1 = torch.minimum(i1, dims - 1)
    cnt = (i1 - i0 + 1).clamp(min=0)
    ncell_per = cnt[:, 0] * cnt[:, 1]
    eid = torch.repeat_interleave(torch.arange(len(tri), device=dev), ncell_per)
    start = torch.cumsum(ncell_per, 0) - ncell_per
    local = torch.arange(int(ncell_per.sum()), device=dev) - torch.repeat_interleave(start, ncell_per)
    cx = i0[eid, 0] + local % cnt[eid, 0]
    cy = i0[eid, 1] + local // cnt[eid, 0]
    cell = cx * ny + cy
    order = torch.argsort(cell, stable=True)
    cell_sorted, eid_sorted = cell[order], eid[order]
    counts = torch.bincount(cell_sorted, minlength=nx * ny)
    cellStart = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)])
    return ElementGrid(gmin, cs, (nx, ny), cellStart, eid_sorted)


def _point_triangle_distance(p, tri):
    """exact distance from points p [P,2] to triangles tri [P,3,2] (0 inside)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    def seg(p, u, v):
        d = v - u
        t = ((p - u) * d).sum(1) / (d * d).sum(1).clamp(min=1e-300)
        t = t.clamp(0, 1)
        return (p - (u + t[:, None] * d)).norm(dim=1)
    dist = torch.minimum(torch.minimum(seg(p, a, b), seg(p, b, c)), seg(p, c, a))
    cr = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    s1, s2, s3 = cr(b - a, p - a), cr(c - b, p - b), cr(a - c, p - c)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    return torch.where(inside, torch.zeros_like(dist), dist)


@dataclass
class BoundaryAdjacency:
    """(query, element) pairs within one support, plus their P1 weights: the analogue of an adjacency list that already carries the geometry."""
    pairQuery: torch.Tensor                 # [P] int32
    pairElement: torch.Tensor               # [P] int32
    weights: torch.Tensor                   # [P,3]   int N_k W dA
    gradWeights: torch.Tensor               # [P,3,2] int N_k grad_x W dA
    numQueries: int
    kernel: str


def buildBoundaryAdjacency(queryParticles, operationProperties: OperationProperties, mesh: BoundaryMesh,
                           grid: Optional[ElementGrid] = None, supportMax: Optional[float] = None) -> BoundaryAdjacency:
    """candidate search through the element grid, exact distance filter, then the P1 pair weights on the device."""
    pos, sup = queryParticles.positions, queryParticles.supports
    if pos.shape[1] != 2:
        raise NotImplementedError("2D only (3D boundary integrals: not implemented)")
    name = kernelName(operationProperties.kernel)
    dev = mesh.device
    pos = pos.to(dev)
    sup = sup.to(dev)
    supportMax = float(sup.max()) if supportMax is None else supportMax
    grid = grid or buildElementGrid(mesh, supportMax)
    nx, ny = grid.dims
    cx = torch.floor((pos[:, 0] - grid.cellMin[0]) / grid.cellSize).long()
    cy = torch.floor((pos[:, 1] - grid.cellMin[1]) / grid.cellSize).long()
    ok = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    cell = torch.where(ok, cx * ny + cy, torch.zeros_like(cx))
    cnt = torch.where(ok, grid.cellStart[cell + 1] - grid.cellStart[cell], torch.zeros_like(cell))
    q = torch.repeat_interleave(torch.arange(len(pos), device=dev), cnt)
    off = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    e = grid.cellElements[grid.cellStart[cell[q]] + off]
    # kind filter (query side) ---------------------------------------------------------------------------------
    keep = _queryAllowed(queryParticles, operationProperties.operationMode, dev)[q]
    q, e = q[keep], e[keep]
    tri = mesh.vertices[mesh.elements[e].long()]
    d = _point_triangle_distance(pos[q], tri)
    keep = d < sup[q]
    q, e = q[keep].to(torch.int32), e[keep].to(torch.int32)
    w, G = warpbc.pair_weights(q, e, pos, sup, mesh.vertices, mesh.elements, name, device=str(dev), as_torch=True)
    return BoundaryAdjacency(q, e, w, G, len(pos), name)


def _queryAllowed(queryParticles, mode: OperationDirection, dev):
    """boundary elements are `Boundary`-kind SOURCES: they act iff the source side of the direction is Boundary or All; the query kind must match the target side."""
    kinds = getattr(queryParticles, "kinds", None)
    n = queryParticles.positions.shape[0]
    allq = torch.ones(n, dtype=torch.bool, device=dev)
    nm = mode.name
    if nm in ("AllToAll", "TrueAllToToAll"):
        return allq if kinds is None else (kinds.to(dev) != 2) if nm == "TrueAllToToAll" else allq
    src, tgt = nm.split("To")
    if src not in ("Boundary", "All"):
        return torch.zeros(n, dtype=torch.bool, device=dev)
    if kinds is None:
        return allq
    k = kinds.to(dev)
    return {"Fluid": k == 0, "Boundary": k == 1, "Ghost": k == 2, "All": k != 2}[tgt]


# ----------------------------------------------------------------------------------------------- the operations
def _nodal(ref, mesh, adj, name):
    """reference values per pair and vertex: [P,3,...] from nodal [V,...] or per-element [E,...] arrays (or a scalar)."""
    if ref is None:
        raise ValueError(f"{name} must be provided")
    if not isinstance(ref, torch.Tensor):
        ref = torch.as_tensor(ref, dtype=torch.float64, device=mesh.device)
    ref = ref.to(mesh.device, torch.float64)
    if ref.ndim == 0:
        return ref.expand(len(adj.pairQuery), 3)
    if ref.shape[0] == mesh.vertices.shape[0] and ref.shape[0] != mesh.elements.shape[0]:
        return ref[mesh.elements[adj.pairElement.long()].long()]
    if ref.shape[0] == mesh.elements.shape[0] and ref.shape[0] != mesh.vertices.shape[0]:
        v = ref[adj.pairElement.long()]
        return v.unsqueeze(1).expand(v.shape[0], 3, *v.shape[1:])
    raise ValueError("ambiguous reference array: V == E; pass nodal values as [V,...] via `referenceNodal=` or per-element via `referenceElement=`")


def boundaryOperation(queryParticles, operationProperties: OperationProperties, mesh: BoundaryMesh,
                      queryValues: Optional[torch.Tensor] = None, referenceValues=None, referenceDensities=1.0,
                      adjacency: Optional[BoundaryAdjacency] = None, returnReaction: bool = False,
                      referenceNodal=None, referenceElement=None):
    """Boundary contribution of the requested operation for every query particle, like `warpOperation` but with exact element integrals.
        Interpolate : sum_e int A W
        Density     : sum_e int rho_b W                       (rho_b = referenceDensities, default 1 -> the kernel integral)
        Gradient    : sum_e int a(A) (x) grad_x W             a = A | A - A_i | A + A_i | rho_b rho_i (A_i/rho_i^2 + A/rho_b^2)   (Naive/Difference/Summation/Symmetric)
        Divergence  : sum_e int a . grad_x W                  (vector A)
        Curl (2D)   : sum_e int (a_y d_x W - a_x d_y W)       (vector A)
    Fields are P1 nodal ([V,...] via referenceValues / referenceNodal) or P0 per element (referenceElement [E,...]).  Returns a tensor on the mesh device
    with leading dimension N (rows of excluded query kinds are zero); with `returnReaction` also the per-VERTEX reaction `[V,2]`:
        R_k = - sum_i m_i * (contribution of vertex k to out_i)          (out interpreted as an acceleration: Gradient of a scalar)
    whose sum over vertices is exactly -sum_i m_i out_i (momentum conservation)."""
    dev = mesh.device
    adj = adjacency or buildBoundaryAdjacency(queryParticles, operationProperties, mesh)
    op, mode = operationProperties.operation, operationProperties.gradientMode
    N = adj.numQueries
    qi = adj.pairQuery.long()
    w, G = adj.weights, adj.gradWeights
    nodal = referenceNodal if referenceNodal is not None else referenceValues
    if referenceElement is not None:
        A = _nodal(referenceElement, BoundaryMesh(mesh.vertices, mesh.elements, mesh.bodyIds) if mesh.vertices.shape[0] != mesh.elements.shape[0] else mesh, adj, "referenceElement")
    elif nodal is not None:
        A = _nodal(nodal, mesh, adj, "referenceValues")
    else:
        A = None

    def zeros(*shape):
        return torch.zeros((N, *shape), dtype=torch.float64, device=dev)

    reaction = None
    if op == WarpOperation.Density:
        rho_b = _nodal(referenceDensities, mesh, adj, "referenceDensities") if not isinstance(referenceDensities, float) else torch.full((len(qi), 3), referenceDensities, dtype=torch.float64, device=dev)
        out = zeros().index_add_(0, qi, (w * rho_b).sum(1))
        return out
    if op == WarpOperation.Interpolate:
        if A is None:
            raise ValueError("referenceValues required")
        contrib = (w.reshape(*w.shape, *([1] * (A.ndim - 2))) * A).sum(1)
        out = zeros(*A.shape[2:]).index_add_(0, qi, contrib)
        return out
    if op in (WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl):
        if A is None or queryValues is None:
            raise ValueError("queryValues and referenceValues are required")
        fi = queryValues.to(dev, torch.float64)[qi]                                   # [P,...]
        fiE = fi.unsqueeze(1)
        if mode == GradientScheme.Naive:
            a = A
        elif mode == GradientScheme.Difference:
            a = A - fiE
        elif mode == GradientScheme.Summation:
            a = A + fiE
        elif mode == GradientScheme.Symmetric:
            rho_i = queryParticles.densities.to(dev, torch.float64)[qi]
            rho_b = _nodal(referenceDensities, mesh, adj, "referenceDensities") if not isinstance(referenceDensities, float) else torch.full((len(qi), 3), referenceDensities, dtype=torch.float64, device=dev)
            shape = (len(qi), 3) + (1,) * (A.ndim - 2)
            a = rho_b.reshape(shape) * rho_i.reshape((-1, 1) + (1,) * (A.ndim - 2)) * (fiE / (rho_i.reshape((-1, 1) + (1,) * (A.ndim - 2)) ** 2) + A / (rho_b.reshape(shape) ** 2))
        else:
            raise ValueError(mode)
        if op == WarpOperation.Gradient:
            # out[i, c..., d] = sum_k a[i,k,c...] G[i,k,d]
            contrib = torch.einsum("pk...,pkd->p...d", a, G)
        elif op == WarpOperation.Divergence:
            contrib = torch.einsum("pkc,pkc->p", a, G)
        else:   # curl, 2D: a_y G_x - a_x G_y
            contrib = (a[..., 1] * G[..., 0] - a[..., 0] * G[..., 1]).sum(1)
        out = zeros(*contrib.shape[1:]).index_add_(0, qi, contrib)
        if returnReaction:
            if op != WarpOperation.Gradient or A.ndim != 2:
                raise NotImplementedError("reaction: Gradient of a scalar field")
            m = queryParticles.masses.to(dev, torch.float64)[qi]
            pervertex = -(m[:, None, None] * a[..., None] * G)                       # [P,3,2]
            reaction = torch.zeros((mesh.vertices.shape[0], 2), dtype=torch.float64, device=dev)
            reaction.index_add_(0, mesh.elements[adj.pairElement.long()].long().reshape(-1), pervertex.reshape(-1, 2))
            return out, reaction
        return out
    raise NotImplementedError(f"{op}: Laplacian / Covariance boundary operations are not implemented yet (see docs/boundary-operations.md)")
