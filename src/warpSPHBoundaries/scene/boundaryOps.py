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

Hard tier switching: explicit meshes are tiers 1/2 (exact elements); `BoundaryDescription` adds implicit bodies (`DiskBody`, `HalfPlaneBody`) evaluated by tier 3
(curvature expansion) or tier 4 (small-obstacle series) per (particle, body) from the thresholds of `TierPolicy`, with a tier-2 polygon fallback in the gap; no blending.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, WarpOperation

from ..edge import warpbc
from .implicitBodies import DiskBody, Tier3, Tier4, TierPolicy, evaluateBody

_KERNEL_NAMES = {KernelFunctions.CubicSpline: "cubic", KernelFunctions.QuarticSpline: "quartic", KernelFunctions.QuinticSpline: "quintic",
                 KernelFunctions.B7: "b7", KernelFunctions.B8: "b8", KernelFunctions.Poly6: "poly6",
                 KernelFunctions.Wendland2: "w2", KernelFunctions.Wendland4: "w4", KernelFunctions.Wendland6: "w6"}


def kernelName(kernel: "KernelFunctions | str") -> str:
    if isinstance(kernel, str):                                            # a registered kernel name (warpSPHBoundaries.kernels.KERNELS)
        from ..edge import kernels
        if kernel in kernels.KERNELS:
            return kernel
        raise KeyError(f"unknown kernel {kernel!r}; expected one of {sorted(kernels.KERNELS)}")
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


@dataclass
class BoundaryDescription:
    """explicit element mesh (tiers 1/2) plus implicit bodies (tiers 3/4 with a hard per-particle tier switch and a tier-2 polygon fallback)."""
    mesh: Optional[BoundaryMesh] = None
    bodies: list = field(default_factory=list)          # DiskBody / HalfPlaneBody, bodyId = index in this list is NOT required (use body.bodyId)
    policy: TierPolicy = field(default_factory=TierPolicy)


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
    mesh: Optional[BoundaryMesh] = None                 # the mesh the weights refer to (explicit + polygon fallbacks)
    numExplicitVertices: int = 0
    polyBodies: Optional[list] = None                   # [(body, vertexOffset, nVertices)] for the tier-2 fallbacks
    bodyLam: Optional[list] = None                      # per implicit body: lambda [N], grad lambda [N,2], tier [N]
    bodyGrad: Optional[list] = None
    bodyTier: Optional[list] = None
    bodies: Optional[list] = None


def buildBoundaryAdjacency(queryParticles, operationProperties: OperationProperties, boundary,
                           grid: Optional[ElementGrid] = None, supportMax: Optional[float] = None) -> BoundaryAdjacency:
    """explicit mesh: candidate search through the element grid, exact distance filter, P1 pair weights on the device.
    implicit bodies: per-(particle, body) hard tier choice (3, 4, or 2 = elements of a polygonisation, generated here and merged into the mesh)."""
    desc = boundary if isinstance(boundary, BoundaryDescription) else BoundaryDescription(mesh=boundary)
    pos, sup = queryParticles.positions, queryParticles.supports
    if pos.shape[1] != 2:
        raise NotImplementedError("2D only (3D boundary integrals: not implemented)")
    name = kernelName(operationProperties.kernel)
    mesh0 = desc.mesh
    dev = mesh0.device if mesh0 is not None else pos.device
    pos, sup = pos.to(dev), sup.to(dev)
    allowed = queryAllowed(queryParticles, operationProperties.operationMode, dev)
    # ---- implicit bodies ------------------------------------------------------------------------------------
    bodyLam, bodyGrad, bodyTier, polyBodies = [], [], [], []
    tables = {}
    if desc.bodies:
        tables["t3"] = tier3Table(name, str(dev))
        tables["t4"] = tier4Table(name, str(dev))
    V = [mesh0.vertices] if mesh0 is not None else []
    E = [mesh0.elements.long()] if mesh0 is not None else []
    bid = [mesh0.bodyIds.long()] if mesh0 is not None else []
    nV = mesh0.vertices.shape[0] if mesh0 is not None else 0
    elementOwner = [torch.full((mesh0.elements.shape[0],), -1, dtype=torch.long, device=dev)] if mesh0 is not None else []
    for bi, body in enumerate(desc.bodies):
        lam, grad, tier, active = evaluateBody(body, pos, sup, name, dev, desc.policy, tables)
        mask = allowed
        bodyLam.append(torch.where(mask, lam, torch.zeros_like(lam)))
        bodyGrad.append(torch.where(mask[:, None], grad, torch.zeros_like(grad)))
        bodyTier.append(tier)
        if isinstance(body, DiskBody) and bool(((tier == 2) & mask).any()):
            Vp, Ep = body.polygon(float(sup[(tier == 2) & mask].min()), desc.policy)
            Vp = torch.as_tensor(Vp, dtype=torch.float64, device=dev)
            Ep = torch.as_tensor(Ep, dtype=torch.long, device=dev) + nV
            polyBodies.append((bi, nV, Vp.shape[0]))
            V.append(Vp); E.append(Ep)
            bid.append(torch.full((Ep.shape[0],), body.bodyId, dtype=torch.long, device=dev))
            elementOwner.append(torch.full((Ep.shape[0],), bi, dtype=torch.long, device=dev))
            nV += Vp.shape[0]
    if not V:                                            # only tier-3/4 bodies, no element anywhere: empty pair list
        if not desc.bodies:
            raise ValueError("empty boundary description")
        mesh = BoundaryMesh(torch.zeros((0, 2), dtype=torch.float64, device=dev), torch.zeros((0, 3), dtype=torch.int32, device=dev))
        e0 = torch.zeros(0, dtype=torch.int32, device=dev)
        return BoundaryAdjacency(e0, e0, torch.zeros((0, 3), dtype=torch.float64, device=dev), torch.zeros((0, 3, 2), dtype=torch.float64, device=dev),
                                 len(pos), name, mesh, 0, polyBodies, bodyLam, bodyGrad, bodyTier, desc.bodies)
    mesh = BoundaryMesh(torch.cat(V), torch.cat(E).to(torch.int32), torch.cat(bid).to(torch.int32))
    owner = torch.cat(elementOwner)
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
    keep = allowed[q]
    # polygon-fallback elements only for the particles whose hard tier for that body is 2
    own = owner[e]
    for bi, tier in enumerate(bodyTier):
        keep &= ~((own == bi) & (tier[q] != 2))
    q, e = q[keep], e[keep]
    tri = mesh.vertices[mesh.elements[e].long()]
    d = _point_triangle_distance(pos[q], tri)
    keep = d < sup[q]
    q, e = q[keep].to(torch.int32), e[keep].to(torch.int32)
    w, G = warpbc.pair_weights(q, e, pos, sup, mesh.vertices, mesh.elements, name, device=str(dev), as_torch=True)
    return BoundaryAdjacency(q, e, w, G, len(pos), name, mesh, mesh0.vertices.shape[0] if mesh0 is not None else 0, polyBodies,
                             bodyLam, bodyGrad, bodyTier, desc.bodies)


_T3, _T4 = {}, {}


def tier3Table(name, dev):
    if (name, dev) not in _T3:
        _T3[(name, dev)] = Tier3(name, dev)
    return _T3[(name, dev)]


def tier4Table(name, dev):
    if (name, dev) not in _T4:
        _T4[(name, dev)] = Tier4(name, dev)
    return _T4[(name, dev)]


def queryAllowed(queryParticles, mode: OperationDirection, dev):
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
def _to(x, dev):
    return x.to(dev, torch.float64) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64, device=dev)


def _extend_nodal(explicit, adj, bodyConst, dev, nExplicit, default=None):
    """nodal array over the COMBINED mesh: user values on the explicit vertices, per-body constants on the polygon-fallback vertices."""
    parts = []
    if adj.numExplicitVertices:
        if explicit is None:
            raise ValueError("explicit-mesh values required")
        e = _to(explicit, dev)
        parts.append(e if e.ndim else e.expand(adj.numExplicitVertices))
    for bi, off, nv in (adj.polyBodies or []):
        if bodyConst is None:
            c = default
        elif isinstance(bodyConst, (int, float)) or (isinstance(bodyConst, torch.Tensor) and bodyConst.ndim == 0):
            c = bodyConst
        else:
            c = bodyConst[bi]
        c = _to(c, dev)
        parts.append(c.expand(nv, *c.shape) if c.ndim else c.expand(nv))
    return torch.cat(parts) if parts else None


def _modeValues(mode, A, fiE, rho_i, rho_b):
    """the field combination `a` of the particle operators (per pair and node), see warpSPHCore wp_gradient.py"""
    if mode == GradientScheme.Naive:
        return A + 0 * fiE
    if mode == GradientScheme.Difference:
        return A - fiE
    if mode == GradientScheme.Summation:
        return A + fiE
    if mode == GradientScheme.Symmetric:
        ex = (1,) * (A.ndim - 2)
        ri = rho_i.reshape((-1, 1) + ex)
        rb = rho_b.reshape(rho_b.shape + ex)
        return rb * ri * (fiE / ri ** 2 + A / rb ** 2)
    raise ValueError(mode)


def boundaryOperation(queryParticles, operationProperties: OperationProperties, boundary,
                      queryValues: Optional[torch.Tensor] = None, referenceValues=None, referenceDensities=1.0,
                      adjacency: Optional[BoundaryAdjacency] = None, returnReaction: bool = False,
                      referenceElement=None, bodyValues=None, bodyDensities=1.0):
    """Boundary contribution of the requested operation for every query particle, like `warpOperation` but with exact integrals.
        Interpolate : sum int A W
        Density     : sum int rho_b W                       (rho_b = referenceDensities / bodyDensities, default 1 -> the kernel integral)
        Gradient    : sum int a (x) grad_x W                a = A | A - A_i | A + A_i | rho_b rho_i (A_i/rho_i^2 + A/rho_b^2)   (Naive/Difference/Summation/Symmetric)
        Divergence  : sum int a . grad_x W                  (vector A)
        Curl (2D)   : sum int (a_y d_x W - a_x d_y W)       (vector A)
    `boundary` is a `BoundaryMesh` (tiers 1/2: fields are P1 nodal `referenceValues[V,...]`, per-element `referenceElement[E,...]`, densities `referenceDensities`
    [V]/[E]/float) or a `BoundaryDescription` that adds implicit bodies (tiers 3/4 chosen per (particle, body) by the hard `TierPolicy`; fields are CONSTANTS per body:
    `bodyValues[B,...]`, `bodyDensities[B]`).  Returns a tensor on the boundary device with leading dimension N (rows of excluded query kinds are zero); with
    `returnReaction` also the per-VERTEX reaction `[V,2]` (and per-body reaction `[B,2]` for descriptions with bodies):
        R = - sum_i m_i * (contribution of that vertex/body to out_i)      (out interpreted as an acceleration: Gradient of a scalar)
    and the total reaction is exactly - sum_i m_i out_i (momentum conservation)."""
    desc = boundary if isinstance(boundary, BoundaryDescription) else None
    dev = (desc.mesh.device if desc is not None and desc.mesh is not None else (boundary.device if desc is None else queryParticles.positions.device))
    adj = adjacency or buildBoundaryAdjacency(queryParticles, operationProperties, boundary)
    op, mode = operationProperties.operation, operationProperties.gradientMode
    N = adj.numQueries
    mesh = adj.mesh
    nodal = referenceValues
    nodalA = None
    if op in (WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl):
        if referenceElement is not None:
            nodalA = ("element", _to(referenceElement, dev))
        else:
            nodalA = ("nodal", _extend_nodal(nodal, adj, bodyValues, dev, mesh.vertices.shape[0]) if (nodal is not None or adj.polyBodies) else None)
    if op == WarpOperation.Density:
        rho_nodal = _extend_nodal(referenceDensities, adj, bodyDensities, dev, mesh.vertices.shape[0], default=1.0) if (adj.polyBodies or not isinstance(referenceDensities, float)) else None
    # ---- pair sets: (query index [P], weights [P,K], gradient weights [P,K,2], nodal field [P,K,...] or None, rho_b [P,K], owner id for reaction)
    sets = []
    qi = adj.pairQuery.long()
    if len(qi):
        w, G = adj.weights, adj.gradWeights
        el = mesh.elements[adj.pairElement.long()].long()                              # [P,3] vertex ids
        Ap = None
        if nodalA is not None:
            kind, arr = nodalA
            if kind == "nodal":
                Ap = arr[el]
            else:
                v = arr[adj.pairElement.long()]
                Ap = v.unsqueeze(1).expand(v.shape[0], 3, *v.shape[1:])
        if op == WarpOperation.Density:
            rho_p = (rho_nodal[el] if rho_nodal is not None else torch.full(w.shape, float(referenceDensities) if isinstance(referenceDensities, float) else 1.0, dtype=torch.float64, device=dev))
        else:
            rb = _extend_nodal(referenceDensities, adj, bodyDensities, dev, mesh.vertices.shape[0], default=1.0) if (not isinstance(referenceDensities, float) or adj.polyBodies) else None
            rho_p = rb[el] if rb is not None else torch.full(w.shape, float(referenceDensities), dtype=torch.float64, device=dev)
        sets.append((qi, w, G, Ap, rho_p, el))
    nb = len(adj.bodies or [])
    for bi in range(nb):
        lam, grad, tier = adj.bodyLam[bi], adj.bodyGrad[bi], adj.bodyTier[bi]
        sel = ((tier == 3) | (tier == 4)) & (lam != 0)
        idx = torch.nonzero(sel).flatten()
        if len(idx) == 0:
            continue
        Ab = None
        if bodyValues is not None and op != WarpOperation.Density:
            c = _to(bodyValues[bi], dev)
            Ab = c.expand(len(idx), 1, *c.shape)
        rho_b = _to(bodyDensities[bi] if not isinstance(bodyDensities, float) else bodyDensities, dev).expand(len(idx), 1)
        sets.append((idx, lam[idx][:, None], grad[idx][:, None, :], Ab, rho_b, ("body", bi)))
    # ---- evaluate each pair set with the operator formulas -------------------------------------------------------
    out = None
    reaction_v = torch.zeros((mesh.vertices.shape[0], 2), dtype=torch.float64, device=dev)
    reaction_b = torch.zeros((nb, 2), dtype=torch.float64, device=dev) if nb else None
    masses = queryParticles.masses.to(dev, torch.float64)
    for (qidx, w, G, Ap, rho_p, owner) in sets:
        if op == WarpOperation.Density:
            contrib = (w * rho_p).sum(1)
        elif op == WarpOperation.Interpolate:
            if Ap is None:
                raise ValueError("referenceValues required")
            contrib = (w.reshape(*w.shape, *([1] * (Ap.ndim - 2))) * Ap).sum(1)
        elif op in (WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl):
            if Ap is None or queryValues is None:
                raise ValueError("queryValues and referenceValues are required")
            fi = queryValues.to(dev, torch.float64)[qidx]
            rho_i = queryParticles.densities.to(dev, torch.float64)[qidx] if getattr(queryParticles, "densities", None) is not None else torch.ones(len(qidx), dtype=torch.float64, device=dev)
            a = _modeValues(mode, Ap, fi.unsqueeze(1), rho_i, rho_p)
            if op == WarpOperation.Gradient:
                contrib = torch.einsum("pk...,pkd->p...d", a, G)
            elif op == WarpOperation.Divergence:
                contrib = torch.einsum("pkc,pkc->p", a, G)
            else:
                contrib = (a[..., 1] * G[..., 0] - a[..., 0] * G[..., 1]).sum(1)
            if returnReaction:
                if op != WarpOperation.Gradient or Ap.ndim != 2:
                    raise NotImplementedError("reaction: Gradient of a scalar field")
                pv = -(masses[qidx][:, None, None] * a[..., None] * G)                       # [P,K,2]
                if isinstance(owner, tuple):
                    reaction_b[owner[1]] += pv.sum((0, 1))
                else:
                    reaction_v.index_add_(0, owner.reshape(-1), pv.reshape(-1, 2))
        else:
            raise NotImplementedError(f"{op}: Laplacian / Covariance boundary operations are not implemented yet (see docs/boundary-operations.md)")
        if out is None:
            out = torch.zeros((N, *contrib.shape[1:]), dtype=torch.float64, device=dev)
        out.index_add_(0, qidx, contrib)
    if out is None:
        if op == WarpOperation.Density:
            out = torch.zeros(N, dtype=torch.float64, device=dev)
        elif op == WarpOperation.Divergence or op == WarpOperation.Curl:
            out = torch.zeros(N, dtype=torch.float64, device=dev)
        elif op == WarpOperation.Gradient and queryValues is not None:
            out = torch.zeros((N, *queryValues.shape[1:], 2), dtype=torch.float64, device=dev)
        else:
            out = torch.zeros((N,) + ((referenceValues.shape[1:]) if referenceValues is not None and hasattr(referenceValues, "shape") else ()), dtype=torch.float64, device=dev)
    if returnReaction:
        return (out, reaction_v) if reaction_b is None else (out, reaction_v, reaction_b)
    return out
