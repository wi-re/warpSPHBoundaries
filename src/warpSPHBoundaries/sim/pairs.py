"""Fluid-fluid building blocks shared by the two solvers: the float64 dtype, the Wendland C2 / C4 pair kernels in 2D, and the ordered neighbour pairs (dense
distance matrix for small systems, cell list otherwise).  Phase 3 of HANDOFF replaces these with warpSPH modules; keep them isolated."""
import math

import torch

from ..scene.scene import buildCellList

F64 = torch.float64


def wendland2(r, h):
    q = r / h
    return torch.where(q < 1, 7.0 / (math.pi * h * h) * (1 - q) ** 4 * (1 + 4 * q), torch.zeros_like(q))


def dwendland2(r, h):
    q = r / h
    return torch.where(q < 1, -7.0 / (math.pi * h ** 3) * 20.0 * q * (1 - q) ** 3, torch.zeros_like(q))


DENSE_PAIRS_MAX = 8000


def neighbor_pairs(pos, h):
    """(i, j, r) for all ordered pairs |x_i - x_j| <= (h_i + h_j) / 2, including i = j (as omniSPH's neighbour lists do)."""
    dev = pos.device
    if len(pos) <= DENSE_PAIRS_MAX:                       # small systems: one distance matrix is much faster than the cell loop (launch / sync bound)
        r = torch.cdist(pos, pos)
        i, j = torch.nonzero(r <= 0.5 * (h[:, None] + h[None, :]), as_tuple=True)
        return i, j, r[i, j]
    cell = 1.01 * float(h.max())        # not exactly the support: a lattice with spacing dx | H puts particles exactly on cell borders, where the insertion and the query round differently
                                        # and ~2 % of the reverse pairs go missing (momentum conservation is lost)
    cl = buildCellList(pos, pos, cell)
    nx, ny = cl.dims
    c = torch.floor((pos - cl.lo) / cell).long()
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = c[:, 0] + dx, c[:, 1] + dy
            ok = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
            cid = torch.where(ok, cx * ny + cy, torch.zeros_like(cx))
            cnt = torch.where(ok, cl.start[cid + 1] - cl.start[cid], torch.zeros_like(cid))
            q = torch.repeat_interleave(torch.arange(len(pos), device=dev), cnt)
            off = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            I.append(q)
            J.append(cl.items[cl.start[cid[q]] + off])
    i, j = torch.cat(I), torch.cat(J)
    r = (pos[i] - pos[j]).norm(dim=1)
    keep = r <= 0.5 * (h[i] + h[j])
    return i[keep], j[keep], r[keep]


def wendland4(r, h):
    """Wendland C4 in 2D, support radius h: 9/(pi h^2) (1 - q)^6 (1 + 6 q + 35 q^2 / 3)."""
    q = r / h
    return torch.where(q < 1, 9.0 / (math.pi * h * h) * (1 - q) ** 6 * (1 + 6 * q + 35.0 / 3.0 * q * q), torch.zeros_like(q))


def dwendland4(r, h):
    q = r / h
    return torch.where(q < 1, -9.0 / (math.pi * h ** 3) * (56.0 / 3.0) * q * (1 + 5 * q) * (1 - q) ** 5, torch.zeros_like(q))
