"""Fluid-fluid building blocks shared by the two solvers: the float64 dtype, the Wendland C2 / C4 pair kernels in 2D, and the ordered neighbour pairs (dense
distance matrix for small systems, cell list otherwise).  Phase 3 of HANDOFF replaces these with warpSPH modules; keep them isolated."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..scene.scene import buildCellList

F64 = torch.float64


def wendland2(r, h):
    q = r / h
    return torch.where(q < 1, 7.0 / (math.pi * h * h) * (1 - q) ** 4 * (1 + 4 * q), torch.zeros_like(q))


def dwendland2(r, h):
    q = r / h
    return torch.where(q < 1, -7.0 / (math.pi * h ** 3) * 20.0 * q * (1 - q) ** 3, torch.zeros_like(q))


@dataclass(frozen=True)
class Periodic:
    """a periodic box: `lo`, `hi` (x, y) and the periodic axes `flags`.  Periodicity is a property of the pair geometry only: the stored positions are never wrapped, clipped or written (the state x is
    the integrated trajectory, dx/dt stays exact for exports and learning), a particle may be any number of box lengths away from the box.  The displacement of a pair is the minimum image,
    `d - L round(d / L)`, at the place it is formed; a search hashes a temporary wrapped copy and the pair distances are computed from the raw positions.  Needs the support below L / 2 on a periodic axis."""
    lo: Tuple[float, float]
    hi: Tuple[float, float]
    flags: Tuple[bool, bool] = (True, True)

    def length(self, ref):
        """the box lengths of the periodic axes (0 on the others) as a tensor like `ref`."""
        return torch.tensor([(h - l) if f else 0.0 for l, h, f in zip(self.lo, self.hi, self.flags)], dtype=ref.dtype, device=ref.device)

    def checkSupport(self, H):
        for a, f in enumerate(self.flags):
            if f and 2.0 * H >= self.hi[a] - self.lo[a]:
                raise ValueError(f"periodic axis {a}: the box length {self.hi[a] - self.lo[a]} must exceed twice the support {H}")


def min_image(d, periodic: Optional[Periodic]):
    """the minimum-image form of the displacements `d` [..., 2] (any integer multiple of the box length removed on the periodic axes); `d` itself without periodicity."""
    if periodic is None:
        return d
    L = periodic.length(d)
    return d - L * torch.round(d / torch.where(L > 0, L, torch.ones_like(L)))


def pair_delta(pos, i, j, periodic: Optional[Periodic] = None):
    """x_i - x_j of the pairs (i, j), the minimum image on the periodic axes; the positions are only read."""
    return min_image(pos[i] - pos[j], periodic)


def _wrapped(pos, periodic: Periodic):
    """a temporary copy of `pos` inside the box on the periodic axes (cell ids only; never stored)."""
    L = periodic.length(pos)
    lo = torch.tensor(periodic.lo, dtype=pos.dtype, device=pos.device)
    return torch.where(L > 0, lo + torch.remainder(pos - lo, torch.where(L > 0, L, torch.ones_like(L))), pos)


DENSE_PAIRS_MAX = 8000
DENSE_PAIRS_MAX_PERIODIC = 2000                          # the minimum-image distance matrix is formed explicitly [N, N, 2]


def neighbor_pairs(pos, h, periodic: Optional[Periodic] = None):
    """(i, j, r) for all ordered pairs |x_i - x_j| <= (h_i + h_j) / 2, including i = j (as omniSPH's neighbour lists do); with `periodic` the distance of the minimum image, from the raw positions."""
    dev = pos.device
    if len(pos) <= (DENSE_PAIRS_MAX if periodic is None else DENSE_PAIRS_MAX_PERIODIC):          # small systems: one distance matrix is much faster than the cell loop (launch / sync bound)
        r = torch.cdist(pos, pos) if periodic is None else min_image(pos[:, None, :] - pos[None, :, :], periodic).norm(dim=2)
        i, j = torch.nonzero(r <= 0.5 * (h[:, None] + h[None, :]), as_tuple=True)
        return i, j, r[i, j]
    cell = 1.01 * float(h.max())        # not exactly the support: a lattice with spacing dx | H puts particles exactly on cell borders, where the insertion and the query round differently
                                        # and ~2 % of the reverse pairs go missing (momentum conservation is lost)
    w = pos if periodic is None else _wrapped(pos, periodic)             # the hash sees the wrapped copy only
    cl = buildCellList(w, w, cell)
    nx, ny = cl.dims
    L = None if periodic is None else periodic.length(pos)
    shifts = [(0.0, 0.0)] if periodic is None else [(sx * float(L[0]), sy * float(L[1])) for sx in ((-1, 0, 1) if periodic.flags[0] else (0,)) for sy in ((-1, 0, 1) if periodic.flags[1] else (0,))]
    I, J = [], []
    for sx, sy in shifts:                                               # a periodic image is a query shifted by a box length against the same cell list
        c = torch.floor((w - torch.tensor([sx, sy], dtype=w.dtype, device=dev) - cl.lo) / cell).long()
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
    if periodic is not None:                                            # in a box of fewer than ~3 cells two images can list the same pair
        key = torch.unique(i * len(pos) + j)
        i, j = key // len(pos), key % len(pos)
    r = pair_delta(pos, i, j, periodic).norm(dim=1)
    keep = r <= 0.5 * (h[i] + h[j])
    return i[keep], j[keep], r[keep]


def wendland4(r, h):
    """Wendland C4 in 2D, support radius h: 9/(pi h^2) (1 - q)^6 (1 + 6 q + 35 q^2 / 3)."""
    q = r / h
    return torch.where(q < 1, 9.0 / (math.pi * h * h) * (1 - q) ** 6 * (1 + 6 * q + 35.0 / 3.0 * q * q), torch.zeros_like(q))


def dwendland4(r, h):
    q = r / h
    return torch.where(q < 1, -9.0 / (math.pi * h ** 3) * (56.0 / 3.0) * q * (1 + 5 * q) * (1 - q) ** 5, torch.zeros_like(q))
