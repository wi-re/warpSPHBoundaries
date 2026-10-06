"""Periodic boxes as a property of the pair geometry (used by the scene layer for the bodies and by the solver for the fluid pairs).

The stored positions are never wrapped, clipped or written: a position is the integrated trajectory (dx/dt exact for exports and learning) and a particle may be any number of box lengths from the box.
The periodic image enters only where a displacement is formed, as the minimum image `d - L round(d / L)` on the periodic axes (support < L / 2).  A body sees a particle at its nearest image relative
to the body centre (`Body.image`, `Body.toLocal`): there is ONE real body, with its pose, loads and kinematics; the images exist only as shifted queries (and, for a bundle that spans the box, as the
tiled copies of its disks inside the one body, `DiskArrayRep.tiled`).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

_LENGTHS = {}


@dataclass(frozen=True)
class Periodic:
    lo: Tuple[float, float]
    hi: Tuple[float, float]
    flags: Tuple[bool, bool] = (True, True)

    def length(self, ref):
        """the box lengths of the periodic axes (0 on the others) as a tensor like `ref` (cached per device and dtype: no host-to-device copy after the first call, so it is legal inside a graph capture)."""
        key = (self, ref.device, ref.dtype)
        t = _LENGTHS.get(key)
        if t is None:
            t = _LENGTHS[key] = torch.tensor([(h - l) if f else 0.0 for l, h, f in zip(self.lo, self.hi, self.flags)], dtype=ref.dtype, device=ref.device)
        return t

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
