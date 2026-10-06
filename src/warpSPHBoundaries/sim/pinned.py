"""A prescribed-velocity (Dirichlet) region of FLUID particles: the wrapping band of a periodic flow past a body (the warpSPH `movingObstacle` setup).

The band is a set of slabs `|min_image(x_axis - centre)| < halfWidth` in the periodic box (a slab at the seam wraps around it), the velocity inside is the free stream `velocity`.  The particles stay fluid particles:
they carry density, pressure and mass in every sum, they are advected with the stream through the periodic box, and the count and the identities never change (no inlet, no outlet, no deletion).  Inside the band
the momentum equation is replaced by the prescription (`weight` 1: no acceleration, the velocity is set to the stream at the end of every step) and the shift is switched off; `ramp` > 0 makes the transition to the
free region a linear weight over that width instead of a step.  The positions are not touched: the slab membership uses the minimum image of the raw position.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..scene.periodic import Periodic, min_image


@dataclass(frozen=True)
class Pinned:
    slabs: Tuple[Tuple[int, float, float], ...]        # (axis, centre, halfWidth) per slab; the union is the band
    velocity: Tuple[float, float] = (0.0, 0.0)
    ramp: float = 0.0

    def weight(self, x, periodic: Optional[Periodic] = None):
        """[N] float64 in [0, 1]: 1 inside the band, 0 outside, the linear ramp of width `ramp` inside the band edge."""
        w = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        for axis, centre, half in self.slabs:
            d = x[:, axis] - centre
            if periodic is not None and periodic.flags[axis]:
                L = periodic.hi[axis] - periodic.lo[axis]
                d = d - L * torch.round(d / L)
            s = half - d.abs()                                                               # > 0 inside
            w = torch.maximum(w, (s > 0).to(x.dtype) if self.ramp <= 0 else (s / self.ramp).clamp(0.0, 1.0))
        return w
