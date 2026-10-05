"""Exact wall part of the delta+ shifting tensile control (Q2).

The wall part of the shifting tensile term (Sun 2017 Eq. 7, in `DeltaSPH2D.shift`) is

    T_i = int_solid W^4 grad_x W dA'

(the integral of W^4 times the kernel gradient over the solid within the support of particle i).  This module
gives it in closed form for the Wendland C2 and C4 kernels, from the identity (which holds for both families)

    grad_x W^5 = 5 W^4 grad_x W,

with W^5 vanishing WITH ITS FIRST DERIVATIVE at the support edge ((1-q)^20 for C2, (1-q)^30 for C4), so there is
no boundary term when the gradient is moved inside the integral:

    T = int_solid W^4 grad_x W dA' = (1/5) grad_x int_solid W^5 dA'.

W^5 is the truncated-power kernel `{family}p5` (the 5th power of the Wendland shape; degree 25 for C2, 40 for C4),
registered in `kernels.KERNELS` on first use and always routed through the Chebyshev-quadrature edge plan
(`warpbc.STABLE_KERNELS[family + "p5"] = (16, 8)`).  With c the family normalisation (c*pi) and c5 the
`{family}p5` normalisation (c5*pi),  W^5(r; H) = (c^5 / (pi^4 c5 H^8)) {family}p5(r; H),  so

    T = (1/5) c^5 / (pi^4 c5 H^8)  *  g,

g the Gradient-operation result of the `{family}p5` kernel at support H -- the same scene operation
`cover_vector_scene` uses for the `cone` kernel.  Sign: T points INTO the wall (a particle just above a flat floor,
solid below it, gets T_y < 0).  Units: the integral over the solid (a 1/length^9 quantity for a 2D kernel).  The
Chebyshev plan is what makes the degree-40 C4 W^5 usable in float64 (the monomial plan is ~1e-3 off there);
`family not in ("w2", "w4")` raises.

Torch / warpSPHCore / scene are imported here so the module stays importable without them (as in cover.py).
"""
import math

import numpy as np


def _register(family):
    """idempotently register the W^5 kernel (degree 25/40 truncated powers of the Wendland `family` shape) as
    `{family}p5` in kernels.KERNELS, and always route it through the Chebyshev-quadrature plan (warpbc.STABLE_KERNELS)."""
    from ..edge import kernels, warpbc
    name = family + "p5"
    if name not in kernels.KERNELS:
        kernels.KERNELS[name] = kernels._from_terms(name, kernels.power_terms(5, family))
    warpbc.STABLE_KERNELS[name] = (16, 8)


def tensile_factor(H, family="w2"):
    """the constant (1/5) c^5 / (pi^4 c5 H^8)  (c = the family normalisation, c5 = the `{family}p5` normalisation)
    turning the `{family}p5` Gradient result into the tensile integral T = int W^4 grad W dA'."""
    from ..edge import kernels
    _register(family)
    c = float(kernels.KERNELS[family].c2_pi)
    c5 = float(kernels.KERNELS[family + "p5"].c2_pi)
    return (1.0 / 5.0) * c ** 5 / (math.pi ** 4 * c5 * float(H) ** 8)


def tensile_vector_scene(scene, positions, H, family="w2"):
    """T_i = int_solid W^4 grad_x W dA' for `positions` [N,2] (the wall part of the shifting tensile control, Q2), the
    scene-layer route: the exact edge kernel W^5 (the kernel `{family}p5`) in the Gradient operation,

        T = (1/5) c^5 / (pi^4 c5 H^8) * grad_x int_solid W^5 dA'.

    Identity: grad_x W^5 = 5 W^4 grad_x W and W^5 vanishes with its first derivative at the support edge (no boundary
    term); both families are routed through the Chebyshev-quadrature plan (warpbc.STABLE_KERNELS).  Sign: T points
    INTO the wall (a particle just above a flat floor, solid below it, gets T_y < 0).  Units: the integral over the
    solid.  SurfaceRep bodies only.  Returns [N,2] float64 on scene.device."""
    if family not in ("w2", "w4"):
        raise NotImplementedError("tensile_vector_scene: Wendland C2 and C4 only")
    import torch
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from .scene import BodyField, SurfaceRep, sceneOperation
    for body in scene.bodies:
        for rep in body.reps:
            if not isinstance(rep, SurfaceRep):
                raise NotImplementedError("tensile_vector_scene: SurfaceRep bodies only")
    _register(family)
    dev = scene.device
    pos = positions.to(dev, torch.float64) if isinstance(positions, torch.Tensor) else torch.as_tensor(np.asarray(positions), dtype=torch.float64, device=dev)
    n = pos.shape[0]
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=dev),
                       masses=torch.ones(n, dtype=torch.float64, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev),
                       densities=torch.ones(n, dtype=torch.float64, device=dev))
    pr = OperationProperties(kernel=family + "p5", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    one = BodyField(torch.tensor(1.0, dtype=torch.float64, device=dev))
    adj = scene.buildAdjacency(ps, pr, channels=(3, 4))                                        # the Naive gradient of a constant needs g0 only
    out = sceneOperation(ps, pr, scene, adj, None, [one] * len(scene.bodies), perBody=True)
    return tensile_factor(H, family) * out.sum(0)
