"""Exact wall part of the delta+ shifting tensile control (Q2).

The wall part of the shifting tensile term (Sun 2017 Eq. 7, in `DeltaSPH2D.shift`) is

    T_i = int_solid W^4 grad_x W dA'

(the integral of W^4 times the kernel gradient over the solid within the support of particle i).  This
module gives it in closed form for the Wendland C2 kernel, from the identity

    grad_x W^5 = 5 W^4 grad_x W,

with W^5 vanishing at the support edge (so no boundary term when the gradient is moved inside the integral):

    T = int_solid W^4 grad_x W dA' = (1/5) grad_x int_solid W^5 dA'.

W^5 is the degree-25 truncated-power kernel `w2p5` (the 5th power of the Wendland C2 shape), registered in
`kernels.KERNELS` on first use.  With c2 the Wendland C2 normalisation (c2*pi) and c25 the w2p5 normalisation
(c25*pi),  W^5(r; H) = (c2^5 / (pi^4 c25 H^8)) w2p5(r; H),  so

    T = (1/5) c2^5 / (pi^4 c25 H^8)  *  g,

g the Gradient-operation result of the `w2p5` kernel at support H -- the same scene operation `cover_vector_scene`
uses for the `cone` kernel.  Sign: T points INTO the wall (a particle just above a flat floor, solid below it,
gets T_y < 0).  Units: the integral over the solid (a 1/length^9 quantity for a 2D kernel).  Wendland C2 only:
the C4 W^5 (degree 40) needs the Chebyshev plan, so `family != "w2"` raises.

Torch / warpSPHCore / scene are imported here so the module stays importable without them (as in cover.py).
"""
import math

import numpy as np


def _register_w2p5():
    """idempotently register the W^5 kernel (degree-25 truncated powers of the Wendland C2 shape) as `w2p5` in kernels.KERNELS."""
    from . import kernels
    if "w2p5" not in kernels.KERNELS:
        from .q2_conditioning import terms
        kernels.KERNELS["w2p5"] = kernels._from_terms("w2p5", terms(5, "w2"))


def tensile_factor(H):
    """the constant (1/5) c2^5 / (pi^4 c25 H^8) turning the w2p5 Gradient result into the tensile integral T = int W^4 grad W dA'."""
    from . import kernels
    _register_w2p5()
    c2 = float(kernels.KERNELS["w2"].c2_pi)
    c25 = float(kernels.KERNELS["w2p5"].c2_pi)
    return (1.0 / 5.0) * c2 ** 5 / (math.pi ** 4 * c25 * float(H) ** 8)


def tensile_vector_scene(scene, positions, H, family="w2"):
    """T_i = int_solid W^4 grad_x W dA' for `positions` [N,2] (the wall part of the shifting tensile control, Q2), the
    scene-layer route: the exact edge kernel W^5 (the kernel `w2p5`) in the Gradient operation,

        T = (1/5) c2^5 / (pi^4 c25 H^8) * grad_x int_solid W^5 dA'.

    Identity: grad_x W^5 = 5 W^4 grad_x W and W^5 vanishes at the support edge (no boundary term).  Sign: T points
    INTO the wall (a particle just above a flat floor, solid below it, gets T_y < 0).  Units: the integral over the
    solid.  Wendland C2 only (the C4 W^5 needs the Chebyshev plan).  SurfaceRep bodies only.  Returns [N,2] float64
    on scene.device."""
    if family != "w2":
        raise NotImplementedError("tensile_vector_scene: Wendland C2 only (C4 needs the Chebyshev plan)")
    import torch
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from .scene import BodyField, SurfaceRep, sceneOperation
    for body in scene.bodies:
        for rep in body.reps:
            if not isinstance(rep, SurfaceRep):
                raise NotImplementedError("tensile_vector_scene: SurfaceRep bodies only")
    _register_w2p5()
    dev = scene.device
    pos = positions.to(dev, torch.float64) if isinstance(positions, torch.Tensor) else torch.as_tensor(np.asarray(positions), dtype=torch.float64, device=dev)
    n = pos.shape[0]
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=dev),
                       masses=torch.ones(n, dtype=torch.float64, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev),
                       densities=torch.ones(n, dtype=torch.float64, device=dev))
    pr = OperationProperties(kernel="w2p5", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    one = BodyField(torch.tensor(1.0, dtype=torch.float64, device=dev))
    out = sceneOperation(ps, pr, scene, None, None, [one] * len(scene.bodies), perBody=True)
    return tensile_factor(H) * out.sum(0)
