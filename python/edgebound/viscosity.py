"""Exact wall part of the artificial viscosity: the wall Laplacian  Delta-lambda_i = int_solid lap W dA'  (Q1, the fourth and last wall quadrature).

The wall term of the naive-Laplacian viscosity  (nu_eff / rho_i) int_solid (v_ghost - v_i) lap W dA'  with the free-slip mirror
(v_ghost - v_i = -2 u_n n, u_n = (v - v_wall).(x - x')-normal, a constant mirror normal per particle as in the pairwise form)
is  -2 nu_eff wallMass u_n / rho  Delta-lambda n  per body, so only the scalar

    Delta-lambda_i = int_solid lap W dA'     (1/length^2, per body; positive for a fluid-side particle, 0 for a particle with the
                                                      support fully inside the solid, 0 beyond H; -> 0 linearly as the particle reaches the wall)

is needed, evaluated WITHOUT any new Warp code, through the unmodified Density and Covariance scene operations on the registered
ORDINARY (normalised) kernel  L(r) = W'(r)/r  (a polynomial for the Wendland kernels: no negative powers).  With

    lap W = div(L y) = 2 L + r L'      and      Cov[L] = int_solid y (x) grad_x L dA'    (tr Cov = -int r L' dA),

    Delta-lambda = 2 lambda[L] - tr Cov[L],       lambda[L] = int_solid L dA'  (the Density operation).

Trap (why lap W is NOT registered as a kernel): the scene adds the body-indicator pseudo-pair with lambda = 1, i.e. it assumes
int K = 1 over a full disk, while int lap W dA = 0 -- a directly registered lap W gives a spurious constant c/H^2 for every
particle INSIDE a body (28.0 = 7/0.5^2 at the centre of a square of side 2H, and kernels._finish divides by int r*shape, which is
0 for a Laplacian).  With L the indicator enters as 2*ind - tr(ind*I) = 0 and cancels exactly.

Conversion (H the support, q = r/H):  W = c/(pi H^2) s(q)  (c = KERNELS[family].c2_pi = 7 / 9)  =>  L = c/(pi H^4) l(q),
l = s'/q:  w2:  l = -20 (1-q)^3;   w4:  l = -(56/3) (6 u^5 - 5 u^6),  u = 1-q.  Register the normalised kernel K_l with shape
(1-q)^3 (w2) / 6 u^5 - 5 u^6 (w4), normalisation C_l = KERNELS["l" + family].c2_pi;  K_l(r; H) = C_l shape(q)/(pi H^2), so
L = f K_l  with  f = P c / (C_l H^2),  P = -20 (w2) / -56/3 (w4)  = lap_factor(H, family),  and

    Delta-lambda = f (2 lambda[K_l] - tr Cov[K_l]).

nu_eff = fac/8 = alpha c0 H/(8 xi)  (fac = alpha c0 H/xi), derived not calibrated: for a smooth field the fluid-fluid pairwise
term of the solver is (fac/8) (lap v + 2 grad div v) in 2D (expand v to second order: the only surviving moment is int r W' dA =
-2 = -d and the isotropic fourth-moment tensor), i.e. fac/(2(d+2)), the nu of the solver's own time-step rule (dtv).  The wall
term uses the same nu_eff, so wall and bulk viscosity are one operator.  The pairwise and the Laplacian wall terms are DIFFERENT
operators near the wall (the Laplacian form damps the wall-normal velocity 3-12x less close to the wall): an intended change of
discretisation (user decision, HANDOFF Part A Q1), see docs/derivations/laplacian-wall.md.  SurfaceRep bodies only; the first
moments of lap W (a position-dependent mirror field) are not implemented.  Torch / warpSPHCore / scene are imported here so the
module stays importable without them (as in tensile.py).
"""
from fractions import Fraction

import numpy as np

# terms of the registered kernel shapes (truncated powers, knot 1) and the P of  L = f K_l,  f = P c / (C_l H^2)
TERMS = {"w2": [(1, 1, 3)], "w4": [(6, 1, 5), (-5, 1, 6)]}
PRE = {"w2": Fraction(-20), "w4": Fraction(-56, 3)}


def _register(family):
    """idempotently register the kernel L = W'/r of the Wendland `family` (shape (1-q)^3 for w2, 6 u^5 - 5 u^6 for w4, u = 1-q)
as `"l" + family` in kernels.KERNELS."""
    if family not in ("w2", "w4"):
        raise NotImplementedError("lap_lambda_scene: Wendland C2 and C4 only")
    from . import kernels
    name = "l" + family
    if name not in kernels.KERNELS:
        kernels.KERNELS[name] = kernels._from_terms(name, TERMS[family])


def lap_factor(H, family="w2"):
    """the constant f = P c / (C_l H^2)  (P = -20 / -56/3, c the family normalisation, C_l the `l+family` normalisation) turning the
    Density / Covariance results of the registered kernel L = W'/r into the wall Laplacian:  Delta-lambda = f (2 lambda - tr Cov)."""
    from . import kernels
    _register(family)
    c = float(kernels.KERNELS[family].c2_pi)
    cl = float(kernels.KERNELS["l" + family].c2_pi)
    return float(PRE[family]) * c / (cl * float(H) ** 2)


def lap_lambda_scene(scene, positions, H, family="w2"):
    """Delta-lambda_i = int_solid lap W dA' (per body) for `positions` [N,2] (the exact wall part of the naive-Laplacian artificial
    viscosity, Q1), the scene-layer route without any Warp code:

        Delta-lambda = f (2 lambda[K_l] - tr Cov[K_l]),   K_l the registered normalised kernel L = W'/r  (kernel `"l" + family`),

    with lambda the Density and Cov the Covariance scene operation (Cov = int y (x) grad_x L dA', tr Cov = -int r L' dA; the
    identity lap W = div(L y) = 2 L + r L').  Sign: positive for a fluid-side particle (z > 0 above a floor), 0 for a particle
    with the support fully inside the solid and beyond H; the indicator pseudo-pair cancels exactly (2*ind - tr(ind*I) = 0) --
    registering lap W directly would not (it would give a spurious c/H^2 inside a body).  Units: 1/length^2, per body.
    SurfaceRep bodies only.  Returns [B, N] float64 on scene.device."""
    from .scene import SurfaceRep
    for body in scene.bodies:
        for rep in body.reps:
            if not isinstance(rep, SurfaceRep):
                raise NotImplementedError("lap_lambda_scene: SurfaceRep bodies only")
    _register(family)
    import torch
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from .scene import BodyField, sceneOperation
    dev = scene.device
    pos = positions.to(dev, torch.float64) if isinstance(positions, torch.Tensor) else torch.as_tensor(np.asarray(positions), dtype=torch.float64, device=dev)
    n = pos.shape[0]
    B = len(scene.bodies)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=dev),
                       masses=torch.ones(n, dtype=torch.float64, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev),
                       densities=torch.ones(n, dtype=torch.float64, device=dev))
    lam = cov = None
    for op in (WarpOperation.Density, WarpOperation.Covariance):
        pr = OperationProperties(kernel="l" + family, operation=op, gradientMode=GradientScheme.Naive,
                                 operationMode=OperationDirection.BoundaryToFluid)
        out = sceneOperation(ps, pr, scene, None, None, [BodyField(rho=1.0)] * B, perBody=True)
        if op == WarpOperation.Density:
            lam = out.reshape(B, n)
        else:
            cov = out.reshape(B, n, 2, 2)
    return lap_factor(H, family) * (2.0 * lam - cov[:, :, 0, 0] - cov[:, :, 1, 1])
