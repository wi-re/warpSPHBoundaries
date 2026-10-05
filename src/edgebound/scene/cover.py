"""Exact wall part of the Barecasco free-surface cover vector (Q3a).

The free-surface detector (deltasph2d._detect_surface) sums the cover vector
C = sum_j unit(x_i - x_j) over fluid neighbours and adds the wall as a continuum
of wall particles of number density n_w.  The wall part is

    C_w = n_w  int_{solid, r <= H} unit(x_i - x') dA'        (now by polar sampling).

This module gives it in closed form.  With the CONTINUOUS kernel K(r) = (r - H) 1[r <= H]
(K vanishes at r = H, so the gradient has neither a circle term nor an indicator term):

    int_T K(|x - x'|) dA'  and  grad_x of it  =  - sum_e n_e int_{chord(e)} K ds,

where n_e is the outward unit normal of edge e of the solid polygon (counter-clockwise),
chord(e) the part of the edge inside the disk of radius H around x, and

    int_chord K ds = [I_1(s)]_{s_lo}^{s_hi} - H (s_hi - s_lo),
    I_1(s) = (s r + z^2 asinh(s/|z|)) / 2,   r = sqrt(s^2 + z^2),   z = n_e . (p_e - x),

with the asinh term 0 at z = 0.  This is exactly `block_grad(n=1, R=H) - H * block_grad(n=0, R=H)`
of core.py (the truncated monomials r^1 1[r<=R] and r^0 1[r<=R]).

Both functions return  grad_x int_{solid} K dA'   in units of length^2 (true geometry;
the `h` used by core is h = 1, i.e. no scaling, so the result needs no rescaling and
scales as H^2 under a uniform scaling of the geometry).  The detector's wall cover vector is

    C_w = n_w * (this result),

and the sign convention is: a positive component points to +x; for a particle just above a
flat floor (solid below it) the result points UP (+y), away from the wall.

Why K = (r - H) 1[r <= H] and not r 1[r <= H]: the edge-only formula  -sum_e n_e int_chord f ds  is the exact
gradient of  int_solid f(|x - x'|) dA'  for ANY locally integrable f.  For f = r 1[r <= H] that gradient is
int_solid (unit(x - x') 1[r <= H] + H unit(x - x') delta(r - H)) dA', i.e. cover vector PLUS a circle term; K
vanishes at r = H, so for K the circle term is absent and the edge formula gives the cover vector alone.
"""
import numpy as np
import mpmath as mp

from ..edge import geometry as G
from ..edge.core import GUARD, block_grad


def cover_gradient_mp(verts, x, H, dps=40):
    """grad_x int_{solid} K(|x - x'|) dA'  (units length^2), K(r) = (r - H) 1[r <= H], by the
    exact edge reduction (mpmath).  `verts`: ccw polygon; `x`: query point; `H`: support radius.
    C_w = n_w * this.  Positive component = +x; above a flat floor it points up (away from the
    wall).  Built from core.block_grad: block_grad(1, H) - H * block_grad(0, H)."""
    with mp.workdps(dps + GUARD):
        P = G.prepare(verts, x, h=1)
        g1 = block_grad(P, 1, H)
        g0 = block_grad(P, 0, H)
        return tuple(+(g1[i] - H * g0[i]) for i in range(2))


def _edges(verts_or_edges):
    """(P [E,2], Q [E,2], n [E,2] outward unit normals) from a ccw polygon [M,2] or edges [E,2,2]."""
    a = np.asarray(verts_or_edges, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        P = a
        x, y = P[:, 0], P[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        if np.sum(x * y2 - x2 * y) < 0:                     # clockwise -> reverse (match G.prepare)
            P = P[::-1]
        Q = np.roll(P, -1, axis=0)
    elif a.ndim == 3 and a.shape[1] == 2 and a.shape[2] == 2:
        P, Q = a[:, 0], a[:, 1]
    else:
        raise ValueError("verts_or_edges must be a [M,2] polygon or [E,2,2] edges (counter-clockwise)")
    d = Q - P
    ell = np.linalg.norm(d, axis=1)
    t = d / ell[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    return P, Q, n


def _I1(s, z):
    """I_1(s) = int r ds = (s r + z^2 asinh(s/|z|)) / 2, r = sqrt(s^2 + z^2); the asinh term is 0 at z = 0."""
    r = np.hypot(s, z)
    az = np.abs(z)
    safe = np.where(az > 0.0, az, 1.0)
    ratio = np.where(az > 0.0, s / safe, 0.0)
    asinh_term = np.where(az > 0.0, z * z * np.arcsinh(ratio), 0.0)
    return 0.5 * (s * r + asinh_term)


def cover_vector_np(points, verts_or_edges, H):
    """grad_x int_{solid} K dA' for `points` [M,2] (units length^2), numpy float64, vectorised over
    points, closed form (own I_1, no mpmath).  Same convention as cover_gradient_mp: C_w = n_w * this,
    positive component = +x, above a flat floor points up (away from the wall)."""
    points = np.asarray(points, dtype=np.float64)
    single = points.ndim == 1
    if single:
        points = points[None, :]
    P, Q, n = _edges(verts_or_edges)
    out = np.zeros((len(points), 2), dtype=np.float64)
    H = float(H)
    for e in range(len(P)):
        ne = n[e]
        t = np.array([-ne[1], ne[0]])                        # unit tangent, n = (t_y, -t_x)
        rel_p = P[e] - points                                # p_e - x   [M,2]
        rel_q = Q[e] - points                                # q_e - x   [M,2]
        z = rel_p @ ne                                       # n . (p_e - x), constant on the edge  [M]
        s0 = rel_p @ t                                       # t . (p_e - x)  [M]
        s1 = rel_q @ t                                       # t . (q_e - x)  [M]
        L = np.sqrt(np.maximum(H * H - z * z, 0.0))
        lo = np.maximum(s0, -L)
        hi = np.minimum(s1, L)
        valid = hi > lo
        if not valid.any():
            continue
        lo_v, hi_v, z_v = lo[valid], hi[valid], z[valid]
        contrib = _I1(hi_v, z_v) - _I1(lo_v, z_v) - H * (hi_v - lo_v)
        out[valid, 0] += -ne[0] * contrib
        out[valid, 1] += -ne[1] * contrib
    return out[0] if single else out


def cover_vector_scene(scene, positions, H, adjacency=None):  # adjacency: a SceneAdjacency of `positions` (e.g. `restrict` of a larger one) or its PairMoments (cone, channels 3, 4)
    """grad_x int_{solid} K dA' for `positions` [N,2] (units length^2), the scene-layer route (Q3a): the unmodified
    Warp edge kernel with the degree-1 kernel `cone`,  W_cone = 3 (1 - q) / (pi h^2)  (normalisation 3),  so that
    W_cone = -3 K(r) / (pi H^3) for  K(r) = (r - H) 1[r <= H]  at h = H  and  grad_x int K dA' = -(pi H^3 / 3) * g0,
    g0 the Gradient-operation result of `cone` at support H.  `H` is the support radius = the cover radius.
    Same sign and units as `cover_vector_np`:  C_w = n_w * this,  positive component = +x,  a particle just above a
    flat floor (solid below it) gets a vector pointing UP (away from the wall).  SurfaceRep bodies only.
    Torch / warpSPHCore / scene are imported here so the numpy and mpmath functions stay importable without torch."""
    import math
    import torch
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

    from .scene import BodyField, SurfaceRep, sceneOperation
    for body in scene.bodies:
        for rep in body.reps:
            if not isinstance(rep, SurfaceRep):
                raise NotImplementedError("cover_vector_scene: SurfaceRep bodies only")
    dev = scene.device
    pos = positions.to(dev, torch.float64) if isinstance(positions, torch.Tensor) else torch.as_tensor(np.asarray(positions), dtype=torch.float64, device=dev)
    n = pos.shape[0]
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=dev),
                       masses=torch.ones(n, dtype=torch.float64, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev),
                       densities=torch.ones(n, dtype=torch.float64, device=dev))
    pr = OperationProperties(kernel="cone", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    one = BodyField(torch.tensor(1.0, dtype=torch.float64, device=dev))
    pm = scene.moments(adjacency, ps, pr, channels=(3, 4))                                       # the Naive gradient of a constant needs g0 only
    out = sceneOperation(ps, pr, scene, pm, None, [one] * len(scene.bodies), perBody=True)
    return -(math.pi * float(H) ** 3 / 3) * out.sum(0)
