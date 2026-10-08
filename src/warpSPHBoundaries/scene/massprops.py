"""Mass properties of an analytic body from its representation (area, centroid, polar moment), the data warpSPH's `RigidBody` takes from the boundary particles' masses (`buildRigidBody`: mass = sum m,
centre of mass = sum m x / M, inertia = sum m |x - com|^2) and an analytic body does not have (docs/audit-warpsph-boundary-hooks.md s.6, two-way coupling).

Moments of the SOLID about the body-frame origin, exact (Green's theorem on the boundary loops, closed forms for the primitives); the representations of a body are disjoint and add:

    m0 = int dA,   m1 = int x dA (2),   J = int |x|^2 dA

    SurfaceRep  per edge (a -> b, solid on the left), c = a x b:  m0 = sum c / 2,  m1 = sum c (a + b) / 6,  J = sum c (|a|^2 + a.b + |b|^2) / 12
    BoxRep, ImplicitRep(DiskBody), DiskArrayRep: rectangle / disk closed forms (J = pi R^4 / 2 + m0 |centre|^2)
    SdfRep      its `fallback` contour

A body whose solid is unbounded (a tank wall, a hole in an infinite solid, a half plane: `solid = 'outside'`, `background = 1`, negative area) has no finite mass: ValueError.
"""
import math

import torch

F64 = torch.float64


def rep_moments(rep):
    """(m0, m1 [2], J) of one representation about the body origin (host floats)."""
    from .implicitBodies import DiskBody, HalfPlaneBody
    from .scene import BoxRep, DiskArrayRep, ImplicitRep, SdfRep, SurfaceRep
    if isinstance(rep, SurfaceRep):
        P, E = rep.vertices.double().cpu(), rep.edges.long().cpu()
        a, b = P[E[:, 0]], P[E[:, 1]]
        c = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
        m0 = float(c.sum()) / 2.0
        m1 = ((c[:, None] * (a + b)).sum(0) / 6.0).tolist()
        J = float((c * ((a * a).sum(1) + (a * b).sum(1) + (b * b).sum(1))).sum()) / 12.0
        return m0, m1, J
    if isinstance(rep, BoxRep):
        (x0, y0), (x1, y1) = rep.lo_h, rep.hi_h
        w, h = x1 - x0, y1 - y0
        m0 = w * h
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        m = (m0, [m0 * cx, m0 * cy], m0 * (w * w + h * h) / 12.0 + m0 * (cx * cx + cy * cy))
        return m if rep.solid == "inside" else tuple(-v if not isinstance(v, list) else [-u for u in v] for v in m)
    if isinstance(rep, DiskArrayRep):
        R = rep.radii.double().cpu()
        C = rep.centres.double().cpu()
        A = math.pi * R * R
        return float(A.sum()), (A[:, None] * C).sum(0).tolist(), float((A * (0.5 * R * R + (C * C).sum(1))).sum())
    if isinstance(rep, ImplicitRep):
        if isinstance(rep.shape, DiskBody):
            s = rep.shape
            A = math.pi * s.radius ** 2
            cx, cy = (float(v) for v in torch.as_tensor(s.center, dtype=F64))
            m = (A, [A * cx, A * cy], A * (0.5 * s.radius ** 2 + cx * cx + cy * cy))
            return m if s.solid == "inside" else tuple(-v if not isinstance(v, list) else [-u for u in v] for v in m)
        if isinstance(rep.shape, HalfPlaneBody):
            raise ValueError("a half plane has no finite mass")
    if isinstance(rep, SdfRep) and rep.fallback is not None:
        return rep_moments(rep.fallback)
    raise NotImplementedError("mass properties: no moments for %s" % type(rep).__name__)


def mass_properties(body, rho=1.0):
    """dict(mass, com [2] (body frame), inertiaOrigin, inertia (about the centre of mass), area) of `body` with the uniform density `rho` (a rigid body: rho_body, the mass is rho x area).  The
    body-frame origin is the body's `center` (Pose), so the world centre of mass is `body.center + R(angle) com`."""
    m0, m1x, m1y, J = 0.0, 0.0, 0.0, 0.0
    for r in body.reps:
        a, (bx, by), j = rep_moments(r)
        m0, m1x, m1y, J = m0 + a, m1x + bx, m1y + by, J + j
    if m0 <= 0.0:
        raise ValueError("the solid of this body is unbounded (negative or zero enclosed area %.6g): no finite mass" % m0)
    com = (m1x / m0, m1y / m0)
    Jc = J - m0 * (com[0] ** 2 + com[1] ** 2)
    return dict(mass=rho * m0, com=com, inertiaOrigin=rho * J, inertia=rho * Jc, area=m0)
