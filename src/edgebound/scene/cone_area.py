"""Q3b: the exact area of solid ∩ disk(p, H) ∩ wedge(p, axis, half-angle) as a closed form per edge (no polygon clipping).

The wall part of the detector's Barecasco cone count is n_w · A, and the "all neighbours" count is n_w · area(solid ∩ disk) (the same function with a full wedge, half-angle >= π).  Derivation (REVIEW-001 §4.5): the disk ∩ wedge is star-shaped about p; on a ray at angle φ the measure of solid ∩ ray ∩ [0,H] is  sum_e s_e min(t_e(φ), H)  (t_e = distance along the ray to the crossing with edge e; s_e = +1 if the edge is traversed counter-clockwise about p, i.e. cross(A,B) > 0 with A = a-p, B = b-p, −1 otherwise), so
    area = background · area(wedge ∩ disk) + sum_e s_e ∫_{swept(e) ∩ wedge} ½ min(t_e(φ), H)² dφ .
Per edge: z = distance of p to the edge line, φ_n = angle of the foot, t_e(φ) = z / cos(φ − φ_n); inside the disk (t_e <= H) ∫ ½ t_e² dφ = ½ z² [tan(φ − φ_n)], outside ∫ ½ H² dφ.  The chord sub-interval is where t_e < H, i.e. |φ − φ_n| < acos(z/H) — strict: z == H (the tangent edge, e.g. a particle at the centre of a square of side 2H) is a pure sector, not a chord.  The loops are oriented with the SOLID ON THE LEFT (SurfaceRep convention; a tank wall is a clockwise loop around the fluid with background = 1, the solid being the unbounded outside).

Units: length² (an area, positive); n_w · result is a count.  The per-edge formula is NON-LOCAL — an edge entirely outside the disk still contributes its sector through the wedge — so edges are not culled by distance (the local form is for the later Warp kernel).
"""
import math

import numpy as np
import torch


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _edge_sector_area(p, a, b, H, th, al):
    """s_e ∫ ½ min(t_e,H)² dφ over the swept angles of edge a->b about p, restricted to the wedge |φ − th| ≤ al (al >= π: no restriction)."""
    A = np.asarray(a, float) - p
    B = np.asarray(b, float) - p
    cr = A[0] * B[1] - A[1] * B[0]
    L = math.hypot(B[0] - A[0], B[1] - A[1])
    if L == 0 or abs(cr) < 1e-300:
        return 0.0                                          # degenerate edge, or p on the edge line (measure-zero contribution)
    s = 1.0 if cr > 0 else -1.0
    z = abs(cr) / L
    pa = math.atan2(A[1], A[0])
    dphi = _wrap(math.atan2(B[1], B[0]) - pa)               # signed swept angle, |dphi| < pi
    lo, hi = (pa, pa + dphi) if dphi >= 0 else (pa + dphi, pa)
    d = (B - A) / L
    foot = A - float(np.dot(A, d)) * d                       # foot of the perpendicular from p onto the line, relative to p
    pn = math.atan2(foot[1], foot[0])
    pn = lo + _wrap(pn - lo)                                 # same branch as [lo, hi] (only differences with pn matter)
    if al >= math.pi:
        wedges = [(lo, hi)]
    else:
        c = lo + _wrap(th - lo)                              # wedge centre in the branch nearest lo; the ±2pi copies cover the straddle
        wedges = []
        for k in (-1, 0, 1):
            w0, w1 = c + 2 * math.pi * k - al, c + 2 * math.pi * k + al
            l, h_ = max(lo, w0), min(hi, w1)
            if l < h_:
                wedges.append((l, h_))
    tot = 0.0
    for (l, h_) in wedges:
        be = math.acos(z / H) if z < H else 0.0
        # chord sub-interval [c_lo, c_hi] = [l, h_] ∩ the copy of (pn − be, pn + be) whose centre is nearest the midpoint
        m = 0.5 * (l + h_)
        k = int(math.floor((m - pn) / (2 * math.pi) + 0.5))
        c_lo = max(l, pn + 2 * math.pi * k - be)
        c_hi = min(h_, pn + 2 * math.pi * k + be)
        if z < H and c_lo < c_hi:
            tot += 0.5 * z * z * (math.tan(c_hi - pn) - math.tan(c_lo - pn))   # chord (triangle)
            tot += 0.5 * H * H * ((c_lo - l) + (h_ - c_hi))                    # the two sector sides
        else:
            tot += 0.5 * H * H * (h_ - l)                                       # pure sector (includes the tangent edge z == H)
    return s * tot


def cone_area_scalar(point, axis_angle, half_angle, H, vertices, edges, background=0):
    """area(solid ∩ disk(point, H) ∩ wedge(point, axis_angle, half_angle)) for one polygon: `vertices` [V,2], `edges` [E,2] int (vertex indices; loops oriented with the solid on the left), `background` = 1 when the solid is the unbounded outside.  plain Python/math, a loop over the edges (one edge at a time).  Units length²; a positive area; n_w · result is a count."""
    p = np.asarray(point, float)
    V = np.asarray(vertices, float)
    E = np.asarray(edges, int)
    al = float(half_angle)
    Hf = float(H)
    tot = background * 0.5 * Hf * Hf * (2 * al if al < math.pi else 2 * math.pi)
    for e0, e1 in E:
        tot += _edge_sector_area(p, V[int(e0)], V[int(e1)], Hf, float(axis_angle), al)
    return float(tot)


def _seg_integral(l, h, z, H, pn, valid):
    """vectorised ∫_l^h ½ min(t_e(φ),H)² dφ for the per-(particle,edge) arrays [l, h, z, pn, valid] (all the same shape): the chord sub-interval [c_lo, c_hi] = [l, h] ∩ (pn − be, pn + be) (be = acos(z/H), the copy whose centre is nearest the midpoint) gives ½ z² (tan(c_hi−pn) − tan(c_lo−pn)) and the two sides a sector ½ H²; a zero-width interval (or z >= H) is a pure sector."""
    Hf = float(H)
    be = torch.where(z < Hf, torch.acos((z / Hf).clamp(max=1.0)), torch.zeros_like(z))
    m = 0.5 * (l + h)
    k = torch.floor((m - pn) / (2.0 * math.pi) + 0.5)
    c_lo = torch.maximum(l, pn + 2.0 * math.pi * k - be)
    c_hi = torch.minimum(h, pn + 2.0 * math.pi * k + be)
    chord_valid = valid & (z < Hf) & (c_lo < c_hi)
    sector_angle = torch.where(chord_valid, (c_lo - l) + (h - c_hi), h - l)
    sector_area = 0.5 * Hf * Hf * sector_angle
    chord_area = 0.5 * z * z * (torch.tan(c_hi - pn) - torch.tan(c_lo - pn))
    chord_area = torch.where(chord_valid, chord_area, torch.zeros_like(chord_area))
    return valid.to(z.dtype) * (sector_area + chord_area)   # an empty interval (l >= h) contributes 0


def cone_area(points, axes, half_angle, H, vertices, edges, background=0):
    """area(solid ∩ disk(p, H) ∩ wedge(p, axis, half-angle)) for `points` [N,2] and per-particle `axes` [N,2] (only the direction matters, atan2 of the components; a zero axis gives a finite, unspecified result — the detector discards it).  `vertices` [V,2], `edges` [E,2] int, loops oriented with the solid on the left, `background` = 1 for the unbounded outside.  Vectorised over the particles and the edges (no Python loop over either).  float64, on the device of `points`.  Returns [N]; units length²; n_w · result is a count.  Same convention as `cone_area_scalar`."""
    dev = points.device if isinstance(points, torch.Tensor) else torch.device("cpu")
    p = torch.as_tensor(points, dtype=torch.float64, device=dev)
    ax = torch.as_tensor(axes, dtype=torch.float64, device=dev)
    V = vertices.to(dev, torch.float64) if isinstance(vertices, torch.Tensor) else torch.as_tensor(np.asarray(vertices, float), dtype=torch.float64, device=dev)
    E = edges.to(dev, torch.int64) if isinstance(edges, torch.Tensor) else torch.as_tensor(np.asarray(edges, int), dtype=torch.int64, device=dev)
    Hf = float(H)
    al = float(half_angle)
    a = V[E[:, 0]]                                   # [E,2]
    b = V[E[:, 1]]
    d = b - a
    L = d.norm(dim=1)                                 # [E]
    A = a[None, :, :] - p[:, None, :]                 # [N,E,2]
    B = b[None, :, :] - p[:, None, :]
    cr = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]   # [N,E]
    sgn = torch.where(cr > 0, torch.ones_like(cr), -torch.ones_like(cr))
    z = cr.abs() / L[None, :].clamp(min=1e-300)        # [N,E]
    pa = torch.atan2(A[..., 1], A[..., 0])
    angB = torch.atan2(B[..., 1], B[..., 0])
    dphi = (angB - pa + math.pi) % (2 * math.pi) - math.pi
    lo = torch.where(dphi >= 0, pa, pa + dphi)
    hi = torch.where(dphi >= 0, pa + dphi, pa)
    dunit = d / L[:, None].clamp(min=1e-300)
    foot = A - (A * dunit[None]).sum(2, keepdim=True) * dunit[None]   # [N,E,2]
    pn = torch.atan2(foot[..., 1], foot[..., 0])
    pn = lo + ((pn - lo + math.pi) % (2 * math.pi) - math.pi)
    th = torch.atan2(ax[:, 1], ax[:, 0])[:, None]      # [N,1]; a zero axis gives atan2(0,0) = 0 (finite)
    two_pi = 2.0 * math.pi
    if al >= math.pi:
        seg = _seg_integral(lo, hi, z, Hf, pn, torch.ones(lo.shape, dtype=torch.bool, device=dev))
        total = sgn * seg
    else:
        c = lo + ((th - lo + math.pi) % (2 * math.pi) - math.pi)     # [N,E] wedge centre in the branch nearest lo
        ks = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, device=dev)   # the three 2pi copies (a fixed loop, not over particles/edges); float64, else the two_pi shift is rounded to float32 (a 1.7e-7 error that a steep chord near pi/2 amplifies)
        w0 = c[None] + two_pi * ks[:, None, None] - al               # [3,N,E]
        w1 = c[None] + two_pi * ks[:, None, None] + al
        l = torch.maximum(lo[None], w0)
        h = torch.minimum(hi[None], w1)
        valid = l < h
        total = (sgn[None] * _seg_integral(l, h, z[None], Hf, pn[None], valid)).sum(0)   # [N,E]
    valid_edge = (L[None, :] > 0) & (cr.abs() >= 1e-300)             # degenerate edge / p on the edge line: 0
    out = (total * valid_edge).sum(1)                                 # [N]
    out = out + background * 0.5 * Hf * Hf * (2 * al if al < math.pi else two_pi)
    return out


def cone_area_scene(scene, points, axes, half_angle, H):
    """area(solid ∩ disk(point, H) ∩ wedge(point, axis, half_angle)) for the SurfaceRep bodies of a `Scene`, summed over the bodies and their reps (units length²; `n_w · result` is a count; `half_angle >= π` = the full disk). The world vertices are `body.pose.toWorld(rep.vertices)`, recomputed on every call, so a moved body works without rebuilding the scene. Overlapping bodies are not supported (the sum would count their overlap twice)."""
    from .scene import SurfaceRep
    for body in scene.bodies:
        for rep in body.reps:
            if not isinstance(rep, SurfaceRep):
                raise NotImplementedError("cone_area_scene: SurfaceRep bodies only")
    dev = scene.device
    total = None
    for body in scene.bodies:
        for rep in body.reps:
            world = body.pose.toWorld(rep.vertices.to(dev))
            term = cone_area(points, axes, half_angle, H, world, rep.edges, rep.background)
            total = term if total is None else total + term
    if total is None:
        total = torch.zeros(points.shape[0] if isinstance(points, torch.Tensor) else len(points), dtype=torch.float64, device=dev)
    return total
