"""Reviewer's probe for WORK-003 T3.1 (throw-away, NOT the deliverable; write your own code, do not import this).
Q3b: area( solid ∩ disk(p, H) ∩ wedge(p, axis angle th, half-angle al) ) in closed form, per edge, no polygon clipping.

Derivation (REVIEW-001 §4.5).  R = disk ∩ wedge is star-shaped about p.  On a ray at angle phi from p the measure of solid ∩ ray ∩ [0,H] is
   sum_e  s_e * min(t_e(phi), H)            (t_e = distance along the ray to the crossing with edge e, s_e = +1 if the edge is traversed
                                              counter-clockwise as seen from p, i.e. cross(A,B) > 0, A = a-p, B = b-p, -1 if clockwise)
and  ∫_0^H t 1[solid] dt = ½ sum_e s_e min(t_e,H)^2 , so
   area = background * area(R) + sum_e s_e ∫_{phi in swept(e) ∩ wedge} ½ min(t_e(phi), H)^2 dphi ,
where the surface loops are oriented with the SOLID ON THE LEFT (SurfaceRep convention; a tank wall = clockwise loop around the fluid with
background = 1: the solid is the unbounded outside).  Per edge: z = |cross(A,B)|/|B-A| (distance of p to the edge line), phi_n = angle of the foot,
t_e(phi) = z / cos(phi - phi_n).  Inside the disk (t_e <= H) ∫ ½ t_e² dphi = ½ z² [tan(phi - phi_n)]; outside ∫ ½ H² dphi.  Breakpoints: the swept interval,
the wedge limits, and phi_n ± acos(z/H) (if z < H).  Test of the middle of each sub-interval decides chord (triangle) or sector.
Full wedge (half-angle >= pi) = the disk: this is the "all-neighbour count" area.

usage: python docs/work/refs/cone_area_probe.py   (numpy only)
"""
import math
import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def edge_sector_area(p, a, b, H, th, al):
    """s_e ∫ ½ min(t_e,H)^2 dphi over the swept angles of edge a->b seen from p, restricted to the wedge |phi - th| <= al (al >= pi: no restriction)."""
    A, B = np.asarray(a, float) - p, np.asarray(b, float) - p
    cr = A[0] * B[1] - A[1] * B[0]
    L = np.linalg.norm(B - A)
    if L == 0 or abs(cr) < 1e-300:
        return 0.0                                              # degenerate edge, or p on the edge line (swept angle 0 or pi: measure-zero contribution)
    s = 1.0 if cr > 0 else -1.0
    z = abs(cr) / L
    pa = math.atan2(A[1], A[0])
    dphi = _wrap(math.atan2(B[1], B[0]) - pa)                   # signed swept angle, |dphi| < pi
    lo, hi = (pa, pa + dphi) if dphi >= 0 else (pa + dphi, pa)  # swept interval (unwrapped), lo < hi
    d = (B - A) / L                                             # foot of the perpendicular from p onto the line
    foot = A - np.dot(A, d) * d
    pn = math.atan2(foot[1], foot[0])
    pn = lo + _wrap(pn - lo) if _wrap(pn - lo) >= 0 else lo + _wrap(pn - lo)   # same branch as the interval (only differences with pn matter)

    # wedge intervals in the unwrapped frame of [lo, hi]
    if al >= math.pi:
        wedges = [(lo, hi)]
    else:
        c = lo + _wrap(th - lo)                                 # wedge centre nearest lo; also try +-2pi
        wedges = []
        for k in (-1, 0, 1):
            w0, w1 = c + 2 * math.pi * k - al, c + 2 * math.pi * k + al
            l, h_ = max(lo, w0), min(hi, w1)
            if l < h_:
                wedges.append((l, h_))
    tot = 0.0
    for (l, h_) in wedges:
        br = [l, h_]
        if z < H:
            be = math.acos(z / H)
            for k in (-1, 0, 1):
                for sgn in (-1, 1):
                    x = pn + 2 * math.pi * k + sgn * be
                    if l < x < h_:
                        br.append(x)
        br = sorted(br)
        for u0, u1 in zip(br[:-1], br[1:]):
            mid = 0.5 * (u0 + u1)
            if z < H and math.cos(mid - pn) > 0 and z / math.cos(mid - pn) < H:          # chord inside the disk (strict: z == H, the tangent edge, is a sector)
                tot += 0.5 * z * z * (math.tan(u1 - pn) - math.tan(u0 - pn))
            else:
                tot += 0.5 * H * H * (u1 - u0)
    return s * tot


def cone_area(p, loops, H, th, al, background=0):
    """loops: list of vertex arrays (closed, solid on the left)."""
    p = np.asarray(p, float)
    tot = background * (0.5 * H * H * (2 * al if al < math.pi else 2 * math.pi))
    for V in loops:
        V = np.asarray(V, float)
        for k in range(len(V)):
            tot += edge_sector_area(p, V[k], V[(k + 1) % len(V)], H, th, al)
    return tot


def brute(p, loops, H, th, al, background=0, nr=1500, nphi=3000):
    """independent: midpoint polar grid over disk ∩ wedge, point-in-polygon by the even-odd rule (no winding, no edge formula)."""
    p = np.asarray(p, float)
    a0, a1 = (th - al, th + al) if al < math.pi else (0.0, 2 * math.pi)
    ph = a0 + (np.arange(nphi) + .5) / nphi * (a1 - a0)
    rr = (np.arange(nr) + .5) / nr * H
    X = p[0] + rr[:, None] * np.cos(ph)[None]
    Y = p[1] + rr[:, None] * np.sin(ph)[None]
    inside = np.zeros(X.shape, bool)
    for V in loops:
        V = np.asarray(V, float)
        for k in range(len(V)):
            (x1, y1), (x2, y2) = V[k], V[(k + 1) % len(V)]
            cond = (y1 > Y) != (y2 > Y)
            xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
            inside ^= cond & (X < xi)
    solid = inside if background == 0 else ~inside
    w = rr[:, None] * (H / nr) * ((a1 - a0) / nphi) * np.ones((1, nphi))
    return float((w * solid).sum())


if __name__ == "__main__":
    SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]                       # CCW: solid inside
    LS = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    TANK_FLUID_LOOP = [(0, 0), (0, 1), (1, 1), (1, 0)]           # CW around the fluid box, background = 1 (solid outside)
    cases = [
        ("square, p outside, wedge toward it", (1.3, 0.5), [SQ], 1.0, math.pi, math.pi / 6, 0),
        ("square, p outside, wedge away", (1.3, 0.5), [SQ], 1.0, 0.0, math.pi / 6, 0),
        ("square, p inside, wedge toward the +x wall", (0.8, 0.5), [SQ], 1.0, 0.0, math.pi / 6, 0),
        ("square, full disk (all-neighbour area)", (0.3, 0.4), [SQ], 1.0, 0.0, math.pi, 0),
        ("square, p near the vertex, wedge on the diagonal", (-0.2, -0.1), [SQ], 0.8, math.pi / 4, math.pi / 6, 0),
        ("L-shape, p in the notch, wedge to the corner", (1.4, 1.4), [LS], 1.0, math.pi * 1.25, math.pi / 6, 0),
        ("L-shape, p right of it, wedge straddling the angle ±pi", (2.3, 0.5), [LS], 1.0, math.pi, math.pi / 6, 0),
        ("tank, fluid at (0.1,0.5) near the left wall, background", (0.1, 0.5), [TANK_FLUID_LOOP], 0.5, math.pi, math.pi / 6, 1),
        ("tank, fluid corner (0.1,0.1), wedge toward the corner", (0.1, 0.1), [TANK_FLUID_LOOP], 0.5, math.pi * 1.25, math.pi / 6, 1),
        ("tank, fluid corner, full disk", (0.1, 0.1), [TANK_FLUID_LOOP], 0.5, 0.0, math.pi, 1),
        ("tank, mid-fluid, no wall in range", (0.5, 0.5), [TANK_FLUID_LOOP], 0.3, 0.0, math.pi / 6, 1),
    ]
    for name, p, loops, H, th, al, bg in cases:
        a, b = cone_area(p, loops, H, th, al, bg), brute(p, loops, H, th, al, bg)
        print("%-62s closed %.10f   brute %.10f   diff %.1e" % (name, a, b, abs(a - b)))

    rng = np.random.default_rng(11)
    worst = 0.0
    for t in range(300):
        loops, bg = ([SQ], 0) if t % 3 == 0 else ([LS], 0) if t % 3 == 1 else ([TANK_FLUID_LOOP], 1)
        p = rng.uniform(-0.6, 2.4, 2) if bg == 0 else rng.uniform(-0.1, 1.1, 2)
        H, th, al = rng.uniform(0.2, 1.2), rng.uniform(-math.pi, math.pi), math.pi / 6 if t % 4 else math.pi
        d = abs(cone_area(p, loops, H, th, al, bg) - brute(p, loops, H, th, al, bg, nr=600, nphi=1200))
        worst = max(worst, d)
    print("random sweep (300 cases, square / L / tank-with-background, random p, H, axis, wedge pi/6 or full): worst |closed - brute| = %.1e" % worst)
