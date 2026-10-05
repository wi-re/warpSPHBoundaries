"""Polygon / edge preparation for the edge-reduction machinery (2D).

Conventions: docs/notation.md.  Everything is expressed relative to the
evaluation point x (x at the origin) and in units of the support radius h:

    w_i = (v_i - x) / h          (exact Fractions)

Per edge e = (p, q):  d = q - p,  ell = |d|,  t = d / ell,  n = (t_y, -t_x)
(outward normal of a COUNTER-CLOCKWISE polygon; clockwise input is reversed first),

    z  = n . p       = (p_x d_y - p_y d_x) / ell     sign is EXACT (rational numerator)
    s0 = t . p       = (d . p) / ell
    s1 = t . q       = (d . q) / ell

so `sign(z_e) = sign(cross(d, x - p))` is the orientation predicate.  The
indicator of x in T is computed from the same exact arithmetic
(`indicator_signs` for convex polygons: all z_e >= 0; `indicator_winding` for
general simple polygons); at the boundary the convention is

    x in the open interior of an edge : 1/2
    x at a vertex                     : interior angle / (2 pi)

and an edge with z_e == 0 contributes 0 to every atan term (average of the
one-sided limits).  Verified: maple/11_*.mpl (vertex wedges), tests/edge.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

import mpmath as mp

from .mpq import mpq, sgn, to_frac


@dataclass
class Edge:
    n: Tuple[mp.mpf, mp.mpf]     # outward unit normal
    t: Tuple[mp.mpf, mp.mpf]     # unit tangent (p -> q)
    z: mp.mpf                    # n.(p - x)/h, sign exact (0 exactly when x on the edge line)
    s0: mp.mpf
    s1: mp.mpf
    zsign: int                   # exact sign(z)
    p: Tuple[mp.mpf, mp.mpf]     # endpoints relative to x, in units of h
    q: Tuple[mp.mpf, mp.mpf]


@dataclass
class Prepared:
    edges: List[Edge]
    indicator: mp.mpf            # 1, 0, 1/2, or interior angle/(2 pi)
    where: str                   # 'inside' | 'outside' | 'edge' | 'vertex'
    h: Fraction
    verts_rel: List[Tuple[Fraction, Fraction]]   # exact, ccw, relative to x, units of h
    dps: int


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def signed_area2(vs) -> Fraction:
    n = len(vs)
    return sum(_cross(vs[i], vs[(i + 1) % n]) for i in range(n))


def _on_segment(p, q) -> bool:
    """origin on the closed segment pq (exact)."""
    if _cross(p, q) != 0:
        return False
    return p[0] * q[0] + p[1] * q[1] <= 0


def locate(vs):
    """Exact classification of the origin w.r.t. the polygon `vs` (ccw, relative coords).
    Returns ('vertex', i) | ('edge', i) | ('inside', None) | ('outside', None)."""
    n = len(vs)
    for i, v in enumerate(vs):
        if v[0] == 0 and v[1] == 0:
            return "vertex", i
    for i in range(n):
        if _on_segment(vs[i], vs[(i + 1) % n]):
            return "edge", i
    crossings = 0
    for i in range(n):
        p, q = vs[i], vs[(i + 1) % n]
        if (p[1] > 0) != (q[1] > 0):
            # x-coordinate of the crossing with the ray y = 0 going to +x:  px + (0-py)(qx-px)/(qy-py)
            xc = p[0] + (0 - p[1]) * (q[0] - p[0]) / (q[1] - p[1])
            if xc > 0:
                crossings += 1
    return ("inside", None) if crossings % 2 == 1 else ("outside", None)


def interior_angle(vs, i) -> mp.mpf:
    """interior angle (0, 2 pi) of the ccw polygon at vertex i."""
    n = len(vs)
    v = vs[i]
    a = (vs[(i + 1) % n][0] - v[0], vs[(i + 1) % n][1] - v[1])     # to next
    b = (vs[(i - 1) % n][0] - v[0], vs[(i - 1) % n][1] - v[1])     # to previous
    ang = mp.atan2(mpq(_cross(a, b)), mpq(a[0] * b[0] + a[1] * b[1]))
    if ang <= 0:
        ang += 2 * mp.pi
    return ang


def prepare(verts, x, h=1, dps: Optional[int] = None) -> Prepared:
    """Build the edge records.  `verts`: sequence of (x, y) coordinate-likes (int,
    Fraction, 'a/b' strings, mpf); `x`: evaluation point; `h`: support radius.
    The working precision is the current mp.dps unless `dps` is given."""
    if dps is not None:
        mp.mp.dps = dps
    h = to_frac(h)
    xf = (to_frac(x[0]), to_frac(x[1]))
    vs = [((to_frac(v[0]) - xf[0]) / h, (to_frac(v[1]) - xf[1]) / h) for v in verts]
    if len(vs) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    A2 = signed_area2(vs)
    if A2 == 0:
        raise ValueError("degenerate (zero-area) polygon")
    if A2 < 0:
        vs = vs[::-1]
    n = len(vs)
    edges = []
    for i in range(n):
        p, q = vs[i], vs[(i + 1) % n]
        d = (q[0] - p[0], q[1] - p[1])
        l2 = d[0] * d[0] + d[1] * d[1]
        ell = mp.sqrt(mpq(l2))
        znum = p[0] * d[1] - p[1] * d[0]        # n.p * ell   (exact)
        s0num = d[0] * p[0] + d[1] * p[1]
        s1num = d[0] * q[0] + d[1] * q[1]
        t = (mpq(d[0]) / ell, mpq(d[1]) / ell)
        nrm = (t[1], -t[0])
        edges.append(Edge(nrm, t, mpq(znum) / ell, mpq(s0num) / ell, mpq(s1num) / ell, sgn(znum),
                          (mpq(p[0]), mpq(p[1])), (mpq(q[0]), mpq(q[1]))))
    where, idx = locate(vs)
    if where == "inside":
        ind = mp.mpf(1)
    elif where == "outside":
        ind = mp.mpf(0)
    elif where == "edge":
        ind = mp.mpf(1) / 2
    else:
        ind = interior_angle(vs, idx) / (2 * mp.pi)
    return Prepared(edges, ind, where, h, vs, mp.mp.dps)


def indicator_signs(P: Prepared) -> mp.mpf:
    """Indicator from the z_e signs alone (valid for CONVEX polygons, e.g. triangles):
    any z_e < 0 -> outside; all > 0 -> 1; one zero -> 1/2; two zeros -> vertex angle / 2 pi."""
    if any(e.zsign < 0 for e in P.edges):
        return mp.mpf(0)
    zeros = [i for i, e in enumerate(P.edges) if e.zsign == 0]
    if not zeros:
        return mp.mpf(1)
    if len(zeros) == 1:
        return mp.mpf(1) / 2
    # vertex: shared vertex of two consecutive zero edges a, b = a + 1 (mod n)
    n = len(P.edges)
    for a in zeros:
        b = (a + 1) % n
        if b in zeros:
            return interior_angle(P.verts_rel, b) / (2 * mp.pi)
    raise ValueError("inconsistent zero pattern for a polygon (x on two non-adjacent edge lines)")


def chord(e: Edge, R) -> Optional[Tuple[mp.mpf, mp.mpf]]:
    """Clip the edge to the support disk of radius R (units of h): chord [lo, hi] in s, or None."""
    az = abs(e.z)
    if az >= R:
        return None
    L = mp.sqrt((R - az) * (R + az))        # stable sqrt(R^2 - z^2)
    lo = max(e.s0, -L)
    hi = min(e.s1, L)
    if lo >= hi:
        return None
    return lo, hi
