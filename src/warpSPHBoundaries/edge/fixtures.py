"""Golden fixtures for the 2D edge reductions (stage-1 mpmath -> every later backend).

    python scripts/make_fixtures.py edge [out.json]        regenerate tests/fixtures/edge2d_golden.json

File layout (JSON):

  meta   : format version, dps, number of printed digits, alpha lists, conventions
  cases  : list of
      id, tags[], kind ("element" | "mesh"),
      polygon  : [[x,y],...] exact rationals as "a/b" strings  (element)     ccw or cw
      elements : [polygon, ...]                                  (mesh)
      x        : [x,y] exact rationals                           evaluation point
      h        : support radius (exact rational string)
      where    : classification of x w.r.t. the polygon (inside|outside|edge|vertex), element only
      kernels  : {kernel -> {value, grad[2], moments{"a,b"}, moment_grad{"a,b"->[2]}}}
                 mesh cases: per-element lists AND the sums
      expect   : exact / independent expectations (mesh sums = disk moments as exact rationals,
                 half-plane lambda_2 from the PLAN track, ...)

All numbers are decimal strings with 40 significant digits.  Values are in PHYSICAL units for the
given h: value ~ h^0, grad ~ h^-1, moment_alpha ~ h^|alpha|, moment_grad ~ h^(|alpha|-1).
Tolerances per backend: docs/backends-and-verification.md.
"""
import json
import time
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

from . import core, geometry
from .. import paths
from .kernels import KERNELS, disk_moment
from .mpq import mpq

DPS = 50          # computation precision
DIGITS = 40       # printed significant digits
KERNEL_NAMES = ["cubic", "w2", "w4", "w6"]
MOMENTS = [(a, k - a) for k in range(1, 5) for a in range(k + 1)]          # k = 1..4
MOMENT_GRADS = [(0, 0)] + [(a, k - a) for k in range(1, 3) for a in range(k + 1)]   # k = 0..2

DEFAULT_PATH = paths.fixtures_dir() / "edge2d_golden.json"


def fs(f: F) -> str:
    f = F(f)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def ns(v) -> str:
    return mp.nstr(v, DIGITS, strip_zeros=False) if v != 0 else "0"


def pt(p):
    return [fs(p[0]), fs(p[1])]


def poly_json(P):
    return [pt(p) for p in P]


# ----------------------------------------------------------------------- evaluation
def evaluate_element(P, x, h, kernels=KERNEL_NAMES):
    out = {}
    with mp.workdps(DPS + core.GUARD):
        prep = geometry.prepare(P, x, h)
        where = prep.where
        for k in kernels:
            ent = {}
            ent["value"] = ns(core.value(P, x, k, h, dps=DPS, prepared=prep))
            g = core.gradient(P, x, k, h, dps=DPS, prepared=prep)
            ent["grad"] = [ns(g[0]), ns(g[1])]
            ent["moments"] = {f"{a},{b}": ns(core.moment(P, x, k, (a, b), h, dps=DPS, prepared=prep)) for a, b in MOMENTS}
            ent["moment_grad"] = {}
            for a, b in MOMENT_GRADS:
                mg = core.moment_gradient(P, x, k, (a, b), h, dps=DPS, prepared=prep)
                ent["moment_grad"][f"{a},{b}"] = [ns(mg[0]), ns(mg[1])]
            out[k] = ent
    return where, out


def _num(s):
    return mp.mpf(s)


def sum_entries(entries):
    """sum of per-element kernel entries (mesh totals), as strings."""
    tot = {}
    for k in entries[0]:
        t = {"value": ns(sum(_num(e[k]["value"]) for e in entries)),
             "grad": [ns(sum(_num(e[k]["grad"][i]) for e in entries)) for i in range(2)],
             "moments": {key: ns(sum(_num(e[k]["moments"][key]) for e in entries)) for key in entries[0][k]["moments"]},
             "moment_grad": {key: [ns(sum(_num(e[k]["moment_grad"][key][i]) for e in entries)) for i in range(2)]
                             for key in entries[0][k]["moment_grad"]}}
        tot[k] = t
    return tot


def disk_expect(h):
    """EXACT full-support moments (rationals) for every kernel and alpha: m_alpha = h^k * disk_moment."""
    out = {}
    for k in KERNEL_NAMES:
        d = {"0,0": "1"}
        for a, b in MOMENTS:
            d[f"{a},{b}"] = fs(disk_moment(KERNELS[k], (a, b)) * F(h) ** (a + b))
        out[k] = d
    return out


# ------------------------------------------------------------------------ geometry
def on_edge(T, i, t):
    p, q = T[i], T[(i + 1) % len(T)]
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def off_edge(T, i, t, eps, sign=1):
    p = on_edge(T, i, t)
    q, r = T[i], T[(i + 1) % len(T)]
    dx, dy = r[0] - q[0], r[1] - q[1]
    return (p[0] - sign * eps * dy, p[1] + sign * eps * dx)


def tiling(a, n):
    tris = []
    xs = [-a + 2 * a * F(i, n) for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            p00, p10, p01, p11 = (xs[i], xs[j]), (xs[i + 1], xs[j]), (xs[i], xs[j + 1]), (xs[i + 1], xs[j + 1])
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
    return tris


def big_tri(d):
    return [(F(-40), F(d)), (F(40), F(d)), (F(0), F(80))]


T0 = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]


def build_cases():
    import random
    rng = random.Random(20261001)
    cases = []

    def add(id_, tags, P, x, h=F(1), expect=None):
        cases.append(dict(kind="element", id=id_, tags=tags, P=P, x=x, h=F(h), expect=expect or {}))

    def rf(lo=-1, hi=1, den=1000):
        return F(rng.randint(int(lo * den), int(hi * den)), den)

    def rtri(scale=1):
        while True:
            T = [(rf(-scale, scale), rf(-scale, scale)) for _ in range(3)]
            A2 = (T[1][0] - T[0][0]) * (T[2][1] - T[0][1]) - (T[1][1] - T[0][1]) * (T[2][0] - T[0][0])
            if abs(A2) > F(scale * scale, 20):
                return T

    # 1. generic
    for i in range(12):
        add(f"generic-{i:02d}", ["generic"], rtri(), (rf(), rf()))
    # 2. h != 1 (physical units)
    for i, h in enumerate([F(7, 5), F(1, 3), F(2)]):
        T = [(a * h, b * h) for a, b in T0]
        add(f"hscaled-{i}", ["h_scaled"], T, (F(1, 20) * h, F(1, 10) * h), h)
    # 3. x on an edge (interior), vertex, edge line outside segment
    for i, t in [(0, F(2, 5)), (1, F(1, 2)), (2, F(1, 10**6)), (0, F(999999, 10**6))]:
        add(f"x_on_edge-{i}-{t.numerator}", ["x_on_edge"], T0, on_edge(T0, i, t))
    for i, v in enumerate(T0):
        add(f"x_at_vertex-{i}", ["x_at_vertex"], T0, v)
    obtuse = [(F(0), F(0)), (F(1, 2), F(1, 10)), (F(-1, 2), F(1, 10))]
    acute = [(F(0), F(0)), (F(1, 2), F(0)), (F(1, 2), F(1, 1000))]
    add("x_at_vertex-obtuse", ["x_at_vertex"], obtuse, obtuse[0])
    add("x_at_vertex-acute", ["x_at_vertex"], acute, acute[0])
    Tq = [(F(0), F(0)), (F(1, 2), F(0)), (F(0), F(1, 2))]
    for j, x in enumerate([(F(-1, 5), F(0)), (F(7, 10), F(0)), (F(0), F(-1, 3)), (F(0), F(9, 10))]):
        add(f"x_on_edge_line-{j}", ["x_on_edge_line"], Tq, x)
    # 4. z -> 0
    for e in (3, 10, 20, 30, 40):
        for sign in (1, -1):
            add(f"z_tiny-1e-{e}-{'in' if sign > 0 else 'out'}", ["z_to_0"], T0, off_edge(T0, 0, F(2, 5), F(1, 10**e), sign))
    # 5. tiny chords
    z = F(1) - F(1, 10**24)
    add("tiny_chord-grazing", ["tiny_chord"], [(F(-5), z), (F(5), z), (F(0), F(9))], (F(0), F(0)))
    add("tiny_chord-edge_end-1e-13", ["tiny_chord"],
        [(F(-2), F(3, 5)), (-F(4, 5) + F(1, 10**13), F(3, 5)), (F(-1), F(4))], (F(0), F(0)))
    add("tiny_chord-edge_end-1e-6", ["tiny_chord"],
        [(F(-2), F(3, 5)), (-F(4, 5) + F(1, 10**6), F(3, 5)), (F(-1), F(4))], (F(0), F(0)))
    add("tiny_chord-knot_circle-1e-13", ["tiny_chord", "cubic_knot"],
        [(F(-2), F(3, 10)), (-F(2, 5) + F(1, 10**13), F(3, 10)), (F(-1), F(4))], (F(0), F(0)))
    add("tiny_chord-both_ends", ["tiny_chord"],
        [(-F(4, 5) - F(1, 10**12), F(3, 5)), (-F(4, 5) + F(1, 10**12), F(3, 5)), (F(-1), F(3))], (F(0), F(0)))
    # 6. elements << h  (L_T / h = 1e-3 ... 1)
    base = [(F(-3), F(-2)), (F(6), F(-1)), (F(1), F(7))]
    for sc in (F(1, 10**6), F(1, 10**3), F(1, 10**2), F(1, 10), F(1, 2), F(1)):
        T = [(a * sc / 10, b * sc / 10) for a, b in base]
        tag = f"1e{len(str(sc.denominator)) - 1}" if sc != 1 else "1"
        add(f"small_element-{tag}-x_near", ["small_element"], T, (F(1, 5), F(-1, 7)))
        add(f"small_element-{tag}-x_vertex", ["small_element", "x_at_vertex"], T, T[0])
        add(f"small_element-{tag}-x_inside", ["small_element"], T, (sum(p[0] for p in T) / 3, sum(p[1] for p in T) / 3))
    # 7. half plane: edge at distance d from x (d < 0: x inside the solid)
    for d in [F(0), F(1, 10**12), F(1, 1000), F(1, 10), F(3, 10), F(1, 2), F(1, 2) + F(1, 10**6),
              F(7, 10), F(9, 10), F(1) - F(1, 10**12), F(1), F(3, 2),
              F(-1, 1000), F(-1, 10), F(-1, 2), F(-9, 10), F(-1), F(-3, 2)]:
        add(f"half_plane-d={fs(d)}", ["half_plane"], big_tri(d), (F(0), F(0)), expect={"half_plane_d": fs(d)})
    # 8. engulfing / covering single element: the whole support inside one big triangle
    add("engulf-big_triangle", ["engulf"], [(F(-5), F(-5)), (F(6), F(-5)), (F(0), F(7))], (F(1, 10), F(1, 7)),
        expect={"covers_support": True})
    # 9. non-convex (L-shape), clockwise input
    L = [(F(0), F(0)), (F(1), F(0)), (F(1), F(1, 2)), (F(1, 2), F(1, 2)), (F(1, 2), F(1)), (F(0), F(1))]
    for j, x in enumerate([(F(1, 4), F(1, 4)), (F(3, 4), F(3, 4)), (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2)), (F(-1, 5), F(1, 2))]):
        add(f"nonconvex-{j}", ["nonconvex"], L, x)
    add("clockwise-0", ["clockwise"], T0[::-1], (F(1, 20), F(1, 10)))
    add("far-outside_support", ["outside_support"], [(F(5), F(5)), (F(6), F(5)), (F(5), F(6))], (F(0), F(0)))
    # 10. covering meshes (exact disk moments)
    meshes = []
    for j, x in enumerate([(F(1, 7), F(-1, 5)), (F(0), F(0)), (F(3, 4), F(0))]):
        meshes.append(dict(kind="mesh", id=f"mesh-cover-{j}", tags=["covering_mesh"], elements=tiling(F(2), 2), x=x, h=F(1)))
    meshes.append(dict(kind="mesh", id="mesh-cover-h3_2", tags=["covering_mesh", "h_scaled"],
                       elements=[[(a * F(3, 2), b * F(3, 2)) for a, b in T] for T in tiling(F(2), 2)],
                       x=(F(1, 7) * F(3, 2), F(-1, 5) * F(3, 2)), h=F(3, 2)))
    fan_R = F(2)
    ring = [(fan_R, F(0)), (fan_R, fan_R), (F(0), fan_R), (-fan_R, fan_R), (-fan_R, F(0)), (-fan_R, -fan_R), (F(0), -fan_R), (fan_R, -fan_R)]
    meshes.append(dict(kind="mesh", id="mesh-fan-vertex", tags=["covering_mesh", "x_at_vertex"],
                       elements=[[(F(0), F(0)), ring[i], ring[(i + 1) % 8]] for i in range(8)], x=(F(0), F(0)), h=F(1)))
    return cases + meshes


def generate(path=DEFAULT_PATH, verbose=True):
    with mp.workdps(DPS):               # EVERYTHING (mpq conversions of expectations, mesh sums) at DPS digits
        return _generate(path, verbose)


def _generate(path, verbose):
    t0 = time.time()
    out = []
    for c in build_cases():
        if c["kind"] == "element":
            where, ker = evaluate_element(c["P"], c["x"], c["h"])
            rec = dict(id=c["id"], tags=c["tags"], kind="element", polygon=poly_json(c["P"]), x=pt(c["x"]),
                       h=fs(c["h"]), where=where, kernels=ker, expect={})
            if "half_plane_d" in c["expect"]:
                from ..curvbound import planar2d
                d = F(c["expect"]["half_plane_d"])
                lam = {}
                for k in KERNEL_NAMES:
                    if d >= 1 or d <= -1:
                        lam[k] = "0" if d >= 1 else "1"
                    elif d == 0:
                        lam[k] = "1/2"
                    elif d > 0:
                        lam[k] = ns(planar2d(k, mpq(d), dps=60))
                    else:
                        lam[k] = ns(1 - planar2d(k, mpq(-d), dps=60))
                rec["expect"]["half_plane_lambda2"] = lam
            if c["expect"].get("covers_support"):
                rec["expect"]["moments_exact"] = disk_expect(c["h"])
        else:
            per, polys = [], []
            for T in c["elements"]:
                _, ker = evaluate_element(T, c["x"], c["h"])
                per.append(ker)
                polys.append(poly_json(T))
            rec = dict(id=c["id"], tags=c["tags"], kind="mesh", elements=polys, x=pt(c["x"]), h=fs(c["h"]),
                       per_element=per, kernels=sum_entries(per),
                       expect={"moments_exact": disk_expect(c["h"]), "value_exact": "1", "grad_exact": ["0", "0"]})
        out.append(rec)
        if verbose:
            print(f"  {rec['id']:42s} {time.time() - t0:6.1f}s", flush=True)
    meta = dict(
        format="edge2d-golden-v1", dps=DPS, digits=DIGITS, guard=core.GUARD,
        kernels=KERNEL_NAMES, moments=[list(a) for a in MOMENTS], moment_grads=[list(a) for a in MOMENT_GRADS],
        conventions=("y = x' - x; value = int_T W dA; grad = grad_x value; moments[a,b] = int_T y1^a y2^b W dA; "
                     "moment_grad[a,b] = grad_x moments[a,b]; indicator at an edge = 1/2, at a vertex = interior angle/2pi; "
                     "values in physical units for support radius h"),
        tolerance_hint="reference accurate to ~1e-35 absolute; independent polar oracle agrees to <1e-30 (tests/edge)",
        generator="python scripts/make_fixtures.py edge")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta=meta, cases=out), indent=1) + "\n")
    if verbose:
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(out)} cases, {time.time() - t0:.0f}s)")
    return path


def load(path=DEFAULT_PATH):
    return json.loads(Path(path).read_text())
