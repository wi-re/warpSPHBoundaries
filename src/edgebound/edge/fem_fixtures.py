"""Golden fixtures for the FEM nodal weights (stage 1 exact -> numpy/torch/warp).

    python scripts/make_fixtures.py fem      # regenerates tests/fixtures/edge2d_fem_golden.json

cases: {id, tags, polygon (exact "a/b" strings), x, h, where, p-> kernel -> {w[n_nodes], G[n_nodes][2]}}   (physical units for h)
mesh : covering mesh with a continuous polynomial field of degree <= 3: per-element nodal values for each p are the exact
       P_p nodal values of the TRUNCATION of the field to degree p; expect.exact = exact rational  sum_alpha a_alpha disk_moment
       (and the gradient integral = disk integral of grad A) for the whole mesh -> reproduction test for every backend.
"""
import json
import time
from fractions import Fraction as F

import mpmath as mp

from . import fem
from .. import paths
from .fixtures import DPS, DIGITS, fs, ns, on_edge, off_edge, tiling, pt, poly_json
from .kernels import disk_moment, kernel as get_kernel

PATH = paths.fixtures_dir() / "edge2d_fem_golden.json"
KERNELS = ["cubic", "w2", "w4", "w6"]
A3 = {(0, 0): F(1), (1, 0): F(2), (0, 1): F(-3), (1, 1): F(1), (2, 0): F(1, 2), (0, 2): F(-1, 4), (0, 3): F(-1), (2, 1): F(1), (3, 0): F(1, 3), (1, 2): F(-2)}
T0 = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]


def trunc(A, p):
    return {k: v for k, v in A.items() if sum(k) <= p}


def cases():
    out = []
    add = lambda id_, tags, T, x, h=F(1): out.append(dict(id=id_, tags=tags, T=T, x=x, h=F(h)))
    add("generic-0", ["generic"], T0, (F(1, 20), F(1, 10)))
    add("generic-1", ["generic"], T0, (F(-1, 2), F(1, 2)))
    add("generic-2", ["generic"], [(F(1, 5), F(-3, 10)), (F(7, 10), F(1, 10)), (F(-1, 5), F(1, 2))], (F(3, 10), F(-1, 5)))
    add("hscaled", ["h_scaled"], [(a * F(3, 2), b * F(3, 2)) for a, b in T0], (F(1, 20) * F(3, 2), F(1, 10) * F(3, 2)), F(3, 2))
    add("x_on_edge", ["x_on_edge"], T0, on_edge(T0, 0, F(2, 5)))
    add("x_at_vertex", ["x_at_vertex"], T0, T0[1])
    add("z_tiny", ["z_to_0"], T0, off_edge(T0, 0, F(2, 5), F(1, 10**10), 1))
    base = [(F(-3), F(-2)), (F(6), F(-1)), (F(1), F(7))]
    for tag, sc in [("1e-3", F(1, 10**3)), ("1e-6", F(1, 10**6))]:
        T = [(a * sc / 10, b * sc / 10) for a, b in base]
        add(f"small-x_inside-{tag}", ["small_element", "x_inside"], T, (sum(p[0] for p in T) / 3, sum(p[1] for p in T) / 3))
        add(f"small-x_vertex-{tag}", ["small_element", "x_at_vertex"], T, T[0])
        for d_label, d in [("0.3", F(3, 10)), ("0.6", F(3, 5)), ("0.95", F(19, 20))]:
            add(f"small-far-d{d_label}-{tag}", ["small_element", "far"], [(a + d, b) for a, b in T], (F(0), F(0)))
    add("moderate-near", ["small_element"], [(a * F(3, 10) + F(3, 5), b * F(3, 10)) for a, b in T0], (F(0), F(0)))
    add("half_plane-d0.4", ["half_plane"], [(F(-40), F(2, 5)), (F(40), F(2, 5)), (F(0), F(80))], (F(0), F(0)))
    return out


def evaluate(c):
    res = {}
    for p in (0, 1, 2, 3):
        res[str(p)] = {}
        for k in KERNELS:
            w, G = fem.weights(c["T"], c["x"], k, p, c["h"], dps=DPS, grad=True)
            res[str(p)][k] = dict(w=[ns(v) for v in w], G=[[ns(g[0]), ns(g[1])] for g in G])
    return res


def build(verbose=True):
    t0 = time.time()
    out = []
    with mp.workdps(DPS):
        for c in cases():
            prep = fem.geometry.prepare(c["T"], c["x"], c["h"])
            out.append(dict(id=c["id"], tags=c["tags"], polygon=poly_json(c["T"]), x=pt(c["x"]), h=fs(c["h"]), where=prep.where,
                            weights=evaluate(c)))
            if verbose:
                print(f"  {c['id']:34s} {time.time() - t0:6.1f}s", flush=True)
        # covering mesh with a continuous polynomial field (exact reproduction)
        tris = tiling(F(2), 2)
        x = (F(1, 7), F(-1, 5))
        exact = {}
        for k in KERNELS:
            for p in (1, 2, 3):
                A = trunc(A3, p)
                dAx, dAy = fem.poly_grad(A)
                ref = lambda B: sum(c * disk_moment(get_kernel(k), al) for al, c in fem.poly_shift(B, x).items())
                exact[f"{k},{p}"] = dict(value=fs(ref(A)), grad=[fs(ref(dAx)), fs(ref(dAy))])
        mesh = dict(id="mesh-cover-poly", tags=["covering_mesh"], elements=[poly_json(T) for T in tris], x=pt(x), h="1",
                    field={f"{i},{j}": fs(c) for (i, j), c in A3.items()},
                    nodal={str(p): [[fs(v) for v in fem.nodal_values(T, p, trunc(A3, p))] for T in tris] for p in (1, 2, 3)},
                    expect=exact)
    data = dict(meta=dict(format="edge2d-fem-golden-v1", dps=DPS, digits=DIGITS, kernels=KERNELS, p=[0, 1, 2, 3],
                          node_order="vertices, then per edge (0,1),(1,2),(2,0) the nodes at 1/3,2/3 from the first vertex (P2: midpoint), then centroid",
                          units="w dimensionless; G ~ 1/h; geometry physical with support radius h"),
                cases=out, mesh=mesh)
    PATH.write_text(json.dumps(data, indent=1) + "\n")
    if verbose:
        print(f"wrote {PATH} ({PATH.stat().st_size / 1e6:.2f} MB, {len(out)} cases, {time.time() - t0:.0f}s)")
