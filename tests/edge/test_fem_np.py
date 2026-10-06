"""FEM nodal weights, stage 2/3 (numpy): golden fixtures, hybrid far-field, conditioning (check 7)."""
import json
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from warpSPHBoundaries.edge import fem, np_fem
from warpSPHBoundaries.edge.fem_fixtures import PATH

DATA = json.loads(Path(PATH).read_text())
CASES = DATA["cases"]
KERNELS = DATA["meta"]["kernels"]


def rel_of(c, dt=np.float64):
    x = (F(c["x"][0]), F(c["x"][1]))
    rel = np.array([[float(F(a) - x[0]), float(F(b) - x[1])] for a, b in c["polygon"]])
    return rel.astype(dt)


def ref_w(c, p, k, rel=None):
    """reference weights for the (possibly rounded) float inputs: recomputed exactly (stage 1) if rel is given."""
    if rel is None:
        e = DATA_ENTRY(c, p, k)
        return np.array([float(mp.mpf(s)) for s in e["w"]]), np.array([[float(mp.mpf(s)) for s in g] for g in e["G"]])
    T = [(F(float(a)), F(float(b))) for a, b in rel.astype(np.float64)]
    w, G = fem.weights(T, (F(0), F(0)), k, p, F(c["h"]), grad=True)
    return np.array([float(v) for v in w]), np.array([[float(g[0]), float(g[1])] for g in G])


def DATA_ENTRY(c, p, k):
    return c["weights"][str(p)][k]


def rel_err(a, ref):
    s = np.abs(ref).max()
    return np.abs(a - ref).max() / s if s > 0 else np.abs(a).max()


@pytest.mark.parametrize("p", [0, 1, 2, 3])
@pytest.mark.parametrize("k", KERNELS)
def test_fixtures_float64_hybrid(p, k):
    """every FEM fixture (x inside / on edge / vertex / z->0 / tiny elements / far / rim / half-plane), float64 hybrid."""
    worst = 0
    for c in CASES:
        h = float(F(c["h"]))
        rel = rel_of(c)
        w, G = np_fem.weights_hybrid(rel[None], np.zeros((1, 2)), k, p, h, grad=True)
        rw, rG = ref_w(c, p, k)
        e = rel_err(w[0], rw)
        eg = np.abs(G[0] * h - rG * h).max() / max(np.abs(rw).max(), 1e-300)          # G error relative to the weight scale ~ int|grad W| bound
        worst = max(worst, e, eg)
        if c["id"] in ("small-far-d0.95-1e-3", "small-far-d0.95-1e-6"):
            continue                               # rim-straddling: weights ~ 1e-9 absolute, relative metric meaningless
        assert e < 1e-10 and eg < 1e-8, (c["id"], e, eg)
    assert worst < 1e-5


@pytest.mark.parametrize("p", [1, 2, 3])
def test_float32_hybrid_stable_meets_float32_tolerance(p):
    """float32 inputs, hybrid + stable: relative error at float32 epsilon level for every case with L_T/h >= 1e-3."""
    for c in CASES:
        if "far" in c["tags"] and "1e-6" in c["id"]:
            continue                               # element not representable in float32 at this distance
        rel = rel_of(c, np.float32)
        h = np.float32(float(F(c["h"])))
        w = np_fem.weights_hybrid(rel[None], np.zeros((1, 2), np.float32), "w4", p, h, dtype=np.float32, stable=(8, 6))[0]
        rw, _ = ref_w(c, p, "w4", rel)                  # exact weights of the ROUNDED float32 inputs
        if np.abs(rw).max() < 1e-12:
            continue
        assert rel_err(w.astype(np.float64), rw) < 2e-4, (c["id"], rel_err(w.astype(np.float64), rw))


def test_conditioning_check7_regression():
    """documented behaviour: the plain closed form loses ~(d/L)^(p+1); the hybrid does not."""
    c = [c for c in CASES if c["id"] == "small-far-d0.3-1e-3"][0]
    rel = rel_of(c)
    rw, _ = ref_w(c, 3, "w4")
    closed = np_fem.weights(rel[None], np.zeros((1, 2)), "w4", 3)[0]
    hybrid = np_fem.weights_hybrid(rel[None], np.zeros((1, 2)), "w4", 3)[0]
    assert rel_err(closed, rw) > 1e-4 and rel_err(hybrid, rw) < 1e-12
    # x inside the tiny element: the inner-potential variant keeps the closed form at machine precision
    c = [c for c in CASES if c["id"] == "small-x_inside-1e-6"][0]
    rw, _ = ref_w(c, 3, "w4")
    assert rel_err(np_fem.weights(rel_of(c)[None], np.zeros((1, 2)), "w4", 3)[0], rw) < 1e-12


def test_covering_mesh_polynomial_reproduction_exact():
    m = DATA["mesh"]
    x = np.array([[float(F(m["x"][0])), float(F(m["x"][1]))]])
    for k in KERNELS:
        for p in (1, 2, 3):
            tot = 0.0
            gx = gy = 0.0
            for poly, nodal in zip(m["elements"], m["nodal"][str(p)]):
                rel = np.array([[float(F(a) - F(m["x"][0])), float(F(b) - F(m["x"][1]))] for a, b in poly])
                w, G = np_fem.weights_hybrid(rel[None], np.zeros((1, 2)), k, p, 1.0, grad=True)
                A = np.array([float(F(v)) for v in nodal])
                tot += (A * w[0]).sum()
                gx += (A * G[0, :, 0]).sum()
                gy += (A * G[0, :, 1]).sum()
            ex = m["expect"][f"{k},{p}"]
            assert abs(tot - float(F(ex["value"]))) < 1e-12
            assert abs(gx - float(F(ex["grad"][0]))) < 1e-11 and abs(gy - float(F(ex["grad"][1]))) < 1e-11


def test_stage1_matches_fixture_regeneration():
    """spot regeneration of two fixture entries (stage 1 is the source of truth)."""
    for cid in ("generic-0", "x_at_vertex"):
        c = [c for c in CASES if c["id"] == cid][0]
        T = [(F(a), F(b)) for a, b in c["polygon"]]
        w = fem.weights(T, (F(c["x"][0]), F(c["x"][1])), "w2", 2, F(c["h"]))
        with mp.workdps(40):
            for a, b in zip(w, c["weights"]["2"]["w2"]["w"]):
                assert abs(a - mp.mpf(b)) < mp.mpf(10) ** -35
