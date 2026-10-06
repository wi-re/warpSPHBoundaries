"""Stage 2 (numpy, vectorised, branch-free) against the golden fixtures and stage 1."""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

import warpSPHBoundaries as eb
from warpSPHBoundaries.edge import fixtures as fx
from warpSPHBoundaries.edge import np2d

DATA = fx.load()
ELEMENTS = [c for c in DATA["cases"] if c["kind"] == "element" and len(c["polygon"]) == 3]
MESHES = [c for c in DATA["cases"] if c["kind"] == "mesh"]
KERNELS = ["cubic", "w2", "w4", "w6"]
MOMS = [tuple(map(int, k.split(","))) for k in DATA["cases"][0]["kernels"]["cubic"]["moments"]]
MGS = [tuple(map(int, k.split(","))) for k in DATA["cases"][0]["kernels"]["cubic"]["moment_grad"]]


def rel_geometry(recs):
    """exact (v - x) rounded ONCE to float64 (physical units); x := 0.  Mixed h per element."""
    rel, h = [], []
    for c in recs:
        x = (F(c["x"][0]), F(c["x"][1]))
        rel.append([[float(F(a) - x[0]), float(F(b) - x[1])] for a, b in c["polygon"]])
        h.append(float(F(c["h"])))
    return np.array(rel), np.zeros((len(recs), 2)), np.array(h)


def ref(c, k, what, key=None):
    e = c["kernels"][k]
    if what == "value":
        return float(mp.mpf(e["value"]))
    if what == "grad":
        return np.array([float(mp.mpf(s)) for s in e["grad"]])
    if what == "m":
        return float(mp.mpf(e["moments"][key]))
    return np.array([float(mp.mpf(s)) for s in e["moment_grad"][key]])


@pytest.mark.parametrize("k", KERNELS)
def test_fixtures_all_cases_one_vectorised_batch(k):
    """every convex-triangle fixture (mixed h, edge/vertex/z->0/tiny chords/tiny elements ...) in ONE call."""
    V, X, H = rel_geometry(ELEMENTS)
    val = np2d.value(V, X, k, H)
    gr = np2d.gradient(V, X, k, H)
    worst = {"value": 0, "grad": 0, "m": 0, "mg": 0}
    for i, c in enumerate(ELEMENTS):
        h = H[i]
        worst["value"] = max(worst["value"], abs(val[i] - ref(c, k, "value")))
        worst["grad"] = max(worst["grad"], np.abs(gr[i] - ref(c, k, "grad")).max() * h)      # grad ~ 1/h
    assert worst["value"] < 1e-12 and worst["grad"] < 1e-11, worst
    for al in MOMS:
        m = np2d.moment(V, X, k, al, H)
        for i, c in enumerate(ELEMENTS):
            worst["m"] = max(worst["m"], abs(m[i] - ref(c, k, "m", f"{al[0]},{al[1]}")) / H[i] ** sum(al))
    for al in MGS:
        mg = np2d.moment_gradient(V, X, k, al, H)
        for i, c in enumerate(ELEMENTS):
            worst["mg"] = max(worst["mg"], np.abs(mg[i] - ref(c, k, "mg", f"{al[0]},{al[1]}")).max() * H[i] ** (1 - sum(al)))
    assert worst["m"] < 1e-12 and worst["mg"] < 1e-11, worst


@pytest.mark.parametrize("k", KERNELS)
def test_covering_meshes(k):
    for c in MESHES:
        V, X, H = rel_geometry([dict(polygon=P, x=c["x"], h=c["h"]) for P in c["elements"]])
        assert abs(np2d.value(V, X, k, H).sum() - 1) < 1e-12
        assert np.abs(np2d.gradient(V, X, k, H).sum(0)).max() < 1e-11
        for al in [(1, 0), (1, 1), (2, 0), (0, 2), (3, 0), (2, 2), (0, 4)]:
            exact = float(F(c["expect"]["moments_exact"][k][f"{al[0]},{al[1]}"]))
            assert abs(np2d.moment(V, X, k, al, H).sum() - exact) < 1e-12 * float(F(c["h"])) ** sum(al)


def test_exact_kernel_weights():
    """the indicator weights are exact rationals; Wendland 1, cubic 8/7 and -1/7."""
    for k in ["w2", "w4", "w6"]:
        _, vals = np2d.compile_moment(k, (0, 0))
        assert {R: np2d.indicator_weight(P, R) for R, P in vals.items()} == {F(1): F(1)}
    _, vals = np2d.compile_moment("cubic", (0, 0))
    assert {R: np2d.indicator_weight(P, R) for R, P in vals.items()} == {F(1): F(8, 7), F(1, 2): F(-1, 7)}


def test_matches_stage1_random_batch():
    rng = np.random.default_rng(7)
    N = 40
    V = rng.uniform(-1, 1, (N, 3, 2))
    X = rng.uniform(-0.5, 0.5, (N, 2))
    H = rng.uniform(0.6, 1.4, N)
    for k in ["cubic", "w6"]:
        val = np2d.value(V, X, k, H)
        for i in range(0, N, 5):
            T = [(F(float(a)), F(float(b))) for a, b in V[i]]
            x = (F(float(X[i, 0])), F(float(X[i, 1])))
            with mp.workdps(40):
                s1 = eb.value(T, x, k, F(float(H[i])))
            assert abs(val[i] - float(s1)) < 1e-12
            al = (2, 1)
            with mp.workdps(40):
                m1 = eb.moment(T, x, k, al, F(float(H[i])))
            assert abs(np2d.moment(V[i:i + 1], X[i:i + 1], k, al, H[i])[0] - float(m1)) < 1e-12 * H[i] ** 3


def test_orientation_and_batch_shapes():
    T = np.array([[[-0.3, -0.2], [0.6, -0.1], [0.1, 0.7]]])
    x = np.array([[0.05, 0.1]])
    a = np2d.value(T, x, "w4")
    b = np2d.value(T[:, ::-1], x, "w4")                  # clockwise input
    assert abs(a - b) < 1e-15
    assert np2d.value(T[0], x[0], "w4").shape == (1,)


def test_tiny_elements_keep_relative_accuracy_with_unsplit_form():
    """L_T/h = 1e-6: the split form loses the value (abs err ~1e-14 on a value of 1e-12); the unsplit form does not."""
    recs = [c for c in ELEMENTS if c["id"].startswith("small_element-1e6") and ("inside" in c["id"] or "vertex" in c["id"])]
    V, X, H = rel_geometry(recs)
    for k in ["w4", "w2"]:
        un = np2d.value(V, X, k, H)
        sp = np2d.value(V, X, k, H, unsplit=False)
        for i, c in enumerate(recs):
            r = ref(c, k, "value")
            assert abs(un[i] - r) / r < 1e-13                                        # relative accuracy retained
        assert max(abs(sp[i] - ref(c, k, "value")) / ref(c, k, "value") for i, c in enumerate(recs)) > 1e-7


def test_longdouble_is_more_accurate_than_double():
    recs = [c for c in ELEMENTS if c["id"].startswith("generic")]
    V, X, H = rel_geometry(recs)
    VL = V.astype(np.longdouble)
    e64, eld = 0, 0
    for k in ["cubic", "w6"]:
        v64 = np2d.value(V, X, k, H)
        vld = np2d.value(VL, X.astype(np.longdouble), k, H.astype(np.longdouble), dtype=np.longdouble)
        for i, c in enumerate(recs):
            e64 = max(e64, abs(v64[i] - ref(c, k, "value")))
            eld = max(eld, abs(float(vld[i]) - ref(c, k, "value")))
    assert eld < 1e-15 and eld < e64


def test_gauss_quadrature_option_converges_for_moderate_z():
    recs = [c for c in ELEMENTS if c["id"].startswith("generic")]
    V, X, H = rel_geometry(recs)
    errs = []
    for m in (4, 8, 16):
        v = np2d.value(V, X, "w4", H, quad=m)
        errs.append(max(abs(v[i] - ref(c, "w4", "value")) for i, c in enumerate(recs)))
    assert errs[-1] < 1e-12 and errs[-1] <= errs[0]


def test_consistent_predicate_gives_continuity_across_an_edge():
    """x swept across an edge by 1e-15 ... 1e-2: sign test and atan use the SAME computed z, so there is no jump
    (|dV| <= |grad| * dx), also at ~1 ulp from the edge where the sign is decided by rounding."""
    T = np.array([[[-0.3, -0.2], [0.6, -0.1], [0.1, 0.7]]])
    p = T[0, 0] + 0.4 * (T[0, 1] - T[0, 0])
    nrm = np.array([T[0, 1, 1] - T[0, 0, 1], -(T[0, 1, 0] - T[0, 0, 0])])
    nrm /= np.linalg.norm(nrm)                           # outward normal of edge 0 (ccw)
    for k in ["cubic", "w4"]:
        v0 = np2d.value(T, p[None], k)[0]
        m0 = np2d.moment(T, p[None], k, (1, 1))[0]
        for dx in [1e-15, 1e-13, 1e-10, 1e-6, 1e-3]:
            for sgn in (1, -1):
                x = (p + sgn * dx * nrm)[None]
                assert abs(np2d.value(T, x, k)[0] - v0) < 1e-12 + 3 * dx
                assert abs(np2d.moment(T, x, k, (1, 1))[0] - m0) < 1e-12 + 3 * dx
