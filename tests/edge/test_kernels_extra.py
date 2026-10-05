"""Additional warpSPHCore kernels (quartic, quintic, B7, B8, poly6): exact tables vs kernel_specs.yaml, and the whole 2D machinery on them."""
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

import edgebound as eb
from edgebound.edge import fem, np2d, np_fem
from edgebound.edge.kernels import KERNELS, WARPSPH_C2_PI, disk_moment, peval
from edgebound.edge.mpq import mpq

from .conftest import rand_point, rand_triangle

NEW = ["quartic", "quintic", "b7", "b8", "poly6"]
SPEC = Path("/home/lu26029/dev/warpSPHCore/scripts/kernels/kernel_specs.yaml")


@pytest.mark.parametrize("name", NEW)
def test_exact_normalisation_and_warpsph_constants(name):
    k = KERNELS[name]
    assert k.c2_pi == WARPSPH_C2_PI[name]                  # C2 * pi from warpSPHCore's table (independent of the construction)
    assert disk_moment(k, (0, 0)) == 1
    # smoothness at every knot: pi W and its first (p-1) derivatives continuous, W(1) = 0
    for (lo, hi, c), (lo2, hi2, c2) in zip(k.pieces[:-1], k.pieces[1:]):
        assert peval(list(c), hi) == peval(list(c2), hi)
    assert peval(list(k.pieces[-1][2]), F(1)) == 0


@pytest.mark.skipif(not SPEC.exists(), reason="warpSPHCore kernel_specs.yaml not available")
@pytest.mark.parametrize("name", [n for n in NEW if n != "poly6"])
def test_shape_matches_kernel_spec_yaml(name):
    import yaml
    spec = yaml.safe_load(SPEC.read_text())["kernels"][name]
    expr = spec["shape"]["expr"]
    ns = {"pos": lambda x: np.maximum(x, 0.0), "step": lambda x: (x >= 0) * 1.0, "pi": np.pi, "exp": np.exp, "sqrt": np.sqrt}
    q = np.linspace(0.0, 0.999, 200)
    shape = eval(expr, {"__builtins__": {}}, dict(ns, q=q))
    k = KERNELS[name]
    # the pieces carry pi*W = C2_pi * shape on each interval: evaluate the piece containing x
    vals = []
    for x in q:
        for lo, hi, c in k.pieces:
            if float(lo) <= x <= float(hi):
                vals.append(float(peval([mp.mpf(v.numerator) / v.denominator for v in c], mp.mpf(float(x)))) / float(k.c2_pi))
                break
    assert np.abs(np.array(vals) - shape).max() < 1e-12


@pytest.mark.parametrize("name", NEW)
def test_value_gradient_moments_vs_polar_oracle(name, rng):
    for _ in range(2):
        T, x = rand_triangle(rng), rand_point(rng)
        assert abs(eb.value(T, x, name) - eb.polar_value(T, x, name)) < mp.mpf(10) ** -28
        g, og = eb.gradient(T, x, name), eb.polar_gradient(T, x, name)
        assert max(abs(g[0] - og[0]), abs(g[1] - og[1])) < mp.mpf(10) ** -28
        for al in [(1, 0), (1, 1), (0, 2), (2, 1)]:
            assert abs(eb.moment(T, x, name, al) - eb.polar_moment(T, x, name, al)) < mp.mpf(10) ** -28


@pytest.mark.parametrize("name", NEW)
def test_numpy_float64_and_fem_p1_and_covering_mesh(name):
    from .test_value import _tiling
    rng = np.random.default_rng(3)
    V = rng.uniform(-1, 1, (30, 3, 2))
    X = rng.uniform(-.4, .4, (30, 2))
    for i in range(0, 30, 6):
        T = [(F(float(a)), F(float(b))) for a, b in V[i]]
        x = (F(float(X[i, 0])), F(float(X[i, 1])))
        assert abs(np2d.value(V[i:i + 1], X[i:i + 1], name)[0] - float(eb.value(T, x, name))) < 5e-12
        w = np_fem.weights_hybrid(V[i:i + 1], X[i:i + 1], name, 1)[0]
        wr = fem.weights(T, x, name, 1)
        assert max(abs(w[k] - float(wr[k])) for k in range(3)) < 5e-12
    tris = [[(float(a), float(b)) for a, b in T] for T in _tiling(F(2), 2)]
    tot = sum(np2d.value(np.array([T]), np.array([[0.1, -0.2]]), name)[0] for T in tris)
    assert abs(tot - 1) < 1e-12
