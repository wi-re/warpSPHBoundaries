"""Shape derivative (vertex gradient): edge-local analytic adjoint vs 40+ digit finite differences of the exact stage 1."""
from fractions import Fraction as F

import numpy as np
import pytest

import warpSPHBoundaries as eb
from warpSPHBoundaries.edge import np2d
from warpSPHBoundaries.edge.mpq import mpq

from .conftest import rand_point, rand_triangle


def fd_vertex(T, x, name, al, k, comp, h=F(1)):
    """4th-order central difference of the exact stage-1 moment w.r.t. coordinate `comp` of vertex k."""
    step = F(1, 10**10)

    def f(m):
        V = [list(v) for v in T]
        V[k][comp] = V[k][comp] + m * step
        return eb.moment([tuple(v) for v in V], x, name, al, h, dps=60)
    return (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / (12 * mpq(step))


@pytest.mark.parametrize("name", ["cubic", "w4"])
@pytest.mark.parametrize("al", [(0, 0), (1, 0), (1, 1), (2, 1)])
def test_shape_gradient_vs_exact_finite_differences(name, al, rng):
    for _ in range(2):
        T, x = rand_triangle(rng), rand_point(rng)
        V = np.array([[float(a), float(b)] for a, b in T])
        X = np.array([[float(x[0]), float(x[1])]])
        # exact geometry = the float inputs (rationals) so that stage 1 and numpy see the same triangle
        Tf = [(F(float(a)), F(float(b))) for a, b in V]
        xf = (F(float(X[0, 0])), F(float(X[0, 1])))
        sg = np2d.shape_gradient(V[None], X, name, al)[0]
        for k in range(3):
            for comp in range(2):
                ref = float(fd_vertex(Tf, xf, name, al, k, comp))
                assert abs(sg[k, comp] - ref) < 1e-11 + 1e-9 * abs(ref), (al, k, comp, sg[k, comp], ref)


def test_translation_identity_for_the_value():
    """moving the polygon is moving x the other way:  sum_k d value/d v_k = - grad_x value."""
    rng = np.random.default_rng(3)
    V = rng.uniform(-1, 1, (20, 3, 2))
    X = rng.uniform(-0.3, 0.3, (20, 2))
    for name in ["cubic", "w2", "w4", "w6"]:
        sg = np2d.shape_gradient(V, X, name)
        g = np2d.gradient(V, X, name)
        assert np.abs(sg.sum(1) + g).max() < 1e-11


def test_orientation_independent_and_scaling():
    T = np.array([[[-0.3, -0.2], [0.6, -0.1], [0.1, 0.7]]])
    x = np.array([[0.05, 0.1]])
    a = np2d.shape_gradient(T, x, "w4")
    b = np2d.shape_gradient(T[:, ::-1], x, "w4")[:, ::-1]           # clockwise input, vertices reversed back
    assert np.abs(a - b).max() < 1e-13
    h = 1.7
    c = np2d.shape_gradient(T * h, x * h, "w4", h=h)
    assert np.abs(c - a / h).max() < 1e-12                           # value ~ h^0  =>  d/d vertex ~ 1/h


def test_float32_stable_shape_gradient():
    rng = np.random.default_rng(4)
    V = rng.uniform(-0.8, 0.8, (30, 3, 2))
    X = rng.uniform(-0.3, 0.3, (30, 2))
    ref = np2d.shape_gradient(V, X, "w6")
    s32 = np2d.shape_gradient(V.astype(np.float32), X.astype(np.float32), "w6", dtype=np.float32, stable=(8, 6))
    assert np.abs(s32.astype(np.float64) - ref).max() < 5e-5
