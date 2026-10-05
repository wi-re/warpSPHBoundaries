"""External benchmark: the test problems of Winchenbach & Kolb, "Solving Boundary Handling Analytically in Two Dimensions for SPH" (arXiv:2507.21686, Sec. 6.2).

Their setup (Wendland C4, support h = 1, evaluation point x = 0, triangulations of [-1,1]^2 so the support disk is covered exactly):
  test 1  piecewise-constant boundary field f = 1 on a coarse mesh of 8 triangles:  integral = 1,  gradient = 0
  test 2  piecewise-linear field f(x, y) = x, 8 large triangles and 96 smaller ones:  integral = 0,  gradient = (1, 0)^T
Reported: their analytic solution L2 error ~1e-13 (integral and gradient), degree-50 Xiao-Gimbutas quadrature ~1e-10 (integral) / 1e-8 (gradient), slowly
converging; quadrature needs up to 3 x 96 x 500 kernel evaluations.  Here: the same problems, exact answers known -> error of OUR implementation (float64 numpy,
float32 numpy, stage 1 at 40 digits).  (The paper's Wendland normalisation constant is not needed: the exact answers fix it.)
"""
import numpy as np
import pytest
from fractions import Fraction as F
from scipy.spatial import Delaunay

import mpmath as mp
import edgebound as eb
from edgebound.edge import fem, np_fem


def mesh8():
    """8 triangles of [-1,1]^2 with the evaluation point 0 as the common vertex (4 corners + 4 edge midpoints on the boundary)."""
    ring = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    return [[(0, 0), ring[i], ring[(i + 1) % 8]] for i in range(8)]


def mesh96(seed=3):
    """irregular Delaunay mesh of [-1,1]^2 with exactly 96 triangles (16 boundary vertices + 41 interior points)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, 5)[:-1]
    bnd = [(x, -1) for x in t] + [(1, y) for y in t] + [(x, 1) for x in -t] + [(-1, y) for y in -t]
    for _ in range(100):
        pts = np.array(bnd + [tuple(p) for p in rng.uniform(-0.93, 0.93, (41, 2))], dtype=float)
        tri = Delaunay(pts)
        if len(tri.simplices) == 96:
            return [[tuple(pts[i]) for i in s] for s in tri.simplices]
        seed += 1
        rng = np.random.default_rng(seed)
    raise RuntimeError("no 96-triangle mesh found")


def nodal_x(T):
    return np.array([v[0] for v in T], dtype=float)


def evaluate_numpy(tris, field, dtype=np.float64, stable=None):
    V = np.array(tris, dtype=float)
    X = np.zeros((len(tris), 2))
    w, G = np_fem.weights_hybrid(V.astype(dtype), X.astype(dtype), "w4", 1, 1.0, dtype=dtype, stable=stable, grad=True)
    A = np.array([field(T) for T in tris], dtype=np.float64)
    integral = (A * w).sum()
    grad = np.array([(A * G[..., 0]).sum(), (A * G[..., 1]).sum()])
    return float(integral), grad


@pytest.mark.parametrize("which", ["mesh8", "mesh96"])
def test_paper_test1_constant_field(which):
    tris = mesh8() if which == "mesh8" else mesh96()
    one = lambda T: np.ones(3)
    I, g = evaluate_numpy(tris, one)
    tol = 1e-13 if which == "mesh8" else 1e-12            # measured: mesh8 0 / 6e-17, mesh96 1.6e-14 / 3.9e-13 (paper: ~1e-13)
    assert abs(I - 1) < tol and np.abs(g).max() < tol, (I, g)


@pytest.mark.parametrize("which", ["mesh8", "mesh96"])
def test_paper_test2_linear_field(which):
    tris = mesh8() if which == "mesh8" else mesh96()
    I, g = evaluate_numpy(tris, nodal_x)
    err = max(abs(I - 0), abs(g[0] - 1), abs(g[1] - 0))
    assert err < (1e-13 if which == "mesh8" else 1e-12), (I, g)        # measured mesh8 0, mesh96 1.8e-13 (paper: ~1e-13)


@pytest.mark.parametrize("which", ["mesh8", "mesh96"])
def test_paper_tests_float32_stable(which):
    tris = mesh8() if which == "mesh8" else mesh96()
    I, g = evaluate_numpy(tris, nodal_x, np.float32, stable=(8, 6))
    assert abs(I) < 5e-6 and abs(g[0] - 1) < 5e-5 and abs(g[1]) < 5e-5, (I, g)


def test_paper_tests_stage1_at_40_digits():
    """exact (mpmath, rational mesh): the covering-mesh identities hold to ~1e-35 for the 8-triangle test."""
    tris = [[(F(a), F(b)) for a, b in T] for T in mesh8()]
    x = (F(0), F(0))
    with mp.workdps(40):
        I = sum(sum(fem.polynomial_field_nodal_values(T, 1, lambda X, Y: X)[i] * w for i, w in enumerate(fem.weights(T, x, "w4", 1))) for T in tris)
        one = sum(eb.value(T, x, "w4") for T in tris)
        gx = sum(sum(fem.polynomial_field_nodal_values(T, 1, lambda X, Y: X)[i] * G[0] for i, G in enumerate(fem.weights(T, x, "w4", 1, grad=True)[1])) for T in tris)
    assert abs(one - 1) < mp.mpf(10) ** -35 and abs(I) < mp.mpf(10) ** -35 and abs(gx - 1) < mp.mpf(10) ** -35


def test_paper_fig2_single_triangle_linear_field_vs_exact():
    """Fig. 2 setup: a large triangle, piecewise-linear field, h = 1; paper references 65536-point quadrature (error ~1e-6..1e-4 vs the analytic one).
    Here the reference is our 40-digit polar oracle (no quadrature-order limit): agreement of the float64 weights on a grid of evaluation points."""
    T = [(-2.0, 2.0), (2.0, 2.0), (0.0, -3.0)]
    A = np.array([1.0, 2.0, -1.0])
    Tf = [(F(a), F(b)) for a, b in T]
    rng = np.random.default_rng(1)
    pts = [(rng.uniform(-3, 3), rng.uniform(-4, 3)) for _ in range(12)] + [(0.0, 1.9), (0.0, 2.0), (-1.0, 2.0), (2.0, 2.0), (0.3, -1.0)]
    worst = 0.0
    for p in pts:
        V = np.array(T)[None]
        X = np.array(p)[None]
        w = np_fem.weights_hybrid(V, X, "w4", 1, 1.0)[0]
        got = float((A * w).sum())
        xf = (F(p[0]), F(p[1]))
        # exact: A(x') = sum A_i lambda_i(x') = a + b.(x'-x)   -> moments k <= 1 of the exact polar oracle
        lg = fem.lambdas_and_grads(Tf)
        a0 = sum(float(A[i]) * float(c + gx * xf[0] + gy * xf[1]) for i, (c, gx, gy) in enumerate(lg))
        b = (sum(float(A[i]) * float(lg[i][1]) for i in range(3)), sum(float(A[i]) * float(lg[i][2]) for i in range(3)))
        with mp.workdps(30):
            ref = a0 * eb.polar_value(Tf, xf, "w4") + b[0] * eb.polar_moment(Tf, xf, "w4", (1, 0)) + b[1] * eb.polar_moment(Tf, xf, "w4", (0, 1))
        worst = max(worst, abs(got - float(ref)))
    assert worst < 1e-12
