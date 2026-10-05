"""fem-nodal-weights.md checks 1-6 (stage 1, mpmath). Conditioning (check 7): tests/edge/test_fem_np.py."""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound.edge import fem
from edgebound.edge.kernels import disk_moment, kernel as get_kernel, pderiv
from edgebound.edge.mpq import mpq

from .conftest import KERNELS, rand_point, rand_triangle
from .conftest import tiling

TOL = mp.mpf(10) ** -30
T0 = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]

# a degree-3 polynomial (and its lower-degree truncations)
A3 = {(0, 0): F(1), (1, 0): F(2), (0, 1): F(-3), (1, 1): F(1), (2, 0): F(1, 2), (0, 2): F(-1, 4), (0, 3): F(-1), (2, 1): F(1), (3, 0): F(1, 3), (1, 2): F(-2)}


def trunc(A, p):
    return {k: v for k, v in A.items() if sum(k) <= p}


def disk_ref(kern, A, x):
    """exact int over R^2 of A(x') W(|x - x'|) = sum a_alpha disk_moment(alpha)  (rational)."""
    return sum(c * disk_moment(get_kernel(kern), al) for al, c in fem.poly_shift(A, x).items())


def test_basis_is_exactly_lagrange():
    for p in range(1, 4):
        nodes, polys = fem.basis(p)
        assert len(nodes) == (p + 1) * (p + 2) // 2
        for i, n in enumerate(nodes):
            for j, P in enumerate(polys):
                assert fem.eval_bary(P, n) == (1 if i == j else 0)
        lam = (F(1, 5), F(3, 10), F(1, 2))
        assert sum(fem.eval_bary(P, lam) for P in polys) == 1


@pytest.mark.parametrize("p", [0, 1, 2, 3])
@pytest.mark.parametrize("name", ["cubic", "w4"])
def test_1_partition_of_unity(name, p, rng):
    for _ in range(2):
        T, x = rand_triangle(rng), rand_point(rng)
        w = fem.weights(T, x, name, p)
        assert abs(sum(w) - eb.value(T, x, name)) < TOL
    tot = sum(sum(fem.weights(Tt, (F(1, 7), F(-1, 5)), name, p)) for Tt in tiling(F(2), 2))
    assert abs(tot - 1) < mp.mpf(10) ** -33


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("name", KERNELS)
def test_2_polynomial_reproduction(name, p, rng):
    A = trunc(A3, p)
    # (a) covering mesh vs exact rational disk integral
    tris = tiling(F(2), 2)
    x = (F(1, 7), F(-1, 5))
    tot = sum(fem.field_integral(Tt, x, name, p, fem.nodal_values(Tt, p, A)) for Tt in tris)
    assert abs(tot - mpq(disk_ref(name, A, x))) < mp.mpf(10) ** -32
    # (b) single element vs independent polar quadrature of the shifted polynomial
    for _ in range(2):
        T, x = rand_triangle(rng), rand_point(rng)
        got = fem.field_integral(T, x, name, p, fem.nodal_values(T, p, A))
        ref = sum(mpq(c) * eb.polar_moment(T, x, name, al) for al, c in fem.poly_shift(A, x).items())
        assert abs(got - ref) < TOL


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("name", ["cubic", "w2", "w6"])
def test_3_gradient_reproduction(name, p, rng):
    A = trunc(A3, p)
    dAx, dAy = fem.poly_grad(A)
    # covering mesh: int A grad_x W = int grad A W  (exact, rational)
    tris = tiling(F(2), 2)
    x = (F(1, 7), F(-1, 5))
    gx = gy = mp.mpf(0)
    for Tt in tris:
        g = fem.field_gradient_integral(Tt, x, name, p, fem.nodal_values(Tt, p, A))
        gx, gy = gx + g[0], gy + g[1]
    assert abs(gx - mpq(disk_ref(name, dAx, x))) < mp.mpf(10) ** -32 and abs(gy - mpq(disk_ref(name, dAy, x))) < mp.mpf(10) ** -32
    # single element vs polar: grad_x W = -(W'/r) y
    T, x = rand_triangle(rng), rand_point(rng)
    got = fem.field_gradient_integral(T, x, name, p, fem.nodal_values(T, p, A))
    from edgebound.edge.oracle import polar_moment_profile
    kern = get_kernel(name)
    pieces = [(lo, hi, {j - 1: -d for j, d in enumerate(pderiv(list(c))) if d != 0}) for lo, hi, c in kern.pieces]
    ref = [mp.mpf(0), mp.mpf(0)]
    for al, c in fem.poly_shift(A, x).items():
        for j in range(2):
            be = (al[0] + (j == 0), al[1] + (j == 1))
            ref[j] += mpq(c) * polar_moment_profile(T, x, be, pieces, dps=40) / mp.pi
    assert abs(got[0] - ref[0]) < TOL and abs(got[1] - ref[1]) < TOL


@pytest.mark.parametrize("p", [1, 2, 3])
def test_4_refinement_invariance(p, rng):
    A = trunc(A3, p)
    T, x = rand_triangle(rng), rand_point(rng)
    a, b, c = T
    mid = lambda u, v: ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
    ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
    whole = fem.field_integral(T, x, "w4", p, fem.nodal_values(T, p, A))
    parts = sum(fem.field_integral(S, x, "w4", p, fem.nodal_values(S, p, A)) for S in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
    assert abs(whole - parts) < mp.mpf(10) ** -33
    gw = fem.field_gradient_integral(T, x, "w4", p, fem.nodal_values(T, p, A))
    gp = [sum(fem.field_gradient_integral(S, x, "w4", p, fem.nodal_values(S, p, A))[j] for S in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])) for j in range(2)]
    assert abs(gw[0] - gp[0]) < mp.mpf(10) ** -33 and abs(gw[1] - gp[1]) < mp.mpf(10) ** -33


@pytest.mark.parametrize("p", [1, 2, 3])
def test_5_continuity_across_shared_edge(p):
    """continuous P_p field on two triangles sharing an edge; x swept across the edge: no jump in value or gradient."""
    A = trunc(A3, p)
    T1 = [(F(-3, 10), F(-1, 5)), (F(3, 5), F(-1, 10)), (F(1, 10), F(7, 10))]
    T2 = [(F(3, 5), F(-1, 10)), (F(7, 10), F(3, 5)), (F(1, 10), F(7, 10))]        # shares edge T1[1]-T1[2]
    P, Q = T1[1], T1[2]
    on = (P[0] + F(2, 5) * (Q[0] - P[0]), P[1] + F(2, 5) * (Q[1] - P[1]))
    nrm = (Q[1] - P[1], -(Q[0] - P[0]))                    # normal to the shared edge
    def total(x):
        v = sum(fem.field_integral(T, x, "w4", p, fem.nodal_values(T, p, A)) for T in (T1, T2))
        g = [sum(fem.field_gradient_integral(T, x, "w4", p, fem.nodal_values(T, p, A))[j] for T in (T1, T2)) for j in range(2)]
        return v, g
    v0, g0 = total(on)
    for e in (F(1, 10**3), F(1, 10**8), F(1, 10**20)):
        for s in (1, -1):
            x = (on[0] + s * e * nrm[0], on[1] + s * e * nrm[1])
            v, g = total(x)
            assert abs(v - v0) < 10 * mpq(e) and abs(g[0] - g0[0]) < 10 * mpq(e) and abs(g[1] - g0[1]) < 10 * mpq(e)


def test_6_degree_exceeded_error_order():
    """field of degree p+1 represented by P_p nodal interpolation on a refined covering mesh: error ~ (mesh size)^(p+1)."""
    A = A3
    x = (F(1, 7), F(-1, 5))
    for p in (1, 2):
        errs = []
        for n in (2, 4, 8):
            tris = tiling(F(2), n)
            # nodal values of the degree-3 (or 2) field, interpolated at degree p (cannot be represented exactly)
            Afull = trunc(A, 3)
            tot = sum(fem.field_integral(Tt, x, "w4", p, fem.nodal_values(Tt, p, Afull)) for Tt in tris)
            errs.append(abs(tot - mpq(disk_ref("w4", Afull, x))))
        order = [mp.log(errs[i] / errs[i + 1], 2) for i in range(2)]
        assert order[-1] > p + 0.7, (p, errs, order)               # observed order ~ p + 1


@pytest.mark.parametrize("p", [1, 2])
def test_p0_is_the_constant_moment(p):
    T, x = T0, (F(1, 20), F(1, 10))
    w = fem.weights(T, x, "w4", 0)
    assert len(w) == 1 and abs(w[0] - eb.value(T, x, "w4")) < mp.mpf(10) ** -35
    # P1: w_i = lambda_i(x) m_0 + grad lambda_i . m_1
    w1 = fem.weights(T, x, "w4", 1)
    lg = fem.lambdas_and_grads(T)
    m0 = eb.value(T, x, "w4")
    m1 = (eb.moment(T, x, "w4", (1, 0)), eb.moment(T, x, "w4", (0, 1)))
    for i, (c, gx, gy) in enumerate(lg):
        lam = c + gx * F(1, 20) + gy * F(1, 10)
        assert abs(w1[i] - (mpq(lam) * m0 + mpq(gx) * m1[0] + mpq(gy) * m1[1])) < mp.mpf(10) ** -35
