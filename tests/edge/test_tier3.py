"""Tier 3 in 2D: closest-point curvature expansion F0 + kappa F1 + kappa^2 F2 vs the exact disk (two independent exact routes)."""
from fractions import Fraction as F

import mpmath as mp
import pytest
import sympy as sp

from warpSPHBoundaries.curvbound import planar2d
from warpSPHBoundaries.edge import tier3
from warpSPHBoundaries.edge.kernels import kernel as get_kernel
from warpSPHBoundaries.edge.mpq import mpq

NAMES = ["cubic", "w2", "w4", "w6"]


@pytest.mark.parametrize("name", NAMES)
def test_F0_is_the_planar_closed_form(name):
    for d in [F(1, 100), F(3, 10), F(1, 2), F(7, 10), F(19, 20)]:
        assert abs(tier3.F_k(name, d, 0) - planar2d(name, mpq(d), dps=50)) < mp.mpf(10) ** -35


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("d", [F(1, 20), F(3, 10), F(3, 5)])
@pytest.mark.parametrize("sign", [1, -1])
def test_truncation_error_orders_convex_and_concave(name, d, sign):
    """error of {F0, +kF1, +k^2 F2} against the exact disk scales as kappa^1, kappa^2, kappa^3 (kappa > 0: convex obstacle, kappa < 0: cavity)."""
    errs = {}
    for k in (F(1, 8), F(1, 16)):
        kap = sign * k
        ex = tier3.lambda_exact_disk(name, d, kap)
        errs[k] = [abs(tier3.lambda_expansion(name, d, kap, o) - ex) for o in range(3)]
    for o in range(3):
        order = mp.log(errs[F(1, 8)][o] / errs[F(1, 16)][o], 2)
        assert o + 1 - 0.35 < order < o + 1 + 0.6, (name, d, sign, o, order)
    # each order improves the answer
    assert errs[F(1, 16)][2] < errs[F(1, 16)][1] < errs[F(1, 16)][0]


@pytest.mark.parametrize("name", ["w4", "cubic"])
def test_gradient_orders(name):
    d = F(3, 10)
    errs = {}
    for k in (F(1, 8), F(1, 16)):
        ex = tier3.gradient_exact_disk(name, d, k)
        errs[k] = [abs(tier3.gradient_expansion(name, d, k, o) - ex) for o in range(3)]
    for o in range(2):
        order = mp.log(errs[F(1, 8)][o] / errs[F(1, 16)][o], 2)
        assert o + 1 - 0.35 < order < o + 1 + 0.7, (name, o, order)
    assert errs[F(1, 16)][2] < errs[F(1, 16)][1] / 5


def test_F1_F2_independent_2d_quadrature():
    """the closed forms (half-plane moments, edge machinery) vs a direct polar quadrature of the Taylor integrands (sympy derivatives of W(sqrt sigma)).
    Independent of the moment recursion, the primitives and Maple."""
    name, d = "w4", F(3, 10)
    lo, hi, c = get_kernel(name).pieces[0]
    r, sig = sp.symbols("r sigma", positive=True)
    W = sum(sp.Rational(cc.numerator, cc.denominator) * r ** n for n, cc in enumerate(c)) / sp.pi
    Phi = W.subs(r, sp.sqrt(sig))
    P0, P1, P2 = Phi, sp.diff(Phi, sig), sp.diff(Phi, sig, 2)
    y1, y2, dd = sp.symbols("y1 y2 dd")
    s_ = y2 - dd
    F1 = P1 * y1 ** 2 * (2 * dd - y2) - s_ * P0
    F2 = P1 * (-dd * s_ * y1 ** 2 - y1 ** 4 / 12) + P2 / 2 * y1 ** 4 * (2 * dd - y2) ** 2 - s_ * P1 * y1 ** 2 * (2 * dd - y2)
    with mp.workdps(22):
        dv = mpq(d)
        for k, Fk in ((1, F1), (2, F2)):
            f = sp.lambdify((y1, y2, sig, dd), Fk.subs(sig, y1 ** 2 + y2 ** 2), "mpmath")

            def radial(rr):
                th0 = mp.asin(dv / rr)
                return mp.quad(lambda th: f(rr * mp.cos(th), rr * mp.sin(th), None, dv) * rr, [th0, mp.pi - th0])
            val = mp.quad(radial, [dv, mp.mpf(1)])
            assert abs(val - tier3.F_k(name, d, k, dps=22)) < mp.mpf(10) ** -15, k


def test_F_k_regular_as_d_to_zero_and_sign_of_F1():
    """F1 < 0 (a convex solid has less volume in the support), finite at d -> 0; F2(0) ~ 0 and F2 = O(d) for w4."""
    for name in NAMES:
        f1 = [tier3.F_k(name, F(1, 10 ** e), 1) for e in (3, 5)]
        assert f1[0] < 0 and abs(f1[0] - f1[1]) < 1e-3 * abs(f1[0])
    f2 = [tier3.F_k("w4", F(1, 10 ** e), 2) for e in (3, 5)]
    assert abs(f2[1]) < abs(f2[0]) / 50


def test_tier3_vs_tier2_accuracy_report():
    """kappa h = 1/4, d = 0.3: second order expansion (no mesh) vs polygons with edge length h/4, h/8 (tier 2)."""
    import numpy as np
    from warpSPHBoundaries.edge import np2d
    d, kap = F(3, 10), F(1, 4)
    ex = float(tier3.lambda_exact_disk("w4", d, kap))
    e2 = abs(float(tier3.lambda_expansion("w4", d, kap, 2)) - ex)
    R = float(1 / kap)
    errs = []
    for ell in (0.25, 0.125):
        N = int(round(2 * np.pi * R / ell))
        t = 2 * np.pi * np.arange(N) / N
        poly = np.stack([R * np.cos(t), R + 0.3 + R * np.sin(t)], axis=1)
        errs.append(abs(np2d.value(poly[None], np.zeros((1, 2)), "w4")[0] - ex))
    assert e2 < errs[1] / 5 < errs[0]
