"""Validation of the planar closed forms against independent oracles.

  * 3-D: exact rational boundary values / joins + 40-digit mpmath quadrature
  * 2-D: boundary values (singular terms d^k ln d -> 0, tested at d ~ 1e-20),
    40-digit 1-D quadrature, exact branch join, and a TRUE 2-D Cartesian
    quadrature (no 1-D shell reduction)
  * h-scaling: physical integral with explicit h equals lambda_3(d/h)
"""
from fractions import Fraction

import mpmath as mp
import pytest

from warpSPHBoundaries.curvbound import (
    physical_planar3d,
    planar2d,
    planar3d,
    quad_planar2d,
    quad_planar2d_cartesian,
    quad_planar3d,
)
from warpSPHBoundaries.curvbound.symbolic import _eval_maple_expr, _load_planar

mp.mp.dps = 40
TOL = mp.mpf("1e-25")
ALL = ["cubic", "w2", "w4", "w6"]


# ------------------------------------------------------------ 3-D planar ----
def test_planar3d_cubic_paper_values():
    """Branch polynomials must be the paper's eq. (18) coefficients."""
    data = _load_planar()
    assert data["3d"][("cubic", "b1")] == [
        Fraction(16, 5), Fraction(-24, 5), 0, Fraction(8, 3), 0,
        Fraction(-7, 5), Fraction(1, 2),
    ]
    assert data["3d"][("cubic", "b2")] == [
        Fraction(-16, 15), Fraction(24, 5), -8, Fraction(16, 3), 0,
        Fraction(-8, 5), Fraction(8, 15),
    ]


@pytest.mark.parametrize("k", ALL)
def test_planar3d_boundary_exact(k):
    assert planar3d(k, Fraction(0)) == Fraction(1, 2)
    assert planar3d(k, Fraction(1)) == 0


def test_planar3d_cubic_join_exact():
    # b1 polynomial at 1/2 vs b2 polynomial at 1/2, both evaluated exactly
    data = _load_planar()

    def ev(coeffs, x):
        acc = Fraction(0)
        for c in coeffs:
            acc = acc * x + c
        return acc

    b1 = ev(data["3d"][("cubic", "b1")], Fraction(1, 2))
    b2 = ev(data["3d"][("cubic", "b2")], Fraction(1, 2))
    assert b1 == b2 == Fraction(1, 30)


@pytest.mark.parametrize("k", ALL)
def test_planar3d_vs_quadrature(k):
    worst = mp.mpf(0)
    for dnum in [1, 3, 4, 5, 7, 9, 99]:
        d = Fraction(dnum, 100)
        err = abs(mp.mpf(str(planar3d(k, d))) - quad_planar3d(k, d))
        worst = max(worst, err)
    assert worst < TOL, f"{k}: worst {worst}"


# ------------------------------------------------------------ 2-D planar ----
@pytest.mark.parametrize("k", ALL)
def test_planar2d_boundary(k):
    # the approach to 1/2 as d -> 0 is O(d |ln d|) (true asymptotics, not
    # numerics), so at d = 1e-20 allow ~1e-18; the EXACT limits lambda(0) =
    # 1/2 and lambda(1) = 0 are proved symbolically in maple/01_planar.mpl
    eps = mp.mpf(10) ** -20
    assert abs(planar2d(k, eps) - mp.mpf(1) / 2) < mp.mpf("1e-18"), k
    assert abs(planar2d(k, 1 - eps)) < mp.mpf("1e-18"), k


@pytest.mark.parametrize("k", ALL)
def test_planar2d_vs_quadrature(k):
    grid = [mp.mpf(1) / 10, mp.mpf(1) / 4, mp.mpf(2) / 5, mp.mpf(49) / 100,
            mp.mpf(51) / 100, mp.mpf(3) / 5, mp.mpf(7) / 10, mp.mpf(9) / 10]
    worst = mp.mpf(0)
    for d in grid:
        err = abs(planar2d(k, d) - quad_planar2d(k, d))
        worst = max(worst, err)
    assert worst < TOL, f"{k}: worst {worst}"


def test_planar2d_cubic_join_exact():
    # both branch expressions are non-singular at d = 1/2; evaluate both
    data = _load_planar()
    half = mp.mpf(1) / 2
    a = _eval_maple_expr(data["2d"][("cubic", "A")], half)
    b = _eval_maple_expr(data["2d"][("cubic", "B")], half)
    assert abs(a - b) < mp.mpf("1e-35")
    exact = _eval_const(data["2d"][("cubic", "join")])
    assert abs(a - exact) < mp.mpf("1e-35")


def _eval_const(expr: str) -> mp.mpf:
    """Constant (d-free) Maple expression -> mp.mpf."""
    return _eval_maple_expr(expr, mp.mpf(0))


@pytest.mark.parametrize("k", ["w2", "cubic"])
@pytest.mark.parametrize("d", [mp.mpf(1) / 5, mp.mpf(1) / 2, mp.mpf(2) / 3])
def test_planar2d_cartesian(k, d):
    """Independent true-2D Cartesian quadrature (30 digits, looser tol)."""
    err = abs(planar2d(k, d) - quad_planar2d_cartesian(k, d, dps=30))
    assert err < mp.mpf("1e-15"), f"{k} d={d}: {err}"


# ------------------------------------------------------------- h-scaling ----
@pytest.mark.parametrize("k", ALL)
def test_h_scaling_3d(k):
    """Physical integral with explicit h equals the unit-support lambda_3(d/h)."""
    for h in [2, 3, 5]:
        d = mp.mpf(h) * mp.mpf(3) / 10        # d/h = 0.3
        err = abs(physical_planar3d(k, d, h) - mp.mpf(str(planar3d(k, Fraction(3, 10)))))
        assert err < mp.mpf("1e-25"), f"{k} h={h}: {err}"
