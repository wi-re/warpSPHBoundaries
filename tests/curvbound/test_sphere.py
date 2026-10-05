"""Validation of the solid-sphere closed forms (Wendland w2/w4/w6).

  * 40-digit quadrature of the defining integral on both branches,
  * EXACT branch join on the curve 2R + d = 1 (rational arithmetic),
  * EXACT boundary values (lambda = 0 at d = 1 on branch 1),
  * planar limit R -> infinity (R = 20 vs the planar oracle).
"""
from fractions import Fraction

import mpmath as mp
import pytest

from curvbound import planar3d, quad_planar3d, quad_sphere, sphere

mp.mp.dps = 40
TOL = mp.mpf("1e-25")
ALL = ["w2", "w4", "w6"]


@pytest.mark.parametrize("k", ALL)
def test_sphere_b1_vs_quadrature(k):
    worst = mp.mpf(0)
    for R in [Fraction(1, 2), 1, 2]:
        for dnum in [0, 1, 3, 5, 7, 9, 10]:
            d = Fraction(dnum, 10)
            err = abs(mp.mpf(str(sphere(k, R, d))) -
                      quad_sphere(k, mp.mpf(str(R)), mp.mpf(str(d))))
            worst = max(worst, err)
    assert worst < TOL, f"{k}: worst {worst}"


@pytest.mark.parametrize("k", ALL)
def test_sphere_b2_vs_quadrature(k):
    worst = mp.mpf(0)
    count = 0
    for Rnum in [1, 2, 3, 4]:
        R = Fraction(Rnum, 10)
        for dnum in [0, 5, 15, 25, 35, 45]:
            d = Fraction(dnum, 100)
            if 2 * R + d >= 1:
                continue
            count += 1
            err = abs(mp.mpf(str(sphere(k, R, d))) -
                      quad_sphere(k, mp.mpf(str(R)), mp.mpf(str(d))))
            worst = max(worst, err)
    assert count > 10
    assert worst < TOL, f"{k}: worst {worst}"


@pytest.mark.parametrize("k", ALL)
def test_sphere_join_exact(k):
    """On 2R+d = 1 the two rational branch forms agree EXACTLY."""
    d = Fraction(1, 3)
    R = (1 - d) / 2                       # R = 1/3: 2R + d = 1 exactly
    v1 = sphere(k, R, d, branch="b1")
    v2 = sphere(k, R, d, branch="b2")
    assert v1 == v2, f"{k}: {v1} != {v2}"


@pytest.mark.parametrize("k", ALL)
def test_sphere_boundary_exact(k):
    assert sphere(k, 1, Fraction(1)) == 0
    assert sphere(k, Fraction(1, 2), Fraction(1)) == 0


@pytest.mark.parametrize("k", ALL)
def test_sphere_planar_limit(k):
    """|lambda_sph(R=20, d) - lambda_3(d)| must be small (O(1/R))."""
    for dnum in [1, 3, 5]:
        d = Fraction(dnum, 10)
        diff = abs(mp.mpf(str(sphere(k, 20, d))) -
                   mp.mpf(str(planar3d(k, d))))
        assert diff < mp.mpf("1e-2"), f"{k} d={d}: {diff}"
        # and the sphere oracle agrees with the closed form there too
        err = abs(mp.mpf(str(sphere(k, 20, d))) -
                  quad_sphere(k, 20, mp.mpf(str(d))))
        assert err < TOL


def test_sphere_matches_planar_oracle_indirectly():
    """R = 20 sphere vs the planar QUADRATURE oracle (cross-oracle check)."""
    for k in ALL:
        d = mp.mpf(1) / 3
        err = abs(mp.mpf(str(sphere(k, 20, Fraction(1, 3)))) -
                  quad_planar3d(k, d))
        assert err < mp.mpf("1e-2")
