"""Tier 4 in 2D: strip / disk mean-value series vs exact references; the circle edge identity (curved boundary)."""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

import edgebound as eb
from edgebound.edge import np2d, tier4

TOL = mp.mpf(10) ** -35


@pytest.mark.parametrize("name", ["cubic", "w4"])
@pytest.mark.parametrize("center,Rd", [((F(0), F(13, 10)), F(1)), ((F(0), F(3, 10)), F(1, 5)), ((F(1, 10), F(1, 20)), F(4, 5)), ((F(0), F(4, 5)), F(1, 10))])
def test_circle_edge_identity_equals_polar_disk_oracle(name, center, Rd):
    """divergence theorem on a CURVED boundary (arcs by Gauss-Legendre) = independent polar oracle: x outside / inside the disk,
    support crossing the circle, cubic knot radius crossing the circle."""
    ref = eb.polar_disk_value(center, Rd, (F(0), F(0)), name)
    assert abs(tier4.arc_value(name, center, Rd) - ref) < TOL


@pytest.mark.parametrize("name", ["w4", "w6", "cubic"])
def test_disk_series_orders(name):
    """error of the K-th partial sum ~ a^(2K+4); D = 0.65 (kernel smooth on [D-a, D+a] for a <= 0.1; cubic knot at 0.5)."""
    D = F(13, 20)
    errs = {}
    for a in (F(1, 10), F(1, 20)):
        ex = tier4.arc_value(name, (F(0), D), a)
        errs[a] = [abs(tier4.disk_series(name, a, D, K) - ex) for K in range(3)]
    for K in range(3):
        order = mp.log(errs[F(1, 10)][K] / errs[F(1, 20)][K], 2)
        assert abs(order - (2 * K + 4)) < 0.6, (name, K, order)
    # each extra term reduces the error
    assert errs[F(1, 20)][2] < errs[F(1, 20)][1] < errs[F(1, 20)][0]


@pytest.mark.parametrize("name", ["w4", "w2"])
def test_strip_series_orders_and_exact_strip_is_two_edges(name):
    d = F(3, 5)
    errs = {}
    for a in (F(1, 10), F(1, 20)):
        ex = tier4.strip_exact(name, a, d)
        errs[a] = [abs(tier4.strip_series(name, a, d, K) - ex) for K in range(2)]
    for K in range(2):
        order = mp.log(errs[F(1, 10)][K] / errs[F(1, 20)][K], 2)
        assert abs(order - (2 * K + 3)) < 0.9, (name, K, order)
    # the exact strip is a polygon: 2 edges of the tier-1/2 machinery (long rectangle)
    a = F(1, 10)
    rect = [(F(-30), d - a), (F(30), d - a), (F(30), d + a), (F(-30), d + a)]
    assert abs(eb.value(rect, (F(0), F(0)), name) - tier4.strip_exact(name, a, d)) < mp.mpf(10) ** -33


def test_disk_series_k0_is_the_point_approximation():
    """leading term = pi a^2 W(D) (a 'point' with its volume) -- the 2D analogue of the slender-limit leading term."""
    name, a, D = "w4", F(1, 20), F(3, 5)
    from edgebound.edge.kernels import kernel, peval
    lo, hi, c = kernel(name).pieces[0]
    W = peval([mp.mpf(v.numerator) / v.denominator for v in c], mp.mpf(3) / 5) / mp.pi
    assert abs(tier4.disk_series(name, a, D, 0) - mp.pi / 400 * W) < mp.mpf(10) ** -35


def test_series_rejects_knot_straddling_disk():
    with pytest.raises(ValueError):
        tier4.disk_series("cubic", F(1, 5), F(1, 2), 1)
    with pytest.raises(ValueError):
        tier4.disk_series("w4", F(1, 5), F(9, 10), 1)


def test_crossover_series_vs_polygon_report():
    """for a/h = 0.1: which of {series K, inscribed N-gon (tier 2)} is more accurate (records the numbers, asserts the ordering)."""
    D, a = F(3, 5), F(1, 10)
    ex = float(tier4.arc_value("w4", (F(0), D), a))
    ser = [abs(float(tier4.disk_series("w4", a, D, K)) - ex) for K in range(3)]
    N = 16
    t = 2 * np.pi * np.arange(N) / N
    poly = np.stack([0.1 * np.cos(t), 0.6 + 0.1 * np.sin(t)], axis=1)
    err_poly = abs(np2d.value(poly[None], np.zeros((1, 2)), "w4")[0] - ex)
    assert ser[1] < err_poly and ser[2] < ser[1]          # K = 1 beats a 16-gon already; the series is cheaper (1 point, no edges)
