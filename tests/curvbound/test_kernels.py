"""Kernel-table integrity: normalization and warpSPHCore transcription."""
from fractions import Fraction

import mpmath as mp

from warpSPHBoundaries.curvbound import KERNELS, kernel
from warpSPHBoundaries.curvbound.oracle import _c2, _c3, _w_hat_mp  # private, exercised here


mp.mp.dps = 40
TOL = mp.mpf("1e-30")


def test_table_values():
    """Guard against transcription drift from kernel_specs.yaml."""
    assert KERNELS["cubic"].c2_pi == Fraction(80, 7)
    assert KERNELS["cubic"].c3_pi == Fraction(16)
    assert KERNELS["w2"].c2_pi == Fraction(7)
    assert KERNELS["w2"].c3_pi == Fraction(21, 2)
    assert KERNELS["w2"].n == 4
    assert KERNELS["w2"].poly == ((0, Fraction(1)), (1, Fraction(4)))
    assert KERNELS["w4"].c2_pi == Fraction(9)
    assert KERNELS["w4"].c3_pi == Fraction(495, 32)
    assert KERNELS["w6"].c2_pi == Fraction(78, 7)
    assert KERNELS["w6"].c3_pi == Fraction(1365, 64)
    assert kernel("w4").kind == "wendland"
    assert kernel("cubic").kind == "cubic"


def test_normalization_2d_3d():
    """2 pi C2 Int W q dq = 1 and 4 pi C3 Int W q^2 dq = 1 for every kernel."""
    for name, kd in KERNELS.items():
        f2 = lambda q: 2 * mp.pi * _c2(kd) * _w_hat_mp(kd, q) * q
        f3 = lambda q: 4 * mp.pi * _c3(kd) * _w_hat_mp(kd, q) * q ** 2
        if kd.kind == "cubic":
            n2 = mp.quad(f2, [0, mp.mpf(1) / 2, 1])
            n3 = mp.quad(f3, [0, mp.mpf(1) / 2, 1])
        else:
            n2 = mp.quad(f2, [0, 1])
            n3 = mp.quad(f3, [0, 1])
        assert abs(n2 - 1) < TOL, f"{name} 2D normalization {n2}"
        assert abs(n3 - 1) < TOL, f"{name} 3D normalization {n3}"


def test_shape_values():
    """Exact shape values at a few points (warpSPHCore-compatible)."""
    cubic = KERNELS["cubic"]
    # (1-q)^3 - 4(1/2-q)_+^3 = 1/2 - 3q^2 + 3q^3 for q <= 1/2 (so W(0) = 1/2)
    assert cubic.w_hat(Fraction(0)) == Fraction(1, 2)
    assert cubic.w_hat(Fraction(1, 2)) == Fraction(1, 8)      # (1/2)^3
    assert cubic.w_hat(Fraction(1)) == 0
    for name in ("w2", "w4", "w6"):
        kd = KERNELS[name]
        assert kd.w_hat(Fraction(0)) == 1
        assert kd.w_hat(Fraction(1)) == 0
        # flat at the support edge for the Wendland family
        q = Fraction(99, 100)
        assert kd.w_hat(q) > 0


def test_wendland_flat_at_edge():
    """Wendland kernels vanish to order >= 2 at q = 1."""
    for name in ("w2", "w4", "w6"):
        kd = KERNELS[name]
        q = mp.mpf(1) - mp.mpf("1e-8")
        assert abs(_w_hat_mp(kd, q)) < mp.mpf("1e-12"), name
