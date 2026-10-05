"""edge-primitives.md checks: derivatives, definite integrals vs quad, z -> 0, short chords."""
from fractions import Fraction as F

import mpmath as mp
import pytest

from edgebound.edge import primitives as pr
from edgebound.edge.mpq import mpq


Z = [F(3, 10), F(-3, 10), F(7, 5), F(1, 1000), F(-1, 10**9)]
S = [F(1, 3), F(-7, 5), F(0), F(9, 10), F(-1, 10**6)]


@pytest.mark.parametrize("m", range(0, 13))
def test_I_derivative(m):
    for z in Z:
        for s in S:
            zz, ss = mpq(z), mpq(s)
            d = mp.diff(lambda t: pr.I(m, t, zz), ss)
            assert abs(d - (ss * ss + zz * zz) ** (mp.mpf(m) / 2)) < mp.mpf(10) ** -28 * (1 + abs(d))


@pytest.mark.parametrize("m", range(-8, 0))
def test_Ig_negative_derivative(m):
    # downward recurrence divides by z^2 at every step: only moderate z (documented failure mode;
    # the production path never uses m < -2)
    for z in [F(3, 10), F(-3, 10), F(7, 5)]:
        zz = mpq(z)
        for s in S:
            ss = mpq(s)
            d = mp.diff(lambda t: pr.Ig(m, t, zz), ss)
            ref = (ss * ss + zz * zz) ** (mp.mpf(m) / 2)
            assert abs(d - ref) < mp.mpf(10) ** -25 * (1 + abs(ref))


def test_I_definite_vs_quad():
    chords = [(F(1, 3), F(5, 4)), (F(-7, 5), F(9, 10)), (F(-2), F(-1, 2)), (F(-1, 10**6), F(1, 10**6))]
    for z in [F(3, 10), F(-1, 2), F(1, 10**5)]:
        zz = mpq(z)
        for lo, hi in chords:
            for m in range(0, 13):
                got = pr.I(m, mpq(hi), zz) - pr.I(m, mpq(lo), zz)
                pts = sorted({mpq(lo), mpq(hi)} | {p for p in (-zz, 0, zz) if mpq(lo) < p < mpq(hi)})
                with mp.workdps(70):
                    ref = mp.quad(lambda t: (t * t + zz * zz) ** (mp.mpf(m) / 2), pts, maxdegree=10)
                assert abs(got - ref) < mp.mpf(10) ** -30 * (1 + abs(ref))


def test_S_definite_vs_quad():
    z = mpq(F(3, 2))
    for j in range(0, 8):
        for m in range(-6, 9):
            for lo, hi in [(F(1, 3), F(5, 4)), (F(-7, 5), F(9, 10)), (F(-2), F(-1, 2))]:
                got = pr.Sg(j, m, mpq(hi), z) - pr.Sg(j, m, mpq(lo), z)
                ref = mp.quad(lambda t: t ** j * (t * t + z * z) ** (mp.mpf(m) / 2),
                              [mpq(lo), 0, mpq(hi)] if lo < 0 < hi else [mpq(lo), mpq(hi)])
                assert abs(got - ref) < mp.mpf(10) ** -30 * (1 + abs(ref))
                if m >= 0:
                    got2 = pr.S_all(j, m, mpq(hi), z) - pr.S_all(j, m, mpq(lo), z)
                    assert abs(got2 - got) < mp.mpf(10) ** -35 * (1 + abs(got))


def test_z_zero_guard():
    for m in range(0, 13):
        for s in [mpq(F(2, 3)), mpq(F(-2, 3))]:
            ref = s * abs(s) ** m / (m + 1)
            assert abs(pr.I(m, s, mp.mpf(0)) - ref) < mp.mpf(10) ** -38


def test_odd_in_s():
    z = mpq(F(2, 7))
    for m in range(0, 13):
        s = mpq(F(5, 11))
        assert abs(pr.I(m, -s, z) + pr.I(m, s, z)) < mp.mpf(10) ** -38


def test_dangle_matches_and_is_stable():
    z = mpq(F(3, 10))
    lo, hi = mpq(F(-1, 3)), mpq(F(5, 4))
    assert abs(pr.dangle(z, lo, hi) - (mp.atan(hi / z) - mp.atan(lo / z))) < mp.mpf(10) ** -38
    assert abs(pr.dangle(-z, lo, hi) - (mp.atan(hi / -z) - mp.atan(lo / -z))) < mp.mpf(10) ** -38
    # far from the foot point and short: compare with the naive difference evaluated at 100 digits
    with mp.workdps(30):
        z = mp.mpf(10) ** -20
        lo = mp.mpf("0.5")
        hi = lo + mp.mpf(10) ** -15
        got = pr.dangle(z, lo, hi)
    with mp.workdps(100):
        ref = mp.atan(hi / z) - mp.atan(lo / z)          # same (exactly representable) inputs
    assert abs(got - ref) / abs(ref) < mp.mpf(10) ** -25         # relative accuracy, no cancellation
    assert pr.dangle(mp.mpf(0), mp.mpf(-1), mp.mpf(1)) == 0
