import random
from fractions import Fraction as F

import mpmath as mp
import pytest


@pytest.fixture(autouse=True)
def _dps():
    old = mp.mp.dps
    mp.mp.dps = 40
    yield
    mp.mp.dps = old


KERNELS = ["cubic", "w2", "w4", "w6"]


def rand_frac(rng, lo=-1, hi=1, den=1000):
    return F(rng.randint(int(lo * den), int(hi * den)), den)


def rand_triangle(rng, scale=1, den=1000):
    while True:
        T = [(rand_frac(rng, -scale, scale, den), rand_frac(rng, -scale, scale, den)) for _ in range(3)]
        A2 = (T[1][0] - T[0][0]) * (T[2][1] - T[0][1]) - (T[1][1] - T[0][1]) * (T[2][0] - T[0][0])
        if abs(A2) > F(scale * scale, 20):
            return T


def rand_point(rng, scale=1, den=1000):
    return (rand_frac(rng, -scale, scale, den), rand_frac(rng, -scale, scale, den))


@pytest.fixture
def rng():
    return random.Random(20261001)

def big_triangle(d):
    """x = origin; the solid is the half plane {y > d} (clipped far away by a huge triangle)."""
    return [(F(-40), d), (F(40), d), (F(0), F(80))]


def tiling(a, n):
    """square [-a, a]^2 split into 2 n^2 triangles (CCW)."""
    tris = []
    xs = [-a + 2 * a * F(i, n) for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            p00, p10, p01, p11 = (xs[i], xs[j]), (xs[i + 1], xs[j]), (xs[i], xs[j + 1]), (xs[i + 1], xs[j + 1])
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
    return tris
