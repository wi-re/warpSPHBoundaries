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
