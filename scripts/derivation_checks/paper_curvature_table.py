# VERIFIED (paper/sections/06-curvature.tex, accuracy table): Wendland C4, d = 0.3.  F0, F1, F2 as half-plane integrals (mpmath) and the exact disk by the shell formula.
# Reproduces every printed entry: errors of F0, F0 + k F1, F0 + k F1 + k^2 F2 for k = 1/2 ... 1/16 (F0, F1, F2 = 0.0991734, -0.0199865, 0.0044970).
from mpmath import mp, mpf, quad, acos, sqrt, pi, diff

mp.dps = 15
C2 = 9 / pi
d = mpf("0.3")
Wd = lambda r: C2 * (1 - r) ** 6 * (1 + 6 * r + mpf(35) / 3 * r ** 2) if r < 1 else mpf(0)
phi = lambda s: Wd(sqrt(s))
dphi = lambda k, s: diff(phi, s, k)


def exact(kap):
    R = 1 / mpf(kap)
    D = R + d
    return quad(lambda q: Wd(q) * 2 * q * acos(max(-1, min(1, (q * q + D * D - R * R) / (2 * D * q)))), [d, 1])


def half(fun):
    return quad(lambda y2: quad(lambda y1: fun(y1, y2), [-sqrt(1 - y2 ** 2), 0, sqrt(1 - y2 ** 2)]), [d, 1])


if __name__ == "__main__":
    sg = lambda y1, y2: y1 * y1 + y2 * y2
    F0 = half(lambda y1, y2: phi(sg(y1, y2)))
    F1 = half(lambda y1, y2: dphi(1, sg(y1, y2)) * y1 ** 2 * (2 * d - y2) - (y2 - d) * phi(sg(y1, y2)))
    F2 = half(lambda y1, y2: dphi(1, sg(y1, y2)) * (-d * (y2 - d) * y1 ** 2 - y1 ** 4 / 12)
              + dphi(2, sg(y1, y2)) / 2 * y1 ** 4 * (2 * d - y2) ** 2 - (y2 - d) * dphi(1, sg(y1, y2)) * y1 ** 2 * (2 * d - y2))
    print("F0, F1, F2", F0, F1, F2)
    for kap in [mpf(1) / 2, mpf(1) / 4, mpf(1) / 8, mpf(1) / 16]:
        ex = exact(kap)
        print(kap, float(ex), float(abs(ex - F0)), float(abs(ex - F0 - kap * F1)), float(abs(ex - F0 - kap * F1 - kap ** 2 * F2)))
