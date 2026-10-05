# VERIFIED (paper §6 Remark varcurv, §7 Prop slender (a)): independent of the library, Wendland C4, h = 1.
#  (1) boundary y = -(k/2) x^2 - (a/6) x^3 (solid below), particle at (0, d): kappa = k, kappa' = a at the closest point (0,0).
#      (i)  lambda(k, a) - lambda(k, 0) = O(a^2): no term linear in kappa';
#      (ii) |lambda - (F0 + k F1 + k^2 F2)| = O(k^3) for the parabola, also with a = k^2;
#      (iii) tangential gradient of lambda at the particle = a F1 + O(a k), more precisely a [F1 + k (2 F2 - d F1)] -- the term the normal-only gradient of Thm curv omits.
#  (2) disk of radius a at distance D: series error  ~ a^(2K+4) for K = 0, 1, 2.
import numpy as np
from scipy.integrate import quad
from mpmath import mp, mpf, quad as mquad, diff as mdiff, sqrt as msqrt

C2 = 9 / np.pi
W = lambda r: C2 * (1 - r) ** 6 * (1 + 6 * r + 35 / 3 * r ** 2) if r < 1 else 0.0
F0, F1, F2 = 0.0991733904185681, -0.0199865486730563, 0.00449697345143767      # d = 0.3, from paper_curvature_table.py

def lam(k, a, d=0.3, px=0.0):
    f = lambda x: -(k / 2) * x ** 2 - (a / 6) * x ** 3
    def col(x):
        dx = x - px
        if abs(dx) >= 1: return 0.0
        L = np.sqrt(1 - dx * dx)
        top = min(f(x), d + L); bot = d - L
        if top <= bot: return 0.0
        return quad(lambda y: W(np.hypot(dx, y - d)), bot, top, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
    return quad(col, px - 1, px + 1, epsabs=1e-14, epsrel=1e-12, limit=400, points=[px])[0]

print("(1)(i)  kappa' enters only quadratically: lambda(k,a) - lambda(k,0)")
for k in [0.25]:
    for a in [0.2, 0.1, 0.05]:
        print(f"   k={k} a={a}:  {lam(k, a) - lam(k, 0):+.3e}   (a^2 = {a*a:.1e})")
print("(1)(ii) parabola vs expansion, a = k^2:")
prev = None
for k in [0.25, 0.125, 0.0625]:
    e = lam(k, k * k) - (F0 + k * F1 + k * k * F2)
    print(f"   k={k}: error {e:+.3e}" + ("" if prev is None else f"   ratio {prev/e:.1f} (8 expected)")); prev = e
print("(1)(iii) tangential gradient [lambda(+eps) - lambda(-eps)]/(2 eps) vs a F1 (leading), and vs a [F1 + k (2 F2 - d F1)]  (d_x kappa = a/(1+k d): the closest point moves by eps/(1+k d)):")
for k, a in [(0.25, 0.0625), (0.125, 0.015625), (0.125, 0.1)]:
    eps = 1e-3
    g = (lam(k, a, px=eps) - lam(k, a, px=-eps)) / (2 * eps)
    p1, p2 = a * F1, a * (F1 + k * (2 * F2 - 0.3 * F1))
    print(f"   k={k} a={a}: measured {g:+.4e}   a F1 {p1:+.4e} (diff {g-p1:+.1e})   corrected {p2:+.4e} (diff {g-p2:+.1e})")

print("(2) disk series (paper Prop slender (a)), W = Wendland C4 at distance D = 0.5")
mp.dps = 40
Wm = lambda r: mpf(9) / mp.pi * (1 - r) ** 6 * (1 + 6 * r + mpf(35) / 3 * r ** 2)
Dd = mpf("0.5")
def lapk(k, x, y):                       # Laplacian^k of W(|.|) at (x, y) via nested mp.diff on the planar function (small k only)
    g = lambda X, Y: Wm(msqrt(X * X + Y * Y))
    def L(f): return lambda X, Y: mdiff(f, (X, Y), (2, 0)) + mdiff(f, (X, Y), (0, 2))
    f = g
    for _ in range(k): f = L(f)
    return f(x, y)
lk = [lapk(k, Dd, mpf(0)) for k in range(3)]
for K in [0, 1, 2]:
    prev = None
    for a in [mpf("0.2"), mpf("0.1"), mpf("0.05")]:
        exact = mquad(lambda rho: mquad(lambda ph: Wm(msqrt((Dd + rho * mp.cos(ph)) ** 2 + (rho * mp.sin(ph)) ** 2)), [0, 2 * mp.pi]) * rho, [0, a])
        ser = mp.pi * a * a * sum((a * a / 4) ** k / (mp.factorial(k) * mp.factorial(k + 1)) * lk[k] for k in range(K + 1))
        e = abs(exact - ser)
        print(f"   K={K} a={float(a)}: err {float(e):.3e}" + ("" if prev is None else f"   ratio {float(prev/e):.1f} (expected {2**(2*K+4)})")); prev = e
