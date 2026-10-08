"""Checks of docs/derivations/disk-element.md (the disk / fibre element).  Run:  python scripts/derivation_checks/disk_element_checks.py

(1) the circle identities (lam, m1x, g0x, g1xx, g1yy) of `edge.disk.disk_channels` against an independent polar (ray) integration from the query with the Wendland C2 kernel written out here
    (7/pi (1-q)^4 (4q+1), support 1): inside, outside, near the surface, R < 1 and R > 1;
(2) the non-analytic term at the surface: sympy expands the planar channel g0x(eps) = int W(sqrt(eps^2 + t^2)) dt and finds  -(3/4) c3 eps^4 ln|eps|  with c3 = 140/pi the cubic coefficient of the kernel;
    the Chebyshev coefficients of g0x(delta) on ONE panel across delta = 0 decay algebraically, split at delta = 0 geometrically;
(3) the radius branches: the lines delta = 0 and D = 1 - R cross at R = 1/2; the point limit lam / R^2 -> pi W(D)... (W normalised to 1) as R -> 0 is regular;
(4) the end of the support D = 1 + R: lam ~ eta^(11/2), g0x ~ eta^(9/2) (the sliver of area eta^(3/2) times the (1-r)^4 of the kernel).
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad

from warpSPHBoundaries.edge.disk import disk_channels

KN = "w2"
Wf = lambda r: np.where(r < 1.0, 7.0 / np.pi * (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
dWf = lambda r: np.where(r < 1.0, 7.0 / np.pi * (-4.0 * (1.0 - r) ** 3 * (4.0 * r + 1.0) + 4.0 * (1.0 - r) ** 4), 0.0)
r_ = sp.symbols("r", positive=True)
_W = sp.Rational(7) / sp.pi * (1 - r_) ** 4 * (4 * r_ + 1)
P0 = sp.lambdify(r_, sp.integrate(sp.expand(_W * r_), r_), "numpy")                      # int W r dr
P1 = sp.lambdify(r_, sp.integrate(sp.expand(_W * r_ ** 2), r_), "numpy")                 # int W r^2 dr
Q1 = sp.lambdify(r_, sp.integrate(sp.expand(sp.diff(_W, r_) * r_ ** 2), r_), "numpy")    # int W' r^2 dr
Q0 = sp.lambdify(r_, sp.integrate(sp.expand(sp.diff(_W, r_) * r_), r_), "numpy")         # int W' r dr


def polar(D, R):
    """the five channels by integrating over the angle theta at the query the radial integrals (exact polynomial antiderivatives) between the entry and exit of the ray through the disk,
    both clipped to the support: lam = int dth int W r dr; m1x = int cos int W r^2 dr; g0x = - int cos int W' r dr; g1xx = - int cos^2 int W' r^2 dr, g1yy = - int sin^2 (same)."""
    inside = D < R

    def seg(th, F):
        s = D * D * np.sin(th) ** 2
        if R * R - s < 0:
            return 0.0
        q = np.sqrt(R * R - s)
        t_out = min(D * np.cos(th) + q, 1.0)
        t_in = 0.0 if inside else max(D * np.cos(th) - q, 0.0)
        return float(F(t_out) - F(t_in)) if t_out > t_in else 0.0

    th_max = np.pi if inside else np.arcsin(min(R / D, 1.0))
    pts = [0.0] if inside else [-th_max, 0.0]
    brk = sorted(set([0.0, th_max] + [a for kn in (1.0,) for a in ([np.arccos(np.clip((kn * kn + D * D - R * R) / (2 * kn * D), -1, 1))] if D > 0 else [])]))
    lim = (0.0, th_max)
    out = []
    for f in (lambda th: seg(th, P0), lambda th: np.cos(th) * seg(th, P1), lambda th: -np.cos(th) * seg(th, Q0),
              lambda th: -np.cos(th) ** 2 * seg(th, Q1), lambda th: -np.sin(th) ** 2 * seg(th, Q1)):
        val = 2.0 * sum(quad(f, a, b, epsabs=1e-13, epsrel=1e-13, limit=400)[0] for a, b in zip(brk[:-1], brk[1:]) if b > a)
        out.append(val)
    return np.array(out)


def lib(D, R):
    o = disk_channels(KN, np.array([D]), np.array([R]))
    return np.array([o[c][0] for c in ("lam", "m1x", "g0x", "g1xx", "g1yy")])


def check_identities():
    print("(1) circle identities vs polar ray integration (max abs difference over the five channels)")
    worst = 0.0
    for D, R in [(0.3, 0.1), (0.5, 0.2), (0.95, 0.2), (0.15, 0.2), (0.6, 0.55), (0.62, 0.6), (0.9 * (1 + 1e-5), 0.9), (0.9 * (1 - 1e-5), 0.9), (1.2, 0.5), (0.4, 1.7), (1.5, 1.0), (2.45, 1.5), (3.0, 2.5), (2.4, 2.5), (2.6, 2.5), (5.0, 6.0)]:
        e = np.abs(lib(D, R) - polar(D, R)).max()
        worst = max(worst, e)
        print(f"    D = {D:5.2f} R = {R:4.2f}: {e:.2e}")
    assert worst < 5e-9, worst
    # exactly ON the circle the reference is not valid: the circle integrand is singular at the query and the quadrature returns the principal value, half of the jump of the indicator
    on = np.abs(lib(0.9, 0.9) - polar(0.9, 0.9)).max()
    print(f"    exactly on the surface (D = R): difference {on:.3f} (= 1/2: the indicator jump; the functions are continuous, evaluate at |D - R| >= 1e-9)")
    assert abs(on - 0.5) < 1e-6
    # conditioning near the surface: the circle integrand is ~ 1/|D - R|, so the absolute error of the reference is ~ 6e-17 R / |D - R| (measured 1e-3 ... 1e-9 relative offsets)
    for off in (1e-5, 1e-7, 1e-9):
        e = np.abs(lib(0.9 * (1 + off), 0.9) - polar(0.9 * (1 + off), 0.9)).max()
        print(f"    relative offset {off:.0e}: {e:.1e}   (6e-17 / offset = {6e-17 / off:.1e})")
        assert e < 20 * 6e-17 / off
    return worst


def check_surface_term():
    print("(2) the eps^4 ln|eps| term")
    eps, T, t = sp.symbols("epsilon T t", positive=True)
    c = sp.Poly(sp.expand(_W.subs(r_, sp.Symbol("q"))), sp.Symbol("q")).all_coeffs()[::-1]
    c3 = c[3]
    assert sp.simplify(c3 - 140 / sp.pi) == 0
    # the odd powers carry the logs: int_{-T}^{T} (eps^2 + t^2)^(k/2) dt
    L = sp.Symbol("L")                                                                          # L = ln(eps): asinh(1/eps) = ln(1 + sqrt(1 + eps^2)) - L
    for k in (3, 5):
        G = sp.integrate((eps ** 2 + t ** 2) ** sp.Rational(k, 2), t)                             # antiderivative in t
        F = 2 * G.subs(t, 1)                                                                      # int_{-1}^{1} (T = 1: the cut at sqrt(1 - eps^2) changes only analytic parts)
        F = F.replace(sp.asinh, lambda a: sp.log(1 + sp.sqrt(1 + eps ** 2)) - L if sp.simplify(a - 1 / eps) == 0 else sp.asinh(a))
        assert not F.has(sp.asinh), F
        ser = sp.expand(sp.series(F, eps, 0, k + 3).removeO())
        coef = sp.simplify(ser.coeff(L, 1))
        print(f"    k = {k}: log part of int_(-1)^1 (eps^2 + t^2)^(k/2) dt = ({coef}) ln(eps)")
        if k == 3:
            assert sp.simplify(coef - (-sp.Rational(3, 4) * eps ** 4)) == 0
        if k == 5:
            assert sp.simplify(coef - (-sp.Rational(5, 8) * eps ** 6)) == 0
    lead = -sp.Rational(3, 4) * c3
    print(f"    => g0x(eps) = ... {lead} eps^4 ln|eps| + O(eps^6 ln|eps|)   (planar limit; curvature changes higher orders)")
    # numerical confirmation on the exact planar function F(eps) = int W(sqrt(eps^2 + t^2)) dt
    Fp = lambda e: 2.0 * quad(lambda s: float(Wf(np.sqrt(e * e + s * s))), 0.0, np.sqrt(1.0 - e * e), epsabs=1e-15, epsrel=1e-14, limit=200)[0]
    es = np.array([2e-2, 1e-2, 5e-3])
    # remove the analytic part by Richardson on even powers: fit F = a0 + a2 e^2 + a4 e^4 + b e^4 ln e + a6 e^6 ... (least squares on 12 points)
    E = np.geomspace(2e-3, 6e-2, 24)
    A = np.stack([np.ones_like(E), E ** 2, E ** 4, E ** 4 * np.log(E), E ** 6, E ** 6 * np.log(E), E ** 8], 1)
    sol = np.linalg.lstsq(A, np.array([Fp(e) for e in E]), rcond=None)[0]
    print(f"    planar fit: coefficient of eps^4 ln eps = {sol[3]:.5f}  (predicted {float(lead):.5f})")
    assert abs(sol[3] - float(lead)) < 2e-3 * abs(float(lead))
    # Chebyshev coefficient decay of g0x(delta) at R = 0.7, delta in [-0.2, 0.2] (n = 40 nodes): (a) ONE panel across the surface, (b) split at the surface (the singularity sits on a panel END),
    # (c) split + the endpoint map u = sin^2(pi w / 2) of the tables (the interval end is then a high-order zero in w)
    R = 0.7
    n = 40
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)

    def coeffs(lo, hi, mapped=False):
        w = 0.5 * (x + 1.0)
        u = np.sin(0.5 * np.pi * w) ** 2 if mapped else w
        d = lo + u * (hi - lo)
        return np.abs(np.polynomial.chebyshev.chebfit(x, disk_channels(KN, R + d, np.full_like(d, R))["g0x"], n - 1))

    ks = (4, 8, 12, 16, 20, 28)
    rows = {"(a) one panel across delta = 0": coeffs(-0.2, 0.2), "(b) split at delta = 0": coeffs(0.0, 0.2), "(c) split + sin^2 map": coeffs(0.0, 0.2, True)}
    for name, c in rows.items():
        print(f"    {name:32s} |c_n|, n = {ks}: " + " ".join(f"{c[k]:.1e}" for k in ks))
    sl = lambda c: np.polyfit(np.log([8, 12, 16, 20]), np.log(c[[8, 12, 16, 20]]), 1)[0]
    sa, sb = sl(rows["(a) one panel across delta = 0"]), sl(rows["(b) split at delta = 0"])
    print(f"    algebraic decay exponents: (a) {sa:.1f} (an eps^4 ln eps singularity inside the panel: n^-5), (b) {sb:.1f} (the same singularity on the panel end: n^-10)")
    assert -6.5 < sa < -4.0 and sb < -8.0 and rows["(c) split + sin^2 map"][28] < 1e-13 and rows["(c) split + sin^2 map"][28] < 1e-3 * rows["(b) split at delta = 0"][28]
    return sol[3]


def check_branches():
    print("(3) radius branches and the point limit")
    # lines delta = 0 and D = 1 - R, i.e. delta = 1 - 2R, meet at R = 1/2
    R = sp.symbols("R")
    assert sp.solve(sp.Eq(0, 1 - 2 * R), R) == [sp.Rational(1, 2)]
    print("    delta = 0 and delta = 1 - 2R cross at R = 1/2: the order of the intervals swaps")
    for D in (0.2, 0.5, 0.8):
        l = lib(D, 1e-3)[0] / 1e-6
        ref = np.pi * float(Wf(np.array(D)))                                                       # lam -> pi R^2 W(D) with int W = 1
        print(f"    D = {D}: lam / R^2 at R = 1e-3 = {l:.6f}   pi W(D) = {ref:.6f}")
        assert abs(l - ref) < 1e-3 * ref                                                      # O(R^2) corrections (second derivatives of W)
    # f / R^2 is analytic in R: Chebyshev decay in R at fixed u = (delta - lo) / (hi - lo) on the middle interval of branch 0 (R in (0, 1/2), delta in [0, 1 - 2R])
    n = 24
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    Rn = 0.25 * (x + 1.0)
    D = Rn + 0.3 * (1.0 - 2.0 * Rn)
    f = disk_channels(KN, D, Rn)["lam"] / Rn ** 2
    c = np.abs(np.polynomial.chebyshev.chebfit(x, f, n - 1))
    print(f"    lam / R^2 in R (fixed interval fraction): |c_n| n = 4, 8, 12, 16 : {[f'{c[k]:.1e}' for k in (4, 8, 12, 16)]}")
    assert c[16] < 1e-9
    g = disk_channels(KN, D, Rn)["lam"]
    cg = np.abs(np.polynomial.chebyshev.chebfit(x, g, n - 1))
    print(f"    (lam itself, without the R^2 scaling, R in (0, 1/4): n = 16 {cg[16]:.1e})")


def check_support_end():
    print("(4) the end of the support")
    R = 0.3
    eta = np.array([4e-2, 2e-2, 1e-2, 5e-3])
    o = [disk_channels(KN, np.array([1.0 + R - e]), np.array([R])) for e in eta]
    for ch, expect in (("lam", 5.5), ("g0x", 4.5)):
        v = np.array([abs(x[ch][0]) for x in o])
        s = np.polyfit(np.log(eta), np.log(v), 1)[0]
        print(f"    {ch}: |{ch}| ~ eta^{s:.2f}   (predicted {expect})")
        assert abs(s - expect) < 0.15, (ch, s)


if __name__ == "__main__":
    a = check_identities()
    b = check_surface_term()
    check_branches()
    check_support_end()
    print("ALL CHECKS PASSED")
