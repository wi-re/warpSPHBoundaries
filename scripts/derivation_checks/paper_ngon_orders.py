# VERIFIED (paper/sections/05-closed-boundaries.tex, Prop. ngon): inscribed / circumscribed regular N-gon vs exact disk, Wendland C4, R = 1.5, d = 0.3.
# Both polygon errors fall by 4 per doubling of N, their ratio tends to -2, and (1/3) lam_in + (2/3) lam_circ falls by 16 (N = 32 -> 64).
# Independent of the library: polar (ray) integration  int dtheta [M(t_out) - M(t_in)]  with M(r) = int_0^r t W dt, against the disk shell formula.
import numpy as np
import sympy as sp
from scipy.integrate import quad

C2 = 9 / np.pi
t = sp.symbols("t")
Mp = sp.lambdify(t, sp.integrate(sp.expand(t * C2 * (1 - t) ** 6 * (1 + 6 * t + sp.Rational(35, 3) * t ** 2)), t), "numpy")
M = lambda r: Mp(np.minimum(r, 1.0))


def lam_polygon(N, apothem, phase, R, d, n=4_000_000):
    """kernel integral over a regular N-gon (edge normals at phase + 2 pi k / N, centre 0), particle at (R + d, 0)."""
    x = np.array([R + d, 0.0])
    ang = phase + 2 * np.pi * np.arange(N) / N
    nk = np.stack([np.cos(ang), np.sin(ang)], 1)
    th = (np.arange(n) + 0.5) / n * 2 * np.pi
    u = np.stack([np.cos(th), np.sin(th)], 1)
    nu = u @ nk.T
    rhs = apothem - nk @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = rhs[None, :] / nu
    tin = np.maximum(np.where(nu < 0, tt, -np.inf).max(1), 0)
    tout = np.where(nu > 0, tt, np.inf).min(1)
    ok = tout > tin
    val = np.where(ok, M(np.where(ok, tout, 0)) - M(np.where(ok, tin, 0)), 0.0)
    return val.sum() * 2 * np.pi / n


def lam_disk(R, d):
    D = R + d
    f = lambda q: C2 * (1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q * q) * 2 * q * np.arccos(np.clip((q * q + D * D - R * R) / (2 * D * q), -1, 1))
    return quad(f, d, 1, epsabs=1e-14, epsrel=1e-14, limit=200)[0]


if __name__ == "__main__":
    R, d = 1.5, 0.3
    ld = lam_disk(R, d)
    print("disk", ld)
    for N in [8, 16, 32, 64]:
        a = np.pi / N
        li, lc = lam_polygon(N, R * np.cos(a), 0.0, R, d), lam_polygon(N, R, 0.0, R, d)
        print(N, "in", li - ld, "circ", lc - ld, "ratio", (li - ld) / (lc - ld), "combination", li / 3 + 2 * lc / 3 - ld)
