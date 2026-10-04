"""Reviewer probe for WORK-005 (Q1): flat wall (solid y<0), particle at (0,z), support H=1, Wendland C2 / C4 (2D, normalised).  Plain numpy/scipy, shares no code with edgebound.
 A(z) = int_solid W'(r)/r (yhat.n)^2 dA   -- the pairwise (Monaghan, free-slip mirror) wall coefficient:  acc_pair = fac * 2 u_n * A * n    (A < 0),  fac = alpha c0 H / xi
 B(z) = int_solid lap W dA = z * int_chord W'(r)/r ds   (Green; polar and edge forms are both evaluated and compared)
 Laplacian form: acc_lap = nu_eff * (-2 u_n) * B * n     (mirror ghost v_j - v_i = -2 u_n n, constant n)
 => the nu_eff that makes the two agree at distance z is  nu(z) = fac * |A| / B.   Printed in units of fac."""
import numpy as np
from scipy import integrate
def kern(fam):
    if fam == "w2":
        shape  = lambda q: (1 - q) ** 4 * (1 + 4 * q)
        dshape = lambda q: -20 * q * (1 - q) ** 3
        d2shape = lambda q: -20 * (1 - q) ** 3 + 60 * q * (1 - q) ** 2
    else:
        shape  = lambda q: (1 - q) ** 6 * (35 / 3 * q * q + 6 * q + 1)
        dshape = lambda q: -6 * (1 - q) ** 5 * (35 / 3 * q * q + 6 * q + 1) + (1 - q) ** 6 * (70 / 3 * q + 6)
        d2shape = lambda q: 30 * (1 - q) ** 4 * (35 / 3 * q * q + 6 * q + 1) - 12 * (1 - q) ** 5 * (70 / 3 * q + 6) + (1 - q) ** 6 * 70 / 3
    c = 1 / (2 * np.pi * integrate.quad(lambda q: shape(q) * q, 0, 1, epsabs=1e-14, epsrel=1e-14)[0])
    return (lambda r: c * shape(r)), (lambda r: c * dshape(r)), (lambda r: c * d2shape(r))
def AB(fam, z):
    W, dW, d2W = kern(fam)
    th = lambda r: np.arccos(np.clip(z / r, -1, 1))
    A = integrate.quad(lambda r: dW(r) * (th(r) + np.sin(th(r)) * np.cos(th(r))), z, 1, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    Bp = integrate.quad(lambda r: (d2W(r) + dW(r) / r) * 2 * th(r) * r, z, 1, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    L = np.sqrt(1 - z * z)
    Be = z * integrate.quad(lambda s: dW(np.hypot(s, z)) / np.hypot(s, z), -L, L, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    return A, Bp, Be
if __name__ == "__main__":
    for fam in ("w2", "w4"):
        print(fam, "  z       A(z)          B polar       B edge       |polar-edge|/B   nu(z)/fac   nu(z)*12/fac")
        for z in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99):
            A, Bp, Be = AB(fam, z)
            print(f"   {z:5.2f} {A:13.6e} {Bp:13.6e} {Be:13.6e} {abs(Bp-Be)/abs(Bp):9.1e} {abs(A)/Be:11.5f} {12*abs(A)/Be:10.4f}")
