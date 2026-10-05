# VERIFIED (paper §9, Prop nueff and the Laplacian estimators): independent of the library.
#  (1) the solver's pairwise form  a_i = fac sum_j V_j [(v_i-v_j).x_ij/|x_ij|^2] grad_i W_ij  on a square lattice (h/dx = 16) against fac/8 (lap v + 2 grad div v),
#      for all four kernels and generic quadratic velocity fields; also against its own continuum integral (polar Gauss quadrature);
#  (2) lap v ~ int (v' - v) lap W dx'  and  lap v ~ 2 int (v - v') W'/r dx' (Morris) converge as O(h^2).
import numpy as np
from numpy.polynomial import Polynomial as Poly
q = Poly([0, 1])
RAW = {"w2": [((1 - q) ** 4 * (1 + 4 * q), 1.0)], "w4": [((1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q ** 2), 1.0)],
       "w6": [((1 - q) ** 8 * (1 + 8 * q + 25 * q ** 2 + 32 * q ** 3), 1.0)], "cubic": [((1 - q) ** 3, 1.0), (-4 * (Poly([0.5]) - q) ** 3, 0.5)]}
def norm(b): return 2 * np.pi * sum((p * q).integ()(rc) for p, rc in b)
KERN = {k: [(p / norm(b), rc) for p, rc in b] for k, b in RAW.items()}
def ev(b, r, d=0):
    r = np.asarray(r, float); return sum(np.where(r <= rc, p.deriv(d)(r) if d else p(r), 0.0) for p, rc in b)

def main():
    # quadratic field v(x) = c + G x + 1/2 x^T H x   (H[k] symmetric, one per component)
    rng = np.random.default_rng(1)
    G = rng.normal(size=(2, 2)); H = rng.normal(size=(2, 2, 2)); H = (H + H.transpose(0, 2, 1)) / 2
    v = lambda y: (G @ y.T).T + 0.5 * np.einsum("kab,na,nb->nk", H, y, y)          # v(x0 + y) - v(x0) with x0 = 0 (v(0) = 0)
    lap = np.array([H[k, 0, 0] + H[k, 1, 1] for k in range(2)])
    graddiv = np.array([sum(H[k, k, i] for k in range(2)) for i in range(2)])      # d_i d_k v_k = sum_k H[k,k,i]
    pred = (lap + 2 * graddiv) / 8                                                      # fac = 1

    print("(1) pairwise bulk term, lattice h/dx=16, vs fac/8 (lap v + 2 grad div v)")
    dx = 1 / 16
    g = np.arange(-17, 18) * dx
    X, Y = np.meshgrid(g, g); P = np.stack([X.ravel(), Y.ravel()], 1); r = np.hypot(P[:, 0], P[:, 1]); m = (r > 0) & (r < 1)
    y = P[m]; r = r[m]
    for k, b in KERN.items():
        dv = v(y)                                                                        # v_j - v_i, y = x_j - x_i
        mu = (dv * y).sum(1) / r ** 2                                                    # (v_i - v_j).(x_i - x_j)/r^2 = dv.y/r^2 (both factors flip sign)
        gradW = ev(b, r, 1)[:, None] * (-y / r[:, None])                                 # grad_i W_ij = W'(r) (x_i - x_j)/r
        a = dx ** 2 * (mu[:, None] * gradW).sum(0)
        # continuum integral (Gauss-Legendre in r, uniform in theta)
        xr, wr = np.polynomial.legendre.leggauss(200); xr = (xr + 1) / 2; wr = wr / 2
        th = (np.arange(256) + 0.5) / 256 * 2 * np.pi
        R_, T_ = np.meshgrid(xr, th); Y2 = np.stack([(R_ * np.cos(T_)).ravel(), (R_ * np.sin(T_)).ravel()], 1); rr = R_.ravel()
        W_ = np.tile(wr, len(th)) * rr * (2 * np.pi / 256)
        mu2 = (v(Y2) * Y2).sum(1) / rr ** 2
        a2 = ((W_ * mu2)[:, None] * (ev(b, rr, 1)[:, None] * (-Y2 / rr[:, None]))).sum(0)
        print(f"  {k:6s} lattice {np.round(a,5)}  continuum {np.round(a2,6)}  predicted {np.round(pred,6)}   |lat-pred| {np.max(np.abs(a-pred)):.1e}  |cont-pred| {np.max(np.abs(a2-pred)):.1e}")

    print("(2) Laplacian estimators for v = cos(x) cos(2y) at x0 = (0.3, 0.2), W_h(r) = W(r/h)/h^2")
    f = lambda x, y_: np.cos(x) * np.cos(2 * y_); exact = -5 * f(0.3, 0.2)
    xr, wr = np.polynomial.legendre.leggauss(300); xr = (xr + 1) / 2; wr = wr / 2
    th = (np.arange(512) + 0.5) / 512 * 2 * np.pi
    R_, T_ = np.meshgrid(xr, th); rr = R_.ravel(); TT = T_.ravel(); w0 = np.tile(wr, len(th)) * 2 * np.pi / 512
    for k in ["w2", "w4", "cubic"]:
        b = KERN[k]; prev = None
        for h in [0.4, 0.2, 0.1]:
            rh = rr * h; area = w0 * rh * h                                              # dA = r dr dtheta, r = h*rr
            dv = f(0.3 + rh * np.cos(TT), 0.2 + rh * np.sin(TT)) - f(0.3, 0.2)
            Wp = ev(b, rr, 1) / h ** 3; Wpp = ev(b, rr, 2) / h ** 4
            est1 = np.sum(area * dv * (Wpp + Wp / rh))
            est2 = -2 * np.sum(area * dv * Wp / rh)
            e1, e2 = abs(est1 - exact), abs(est2 - exact)
            print(f"  {k:6s} h={h}: int (v'-v) lap W err {e1:.2e}, Morris err {e2:.2e}" + ("" if prev is None else f"   ratios {prev[0]/e1:.1f}, {prev[1]/e2:.1f}"))
            prev = (e1, e2)


if __name__ == "__main__":
    main()
