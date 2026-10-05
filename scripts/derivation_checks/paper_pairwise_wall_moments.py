# E4 of docs/plan-exact-wall-laplacian.md: closed-form wall moments of the PAIRWISE viscosity operator.
#
# The solver's pairwise term for a continuum of solid, a_k = int_S (g(r)/r^2) y_k y_l dv_l dx', g = -W'/r (a polynomial for the Wendland kernels),
# y = x' - x_i, dv = v_ext(x') - v_i = sum_alpha c_alpha y^alpha, needs the tensor moments
#       T_beta = int_S g(r) y^beta / r^2 dx',     |beta| = d = |alpha| + 2.
# Closed form (divergence theorem, no quadrature):  Q = y^beta y_m phi_d(r) has div Q = y^beta [(d+2) phi + r phi'] = y^beta g / r^2  iff
#       phi_d(r) = - r^-(d+2) int_r^H s^(d-1) g(s) ds          (zero for r >= H: compact support, singular r^-(d+2) only at the origin),
# and y.n_out = z is constant along an edge, so
#       T_beta = sum_edges z int_chord y^beta phi_d(r) ds  +  ind(x_i in S) C_d  oint yhat^beta dtheta,     C_d = int_0^H s^(d-1) g ds
# (the second term is the small circle around the origin when x_i lies in S; it equals the full-disk value).  With g = sum_j g_j r^j,
#       phi_d(r) = -C_d r^-(d+2) + sum_j g_j r^(j-2) / (d+j),
# i.e. only the chord integrals J(i, m) = int s^i r^m ds, m >= -(d+2): rational / atan / asinh / log primitives (m = -2 with i = 1 and m = -1 with
# i = 0 are the log and asinh cases; no divergence for z != 0).
# Reference: independent ray quadrature, T_beta = oint yhat^beta [sum over inside radial intervals of P_d(min(r, H))] dtheta, P_d(r) = int_0^r s^(d-1) g ds.
import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial as Poly
mp.mp.dps = 30
q = Poly([0, 1])
KERN = {"w2": (1 - q) ** 4 * (1 + 4 * q), "w4": (1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q ** 2)}      # H = 1
def kernel_g(name):
    W = KERN[name]; W = W / (2 * np.pi * (W * q).integ()(1.0))
    g = -W.deriv() / q if False else None
    c = W.deriv().coef                                  # W' = sum c_k r^k, c_0 = 0 for C2 and smoother
    assert abs(c[0]) < 1e-12
    return Poly(-c[1:])                                 # g = -W'/r (polynomial)

# ---- chord integrals J(i, m) = int_lo^hi s^i (s^2 + z^2)^(m/2) ds, z != 0
def Jprim(i, m, s, z):
    """antiderivative at s"""
    r2 = s * s + z * z; r = mp.sqrt(r2)
    if i >= 2:
        return Jprim(i - 2, m + 2, s, z) - z * z * Jprim(i - 2, m, s, z)
    if i == 1:
        return mp.log(r) if m == -2 else r ** (m + 2) / (m + 2)
    # i == 0
    if m == 0: return s
    if m == -1: return mp.asinh(s / abs(z))
    if m == -2: return mp.atan(s / abs(z)) / abs(z)
    if m <= -3: return ((m + 3) * Jprim(0, m + 2, s, z) - s * r ** (m + 2)) / ((m + 2) * z * z)       # from d/ds[s r^(m+2)] = (m+3) r^(m+2) - (m+2) z^2 r^m
    if m > 0: return s * r ** m / (m + 1) + m * z * z / (m + 1) * Jprim(0, m - 2, s, z)
    raise NotImplementedError((i, m))
def J(i, m, lo, hi, z): return Jprim(i, m, hi, z) - Jprim(i, m, lo, z)

def chord_term(beta, g, H, p, qv, x_clip=True):
    """z * int_chord y^beta phi_d(r) ds for the edge p -> qv (vectors relative to the particle, lengths in units of H = 1)."""
    d = beta[0] + beta[1]
    t = qv - p; ell = np.hypot(*t); t = t / ell; n = np.array([t[1], -t[0]])
    z = float(p @ n); s0 = float(p @ t); s1 = float(qv @ t)
    az = abs(z)
    if az >= H or az < 1e-14: return mp.mpf(0)          # z = 0: the edge line passes through the particle; the term carries the factor z and the chord excludes s = 0
    L = np.sqrt(H * H - z * z); lo = max(s0, -L); hi = min(s1, L)
    if lo >= hi: return mp.mpf(0)
    P = (Poly([0, 1]) ** (d - 1) * g).integ()           # P_d(r) = int_0^r s^(d-1) g
    Cd = P(H)
    # phi_d = -C_d r^-(d+2) + sum_j g_j r^(j-2)/(d+j)
    terms = [(-Cd, -(d + 2))] + [(float(gj) / (d + j), j - 2) for j, gj in enumerate(g.coef) if gj != 0]
    # y^beta = (z n + s t)^beta = prod over components; expand in powers of s by polynomial multiplication
    px = Poly([z * n[0], t[0]]) ** beta[0] * Poly([z * n[1], t[1]]) ** beta[1]
    tot = mp.mpf(0)
    for i, ci in enumerate(px.coef):
        for cj, m in terms:
            tot += mp.mpf(float(ci)) * mp.mpf(float(cj)) * J(i, m, mp.mpf(lo), mp.mpf(hi), mp.mpf(z))
    return z * tot

def inside(poly, pt):
    w = 0
    n = len(poly)
    for k in range(n):
        a, b = poly[k] - pt, poly[(k + 1) % n] - pt
        if a[1] <= 0 < b[1] and a[0] * b[1] - a[1] * b[0] > 0: w += 1
        elif b[1] <= 0 < a[1] and a[0] * b[1] - a[1] * b[0] < 0: w -= 1
    return w

def T_closed(beta, g, poly, H=1.0):
    """poly: counter-clockwise vertices relative to the particle (solid on the left)."""
    d = beta[0] + beta[1]
    tot = mp.mpf(0)
    for k in range(len(poly)):
        tot += chord_term(beta, g, H, poly[k], poly[(k + 1) % len(poly)])
    w = inside(poly, np.zeros(2))
    if w:
        P = (Poly([0, 1]) ** (d - 1) * g).integ(); Cd = P(H)
        th = np.linspace(0, 2 * np.pi, 4097)[:-1]
        ang = np.mean(np.cos(th) ** beta[0] * np.sin(th) ** beta[1]) * 2 * np.pi       # exact for trigonometric polynomials (periodic trapezoid)
        tot += w * Cd * ang
    return float(tot)

def T_ray(beta, g, poly, H=1.0, n_gl=800):
    d = beta[0] + beta[1]
    P = (Poly([0, 1]) ** (d - 1) * g).integ()
    nv = len(poly)
    br = {0.0, 2 * np.pi}
    for k in range(nv):
        a, b = poly[k], poly[(k + 1) % nv]
        br.add(np.arctan2(a[1], a[0]) % (2 * np.pi))
        # edge / circle r = H intersections
        dd = b - a; A = dd @ dd; B = 2 * a @ dd; C = a @ a - H * H
        disc = B * B - 4 * A * C
        if disc > 0:
            for sg in (-1, 1):
                u = (-B + sg * np.sqrt(disc)) / (2 * A)
                if 0 < u < 1:
                    pnt = a + u * dd; br.add(np.arctan2(pnt[1], pnt[0]) % (2 * np.pi))
    br = sorted(br)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    tot = 0.0
    w0 = inside(poly, np.zeros(2))
    for a_, b_ in zip(br[:-1], br[1:]):
        if b_ - a_ < 1e-15: continue
        for x, w in zip(xg, wg):
            th = 0.5 * (a_ + b_) + 0.5 * (b_ - a_) * x
            u = np.array([np.cos(th), np.sin(th)])
            rs = []
            for k in range(nv):
                a, b = poly[k], poly[(k + 1) % nv]
                e = b - a; den = u[0] * e[1] - u[1] * e[0]
                if abs(den) < 1e-14: continue
                tt = (a[0] * e[1] - a[1] * e[0]) / den       # ray parameter
                ss = (a[0] * u[1] - a[1] * u[0]) / den       # edge parameter
                if tt > 0 and 0 <= ss < 1: rs.append(tt)
            rs = sorted(rs)
            # inside intervals: starts inside iff winding(origin) != 0
            pts = ([0.0] if w0 else []) + rs
            if len(pts) % 2: pts.append(1e9)
            rad = sum(P(min(pts[i + 1], H)) - P(min(pts[i], H)) for i in range(0, len(pts), 2))
            tot += 0.5 * (b_ - a_) * w * u[0] ** beta[0] * u[1] ** beta[1] * rad
    return tot

def main():
    rng = np.random.default_rng(3)
    shapes = {
        "triangle, x outside": np.array([[0.3, -0.2], [0.9, 0.1], [0.35, 0.6]]),
        "triangle, x inside": np.array([[-0.5, -0.4], [0.6, -0.3], [0.1, 0.7]]),
        "L-shape, x outside (nearest edge z=0.05)": np.array([[0.05, -0.8], [0.9, -0.8], [0.9, 0.013], [0.5, 0.013], [0.5, 0.6], [0.05, 0.6]]),
        "L-shape, an edge collinear with x (z = 0)": np.array([[0.05, -0.8], [0.9, -0.8], [0.9, 0.0], [0.5, 0.0], [0.5, 0.6], [0.05, 0.6]]),
        "half-plane-like slab, x just outside (z = 1e-3)": np.array([[0.001, -2.0], [3.0, -2.0], [3.0, 2.0], [0.001, 2.0]]),
    }
    betas = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (2, 2), (0, 4)]    # d = 2 (L0 constant ghost), 3 (gradient), 4 (Hessian)
    for kn in KERN:
        g = kernel_g(kn)
        print("kernel", kn, " g = -W'/r coefficients", np.round(g.coef, 4))
        for name, poly in shapes.items():
            worst = 0.0; scale = 0.0
            for be in betas:
                a, b = T_closed(be, g, poly), T_ray(be, g, poly)
                worst = max(worst, abs(a - b)); scale = max(scale, abs(b))
            print("  %-48s max |closed - ray| %.1e  (largest moment %.3g)" % (name, worst, scale))
    # sanity: the full disk (x inside a huge body) gives T = C_d oint yhat^beta;  d = 2: (pi C_2) delta_kl
    g = kernel_g("w2"); big = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], float)
    P = (Poly([0, 1]) * g).integ()
    print("full disk: T_xx %.12f  T_xy %.2e  pi*C_2 %.12f" % (T_closed((2, 0), g, big), T_closed((1, 1), g, big), np.pi * P(1.0)))
    # the L0 free-slip wall acceleration  a_k = int (g/r^2) y_k y_l dv_l with dv = -2 u_n n: flat wall at distance z, solver form vs this
    g = kernel_g("w2")
    for z in (0.1, 0.4, 0.8):
        slab = np.array([[z, -3.0], [4.0, -3.0], [4.0, 3.0], [z, 3.0]])           # solid x > z: normal of the wall pointing into the fluid is -x
        Txx = T_closed((2, 0), g, slab)
        print("  flat wall z=%.1f: T_xx = %.8f (closed form), ray %.8f" % (z, Txx, T_ray((2, 0), g, slab)))

if __name__ == "__main__":
    main()
