# VERIFIED (paper §3, Thm value / Thm grad / Thm recursion, Cor fullspace, Prop lapwall, Prop FEM g_alpha identity):
# independent of the library.  Edge side: value identity + compact-potential recursion with scipy quad on chords (all moments |alpha| <= 6,
# all four kernels, non-convex polygon, particle inside and outside).  Reference: ray (polar) quadrature of the area integral, exact radial integrals of
# piecewise polynomials, 2e6 rays.  Gradient identity and g_alpha = d_x m_alpha + sum alpha_j e_j m_{alpha-e_j}: edge formulas vs finite differences / direct ray quadrature.
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.integrate import quad

# ---- kernels as truncated blocks [(poly in r, cut radius)], h = 1, 2D
q = Poly([0, 1])
def scale(blocks, c): return [(p * c, rc) for p, rc in blocks]
def norm2d(blocks):  # 2 pi int_0^inf t W
    return 2 * np.pi * sum(((p * q).integ()(rc)) for p, rc in blocks)
RAW = {
    "w2": [((1 - q) ** 4 * (1 + 4 * q), 1.0)],
    "w4": [((1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q ** 2), 1.0)],
    "w6": [((1 - q) ** 8 * (1 + 8 * q + 25 * q ** 2 + 32 * q ** 3), 1.0)],
    "cubic": [((1 - q) ** 3, 1.0), (-4 * (Poly([0.5]) - q) ** 3, 0.5)],
}
KERN = {k: scale(b, 1 / norm2d(b)) for k, b in RAW.items()}

def evalb(blocks, r):
    r = np.asarray(r, float)
    return sum(np.where(r <= rc, p(r), 0.0) for p, rc in blocks)

# ---- polygon (CCW, non-convex)
POLY = np.array([[0, 0], [1.6, 0], [1.6, 0.7], [0.9, 0.5], [0.8, 1.5], [0, 1.2]], float)
def edges(P):
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        t = (b - a) / np.linalg.norm(b - a)
        yield a, b, t, np.array([t[1], -t[0]])      # outward normal for CCW
def inside(P, x):
    c = False
    for a, b, *_ in edges(P):
        if (a[1] > x[1]) != (b[1] > x[1]) and x[0] < a[0] + (x[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]): c = not c
    return c

# ---- reference: ray quadrature of  int_T y^alpha g(r) dA  with g given as blocks of the FULL radial integrand r^(|alpha|+1) f(r) (so int dtheta ang int g dr)
def ray(P, x, alpha, gblocks, n=2_000_000):
    th = (np.arange(n) + 0.5) / n * 2 * np.pi
    u = np.stack([np.cos(th), np.sin(th)], 1)
    ts = []
    for a, b, *_ in edges(P):
        e = b - a
        den = u[:, 0] * e[1] - u[:, 1] * e[0]
        w = a - x
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[0] * e[1] - w[1] * e[0]) / den
            sp = (w[0] * u[:, 1] - w[1] * u[:, 0]) / den
        ts.append(np.where((den != 0) & (t > 0) & (sp >= 0) & (sp < 1), t, np.inf))
    ts = np.sort(np.stack(ts, 1), 1)
    if inside(P, x):                                  # rays start inside the solid: first interval is [0, t_1]
        ts = np.concatenate([np.zeros((n, 1)), ts], 1)
    prim = lambda r: sum(np.where(True, (p.integ())(np.minimum(r, rc)), 0.0) for p, rc in gblocks)
    tot = np.zeros(n)
    k = 0
    while k + 1 < ts.shape[1]:
        t1, t2 = ts[:, k], ts[:, k + 1]
        ok = np.isfinite(t2)
        tot += np.where(ok, prim(np.where(ok, t2, 0)) - prim(np.where(ok, t1, 0)), 0.0)
        k += 2
    ang = u[:, 0] ** alpha[0] * u[:, 1] ** alpha[1]
    return np.sum(tot * ang) * 2 * np.pi / n

def times_r(blocks, power): return [(p * Poly([0] * power + [1]), rc) for p, rc in blocks]

# ---- edge side
def chord(z, s0, s1, rc):
    if abs(z) >= rc: return None
    l = np.sqrt(rc * rc - z * z); lo, hi = max(s0, -l), min(s1, l)
    return (lo, hi) if hi > lo else None

def value(P, x, blocks):
    tot = 0.0
    for p, rc in blocks:
        M = (p * q).integ(); Mrc = M(rc)
        if inside(P, x): tot += 2 * np.pi * Mrc
        for a, b, t, n in edges(P):
            z, s0, s1 = n @ (a - x), t @ (a - x), t @ (b - x)
            c = chord(z, s0, s1, rc)
            if c: tot += z * quad(lambda s: (M(np.hypot(s, z)) - Mrc) / (s * s + z * z), *c, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    return tot

def mom(P, x, alpha, blocks):
    """int_T y^alpha f via the recursion (7)."""
    if sum(alpha) == 0: return value(P, x, blocks)
    i = 0 if alpha[0] > 0 else 1
    beta = list(alpha); beta[i] -= 1; beta = tuple(beta)
    Phi = []
    for p, rc in blocks:
        Pp = (p * q).integ(); Phi.append((Pp - Pp(rc), rc))
    tot = 0.0
    for p, rc in Phi:
        for a, b, t, n in edges(P):
            z, s0, s1 = n @ (a - x), t @ (a - x), t @ (b - x)
            c = chord(z, s0, s1, rc)
            if c:
                f = lambda s: (z * n[0] + s * t[0]) ** beta[0] * (z * n[1] + s * t[1]) ** beta[1] * p(np.hypot(s, z))
                tot += n[i] * quad(f, *c, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    if beta[i] > 0:
        b2 = list(beta); b2[i] -= 1
        tot -= beta[i] * mom(P, x, tuple(b2), Phi)
    return tot

def mom_ref(P, x, alpha, blocks, n=2_000_000):
    return ray(P, x, alpha, times_r(blocks, sum(alpha) + 1), n)

if __name__ == "__main__":
    worst = 0.0
    xs = {"inside": np.array([0.7, 0.45]), "outside": np.array([1.2, 1.05]), "near-edge": np.array([0.3, -0.12])}
    alphas = [a for k in range(0, 7) for a in [(k - j, j) for j in range(k + 1)]]
    for kname, blocks in KERN.items():
        for xn, x in xs.items():
            assert (xn != "inside") or inside(POLY, x)
            errs = []
            for al in alphas:
                e = mom(POLY, x, al, blocks); r = mom_ref(POLY, x, al, blocks)
                errs.append(abs(e - r) / max(abs(r), 1e-3))
            worst = max(worst, max(errs))
            print(f"{kname:6s} {xn:9s} recursion k<=6 ({len(alphas)} multi-indices): max rel err {max(errs):.2e}")
    # gradient identity (Cor fullspace) vs central differences of the reference, and g_alpha identity vs direct ray quadrature of y^alpha grad_x W
    x = xs["inside"]; h = 2e-3
    for kname, blocks in KERN.items():
        dW = [(p.deriv(), rc) for p, rc in blocks]                       # W' (continuous at the cuts for all four kernels)
        for al in [(0, 0), (1, 0), (0, 2), (2, 1), (3, 0)]:
            fd = np.array([(mom_ref(POLY, x + h * e, al, blocks) - mom_ref(POLY, x - h * e, al, blocks)) / (2 * h) for e in np.eye(2)])
            ed = np.zeros(2)
            for a, b, t, nn in edges(POLY):
                z, s0, s1 = nn @ (a - x), t @ (a - x), t @ (b - x)
                c = chord(z, s0, s1, 1.0)
                if c: ed -= nn * quad(lambda s: (z * nn[0] + s * t[0]) ** al[0] * (z * nn[1] + s * t[1]) ** al[1] * evalb(blocks, np.hypot(s, z)), *c, epsabs=1e-14, epsrel=1e-12)[0]
            # g_alpha,j = int y^alpha d_xj W = - int y^(alpha+e_j) W'/r  -> radial integrand r^(|alpha|+2) * (W'/r) = r^(|alpha|+1) W'
            g = np.array([-ray(POLY, x, (al[0] + (j == 0), al[1] + (j == 1)), times_r(dW, sum(al) + 1)) for j in range(2)])
            mm1 = np.array([al[j] * (mom(POLY, x, tuple(al[k] - (k == j) for k in range(2)), blocks) if al[j] > 0 else 0.0) for j in range(2)])
            e1, e2 = np.max(np.abs(fd - ed)), np.max(np.abs(g - (ed + mm1)))
            print(f"{kname:6s} alpha={al}: |grad m (FD) - edge| = {e1:.1e},  |g_alpha (ray) - (edge d_x m + sum a m_(a-e))| = {e2:.1e}")
    # Green form of the wall Laplacian: int_T lap W = sum z int_chord W'/r
    for kname, blocks in KERN.items():
        dW = [(p.deriv(), rc) for p, rc in blocks]; ddW = [(p.deriv().deriv(), rc) for p, rc in blocks]
        for xn, x in xs.items():
            # r * (W'' + W'/r) = r W'' + W'
            ref = ray(POLY, x, (0, 0), times_r(ddW, 1) + dW)
            ed = 0.0
            for a, b, t, nn in edges(POLY):
                z, s0, s1 = nn @ (a - x), t @ (a - x), t @ (b - x)
                c = chord(z, s0, s1, 1.0)
                if c: ed += z * quad(lambda s: evalb(dW, np.hypot(s, z)) / np.hypot(s, z), *c, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
            print(f"{kname:6s} {xn:9s} Green form lambda_Delta: edge {ed:+.9f} ray {ref:+.9f} diff {abs(ed-ref):.1e}")
