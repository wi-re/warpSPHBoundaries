# Direct treatment of axis-aligned boxes (tanks, channels, flumes, baffles): the integral of a radial kernel over a rectangle is four lookups of ONE 2D function,
#   Phi(a, b) = int_{x' <= a, y' <= b} W(|x' - x|) dA'   (a, b relative to the particle, clamped to [-H, H]: Phi = 0 for a or b <= -H, Phi(a, b) = Phi(H, b) for a >= H),
#   int_box W = Phi(x1, y1) - Phi(x0, y1) - Phi(x1, y0) + Phi(x0, y0),
# and the gradient channel is a derivative of the same table: grad_x int_box W(x' - x) = -sum_corners sign * (dPhi/da, dPhi/db).  No edge loop, no cell list, no atan / asinh / log:
# one table per kernel (symmetric Phi(a, b) = Phi(b, a), the three smooth regions of the support), built once by the exact edge machinery.  This study measures the table accuracy.
# Reference: E.value over the polygon [-2, a] x [-2, b] (exact closed form of the library's stage-1 machinery).
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derivation_checks"))
import numpy as np
import paper_edge_identities_k6 as E
from numpy.polynomial import chebyshev as C

B = E.KERN["w2"]                                                           # Wendland C2, H = 1
x0 = np.zeros(2)
def Phi_exact(a, b):
    a, b = min(max(a, -1.0), 1.0), min(max(b, -1.0), 1.0)
    if a <= -1 or b <= -1: return 0.0
    return E.value(np.array([[-2, -2], [a, -2], [a, b], [-2, b]], float), x0, B)

def build(N):
    """Chebyshev tensor tables on the four panels [-1,0], [0,1] per axis (the kernel's r^3 is non-analytic at the origin only: panel corner)."""
    nodes = np.cos(np.pi * (np.arange(N + 1) + 0.5 / 1) / (N + 1))        # Chebyshev points of the first kind, N + 1 of them
    pan = {0: (-1.0, 0.0), 1: (0.0, 1.0)}
    tab = {}
    for i in (0, 1):
        for j in (0, 1):
            (ax0, ax1), (by0, by1) = pan[i], pan[j]
            xs = 0.5 * (ax0 + ax1) + 0.5 * (ax1 - ax0) * nodes; ys = 0.5 * (by0 + by1) + 0.5 * (by1 - by0) * nodes
            V = np.array([[Phi_exact(a, b) for b in ys] for a in xs])
            # Chebyshev coefficients of the interpolant (tensor)
            c = C.chebfit(nodes, V.T, N); c = C.chebfit(nodes, c.T, N)       # c[k, l] over (x, y)
            tab[(i, j)] = c
    return tab, pan

def Phi_tab(tab, pan, a, b, deriv=(0, 0)):
    a, b = np.clip(a, -1, 1), np.clip(b, -1, 1)
    i, j = int(a >= 0), int(b >= 0)
    c = tab[(i, j)]; (ax0, ax1), (by0, by1) = pan[i], pan[j]
    ta, tb = (2 * a - ax0 - ax1) / (ax1 - ax0), (2 * b - by0 - by1) / (by1 - by0)
    cc = c
    for _ in range(deriv[0]): cc = C.chebder(cc, axis=0) * 2 / (ax1 - ax0)
    for _ in range(deriv[1]): cc = C.chebder(cc, axis=1) * 2 / (by1 - by0)
    return C.chebval2d(ta, tb, cc)

def box_lambda(tab, pan, rect, x):
    (X0, Y0), (X1, Y1) = rect; a0, a1, b0, b1 = X0 - x[0], X1 - x[0], Y0 - x[1], Y1 - x[1]
    return Phi_tab(tab, pan, a1, b1) - Phi_tab(tab, pan, a0, b1) - Phi_tab(tab, pan, a1, b0) + Phi_tab(tab, pan, a0, b0)

def box_grad(tab, pan, rect, x):
    """grad_x int_box W(x' - x) dA' = - sum sign (dPhi/da, dPhi/db) at the four corners"""
    (X0, Y0), (X1, Y1) = rect; a0, a1, b0, b1 = X0 - x[0], X1 - x[0], Y0 - x[1], Y1 - x[1]
    g = np.zeros(2)
    for (a, b, sg) in ((a1, b1, 1), (a0, b1, -1), (a1, b0, -1), (a0, b0, 1)):
        g -= sg * np.array([Phi_tab(tab, pan, a, b, (1, 0)), Phi_tab(tab, pan, a, b, (0, 1))])
    return g

def lam_exact(rect, x):
    (X0, Y0), (X1, Y1) = rect
    return E.value(np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], float), x, B)

if __name__ == "__main__":
    rng = np.random.default_rng(2)
    rect = ((0.0, 0.0), (3.0, 2.0))                                       # the box; queries inside, near walls, near corners, outside
    pts = np.concatenate([rng.uniform(-0.6, 0.6, (60, 2)) + [0.0, 0.0], rng.uniform(-0.3, 1.5, (60, 2)), rng.uniform(0, 3, (20, 2)) * [1, 0.6]])
    pts = np.clip(pts, [-1.0, -1.0], [3.5, 3.0])
    for N in (8, 16, 24, 32):
        t0 = time.time(); tab, pan = build(N); tb = time.time() - t0
        el = eg = 0.0
        for x in pts:
            ex = lam_exact(rect, x)
            el = max(el, abs(box_lambda(tab, pan, rect, x) - ex))
            h = 1e-5; g_ref = np.array([(lam_exact(rect, x + [h, 0]) - lam_exact(rect, x - [h, 0])), (lam_exact(rect, x + [0, h]) - lam_exact(rect, x - [0, h]))]) / (2 * h)
            eg = max(eg, np.max(np.abs(box_grad(tab, pan, rect, x) - g_ref)))
        print(f"Chebyshev degree {N:2d} per panel (4 panels, {4 * (N + 1) ** 2} coefficients, built in {tb:5.1f} s): max |lambda - exact| {el:.2e}, max |grad lambda - FD of exact| {eg:.2e}   ({len(pts)} queries)")


# ------------------------------------------------------------------------------------------------------------------ all channels from two tables
# Quadrant integrals F_f(a, b) = int_{y1 <= a, y2 <= b} f(y) dy, y = x' - x.  With W(x - x') radial:  grad_x W = -grad_y W,  so
#   lam = int W                 = corners of Phi
#   m_d = int y_d W             = corners of Phi_d          (Phi_2(a, b) = Phi_1(b, a))
#   g0_j = int d_xj W = -int d_yj W   -> -corners of dPhi/da_j     (int_{y2<=b} d_y1 W dy = int_{y2<=b} W(a, y2) dy2 = dPhi/da)
#   g1_dj = int y_d d_xj W = -int y_d d_yj W = -[boundary term - delta_dj Phi];  boundary term of (d, j) = a_j dPhi/da_j (d = j),  dPhi_d/da_j (d != j)
# i.e. the nine channels of warpbc.edge_channels are first derivatives of TWO 2D tables (value and first moment) at the four corners.
def build2(N):
    nodes = np.cos(np.pi * (np.arange(N + 1) + 0.5) / (N + 1))
    pan = {0: (-1.0, 0.0), 1: (0.0, 1.0)}
    tabs = ({}, {})
    for i in (0, 1):
        for j in (0, 1):
            (ax0, ax1), (by0, by1) = pan[i], pan[j]
            xs = 0.5 * (ax0 + ax1) + 0.5 * (ax1 - ax0) * nodes; ys = 0.5 * (by0 + by1) + 0.5 * (by1 - by0) * nodes
            for k, fn in enumerate((lambda a, b: Phi_exact(a, b), lambda a, b: Phi1_exact(a, b))):
                V = np.array([[fn(a, b) for b in ys] for a in xs])
                c = C.chebfit(nodes, V.T, N); tabs[k][(i, j)] = C.chebfit(nodes, c.T, N)
    return tabs, pan

def Phi1_exact(a, b):
    a, b = min(max(a, -1.0), 1.0), min(max(b, -1.0), 1.0)
    if a <= -1 or b <= -1: return 0.0
    return E.mom(np.array([[-2, -2], [a, -2], [a, b], [-2, b]], float), x0, (1, 0), B)

def tabval(tab, pan, a, b, deriv=(0, 0)):
    return Phi_tab(tab, pan, a, b, deriv)

def channels_box(tabs, pan, rect, x):
    """(lam, m [2], g0 [2], g1 [2,2]) of the rectangle for the particle x, from the corner tables (the library's channel conventions: g0 = int grad_x W, g1[d, j] = int y_d d_xj W)."""
    (X0, Y0), (X1, Y1) = rect; a0, a1, b0, b1 = X0 - x[0], X1 - x[0], Y0 - x[1], Y1 - x[1]
    lam = 0.0; m = np.zeros(2); g0 = np.zeros(2); g1 = np.zeros((2, 2))
    P, P1 = tabs
    for (a, b, sg) in ((a1, b1, 1), (a0, b1, -1), (a1, b0, -1), (a0, b0, 1)):
        ac, bc = float(np.clip(a, -1, 1)), float(np.clip(b, -1, 1))
        Ph, Pa, Pb = tabval(P, pan, a, b), tabval(P, pan, a, b, (1, 0)), tabval(P, pan, a, b, (0, 1))
        P1v, P1a, P1b = tabval(P1, pan, a, b), tabval(P1, pan, a, b, (1, 0)), tabval(P1, pan, a, b, (0, 1))   # Phi_1 = int y1 W
        # Phi_2(a, b) = Phi_1(b, a):  Phi_2, dPhi_2/da = dPhi_1/db at the swapped argument, dPhi_2/db = dPhi_1/da at the swapped argument
        P2v, P2a, P2b = tabval(P1, pan, b, a), tabval(P1, pan, b, a, (0, 1)), tabval(P1, pan, b, a, (1, 0))
        lam += sg * Ph
        m += sg * np.array([P1v, P2v])
        g0 -= sg * np.array([Pa, Pb])
        # int y_d d_yj W over the quadrant:  j = 1: boundary a * dPhi_d/da ... (d = 1: a * Pa - Phi ; d = 2: dPhi_2/da)  etc.
        # (the boundary at y_j = a_j: the integrand y_d W(a_j, .) integrated over the other variable; for d = j the factor y_d = a_j (clamped to the support: 0 beyond it))
        ya, yb = (a if abs(a) <= 1 else 0.0), (b if abs(b) <= 1 else 0.0)
        q = np.array([[ya * Pa - Ph, P1a if False else 0.0], [0.0, yb * Pb - Ph]])
        q[1, 0] = P2a                                  # d = 2, j = 1: boundary y1 = a:  int_{y2<=b} y2 W(a, y2) dy2 = dPhi_2/da
        q[0, 1] = P1b                                  # d = 1, j = 2: boundary y2 = b:  int_{y1<=a} y1 W(y1, b) dy1 = dPhi_1/db
        g1 -= sg * q
    return lam, m, g0, g1
