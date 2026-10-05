# E5 of docs/plan-exact-wall-laplacian.md: where the free-slip mirror ghost fails at geometry beyond a flat wall -- a right-angle corner and a curved (circular) wall.
#  Solid part of the wall Laplacian  int_S (v_ghost - v_i) lap W dA'  with  v_ghost(x_i + y) = R v(x_i + y_m)  Taylor-truncated at order p in the MIRROR displacement
#  y_m = M(x_i + y) - x_i  (as paper_exact_wall_mirror.py), moments of lap W over polygon regions by Green's second identity (closed form).
#  A: fluid quadrant x, y > 0 (corner at the origin), free-slip symmetric cubic field v_x = x f(x^2, y^2), v_y = y g(x^2, y^2).  Estimators for the solid part:
#     images   region-wise exact images: behind wall x = 0 mirror in x, behind y = 0 mirror in y, corner quadrant point reflection
#     flat     the solid is mirrored as ONE flat wall (the plane of the nearest wall), the other wall ignored
#     double   each wall mirrors its whole half-plane (the corner quadrant is counted twice)
#  B: circular wall (fluid inside the disk of radius Rc), mirror across the tangent line at the closest wall point, free-slip swirl field v = (-y, x)(1 + a r^2 + b r^4)
#     (v_r = 0 and zero shear at r = Rc), against the analytic continuation of the field into the solid; polar quadrature (the solid is a smooth region, no polygon).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sympy as sp
from scipy.integrate import quad
from numpy.polynomial import Polynomial as Poly
import paper_edge_identities_k6 as E

B4 = E.KERN["w4"]                                                    # support H = 1
y1, y2, zz1, zz2, X, Y = sp.symbols("y1 y2 zz1 zz2 X Y")

def lapW_blocks(blocks, extra):
    dW = [(p.deriv(), rc) for p, rc in blocks]; ddW = [(p.deriv().deriv(), rc) for p, rc in blocks]
    sh = lambda bl, k: [(p * Poly([0] * k + [1]), rc) for p, rc in bl]
    return sh(ddW, extra + 1) + sh(dW, extra)

_cache = {}
def moment(P, x, alpha, blocks=B4):
    """int_P y^alpha lap W (y = x' - x) by Green's second identity: sum_e int_chord [y^a W' z/r - W n.grad y^a] ds + int_P W lap y^a."""
    key = (P.tobytes(), tuple(x), alpha)
    if key in _cache: return _cache[key]
    a1, a2 = alpha; tot = 0.0
    dW = [(p.deriv(), rc) for p, rc in blocks]
    for a, b, t, n in E.edges(P):
        z, s0, s1 = n @ (a - x), t @ (a - x), t @ (b - x); c = E.chord(z, s0, s1, 1.0)
        if not c: continue
        def f(s):
            yv = z * n + s * t; r = np.hypot(s, z)
            ya = yv[0] ** a1 * yv[1] ** a2
            gr = np.array([a1 * yv[0] ** max(a1 - 1, 0) * yv[1] ** a2 if a1 else 0.0, a2 * yv[0] ** a1 * yv[1] ** max(a2 - 1, 0) if a2 else 0.0])
            return ya * E.evalb(dW, r) * z / r - E.evalb(blocks, r) * (n @ gr)
        tot += quad(f, *c, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
    if a1 >= 2: tot += a1 * (a1 - 1) * E.mom(P, x, (a1 - 2, a2), blocks)
    if a2 >= 2: tot += a2 * (a2 - 1) * E.mom(P, x, (a1, a2 - 2), blocks)
    _cache[key] = tot
    return tot

def monos(expr):
    Pp = sp.Poly(sp.expand(expr), y1, y2); return {m: float(c) for m, c in Pp.terms()}

def solid_part(P, x, v, Mdiag, p):
    """int_P (v_ghost - v(x)) lap W, v_ghost(x + y) = R v(x + y_m), R = M diagonal, y_m = M (x + y) - x, v Taylor-truncated at order p in the displacement."""
    vx0 = [float(c.subs({X: x[0], Y: x[1]})) for c in v]
    vz = [sp.expand(c.subs({X: x[0] + zz1, Y: x[1] + zz2}, simultaneous=True)) for c in v]
    def trunc(e):
        Pp = sp.Poly(e, zz1, zz2); return sum(c * zz1 ** m[0] * zz2 ** m[1] for m, c in Pp.terms() if sum(m) <= p)
    ym = [Mdiag[0] * (x[0] + y1) - x[0], Mdiag[1] * (x[1] + y2) - x[1]]
    vg = [Mdiag[i] * trunc(vz[i]).subs({zz1: ym[0], zz2: ym[1]}, simultaneous=True) for i in range(2)]
    return np.array([sum(cf * moment(P, x, m) for m, cf in monos(sp.expand(vg[i] - vx0[i])).items()) for i in range(2)])

def fluid_part(P, x, v):
    vx0 = [float(c.subs({X: x[0], Y: x[1]})) for c in v]
    d = [sp.expand(c.subs({X: x[0] + y1, Y: x[1] + y2}, simultaneous=True) - vx0[i]) for i, c in enumerate(v)]
    return np.array([sum(cf * moment(P, x, m) for m, cf in monos(e).items()) for e in d])

rect = lambda x0, x1, y0, y1_: np.array([[x0, y0], [x1, y0], [x1, y1_], [x0, y1_]], float)

def part_A():
    c = dict(c1=0.8, c2=0.5, c3=-0.7, d1=1.1, d2=-0.4, d3=0.6)
    v = [c["c1"] * X + c["c2"] * X ** 3 + c["c3"] * X * Y ** 2, c["d1"] * Y + c["d2"] * Y ** 3 + c["d3"] * X ** 2 * Y]
    lap = lambda x: np.array([float((sp.diff(e, X, 2) + sp.diff(e, Y, 2)).subs({X: x[0], Y: x[1]})) for e in v])
    F = rect(0, 3, 0, 3)
    RA, RB, RC = rect(-3, 0, 0, 3), rect(0, 3, -3, 0), rect(-3, 0, -3, 0)
    LSHAPE = np.array([[-3, -3], [3, -3], [3, 0], [0, 0], [0, 3], [-3, 3]], float)
    HP_A, HP_B = rect(-3, 0, -3, 3), rect(-3, 3, -3, 0)
    MA, MB, MC = (-1, 1), (1, -1), (-1, -1)
    print("A. right-angle corner, fluid quadrant x, y > 0, kernel w4 (H = 1), symmetric cubic field; error of the total lap v (max over components) and of the solid part alone")
    print("   particle (x, y)      | p | images    flat      double   |  |solid part| (images)")
    for x in [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (0.8, 0.8), (1.1, 1.1), (0.15, 0.3), (0.15, 0.8), (0.15, 1.5)]:
        x = np.array(x, float); fl = fluid_part(F, x, v); ex = lap(x)
        for p in (0, 2, 3):
            im = solid_part(RA, x, v, MA, p) + solid_part(RB, x, v, MB, p) + solid_part(RC, x, v, MC, p)
            near = MA if x[0] <= x[1] else MB
            flat = solid_part(LSHAPE, x, v, near, p)
            dbl = solid_part(HP_A, x, v, MA, p) + solid_part(HP_B, x, v, MB, p)
            e = lambda s: np.max(np.abs(fl + s - ex))
            print(f"   ({x[0]:.2f}, {x[1]:.2f})  |x|={np.hypot(*x):.2f} | {p} | {e(im):.2e}  {e(flat):.2e}  {e(dbl):.2e}  | {np.max(np.abs(im)):.3f}")

# ---------------------------------------------------------------------------------------------------------------- B: circular wall
def part_B():
    xg, wg = np.polynomial.legendre.leggauss(40)
    dW1 = [(p.deriv(), rc) for p, rc in B4]; dW2 = [(p.deriv().deriv(), rc) for p, rc in B4]
    lapW = lambda r: E.evalb(dW2, r) + E.evalb(dW1, r) / r                        # W'' + W'/r  (H = 1)
    def solid_integral(Rc, x, fn, nth=3000):
        """int over {|x'| > Rc, |x' - x| <= 1} of fn(y) lapW(|y|) dA' (y = x' - x, x inside the disk): on every ray the solid is r > r+, r+ the root of |x + r u| = Rc."""
        th = (np.arange(nth) + 0.5) / nth * 2 * np.pi
        u = np.stack([np.cos(th), np.sin(th)], 1)
        bq = u @ x; cq = x @ x - Rc ** 2
        rp = -bq + np.sqrt(bq * bq - cq)
        keep = rp < 1
        u, rp = u[keep], rp[keep]
        r = 0.5 * (rp[:, None] + 1) + 0.5 * (1 - rp[:, None]) * xg[None, :]       # [T, G]
        w = 0.5 * (1 - rp[:, None]) * wg[None, :]
        y = r[:, :, None] * u[:, None, :]                                         # [T, G, 2]
        f = fn(y.reshape(-1, 2)).reshape(r.shape + (2,))
        return np.einsum("tg,tgc->c", w * r * lapW(r), f) * 2 * np.pi / nth
    print("B. circular wall (fluid inside r < Rc), x at distance d = 0.3 from the wall, kernel w4 (H = 1).  Field symmetric in the wall-attached coordinates (theta, rho = Rc - r):")
    print("   v_theta = F(theta, rho^2), v_r = -rho G(theta, rho^2) (even / odd in rho: free-slip, zero shear to O(kappa)); its continuation to rho < 0 is the CURVILINEAR mirror image")
    print("   (r -> 2 Rc - r at fixed theta), the exact ghost for this field.  Compared: the flat tangent-line mirror at the closest wall point with Taylor order p, error of the solid part")
    print("   Rc     kappa H | p | error        | ratio to previous Rc | |solid part|")
    Xs, Ys = sp.symbols("Xs Ys")
    r_s, th_s = sp.sqrt(Xs ** 2 + Ys ** 2), sp.atan2(Ys, Xs)
    prev = {}
    for Rc in (32.0, 16.0, 8.0, 4.0, 2.0, 1.5):
        rho_s = Rc - r_s
        F = 1 + 0.5 * sp.cos(2 * th_s) + (0.8 + 0.3 * sp.cos(th_s)) * rho_s ** 2
        G = 0.6 * sp.sin(th_s) + 0.4 * rho_s ** 2
        vth, vr = F, -rho_s * G
        vs = [vr * Xs / r_s - vth * Ys / r_s, vr * Ys / r_s + vth * Xs / r_s]       # v = v_r rhat + v_theta thetahat
        vf = sp.lambdify((Xs, Ys), vs, "numpy")
        def vfun(q_):
            o = vf(q_[:, 0], q_[:, 1]); return np.stack([np.broadcast_to(o[0], q_[:, 0].shape), np.broadcast_to(o[1], q_[:, 0].shape)], 1)
        x = np.array([Rc - 0.3, 0.0]); vi = vfun(x[None])[0]
        truth = solid_integral(Rc, x, lambda y: vfun(x + y) - vi)                 # the same formulas continued to rho < 0 = the curvilinear mirror image
        n = x / np.linalg.norm(x); R = np.eye(2) - 2 * np.outer(n, n); d = Rc - np.linalg.norm(x)
        for p in (0, 2, 3):
            coef = [[(k1, k2, float(sp.diff(e, Xs, k1, Ys, k2).subs({Xs: x[0], Ys: x[1]})) / float(sp.factorial(k1) * sp.factorial(k2)))
                     for k1 in range(p + 1) for k2 in range(p + 1 - k1)] for e in vs]
            def ghost(y):
                z = y @ R.T + 2 * d * n                                              # mirror displacement  y_m = R y + 2 d n
                out = np.zeros_like(y)
                for i in range(2):
                    for k1, k2, cf in coef[i]:
                        out[:, i] += cf * z[:, 0] ** k1 * z[:, 1] ** k2
                return out @ R.T                                                     # R v
            est = solid_integral(Rc, x, lambda y: ghost(y) - vi)
            err = np.max(np.abs(est - truth)); sc = np.max(np.abs(truth))
            ratio = "" if p not in prev else f"{prev[p] / err:5.2f}"
            prev[p] = err
            print(f"   {Rc:5.1f}  {1 / Rc:6.3f}  | {p} | {err:.3e}   | {ratio:>8}             | {sc:.3f}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "A" in which: part_A()
    if "B" in which: part_B()
