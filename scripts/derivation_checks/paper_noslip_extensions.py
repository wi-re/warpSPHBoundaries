# EXPLORATORY (docs/plan-exact-wall-laplacian.md): bias of wall-Laplacian estimators near a NO-SLIP flat wall, noise-free, 1D reduction.
# Fields v(s) depend on the wall distance s only (v_w = 0), so the 2D kernel reduces to its marginal w(u) = int W(sqrt(y1^2+u^2)) dy1 and
#   estimator E = int (v_ext(s') - v(d)) w''(s' - d) ds'   over the support (fluid s' > 0 exactly, extension for s' < 0),   target v''(d).
# ANALYTIC = the field continued analytically into the solid: the smoothing bias that no extension can beat.
# Estimators:
#   fluid only         v_ext = 0 on the solid (no wall term)
#   constant ghost     v_ext = 2 v_w - v(d)                      (the "constant ghost" no-slip, docs/derivations/noslip-wall.md section 3.5)
#   odd mirror         v_ext(s') = 2 v_w - v(-s')                (exact field mirrored: best a mirror can do; kink in v'' at the wall)
#   Chiron-type        renormalised Green form: [fluid sum + (v_w - v_i) w'(-d) - v'(0) w(d)] / gamma, slope model v'(0) ~ (v_i - v_w)/d
#   BC polynomial(m)   weighted LS fit of v on the fluid with basis {s, s^2, ..., s^m} (v(0)=v_w built in), CONTINUED into s' < 0
import numpy as np
from scipy.integrate import quad
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_viscosity_limits as V          # KERN, ev (kernels as polynomial blocks)
B = V.KERN["w4"]

def _marg(f, u):
    L = np.sqrt(max(1 - u * u, 0)); 
    return 2 * quad(lambda y: f(np.hypot(y, u), y, u), 0, L, epsabs=1e-14, epsrel=1e-13)[0]
w   = lambda u: _marg(lambda r, y, u_: V.ev(B, r), abs(u))
wp  = lambda u: np.sign(u) * _marg(lambda r, y, u_: V.ev(B, r, 1) * u_ / r, abs(u))
wpp = lambda u: _marg(lambda r, y, u_: V.ev(B, r, 2) * u_ ** 2 / r ** 2 + V.ev(B, r, 1) * y ** 2 / r ** 3, abs(u))

FIELDS = {
    "boundary layer 1-exp(-s/0.5)": None,
    "sin(1.5 s)": (lambda s: np.sin(1.5 * s), lambda s: -2.25 * np.sin(1.5 * s), None),
    "Poiseuille s(3-s)": (lambda s: s * (3 - s), lambda s: -2.0 + 0 * s, None),
    "sin(3 s)": (lambda s: np.sin(3 * s), lambda s: -9 * np.sin(3 * s), None),
}
FIELDS["boundary layer 1-exp(-s/0.5)"] = (lambda s: 1 - np.exp(-s / 0.5), lambda s: -4 * np.exp(-s / 0.5), None)   # v'' = -(1/0.5^2) exp(-s/0.5)

def est(vext, d, lo=None):
    f = lambda s: (vext(s)) * wpp(s - d)
    a, b = d - 1, d + 1
    return quad(f, a, b, points=[0.0] if a < 0 < b else None, epsabs=1e-12, epsrel=1e-10, limit=200)[0]

def run(d):
    print(f"\n--- wall distance d = {d} (kernel w4, h = 1) ---")
    print(f"{'field':32s} {'target':>9s} | fluid-only  const-ghost  odd-mirror  Chiron-type  ANALYTIC | BC-poly m=1  m=2  m=3  m=4   (errors)")
    gam = quad(lambda s: w(s - d), 0, d + 1, epsabs=1e-13)[0]
    for name, (v, d2v, _) in FIELDS.items():
        vi = v(d); tgt = d2v(d)
        fl = quad(lambda s: (v(s) - vi) * wpp(s - d), 0, d + 1, epsabs=1e-12, epsrel=1e-10, limit=200)[0]      # fluid half of the integral
        e_fluid = fl - tgt
        e_const = fl + quad(lambda s: ((0 - vi) - vi) * wpp(s - d), d - 1, 0, epsabs=1e-12)[0] - tgt if d < 1 else fl - tgt
        e_odd = fl + quad(lambda s: ((-v(-s)) - vi) * wpp(s - d), d - 1, 0, epsabs=1e-12)[0] - tgt if d < 1 else fl - tgt
        # Chiron-type renormalised Green form
        slope = (vi - 0) / d
        ch = (fl + (0 - vi) * wp(-d) - slope * w(d)) / gam
        e_ch = ch - tgt
        e_an = fl + quad(lambda s: (v(s) - vi) * wpp(s - d), d - 1, 0, epsabs=1e-12)[0] - tgt   # analytic continuation: pure smoothing bias
        polys = []
        for m in [1, 2, 3, 4]:
            ks = np.arange(1, m + 1)
            A = np.array([[quad(lambda s: w(s - d) * s ** (i + j), 0, d + 1, epsabs=1e-13)[0] for j in ks] for i in ks])
            rhs = np.array([quad(lambda s: w(s - d) * v(s) * s ** i, 0, d + 1, epsabs=1e-13)[0] for i in ks])
            co = np.linalg.solve(A, rhs)
            p = lambda s, co=co, ks=ks: sum(c * s ** k for c, k in zip(co, ks))
            ext = quad(lambda s: (p(s) - vi) * wpp(s - d), d - 1, 0, epsabs=1e-12)[0]
            polys.append(fl + ext - tgt)
        print(f"{name:32s} {tgt:+9.4f} | {e_fluid:+10.3e} {e_const:+11.3e} {e_odd:+11.3e} {e_ch:+11.3e} {e_an:+11.3e} | " + " ".join(f"{e:+10.2e}" for e in polys))

if __name__ == "__main__":
    for d in [0.1, 0.3, 0.6]:
        run(d)
