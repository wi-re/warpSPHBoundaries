"""Check 7 of fem-nodal-weights.md: conditioning of the nodal weights vs L_T/h and the order p.

    python -m edgebound.fem_study

Error metric: max_i |dw_i| / max_i |w_i|  (relative to the weights), where the reference is the exact (stage-1) result
for the *exact rational value of the float inputs* (so input rounding is excluded: only arithmetic is measured).
Element: a fixed triangle shape scaled by L = L_T (edge length ~ L), evaluation point at a distance d from its centroid
(d = 0: inside; d / h = 0.3, 0.6).
"""
from fractions import Fraction as F

import mpmath as mp
import numpy as np

from . import fem, np_fem

BASE = [(-0.3, -0.2), (0.6, -0.1), (0.1, 0.7)]


def element(L, d, dtype=np.float64, ang=0.7):
    """float inputs (physical, h = 1): triangle of size ~L around the origin, x at distance d along direction ang."""
    c = np.array([sum(v[0] for v in BASE) / 3, sum(v[1] for v in BASE) / 3])
    V = np.array([[(v[0] - c[0]) * L, (v[1] - c[1]) * L] for v in BASE])
    x = np.array([d * np.cos(ang), d * np.sin(ang)])
    rel = (V - x)                                           # relative coordinates computed once in float64
    return rel.astype(dtype)


def reference(rel, kernel, p):
    T = [(F(float(a)), F(float(b))) for a, b in rel.astype(np.float64)]
    w = fem.weights(T, (F(0), F(0)), kernel, p, dps=40)
    return np.array([float(v) for v in w])


def study(kernel="w4", verbose=True):
    rows = []
    np.seterr(all="ignore")
    Ls = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4]
    for d_label, d in [("x inside (d=0)", 0.0), ("d = 0.3 h", 0.3), ("d = 0.6 h", 0.6), ("d = 0.95 h (rim)", 0.95)]:
        for p in (1, 2, 3):
            for L in Ls:
                if d > 0 and L > 0.5 * d:
                    continue                               # element must not contain x; keep L << d
                rel = element(L, d, np.float64)
                ref = reference(rel, kernel, p)
                ref32 = reference(rel.astype(np.float32).astype(np.float64), kernel, p)       # reference for the ROUNDED float32 inputs
                scale = np.abs(ref).max()
                if scale == 0:
                    continue
                zero = np.zeros((1, 2))
                out = []
                for label, dt, fn, kw in [("f64 closed", np.float64, np_fem.weights, {}), ("f32 closed", np.float32, np_fem.weights, {}),
                                          ("f64 hybrid", np.float64, np_fem.weights_hybrid, {}), ("f32 hybrid", np.float32, np_fem.weights_hybrid, {}),
                                          ("f32 hybrid stable", np.float32, np_fem.weights_hybrid, dict(stable=(8, 6)))]:
                    w = fn(rel[None].astype(dt), zero.astype(dt), kernel, p, 1, dtype=dt, **kw)[0]
                    out.append(float(np.abs(w - (ref32 if dt is np.float32 else ref)).max() / scale))
                rows.append((d_label, p, L, scale, *out))
    if verbose:
        print(f"\nrelative error of the nodal weights, kernel {kernel}\n")
        print("| x | p | L_T/h | max |w_i| | f64 closed | f32 closed | f64 hybrid | f32 hybrid | f32 hybrid+stable |\n|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]:g} | {r[3]:.1e} | " + " | ".join(f"{v:.1e}" for v in r[4:]) + " |")
    return rows


if __name__ == "__main__":
    study()
