"""2D tier-selection study for a CONVEX CIRCULAR OBSTACLE (disk radius R, particle at distance d from the surface, support h = 1, kernel w4):
absolute error of every applicable model against the exact disk (circle edge identity), as a function of R/h, and the size of the jump when
switching between neighbouring tiers (the continuity question of HANDOFF §10).

    python -m edgebound.edge.tier_select
"""
from fractions import Fraction as F

import mpmath as mp
import numpy as np

from . import np2d, tier3, tier4


def run(d=F(3, 10), kernel="w4", verbose=True):
    rows = []
    for R in [F(1, 20), F(1, 10), F(1, 5), F(7, 20), F(1, 2), F(1), F(2), F(4), F(8), F(16)]:
        ex = float(tier4.arc_value(kernel, (F(0), R + d), R))
        r = {"R": float(R), "exact": ex}
        # tier 4 (disk series): needs the kernel smooth on [D - a, D + a]
        try:
            r["t4_K2"] = abs(float(tier4.disk_series(kernel, R, R + d, 2)) - ex)
        except ValueError:
            r["t4_K2"] = None
        # tier 3: planar and second order (needs R > 1 - d)
        r["t3_0"] = abs(float(tier3.lambda_expansion(kernel, d, 1 / R, 0)) - ex) if R > 1 - d else None
        r["t3_2"] = abs(float(tier3.lambda_expansion(kernel, d, 1 / R, 2)) - ex) if R > 1 - d else None
        # tier 2: regular polygon with edge length h/4 and h/8 (at least 8 edges)
        for ell, key in ((0.25, "t2_h4"), (0.125, "t2_h8")):
            N = max(8, int(round(2 * np.pi * float(R) / ell)))
            t = 2 * np.pi * np.arange(N) / N
            poly = np.stack([float(R) * np.cos(t), float(R + d) + float(R) * np.sin(t)], axis=1)
            r[key] = abs(np2d.value(poly[None], np.zeros((1, 2)), kernel)[0] - ex)
        rows.append(r)
    if verbose:
        f = lambda v: "n/a" if v is None else f"{v:.1e}"
        print(f"\nabsolute error of the value, {kernel}, d = {float(d)} h, disk obstacle radius R\n")
        print("| R/h | exact | tier 4 series K=2 | tier 3 planar (2020) | tier 3 order 2 | tier 2 edge h/4 | tier 2 edge h/8 |\n|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['R']:g} | {r['exact']:.4f} | {f(r['t4_K2'])} | {f(r['t3_0'])} | {f(r['t3_2'])} | {f(r['t2_h4'])} | {f(r['t2_h8'])} |")
    return rows


if __name__ == "__main__":
    run()
