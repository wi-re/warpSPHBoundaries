"""How much accuracy does the (guard-free) algorithm lose at 53-bit precision?

Emulates float64 arithmetic with mpmath at prec = 53 and GUARD = 0 (same algorithm, same
primitives, no extra digits) and compares against the 40-digit golden fixtures.  The result is the
numpy-float64 accuracy to expect per hard case (before any stability improvement):

    python scripts/studies/precision_probe.py
"""
from fractions import Fraction as F

import mpmath as mp

from edgebound.edge import core, fixtures as fx, geometry
from edgebound.edge.mpq import mpq


def probe(verbose=True):
    data = fx.load()
    rows = []
    old_guard = core.GUARD
    core.GUARD = 0
    try:
        for c in data["cases"]:
            if c["kind"] != "element":
                continue
            T = [(F(a), F(b)) for a, b in c["polygon"]]
            x = (F(c["x"][0]), F(c["x"][1]))
            h = F(c["h"])
            worst = {"value": 0.0, "grad": 0.0, "m1": 0.0, "m2": 0.0, "m3": 0.0, "m4": 0.0}
            for k in ("cubic", "w4"):
                ref = c["kernels"][k]
                with mp.workprec(53):
                    prep = geometry.prepare(T, x, h)
                    v = core.value(T, x, k, h, dps=15, prepared=prep)
                    g = core.gradient(T, x, k, h, dps=15, prepared=prep)
                with mp.workprec(200):
                    rv = mp.mpf(ref["value"])
                    worst["value"] = max(worst["value"], float(abs(v - rv)))
                    for i in range(2):
                        worst["grad"] = max(worst["grad"], float(abs(g[i] - mp.mpf(ref["grad"][i]))))
                for al in [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (2, 2), (0, 4)]:
                    with mp.workprec(53):
                        m = core.moment(T, x, k, al, h, dps=15, prepared=prep)
                    kk = sum(al)
                    with mp.workprec(200):
                        rm = mp.mpf(ref["moments"][f"{al[0]},{al[1]}"])
                        # relative to the natural scale h^k of the moment
                        e = float(abs(m - rm) / mpq(h) ** kk)
                    worst[f"m{kk}"] = max(worst[f"m{kk}"], e)
            rows.append((c["id"], worst))
    finally:
        core.GUARD = old_guard
    if verbose:
        print(f"{'case':44s} {'value':>9s} {'grad':>9s} {'m1':>9s} {'m2':>9s} {'m3':>9s} {'m4':>9s}")
        for cid, w in rows:
            print(f"{cid:44s} " + " ".join(f"{w[k]:9.1e}" for k in ("value", "grad", "m1", "m2", "m3", "m4")))
    return rows


if __name__ == "__main__":
    probe()
