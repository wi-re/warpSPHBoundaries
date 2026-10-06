"""Accuracy / cost study of the stage-2 numpy implementation:  exact core vs cheap approximations.

    python scripts/studies/np2d_study.py            (prints markdown tables, ~1 min)

Reference = the 40-digit golden fixtures (hard cases) and, for random batches, the stage-1 mpmath code.
Errors are ABSOLUTE in the natural units (value: 1; gradient * h; moment_alpha / h^k) and the max over
the case set; "hard" = all convex-triangle fixtures (x on edge/vertex, z -> 1e-40, tiny chords, elements
1e-6 h, half-planes, ...), "generic" = the 12 generic fixtures.
"""
import time
from fractions import Fraction as F

import mpmath as mp
import numpy as np

from warpSPHBoundaries.edge import fixtures as fx
from warpSPHBoundaries.edge import np2d

KERNELS = ["cubic", "w2", "w4", "w6"]


def _rel(recs, dtype=np.float64):
    rel, h = [], []
    for c in recs:
        x = (F(c["x"][0]), F(c["x"][1]))
        rel.append([[float(F(a) - x[0]), float(F(b) - x[1])] for a, b in c["polygon"]])
        h.append(float(F(c["h"])))
    V = np.array(rel)
    return V.astype(dtype), np.zeros((len(recs), 2), dtype=dtype), np.array(h, dtype=dtype)


LOST = [0]


def _fin(e):
    """non-finite results (element not representable in the dtype) are skipped and counted"""
    if np.isfinite(e):
        return e
    LOST[0] += 1
    return 0.0


def _errors(recs, dtype=np.float64, **kw):
    V, X, H = _rel(recs, dtype)
    worst = dict(value=0.0, grad=0.0, m2=0.0, m4=0.0)
    np.seterr(all="ignore")
    for k in KERNELS:
        v = np2d.value(V, X, k, H, dtype=dtype, **kw)
        g = np2d.gradient(V, X, k, H, dtype=dtype, **kw)
        m2 = np2d.moment(V, X, k, (1, 1), H, dtype=dtype, **kw)
        m4 = np2d.moment(V, X, k, (2, 2), H, dtype=dtype, **kw)
        for i, c in enumerate(recs):
            e = c["kernels"][k]
            worst["value"] = max(worst["value"], _fin(abs(float(v[i]) - float(mp.mpf(e["value"])))))
            worst["grad"] = max(worst["grad"], _fin(max(abs(float(g[i, j]) - float(mp.mpf(e["grad"][j]))) for j in range(2)) * float(H[i])))
            worst["m2"] = max(worst["m2"], _fin(abs(float(m2[i]) - float(mp.mpf(e["moments"]["1,1"]))) / float(H[i]) ** 2))
            worst["m4"] = max(worst["m4"], _fin(abs(float(m4[i]) - float(mp.mpf(e["moments"]["2,2"]))) / float(H[i]) ** 4))
    return worst


def run(verbose=True):
    data = fx.load()
    hard = [c for c in data["cases"] if c["kind"] == "element" and len(c["polygon"]) == 3 and "nonconvex" not in c["tags"]]
    gen = [c for c in hard if c["id"].startswith("generic")]
    rows = []
    variants = [("float64 closed form (split + unsplit)", dict()),
                ("float64 closed form, no unsplit", dict(unsplit=False)),
                ("longdouble closed form", dict(dtype=np.longdouble)),
                ("float32 closed form", dict(dtype=np.float32))]
    for m in (2, 3, 4, 6, 8):
        variants.append((f"float64, Gauss {m} nodes x 4 panels", dict(quad=m)))
    for s in [(4, 4), (6, 5), (8, 6)]:
        variants.append((f"float64, stable Chebyshev quadrature {s[0]}x{s[1]}", dict(stable=s)))
        variants.append((f"float32, stable Chebyshev quadrature {s[0]}x{s[1]}", dict(stable=s, dtype=np.float32)))
    for name, kw in variants:
        row = [name]
        LOST[0] = 0
        for recs in (gen, hard):
            e = _errors(recs, **kw)
            row += [e["value"], e["grad"], e["m2"], e["m4"]]
        if LOST[0]:
            row[0] += f" ({LOST[0] // 4} of 4x{len(hard)} evaluations not representable, skipped)"
        rows.append(row)
    if verbose:
        print("| variant | generic: value | grad | m_11 | m_22 | hard: value | grad | m_11 | m_22 |")
        print("|---|" + "---|" * 8)
        for r in rows:
            print(f"| {r[0]} | " + " | ".join(f"{x:.1e}" for x in r[1:]) + " |")
    return rows


def float32_table(verbose=True):
    """Stage 3: float32 tolerance table per fixture class, closed form vs stable (Chebyshev) quadrature."""
    data = fx.load()
    hard = [c for c in data["cases"] if c["kind"] == "element" and len(c["polygon"]) == 3 and "nonconvex" not in c["tags"]]
    classes = [("generic", ["generic", "h_scaled", "clockwise"]), ("x on edge / vertex / edge line", ["x_on_edge", "x_at_vertex", "x_on_edge_line"]),
               ("z -> 0", ["z_to_0"]), ("tiny chords", ["tiny_chord"]), ("elements <= 1e-2 h", ["small_element"]),
               ("half-plane (d = 0 ... 1, d < 0)", ["half_plane"]), ("engulf / outside support", ["engulf", "outside_support"])]
    modes = [("closed form", dict()), ("stable (8 nodes x 6 panels)", dict(stable=(8, 6))), ("stable (5 nodes x 4 panels)", dict(stable=(5, 4)))]
    rows = []
    np.seterr(all="ignore")
    for cname, tags in classes:
        recs = [c for c in hard if any(t in c["tags"] for t in tags)]
        if cname.startswith("elements"):
            recs = [c for c in recs if not c["id"].startswith(("small_element-1e0", "small_element-1e1-", "small_element-1-"))]
        V, X, H = _rel(recs, np.float32)
        for mname, kw in modes:
            w = dict(value=0.0, grad=0.0, m11=0.0, m22=0.0)
            lost = 0
            for k in KERNELS:
                v = np2d.value(V, X, k, H, dtype=np.float32, **kw)
                g = np2d.gradient(V, X, k, H, dtype=np.float32, **kw)
                m1 = np2d.moment(V, X, k, (1, 1), H, dtype=np.float32, **kw)
                m2 = np2d.moment(V, X, k, (2, 2), H, dtype=np.float32, **kw)
                for i, c in enumerate(recs):
                    e = c["kernels"][k]
                    ds = [abs(float(v[i]) - float(mp.mpf(e["value"]))),
                          max(abs(float(g[i, j]) - float(mp.mpf(e["grad"][j]))) for j in range(2)) * float(H[i]),
                          abs(float(m1[i]) - float(mp.mpf(e["moments"]["1,1"]))) / float(H[i]) ** 2,
                          abs(float(m2[i]) - float(mp.mpf(e["moments"]["2,2"]))) / float(H[i]) ** 4]
                    if not all(np.isfinite(d) for d in ds):
                        lost += 1
                        continue
                    for key, d in zip(("value", "grad", "m11", "m22"), ds):
                        w[key] = max(w[key], d)
            rows.append((cname, len(recs), mname, w, lost // 4 if lost else 0))
    if verbose:
        print("\nfloat32 tolerance table (max abs error over fixtures of the class, 4 kernels)\n")
        print("| class (cases) | method | value | grad x h | m_11 | m_22 | cases not representable in f32 |\n|---|---|---|---|---|---|---|")
        for cname, n, mname, w, lost in rows:
            print(f"| {cname} ({n}) | {mname} | {w['value']:.1e} | {w['grad']:.1e} | {w['m11']:.1e} | {w['m22']:.1e} | {lost} |")
    return rows


def tiny_relative(verbose=True):
    """relative error of the VALUE for elements 1e-6 h and 1e-3 h with x inside / at a vertex (w2, w4)."""
    data = fx.load()
    recs = [c for c in data["cases"] if c["kind"] == "element" and c["id"].startswith(("small_element-1e6", "small_element-1e3"))
            and ("inside" in c["id"] or "vertex" in c["id"])]
    V, X, H = _rel(recs)
    rows = []
    for label, kw in [("split form only", dict(unsplit=False)), ("with unsplit form (default)", dict())]:
        row = [label]
        for tag in ("1e6", "1e3"):
            idx = [i for i, c in enumerate(recs) if f"-{tag}-" in c["id"]]
            worst = 0
            for k in ("w2", "w4"):
                v = np2d.value(V[idx], X[idx], k, H[idx], **kw)
                for j, i in enumerate(idx):
                    r = float(mp.mpf(recs[i]["kernels"][k]["value"]))
                    worst = max(worst, abs(v[j] - r) / r)
            row.append(worst)
        rows.append(row)
    if verbose:
        print("\nRelative error of the value for tiny elements, x inside / at a vertex (w2, w4)\n")
        print("| form | L_T/h = 1e-6 | L_T/h = 1e-3 |\n|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]:.1e} | {r[2]:.1e} |")
    return rows


def _area_gauss(V, X, kname, n):
    """baseline: tensor-Gauss (Duffy) quadrature of int_T W over the triangle, n x n points, no clipping."""
    from warpSPHBoundaries.edge.kernels import kernel as get_kernel
    k = get_kernel(kname)
    xg, wg = np.polynomial.legendre.leggauss(n)
    u = (xg + 1) / 2
    wu = wg / 2
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    tot = np.zeros(len(V))
    for ui, wi in zip(u, wu):
        for vi, wj in zip(u, wu):
            s, t_ = ui, vi * (1 - ui)                        # Duffy: jacobian (1 - ui)
            p = a + s * (b - a) + t_ * (c - a) - X
            r = np.hypot(p[:, 0], p[:, 1])
            W = np.zeros(len(V))
            for lo, hi, cc in k.pieces:
                m = (r >= float(lo)) & (r <= float(hi)) & (r < 1)
                W = np.where(m, sum(float(ci) * r ** i for i, ci in enumerate(cc)) / np.pi, W)
            tot += wi * wj * (1 - ui) * W
    return tot * area2


def vs_area_quadrature(N=300, verbose=True):
    """error of the VALUE for random triangles (x within the support, element sizes ~ h) vs number of kernel evaluations:
    plain 2D Gauss on the triangle (algebraic convergence: the kernel is only C^k at r = h)
    vs the edge reduction with 1D Gauss (cost counted as nodes x panels x edges)."""
    rng = np.random.default_rng(3)
    V = rng.uniform(-1, 1, (N, 3, 2))
    X = rng.uniform(-0.4, 0.4, (N, 2))
    rows = []
    for kname in ("cubic", "w4"):
        with mp.workdps(30):
            ref = np2d.value(V.astype(np.longdouble), X.astype(np.longdouble), kname, np.longdouble(1), dtype=np.longdouble)
        ref = np.asarray(ref, dtype=np.float64)
        for n in (4, 8, 16, 32, 64):
            e = np.abs(_area_gauss(V, X, kname, n) - ref).max()
            rows.append((kname, f"2D Gauss, {n}x{n}", n * n, e))
        for m in (2, 3, 4, 6, 8):
            e = np.abs(np2d.value(V, X, kname, 1.0, quad=m) - ref).max()
            rows.append((kname, f"edge reduction, Gauss {m}x4 panels", 3 * 4 * m, e))
        e = np.abs(np2d.value(V, X, kname, 1.0) - ref).max()
        rows.append((kname, "edge reduction, closed form", 3 * 2, e))
    if verbose:
        print("\nMax error of the value, random triangles (N = %d), vs kernel evaluations per triangle\n" % N)
        print("| kernel | method | evaluations | max abs error |\n|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1e} |")
    return rows


def timing(N=200_000, verbose=True):
    rng = np.random.default_rng(1)
    V = rng.uniform(-1, 1, (N, 3, 2))
    X = rng.uniform(-0.5, 0.5, (N, 2))
    out = []
    for label, kw in [("closed form", {}), ("Gauss 3x4", dict(quad=3)), ("Gauss 4x4", dict(quad=4)),
                      ("stable 6x5", dict(stable=(6, 5))), ("stable 8x6", dict(stable=(8, 6)))]:
        row = [label]
        for what, fn in [("value", lambda: np2d.value(V, X, "w4", 1.0, **kw)),
                         ("grad", lambda: np2d.gradient(V, X, "w4", 1.0, **kw)),
                         ("m_11", lambda: np2d.moment(V, X, "w4", (1, 1), 1.0, **kw)),
                         ("m_22", lambda: np2d.moment(V, X, "w4", (2, 2), 1.0, **kw))]:
            fn()
            t = time.perf_counter()
            fn()
            row.append((time.perf_counter() - t) / N * 1e6)
        out.append(row)
    if verbose:
        print(f"\nCost, w4, N = {N} triangle/point pairs, one thread (microseconds per pair)\n")
        print("| variant | value | gradient | m_11 | m_22 |\n|---|---|---|---|---|")
        for r in out:
            print(f"| {r[0]} | " + " | ".join(f"{x:.2f}" for x in r[1:]) + " |")
    return out


if __name__ == "__main__":
    run()
    float32_table()
    tiny_relative()
    vs_area_quadrature()
    timing()
