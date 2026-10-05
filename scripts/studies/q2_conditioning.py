"""Q2 conditioning study: the shifting tensile term  T = int_solid W^4 grad_x W dA' = (1/5) grad_x int_solid W^5 dA'
(the identity: grad_x W^5 = 5 W^4 grad_x W, and W^5 vanishes at the support edge, so no boundary term).

For each Wendland family of this repo (w2: shape (1-q)^4 (1+4q), degree 5;  w4: shape (1-q)^6 (1+6q + 35/3 q^2), degree 8)
and k = 1..5 (kernel W^k = C_k shape^k, degree 5k / 8k), compare the g(0,0) gradient channel on the unit square
(support h = 1) against the exact mpmath reference (core.block_grad at 150 + GUARD dps), via:
  (A) Warp `warpbc.edge_channels` (the current monomial-basis plan -- the route the solver uses),
  (B) `np2d.gradient` float64 plain,
  (C) `np2d.gradient` stable=(8, 6),
  (D) `np2d.gradient` stable=(16, 8).
All routes and the reference return  grad_x int_solid (pi W^k) dA'  (the common 1/pi is applied at evaluation).

Point set: the reviewer probe's 6 points + 200 random points in [-0.5, 1.5]^2 (seed 5) + the 4 vertices and 4 edge
midpoints of the unit square (214 points).  Report per (family, k, route): scale = max|ref|, worst |err|/scale, worst
absolute error and where; plus the DevicePlan build time and (nE, nV) for w2p5 / w4p5 and the mpmath reference time;
the fixed classification at k = 5 (<= 1e-8 GOOD, <= 1e-5 ACCEPTABLE, > 1e-5 NOT USABLE); the reproduction of the
reviewer's numbers (within a factor 10); and the flat-floor sign check of docs/q2-conditioning.md.

usage: python scripts/studies/q2_conditioning.py
"""
import time
from fractions import Fraction as F

import numpy as np

from edgebound.edge import geometry as G
from edgebound.edge import kernels, np2d, warpbc
from edgebound.edge.core import GUARD, block_grad

POLY = [(0, 0), (1, 0), (1, 1), (0, 1)]
PTS6 = [(0.3, 0.4), (1.02, 0.5), (0.5, -0.3), (0.97, 0.03), (1.3, 0.5), (0.5, -0.9)]
# (base coeffs in the u = 1-q basis, p0, degree per k)
from edgebound.edge.kernels import POWER_FAMILIES as FAM, power_monomials as ref_coeffs, power_terms as terms      # library code (tensile.py needs it)
KS = (1, 2, 3, 4, 5)
DEGMAX = 40                                     # w4 k = 5
ROUTES = [("A warp", None), ("B np plain", None), ("C np stable(8,6)", (8, 6)), ("D np stable(16,8)", (16, 8))]


def main():
    import mpmath as mp
    import torch

    dev = "cuda:0"
    t_start = time.time()

    def _mpf(x):                                   # Fraction -> mpf (the mpmath constructor rejects Fraction)
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)

    rng = np.random.default_rng(5)
    X = np.vstack([np.asarray(PTS6, dtype=np.float64), rng.uniform(-0.5, 1.5, (200, 2)),
                   np.asarray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float64),
                   np.asarray([(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)], dtype=np.float64)])
    n = len(X)
    print("points: %d (6 probe + 200 random seed 5 + 4 vertices + 4 midpoints), unit square, h = 1" % n, flush=True)

    # register the W^k kernels (in memory only; nothing cached in the repo)
    names = []
    for fam in FAM:
        for k in KS:
            nm = "%sp%d" % (fam, k)
            kernels.KERNELS[nm] = kernels._from_terms(nm, terms(k, fam))
            names.append(nm)
    acoefs = {nm: ref_coeffs(k, fam) for fam in FAM for k in KS for nm in ["%sp%d" % (fam, k)]}

    # ---------------- mpmath reference (shared across families and k): gs[j] = block_grad(j) per point
    t0 = time.time()
    with mp.workdps(150 + GUARD):
        refs_raw = []
        for (px, py) in X:
            P = G.prepare(POLY, (px, py), h=1)
            refs_raw.append([block_grad(P, j, 1) for j in range(DEGMAX + 1)])
    t_ref = time.time() - t0
    print("mpmath reference (150 + GUARD dps, %d block_grads per point): %.1f s" % (DEGMAX + 1, t_ref), flush=True)

    def ref_for(nm):
        a = acoefs[nm]
        out = np.zeros((n, 2))
        with mp.workdps(150 + GUARD):
            C = _mpf(F(kernels.KERNELS[nm].c2_pi)) / mp.pi
            for i, gs in enumerate(refs_raw):
                tot = [mp.mpf(0), mp.mpf(0)]
                for j, aj in enumerate(a):
                    if aj:
                        tot[0] += aj * gs[j][0]
                        tot[1] += aj * gs[j][1]
                out[i, 0] = float(C * tot[0])
                out[i, 1] = float(C * tot[1])
        return out

    # ---------------- route A: Warp edge_channels (current plan)
    P = torch.tensor(X, dtype=torch.float64, device=dev)
    V = torch.tensor([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=torch.float64, device=dev)
    E = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=torch.int32, device=dev)
    qi = torch.arange(n).repeat_interleave(4).to(torch.int32).to(dev)
    ee = torch.arange(4).repeat(n).to(torch.int32).to(dev)
    sup = torch.ones(n, dtype=torch.float64, device=dev)
    planinfo = {}

    def route_a(nm):
        t0 = time.time()
        plan = warpbc.DevicePlan(nm, dev)                       # the plan builder resolves the kernel by NAME (get_kernel)
        tplan = time.time() - t0
        planinfo[nm] = (tplan, plan.nE, plan.nV)
        c = warpbc.edge_channels(qi, ee, P, sup, V, E, nm, device=dev, plan=plan)
        g = torch.zeros((n, 2), dtype=torch.float64, device=dev).index_add_(0, qi.long(), c[:, 3:5])
        return g.cpu().numpy()

    def route_np(nm, stable=None):
        verts_rep = np.repeat(np.asarray(POLY, dtype=np.float64)[None], n, axis=0)      # one query point per polygon
        return np2d.gradient(verts_rep, X, nm, h=1, dtype=np.float64, stable=stable)

    # ---------------- report
    res = {}
    for fam in FAM:
        for k in KS:
            nm = "%sp%d" % (fam, k)
            ref = ref_for(nm)
            scale = float(np.abs(ref).max())
            line = {}
            for rname, stable in ROUTES:
                got = route_a(nm) if rname == "A warp" else route_np(nm, stable=stable)
                err = np.abs(got - ref)
                iworst = int(np.abs(err).max(axis=1).argmax())
                rel = float(err.max()) / scale
                line[rname] = (rel, float(err.max()), iworst)
                res[nm + "/" + rname] = rel
            print("%s k=%d degree=%-2d scale=% .3e | A % .2e  B % .2e  C % .2e  D % .2e"
                  % (fam, k, FAM[fam][2] * k, scale,
                     line["A warp"][0], line["B np plain"][0], line["C np stable(8,6)"][0], line["D np stable(16,8)"][0]), flush=True)
            for rname, _ in ROUTES:
                rel, ae, iw = line[rname]
                print("    %-14s worst |err|/scale = % .3e   worst abs err = % .3e   at point #%d (%.3f, %.3f)"
                      % (rname, rel, ae, iw, X[iw, 0], X[iw, 1]), flush=True)

    print("\nDevicePlan (route A) build time and plan size:", flush=True)
    for nm in names:
        tplan, nE, nV = planinfo[nm]
        print("  %s: %.2f s   nE = %d   nV = %d" % (nm, tplan, nE, nV), flush=True)

    # ---------------- reviewer reproduction (within a factor 10)
    print("\nreviewer reproduction (worst |err|/scale; reviewer 2026-10-03, 6 points):", flush=True)
    reviewer = {("w2", 1): 4.0e-15, ("w2", 2): 1.5e-13, ("w2", 3): 6.7e-12, ("w2", 5): 1.8e-8,
                ("w4", 1): 2.5e-14, ("w4", 2): 1.1e-11, ("w4", 3): 2.8e-9, ("w4", 5): 7.1e-4}
    for (fam, k), rv in reviewer.items():
        mine = res["%sp%d/A warp" % (fam, k)]
        ratio = mine / rv if rv > 0 else float("inf")
        print("  %sp%d (A): reviewer % .2e   mine % .2e   ratio % .3f   %s" % (fam, k, rv, mine, ratio, "(OK <= 10)" if ratio <= 10 else "(FINDING: > 10)"), flush=True)
    for rname, rv in (("B np plain", 8.4e-4), ("C np stable(8,6)", 3.2e-11), ("D np stable(16,8)", 5e-17)):
        mine = res["w4p5/" + rname]
        ratio = mine / rv
        print("  w4p5 %-14s: reviewer % .2e   mine % .2e   ratio % .3f   %s" % (rname, rv, mine, ratio, "(OK <= 10)" if ratio <= 10 else "(FINDING: > 10)"), flush=True)

    # ---------------- classification at k = 5
    print("\nclassification at k = 5 (fixed: <= 1e-8 GOOD, <= 1e-5 ACCEPTABLE, > 1e-5 NOT USABLE):", flush=True)
    for fam in FAM:
        nm = "%sp5" % fam
        cls = []
        for rname, _ in ROUTES:
            rel = res[nm + "/" + rname]
            c = "GOOD" if rel <= 1e-8 else ("ACCEPTABLE" if rel <= 1e-5 else "NOT USABLE")
            cls.append("%s %s" % (rname, c))
        print("  %s: %s" % (fam, ";  ".join(cls)), flush=True)

    # ---------------- flat-floor sign check (docs/q2-conditioning.md)
    # point (0, 0.3), support 1, solid below y = 0: the square [-2,2] x [-2,0] is exactly the half-plane inside the disk
    sq = [(-2.0, -2.0), (2.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]
    print("\nflat-floor sign check: point (0, 0.3), support 1, solid below y = 0 (square [-2,2] x [-2,0]):", flush=True)
    with mp.workdps(150 + GUARD):
        P = G.prepare(sq, (0.0, 0.3), h=1)
        gs = [block_grad(P, j, 1) for j in range(DEGMAX + 1)]
        for fam in FAM:
            a = acoefs["%sp5" % fam]
            tot = mp.mpf(0)
            for j, aj in enumerate(a):
                if aj:
                    tot += aj * gs[j][1]
            g0y = tot * (_mpf(F(kernels.KERNELS["%sp5" % fam].c2_pi)) / mp.pi)
            c2 = _mpf(F(kernels.KERNELS[fam].c2_pi))
            c25 = _mpf(F(kernels.KERNELS["%sp5" % fam].c2_pi))
            Ty = (mp.mpf(1) / 5) * (c2 / c25) ** 5 / mp.pi ** 4 * g0y
            print("  %s: g0_y = % .10e   T_y = (1/5)(c2/c2^5)^5/pi^4 * g0_y = % .10e   T_y < 0: %s" % (fam, float(g0y), float(Ty), float(Ty) < 0), flush=True)

    print("\ntotal: %.1f s" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
