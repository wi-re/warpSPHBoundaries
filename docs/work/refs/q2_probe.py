"""Reviewer's probe for WORK-002 T2.4 (throw-away, NOT the deliverable; copy to .tmp/ to run).  Error of the current Warp edge-channel compile for the
gradient channel g0 of the powers W^k (k = 1..5) of Wendland C2 ('w2', degree 5k) and C4 ('w4', degree 8k) against an exact mpmath reference, on a
unit square, support h = 1.  W^k as an EdgeKernel in the truncated-power basis u = 1 - q:  w2: (1+4q) = 5 - 4u;  w4: 1 + 6q + 35/3 q^2 = 56/3 - 88/3 u + 35/3 u^2.
edge_channels folds 1/pi into the channels: compare with  C/pi * grad int shape.   usage: python q2_probe.py w2|w4
Reviewer's result 2026-10-03 (worst |abs err| / max|ref| over the 6 points; scale = max|ref| = 1.4 .. 3.7):
   w2:  k=1 4.0e-15   k=2 1.5e-13   k=3 6.7e-12   k=5 (degree 25) 1.8e-08
   w4:  k=1 2.5e-14   k=2 1.1e-11   k=3 2.8e-09   k=5 (degree 40) 7.1e-04   <- worst abs error 2.6e-03
i.e. W^5/5 is fine for C2 but NOT for C4 with the current (monomial-basis) plan; the exact Chebyshev compile of np2d (stable mode) is the candidate remedy."""
import sys
from fractions import Fraction as F
sys.path.insert(0, "python")
import numpy as np, torch, math, mpmath as mp
from warpSPHBoundaries.edge import kernels, warpbc, geometry as G
from warpSPHBoundaries.edge.core import GUARD, block_grad

fam = sys.argv[1] if len(sys.argv) > 1 else "w2"
dev = "cuda:0"
poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
V = torch.tensor(poly, dtype=torch.float64, device=dev); E = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=torch.int32, device=dev)
pts = [(0.3, 0.4), (1.02, 0.5), (0.5, -0.3), (0.97, 0.03), (1.3, 0.5), (0.5, -0.9)]
base = {"w2": [F(5), F(-4)], "w4": [F(56, 3), F(-88, 3), F(35, 3)]}[fam]
p0 = {"w2": 4, "w4": 6}[fam]; dg = {"w2": 5, "w4": 8}[fam]

def terms(k):                         # [(coef, knot=1, power)] of W^k in the basis u^power
    pol = [F(1)]
    for _ in range(k):
        new = [F(0)] * (len(pol) + len(base) - 1)
        for i, a in enumerate(pol):
            for j, b in enumerate(base): new[i + j] += a * b
        pol = new
    return [(c, F(1), p0 * k + m) for m, c in enumerate(pol)]

def ref_g0(k, x, C):                  # exact: expand u^p = (1-r)^p into monomials r^j, block_grad for each (core.py, mpmath, h = 1)
    from math import comb
    with mp.workdps(150 + GUARD):
        P = G.prepare(poly, x, h=1)
        gs = [block_grad(P, j, 1) for j in range(dg * k + 1)]
        tot = [mp.mpf(0), mp.mpf(0)]
        for c, _, p in terms(k):
            for j in range(p + 1):
                co = c * comb(p, j) * (-1) ** j
                tot[0] += co * gs[j][0]; tot[1] += co * gs[j][1]
        return np.array([float(tot[0]), float(tot[1])]) * float(C) / math.pi

n = len(pts); P = torch.tensor(pts, dtype=torch.float64, device=dev)
qi = torch.arange(n).repeat_interleave(4).to(torch.int32).to(dev); ee = torch.arange(4).repeat(n).to(torch.int32).to(dev)
for k in (1, 2, 3, 5):
    name = f"{fam}p{k}"
    kernels.KERNELS[name] = kernels._from_terms(name, terms(k)); C = kernels.KERNELS[name].c2_pi
    c = warpbc.edge_channels(qi, ee, P, torch.ones(n, dtype=torch.float64, device=dev), V, E, name, device=dev)
    g = torch.zeros((n, 2), dtype=torch.float64, device=dev).index_add_(0, qi.long(), c[:, 3:5]).cpu().numpy()
    ref = np.array([ref_g0(k, x, C) for x in pts]); sc = np.abs(ref).max(); err = np.abs(g - ref).max(axis=1)
    print(f"{fam} k={k} degree={dg*k} scale={sc:.3e}  worst abs/scale={err.max()/sc:.2e}  abs per point: {[f'{e:.1e}' for e in err]}")
