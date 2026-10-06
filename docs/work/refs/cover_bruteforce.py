"""Reviewer's independent check of the T1.1 cover vector (REVIEW-001): midpoint grid of  int_{solid ∩ disk} unit(x - x') dA'  (matplotlib Path for
point-in-polygon), shares no code with cover.py.  Reviewer's result: agreement with cover.cover_vector_np to the grid resolution, worst 6e-5 absolute over
7 cases (unit square, L-shape, 30-degree triangle; H = 0.3 .. 1; points inside/outside/near an edge) at n = 3000.   Run from the repo root."""
import math, sys
sys.path.insert(0, "python")
import numpy as np
from matplotlib.path import Path
from warpSPHBoundaries.scene import cover

def brute(poly, x, H, n=3000):
    x = np.array(x); g = (np.arange(n) + 0.5) / n * 2 * H - H
    X, Y = np.meshgrid(x[0] + g, x[1] + g); P = np.stack([X.ravel(), Y.ravel()], 1)
    d = x - P; r = np.hypot(d[:, 0], d[:, 1]); m = (r <= H) & (r > 0) & Path(np.array(poly)).contains_points(P)
    return (d[m] / r[m, None]).sum(0) * (2 * H / n) ** 2
SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]; L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]; T = [(0, 0), (1, 0), (math.cos(math.pi / 6), math.sin(math.pi / 6))]
for poly, x, H in [(SQ, (0.3, 0.4), 1.0), (SQ, (1.3, 0.5), 1.0), (L, (1.2, 1.2), 1.0), (L, (0.9, 0.9), 0.6), (T, (0.5, 0.1), 0.7), (T, (0.05, 0.02), 0.3), (SQ, (0.5, -0.02), 0.3)]:
    b, c = brute(poly, x, H), cover.cover_vector_np(np.array(x, float), poly, H)
    print(x, H, "brute", b, "closed form", c, "diff", np.abs(b - c).max())
