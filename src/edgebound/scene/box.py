"""Corner tables for axis-aligned boxes (docs/box-domain-primitive.md): the moment channels of a radial kernel over a rectangle from four lookups of two 2D functions.

With y = x' - x (units of the support h) and the quadrant integrals

    Phi(a, b)  = int_{y1 <= a, y2 <= b} W(|y|) dy,        Phi1(a, b) = int_{y1 <= a, y2 <= b} y1 W(|y|) dy      (Phi2(a, b) = Phi1(b, a)),

a rectangle [a0, a1] x [b0, b1] (limits relative to the particle) integrates to the inclusion-exclusion of the four corners, and the nine channels of
`warpbc.edge_channels` are first derivatives of the two tables (by parts the boundary terms are 1D marginals = derivatives):

    lam = Phi,   m_d = Phi_d,   g0_j = -dPhi/da_j,   g1_dj = -[boundary_dj - delta_dj Phi],   boundary_dd = a_d dPhi/da_d,   boundary_21 = dPhi_2/da,   boundary_12 = dPhi_1/db.

The arguments are clamped to [-1, 1] (the support): Phi = 0 below, saturated above, so the far field is exact without a branch.  The tables are Chebyshev tensors on the
a geometrically graded panel grid toward the origin (the kernel's odd powers of r make Phi non-analytic at (0, 0)), built once per kernel by `BoxRep`.
"""
import numpy as np
import torch
from numpy.polynomial import chebyshev as C

F64 = torch.float64


def chebyshev_nodes(N):
    return np.cos(np.pi * (np.arange(N + 1) + 0.5) / (N + 1))


def fit_panels(values, N):
    """Chebyshev coefficients [N+1, N+1] of the tensor interpolant of `values[i, j]` = f(x_i, y_j) at the first-kind nodes (x, y panel-mapped to [-1, 1])."""
    nodes = chebyshev_nodes(N)
    c = C.chebfit(nodes, np.asarray(values).T, N)
    return C.chebfit(nodes, c.T, N)


def _basis(t, N):
    """T_k(t), dT_k/dt for k = 0 .. N, t [P] in [-1, 1], in closed form (a handful of launches instead of an N-step recurrence): T_k = cos(k theta), dT_k/dt = k sin(k theta) / sin(theta)
    with the limits k^2 (-1)^(k-1)... at the ends handled by the Taylor expansion sin(k th) / sin(th) = k (1 - (k^2 - 1) th^2 / 6) for th < 1e-4 (and the mirror image at pi)."""
    k = torch.arange(N + 1, dtype=F64, device=t.device)
    th = torch.acos(t.clamp(-1.0, 1.0))[:, None]
    T = torch.cos(k * th)
    sth = torch.sin(th)
    d = torch.minimum(th, torch.pi - th)
    sgn = torch.where(th > 0.5 * torch.pi, (-1.0) ** (k + 1), torch.ones_like(k))                      # sin(k (pi - d)) / sin(d) = (-1)^(k+1) sin(k d) / sin(d)
    ratio = torch.where(d < 1e-4, k * (1.0 - (k * k - 1.0) * d * d / 6.0) * sgn, torch.sin(k * th) / sth.clamp(min=1e-300))
    return T, k * ratio


class BoxTables:
    """the two corner tables of one kernel on one device, Chebyshev tensors on the graded panel grid `breaks` (per axis): `eval(name, a, b)` -> (value, d/da, d/db) at points (a, b)
    in units of the support (clamped).  Kernels with a cusp at the origin (`cone`, `lw2`: a term linear in r) make Phi non-analytic at (0, 0); the geometric grading toward the origin
    restores the spectral rate."""

    def __init__(self, coeffs, breaks, N, device):
        self.N = N
        self.breaks = torch.as_tensor(breaks, dtype=F64, device=device)                 # [np + 1]
        self.inner = self.breaks[1:-1].contiguous()
        self.coef = {k: torch.as_tensor(c, dtype=F64, device=device) for k, c in coeffs.items()}        # name -> [np, np, N+1, N+1]

    def corner(self, a, b):
        """everything one corner needs, for points (a, b) in units of the support (clamped), the basis computed once: Phi (value, d/da, d/db), Phi1 (value, d/da, d/db) and Phi_2 = Phi1(b, a)
        (value, d/da, d/db)."""
        a, b = a.clamp(-1.0, 1.0), b.clamp(-1.0, 1.0)
        ia, ib = torch.bucketize(a, self.inner, right=True), torch.bucketize(b, self.inner, right=True)
        wa, wb = self.breaks[ia + 1] - self.breaks[ia], self.breaks[ib + 1] - self.breaks[ib]
        ta, tb = (2.0 * a - (self.breaks[ia] + self.breaks[ia + 1])) / wa, (2.0 * b - (self.breaks[ib] + self.breaks[ib + 1])) / wb
        sa, sb = 2.0 / wa, 2.0 / wb
        Ta, dTa = _basis(ta, self.N)
        Tb, dTb = _basis(tb, self.N)
        out = {}
        for name in ("Phi", "Phi1"):
            Cg = self.coef[name][ia, ib]                                                # [P, N+1, N+1], first index: a
            CTb = torch.einsum("pkl,pl->pk", Cg, Tb)
            out[name] = ((Ta * CTb).sum(1), sa * (dTa * CTb).sum(1), sb * (Ta * torch.einsum("pkl,pl->pk", Cg, dTb)).sum(1))
        Cg = self.coef["Phi1"][ib, ia]                                                  # Phi_2(a, b) = Phi1(b, a): first index is now b
        CTa = torch.einsum("pkl,pl->pk", Cg, Ta)
        v2, d_first, d_second = (Tb * CTa).sum(1), sb * (dTb * CTa).sum(1), sa * (Tb * torch.einsum("pkl,pl->pk", Cg, dTa)).sum(1)
        out["Phi2"] = (v2, d_second, d_first)                                           # (value, d/da, d/db) of Phi_2
        return out

    def channels(self, a0, a1, b0, b1):
        """(lam [P], m [P,2], g0 [P,2], g1 [P,2,2]) of the rectangle [a0, a1] x [b0, b1] (relative to the particle, units of the support), library conventions
        (m = int y W, g0 = int grad_x W, g1[d, j] = int y_d d_xj W); the four corners are one batch of 4P points."""
        P = a0.shape[0]
        a = torch.cat([a1, a0, a1, a0]); b = torch.cat([b1, b1, b0, b0])
        sg = torch.cat([torch.ones(P), -torch.ones(P), -torch.ones(P), torch.ones(P)]).to(device=a0.device, dtype=F64)
        c = self.corner(a, b)
        (ph, pa, pb), (p1, p1a, p1b), (p2, p2a, p2b) = c["Phi"], c["Phi1"], c["Phi2"]
        ya = torch.where(a.abs() <= 1.0, a, torch.zeros_like(a))                      # the boundary term carries the factor y_j = a_j only inside the support
        yb = torch.where(b.abs() <= 1.0, b, torch.zeros_like(b))
        red = lambda v: (sg * v).reshape(4, P).sum(0)
        lam = red(ph)
        m = torch.stack([red(p1), red(p2)], 1)
        g0 = -torch.stack([red(pa), red(pb)], 1)
        g1 = -torch.stack([torch.stack([red(ya * pa - ph), red(p1b)], 1),               # d = 1: j = 1 boundary a dPhi/da - Phi, j = 2 boundary dPhi_1/db
                           torch.stack([red(p2a), red(yb * pb - ph)], 1)], 1)           # d = 2: j = 1 boundary dPhi_2/da,       j = 2 boundary b dPhi/db - Phi
        return lam, m, g0, g1
