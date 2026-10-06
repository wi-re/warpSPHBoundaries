"""Moments of the pair-viscosity weight over the solid half-plane: the wall term of the no-slip viscous closure for a field that is a polynomial of the wall distance.

The fluid pair sum of the solver is  acc_i = fac sum_j V_j / rho_ij  P(r_ij) (v_i - v_j),  P(r) = (L(r) / r^2) r (x) r,  L = W'(r) / r  (a negative semi-definite tensor weight; `mu gW` of the code).  Near a wall
the sum is truncated: the missing part is the same integral over the solid region S with the continuation u_ext of the velocity field,

    A_w = fac (wm / rho_i) int_S P(y) (u_i - u_ext(x')) dA',     y = x' - x_i,

and a continuation that is a polynomial of the signed wall distance s' (s' = y.n + d, n the unit normal into the fluid, d the distance of the particle), relative to the wall velocity w = u - u_wall,

    w_ext(s') = a s' + (L2 / 2) s'^2     (w = 0 at the wall: no slip)     =>     A_w = - beta [ T1 a + (T2 / 2) L2 - T0 w_i ],     beta = fac wm / rho_i,   Tk = int_S P(y) s'^k dA'.

(A rigid wall motion adds nothing: P annihilates every field with u_ij . r_ij = 0, rotation and translation.)  For a planar wall the tensors are diagonal in the frame (n, t) and depend on d only: `WallPairMoments` tabulates
T_k^{nn}, T_k^{tt}(d) (k = 0, 1, 2) once per kernel and support.  Units: T_k ~ H^(k - 2) / H^2 ... (the table is built at the solver's own H, no scaling needed).
"""
import math

import torch

F64 = torch.float64


class WallPairMoments:
    """T_k^{nn}(d), T_k^{tt}(d), k = 0, 1, 2 of the half-plane solid at distance d for the kernel gradient `dW(r, H)` (d in [0, H]; zero beyond)."""

    def __init__(self, dW, H, device, nd=257, nth=128, nr=64):
        import numpy as np
        gl_t, gl_w = (torch.as_tensor(a, dtype=F64) for a in np.polynomial.legendre.leggauss(nth))
        gr_t, gr_w = (torch.as_tensor(a, dtype=F64) for a in np.polynomial.legendre.leggauss(nr))
        d = torch.linspace(0.0, H, nd, dtype=F64)
        out = torch.zeros((2, 3, nd), dtype=F64)
        for i in range(nd):
            di = float(d[i])
            if di >= H:
                continue
            tm = math.acos(di / H)                                               # the solid is cos(theta) >= d / r (theta from the inward normal)
            th = 0.5 * tm * (gl_t + 1.0)                                         # theta in [0, tm], twice the integral (symmetric in theta)
            wth = 0.5 * tm * gl_w * 2.0
            r0 = di / torch.cos(th)                                              # [nth]
            r = r0[:, None] + 0.5 * (H - r0)[:, None] * (gr_t[None, :] + 1.0)    # [nth, nr]
            wr = 0.5 * (H - r0)[:, None] * gr_w[None, :]
            L = dW(r.reshape(-1), H).reshape(r.shape) / r                         # W'(r) / r
            s = di - r * torch.cos(th)[:, None]                                  # s' = d + y.n with y.n = -r cos(theta)
            for k in range(3):
                base = L * r * s ** k * wr                                       # L(r) s'^k r dr
                out[0, k, i] = (wth * (torch.cos(th) ** 2) * base.sum(1)).sum()
                out[1, k, i] = (wth * (torch.sin(th) ** 2) * base.sum(1)).sum()
        self.H = float(H)
        self.nd = nd
        self.table = out.to(device)                                              # [2 (nn, tt), 3 (k), nd]

    def eval(self, d):
        """(T [2, 3, N]) interpolated linearly in d; zero for d >= H."""
        f = (d / self.H).clamp(0.0, 1.0) * (self.nd - 1)
        i0 = f.floor().long().clamp(max=self.nd - 2)
        w = (f - i0)
        t = self.table[:, :, i0] * (1.0 - w) + self.table[:, :, i0 + 1] * w
        return t * (d < self.H).to(F64)
