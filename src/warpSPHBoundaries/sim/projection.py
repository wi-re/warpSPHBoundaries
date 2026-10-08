"""The compact approximate projection shared by the schemes (docs/plan-next-steps.md "DFSPH in closed periodic domains": a CONVERGED projection that does not damp the resolved flow).

The divergence-free correction  v = v* - dt grad P / rho  with P from the compact (Morris / Brookshaw) Laplacian instead of the composed div(grad) of the scheme's own operators (Cummins & Rudman 1999): the composed operator,
converged, is an exact discrete projection that also removes the divergence noise of moving particles and damps the resolved flow (TGV decay 1.38x analytic at 100 iterations); the compact one does not (1.00-1.01x
converged).  The schemes supply the source and apply their own pressure gradient; this module solves

    sum_j w_ij (P_i - P_j) + D_i P_i = rhs_i ,   w_ij = 2 Vt_i Vt_j (x_ij . grad W_ij) / (r_ij^2 + eta^2 h_i^2) / cal  <= 0   (symmetric; ~ Vt_i lap P_i)

D_i <= 0 is the wall row: the response of the scheme's own wall divergence term to the scheme's own wall pressure force on particle i (a Robin condition; without it the walls are Neumann, and a scheme whose wall
correction moves the near-wall flux more than the fluid Laplacian predicts overshoots every step: delta+ on a disk grew 1.4x per step).

for P by Jacobi-preconditioned CG to a relative residual `tol` (closed domains: the constant null space, the right-hand side's mean removed by the caller).  On CUDA the CG runs as Warp kernels over the CSR pairs
(dfsph_kernels.py), `block` iterations per CUDA graph replay, with the device flag `active` making iterations past convergence no-ops; the host reads the flag only at the check schedule of warpSPH's ACSPH
solve (0.50 / 0.75 / 0.85 of the previous solve's first-converged count, then every block).  Warm start from the previous solution (the true residual decides convergence).
"""
import math
from typing import Optional

import torch

F64 = torch.float64


def compact_weights(d, gW, Vt_i, Vt_j, h_i, cal, eta2):
    """w_ij of the compact Laplacian for pairs with separation d = x_i - x_j and gW = grad_i W_ij (self pairs: 0)."""
    return 2.0 * Vt_i * Vt_j * (d * gW).sum(1) / ((d * d).sum(1) + eta2 * h_i ** 2) / cal


class CompactProjection:
    def __init__(self, device, block: int = 8, warmStart: bool = True, graphs: bool = True):
        self.dev = device
        self.block = max(1, int(block))
        self.warmStart = warmStart
        self.graphs = graphs and str(device).startswith("cuda") and torch.cuda.is_available()
        self.prevCount = None
        self.prevP = None
        self._buf = {}
        self._graph = None
        self.stats = {"solves": 0, "reads": 0, "captures": 0, "iterations": 0}

    def _pin(self, name, t):
        b = self._buf.get(name)
        if b is None or b.shape != t.shape or b.dtype != t.dtype:
            self._buf[name] = t.clone()
            self._graph = None
            return self._buf[name]
        b.copy_(t)
        return b

    def _pin_pairs(self, nbr, w):
        P = len(nbr)
        cap = self._buf["nbr"].shape[0] if "nbr" in self._buf else 0
        if P > cap:
            cap = int(math.ceil(1.25 * P / 1024.0) * 1024)
            self._buf["nbr"] = torch.zeros(cap, dtype=torch.int64, device=self.dev)
            self._buf["w"] = torch.zeros(cap, dtype=F64, device=self.dev)
            self._graph = None
        nb, wb = self._buf["nbr"], self._buf["w"]
        nb[:P].copy_(nbr.to(torch.int64))
        nb[P:].zero_()
        wb[:P].copy_(w)
        wb[P:].zero_()
        return nb, wb

    def solve(self, offsets, counts, nbr, w, rhs, tol=1e-8, maxIterations=2000, diag=None, dirichlet=None):
        """P with sum_j w_ij (P_i - P_j) + diag_i P_i = rhs_i over the CSR pairs (offsets / counts per particle into nbr / w, i-sorted).  `dirichlet` (bool mask): P_i = 0 there (a free surface), eliminated
        symmetrically (the rows become P_i = 0, a neighbour's coupling to a fixed P_j = 0 moves to its diagonal).  Returns (P, iterations, relative residual)."""
        idx_i = torch.repeat_interleave(torch.arange(len(rhs), device=self.dev), counts.long())
        D = diag.clone() if diag is not None else torch.zeros_like(rhs)
        if dirichlet is not None:
            fixed = dirichlet.to(torch.bool)
            jl = nbr.long()
            toFixed = fixed[jl] & ~fixed[idx_i]
            D = D.index_add(0, idx_i, torch.where(toFixed, w, torch.zeros_like(w)))               # w_ij (P_i - 0): the coupling becomes diagonal
            w = torch.where(fixed[idx_i] | fixed[jl], torch.zeros_like(w), w)
            ref = D.abs().mean() + torch.zeros_like(rhs).index_add_(0, idx_i, w.abs()).mean()
            D = torch.where(fixed, -ref.expand_as(D), D)                                            # P_i = 0 (any scale: the row decouples)
            rhs = torch.where(fixed, torch.zeros_like(rhs), rhs)
        Aop = lambda p: torch.zeros_like(rhs).index_add_(0, idx_i, w * (p[idx_i] - p[nbr.long()])) + D * p
        scale = float(rhs.norm())
        self.stats["solves"] += 1
        if scale <= 1e-30:
            self.prevP = torch.zeros_like(rhs)
            return self.prevP.clone(), 0, 0.0
        b = -rhs / scale                                                                            # (-A) P = -rhs, normalised (a near-solenoidal field underflows the inner products otherwise)
        x0 = self.prevP / scale if (self.warmStart and self.prevP is not None and self.prevP.shape == rhs.shape) else None
        dg = torch.zeros_like(rhs).index_add_(0, idx_i, w) + D
        Minv = torch.where(dg.abs() > 0, -1.0 / dg, torch.zeros_like(dg))
        if self.graphs:
            P, it, rel = self._cg_graphed(offsets, counts, nbr, w, D, Minv, b, x0, Aop, tol, maxIterations)
        else:
            P, it, rel = self._cg_eager(Minv, b, x0, Aop, tol, maxIterations)
        P = P * scale
        self.prevP = P.clone()
        self.stats["iterations"] += it
        return P, it, rel

    def _cg_eager(self, Minv, b, x0, Aop, tol, maxIterations):
        p = x0.clone() if x0 is not None else torch.zeros_like(b)
        r = b + Aop(p) if x0 is not None else b.clone()
        z = Minv * r
        q = z.clone()
        rz = (r * z).sum()
        it, rel = 0, float(r.norm())
        if rel < tol:
            return p, 0, rel
        for it in range(1, maxIterations + 1):
            Aq = -Aop(q)
            alpha = rz / (q * Aq).sum()
            p = p + alpha * q
            r = r - alpha * Aq
            if it % 10 == 0:
                rel = float(r.norm())
                if not math.isfinite(rel) or rel < tol:
                    break
            z = Minv * r
            rz2 = (r * z).sum()
            q = z + (rz2 / rz) * q
            rz = rz2
        return p, it, float(r.norm())

    def _cg_graphed(self, offsets, counts, nbr, w, D, Minv, b, x0, Aop, tol, maxIterations):
        import warp as wp
        from .dfsph_kernels import cg_matvec_kernel, cg_update_kernel, cg_direction_kernel, cg_scalars_kernel
        N = len(b)
        nb, wb = self._pin_pairs(nbr, w)
        off = self._pin("off", offsets.to(torch.int32))
        cnt = self._pin("cnt", counts.to(torch.int32))
        Mv = self._pin("Minv", Minv)
        Dv = self._pin("D", D)
        pv = self._pin("p", x0 if x0 is not None else torch.zeros_like(b))
        r = self._pin("r", b + Aop(x0) if x0 is not None else b)                                    # r0 = b - (-A) x0
        z = self._pin("z", Mv * r)
        q = self._pin("q", z)
        Aq = self._pin("Aq", torch.zeros_like(b))
        zero = torch.zeros((), dtype=F64, device=self.dev)
        rr0 = (r * r).sum()
        S = self._pin("S", torch.stack([(r * z).sum(), zero, zero, rr0, (rr0 > tol ** 2).to(F64), zero, torch.full((), tol ** 2, dtype=F64, device=self.dev), rr0]))
        f1 = lambda t: wp.from_torch(t, dtype=wp.float64)
        B = self.block

        def block():
            for _ in range(B):
                wp.launch(cg_matvec_kernel, dim=N, inputs=[wp.from_torch(off, dtype=wp.int32), wp.from_torch(cnt, dtype=wp.int32), wp.from_torch(nb, dtype=wp.int64), f1(wb), f1(Dv), f1(q), f1(Aq)])
                torch.dot(q, Aq, out=S[1])
                wp.launch(cg_update_kernel, dim=N, inputs=[f1(S), f1(q), f1(Aq), f1(Mv), f1(pv), f1(r), f1(z)])
                torch.dot(r, z, out=S[2])
                torch.dot(r, r, out=S[3])
                wp.launch(cg_direction_kernel, dim=N, inputs=[f1(S), f1(z), f1(q)])
                wp.launch(cg_scalars_kernel, dim=1, inputs=[f1(S)])

        if self._graph is None:                                                                     # a warm-up compiles the kernels, the state is restored, then one capture (the buffers are persistent)
            saved = [t.clone() for t in (S, pv, r, z, q)]
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
                block()
            for t, t0 in zip((S, pv, r, z, q), saved):
                t.copy_(t0)
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream()
            cap.wait_stream(torch.cuda.current_stream())
            with wp.ScopedStream(wp.stream_from_torch(cap)), torch.cuda.graph(g, stream=cap, capture_error_mode="thread_local"):
                block()
            torch.cuda.current_stream().wait_stream(cap)
            self._graph = g
            self.stats["captures"] += 1
        prev = self.prevCount
        marks = sorted({max(1, math.ceil(fr * prev)) for fr in (0.50, 0.75, 0.85)}) if prev else []
        done = 0
        if float(S[4].item()) > 0.5:
            while done < maxIterations:
                self._graph.replay()
                done += B
                if marks and done < marks[0]:
                    continue
                while marks and done >= marks[0]:
                    marks.pop(0)
                self.stats["reads"] += 1
                if S[4].item() < 0.5:                                                                # the one host read of a check
                    break
        vals = S.tolist()
        self.prevCount = int(vals[5]) if vals[5] > 0 else self.prevCount
        return pv.clone(), int(vals[5]), math.sqrt(max(vals[7], 0.0))
