"""The PARTICLE representation of a boundary as a provider with the same contract as the analytic one (`provider.AnalyticBoundary`, docs/audit-warpsph-boundary-hooks.md §4): the solid of every body sampled by a
lattice of wall particles (spacing s, area s^2 each, in the body frame, moving with the body), and the aggregates as plain pair sums over them,

    lam   = sum_b dA W(|x - x_b|)            G = sum_b dA grad_x W            Cov = sum_b dA y (x) grad_x W   (y = x_b - x)
    cover, lap, tens                         the same sums of the cover kernel, the Laplacian and W^4 grad W, divided by the factors the consumers multiply back (scene/viscosity.py, scene/tensile.py)
    A     = sum_b dA (a1 . y) grad_x W       cone_area(axis, half angle) = sum of dA over wall particles within one support, inside the wedge

i.e. exactly the integrals the edge reductions evaluate in closed form, as quadratures with the lattice as the rule.  Two uses: the oracle of the cross-representation convergence test (the analytic integrals
are the continuum limit of the wall-particle sums: first order in s / h, the indicator of the solid is sampled per cell), and the aggregate a boundary-PARTICLE scheme (mDBC) can hand to the same scheme code.  Brute
force over queries x wall particles within a support (cell-free, chunked): a test oracle, not a production neighbour search.
"""
import math

import numpy as np
import torch

from ..edge import kernels
from .boundaryOps import kernelName
from .fused import WallAggregate
from .provider import BoundaryProvider, _FAMILY
from .scene import Scene
from .tensile import tensile_factor
from .viscosity import lap_factor

F64 = torch.float64


def _profile(name, r, h, order):
    """W_h(r) (order 0), W_h'(r) (1) or W_h''(r) (2) of the registered kernel `name` with support radius h (units 1/length^2, ^3, ^4); r > 0."""
    q = r / h
    out = torch.zeros_like(q)
    for b in kernels.kernel(name).blocks:
        R = float(b.R)
        c = [float(x) for x in b.coeffs]
        d = c
        for _ in range(order):
            d = [n * d[n] for n in range(1, len(d))] or [0.0]
        v = torch.zeros_like(q)
        for coef in reversed(d):
            v = v * q + coef
        out = out + torch.where(q < R, v, torch.zeros_like(v))
    return out / (math.pi * h ** (2 + order))


class ParticleAggregate(WallAggregate):
    """the aggregate of a `ParticleBoundary` at one position set: `.out` (as `FusedWall.out`), `evaluate` for the hydrostatic term, `cone_area`."""

    def __init__(self, provider, ps, support, kernel, laplacian):
        self.provider, self.H, self.kernel, self.N = provider, float(support), kernel, len(ps.positions)
        self.name = kernelName(kernel)
        self.fam = provider.family(kernel)
        self.x = ps.positions.to(provider.scene.device, F64)
        self._pairs = None
        self.out = self._static(laplacian)

    def _bodies(self):
        """per body: the wall particles in the world frame [P, 2] and their area."""
        for b, (pts, dA) in zip(self.provider.scene.bodies, self.provider.points(self.H)):
            yield b, b.pose.toWorld(pts), dA

    def _per_body(self, fn):
        """[B, N, ...] with fn(y [n,P,2], r [n,P], mask [n,P], dA) -> [n, ...] summed over the wall particles of every body, in chunks of queries."""
        outs = []
        for b, X, dA in self._bodies():
            rows = []
            for lo in range(0, self.N, 64):
                x = self.x[lo:lo + 64]
                y = X[None, :, :] - x[:, None, :]
                r = y.norm(dim=2)
                m = (r < self.H) & (r > 1e-14)
                rows.append(fn(y, r.clamp(min=1e-14), m, dA))
            outs.append(torch.cat(rows))
        return torch.stack(outs)

    def _static(self, laplacian):
        H, nm, fam = self.H, self.name, self.fam
        zero = lambda t, m: torch.where(m, t, torch.zeros_like(t))
        w = lambda r, m: zero(_profile(nm, r, H, 0), m)
        dw = lambda r, m: zero(_profile(nm, r, H, 1), m)
        out = {}
        out["lam"] = self._per_body(lambda y, r, m, dA: dA * w(r, m).sum(1))
        grad = lambda y, r, m: (dw(r, m) / r)[..., None] * (-y)                       # grad_x W = W'(r) (x - x_b) / r
        out["G"] = self._per_body(lambda y, r, m, dA: dA * grad(y, r, m).sum(1))
        out["Cov"] = self._per_body(lambda y, r, m, dA: dA * torch.einsum("npa,npb->nab", y, grad(y, r, m)))
        cone = lambda r, m: zero(_profile("cone", r, H, 1), m)
        out["cover"] = self._per_body(lambda y, r, m, dA: dA * ((cone(r, m) / r)[..., None] * (-y)).sum(1))     # grad_x of the cone-kernel integral: the registered kernel's own normalisation, as FusedWall's `cone` group
        if laplacian:
            lapW = lambda r, m: zero(_profile(nm, r, H, 2) + _profile(nm, r, H, 1) / r, m)
            out["lap"] = self._per_body(lambda y, r, m, dA: dA * lapW(r, m).sum(1)) / lap_factor(H, fam)
        out["tens"] = self._per_body(lambda y, r, m, dA: dA * ((w(r, m) ** 4 * dw(r, m) / r)[..., None] * (-y)).sum(1)) / tensile_factor(H, fam)
        return out

    def evaluate(self, outputs, a1=None):
        """{name: [B, N, ...]} for the 'a1g1' output (A = sum_b dA (a1 . y) grad_x W with a1 [B, N, 2]) and the plain 'lam' / 'g0' / 'cov' of the main kernel."""
        res = {}
        for o in outputs:
            if o.kind == "a1g1":
                res[o.name] = torch.stack([self._a1g1(bi, a1[bi]) for bi in range(len(self.provider.scene.bodies))])
            else:
                res[o.name] = self.out[{"lam": "lam", "g0": "G", "cov": "Cov"}[o.kind]]
        return res

    def _a1g1(self, bi, a1):
        H, nm = self.H, self.name
        b, X, dA = list(self._bodies())[bi]
        rows = []
        for lo in range(0, self.N, 64):
            x = self.x[lo:lo + 64]
            y = X[None, :, :] - x[:, None, :]
            r = y.norm(dim=2)
            m = (r < H) & (r > 1e-14)
            g = (_profile(nm, r.clamp(min=1e-14), H, 1) / r.clamp(min=1e-14) * m)[..., None] * (-y)
            rows.append(dA * torch.einsum("nd,npd,npj->nj", a1[lo:lo + 64], y, g))
        return torch.cat(rows)

    def cone_area(self, axes, half_angle):
        """[2, N]: area of the solid within one support and within `half_angle` of `axes` (row 0), and of the solid within one support (row 1), summed over the bodies."""
        ax = axes.to(self.x.device, F64)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-300)
        res = torch.zeros((2, self.N), dtype=F64, device=self.x.device)
        for b, X, dA in self._bodies():
            for lo in range(0, self.N, 64):
                x = self.x[lo:lo + 64]
                y = X[None, :, :] - x[:, None, :]
                r = y.norm(dim=2)
                m = r < self.H
                cosang = (y * ax[lo:lo + 64, None, :]).sum(2) / r.clamp(min=1e-300)
                inside = m & (torch.acos(cosang.clamp(-1.0, 1.0)) <= half_angle)
                res[0, lo:lo + 64] += dA * inside.sum(1)
                res[1, lo:lo + 64] += dA * m.sum(1)
        return res


class ParticleBoundary:
    """the bodies of a `Scene` sampled by wall particles on a lattice of spacing `spacing`: a `BoundaryProvider` whose aggregates are quadratures (see the module docstring)."""

    def __init__(self, scene: Scene, spacing: float):
        self.scene = scene
        self.spacing = float(spacing)
        self._pts = {}

    @property
    def bodies(self):
        return self.scene.bodies

    def family(self, kernel):
        fam = _FAMILY.get(kernel)
        if fam is None:
            raise NotImplementedError("exact wall operations: Wendland C2 and C4 only")
        return fam

    def points(self, support):
        """per body: (wall particles in the body frame [P, 2], area per particle): the cell centres of the lattice, within one support of the body's bounding box, that lie inside the solid of the body."""
        key = float(support)
        if key not in self._pts:
            s, out = self.spacing, []
            for bi, b in enumerate(self.scene.bodies):
                lo, hi = [t.cpu().numpy() for t in b.obb()]
                lo, hi = np.maximum(lo, -1e6) - support, np.minimum(hi, 1e6) + support
                ax = [lo[d] + s * (np.arange(int(math.ceil((hi[d] - lo[d]) / s))) + 0.5) for d in range(2)]
                P = torch.as_tensor(np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 2), dtype=F64, device=self.scene.device)
                keep = self._inside_local(b, P)
                out.append((P[keep], s * s))
            self._pts[key] = out
        return self._pts[key]

    @staticmethod
    def _inside_local(body, P):
        """the solid of the body at body-frame points P: the union over its representations (a `Scene.inside` for a body at the identity pose)."""
        from .scene import Body, Scene as _Scene
        probe = Body(bodyId=body.bodyId, reps=body.reps)
        probe.center = probe.center.to(P.device)
        return _Scene([probe], str(P.device)).inside(P)

    def aggregate(self, ps, support, kernel=None, laplacian=False, fixedAdjacency=True):
        from warpSPHCore import KernelFunctions
        return ParticleAggregate(self, ps, support, KernelFunctions.Wendland2 if kernel is None else kernel, laplacian)

    def signed_distance(self, x, body=None, supportMax=None, want_body=False):
        return self.scene.signed_distance(x, body=body, supportMax=supportMax, want_body=want_body)
