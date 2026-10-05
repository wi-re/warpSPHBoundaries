"""Tiers 3 and 4 for implicit (SDF) bodies in the boundary-operation interface, with HARD tier switches (no blending).

For a body the three models give, per (particle, body) pair, the kernel integral  lambda = int_body W  and its gradient  grad_x lambda  (support radius
h = supports[i], everything in units of h inside):

    tier 3  closest-point curvature expansion  lambda = F0(q) + kappa F1(q) + kappa^2 F2(q),  q = d/h, kappa -> h*kappa     (`tier3.py`, Maple-derived)
    tier 4  small obstacle: disk mean-value series  pi a^2 sum_k a^2k Delta^k W / (4^k k!(k+1)!)  at the centre distance    (`tier4.py`)
    tier 2  exact elements of a polygonisation of the body (handled by `boundaryOps`)

The 1D functions F_k(q), F_k'(q) are tabulated once per kernel (exact mpmath evaluation of the half-plane moments, cached in results/tables/) and
evaluated by cubic Hermite interpolation; tier 4 uses exact polynomial coefficients of Delta^k W.  Supported bodies: `DiskBody` (solid disk or cavity) and
`HalfPlaneBody` (tier 3 with kappa = 0 is EXACT = the planar closed form).  Supported fields: constants per body (what a static or uniformly moving wall needs):
the operations are  lambda * rho_b (Density), lambda * A (Interpolate),  a (x) grad lambda (Gradient), a . grad lambda, curl (Divergence, Curl).
Fields varying along the surface need the moment versions of F_k (same derivation, not done): use tier 2 elements for those.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np
import sympy as sp
import torch

from ..edge.kernels import kernel as get_kernel
from .. import paths

TABLE_DIR = paths.tables_dir()
NGRID = 512
QMIN = 1e-6


# ------------------------------------------------------------------------------------------------ tier 3 tables
def build_tier3_table(kname: str, n: int = NGRID, verbose=False):
    """F_k(q), F_k'(q) (k = 0,1,2) at q = j/n (q_0 = QMIN), exact half-plane moments in mpmath, 4th-order differences for F'."""
    import mpmath as mp
    from ..edge import tier3
    qs = [QMIN] + [j / n for j in range(1, n + 1)]
    F = np.zeros((3, len(qs)))
    dF = np.zeros((3, len(qs)))
    hh = Fraction(1, 10 ** 7)
    for j, q in enumerate(qs):
        d = Fraction(q).limit_denominator(10 ** 12) if j else Fraction(1, 10 ** 6)
        for k in range(3):
            f = lambda dd: tier3.F_k(kname, dd, k, dps=30)
            F[k, j] = float(f(d))
            dF[k, j] = float((8 * (f(d + hh) - f(d - hh)) - (f(d + 2 * hh) - f(d - 2 * hh))) / (12 * mp.mpf(hh.numerator) / hh.denominator)) if d + 2 * hh < 1 else (0.0 if d >= 1 else float((f(d) - f(d - hh)) / (mp.mpf(hh.numerator) / hh.denominator)))
        if verbose and j % 64 == 0:
            print("tier3 table", kname, j, "/", len(qs))
    return np.array(qs), F, dF


def load_tier3_table(kname: str):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    f = TABLE_DIR / f"tier3_{kname}.npz"
    if not f.exists():
        q, F, dF = build_tier3_table(kname)
        np.savez(f, q=q, F=F, dF=dF)
    z = np.load(f)
    return z["q"], z["F"], z["dF"]


class Tier3:
    def __init__(self, kname: str, device):
        q, F, dF = load_tier3_table(kname)
        self.q = torch.as_tensor(q, dtype=torch.float64, device=device)
        self.F = torch.as_tensor(F, dtype=torch.float64, device=device)
        self.dF = torch.as_tensor(dF, dtype=torch.float64, device=device)
        # Lambda(q) = int_q^1 F0(s) ds (exact integral of the Hermite spline), for the first moment of a half plane
        hh = self.q[1:] - self.q[:-1]
        seg = 0.5 * hh * (self.F[0][:-1] + self.F[0][1:]) + hh * hh * (self.dF[0][:-1] - self.dF[0][1:]) / 12.0
        self.Lam = torch.cat([torch.flip(torch.cumsum(torch.flip(seg, [0]), 0), [0]), torch.zeros(1, dtype=torch.float64, device=device)])

    def planar_moments(self, d):
        """first moments of the half plane solid at SIGNED distance d / h (positive on the fluid side, units of h), valid for d >= 0 and (by the complement) d < 0:
        returns (lam, dlam/dq, m1n, g1nn) with
            m1 = int_solid y W dA  = -m1n n           (n: unit normal into the fluid, y = x' - x; no tangential part)
            g1 = int_solid y (x) grad_x W dA = g1nn n (x) n + g1tt t (x) t,   g1tt = lam,   g1nn = lam - q lam'
        all in units of h (m1n: h, g1: dimensionless)."""
        a = d.abs()
        lam0, dl0 = self._hermite(0, a)
        # Lambda by Hermite interpolation (derivative -F0)
        qq = a.clamp(float(self.q[0]), 1.0)
        idx = torch.bucketize(qq, self.q[1:-1].contiguous())
        q0, q1 = self.q[idx], self.q[idx + 1]
        hq = q1 - q0
        t_ = (qq - q0) / hq
        f0, f1 = self.Lam[idx], self.Lam[idx + 1]
        m0, m1_ = -self.F[0][idx] * hq, -self.F[0][idx + 1] * hq
        t2, t3 = t_ * t_, t_ * t_ * t_
        Lam = (2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t_) * m0 + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * m1_
        Lam = torch.where(a < 1.0, Lam, torch.zeros_like(Lam))
        m1n_c = a * lam0 + Lam                       # complement / solid-side half plane moment (positive d)
        g1nn_c = lam0 - a * dl0
        g1tt_c = lam0
        pos = d >= 0
        lam = torch.where(pos, lam0, 1.0 - lam0)
        dlam = torch.where(pos, dl0, dl0)           # d lambda / d d is the same for both sides (see `lam`)
        m1n = m1n_c                                 # d < 0: the solid contains the particle, m1 = -(m1 of the fluid complement) = -m1n_c n as well (the full-plane first moment is 0)
        g1nn = torch.where(pos, g1nn_c, 1.0 - g1nn_c)
        g1tt = torch.where(pos, g1tt_c, 1.0 - g1tt_c)
        return lam, dlam, m1n, g1nn, g1tt

    def _hermite(self, k, q):
        """F_k(q), F_k'(q) by cubic Hermite interpolation (q clamped to the table, F = 0 for q >= 1)."""
        qq = q.clamp(float(self.q[0]), 1.0)
        idx = torch.bucketize(qq, self.q[1:-1].contiguous())          # interval index 0..n-1
        q0, q1 = self.q[idx], self.q[idx + 1]
        h = q1 - q0
        t = (qq - q0) / h
        f0, f1 = self.F[k][idx], self.F[k][idx + 1]
        m0, m1 = self.dF[k][idx] * h, self.dF[k][idx + 1] * h
        t2, t3 = t * t, t * t * t
        val = (2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * m1
        der = ((6 * t2 - 6 * t) * f0 + (3 * t2 - 4 * t + 1) * m0 + (-6 * t2 + 6 * t) * f1 + (3 * t2 - 2 * t) * m1) / h
        # below the first node (q_0 = QMIN): linear extrapolation with the node derivative (exact at the wall to ~1e-12)
        below = q < qq
        val = torch.where(below, self.F[k][0] + self.dF[k][0] * (q - self.q[0]), val)
        der = torch.where(below, self.dF[k][0].expand_as(der), der)
        out = q < 1.0
        return torch.where(out, val, torch.zeros_like(val)), torch.where(out, der, torch.zeros_like(der))

    def lam(self, d, kappa):
        """lambda and d lambda/d d for SIGNED distance d (positive outside the solid, units of h) and curvature kappa (units 1/h; > 0 convex solid):
        d >= 0:  F0 + k F1 + k^2 F2 ;   d < 0 (inside the solid):  1 - [F0(|d|) - k F1(|d|) + k^2 F2(|d|)]  (complement body, curvature -k)."""
        a = d.abs()
        s = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))            # sign of kappa seen from the particle's side
        kk = kappa * s
        v0, d0 = self._hermite(0, a)
        v1, d1 = self._hermite(1, a)
        v2, d2 = self._hermite(2, a)
        v = v0 + kk * v1 + kk * kk * v2
        dv = d0 + kk * d1 + kk * kk * d2                                            # derivative wrt |d|
        lam = torch.where(d >= 0, v, 1.0 - v)
        # outside: d lambda/d d = dv ; inside: lambda = 1 - v(|d|), d|d|/dd = -1  ->  d lambda/d d = + dv
        return lam, dv


# ------------------------------------------------------------------------------------------------ tier 4 coefficients
def _laurent(expr):
    """sympy expression in r -> [(power, coefficient)] (powers may be negative)."""
    r = sp.Symbol("r", real=True)
    out = []
    for term in sp.expand(expr).as_ordered_terms():
        c, rest = term.as_independent(r)
        p = 0 if rest == 1 else int(sp.degree(rest, r)) if rest.is_polynomial(r) else int(sp.Poly(rest * r ** 6, r).degree() - 6)
        out.append((p, float(c)))
    return out


class Tier4:
    """disk mean-value series, K = 2, exact polynomial coefficients of Delta^k (pi W) per kernel piece."""

    def __init__(self, kname: str, device, K: int = 2):
        from ..edge import tier4 as t4
        self.K = K
        self.knots = [float(p[1]) for p in get_kernel(kname).pieces]                # upper knots, last = 1
        self.device = device
        self.terms = []                                                              # [piece][k] -> (powers, coefs)
        for w in range(len(get_kernel(kname).pieces)):
            row = []
            for k in range(K + 1):
                lau = _laurent(t4._lap_k(kname, w, k))
                row.append((torch.as_tensor([p for p, _ in lau], dtype=torch.float64, device=device),
                            torch.as_tensor([c for _, c in lau], dtype=torch.float64, device=device)))
            self.terms.append(row)
        self.kn = torch.as_tensor([0.0] + self.knots, dtype=torch.float64, device=device)

    def lam(self, D, a):
        """(lambda, d lambda / d D, valid) for disks of radius a at centre distance D (units of h); valid: x outside and [D-a, D+a] within one kernel piece."""
        lo, hi = D - a, D + a
        piece_lo = torch.bucketize(lo, self.kn[1:-1].contiguous(), right=True)
        piece_hi = torch.bucketize(hi, self.kn[1:-1].contiguous(), right=False)
        valid = (lo > 0) & (hi < 1.0) & (piece_lo == piece_hi)
        val = torch.zeros_like(D)
        der = torch.zeros_like(D)
        Dc = D.clamp(min=1e-12)
        for w, row in enumerate(self.terms):
            sel = valid & (piece_lo == w)
            if not bool(sel.any()):
                continue
            tot = torch.zeros_like(D)
            dtot = torch.zeros_like(D)
            for k, (pw, cf) in enumerate(row):
                f = sum(c * Dc ** p for p, c in zip(pw.tolist(), cf.tolist()))
                df = sum(c * p * Dc ** (p - 1) for p, c in zip(pw.tolist(), cf.tolist()))
                fac = a ** (2 * k + 2) / (4 ** k * _fact(k) * _fact(k + 1))
                tot = tot + fac * f
                dtot = dtot + fac * df
            val = torch.where(sel, tot, val)
            der = torch.where(sel, dtot, der)
        return val, der, valid


def _fact(n):
    return float(np.prod(np.arange(1, n + 1))) if n > 0 else 1.0


# ------------------------------------------------------------------------------------------------ bodies
@dataclass
class TierPolicy:
    """hard thresholds (units of the particle's own h), from docs/tier-selection-2d.md:  R/h <= tier4MaxRadius -> series, R/h >= tier3MinRadius -> curvature expansion,
    else exact elements; per particle (adaptive h), no blending."""
    tier4MaxRadius: float = 0.2
    tier3MinRadius: float = 2.0
    maxPolygonEdgeOverH: float = 1.0 / 16.0


@dataclass
class DiskBody:
    center: Tuple[float, float]
    radius: float
    solid: str = "inside"           # 'inside': solid disk (obstacle, kappa = +1/R); 'outside': cavity wall (fluid inside the circle, kappa = -1/R)
    bodyId: int = 0

    def signed(self, pos):
        """(d, n, kappa): signed distance (positive in the fluid), unit normal pointing into the fluid, curvature."""
        c = torch.as_tensor(self.center, dtype=torch.float64, device=pos.device)
        v = pos - c
        r = v.norm(dim=1).clamp(min=1e-300)
        e = v / r[:, None]
        if self.solid == "inside":
            return r - self.radius, e, 1.0 / self.radius
        return self.radius - r, -e, -1.0 / self.radius

    def polygon(self, h_min, policy: TierPolicy):
        """inscribed fan triangulation used by tier 2 (solid disk only; cavities use the complement mesh outside: not supported -> raise)."""
        if self.solid != "inside":
            raise NotImplementedError("tier-2 fallback for cavity walls: supply an explicit BoundaryMesh for the gap regime")
        n = max(24, int(np.ceil(2 * np.pi * self.radius / (policy.maxPolygonEdgeOverH * h_min))))
        # area-preserving radius: polygon area equals the disk area (removes the O(1/N^2) mean bias)
        rN = self.radius * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
        t = 2 * np.pi * np.arange(n) / n
        V = np.vstack([np.array([self.center]), np.stack([self.center[0] + rN * np.cos(t), self.center[1] + rN * np.sin(t)], 1)])
        E = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
        return V, E


@dataclass
class HalfPlaneBody:
    point: Tuple[float, float]
    normal: Tuple[float, float]     # unit normal pointing INTO THE FLUID
    bodyId: int = 0

    def signed(self, pos):
        n = torch.as_tensor(self.normal, dtype=torch.float64, device=pos.device)
        p = torch.as_tensor(self.point, dtype=torch.float64, device=pos.device)
        d = ((pos - p) * n).sum(1)
        return d, n.expand_as(pos), 0.0


def evaluateBody(body, positions, supports, kernel_name, device, policy: TierPolicy, tables):
    """(lambda [N], grad lambda [N,2], tier [N] int, active [N] bool).  tier: 3, 4, or 2 (= use the exact elements of the polygonisation; lambda/grad zero here)."""
    pos = positions.to(device, torch.float64)
    h = supports.to(device, torch.float64)
    d, n, kappa = body.signed(pos)
    N = len(pos)
    q = d / h
    lam = torch.zeros(N, dtype=torch.float64, device=device)
    dl = torch.zeros(N, dtype=torch.float64, device=device)                     # d lambda / d q  (dimensionless in q)
    tier = torch.full((N,), 3, dtype=torch.int32, device=device)
    active = q < 1.0                                                           # beyond one support: no overlap (solid) ... or fully inside handled by lambda
    if isinstance(body, HalfPlaneBody):
        t3 = tables["t3"]
        lam, dl = t3.lam(q, torch.zeros_like(q))
        lam = torch.where(q <= -1.0, torch.ones_like(lam), lam)
        dl = torch.where(q <= -1.0, torch.zeros_like(dl), dl)
        grad = (dl / h)[:, None] * n
        return lam, grad, tier, active
    R = body.radius
    Rh = R / h
    use4 = Rh <= policy.tier4MaxRadius
    use3 = Rh >= policy.tier3MinRadius
    # tier 3
    t3 = tables["t3"]
    kap = (kappa * h)                                                          # kappa h  (dimensionless)
    lam3, dl3 = t3.lam(q, kap)
    # tier 4 (solid disks only; x outside; series valid on one kernel piece)
    lam4 = torch.zeros_like(lam)
    dl4 = torch.zeros_like(lam)
    ok4 = torch.zeros(N, dtype=torch.bool, device=device)
    if body.solid == "inside" and bool(use4.any()):
        D = (pos - torch.as_tensor(body.center, dtype=torch.float64, device=device)).norm(dim=1) / h
        v4, dv4, ok4 = tables["t4"].lam(D, Rh)
        lam4, dl4 = v4, dv4
    sel4 = use4 & ok4
    sel3 = use3
    sel2 = ~(sel3 | sel4)
    lam = torch.where(sel3, lam3, torch.where(sel4, lam4, lam))
    dl = torch.where(sel3, dl3, torch.where(sel4, dl4, dl))
    tier = torch.where(sel3, torch.full_like(tier, 3), torch.where(sel4, torch.full_like(tier, 4), torch.full_like(tier, 2)))
    # gradient: tier 3: d lambda/d d along n ; tier 4: d lambda/d D along the radial direction (= n for solid disks)
    grad = (dl / h)[:, None] * n
    grad = torch.where(sel2[:, None], torch.zeros_like(grad), grad)
    lam = torch.where(sel2, torch.zeros_like(lam), lam)
    return lam, grad, tier, active


if __name__ == "__main__":      # python -m edgebound.scene.implicitBodies : build/cache the tier-3 tables of all supported kernels (~30 s each)
    import time
    for name in ["cubic", "quartic", "quintic", "b7", "b8", "poly6", "w2", "w4", "w6"]:
        t = time.time()
        load_tier3_table(name)
        print(f"{name}: {time.time() - t:.0f}s", flush=True)
