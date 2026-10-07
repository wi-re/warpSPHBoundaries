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


# ------------------------------------------------------------------------------------------------------------------------ curved walls
def _gl(n, device):
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.as_tensor(x, dtype=F64, device=device), torch.as_tensor(w, dtype=F64, device=device)


MORRIS_ETA2 = 0.0025                                                             # eta^2 / H^2 of the Morris pair weight (warpSPH wp_viscosityDelta)


def curved_moments(dW, H, d, kw, nth=160, nr=48, weight="alpha"):
    """T_k^{c' c}(d, kappa_w) for a circular wall of signed curvature `kw` (> 0: solid disk of radius 1/kw, fluid outside; < 0: cavity of radius 1/|kw|, fluid inside; 0: the plane) at the distance `d` of the particle, by a
    polar quadrature of the actual solid region about the particle.  The continuation of the velocity relative to the wall is  w_ext(x') = sum_c (a_c s' + L_c s'^2 / 2) e_c(x')  with s' the signed distance and e_c(x') the (n, t) frame AT x'
    (the frame turns with the position, as in a flow along the wall); the tensor entries are  T_k^{c'c} = int_S L(r) (e_c' . y_hat)(y_hat . e_c(x')) s'^k dA'  with the fixed frame of the particle on the left;  k = 0 uses the fixed frame
    on both sides (the particle's own velocity).  Diagonal by the mirror symmetry t -> -t.  Returns [B, 3 (k), 2 (n, t)].  `d`, `kw` are [B] tensors, d in [0, H).
    `weight = "morris"`: the scalar Morris pair weight K(r) = W'(r) r / (r^2 + eta^2) times the identity instead of L(r) y_hat (x) y_hat: T_k^{c'c} = int_S K(r) (e_c' . e_c(x')) s'^k dA' (k = 0: int_S K dA' on both components).
    Slot k = 3 is the first moment of the solid M1_n = int_S K (y . n) dA' (Morris only; the tangential part vanishes by symmetry): the Morris weight does not annihilate a rigid ROTATION of the wall (the alpha weight does),
    the wall integral of v_wall(x_i + y) - v_wall(x_i) = Omega x y is Omega x M1.  Returns [B, 4, 2]."""
    dev = d.device
    B = d.shape[0]
    out = torch.zeros((B, 4, 2), dtype=F64, device=dev)
    xr, wr = _gl(nr, dev)
    xt, wt = _gl(nth, dev)
    flat = kw.abs() * H < 1e-6
    for sign in (1, -1, 0):
        sel = (flat if sign == 0 else (~flat & (kw * sign > 0))) & (d < H)
        if not bool(sel.any()):
            continue
        dd, kk = d[sel], kw[sel]
        nb = dd.shape[0]
        if sign == 0:
            R = torch.full_like(dd, 1e9)
        else:
            R = 1.0 / kk.abs()
        if sign >= 0:                                                            # convex (or plane): a solid disk
            cm2 = dd * (2 * R + dd) / (R + dd) ** 2                               # cos^2 of the tangent angle
            thmax = torch.acos(torch.sqrt(cm2.clamp(max=1.0)))
            v = 0.5 * (xt + 1.0)                                                  # theta = thmax (1 - v^2), v in [0, 1]
            th = thmax[:, None] * (1.0 - v[None, :] ** 2)                         # [nb, nth]
            jac = 2.0 * thmax[:, None] * v[None, :] * (0.5 * wt)[None, :] * 2.0   # |dtheta| = 2 thmax v dv, dv = 0.5 dx, times 2 (theta < 0 symmetric)
            Rp = (R + dd)[:, None]
            disc = (Rp * torch.cos(th)) ** 2 - (dd * (2 * R + dd))[:, None]
            sq = torch.sqrt(disc.clamp(min=0.0))
            r_lo = Rp * torch.cos(th) - sq
            r_hi = torch.minimum(Rp * torch.cos(th) + sq, torch.full_like(th, H))
            ok = r_lo < H
        else:                                                                       # cavity: the solid is outside the circle of radius R, centre at distance R - d along n
            th = 0.5 * math.pi * (xt + 1.0)[None, :].expand(nb, -1)                  # theta in [0, pi], symmetric in the sign of sin(theta)
            jac = (0.5 * math.pi * wt)[None, :].expand(nb, -1) * 2.0
            Rd = (R - dd)[:, None]
            b = Rd * torch.cos(th)
            r_lo = b + torch.sqrt(b ** 2 + (dd * (2 * R - dd))[:, None])
            r_hi = torch.full_like(th, H)
            ok = r_lo < H
        r = r_lo[:, :, None] + 0.5 * (r_hi - r_lo).clamp(min=0.0)[:, :, None] * (xr[None, None, :] + 1.0)      # [nb, nth, nr]
        wrr = 0.5 * (r_hi - r_lo).clamp(min=0.0)[:, :, None] * wr[None, None, :]
        if weight == "morris":
            L = dW(r.reshape(-1), H).reshape(r.shape) * r / (r * r + MORRIS_ETA2 * H * H)
        else:
            L = dW(r.reshape(-1), H).reshape(r.shape) / r
        ct, st = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
        if sign >= 0:                                                               # y = r (-cos, -sin) (towards the solid); c = -(R + d) n; frame at x': n' = (y - c) / |y - c|
            yx, yy = -r * ct, -r * st
            cx = -(R + dd)[:, None, None]
            ux, uy = yx - cx, yy
            rho = torch.sqrt(ux * ux + uy * uy)
            sp = rho - R[:, None, None]
            nn_x, nn_y = ux / rho, uy / rho
        else:
            yx, yy = r * ct, r * st                                                 # y = r (cos, sin); centre c = (R - d) n
            cx = (R - dd)[:, None, None]
            ux, uy = yx - cx, yy
            rho = torch.sqrt(ux * ux + uy * uy)
            sp = R[:, None, None] - rho
            nn_x, nn_y = -ux / rho, -uy / rho                                       # towards the centre = into the fluid
        yh_x, yh_y = yx / r, yy / r
        tt_x, tt_y = -nn_y, nn_x                                                    # tau' = J n'
        if weight == "morris":
            enn, ett = nn_x, tt_y                                                   # e_n . n', e_t . tau'
            f0n = f0t = torch.ones_like(yh_x)
        else:
            enn = yh_x * (yh_x * nn_x + yh_y * nn_y)                                # (e_n . y_hat)(y_hat . n')
            ett = yh_y * (yh_x * tt_x + yh_y * tt_y)                                # (e_t . y_hat)(y_hat . tau')
            f0n, f0t = yh_x * yh_x, yh_y * yh_y                                     # fixed frame (k = 0)
        w = (jac[:, :, None] * wrr * L * r) * ok[:, :, None]
        for k in range(3):
            sk = sp ** k
            if k == 0:
                out[sel, 0, 0] = (w * f0n).sum((1, 2))
                out[sel, 0, 1] = (w * f0t).sum((1, 2))
            else:
                out[sel, k, 0] = (w * enn * sk).sum((1, 2))
                out[sel, k, 1] = (w * ett * sk).sum((1, 2))
        if weight == "morris":
            out[sel, 3, 0] = (w * yx).sum((1, 2))                                   # y . n with n = e_x of the particle frame
    return out


class CurvedWallMoments:
    """`curved_moments` on a grid (d / H, kappa_w H) built once, bilinear lookup; kappa_w H in [-kmax, kmax] (R / H >= 1 / kmax)."""

    def __init__(self, dW, H, device, nd=65, nk=29, kmax=0.7, weight="alpha"):
        self.H, self.kmax, self.nd, self.nk, self.weight = float(H), float(kmax), nd, nk, weight
        dg = torch.linspace(0.0, 1.0, nd, dtype=F64, device=device) * H
        kg = torch.linspace(-kmax, kmax, nk, dtype=F64, device=device) / H
        D, K = torch.meshgrid(dg, kg, indexing="ij")
        tab = curved_moments(dW, H, D.reshape(-1), K.reshape(-1), weight=weight).reshape(nd, nk, 4, 2)
        self.table = tab.permute(3, 2, 0, 1).contiguous()                            # [2 (n, t), 4 (k = 0, 1, 2; 3 = M1), nd, nk]
        self.table[:, :, -1, :] = 0.0                                                # d = H: no solid within the support

    def eval(self, d, kw):
        """[2, 4, N] at distances `d` and wall curvatures `kw` (clamped to the grid)."""
        fd = (d / self.H).clamp(0.0, 1.0) * (self.nd - 1)
        fk = ((kw * self.H).clamp(-self.kmax, self.kmax) + self.kmax) / (2 * self.kmax) * (self.nk - 1)
        i0 = fd.floor().long().clamp(max=self.nd - 2)
        j0 = fk.floor().long().clamp(max=self.nk - 2)
        a, b = fd - i0, fk - j0
        t = self.table
        v = (t[:, :, i0, j0] * ((1 - a) * (1 - b)) + t[:, :, i0 + 1, j0] * (a * (1 - b)) + t[:, :, i0, j0 + 1] * ((1 - a) * b) + t[:, :, i0 + 1, j0 + 1] * (a * b))
        return v * (d < self.H).to(F64)


# ------------------------------------------------------------------------------------------------------------------------ corners (wedges)
def _wedge_nearest(P, beta):
    """nearest boundary point of the wedge solid {|atan2(y, x)| >= pi - beta / 2} (vertex at the origin, fluid bisector +x; beta < pi convex, > pi re-entrant) for points P [..., 2]:
    returns (s signed distance, + in the fluid, - in the solid; n the unit normal into the fluid at the nearest boundary point)."""
    a = math.pi - 0.5 * beta
    dA = torch.tensor([math.cos(a), math.sin(a)], dtype=F64, device=P.device)
    dB = torch.tensor([math.cos(a), -math.sin(a)], dtype=F64, device=P.device)
    out = []
    for dd in (dA, dB):
        t = (P * dd).sum(-1).clamp(min=0.0)                                         # projection on the ray (the vertex when negative)
        q = t[..., None] * dd
        out.append((q, (P - q).norm(dim=-1)))
    (qA, rA), (qB, rB) = out
    useA = rA <= rB
    q = torch.where(useA[..., None], qA, qB)
    r = torch.where(useA, rA, rB)
    ang = torch.atan2(P[..., 1], P[..., 0]).abs()
    solid = ang >= a
    v = P - q
    n = v / r.clamp(min=1e-300)[..., None]
    n = torch.where(solid[..., None], -n, n)                                         # into the fluid
    return torch.where(solid, -r, r), n, solid


def wedge_moments(dW, H, rho, phi, beta, weight="alpha", nth=1440, ng=24, fixed=False):
    """T_k^{c'c} (k = 0, 1, 2; full 2 x 2 in the particle frame (n, t), t = J n) and the first moment M1 (Morris) of the pair weight over the wedge solid within the support, for particles at polar coordinates (rho, phi)
    about the vertex (fluid bisector phi = 0).  Same continuation as `curved_moments`: w_ext(x') = sum_c (a_c s' + L_c s'^2 / 2) e_c(x') with s' the signed distance to the wedge boundary and e_c(x') the frame of its
    nearest boundary point; k = 0 uses the particle's frame on both sides.  Polar quadrature about the particle: midpoint in theta, Gauss-Legendre on the radial pieces between the crossings with the lines through the vertex
    where the integrand is not smooth (the two faces, the solid bisector, the face normals through the vertex).  Returns T [B, 3, 2, 2] (k, c', c), M1 [B, 2]; with `fixed=True` also Tf [B, 3, 2, 2], the same moments
    for the FIXED-frame continuation (the particle's frame on both sides, s~ = y . n + d: the continuation of the complement closure), so that T - Tf is the continuum difference of the two continuation models over
    the actual solid (the hybrid corner closure adds it to the complement moments)."""
    dev = rho.device
    B = rho.shape[0]
    xg, wg = _gl(ng, dev)
    a = math.pi - 0.5 * beta
    P0 = torch.stack([rho * torch.cos(phi), rho * torch.sin(phi)], 1)                # [B, 2]
    s0, n0, _ = _wedge_nearest(P0, beta)
    t0 = torch.stack([-n0[:, 1], n0[:, 0]], 1)
    th = (torch.arange(nth, dtype=F64, device=dev) + 0.5) * (2 * math.pi / nth)
    e = torch.stack([torch.cos(th), torch.sin(th)], 1)                               # [nth, 2]
    lines = [a, -a, math.pi, a + 0.5 * math.pi, a - 0.5 * math.pi, -a + 0.5 * math.pi, -a - 0.5 * math.pi]
    U = torch.tensor([[math.cos(g), math.sin(g)] for g in lines], dtype=F64, device=dev)  # [nl, 2] directions of the lines through the vertex
    cr = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    num = -cr(U[None, None], P0[:, None, None, :])                                   # [B, 1, nl]
    den = cr(U[None, None], e[None, :, None, :])                                     # [1, nth, nl]
    rc = num / torch.where(den.abs() < 1e-14, torch.full_like(den, 1e-14), den)
    rc = torch.where((rc > 0) & (rc < H), rc, torch.full_like(rc, H))
    bp = torch.cat([torch.zeros_like(rc[..., :1]), rc, torch.full_like(rc[..., :1], H)], -1).sort(-1).values   # [B, nth, nl + 2]
    lo, hi = bp[..., :-1], bp[..., 1:]                                               # segments [B, nth, ns]
    r = lo[..., None] + 0.5 * (hi - lo)[..., None] * (xg + 1.0)                      # [B, nth, ns, ng]
    wr = 0.5 * (hi - lo)[..., None] * wg
    y = r[..., None] * e[None, :, None, None, :]                                     # [B, nth, ns, ng, 2]
    X = P0[:, None, None, None, :] + y
    s, nq, solid = _wedge_nearest(X, beta)
    tq = torch.stack([-nq[..., 1], nq[..., 0]], -1)
    rr = r.clamp(min=1e-300)
    if weight == "morris":
        L = dW(rr.reshape(-1), H).reshape(rr.shape) * rr / (rr * rr + MORRIS_ETA2 * H * H)
    else:
        L = dW(rr.reshape(-1), H).reshape(rr.shape) / rr
    w = L * rr * wr * solid.to(F64) * (2 * math.pi / nth)                             # area element r dr dtheta
    fr = [n0[:, None, None, None, :], t0[:, None, None, None, :]]                    # particle frame (left)
    fq = [nq, tq]                                                                    # frame at x' (right)
    yh = y / rr[..., None]
    T = torch.zeros((B, 3, 2, 2), dtype=F64, device=dev)
    for c1 in range(2):
        for c2 in range(2):
            if weight == "morris":
                g0 = torch.full_like(L, 1.0 if c1 == c2 else 0.0)
                g = (fr[c1] * fq[c2]).sum(-1)
            else:
                g0 = (fr[c1] * yh).sum(-1) * (yh * fr[c2]).sum(-1)
                g = (fr[c1] * yh).sum(-1) * (yh * fq[c2]).sum(-1)
            T[:, 0, c1, c2] = (w * g0).sum((1, 2, 3))
            T[:, 1, c1, c2] = (w * g * s).sum((1, 2, 3))
            T[:, 2, c1, c2] = (w * g * s * s).sum((1, 2, 3))
    M1 = torch.stack([(w * (y * fr[c]).sum(-1)).sum((1, 2, 3)) for c in range(2)], 1) if weight == "morris" else torch.zeros((B, 2), dtype=F64, device=dev)
    if not fixed:
        return T, M1
    sf = (y * n0[:, None, None, None, :]).sum(-1) + s0[:, None, None, None]                  # s~ = y . n + d (fixed frame)
    Tf = torch.zeros((B, 3, 2, 2), dtype=F64, device=dev)
    Tf[:, 0] = T[:, 0]
    for c1 in range(2):
        for c2 in range(2):
            if weight == "morris":
                g = torch.full_like(L, 1.0 if c1 == c2 else 0.0)
            else:
                g = (fr[c1] * yh).sum(-1) * (yh * fr[c2]).sum(-1)
            Tf[:, 1, c1, c2] = (w * g * sf).sum((1, 2, 3))
            Tf[:, 2, c1, c2] = (w * g * sf * sf).sum((1, 2, 3))
    return T, M1, Tf


class WedgeWallMoments:
    """`wedge_moments` of one corner angle `beta` on a grid (rho / H, phi / phi_max), phi_max = pi - beta / 2 (the fluid edge), built once (chunked), bilinear lookup.  A particle mirrored about the bisector (phi < 0)
    takes the table at |phi| with the n-t couplings and M1_t negated (t = J n flips under the reflection)."""

    def __init__(self, dW, H, device, beta, weight="alpha", nr=33, nphi=33, chunk=8):
        self.H, self.beta, self.nr, self.nphi = float(H), float(beta), nr, nphi
        self.phimax = math.pi - 0.5 * beta
        rg = torch.linspace(0.0, 1.0, nr, dtype=F64, device=device).clamp(min=0.01) * H
        pg = torch.linspace(0.0, 1.0 - 1e-3, nphi, dtype=F64, device=device) * self.phimax          # stops short of the face line (d = 0: the particle frame is undefined there)
        self.pscale = (1.0 - 1e-3) * self.phimax
        R, Pp = torch.meshgrid(rg, pg, indexing="ij")
        R, Pp = R.reshape(-1), Pp.reshape(-1)
        Ts, Ms, Tfs = [], [], []
        for i in range(0, len(R), chunk):
            T, M, Tf = wedge_moments(dW, H, R[i:i + chunk], Pp[i:i + chunk], beta, weight, fixed=True)
            Ts.append(T)
            Ms.append(M)
            Tfs.append(Tf)
        self.T = torch.cat(Ts).reshape(nr, nphi, 3, 2, 2)
        self.M1 = torch.cat(Ms).reshape(nr, nphi, 2)
        self.Tf = torch.cat(Tfs).reshape(nr, nphi, 3, 2, 2)

    def eval(self, rho, phi, fixed=False):
        """(T [N, 3, 2, 2], M1 [N, 2]) at signed angle phi (mirrored below the bisector); with `fixed` also the fixed-frame moments Tf."""
        neg = phi < 0
        fr = (rho / self.H).clamp(0.0, 1.0) * (self.nr - 1)
        fp = (phi.abs() / self.pscale).clamp(0.0, 1.0) * (self.nphi - 1)
        i0 = fr.floor().long().clamp(max=self.nr - 2)
        j0 = fp.floor().long().clamp(max=self.nphi - 2)
        a, b = (fr - i0), (fp - j0)

        def lerp(t):
            sh = (-1,) + (1,) * (t.dim() - 2)
            A, Bb = a.reshape(sh), b.reshape(sh)
            return t[i0, j0] * (1 - A) * (1 - Bb) + t[i0 + 1, j0] * A * (1 - Bb) + t[i0, j0 + 1] * (1 - A) * Bb + t[i0 + 1, j0 + 1] * A * Bb

        T, M = lerp(self.T), lerp(self.M1)
        sgn = torch.where(neg, -1.0, 1.0).to(F64)

        def mirror(t):
            t = t.clone()
            t[:, :, 0, 1] = t[:, :, 0, 1] * sgn[:, None]
            t[:, :, 1, 0] = t[:, :, 1, 0] * sgn[:, None]
            return t

        T = mirror(T)
        M = torch.stack([M[:, 0], M[:, 1] * sgn], 1)
        if fixed:
            return T, M, mirror(lerp(self.Tf))
        return T, M
