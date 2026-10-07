"""The no-slip wall closure of the viscous term, shared by the schemes (delta+ `DeltaSPH2D`, the omniSPH-style `DFSPH2D`): the wall part of the viscous acceleration of a particle near a wall from the moments of the
pair weight over the solid (curved tables, wedge tables at corners) or, with `complement`, from the complement of the particle's own discrete fluid moments (docs/plan-next-steps.md "Complement moments"); the
signed solid area of a body for the Morris torque correction.  Extracted from `DeltaSPH2D` unchanged (docs/plan-next-steps.md "DFSPH track", D3).

    wc = NoSlipClosure(scene, dW, H, dx, device, morris=True, cal=0.985, wallMass=1.0, complement=True, sums=fn)
    a_wall = wc.term(bi, x, w, n, d, viscf, rho, fac)       # w: velocity relative to the rigid wall motion at the particle, n / d: the wall normal / distance, viscf: the fluid pair viscous acceleration,
                                                            # fac: the alpha-form factor (Morris: nu_used = fac / 8 / cal is the coefficient of the fluid operator)
`sums(x, n, d, rho) -> (S0, S1, S2, SM)`: the scheme's discrete fluid moments of the Morris weight in the frame of each particle's wall (only with `complement`).
"""
import math

import torch

from .wallmoments import CurvedWallMoments, WedgeWallMoments

F64 = torch.float64


class NoSlipClosure:
    def __init__(self, scene, dW, H, dx, device, morris, cal=1.0, wallMass=1.0, complement=False, sums=None, cornerWedgeTables=False, cornerAngleTol=20.0):
        if complement and not morris:
            raise NotImplementedError("complementMoments: Morris viscosity only (the alpha weight needs the lattice-anisotropic full-plane moments)")
        self.scene, self.dW, self.H, self.dx, self.dev = scene, dW, float(H), float(dx), device
        self.morris, self.cal, self.wallMass, self.complement, self.sums = bool(morris), float(cal), float(wallMass), bool(complement), sums
        self.cornerWedgeTables, self.cornerAngleTol = bool(cornerWedgeTables), float(cornerAngleTol)

    @staticmethod
    def solid_area(b):
        """signed area enclosed by the boundary of the body's solid, oriented with the solid on the left (+ for an obstacle, - for a cavity / a domain); cached per body (host float)."""
        from ..scene.implicitBodies import DiskBody
        from ..scene.scene import BoxRep, DiskArrayRep, ImplicitRep, SdfRep, SurfaceRep
        if getattr(b, "_solidAreaCache", None) is not None:                                     # cached on the body itself (an id() key is reused after garbage collection)
            return b._solidAreaCache

        def area(r):
            if isinstance(r, DiskArrayRep):
                return math.pi * float((r.radii.double() ** 2).sum())
            if isinstance(r, SurfaceRep):
                P, E = r.vertices.double().cpu(), r.edges.long().cpu()
                a, c = P[E[:, 0]], P[E[:, 1]]
                return 0.5 * float((a[:, 0] * c[:, 1] - c[:, 0] * a[:, 1]).sum())
            if isinstance(r, BoxRep):
                A = (r.hi_h[0] - r.lo_h[0]) * (r.hi_h[1] - r.lo_h[1])
                return A if r.solid == "inside" else -A
            if isinstance(r, ImplicitRep) and isinstance(r.shape, DiskBody):
                A = math.pi * r.shape.radius ** 2
                return A if r.shape.solid == "inside" else -A
            if isinstance(r, SdfRep) and r.fallback is not None:
                return area(r.fallback)
            raise NotImplementedError("Morris torque correction: no enclosed area for %s" % type(r).__name__)

        b._solidAreaCache = sum(area(r) for r in b.reps)
        return b._solidAreaCache


    def term(self, bi, x, w, n, d, viscf, rho, fac):
        """the wall part of the viscous acceleration of a no-slip wall from the moments of the pair weight over the solid (sim/wallmoments.py): the velocity relative to the wall is a polynomial of the wall distance,
        w(s) = a s + (L / 2) s^2 (a, L per component in the frame (n, t)), with  w(d) = w_i  and the viscous balance of the particle (the fluid pair sum `viscf` plus this wall term = nu_p (lap w [+ 2 grad div w]),
        nu_p = fac / 8; the normal component carries the factor 3 of the pair form, the tangential one the curvature terms of the wall kappa = div n: lap w_t = w_t'' + kappa w_t' - kappa^2 w_t)."""
        morris = self.morris
        cache = self.__dict__.setdefault("_wpmCache", {})
        if morris not in cache:
            cache[morris] = CurvedWallMoments(self.dW, self.H, self.dev, weight="morris" if morris else "alpha")
        t = torch.stack([-n[:, 1], n[:, 0]], 1)
        kap = self.curvature(bi, x, n, t)                                                       # div n at the particle: 1 / (R + d) convex, -1 / (R - d) concave
        Tc = cache[morris].eval(d, kap / (1.0 - kap * d))                                        # [2 (n, t), 4, N]: moments over the actual circular solid of curvature kappa_w = 1 / R (signed)
        N = x.shape[0]
        T = torch.zeros((N, 3, 2, 2), dtype=F64, device=self.dev)                                # full tensors T_k^{c'c} (k, c', c): diagonal from the curved tables, full at corners
        T[:, :, 0, 0] = Tc[0, :3].T
        T[:, :, 1, 1] = Tc[1, :3].T
        M1 = torch.stack([Tc[0, 3], torch.zeros_like(Tc[0, 3])], 1)
        comp = self.complement
        corner = self.corner_moments(bi, x, morris, fixed=comp) if self.cornerWedgeTables else None
        if comp:
            if not morris:
                raise NotImplementedError("complementMoments: Morris viscosity only (the alpha weight needs the lattice-anisotropic full-plane moments)")
            T, M1 = self.complement_moments(x, n, d, rho)
            kap = torch.zeros_like(kap)                                                          # fixed-frame continuation: no curvature terms
            if corner is not None:                                                               # HYBRID at corners: the complement (discrete quadrature) plus the continuum difference of the turning-frame and the fixed-frame
                cm, Tw, _, kw, Tf = corner                                                       # continuation over the wedge solid (k = 1, 2; k = 0 and M1 do not depend on the continuation), the turning-frame curvature
                dT = torch.zeros_like(T)
                dT[:, 1:] = Tw[:, 1:] - Tf[:, 1:]
                T = T + torch.where(cm[:, None, None, None], dT, torch.zeros_like(dT))
                kap = torch.where(cm, kw, kap)
            corner = None
            # the complement is the particle's WHOLE missing region: it belongs to the nearest body only, and only within the support (prototype: several bodies inside one support share one missing region)
            dall = torch.stack([self.scene.signed_distance(x, body=k, supportMax=self.H)[0] for k in range(len(self.scene.bodies))])
            own = (dall.argmin(0) == bi) & (dall[bi] < self.H)
            T = T * own[:, None, None, None].to(F64)
            M1 = M1 * own[:, None].to(F64)
        if corner is not None:                                                                   # particles within H of a polygon corner: the wedge tables, and the curvature of the distance field (1 / rho where the nearest point is the vertex, 0 on faces)
            cm, Tw, Mw, kw = corner
            T = torch.where(cm[:, None, None, None], Tw, T)
            M1 = torch.where(cm[:, None], Mw, M1)
            kap = torch.where(cm, kw, kap)
        if morris:                                                                               # Morris pair weight 2 nu_used V_w K(r) (identity), rho_w = rho_i; its discrete operator realises nu_used * morrisCalibration
            nu_used = fac / 8.0 / self.cal
            beta = 2.0 * nu_used * self.wallMass / rho
            nup = nu_used * self.cal
            cn = 1.0                                                                             # no grad div part: the normal component has no factor 3
        else:
            beta = fac * self.wallMass / rho
            nup = fac / 8.0
            cn = 3.0
        bn = beta / nup                                                                          # beta / nu_p
        wv = torch.stack([(w * n).sum(1), (w * t).sum(1)], 1)                                    # [N, 2] in (n, t)
        fv = torch.stack([(viscf * n).sum(1), (viscf * t).sum(1)], 1)
        rot = None
        if morris:                                                                               # a rigid rotation Omega of the wall: the Morris weight does not annihilate it.  Wall part: -beta Omega J M1 (J M1 = M1_n t - M1_t n);
            Om = beta * self.scene.bodies[bi].angularVelocity                                    # the fluid sum carries the opposite (the full-plane integral vanishes), removed before the balance of w
            rot = torch.stack([-Om * M1[:, 1], Om * M1[:, 0]], 1)
            fv = fv - rot
        # w(s) = a s + L s^2 / 2 per component, w(d) = w_i  ->  a = w / d - L d / 2;  wall term A = -beta (G1 w + Hm L), G1 = T1 / d - T0, Hm = (T2 - d T1) / 2 (2 x 2);  balance f + A = nu_p lap w with
        # lap w = (cn L_n, (1 + kappa d / 2) L_t + kappa w_t / d - kappa^2 w_t):  (D + bn Hm) L = f / nu_p - bn G1 w - r.  Diagonal T (no corner) = the former per-component solve.
        dd = d[:, None, None]
        G1 = T[:, 1] / dd - T[:, 0]
        Hm = 0.5 * (T[:, 2] - dd * T[:, 1])
        Dm = torch.zeros_like(Hm)
        Dm[:, 0, 0] = cn
        Dm[:, 1, 1] = 1.0 + 0.5 * kap * d
        rv = torch.stack([torch.zeros_like(d), kap * wv[:, 1] / d - kap * kap * wv[:, 1]], 1)
        rhs = fv / nup - bn[:, None] * (G1 @ wv[:, :, None])[:, :, 0] - rv
        Mm = Dm + bn[:, None, None] * Hm                                                         # 2 x 2 by Cramer's rule (torch.linalg.solve checks singularity on the host: not capturable)
        det = Mm[:, 0, 0] * Mm[:, 1, 1] - Mm[:, 0, 1] * Mm[:, 1, 0]
        Lv = torch.stack([Mm[:, 1, 1] * rhs[:, 0] - Mm[:, 0, 1] * rhs[:, 1], Mm[:, 0, 0] * rhs[:, 1] - Mm[:, 1, 0] * rhs[:, 0]], 1) / det[:, None]
        Av = -beta[:, None] * ((G1 @ wv[:, :, None])[:, :, 0] + (Hm @ Lv[:, :, None])[:, :, 0])
        if rot is not None:
            Av = Av - rot
        return Av[:, 0:1] * n + Av[:, 1:2] * t


    def complement_moments(self, x, n, d, rho):
        """the Morris wall moments as complements of the particle's own discrete fluid moments, fixed particle frame (y = x_j - x_i, s~ = y . n + d, mu_j = V_j (rho_i + rho_j) / (2 rho_i), K = W' r / (r^2 + eta^2)):
        T_0 = I_0 - sum mu K,  T_1 = d I_0 - sum mu K s~,  T_2 = (I_2 / 2 + d^2 I_0) - sum mu K s~^2,  M1 = - sum mu K y.  I_0 cancels in the closure (G1 = S_0 - S_1 / d, Hm = (I_2 / 2 - S_2 + d S_1) / 2); I_2 = -2 cal makes
        the full-plane operator the calibrated bulk one.  Warp modules on the fluid adjacency (graph-capturable), the torch pair list with fluidWarp = False (oracle)."""
        N = len(x)
        S0, S1, S2, SM = self.sums(x, n, d, rho)                                                 # the scheme's discrete fluid moments (callback)
        I2 = -2.0 * self.cal
        T0 = -S0                                                                                # + I_0 (cancels)
        T1 = -S1                                                                                # + d I_0 (cancels)
        T2 = 0.5 * I2 - S2                                                                      # + d^2 I_0 (cancels)
        T = torch.zeros((N, 3, 2, 2), dtype=F64, device=self.dev)
        for k, Tk in enumerate((T0, T1, T2)):
            T[:, k, 0, 0] = Tk
            T[:, k, 1, 1] = Tk
        t = torch.stack([-n[:, 1], n[:, 0]], 1)
        M1 = torch.stack([-(SM * n).sum(1), -(SM * t).sum(1)], 1)
        return T, M1


    def corners(self, b):
        """the sharp corners of a body (body frame, cached): vertices of its polygon loops (`SurfaceRep`, `BoxRep` via its polygon) whose solid angle beta differs from pi by more than cfg.cornerAngleTol, as
        (V [m, 2] vertex, E [m, 2] unit bisector into the fluid, betas [m] host floats, the same on the device); None without corners.  The loops have the solid on the left (counter-clockwise around an obstacle, clockwise around a cavity)."""
        from ..scene.scene import BoxRep, SurfaceRep
        key = (float(self.cornerAngleTol), str(self.dev))
        if getattr(b, "_cornerCache", None) is not None and b._cornerCache[0] == key:            # cached on the body itself (an id() key is reused after garbage collection)
            return b._cornerCache[1]
        tol = math.radians(self.cornerAngleTol)
        V, E, betas = [], [], []
        for r in b.reps:
            if isinstance(r, BoxRep):
                r = r.surface()
            if not isinstance(r, SurfaceRep):
                continue
            P, ed = r.vertices.double().cpu(), r.edges.long().cpu()
            nxt = {int(e[0]): k for k, e in enumerate(ed)}
            for k, e in enumerate(ed):
                k2 = nxt.get(int(e[1]))
                if k2 is None:
                    continue
                v = P[e[1]]
                tin = P[e[1]] - P[e[0]]
                tout = P[ed[k2][1]] - P[ed[k2][0]]
                tin, tout = tin / tin.norm(), tout / tout.norm()
                turn = math.atan2(float(tin[0] * tout[1] - tin[1] * tout[0]), float((tin * tout).sum()))      # left turn > 0: a convex corner of the solid
                beta = math.pi - turn
                if abs(beta - math.pi) <= tol:
                    continue
                nin = torch.stack([tin[1], -tin[0]])                                                      # outward (into the fluid): right-hand normal, the solid is on the left
                nout = torch.stack([tout[1], -tout[0]])
                bis = nin + nout
                if float(bis.norm()) < 1e-9:
                    continue
                V.append(v)
                E.append(bis / bis.norm())
                betas.append(beta)
        b._cornerCache = (key, None if not V else (torch.stack(V).to(self.dev), torch.stack(E).to(self.dev), betas, torch.tensor(betas, dtype=F64, device=self.dev)))   # device copy made once (graph capture)
        return b._cornerCache[1]


    def corner_moments(self, bi, x, morris, fixed=False):
        """for the particles within H of a corner of body `bi` (nearest corner): (mask [N], T [N, 3, 2, 2], M1 [N, 2], kappa [N]) from the wedge tables of that corner's angle, with `fixed` also the fixed-frame
        moments Tf [N, 3, 2, 2] (for the hybrid with the complement closure); None without corners."""
        b = self.scene.bodies[bi]
        cs = self.corners(b)
        if cs is None:
            return None
        Vc, Ec, betas, betaDev = cs
        lx = b.toLocal(x)                                                                        # body frame, nearest periodic image
        rel = lx[:, None, :] - Vc[None]                                                          # [N, m, 2]
        dist = rel.norm(dim=2)
        k = dist.argmin(1)
        rho = dist.gather(1, k[:, None])[:, 0]
        mask = rho < self.H
        e = Ec[k]
        r = rel[torch.arange(len(x), device=self.dev), k]
        phi = torch.atan2(e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0], (e * r).sum(1))                 # signed angle from the fluid bisector
        tabs = self.__dict__.setdefault("_wedgeCache", {})
        T = torch.zeros((len(x), 3, 2, 2), dtype=F64, device=self.dev)
        M1 = torch.zeros((len(x), 2), dtype=F64, device=self.dev)
        kap = torch.zeros(len(x), dtype=F64, device=self.dev)
        Tf = torch.zeros((len(x), 3, 2, 2), dtype=F64, device=self.dev)
        bk = betaDev[k]
        for beta in sorted(set(round(v, 6) for v in betas)):
            key = (beta, morris)
            if key not in tabs:
                tabs[key] = WedgeWallMoments(self.dW, self.H, self.dev, beta, weight="morris" if morris else "alpha")
            sel = (bk - beta).abs() < 1e-6
            if fixed:
                Tg, Mg, Tfg = tabs[key].eval(rho, phi, fixed=True)
                Tf = torch.where(sel[:, None, None, None], Tfg, Tf)
            else:
                Tg, Mg = tabs[key].eval(rho, phi)
            T = torch.where(sel[:, None, None, None], Tg, T)
            M1 = torch.where(sel[:, None], Mg, M1)
            a = math.pi - 0.5 * beta                                                             # the vertex region of the fluid: both projections on the face rays negative (convex corners only)
            pa = rho * torch.cos(phi - a)
            pb = rho * torch.cos(phi + a)
            vq = (pa < 0) & (pb < 0)
            kap = torch.where(sel & vq, 1.0 / rho.clamp(min=0.25 * self.dx), kap)
        if fixed:
            return mask, T, M1, kap, Tf
        return mask, T, M1, kap


    def curvature(self, bi, x, n, t):
        """kappa = div n of the signed distance at the particles (tangential derivative of the wall normal, central difference over half a spacing): 1 / r for a convex circle, -1 / r concave, 0 on a plane."""
        eps = 0.5 * self.dx
        npl = self.scene.signed_distance(x + eps * t, body=bi, supportMax=self.H)[1]
        nmi = self.scene.signed_distance(x - eps * t, body=bi, supportMax=self.H)[1]
        return ((npl - nmi) * t).sum(1) / (2.0 * eps)

