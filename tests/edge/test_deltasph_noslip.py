"""Tests for the wallViscosityForm = "noslip" (WORK-007): the no-slip wall term is the Chiron-style flux term
    acc_wall = -2 nu_eff (v_i - v_w) |G_b| / (rho_i d_b)      per body b,   nu_eff = alpha c0 H/(8 xi) = fac/8,   d_b = max(d_signed, 0.25 dx),
from a one-sided finite difference of the normal derivative (Chiron et al. 2019 Eq. 91-92) with v_w = b.velocityAt(x_i) (the body velocity, 0 for a resting wall).
It opposes the ALL-components relative velocity (no-slip), unlike the free-slip forms which damp only the normal component.

(a) flat wall, absolute: a particle dp/2 above a flat wall, v = (1,0) / (0,-1) / (0.6,-0.8): the term equals
    -2 (fac/8) wallMass v F / (rho z)  with z = dp/2 and F = int_{-L}^{L} W(sqrt(s^2+z^2)) ds (OWN scipy quad, epsabs = epsrel = 1e-13,
    W = 7/(pi H^2) (1-q)^4 (1+4q), L = sqrt(H^2 - z^2)); the solver's |G_b| = wallMass F (the 8.6e-16 identity).
(b) distance floor + Galilean: a particle at z = 0.1 dx (inside the d_min) uses d_b = 0.25 dx in the denominator but the TRUE F(z);
    with the wall moving at v0 and the particle at v0 the relative velocity is 0 (no term), at rest it is non-zero.
(c) Couette and Poiseuille through the solver (the probe configuration): a_bulk = rhs(wallViscosity=False) - rhs(viscosity=False),
    a_flux = rhs(form="noslip") - rhs(viscosity=False) (first component, mean over the central columns |x| < 0.3 in the rows
    s = (k+0.5) dp, k = 0..3); exact targets from the continuum: Couette 0, Poiseuille nu_eff d^2_y u = -2 nu_eff.  The free-slip
    forms (laplacian, pairwise) are the negative controls: on a tangential field their wall term is exactly 0, so they give |a| = |a_bulk|.
(d) wallViscosity = False gives no wall term for "noslip" (the form is irrelevant); five sim.step() stay finite on the C2 and the C4 tank
    (the form is kernel-agnostic: |G_b| comes from the scene).

tolerances (stated with their reason, BEFORE looking at the results):
  * (a)  max|got - pred| <= 1e-9 * |pred| (float64 round-off; the reviewer measured |G| vs wallMass F at 8.6e-16; a wrong factor or sign is >= 1e-2).
         smoke (rtol 1e-4, at z/H = 0.125): nu_eff = 2.432164e-3, F = 8.29977090 / wallMass, pred_x = -2.018640 (v = (1,0)), pred_y = +2.018640 (v = (0,-1)).
  * (b)  floor: max|got - pred| <= 1e-9 * |pred|; Galilean: co-moving wall term < 1e-10, resting wall term > 0.1.
  * (c)  Couette: |a_flux(row k)| <= 0.2 |a_bulk(row 0)| in every row (measured 0.00815 / -0.00855 / -0.00280 / 0.00001 vs a_bulk(row 0) = 0.06014: <= 0.142);
         Poiseuille: rows 2,3 |a_flux - target| <= 0.15 |target| (measured 7 % / 4 %); rows 0,1 |a_flux - target| <= 0.6 |a_bulk - target| (measured 0.22 / 0.42).
         The first-order flux term is not exact in the first two rows (the point of the controls) and the lattice ghost truth is only ~4 % accurate on row 3.
         negative controls: the laplacian and pairwise forms give |a(row 0)| > 0.5 |a_bulk(row 0)| (they fail the Couette tolerance).
  * (d)  wallViscosity = False: |a(noslip, off) - a(laplacian, off)| < 1e-12 (the form is irrelevant with no wall term); five steps finite (x, v, rho).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions

from edgebound.sim import deltasph2d as D
from edgebound.sim.deltasph2d import DeltaSPHConfig, hydrostatic_tank

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64


def small_tank(device, dp=0.04, L=0.8, Ht=0.6, Hw=0.3, domain="surface", **cfgkw):
    sim, info = D.hydrostatic_tank(dp=dp, L=L, Htank=Ht, Hwater=Hw, domain=domain, device=device)
    sim.cfg.__dict__.update(cfgkw)
    return sim, info


@pytest.mark.parametrize("device", DEVICES)
def test_noslip_wall_term_is_the_chiron_flux_on_a_flat_wall(device):
    """(a) the no-slip term on a flat wall equals -2 nu_eff wallMass v F/(rho z) for v = (1,0), (0,-1), (0.6,-0.8); the OWN prediction pins itself."""
    from scipy import integrate
    sim, info = small_tank(device)
    sim.cfg.wallViscosityForm = "noslip"
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    dp, bed = sim.dx, info["bed"]
    x = torch.tensor([[0.0, bed + 0.5 * dp]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    H = sim.H
    z = 0.5 * dp
    L = math.sqrt(max(H * H - z * z, 0.0))
    def W(rr):
        q = rr / H
        return 7.0 / (math.pi * H * H) * (1.0 - q) ** 4 * (1.0 + 4.0 * q)          # the normalised Wendland C2
    F = integrate.quad(lambda s: W(math.sqrt(s * s + z * z)), -L, L, epsabs=1e-13, epsrel=1e-13)[0]
    fac = sim.cfg.alpha * sim.cfg.c0 * H / sim.xi
    nu = fac / 8.0
    got = {}
    for name, v in (("x", (1.0, 0.0)), ("y", (0.0, -1.0)), ("diag", (0.6, -0.8))):
        vel = torch.tensor([v], dtype=F64, device=device)
        sim.cfg.wallViscosity = True
        a_on = sim.rhs(x, vel, rho)[0]
        sim.cfg.wallViscosity = False
        a_off = sim.rhs(x, vel, rho)[0]
        got[name] = (a_on - a_off)[0].cpu().numpy()
    for name, v in (("x", (1.0, 0.0)), ("y", (0.0, -1.0)), ("diag", (0.6, -0.8))):
        pred = -2.0 * nu * sim.cfg.wallMass * F / z * np.array(v)                 # -2 nu_eff wallMass v F/(rho z), rho = 1
        diff = float(np.abs(got[name] - pred).max())
        assert diff <= 1e-9 * float(np.abs(pred).max()), (name, got[name], pred)
        print("(a) v = %s: got = (%.6f, %.6f)  pred = (%.6f, %.6f)  max|diff|/max|pred| = %.2e (tol 1e-9)"
              % (v, *got[name], *pred, diff / float(np.abs(pred).max())))
    # (a-smoke) the OWN prediction itself (nu_eff, F, pred for v = (1,0) and v = (0,-1) at z/H = 0.125)
    assert abs(nu - 2.432164e-3) <= 1e-4 * 2.432164e-3, nu
    assert abs(F - 8.29977090) <= 1e-4 * 8.29977090, F                            # wallMass = 1, so wallMass * F = F
    pred_x = -2.0 * nu * sim.cfg.wallMass * F / z * 1.0
    pred_y = -2.0 * nu * sim.cfg.wallMass * F / z * (-1.0)
    assert abs(pred_x - (-2.018640)) <= 1e-4 * 2.018640, pred_x
    assert abs(pred_y - 2.018640) <= 1e-4 * 2.018640, pred_y
    print("(a-smoke) nu_eff = %.6e  F = %.6f  pred_x = %.6f  pred_y = %.6f" % (nu, F, pred_x, pred_y))


@pytest.mark.parametrize("device", DEVICES)
def test_noslip_distance_floor_and_galilean(device):
    """(b) a particle at z = 0.1 dx (inside the d_min) uses d_b = 0.25 dx in the denominator with the TRUE F(z); the co-moving wall (v = v_w) gives no term, the resting wall does."""
    from scipy import integrate
    sim, info = small_tank(device)
    sim.cfg.wallViscosityForm = "noslip"
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    dp, bed = sim.dx, info["bed"]
    H = sim.H
    z = 0.1 * dp
    L = math.sqrt(max(H * H - z * z, 0.0))
    def W(rr):
        q = rr / H
        return 7.0 / (math.pi * H * H) * (1.0 - q) ** 4 * (1.0 + 4.0 * q)
    F = integrate.quad(lambda s: W(math.sqrt(s * s + z * z)), -L, L, epsabs=1e-13, epsrel=1e-13)[0]
    nu = sim.cfg.alpha * sim.cfg.c0 * H / sim.xi / 8.0
    x = torch.tensor([[0.0, bed + z]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    v = (1.0, 0.0)
    vel = torch.tensor([v], dtype=F64, device=device)
    sim.cfg.wallViscosity = True
    a_on = sim.rhs(x, vel, rho)[0]
    sim.cfg.wallViscosity = False
    a_off = sim.rhs(x, vel, rho)[0]
    got = (a_on - a_off)[0].cpu().numpy()
    dd = 0.25 * dp                                                                 # z = 0.1 dp < 0.25 dp: the floor engages
    pred = -2.0 * nu * sim.cfg.wallMass * F / dd * np.array(v)                     # |G| = wallMass F(z_true), d_b = dd
    diff = float(np.abs(got - pred).max())
    assert diff <= 1e-9 * float(np.abs(pred).max()), (got, pred)
    print("(b-floor) z = %.1f dx: got = (%.6f, %.6f)  pred = (%.6f, %.6f)  max|diff|/max|pred| = %.2e (tol 1e-9)"
          % (z / dp, *got, *pred, diff / float(np.abs(pred).max())))
    # (b-Galilean) co-moving wall: v = v_w -> no relative velocity -> no term
    v0 = (0.7, -0.3)
    vel0 = torch.tensor([v0], dtype=F64, device=device)
    sim.scene.bodies[0].linearVelocity = torch.tensor(list(v0), dtype=F64, device=device)
    try:
        sim.cfg.wallViscosity = True
        a_on = sim.rhs(x, vel0, rho)[0]
        sim.cfg.wallViscosity = False
        a_off = sim.rhs(x, vel0, rho)[0]
        gal = float((a_on - a_off)[0].norm())
        assert gal < 1e-10, gal
        # resting wall, v = v0: non-zero
        sim.scene.bodies[0].linearVelocity = torch.zeros(2, dtype=F64, device=device)
        sim.cfg.wallViscosity = True
        a_on2 = sim.rhs(x, vel0, rho)[0]
        sim.cfg.wallViscosity = False
        a_off2 = sim.rhs(x, vel0, rho)[0]
        nz = float((a_on2 - a_off2)[0].norm())
        assert nz > 0.1, nz
    finally:
        sim.scene.bodies[0].linearVelocity = torch.zeros(2, dtype=F64, device=device)
    print("(b-Galilean) co-moving |term| = %.2e (< 1e-10)  resting |term| = %.6f (> 0.1)" % (gal, nz))


@pytest.mark.parametrize("device", DEVICES)
def test_noslip_couette_and_poiseuille_through_the_solver(device):
    """(c) the probe configuration: a_bulk = rhs(wallViscosity=False) - rhs(viscosity=False), a_flux = rhs(form='noslip') - rhs(viscosity=False)
    (first component, mean over the central columns |x| < 0.3 in the rows s = (k+0.5) dp, k = 0..3); the free-slip forms are the negative controls."""
    sim, info = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=DeltaSPHConfig())
    x = sim.x
    dp, H = sim.dx, sim.H
    nu = sim.cfg.alpha * sim.cfg.c0 * H / sim.xi / 8.0
    yf = float(x[:, 1].min()) - 0.5 * dp
    s = x[:, 1] - yf
    cols = (x[:, 0].abs() < 0.3)
    ncols = int(cols.sum())                                                          # central-column particles over all rows (the mask, not a pin)
    assert ncols > 0
    def rowmask(k):
        return cols & ((s - (k + 0.5) * dp).abs() < 1e-9)
    def mean0(d, k):
        return float(d[rowmask(k), 0].mean())
    perrow = int(rowmask(0).sum())                                                  # per-row count (15)
    def all_forms(v):
        """the viscous part of rhs (bulk + wall, by form) minus the all-off rhs, for each form ('bulk' = wallViscosity off)."""
        try:
            sim.cfg.viscosity = False
            a0, _, _ = sim.rhs(x, v, sim.rho)
        finally:
            sim.cfg.viscosity = True
        out = {}
        for form in ("bulk", "noslip", "laplacian", "pairwise"):
            try:
                sim.cfg.viscosity = True
                sim.cfg.wallViscosity = form != "bulk"
                if form != "bulk":
                    sim.cfg.wallViscosityForm = form
                a, _, _ = sim.rhs(x, v, sim.rho)
                out[form] = a - a0
            finally:
                sim.cfg.viscosity = True
                sim.cfg.wallViscosity = True
                sim.cfg.wallViscosityForm = "laplacian"
        return out
    # Couette u = s (target 0): |a_flux| small in every row; the free-slip forms give |a| = |a_bulk| (no tangential damping)
    vC = torch.stack([s, torch.zeros_like(s)], 1)
    rC = all_forms(vC)
    ab0 = abs(mean0(rC["bulk"], 0))
    assert ab0 > 0.0
    print("(c) nu_eff = %.5f  central columns |x|<0.3: %d/row (%d total)  a_bulk(row 0) = %.5f" % (nu, perrow, ncols, mean0(rC["bulk"], 0)))
    for k in range(4):
        an = abs(mean0(rC["noslip"], k))
        assert an <= 0.2 * ab0, (k, an, ab0)
        print("(c) Couette row %.1f: a_bulk = %+.5f  a_flux = %+.5f  (|a_flux|/|a_bulk(row 0)| = %.3f <= 0.2)"
              % (k + 0.5, mean0(rC["bulk"], k), mean0(rC["noslip"], k), an / ab0))
    for form in ("laplacian", "pairwise"):
        a0f = abs(mean0(rC[form], 0))
        assert a0f > 0.5 * ab0, (form, a0f, ab0)
        print("(c) negative control %s: |a(row 0)| = %.5f > 0.5 |a_bulk(row 0)| = %.5f (fails the Couette tolerance)" % (form, a0f, 0.5 * ab0))
    # Poiseuille u = s (0.36 - s) (target -2 nu): rows 2,3 within 15 % of the target; rows 0,1 much closer than the bulk
    vP = torch.stack([s * (0.36 - s), torch.zeros_like(s)], 1)
    rP = all_forms(vP)
    target = -2.0 * nu
    print("(c) Poiseuille target = -2 nu = %.5f" % target)
    for k in (2, 3):
        a = mean0(rP["noslip"], k)
        assert abs(a - target) <= 0.15 * abs(target), (k, a, target)
        print("(c) Poiseuille row %.1f: a_bulk = %+.5f  a_flux = %+.5f  target = %+.5f  (|a_flux - target|/|target| = %.3f <= 0.15)"
              % (k + 0.5, mean0(rP["bulk"], k), a, target, abs(a - target) / abs(target)))
    for k in (0, 1):
        a = mean0(rP["noslip"], k)
        ab = mean0(rP["bulk"], k)
        assert abs(a - target) <= 0.6 * abs(ab - target), (k, a, ab, target)
        print("(c) Poiseuille row %.1f: a_bulk = %+.5f  a_flux = %+.5f  target = %+.5f  (|a_flux - target|/|a_bulk - target| = %.3f <= 0.6)"
              % (k + 0.5, ab, a, target, abs(a - target) / abs(ab - target)))


@pytest.mark.parametrize("device", DEVICES)
def test_noslip_no_wall_term_when_off_and_five_steps_finite(device):
    """(d) wallViscosity = False gives no wall term for 'noslip' (the form is irrelevant with no wall term); five sim.step() stay finite on the C2 and the C4 tank."""
    sim, info = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=DeltaSPHConfig())
    x = sim.x
    s = x[:, 1] - (float(x[:, 1].min()) - 0.5 * sim.dx)
    v = torch.stack([s, torch.zeros_like(s)], 1)
    try:
        sim.cfg.viscosity = True
        sim.cfg.wallViscosity = False
        sim.cfg.wallViscosityForm = "noslip"
        a_ns_off, _, _ = sim.rhs(x, v, sim.rho)
        sim.cfg.wallViscosityForm = "laplacian"
        a_lap_off, _, _ = sim.rhs(x, v, sim.rho)
    finally:
        sim.cfg.viscosity = True
        sim.cfg.wallViscosity = True
        sim.cfg.wallViscosityForm = "laplacian"
    d = float((a_ns_off - a_lap_off).norm())
    assert d < 1e-12, d
    print("(d) wallViscosity=False: |a(noslip) - a(laplacian)| = %.2e (< 1e-12, the form is irrelevant)" % d)
    for kern in (KernelFunctions.Wendland2, KernelFunctions.Wendland4):
        sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=DeltaSPHConfig(kernel=kern))
        sim.cfg.wallViscosityForm = "noslip"
        for _ in range(5):
            sim.step()
        for a in (sim.x, sim.v, sim.rho):
            assert bool(torch.isfinite(a).all()), kern
    print("(d) five sim.step() finite on the C2 and the C4 tank")
