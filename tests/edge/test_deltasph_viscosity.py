"""Tests for the wallViscosityForm = "laplacian" default (WORK-005 T5.2 / WORK-006): the wall part of the artificial
viscosity from the exact wall Laplacian (viscosity.lap_lambda_scene, free-slip mirror, nu_eff = alpha c0 H/(8 xi)); the
pairwise polar-quadrature term is the "pairwise" form.

d_ex = rhs(wallViscosityForm="laplacian") - rhs(wallViscosity=False) must equal the exact wall-Laplacian term
    -2 nu_eff wallMass u_n / rho  Delta-lambda n
with n, u_n from the solver's own wall gradient G (not under test) and Delta-lambda from the OWN polar midpoint brute force
(600 x 1200, solid = the exterior of the tank box).  The pairwise term d_pair is a DIFFERENT operator near the wall (the
context finding, not an error), so d_ex and d_pair must differ strongly.

tolerances (fixed in the WORK-005 T5.2 spec, stated BEFORE looking at the results):
  * (b)  max|d_ex - pred| <= 5e-4 * max|pred| (measured 1.0e-4 / 1.3e-4 = the brute grid's own error, as in T5.1 (b));
         max|pred| > 0.1 (measured 0.289 / 0.293); >= 500 particles with d_ex != 0 (measured 692 / 693).
  * (c)  max|d_ex - d_pair| > 0.5 * max|d_pair| (measured 0.899 / 0.886); the sign-flipped prediction differs from d_ex by
         > 1.0 * max|pred|; the prediction with nu_eff replaced by fac/12 differs by > 0.2 * max|pred| (arithmetic: 1/3 of the
         term; this pins the /8).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions

from edgebound.sim.deltasph2d import DeltaSPHConfig, hydrostatic_tank

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


def lapw(fam, q, H):
    """the Laplacian of the normalised Wendland kernel  W = c/(pi H^2) s(r/H)  (c = 7 / 9) at q = r/H, written out as in
    test_viscosity_scene.py (product-rule derivatives, verified there against 5-point finite differences)."""
    c = 7.0 if fam == "w2" else 9.0
    if fam == "w2":
        sp = lambda q: -4 * (1 - q) ** 3 * (1 + 4 * q) + 4 * (1 - q) ** 4
        spp = lambda q: 12 * (1 - q) ** 2 * (1 + 4 * q) - 32 * (1 - q) ** 3
    else:
        def sp(q):
            f = 35.0 / 3.0 * q * q + 6 * q + 1
            return -6 * (1 - q) ** 5 * f + (1 - q) ** 6 * (70.0 / 3.0 * q + 6)
        def spp(q):
            f = 35.0 / 3.0 * q * q + 6 * q + 1
            return 30 * (1 - q) ** 4 * f - 12 * (1 - q) ** 5 * (70.0 / 3.0 * q + 6) + (1 - q) ** 6 * 70.0 / 3.0
    return c / (math.pi * H ** 4) * (spp(q) + sp(q) / q)


def dl_brute(fam, p, H, L=2.4, Ht=1.2, nr=600, nt=1200):
    """the OWN polar midpoint brute force of  Delta-lambda = int_solid lap W dA'  for each particle of p [N,2]: a 600 x 1200
    grid around each particle, solid = the EXTERIOR of the tank box |x| <= L/2, |y| <= Ht/2 (no edge formulas, no winding numbers)."""
    r = (np.arange(nr) + 0.5) / nr * H
    th = (np.arange(nt) + 0.5) / nt * 2.0 * math.pi
    R, T = np.meshgrid(r, th, indexing="ij")
    lap = lapw(fam, R / H, H)
    out = np.zeros(len(p))
    for i, (px, py) in enumerate(p):
        X = px + R * np.cos(T)
        Y = py + R * np.sin(T)
        solid = (np.abs(X) > L / 2) | (np.abs(Y) > Ht / 2)
        out[i] = np.sum(lap * solid * R) * (H / nr) * (2.0 * math.pi / nt)
    return out


def test_viscosity_exact_default_off():
    """(a) the wall-viscosity form defaults to "laplacian" (the exact wall Laplacian)."""
    assert DeltaSPHConfig().wallViscosityForm == "laplacian"


@pytest.mark.parametrize("device", DEVICES)
def test_viscosity_exact_matches_wall_laplacian(device):
    """(b) d_ex = rhs(wallViscosityForm="laplacian") - rhs(wallViscosity=False) equals -2 nu_eff wallMass u_n/rho Delta-lambda_brute n (the own
    600 x 1200 brute force over the tank exterior), max|diff| <= 5e-4 max|pred|, max|pred| > 0.1, >= 500 non-zero particles;
    (c) negative controls (the pairwise operator differs > 0.5 max; sign flip > 1.0 max; nu_eff -> fac/12 differs > 0.2 max);
    (d) five sim.step() stay finite (x, v, rho), both tanks."""
    for kern, fam in ((KernelFunctions.Wendland2, "w2"), (KernelFunctions.Wendland4, "w4")):
        sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device,
                                  cfg=DeltaSPHConfig(kernel=kern))
        x = sim.x
        v = torch.stack([0.3 * torch.sin(2 * x[:, 0]) + 0.1, -0.2 * torch.cos(3 * x[:, 1]) + 0.05], 1)
        sim.v = v
        a_ex, _, _ = sim.rhs(sim.x, v, sim.rho)                                # default form: laplacian
        sim.cfg.wallViscosity = False
        try:
            a_no, _, _ = sim.rhs(sim.x, v, sim.rho)
            sim.cfg.wallViscosity = True
            sim.cfg.wallViscosityForm = "pairwise"
            a_pair, _, _ = sim.rhs(sim.x, v, sim.rho)
        finally:
            sim.cfg.wallViscosity = True
            sim.cfg.wallViscosityForm = "laplacian"
        d_ex = (a_ex - a_no).cpu().numpy()
        d_pair = (a_pair - a_no).cpu().numpy()
        near = np.nonzero(np.abs(d_ex).max(1) > 0)[0]
        assert len(near) >= 500, (fam, len(near))
        # the independent prediction: n, u_n from the solver's own wall gradient (not under test); Delta-lambda the own brute force
        st = sim._surface_state(sim.x, sim.rho)
        G = st["G"]
        n_ = (G[0] / G[0].norm(dim=1).clamp(min=1e-300)[:, None]).cpu().numpy()
        un = (v.cpu().numpy() * n_).sum(1)
        nu = sim.cfg.alpha * sim.cfg.c0 * sim.H / sim.xi / 8
        dlb = dl_brute(fam, x.cpu().numpy()[near], sim.H)
        pred = (-2 * nu * sim.cfg.wallMass * un[near] / sim.rho.cpu().numpy()[near] * dlb)[:, None] * n_[near]
        scale = float(np.abs(pred).max())
        assert scale > 0.1, (fam, scale)
        err = float(np.abs(d_ex[near] - pred).max())
        assert err <= 5e-4 * scale, (fam, err, scale)
        print("(b) %s: near=%d  max|pred| = %.4f  max|d_ex - pred|/max|pred| = %.3e (tol 5e-4)" % (fam, len(near), scale, err / scale))
        # (c) negative controls
        d_op = float(np.abs(d_ex - d_pair).max()) / float(np.abs(d_pair).max())
        assert d_op > 0.5, (fam, d_op)
        d_sign = float(np.abs(-pred - d_ex[near]).max()) / scale
        assert d_sign > 1.0, (fam, d_sign)
        pred12 = pred * (8.0 / 12.0)                                   # nu_eff -> fac/12: the term times (fac/12)/(fac/8) = 2/3
        d_nu = float(np.abs(pred12 - d_ex[near]).max()) / scale
        assert d_nu > 0.2, (fam, d_nu)
        print("(c) %s: max|d_ex - d_pair|/max|d_pair| = %.3f (> 0.5)   sign flip %.3f (> 1.0)   nu_eff -> fac/12: %.3f (> 0.2)" % (fam, d_op, d_sign, d_nu))
        # (d) five steps stay finite (no NaN / inf in x, v, rho)
        for _ in range(5):
            sim.step()
        for a in (sim.x, sim.v, sim.rho):
            assert torch.isfinite(a).all()
        print("(d) %s: five steps finite" % fam)
