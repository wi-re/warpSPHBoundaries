"""Tests for the delta+ shifting tensile control wall part with the Wendland C4 kernel (WORK-004 T4.3): the exact edge
reduction is the only path.

The wall part of the shifting tensile term  T_i = int_solid W^4 grad_x W dA'  is evaluated by the solver through the exact
edge reduction (tensile.tensile_vector_scene, family="w4"); this test compares it against the polar quadrature written out
in `_polar_T` (`sim._solid_samples` still exists, used only by the `wallViscosityForm = "pairwise"` wall term), so the
difference is the polar quadrature's own error.  The prefactor wallMass*shiftR/w0^4 is not part of T and is left out of the
comparison.  Mirrors test_deltasph_tensile.py (C2); this is the C4 case (the C4 tank, 304 near particles) and it replaces the
deleted test_tensile_exact_wendland4_raises guard (the exact tensile works for C4, through the Chebyshev-quadrature plan).

tolerance (fixed in the WORK-004 T4.3 spec, stated BEFORE looking at the results):
  * (a)  max|T_quad - T_exact| <= 3e-2 * max|T_exact| (measured 1.749e-2 = the polar quadrature's own error);
         max|T_exact| > 1e7 (measured 7.35e7); sign control max|T_quad + T_exact| > 1.0 * max|T_exact| (measured 2.017).
"""
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions

from edgebound.deltasph2d import DeltaSPHConfig, hydrostatic_tank
from edgebound.dfsph2d import F64
from edgebound.tensile import tensile_vector_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


def _polar_T(sim, near):
    """the wall tensile integral T = int_solid W^4 grad W dA' by the polar quadrature exactly as DeltaSPH2D.shift
    computes it (without the wallMass*shiftR/w0^4 prefactor), for the solver's W/dW: [Q,2]."""
    H = sim.H
    ins, u, rk, dr, dphi = sim._solid_samples(sim.x, near)
    Fr = sim.W(rk, H) ** 4 * sim.dW(rk, H) * rk * dr
    return -torch.einsum("bqrp,r,pa->qa", ins.to(F64), Fr, u) * dphi


@pytest.mark.parametrize("device", DEVICES)
def test_tensile_exact_c4(device):
    """(a) polar vs exact on the near particles of the C4 tank (304 near); (b) five steps stay finite (x, v, rho)."""
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device,
                              cfg=DeltaSPHConfig(kernel=KernelFunctions.Wendland4))
    x, H = sim.x, sim.H
    st = sim._surface_state(x, sim.rho)
    near = st["near"]

    # (a) the wall tensile integral: polar quadrature (own `_polar_T`) vs the exact edge reduction (family="w4")
    T_quad = _polar_T(sim, near)
    T_exact = tensile_vector_scene(sim.scene, x[near], H, "w4")
    scale = float(T_exact.abs().max())
    assert scale > 1e7, scale
    assert float((T_quad - T_exact).abs().max()) <= 3e-2 * scale, (float((T_quad - T_exact).abs().max()), scale)
    assert float((T_quad + T_exact).abs().max()) > 1.0 * scale
    print("(a) C4: near=%d  max|T_exact|=%.3e  |T_quad - T_exact|/scale=%.3e  |T_quad + T_exact|/scale=%.3e"
          % (len(near), scale,
             float((T_quad - T_exact).abs().max()) / scale,
             float((T_quad + T_exact).abs().max()) / scale))

    # (b) five steps stay finite (no NaN / inf in x, v, rho)
    for _ in range(5):
        sim.step()
    for a in (sim.x, sim.v, sim.rho):
        assert torch.isfinite(a).all()
    print("(b) C4: five steps finite")


@pytest.mark.parametrize("device", DEVICES)
def test_tensile_exact_c4_and_c2_both_work(device):
    """(c) (replaces the deleted C4 guard) the exact shift runs and is finite on the C4 and C2 tanks (both kernels work)."""
    for kernel in (KernelFunctions.Wendland4, KernelFunctions.Wendland2):
        sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device,
                                  cfg=DeltaSPHConfig(kernel=kernel))
        u = sim.shift(sim.dt)                         # the exact edge reduction (the only path)
        assert torch.isfinite(u).all()
        print("(c) %s: exact shift finite" % ("C4" if kernel == KernelFunctions.Wendland4 else "C2"))
