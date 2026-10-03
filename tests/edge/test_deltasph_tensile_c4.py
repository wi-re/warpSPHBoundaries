"""Tests for the cfg.tensileExact switch with the Wendland C4 kernel (WORK-004 T4.3).

The wall part of the shifting tensile term  T_i = int_solid W^4 grad_x W dA'  is computed two ways:
  - polar quadrature (the default):  -dphi * sum_samples ins * W(r)^4 W'(r) r dr u_hat     (as in DeltaSPH2D.shift)
  - exact edge reduction (tensile.tensile_vector_scene, family="w4"):  (1/5) c4^5/(pi^4 c45 H^8) grad_x int_solid W^5 dA'

The polar sum is a quadrature of the same integral (grad W = -W'(r) u_hat), so the difference is the quadrature's own
error.  The prefactor wallMass*shiftR/w0^4 is not part of T and is left out of the comparison.  Mirrors
test_deltasph_tensile.py (C2); this is the C4 case (the C4 tank, 304 near particles) and it replaces the deleted
test_tensile_exact_wendland4_raises guard (tensileExact now works for C4, through the Chebyshev-quadrature plan).

tolerances (fixed in the WORK-004 T4.3 spec, stated BEFORE looking at the results):
  * (a)  max|T_quad - T_exact| <= 3e-2 * max|T_exact| (measured 1.749e-2 = the polar quadrature's own error);
         max|T_exact| > 1e7 (measured 7.35e7); sign control max|T_quad + T_exact| > 1.0 * max|T_exact| (measured 2.017).
  * (b)  the effect on shift, 5e-3 <= max|u_q - u_e| / max||u_q|| <= 0.15 (measured 5.91e-2).
  * (d)  (replaces the deleted guard) tensileExact = True on a C2 tank still works (shift runs, u finite), and the C4
         and C2 exact results differ from their polar counterparts (max||u_q - u_e|| > 0 for both).
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
    """(a) polar vs exact on the near particles of the C4 tank (304 near); (b) the effect on shift; (c) five steps with
    tensileExact = True stay finite (x, v, rho)."""
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device,
                              cfg=DeltaSPHConfig(kernel=KernelFunctions.Wendland4))
    x, H = sim.x, sim.H
    st = sim._surface_state(x, sim.rho)
    near, (ins, u, rk, dr, dphi) = st["samples"]

    # (a) the wall tensile integral: polar quadrature vs the exact edge reduction (family="w4")
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

    # (b) the effect on shift: the switch changes the C4 shift by a visible but small amount
    u_q = sim.shift(sim.dt)
    sim.cfg.tensileExact = True
    try:
        u_e = sim.shift(sim.dt)
    finally:
        sim.cfg.tensileExact = False
    rel = float((u_q - u_e).norm(dim=1).max()) / float(u_q.norm(dim=1).max())
    assert 5e-3 <= rel <= 0.15, rel
    print("(b) C4: max|u_q - u_e| / max||u_q|| = %.4e" % rel)

    # (c) five steps with tensileExact = True stay finite (no NaN / inf in x, v, rho)
    sim.cfg.tensileExact = True
    try:
        for _ in range(5):
            sim.step()
    finally:
        sim.cfg.tensileExact = False
    for a in (sim.x, sim.v, sim.rho):
        assert torch.isfinite(a).all()
    print("(c) C4: five steps with tensileExact=True finite")


@pytest.mark.parametrize("device", DEVICES)
def test_tensile_exact_c4_and_c2_both_work(device):
    """(d) (replaces the deleted C4 guard) tensileExact = True on a C2 tank still works (shift runs, u finite), and the C4
    and C2 exact results differ from their polar counterparts (max||u_q - u_e|| > 0 for both)."""
    for kernel in (KernelFunctions.Wendland4, KernelFunctions.Wendland2):
        sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device,
                                  cfg=DeltaSPHConfig(kernel=kernel, tensileExact=True))
        u_e = sim.shift(sim.dt)                       # exact (tensileExact is True in this sim's cfg)
        assert torch.isfinite(u_e).all()
        sim.cfg.tensileExact = False
        u_q = sim.shift(sim.dt)                       # polar
        d = float((u_q - u_e).norm(dim=1).max())
        assert d > 0.0, d
        print("(d) %s: exact shift finite; max||u_q - u_e|| = %.4e (> 0)" % ("C4" if kernel == KernelFunctions.Wendland4 else "C2", d))
