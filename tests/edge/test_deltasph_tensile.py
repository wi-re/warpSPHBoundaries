"""Tests for the cfg.tensileExact switch (Q2 in the delta+ shifting tensile control, Wendland C2).

The wall part of the shifting tensile term (Sun 2017 Eq. 7) is the integral

    T_i = int_solid W^4 grad_x W dA'

over the solid within the support of particle i.  It is computed two ways:
  - polar quadrature (the default):  -dphi * sum_samples ins * W(r)^4 W'(r) r dr u_hat     (as in DeltaSPH2D.shift)
  - exact edge reduction (tensile.tensile_vector_scene):  (1/5) c2^5/(pi^4 c25 H^8) grad_x int_solid W^5 dA'

The polar sum is a quadrature of the same integral (grad W = -W'(r) u_hat), so the difference is the quadrature's
own error.  The prefactor wallMass*shiftR/w0^4 is not part of T and is left out of the comparison.

Tolerances (stated with their reason):
  * (b)  max |T_quad - T_exact| <= 3e-2 * max|T_exact| (the reviewer measured 1.389e-2 = the polar quadrature's own
    error, so a tighter bound would be false); max|T_exact| > 1e6 (not two zeros; measured 2.5e7); the sign control
    max|T_quad + T_exact| > 1.0 * max|T_exact| (measured 2.01): a correct, definite, matching sign.
  * (c)  the effect on shift, 5e-3 <= max|u_q - u_e| / max||u_q|| <= 0.15 (measured 3.59e-2): the lower bound proves the
    switch is not a no-op, the upper one that nothing else moved.
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
    computes it (without the wallMass*shiftR/w0^4 prefactor): [Q,2]."""
    H = sim.H
    ins, u, rk, dr, dphi = sim._solid_samples(sim.x, near)
    Fr = sim.W(rk, H) ** 4 * sim.dW(rk, H) * rk * dr
    return -torch.einsum("bqrp,r,pa->qa", ins.to(F64), Fr, u) * dphi


@pytest.mark.parametrize("device", DEVICES)
def test_tensile_exact_switch(device):
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    x, H = sim.x, sim.H
    st = sim._surface_state(x, sim.rho)
    near, (ins, u, rk, dr, dphi) = st["samples"]

    # (a) default off
    assert DeltaSPHConfig().tensileExact is False

    # (b) the wall tensile integral: polar quadrature vs the exact edge reduction (the quadrature's own error)
    T_quad = _polar_T(sim, near)
    T_exact = tensile_vector_scene(sim.scene, x[near], H)
    scale = float(T_exact.abs().max())
    assert scale > 1e6
    assert float((T_quad - T_exact).abs().max()) <= 3e-2 * scale, (float((T_quad - T_exact).abs().max()), scale)
    assert float((T_quad + T_exact).abs().max()) > 1.0 * scale

    # (c) the effect on shift: the switch changes the shift by a visible but small amount
    u_q = sim.shift(sim.dt)
    sim.cfg.tensileExact = True
    try:
        u_e = sim.shift(sim.dt)
    finally:
        sim.cfg.tensileExact = False
    rel = float((u_q - u_e).norm(dim=1).max()) / float(u_q.norm(dim=1).max())
    assert 5e-3 <= rel <= 0.15, rel
    print("(c) max|u_q - u_e| / max||u_q|| = %.4e" % rel)

    # (d) five steps with tensileExact = True stay finite (no NaN / inf in x, v, rho)
    sim.cfg.tensileExact = True
    try:
        for _ in range(5):
            sim.step()
    finally:
        sim.cfg.tensileExact = False
    for a in (sim.x, sim.v, sim.rho):
        assert torch.isfinite(a).all()
