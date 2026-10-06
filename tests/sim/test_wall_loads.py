"""Force / torque bookkeeping of the wall terms (`DeltaSPH2D.wallLoads` [3, B, 3] = pressure, wall viscous, no-penetration impulse; Fx, Fy, torque z about the body centre; `rhs(want_forces=True)` returns the
first two).  The load of the fluid on a body is the reaction of the particle accelerations the wall terms produce: -m a, acting at the particle (pressure: a central pair force has the torque of the force at
the particle), or at the contact point (tangential no-slip friction, no-penetration impulse).

(a) absolute: the half-filled hydrostatic tank at t = 0 (dp = 0.025): the pressure load on the tank is the analytic one: Fy = -rho g H L/2 (1.5 %), torque about the centre (floor + left wall, 3 %), Fx = -rho g H^2/2
    (left wall, 6 %, first order in dp: the error halves from dp = 0.05 to 0.025), viscous load zero at rest;
(b) accounting identity, all wall viscosity forms and both wall paths (fused / sceneOperation): the viscous load equals -m sum (a_on - a_off) and its torque the one about the lever arms, to round-off;
(c) the no-slip friction uses the wall velocity at the wall point: a particle that moves with a rotating floor at its contact point feels no friction, one that moves with the floor velocity at ITS OWN position does;
(d) the no-penetration load is the impulse reaction: F = -m dv / dt summed over the corrected particles, torque about the centre at the contact point, zero without corrections; a graph replay has the same loads
    as the eager step (bit for bit).
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim import cases
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64


def half_tank(dp, device, cfg=None):
    sim = cases.hydrostatic_tank(dp=dp, domain="surface", device=device, cfg=cfg)[0]
    m = sim.x[:, 0] < 0
    return DeltaSPH2D(sim.x[m], torch.zeros_like(sim.x[m]), sim.rho[m], dp, sim.scene, sim.cfg, device)


@pytest.mark.parametrize("device", DEVICES)
def test_hydrostatic_load_and_torque(device):
    g, Hw = 9.81, 0.5
    Fy, Fx, tau = -g * Hw * 1.2, -0.5 * g * Hw ** 2, g * Hw * 0.72 + g * (-0.054166667)
    err = {}
    for dp in (0.05, 0.025):
        sim = half_tank(dp, device)
        _, _, loads = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
        p = loads[0, 0].tolist()
        err[dp] = abs(p[0] - Fx) / abs(Fx)
        assert float(loads[1].abs().max()) == 0.0                                          # fluid at rest: no viscous load
    assert abs(p[1] - Fy) / abs(Fy) < 0.015, p
    assert abs(p[2] - tau) / abs(tau) < 0.03, p
    assert err[0.025] < 0.06 and err[0.025] < 0.7 * err[0.05], err


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("form", ["laplacian", "pairwise", "noslip"])
def test_viscous_load_is_the_reaction_of_the_wall_term(form, fused):
    device = DEVICES[0]
    cfg = DeltaSPHConfig(wallViscosityForm=form, fusedWall=fused, graphStep=False, fluidWarp=False)
    sim = cases.hydrostatic_tank(dp=0.06, domain="surface", device=device, cfg=cfg)[0]
    rng = np.random.default_rng(0)
    v = torch.as_tensor(rng.normal(0, 0.3, sim.x.shape), dtype=F64, device=device)
    a_on, _, loads = sim.rhs(sim.x, v, sim.rho, want_forces=True)
    sim.cfg.wallViscosity = False
    a_off, _, _ = sim.rhs(sim.x, v, sim.rho)
    dA = a_on - a_off
    F = -sim.m * dA.sum(0)
    c = sim.scene.bodies[0].center
    got = loads[1, 0]
    assert float(dA.abs().max()) > 1e-3                                                   # the wall term does something
    assert float((got[:2] - F).abs().max()) <= 1e-9 * float(F.abs().max()), (got[:2].tolist(), F.tolist())
    if form != "noslip":                                                                   # lever = the particle
        r = sim.x - c
        tau = (-sim.m * (r[:, 0] * dA[:, 1] - r[:, 1] * dA[:, 0])).sum()
        assert abs(float(got[2]) - float(tau)) <= 1e-9 * max(abs(float(tau)), 1e-3), (float(got[2]), float(tau))


@pytest.mark.parametrize("device", DEVICES)
def test_noslip_friction_uses_the_wall_velocity_at_the_wall_point(device):
    sim = cases.hydrostatic_tank(dp=0.06, domain="surface", device=device, cfg=DeltaSPHConfig(wallViscosityForm="noslip", graphStep=False, fluidWarp=False))[0]
    b = sim.scene.bodies[0]
    b.angularVelocity = 1.5                                                                # the tank rotates about its centre
    d, n, hit = sim.scene.signed_distance(sim.x)
    cp = sim.x - d[:, None] * n
    near = d < 0.6 * sim.dx                                                                # the first row of every wall
    assert int(near.sum()) > 10
    sim.cfg.wallViscosity = False
    a_off, _, _ = sim.rhs(sim.x, b.velocityAt(cp), sim.rho)
    sim.cfg.wallViscosity = True
    a_cp, _, _ = sim.rhs(sim.x, b.velocityAt(cp), sim.rho)
    a_x, _, _ = sim.rhs(sim.x, b.velocityAt(sim.x), sim.rho)
    # a particle moving with the wall at its wall point: no friction beyond the fluid-fluid / pressure terms (which do not depend on this wall term: compare the wall viscous part only)
    assert float((a_x - a_off)[near].abs().max()) > 1e-3                                   # negative control: moving with the floor velocity at the particle's own position leaves a relative velocity omega d
    assert float((a_cp - a_off)[near].abs().max()) < 1e-9 * max(1.0, float(a_off.abs().max()))


@pytest.mark.parametrize("device", DEVICES)
def test_no_penetration_load_is_the_impulse_reaction(device):
    sim = cases.hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=DeltaSPHConfig(noPen="impulse", graphStep=False))[0]
    b = sim.scene.bodies[0]
    b.linearVelocity = torch.tensor([0.2, 0.0], dtype=F64, device=device)
    p = torch.tensor([[0.3, -0.6 + 0.004]], dtype=F64, device=device)                       # 0.004 above the floor
    sim.x = sim.x.clone(); sim.x[0] = p[0]
    sim.v = torch.zeros_like(sim.v); sim.v[0] = torch.tensor([0.5, -1.0], dtype=F64, device=device)
    v0 = sim.v.clone()
    sim.dt_t.fill_(1e-3)
    n_act = int(sim.no_penetration())
    dv = sim.v - v0
    assert n_act >= 1 and float(dv[0].abs().max()) > 0
    L = sim._nopenLoad[0]
    F = -sim.m * dv.sum(0) / 1e-3
    assert float((L[:2] - F).abs().max()) <= 1e-12 * float(F.abs().max())
    d, n, _ = sim.scene.signed_distance(sim.x)
    cp = sim.x - d[:, None] * n
    r = cp - b.center
    tau = (-sim.m * dv / 1e-3 * 1.0)
    tau = (r[:, 0] * tau[:, 1] - r[:, 1] * tau[:, 0]).sum()
    assert abs(float(L[2]) - float(tau)) <= 1e-9 * abs(float(tau))
    sim.v = torch.zeros_like(sim.v); sim.v[0] = torch.tensor([0.2, 0.7], dtype=F64, device=device)     # moving away from the floor (relative to it): no impulse, no load
    assert int(sim.no_penetration()) == 0 and float(sim._nopenLoad.abs().max()) == 0.0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
def test_graph_replay_has_the_same_loads():
    def go(graph):
        sim = cases.marrone_dambreak(nx=24, shifting=True, noPen="impulse", device="cuda:0", graphStep=graph)[0]
        for _ in range(30):
            sim.step()
        return sim
    a, b = go(False), go(True)
    assert torch.equal(a.wallLoads, b.wallLoads) and torch.equal(a.wallForce, b.wallForce)
    assert float(a.wallLoads[0].abs().max()) > 0
