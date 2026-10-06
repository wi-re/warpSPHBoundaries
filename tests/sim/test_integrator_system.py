"""The step as a warpSPHIntegrators system (`sim/system.py`, phase 4): `symplecticEuler(DeltaSPHSystem, dt, deltaSPHRhs)` replaced the hand-coded predictor-corrector.

(a) the library step equals the hand-coded step it replaced (kept here as the oracle, written out from `DeltaSPH2D.rhs / shift / no_penetration`): dam break with shifting and the no-penetration impulse, with and
    without the time-centred continuity (the drift field), and with a body moving at constant velocity and angular velocity (the old two `Body.move(dt/2)` and the integrated body state agree for a = 0);
(b) the bodies are integrated state: the right-hand sides see the stage poses (c, c + dt/2 v; v, v + dt/2 a), the step ends at c + dt v + dt^2/2 a, v + dt a (exact for the constant acceleration; the two explicit half
    moves of the old step were first order in a);
(c) the step leaves the host bodies as plain bodies (python angle / omega, no stage tensors bound).
"""
import pytest
import torch
import warp as wp

import edgebound  # noqa: F401
from edgebound.sim import cases

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
DEV = "cuda:0"
F64 = torch.float64


def hand_step(sim):
    """the DualSPHysics symplectic Euler as DeltaSPH2D._step_core had it before phase 4 (time-centred continuity, shifting, no-penetration, two half body moves)."""
    cfg, dt = sim.cfg, sim.dt
    sim.dt_t.fill_(dt)
    a0, d0, _ = sim.rhs(sim.x, sim.v, sim.rho)
    xh, vh, rh = sim.x + 0.5 * dt * sim.v, sim.v + 0.5 * dt * a0, sim.rho + 0.5 * dt * d0
    for b in sim.scene.bodies:
        b.move(0.5 * dt)
    a1, d1, _ = sim.rhs(xh, vh, rh)
    vn = sim.v + dt * a1
    x = sim.x + 0.5 * dt * (sim.v + vn)
    rho = sim.rho + dt * d1
    if cfg.timeCentred:
        rho = rho + dt * (sim._kinematic(0.5 * (sim.v + vn)) - sim._kinematic(vh))
    sim.x, sim.v, sim.rho = x, vn, rho
    for b in sim.scene.bodies:
        b.move(0.5 * dt)
    if cfg.shifting:
        sim.x = sim.x + sim.shift(sim.dt_t)
    sim.no_penetration()
    sim.dt_t.copy_(sim._next_dt(a1))
    sim.time += dt
    sim.dt = float(sim.dt_t)


def make(timeCentred=True):
    sim = cases.marrone_dambreak(nx=24, shifting=True, noPen="impulse", device=DEV, fluidWarp=True, graphStep=False)[0]       # the case switches the time-centred continuity on
    sim.cfg.timeCentred = timeCentred
    return sim


def move(sim):
    b = sim.scene.bodies[0]
    b.linearVelocity = torch.tensor([0.05, -0.02], dtype=F64, device=DEV)
    b.angularVelocity = 0.03


@pytest.mark.parametrize("timeCentred,moving", [(False, False), (True, False), (False, True)])
def test_library_step_equals_the_hand_coded_step(timeCentred, moving):
    a, b = make(timeCentred), make(timeCentred)
    if moving:
        move(a), move(b)
    for _ in range(30):
        a.step()
        hand_step(b)
    for name in ("x", "v", "rho"):
        err = float((getattr(a, name) - getattr(b, name)).abs().max())
        assert err < 1e-10, (name, err)
    assert abs(a.time - b.time) < 1e-14 and abs(a.dt - b.dt) <= 1e-12 * a.dt
    if moving:
        ba, bb = a.scene.bodies[0], b.scene.bodies[0]
        assert float((ba.center - bb.center).abs().max()) < 1e-14 and abs(float(ba.angle) - float(bb.angle)) < 1e-14 and float(ba.angle) > 0.0


def test_bodies_are_integrated_state():
    sim = make()
    b = sim.scene.bodies[0]
    c0, v0 = b.center.clone(), torch.tensor([0.05, -0.02], dtype=F64, device=DEV)
    a0 = torch.tensor([0.3, -0.1], dtype=F64, device=DEV)
    b.linearVelocity, b.linearAcceleration = v0.clone(), a0.clone()
    b.angle, b.angularVelocity, b.angularAcceleration = 0.1, 0.2, 0.4
    seen = []
    rhs = sim.rhs

    def spy(x, v, rho, want_forces=False):
        seen.append((b.center.clone(), b.linearVelocity.clone(), float(b.angle), float(b.angularVelocity)))
        return rhs(x, v, rho, want_forces)
    sim.rhs = spy
    dt = sim.dt
    sim.step()
    h = 0.5 * dt
    assert len(seen) == 2
    assert float((seen[0][0] - c0).abs().max()) == 0.0 and float((seen[0][1] - v0).abs().max()) == 0.0
    assert float((seen[1][0] - (c0 + h * v0)).abs().max()) < 1e-15 and float((seen[1][1] - (v0 + h * a0)).abs().max()) < 1e-15           # stage 1: explicit half step
    assert abs(seen[1][2] - (0.1 + h * 0.2)) < 1e-15 and abs(seen[1][3] - (0.2 + h * 0.4)) < 1e-15
    assert float((b.center - (c0 + dt * v0 + 0.5 * dt * dt * a0)).abs().max()) < 1e-15                                                    # the end of the step: exact for a constant acceleration
    assert float((b.linearVelocity - (v0 + dt * a0)).abs().max()) < 1e-15
    assert abs(b.angle - (0.1 + dt * 0.2 + 0.5 * dt * dt * 0.4)) < 1e-15 and abs(b.angularVelocity - (0.2 + dt * 0.4)) < 1e-15


def test_host_bodies_are_plain_after_a_step():
    sim = make()
    move(sim)
    for _ in range(3):
        sim.step()
    b = sim.scene.bodies[0]
    assert isinstance(b.angle, float) and isinstance(b.angularVelocity, float) and getattr(b, "_cs", None) is None
    assert b.center.shape == (2,) and b.linearVelocity.shape == (2,)
