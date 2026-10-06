"""CUDA-graph replay of the whole step (`sim/graphstep.py`, cfg.graphStep) against the eager step it records.

(a) dam break and sloshing (rolling gravity = a static input updated every call), 40 steps: positions, velocities, densities, the wall force, the host time and dt are torch.equal to the eager run with
    the same cfg (the graph replays the very kernels of the eager step: a bit-level contract, not a tolerance), one capture and 40 replays;
(b) a run long enough for the Verlet list to become invalid: the step that finds the flag set is run eagerly, the next one captures against the new list, and the state stays equal to the eager run;
(c) the box domain and bodies with prescribed motion are captured (the bodies are part of the integrated state: their state at the start of the step is a device input); (d) a configuration the graph cannot take (pairwise wall viscosity) uses the eager step silently.
CUDA only.
"""
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim import cases

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
DEV = "cuda:0"


def run(maker, steps, graph, **kw):
    sim = maker(device=DEV, fluidWarp=True, graphStep=graph, **kw)[0]
    for _ in range(steps):
        sim.step()
    return sim


def same(a, b):
    for name in ("x", "v", "rho", "wallForce", "wallLoads"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.time == b.time and a.dt == b.dt


@pytest.mark.parametrize("case", ["dambreak", "sloshing"])
def test_replay_equals_eager(case):
    maker, kw = (cases.marrone_dambreak, dict(nx=30, shifting=True, noPen="impulse")) if case == "dambreak" else (cases.sloshing_tank, dict(nx=40))
    a, b = run(maker, 40, False, **kw), run(maker, 40, True, **kw)
    same(a, b)
    assert b._graphed and b._graphed.stats["captures"] == 1 and b._graphed.stats["replays"] == 40 and b._graphed.stats["eager"] == 0


def test_verlet_rebuild_falls_back_to_one_eager_step_and_recaptures():
    kw = dict(nx=30, shifting=True, noPen="impulse")
    a = run(cases.marrone_dambreak, 400, False, **kw)
    b = run(cases.marrone_dambreak, 400, True, **kw)
    same(a, b)
    st = b._graphed.stats
    assert st["eager"] >= 1 and st["captures"] >= 2 and st["captures"] + st["replays"] + st["eager"] >= 400, st


@pytest.mark.parametrize("domain", ["box", "surface"])
def test_box_domain_is_captured(domain):
    kw = dict(nx=30, shifting=True, noPen="impulse")
    a, b = run(cases.marrone_dambreak, 30, False, domain=domain, **kw), run(cases.marrone_dambreak, 30, True, domain=domain, **kw)
    same(a, b)
    assert b._graphed and b._graphed.stats["captures"] == 1 and b._graphed.stats["replays"] == 30


@pytest.mark.parametrize("domain", ["box", "surface"])
def test_moving_body_poses_are_graph_inputs(domain):
    """a tank body with a prescribed motion (velocity, acceleration, rotation, angular acceleration): the body state is integrated on the device with the particles and the replay equals the eager step bit for bit
    (the pose changes every step, so a baked pose would be wrong from the second step on); the host bodies end where the eager bodies end."""
    def go(graph):
        sim = cases.marrone_dambreak(nx=24, shifting=True, noPen="impulse", domain=domain, device=DEV, fluidWarp=True, graphStep=graph)[0]
        b = sim.scene.bodies[0]
        b.linearVelocity = torch.tensor([0.05, -0.02], dtype=torch.float64, device=DEV)
        b.linearAcceleration = torch.tensor([0.01, 0.0], dtype=torch.float64, device=DEV)
        b.angularVelocity, b.angularAcceleration = 0.02, 0.005
        for _ in range(25):
            sim.step()
        return sim
    a, b = go(False), go(True)
    same(a, b)
    ba, bb = a.scene.bodies[0], b.scene.bodies[0]
    assert b._graphed and b._graphed.stats["replays"] >= 20
    assert torch.equal(ba.center, bb.center) and torch.equal(ba.linearVelocity, bb.linearVelocity) and float(ba.angle) == float(bb.angle) and float(ba.angularVelocity) == float(bb.angularVelocity)
    assert float(ba.angle) > 0.0 and float(ba.center.abs().max()) > 0.0


def test_unsupported_configurations_run_eagerly():
    """a configuration the graph cannot take (the pairwise wall viscosity builds index lists) uses the eager step silently."""
    kw = dict(nx=30, shifting=True, noPen="impulse", wallViscosityForm="pairwise")
    a, b = run(cases.marrone_dambreak, 10, False, **kw), run(cases.marrone_dambreak, 10, True, **kw)
    assert b._graphed is False
    same(a, b)
