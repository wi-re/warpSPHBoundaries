"""CUDA-graph replay of the whole step (`sim/graphstep.py`, cfg.graphStep) against the eager step it records.

(a) dam break and sloshing (rolling gravity = a static input updated every call), 40 steps: positions, velocities, densities, the wall force, the host time and dt are torch.equal to the eager run with
    the same cfg (the graph replays the very kernels of the eager step: a bit-level contract, not a tolerance), one capture and 40 replays;
(b) a run long enough for the Verlet list to become invalid: the step that finds the flag set is run eagerly, the next one captures against the new list, and the state stays equal to the eager run;
(c) a configuration the graph cannot take (box domain, moving body) uses the eager step silently: `_graphed` is False and the state equals the eager run.
CUDA only.
"""
import pytest
import torch
import warp as wp

import edgebound  # noqa: F401
from edgebound.sim import cases

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
DEV = "cuda:0"


def run(maker, steps, graph, **kw):
    sim = maker(device=DEV, fluidWarp=True, graphStep=graph, **kw)[0]
    for _ in range(steps):
        sim.step()
    return sim


def same(a, b):
    for name in ("x", "v", "rho", "wallForce"):
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


def test_unsupported_configurations_run_eagerly():
    kw = dict(nx=30, shifting=True, noPen="impulse")
    a, b = run(cases.marrone_dambreak, 20, False, domain="box", **kw), run(cases.marrone_dambreak, 20, True, domain="box", **kw)
    assert b._graphed is False
    same(a, b)
