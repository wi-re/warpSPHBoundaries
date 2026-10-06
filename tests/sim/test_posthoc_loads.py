"""Wall loads recovered post hoc from an exported trajectory (`DeltaSPH2D.loadsAt`): a fresh solver on the same scene, fed with the exported states (x, v, rho) of consecutive steps only, reproduces the loads the
running solver books (`wallLoads`: pressure and wall viscous force and torque on the tank of the dam break during the impact).  The wall terms are a pure function of the fluid state, the body poses and gravity.
The booked loads are those of the half-step state of the step, so the average of the loads of the two exported frames n, n + 1 matches them to 2e-3 of the peak (measured 5e-4), a single frame to 3e-2 (measured 1.2e-2);
the no-penetration impulse is not recoverable from a state (zero here: no contact).
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim import cases

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")


def make():
    return cases.marrone_dambreak(nx=24, shifting=True, noPen="impulse", device="cuda:0", fluidWarp=True, graphStep=False)[0]


def test_loads_from_exported_states_equal_the_booked_loads():
    run, post = make(), make()
    for _ in range(240):
        run.step()
    prev = (run.x.clone(), run.v.clone(), run.rho.clone())
    booked, mean, single = [], [], []
    for _ in range(60):
        run.step()
        cur = (run.x.clone(), run.v.clone(), run.rho.clone())
        la, lb = post.loadsAt(*prev), post.loadsAt(*cur)
        booked.append(run.wallLoads[:2].cpu().numpy())
        mean.append((0.5 * (la + lb)).cpu().numpy())
        single.append(la.cpu().numpy())
        prev = cur
    b, m, s = np.array(booked), np.array(mean), np.array(single)
    scale = np.abs(b[:, 0, 0, :2]).max()
    assert scale > 1.0                                                                       # the water pushes on the tank
    for k in (0, 1):                                                                         # pressure, wall viscous
        sc = max(np.abs(b[:, k, 0, :2]).max(), 1e-12)
        assert np.abs(m[:, k, 0, :2] - b[:, k, 0, :2]).max() <= 2e-3 * sc, k
        assert np.abs(s[:, k, 0, :2] - b[:, k, 0, :2]).max() <= 3e-2 * sc, k
    assert np.abs(m[:, 0, 0, 2] - b[:, 0, 0, 2]).max() <= 2e-3 * np.abs(b[:, 0, 0, 2]).max()
    assert float(run.wallLoads[2].abs().max()) == 0.0                                      # no contact correction in this window
