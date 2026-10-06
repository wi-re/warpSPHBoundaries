"""Step 1d of docs/plan-next-steps.md: `sim/loads.py`.

(a) coefficients: drag along the stream, lift 90 degrees counter-clockwise, the 1/2 rho U^2 D normalisation, any stream direction;
(b) `strouhal` recovers the frequency of a sampled sine (also with a mean and a ramp) to 0.5 %;
(c) `LoadHistory` books the solver's loads, and `loads_from_frames` recovers them from the exported frames of a periodic fibre run (raw, unwrapped trajectory: the frames are the stored positions): pressure and viscous within
    2e-2 of the peak of the interval-mean (a smooth, unforced run; the booked loads are the half-step state).
Float64 contracts.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.sim.loads import LoadHistory, coefficients, loads_from_frames, strouhal, window_mean

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


def test_coefficients_follow_the_stream_direction():
    F = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    cd, cl = coefficients(F, rho=1.0, U=2.0, D=0.5)                       # q = 0.5 * 1 * 4 * 0.5 = 1
    assert np.allclose(cd, [2.0, 0.0]) and np.allclose(cl, [1.0, 3.0])
    th = 0.7
    e = (math.cos(th), math.sin(th))
    Fr = np.array([[2.0 * e[0] - 1.0 * e[1], 2.0 * e[1] + 1.0 * e[0], 0.0]])           # the same loads in a frame rotated by th
    cd2, cl2 = coefficients(Fr, 1.0, 2.0, 0.5, direction=e)
    assert np.allclose(cd2, [2.0]) and np.allclose(cl2, [1.0])


def test_strouhal_of_a_sampled_sine():
    t = np.arange(0, 40.0, 0.01)
    y = 0.3 + 0.002 * t + np.sin(2 * math.pi * 0.165 * t + 0.4)
    assert abs(strouhal(t, y, D=1.0, U=1.0) - 0.165) < 0.005 * 0.165
    assert abs(strouhal(t, y, D=2.0, U=4.0, t0=5.0) - 0.0825) < 0.005 * 0.0825
    assert abs(window_mean(t, np.ones_like(t), 10.0) - 1.0) < 1e-12


@pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
@pytest.mark.parametrize("device", DEVICES)
def test_booked_loads_and_loads_from_frames_agree(device):
    import sys
    sys.path.insert(0, "tests/sim")
    from test_pinned_frame import fibre_sim
    run, _ = fibre_sim(device)
    post, _ = fibre_sim(device)
    run.cfg.bodyForce = (0.0, 0.0)                                                                  # a smooth run without the driver: the band and the stream do the work
    hist = LoadHistory()
    for _ in range(30):
        run.step()
    frames, times = [(run.x.clone(), run.v.clone(), run.rho.clone())], [run.time]
    for _ in range(12):
        run.step()
        hist.record(run)
        frames.append((run.x.clone(), run.v.clone(), run.rho.clone()))
        times.append(run.time)
    t, mid = loads_from_frames(post, frames, times)
    booked = hist.total()[:, :3]
    assert booked.shape == mid.shape and np.abs(booked[:, :2]).max() > 1e-6
    # booked = pressure + viscous + impulse; the impulse is not recoverable (zero away from contact)
    imp = np.array(hist.loads)[:, 2, 0, :]
    sc = np.abs(booked[:, :2]).max()
    assert np.abs(mid[:, :2] - (booked[:, :2] - imp[:, :2])).max() <= 2e-2 * sc
    assert np.allclose(hist.series()[0], np.array(times[1:]))
