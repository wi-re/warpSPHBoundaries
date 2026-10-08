"""Per-body boundary-condition policies in the delta+-SPH solver (`Body.bc`, sim/bcclosures.py): periodic channel between two plates.

(a) static fluid: at rest stays at rest for every policy (no spurious wall velocity);
(b) uniform stream U along the plates, no forcing: freeSlip and a pinned 'constant' wall at U exert no shear (the mean velocity stays at U to 1e-5), noSlip drags it down (0.18 U after 1 s, measured),
    'zeros' on a plate at rest is the same wall as noSlip (equal to 1e-12);
(c) Couette: the upper plate moves with U.  noSlip drags the fluid (mean velocity of the top layer 0.85 U, measured); 'constant' pinned at the plate's velocity is the same wall (equal to 1e-12); 'zeros'
    ignores the plate's motion (the fluid stays at rest, max|v| < 1e-9 U) and so does freeSlip (no tangential traction);
(d) the policy is validated, freeSlip refuses the Morris operator.
Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"
PER = Periodic((0.0, -9.0), (1.0, 9.0), (True, False))
U = 0.5


def channel(bc=("noSlip", "noSlip"), upper_velocity=0.0, v0=0.0, bcValue=(U, 0.0), n=24, W=0.5, wallViscosityForm="noslipMirror", **cfgkw):
    dx = 1.0 / n
    ny = int(round(W / dx))
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    vel = np.zeros_like(pos)
    vel[:, 0] = v0
    plate = lambda i, yc, v, pol: Body(bodyId=i, center=(0.5, yc), linearVelocity=(v, 0.0), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))], bc=pol, bcValue=bcValue)
    scene = Scene([plate(0, -0.15, 0.0, bc[0]), plate(1, W + 0.15, upper_velocity, bc[1])], DEV)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, periodic=PER, graphStep=True, shifting=True, wallViscosityForm=wallViscosityForm, **cfgkw)
    return DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, scene, cfg, DEV, support=4 * dx), W


def run(sim, T):
    while sim.time < T:
        sim.step()
    return sim


@pytest.mark.parametrize("pol", ["noSlip", "freeSlip", "zeros", "constant"])
def test_static_fluid_stays_at_rest(pol):
    sim, _ = channel(bc=(pol, pol), bcValue=(0.0, 0.0))
    run(sim, 0.5)
    assert float(sim.v.abs().max()) < 1e-6


def test_uniform_stream():
    mean = {}
    for name, bc, val in (("noSlip", ("noSlip",) * 2, (0.0, 0.0)), ("freeSlip", ("freeSlip",) * 2, (0.0, 0.0)), ("zeros", ("zeros",) * 2, (0.0, 0.0)), ("constant", ("constant",) * 2, (U, 0.0))):
        sim, _ = channel(bc=bc, v0=U, bcValue=val)
        run(sim, 1.0)
        mean[name] = float(sim.v[:, 0].mean()) / U
        if name == "noSlip":
            vNo = sim.v.clone()
        if name == "zeros":
            assert float((sim.v - vNo).abs().max()) < 1e-12
    assert mean["freeSlip"] > 0.99999 and mean["constant"] > 0.99999, mean                          # measured 1 - 4e-16: no shear at all
    assert mean["noSlip"] < 0.3, mean                                                                # measured 0.178


def test_couette_drag_pinned_and_free_walls():
    out = {}
    for name, bc, val in (("noSlip", ("noSlip",) * 2, (0.0, 0.0)), ("constant", ("noSlip", "constant"), (U, 0.0)), ("zeros", ("noSlip", "zeros"), (0.0, 0.0)), ("freeSlip", ("noSlip", "freeSlip"), (0.0, 0.0))):
        sim, W = channel(bc=bc, upper_velocity=U, bcValue=val)
        run(sim, 1.5)
        y = sim.x[:, 1]
        top = y > 0.75 * W
        out[name] = (sim.v.clone(), float(sim.v[top, 0].mean()) / U, float(sim.v.abs().max()) / U)
    assert out["noSlip"][1] > 0.7                                                                  # measured 0.846
    assert float((out["constant"][0] - out["noSlip"][0]).abs().max()) < 1e-12
    assert out["zeros"][2] < 1e-9 and out["freeSlip"][2] < 1e-9, (out["zeros"][2], out["freeSlip"][2])      # measured 2e-12


def test_policy_validation():
    with pytest.raises(ValueError):
        channel(bc=("noSlip", "extended"))
    sim, _ = channel(bc=("freeSlip", "freeSlip"), wallViscosityForm="noslipMoment", fluidViscosity="morris")
    with pytest.raises(NotImplementedError):
        sim.step()


def _boosted(boost, frame, form="noslipMirror", T_end=1.0):
    sim, W = channel(upper_velocity=U + boost, v0=boost, wallViscosityForm=form, shiftMachFrame=frame)
    sim.scene.bodies[0].linearVelocity = torch.tensor([boost, 0.0], dtype=torch.float64, device=DEV)
    run(sim, T_end)
    v = sim.v.clone()
    v[:, 0] -= boost
    return v, sim.rho.clone()


def test_galilean_boost():
    """a uniform boost of the fluid AND the plates leaves the dynamics (in the boosted frame) unchanged: the wall closures are relative (to 1e-14 without shifting, measured 4e-15); the shifting's Mach
    number of absolute speeds is the only break (measured 8e-5 of U = 0.5), `shiftMachFrame='wall'` removes it (1e-13)."""
    v0, r0 = _boosted(0.0, "absolute")
    vb, rb = _boosted(0.7, "absolute")
    absolute = float((vb - v0).abs().max())
    assert 1e-7 < absolute < 1e-2, absolute                                                             # the break exists, and is small
    w0, _ = _boosted(0.0, "wall")
    wb, rwb = _boosted(0.7, "wall")
    assert float((wb - w0).abs().max()) < 1e-9


def test_wall_frame_is_the_absolute_frame_for_walls_at_rest():
    out = []
    for frame in ("absolute", "wall"):
        sim, _ = channel(v0=0.3, shiftMachFrame=frame, bodyForce=(0.05, 0.0))                          # static plates, driven flow
        run(sim, 0.8)
        out.append(sim.v.clone())
    assert torch.equal(out[0], out[1])
    with pytest.raises(ValueError):
        sim, _ = channel(shiftMachFrame="nearest")
        sim.step()
