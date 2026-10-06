"""Torque bookkeeping of the no-slip wall friction (`wallFrictionLever`): the fluid loses angular momentum where the friction acts on the PARTICLE, so the load on the bodies must be booked at the particle to conserve
angular momentum between the fluid and the walls.  Rigid rotation of the fluid in an annulus (inner cylinder rotating with it, outer fixed): fluid-fluid terms and the (radial) wall pressure exert no torque, the
wall friction is the only one, so  sum_bodies torque + sum m x cross acc = 0  must hold with `particle` and fails by the factor r_contact / r_particle with `contact`.  (Study: scripts/studies/taylor_couette.py;
n = 32, torque on the outer / inner 1.2103 with `contact`, 1.0004 with `particle`.)  Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"


def annulus(form, lever, n=24, r1=0.2, r2=0.5, omega=1.0):
    dx = 1.0 / n
    g = np.arange(-int(r2 / dx) - 1, int(r2 / dx) + 2)
    X, Y = np.meshgrid(dx * (g + 0.5), dx * (g + 0.5), indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    r = np.linalg.norm(P, axis=1)
    pos = P[(r >= r1 + 0.5 * dx) & (r <= r2 - 0.5 * dx)]
    vel = omega * np.stack([-pos[:, 1], pos[:, 0]], 1)                     # rigid rotation with the inner cylinder
    inner = Body(bodyId=0, center=(0.0, 0.0), angularVelocity=omega, reps=[DiskArrayRep([(0.0, 0.0)], [r1])])
    outer = Body(bodyId=1, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=r2, solid="outside"))])
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, graphStep=False, shifting=False, wallViscosityForm=form, wallFrictionLever=lever)
    return DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, Scene([inner, outer], DEV), cfg, DEV, support=4 * dx)


def balance(form, lever):
    sim = annulus(form, lever)
    acc, _, forces = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    x = sim.x
    dLdt = sim.m * (x[:, 0] * acc[:, 1] - x[:, 1] * acc[:, 0]).sum()                  # angular momentum rate of the fluid
    T = forces[:, :, 2].sum()                                                          # torque of the fluid on both bodies (pressure + viscous)
    Tin = forces[:, 0, 2].sum()
    return float(dLdt + T), float(Tin), float(T)


@pytest.mark.parametrize("form", ["noslip", "noslipCurv"])
def test_particle_lever_conserves_angular_momentum_contact_lever_does_not(form):
    res, Tin, _ = balance(form, "particle")
    assert abs(Tin) > 1e-4
    assert abs(res) <= 1e-9 * abs(Tin), (res, Tin)
    resc, Tinc, _ = balance(form, "contact")
    assert abs(resc) > 1e-2 * abs(Tinc), (resc, Tinc)                                  # the contact-point lever misses the arm difference: a few per cent of the torque
