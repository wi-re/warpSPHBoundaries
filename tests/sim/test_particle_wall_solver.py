"""One scheme, two boundary representations (`cfg.wallParticleSpacing`): the delta+-SPH right-hand side and steps with the wall as a LATTICE OF WALL PARTICLES (`ParticleBoundary`, aggregates as pair sums) against the
same scheme with the ANALYTIC bodies (exact edge integrals), on a disk in a hydrostatic tank with a translating disk.  The boundary-condition physics (pressure extrapolation and clamp, free-slip mirror,
no-penetration impulse, shifting) is the scheme's and identical; only the integrals differ, by the quadrature error of the lattice (first order in s / H, s = dx / 8 -> s / H = 1/32).

Measured at s = dx / 8: the acceleration differs by 2.5 % of its maximum, the pressure / viscous loads by 0.06 %, after 15 steps with the disk moving at 0.3 m/s and particle shifting on (the wall tensile term is a wall-particle sum too) the positions by 0.018 dx (7e-4 dx without shifting),
the velocities by 1.1 % of the maximum speed, the densities by 1.3e-4; at the coarser s = dx the differences are several times larger (the test can fail).  The wall-particle form is eager only (the graph step is not used with it).
"""
import math

import pytest
import torch
import warp as wp

import edgebound  # noqa: F401
from edgebound.sim.deltasph2d import DeltaSPHConfig
import test_implicit_disk_solver as disk

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
G = 9.81


def run(spacing, steps=15):
    cfg = DeltaSPHConfig(gravity=(0.0, -G), c0=20.0 * math.sqrt(G * 0.5), graphStep=False, fluidWarp=False, wallParticleSpacing=spacing, noPen="impulse", shifting=True)
    sim = disk.make("implicit", dp=0.03, cfg=cfg, motion=((0.3, 0.0), (0.0, 0.0)))
    a, _, loads = sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    for _ in range(steps):
        sim.step()
    return a, loads, sim


def diffs(ref, got):
    a0, l0, s0 = ref
    a1, l1, s1 = got
    return dict(a=float((a1 - a0).abs().max() / a0.abs().max()), loads=float((l1 - l0).abs().max() / l0.abs().max()), x=float((s1.x - s0.x).abs().max()) / s0.dx,
                v=float((s1.v - s0.v).abs().max() / s0.v.abs().max()), rho=float((s1.rho - s0.rho).abs().max()))


def test_wall_particles_reproduce_the_analytic_scheme():
    ref = run(0.0)
    fine, coarse = diffs(ref, run(0.125)), diffs(ref, run(1.0))
    assert fine["a"] < 0.05 and fine["loads"] < 0.01 and fine["x"] < 0.05 and fine["v"] < 0.03 and fine["rho"] < 1e-3, fine
    assert coarse["a"] > 2 * fine["a"] and coarse["v"] > 2 * fine["v"], (coarse, fine)                # the lattice spacing matters: negative control
    b = disk.make("implicit", dp=0.03, cfg=DeltaSPHConfig(graphStep=True, wallParticleSpacing=0.125))
    from edgebound.sim.graphstep import graphable
    assert not graphable(b)                                                                        # wall particles are not captured
