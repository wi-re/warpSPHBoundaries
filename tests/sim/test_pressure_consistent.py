"""`pressureConsistent`: a uniform pressure of ANY level and sign exerts no force near a wall (difference form of the pressure force), and the loads on the body do not depend on the pressure level.

Fluid at rest around a cylinder in a periodic box on a square lattice cut at dx/2 (the layout leaves the wall-consistency residual S_i = sum_j V_j grad W_ij + mu grad lambda non-zero: a uniform pressure P exerts -2 P S_i / rho_i):
(a) default: the acceleration of the near-wall particles is O(P) (>= 0.1 P at the first layers); `pressureConsistent`: zero to round-off for P = -2, 0.5, 5 (the clamp and the Antuono switch included);
(b) the pressure load on the body of a non-uniform smooth pressure (a linear ramp across the box) is the same with and without a uniform offset P_b (the consistent load uses P - mean(P)).
Float64 contracts, CUDA.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, CUDA")
DEV = "cuda:0"


def cylinder(consistent, n=32, R=0.2, **kw):
    dx = 1.0 / n
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    pos = pos[np.linalg.norm(pos - 0.5, axis=1) >= R + 0.5 * dx]
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [R])])], DEV)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, periodic=Periodic((0, 0), (1, 1)), graphStep=False, shifting=False, pressureConsistent=consistent, **kw)
    return DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, DEV, support=4 * dx), R, dx


@pytest.mark.parametrize("Pb", [-2.0, 0.5, 5.0])
def test_uniform_pressure_exerts_no_force(Pb):
    for consistent in (False, True):
        sim, R, dx = cylinder(consistent, backgroundPressure=Pb)
        acc = sim.rhs(sim.x, sim.v, sim.rho)[0]
        near = ((sim.x - 0.5).norm(dim=1) - R) < 3 * dx
        a = float(acc[near].norm(dim=1).max())
        if consistent:
            assert a < 1e-9, a
        else:
            assert a > 0.1 * abs(Pb), (a, Pb)


def test_loads_do_not_depend_on_the_pressure_level():
    loads = []
    for Pb in (0.0, 5.0):
        sim, R, dx = cylinder(True, backgroundPressure=Pb)
        rho = sim.rho * (1.0 + 1e-3 * (sim.x[:, 0] - 0.5))                                    # a smooth pressure ramp across the box (P = c0^2 (rho - 1))
        loads.append(sim.rhs(sim.x, sim.v, rho, want_forces=True)[2][0, 0, :2].clone())
    assert float(loads[0].norm()) > 1e-6
    assert float((loads[0] - loads[1]).norm()) <= 1e-9 * float(loads[0].norm())
