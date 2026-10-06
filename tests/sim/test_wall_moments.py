"""The exact-moment no-slip wall closure (`wallViscosityForm="noslipMoment"`, sim/wallmoments.py).

(a) `WallPairMoments`: the tabulated moments T_k^{nn, tt}(d) = int_S P(y) s'^k dA' of the pair weight P = (W'/r^3) y (x) y over the solid half-plane (s' the distance from the wall) agree with a direct 2D quadrature;
(b) operator consistency on the exact Poiseuille parabola (fluid at rest density, no dynamics): the total viscous acceleration of the rows next to the wall is within 20 % of the exact value 2 nu A of every row
    (`noslip` flux form: -0.15 to 1.8 times; the closure is exact for a profile that is a polynomial of the wall distance up to the lattice sum of the fluid pairs), the bulk rows within 1 % of the shear-wave viscosity;
(c) the plane Poiseuille channel (n = 32, momentum balance 0.5 %, profile amplitude 0.96-1.02 of the bulk-viscosity parabola; flux form 0.95, mirror 0.90; study: scripts/studies/periodic_channel.py).
Float64 contracts, CUDA for (b), (c).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene
from warpSPHBoundaries.sim.deltasph2d import KERNELS, DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory
from warpSPHBoundaries.sim.wallmoments import WallPairMoments

pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
F64 = torch.float64
DEV = "cuda:0"
PER = Periodic((0.0, -9.0), (1.0, 9.0), (True, False))


def test_tables_equal_a_direct_quadrature():
    W, dW, xi, ks = KERNELS[DeltaSPHConfig().kernel]
    H = 0.1
    wm = WallPairMoments(dW, H, "cpu")
    d = torch.tensor([0.0, 0.0125, 0.05, 0.09], dtype=F64)
    T = wm.eval(d)
    n = 2400
    xs = (np.arange(n) + 0.5) / n * 2 * H - H
    X, Y = np.meshgrid(xs, xs, indexing="ij")                                           # X: normal component of y (the solid is X <= -d), Y: tangential
    r = np.hypot(X, Y)
    for i, di in enumerate(d.tolist()):
        m = (r < H) & (r > 1e-9) & (X <= -di)
        L = dW(torch.tensor(r[m]), H).numpy() / r[m]
        s = di + X[m]
        dA = (2 * H / n) ** 2
        for k in range(3):
            for c, comp in enumerate((X[m], Y[m])):
                ref = (L * comp ** 2 / r[m] ** 2 * s ** k).sum() * dA
                assert abs(float(T[c, k, i]) - ref) <= 5e-3 * abs(ref) + 2e-3 * (1.0 if k == 0 else 0.1 ** k), (di, k, c, float(T[c, k, i]), ref)
    assert float(wm.eval(torch.tensor([H * 1.2], dtype=F64)).abs().max()) == 0.0         # beyond the support: no wall


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
@pytest.mark.parametrize("form,lo,hi", [("noslipMoment", 0.8, 1.2), ("noslip", -1.0, 3.0)])
def test_viscous_acceleration_of_the_exact_parabola(form, lo, hi):
    n = 32
    dx = 1.0 / n
    ny = n // 2
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    vel = np.zeros_like(pos)
    vel[:, 0] = pos[:, 1] * (W - pos[:, 1])                                             # A = 1
    plate = lambda i, yc: Body(bodyId=i, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(0, -0.15), plate(1, W + 0.15)], DEV)
    alpha = 0.5
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=alpha, periodic=PER, graphStep=False, shifting=False, wallViscosityForm=form)
    sim = DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, scene, cfg, DEV, support=4 * dx)
    acc = sim.rhs(sim.x, sim.v, sim.rho)[0].cpu().numpy()[:, 0].reshape(n, ny).mean(0)
    nu = alpha * 10.0 * 4 * dx / (8 * 2.821384729) * 0.955                              # bulk: the shear-wave law of the discretisation
    ratio = acc / (-2.0 * nu)
    assert np.abs(ratio[6:ny - 6] - 1.0).max() < 0.02                                    # the bulk rows
    assert (ratio[:3] > lo).all() and (ratio[:3] < hi).all(), ratio[:4]                  # the wall rows
    assert (ratio[-3:] > lo).all() and (ratio[-3:] < hi).all()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
def test_poiseuille_channel_with_the_moment_closure():
    n, W0 = 32, 0.5
    dx = 1.0 / n
    ny = int(round(W0 / dx))
    W = ny * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(ny) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    plate = lambda i, yc: Body(bodyId=i, center=(0.5, yc), reps=[BoxRep((-0.8, -0.15), (0.8, 0.15))])
    scene = Scene([plate(0, -0.15), plate(1, W + 0.15)], DEV)
    f = 0.05
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, periodic=PER, bodyForce=(f, 0.0), graphStep=True, shifting=True, wallViscosityForm="noslipMoment")
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, DEV, support=4 * dx)
    hist = LoadHistory()
    while sim.time < 6.0:
        sim.step()
        hist.record(sim)
    F = sum(hist.total(b)[-200:, 0].mean() for b in range(2))
    assert abs(F / (f * len(pos) * dx * dx) - 1.0) < 5e-3
    y, u = sim.x[:, 1].cpu().numpy(), sim.v[:, 0].cpu().numpy()
    nu = 0.5 * 10.0 * 4 * dx / (8 * 2.821384729) * 0.955
    ex = f * y * (W - y) / (2 * nu)
    amp = float((u * ex).sum() / (ex * ex).sum())
    assert 0.95 < amp < 1.03, amp
