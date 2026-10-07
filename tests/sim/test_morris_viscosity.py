"""The Morris fluid viscosity (`cfg.fluidViscosity = "morris"`) and its no-slip wall closure (docs/plan-next-steps.md, "Morris viscosity instead of the alpha form").

(a) operator = lattice symbol: on a periodic square lattice at uniform density, a shear wave v = A sin(k.x) e (e perpendicular to k) gives a_i = M(k) v_i with the lattice sum M(k) = sum_r w(r) (1 - cos k.r) of the pair weight
    (alpha form: fac W'/r^3 (r.e)^2; Morris: 2 nu W' r / (r^2 + eta^2), nu = alpha c0 H / (8 xi) / morrisCalibration), for a wave along the lattice axis and along the diagonal, Warp modules and the torch oracle,
    to 1e-9 of max|a|;  the Morris symbol is isotropic to 1e-3 at this |k| and the alpha symbol is not (> 3 % at H = 4 dx): a negative control that the two forms are distinct;
(b) the Morris moment tables of the solid (planar, convex, concave) equal a brute-force 2D integral over the actual solid to 2e-3 of the largest entry;
(c) the alpha-specific wall closures refuse the Morris operator.
Float64 contracts.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.sim.cases import hydrostatic_tank
from warpSPHBoundaries.sim.deltasph2d import KERNELS, DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.wallmoments import MORRIS_ETA2, curved_moments

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
W2 = KERNELS[KernelFunctions.Wendland2]


def symbol(form, k, e, H, dx, fac, cal):
    """lattice sum of the pair weight: the rate matrix applied to the polarisation e (a scalar for a shear wave)."""
    _, dW, _, _ = W2
    m = int(math.ceil(H / dx)) + 1
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    r = np.stack([i.ravel(), j.ravel()], 1) * dx
    d = np.linalg.norm(r, axis=1)
    keep = (d > 1e-12) & (d < H)
    r, d = r[keep], d[keep]
    wp_ = dW(torch.tensor(d), H).numpy()
    c = 1 - np.cos(r @ k)
    V = dx * dx
    if form == "alpha":
        return float(np.sum(fac * V * wp_ / d ** 3 * (r @ e) ** 2 * c))
    nu = fac / 8.0 / cal
    return float(np.sum(2 * nu * V * wp_ * d / (d * d + MORRIS_ETA2 * H * H) * c))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", ["alpha", "morris"])
@pytest.mark.parametrize("warp", [True, False])
def test_shear_wave_response_is_the_lattice_symbol(device, form, warp):
    n = 24
    dx = 1.0 / n
    H = 4 * dx
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(n) + .5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    cal = 0.985
    for kv in ((1, 0), (1, 1)):
        k = 2 * math.pi * np.asarray(kv, float)
        e = np.array([-kv[1], kv[0]], float) / math.hypot(*kv)
        vel = 0.1 * np.sin(pos @ k)[:, None] * e[None]
        cfg = DeltaSPHConfig(gravity=(0, 0), c0=10.0, alpha=0.5, periodic=Periodic((0, 0), (1, 1)), graphStep=False, shifting=False, fluidViscosity=form, morrisCalibration=cal, fluidWarp=warp)
        sim = DeltaSPH2D(pos, vel, np.ones(len(pos)), dx, None, cfg, device, support=H)
        acc, _, _ = sim.rhs(sim.x, sim.v, sim.rho)
        fac = cfg.alpha * cfg.c0 * H / sim.xi
        M = symbol(form, k, e, H, dx, fac, cal)
        pred = M * sim.v
        assert float((acc - pred).abs().max()) <= 1e-9 * float(pred.abs().max())


@pytest.mark.parametrize("form", ["alpha", "morris"])
def test_isotropy_of_the_pair_weights(form):
    """the same |k| along the lattice axis and the diagonal: the Morris symbol is isotropic (< 1e-3), the alpha symbol is not (> 3 % at H = 4 dx)."""
    dx, H, kdx = 1.0 / 48, 4.0 / 48, 0.3
    kk = kdx / dx
    ax = symbol(form, kk * np.array([1.0, 0.0]), np.array([0.0, 1.0]), H, dx, 1.0, 1.0)
    dg = symbol(form, kk * np.array([1.0, 1.0]) / math.sqrt(2), np.array([-1.0, 1.0]) / math.sqrt(2), H, dx, 1.0, 1.0)
    aniso = dg / ax - 1
    if form == "morris":
        assert abs(aniso) < 1e-3
    else:
        assert aniso > 0.03


def _brute(dW, d, kw, H=1.0, n=1600):
    g = (np.arange(n) + 0.5) / n * 2 * H - H
    X, Y = np.meshgrid(g, g, indexing="ij")
    A = (2 * H / n) ** 2
    r = np.hypot(X, Y)
    m = (r < H) & (r > 1e-9)
    if kw == 0:
        solid, sp, nx, ty = X < -d, X + d, np.ones_like(X), np.ones_like(X)
    elif kw > 0:
        R = 1 / kw
        cx = -(R + d)
        rho = np.hypot(X - cx, Y)
        solid, sp, nx, ty = rho < R, rho - R, (X - cx) / rho, (X - cx) / rho
    else:
        R = -1 / kw
        cx = R - d
        rho = np.hypot(X - cx, Y)
        solid, sp, nx, ty = rho > R, R - rho, -(X - cx) / rho, -(X - cx) / rho
    K = np.zeros_like(X)
    K[m] = dW(torch.tensor(r[m]), H).numpy() * r[m] / (r[m] ** 2 + MORRIS_ETA2 * H * H)
    w = K * solid * m * A
    return np.array([[w.sum(), w.sum()]] + [[(w * nx * sp ** k).sum(), (w * ty * sp ** k).sum()] for k in (1, 2)] + [[(w * X).sum(), 0.0]])     # slot 3: M1_n = int_S K (y . n)


@pytest.mark.parametrize("d,kw", [(0.1, 0.0), (0.35, 0.0), (0.1, 0.6), (0.3, -0.6)])
def test_morris_moment_tables_match_the_solid_integral(d, kw):
    _, dW, _, _ = W2
    t = curved_moments(dW, 1.0, torch.tensor([d], dtype=torch.float64), torch.tensor([kw], dtype=torch.float64), weight="morris")[0].numpy()
    b = _brute(dW, d, kw)
    assert np.abs(t - b).max() <= 2e-3 * np.abs(b).max()


@pytest.mark.parametrize("device", DEVICES)
def test_alpha_wall_closures_refuse_morris(device):
    sim, _ = hydrostatic_tank(dp=0.08, domain="surface", device=device, cfg=DeltaSPHConfig(fluidViscosity="morris", wallViscosityForm="laplacian", graphStep=False))
    with pytest.raises(NotImplementedError):
        sim.rhs(sim.x, sim.v, sim.rho, want_forces=True)
    sim.cfg.fluidViscosity = "laplace"
    with pytest.raises(ValueError):
        sim.rhs(sim.x, sim.v, sim.rho)


def test_signed_solid_area_for_the_torque_correction():
    """the Morris torque correction -2 mu Omega A needs the signed area enclosed by the solid's boundary (solid on the left): + for obstacles, - for cavities / domains."""
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import Body, BoxRep, DiskArrayRep, ImplicitRep, SurfaceRep
    sim = DeltaSPH2D.__new__(DeltaSPH2D)
    cases = [(DiskArrayRep([(0.0, 0.0), (1.0, 0.0)], [0.2, 0.1]), math.pi * (0.04 + 0.01)),
             (SurfaceRep.polygon([(-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)]), 0.5),
             (SurfaceRep.polygon([(-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)], solid="outside"), -0.5),
             (BoxRep((0.0, 0.0), (2.0, 1.0)), 2.0), (BoxRep((0.0, 0.0), (2.0, 1.0), solid="outside"), -2.0),
             (ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.3)), math.pi * 0.09), (ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.3, solid="outside")), -math.pi * 0.09)]
    for rep, A in cases:
        b = Body(bodyId=0, center=(0.0, 0.0), reps=[rep])
        assert abs(sim._solid_area(b) - A) < 1e-12 * max(1.0, abs(A))
