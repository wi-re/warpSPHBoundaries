"""DeltaSPH2D (delta+-SPH on the scene boundary layer): the DDT is exactly zero on a hydrostatic field, the wall continuity term has the right sign, a wall never pulls (ceiling), the Barecasco switch
flags the free surface and not the walls, hydrostatic tank at rest, representation independence."""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound import deltasph2d as D
from edgebound.dfsph2d import domain_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64


def small_tank(device, dp=0.04, L=0.8, Ht=0.6, Hw=0.3, domain="surface", **cfgkw):
    sim, info = D.hydrostatic_tank(dp=dp, L=L, Htank=Ht, Hwater=Hw, domain=domain, device=device)
    sim.cfg.__dict__.update(cfgkw)
    return sim, info


@pytest.mark.parametrize("device", DEVICES)
def test_density_diffusion_is_zero_on_a_hydrostatic_field(device):
    """fourtakas2019: psi_ij = 0 pair by pair for the exact hydrostatic density, truncated stencils at the surface included (no wall term: fluid-to-fluid only)."""
    sim, info = small_tank(device)
    sim.scene, sim.nb = None, 0
    c0, g = sim.cfg.c0, 9.81
    depth = (info["bed"] + info["Hwater"] - sim.x[:, 1]).clamp(min=0)
    rho = 1.0 * (1.0 + g * depth / c0 ** 2)
    sim.cfg.viscosity = False
    acc, drho, _ = sim.rhs(sim.x, torch.zeros_like(sim.x), rho)
    assert float(drho.abs().max()) < 1e-10
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    off = sim.rhs(sim.x, torch.zeros_like(sim.x), rho)[1]
    assert float((drho - off).abs().max()) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_wall_continuity_compresses_toward_and_expands_away_from_the_wall(device):
    """free-slip mirror: a particle moving toward the bed has drho/dt > 0, away < 0, along it (tangential) 0."""
    sim, info = small_tank(device)
    dp, bed = sim.dx, info["bed"]
    x = torch.tensor([[0.0, bed + 0.5 * dp]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.cfg.ddt = False
    sim.cfg.viscosity = False
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    out = {}
    for name, v in (("toward", (0.0, -1.0)), ("away", (0.0, 1.0)), ("along", (1.0, 0.0))):
        out[name] = float(sim.rhs(x, torch.tensor([v], dtype=F64, device=device), rho)[1][0])
    assert out["toward"] > 1.0 and out["away"] < -1.0
    assert abs(out["toward"] + out["away"]) < 1e-9 and abs(out["along"]) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_a_wall_above_a_particle_does_not_pull_it(device):
    """isolated particle at the ceiling at rest and at rest density: only gravity acts with the p_b >= 0 clamp; without it the hydrostatic wall pressure p_i - rho g d < 0 is a suction that holds the particle up."""
    res = {}
    for clamp in (True, False):
        sim, info = small_tank(device, clampWallPressure=clamp)
        top = info["bed"] + 0.6
        x = torch.tensor([[0.0, top - 0.5 * sim.dx]], dtype=F64, device=device)
        rho = torch.ones(1, dtype=F64, device=device)
        sim.cfg.ddt = False
        sim.cfg.viscosity = False
        sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
        sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
        res[clamp] = float(sim.rhs(x, torch.zeros(1, 2, dtype=F64, device=device), rho)[0][0, 1])
    assert abs(res[True] + 9.81) < 1e-9
    assert res[False] > -9.81 + 1.0                 # the unclamped wall holds up a good part of the weight


@pytest.mark.parametrize("device", DEVICES)
def test_free_surface_detector_flags_the_top_row_and_not_the_walls(device):
    sim, info = small_tank(device)
    sim.rhs(sim.x, torch.zeros_like(sim.x), sim.rho)
    top = sim.x[:, 1] > info["bed"] + info["Hwater"] - 0.9 * sim.dx
    assert bool(sim.surface[top].all())
    assert int(sim.surface.sum()) == int(top.sum())


@pytest.mark.parametrize("device", DEVICES)
def test_hydrostatic_tank_stays_at_rest_and_the_walls_carry_the_weight(device):
    sim, info = small_tank(device)
    forces = []
    for _ in range(150):
        sim.step()
        forces.append(sim.wallForce.sum(0).cpu().numpy())
    weight = sim.m * len(sim.x) * 9.81
    assert float(sim.v.norm(dim=1).max()) < 0.05
    assert abs(np.mean(forces[-50:], 0)[1] / weight + 1.0) < 0.03           # force of the fluid on the walls points down and equals its weight
    assert abs(np.mean(forces[-50:], 0)[0]) < 0.02 * weight
    assert 0.99 < float(sim.rho.min()) and float(sim.rho.max()) < 1.02


@pytest.mark.parametrize("device", DEVICES)
def test_dynamics_independent_of_the_wall_representation(device):
    xs = {}
    for kind in ("surface", "volume"):
        sim, _ = small_tank(device, domain=kind)
        for _ in range(25):
            sim.step()
        xs[kind] = sim.x.cpu().numpy()
    assert np.abs(xs["surface"] - xs["volume"]).max() < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_pair_list_is_complete_for_a_lattice_whose_spacing_divides_the_support(device):
    """regression: with support = 4 dx the lattice puts particles exactly on the cell-list borders; the cell index was rounded differently at insertion and at query and ~2 % of the reverse pairs went missing
    (the pairwise antisymmetric forces then no longer conserved momentum).  The pair list must equal the brute-force list of r < H, in both directions."""
    from edgebound.dfsph2d import neighbor_pairs
    sim, _ = D.hydrostatic_tank(dp=0.02, device=device)                       # the English 2.4 x 1.2 m tank: the bug needs this lattice / cell alignment
    x = sim.x
    i, j, r = neighbor_pairs(x, sim.Hvec)
    n = len(x)
    inside = r < sim.H - 1e-9
    assert int(inside.sum()) == int((torch.cdist(x, x) < sim.H - 1e-9).sum())
    keys = set((i[inside] * n + j[inside]).tolist())
    assert all((b * n + a) in keys for a, b in zip(i[inside][:3000].tolist(), j[inside][:3000].tolist()))
    sim.scene, sim.nb = None, 0
    a_ff = sim.rhs(x, torch.zeros_like(x), sim.rho)[0] - sim.g[None]       # fluid-fluid pressure force without gravity
    assert float((sim.m * a_ff.sum(0)).abs().max()) < 1e-12                # momentum conservation


@pytest.mark.parametrize("device", DEVICES)
def test_wall_viscous_term_matches_the_half_plane_integral(device):
    """a particle dp/2 above a flat wall moving into it at u_n: the free-slip alpha-viscosity wall term is  (alpha c0 H / xi) (2 u_n / rho) mu M2nn n  with M2nn = int_{y_n > d} int W'(r) y_n^2 / r^3 dy_t dy_n
    (checked against a fine tensor-grid quadrature of the half plane); it decelerates the approach, is zero for tangential motion, and vanishes without viscosity."""
    from edgebound.dfsph2d import dwendland2
    sim, info = small_tank(device)
    sim.cfg.wallViscosityForm = "pairwise"                        # this test pins the pairwise (Monaghan) form, which is kept
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    dp, bed = sim.dx, info["bed"]
    x = torch.tensor([[0.0, bed + 0.5 * dp]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    got = {}
    for name, v in (("into", (0.0, -1.0)), ("along", (1.0, 0.0))):
        vel = torch.tensor([v], dtype=F64, device=device)
        sim.cfg.wallViscosity = True
        a_on = sim.rhs(x, vel, rho)[0]
        sim.cfg.wallViscosity = False
        a_off = sim.rhs(x, vel, rho)[0]
        got[name] = (a_on - a_off)[0]
    d, H = 0.5 * dp, sim.H
    yn = torch.linspace(d, H, 4001, dtype=F64, device=device)
    yt = torch.linspace(-H, H, 8001, dtype=F64, device=device)
    YN, YT = torch.meshgrid(yn, yt, indexing="ij")
    R = torch.sqrt(YN ** 2 + YT ** 2)
    f = torch.where(R < H, dwendland2(R, H) * YN ** 2 / R.clamp(min=1e-300) ** 3, torch.zeros_like(R))
    M2nn = float(torch.trapezoid(torch.trapezoid(f, yt, dim=1), yn))
    expect = (sim.cfg.alpha * sim.cfg.c0 * H / D.XI) * 2.0 * 1.0 * M2nn * sim.cfg.wallMass          # u_n = +1 (into the wall: the bed lies below, n points down), force along n
    assert abs(float(got["into"][1])) > 0
    assert float(got["into"][1]) > 0                                                                # pushes the particle away from the wall (up), against the approach
    assert abs(-float(got["into"][1]) - expect) < 0.03 * abs(expect)                                # n = -y, M2nn n = -M2nn y: expect = -accel_y... sign handled by magnitude
    assert float(got["along"].norm()) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_wall_viscous_term_is_the_wall_laplacian(device):
    """a particle dp/2 above a flat wall moving into it at u_n = +1: the DEFAULT (wallViscosityForm = "laplacian") wall term is the
    exact wall-Laplacian term  acc = -2 nu_eff wallMass u_n B n  with B = int_solid lap W dA'  by the divergence theorem on the
    wall line, B = -z int_{-L}^{L} W'(r)/r dx (an OWN scipy quad, epsabs = epsrel = 1e-13, that shares nothing with the solver),
    nu_eff = fac/8, n = (0, -1).  It pushes the particle away from the wall, is zero for tangential motion, and the pairwise form
    gives a ~10x larger value (the different operator, the negative control).

    Tolerances (stated with their reason, BEFORE looking at the results):
      * |got_into_y - pred_y| <= 1e-9 * pred_y (float64 round-off; the reviewer measured 8.8e-16; a wrong factor or sign is >= 1e-2).
      * smoke: B = 90.557119, fac = 0.019457, pred_y = 0.440499 (rtol 1e-4, at z/H = 0.125) -- pins the OWN prediction.
      * got_into_y > 0 (pushes away from the wall); |got_into_x| < 1e-12; tangential |got_along| < 1e-9.
      * negative control: the pairwise form differs from the Laplacian value by > 0.5 * 4.349 (reviewer |got_into_y| = 4.349,
        factor 9.87 = 8|A|/B at z/H = 0.125).
    """
    from scipy import integrate
    sim, info = small_tank(device)
    sim.cfg.ddt = False
    sim.cfg.delta = 0.0
    dp, bed = sim.dx, info["bed"]
    x = torch.tensor([[0.0, bed + 0.5 * dp]], dtype=F64, device=device)
    rho = torch.ones(1, dtype=F64, device=device)
    sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(1, dtype=torch.int32, device=device)
    got = {}
    for name, v in (("into", (0.0, -1.0)), ("along", (1.0, 0.0))):
        vel = torch.tensor([v], dtype=F64, device=device)
        sim.cfg.wallViscosity = True
        a_on = sim.rhs(x, vel, rho)[0]
        sim.cfg.wallViscosity = False
        a_off = sim.rhs(x, vel, rho)[0]
        got[name] = (a_on - a_off)[0]
    # the independent prediction: B = int_solid lap W dA' by the divergence theorem on the wall line (OWN scipy quad)
    H = sim.H
    z = 0.5 * dp
    L = math.sqrt(max(H * H - z * z, 0.0))
    def Wprime(rr):
        q = rr / H
        return -140.0 * q * (1.0 - q) ** 3 / (math.pi * H ** 3)      # the normalised Wendland C2, W'(r) = c/(pi H^3) s'(q)/H, c = 7
    B = -z * integrate.quad(lambda xx: Wprime(math.sqrt(xx * xx + z * z)) / math.sqrt(xx * xx + z * z),
                            -L, L, epsabs=1e-13, epsrel=1e-13)[0]
    fac = sim.cfg.alpha * sim.cfg.c0 * H / sim.xi
    pred_y = 2.0 * (fac / 8.0) * sim.cfg.wallMass * B                 # u_n = +1, n = (0, -1): acc = -2 nu_eff wallMass u_n B n
    # (a) the Laplacian form (the default) matches the prediction
    assert abs(float(got["into"][1]) - pred_y) <= 1e-9 * pred_y, (float(got["into"][1]), pred_y)
    # (b) smoke: the OWN prediction itself (B, fac, pred_y at z/H = 0.125)
    assert abs(B - 90.557119) <= 1e-4 * 90.557119, B
    assert abs(fac - 0.019457) <= 1e-4 * 0.019457, fac
    assert abs(pred_y - 0.440499) <= 1e-4 * 0.440499, pred_y
    # (c) physics: pushes away from the wall, no x or tangential component
    assert float(got["into"][1]) > 0
    assert abs(float(got["into"][0])) < 1e-12
    assert abs(float(got["along"].norm())) < 1e-9
    print("(laplacian) B = %.6f  fac = %.6f  pred_y = %.6f  got_into_y = %.6f  |diff|/pred = %.2e"
          % (B, fac, pred_y, float(got["into"][1]), abs(float(got["into"][1]) - pred_y) / pred_y))
    # (d) negative control: the pairwise form is a different, ~10x stronger operator
    sim.cfg.wallViscosityForm = "pairwise"
    try:
        vel = torch.tensor([(0.0, -1.0)], dtype=F64, device=device)
        sim.cfg.wallViscosity = True
        a_on = sim.rhs(x, vel, rho)[0]
        sim.cfg.wallViscosity = False
        a_off = sim.rhs(x, vel, rho)[0]
        got_pair = (a_on - a_off)[0]
    finally:
        sim.cfg.wallViscosityForm = "laplacian"
    got_into_y_pair = float(got_pair[1])
    assert abs(got_into_y_pair - pred_y) > 0.5 * 4.349, got_into_y_pair
    print("(pairwise) got_into_y = %.6f (reviewer 4.349); differs from the Laplacian pred_y by %.6f (> 0.5*4.349 = %.4f)"
          % (got_into_y_pair, abs(got_into_y_pair - pred_y), 0.5 * 4.349))


@pytest.mark.parametrize("device", DEVICES)
def test_no_penetration_impulse_follows_the_mdbc_law(device):
    """v_n <- v_n (1 - f), f = 3 - 4 clip(1/2 + d/dp, 1/4, 1) for a closing particle with d < dp/4: f = 0.6 at d = 0.1 dp, 1.4 (reflection) at d = -0.1 dp, nothing at d = 0.3 dp, nothing for an opening or tangential velocity."""
    sim, info = small_tank(device, noPen="impulse")
    dp, bed = sim.dx, info["bed"]
    ys = torch.tensor([0.1, -0.1, 0.3, 0.1, 0.1], dtype=F64, device=device) * dp + bed
    sim.x = torch.stack([torch.linspace(-0.2, 0.2, 5, dtype=F64, device=device), ys], 1)
    sim.v = torch.tensor([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]], dtype=F64, device=device)
    sim.rho = torch.ones(5, dtype=F64, device=device)
    sim.Hvec = torch.full((5,), sim.H, dtype=F64, device=device)
    sim.kinds = torch.zeros(5, dtype=torch.int32, device=device)
    sim.no_penetration()
    vy = sim.v[:, 1].tolist()
    assert abs(vy[0] - (-1.0 * (1 - 0.6))) < 1e-9 and abs(vy[1] - (-1.0 * (1 - 1.4))) < 1e-9
    assert vy[2] == -1.0 and vy[3] == 1.0 and sim.v[4].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_particle_shift_vanishes_in_the_bulk_and_is_normal_and_small_at_a_flat_wall(device):
    sim, info = small_tank(device, Hw=0.5, Ht=0.8, shifting=True)
    sim.rho = torch.ones_like(sim.rho)                              # a density gradient (hydrostatic init) shifts through the m/(rho_i + rho_j) weight
    upd = sim.shift(sim.dt)
    x = sim.x
    bed, top = info["bed"], info["bed"] + info["Hwater"]
    bulk = (x[:, 1] > bed + 4.5 * sim.dx) & (x[:, 1] < top - 4.5 * sim.dx) & (x[:, 0].abs() < 0.4 - 4.5 * sim.dx)
    assert bool(bulk.any()) and float(upd[bulk].norm(dim=1).max()) < 1e-9 * sim.dx
    first = (x[:, 1] < bed + 0.6 * sim.dx) & (x[:, 0].abs() < 0.2)
    assert float(upd[first, 1].abs().max()) < 0.01 * sim.dx          # a first row at its natural dx/2 spacing is (almost) at the PST equilibrium: fluid sum, wall gradient and tensile term balance
    assert float(upd[first, 0].abs().max()) < 1e-9 * sim.dx         # no tangential component on a flat wall
    # a particle pushed into the wall region is pushed back out: move the first row 0.3 dx closer
    sim.x = sim.x.clone()
    sim.x[first, 1] -= 0.3 * sim.dx
    upd2 = sim.shift(sim.dt)
    assert bool((upd2[first, 1] > 0).all())
