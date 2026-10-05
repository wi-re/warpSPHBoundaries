"""DFSPH2D on the scene boundary layer: operators vs warpSPHCore, lattice calibration, representation independence of the dynamics, agreement with the
compiled omniSPH reference (skipped when `omnySPH` is not importable)."""
import os
import sys

import numpy as np
import pytest
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation, DomainDescription
import warpSPHCore as core
from edgebound.sim import dfsph2d as D
from edgebound import paths

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64


def lattice(nx, ny, dx, dy, x0=0.0, y0=0.0):
    X, Y = np.meshgrid(x0 + dx * (np.arange(nx) + 0.5), y0 + dy * (np.arange(ny) + 0.5), indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


@pytest.mark.parametrize("device", DEVICES)
def test_pair_sums_match_warpoperation(device):
    """density and the symmetric pressure gradient of the torch pair sums equal `warpSPHCore.warpOperation` (same kernel, same support)."""
    rng = np.random.default_rng(0)
    dx = 0.01
    pos = lattice(14, 14, dx, dx) + rng.normal(0, 0.1 * dx, (196, 2))
    h = 2.5 * dx
    t = lambda a: torch.as_tensor(a, dtype=TD, device=device)
    n = len(pos)
    V = np.full(n, dx * dx)
    p = rng.uniform(0, 1, n)
    sim = D.DFSPH2D(pos, np.zeros_like(pos), V, np.full(n, h), None, D.DFSPHConfig(), device)
    sim._prepare()
    rho = sim._sum(sim.V[sim.pj] * sim.W)
    sim.rho = rho
    acc = sim._fluid_accel(t(p))
    # warpSPHCore reference
    from warpSPHCore.type_config import get_torch_precision
    t32 = lambda a: torch.as_tensor(a, dtype=get_torch_precision(), device=device)     # warpOperation computes in warpSPHCore's precision (float64 under edgebound's default, float32 otherwise)
    ps = ParticleState(positions=t32(pos), supports=t32(np.full(n, h)), masses=t32(V), kinds=torch.zeros(n, dtype=torch.int32, device=device), densities=rho.to(get_torch_precision()))
    lo, hi = pos.min(0) - 2 * h, pos.max(0) + 2 * h
    domain = DomainDescription(t32(lo), t32(hi), torch.zeros(2, dtype=torch.bool, device=device), 2)
    props_d = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density, operationMode=OperationDirection.AllToAll)
    rho_ref = core.warpOperation(ps, props_d, domain)
    np.testing.assert_allclose(rho.cpu().numpy(), rho_ref.double().cpu().numpy(), rtol=2e-5)
    props_g = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Symmetric,
                                  operationMode=OperationDirection.AllToAll)
    g = core.warpOperation(ps, props_g, domain, queryValues=t32(p), referenceValues=t32(p))
    # warp Symmetric: m_j rho_i (p_i/rho_i^2 + p_j/rho_j^2) grad W; our acceleration is -(1/rho_i) of it
    np.testing.assert_allclose(acc.cpu().numpy(), (-g / rho[:, None]).double().cpu().numpy(), rtol=1e-3, atol=3e-4 * float(acc.abs().max()))


@pytest.mark.parametrize("device", DEVICES)
def test_lattice_calibration_is_a_rest_state(device):
    dx, dy, h = 0.0089, 0.0088, 0.02236
    cal = D.lattice_calibration(dx, dy, h)
    pos = lattice(40, 12, dx, dy, x0=0.1, y0=0.1)
    lo = pos.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"], 0.6])
    for kind in ["surface", "volume", "sdf", "box"]:
        sc = D.domain_scene(kind, lo, hi, h, device)
        sim = D.DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, D.DFSPHConfig(wallMass=cal["mu"]), device)
        sim._prepare()
        rho = sim._sum(sim.V[sim.pj] * sim.W) + sim.lam
        x = sim.x
        bulk = (x[:, 0] > 0.1 + 6 * dx) & (x[:, 0] < 0.1 + 34 * dx) & (x[:, 1] > 0.1 + 4 * dy) & (x[:, 1] < 0.1 + 8 * dy)
        assert abs(float(rho[bulk].mean()) - 1) < 1e-9
        row0 = (x[:, 1] < 0.1 + dy) & (x[:, 0] > 0.1 + 6 * dx) & (x[:, 0] < 0.1 + 34 * dx)
        assert abs(float(rho[row0].mean()) - 1) < 3e-4


@pytest.mark.parametrize("device", DEVICES)
def test_dynamics_independent_of_the_representation(device):
    """30 steps of a settling block: volume slabs, surface loop, tier-3 SDF (+ corner fallback) and the BoxRep corner tables give the same trajectories (3e-13 / 3e-13 / 1e-5 / 2e-10)."""
    dx, dy, h = 0.0089, 0.0088, 0.02236
    cal = D.lattice_calibration(dx, dy, h)
    pos = lattice(30, 10, dx, dy, x0=0.1, y0=0.1)
    lo = pos.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"], 0.5])
    xs = {}
    for kind in ["surface", "volume", "sdf", "box"]:
        sc = D.domain_scene(kind, lo, hi, h, device)
        sim = D.DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, D.DFSPHConfig(wallMass=cal["mu"]), device)
        for _ in range(30):
            sim.step()
        xs[kind] = sim.x.cpu().numpy()
    assert np.abs(xs["volume"] - xs["surface"]).max() < 1e-10
    assert np.abs(xs["sdf"] - xs["surface"]).max() < 1e-4
    box_dev = np.abs(xs["box"] - xs["surface"]).max()
    print("box vs surface max|dx| =", box_dev)
    assert box_dev < 1e-8                                        # measured 1.9e-10
    assert np.abs(xs["surface"] - pos).max() > 1e-6          # something actually happened


def test_agrees_with_omnisph_reference():
    sys.path.insert(0, str(paths.OMNISPH_HOME / "omnySPH" / "src"))
    try:
        import omnySPH  # noqa: F401
    except Exception:
        pytest.skip("omnySPH not importable")
    scratch = str(paths.tmp_dir() / "omni")
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "cfg")
    if not os.path.exists(link):
        os.symlink(str(paths.OMNISPH_HOME / "cfg"), link)
    old = os.getcwd()
    os.chdir(scratch)
    try:
        from edgebound.sim import dfsph_ref as R
        r = 0.005
        fmin, fmax = (0.02, 0.02), (0.5, 0.15)
        c = R.omni_case(r, fmin, fmax, top=0.4)
        sp = np.array([c["dx"], c["dy"]])
        lo = np.array(fmin) - sp
        hi = np.array([fmax[0], 0.4]) + np.array([sp[0], 0])
        import omnySPH as O
        so = O.SPHSimulation(R._yaml(r, fmin, fmax, lo - c["epsAdj"], hi + c["epsAdj"], 0.06))
        dev = DEVICES[0]
        sc = D.domain_scene("volume", lo, hi, float(c["h"][0]), dev)
        sim = D.DFSPH2D(c["x"], np.zeros_like(c["x"]), c["V"], c["h"], sc, D.DFSPHConfig(), dev)
        for _ in range(120):
            so.timestep()
            sim.step()
        st = R.omni_state(so, c["n"])
        xo, xs = st["x"], sim.x.cpu().numpy()
        # a settling block: bulk statistics must agree closely (particle-by-particle agreement is not expected: different wall closures)
        assert abs(xo[:, 1].mean() - xs[:, 1].mean()) < 1e-3
        assert abs(xo[:, 0].mean() - xs[:, 0].mean()) < 1e-3
        rms = lambda v: float(np.sqrt((np.asarray(v) ** 2).sum(1).mean()))
        assert abs(rms(st["v"]) / rms(sim.v.cpu().numpy()) - 1) < 0.35
        assert np.sqrt(((xo - xs) ** 2).sum(1).mean()) < 3e-3
    finally:
        os.chdir(old)


# ----------------------------------------------------------------------------------------------------------------------------- force tracking
from edgebound.sim import dfsph_cases as C
from edgebound.scene.scene import BodyField, sceneOperation, Body, SurfaceRep, Scene


@pytest.mark.parametrize("device", DEVICES)
def test_momentum_bookkeeping_with_a_rotating_obstacle(device):
    """sum m dv = dt (m g - sum F_pressure - sum F_friction) to round-off, every step: the tracked forces are exactly the momentum exchanged with the bodies."""
    sim, _ = C.tank_with_obstacle(L=0.3, H=0.3, fill=0.15, Rh=0.04, omega=4.0, r=0.006, device=device)
    for _ in range(25):
        sim.step()
    assert max(h["balance"] for h in sim.history) < 1e-14
    fo = np.array([h["pressure"][1] for h in sim.history])
    assert np.abs(fo).max() > 1e-4                       # the obstacle really feels the fluid


@pytest.mark.parametrize("device", DEVICES)
def test_archimedes_force_on_a_submerged_body(device):
    """a fixed submerged hexagon: the time-averaged vertical force is the buoyancy rho g A, the forces on all bodies add up to minus the weight."""
    sim, info = C.tank_with_obstacle(L=0.4, H=0.35, fill=0.2, Rh=0.04, omega=0.0, r=0.005, device=device)
    for _ in range(320):
        sim.step()
    t = np.array([h["t"] for h in sim.history])
    F = np.array([h["pressure"] + h["friction"] for h in sim.history])      # [T,2 bodies,2]
    m = t > 0.15
    fo, fd = F[m, 1].mean(0), F[m, 0].mean(0)
    assert abs(fo[1] / info["buoyancy"] - 1) < 0.06
    assert abs(fo[0]) < 0.1 * info["buoyancy"]
    assert abs((fo + fd)[1] / (-info["weight"]) - 1) < 5e-3


@pytest.mark.parametrize("device", DEVICES)
def test_surface_and_volume_obstacles_give_the_same_forces(device):
    sims = [C.tank_with_obstacle(L=0.3, H=0.3, fill=0.15, Rh=0.04, omega=3.0, rep=rep, r=0.006, device=device)[0] for rep in ("surface", "volume")]
    for _ in range(20):
        for s in sims:
            s.step()
    fs = np.array([h["pressure"] for h in sims[0].history])
    fv = np.array([h["pressure"] for h in sims[1].history])
    np.testing.assert_allclose(fs, fv, atol=1e-9, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_per_body_operation_sums_to_the_total(device):
    sc = Scene([Body(bodyId=0, center=(0.0, 0.0), reps=[SurfaceRep.box((-0.3, -0.1), (0.3, 0.0))]),
                Body(bodyId=1, center=(0.0, 0.25), angle=0.3, reps=[C.hexagon(0.05)])], device)
    rng = np.random.default_rng(2)
    P = rng.uniform(-0.4, 0.4, (80, 2))
    ps = ParticleState(positions=torch.as_tensor(P, dtype=TD, device=device), supports=torch.full((80,), 0.05, dtype=TD, device=device),
                       masses=torch.ones(80, dtype=TD, device=device), kinds=torch.zeros(80, dtype=torch.int32, device=device), densities=torch.ones(80, dtype=TD, device=device))
    pr = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, operationMode=OperationDirection.BoundaryToFluid)
    f = [BodyField(torch.tensor(1.3, dtype=TD, device=device)), BodyField(torch.tensor(0.7, dtype=TD, device=device))]
    tot = sceneOperation(ps, pr, sc, bodyFields=f)
    per = sceneOperation(ps, pr, sc, bodyFields=f, perBody=True)
    assert per.shape == (2, 80, 2)
    np.testing.assert_allclose(per.sum(0).cpu().numpy(), tot.cpu().numpy(), atol=1e-13)


# ----------------------------------------------------------------------------------------------------------------------------- wall closure
def _tank_sim(device, **cfgkw):
    dx, dy, h = 0.0089, 0.0088, 0.02236
    cal = D.lattice_calibration(dx, dy, h)
    pos = lattice(30, 14, dx, dy, x0=0.1, y0=0.1)
    lo = pos.min(0) - np.array([cal["dwallX"], cal["dwallY"]])
    hi = np.array([pos[:, 0].max() + cal["dwallX"], 0.5])
    sc = D.domain_scene("surface", lo, hi, h, device)
    sim = D.DFSPH2D(pos, np.zeros_like(pos), cal["V"], np.full(len(pos), h), sc, D.DFSPHConfig(wallMass=cal["mu"], **cfgkw), device)
    return sim, pos, dy


@pytest.mark.parametrize("device", DEVICES)
def test_wall_closure_removes_the_static_wall_residual(device):
    """exact hydrostatic pressure on the calibrated lattice: the residual acceleration of the wall rows drops by more than an order of magnitude with the closure,
    the interior (no wall contact) is untouched."""
    res = {}
    for gc in ("none", "wall"):
        sim, pos, dy = _tank_sim(device, gradientCorrection=gc)
        sim._prepare()
        sim.rho = sim._sum(sim.V[sim.pj] * sim.W) + sim.lam
        x = sim.x
        ytop = float(x[:, 1].max()) + dy / 2
        p = 9.81 * (ytop - x[:, 1])
        acc = torch.tensor([0.0, -9.81], dtype=TD, device=device) + sim._fluid_accel(p) + sim._boundary_accel(p, True)
        r = acc.norm(dim=1)
        res[gc] = (float(r[x[:, 1] < 0.1 + 0.6 * dy].mean()), float(r[(x[:, 1] > 0.1 + 6 * dy) & (x[:, 0] > 0.2) & (x[:, 0] < 0.3)].mean()))
    assert res["none"][0] > 8.0 and res["wall"][0] < 1.0
    assert abs(res["wall"][1] - res["none"][1]) < 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_closure_keeps_the_force_bookkeeping_and_tight_solves_with_a_moving_body(device):
    cfg = D.DFSPHConfig(gradientCorrection="wall", densityEta=1e-5, divergenceEta=1e-5, divergenceMaxIterations=60, maxIterations=500, divergenceClamp=True)
    sim, info = C.tank_with_obstacle(L=0.3, H=0.3, fill=0.15, Rh=0.04, omega=4.0, r=0.006, device=device, cfg=None)
    sim.cfg.__dict__.update(dict(cfg.__dict__, wallMass=info["cal"]["mu"]))
    for _ in range(60):
        sim.step()
    assert max(h["balance"] for h in sim.history) < 1e-13
    assert torch.isfinite(sim.x).all() and float(sim.v.norm(dim=1).max()) < 5.0


@pytest.mark.parametrize("device", DEVICES)
def test_a_particle_leaving_a_wall_is_not_pulled_back_when_a_body_moves(device):
    """a moving body puts the wall into the divergence solve; with an unclamped divergence pressure a particle that leaves the ceiling feels the wall term as suction and is stopped (fluid sticks to every wall).
    Isolated particle at the ceiling moving away at 1 m/s, no gravity, a rotating hexagon elsewhere in the box: the default (clamped) keeps the velocity, the unclamped solve cancels it."""
    r = 0.006
    h = r * np.sqrt(20.0)
    dx = D.PACKING * h
    cal = D.lattice_calibration(dx, dx, h)
    H = 0.3
    out = {}
    for clamp in (None, False):
        dom = D.domain_scene("surface", (0.0, 0.0), (0.4, H), h, device).bodies[0]
        hexa = Body(bodyId=1, center=(0.3, 0.1), angularVelocity=3.0, reps=[C.hexagon(0.04, "surface")])
        sim = D.DFSPH2D(np.array([[0.1, H - 0.5 * dx]]), np.array([[0.0, -1.0]]), cal["V"], np.array([h]), Scene([dom, hexa], device),
                        D.DFSPHConfig(gravity=(0.0, 0.0), wallMass=cal["mu"], wallDivergenceClamp=clamp), device)
        for _ in range(30):
            sim.step()
        out[clamp] = float(sim.v[0, 1])
    assert out[None] < -0.99                    # keeps leaving the wall
    assert out[False] > -0.1                    # the old behaviour: stopped at the wall


@pytest.mark.parametrize("device", DEVICES)
def test_divergence_pressure_keeps_its_sign_between_fluid_particles(device):
    """the negative divergence pressure is the cohesion of omniSPH's solver (a stretching blob is pulled back together).  With a moving body (wall in the divergence solve) the default must keep it:
    same damping of the stretching as the wall-blind solve; clamping the divergence pressure of all particles (`divergenceClamp`) removes it and the fluid expands (mean density 0.7 after a splash)."""
    r = 0.006
    h = r * np.sqrt(20.0)
    dx = D.PACKING * h
    cal = D.lattice_calibration(dx, dx, h)
    pos = lattice(12, 12, dx, dx, 0.2, 0.1)
    c = pos.mean(0)
    vr = {}
    for name, kw in (("default", {}), ("global", dict(divergenceClamp=True)), ("blind", dict(boundaryInDivergence=False))):
        dom = D.domain_scene("surface", (0, 0), (0.6, 0.4), h, device).bodies[0]
        hexa = Body(bodyId=1, center=(0.45, 0.2), angularVelocity=3.0, reps=[C.hexagon(0.04, "surface")])
        sim = D.DFSPH2D(pos, 0.5 * (pos - c), cal["V"], np.full(len(pos), h), Scene([dom, hexa], device), D.DFSPHConfig(gravity=(0, 0), wallMass=cal["mu"], **kw), device)
        for _ in range(40):
            sim.step()
        d = sim.x - sim.x.mean(0)
        vr[name] = float((((sim.v - sim.v.mean(0)) * d).sum(1) / d.norm(dim=1).clamp(min=1e-9)).mean())
    assert abs(vr["default"] - vr["blind"]) < 0.1 * vr["blind"] + 1e-4
    assert vr["global"] > 5 * vr["default"]
