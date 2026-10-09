"""The wall terms of the analytic-boundary flavour (modules/analyticBoundary/wallTerms.py) against the reference
implementation in warpSPHBoundaries (`DeltaSPH2D.rhs`): the wall contribution to d rho / dt and to the acceleration.

Reference: `DeltaSPH2D.rhs(x, v, rho)` with the tank minus the same call without any wall (the fluid-fluid terms
cancel). The state has P >= 0 everywhere (rho >= rho0), so the Antuono switch is +1 for every particle whatever the
surface detection says. Compared: continuity (free-slip mirror), the pressure force (with the p_b >= 0 clamp) and the
viscous wall term, for a free-slip tank and a no-slip tank; the sum of the three equals the difference of the
reference right-hand sides.
Lives in warpSPHBoundaries, not in warpSPH: it compares the analytic-boundary code of warpSPH (patches/warpsph/, `warpSPH.modules.analyticBoundary`) with this
package's reference solver `DeltaSPH2D`, and warpSPH does not import the package's `sim` layer.  Skipped when the installed warpSPH has no analytic-boundary code (warpSPH >= 0.6.0 has it; for an older tree put a patched `src` on PYTHONPATH, patches/warpsph/).
Run in both precisions (the tolerances follow the state's dtype):  `pytest tests/warpsph`  and  `warpSPHCore_PRECISION=float32 pytest tests/warpsph`.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPH.modules.analyticBoundary')

from warpSPHCore import KernelFunctions  # noqa: E402
from warpSPHCore.type_config import get_torch_precision  # noqa: E402

from warpSPH.boundary import buildBoundaryProvider  # noqa: E402
from warpSPH.configurations.region import BCType, RegionType  # noqa: E402
from warpSPH.modules.analyticBoundary import (evaluateWall, wallContinuity, wallLoads, wallPressureAcceleration, wallViscousAcceleration)  # noqa: E402
from warpSPH.rigidBody import buildAnalyticRigidBody  # noqa: E402
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState  # noqa: E402
from warpSPHBoundaries.scene import Body  # noqa: E402
from warpSPHBoundaries.scene.scene import BoxRep  # noqa: E402
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig  # noqa: E402
from warpSPHBoundaries.scene.scene import Scene  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
DEV = 'cuda'
F64 = torch.float64
LO, HI = (0.0, 0.0), (0.625, 0.5)                    # dyadic: the lattice, the support 4 dx and the walls are exact in float32
G = (0.0, -9.81)
ALPHA, C0 = 0.1, 20.0


def block(dx=1.0 / 32):
    nx, ny = 20, 10
    X, Y = np.meshgrid(dx * (np.arange(nx) + 0.5), dx * (np.arange(ny) + 0.5), indexing='ij')
    return np.stack([X.ravel(), Y.ravel()], 1), dx


def state_of(pos, v, rho, H):
    """the particle state in warpSPHCore's precision (the Warp neighbour search follows it; float32 unless configured)."""
    n = len(pos)
    tp = get_torch_precision()
    z = torch.zeros(n, device=DEV, dtype=tp)
    t = lambda a: torch.as_tensor(a, device=DEV, dtype=tp)
    return WeaklyCompressibleState(
        positions=t(pos), velocities=t(v), supports=z + H, masses=z + (1.0 / 32) ** 2, densities=t(rho),
        kinds=torch.zeros(n, dtype=torch.int32, device=DEV), materials=torch.zeros(n, dtype=torch.int32, device=DEV),
        UIDs=torch.arange(n, device=DEV), UIDcounter=n, pressures=C0 ** 2 * (t(rho) - 1.0), soundspeeds=z + C0,
        ghostIndices=-torch.ones(n, dtype=torch.int32, device=DEV), ghostOffsets=torch.zeros((n, 2), device=DEV, dtype=tp))


def setup(kind, jitter=0.0):
    pos, dx = block()
    H = 4 * dx
    rng = np.random.default_rng(3)
    if jitter:
        pos = pos + jitter * dx * rng.uniform(-1.0, 1.0, size=pos.shape)
    v = rng.normal(scale=0.3, size=pos.shape)
    rho = 1.0 + 0.002 * (1.0 + 0.5 * np.sin(8 * pos[:, 0]) * np.cos(5 * pos[:, 1])) + 9.81 * np.clip(0.3 - pos[:, 1], 0, None) / C0 ** 2
    st = state_of(pos, v, rho, H)
    pos, v, rho = (a.double().cpu().numpy() for a in (st.positions, st.velocities, st.densities))        # the reference sees exactly the rounded inputs
    body = Body(bodyId=0, reps=[BoxRep(LO, HI, solid='outside')])
    region = SimpleNamespace(type=RegionType.Boundary, representation=body, sdf=None, kind=kind)
    rb = buildAnalyticRigidBody(region, 0, st)
    provider = buildBoundaryProvider([region], DEV)
    provider.rigidBodies = [rb]
    config = SimpleNamespace(kernel=KernelFunctions.Wendland2)
    schemeConfig = SimpleNamespace(analyticWallMass=1.0, fluid=SimpleNamespace(restDensity=1.0))
    wall = evaluateWall(provider, st, config, schemeConfig, torch.tensor(G, device=DEV, dtype=F64))
    wall.provider = provider                               # for the tests that need to run the provider again
    return pos, v, rho, H, dx, st, wall


def reference_state_dict(pos, v, rho, H, dx):
    """`DeltaSPH2D._surface_state` (surface, dilated set F, fluid and fluid + wall renormalisation matrices, the fluid kernels) of the tank at this state."""
    kw = dict(gravity=G, c0=C0, alpha=ALPHA, noPen='impulse', shifting=True, graphStep=False, fluidWarp=True, fusedWall=True)
    tank = Scene([Body(bodyId=0, reps=[BoxRep(LO, HI, solid='outside')])], DEV)
    sim = DeltaSPH2D(pos, v, np.ones(len(pos)), dx, tank, DeltaSPHConfig(**kw), DEV, support=H)
    xt, rt = torch.as_tensor(pos, device=DEV, dtype=F64), torch.as_tensor(rho, device=DEV, dtype=F64)
    return sim._surface_state(xt, rt)


def reference(pos, v, rho, H, dx, form):
    kw = dict(gravity=G, c0=C0, alpha=ALPHA, wallViscosityForm=form, noPen='impulse', shifting=False, graphStep=False, fluidWarp=False, fusedWall=True)
    tank = Scene([Body(bodyId=0, reps=[BoxRep(LO, HI, solid='outside')])], DEV)
    withWall = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, tank, DeltaSPHConfig(**kw), DEV, support=H)
    noWall = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, None, DeltaSPHConfig(**kw), DEV, support=H)
    xt, vt, rt = (torch.as_tensor(a, device=DEV, dtype=F64) for a in (pos, v, rho))
    a1, d1, _ = withWall.rhs(xt, vt, rt)
    a0, d0, _ = noWall.rhs(xt, vt, rt)
    return (a1 - a0), (d1 - d0)


@pytest.mark.parametrize('kind,form', [(BCType.freeSlip, 'laplacian'), (BCType.noSlip, 'noslipMirror')])
def test_wall_terms_equal_the_reference_wall_contribution(kind, form):
    pos, v, rho, H, dx, st, wall = setup(kind)
    dacc, ddrho = reference(pos, v, rho, H, dx, form)
    ours_drho = wallContinuity(wall, st.densities, st.velocities)
    s = torch.ones_like(st.densities)
    P64 = C0 ** 2 * (st.densities.to(F64) - 1.0)                                         # the pressure of the rounded density, in double (the state's own is rounded again)
    ours_acc = (wallPressureAcceleration(wall, P64, s, st.densities, wallMass=1.0, h=H)
                + wallViscousAcceleration(wall, st.densities, st.velocities, ALPHA * C0 * H / 2.8213846683502197, H, wallMass=1.0))
    scale = float(dacc.abs().max())
    assert scale > 1.0                                                                     # the walls act: hydrostatic support and viscous drag
    tol = 5e-7 if st.positions.dtype == torch.float32 else 1e-9                           # the terms are returned in the state's precision
    assert float((ours_drho - ddrho).abs().max()) <= tol * max(1.0, float(ddrho.abs().max()))
    assert float((ours_acc - dacc).abs().max()) <= tol * scale


@pytest.mark.parametrize('kind,form', [(BCType.freeSlip, 'laplacian'), (BCType.noSlip, 'noslipMirror')])
def test_wall_loads_equal_the_reference_loads(kind, form):
    """The load of the fluid on the tank (reaction -m a to the pressure and viscous wall terms, acting at the fluid particle, torque about the body's centre) against `DeltaSPH2D.loadsAt`: [2, B, 3] =
    (pressure, wall viscous) x (Fx, Fy, torque). The hydrostatic part carries the weight of the fluid: the vertical pressure load is ~ -m_total g in magnitude."""
    pos, v, rho, H, dx, st, wall = setup(kind)
    kw = dict(gravity=G, c0=C0, alpha=ALPHA, wallViscosityForm=form, noPen='impulse', shifting=False, graphStep=False, fluidWarp=False, fusedWall=True)
    tank = Scene([Body(bodyId=0, reps=[BoxRep(LO, HI, solid='outside')])], DEV)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, tank, DeltaSPHConfig(**kw), DEV, support=H)
    ref = sim.loadsAt(*(torch.as_tensor(a, device=DEV, dtype=F64) for a in (pos, v, rho)))              # [2, 1, 3]
    s = torch.ones_like(st.densities)
    P64 = C0 ** 2 * (st.densities.to(F64) - 1.0)
    accP = wallPressureAcceleration(wall, P64, s, st.densities, wallMass=1.0, h=H, perBody=True)
    accV = wallViscousAcceleration(wall, st.densities, st.velocities, ALPHA * C0 * H / 2.8213846683502197, H, wallMass=1.0, perBody=True)
    ours = wallLoads(accP, accV, st.positions, st.masses, torch.tensor([[0.0, 0.0]], device=DEV, dtype=F64))
    assert ours.shape == ref.shape == (2, 1, 3)
    scale = float(ref.abs().max())
    tol = 2e-6 if st.positions.dtype == torch.float32 else 1e-9
    assert scale > 0.1
    assert float((ours - ref).abs().max()) <= tol * scale, (ours, ref)
    weight = float(st.masses.sum()) * 9.81
    assert abs(float(ours[0, 0, 1])) > 0.5 * weight                                                          # the wall holds the fluid up (the column is nearly at rest)
