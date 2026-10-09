"""Shifting with analytic walls: warpSPH's own `solveShifting` (surfaceNormal projection, Sun et al. 2017 Eq. 7 constants)
on the dam break's initial state with analytic tank walls, against the reference `DeltaSPH2D.shift` of warpSPHBoundaries
on the same particles. The reference's tensile value W0 is set to warpSPH's (the kernel at dx / kernelScale, which the
reference otherwise takes at dx: 3.7 % of the tensile term, docs/audit-warpsph-boundary-hooks.md), so the comparison isolates
the wall part: raw sum with the wall continuum, the detector, the curvature / lambda gates, the cap and the clamp.
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

from warpSPHCore import SupportScheme, buildVerletList  # noqa: E402

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme  # noqa: E402
from warpSPH.modules.analyticBoundary import kernelAtSpacing  # noqa: E402
from warpSPH.modules.shifting import solveShifting  # noqa: E402
from warpSPH.runner import buildContext, getCase  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402
from warpSPHBoundaries.scene import Body  # noqa: E402
from warpSPHBoundaries.scene.scene import BoxRep, Scene  # noqa: E402
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
F64 = torch.float64
DEV = 'cuda'


def build(nx=32):
    importAll()
    case = getCase('dambreak')
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params={**case.params, 'wallRepresentation': 'analytic'})
    spec = spec.merged(**case.defaults).merged(nx=nx)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    return ctx, case.buildSystem(ctx)


@pytest.mark.parametrize('curvatureGate', [False, True])
def test_solveShifting_with_analytic_walls_equals_the_reference_shift(curvatureGate):
    """curvatureGate=False: everything but the Sun 2019 Eq. (21) curvature gate agrees to round-off. With the gate on, a few free-surface
    particles differ: warpSPH's `_curvatureGate` takes the minimum normal dot product over the RAW Verlet list (pairs up to verletScale x
    support), the reference over the pairs inside the support, so warpSPH gates more (a property of the shared warpSPH code path, the same for
    boundary particles; the analytic flavour uses warpSPH's own `solveShifting`)."""
    ctx, system = build()
    st, config, sc = system.state, ctx.config, ctx.schemeConfig
    n = st.positions.shape[0]
    rng = np.random.default_rng(5)
    st.velocities = torch.as_tensor(rng.normal(scale=0.5, size=(n, 2)), device=st.positions.device, dtype=st.positions.dtype)
    sc.fluid.fixedSoundSpeed = 20.0
    sc.surfaceDetectionConfig.active = True
    sp = sc.shiftProperties
    sp.projectionScheme, sp.sun2017Eq7Shift, sp.reuseNormals, sp.iterations = ShiftingProjectionScheme.surfaceNormal, True, False, 1
    sp.surfaceCurvatureAngle = 15.0 if curvatureGate else 0.0                              # 0: no gate in warpSPH
    dt = 1e-3
    adj = buildVerletList(st, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric, verbose=False)
    got = solveShifting(st, config, sc, adj, dt).to(F64)

    # the reference: same particles, same tank
    pos = st.positions.double().cpu().numpy()
    vel = st.velocities.double().cpu().numpy()
    dx = float(config.dx)
    H = float(st.supports.max())
    lo, hi = [float(a) for a in config.domain.min.cpu()], [float(a) for a in config.domain.max.cpu()]
    interior = ctx.scratch['interiorDomain']
    lo, hi = [float(a) for a in interior.min], [float(a) for a in interior.max]
    tank = Scene([Body(bodyId=0, reps=[BoxRep(tuple(lo), tuple(hi), solid='outside')])], DEV)
    cfg = DeltaSPHConfig(gravity=(0.0, -9.81), c0=20.0, shiftCFL=sp.CFL, shiftR=0.2, shiftCapFraction=sp.maxShiftVelocityFraction, shiftThreshold=sp.threshold,
                         shiftCurvatureAngle=sp.surfaceCurvatureAngle if curvatureGate else 180.0, shiftLambda=sp.surfaceLambdaThreshold, noPen='impulse', shifting=True, graphStep=False, fluidWarp=True, fusedWall=True)
    sim = DeltaSPH2D(pos, vel, np.ones(n), dx, tank, cfg, DEV, support=H)
    sim._w0 = kernelAtSpacing(config.kernel, H, float(st.masses.mean()), sc.fluid.restDensity)
    ref = sim.shift(dt)

    scale = float(ref.abs().max())
    assert scale > 0
    err = float((got - ref).abs().max())
    near = (got.norm(dim=1) > 0).sum()
    print(f'shift scale {scale:.3e}, max |difference| {err:.3e}, nonzero {int(near)} of {n}')
    if curvatureGate:                                                                      # warpSPH gates more: the difference is bounded by the largest shift of the gated particles
        assert err <= 0.25 * scale
        return
    tol = 2e-3 if st.positions.dtype == torch.float32 else 1e-8
    assert err <= tol * scale
