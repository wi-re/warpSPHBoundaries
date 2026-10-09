"""Free-surface detection with analytic walls (modules/analyticBoundary/detector.py) against the reference in
warpSPHBoundaries (`DeltaSPH2D._surface_state`): the same raw and dilated surface masks, the minimum eigenvalue of the
renormalisation matrix of fluid + wall, and the lambda-gradient normals, for a fluid block with a free surface and wall
contact on three sides (and at the corners), regular and jittered.
Lives in warpSPHBoundaries, not in warpSPH: it compares the analytic-boundary code of warpSPH (patches/warpsph/, `warpSPH.modules.analyticBoundary`) with this
package's reference solver `DeltaSPH2D`, and warpSPH does not import the package's `sim` layer.  Skipped when the installed warpSPH has no analytic-boundary code.
Run in both precisions (the tolerances follow the state's dtype):  `pytest tests/warpsph`  and  `warpSPHCore_PRECISION=float32 pytest tests/warpsph`
with `PYTHONPATH=<patched warpSPH>/src`.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPH.modules.analyticBoundary')

from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList  # noqa: E402

from warpSPH.configurations.moduleConfigurations.surfaceDetection import buildDefaultSurfaceDetectionConfig  # noqa: E402
from warpSPH.configurations.moduleConfigurations.gravity import GravityType  # noqa: E402
from warpSPH.configurations.region import BCType  # noqa: E402
from warpSPH.modules.analyticBoundary import detectFreeSurfaceAnalytic, sym2LamPinv  # noqa: E402
from warpSPH.modules.surfaceDetection import detectFreeSurface  # noqa: E402

from test_analyticWallTerms import C0, DEV, F64, G, HI, LO, block, reference_state_dict, setup  # noqa: E402


def adjacency_of(st, H, tp_domain=None):
    tp = st.positions.dtype
    dom = DomainDescription(min=torch.tensor([-1.0, -1.0], device=DEV, dtype=tp), max=torch.tensor([2.0, 2.0], device=DEV, dtype=tp),
                            periodic=torch.tensor([False, False], device=DEV), dim=2)
    adj = buildVerletList(st, dom, verletScale=1.2, supportMode=SupportScheme.SuperSymmetric, verbose=False)
    return dom, adj


@pytest.mark.parametrize('jitter', [0.0, 0.15])
def test_detector_equals_the_reference_state(jitter):
    pos, v, rho, H, dx, st, wall = setup(BCType.freeSlip, jitter=jitter)
    dom, adj = adjacency_of(st, H)
    provider = wall.provider
    config = SimpleNamespace(kernel=KernelFunctions.Wendland2, domain=dom, dx=dx, verletScale=1.2)
    schemeConfig = SimpleNamespace(boundaryProvider=provider, analyticWallMass=1.0, fluid=SimpleNamespace(restDensity=1.0), _analyticSupport=H)
    surfaceConfig = buildDefaultSurfaceDetectionConfig()
    raw, dilated, normals, renorm, lMin = detectFreeSurfaceAnalytic(st, config, schemeConfig, surfaceConfig, adj, wall=wall)
    ref = reference_state_dict(pos, v, rho, H, dx)
    assert torch.equal(raw.cpu(), ref['surface'].cpu())
    assert torch.equal((dilated > 0.5).cpu(), ref['F'].cpu())
    assert int(raw.sum()) > 10 and int(raw.sum()) < len(pos) // 2                         # a real surface layer, not everything
    lamRef, L = sym2LamPinv(ref['Mt'])
    assert float((lMin.to(F64) - lamRef).abs().max()) < (1e-5 if st.positions.dtype == torch.float32 else 1e-9)
    gl = torch.einsum('nab,nb->na', L, ref['fk'].lam_gradient(lamRef))
    nref = -gl / gl.norm(dim=1, keepdim=True).clamp(min=1e-300)
    tol = 1e-3 if st.positions.dtype == torch.float32 else 1e-7                          # the gradient sum runs in warpSPHCore's precision
    assert float((normals.to(F64) - nref).abs().max()) < tol


def test_hook_in_detectFreeSurface_dispatches_to_the_analytic_detector():
    pos, v, rho, H, dx, st, wall = setup(BCType.freeSlip)
    dom, adj = adjacency_of(st, H)
    config = SimpleNamespace(kernel=KernelFunctions.Wendland2, domain=dom, dx=dx, verletScale=1.2)
    schemeConfig = SimpleNamespace(boundaryProvider=wall.provider, analyticWallMass=1.0, fluid=SimpleNamespace(restDensity=1.0), _analyticSupport=H)
    surfaceConfig = buildDefaultSurfaceDetectionConfig()
    surfaceConfig.active = True
    schemeConfig.gravityConfig = SimpleNamespace(active=True, type=GravityType.Directional, magnitude=9.81, direction=[0.0, -1.0])
    out = detectFreeSurface(st, config, schemeConfig, surfaceConfig, adj, returnNormals=True)
    direct = detectFreeSurfaceAnalytic(st, config, schemeConfig, surfaceConfig, adj, wall=wall)
    assert torch.equal(out[0], direct[0]) and torch.equal(out[1], direct[1]) and torch.allclose(out[2], direct[2])
