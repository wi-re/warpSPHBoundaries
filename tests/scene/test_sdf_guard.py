"""Guard the interpolated jets over entire supports, before series arithmetic."""
import numpy as np
import pytest
import torch

from warpSPHBoundaries.scene.scene import SdfRep, _sdfTier3
from warpSPHBoundaries.scene.sdf_guard import cell_support_limits, square_minima, support_validity
from warpSPHBoundaries.validation.sdf_stress import CASES


DEVICES = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_sparse_support_minimum_matches_exhaustive_cell_scan(device):
    rng = np.random.default_rng(72)
    field = torch.tensor(rng.uniform(.01, 2., (37, 43)), device=device)
    field[12:14, 27:29] = 0
    points = torch.tensor(rng.uniform(-.1, 4.2, (800, 2)), device=device)
    supports = torch.tensor(rng.uniform(.001, 1.5, 800), device=device)
    origin = torch.tensor([0., 0.], dtype=torch.float64, device=device)
    got = support_validity(square_minima(field), origin, .1, points, supports).cpu().numpy()
    expected = []
    for p, h in zip(points.cpu().numpy(), supports.cpu().numpy()):
        lo, hi = np.floor((p-h)/.1).astype(int), np.floor((p+h)/.1).astype(int)
        inside = (lo >= 0).all() and (hi < field.shape).all()
        expected.append(inside and h <= float(field[lo[0]:hi[0]+1, lo[1]:hi[1]+1].min()))
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('device', DEVICES)
def test_cell_interval_limits_bound_every_random_bilinear_jet(device):
    rng = np.random.default_rng(2)
    nodes = np.empty((4, 13, 15))
    nodes[0] = rng.uniform(-.3, .3, nodes.shape[1:])
    nodes[1] = rng.uniform(.94, 1.02, nodes.shape[1:])
    nodes[2] = rng.uniform(-.03, .03, nodes.shape[1:])
    nodes[3] = rng.uniform(-.4, .4, nodes.shape[1:])
    tensor = torch.tensor(nodes, device=device)
    limits = cell_support_limits(tensor, .1, .5, .1, .25).cpu().numpy()
    assert np.count_nonzero(limits > 0) > 20
    for _ in range(1000):
        i, j = rng.integers([0, 0], np.array(limits.shape))
        if limits[i, j] <= 0:
            continue
        x, y = rng.random(2)
        d, gx, gy, lap = ((1-x)*(1-y)*nodes[:, i, j]+x*(1-y)*nodes[:, i+1, j]+
                          (1-x)*y*nodes[:, i, j+1]+x*y*nodes[:, i+1, j+1])
        den = 1-d*lap
        assert den >= .25
        assert .9 <= np.hypot(gx, gy) <= 1.1
        assert abs(lap/den)*limits[i, j] <= .5


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('case_name,point', [
    ('sharp-corner', [.2, .2]), ('thin-strip', [0., .2]),
    ('narrow-gap', [.05, .3]), ('tight-disk', [.45, 0.]),
])
def test_corner_and_features_between_old_probes_require_fallback(device, case_name, point):
    case = next(c for c in CASES if c.name == case_name)
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64).to(device)
    p = torch.tensor([point], dtype=torch.float64, device=device)
    h = torch.ones(1, dtype=torch.float64, device=device)
    lam, grad, fallback = _sdfTier3(rep, p, h, 'w4', device)
    assert fallback.item()
    torch.testing.assert_close(lam, torch.zeros_like(lam))
    torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize('device', DEVICES)
def test_flat_and_smooth_regions_remain_available_with_variable_support(device):
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-3., -3.), (3., 3.), 1/32).to(device)
    points = torch.tensor([[0., .2], [.3, -.1], [.1, 1.5], [.1, -1.5]], dtype=torch.float64, device=device)
    h = torch.tensor([1., .2, .4, .4], dtype=torch.float64, device=device)
    lam, grad, fallback = _sdfTier3(rep, points, h, 'w4', device)
    assert not fallback.any()
    assert 0 < lam[0] < .5 and .5 < lam[1] < 1
    torch.testing.assert_close(lam[2:], torch.tensor([0., 1.], dtype=torch.float64, device=device))
    torch.testing.assert_close(grad[2:], torch.zeros_like(grad[2:]))
    # Sampled circle R=3h is an allowed smooth regime, not a blanket fallback.
    circle = SdfRep.fromFunction(lambda p: p.norm(dim=1)-3., (-5., -5.), (5., 5.), 1/16).to(device)
    theta = torch.linspace(0, 2*np.pi, 31, dtype=torch.float64, device=device)[:-1]
    p = 3.2*torch.stack((theta.cos(), theta.sin()), 1)
    assert not _sdfTier3(circle, p, torch.ones(len(p), dtype=torch.float64, device=device), 'w4', device)[2].any()


@pytest.mark.parametrize('device', DEVICES)
def test_incomplete_sample_coverage_and_nonfinite_inputs_fail_closed(device):
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), .125).to(device)
    p = torch.tensor([[0., .2], [1.5, .2], [2.5, .2], [float('nan'), 0.], [0., .1]],
                     dtype=torch.float64, device=device)
    h = torch.tensor([1., 1., 1., 1., float('inf')], dtype=torch.float64, device=device)
    lam, grad, fallback = _sdfTier3(rep, p, h, 'w4', device)
    assert fallback.tolist() == [False, True, True, True, True]
    assert torch.isfinite(lam).all() and torch.isfinite(grad).all()
    assert (lam[1:] == 0).all() and (grad[1:] == 0).all()


def test_denominator_sign_and_pole_rejected_even_with_relaxed_curvature_gate():
    # Unit gradient alone cannot exclude the exact exterior-corner pole.
    nodes = torch.zeros((4, 5, 5), dtype=torch.float64)
    nodes[0] = 1.
    nodes[1] = 1.
    for lap in (1., 1+1e-8, 1-1e-8):
        nodes[3] = lap
        assert not cell_support_limits(nodes, .1, 1e12, .1, .25).any()


def test_rejected_curvature_never_reaches_table_series(monkeypatch):
    from warpSPHBoundaries.scene import scene as module
    original = module.tier3Table
    class CheckedTable:
        def __init__(self, inner):
            self.inner = inner
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def curvature_moments(self, q, n, k, b):
            assert torch.isfinite(q).all() and torch.isfinite(k).all()
            assert torch.isfinite(b).all() and torch.isfinite(n).all()
            assert (k.abs() <= .5).all()
            return self.inner.curvature_moments(q, n, k, b)
    monkeypatch.setattr(module, 'tier3Table', lambda name, dev: CheckedTable(original(name, dev)))
    case = next(c for c in CASES if c.name == 'sharp-corner')
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64)
    assert _sdfTier3(rep, torch.tensor([[.2, .2]], dtype=torch.float64), torch.ones(1), 'w4', 'cpu', backend='torch')[2].item()


def test_guard_cache_tracks_policy_changes_and_device_reset():
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), .125)
    p, h = torch.tensor([[0., .2]], dtype=torch.float64), torch.ones(1)
    assert rep.supportValidity(p, h).item()
    first = rep._validity
    rep.gradTol = .05
    assert rep.supportValidity(p, h).item()
    assert rep._validity is not first
    rep.to('cpu')
    assert rep._validity is None


@pytest.mark.parametrize('mass', [-1., 2., float('inf'), float('nan')])
def test_invalid_evaluated_mass_requests_fallback_instead_of_clamping(monkeypatch, mass):
    from warpSPHBoundaries.scene import scene as module
    class BrokenTable:
        variance = .05
        def curvature_moments(self, q, n, k, b):
            out = torch.zeros((len(q), 12), dtype=q.dtype, device=q.device)
            out[:, 0] = mass
            return out
    monkeypatch.setattr(module, 'tier3Table', lambda *args: BrokenTable())
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), .125)
    lam, grad, fallback = _sdfTier3(rep, torch.tensor([[0., .2]], dtype=torch.float64),
                                   torch.ones(1), 'w4', 'cpu', backend='torch')
    assert fallback.item()
    assert lam.item() == 0 and (grad == 0).all()


def test_missing_fallback_has_explicit_error_for_rejected_model():
    from warpSPHBoundaries.validation.adversarial import scene_channels
    case = next(c for c in CASES if c.name == 'sharp-corner')
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64)
    with pytest.raises(ValueError, match='local model.*no fallback'):
        scene_channels(rep, [[.2, .2]], 'w4', 'cpu')


@pytest.mark.parametrize('device', DEVICES)
def test_whole_solid_support_at_zero_gradient_has_identity_moment(device):
    from warpSPHBoundaries.validation.adversarial import scene_channels
    rep = SdfRep.fromFunction(lambda p: p.norm(dim=1)-3., (-5., -5.), (5., 5.), 1/16)
    result = scene_channels(rep, [[0., 0.]], 'w4', device)[0]
    np.testing.assert_allclose(result[[0, 1, 2, 6, 7]], [1, 0, 0, 0, 0], atol=0, rtol=0)
    np.testing.assert_allclose(result[8:], [1, 0, 0, 1], atol=0, rtol=0)
