"""Fused SDF parity, requested moments, failure policy and live graph inputs."""
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.scene.scene import SdfRep, _sdfCurvature
from warpSPHBoundaries.scene.sdf_warp import CANONICAL, evaluate
from warpSPHBoundaries.scene.implicitBodies import Tier3

DEVICES = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
PROFILES = [None, (0,), (0, 1, 2), (3, 4), (5, 6, 7, 8), (9, 10, 11), ()]


@pytest.mark.parametrize('device', DEVICES)
def test_power_of_two_support_width_uses_integer_level_selection(device):
    from warpSPHBoundaries.scene.sdf_guard import square_minima, support_validity
    levels = square_minima(torch.full((37, 43), 5., dtype=torch.float64, device=device))
    points = torch.tensor([[5., 5.5], [5.5, 5.]], dtype=torch.float64, device=device)
    h = torch.full((2,), 3.75, dtype=torch.float64, device=device)
    assert support_validity(levels, torch.zeros(2, dtype=torch.float64, device=device), 1., points, h).all()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_fused_curved_jets_match_torch_values_selection_and_requested_channels(device, kernel):
    rng = np.random.default_rng(921)
    rep = SdfRep.fromFunction(lambda p: p.norm(dim=1)-3, (-5., -5.), (5., 5.), 1/16).to(device)
    angle = rng.uniform(-np.pi, np.pi, 100)
    radius = rng.uniform(2.1, 3.9, 100)
    p = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
    p = np.vstack((p, [[0, 0], [4.8, 0], [8, 0], [np.nan, 0], [np.inf, 0], [0, 3]]))
    h = np.r_[rng.uniform(.1, 1.2, 100), [1, 1, 1, 1, 1, np.nan]]
    pos = torch.tensor(p, dtype=torch.float64, device=device)
    support = torch.tensor(h, dtype=torch.float64, device=device)
    for profile in PROFILES:
        actual, low = _sdfCurvature(rep, pos, support, kernel, device, profile)
        reference, reference_low = _sdfCurvature(rep, pos, support, kernel, device, profile, backend='torch')
        torch.testing.assert_close(low, reference_low)
        torch.testing.assert_close(actual, reference, atol=3e-13, rtol=3e-12)
        requested = set(range(12)) if profile is None else {CANONICAL[c] for c in profile}
        assert not actual[:, [c for c in range(12) if c not in requested]].any()
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize('device', DEVICES)
def test_corners_cavities_inflections_and_nonfinite_field_nodes_fail_or_match_oracle(device):
    from warpSPHBoundaries.validation.sdf_stress import CASES
    functions = [lambda p: 3-p.norm(dim=1), lambda p: p[:, 1],
                 lambda p: torch.tensor(next(c for c in CASES if c.name == 'sharp-corner').distance(p.numpy()))]
    points = torch.tensor([[.2, .2], [-2., .2], [-2., -2.], [0., 3.2], [0., 0.]], dtype=torch.float64, device=device)
    supports = torch.ones(len(points), dtype=torch.float64, device=device)
    for function in functions:
        rep = SdfRep.fromFunction(function, (-5., -5.), (5., 5.), 1/32).to(device)
        actual, low = _sdfCurvature(rep, points, supports, 'w4', device)
        expected, elow = _sdfCurvature(rep, points, supports, 'w4', device, backend='torch')
        torch.testing.assert_close(low, elow)
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-3., -3.), (3., 3.), .125).to(device)
    rep.d[24, 24] = torch.nan
    actual, low = _sdfCurvature(rep, points[:1], supports[:1], 'w4', device)
    assert low.item() and not actual.any()


@pytest.mark.parametrize('device', DEVICES)
def test_curvature_variation_moments_match_oracle_on_non_circular_geometry(device):
    from warpSPHBoundaries.validation.sdf_accuracy import SmoothCase
    cases = [SmoothCase('cubic', kind='graph', kappa=.15, cubic=.12),
             SmoothCase('ellipse', a=8., b=4., angle=.63)]
    for case in cases:
        points = case.points([-.5, 0., .3, .7])
        rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                                 points.min(0)-1.4, points.max(0)+1.4, 1/32).to(device)
        pos = torch.tensor(points, dtype=torch.float64, device=device)
        h = torch.ones(len(points), dtype=torch.float64, device=device)
        actual, low = _sdfCurvature(rep, pos, h, 'w4', device)
        expected, elow = _sdfCurvature(rep, pos, h, 'w4', device, backend='torch')
        assert (~low).all() and rep.curvatureVariation(pos)[0].abs().max() > 1e-3
        torch.testing.assert_close(low, elow)
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)


def test_fused_posterior_checks_mass_even_if_only_first_moment_was_requested():
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), .125)
    table = Tier3('w4', 'cpu')
    table.F = table.F.clone()
    table.F[0] = torch.nan
    result, low = evaluate(rep, torch.tensor([[0., .2]], dtype=torch.float64),
                           torch.ones(1, dtype=torch.float64), table, (1, 2))
    assert low.item() and not result.any()


def test_static_cache_reset_policy_change_and_no_torch_jet_query(monkeypatch):
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-3., -3.), (3., 3.), .125)
    pos, h = torch.tensor([[0., .2]], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    _sdfCurvature(rep, pos, h, 'w4', 'cpu')
    original = rep._warpSdf
    def unexpected(*args):
        raise AssertionError('query-time Torch jet sampling reached the fused path')
    monkeypatch.setattr(rep, '_signed', unexpected)
    monkeypatch.setattr(rep, 'curvatureVariation', unexpected)
    _sdfCurvature(rep, pos, h, 'w4', 'cpu', (0, 1, 2))
    assert rep._warpSdf is original
    rep.gradTol = .05
    _sdfCurvature(rep, pos, h, 'w4', 'cpu')
    assert rep._warpSdf is not original
    rep.to('cpu')
    assert rep._warpSdf is None


@pytest.mark.parametrize('profile', [(1, 2), (9, 10, 11)])
def test_scene_preserves_requested_non_mass_moments_and_masks_fallback(profile):
    from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
    from warpSPHBoundaries.scene.scene import Body, Scene, SurfaceRep
    points = torch.tensor([[-3., .2], [.2, .2], [-3., -3.]], dtype=torch.float64)
    from warpSPHBoundaries.validation.sdf_stress import CASES
    case = next(c for c in CASES if c.name == 'sharp-corner')
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())), (-5., -5.), (2., 2.), .125,
        fallback=SurfaceRep.polygon([[-5., -5.], [0., -5.], [0., 0.], [-5., 0.]]))
    scene = Scene([Body(bodyId=0, reps=[rep])], 'cpu')
    h = torch.ones(len(points), dtype=torch.float64)
    ps = ParticleState(positions=points, supports=h, masses=h, densities=h,
                       kinds=torch.zeros(len(points), dtype=torch.int32))
    props = OperationProperties(kernel='w4', operation=WarpOperation.Density,
        gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    full = scene.pairMoments(ps, props).entries[0]
    partial = scene.pairMoments(ps, props, profile).entries[0]
    for key in ('implicit', 'surface'):
        field, shape = ('m1', (3, 2)) if profile == (1, 2) else ('m2', (3, 2, 2))
        expected_sum, actual_sum = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
        for expected in full[key]:
            expected_sum.index_add_(0, expected.q.long(), getattr(expected, field))
        for actual in partial[key]:
            actual_sum.index_add_(0, actual.q.long(), getattr(actual, field))
            assert not actual.lam.any() and not actual.g0.any() and not actual.g1.any()
            if profile == (1, 2):
                assert actual.m2 is None
            else:
                assert not actual.m1.any()
        torch.testing.assert_close(actual_sum, expected_sum)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA capture needs the host GPU')
@pytest.mark.parametrize('profile', [None, (0, 1, 2), (3, 4)])
@pytest.mark.parametrize('backend', ['warp', 'torch'])
def test_fused_guarded_sdf_capture_replays_changed_positions_and_supports(profile, backend):
    wp.init()
    device = 'cuda:0'
    rep = SdfRep.fromFunction(lambda p: p.norm(dim=1)-3, (-5., -5.), (5., 5.), 1/16, backend=backend).to(device)
    pos = torch.tensor([[0., 3.2], [0., 2.9], [4.7, 0.]], dtype=torch.float64, device=device)
    h = torch.ones(3, dtype=torch.float64, device=device)
    _sdfCurvature(rep, pos, h, 'w4', device, profile)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            actual, low = _sdfCurvature(rep, pos, h, 'w4', device, profile)
    torch.cuda.current_stream().wait_stream(stream)
    pos[:, 1].add_(.13)
    h.mul_(.7)
    graph.replay()
    torch.cuda.synchronize()
    expected, elow = _sdfCurvature(rep, pos, h, 'w4', device, profile, backend='torch')
    torch.testing.assert_close(low, elow)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
