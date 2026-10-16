"""Order-two moment family versus independent boundary/shape integrals."""
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import torch

from warpSPHBoundaries.edge.curvature import line_moment, moments as closed_moments
from warpSPHBoundaries.scene.implicitBodies import Tier3
from warpSPHBoundaries.validation.curvature import model_reference, exact_variation
from warpSPHBoundaries.validation.sdf_accuracy import SmoothCase, area_reference

DEVICES = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_closed_line_primitives_and_cumulative_spline(kernel):
    table = Tier3(kernel, 'cpu')
    d = torch.tensor([0., 1e-9, 1e-6, .01, .3, .5, .9, 1.], dtype=torch.float64)
    b2 = -2*table._hermite(1, d)[0].numpy()
    b4 = -6*table._integral(1, d, 1).numpy()
    for i, q in enumerate(d.numpy()):
        for power, actual in ((2, b2[i]), (4, b4[i])):
            fine = line_moment(kernel, Fraction(float(q)), power, dps=60)
            coarse = line_moment(kernel, Fraction(float(q)), power, dps=40)
            assert abs(fine-coarse) < mp.mpf('1e-35')
            np.testing.assert_allclose(actual, float(fine), atol=1e-11, rtol=0)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('distance', [Fraction(-3, 10), Fraction(0), Fraction(3, 10)])
def test_closed_form_family_precision_refinement_and_independent_shape_integrals(kernel, distance):
    fine = closed_moments(kernel, distance, Fraction(1, 8), Fraction(1, 30),
                          (Fraction(3, 5), Fraction(4, 5)), dps=65)
    coarse = closed_moments(kernel, distance, Fraction(1, 8), Fraction(1, 30),
                            (Fraction(3, 5), Fraction(4, 5)), dps=45)
    assert max(abs(a-b) for a, b in zip(fine, coarse)) < mp.mpf('1e-40')
    target = model_reference([float(distance)], [[.6, .8]], [.125], [1/30], kernel, 160)[0]
    np.testing.assert_allclose([float(x) for x in fine], target, atol=3e-14, rtol=3e-13)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_all_twelve_moments_match_independent_shape_quadrature(kernel, device):
    d = np.array([-1.1, -.8, -.3, -1e-9, 0., 1e-9, .2, .6, 1.1])
    normal = np.broadcast_to([.6, .8], (len(d), 2)).copy()
    k = np.linspace(-.2, .2, len(d))
    b = np.linspace(.03, -.04, len(d))
    table = Tier3(kernel, device)
    tensor = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    actual = table.curvature_moments(*map(tensor, (d, normal, k, b))).cpu().numpy()
    target = model_reference(d, normal, k, b, kernel, 160)
    np.testing.assert_allclose(actual[:, :6], target[:, :6], atol=3e-11, rtol=0)
    # Shipped F0 Hermite derivatives already have ~1e-8 interpolation error;
    # this is numerical error, separate from the shape expansion under test.
    np.testing.assert_allclose(actual[:, 6:], target[:, 6:], atol=2e-8, rtol=0)


@pytest.mark.parametrize('distance', [-.3, 0., .3])
def test_circle_curvature_corrections_have_third_order_model_error(distance):
    errors = []
    for radius in (16., 32.):
        case = SmoothCase('circle', a=radius, b=radius)
        point = case.points([distance])[0]
        target, _ = area_reference(case, point)
        _, normal, _, k, _ = case.frame
        actual = model_reference([distance], normal[None], [k], kernel='w4', order=160)[0]
        errors.append(np.abs(actual-target))
    # Components that vanish by symmetry are covered by all-moment shape tests;
    # check convergence of the nonzero scalar/normal/diagonal channels.
    channels = [0, 2, 3, 5, 7, 8, 11]
    for j in channels:
        if errors[0][j] > 1e-9:
            assert errors[0][j]/errors[1][j] > 6, (distance, j, errors)


def test_inflection_tangential_gradient_is_restored():
    table = Tier3('w4', 'cpu')
    case = SmoothCase('inflection', kind='graph', cubic=.03)
    normal = np.array([[0., 1.]])
    d = np.array([.3])
    target, _ = area_reference(case, case.points(d)[0])
    actual = table.curvature_moments(torch.tensor(d), torch.tensor(normal), torch.zeros(1),
                                     torch.tensor(exact_variation(case, 1))).numpy()[0]
    assert abs(actual[6]-target[6]) < abs(target[6])*1e-4
    assert abs(actual[1]-target[1]) < abs(target[1])*1e-4


def test_complement_reverses_normal_curvature_and_variation_consistently():
    table = Tier3('w4', 'cpu')
    d = torch.tensor([-.3, 0., .3], dtype=torch.float64)
    n = torch.tensor([[.6, .8]], dtype=torch.float64).expand(3, -1)
    k, b = torch.full((3,), .1, dtype=torch.float64), torch.full((3,), .03, dtype=torch.float64)
    inside = table.curvature_moments(d, n, k, b)
    # Complement flips n and t, hence k -> -k but k_s -> k_s.
    outside = table.curvature_moments(-d, -n, -k, b)
    whole = table.curvature_moments(torch.full((3,), -2.), torch.zeros_like(n), k, b)
    torch.testing.assert_close(inside+outside, whole, atol=5e-12, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_sampled_curvature_variation_refines_and_resets_cache(device):
    from warpSPHBoundaries.scene.scene import SdfRep
    case = SmoothCase('inflection', kind='graph', cubic=.12)
    points = case.points([-.3, 0., .3])
    errors = []
    for resolution in (32, 64):
        rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                                 (-2.5, -2.5), (2.5, 2.5), 1/resolution).to(device)
        value, covered = rep.curvatureVariation(torch.tensor(points, device=device))
        assert covered.all()
        errors.append(float((value-.12).abs().max()))
        assert rep._lapGradient is not None
        rep.to(device)
        assert rep._lapGradient is None
    assert errors[1] < errors[0]/2
    assert errors[1] < 5e-4


def test_nonfinite_variation_is_rejected_before_all_moment_evaluation(monkeypatch):
    from warpSPHBoundaries.scene.scene import SdfRep, _sdfCurvature
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), 1/16)
    monkeypatch.setattr(rep, 'curvatureVariation', lambda p: (torch.full((len(p),), float('inf'), dtype=p.dtype), torch.ones(len(p), dtype=torch.bool)))
    values, low = _sdfCurvature(rep, torch.tensor([[0., .2]], dtype=torch.float64), torch.ones(1), 'w4', 'cpu', backend='torch')
    assert low.item()
    assert torch.isfinite(values).all() and not values.any()


def test_variation_requires_two_node_stencil_coverage():
    from warpSPHBoundaries.scene.scene import SdfRep
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-2., -2.), (2., 2.), 1/8)
    _, covered = rep.curvatureVariation(torch.tensor([[-1.875, 0.], [-1.75, 0.]], dtype=torch.float64))
    assert covered.tolist() == [False, True]


def test_rapid_ripple_passes_curvature_only_guard_but_rejects_variation_model():
    from warpSPHBoundaries.validation.curvature import RippleWall
    from warpSPHBoundaries.scene.scene import SdfRep, _sdfCurvature
    from warpSPHBoundaries.scene.sdf_guard import cell_support_limits, square_minima, support_validity
    wall = RippleWall()
    point = wall.points([.3])
    rep = SdfRep.fromFunction(lambda p: torch.tensor(wall.distance(p.numpy())), (-1.4, -1.4), (1.4, 1.8), 1/128)
    position = torch.tensor(point)
    h = torch.ones(1, dtype=torch.float64)
    old_levels = square_minima(cell_support_limits(rep._node_fields(), rep.spacing, .5, .1, .25))
    assert support_validity(old_levels, rep.origin, rep.spacing, position, h).item()
    assert abs(rep.signed(position)[2].item()) < .01
    assert abs(rep.curvatureVariation(position)[0].item()) > 5
    _, low = _sdfCurvature(rep, position, h, 'w4', 'cpu')
    assert low.item()


def test_variation_cell_bound_covers_random_bilinear_jets():
    from warpSPHBoundaries.scene.sdf_guard import cell_support_limits
    rng = np.random.default_rng(71)
    nodes = np.zeros((4, 9, 11))
    nodes[0] = rng.uniform(-.2, .2, (9, 11))
    nodes[1] = rng.uniform(.94, 1.02, (9, 11))
    nodes[2] = rng.uniform(-.02, .02, (9, 11))
    nodes[3] = rng.uniform(-.1, .1, (9, 11))
    gradient = rng.uniform(-.2, .2, (2, 9, 11))
    limits = cell_support_limits(torch.tensor(nodes), .1, .5, .1, .25, .5, torch.tensor(gradient)).numpy()
    for _ in range(500):
        i, j = rng.integers([0, 0], limits.shape)
        if limits[i, j] == 0:
            continue
        x, y = rng.random(2)
        blend = lambda f: (1-x)*(1-y)*f[:, i, j]+x*(1-y)*f[:, i+1, j]+(1-x)*y*f[:, i, j+1]+x*y*f[:, i+1, j+1]
        d, gx, gy, lap = blend(nodes)
        normal = np.array([gx, gy])/np.hypot(gx, gy)
        tangent = np.array([-normal[1], normal[0]])
        beta = np.dot(tangent, blend(gradient))/(1-d*lap)**3
        assert abs(beta)*limits[i, j]**2 <= .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph check')
def test_all_moment_tensor_evaluator_replays_without_stale_inputs():
    device = 'cuda:0'
    table = Tier3('w4', device)
    d = torch.tensor([-.3, .1, .4], dtype=torch.float64, device=device)
    n = torch.tensor([[.6, .8]], dtype=torch.float64, device=device).expand(3, -1)
    k = torch.full_like(d, .1)
    b = torch.full_like(d, .03)
    table.curvature_moments(d, n, k, b)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = table.curvature_moments(d, n, k, b)
    d.add_(.03)
    b.mul_(-2)
    graph.replay()
    torch.testing.assert_close(output, table.curvature_moments(d, n, k, b), atol=2e-14, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_curved_scene_exposes_second_moments_across_local_full_and_fallback_rows(device):
    from warpSPHBoundaries.scene.scene import SdfRep, SurfaceRep
    from warpSPHBoundaries.validation.adversarial import scene_channels
    from warpSPHBoundaries.validation.sdf_stress import CASES
    case = next(c for c in CASES if c.name == 'sharp-corner')
    vertices = np.array([[-4., -4.], [0., -4.], [0., 0.], [-4., 0.]])
    points = np.array([[-2., .2], [.2, .2], [-2., -2.]])
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-5., -5.), (2., 2.), 1/32, fallback=SurfaceRep.polygon(vertices)).to(device)
    from warpSPHBoundaries.scene.scene import _sdfTier3
    mask = _sdfTier3(rep, torch.tensor(points, device=device), torch.ones(3, dtype=torch.float64, device=device), 'w4', device)[2]
    assert mask.cpu().tolist() == [False, True, False]
    actual = scene_channels(rep, points, 'w4', device, second=True)
    target = case.reference(points, 'w4')
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, target, atol=2e-8, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_scene_curved_moment_rotation_and_varying_support_scale(device):
    from warpSPHCore import ParticleState, OperationProperties, KernelFunctions, WarpOperation
    from warpSPHBoundaries.scene.scene import Scene, Body, SdfRep
    from warpSPHBoundaries.validation.distance_only import halfplane_channels
    n = np.array([.6, .8])
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = np.array([.3, -.2])
    h = np.array([.5, 1., 2.])
    distance = np.array([-.2, .0, .3])
    local = (h*distance)[:, None]*n
    positions = local@rotation.T+centre
    tensor = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    rep = SdfRep.fromFunction(lambda p: p@torch.tensor(n), (-6., -6.), (6., 6.), 1/32)
    scene = Scene([Body(center=tuple(centre), angle=angle, reps=[rep])], device)
    one = torch.ones(3, dtype=torch.float64, device=device)
    ps = ParticleState(positions=tensor(positions), supports=tensor(h), masses=one, densities=one,
                       kinds=torch.zeros(3, dtype=torch.int32, device=device))
    props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Density)
    actual = np.zeros((3, 12))
    for entry in scene.pairMoments(ps, props).entries:
        for m in entry['implicit']:
            q = m.q.cpu().numpy()
            assert m.m2 is not None
            actual[q, 0] = m.lam.cpu().numpy()
            actual[q, 1:3] = m.m1.cpu().numpy()/h[q, None]
            actual[q, 3:6] = m.m2.cpu().numpy().reshape(-1, 4)[:, [0, 1, 3]]/h[q, None]**2
            actual[q, 6:8] = m.g0.cpu().numpy()*h[q, None]
            actual[q, 8:] = m.g1.cpu().numpy().reshape(-1, 4)
    target = halfplane_channels(distance, np.broadcast_to(n@rotation.T, (3, 2)), 'w4', np.longdouble)
    np.testing.assert_allclose(actual, target, atol=2e-8, rtol=0)
