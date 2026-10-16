"""Independent domain and same-model references for accepted smooth SDFs."""
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy.optimize import differential_evolution

from warpSPHBoundaries.edge.tier3 import F_k, gradient_expansion
from warpSPHBoundaries.validation.distance_only import halfplane_channels
from warpSPHBoundaries.validation.sdf_accuracy import (
    SmoothCase, accepted_channel_gate, area_reference, boundary_reference_mp, local_model_reference,
    rotate_channels, shape_coefficients)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_direct_area_plane_all_twelve_channels(kernel):
    case = SmoothCase('plane', kind='graph')
    for d in (-.3, .0, .3):
        target, estimate = area_reference(case, [0., d], kernel, order=64)
        exact = halfplane_channels([d], kernel=kernel, dtype=np.longdouble)[0]
        np.testing.assert_allclose(target, exact, atol=2e-12, rtol=2e-12)
        assert estimate < 2e-11


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_independent_shape_derivatives_match_existing_tubular_expansion(kernel):
    d = Fraction(3, 10)
    f1, f2, *_ = shape_coefficients(float(d), kernel, 128)
    np.testing.assert_allclose([f1, f2], [float(F_k(kernel, d, j, dps=40)) for j in (1, 2)], atol=2e-15, rtol=3e-13)
    for sign in (-1, 1):
        values = local_model_reference([sign*.3], [[0, 1]], [.125], kernel, order=128)[0]
        with mp.workdps(40):
            # Inside continuation: 1 - exterior with reversed curvature.
            f = sum(mp.mpf(sign*.125)**j*F_k(kernel, d, j, dps=40) for j in range(3))
            value = f if sign == 1 else 1-f
            derivative = gradient_expansion(kernel, d, Fraction(sign, 8), dps=35)
        np.testing.assert_allclose(values[[0, 7]], [float(value), float(derivative)], atol=3e-13, rtol=1e-11)


@pytest.mark.parametrize('case', [SmoothCase('disk', a=3, b=3),
                                 SmoothCase('cavity', a=3, b=3, cavity=True),
                                 SmoothCase('ellipse', a=5, b=3, angle=.63),
                                 SmoothCase('cubic', kind='graph', kappa=.15, cubic=.3)])
@pytest.mark.parametrize('distance', [-.3, .3])
def test_area_against_independent_high_precision_boundary_all_twelve(case, distance):
    point = case.points([distance])[0]
    area, estimate = area_reference(case, point, 'w4', 64)
    boundary = boundary_reference_mp(case, point, 'w4', 32)
    np.testing.assert_allclose(area, boundary, atol=2e-12, rtol=3e-11)
    assert estimate < 2e-11


def test_true_cubic_distance_minimizes_over_every_real_stationary_point():
    case = SmoothCase('cubic', kind='graph', kappa=.15, cubic=.3)
    points = np.array([[-2., 2.], [.2, -.3], [0., 2.], [2., -2.], [0., 0.]])
    d = case.distance(points)
    for p, distance in zip(points, d):
        result = differential_evolution(lambda x: (x[0]-p[0])**2+(case.height(x[0])-p[1])**2,
                                        [(-8, 8)], seed=1, tol=1e-12, polish=True)
        np.testing.assert_allclose(distance**2, result.fun, atol=2e-12, rtol=2e-12)
    offsets = np.array([-.8, -.3, 0, .3, .8])
    np.testing.assert_allclose(case.distance(case.points(offsets)), offsets, atol=2e-14, rtol=0)


def test_rotation_and_complement_keep_moments_and_reference_independent():
    solid = SmoothCase('ellipse', a=5, b=3, angle=.63)
    cavity = SmoothCase('cavity', a=5, b=3, angle=.63, cavity=True)
    rotated = SmoothCase('rotated', a=5, b=3, angle=.63, rotation=.37)
    point = solid.points([.2])[0]
    a, _ = area_reference(solid, point)
    b, _ = area_reference(cavity, point)
    whole = halfplane_channels([-2])[0]
    np.testing.assert_allclose(a+b, whole, atol=2e-12, rtol=2e-12)
    c, _ = area_reference(rotated, rotated.points([.2])[0])
    np.testing.assert_allclose(c, rotate_channels(a, rotated.matrix), atol=2e-12, rtol=2e-12)


def test_inflection_has_tangential_gradient_despite_zero_closest_curvature():
    case = SmoothCase('inflection', kind='graph', cubic=.3)
    point = case.points([.3])[0]
    target, _ = area_reference(case, point)
    local = local_model_reference([.3], [[0., 1.]], [0.])[0]
    assert abs(target[6]) > 1e-4
    assert local[6] == 0
    assert np.isnan(local[3:6]).all()


@pytest.mark.parametrize('device', ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
def test_production_accepts_inflection_but_accuracy_remains_a_separate_gate(device):
    from warpSPHBoundaries.scene.scene import SdfRep, _sdfTier3
    from warpSPHBoundaries.validation.adversarial import scene_channels
    case = SmoothCase('inflection', kind='graph', cubic=.12)
    point = case.points([.3])
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-1.5, -1.2), (1.5, 1.8), 1/32).to(device)
    fallback = _sdfTier3(rep, torch.tensor(point, device=device), torch.ones(1, device=device), 'w4', device)[2]
    assert not fallback.item()
    output = scene_channels(rep, point, 'w4', device)
    reference, _ = area_reference(case, point[0], 'w4')
    assert np.isfinite(output[:, [0, 1, 2, 6, 7, 8, 9, 10, 11]]).all()
    error = abs(output[0, 6]-reference[6])
    # Curvature variation restores the leading tangential gradient. Sampling
    # and higher model terms still exceed epsilon32 on this coarse grid.
    assert np.finfo(np.float32).eps < error < .01*abs(reference[6])


def test_accuracy_gate_never_passes_missing_rejected_or_uncertain_data():
    errors = np.zeros((3, 12))
    errors[1] = 1
    uncertainty = np.full_like(errors, 1e-12)
    scale = np.ones(12)
    assert accepted_channel_gate(errors, uncertainty, scale, [1, 0, 1], 1e-7).all()
    assert not accepted_channel_gate(errors, uncertainty, scale, [0, 0, 0], 1e-7).any()
    assert not accepted_channel_gate(errors, uncertainty, scale, [1, 1, 1], 1e-7).any()
    errors[0, 0] = np.nan
    uncertainty[0, 1] = np.inf
    uncertainty[0, 2] = -1
    uncertainty[0, 3] = 1e-5
    passed = accepted_channel_gate(errors, uncertainty, scale, [1, 0, 1], 1e-7)
    assert not passed[:4].any() and passed[4:].all()


@pytest.mark.parametrize('device', ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
def test_planar_first_moment_includes_missing_subnode_integral(kernel, device):
    from warpSPHBoundaries.scene.implicitBodies import Tier3
    table = Tier3(kernel, device)
    q0 = float(table.q[0])
    offsets = np.array([-2*q0, -q0, -1e-9, 0., 1e-9, q0, 2*q0])
    d = torch.tensor(offsets, dtype=torch.float64, device=device)
    first = table.planar_moments(d)[2].cpu().numpy()
    reference = -halfplane_channels(offsets, kernel=kernel, dtype=np.longdouble)[:, 2]
    np.testing.assert_allclose(first, reference, atol=6e-12, rtol=0)
    # An independent radial identity at the wall, not the edge machinery.
    from warpSPHBoundaries.edge.kernels import kernel as get_kernel
    radial = 2*sum(sum(c*(hi**(j+3)-lo**(j+3))/Fraction(j+3) for j, c in enumerate(coeff))
                   for lo, hi, coeff in get_kernel(kernel).pieces)
    with mp.workdps(40):
        exact = mp.mpf(radial.numerator)/radial.denominator/mp.pi
    assert abs(first[3]-float(exact)) < 6e-12
    # The even moment has zero wall slope. The omitted strip gave a spurious
    # one-sided slope of approximately 1/2 below the first table node.
    assert abs(first[4]-first[3])/1e-9 < 1e-5
