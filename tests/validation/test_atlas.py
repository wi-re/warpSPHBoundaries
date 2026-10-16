"""Check the atlas's independent reference and error accounting, not its pixels."""
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from warpSPHBoundaries.edge import oracle
from warpSPHBoundaries.validation.atlas import (
    ALPHAS, CHANNELS, Evaluation, disk_reference, disk_study, error_budget,
    polygon_channels,
)


@pytest.mark.parametrize('kernel', ['cubic', 'w4'])
@pytest.mark.parametrize('point', [(1.1, .2), (.2, -.1), (.7, 0.)])
def test_polar_reference_matches_independent_multiprecision_oracle(kernel, point):
    # Outside, inside, and exactly on the disk boundary; every scalar moment.
    got = disk_reference([point], .7, kernel, nodes=96)[0]
    for j, alpha in enumerate(ALPHAS):
        ref = oracle.polar_disk_moment((0, 0), Fraction(.7), point, kernel, alpha, dps=30)
        assert abs(float(got[j])-float(ref)) < 2e-13, (kernel, point, alpha)
    gradient = oracle.polar_disk_gradient((0, 0), Fraction(.7), point, kernel, dps=30)
    assert np.max(np.abs(got[6:8]-np.asarray(gradient, float))) < 2e-12


def test_reference_rotation_scaling_and_exact_support_limits():
    points = np.array([[1.1, .3], [.1, .2]])
    a = disk_reference(points, .7)
    rotation = np.array([[0., -1.], [1., 0.]])
    b = disk_reference(points@rotation.T, .7)
    assert np.allclose(b[:, 1:3], a[:, 1:3]@rotation.T, atol=1e-15)
    assert np.allclose(b[:, 3], a[:, 5], atol=1e-15)
    assert np.allclose(b[:, 4], -a[:, 4], atol=1e-15)
    assert np.allclose(b[:, 8:].reshape(-1, 2, 2),
                       rotation@a[:, 8:].reshape(-1, 2, 2)@rotation.T, atol=1e-14)
    full = disk_reference([[0, 0]], 2.)[0]
    assert abs(full[0]-1) < 1e-15
    assert np.max(np.abs(full[1:3])) < 1e-15
    assert np.allclose(full[8:].reshape(2, 2), np.eye(2), atol=1e-14)
    assert np.all(disk_reference([[3., 0]], .7) == 0)


def test_rotated_nearly_on_surface_reference_resolves_interior_half_ray():
    radius = .2
    point = radius*np.array([np.cos(.17), np.sin(.17)])
    a = disk_reference([point], radius, nodes=48)[0]
    b = disk_reference([point], radius, nodes=96)[0]
    assert np.max(np.abs(a-b)) < 2e-15
    for j, alpha in enumerate(ALPHAS):
        ref = oracle.polar_disk_moment((0, 0), Fraction(radius), point, 'w4', alpha, dps=35)
        assert abs(float(b[j])-float(ref)) < 2e-14


def test_gradient_moments_match_finite_differences_with_moving_origin():
    point = np.array([[1., .2]])
    base = disk_reference(point, .7)[0]
    for axis in (0, 1):
        shift = np.zeros((1, 2)); shift[0, axis] = 1e-5
        derivative = (disk_reference(point+shift, .7)[0]-disk_reference(point-shift, .7)[0])/(2e-5)
        assert abs(float(derivative[0]-base[6+axis])) < 2e-9
        for d in (0, 1):
            # partial_x m_d = g1[d,x] - delta[d,x] m00.
            assert abs(float(derivative[1+d]-(base[8+2*d+axis]-(d == axis)*base[0]))) < 2e-9


def test_surface_and_volume_use_same_geometry_and_channel_order():
    theta = np.arange(8)*2*np.pi/8
    polygon = .7*np.column_stack([np.cos(theta), np.sin(theta)])
    points = np.array([[1., .1], [.2, -.1], [-.5, .3]])
    a = polygon_channels(polygon, points)
    b = polygon_channels(polygon, points, triangulated=True)
    assert CHANNELS[:6] == ('m00', 'm10', 'm01', 'm20', 'm11', 'm02')
    assert a.shape == (3, 12)
    assert np.max(np.abs(a-b)) < 2e-12
    offset = np.array([2., -.4])
    moved = polygon_channels(polygon+offset, points+offset, triangulated=True)
    assert np.max(np.abs(b-moved)) < 2e-12


def test_error_budget_preserves_sign_and_missing_terms():
    target = np.array([[2., 3., 4.]])
    model = np.array([[2.5, 2., np.nan]])
    values = np.array([[2.4, 2.2, np.nan]])
    evaluation = Evaluation('example', values, model)
    numerical, model_error, total = error_budget(evaluation, target)
    assert np.allclose((numerical+model_error)[:, :2], total[:, :2])
    assert numerical[0, 0] < 0 and model_error[0, 0] > 0
    assert np.isnan(total[0, 2])


def test_adapters_do_not_fill_in_missing_moments_or_invalid_models():
    _, target, estimate, evaluations = disk_study(np.array([[1., .1]]), .7)
    assert target.shape == estimate.shape == (1, 12)
    assert np.isnan(evaluations[2].values).all()           # invalid curvature regime
    assert np.isnan(evaluations[3].values[:, 3:6]).all()  # missing second moments
    _, model_error, _ = error_budget(evaluations[3], target)
    assert np.all(model_error[:, [0, 1, 2, 6, 7, 8, 9, 10, 11]] == 0)


def test_invalid_reference_arguments_are_rejected():
    with pytest.raises(ValueError):
        disk_reference([[0, 0]], 0)
    with pytest.raises(ValueError):
        disk_reference([[0, 0]], 1, nodes=1)
