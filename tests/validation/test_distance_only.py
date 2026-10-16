"""The curvature-free reference is independent of curvature recovery/selection."""
import numpy as np
import pytest
import torch

from warpSPHBoundaries.scene.scene import SdfRep
from warpSPHBoundaries.validation.atlas import polygon_channels
from warpSPHBoundaries.validation.distance_only import (
    exact_distance_normal, halfplane_channels, sampled_distance_only)
from warpSPHBoundaries.validation.sdf_stress import CASES


@pytest.mark.parametrize('kernel', ['w2', 'w4', 'w6', 'cubic'])
def test_all_halfplane_moments_match_rotated_polygon_integrals(kernel):
    d = np.array([-1.1, -.8, -.2, 0., .2, .8, 1.1])
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    normal = np.broadcast_to(rotation[:, 1], (len(d), 2))
    values = halfplane_channels(d, normal, kernel)
    vertices = np.array([[-4, -4], [4, -4], [4, 0], [-4, 0]])@rotation.T
    points = d[:, None]*normal
    target = polygon_channels(vertices, points, kernel, dtype=np.longdouble)
    np.testing.assert_allclose(values, target, atol=2e-11, rtol=2e-10)


def test_sampler_and_reference_never_call_curvature_or_guard(monkeypatch):
    case = next(c for c in CASES if c.name == 'sharp-corner')
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64)
    for name in ('signed', '_signed', '_node_fields', 'supportValidity'):
        monkeypatch.setattr(rep, name, lambda *args: pytest.fail('curvature/guard path called'))
    points = np.array([[-1e-7, .2], [1e-7, .2], [.2, .2], [0., 0.]])
    result = sampled_distance_only(rep, points)
    assert np.isfinite(result['values']).all()
    assert np.max(np.abs(result['values'][0]-result['values'][1])) < 2e-6
    # 50-digit closed-form F0 at the exact sampled float64 distance. The
    # earlier table-based ablation differs by approximately 1.08e-12 here.
    np.testing.assert_allclose(result['values'][2, 0], .112848419898379660505, atol=3e-15, rtol=0)
    np.testing.assert_allclose(result['values'][3, 0], .5, atol=1e-13, rtol=0)


def test_mass_remains_defined_when_medial_axis_normal_is_undefined():
    case = next(c for c in CASES if c.name == 'narrow-gap')
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64)
    result = sampled_distance_only(rep, [[0., 0.]])
    assert not result['normal_defined'][0]
    assert 0 < result['values'][0, 0] < .5
    assert np.isnan(result['values'][0, 1:]).all()
    # Empty/full support is isotropic, even without a normal.
    full = halfplane_channels([-1.1, 1.1], np.full((2, 2), np.nan))
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(full[:, 8:], [[1, 0, 0, 1], [0, 0, 0, 0]])


def test_analytic_corner_normals_distinguish_vertex_projection_and_medial_axis():
    case = next(c for c in CASES if c.name == 'sharp-corner')
    d, n = exact_distance_normal(case, [[.2, .2], [-.5, .2], [-.4, -.4], [0., 0.]])
    np.testing.assert_allclose(d, [np.sqrt(.08), .2, -.4, 0.], atol=2e-17, rtol=1e-16)
    np.testing.assert_allclose(n[:2], [[2**-.5, 2**-.5], [0., 1.]], atol=2e-16, rtol=0)
    assert np.isnan(n[2:]).all()
    plane = halfplane_channels(d, n)
    assert np.isfinite(plane[:, 0]).all()


def test_sampled_distance_baseline_rejects_extrapolated_grid_values():
    rep = SdfRep.fromFunction(lambda p: p[:, 1], (-1., -1.), (1., 1.), 1/16)
    result = sampled_distance_only(rep, [[0., .2], [2., .2]])
    assert result['distance_defined'].tolist() == [True, False]
    assert np.isfinite(result['values'][0]).all()
    assert np.isnan(result['values'][1]).all()
