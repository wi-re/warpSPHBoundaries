"""Physical counterexamples behind the documentation plots, using the real policy."""
import numpy as np
import pytest

from warpSPHBoundaries.validation import atlas
from warpSPHBoundaries.validation.sdf_stress import CASES, sampled_model


def test_support_guard_rejects_thin_strip_missed_by_discrete_probes():
    case = next(c for c in CASES if c.name == 'thin-strip')
    points = np.array([[0., .2]])
    prediction = sampled_model(case, points)
    target = case.reference(points, 'w4')
    assert prediction['fallback'][0] and not prediction['accepted'][0]
    assert abs(prediction['curvature'][0]) < 1e-10
    assert abs(prediction['gradient_norm'][0]-1) < 1e-12
    assert prediction['values'][0, 0]-target[0, 0] > .05  # opposite sheet truncates the solid


@pytest.mark.parametrize('name,point', [
    ('tight-disk', [.45, 0.]),
    ('sharp-corner', [.25, .25]),
    ('narrow-gap', [0., 0.]),
])
def test_invalid_local_geometry_requests_fallback(name, point):
    case = next(c for c in CASES if c.name == name)
    prediction = sampled_model(case, [point])
    assert prediction['fallback'][0]
    assert not prediction['accepted'][0]
    assert prediction['values'].shape == (1, 12)
    assert len(prediction['implemented_channels']) == 12


def test_gap_distance_has_medial_axis_and_reference_is_disjoint_union():
    case = next(c for c in CASES if c.name == 'narrow-gap')
    prediction = sampled_model(case, [[0., 0.]])
    assert abs(prediction['distance'][0]-.075) < 1e-12
    assert prediction['gradient_norm'][0] < .01
    target = case.reference([[0., 0.]], 'w4')
    assert 0 < target[0, 0] < 1
    assert abs(target[0, 6]) < 1e-13               # cancellation by left/right symmetry


def test_support_guard_rejects_narrow_gap_despite_moderate_query_curvature():
    case = next(c for c in CASES if c.name == 'narrow-gap')
    points = np.array([[.05, .3]])
    prediction = sampled_model(case, points)
    target = case.reference(points, 'w4')
    assert prediction['fallback'][0] and not prediction['accepted'][0]
    assert abs(prediction['curvature'][0]-1/3) < 1e-4
    assert abs(prediction['values'][0, 0]-target[0, 0]) > .2


def test_high_resolution_batching_preserves_reference_and_channels(monkeypatch):
    points = np.array([[.4, .1], [.7, 0.], [1., .2], [-.8, .2], [0., 0.]])
    polygon = np.array([[-.4, -.3], [.6, -.2], [.2, .7]])
    reference = atlas.disk_reference(points, .7)
    surface = atlas.polygon_channels(polygon, points)
    volume = atlas.polygon_channels(polygon, points, triangulated=True)
    monkeypatch.setattr(atlas, 'RASTER_BATCH', 2)
    np.testing.assert_array_equal(atlas.disk_reference(points, .7), reference)
    np.testing.assert_array_equal(atlas.polygon_channels(polygon, points), surface)
    np.testing.assert_array_equal(atlas.polygon_channels(polygon, points, triangulated=True), volume)
