"""Physical invariants and adapter regressions for the interactive laboratory."""
import json
import numpy as np
import pytest

from warpSPHBoundaries.visualizer.compute import (
    CHANNELS, DEFAULT, compute, evaluate_model, geometry, sdf, validate,
)


def arrays(result):
    return [np.array(r['values'], dtype=float) for r in result['results']]


@pytest.mark.parametrize('field', [*CHANNELS, 'grad_norm'])
@pytest.mark.parametrize('kernel', ['w4', 'cubic'])
def test_triangle_sum_matches_rectangle_for_every_channel(field, kernel):
    # Includes samples on the internal diagonal; magnitudes must be taken AFTER
    # summing triangle gradients, rather than adding per-triangle magnitudes.
    result = compute(dict(resolution=17, field=field, kernel=kernel,
                          models=['mesh', 'rectangle']))
    a, b = arrays(result)
    np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('resolution', [25, 41])
def test_circle_fan_seams_do_not_double_count_solid(resolution):
    result = compute(dict(geometry='circle', resolution=resolution,
                          models=['mesh', 'exact']))
    mesh, polygon = arrays(result)
    np.testing.assert_allclose(mesh, polygon, atol=2e-12)
    assert mesh.min() >= -2e-12
    assert mesh.max() <= 1+2e-12


def test_support_normalization_is_scale_invariant_for_all_models():
    common = dict(geometry='circle', resolution=17, models=['mesh', 'sdf', 'curvature', 'disk'], field='Gy')
    a = compute({**common, 'h': .3, 'radius': 1.})
    b = compute({**common, 'h': .6, 'radius': 2.})
    for x, y in zip(arrays(a), arrays(b)):
        np.testing.assert_allclose(x, y, atol=1e-11, rtol=1e-11)


def test_circle_curvature_improves_over_local_plane():
    r = compute(dict(geometry='circle', resolution=25, models=['sdf', 'curvature', 'disk']))
    plane, curved, reference = arrays(r)
    assert np.max(abs(curved-reference)) < np.max(abs(plane-reference))*.1
    assert np.max(abs(curved-reference)) < .002


def test_sampled_sdf_refines_toward_analytic_circle_jets():
    base = dict(geometry='circle', resolution=17, models=['sdf', 'curvature'])
    exact = arrays(compute(base))
    coarse = arrays(compute({**base, 'sdf_source': 'sampled', 'sdf_spacing': .4}))
    fine = arrays(compute({**base, 'sdf_source': 'sampled', 'sdf_spacing': .08}))
    for a, b, ref in zip(coarse, fine, exact):
        assert np.nanmax(abs(b-ref)) < np.nanmax(abs(a-ref))


def test_plane_mass_is_defined_at_an_ambiguous_polygon_normal():
    c = validate({'models': ['sdf']})
    vertices, triangles = geometry(c)
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 2.]])
    jets = sdf(c, vertices, points)
    values, _ = evaluate_model('sdf', c, vertices, triangles, points, jets)
    np.testing.assert_allclose(values[:3], [1., .5, .5], atol=1e-12)
    assert values[3] == 0


def test_incompatible_models_are_explicit_and_do_not_hide_valid_views():
    r = compute(dict(resolution=17, models=['mesh', 'curvature', 'disk']))
    assert 'values' in r['results'][0]
    assert 'polygon corners' in r['results'][1]['error']
    assert 'requires circle' in r['results'][2]['error']
    r = compute(dict(geometry='circle', radius=.2, resolution=17, models=['curvature', 'rectangle']))
    assert 'R/h' in r['results'][0]['error']
    assert 'requires rectangle' in r['results'][1]['error']


@pytest.mark.parametrize('bad', [
    {'resolution': 10000}, {'resolution': 17.5}, {'h': 0}, {'h': float('nan')},
    {'geometry': 'tetrahedron'}, {'models': []}, {'models': ['mesh', 'mesh']},
    {'models': ['oops']}, {'vertices': [[0, 0]]}, {'kernel': ['w4']},
    {'models': [False]}, {'field': 'not-a-channel'}, {'resolution': True},
    {'sdf_source': 'unknown'},
])
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(ValueError):
        compute(bad)


def test_degenerate_triangle_is_rejected_without_mutating_defaults():
    with pytest.raises(ValueError, match='collinear'):
        compute({'geometry': 'triangle', 'vertices': [[0, 0], [1, 1], [2, 2], [0, 1]]})
    assert DEFAULT['geometry'] == 'rectangle'


def test_results_are_strict_json_and_report_units_and_grid():
    r = compute({'resolution': 17, 'field': 'Gy'})
    json.dumps(r, allow_nan=False)
    assert r['dimension'] == 2 and r['schema_version'] == 1
    assert len(r['distance']) == len(r['x'])*len(r['y']) == 17**2
    assert r['config']['field'] == 'Gy'
    assert all(len(v['values']) == 17**2 for v in r['results'])
