"""Adversarial findings are reproducible; missing/rejected results stay explicit."""
import numpy as np
import pytest
import torch

from warpSPHBoundaries.validation.adversarial import (
    Measurement, polygon_cases, target_reference, triangle_area_reference,
    make_sdf, report, measured_budget, scene_channels, SUPPORTED)
from warpSPHBoundaries.validation.robustness import polygon_measurements, FAMILIES, coverage_status
from warpSPHBoundaries.validation.specialized_audit import (
    factored_radial, register_kernels, table_reference, representation_audit, ellipse_distance)


def case(name):
    return next(c for c in polygon_cases() if c.name == name)


@pytest.mark.parametrize('name', ['triangle-size-1e-09', 'triangle-angle-0.1', 'triangle-angle-179', 'sliver-1e-07', 'reflex-notch', 'acute-notch'])
def test_geometry_probes_include_exact_vertices_and_both_edge_sides(name):
    c = case(name)
    p = c.probes()
    np.testing.assert_array_equal(p[:len(c.vertices)], c.vertices)
    np.testing.assert_allclose(c.distance(c.vertices), 0, atol=1e-16)
    assert np.isfinite(c.distance(p)).all()
    assert c.distance(p).min() < 0 and c.distance(p).max() > 0


def test_tiny_triangle_reference_keeps_the_nonzero_signal_without_indicator_cancellation():
    c = case('triangle-size-1e-05')
    p = c.vertices.mean(0, keepdims=True)+[.2, .3]
    a = triangle_area_reference(c.vertices, p, 'w4', 32)
    b = triangle_area_reference(c.vertices, p, 'w4', 64)
    assert 0 < a[0, 0] < 1e-9
    np.testing.assert_allclose(a, b, rtol=2e-17, atol=1e-28)


def test_subgrid_triangle_is_reported_as_lost_instead_of_skipped():
    c = case('triangle-size-0.001')
    p = c.probes()
    target, uncertainty, _ = target_reference(c, p, 'w4')
    rep = make_sdf(c, p, 1/64)
    methods, prediction, lost = polygon_measurements(c, p, target, rep, 'w4', None)
    assert lost
    contour = next(m for m in methods if m.name == 'SDF-contour')
    assert 'geometry lost' in contour.note
    assert np.all(contour.values == 0)
    numerical, model, total = measured_budget(contour, target)
    np.testing.assert_array_equal(numerical, 0)
    np.testing.assert_array_equal(model, total)
    assert report(contour, target)['max_error_over_signal_scale'][0] == 1
    assert prediction['fallback'].any()


def test_report_distinguishes_rejected_domains_missing_channels_and_real_nonfinite_failures():
    target = np.ones((2, 12))*1e-20
    value = target.copy()
    value[0, 0] = np.nan
    value[:, 3:6] = np.nan
    m = Measurement('example', value, supported=SUPPORTED, domain=np.array([False, True]))
    assert report(m, target)['nonfinite_implemented_values'] == 0
    m.domain = None
    assert report(m, target)['nonfinite_implemented_values'] == 1
    assert report(m, target)['status'] == 'nonfinite_result'
    value[0, 0] = 0
    r = report(m, target)
    assert r['max_error_over_signal_scale'][0] == 1
    assert r['max_absolute_error'][0] == 1e-20


def test_refinement_difference_does_not_hide_shared_bias():
    target = np.ones((2, 12))
    m = Measurement('biased rule', target+.1, estimate=np.zeros_like(target))
    assert report(m, target)['refinement_underestimates_above_roundoff_floor'] == 24


@pytest.mark.parametrize('kernel', ['w2p5', 'w4p5'])
def test_high_degree_reference_uses_stable_factors_and_has_unit_mass(kernel):
    register_kernels()
    mass = factored_radial(np.array([0.], dtype=np.longdouble), np.array([1.], dtype=np.longdouble), kernel, 1)
    np.testing.assert_allclose(mass*2*np.arccos(np.longdouble(-1)), 1, rtol=0, atol=2e-17)
    points = np.array([[0., 0.], [.7, .1]])
    a = table_reference(points, .3, kernel, 48)
    b = table_reference(points, .3, kernel, 96)
    np.testing.assert_allclose(a, b, atol=5e-16, rtol=0)


def test_input_representation_failures_are_separate_from_evaluator_error(tmp_path):
    rows = representation_audit(tmp_path)
    assert any(r['collapsed'] and r['dtype'] == 'float32' for r in rows)
    assert all(r['local_reference_m00'] > 0 for r in rows)
    assert all(r['input_geometry_m00_error'] is None for r in rows if r['collapsed'])


def test_coverage_registry_includes_every_current_approximation_family():
    assert {'edge arithmetic', 'edge quadrature', 'volume / FEM', 'SDF curvature', 'SDF distance-only',
            'SDF contour geometry', 'production edge evaluation', 'boundary particles',
            'disk lookup', 'box lookup', 'T3 circle / cavity model', 'T4 disk series',
            'T4 strip series', 'polygon geometry approximation', 'implicit tier switches',
            'input representation', 'noncircular curvature', 'circle arc quadrature'} == set(FAMILIES)


def test_arc_quadrature_agrees_with_independent_reference_on_the_circle():
    from warpSPHBoundaries.edge.disk import disk_channels
    reference = table_reference(np.array([[.7, 0.]]), .7, 'w4', 96)
    actual = disk_channels('w4', np.array([.7]), .7)['lam']
    np.testing.assert_allclose(actual, reference[:, 0], atol=1e-12, rtol=0)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('radius', [.1, .5, .7, 1., 2., 6.])
def test_arc_boundary_channels_and_both_sides_against_independent_polar_reference(kernel, radius):
    from warpSPHBoundaries.validation.specialized_audit import arc_measurements
    distances = np.array([radius-1e-8, np.nextafter(radius, 0.), radius,
                          np.nextafter(radius, np.inf), radius+1e-8])
    points = np.column_stack((distances, np.zeros_like(distances)))
    # Interior rays develop a narrow transition near pi/2 even when the
    # regular circle-flux integrand is smooth. The old 96-node reference is
    # itself inaccurate at R-D=1e-8 for large disks.
    reference = table_reference(points, radius, kernel, 768)
    coarse = table_reference(points, radius, kernel, 384)
    np.testing.assert_allclose(reference[:, SUPPORTED], coarse[:, SUPPORTED], atol=5e-13, rtol=0)
    for m in arc_measurements(points, radius, kernel, reference):
        np.testing.assert_allclose(m.values[:, SUPPORTED], reference[:, SUPPORTED], atol=3e-12, rtol=0)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('radius', [.001, .1, .7, 2.])
def test_arc_centre_preserves_exact_symmetry_without_query_perturbation(kernel, radius):
    from warpSPHBoundaries.edge.disk import disk_channels
    ref = table_reference(np.array([[0., 0.]]), radius, kernel, 96)[0]
    actual = disk_channels(kernel, 0., radius)
    assert actual['m1x'] == actual['g0x'] == 0.
    assert actual['g1xx'] == actual['g1yy']
    np.testing.assert_allclose([actual['lam'], actual['g1xx'], actual['g1yy']],
                               ref[[0, 8, 11]], atol=2e-13, rtol=0)


@pytest.mark.parametrize('kernel', ['w2p5', 'w4p5'])
@pytest.mark.parametrize('radius', [.2, .7, .99999999, 2.])
def test_high_degree_centre_profiles_against_independent_factored_radial_integrals(kernel, radius):
    from warpSPHBoundaries.edge.disk import disk_channels
    register_kernels()
    ref = table_reference(np.array([[0., 0.]]), radius, kernel, 96)[0]
    actual = disk_channels(kernel, 0., radius)
    assert actual['m1x'] == actual['g0x'] == 0.
    np.testing.assert_allclose([actual['lam'], actual['g1xx'], actual['g1yy']],
                               ref[[0, 8, 11]], atol=3e-12, rtol=0)


def test_execution_coverage_does_not_claim_production_paths_were_run():
    status = coverage_status(dict(records=[dict(methods=[dict(name='closed-f64')])], specialized=[], representation=[]))
    assert status['edge arithmetic'].startswith('executed')
    assert status['production edge evaluation'] == 'not run'
    assert status['box lookup'] == 'not run'
    assert status['noncircular curvature'] == 'not run'
    assert status['SDF distance-only'] == 'not run'
    distance = coverage_status(dict(records=[dict(methods=[dict(name='SDF-distance-only')])], specialized=[], representation=[]))
    assert distance['SDF distance-only'].startswith('executed')
    assert distance['SDF curvature'] == 'not run'


@pytest.mark.parametrize('ratio', [.2, .75, 1.])
def test_ellipse_sdf_matches_known_normal_offsets_and_axis_branches(ratio):
    a, b = 2., 2.*ratio
    theta = np.array([0., .1, .7, np.pi/2])
    boundary = np.column_stack((a*np.cos(theta), b*np.sin(theta)))
    normal = np.column_stack((np.cos(theta)/a, np.sin(theta)/b))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    for offset in (-.01, 0., .01):
        np.testing.assert_allclose(ellipse_distance(boundary+offset*normal, a, b), offset, atol=3e-14, rtol=0)
    np.testing.assert_allclose(ellipse_distance([[0., 0.], [2.2, 0.]], a, b), [-b, .2], atol=3e-14, rtol=0)


@pytest.mark.parametrize('device', ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('name', ['triangle-size-0.001', 'triangle-angle-3', 'reflex-notch'])
def test_guarded_and_fused_production_paths_execute_on_adversarial_geometry(name, device):
    from warpSPHBoundaries.scene.scene import SurfaceRep
    c = case(name)
    p = c.probes()[:10]
    target, _, _ = target_reference(c, p, 'w4')
    for fused in (False, True):
        values = scene_channels(SurfaceRep.polygon(c.vertices), p, 'w4', device=device, fused=fused)
        assert np.isnan(values[:, 3:6]).all()
        np.testing.assert_allclose(values[:, SUPPORTED], target[:, SUPPORTED], atol=3e-11, rtol=0)


def test_sliver_reference_carries_precision_refinement_and_storage_uncertainty():
    c = case('sliver-1e-07')
    point = np.array([[.16, .21]])
    target, uncertainty, description = target_reference(c, point, 'w4')
    assert '70/45-digit' in description
    assert (uncertainty > 0).all()
    from warpSPHBoundaries.edge.triangle import integrate
    np.testing.assert_allclose(integrate(c.vertices, point[0], 'w4').moments, target[0], rtol=2e-12, atol=0)


def test_device_refinement_failure_remains_visible_in_saved_audit(tmp_path):
    from warpSPHBoundaries.validation.robustness import save_measurements
    target=np.ones((3,12))
    values=target.copy();values[0]=np.nan
    status=np.array([-1,0,1],dtype=np.int32)
    m=Measurement('device',values,estimate=np.full_like(values,.01),status=status)
    summaries=save_measurements(tmp_path,target,np.zeros_like(target),[m])
    assert summaries[0]['quadrature_queries']==2
    assert summaries[0]['quadrature_failed_queries']==1
    assert summaries[0]['nonfinite_implemented_values']==12
    assert summaries[0]['status']=='nonfinite_result'
    with np.load(tmp_path/'data.npz') as a:
        np.testing.assert_array_equal(a['device_status'],status)
        assert np.isnan(a['device'][0]).all()
