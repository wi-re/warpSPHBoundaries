"""Corner pole, and edge/vertex continuity of the executed fallback paths."""
import numpy as np
import pytest
import torch

from warpSPHBoundaries.scene.scene import SurfaceRep
from warpSPHBoundaries.validation.corner_audit import CORNER, SUPPORTED, production_channels
from warpSPHBoundaries.validation.sdf_stress import sampled_model
from warpSPHBoundaries.validation.sdf_corner_mechanism import planar_mass_reference


@pytest.mark.parametrize('kernel', ['w2', 'w4', 'w6', 'cubic'])
def test_distance_only_corner_ablation_is_continuous_at_face_vertex_seam(kernel):
    points = np.array([[-1e-7, .2], [1e-7, .2], [.2, .2]])
    result = sampled_model(CORNER, points, kernel)
    planar = result['planar_values']
    # The curvature pole must not contaminate the distance-only ablation.
    assert np.isfinite(planar[:, SUPPORTED]).all()
    assert np.isfinite(planar[:, 3:6]).all()
    np.testing.assert_allclose(planar[0, SUPPORTED], planar[1, SUPPORTED], atol=2e-6, rtol=0)
    exact = planar_mass_reference(CORNER.distance(points), kernel)
    np.testing.assert_allclose(planar[:, 0], exact, atol=2e-4, rtol=0)  # includes h/64 distance sampling
    assert abs(result['values'][2, 0]) > 100


@pytest.mark.parametrize('kernel', ['w2', 'w4', 'w6', 'cubic'])
def test_constant_distance_does_not_imply_constant_corner_integral(kernel):
    points = np.array([[0., .2], [.2/np.sqrt(2), .2/np.sqrt(2)], [.2, 0.]])
    distance = CORNER.distance(points)
    np.testing.assert_allclose(distance, .2, atol=5e-16, rtol=0)
    plane = planar_mass_reference(distance, kernel)
    np.testing.assert_allclose(plane, plane[0], atol=3e-15, rtol=0)
    corner = CORNER.reference(points, kernel)[:, 0]
    # At the axis, reflection in the other axis bisects the half-plane mass.
    np.testing.assert_allclose(corner[[0, 2]], plane[[0, 2]]/2, atol=3e-15, rtol=0)
    assert corner[0]-corner[1] > .01


@pytest.mark.parametrize('solid', ['inside', 'outside'])
def test_surface_indicator_uses_solid_angle_on_edges_and_vertices(solid):
    vertices = np.array([[0., 0.], [2., 0.], [2., 1.], [1., 1.], [1., 2.], [0., 2.]])
    points = np.vstack((vertices, (vertices+np.roll(vertices, -1, axis=0))/2,
                        [[2., 2.], [1., -.000001], [1., .000001]]))
    target = np.array([.25, .25, .25, .75, .25, .25]+[.5]*6+[0., 0., 1.])
    rep = SurfaceRep.polygon(vertices, solid=solid)
    if solid == 'outside':
        target = 1-target
    tensor = torch.as_tensor(points, dtype=torch.float64)
    np.testing.assert_allclose(rep.indicator(tensor).numpy(), target, atol=2e-15, rtol=0)
    # Cell caching must preserve the same convention, including y = y_max.
    np.testing.assert_allclose(rep.indicatorFast(tensor, 1.).numpy(), target, atol=2e-15, rtol=0)


def test_corner_refinement_cannot_cure_the_exterior_curvature_pole():
    point = [[.2, .2]]
    reference = CORNER.reference(point, 'w4')[0, 0]
    coarse = sampled_model(CORNER, point, spacing=1/32)
    fine = sampled_model(CORNER, point, spacing=1/128)
    assert coarse['fallback'][0] and fine['fallback'][0]
    assert abs(coarse['values'][0, 0]-reference) > 1000
    assert abs(fine['values'][0, 0]-reference) > 100*abs(coarse['values'][0, 0]-reference)
    np.testing.assert_allclose(reference, .0380174930325047334145, rtol=0, atol=1e-16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='production Warp paths need host CUDA')
def test_guarded_corner_and_provider_agree_with_reference_on_edges_and_vertex():
    points = np.array([[.2, .2], [.4, .2], [.2, -.2], [-.4, -.4],
                       [-.4, -.39], [0., 0.], [0., -.73125], [-.95625, 0.]])
    assert sampled_model(CORNER, points)['fallback'].all()
    reference = CORNER.reference(points, 'w4')
    for result in production_channels(points):
        assert np.isnan(result[:, 3:6]).all()
        np.testing.assert_allclose(result[:, SUPPORTED], reference[:, SUPPORTED], atol=5e-12, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fixed topology uses host CUDA')
@pytest.mark.parametrize('solid', ['inside', 'outside'])
def test_fixed_indicator_matches_boundary_angles_including_reflex_vertices(solid):
    from warpSPHBoundaries.scene.fixedadj import indicator_device
    vertices = np.array([[0., 0.], [2., 0.], [2., 1.], [1., 1.], [1., 2.], [0., 2.]])
    points = np.vstack((vertices, (vertices+np.roll(vertices, -1, axis=0))/2))
    rep = SurfaceRep.polygon(vertices, solid=solid).to('cuda:0')
    p = torch.as_tensor(points, dtype=torch.float64, device='cuda:0')
    torch.testing.assert_close(indicator_device(rep, p, 1.), rep.indicator(p), atol=2e-15, rtol=0)
