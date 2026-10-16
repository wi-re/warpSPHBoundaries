"""Geometry-conditioned fallback: independent integrals and physical identities."""
import numpy as np
import pytest

from warpSPHBoundaries.edge import fem, np_fem
from warpSPHBoundaries.edge.triangle import integrate
from warpSPHBoundaries.validation.adversarial import polygon_cases, triangle_area_reference


def case(name):
    return next(c for c in polygon_cases() if c.name == name)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('name', ['triangle-size-1e-09', 'triangle-size-0.001'])
def test_tiny_area_matches_independent_extended_precision_tensor_rule(kernel, name):
    c = case(name)
    points = c.probes()[[0, 6, 44, 45, 47]]
    ref = triangle_area_reference(c.vertices, points, kernel, 64)
    actual = [integrate(c.vertices, x, kernel) for x in points]
    assert all(r.converged for r in actual)
    scale = np.max(abs(ref), axis=0)
    assert np.max(abs(np.array([r.moments for r in actual])-ref)/scale) < 2e-10


@pytest.mark.parametrize('kernel', ['cubic', 'w4'])
def test_sliver_moments_and_nodal_weights_obey_rotation_orientation_and_support_scaling(kernel):
    c = case('sliver-1e-07')
    x = np.array([.16, .21])
    base = integrate(c.vertices, x, kernel)
    reverse = integrate(c.vertices[::-1], x, kernel)
    np.testing.assert_allclose(reverse.moments, base.moments, rtol=2e-12, atol=1e-25)
    np.testing.assert_allclose(base.weights.sum(), base.moments[0], rtol=1e-14)
    np.testing.assert_allclose(base.gradients.sum(0), base.moments[6:8], rtol=2e-12)
    for h in (.125, 8):
        r = integrate(c.vertices*h, x*h, kernel, h=h)
        scale = np.array([1,h,h,h*h,h*h,h*h,1/h,1/h,1,1,1,1])
        np.testing.assert_allclose(r.moments, base.moments*scale, rtol=3e-12, atol=1e-25)
    # Ninety-degree rotation permutes/signs scalar and gradient tensors.
    rotate = lambda y: np.stack((-y[..., 1], y[..., 0]), axis=-1)
    r = integrate(rotate(c.vertices), rotate(x), kernel)
    expected = base.moments[[0,2,1,5,4,3,7,6,11,10,9,8]]*np.array([1,-1,1,1,-1,1,-1,1,1,-1,-1,1])
    np.testing.assert_allclose(r.moments, expected, rtol=2e-12, atol=1e-25)


def test_hybrid_dispatch_preserves_p2_moments_on_extreme_sliver_and_origin():
    for name, x in [('sliver-1e-07', [.1,.2]), ('triangle-size-1e-09', [.2,.1])]:
        c = case(name)
        r = integrate(c.vertices, x, 'w4')
        w, g = np_fem.weights_hybrid(c.vertices, np.array(x), 'w4', 2, grad=True)
        relative = (c.vertices[0]-x)+(np.array(fem.basis(2)[0]) @ (c.vertices-c.vertices[0]))
        m = [(w[0]*relative[:,0]**a*relative[:,1]**b).sum() for a,b in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]]
        m += list(g[0].sum(0))
        m += [(relative[:,d]*g[0,:,j]).sum() for d in range(2) for j in range(2)]
        np.testing.assert_allclose(m, r.moments, rtol=1e-10, atol=3e-42 if 'size' in name else 1e-25)


def test_degenerate_and_invalid_geometry():
    r = integrate([[0,0],[1,0],[2,0]], [.1,.2], 'w4')
    assert r.converged and not r.moments.any() and not r.weights.any()
    with pytest.raises(ValueError):
        integrate([[0,0],[1,0],[0,1]], [0,0], 'w4', h=0)


@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('geometry', ['tiny-1e-12', 'sliver-1e-12', 'translated-sliver'])
def test_extended_conditioning_against_precision_refined_exact_float_geometry(kernel, geometry):
    from warpSPHBoundaries.validation.adversarial import precision_polygon_reference
    if geometry == 'tiny-1e-12':
        vertices = case('triangle-size-1').vertices*1e-12+[.2,.1]
    else:
        vertices = np.array([[-.8,-.4e-12],[.8,0],[-.8,.4e-12]])
        if geometry == 'translated-sliver':
            vertices = np.array([[-.8,-.4e-6],[.8,0],[-.8,.4e-6]])+[1e8,-1e8]
    point = vertices.mean(0)+[.16,.21]
    ref = precision_polygon_reference(vertices, point[None], kernel)[0]
    result = integrate(vertices, point, kernel)
    assert result.converged
    np.testing.assert_allclose(result.moments, ref, rtol=2e-11, atol=0)
