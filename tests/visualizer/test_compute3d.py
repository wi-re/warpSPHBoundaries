"""3D mass conservation, exact-degree integrals, grid layout and model errors."""
import base64
from fractions import Fraction
import json

import numpy as np
import pytest

from warpSPHBoundaries.visualizer.compute import compute
from warpSPHBoundaries.visualizer.compute3d import (
    planar_mass, sphere_mass, signed_distance, validate3d,
)
from warpSPHBoundaries.curvbound.symbolic import planar3d, sphere


def decode(encoded):
    return np.frombuffer(base64.b64decode(encoded), dtype='<f4')


@pytest.mark.parametrize('name', ['w2','w4','w6','cubic'])
def test_planar_formula_agrees_with_existing_exact_rational_evaluator(name):
    distances = np.array([0., .1, .3, .5, .9, 1.])
    expected = [float(planar3d(name,Fraction(float(d)))) for d in distances]
    np.testing.assert_allclose(planar_mass(distances,name),expected,atol=6e-16)
    np.testing.assert_allclose(planar_mass(-distances,name),1-np.array(expected),atol=6e-16)
    np.testing.assert_allclose(planar_mass(np.array([-2.,2.]),name),[1.,0.])


@pytest.mark.parametrize('name', ['w2','w4','w6'])
@pytest.mark.parametrize('radius', [.2,.7,3.])
def test_sphere_exterior_matches_existing_rational_closed_forms(name,radius):
    d = np.array([0.,.1,.4,.8,1.])
    points = np.column_stack([radius+d,np.zeros_like(d),np.zeros_like(d)])
    expected = [float(sphere(name,Fraction(radius),Fraction(float(x)))) for x in d]
    np.testing.assert_allclose(sphere_mass(points,radius,name),expected,atol=8e-16)


@pytest.mark.parametrize('name', ['w2','w4','w6','cubic'])
def test_sphere_inside_support_and_centre(name):
    # Independent radial quadrature of the centre and antipodal-complement check.
    from scipy.integrate import quad
    from warpSPHBoundaries.visualizer.compute3d import profile
    radius = .3
    expected = quad(lambda q:4*np.pi*q*q*float(profile(np.array(q),name)),0,radius,epsabs=1e-13)[0]
    result = sphere_mass(np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[1.31,0.,0.]]),radius,name)
    assert abs(result[0]-expected)<1e-13
    assert result[0]>result[1]>result[2]>result[3]==0
    np.testing.assert_allclose(sphere_mass(np.array([[0.,0.,0.],[.2,0.,0.]]),2.,name),[1.,1.],atol=1e-15)


def test_3d_payload_layout_and_serialization():
    result = compute({'dimension':3,'resolution':17})
    json.dumps(result,allow_nan=False)
    assert result['encoding']=='base64-f32-le' and result['order']=='zyx'
    assert result['shape']==[17]*3 and result['dimension']==3
    distance = decode(result['distance']).reshape(result['shape'])
    centre = 8
    assert distance[centre,centre,centre]==-1
    for axis in [distance[centre,centre],distance[centre,:,centre],distance[:,centre,centre]]:
        np.testing.assert_allclose(axis,abs(np.array(result['x']))-1,atol=1e-7)
    assert all(len(decode(r['values']))==17**3 for r in result['results'])


def test_voxel_convolution_converges_to_sphere_integral():
    errors = []
    for resolution in [17,49]:
        result = compute({'dimension':3,'resolution':resolution,'models':['sphere','voxel']})
        exact,voxel = [decode(r['values']) for r in result['results']]
        errors.append(np.mean(abs(voxel-exact)))
    assert errors[1]<errors[0]*.35
    assert errors[1]<.002


@pytest.mark.parametrize('geometry', ['sphere','box','tetrahedron'])
def test_scale_invariance_and_gradient_axis_order(geometry):
    cfg = dict(dimension=3,resolution=17,geometry=geometry,models=['sdf','voxel'],field='Gz')
    a = compute({**cfg,'h':.3,'radius':.5,'width':1.,'height':1.5,'depth':.75})
    b = compute({**cfg,'h':.6,'radius':1.,'width':2.,'height':3.,'depth':1.5})
    for x,y in zip(a['results'],b['results']):
        np.testing.assert_allclose(decode(x['values']),decode(y['values']),atol=1e-6)
    if geometry!='tetrahedron':
        g=decode(a['results'][0]['values']).reshape([17]*3)
        np.testing.assert_allclose(g,-g[::-1],atol=1e-6)
        np.testing.assert_allclose(g[8],0,atol=1e-6)


def test_tetrahedron_distance_is_euclidean_outside_and_negative_inside():
    c=validate3d({'geometry':'tetrahedron'})
    points=np.array([[0.,0.,0.],[-1.,-1.,-1.],[-2.,-2.,-2.],[1.,1.,1.]])
    d=signed_distance(points,c)
    np.testing.assert_allclose(d,[-1/np.sqrt(3),0,np.sqrt(3),2/np.sqrt(3)],atol=1e-14)


def test_unsupported_sphere_view_does_not_hide_box_voxels():
    r=compute({'dimension':3,'geometry':'box','resolution':17,'models':['sphere','voxel']})
    assert 'requires sphere' in r['results'][0]['error']
    assert len(decode(r['results'][1]['values']))==17**3


@pytest.mark.parametrize('bad', [
    {'dimension':4},{'dimension':True},{'dimension':'3'},
    {'resolution':130},{'resolution':0},{'models':['sphere','sphere']},
    {'geometry':'mesh'},{'models':[]},{'field':'m20'},{'supersampling':3},
    {'h':float('nan')},{'radius':-1},{'models':['not-a-model']},
])
def test_3d_request_bounds(bad):
    with pytest.raises(ValueError):
        compute({'dimension':3,**bad})
