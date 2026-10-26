"""Independent mesh, differential, asymptotic and whole-support checks."""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

from warpSPHBoundaries.mesh3d import Tetrahedron
from warpSPHBoundaries.mesh3d.radial import axisymmetric, select
from warpSPHBoundaries.mesh3d.slender import ball_series
from warpSPHBoundaries.visualizer.terms3d import FIELDS3D
from warpSPHBoundaries.visualizer.compute3d import sphere_mass

ALL = ['w2','w4','w6','cubic']


def exact_field(t, name, field):
    if field == 'm00':
        return t.moment(name)
    if field.startswith('m'):
        return t.moment(name, tuple(int(a) for a in field[1:]))
    if field.startswith('G'):
        return t.kernel_gradient(name)['xyz'.index(field[1])]
    if field.startswith('g1'):
        i,j = ['xyz'.index(a) for a in field[-2:]]
        return t.kernel_gradient(name, tuple(int(k == i) for k in range(3)))[j]
    if field.startswith('H'):
        i,j = ['xyz'.index(a) for a in field[-2:]]
        return t.kernel_hessian(name)[i][j]
    if field == 'laplacian':
        return t.laplacian(name)
    return mp.sqrt(sum(a*a for a in t.kernel_gradient(name)))


@pytest.mark.parametrize('name', ALL)
@pytest.mark.parametrize('d',[F(3,10),F(-3,10)])
def test_all_radial_halfspace_channels_against_analytic_mesh(name,d):
    # Exact rational proper rotation with n=(2,-2,1)/3; avoids only testing
    # the trivially zero off-diagonal entries of an axis-aligned half-space.
    rot = [[F(1,3),F(2,3),F(2,3)],[F(2,3),F(1,3),F(-2,3)],[F(-2,3),F(2,3),F(-1,3)]]
    # This orthogonal map has n equal to its last column; either determinant
    # is accepted since Tetrahedron explicitly orients each face outwards.
    def transform(v):
        return tuple(sum(rot[i][j]*v[j] for j in range(3)) for i in range(3))
    vertices = [transform(v) for v in [(-10,-10,0),(10,-10,0),(0,10,0),(0,0,-10)]]
    normal = np.array([float(row[2]) for row in rot])
    with mp.workdps(45):
        t = Tetrahedron(vertices,transform((0,0,d)))
        inv = axisymmetric([float(d)],name)
        for field in FIELDS3D:
            actual = select(inv,normal[None,:],field)[0]
            expected = float(exact_field(t,name,field))
            assert abs(actual-expected)<3e-12,(name,d,field,actual,expected)


@pytest.mark.parametrize('name',ALL)
def test_sphere_mass_and_hessian_match_independent_derivatives(name):
    point = np.array([.73,.31,-.2])
    R = .5
    r = np.linalg.norm(point)
    inv = axisymmetric([r-R],name,radius=R)
    normal = point[None,:]/r
    mass = select(inv,normal,'m00')[0]
    assert abs(mass-sphere_mass(point[None,:],R,name)[0])<1e-14
    step = 2e-5
    for i in range(3):
        left,right = point.copy(),point.copy()
        left[i]-=step;right[i]+=step
        derivative = (sphere_mass(right[None,:],R,name)[0]-sphere_mass(left[None,:],R,name)[0])/(2*step)
        assert abs(derivative-select(inv,normal,'G'+'xyz'[i])[0])<2e-8
        for j in range(3):
            def grad(p):
                length = np.linalg.norm(p)
                return select(axisymmetric([length-R],name,radius=R),p[None,:]/length,'G'+'xyz'[j])[0]
            hessian = (grad(right)-grad(left))/(2*step)
            field='H'+''.join(sorted('xyz'[i]+'xyz'[j]))
            assert abs(hessian-select(inv,normal,field)[0])<3e-7


@pytest.mark.parametrize('name',ALL)
def test_curvature_order_two_converges_cubically_for_complete_family(name):
    errors = []
    for R in (4.,8.,16.):
        exact = axisymmetric([.27],name,radius=R)
        local = axisymmetric([.27],name,curvature=1/R)
        errors.append(np.max(abs(exact-local)))
    assert errors[1]<errors[0]/6
    assert errors[2]<errors[1]/6
    assert errors[2]<1e-4


@pytest.mark.parametrize('name',ALL)
def test_small_ball_series_all_terms_against_exact_spherical_family(name):
    p = np.array([.31,.18,.11])
    q = np.linalg.norm(p)
    normal = p[None,:]/q
    for a in (.04,.02):
        inv = axisymmetric([q-a],name,radius=a)
        for field in FIELDS3D:
            expected = select(inv,normal,field)[0]
            actual = ball_series(p,a,name,field)
            # An order-four asymptotic expansion, not an exact quadrature.
            scale = max(abs(expected),4*np.pi*a**3/3)
            assert abs(actual-expected)/scale < (2e-3 if a==.04 else 4e-5),(name,a,field,actual,expected)
    assert np.isnan(ball_series([0,0,0],.1,name))
    assert np.isnan(ball_series([.95,0,0],.1,name))
    if name=='cubic':
        assert np.isnan(ball_series([.49,0,0],.03,name))
