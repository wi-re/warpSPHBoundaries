import numpy as np
import pytest
from warpSPHBoundaries.fluxbridge import Geometry
from warpSPHBoundaries.fluxbridge.material import ResistanceMaterial,local_response


@pytest.mark.parametrize("dimension",[2,3])
@pytest.mark.parametrize("degree",[0,1,2,3])
def test_common_work_reaction_and_dissipation(dimension,degree):
    d=dimension
    geo=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])],d)
    from math import comb
    n=comb(degree+d,d)
    rng=np.random.default_rng(17)
    controls=rng.normal(size=(1,n,d))
    R=np.diag(np.arange(1,d+1))
    material=ResistanceMaterial(geo,[[R]],controls,velocity_degree=degree)
    u=rng.normal(size=(2,d))
    masses=np.array([.5,.8])
    batch=material.prepare([[.2]*d,[.3]*d],1)
    result=batch.apply(u,masses)
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-9)
    work=np.sum(result.fluid_force*u)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(work,-result.dissipation,atol=1e-9)
    assert result.dissipation>=-1e-10
    np.testing.assert_allclose(result.resistance,result.resistance.swapaxes(-1,-2),atol=1e-10)


@pytest.mark.parametrize("precision,tol",[("float64",1e-12),("float32",2e-6)])
def test_local_mobility_stiff_zero_and_rotated_tensor(precision,tol):
    angle=.7
    Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    D=np.array([np.zeros((2,2)),Q@np.diag([2.,20.])@Q.T,np.eye(2)*1e8])
    u=np.ones((3,2)); b=np.ones((3,2))*3
    result,mobility=local_response(D,u,b,.1,precision=precision)
    expected=np.linalg.inv(np.eye(2)+.1*D)
    np.testing.assert_allclose(mobility,expected,atol=tol,rtol=tol)
    np.testing.assert_allclose(result,np.einsum('nij,nj->ni',expected,u+.1*b),atol=tol,rtol=tol)
    # A driven terminal state is fixed by the supplied local algebra.
    u=np.linalg.solve(D[1],b[1])
    terminal,_=local_response(D[1:2],u[None],b[1:2],100,precision=precision)
    np.testing.assert_allclose(terminal[0],u,atol=tol,rtol=tol)


def test_reject_indefinite_resistance_and_surface_bulk_law():
    geo=Geometry([[[0,0],[1,0],[0,1]]],2)
    with pytest.raises(ValueError,match="semidefinite"):
        ResistanceMaterial(geo,[[np.diag([1,-1])]],np.zeros((1,1,2)),velocity_degree=0)
    surface=Geometry([[[0,0,0],[1,0,0],[0,1,0]]],3,"surface")
    with pytest.raises(ValueError,match="volume"):
        ResistanceMaterial(surface,[[np.eye(3)]],np.zeros((1,1,3)),velocity_degree=0)


def test_noncentral_drag_torque_is_explicitly_reported():
    geometry=Geometry([[[0,0],[2,0],[0,2]]],2)
    material=ResistanceMaterial(geometry,[[np.eye(2)]],np.zeros((1,3,2)))
    queries=np.array([[.2,.2]])
    result=material.prepare(queries,1).apply([[1,0]],[1])
    defect=result.torque_defect(queries,np.array(geometry.elements))
    assert np.isfinite(defect) and abs(defect)>1e-4
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-10)

@pytest.mark.parametrize("dimension",[2,3])
def test_cubic_resistance_velocity_test_degree_nine_common_work(dimension):
    from math import comb
    d=dimension
    geometry=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])],d)
    count=comb(3+d,d)
    rng=np.random.default_rng(22)
    factors=rng.normal(size=(1,count,d,d))
    resistance=factors@factors.swapaxes(-1,-2)+np.eye(d)*.2
    controls=rng.normal(size=(1,count,d))
    material=ResistanceMaterial(geometry,resistance,controls,resistance_degree=3,velocity_degree=3)
    assert material.field.degree==9
    u=rng.normal(size=(1,d))
    result=material.prepare([[.2]*d],1).apply(u,[.7])
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-8)
    power=np.sum(result.fluid_force*u)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(power,-result.dissipation,atol=1e-8)
    assert result.dissipation>=0
