import numpy as np
import pytest
from warpSPHBoundaries.fluxbridge import Geometry,AffineResistanceMaterial,pullback_gradient


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('precision,tol',[('float64',1e-9),('float32_compensated',1e-5)])
def test_affine_common_work_torque_and_rigid_rotation(dimension,precision,tol):
    d=dimension
    vertices=np.vstack([np.zeros(d),2*np.eye(d)])
    geometry=Geometry([vertices],d)
    rng=np.random.default_rng(8)
    R=np.diag(np.arange(1,d+1))
    controls=rng.normal(size=(1,d+1,d))
    material=AffineResistanceMaterial(geometry,[[R]],controls)
    queries=np.array([[.2]*d,[.3]*d])
    u=rng.normal(size=(2,d)); L=rng.normal(size=(2,d,d))
    batch=material.prepare(queries,1,precision=precision)
    result=batch.apply(u,[.7,.3],L)
    power=np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(power,-result.dissipation,atol=tol,rtol=tol)
    assert result.dissipation>=-tol
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=tol)
    np.testing.assert_allclose(result.torque_defect(queries,vertices[None]),0,atol=tol)
    omega=rng.normal(size=(d,d)); omega=omega-omega.T
    shift=rng.normal(size=d)
    moving=AffineResistanceMaterial(geometry,[[R]],vertices[None]@omega.T+shift)
    rigid=moving.prepare(queries,1,precision=precision).apply(queries@omega.T+shift,[.7,.3],np.broadcast_to(omega,(2,d,d)))
    np.testing.assert_allclose(rigid.fluid_force,0,atol=tol)
    np.testing.assert_allclose(rigid.gradient_load,0,atol=tol)
    np.testing.assert_allclose(rigid.reaction,0,atol=tol)
    assert abs(rigid.dissipation)<tol


@pytest.mark.parametrize('dimension',[2,3])
def test_supplied_gradient_transpose_conserves_particle_work_and_torque(dimension):
    d=dimension
    vertices=np.vstack([np.zeros(d),2*np.eye(d)])
    geometry=Geometry([vertices],d)
    queries=np.vstack([np.full(d,.1),np.full(d,.1)+np.eye(d)*.3])
    gradients=np.vstack([-np.ones(d),np.eye(d)])/.3
    rng=np.random.default_rng(12)
    u=rng.normal(size=(d+1,d)); controls=rng.normal(size=(1,d+1,d))
    L0=u.T@gradients
    L=np.broadcast_to(L0,(d+1,d,d))
    material=AffineResistanceMaterial(geometry,[[np.eye(d)]],controls)
    result=material.prepare(queries,1).apply(u,np.ones(d+1),L)
    force=pullback_gradient(result,lambda M:gradients@M.sum(axis=0).T)
    np.testing.assert_allclose(force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-9)
    work=np.sum(force*u)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(work,-result.dissipation,atol=1e-9)
    def torque(x,f):
        return (x[:,0]*f[:,1]-x[:,1]*f[:,0]).sum() if d==2 else np.cross(x,f).sum(axis=0)
    np.testing.assert_allclose(torque(queries,force)+torque(vertices,result.reaction[0]),0,atol=1e-9)

@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('precision,tol',[('float64',1e-8),('float32_compensated',1e-5)])
def test_affine_rotation_covariance(dimension,precision,tol):
    d=dimension
    vertices=np.vstack([np.zeros(d),2*np.eye(d)])
    queries=np.array([[.2]*d])
    rng=np.random.default_rng(7)
    Q,_=np.linalg.qr(rng.normal(size=(d,d)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    R=np.diag(np.arange(1,d+1))
    controls=rng.normal(size=(1,d+1,d)); u=rng.normal(size=(1,d)); L=rng.normal(size=(1,d,d))
    original=AffineResistanceMaterial(Geometry([vertices],d),[[R]],controls).prepare(queries,1,precision=precision).apply(u,[.7],L)
    moved=AffineResistanceMaterial(Geometry([vertices@Q.T],d),[[Q@R@Q.T]],controls@Q.T).prepare(queries@Q.T,1,precision=precision).apply(u@Q.T,[.7],Q@L@Q.T)
    np.testing.assert_allclose(moved.fluid_force,original.fluid_force@Q.T,atol=tol,rtol=tol)
    np.testing.assert_allclose(moved.gradient_load,Q@original.gradient_load@Q.T,atol=tol,rtol=tol)
    np.testing.assert_allclose(moved.reaction,original.reaction@Q.T,atol=tol,rtol=tol)
    np.testing.assert_allclose(moved.dissipation,original.dissipation,atol=tol,rtol=tol)

@pytest.mark.parametrize('dimension',[2,3])
def test_affine_virtual_work_directional_derivative(dimension):
    d=dimension
    vertices=np.vstack([np.zeros(d),2*np.eye(d)])
    geometry=Geometry([vertices],d)
    rng=np.random.default_rng(16)
    u=rng.normal(size=(1,d)); L=rng.normal(size=(1,d,d)); controls=rng.normal(size=(1,d+1,d))
    du=rng.normal(size=u.shape); dL=rng.normal(size=L.shape); dc=rng.normal(size=controls.shape)
    query=[[.2]*d]
    def result(u,L,c):
        return AffineResistanceMaterial(geometry,[[np.eye(d)]],c).prepare(query,1).apply(u,[.7],L)
    original=result(u,L,controls)
    work=np.sum(original.fluid_force*du)+np.sum(original.gradient_load*dL)+np.sum(original.reaction*dc)
    eps=1e-5
    derivative=(result(u+eps*du,L+eps*dL,controls+eps*dc).dissipation-result(u-eps*du,L-eps*dL,controls-eps*dc).dissipation)/(4*eps)
    np.testing.assert_allclose(work,-derivative,atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(original.affine_forcing,original.fluid_force/.7+np.einsum('nij,nj->ni',original.resistance,u),atol=1e-10)


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('precision,tol',[('float64',1e-8),('float32_compensated',1e-5)])
def test_cubic_affine_material_through_degree_nine(dimension,precision,tol):
    from math import comb
    from warpSPHBoundaries.mesh3d.fem import expansion
    d=dimension
    vertices=np.vstack([np.zeros(d),2*np.eye(d)])
    geometry=Geometry([vertices],d)
    n=comb(3+d,d)
    rng=np.random.default_rng(52)
    factors=rng.normal(size=(1,n,d,d))
    D=factors@factors.swapaxes(-1,-2)+np.eye(d)*.2
    controls=rng.normal(size=(1,n,d))
    material=AffineResistanceMaterial(geometry,D,controls,resistance_degree=3,velocity_degree=3)
    assert material.field.degree==9
    u=rng.normal(size=(1,d)); L=rng.normal(size=(1,d,d)); q=np.full((1,d),.2)
    result=material.prepare(q,1,precision=precision).apply(u,[.7],L)
    work=np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(work,-result.dissipation,atol=tol,rtol=tol)
    assert result.dissipation>=-tol
    padded=np.pad(vertices,((0,0),(0,3-d))).tolist()
    nodes,_=expansion(padded,padded[0],3)
    nodes=np.array([[float(t) for t in p[:d]] for p in nodes])
    np.testing.assert_allclose(result.torque_defect(q,nodes[None]),0,atol=tol)
