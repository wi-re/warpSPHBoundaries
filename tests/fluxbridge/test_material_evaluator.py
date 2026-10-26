"""Reusable static material compilation with independent query batches/snapshots."""
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,ResistanceMaterial,AffineResistanceMaterial

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
@pytest.mark.parametrize('affine',[False,True])
def test_compiled_material_reuses_static_fields_and_copies_source(device,precision,tol,affine):
    geo=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    rng=np.random.default_rng(72);controls=rng.normal(size=(1,4,3))
    factory=AffineResistanceMaterial if affine else ResistanceMaterial
    material=factory(geo,[[np.diag([1.,2.,3.])]],controls)
    options=dict(method='tet_quadrature',preparation='device',precision=precision,device=device,kernel='w6')
    compiled=material.compile(**options)
    q=np.array([[.2,.2,.2],[.3,.21,.19]]);u=rng.normal(size=(2,3));mass=np.array([.2,.3]);L=rng.normal(size=(2,3,3))
    apply=lambda b:b.apply(u,mass,L) if affine else b.apply(u,mass)
    a=compiled.prepare(q,1);result=apply(a);reference=apply(material.prepare(q,1,**options))
    for key in ['fluid_force','reaction','dissipation']+(['gradient_load'] if affine else []):
        np.testing.assert_allclose(getattr(result,key),getattr(reference,key),rtol=tol,atol=tol)
    other=compiled.prepare(q+.01,1)
    assert a.batch.owner is other.batch.owner is compiled.evaluator
    assert a.batch.owner.coefficients.ptr==other.batch.owner.coefficients.ptr
    apply(other)
    np.testing.assert_array_equal(apply(a).fluid_force,result.fluid_force)
    # Source mutation cannot alter an existing compiled field/offset snapshot.
    for poly in material.field.coefficients:
        for alpha,value in tuple(poly.items()):poly[alpha]=1.5*value
    changed=apply(material.prepare(q,1,**options))
    np.testing.assert_allclose(changed.fluid_force,1.5*reference.fluid_force,rtol=tol,atol=tol)
    material.controls.fill(123);material.offsets.fill(0)
    np.testing.assert_array_equal(apply(compiled.prepare(q,1)).fluid_force,result.fluid_force)
    assert not compiled.material.controls.flags.writeable
    assert not compiled.material.offsets.flags.writeable
