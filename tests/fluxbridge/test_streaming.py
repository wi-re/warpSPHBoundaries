"""Bounded native batches preserve frozen inputs, periodic CSR and material work."""
import gc
import weakref
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import (Geometry,Evaluator,FieldEvaluator,PolynomialField,
    ResistanceMaterial,AffineResistanceMaterial,baseline,adjacency)

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


def options(d,device,precision):
    return dict(device=device,precision=precision,search='device',
        method='conditioned_boundary' if d==2 else 'tet_quadrature',preparation='host' if d==2 else 'device')


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
@pytest.mark.parametrize('dimension',[2,3])
def test_values_and_field_shapes_preserve_full_batch(device,precision,tol,dimension):
    d=dimension;v=np.vstack([np.zeros(d),2*np.eye(d)])
    g=Geometry([v+np.eye(d)[0]*4*i for i in range(3)],d)
    q=np.full((5,d),.2);q[:,0]+=[0,4,8,0,30];h=np.linspace(.7,1.,len(q))
    providers=[Evaluator(g,baseline(d),**options(d,device,precision)),
        FieldEvaluator(PolynomialField.basis(g,2),derivative=(0,1),**options(d,device,precision)),
        FieldEvaluator(PolynomialField.constant(g,[1.,2.,3.]),**options(d,device,precision))]
    for owner in providers:
        expected=owner.evaluate(q,h)
        value,execution=owner.evaluate_streamed(q,h,chunk_size=2,return_metrics=True)
        np.testing.assert_allclose(value,expected,atol=tol,rtol=tol)
        assert value.shape==expected.shape and value.dtype==expected.dtype
        assert execution['chunks']==3 and execution['total_interactions']==4
        assert not execution['graph_compatible'] and execution['peak_owned_batch_bytes']>0
        empty=owner.evaluate_streamed([],1,chunk_size=2)
        assert empty.shape==(0,)+expected.shape[1:]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol,accepted',[('float64',1e-12,16),('float32',1e-5,8)])
def test_lower_order_provenance_excludes_empty_chunks(device,precision,tol,accepted):
    vertices=np.vstack([np.zeros(3),2*np.eye(3)])
    geometry=Geometry([vertices,vertices+[4,0,0]],3)
    settings=options(3,device,precision)
    q=np.array([[.2,.2,.2],[4.2,.2,.2],[30,30,30]])
    reference=Evaluator(geometry,baseline(3),nodes=16,**settings).evaluate(q,1)
    evaluator=Evaluator(geometry,baseline(3),nodes=8,**settings)
    values,execution=evaluator.evaluate_streamed(q,1,chunk_size=1,return_metrics=True)
    np.testing.assert_allclose(values,reference,atol=tol,rtol=tol)
    assert execution['requested_nodes']==8 and execution['chunks']==3
    assert execution['quadrature_orders']=={str(accepted):2}
    assert execution['quadrature_interactions_by_order']=={str(accepted):2}
    assert execution['quadrature_promotions']==(2 if accepted==16 else 0)
    assert execution['accepted_rule_samples']>0 and execution['comparison_rule_samples']>0
    assert execution['field_promotions']==0


@pytest.mark.parametrize('device',DEVICES)
def test_iterator_snapshots_periodic_adjacency_and_releases_batches(device,monkeypatch):
    g=Geometry([[[0,0],[1,0],[1,1],[0,1]]],2)
    owner=Evaluator(g,baseline(2),device=device,search='device')
    q=np.array([[.1,.2],[3.1,-4.8],[.4,.3]]);h=np.array([1.5,1.1,.7])
    neighbors=adjacency(g,q,h,periods=(1,1));expected=owner.evaluate(q,h,neighbors=neighbors)
    live=weakref.WeakSet();original=owner.prepare
    def prepare(*args,**kwargs):
        b=original(*args,**kwargs);live.add(b);return b
    monkeypatch.setattr(owner,'prepare',prepare)
    iterator=owner.iter_evaluate(q,h,neighbors=neighbors,chunk_size=2)
    q+=.13;h*=.5  # snapshot is taken when the iterator is created
    parts=[]
    for chunk in iterator:
        gc.collect();assert len(live)==0
        parts.append(chunk.values)
        assert chunk.stop-chunk.start<=2
    np.testing.assert_allclose(np.concatenate(parts),expected,atol=1e-12,rtol=1e-12)
    with pytest.raises(ValueError,match='stale'):owner.iter_evaluate(q,h,neighbors=neighbors)
    with pytest.raises(ValueError,match='periodicity'):
        owner.iter_evaluate(q-.13,h*2,neighbors=adjacency(g,q-.13,h*2,periods=(1,1)),periods=(2,2))


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('affine',[False,True])
def test_streamed_material_work_reaction_and_gradient(device,precision,tol,dimension,affine):
    d=dimension;g=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])+np.eye(d)[0]*4*i for i in range(3)],d)
    rng=np.random.default_rng(81);f=rng.normal(size=(3,d+1,d,d));controls=rng.normal(size=(3,d+1,d))
    cls=AffineResistanceMaterial if affine else ResistanceMaterial
    material=cls(g,f@f.swapaxes(-1,-2)+np.eye(d)*.2,controls,resistance_degree=1,velocity_degree=1)
    owner=material.compile(**options(d,device,precision))
    q=np.full((7,d),.2);q[:,0]+=[0,4,8,0,4,8,30]
    h=np.linspace(.7,1.,len(q));u=rng.normal(size=q.shape);m=rng.uniform(.1,.4,len(q));L=rng.normal(size=(len(q),d,d))
    batch=owner.prepare(q,h)
    reference=batch.apply(u,m,L) if affine else batch.apply(u,m)
    result=owner.apply_streamed(q,h,u,m,L if affine else None,chunk_size=3)
    for name in ['fluid_force','reaction','resistance','skeleton_product','dissipation']+(['gradient_load','affine_forcing'] if affine else []):
        np.testing.assert_allclose(getattr(result,name),getattr(reference,name),atol=tol,rtol=tol)
    work=np.sum(u*result.fluid_force)+np.sum(controls*result.reaction)
    if affine:work+=np.sum(L*result.gradient_load)
    assert abs(work+result.dissipation)/(1+abs(result.dissipation))<tol
    assert result.dissipation>=0
    assert result.execution['chunks']==3
    assert result.execution['peak_owned_batch_bytes']<batch.bytes
    again=owner.apply_streamed(q,h,u,m,L if affine else None,chunk_size=3)
    np.testing.assert_array_equal(result.reaction,again.reaction)
    np.testing.assert_array_equal(result.fluid_force,again.fluid_force)
    empty=owner.apply_streamed([],1,np.empty((0,d)),1,np.empty((0,d,d)) if affine else None)
    assert empty.fluid_force.shape==(0,d) and empty.execution['chunks']==0
    np.testing.assert_array_equal(empty.reaction,0)
    assert empty.dissipation==0


@pytest.mark.parametrize('device',DEVICES)
def test_explicit_rule_budget_is_bounded_without_fallback(device,monkeypatch):
    from warpSPHBoundaries.fluxbridge import device_panels
    g=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    ev=Evaluator(g,baseline(3),**options(3,device,'float32'))
    q=.2+np.random.default_rng(42).uniform(-.01,.01,(32,3));reference=ev.evaluate(q,1)
    monkeypatch.setattr(device_panels,'MAX_RULE_BYTES',8*2**20)
    with pytest.raises(ValueError,match='storage exceeds'):ev.prepare(q,1)
    value,execution=ev.evaluate_streamed(q,1,chunk_size=1,return_metrics=True)
    np.testing.assert_allclose(value,reference,atol=1e-5,rtol=1e-5)
    assert execution['peak_owned_batch_bytes']<8*2**20
    assert execution['chunks']==32 and execution['precision']=='float32'
    monkeypatch.setattr(device_panels,'MAX_RULE_BYTES',1)
    with pytest.raises(ValueError,match=r'stream chunk \[0:1\].*storage exceeds'):
        ev.evaluate_streamed(q,1,chunk_size=1)


@pytest.mark.parametrize('device',DEVICES)
def test_ordinary_periodic_disk_streaming(device):
    from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
    ev=DiskArrayEvaluator([[0,0],[.8,0]],[.2,.15],device=device,precision='float32',search='device')
    q=np.array([[.25,.1],[.75,.2],[2.25,-1.9]])
    np.testing.assert_allclose(ev.evaluate_streamed(q,.3,periods=(2,2),chunk_size=2),
        ev.evaluate(q,.3,periods=(2,2)),atol=1e-5,rtol=1e-5)


def test_streaming_input_validation_and_geometry_revision():
    g=Geometry([[[0,0],[2,0],[0,2]]],2);ev=Evaluator(g,baseline(2))
    for size in [0,-1,1.5,True,2**31]:
        with pytest.raises(ValueError,match='chunk_size'):ev.iter_evaluate([[.2,.2]],1,chunk_size=size)
    with pytest.raises(ValueError,match='periods'):ev.evaluate_streamed([],1,periods=(-1,0))
    iterator=ev.iter_evaluate([[.2,.2],[.3,.3]],1,chunk_size=1);next(iterator);g.revision='changed'
    with pytest.raises(ValueError,match='geometry changed'):next(iterator)
    g=Geometry([[[0,0],[2,0],[0,2]]],2)
    material=ResistanceMaterial(g,[[np.eye(2)]],np.zeros((1,3,2))).compile(precision='float32',method='conditioned_boundary')
    with pytest.raises(ValueError,match='storage range'):material.apply_streamed([[.2,.2]],1,[[1e40,0]],1)
    with pytest.raises(ValueError,match='velocity_gradient'):material.apply_streamed([[.2,.2]],1,[[0,0]],1,np.zeros((1,2,2)))
    affine=AffineResistanceMaterial(g,[[np.eye(2)]],np.zeros((1,3,2))).compile()
    with pytest.raises(ValueError,match='velocity_gradient'):affine.apply_streamed([[.2,.2]],1,[[0,0]],1)
