import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import SolidEvaluator


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('precision',['float64','float32'])
def test_search_assembly_and_periodic_images(device,d,precision):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    v=np.vstack([np.zeros(d),np.eye(d)])
    vertices=np.array([v,v+2,v+10])
    q=np.array([np.full(d,.25),np.full(d,2.125),np.full(d,30.),np.zeros(d)])
    h=np.array([.6,1.2,.5,1.])
    evaluator=SolidEvaluator(vertices,device=device,precision=precision,bundle='wcsph_extended')
    fast=evaluator.prepare(q,h).evaluate();full=evaluator.prepare(q,h,exhaustive=True).evaluate()
    assert fast.pair_count<full.pair_count
    tolerance=1e-10 if precision=='float64' else 1e-5
    assert np.max(abs(fast.aggregate.numpy()-full.aggregate.numpy()))<tolerance
    assert np.all(fast.aggregate.numpy()[:,2]==0)
    first=fast.aggregate.numpy();assert np.array_equal(first,fast.evaluate().aggregate.numpy())
    period=(4.,)*d
    periodic=evaluator.prepare(q,h,periods=period).evaluate()
    shifted=evaluator.prepare(q+np.array(period),h,periods=period).evaluate()
    assert np.max(abs(periodic.aggregate.numpy()-shifted.aggregate.numpy()))<tolerance
    with pytest.raises(ValueError):evaluator.prepare(q+.1,h,adj=fast.adjacency)
    with pytest.raises(ValueError):evaluator.prepare(q,h,adj=fast.adjacency,periods=period)
    gpu_search=SolidEvaluator(vertices,device=device,precision=precision,bundle='wcsph_extended',search='device')
    from_device=gpu_search.prepare(q,h).evaluate()
    assert np.array_equal(from_device.adjacency.element,fast.adjacency.element)
    assert np.max(abs(from_device.aggregate.numpy()-first))<tolerance


@pytest.mark.parametrize('d',[2,3])
def test_solid_support_interior_and_empty(d):
    # Mesh the large box around the particle using a Delaunay decomposition.
    from itertools import product
    from scipy.spatial import Delaunay
    vertices=np.array(list(product((-3.,3.),repeat=d)))
    mesh=vertices[Delaunay(vertices).simplices]
    e=SolidEvaluator(mesh,bundle='wcsph_extended')
    b=e.prepare([np.zeros(d)],1).evaluate();value=b.aggregate.numpy()[:,0]
    target=np.zeros(len(value));target[0]=1
    for j,name in enumerate(b.labels):
        if name in ('Covxx','Covyy','Covzz'):target[j]=1
    assert np.max(abs(value-target))<1e-10
    empty=e.prepare([],[]).evaluate();assert empty.aggregate.shape==(len(b.labels),0)
    for bad in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):e.prepare([np.zeros(d)],bad)
