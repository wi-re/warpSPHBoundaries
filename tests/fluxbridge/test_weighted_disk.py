"""Independent whole-circle references versus native interior-volume nodes."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import WeightedDiskEvaluator,Request,baseline
from warpSPHBoundaries.fluxbridge.disk import DiskQuadratureEvaluator

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])
CASES=json.loads((Path(__file__).parents[1]/'fixtures/fluxbridge-weighted-disk-v1.json').read_text())['cases']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,budget',[('float32',1e-5),('float64',1e-14)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_full_degree_nine_jets_against_whole_circle_reference(device,precision,budget,kernel):
    for c in CASES:
        if c['kernel']!=kernel:continue
        e=WeightedDiskEvaluator([Request(**r) for r in c['requests']],kernel=kernel,precision=precision,device=device)
        b=e.prepare(float(c['distance']),c['normal'],1,radii=float(c['radius']))
        value=b.numpy()[0];expected=np.array(c['values'],float)
        error=np.max(abs(value-expected)/(1+abs(expected)))
        assert np.isfinite(value).all() and error<budget,(c['label'],error,b.quadrature)
        b.launch();np.testing.assert_array_equal(b.numpy()[0],value)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float32',1e-5),('float64',1e-14)])
def test_units_subsets_freeze_empty_and_baseline_reduction(device,precision,tol):
    n=[.6,.8];d=np.array([-.5,-.2,0,.5,1]);h=np.ones(5);r=np.full(5,.5)
    e=WeightedDiskEvaluator(kernel='w6',precision=precision,device=device)
    b=e.prepare(d,n,h,radii=r);value=b.numpy().copy();d[:]=0;h[:]=2;r[:]=1;b.launch()
    np.testing.assert_array_equal(b.numpy(),value)
    original_d=np.array([-.5,-.2,0,.5,1])
    circle=DiskQuadratureEvaluator(kernel='w6',precision=precision,device=device)
    np.testing.assert_allclose(value,circle.evaluate(original_d,n,1,radii=.5),atol=1e-5 if precision=='float32' else 1e-10,rtol=tol)
    req=[Request((4,5),(0,1)),Request((9,0)),Request((0,0)),Request((4,5),(0,1))]
    sparse=WeightedDiskEvaluator(req,precision=precision,device=device)
    x=sparse.evaluate(original_d,n,1,radii=.5)
    y=sparse.evaluate(original_d*3,n,3,radii=1.5)
    np.testing.assert_allclose(y/np.array([3.**r.scale for r in req]),x,atol=tol,rtol=tol)
    np.testing.assert_array_equal(x[:,0],x[:,-1]);np.testing.assert_array_equal(x[-1],0)
    assert e.evaluate([],n,1,radii=.5).shape==(0,15)
    assert WeightedDiskEvaluator([],device=device).evaluate(original_d,n,1,radii=.5).shape==(5,0)
    centre=e.evaluate(-.5,[1,0],1,radii=.5)
    np.testing.assert_allclose(centre,value[:1],atol=tol,rtol=tol)


@pytest.mark.parametrize('device',DEVICES)
def test_complete_query_derivatives_polynomial_work_and_composed_call(device):
    alpha=(3,2);requests=[Request(alpha),Request(alpha,(0,)),Request(alpha,(1,)),Request((2,2)),Request((3,1))]
    e=WeightedDiskEvaluator(requests,device=device);q=np.array([.35,-.2]);radius=.3;eps=1e-5
    def moments(p):
        length=np.linalg.norm(p);return e.evaluate(length-radius,p/length,1,radii=radius)[0]
    value=moments(q)
    for j in range(2):
        offset=np.eye(2)[j]*eps;diff=(moments(q+offset)[0]-moments(q-offset)[0])/(2*eps)
        np.testing.assert_allclose(diff,value[1+j]-alpha[j]*value[3+j],atol=1e-10,rtol=2e-7)
    coeff=np.arange(15).reshape(3,5)/13;loads=np.array([.7,-.1,.3])
    np.testing.assert_allclose(np.dot(coeff@value,loads),np.dot(value,coeff.T@loads),atol=1e-14)
    b=e.prepare(0,[0,1],1,radii=.5);expected=b.numpy()[0]
    api=e.api;Plan,Rules,real,vec=api.Plan,api.Rules,api.real,api.vec;element,output=api.element,api.output
    @wp.kernel
    def consumer(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec),work:wp.array2d(dtype=real),out:wp.array(dtype=real)):
        element(plan,rules,scratch,work,0)
        out[0]=output(plan,work,0,0)+real(.2)*output(plan,work,0,1)
    out=wp.empty(1,dtype=real,device=device)
    wp.launch(consumer,1,inputs=[e.plan,b.rules,b.scratch,b.work,out],device=device)
    np.testing.assert_allclose(out.numpy()[0],expected[0]+.2*expected[1],atol=1e-14)


@pytest.mark.skipif(not wp.is_cuda_available(),reason='CUDA hardware unavailable')
def test_frozen_cuda_graph():
    e=WeightedDiskEvaluator([Request((9,0)),Request((2,3),(0,1))],precision='float32',device='cuda:0')
    b=e.prepare([-.5,0,.5,1],[0,1],1,radii=.5);expected=b.numpy().copy()
    with wp.ScopedCapture(device='cuda:0') as capture:b.launch()
    b.work.fill_(123);wp.capture_launch(capture.graph);np.testing.assert_array_equal(b.numpy(),expected)


def test_invalid_inputs_and_explicit_resource_rejection(monkeypatch):
    from warpSPHBoundaries.fluxbridge import weighted_disk
    for kwargs in [dict(precision='float32_compensated'),dict(nodes=7),dict(kernel='bad'),dict(requests=[Request((0,0,0))])]:
        with pytest.raises(ValueError):WeightedDiskEvaluator(**kwargs)
    e=WeightedDiskEvaluator()
    for d,n,h,r in [(-1,[0,1],1,.5),(0,[0,2],1,.5),(0,[0,1],0,.5),(0,[0,1],1,-1)]:
        with pytest.raises(ValueError):e.prepare(d,n,h,radii=r)
    with pytest.raises(ValueError,match='conditioning'):e.prepare(0,[0,1],1,radii=1e-6)
    with pytest.raises(ValueError,match='higher comparison'):WeightedDiskEvaluator(nodes=64).prepare(0,[0,1],1,radii=.5)
    monkeypatch.setattr(weighted_disk,'MAX_BATCH_BYTES',1)
    with pytest.raises(ValueError,match='storage'):e.prepare(0,[0,1],1,radii=.5)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,h',[('float64',8e-155),('float32',6e-20)])
def test_physical_storage_overflow_rejected(device,precision,h):
    with pytest.raises(ValueError,match='physical output.*storage'):
        WeightedDiskEvaluator(kernel='w6',precision=precision,device=device).prepare(0,[0,1],h,radii=.5*h)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float32',1e-5),('float64',1e-14)])
@pytest.mark.parametrize('search',['host','device'])
def test_weighted_periodic_fibre_arrays_exhaustive_and_streamed(device,precision,tol,search):
    from itertools import product
    from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
    centres=np.array([[0,0],[.9,.3]]);radii=np.array([.2,.15])
    requests=[Request((0,0)),Request((4,5)),Request((3,2),(0,1))]
    q=np.array([[.1,.1],[1.8,.1],[2.1,-1.9]]);h=.8
    e=DiskArrayEvaluator(centres,radii,requests,precision=precision,device=device,search=search,method='volume_quadrature')
    value=e.evaluate(q,h,periods=(2,2))
    exhaustive=e.evaluate(q,h,periods=(2,2),exhaustive=True)
    np.testing.assert_allclose(value,exhaustive,atol=tol,rtol=tol)
    oracle=WeightedDiskEvaluator(requests,precision=precision,device=device)
    expected=np.zeros_like(value)
    for centre,radius in zip(centres,radii):
        for image in product((-2,0,2),repeat=2):
            delta=q-centre-image;length=np.linalg.norm(delta,axis=1)
            expected+=oracle.evaluate(length-radius,delta/length[:,None],h,radii=radius)
    np.testing.assert_allclose(value,expected,atol=tol,rtol=tol)
    streamed,execution=e.evaluate_streamed(q,h,periods=(2,2),chunk_size=2,return_metrics=True)
    np.testing.assert_allclose(streamed,value,atol=tol,rtol=tol)
    assert execution['requested_nodes']==24 and execution['accepted_rule_samples']>0
    assert execution['comparison_rule_samples']>execution['accepted_rule_samples']
    assert sum(execution['quadrature_interactions_by_order'].values())==execution['total_interactions']
    assert not hasattr(e,'resistance')  # Resolved solid fibres have no effective porous drag.


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float32',1e-5),('float64',1e-14)])
def test_all_whole_support_polynomial_moments_and_kernel_integration_by_parts(device,precision,tol):
    from fractions import Fraction as F
    from math import factorial
    from warpSPHBoundaries.fluxbridge import monomials
    from warpSPHBoundaries.edge.kernels import KERNELS
    derivatives=[(),(0,),(1,),(0,0),(0,1),(1,1)]
    requests=[Request(a,ds) for a in monomials(2,9) for ds in derivatives]
    def moment(alpha):
        if min(alpha)<0 or any(a%2 for a in alpha):return F(0)
        i,j=(a//2 for a in alpha);degree=sum(alpha)
        angular=F(2*factorial(2*i)*factorial(2*j),4**(i+j)*factorial(i)*factorial(j)*factorial(i+j))
        radial=sum(c*(hi**(p+degree+2)-lo**(p+degree+2))/(p+degree+2)
            for lo,hi,coeff in KERNELS['w6'].pieces for p,c in enumerate(coeff))
        return angular*radial
    reference=[]
    for r in requests:
        alpha=list(r.alpha);factor=1
        for j in r.derivative:factor*=alpha[j];alpha[j]-=1
        reference.append(float(factor*moment(alpha)))
    e=WeightedDiskEvaluator(requests,kernel='w6',precision=precision,device=device)
    h=.7;value=e.evaluate(-2,[.6,.8],h,radii=2)[0]/np.array([h**r.scale for r in requests])
    expected=np.array(reference)
    assert np.max(abs(value-expected)/(1+abs(expected)))<tol
