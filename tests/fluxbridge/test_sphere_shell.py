"""Weighted spherical geometry, independent cap fixtures and composed consumers."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import SphereShellEvaluator,Request,baseline
from warpSPHBoundaries.fluxbridge.radial import RadialEvaluator

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])
CASES=json.loads((Path(__file__).parents[1]/'fixtures/fluxbridge-sphere-v1.json').read_text())['cases']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,budget',[('float32',1e-5),('float64',1e-14)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_degree_nine_and_second_derivatives_against_independent_caps(device,precision,budget,kernel):
    for case in CASES:
        if case['kernel']!=kernel:continue
        evaluator=SphereShellEvaluator([Request(**r) for r in case['requests']],kernel=kernel,precision=precision,device=device)
        b=evaluator.prepare(float(case['distance']),case['normal'],1,radii=float(case['radius']))
        b.launch();value=b.numpy()[0];expected=np.array(case['values'],float)
        error=np.max(abs(value-expected)/(1+abs(expected)))
        assert np.isfinite(value).all() and error<budget,(case['label'],error,b.quadrature)
        assert b.quadrature['worst_mixed_difference']<=budget
        b.work.fill_(123);b.launch();np.testing.assert_array_equal(b.numpy()[0],value)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float32',1e-5),('float64',1e-14)])
def test_baseline_units_centre_subsets_empty_and_polynomial_work(device,precision,tol):
    normal=np.array([2/3,-1/3,2/3]);d=np.array([-.5,-.2,0.,.5,1.]);radius=.5
    evaluator=SphereShellEvaluator(kernel='w6',precision=precision,device=device)
    value=evaluator.evaluate(d,normal,1,radii=radius)
    analytic=RadialEvaluator('sphere',kernel='w6',precision=precision,device=device)
    np.testing.assert_allclose(value,analytic.evaluate(d,normal,1,radii=radius),atol=tol,rtol=tol)
    scaled=evaluator.evaluate(d*3,normal,3,radii=radius*3)
    np.testing.assert_allclose(scaled,value*np.array([3.**r.scale for r in evaluator.requests]),atol=tol,rtol=tol)
    rotated=evaluator.evaluate(-radius,[0,0,1],1,radii=radius)
    np.testing.assert_allclose(rotated,value[:1],atol=tol,rtol=tol)
    req=[Request((3,3,3)),Request((2,3,4),(0,2)),Request((0,0,0)),Request((3,3,3))]
    subset=SphereShellEvaluator(req,precision=precision,device=device)
    x=subset.evaluate(d,normal,1,radii=radius)
    np.testing.assert_array_equal(x[:,0],x[:,-1]);np.testing.assert_array_equal(x[-1],0)
    coeff=np.array([[1,.2,-.1,.3],[.4,-.2,.5,1.]])
    load=np.arange(10).reshape(5,2)/7
    np.testing.assert_allclose(np.sum((x@coeff.T)*load),np.sum(x*(load@coeff)),atol=tol,rtol=tol)
    assert evaluator.evaluate([],normal,1,radii=radius).shape==(0,28)
    assert SphereShellEvaluator([],device=device).evaluate(d,normal,1,radii=radius).shape==(5,0)


@pytest.mark.parametrize('device',DEVICES)
def test_query_derivatives_and_composed_native_function(device):
    q=np.array([.4,-.2,.3]);radius=.35;eps=1e-5
    e=SphereShellEvaluator(device=device);ids={r:i for i,r in enumerate(e.requests)}
    def evaluate(p):
        length=np.linalg.norm(p)
        return e.evaluate(length-radius,p/length,1,radii=radius)[0]
    value=evaluate(q)
    for j in range(3):
        offset=np.eye(3)[j]*eps;diff=(evaluate(q+offset)-evaluate(q-offset))/(2*eps)
        np.testing.assert_allclose(diff[0],value[ids[Request((0,0,0),(j,))]],atol=2e-8,rtol=2e-7)
    alpha=(3,2,1)
    lower=[tuple(a-(i==j) for i,a in enumerate(alpha)) for j in range(3)]
    requests=[Request(alpha),*[Request(alpha,(j,)) for j in range(3)],*[Request(a) for a in lower]]
    weighted=SphereShellEvaluator(requests,device=device)
    def moments(p):
        length=np.linalg.norm(p)
        return weighted.evaluate(length-radius,p/length,1,radii=radius)[0]
    weighted_value=moments(q)
    for j in range(3):
        offset=np.eye(3)[j]*eps
        complete=(moments(q+offset)[0]-moments(q-offset)[0])/(2*eps)
        np.testing.assert_allclose(complete,weighted_value[1+j]-alpha[j]*weighted_value[4+j],atol=1e-10,rtol=2e-7)
    e=SphereShellEvaluator([Request((0,0,0)),Request((3,3,3))],device=device)
    b=e.prepare(0.,[0,0,1],1,radii=.5);b.launch();expected=b.numpy()[0]
    api=e.api;real,vec,Plan=api.real,api.vec,api.Plan
    element,output=api.element,api.output
    @wp.kernel
    def consume(plan:Plan,indices:wp.array2d(dtype=int),facets:wp.array2d(dtype=vec),
                work:wp.array2d(dtype=real),out:wp.array(dtype=real)):
        element(real(0),real(.5),vec(real(0),real(0),real(1)),plan,indices,facets,work,0)
        out[0]=output(plan,work,0,0)+real(.2)*output(plan,work,0,1)
    out=wp.empty(1,dtype=real,device=device)
    wp.launch(consume,1,inputs=[e.plan,e.indices,b.facets,b.work,out],device=device)
    np.testing.assert_allclose(out.numpy()[0],expected[0]+.2*expected[1],atol=1e-14)


@pytest.mark.skipif(not wp.is_cuda_available(),reason='CUDA hardware unavailable')
def test_frozen_cuda_graph():
    e=SphereShellEvaluator(precision='float32',device='cuda:0')
    b=e.prepare([-.5,0,.5,1],[0,0,1],1,radii=.5);b.launch();expected=b.numpy()
    with wp.ScopedCapture(device='cuda:0') as capture:b.launch()
    b.work.fill_(123);wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(b.numpy(),expected)


def test_invalid_contract_and_resource_rejection(monkeypatch):
    from warpSPHBoundaries.fluxbridge import sphere_shell
    for kwargs in [dict(precision='float32_compensated'),dict(nodes=7),dict(kernel='bad'),dict(requests=[Request((0,0))])]:
        with pytest.raises(ValueError):SphereShellEvaluator(**kwargs)
    e=SphereShellEvaluator()
    for distance,normal,h,r in [(-1,[0,0,1],1,.5),(0,[0,0,2],1,.5),(0,[0,0,1],0,.5),(0,[0,0,1],1,-1)]:
        with pytest.raises(ValueError):e.prepare(distance,normal,h,radii=r)
    with pytest.raises(ValueError,match='storage'):SphereShellEvaluator(precision='float32').prepare(1,[0,0,1],1e-40,radii=.5)
    monkeypatch.setattr(sphere_shell,'MAX_BATCH_BYTES',1)
    with pytest.raises(ValueError,match='budget'):e.prepare(0,[0,0,1],1,radii=.5)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,h',[('float64',1e-154),('float32',7e-20)])
def test_finite_normalized_values_cannot_hide_physical_overflow(device,precision,h):
    e=SphereShellEvaluator(kernel='w6',precision=precision,device=device)
    # h^-2 is representable, but the physical Hessian times that scale is not.
    with pytest.raises(ValueError,match='physical output.*storage'):
        e.prepare(0.,[0,0,1],h,radii=h*.5)
