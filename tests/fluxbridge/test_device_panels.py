"""Independent acceptance of device-generated angular geometry and shared caches."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,AffineResistanceMaterial,PolynomialField,FieldEvaluator

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])
FIXTURES=Path(__file__).parents[1]/'fixtures'


def tetra():return Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,budget',[('float64',1e-14),('float32',1e-5)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_device_preparation_against_frozen_reference(device,precision,budget,kernel):
    cases=json.loads((FIXTURES/'fluxbridge-v1.json').read_text())['cases']
    cases=[c for c in cases if c['dimension']==3 and c['measure']=='volume']
    cases+=json.loads((FIXTURES/'fluxbridge-tetquad-v1.json').read_text())['cases']
    cases+=json.loads((FIXTURES/'fluxbridge-nearplane-v1.json').read_text())['cases']
    for c in cases:
        if c['kernel']!=kernel:continue
        ev=Evaluator(Geometry([c['vertices']],3),[Request(**r) for r in c['requests']],kernel=kernel,
            precision=precision,device=device,method='tet_quadrature',preparation='device')
        b=ev.prepare([c['query']],c['h'],exhaustive=True);b.launch()
        reference=np.array(c['values'],float);value=b.numpy()[0]
        scales=np.array([float(c['h'])**r.scale for r in ev.host_plan.requests]);value=value/scales;reference=reference/scales
        assert np.isfinite(value).all()
        assert max(abs(value-reference)/(1+abs(reference)))<budget,(c.get('label',c['query']),kernel,precision)
        assert b.quadrature['preparation']=='device' and b.quadrature['geometry_precision']=='float64'


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
def test_device_rule_batches_fields_periodicity_and_reuse(device,precision,tol):
    g=tetra();q=np.array([[.2,.2,.2],[0,0,0],[3,3,3]])
    options=dict(kernel='w6',method='tet_quadrature',precision=precision,device=device)
    ev=Evaluator(g,baseline(3),preparation='device',**options)
    b=ev.prepare(q,1,exhaustive=True);b.launch();expected=b.numpy()
    np.testing.assert_allclose(expected,Evaluator(g,baseline(3),**options).evaluate(q,1,exhaustive=True),atol=tol,rtol=tol)
    b.integration_plan.profiles.fill_(float('nan'));b.work.fill_(123);b.launch()
    np.testing.assert_array_equal(b.numpy(),expected)
    np.testing.assert_array_equal(expected[-1],0)
    assert ev.prepare([],1).numpy().shape==(0,len(baseline(3)))
    empty=Evaluator(g,[],preparation='device',**options).prepare(q,1);empty.launch();assert empty.numpy().shape==(3,0)
    np.testing.assert_allclose(ev.evaluate([[.1,.1,.1],[2.1,-1.9,4.1]],.3,periods=(2,2,2))[0],
        ev.evaluate([[.1,.1,.1],[2.1,-1.9,4.1]],.3,periods=(2,2,2))[1],atol=tol,rtol=tol)
    field=PolynomialField.basis(g,3)
    fe=FieldEvaluator(field,preparation='device',**options);fb=fe.prepare(q,1);fb.launch()
    np.testing.assert_allclose(fb.numpy(),FieldEvaluator(field,**options).evaluate(q,1),atol=tol,rtol=tol)
    loads=np.random.default_rng(33).normal(size=fb.values.shape);volumes=np.array([.3,.2,.1])
    transfer=fb.transpose(loads,volumes)
    np.testing.assert_allclose(np.sum(fb.numpy()*loads*volumes[:,None]),np.sum(fe.coefficients.numpy()*transfer),atol=tol,rtol=tol)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
def test_real_material_device_preparation_work_and_graph(device,precision,tol):
    g=tetra();rng=np.random.default_rng(52);f=rng.normal(size=(1,20,3,3));controls=rng.normal(size=(1,20,3))
    m=AffineResistanceMaterial(g,f@f.swapaxes(-1,-2)+np.eye(3)*.2,controls,resistance_degree=3,velocity_degree=3)
    q=np.array([[.2,.2,.2],[.21,.19,.18]]);u=rng.normal(size=q.shape);L=rng.normal(size=(len(q),3,3));mass=np.full(len(q),.3)
    options=dict(kernel='w6',method='tet_quadrature',precision=precision,device=device)
    b=m.prepare(q,1,preparation='device',**options);result=b.apply(u,mass,L)
    reference=m.prepare(q,1,**options).apply(u,mass,L)
    for key in ['fluid_force','gradient_load','reaction','dissipation']:
        np.testing.assert_allclose(getattr(result,key),getattr(reference,key),atol=tol,rtol=tol)
    work=np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls)
    assert abs(work+result.dissipation)/(1+abs(result.dissipation))<tol
    assert result.dissipation>=0
    if wp.get_device(device).is_cuda:
        real=b.batch.evaluator.api.real;ud=wp.array(u,dtype=real,device=device);Ld=wp.array(L,dtype=real,device=device);md=wp.array(mass,dtype=real,device=device)
        b.apply_device(ud,md,Ld);arrays=(b.force,b.gradient_load,b.reaction,b.power);values=[a.numpy() for a in arrays]
        with wp.ScopedCapture(device=device) as capture:b.apply_device(ud,md,Ld)
        for a in arrays:a.fill_(123)
        wp.capture_launch(capture.graph)
        for a,value in zip(arrays,values):np.testing.assert_array_equal(a.numpy(),value)


def test_device_preparation_contract_and_resource_policy(monkeypatch):
    from warpSPHBoundaries.fluxbridge import device_panels
    with pytest.raises(ValueError,match='tetrahedral'):
        Evaluator(tetra(),baseline(3),preparation='device')
    ev=Evaluator(tetra(),baseline(3),method='tet_quadrature',preparation='device')
    monkeypatch.setattr(device_panels,'MAX_RULE_BYTES',1)
    with pytest.raises(ValueError,match='storage exceeds'):
        ev.prepare([[.2]*3],1)
