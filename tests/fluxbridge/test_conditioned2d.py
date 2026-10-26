"""Explicit numerical profile route: raw-fp32 gates, locality and work."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,PolynomialField,FieldEvaluator,ResistanceMaterial,AffineResistanceMaterial
from warpSPHBoundaries.fluxbridge.box import BoxEvaluator
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])
FIXTURES=Path(__file__).parents[1]/'fixtures'


def cases(family,kernel):
    data=json.loads((FIXTURES/f'fluxbridge-{family}.json').read_text())['cases']
    return [c for c in data if c['kernel']==kernel and
            (c['kind']=='box' if family=='diskbox-v1' else c['dimension']==2 and c.get('measure','volume')=='volume')]


def evaluate(c,family,precision,nodes,device):
    req=[Request(**r) for r in c['requests']]
    kw=dict(kernel=c['kernel'],precision=precision,method='conditioned_boundary',nodes=nodes,device=device)
    if family=='v1':return Evaluator(Geometry([c['vertices']],2),req,**kw).evaluate([c['query']],c['h'])[0]
    if family=='diskbox-v1':return BoxEvaluator([.2,-.05],[.7,.65],req,**kw).evaluate([c['query']],1)[0]
    return PlanarEvaluator(c['kind'],2,req,**kw).evaluate(float(c['distance']),c['normal'],1,
        **({} if c['half_width'] is None else dict(half_widths=float(c['half_width']))))[0]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,budget',[('float64',1e-10),('float32',1e-5)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
@pytest.mark.parametrize('family',['v1','diskbox-v1','planar-v1'])
def test_frozen_full_weighted_contract(device,precision,budget,kernel,family):
    for c in cases(family,kernel):
        v=evaluate(c,family,precision,16,device);ref=np.array(c['values'],float)
        assert np.isfinite(v).all()
        assert np.max(abs(v-ref)/(1+abs(ref)))<=budget


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_convergence_complete_derivatives_and_units(device,kernel):
    geo=Geometry([[[0,0],[2,0],[0,2]]],2);req=baseline(2)
    e=Evaluator(geo,req,kernel=kernel,method='conditioned_boundary',device=device)
    q=np.array([[.2,.3],[0,0],[1,0],[1e-10,.3]])
    low=e.evaluate(q,1)
    high=Evaluator(geo,req,kernel=kernel,method='conditioned_boundary',nodes=32,device=device).evaluate(q,1)
    np.testing.assert_allclose(low,high,atol=1e-10,rtol=1e-10)
    ids={r:i for i,r in enumerate(req)};eps=2e-5
    for j in range(2):
        delta=np.eye(2)[j:j+1]*eps
        diff=(e.evaluate(q[:1]+delta,1)-e.evaluate(q[:1]-delta,1))/(2*eps)
        np.testing.assert_allclose(diff[0,0],low[0,ids[Request((0,0),(j,))]],atol=1e-8,rtol=1e-7)
        for i in range(2):
            a=tuple(int(k==i) for k in range(2))
            expected=low[0,ids[Request(a,(j,))]]-(low[0,0] if i==j else 0)
            np.testing.assert_allclose(diff[0,ids[Request(a)]],expected,atol=1e-8,rtol=1e-7)
            np.testing.assert_allclose(diff[0,ids[Request((0,0),(i,))]],low[0,ids[Request((0,0),tuple(sorted((i,j))))]],atol=1e-8,rtol=1e-7)
    for precision,tol in (('float64',1e-10),('float32',1e-5)):
        base=Evaluator(geo,req,kernel=kernel,precision=precision,method='conditioned_boundary',device=device)
        b=base.prepare(q,1);b.work.fill_(1234);b.launch();a=b.numpy();b.launch()
        np.testing.assert_array_equal(a,b.numpy())
        scaled=Evaluator(Geometry([np.array(geo.elements[0])*3],2),req,kernel=kernel,precision=precision,
                         method='conditioned_boundary',device=device).evaluate(q*3,3)
        np.testing.assert_allclose(scaled,a*np.array([3.**r.scale for r in req]),atol=tol,rtol=tol)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('degree',range(4))
def test_fields_periodic_locality_and_transpose(device,degree):
    geo=Geometry([np.array([[0,0],[2,0],[0,2]])+[4*i,0] for i in range(20)],2)
    f=PolynomialField.basis(geo,degree);q=np.array([[.2,.2],[100,.2],[4.2,.2]])
    e=FieldEvaluator(f,kernel='w6',precision='float32',method='conditioned_boundary',device=device)
    b=e.prepare(q,1);b.launch();ref=FieldEvaluator(f,kernel='w6',device=device).evaluate(q,1)
    np.testing.assert_allclose(b.numpy(),ref,atol=1e-5,rtol=1e-5)
    full=e.prepare(q,1,exhaustive=True);full.launch()
    np.testing.assert_allclose(b.numpy(),full.numpy(),atol=1e-5,rtol=1e-5)
    loads=np.random.default_rng(8).normal(size=ref.shape);v=np.array([.7,.2,.4])
    transfer=b.transpose(loads,v)
    np.testing.assert_allclose(np.sum(b.numpy()*loads*v[:,None]),np.sum(e.coefficients.numpy()*transfer),atol=1e-5,rtol=1e-5)
    tile=Geometry([[[0,0],[1,0],[1,1],[0,1]]],2)
    ev=Evaluator(tile,baseline(2),precision='float32',method='conditioned_boundary',device=device)
    a=ev.evaluate([[.1,.2],[3.1,-4.8]],1.5,periods=(1,1))
    np.testing.assert_allclose(a[0],a[1],atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(a[:,0],1,atol=1e-5)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('affine',[False,True])
def test_cubic_material_degree_nine_common_work(device,affine):
    geo=Geometry([[[0,0],[2,0],[0,2]]],2);rng=np.random.default_rng(52)
    factors=rng.normal(size=(1,10,2,2));D=factors@factors.swapaxes(-1,-2)+np.eye(2)*.2
    controls=rng.normal(size=(1,10,2));cls=AffineResistanceMaterial if affine else ResistanceMaterial
    material=cls(geo,D,controls,resistance_degree=3,velocity_degree=3)
    q=[[.2,.2]];u=rng.normal(size=(1,2));gradient=rng.normal(size=(1,2,2));args=[gradient] if affine else []
    result=material.prepare(q,1,kernel='w6',precision='float32',method='conditioned_boundary',device=device).apply(u,[.7],*args)
    ref=material.prepare(q,1,kernel='w6',device=device).apply(u,[.7],*args)
    np.testing.assert_allclose(result.fluid_force,ref.fluid_force,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.reaction,ref.reaction,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.dissipation,ref.dissipation,atol=1e-5,rtol=1e-5)
    work=np.sum(result.fluid_force*u)+np.sum(result.reaction*controls)
    if affine:work+=np.sum(result.gradient_load*gradient)
    np.testing.assert_allclose(work,-result.dissipation,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-5)
    assert result.dissipation>=0


def test_explicit_limits_and_empty():
    from warpSPHBoundaries.fluxbridge.fields import field_functions
    from warpSPHBoundaries.fluxbridge.planar import functions as planar_functions
    with pytest.raises(ValueError):field_functions('float32_compensated','conditioned_boundary')
    with pytest.raises(ValueError):planar_functions(3,'float32','conditioned_boundary')
    for geo in (Geometry([[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]],3),Geometry([[[0,0,0],[1,0,0],[0,1,0]]],3,'surface')):
        with pytest.raises(ValueError):Evaluator(geo,baseline(3),method='conditioned_boundary')
    geo=Geometry([[[0,0],[1,0],[0,1]]],2)
    for opts in (dict(precision='float32_compensated'),dict(nodes=7),dict(nodes=65)):
        with pytest.raises(ValueError):Evaluator(geo,baseline(2),method='conditioned_boundary',**opts)
    for p in ('float64','float32'):
        assert Evaluator(Geometry([],2),baseline(2),precision=p,method='conditioned_boundary').evaluate([[0,0]],1).sum()==0
        assert Evaluator(geo,[],precision=p,method='conditioned_boundary').evaluate([[0,0]],1).shape==(1,0)
    with pytest.raises(ValueError,match='support scaling'):
        Evaluator(geo,baseline(2),precision='float32',method='conditioned_boundary').evaluate([[100,100]],1e-30)


@pytest.mark.parametrize('device',DEVICES)
def test_supported_contacts_nonconvexity_and_disjoint_polynomial_rows(device):
    from math import comb
    shapes=[np.array([[0,0],[2,0],[0,2]]),np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]])]
    for vertices in shapes:
        q=np.array([[.2,.2],[0,0],[1,0],[1.8,1.8],[4,4],[2+1e-6,0],[2-1e-6,0]])
        geo=Geometry([vertices],2)
        ev=Evaluator(geo,baseline(2),kernel='w6',precision='float32',method='conditioned_boundary',device=device)
        b=ev.prepare(q,1,exhaustive=True);b.launch()
        ref=Evaluator(geo,baseline(2),kernel='w6',device=device).evaluate(q,1,exhaustive=True)
        np.testing.assert_allclose(b.numpy(),ref,atol=1e-5,rtol=1e-5)
        # Disjoint rows must stay exactly zero before a degree-nine field's
        # large query-origin translation can amplify exterior flux roundoff.
        f=PolynomialField(geo,[{(9,0):np.array(1.)}])
        value=FieldEvaluator(f,precision='float32',method='conditioned_boundary',device=device).evaluate([[100,100]],1,exhaustive=True)
        np.testing.assert_array_equal(value,0)
        shift=2**24
        moved=Evaluator(Geometry([vertices+shift],2),baseline(2),kernel='w6',precision='float32',method='conditioned_boundary',device=device)
        np.testing.assert_allclose(moved.evaluate(q+shift,1,exhaustive=True),b.numpy(),atol=1e-5,rtol=1e-5)


@pytest.mark.skipif(not wp.is_cuda_available(),reason='CUDA hardware unavailable')
def test_conditioned_cuda_graph():
    e=Evaluator(Geometry([[[0,0],[2,0],[0,2]]],2),baseline(2),precision='float32',method='conditioned_boundary',device='cuda:0')
    b=e.prepare([[.2,.2]],1);b.launch();reference=b.numpy()
    with wp.ScopedCapture(device='cuda:0') as capture:b.launch()
    wp.capture_launch(capture.graph);np.testing.assert_array_equal(b.numpy(),reference)
