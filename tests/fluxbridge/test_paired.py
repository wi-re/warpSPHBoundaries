import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge.paired import arithmetic

DEVICES=["cpu"]+(["cuda:0"] if wp.is_cuda_available() else [])


def split(values):
    values=np.asarray(values,dtype=np.float64)
    high=values.astype(np.float32)
    return np.stack([high,(values-high.astype(np.float64)).astype(np.float32)],axis=-1)


@pytest.mark.parametrize('op',range(8))
@pytest.mark.parametrize("device",DEVICES)
def test_paired_arithmetic_primitives(op,device):
    rng=np.random.default_rng(41)
    a=np.r_[np.exp(rng.uniform(-12,12,200)),[1.,2.,.25,1e-14]]
    if op in (0,1,2,3,6,7):
        a[::3]*=-1
    b=np.exp(rng.uniform(-8,8,len(a)))
    b[::4]*=-1
    operations=[np.add,np.subtract,np.multiply,np.divide,lambda a,b:np.sqrt(a),lambda a,b:np.log(a),np.arctan2,lambda a,b:np.arcsinh(a)]
    ref=operations[op](a,b)
    out=wp.empty(len(a),dtype=wp.vec2f,device=device)
    wp.launch(arithmetic,len(a),inputs=[wp.array(split(a),dtype=wp.vec2f,device=device),wp.array(split(b),dtype=wp.vec2f,device=device),op,out],device=device)
    values=out.numpy().astype(np.float64).sum(axis=-1)
    np.testing.assert_allclose(values,ref,rtol=2e-12,atol=1e-13)

@pytest.mark.parametrize('compact',[False,True])
@pytest.mark.parametrize("device",DEVICES)
def test_compensated_fixtures(compact,device):
    import json
    from pathlib import Path
    from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request
    cases=json.loads((Path(__file__).parents[1]/'fixtures/fluxbridge-v1.json').read_text())['cases']
    worst=0.
    for case in cases:
        req=[Request(**r) for r in case['requests']]
        ev=Evaluator(Geometry([case['vertices']],case['dimension'],case['measure']),req,
                     kernel=case['kernel'],precision='float32_compensated',compact=compact,device=device)
        value=ev.evaluate([case['query']],case['h'],exhaustive=True)[0]
        ref=np.array(case['values'],float)
        assert np.isfinite(value).all(),case
        err=np.max(np.abs(value-ref)/(1+np.abs(ref)))
        worst=max(worst,float(err))
        assert err<1e-5,(case['dimension'],case['kernel'],case['query'],err)
    print('Worst compensated mixed error:',worst)

@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('degree',[0,1,2,3])
@pytest.mark.parametrize("device",DEVICES)
def test_compensated_fused_fields_and_transpose(dimension,degree,device):
    from warpSPHBoundaries.fluxbridge import Geometry,PolynomialField,FieldEvaluator
    d=dimension
    primitive=np.vstack([np.zeros(d),2*np.eye(d)])
    geometry=Geometry([primitive,primitive+np.array([4.]+[0.]*(d-1))],d)
    field=PolynomialField.basis(geometry,degree)
    queries=np.array([[.2]*d,[.3]*d,[4.2]+[.2]*(d-1)])
    supports=[1.,.8,1.]
    ev=FieldEvaluator(field,precision='float32_compensated',kernel='w6',device=device)
    batch=ev.prepare(queries,supports)
    batch.launch()
    value=batch.numpy()
    ref=FieldEvaluator(field,kernel='w6',device=device).evaluate(queries,supports)
    np.testing.assert_allclose(value,ref,atol=1e-5,rtol=1e-5)
    loads=np.random.default_rng(17).normal(size=value.shape)
    volumes=np.array([.7,.3,.6])
    trans=batch.transpose(loads,volumes)
    coefficients=ev.coefficients.numpy().astype(np.float64).sum(axis=-1)
    left=np.sum(value*loads*volumes[:,None])
    right=np.sum(coefficients*trans)
    np.testing.assert_allclose(left,right,atol=1e-5,rtol=1e-5)
    # Same geometry and support, only requested kernel derivatives change.
    for derivative in [(0,),(0,1)]:
        out=FieldEvaluator(field,precision='float32_compensated',kernel='w6',derivative=derivative,device=device).evaluate(queries,supports)
        ref=FieldEvaluator(field,kernel='w6',derivative=derivative,device=device).evaluate(queries,supports)
        assert np.max(np.abs(out-ref)/(1+np.abs(ref)))<1e-5


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize("device",DEVICES)
def test_compensated_degree_nine_material_work(dimension,device):
    from math import comb
    from warpSPHBoundaries.fluxbridge import Geometry,ResistanceMaterial
    d=dimension
    geometry=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])],d)
    n=comb(3+d,d)
    rng=np.random.default_rng(22)
    factors=rng.normal(size=(1,n,d,d))
    resistance=factors@factors.swapaxes(-1,-2)+np.eye(d)*.2
    controls=rng.normal(size=(1,n,d))
    material=ResistanceMaterial(geometry,resistance,controls,resistance_degree=3,velocity_degree=3)
    u=rng.normal(size=(1,d))
    batch=material.prepare([[.2]*d],1,precision='float32_compensated',kernel='w6',device=device)
    result=batch.apply(u,[.7])
    ref=material.prepare([[.2]*d],1,kernel='w6',device=device).apply(u,[.7])
    np.testing.assert_allclose(result.fluid_force,ref.fluid_force,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.reaction,ref.reaction,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.dissipation,ref.dissipation,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=1e-5)
    power=np.sum(result.fluid_force*u)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(power,-result.dissipation,atol=1e-5,rtol=1e-5)
    assert result.dissipation>=0


def test_compensated_precision_range_rejections_and_large_asinh():
    from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request
    ev=Evaluator(Geometry([[[0,0],[1,0],[0,1]]],2),[Request((0,0))],precision='float32_compensated')
    for support in (1e-50,1e40):
        with pytest.raises(ValueError,match='fp32 storage range'):
            ev.prepare([[0,0]],support)
    a=np.array([1e20,-1e25])
    out=wp.empty(2,dtype=wp.vec2f,device='cpu')
    wp.launch(arithmetic,2,inputs=[wp.array(split(a),dtype=wp.vec2f,device='cpu'),wp.zeros(2,dtype=wp.vec2f,device='cpu'),7,out],device='cpu')
    np.testing.assert_allclose(out.numpy().astype(np.float64).sum(axis=-1),np.arcsinh(a),rtol=2e-12)


def test_compensated_element_local_coordinates_and_periodic_multiple_interactions():
    from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,PolynomialField,FieldEvaluator
    primitive=np.array([[0.,0.],[2.,0.],[0.,2.]])
    geometry=Geometry([primitive,primitive+np.array([1e6,0.])],2)
    queries=[[.2,.2],[1e6+.2,.2]]
    for kernel in ('w2','w4','w6','cubic'):
        ev=Evaluator(geometry,baseline(2),precision='float32_compensated',kernel=kernel)
        value=ev.evaluate(queries,1)
        ref=Evaluator(geometry,baseline(2),kernel=kernel).evaluate(queries,1)
        assert np.max(np.abs(value-ref)/(1+np.abs(ref)))<1e-5
    cell=Geometry([[[0,0],[1,0],[1,1],[0,1]]],2)
    value=Evaluator(cell,[Request((0,0))],precision='float32_compensated',compact=True).evaluate([[.1,.2],[3.1,-4.8]],1.5,periods=(1,1))
    np.testing.assert_allclose(value,1,atol=1e-5)
    tris=Geometry([[[0,0],[1,0],[0,1]],[[1,1],[0,1],[1,0]]],2)
    field=PolynomialField.constant(tris,[1.,3.])
    ev=FieldEvaluator(field,precision='float32_compensated')
    value=ev.evaluate([[.2,.2],[.5,.5]],1)
    ref=FieldEvaluator(field).evaluate([[.2,.2],[.5,.5]],1)
    np.testing.assert_allclose(value,ref,atol=1e-5)
