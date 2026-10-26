"""Shared monomial contractions retain frozen jets, units and transpose work."""
from functools import lru_cache
from itertools import product
import json
from math import comb,prod
from pathlib import Path
import mpmath as mp
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,PolynomialField,FieldEvaluator,AffineResistanceMaterial,Request,monomials

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


@lru_cache(None)
def frozen_weights(kernel,derivative):
    cases=json.loads((Path(__file__).parents[1]/'fixtures/fluxbridge-v1.json').read_text())['cases']
    case=next(c for c in cases if c['dimension']==3 and c['measure']=='volume' and c['kernel']==kernel and len(c['requests'])==2200)
    with mp.workdps(70):
        moments={Request(**r):mp.mpf(v) for r,v in zip(case['requests'],case['values'])}
        result=[]
        for alpha in monomials(3,9):
            result.append(float(mp.fsum(mp.mpf(prod(comb(a,b) for a,b in zip(alpha,beta)))*
                mp.mpf('.2')**sum(a-b for a,b in zip(alpha,beta))*moments[Request(beta,derivative)]
                for beta in product(*(range(a+1) for a in alpha)))))
    return np.array(result)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
@pytest.mark.parametrize('kernel',['w2','w6'])
@pytest.mark.parametrize('derivative',[(),(0,1)])
@pytest.mark.parametrize('scale',[1.,.7])
@pytest.mark.parametrize('nodes',[8,16])
def test_shared_degree_nine_integrals_match_frozen_jets(device,precision,tol,kernel,derivative,scale,nodes):
    alphas=monomials(3,9);eye=np.eye(len(alphas))
    geo=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])*scale],3)
    field=PolynomialField(geo,[dict(zip(alphas,eye))])
    owner=FieldEvaluator(field,derivative=derivative,kernel=kernel,method='tet_quadrature',
        preparation='device',precision=precision,device=device,nodes=nodes)
    b=owner.prepare([[.2*scale]*3],scale);b.launch();value=b.numpy()
    expected=frozen_weights(kernel,derivative)*np.array([scale**(sum(a)-len(derivative)) for a in alphas])
    assert max(abs(value[0]-expected)/(1+abs(expected)))<tol
    # The correction workspace is refreshed, not a cache of completed values.
    b.field_weights.fill_(float('nan'));b.field_weight_errors.fill_(float('nan'));b.launch()
    np.testing.assert_array_equal(b.numpy(),value)
    loads=np.random.default_rng(6).normal(size=value.shape)
    transferred=b.transpose(loads,[.7])
    np.testing.assert_allclose(np.sum(value*loads*.7),np.sum(owner.coefficients.numpy()*transferred),atol=tol,rtol=tol)
    assert b.field_weight_errors.dtype==b.work.dtype==owner.coefficients.dtype
    if precision=='float32':assert b.work.dtype==wp.float32
    b.evaluation_queries.fill_(owner.evaluator.api.vec(9.))
    assert np.isnan(b.transpose(loads,[.7])).all()
    b.launch();assert np.isnan(b.numpy()).all()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('nodes',[8,16])
def test_p3_product_regression_keeps_fp32_operands_and_budget(device,nodes):
    # This element/control/query combination exposed loss from rounding shared
    # translated weights before projection. It must not regress to that path.
    rng=np.random.default_rng(52);f=rng.normal(size=(16,20,3,3));u=rng.normal(size=(16,20,3))
    geometry=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    material=AffineResistanceMaterial(geometry,(f@f.swapaxes(-1,-2)+np.eye(3)*.2)[8:9],u[8:9],
        resistance_degree=3,velocity_degree=3)
    query_rng=np.random.default_rng(852);q=.2+query_rng.uniform(-.015,.015,(32,3));q=q[[24]]
    values=[]
    for precision in ['float64','float32']:
        batch=material.prepare(q,1,precision=precision,device=device,method='tet_quadrature',preparation='device',nodes=nodes).batch
        batch.launch();values.append(batch.numpy())
    assert np.max(abs(values[0]-values[1])/(1+abs(values[0])))<1e-5
    assert batch.field_weight_errors.dtype==wp.float32
    assert np.any(batch.field_weight_errors.numpy()!=0)


def test_field_storage_overflow_rejected_before_launch():
    g=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    field=PolynomialField.constant(g,[1e40])
    with pytest.raises(ValueError,match='field coefficients exceed finite fp32'):
        FieldEvaluator(field,precision='float32',method='tet_quadrature')
