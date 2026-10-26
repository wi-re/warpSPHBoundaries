import numpy as np
import pytest
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request
from warpSPHBoundaries.fluxbridge.fields import PolynomialField,FieldEvaluator


@pytest.mark.parametrize("dimension",[2,3])
@pytest.mark.parametrize("degree",[0,1,2,3])
def test_fem_reproduction_and_weighted_transpose(dimension,degree):
    geo=Geometry([np.vstack([np.zeros(dimension),2*np.eye(dimension)])],dimension)
    basis=PolynomialField.basis(geo,degree)
    ev=FieldEvaluator(basis)
    q=np.array([[.2]*dimension,[.3]*dimension])
    batch=ev.prepare(q,1)
    batch.launch()
    val=batch.numpy()
    mass=Evaluator(geo,[Request((0,)*dimension)]).evaluate(q,1)[:,0]
    np.testing.assert_allclose(val.sum(axis=1),mass,atol=1e-10)
    rng=np.random.default_rng(7)
    loads=rng.normal(size=val.shape)
    volumes=np.array([.4,.7])
    trans=batch.transpose(loads,volumes)
    coeff=ev.coefficients.numpy()
    np.testing.assert_allclose(np.sum(val*loads*volumes[:,None]),np.sum(coeff*trans),rtol=1e-10,atol=1e-10)
    if degree:
        # A global linear field is reproduced independently of the basis order.
        from warpSPHBoundaries.mesh3d.fem import expansion
        verts=np.pad(geo.elements[0],((0,0),(0,3-dimension))).tolist()
        nodes,_=expansion(verts,verts[0],degree)
        controls=np.array([[float(v[0]+2*v[1]) for v in nodes]])
        field=PolynomialField.from_controls(geo,controls,degree,kind="lagrange")
        out=FieldEvaluator(field).evaluate(q,1)
        refs=Evaluator(geo,[Request((1,)+(0,)*(dimension-1)),Request((0,1)+(0,)*(dimension-2)),Request((0,)*dimension)]).evaluate(q,1)
        ref=refs[:,0]+2*refs[:,1]+(q[:,0]+2*q[:,1])*refs[:,2]
        np.testing.assert_allclose(out,ref,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("dimension",[2,3])
def test_degree_nine_mixed_product_and_kernel_gradient(dimension):
    geo=Geometry([np.vstack([np.zeros(dimension),2*np.eye(dimension)])],dimension)
    zero=(0,)*dimension
    cubic=PolynomialField(geo,[{(3,)+(0,)*(dimension-1):np.array(1.)}])
    product=cubic.product(cubic).product(cubic)
    q=np.full((1,dimension),.2)
    val=FieldEvaluator(product).evaluate(q,1)[0]
    from math import comb
    req=[Request((k,)+(0,)*(dimension-1)) for k in range(10)]
    out=Evaluator(geo,req).evaluate(q,1)[0]
    ref=sum(comb(9,k)*.2**(9-k)*out[k] for k in range(10))
    np.testing.assert_allclose(val,ref,atol=1e-10,rtol=1e-10)
    g=FieldEvaluator(product,derivative=(0,)).evaluate(q,1)[0]
    delta=np.zeros_like(q); delta[0,0]=1e-5
    diff=(FieldEvaluator(product).evaluate(q+delta,1)[0]-FieldEvaluator(product).evaluate(q-delta,1)[0])/(2e-5)
    np.testing.assert_allclose(g,diff,atol=1e-9,rtol=1e-7)

@pytest.mark.parametrize("dimension",[2,3])
def test_vector_differential_index_order_and_hessian_trace(dimension):
    from warpSPHBoundaries.fluxbridge.fields import differential
    d=dimension
    geo=Geometry([np.vstack([np.zeros(d),2*np.eye(d)])],d)
    field=PolynomialField.constant(geo,[np.arange(1,d+1)])
    q=np.full((1,d),.2)
    result=differential(field,q,1,hessian=True)
    scalar=differential(PolynomialField.constant(geo,[1]),q,1,hessian=True)
    np.testing.assert_allclose(result['gradient'],np.arange(1,d+1)[None,:,None]*scalar['gradient'][:,None,:],atol=1e-10)
    np.testing.assert_allclose(result['divergence'],np.einsum('i,ni->n',np.arange(1,d+1),scalar['gradient']),atol=1e-10)
    np.testing.assert_allclose(result['laplacian'],np.arange(1,d+1)[None,:]*scalar['laplacian'][:,None],atol=1e-10)


def test_discontinuous_material_traces_keep_internal_face_contributions():
    geometry=Geometry([[[0,0],[1,0],[0,1]],[[1,1],[0,1],[1,0]]],2)
    field=PolynomialField.constant(geometry,[1.,3.])
    queries=[[.5,.5],[.2,.3]]
    out=FieldEvaluator(field).evaluate(queries,1)
    first=Evaluator(Geometry([geometry.elements[0]],2),[Request((0,0))]).evaluate(queries,1)[:,0]
    second=Evaluator(Geometry([geometry.elements[1]],2),[Request((0,0))]).evaluate(queries,1)[:,0]
    np.testing.assert_allclose(out,first+3*second,atol=1e-10)
    homogeneous=FieldEvaluator(PolynomialField.constant(geometry,[1.,1.])).evaluate(queries,1)
    boundary=Evaluator(Geometry([[[0,0],[1,0],[1,1],[0,1]]],2),[Request((0,0))]).evaluate(queries,1)[:,0]
    np.testing.assert_allclose(homogeneous,boundary,atol=1e-10)


def test_exact_zero_coefficients_do_not_request_unused_high_moments():
    geometry=Geometry([[[0,0],[1,0],[0,1]]],2)
    field=PolynomialField(geometry,[{(0,0):np.array([1.,2.]),(9,0):np.zeros(2)}])
    assert field.degree==0
    zero=PolynomialField(geometry,[{(9,0):np.zeros((2,2))}])
    assert zero.degree==0 and zero.shape==(2,2)
    tiny=PolynomialField(geometry,[{(9,0):np.array(1e-100)}])
    assert tiny.degree==9
    np.testing.assert_array_equal(FieldEvaluator(zero).evaluate([[.2,.2]],1),np.zeros((1,2,2)))
