"""Whole-element search preserves traces, containment and periodic ordering."""
from itertools import product
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import (Geometry, DeviceSearch, adjacency,
    Evaluator, Request, baseline, PolynomialField, FieldEvaluator, AffineResistanceMaterial)

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


def tuples(a):
    return list(zip(a.query.tolist(),a.element.tolist(),map(tuple,a.shift)))


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('periodic',[False,True])
def test_random_frozen_csr_against_host_and_brute(device,dimension,periodic):
    rng=np.random.default_rng(851)
    g=Geometry([np.vstack([np.zeros(dimension),np.eye(dimension)])*.4+x
        for x in rng.uniform(-3,3,(87,dimension))],dimension)
    q=rng.uniform(-4,4,(39,dimension));h=rng.uniform(.1,1.1,len(q))
    periods=(5.,)*(dimension-1)+(0.,) if periodic else None
    search=DeviceSearch(g,device=device); a=search.prepare(q,h,periods=periods)
    ref=adjacency(g,q,h,periods=periods)
    assert tuples(a)==tuples(ref)
    np.testing.assert_array_equal(a.offsets,ref.offsets)
    assert len(tuples(a))==len(set(tuples(a)))
    assert tuples(search.prepare(q,h,periods=periods))==tuples(a)
    assert tuples(search.prepare(q[::-1],h[::-1],periods=periods))==tuples(adjacency(g,q[::-1],h[::-1],periods=periods))
    # Independent enumeration uses all element/image boxes, no hierarchy.
    brute=[]
    images=list(product(*[range(-2,3) if L else [0] for L in (periods or (0.,)*dimension)]))
    for qi,(x,r) in enumerate(zip(q,h)):
        for ei,bounds in enumerate(g.bounds):
            for image in images:
                shift=np.array(image)*(periods or (0.,)*dimension)
                delta=np.maximum(np.maximum(bounds[0]-(x-shift),(x-shift)-bounds[1]),0)/r
                if np.dot(delta,delta)<=1:brute.append((qi,ei,tuple(shift)))
    assert tuples(a)==sorted(brute)
    for array in (a.query,a.element,a.shift,a.offsets):assert not array.flags.writeable
    with pytest.raises(ValueError,match='stale'):a.validate(g,q+.01,h)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dimension',[2,3])
def test_touching_slivers_translation_and_containment(device,dimension):
    # Keep whole elements even when every facet is outside support.
    v=np.vstack([np.full(dimension,-10.),np.eye(dimension)*60-10])
    small=np.vstack([np.zeros(dimension),np.eye(dimension)])*.25+20
    sliver=np.vstack([np.zeros(dimension),np.eye(dimension)])
    sliver[-1,-1]=1e-10
    for shift in [0.,2.**40]:
        g=Geometry([v+shift,small+shift,(sliver if shift==0 else np.vstack([np.zeros(dimension),np.eye(dimension)]))+shift],dimension)
        q=np.array([np.zeros(dimension),np.ones(dimension)*20,np.ones(dimension)*1000])+shift
        a=DeviceSearch(g,device=device).prepare(q,[.1,.1,1])
        assert tuples(a)==tuples(adjacency(g,q,[.1,.1,1]))
        assert (0,0,(0.,)*dimension) in tuples(a)
    g=Geometry([np.vstack([np.zeros(dimension),np.eye(dimension)])],dimension)
    q=np.zeros((6,dimension));q[:,0]=[-1.,np.nextafter(-1.,-np.inf),-1-1e-5,2.,2+1e-5,0.]
    a=DeviceSearch(g,device=device).prepare(q,1.)
    assert set(tuples(adjacency(g,q,1)))<=set(tuples(a))
    assert not set(a.query)&{2,4}


@pytest.mark.parametrize('device',DEVICES)
def test_closed_skin_cavity_and_standalone_surface(device):
    from scipy.spatial import ConvexHull
    cube=np.array(list(product([-10.,10.],repeat=3)));hull=ConvexHull(cube)
    faces=[]
    for f,equation in zip(hull.simplices,hull.equations):
        f=list(f);a,b,c=cube[f]
        if np.dot(np.cross(b-a,c-a),equation[:3])<0:f[1],f[2]=f[2],f[1]
        faces.append(f)
    # An inner closed skin has opposite orientation and defines a cavity.
    vertices=np.vstack([cube,cube*.1]);faces+= [[i+8 for i in f[::-1]] for f in faces.copy()]
    g=Geometry.closed_surface(vertices,faces)
    q=np.array([[0,0,0],[3,0,0],[100,0,0]])
    a=DeviceSearch(g,device=device).prepare(q,.2)
    assert a.query.tolist()==[0,1]
    requests=[Request((0,0,0))]
    ev=Evaluator(g,requests,device=device,search='device')
    np.testing.assert_allclose(ev.evaluate(q,.2),ev.evaluate(q,.2,exhaustive=True),atol=1e-10)
    np.testing.assert_allclose(ev.evaluate(q,.2).ravel(),[0,1,0],atol=1e-10)
    surface=Geometry([[[0,0,0],[1,0,0],[0,1,0]]],3,'surface')
    assert tuples(DeviceSearch(surface,device=device).prepare(q,.2))==tuples(adjacency(surface,q,.2))


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
def test_evaluator_fields_and_material_consume_search(device,precision,tol):
    v=np.vstack([np.zeros(3),2*np.eye(3)])
    g=Geometry([v+np.array([i*4,0,0]) for i in range(3)],3)
    q=np.array([[.2,.2,.2],[4.2,.2,.2],[30,30,30]])
    opts=dict(device=device,precision=precision,method='tet_quadrature',preparation='device',search='device')
    ev=Evaluator(g,baseline(3),**opts);a=ev.evaluate(q,1);b=ev.evaluate(q,1,exhaustive=True)
    np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
    field=PolynomialField.basis(g,2);fe=FieldEvaluator(field,**opts);fb=fe.prepare(q,1);fb.launch()
    np.testing.assert_allclose(fb.numpy(),FieldEvaluator(field,**{**opts,'search':'host'}).evaluate(q,1),atol=tol,rtol=tol)
    loads=np.random.default_rng(8).normal(size=fb.values.shape);vol=np.array([.2,.3,.5]);transpose=fb.transpose(loads,vol)
    np.testing.assert_allclose(np.sum(fb.numpy()*loads*vol[:,None]),np.sum(fe.coefficients.numpy()*transpose),atol=tol,rtol=tol)
    rng=np.random.default_rng(39);controls=rng.normal(size=(3,4,3));resistance=np.stack([np.diag([1.,2.,3.])*(i+1) for i in range(3)])[:,None]
    material=AffineResistanceMaterial(g,resistance,controls,resistance_degree=0,velocity_degree=1)
    compiled=material.compile(**opts);batch=compiled.prepare(q,1)
    u=np.tile([1.,2.,3.],(3,1));L=np.zeros((3,3,3));m=np.array([.2,.3,.5])
    result=batch.apply(u,m,L);reference=material.prepare(q,1,**{**opts,'search':'host'}).apply(u,m,L)
    for name in ['fluid_force','gradient_load','reaction','dissipation']:
        np.testing.assert_allclose(getattr(result,name),getattr(reference,name),atol=tol,rtol=tol)
    if wp.get_device(device).is_cuda:
        real=batch.batch.evaluator.api.real;ud=wp.array(u,dtype=real,device=device);ld=wp.array(L,dtype=real,device=device);md=wp.array(m,dtype=real,device=device)
        batch.apply_device(ud,md,ld);expected=batch.force.numpy()
        with wp.ScopedCapture(device=device) as capture:batch.apply_device(ud,md,ld)
        batch.force.fill_(123);wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(batch.force.numpy(),expected)


@pytest.mark.parametrize('device',DEVICES)
def test_periodic_multiple_images_and_tiled_cell(device):
    g=Geometry([[[0,0],[1,0],[1,1],[0,1]]],2)
    q=np.array([[.1,.2],[3.1,-4.8]])
    s=DeviceSearch(g,device=device);a=s.prepare(q,1.5,periods=(1,1))
    expected=set(tuples(adjacency(g,q,1.5,periods=(1,1))))
    assert expected<=set(tuples(a))
    assert tuples(a)==sorted(set(tuples(a)))
    for qi,ei,shift in set(tuples(a))-expected:
        delta=np.maximum(np.maximum(g.bounds[ei,0]-(q[qi]-shift),(q[qi]-shift)-g.bounds[ei,1]),0)/1.5
        assert abs(np.dot(delta,delta)-1)<2e-14
    ev=Evaluator(g,[Request((0,0))],device=device,search='device')
    np.testing.assert_allclose(ev.evaluate(q,1.5,periods=(1,1)),1,atol=1e-10)


@pytest.mark.parametrize('device',DEVICES)
def test_empty_and_explicit_resource_rejection(device,monkeypatch):
    from warpSPHBoundaries.fluxbridge import device_search
    g=Geometry([[[0,0],[1,0],[0,1]]],2);s=DeviceSearch(g,device=device)
    assert s.prepare([],1).offsets.tolist()==[0]
    assert DeviceSearch(Geometry([],3),device=device).prepare([[0,0,0]],1).offsets.tolist()==[0,0]
    with pytest.raises(ValueError,match='periods'):s.prepare([[0,0]],1,periods=(-1,1))
    with pytest.raises(ValueError,match='4096'):s.prepare([[0,0]],100,periods=(1,1))
    with pytest.raises(ValueError,match='integer'):s.prepare([[1e20,0]],1,periods=(1,0))
    with pytest.raises(ValueError,match='search'):Evaluator(g,[Request((0,0))],search='invalid')
    with pytest.raises(ValueError,match='box range'):
        DeviceSearch(Geometry([[[0,0],[1e38,0],[0,1e38]]],2),device=device)
    monkeypatch.setattr(device_search,'MAX_SEARCH_BYTES',1)
    with pytest.raises(ValueError,match='storage exceeds'):s.prepare([[0,0]],1)
    g.revision='changed'
    with pytest.raises(ValueError,match='stale'):s.prepare([[0,0]],1)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-10),('float32',1e-5)])
def test_ordinary_solid_disks_and_periodic_fibres(device,precision,tol):
    from warpSPHBoundaries.fluxbridge.disks import DiskArrayEvaluator
    centres=[[0.,0.],[.8,0.]];radii=[.2,.15]
    options=dict(precision=precision,device=device,nodes=24)
    device_array=DiskArrayEvaluator(centres,radii,search='device',**options)
    host_array=DiskArrayEvaluator(centres,radii,**options)
    q=np.array([[.25,.1],[.75,.2],[2.25,-1.9],[10,10]])
    np.testing.assert_allclose(device_array.evaluate(q,.3),host_array.evaluate(q,.3,exhaustive=True),atol=tol,rtol=tol)
    q=q[:3]
    value=device_array.evaluate(q,.3,periods=(2,2))
    np.testing.assert_allclose(value,host_array.evaluate(q,.3,periods=(2,2),exhaustive=True),atol=tol,rtol=tol)
    np.testing.assert_allclose(value[0],value[2],atol=tol,rtol=tol)
    assert device_array.device_search.last_metrics['host_csr_readback']


@pytest.mark.parametrize('device',DEVICES)
def test_two_dimensional_differential_reuses_explicit_search(device,monkeypatch):
    from warpSPHBoundaries.fluxbridge import differential
    from warpSPHBoundaries.fluxbridge import fields
    g=Geometry([[[0,0],[2,0],[0,2]],[[4,0],[6,0],[4,2]]],2)
    field=PolynomialField.constant(g,[[1.,2.],[3.,4.]])
    q=np.array([[.2,.2],[4.2,.2]])
    options=dict(device=device,method='conditioned_boundary',precision='float32')
    reference=differential(field,q,1,hessian=True,**options)
    def forbidden(*args,**kwargs):raise AssertionError('device differential dispatched host adjacency')
    monkeypatch.setattr(fields,'adjacency',forbidden)
    from warpSPHBoundaries.fluxbridge import evaluate
    monkeypatch.setattr(evaluate,'adjacency',forbidden)
    actual=differential(field,q,1,search='device',hessian=True,**options)
    for k in reference:np.testing.assert_array_equal(actual[k],reference[k])
