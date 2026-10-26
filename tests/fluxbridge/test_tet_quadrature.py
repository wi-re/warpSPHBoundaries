"""Independent accuracy and real material-consumer gates for tetrahedral quadrature."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import (Geometry,Evaluator,Request,baseline,PolynomialField,
    FieldEvaluator,AffineResistanceMaterial)

DEVICES=["cpu"]+(["cuda:0"] if wp.is_cuda_available() else [])
FIXTURES=Path(__file__).parents[1]/"fixtures"


def tetra(): return Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,budget",[("float64",1e-14),("float32",1e-5)])
@pytest.mark.parametrize("kernel",["w2","w4","w6","cubic"])
def test_frozen_full_degree_nine_and_adversarial_cases(device,precision,budget,kernel):
    main=json.loads((FIXTURES/"fluxbridge-v1.json").read_text())["cases"]
    extra=json.loads((FIXTURES/"fluxbridge-tetquad-v1.json").read_text())["cases"]
    extra+=json.loads((FIXTURES/'fluxbridge-nearplane-v1.json').read_text())['cases']
    selected=[c for c in main if c["dimension"]==3 and c["measure"]=="volume"]+extra
    for c in selected:
        if c["kernel"]!=kernel:continue
        req=[Request(**r) for r in c["requests"]]
        ev=Evaluator(Geometry([c["vertices"]],3),req,kernel=kernel,precision=precision,
            method="tet_quadrature",device=device)
        b=ev.prepare([c["query"]],c["h"],exhaustive=True);b.launch()
        value=b.numpy()[0];ref=np.array(c["values"],float)
        assert np.isfinite(value).all()
        scales=np.array([float(c['h'])**r.scale for r in req]);value=value/scales;ref=ref/scales
        assert np.max(abs(value-ref)/(1+abs(ref))) < budget,(c.get("label",c["query"]),value,ref)
        assert b.quadrature["worst_mixed_difference"] <= budget


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,tol",[("float64",1e-12),("float32",1e-5)])
def test_reuse_units_subsets_empty_and_stale_prepared_queries(device,precision,tol):
    geo=tetra();req=baseline(3)
    ev=Evaluator(geo,req,kernel="w6",precision=precision,method="tet_quadrature",device=device)
    q=np.array([[.2,.2,.2],[0,0,0],[1,0,0],[3,3,3]])
    b=ev.prepare(q,1,exhaustive=True);b.work.fill_(123);b.launch();value=b.numpy()
    b.launch();np.testing.assert_array_equal(b.numpy(),value)
    other=ev.prepare([[.3]*3],1);other.launch();b.launch();np.testing.assert_array_equal(b.numpy(),value)
    scaled=Evaluator(Geometry([geo.elements[0]*3],3),req,kernel="w6",precision=precision,
        method="tet_quadrature",device=device).evaluate(q*3,3,exhaustive=True)
    np.testing.assert_allclose(scaled,value*np.array([3.**r.scale for r in req]),atol=tol,rtol=tol)
    indices=[-1,0,4,-1]
    subset=Evaluator(geo,[req[i] for i in indices],kernel="w6",precision=precision,method="tet_quadrature",device=device)
    np.testing.assert_allclose(subset.evaluate(q,1,exhaustive=True),value[:,indices],atol=tol,rtol=tol)
    assert ev.evaluate([],1).shape==(0,len(req))
    assert Evaluator(geo,[],method="tet_quadrature",device=device).evaluate(q,1).shape==(4,0)
    np.testing.assert_array_equal(Evaluator(Geometry([],3),req,method="tet_quadrature",device=device).evaluate(q,1),0)
    b.evaluation_queries.fill_(ev.api.vec(9.))
    b.launch();assert np.isnan(b.numpy()).all()


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("degree",range(4))
def test_fields_transpose_locality_and_disjoint_rows(device,degree):
    geo=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])+[4*i,0,0] for i in range(4)],3)
    field=PolynomialField.basis(geo,degree)
    ev=FieldEvaluator(field,kernel="w6",method="tet_quadrature",device=device)
    q=np.array([[.2,.2,.2],[4.2,.2,.2],[100,100,100]])
    b=ev.prepare(q,1);b.launch();value=b.numpy()
    full=ev.prepare(q,1,exhaustive=True);full.launch()
    np.testing.assert_allclose(full.numpy(),value,atol=1e-12,rtol=1e-12)
    reference=FieldEvaluator(field,kernel="w6",device=device).evaluate(q,1)
    np.testing.assert_allclose(value,reference,atol=1e-10,rtol=1e-10)
    loads=np.random.default_rng(8).normal(size=value.shape);mass=np.array([.7,.3,.2])
    transferred=b.transpose(loads,mass)
    np.testing.assert_allclose(np.sum(value*loads*mass[:,None]),
        np.sum(ev.coefficients.numpy()*transferred),atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(value[-1],0)
    remote=FieldEvaluator(PolynomialField(tetra(),[{(9,0,0):np.array(1.)}]),
        method="tet_quadrature",device=device).evaluate([[1e100]*3],1,exhaustive=True)
    np.testing.assert_array_equal(remote,0)


@pytest.mark.parametrize("device",DEVICES)
def test_periodic_images_and_complete_derivatives(device):
    geo=tetra();req=baseline(3)
    ev=Evaluator(geo,req,kernel="w6",method="tet_quadrature",device=device)
    q=np.array([[.1,.1,.1],[2.1,-1.9,4.1]])
    a=ev.evaluate(q,.3,periods=(2,2,2));np.testing.assert_allclose(a[0],a[1],atol=1e-10,rtol=1e-10)
    reference=Evaluator(geo,req,kernel="w6",device=device).evaluate(q,.3,periods=(2,2,2))
    np.testing.assert_allclose(a,reference,atol=1e-9,rtol=1e-9)
    q=np.array([[.2,.2,.2]]);value=ev.evaluate(q,1)[0];eps=2e-5
    ids={r:i for i,r in enumerate(req)}
    for j in range(3):
        diff=(ev.evaluate(q+np.eye(3)[j]*eps,1)-ev.evaluate(q-np.eye(3)[j]*eps,1))[0]/(2*eps)
        np.testing.assert_allclose(diff[0],value[ids[Request((0,0,0),(j,))]],atol=1e-8,rtol=1e-7)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,tol",[("float64",1e-10),("float32",1e-5)])
def test_real_degree_nine_affine_material_work_and_torque(device,precision,tol):
    geo=tetra();rng=np.random.default_rng(52)
    factors=rng.normal(size=(1,20,3,3));D=factors@factors.swapaxes(-1,-2)+np.eye(3)*.2
    controls=rng.normal(size=(1,20,3));u=rng.normal(size=(1,3));L=rng.normal(size=(1,3,3))
    material=AffineResistanceMaterial(geo,D,controls,resistance_degree=3,velocity_degree=3)
    q=np.array([[.2]*3])
    b=material.prepare(q,1,kernel="w6",precision=precision,method="tet_quadrature",device=device)
    result=b.apply(u,[.7],L)
    reference=material.prepare(q,1,kernel="w6",device=device).apply(u,[.7],L)
    for name in ("fluid_force","gradient_load","reaction","dissipation"):
        np.testing.assert_allclose(getattr(result,name),getattr(reference,name),atol=tol,rtol=tol)
    work=np.sum(result.fluid_force*u)+np.sum(result.gradient_load*L)+np.sum(result.reaction*controls)
    np.testing.assert_allclose(work,-result.dissipation,atol=tol,rtol=tol)
    assert result.dissipation>=-tol
    np.testing.assert_allclose(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)),0,atol=tol)
    from warpSPHBoundaries.mesh3d.fem import expansion
    nodes,_=expansion(geo.elements[0].tolist(),[0,0,0],3)
    np.testing.assert_allclose(result.torque_defect(q,np.array(nodes,float)[None]),0,atol=tol)
    assert b.batch.quadrature["field_mixed_difference"]<=b.batch.quadrature["field_budget"]


def test_promotion_and_unresolved_convergence(monkeypatch):
    from warpSPHBoundaries.fluxbridge import tet_quadrature as tq
    ev=Evaluator(tetra(),baseline(3),kernel="w6",method="tet_quadrature",nodes=8)
    b=ev.prepare([[.2]*3],1);assert b.quadrature["promotions"]>0
    def bad_rule(facets,query,h,kernel,nodes):
        return np.array([[0.,0.,1.]]),np.array([1.]),np.array([1.+.1/nodes])
    monkeypatch.setattr(tq,"angular_samples",bad_rule)
    with pytest.raises(ValueError,match="did not converge"):
        Evaluator(tetra(),[Request((0,0,0))],method="tet_quadrature").prepare([[.2]*3],1)


def test_unsupported_contracts():
    for geometry in (Geometry([[[0,0],[1,0],[0,1]]],2),Geometry([[[0,0,0],[1,0,0],[0,1,0]]],3,"surface")):
        with pytest.raises(ValueError):Evaluator(geometry,baseline(geometry.dimension),method="tet_quadrature")
    for kwargs in (dict(precision="float32_compensated"),dict(nodes=7),dict(nodes=65)):
        with pytest.raises(ValueError):Evaluator(tetra(),baseline(3),method="tet_quadrature",**kwargs)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,tol",[("float64",1e-14),("float32",1e-5)])
def test_prepared_composed_device_function(device,precision,tol):
    req=[Request((0,0,0)),Request((3,3,3)),Request((2,2,2),(0,2))]
    ev=Evaluator(tetra(),req,kernel="w6",method="tet_quadrature",precision=precision,device=device)
    b=ev.prepare([[.2]*3],1);b.launch();reference=b.numpy()[0]
    api=ev.api;real,vec,Plan=api.real,api.vec,api.Plan
    element,output=api.element,api.output
    @wp.kernel
    def consumer(query:vec,plan:Plan,facets:wp.array2d(dtype=vec),work:wp.array2d(dtype=real),out:wp.array(dtype=real)):
        element(query,real(1.),facets,0,4,plan,work,0)
        out[0]=output(plan,work,0,0)+real(.2)*output(plan,work,0,1)+real(.3)*output(plan,work,0,2)
    out=wp.empty(1,dtype=real,device=device)
    wp.launch(consumer,1,inputs=[vec(.2),b.integration_plan,ev.facets,b.work,out],device=device)
    np.testing.assert_allclose(out.numpy()[0],reference@[1.,.2,.3],atol=tol,rtol=tol)
    wp.launch(consumer,1,inputs=[vec(.3),b.integration_plan,ev.facets,b.work,out],device=device)
    assert np.isnan(out.numpy()).all()
    b.launch();np.testing.assert_array_equal(b.numpy()[0],reference)


@pytest.mark.skipif(not wp.is_cuda_available(),reason="CUDA hardware unavailable")
def test_tet_quadrature_cuda_graph():
    ev=Evaluator(tetra(),baseline(3),method="tet_quadrature",device="cuda:0")
    b=ev.prepare([[.2]*3],1);b.launch();expected=b.numpy()
    with wp.ScopedCapture(device="cuda:0") as capture:b.launch()
    b.work.fill_(123)
    wp.capture_launch(capture.graph);np.testing.assert_array_equal(b.numpy(),expected)


@pytest.mark.skipif(not wp.is_cuda_available(),reason="CUDA hardware unavailable")
@pytest.mark.parametrize("precision",["float64","float32"])
@pytest.mark.parametrize("nodes",[8,16])
def test_real_material_cuda_graph(precision,nodes):
    geo=tetra();rng=np.random.default_rng(52)
    factors=rng.normal(size=(1,20,3,3));D=factors@factors.swapaxes(-1,-2)+np.eye(3)*.2
    controls=rng.normal(size=(1,20,3))
    material=AffineResistanceMaterial(geo,D,controls,resistance_degree=3,velocity_degree=3)
    b=material.prepare([[.2]*3],1,precision=precision,method="tet_quadrature",device="cuda:0",nodes=nodes)
    real=b.batch.evaluator.api.real
    u=wp.array([[.4,-.2,.7]],dtype=real,device="cuda:0")
    mass=wp.array([.7],dtype=real,device="cuda:0")
    gradient=wp.array([np.eye(3)*.1],dtype=real,device="cuda:0")
    b.apply_device(u,mass,gradient)
    arrays=(b.force,b.gradient_load,b.reaction,b.power)
    expected=[a.numpy() for a in arrays]
    with wp.ScopedCapture(device="cuda:0") as capture:b.apply_device(u,mass,gradient)
    for a in arrays:a.fill_(123)
    wp.capture_launch(capture.graph)
    for a,value in zip(arrays,expected):np.testing.assert_array_equal(a.numpy(),value)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
@pytest.mark.parametrize('nodes',[8,16])
def test_shared_radius_cache_preserves_independent_composed_rows(device,precision,tol,nodes):
    ev=Evaluator(tetra(),baseline(3),method='tet_quadrature',precision=precision,device=device,nodes=nodes)
    b=ev.prepare([[.2,.2,.2],[.21,.19,.18]],1);b.launch();expected=b.numpy()
    plan=b.integration_plan
    assert b.quadrature['unique_profile_arguments'] < b.quadrature['samples']/3
    offsets=plan.profile_offsets.numpy();ids=plan.profile_index.numpy();nodes=plan.node_offsets.numpy()
    for row in range(b.count):
        assert ids[nodes[row]:nodes[row+1]].min()>=offsets[row]
        assert ids[nodes[row]:nodes[row+1]].max()<offsets[row+1]
    real,vec,Plan=ev.api.real,ev.api.vec,ev.api.Plan
    element,output=ev.api.element,ev.api.output
    @wp.kernel
    def composed(queries:wp.array(dtype=vec),facets:wp.array2d(dtype=vec),p:Plan,
                 work:wp.array2d(dtype=real),out:wp.array2d(dtype=real)):
        row=wp.tid()
        element(queries[row],real(1.),facets,0,4,p,work,row)
        for j in range(out.shape[1]):out[row,j]=output(p,work,row,j)
    plan.profiles.fill_(float('nan'));b.work.fill_(123)
    out=wp.empty(expected.shape,dtype=real,device=device)
    wp.launch(composed,b.count,inputs=[b.evaluation_queries,ev.facets,plan,b.work,out],device=device)
    np.testing.assert_allclose(out.numpy(),expected,rtol=tol,atol=tol)
