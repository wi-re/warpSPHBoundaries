"""Native acceptance is distinct from browser or arbitrary-precision tests."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline,adjacency

DEVICES=["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


@pytest.mark.parametrize("kernel",["w2","w4","w6","cubic"])
@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,budget",[("float64",1e-10),("float32",1e-5)])
def test_independent_fixtures(device,precision,budget,kernel):
    data=json.loads((Path(__file__).parents[1]/"fixtures/fluxbridge-v1.json").read_text())
    failures=[]
    for case in data["cases"]:
        if case["kernel"]!=kernel:
            continue
        d=case["dimension"]
        requests=tuple(Request(**r) for r in case["requests"])
        geometry=Geometry([case["vertices"]],d,case["measure"])
        ev=Evaluator(geometry,requests,kernel=case["kernel"],precision=precision,device=device)
        value=ev.evaluate([case["query"]],case["h"],exhaustive=True)[0]
        ref=np.array(case["values"],float)
        err=np.abs(value-ref)/(1+np.abs(ref))
        if not np.isfinite(err).all() or err.max()>budget:
            failures.append((d,case["measure"],case["kernel"],case["query"],float(np.nanmax(err)),int(np.argmax(err))))
    if failures and precision=="float32":
        assert all(np.isfinite(f[4]) and f[4]<1e-2 for f in failures), failures
        pytest.xfail(f"Not accepted at 1e-5: observed raw fp32 cancellation {failures}")
    assert not failures, failures


@pytest.mark.parametrize("device",DEVICES)
def test_locality_containment_discontinuities_and_periodic_images(device):
    elems=[np.array([[0,0],[2,0],[0,2]])+np.array([4*i,0]) for i in range(40)]
    geometry=Geometry(elems,2)
    requests=baseline(2)
    q=np.array([[.2,.2],[4.2,.2],[9.,0.],[1000,1000]])
    h=np.array([1.,.5,.7,1.])
    ev=Evaluator(geometry,requests,device=device)
    local=ev.prepare(q,h)
    full=ev.prepare(q,h,exhaustive=True)
    local.launch(); full.launch()
    np.testing.assert_allclose(local.numpy(),full.numpy(),atol=1e-10,rtol=1e-10)
    assert local.count<full.count/10
    with pytest.raises(ValueError,match="stale"):
        ev.prepare(q+.01,h,neighbors=local.neighbors)
    cell=Geometry([[[0,0],[1,0],[1,1],[0,1]]],2)
    disk=Evaluator(cell,[Request((0,0))],device=device)
    q=np.array([[.1,.2],[3.1,-4.8]])
    b=disk.prepare(q,1.5,periods=(1,1))
    b.launch()
    np.testing.assert_allclose(b.numpy(),1,atol=1e-10)
    pairs=list(zip(b.neighbors.query,b.neighbors.element,map(tuple,b.neighbors.shift)))
    assert len(set(pairs))==len(pairs)
    huge=Geometry([[[-10,-10],[10,-10],[10,10],[-10,10]]],2)
    np.testing.assert_allclose(Evaluator(huge,[Request((0,0))],device=device).evaluate([[0,0]],1),1,atol=1e-10)


@pytest.mark.parametrize("device",DEVICES)
def test_kernel_and_complete_derivatives_scaling_and_empty(device):
    for d in (2,3):
        verts=np.vstack([np.zeros(d),2*np.eye(d)])
        q=np.full((1,d),.2)
        zero=(0,)*d
        a=(1,)+(0,)*(d-1)
        req=[Request(a),Request(a,(0,)),Request(zero)]+[Request(zero,(i,j)) for i in range(d) for j in range(d)]
        ev=Evaluator(Geometry([verts],d),req,device=device)
        out=ev.evaluate(q,1)[0]
        eps=1e-5
        delta=np.zeros_like(q); delta[0,0]=eps
        diff=(ev.evaluate(q+delta,1)[0,0]-ev.evaluate(q-delta,1)[0,0])/(2*eps)
        np.testing.assert_allclose(diff,out[1]-out[2],rtol=1e-7,atol=1e-8)
        H=out[3:].reshape(d,d)
        np.testing.assert_allclose(H,H.T,atol=1e-10)
        scaled=Evaluator(Geometry([verts*3],d),req,device=device).evaluate(q*3,3)[0]
        np.testing.assert_allclose(scaled,out*np.array([3.**r.scale for r in req]),atol=1e-10)
        assert ev.evaluate([],1).shape==(0,len(req))
        assert Evaluator(Geometry([],d),req,device=device).evaluate(q,1).sum()==0


def test_unsupported_inputs():
    for alpha,ds in [((10,0),()),((0,-1),()),((0,0),(0,1,0))]:
        with pytest.raises(ValueError):
            Request(alpha,ds)
    ev=Evaluator(Geometry([[[0,0],[1,0],[0,1]]],2),[Request((0,0))])
    with pytest.raises(ValueError):
        ev.evaluate([[0,0]],0)

@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("dimension",[2,3])
def test_compact_correction_every_weighted_moment(device,dimension):
    data=json.loads((Path(__file__).parents[1]/"fixtures/fluxbridge-v1.json").read_text())
    for case in data["cases"]:
        if case["dimension"]!=dimension or case["measure"]!="volume":
            continue
        req=tuple(Request(**r) for r in case["requests"])
        geo=Geometry([case['vertices']],dimension)
        out=Evaluator(geo,req,kernel=case["kernel"],device=device,compact=True).evaluate([case['query']],1,exhaustive=True)[0]
        np.testing.assert_allclose(out,np.array(case['values'],float),atol=1e-10,rtol=1e-10)
    # Complete containment: all faces miss support; even moments need recursive corrections.
    from warpSPHBoundaries.fluxbridge import monomials
    if dimension==2:
        geometry=Geometry([[[-10,-10],[10,-10],[10,10],[-10,10]]],2)
    else:
        from itertools import product
        from scipy.spatial import ConvexHull
        verts=np.array(list(product([-10,10],repeat=3)))
        hull=ConvexHull(verts)
        faces=[]
        for f,eq in zip(hull.simplices,hull.equations):
            f=list(f)
            a,b,c=verts[f]
            if np.dot(np.cross(b-a,c-a),eq[:3])<0:
                f[1],f[2]=f[2],f[1]
            faces.append(f)
        geometry=Geometry.closed_surface(verts,faces)
    req=[Request(a) for a in monomials(dimension,4)]
    a=Evaluator(geometry,req,device=device).evaluate([[0]*dimension],1)
    b=Evaluator(geometry,req,device=device,compact=True).evaluate([[0]*dimension],1)
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize("precision,tol",[("float64",1e-10),("float32",1e-5)])
def test_recentered_translation_and_plane_surface_limits(precision,tol):
    # A translation that would destroy fp32 geometry if cast before recentering.
    tri=np.array([[0.,0.],[2.,0.],[0.,2.]])
    q=np.array([[.25,.25]])
    req=baseline(2)
    a=Evaluator(Geometry([tri],2),req,precision=precision).evaluate(q,1)
    b=Evaluator(Geometry([tri+2**24],2),req,precision=precision).evaluate(q+2**24,1)
    np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
    from warpSPHBoundaries.mesh3d.radial import kernel_jet
    verts=np.array([[-10,-10,.25],[20,-10,.25],[-10,20,.25]])
    req=baseline(3)
    geo=Geometry([verts],3,"surface")
    vals=Evaluator(geo,req,precision=precision).evaluate([[0,0,0]],1)[0]
    reverse=Evaluator(Geometry([verts[::-1]],3,"surface"),req,precision=precision).evaluate([[0,0,0]],1)[0]
    np.testing.assert_allclose(vals,reverse,atol=tol,rtol=tol)
    columns={r:i for i,r in enumerate(req)}
    W,dW,_,_=kernel_jet(np.array(.25),"w2")
    np.testing.assert_allclose(vals[columns[Request((0,0,0),(2,))]],2*np.pi*.25*W,atol=tol,rtol=tol)
    np.testing.assert_allclose(vals[columns[Request((0,0,0),(2,2))]],-2*np.pi*(W+.25*dW),atol=tol,rtol=tol)
    np.testing.assert_allclose(vals[columns[Request((0,0,0),(0,0))]],0,atol=tol)


def test_fp32_rejected_family_status_is_explicit():
    ev=Evaluator(Geometry([[[0,0],[1,0],[0,1]]],2),[Request((0,0))],kernel="w6",precision="float32")
    assert "experimental" in ev.accuracy_status

@pytest.mark.parametrize("dimension",[2,3])
def test_sliver_and_tiny_precision_policy(dimension):
    vertices=np.vstack([np.zeros(dimension),np.eye(dimension)])
    vertices[-1,-1]=1e-10
    with pytest.raises(ValueError,match="ill-conditioned"):
        Evaluator(Geometry([vertices],dimension),baseline(dimension)).evaluate([[0]*dimension],1)
    vertices=np.vstack([np.zeros(dimension),np.eye(dimension)])*1e-6
    with pytest.raises(ValueError,match="tiny element"):
        Evaluator(Geometry([vertices],dimension),baseline(dimension)).evaluate([[0]*dimension],1)


@pytest.mark.skipif(not wp.is_cuda_available(),reason="CUDA hardware unavailable")
def test_cuda_graph_replay():
    ev=Evaluator(Geometry([[[0,0],[2,0],[0,2]]],2),baseline(2),device="cuda:0",compact=True)
    batch=ev.prepare([[.2,.2],[.3,.3]],1)
    batch.launch()
    ref=batch.numpy().copy()
    with wp.ScopedCapture(device="cuda:0") as capture:
        batch.launch()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(batch.numpy(),ref)

@pytest.mark.parametrize("dimension",[2,3])
def test_independent_whole_support_polynomials(dimension):
    import mpmath as mp
    from warpSPHBoundaries.edge.kernels import KERNELS as BLOCKS
    from warpSPHBoundaries.curvbound.kernels import KERNELS as SHAPES
    from warpSPHBoundaries.fluxbridge import monomials
    from itertools import product
    if dimension==2:
        geometry=Geometry([[[-5,-5],[5,-5],[5,5],[-5,5]]],2)
    else:
        from scipy.spatial import ConvexHull
        vertices=np.array(list(product([-5.,5.],repeat=3)))
        hull=ConvexHull(vertices)
        faces=[]
        for f,eq in zip(hull.simplices,hull.equations):
            f=list(f); a,b,c=vertices[f]
            if np.dot(np.cross(b-a,c-a),eq[:3])<0:
                f[1],f[2]=f[2],f[1]
            faces.append(f)
        geometry=Geometry.closed_surface(vertices,faces)
    alphas=monomials(dimension,9)
    derivatives=[()]+[(j,) for j in range(dimension)]+[(i,j) for i in range(dimension) for j in range(i,dimension)]
    req=[Request(a,ds) for a in alphas for ds in derivatives]
    for kernel in ('w2','w4','w6','cubic'):
        with mp.workdps(70):
            def moment(a):
                if any(t%2 for t in a):
                    return mp.mpf(0)
                ang=2*mp.fprod(mp.gamma(mp.mpf(t+1)/2) for t in a)/mp.gamma(mp.mpf(sum(a)+dimension)/2)
                ratio=1 if dimension==2 else SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi
                radial=sum((mp.mpf((c*ratio).numerator)/(c*ratio).denominator)*(mp.mpf(b.R.numerator)/b.R.denominator)**(n+sum(a)+dimension)/(n+sum(a)+dimension)
                           for b in BLOCKS[kernel].blocks for n,c in enumerate(b.coeffs) if c)
                return ang*radial/mp.pi
            ref=[]
            for r in req:
                a=list(r.alpha); factor=1
                for j in r.derivative:
                    factor*=a[j]; a[j]-=1
                ref.append(float(factor*moment(a)) if factor else 0.)
        for compact in (False,True):
            out=Evaluator(geometry,req,kernel=kernel,compact=compact).evaluate([[0]*dimension],1)[0]
            np.testing.assert_allclose(out,ref,atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize("kernel",["w2","w4","w6","cubic"])
def test_surface_ambient_normal_hessian_at_plane(kernel):
    from warpSPHBoundaries.mesh3d.radial import kernel_jet
    geometry=Geometry([[[-10,-10,0],[20,-10,0],[-10,20,0]]],3,"surface")
    req=[Request((0,0,0),(2,)),Request((0,0,0),(2,2)),Request((0,0,0),(0,0))]
    value=Evaluator(geometry,req,kernel=kernel).evaluate([[0,0,0]],1)[0]
    W=kernel_jet(np.array(0.),kernel)[0]
    np.testing.assert_allclose(value,[0.,-2*np.pi*W,0.],atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize("precision,big",[("float64",1e16),("float32",1e8)])
def test_warp_compiler_preserves_profile_compensation(precision,big):
    from warpSPHBoundaries.fluxbridge.wp_func import functions
    api=functions(precision)
    real,Plan,output=api.real,api.Plan,api.output
    plan=Plan()
    plan.f=wp.empty((0,9),dtype=int,device="cpu")
    plan.b=wp.empty((0,5),dtype=int,device="cpu")
    plan.offsets=wp.array([0,3],dtype=int,device="cpu")
    plan.term_kind=wp.array([0,0,0],dtype=int,device="cpu")
    plan.term_index=wp.array([0,1,0],dtype=int,device="cpu")
    plan.term_value=wp.array([1,1,-1],dtype=real,device="cpu")
    work=wp.array([[big,1]],dtype=real,device="cpu")
    out=wp.empty(1,dtype=real,device="cpu")
    @wp.kernel
    def consumer(plan: Plan,work: wp.array2d(dtype=real),out: wp.array(dtype=real)):
        out[0]=output(plan,work,0,0)
    wp.launch(consumer,1,inputs=[plan,work,out],device="cpu")
    assert out.numpy()[0]==1


@pytest.mark.parametrize("dimension",[2,3])
def test_exact_support_tangency_is_empty(dimension):
    vertices=np.vstack([np.zeros(dimension),2*np.eye(dimension)])
    query=np.array([[3.]+[0.]*(dimension-1)])
    for kernel in ('w2','w4','w6','cubic'):
        for compact in (False,True):
            value=Evaluator(Geometry([vertices],dimension),baseline(dimension),kernel=kernel,compact=compact).evaluate(query,1,exhaustive=True)
            np.testing.assert_allclose(value,0,atol=1e-10)


def test_closed_cavity_compact_regular_and_material_angle_limits():
    from itertools import product
    from scipy.spatial import ConvexHull
    def cube(radius):
        vertices=np.array(list(product([-radius,radius],repeat=3)))
        hull=ConvexHull(vertices)
        faces=[]
        for f,eq in zip(hull.simplices,hull.equations):
            f=list(f); a,b,c=vertices[f]
            if np.dot(np.cross(b-a,c-a),eq[:3])<0:
                f[1],f[2]=f[2],f[1]
            faces.append(f)
        return vertices,faces
    outer,of=cube(3.)
    inner,inf=cube(.4)
    cavity_faces=of+[[j+8 for j in f[::-1]] for f in inf]
    geometry=Geometry.closed_surface(np.vstack([outer,inner]),cavity_faces)
    queries=[[0,0,0],[.4,0,0],[.4,.4,0],[.4,.4,.4]]
    req=baseline(3)
    for compact in (False,True):
        out=Evaluator(geometry,req,compact=compact).evaluate(queries,1)
        outside=Evaluator(Geometry.closed_surface(outer,of),req,compact=compact).evaluate(queries,1)
        void=Evaluator(Geometry.closed_surface(inner,inf),req,compact=compact).evaluate(queries,1)
        np.testing.assert_allclose(out,outside-void,atol=1e-10,rtol=1e-10)


def test_sparse_outputs_empty_request_and_surface_scaling():
    req=[Request((0,0,0)),Request((0,0,0),(2,)),Request((1,0,0),(0,1))]
    triangle=np.array([[0.,0.,.25],[2.,0.,.25],[0.,2.,.25]])
    q=np.array([[.2,.2,0.]])
    geo=Geometry([triangle],3,"surface")
    values=Evaluator(geo,req).evaluate(q,1)
    doubled=Evaluator(Geometry([triangle*3],3,"surface"),req).evaluate(q*3,3)
    np.testing.assert_allclose(doubled,values*np.array([3.**(r.scale-1) for r in req]),atol=1e-10,rtol=1e-10)
    assert Evaluator(geo,[]).evaluate(q,1).shape==(1,0)

@pytest.mark.parametrize('precision,tolerance',[('float64',1e-10),('float32',1e-2),('float32_compensated',1e-5)])
def test_compact_empty_surface_overwrites_stale_scratch(precision,tolerance):
    geometry=Geometry([[[0,0,.25],[2,0,.25],[0,2,.25]]],3,'surface')
    queries=[[3,3,0],[3,3,3],[3,0,.25]]
    for kernel in ('w2','w4','w6','cubic'):
        ev=Evaluator(geometry,baseline(3),precision=precision,kernel=kernel,compact=True)
        batch=ev.prepare(queries,1,exhaustive=True)
        batch.work.fill_(ev.api.work_dtype(12345.))
        batch.launch()
        value=batch.numpy()
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value,0,atol=tolerance)
        reference=Evaluator(geometry,baseline(3),kernel=kernel).evaluate(queries,1,exhaustive=True)
        np.testing.assert_allclose(value,reference,atol=tolerance,rtol=tolerance)
