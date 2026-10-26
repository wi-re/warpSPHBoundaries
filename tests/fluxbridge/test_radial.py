"""Independent native 3D special-form baseline acceptance."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Request,baseline,Geometry,Evaluator
from warpSPHBoundaries.fluxbridge.radial import RadialEvaluator

DEVICES=["cpu"]+(["cuda:0"] if wp.is_cuda_available() else [])
FIXTURES=Path(__file__).parents[1]/"fixtures/fluxbridge-radial-v1.json"


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("kind",["plane","sphere"])
@pytest.mark.parametrize("kernel",["w2","w4","w6","cubic"])
@pytest.mark.parametrize("precision,budget",[("float64",1e-10),("float32",1e-5),("float32_compensated",1e-5)])
def test_independent_radial_fixtures(device,kind,kernel,precision,budget):
    cases=[c for c in json.loads(FIXTURES.read_text())["cases"] if c['kind']==kind and c['kernel']==kernel]
    kw={} if kind=="plane" else dict(radii=np.array([c['radius'] for c in cases],float))
    values=RadialEvaluator(kind,kernel=kernel,precision=precision,device=device).evaluate(
        np.array([c['distance'] for c in cases],float),np.array(cases[0]['normal'],float),1,**kw)
    ref=np.array([c['values'] for c in cases],float)
    error=np.abs(values-ref)/(1+np.abs(ref))
    assert np.isfinite(error).all()
    assert error.max()<=budget


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("precision,tol",[("float64",1e-10),("float32",1e-5),("float32_compensated",1e-5)])
def test_radial_scaling_subset_empty_and_reuse(device,precision,tol):
    n=np.array([2.,-1.,2.])/3
    req=baseline(3)
    for kind in ("plane","sphere"):
        ev=RadialEvaluator(kind,kernel="w6",precision=precision,device=device)
        kw={} if kind=="plane" else dict(radii=[.5,2.,1.])
        d=np.array([-.2,.2,1.])
        batch=ev.prepare(d,n,1,**kw)
        batch.work.fill_(12345)
        batch.launch();value=batch.numpy()
        batch.launch();np.testing.assert_array_equal(value,batch.numpy())
        scaled={} if kind=="plane" else dict(radii=np.array(kw['radii'])*3)
        out=ev.evaluate(d*3,n,3,**scaled)
        np.testing.assert_allclose(out,value*np.array([3.**r.scale for r in req]),rtol=tol,atol=tol)
        assert np.all(value[-1]==0)  # exact external tangency
        subset=(req[2],req[25],req[0],Request((0,0,0),(2,0)))
        ev2=RadialEvaluator(kind,subset,kernel="w6",precision=precision,device=device)
        actual=ev2.evaluate(d,n,1,**kw)
        expected=value[:,[2,25,0,req.index(Request((0,0,0),(0,2)))]]
        np.testing.assert_allclose(actual,expected,atol=tol,rtol=tol)
        assert ev.evaluate([],n,1,**({} if kind=="plane" else dict(radii=1))).shape==(0,28)
        assert RadialEvaluator(kind,[],device=device).evaluate([0],n,1,**({} if kind=="plane" else dict(radii=1))).shape==(1,0)


@pytest.mark.parametrize("kernel",["w2","w4","w6","cubic"])
@pytest.mark.parametrize("device",DEVICES)
def test_complete_derivatives_fixed_sphere_and_plane(kernel,device):
    req=baseline(3);z=(0,0,0)
    columns={r:i for i,r in enumerate(req)}
    for kind in ("plane","sphere"):
        ev=RadialEvaluator(kind,kernel=kernel,device=device)
        def value(q):
            if kind=="plane":
                return ev.evaluate([q[2]],[0,0,1],1)[0]
            length=np.linalg.norm(q)
            return ev.evaluate([length-.7],q/length,1,radii=.7)[0]
        q=np.array([.2,-.3,.6]);out=value(q)
        eps=2e-5
        for j in range(3):
            delta=np.eye(3)[j]*eps
            plus,minus=value(q+delta),value(q-delta)
            diff=(plus-minus)/(2*eps)
            np.testing.assert_allclose(diff[columns[Request(z)]],out[columns[Request(z,(j,))]],rtol=2e-7,atol=2e-8)
            for i in range(3):
                np.testing.assert_allclose(diff[columns[Request(z,(i,))]],out[columns[Request(z,tuple(sorted((i,j))))]],rtol=2e-7,atol=2e-8)
                alpha=tuple(int(k==i) for k in range(3))
                expected=out[columns[Request(alpha,(j,))]]-(out[0] if i==j else 0)
                np.testing.assert_allclose(diff[columns[Request(alpha)]],expected,rtol=2e-7,atol=2e-8)


@pytest.mark.parametrize("device",DEVICES)
def test_plane_matches_closed_mesh_and_rotated_radial_oracle(device):
    from itertools import product
    from scipy.spatial import ConvexHull
    from warpSPHBoundaries.mesh3d.radial import axisymmetric
    vertices=np.array(list(product([-4.,4.],[-4.,4.],[-4.,0.])))
    hull=ConvexHull(vertices);faces=[]
    for f,eq in zip(hull.simplices,hull.equations):
        f=list(f);a,b,c=vertices[f]
        if np.dot(np.cross(b-a,c-a),eq[:3])<0:
            f[1],f[2]=f[2],f[1]
        faces.append(f)
    geometry=Geometry.closed_surface(vertices,faces)
    for kernel in ("w2","w4","w6","cubic"):
        d=np.array([-.5,0,.2,.75])
        radial=RadialEvaluator("plane",kernel=kernel,device=device).prepare(d,[0,0,1],1)
        radial.launch()
        mesh=Evaluator(geometry,baseline(3),kernel=kernel,device=device,compact=True).evaluate(np.array([[0,0,x] for x in d]),1)
        np.testing.assert_allclose(radial.numpy(),mesh,rtol=1e-10,atol=1e-10)
        for radius in (.1,.5,2.,100.):
            d=np.array([-radius,-radius+1e-6,-.5*radius,0.,.5,.999,1.])
            batch=RadialEvaluator("sphere",kernel=kernel,device=device).prepare(d,[0,0,1],1,radii=radius)
            batch.launch()
            oracle=axisymmetric(d,kernel,radius=radius)
            np.testing.assert_allclose(batch.work.numpy(),oracle,rtol=1e-10,atol=1e-10)
    # Isotropic centre and full-support limits must not depend on supplied axis.
    ev=RadialEvaluator("sphere",kernel="w6",device=device)
    for distance,radius in ((-.5,.5),(-2.,2.)):
        a=ev.evaluate(distance,[0,0,1],1,radii=radius)
        b=ev.evaluate(distance,[2/3,-1/3,2/3],1,radii=radius)
        np.testing.assert_allclose(a,b,atol=1e-10)


@pytest.mark.parametrize("precision",["float64","float32","float32_compensated"])
@pytest.mark.parametrize("device",DEVICES)
def test_device_functions_are_composable(device,precision):
    ev=RadialEvaluator("plane",precision=precision,device=device)
    api=ev.api;real=api.real;vec=api.vec;Plan=api.Plan
    element,select=api.element,api.select
    @wp.kernel
    def consumer(plan:Plan,work:wp.array2d(dtype=real),out:wp.array(dtype=real)):
        row=wp.tid()
        element(real(.2),real(0),real(1),0,plan,work,row)
        out[row]=real(2)*select(work,row,vec(real(0),real(0),real(1)),0,0,0,0)
    scratch=wp.empty((1,9),dtype=real,device=device)
    out=wp.empty(1,dtype=real,device=device)
    wp.launch(consumer,1,inputs=[ev.plan,scratch,out],device=device)
    ref=ev.evaluate(.2,[0,0,1],1)[0,0]*2
    np.testing.assert_allclose(out.numpy()[0],ref,atol=1e-6,rtol=1e-6)


def test_explicit_radial_domains():
    for requests in ([Request((0,0))],[Request((3,0,0))],[Request((1,0,0),(1,2))]):
        with pytest.raises(ValueError,match="baseline"):
            RadialEvaluator("sphere",requests)
    ev=RadialEvaluator("sphere")
    for kw in ({},{"radii":-1},{"radii":1e-6},{"radii":1e6}):
        with pytest.raises(ValueError):
            ev.prepare([0],[0,0,1],1,**kw)
    with pytest.raises(ValueError):ev.prepare([-2],[0,0,1],1,radii=1)
    with pytest.raises(ValueError,match="unit"):ev.prepare([0],[1,1,1],1,radii=1)
    with pytest.raises(ValueError):ev.prepare([0],[0,0,1],0,radii=1)
    with pytest.raises(ValueError):RadialEvaluator("plane").prepare([0],[0,0,1],1,radii=1)
    with pytest.raises(ValueError,match="fp32"):
        RadialEvaluator("plane",precision="float32_compensated").prepare([1e100],[0,0,1],1)
    # Even a disjoint support must reject unrepresentable h^+-2 scratch scaling
    # instead of evaluating zero*infinity to NaN.
    for precision,h in (("float32",1e-30),("float32_compensated",1e20),("float64",1e200)):
        with pytest.raises(ValueError,match="support scaling"):
            RadialEvaluator("plane",precision=precision).prepare([10],[0,0,1],h)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("kernel",['w2','w4','w6','cubic'])
def test_raw_fp32_random_radial_domain_against_independent_shells(device,kernel):
    from warpSPHBoundaries.mesh3d.radial import axisymmetric
    rng=np.random.default_rng(20261022)
    d=np.r_[rng.uniform(-1,1,120),0.,1.,-1.,1.-1e-6,-1.+1e-6]
    plane=RadialEvaluator('plane',kernel=kernel,precision='float32',device=device).prepare(d,[0,0,1],1)
    plane.launch();ref=axisymmetric(d,kernel)
    assert np.max(abs(plane.work.numpy()-ref)/(1+abs(ref)))<1e-5
    sphere=RadialEvaluator('sphere',kernel=kernel,precision='float32',device=device)
    for radius in np.geomspace(.01,1e4,13):
        distance=np.maximum(d,-radius)
        b=sphere.prepare(distance,[0,0,1],1,radii=radius);b.launch()
        ref=axisymmetric(distance,kernel,radius=radius)
        assert np.isfinite(b.work.numpy()).all()
        assert np.max(abs(b.work.numpy()-ref)/(1+abs(ref)))<1e-5


@pytest.mark.parametrize("device",DEVICES)
def test_raw_beta_early_exit_matches_fixed_64_steps(device):
    from warpSPHBoundaries.fluxbridge.radial import functions
    fixed=functions('float32',beta_early_exit=False)
    for kind,radius in (('plane',None),('sphere',.01),('sphere',.3),('sphere',1.),('sphere',100.)):
        distance=np.linspace(-1,1,65)
        if radius is not None:distance=np.maximum(distance,-radius)
        ev=RadialEvaluator(kind,kernel='cubic',precision='float32',device=device)
        batch=ev.prepare(distance,[0,0,1],1,**({} if radius is None else dict(radii=radius)))
        batch.launch()
        plan=fixed.Plan()
        for name in ('powers','radial_powers','profiles','bounds','coefficients'):
            setattr(plan,name,getattr(ev.plan,name))
        work=wp.empty_like(batch.work);values=wp.empty_like(batch.values)
        wp.launch(fixed.batch,batch.count,inputs=[batch.distances,batch.radii,batch.normals,batch.supports,
            0 if kind=='plane' else 1,plan,ev.metadata,work,values],device=device)
        np.testing.assert_array_equal(work.numpy(),batch.work.numpy())
        np.testing.assert_array_equal(values.numpy(),batch.numpy())


@pytest.mark.skipif(not wp.is_cuda_available(),reason="CUDA hardware unavailable")
def test_radial_cuda_graph():
    ev=RadialEvaluator("sphere",precision="float32_compensated",device="cuda:0")
    batch=ev.prepare([-.2,0,.2],[0,0,1],1,radii=1)
    batch.launch();reference=batch.numpy()
    with wp.ScopedCapture(device="cuda:0") as capture:
        batch.launch()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(batch.numpy(),reference)
