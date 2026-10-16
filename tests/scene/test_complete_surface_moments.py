"""Executed twelve-moment surface contract and conditioned device path."""
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme,OperationDirection,OperationProperties,ParticleState,WarpOperation

from warpSPHBoundaries.edge import warptriangle
from warpSPHBoundaries.scene.scene import Scene,Body,SurfaceRep,BoxRep
from warpSPHBoundaries.scene.fixedadj import fixed_adjacency
from warpSPHBoundaries.scene.fused import FusedWall,WallOutput
from warpSPHBoundaries.edge.warpfused import FusedGroup
from warpSPHBoundaries.validation.adversarial import polygon_cases,scene_channels,precision_polygon_reference

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])
ORDER=[0,1,2,9,10,11,3,4,5,6,7,8]
OUTS=tuple(WallOutput(k,0,k) for k in ['lam','m1','g0','cov','m2'])


def setup(device, vertices, points, *, solid='inside', center=(0.,0.), angle=0., supports=None):
    rep=SurfaceRep.polygon(vertices,solid)
    scene=Scene([Body(bodyId=0,reps=[rep],center=center,angle=angle)],device)
    x=torch.as_tensor(points,dtype=torch.float64,device=device)
    h=torch.ones(len(x),dtype=torch.float64,device=device) if supports is None else torch.as_tensor(supports,dtype=torch.float64,device=device)
    ps=ParticleState(positions=x,supports=h,masses=torch.ones_like(h),densities=torch.ones_like(h),kinds=torch.zeros(len(x),dtype=torch.int32,device=device))
    props=OperationProperties(kernel='w4',operation=WarpOperation.Density,gradientMode=GradientScheme.Naive,operationMode=OperationDirection.BoundaryToFluid)
    return scene,ps,props


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('name',['triangle-size-1e-09','triangle-size-0.01','triangle-angle-0.1','triangle-angle-179','sliver-1e-07','acute-notch'])
def test_complete_executed_surface_paths_against_saved_precision_references(device,name):
    from warpSPHBoundaries.validation.paths import evidence_path
    c=next(c for c in polygon_cases() if c.name==name)
    path=evidence_path('triangle-conditioning')/'w4'/name/'reference.npz'
    points=c.probes()
    if path.exists():
        with np.load(path) as a:target=a['target']
    else:
        target=precision_polygon_reference(c.vertices,points,'w4')
    for fused in [False,True]:
        values,estimate,status=scene_channels(SurfaceRep.polygon(c.vertices),points,'w4',device,
                                            fused=fused,second=True,return_diagnostics=True)
        assert (status>=0).all()
        error=abs(values-target)/np.max(abs(target),axis=0)
        assert np.max(error)<=np.finfo(np.float32).eps
        assert np.isfinite(values).all()
        if name!='acute-notch': assert (status==1).any()
        assert not np.isfinite(estimate[status==0]).any()


@pytest.mark.parametrize('device',DEVICES)
def test_second_moments_whole_support_cavity_rotation_and_support_scale(device):
    vertices=np.array([[-3.,-3.],[3.,-3.],[3.,3.],[-3.,3.]])
    h=np.array([.5,1.,2.])
    sc,ps,p=setup(device,vertices,np.zeros((3,2)),supports=h,angle=.43)
    pm=sc.pairMoments(ps,p).entries[0]['surface'][0]
    plane=warptriangle.plane_second_moment('w4')
    target=torch.eye(2,dtype=torch.float64,device=device)[None]*torch.tensor(h*h*plane,device=device)[:,None,None]
    torch.testing.assert_close(pm.m2,target,atol=2e-12,rtol=0)
    wall=FusedWall(sc,fixed_adjacency(sc,ps,p,2.),(FusedGroup('w4'),))
    got=wall.evaluate(OUTS)
    torch.testing.assert_close(got['m2'][0],target,atol=2e-12,rtol=0)
    tiny=next(c for c in polygon_cases() if c.name=='triangle-size-1e-05')
    points=tiny.probes()[[0,6,44]]
    ins=scene_channels(SurfaceRep.polygon(tiny.vertices),points,device=device,fused=True,second=True)
    ext=scene_channels(SurfaceRep.polygon(tiny.vertices,'outside'),points,device=device,fused=True,second=True)
    expected=np.zeros_like(ins);expected[:,[0,8,11]]=1.;expected[:,[3,5]]=plane
    np.testing.assert_allclose(ins+ext,expected,atol=3e-15,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_failed_refinement_is_explicit_not_a_successful_zero(device):
    c=next(c for c in polygon_cases() if c.name=='sliver-1e-07')
    v=torch.tensor(c.vertices,dtype=torch.float64,device=device)
    e=torch.tensor([[0,1],[1,2],[2,0]],dtype=torch.int32,device=device)
    metadata=warptriangle.prepare(v,e)
    values,estimate,status=warptriangle.channels(v,torch.tensor([[.1,.2]],dtype=torch.float64,device=device),
        torch.ones(1,dtype=torch.float64,device=device),metadata,'w4',tolerance=1e-30,max_panels=1)
    assert status.item()==-1
    assert torch.isnan(values).all()
    assert torch.isfinite(estimate).all()


@pytest.mark.skipif(not wp.is_cuda_available(),reason='CUDA capture/replay needs the host GPU')
def test_conditioned_m2_graph_replay_and_copy_from_stay_on_device():
    c=next(c for c in polygon_cases() if c.name=='sliver-1e-07')
    points=c.probes()[[0,4,6,44,45]]
    sc,ps,p=setup('cuda:0',c.vertices,points)
    def build(state):return FusedWall(sc,fixed_adjacency(sc,state,p,1.),(FusedGroup('w4'),))
    current=build(ps);current.evaluate(OUTS)
    current.reset_derived()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with torch.cuda.graph(graph,stream=stream):out=current.evaluate(OUTS)
    torch.cuda.current_stream().wait_stream(stream)
    ps.positions.add_(torch.tensor([.02,.03],device='cuda:0'))
    updated=build(ps)
    expected=updated.evaluate(OUTS)
    current.copy_from(updated)
    graph.replay();torch.cuda.synchronize()
    for key in expected:torch.testing.assert_close(out[key],expected[key],rtol=2e-12,atol=1e-24)
    # Warm static plans, then the construction/evaluation must not host-sync.
    torch.cuda.set_sync_debug_mode('error')
    try:
        another=build(ps);another.evaluate(OUTS)
    finally:torch.cuda.set_sync_debug_mode('default')
    full_graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with torch.cuda.graph(full_graph,stream=stream):
            captured_wall=build(ps)
            captured=captured_wall.evaluate(OUTS)
    torch.cuda.current_stream().wait_stream(stream)
    ps.positions.add_(torch.tensor([-.01,.04],device='cuda:0'))
    full_graph.replay();torch.cuda.synchronize()
    expected=build(ps).evaluate(OUTS)
    for key in expected:torch.testing.assert_close(captured[key],expected[key],rtol=2e-12,atol=1e-24)


@pytest.mark.parametrize('device',DEVICES)
def test_rotated_conditioned_surface_with_variable_supports_and_multiple_representations(device):
    from warpSPHBoundaries.edge.triangle import integrate
    c=next(c for c in polygon_cases() if c.name=='triangle-size-0.01')
    local=c.vertices.mean(0)+np.array([[.1,.2],[-.4,.25],[.03,-.2]])
    angle=.43;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=np.array([.7,-.5]);h=np.array([.5,1.,1.7])
    sc,ps,p=setup(device,c.vertices,local@R.T+center,center=center,angle=angle,supports=h)
    # More than one representation contributes to the same body/output pool.
    sc.bodies[0].reps.append(SurfaceRep.polygon(c.vertices+[.03,.08]).to(device))
    ref=np.array([integrate(c.vertices,q,'w4',h=hh).moments+integrate(c.vertices+[.03,.08],q,'w4',h=hh).moments
                  for q,hh in zip(local,h)])
    expected=dict(lam=ref[:,0],m1=ref[:,1:3]@R.T,g0=ref[:,6:8]@R.T,
                  cov=R@ref[:,8:].reshape(-1,2,2)@R.T)
    mat=np.stack([ref[:,3],ref[:,4],ref[:,4],ref[:,5]],1).reshape(-1,2,2)
    expected['m2']=R@mat@R.T
    wall=FusedWall(sc,fixed_adjacency(sc,ps,p,1.7),(FusedGroup('w4'),))
    actual=wall.evaluate(OUTS)
    for key in expected:
        np.testing.assert_allclose(actual[key][0].cpu().numpy(),expected[key],rtol=2e-10,atol=1e-20)
    pm=sc.pairMoments(ps,p)
    for key,field in [('lam','lam'),('m1','m1'),('g0','g0'),('cov','g1'),('m2','m2')]:
        out=torch.zeros_like(actual[key][0])
        for m in pm.entries[0]['surface']:out.index_add_(0,m.q.long(),getattr(m,field))
        torch.testing.assert_close(out,actual[key][0],rtol=2e-10,atol=1e-20)


def test_second_moment_coverage_rejects_unimplemented_lookup_geometry():
    sc,ps,p=setup('cpu',[[-1,-1],[1,-1],[0,1]],[[.1,.2]])
    sc.bodies[0].reps=[BoxRep((-.3,-.2),(.3,.2)).to('cpu')]
    wall=FusedWall(sc,fixed_adjacency(sc,ps,p,1.),(FusedGroup('w4'),))
    with pytest.raises(NotImplementedError,match='explicit SurfaceRep'):
        wall.evaluate((WallOutput('m2',0,'m2'),))


def test_closed_form_surface_route_remains_available_explicitly():
    c=next(c for c in polygon_cases() if c.name=='triangle-size-1e-09')
    rep=SurfaceRep.polygon(c.vertices,conditioned=False)
    assert rep._triangle is None
    actual,estimate,status=scene_channels(rep,c.probes(),device='cpu',second=True,return_diagnostics=True)
    assert not status.any() and not np.isfinite(estimate).any()
    assert np.isfinite(actual).all()


def test_static_triangle_orientation_resolves_sub_longdouble_determinants():
    eps=np.finfo(float).eps
    vertices=np.array([[0.,0.],[1.,1.-eps],[1.+eps,1.]])
    for order in [vertices,vertices[::-1]]:
        rep=SurfaceRep.polygon(order)
        assert rep._triangle is not None and rep._triangle[2]>0
        assert rep._triangle[1]==eps*eps
        outside=SurfaceRep.polygon(order,'outside')
        assert outside._triangle[2]<0 and outside.background==1


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kernel',['cubic','w2','w4','w6'])
def test_requested_chebyshev_route_also_evaluates_second_moments(device,kernel):
    c=next(c for c in polygon_cases() if c.name=='acute-notch')
    values=scene_channels(SurfaceRep.polygon(c.vertices),c.probes()[[0,6,8,30,44]],kernel,device,
                          fused=True,stable=True,second=True)
    reference=precision_polygon_reference(c.vertices,c.probes()[[0,6,8,30,44]],kernel)
    error=abs(values-reference)/np.max(abs(reference),axis=0)
    assert np.max(error) <= np.finfo(np.float32).eps


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kernel',['cubic','w2','w4','w6'])
def test_exterior_solid_keeps_full_support_far_outside_its_loop_box(device,kernel):
    c=next(c for c in polygon_cases() if c.name=='triangle-size-1e-05')
    points=np.array([[2.,3.],[-3.,2.],[3.,-2.]])
    expected=np.zeros((len(points),12));expected[:,[0,8,11]]=1.
    expected[:,[3,5]]=warptriangle.plane_second_moment(kernel)
    for fused in [False,True]:
        actual=scene_channels(SurfaceRep.polygon(c.vertices,'outside'),points,kernel,device,fused=fused,second=True)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-15)
    from dataclasses import replace
    sc,ps,p=setup(device,c.vertices,points,solid='outside');p=replace(p,kernel=kernel)
    wall=FusedWall(sc,fixed_adjacency(sc,ps,p,1.),(FusedGroup(kernel),))
    lap=wall.evaluate((WallOutput('lap',0,'lap'),))['lap']
    assert torch.equal(lap,torch.zeros_like(lap))
