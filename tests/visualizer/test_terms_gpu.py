"""Execute the actual browser compute WGSL through optional wgpu-native.

This validates computation, separately from the existing ray-caster tests.
Mesa software execution is useful for correctness, not hardware performance.
"""
from pathlib import Path

import numpy as np
import pytest

from warpSPHBoundaries.curvbound.kernels import KERNELS
from warpSPHBoundaries.edge.kernels import KERNELS as BLOCKS
from warpSPHBoundaries.mesh3d.radial import axisymmetric, select
from warpSPHBoundaries.visualizer.compute3d import signed_distance, validate3d
from warpSPHBoundaries.visualizer.terms3d import FIELDS3D

wgpu = pytest.importorskip('wgpu')
# IDs deliberately match the frontend scalar selector; exported metadata is
# checked separately against the Python public field set.
FIELDS = ['m00','m100','m010','m001','m200','m110','m101','m020','m011','m002',
          'Gx','Gy','Gz',*[f'g1{a}{b}' for a in 'xyz' for b in 'xyz'],
          'Hxx','Hxy','Hxz','Hyy','Hyz','Hzz','laplacian','grad_norm']


@pytest.fixture(scope='module')
def execute():
    try:
        adapter = wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
        device = adapter.request_device_sync()
    except Exception as exc:
        pytest.skip(str(exc))
    source = Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static/terms-compute.wgsl'
    module = device.create_shader_module(code=source.read_text())
    pipeline = device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'compute_terms'})
    def run(c, model, field, samples=None):
        n=9
        half=(c['radius'] if c['geometry']=='sphere' else max(c['width'],c['height'],c['depth'])/2)+c['padding']*c['h']
        axis=np.linspace(-half,half,n)
        z,y,x=np.meshgrid(axis,axis,axis,indexing='ij')
        points=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
        k=KERNELS[c['kernel']]
        co=[float(v*k.c3_pi/k.c2_pi)/np.pi for v in BLOCKS[c['kernel']].pieces[0][2]]
        nodes,weights=np.polynomial.legendre.leggauss(16)
        table=np.zeros(44+(0 if samples is None else samples.size),dtype='<f4')
        table[:len(co)]=co;table[12:44]=np.column_stack([nodes,weights]).ravel()
        if samples is not None:table[44:]=samples.ravel()
        raw=bytearray(64);u=np.frombuffer(raw,dtype='<u4');f=np.frombuffer(raw,dtype='<f4')
        u[:4]=[n,FIELDS.index(field),['sphere','box','tetrahedron'].index(c['geometry']),model]
        P=dict(k.poly) if c['kernel']!='cubic' else {}
        f[4:]=[c['h'],c['radius'],c['width'],c['height'],c['depth'],half,
               ['cubic','w2','w4','w6'].index(c['kernel']),c['curvature_order'] if model==2 else (0 if samples is None else len(samples)),
               k.n if c['kernel']!='cubic' else 3,*[float(P.get(j,0)) for j in (1,2,3)]]
        uniform=device.create_buffer_with_data(data=raw,usage=wgpu.BufferUsage.UNIFORM)
        lookup=device.create_buffer_with_data(data=table,usage=wgpu.BufferUsage.STORAGE)
        output=device.create_buffer(size=n**3*8,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
        group=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[
            {'binding':0,'resource':{'buffer':uniform}}, {'binding':1,'resource':{'buffer':lookup}}, {'binding':2,'resource':{'buffer':output}}])
        encoder=device.create_command_encoder();p=encoder.begin_compute_pass();p.set_pipeline(pipeline);p.set_bind_group(0,group);p.dispatch_workgroups((n**3+63)//64);p.end();device.queue.submit([encoder.finish()])
        result=np.frombuffer(device.queue.read_buffer(output),dtype='<f4').reshape(-1,2).copy()
        uniform.destroy();lookup.destroy();output.destroy()
        return points,result
    yield run
    device.destroy()


@pytest.mark.parametrize('name',['w2','w4','w6','cubic'])
def test_gpu_all_moment_derivative_channels_against_float64_reference(execute,name):
    c=validate3d(dict(kernel=name,radius=.7,h=.5))
    for model in (0,1,2):
        if model==2:c=validate3d(dict(kernel=name,radius=2.,h=.5))
        for field in FIELDS:
            points,result=execute(c,model,field)
            distance,normal=signed_distance(points,c,True)
            inv=axisymmetric(distance/c['h'],name,radius=c['radius']/c['h'] if model==0 else None,
                             curvature=c['h']/c['radius'] if model==2 else None)
            expected=select(inv,normal,field)
            np.testing.assert_allclose(result[:,1],distance,atol=2e-6)
            assert np.max(abs(result[:,0]-expected)) < 3e-5*max(1,float(np.max(abs(expected)))),(name,model,field,np.max(abs(result[:,0]-expected)))


@pytest.mark.parametrize('geometry',['sphere','box','tetrahedron'])
def test_gpu_support_cubature_and_geometry_distances(execute,geometry):
    from warpSPHBoundaries.visualizer.terms3d import integrand
    from warpSPHBoundaries.visualizer.compute3d import geometry3d
    c=validate3d(dict(geometry=geometry,kernel='w4',radius=.713,h=.517,width=1.37,height=1.09,depth=.71))
    radial,wr=np.polynomial.legendre.leggauss(8)
    polar,wp=np.polynomial.legendre.leggauss(6)
    yy=[];volume=[]
    for r,w in zip((radial+1)/2,wr/2):
        for mu,wm in zip(polar,wp):
            for j in range(12):
                phi=2*np.pi*(j+.5)/12
                yy.append([r*np.sqrt(1-mu*mu)*np.cos(phi),r*np.sqrt(1-mu*mu)*np.sin(phi),r*mu])
                volume.append(w*r*r*wm*2*np.pi/12)
    yy=np.array(yy);volume=np.array(volume)
    for field in ('m100','g1xy','Hxz','laplacian','grad_norm'):
        weights=integrand(yy,'w4',field)*volume if field!='grad_norm' else np.zeros(len(yy))
        grad=np.column_stack([integrand(yy,'w4','G'+a)*volume for a in 'xyz'])
        samples=np.column_stack([yy,weights,grad,np.zeros(len(yy))])
        points,result=execute(c,3,field,samples)
        source=points[:,None,:]+c['h']*yy[None,:,:]
        if geometry=='sphere':mask=np.sum(source*source,axis=-1)<=c['radius']**2
        elif geometry=='box':mask=np.all(abs(source)<=np.array([c['width'],c['height'],c['depth']])/2,axis=-1)
        else:
            g=geometry3d(c);vertices=np.array(g['vertices']);mask=np.ones(source.shape[:-1],dtype=bool)
            for face in g['faces']:
                a,b,d=vertices[face];normal=np.cross(b-a,d-a);normal/=np.linalg.norm(normal)
                if normal@a<0:normal=-normal
                mask &= np.einsum('...i,i->...',source-a,normal)<=0
        expected=np.linalg.norm(mask@grad,axis=-1) if field=='grad_norm' else mask@weights
        np.testing.assert_allclose(result[:,0],expected,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(result[:,1],signed_distance(points,c),atol=2e-6)


@pytest.mark.parametrize('order',[0,1,2])
def test_gpu_selected_curvature_order(execute,order):
    c=validate3d(dict(kernel='w4',radius=2.,h=.5,curvature_order=order))
    points,result=execute(c,2,'Hxx')
    distance,normal=signed_distance(points,c,True)
    expected=select(axisymmetric(distance/c['h'],'w4',curvature=c['h']/c['radius'],curvature_order=order),normal,'Hxx')
    np.testing.assert_allclose(result[:,0],expected,atol=3e-5,rtol=3e-5)
