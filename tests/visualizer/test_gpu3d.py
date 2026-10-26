"""Optional real WebGPU/WGSL rendering checks (pip install wgpu for this test).

The app itself has no wgpu-py dependency. This runs its *actual* WGSL through
wgpu-native, including texture filtering, uniforms, compositing and readback.
"""
from pathlib import Path

import numpy as np
import pytest

wgpu = pytest.importorskip('wgpu')


@pytest.fixture(scope='module')
def render_volume():
    try:
        adapter=wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
        device=adapter.request_device_sync()
    except Exception as exc:
        pytest.skip(f'No offscreen WebGPU adapter: {exc}')
    shader=device.create_shader_module(code=(Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static/volume.wgsl').read_text())
    pipeline=device.create_render_pipeline(layout='auto',vertex={'module':shader,'entry_point':'vertexMain'},fragment={'module':shader,'entry_point':'fragmentMain','targets':[{'format':'rgba8unorm'}]},primitive={'topology':'triangle-list'})
    n=25
    axis=np.linspace(-.5,.5,n)
    z,y,x=np.meshgrid(axis,axis,axis,indexing='ij')
    radius=np.sqrt(x*x+y*y+z*z)
    voxels=np.stack([np.clip(.5-(radius-.28)*4,0,1),(radius-.28)/.2,np.ones_like(radius),np.zeros_like(radius)],axis=-1).astype('<f2')
    texture=device.create_texture(size=(n,n,n),dimension='3d',format='rgba16float',usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
    device.queue.write_texture({'texture':texture},voxels,{'bytes_per_row':n*8,'rows_per_image':n},(n,n,n))
    sampler=device.create_sampler(mag_filter='linear',min_filter='linear')
    lut_texture=device.create_texture(size=(256,1,1),format='rgba8unorm',usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
    uniform=device.create_buffer(size=128,usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
    group=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':{'buffer':uniform}},{'binding':1,'resource':texture.create_view()},{'binding':2,'resource':sampler},{'binding':3,'resource':lut_texture.create_view()}])
    output=device.create_texture(size=(64,64,1),format='rgba8unorm',usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
    def render(steps=128,opacity=.15,mode=1.,plane=0.,offset=0.,lighting=0.,yaw=0.,pitch=0.,constant_color=False,gradient=False):
        samples=voxels.copy()
        if gradient:
            samples[:,:,:,0]=-x*np.exp(-((radius-.28)/.07)**2)
        device.queue.write_texture({'texture':texture},samples,{'bytes_per_row':n*8,'rows_per_image':n},(n,n,n))
        data=np.zeros((8,4),dtype='<f4')
        data[0]=[2*np.cos(pitch)*np.sin(yaw),2*np.sin(pitch),2*np.cos(pitch)*np.cos(yaw),1]
        data[1,:3]=[np.cos(yaw),0,-np.sin(yaw)]
        data[2,:3]=[-np.sin(pitch)*np.sin(yaw),np.cos(pitch),-np.sin(pitch)*np.cos(yaw)]
        data[3,:3]=-data[0,:3]/2
        data[4]=[-.35,.35,.5,steps] if gradient else [0,1,.5,steps];data[5]=[mode,.75,0,lighting];data[6,2]=offset;data[7,2]=plane
        device.queue.write_buffer(uniform,0,data)
        lut=np.array([[90,170,80,round(opacity*255)] if constant_color else [i,255-i,80,round(opacity*255)] for i in range(256)],dtype='uint8')
        if gradient:lut[:,3]=np.round(opacity*255*abs(np.linspace(-1,1,256))).astype('uint8')
        device.queue.write_texture({'texture':lut_texture},lut,{'bytes_per_row':1024},(256,1,1))
        encoder=device.create_command_encoder()
        render_pass=encoder.begin_render_pass(color_attachments=[{'view':output.create_view(),'clear_value':(0,0,0,1),'load_op':'clear','store_op':'store'}])
        render_pass.set_pipeline(pipeline);render_pass.set_bind_group(0,group);render_pass.draw(3);render_pass.end();device.queue.submit([encoder.finish()])
        return np.frombuffer(device.queue.read_texture({'texture':output},{'bytes_per_row':256,'rows_per_image':64},(64,64,1)),dtype='uint8').reshape(64,64,4).copy()
    yield render
    texture.destroy();lut_texture.destroy();uniform.destroy();output.destroy();device.destroy()


def test_shader_volume_and_transfer_opacity_produce_nonempty_different_images(render_volume):
    transparent=render_volume(opacity=0)
    volume=render_volume(opacity=.15)
    assert np.mean(abs(volume.astype(float)-transparent))>10
    # Constant opacity makes the low-value front samples dominate. Check color
    # displacement, rather than assuming a particular colormap channel is high.
    assert np.linalg.norm(volume[32,32,:3].astype(float)-transparent[32,32,:3])>80
    assert np.all(volume[:,:,3]==255)


def test_step_correction_keeps_opacity_stable_across_quality_settings(render_volume):
    coarse=render_volume(steps=64)
    fine=render_volume(steps=512)
    assert np.mean(abs(coarse.astype(float)-fine))<1.2


def test_legacy_slice_options_do_not_affect_volume_pixels(render_volume):
    reference=render_volume(mode=1,plane=0)
    for offset in [0,.35]:
        np.testing.assert_array_equal(render_volume(mode=0,plane=1,offset=offset),reference)


def test_gradient_lighting_changes_volume_pixels(render_volume):
    unlit=render_volume(lighting=0)
    lit=render_volume(lighting=1)
    assert np.mean(abs(unlit.astype(float)-lit))>1


def test_zero_opacity_slice_cannot_obscure_volume(render_volume):
    for yaw,pitch in [(0,0),(.7,.4),(2.8,-.8)]:
        background=render_volume(opacity=0,yaw=yaw,pitch=pitch)
        sliced=render_volume(opacity=0,plane=1,yaw=yaw,pitch=pitch)
        np.testing.assert_array_equal(sliced,background)


def test_rotating_volume_keeps_central_ray_continuous(render_volume):
    centres=[]
    for yaw,pitch in [(0,0),(.3,.2),(1.57,0),(2.8,-.8),(3.14,0),(-1.57,.7)]:
        coarse=render_volume(steps=64,yaw=yaw,pitch=pitch,constant_color=True)
        fine=render_volume(steps=512,yaw=yaw,pitch=pitch,constant_color=True)
        assert np.mean(abs(coarse.astype(float)-fine))<1.3
        centres.append(coarse[32,32,:3])
    assert np.ptp(np.array(centres,dtype=float),axis=0).max()<4


def test_signed_gradient_lobes_remain_visible_across_rotations(render_volume):
    for yaw,pitch in [(0,0),(.7,.4),(1.57,0),(2.8,-.8)]:
        background=render_volume(gradient=True,opacity=0,yaw=yaw,pitch=pitch)
        coarse=render_volume(gradient=True,steps=64,yaw=yaw,pitch=pitch)
        fine=render_volume(gradient=True,steps=512,yaw=yaw,pitch=pitch)
        assert np.mean(abs(coarse.astype(float)-background))>2
        assert np.mean(abs(coarse.astype(float)-fine))<1.5
