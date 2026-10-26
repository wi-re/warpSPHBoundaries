"""Run the actual browser 2D WGSL with its production JS host layout.

A software Vulkan adapter tests correctness; its timings are not hardware GPU
performance measurements. Independent Python/JS parity is in test_browser_terms.
"""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

wgpu=pytest.importorskip('wgpu')
ROOT=Path(__file__).parents[2]
STATIC=ROOT/'src/warpSPHBoundaries/visualizer/static'


@pytest.fixture(scope='module')
def execute():
    if not shutil.which('node'):
        pytest.skip('Node is needed to exercise the actual JS dispatch layout')
    try:
        device=wgpu.gpu.request_adapter_sync(force_fallback_adapter=True).request_device_sync()
    except Exception as exc:
        pytest.skip(str(exc))
    module=device.create_shader_module(code=(STATIC/'terms2d-compute.wgsl').read_text())
    pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'compute_terms2d'})
    def run(c,fields=None):
        script=f'''
import {{compile2d}} from {json.dumps((STATIC/'compute2d-plan.js').as_uri())};
import {{FIELDS2D,polygonChannels,planeChannels,diskChannels,signed2d,compute2d,whole2d}} from {json.dumps((STATIC/'browser-compute2d.js').as_uri())};
import fs from 'node:fs';
import {{diskTableChannels}} from {json.dumps((STATIC/'disk-table.js').as_uri())};
const payload={json.dumps(c)},file=payload.models.includes('disk_table')?fs.readFileSync({json.dumps(str(STATIC))}+'/disk-table-'+payload.kernel+'.bin'):null,lookup=file?new Float32Array(file.buffer,file.byteOffset,file.byteLength/4):null;
const plan=compile2d({json.dumps(c)},lookup),{{c:config,geometry:g,x,y}}=plan,vertices=g.vertices.map(p=>p.map(v=>v/config.h)),triangles=g.triangles.map(t=>t.map(p=>p.map(v=>v/config.h))),expected={{}};
for(const model of config.models){{
 const rows=[];for(const py of y)for(const px of x){{const p=[px/config.h,py/config.h];let v;
  if(model==='mesh'){{v=Array(12).fill(0);for(const t of triangles){{const a=polygonChannels(t,p,config.kernel,config.quadrature);v=v.map((x,i)=>x+a[i]);}}}}
  else if(model==='exact'||model==='rectangle')v=polygonChannels(vertices,p,config.kernel,config.quadrature);
  else if(model==='disk_table')v=diskTableChannels(lookup,p,config.radius/config.h,whole2d(config.kernel));
  else if(model==='disk')v=diskChannels(p,config.radius/config.h,config.kernel,config.quadrature);
  else{{const [d,n]=signed2d([px,py],config,g.vertices);v=planeChannels(d/config.h,n,config.kernel,config.quadrature,model==='curvature'?config.h/config.radius:0,config.curvature_order);}}
  rows.push([...v,Math.hypot(v[6],v[7])]);
 }}
 if(config.sdf_source==='sampled'&&['sdf','curvature'].includes(model)){{for(let field=0;field<FIELDS2D.length;field++){{const data=compute2d({{...config,models:[model],field:FIELDS2D[field]}}).results[0].values;for(let i=0;i<rows.length;i++)rows[i][field]=data[i];}}}}expected[model]=rows;
}}
console.log(JSON.stringify({{config,table:[...plan.table],uniform:[...plan.uniform],expected}}));
'''
        raw=subprocess.run(['node','--input-type=module','--eval',script],check=True,capture_output=True,text=True)
        plan=json.loads(raw.stdout);cfg=plan['config'];count=cfg['resolution']**2
        uniform=device.create_buffer(size=96,usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
        table=device.create_buffer_with_data(data=np.array(plan['table'],dtype='<f4'),usage=wgpu.BufferUsage.STORAGE)
        output=device.create_buffer(size=count*16,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
        group=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':{'buffer':uniform}},{'binding':1,'resource':{'buffer':table}},{'binding':2,'resource':{'buffer':output}}])
        results={}
        try:
            for model in cfg['models']:
                results[model]=np.full((count,13),np.nan)
                for field in (range(13) if fields is None else fields):
                    data=bytearray(plan['uniform']);ids=np.frombuffer(data,dtype='<u4');ids[1]=field;ids[3]=['mesh','exact','rectangle','sdf','curvature','disk','disk_table'].index(model)
                    device.queue.write_buffer(uniform,0,data)
                    encoder=device.create_command_encoder();p=encoder.begin_compute_pass();p.set_pipeline(pipeline);p.set_bind_group(0,group);p.dispatch_workgroups((count+63)//64);p.end();device.queue.submit([encoder.finish()])
                    pairs=np.frombuffer(device.queue.read_buffer(output),dtype='<f4').reshape(-1,4)
                    assert np.isfinite(pairs).all(), 'GPU validity uses finite placeholders, never NaN payloads'
                    assert np.isin(pairs[:,2],[0,1]).all()
                    results[model][:,field]=np.where(pairs[:,2]>.5,pairs[:,0],np.nan)
            return results,{k:np.array(v,dtype=float) for k,v in plan['expected'].items()}
        finally:
            uniform.destroy();table.destroy();output.destroy()
    yield run
    device.destroy()


@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_every_polygon_channel_and_mesh_assembly(kernel,execute):
    gpu,reference=execute(dict(resolution=17,kernel=kernel,width=1.8,height=.65,angle=27,h=.6,models=['mesh','exact','rectangle','sdf']))
    for model in gpu:
        np.testing.assert_allclose(gpu[model],reference[model],atol=3e-5,rtol=3e-5,equal_nan=True)
    np.testing.assert_array_equal(gpu['mesh'],gpu['exact'])


@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_every_circle_and_curvature_channel(kernel,execute):
    gpu,reference=execute(dict(geometry='circle',resolution=17,kernel=kernel,radius=.9,h=.3,models=['disk','sdf','curvature']))
    for model in gpu:
        np.testing.assert_allclose(gpu[model],reference[model],atol=5e-5,rtol=5e-5,equal_nan=True)


def test_clockwise_triangle_is_oriented_before_gpu_dispatch(execute):
    gpu,reference=execute(dict(geometry='triangle',resolution=17,vertices=[[-1,-.8],[0,1],[1,-.8],[-1,1]],models=['mesh','sdf']),fields=[0,6,8,12])
    for model in gpu:
        np.testing.assert_allclose(gpu[model][:,[0,6,8,12]],reference[model][:,[0,6,8,12]],atol=4e-5,rtol=4e-5,equal_nan=True)


def test_sampled_sdf_and_curvature_use_the_same_lattice_as_float64(execute):
    gpu,reference=execute(dict(geometry='circle',resolution=17,radius=.9,h=.3,sdf_source='sampled',sdf_spacing=.15,models=['sdf','curvature']))
    for model in gpu:
        np.testing.assert_allclose(gpu[model],reference[model],atol=5e-4,rtol=5e-4,equal_nan=True)


def test_small_circle_preserves_original_disk_and_polygon_domains(execute):
    gpu,reference=execute(dict(geometry='circle',resolution=17,radius=.2,h=1.2,models=['mesh','disk']))
    for model in gpu:
        np.testing.assert_allclose(gpu[model],reference[model],atol=3e-5,rtol=3e-5,equal_nan=True)


def test_sampled_polygon_normals_and_gradients(execute):
    gpu,reference=execute(dict(resolution=17,width=1.8,height=.65,angle=27,h=.6,sdf_source='sampled',sdf_spacing=.15,models=['sdf']),fields=[0,6,8,12])
    np.testing.assert_allclose(gpu['sdf'][:,[0,6,8,12]],reference['sdf'][:,[0,6,8,12]],atol=5e-4,rtol=5e-4,equal_nan=True)


def test_editable_triangle_has_no_exterior_normal_holes_at_fine_resolution(execute):
    # A coarse grid misses near-equal edge/vertex distances outside the triangle.
    # These have a unique projection, whereas interior medial ties stay missing.
    fields=[6,7,8,9,12]
    gpu,reference=execute(dict(geometry='triangle',resolution=193,
        vertices=[[-1,-.9],[1.1,-.7],[-.3,1.1],[-1,1]],models=['sdf']),fields=fields)
    assert np.isfinite(reference['sdf'][:,fields]).all()
    assert np.isfinite(gpu['sdf'][:,fields]).all()
    np.testing.assert_allclose(gpu['sdf'][:,fields],reference['sdf'][:,fields],atol=2e-5,rtol=2e-5)


def test_true_medial_and_boundary_corner_normals_remain_missing(execute):
    gpu,reference=execute(dict(resolution=25,h=2,padding=1,models=['sdf']),fields=[0,6])
    centre=12*25+12
    assert np.isfinite(gpu['sdf'][centre,0])
    assert np.isnan(gpu['sdf'][centre,6])
    np.testing.assert_allclose(gpu['sdf'][:,[0,6]],reference['sdf'][:,[0,6]],atol=3e-5,rtol=3e-5,equal_nan=True)


@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_gpu_disk_lookup_all_channels(execute,kernel):
    gpu,reference=execute(dict(geometry='circle',resolution=17,kernel=kernel,radius=.4,h=.8,models=['disk_table']))
    np.testing.assert_allclose(gpu['disk_table'],reference['disk_table'],atol=3e-6,rtol=3e-6)

@pytest.mark.parametrize('order',[0,1,2])
def test_gpu_2d_curvature_selected_order(execute,order):
    gpu,reference=execute(dict(geometry='circle',resolution=17,kernel='w4',radius=.9,h=.3,curvature_order=order,models=['curvature']))
    np.testing.assert_allclose(gpu['curvature'],reference['curvature'],atol=5e-5,rtol=5e-5)
