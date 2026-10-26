"""Actual compensated-fp32 shader with production host plans, not a CPU emulator.

Coarse samples include aligned face/edge/vertex queries. Software Vulkan checks
numerics, not hardware throughput. Float64 core is separately audited vs mpmath.
"""
import json
from pathlib import Path
import shutil,subprocess
import numpy as np
import pytest
wgpu=pytest.importorskip('wgpu')
STATIC=Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static'
FIELDS=['m00','m100','m010','m001','m200','m110','m101','m020','m011','m002','Gx','Gy','Gz',*[f'g1{a}{b}' for a in 'xyz' for b in 'xyz'],*[f'H{ij}' for ij in ['xx','xy','xz','yy','yz','zz']],'laplacian','grad_norm']
FIELDS2=['m00','m10','m01','m20','m11','m02','Gx','Gy','g1xx','g1xy','g1yx','g1yy','grad_norm']

@pytest.fixture(scope='module')
def execute():
    if not shutil.which('node'):pytest.skip('Node needed for actual production host plans')
    try:d=wgpu.gpu.request_adapter_sync(force_fallback_adapter=True).request_device_sync()
    except Exception as e:pytest.skip(str(e))
    m=d.create_shader_module(code=(STATIC/'analytic3d-compute.wgsl').read_text())
    pipeline=d.create_compute_pipeline(layout='auto',compute={'module':m,'entry_point':'compute_analytic3d'})
    def run(config,fields,dimension=3,series=False,local=False):
        config={**dict(radius=1,width=2,height=2,depth=2,h=1,resolution=17,padding=1,quality=1,models=['analytic']),**config}
        script=f'''
import {{compileAnalytic3d,split32}} from {json.dumps((STATIC/'analytic3d-plan.js').as_uri())};
import {{compileAnalytic2d,compileStrip2d}} from {json.dumps((STATIC/'analytic2d-plan.js').as_uri())};
import {{compileLocal2d}} from {json.dumps((STATIC/'local2d-plan.js').as_uri())};
import {{compileSeries}} from {json.dumps((STATIC/'series-plan.js').as_uri())};
import {{pointTerm,prepare}} from {json.dumps((STATIC/'browser-compute.js').as_uri())};
import {{compute2d}} from {json.dumps((STATIC/'browser-compute2d.js').as_uri())};
const config={json.dumps(config)},dimension={dimension},series={json.dumps(series)},local={json.dumps(local)},rows=[];
for(const field of {json.dumps(fields)}){{const c={{...config,field}},plan=series?compileSeries(c,dimension):local?compileLocal2d(c,c.models[0]==='curvature_analytic'):c.models[0]==='strip'?compileStrip2d(c):dimension===3?compileAnalytic3d(c):compileAnalytic2d(c),f=new Float32Array(plan.uniform.buffer),u=new Uint32Array(plan.uniform.buffer);u[0]=5;const step=split32(((plan.axis??plan.x)[4]-(plan.axis??plan.x)[0])/c.h);f[6]=step[0];f[7]=step[1];if(local){{const start=u[3]*6+f[14]*4,packed=[...plan.table.slice(0,start)];for(const j of [0,4,8,12,16])for(const i of [0,4,8,12,16])packed.push(...plan.table.slice(start+(j*17+i)*12,start+(j*17+i+1)*12));plan.table=new Float32Array(packed);}}let expected=[];
 if(dimension===3){{const g=prepare(c),axis=[0,4,8,12,16].map(i=>plan.axis?plan.axis[i]:plan.x[i]);for(const z of axis)for(const y of axis)for(const x of axis)expected.push(pointTerm([x,y,z],c,series?'ball_series':'analytic',g));}}
 else{{const result=compute2d(c),values=result.results[0].values;for(const j of [0,4,8,12,16])for(const i of [0,4,8,12,16])expected.push(values[j*17+i]);}}
 rows.push({{uniform:[...plan.uniform],table:[...plan.table],expected,field}});
}}console.log(JSON.stringify(rows));
'''
        rows=json.loads(subprocess.run(['node','--input-type=module','--eval',script],check=True,capture_output=True,text=True).stdout)
        output=[]
        for row in rows:
            uniform=d.create_buffer_with_data(data=bytes(row['uniform']),usage=wgpu.BufferUsage.UNIFORM)
            table=d.create_buffer_with_data(data=np.array(row['table'],dtype='<f4'),usage=wgpu.BufferUsage.STORAGE)
            count=5**dimension;out=d.create_buffer(size=count*16,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
            try:
                group=d.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':b}} for i,b in enumerate([uniform,table,out])])
                e=d.create_command_encoder();p=e.begin_compute_pass();p.set_pipeline(pipeline);p.set_bind_group(0,group);p.dispatch_workgroups((count+63)//64);p.end();d.queue.submit([e.finish()]);raw=np.frombuffer(d.queue.read_buffer(out),dtype='<f4').reshape(-1,4)
                assert np.isfinite(raw).all(),row['field']
                actual=np.where(raw[:,2]>.5,raw[:,0],np.nan);expected=np.array(row['expected'],float)
                output.append((row['field'],actual,expected))
            finally:uniform.destroy();table.destroy();out.destroy()
        return output
    yield run
    d.destroy()

@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_analytic_polyhedron_every_field_and_aligned_boundaries(execute,kernel):
    for field,actual,expected in execute(dict(geometry='tetrahedron',kernel=kernel),FIELDS):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)

@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_analytic_2d_polygon_every_channel(execute,kernel):
    c=dict(geometry='triangle',vertices=[[-1,-.9],[1.1,-.7],[-.3,1.1],[-1,1]],kernel=kernel)
    for field,actual,expected in execute(c,FIELDS2,dimension=2):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)

@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('order',[0,2,4])
def test_mean_value_series_all_fields_and_missing_domain(execute,dimension,order,kernel):
    config=dict(geometry='circle' if dimension==2 else 'sphere',kernel=kernel,radius=.2,h=1.2,padding=1.15,series_order=order,models=['disk_series' if dimension==2 else 'ball_series'])
    for field,actual,expected in execute(config,FIELDS2 if dimension==2 else FIELDS,dimension=dimension,series=True):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)


@pytest.mark.parametrize('order',[0,1,2])
def test_analytic_local_curvature_coefficients_all_channels(execute,order):
    config=dict(geometry='circle',radius=.9,h=.3,kernel='w4',models=['curvature_analytic'],curvature_order=order)
    for field,actual,expected in execute(config,FIELDS2,dimension=2,local=True):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)

def test_analytic_polygon_sdf_keeps_undefined_normals_missing(execute):
    config=dict(geometry='rectangle',kernel='w6',models=['sdf_analytic'],height=1)
    for field,actual,expected in execute(config,FIELDS2,dimension=2,local=True):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)


def test_box_analytic_second_moments_gradients_and_hessians(execute):
    fields=['m00','m200','m110','m002','Gx','Gy','Gz','g1xy','g1zz','Hxx','Hxy','Hzz','laplacian','grad_norm']
    for field,actual,expected in execute(dict(geometry='box',kernel='w6',width=1.4,height=.6,depth=1.1,h=.51),fields):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)


def test_analytic_local_sampled_curvature_preserves_the_lattice_model(execute):
    config=dict(geometry='circle',radius=.9,h=.3,kernel='cubic',models=['curvature_analytic'],sdf_source='sampled',sdf_spacing=.15,curvature_order=2)
    for field,actual,expected in execute(config,FIELDS2,dimension=2,local=True):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)


def test_infinite_strip_special_form_all_channels(execute):
    config=dict(geometry='rectangle',kernel='w4',models=['strip'],height=.4,angle=27)
    for field,actual,expected in execute(config,FIELDS2,dimension=2):
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6,equal_nan=True,err_msg=field)
