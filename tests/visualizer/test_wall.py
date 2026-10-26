"""Independent integral, common-work, regional closure and web transport gates."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry, Request
from warpSPHBoundaries.visualizer.wall import validate,generate,integrate,contractions,compute_wall,compare_wall
from test_server import request

FIXTURES=Path(__file__).parents[1]/'fixtures'
DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('fraction',[0,.37,1])
def test_generator_exact_volume_and_interface(dimension,fraction):
    c=validate(dict(dimension=dimension,fraction=fraction,subdivisions=3,grading=1.5,angle=27))
    geo,labels,R=generate(c)
    volumes=np.array([abs(np.linalg.det((v[1:]-v[0]).T))/(2 if dimension==2 else 6) for v in geo.elements])
    total=c['width']*c['thickness']*(c['depth'] if dimension==3 else 1)
    np.testing.assert_allclose(volumes.sum(),total,rtol=1e-13)
    np.testing.assert_allclose(volumes[labels=='porous'].sum(),total*fraction,rtol=1e-13,atol=1e-14)
    cut=c['width']*(fraction-.5)
    for vertices,label in zip(geo.elements,labels):
        x=(vertices@R)[:,0]
        assert max(x)<=cut+1e-14 if label=='porous' else min(x)>=cut-1e-14


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float32',1e-5),('float64',1e-10)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_native_transport_matches_independent_baseline(dimension,precision,tol,kernel,device):
    cases=json.loads((FIXTURES/'fluxbridge-v1.json').read_text())['cases']
    c=next(v for v in cases if v['dimension']==dimension and v['measure']=='volume' and v['kernel']==kernel and v['query']==[0]*dimension)
    options=validate(dict(dimension=dimension,kernel=kernel,precision=precision,h=c['h'],device=device))
    req,b,pairs,_=integrate(options,Geometry([c['vertices']],dimension),np.array([c['query']],float))
    lookup={r:i for i,r in enumerate(req)}
    chosen=[(lookup[Request(**r)],v) for r,v in zip(c['requests'],c['values']) if Request(**r) in lookup]
    ids=[i for i,_ in chosen]
    ref=np.array([v for _,v in chosen],float)
    assert max(abs(pairs[0,ids]-ref)/(1+abs(ref)))<tol


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('degree',[0,1,2,3])
def test_polynomial_material_contractions_common_work(dimension,degree,device):
    c=validate(dict(dimension=dimension,device=device,degree=degree,precision='float64',fluid=[.3,.7,-.2],skeleton=[.1,-.2,.05],
        fluid_gradient=[[.1,.2,.3],[-.2,.1,.4],[.3,.1,-.1]],skeleton_gradient=[[.05,0,.1],[.1,0,.05],[0,.1,.2]],
        angle=13,tensor_angle=33))
    geo,_,R=generate(c);q=np.array([[.1,.07]+([.13] if dimension==3 else [])])
    req,b,pairs,_=integrate(c,geo,q)
    moments=pairs.sum(axis=0,keepdims=True);result=contractions(c,q,req,moments,R)
    # Independently contract the stored moments by dense polynomial algebra.
    lookup={r:i for i,r in enumerate(req)}
    idx=lambda a:moments[0,lookup[Request(tuple(a))]]
    zero=(0,)*dimension;t=R[:,0];xi=q[0]@t/c['width']+.5
    linear={zero:1+.25*xi}
    for j in range(dimension):
        a=[0]*dimension;a[j]=1;linear[tuple(a)]=.25*t[j]/c['width']
    def mul(a,b):
        out={}
        for i,v in a.items():
            for j,w in b.items():
                key=tuple(x+y for x,y in zip(i,j));out[key]=out.get(key,0)+v*w
        return out
    weight={zero:1.}
    for _ in range(degree):weight=mul(weight,linear)
    angle=np.deg2rad(c['tensor_angle']);a=np.zeros(dimension);a[:2]=[np.cos(angle),np.sin(angle)]
    tensor=R@(c['resistance']*(np.eye(dimension)+(c['anisotropy']-1)*np.outer(a,a)))@R.T
    A=np.array(c['fluid_gradient'])[:dimension,:dimension]-np.array(c['skeleton_gradient'])[:dimension,:dimension]
    v=np.array(c['fluid'][:dimension])-np.array(c['skeleton'][:dimension])+A@q[0]
    velocity=[]
    for i in range(dimension):
        p={zero:v[i]}
        for j in range(dimension):
            a=[0]*dimension;a[j]=1;p[tuple(a)]=A[i,j]
        velocity.append(p)
    reference_force=np.zeros(dimension);energy=0.
    for i in range(dimension):
        for j in range(dimension):
            reference_force[i]-=tensor[i,j]*sum(co*idx(a) for a,co in mul(weight,velocity[j]).items())
            energy+=tensor[i,j]*sum(co*idx(a) for a,co in mul(mul(weight,velocity[i]),velocity[j]).items())
    np.testing.assert_allclose(result['drag'][0],reference_force,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(result['dissipation'][0],energy,atol=1e-12,rtol=1e-12)
    assert energy>=0
    np.testing.assert_array_equal(result['reaction'],-result['drag'])
    assert np.linalg.eigvalsh(result['mobility'][0]).min()>0
    assert np.linalg.eigvalsh(result['mobility'][0]).max()<=1+1e-12


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('degree',[0,3])
def test_regional_closure_locality_and_mesh_refinement(dimension,degree):
    config=dict(dimension=dimension,degree=degree,precision='float64',queries=[[.13,.09]+([.11] if dimension==3 else [])],subdivisions=1)
    base=compute_wall(config);full=compute_wall({**config,'exhaustive':True});refined=compute_wall({**config,'subdivisions':2})
    for region in ('solid','porous','total','free'):
        for key,value in base['terms'][region].items():
            np.testing.assert_allclose(value,full['terms'][region][key],atol=1e-11,rtol=1e-11)
            np.testing.assert_allclose(value,refined['terms'][region][key],atol=1e-11,rtol=1e-11)
    parts=base['terms'];np.testing.assert_allclose(sum(np.array(parts[r]['occupancy']) for r in ('solid','porous','free')),1,atol=1e-15)
    np.testing.assert_allclose(sum(np.array(parts[r]['gradient']) for r in ('solid','porous','free')),0,atol=1e-15)
    disabled=compute_wall({**config,'porous_enabled':False})
    np.testing.assert_array_equal(disabled['terms']['total']['occupancy'],disabled['terms']['solid']['occupancy'])
    np.testing.assert_array_equal(disabled['terms']['total']['drag'],0)
    zero=compute_wall({**config,'resistance':0})
    np.testing.assert_array_equal(zero['terms']['total']['drag'],0)
    np.testing.assert_array_equal(zero['terms']['total']['dissipation'],0)


def test_http_wall_validation_and_finite_json():
    h=request('/api/wall',json.dumps(dict(resolution=3,subdivisions=1)).encode())
    h.do_POST();assert h.status==200
    result=json.loads(h.wfile.getvalue());assert result['kind']=='fluxbridge-wall'
    assert len(result['queries'])==9
    assert 'arXiv:2504.07739v2' in result['contract']['reference']
    assert 'not solid fraction or porosity' in result['contract']['occupancy']
    for config in [[],dict(dimension=4),dict(precision='float32',backend='analytic'),dict(fraction=-1),dict(geometry='fibres',dimension=3),dict(porosity=.5)]:
        h=request('/api/wall',json.dumps(config).encode());h.do_POST();assert h.status==400
    h=request('/api/wall',b'{}',origin='https://example.invalid');h.do_POST();assert h.status==403


@pytest.mark.parametrize('geometry',['disk','fibres'])
def test_standalone_cylinder_sections_have_no_effective_drag(geometry):
    result=compute_wall(dict(geometry=geometry,queries=[[0.,.1]],precision='float64'))
    assert set(result['regions'])=={'solid'}
    np.testing.assert_array_equal(result['terms']['total']['drag'],0)
    assert 'transverse' in result['contract']['fibre_flow']
    assert 'normal_response' not in result['terms']['total']
    assert 'normal_relative' not in result['terms']['solid']


@pytest.mark.parametrize('plane,axes',[('xy',[0,1]),('xz',[0,2]),('yz',[1,2])])
def test_coordinate_slices_preserve_physical_axes(plane,axes):
    r=compute_wall(dict(dimension=3,slice_plane=plane,slice=.13,resolution=3,subdivisions=1))
    q=np.array(r['queries']);assert r['plot_axes']==axes
    np.testing.assert_array_equal(q[:,next(i for i in range(3) if i not in axes)],.13)
    xx,yy=np.meshgrid(*r['axes'])
    np.testing.assert_array_equal(q[:,axes[0]],xx.ravel())
    np.testing.assert_array_equal(q[:,axes[1]],yy.ravel())


@pytest.mark.parametrize('dimension',[2,3])
def test_static_wall_export_reopens_native_data(tmp_path,dimension):
    import subprocess,sys
    config=tmp_path/'wall.json';config.write_text(json.dumps(dict(dimension=dimension,resolution=3,subdivisions=1)))
    site=tmp_path/'site'
    subprocess.run([sys.executable,'-m','warpSPHBoundaries.visualizer','--wall','--export',str(site),'--config',str(config)],check=True,capture_output=True)
    saved=json.loads((site/'wall-example.json').read_text())
    assert saved['kind']=='fluxbridge-wall'
    assert saved['dimension']==dimension
    assert saved['contract']['contraction_precision'].startswith('host float64')
    assert (site/'wall.html').is_file() and (site/'wall-app.js').is_file()
    ordinary=json.loads((site/'example.json').read_text())
    assert ordinary['dimension']==2 and ordinary['schema_version']==1
    assert ordinary.get('kind')!='fluxbridge-wall'


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-12),('float32',1e-5)])
def test_web_provider_uses_explicit_device_preparation(device,precision,tol):
    config=dict(dimension=3,device=device,precision=precision,queries=[[.13,.09,.11]],subdivisions=1)
    host=compute_wall(config);native=compute_wall({**config,'preparation':'device'})
    assert native['contract']['preparation']=='device'
    assert native['timing'][0]['quadrature']['preparation']=='device'
    for region,fields in host['terms'].items():
        for key,value in fields.items():
            np.testing.assert_allclose(native['terms'][region][key],value,atol=tol,rtol=tol)


@pytest.mark.parametrize('dimension',[2,3])
def test_wall_whole_support_containment_and_physical_response(dimension):
    c=dict(dimension=dimension,precision='float64',fraction=1,thickness=2,depth=1,h=.05,
        subdivisions=1,queries=[[.13,.09]+([.11] if dimension==3 else [])])
    if dimension==3:c['preparation']='device'
    r=compute_wall(c);v=r['terms']
    np.testing.assert_allclose(v['porous']['occupancy'],1,atol=1e-12)
    np.testing.assert_allclose(np.array(v['porous']['gradient'])*c['h'],0,atol=1e-12)
    np.testing.assert_allclose(np.array(v['porous']['hessian'])*c['h']**2,0,atol=1e-12)
    np.testing.assert_allclose(v['free']['second'],0,atol=1e-12)
    np.testing.assert_allclose(v['porous']['dissipation'],4,atol=1e-11)
    np.testing.assert_allclose(v['porous']['normal_response'],1/1.4,atol=1e-12)


@pytest.mark.parametrize('device',DEVICES)
def test_chunked_wall_matches_one_native_batch(device):
    c=validate(dict(dimension=3,device=device,precision='float64',resolution=3,subdivisions=1,preparation='device'))
    result=compute_wall(c);geo,labels,_=generate(c)
    req,b,pairs,_=integrate(c,geo,np.array(result['queries']))
    expected={r:np.zeros((len(result['queries']),len(req))) for r in ('solid','porous')}
    for row,(qi,ei) in enumerate(zip(b.neighbors.query,b.neighbors.element)):expected[labels[ei]][qi]+=pairs[row]
    for r,value in expected.items():np.testing.assert_allclose(result['moments'][r],value,atol=1e-12,rtol=1e-12)
    assert len(result['timing'])==3 and result['static_prepare_ms']>=0
    assert result['elapsed_ms']>=result['static_prepare_ms']


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('device',DEVICES)
def test_native_comparison_preserves_physical_setup_and_probes(dimension,device):
    c=dict(dimension=dimension,degree=3,precision='float32',subdivisions=1,device=device,
        search='device',queries=[[.13,.09]+([.11] if dimension==3 else []),[0.,.19999999999999996]+([0.] if dimension==3 else [])])
    if dimension==3:c['preparation']='device'
    result=compare_wall(c);base=compute_wall(c)
    assert result['kind']=='fluxbridge-wall' and result['contract']['search']=='device'
    assert [r['key'] for r in result['comparisons']['rows']]==['support_search','mesh_refinement','precision']
    for row in result['comparisons']['rows']:
        assert row['status']=='measured'
        assert row['reference_config']['h']==base['config']['h']
        assert row['reference_config']['width']==base['config']['width']
        assert max(row['normalized_moment_mixed'].values())<1e-5
        reference=compute_wall({**c,**row['changed']})
        for region,fields in base['terms'].items():
            for key,value in fields.items():
                delta=abs(np.asarray(value)-reference['terms'][region][key])
                assert row['terms'][region][key]['max_absolute']==float(delta.max())
    np.testing.assert_array_equal(result['queries'],base['queries'])


def test_comparison_chord_error_and_unavailable_refinement():
    disk=compare_wall(dict(geometry='disk',segments=8,precision='float64',queries=[[.85,.1]]))
    rows=disk['comparisons']['rows']
    assert rows[1]['key']=='geometry_refinement' and 'changes circle geometry' in rows[1]['label']
    assert rows[1]['terms']['solid']['occupancy']['max_absolute']>1e-6
    assert rows[2]['status']=='unavailable'
    capped=compare_wall(dict(subdivisions=12,precision='float64',queries=[[10,10]]))
    assert capped['comparisons']['rows'][1]['status']=='unavailable'


def test_comparison_http_validation_and_static_export(tmp_path):
    h=request('/api/wall/compare',json.dumps(dict(resolution=3,subdivisions=1)).encode());h.do_POST()
    assert h.status==200 and len(json.loads(h.wfile.getvalue())['comparisons']['rows'])==3
    h=request('/api/wall/compare',b'{}',origin='https://example.invalid');h.do_POST();assert h.status==403
    import subprocess,sys
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(resolution=3,subdivisions=1)))
    site=tmp_path/'site'
    subprocess.run([sys.executable,'-m','warpSPHBoundaries.visualizer','--wall','--compare','--export',str(site),'--config',str(config)],check=True,capture_output=True)
    saved=json.loads((site/'wall-example.json').read_text())
    assert len(saved['comparisons']['rows'])==3
    assert saved['comparisons']['rows'][2]['reference_config']['precision']=='float64'
