"""Portable plans, native contracts and production WGSL numerical execution.

Dawn is optional; its concrete adapter is reported, never inferred to be hardware.
"""
import fcntl
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import numpy as np
import pytest
from warpSPHBoundaries.visualizer.wall import validate, generate, requests_for, contractions, compute_wall

ROOT=Path(__file__).parents[2]
STATIC=ROOT/'src/warpSPHBoundaries/visualizer/static'
RUNNER=ROOT/'tests/visualizer/js/fluxbridge-gpu-runner.mjs'
DAWN=Path(os.environ.get('FLUXBRIDGE_WEBGPU_MODULE',ROOT/'.tmp/dawn-check/node_modules/webgpu/index.js'))


def run(payload,gpu=False):
    if not shutil.which('node'):pytest.skip('Node required for browser plans')
    if gpu and not DAWN.is_file():pytest.skip('Optional Dawn WebGPU runtime is not installed')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
        result=subprocess.run(['node',str(RUNNER)],input=json.dumps(payload),capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)


def test_generated_profiles_match_exact_native_plans():
    module=runpy.run_path(str(ROOT/'scripts/visualizer/build_fluxbridge_profiles.py'))
    assert json.loads((STATIC/'fluxbridge-profiles.json').read_text())==json.loads(json.dumps(module['build']()))


@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('geometry',['wall','disk','fibres'])
def test_browser_geometry_request_and_contraction_contract(d,geometry):
    if d==3 and geometry!='wall':pytest.skip('3D fibre geometry is a separate contract')
    c=validate(dict(dimension=d,geometry=geometry,degree=3,angle=27,grading=1.5,fraction=.37,center=[.2,-.3,.1],
        fluid_gradient=[[.1,.2,.3],[-.1,.3,.2],[.2,.1,-.1]],skeleton_gradient=[[.03,.01,0],[0,.02,.01],[0,0,.01]]))
    geo,regions,R=generate(c);req=requests_for(d,c['degree']);queries=np.array([[.13,.07]+([.12] if d==3 else [])])
    # Arbitrary moments check algebra independently of native/browser integrators.
    moments=np.random.default_rng(74).uniform(.001,.1,(len(queries),len(req)))
    actual=run(dict(host=dict(config=c,queries=queries.tolist(),moments=moments.tolist())))
    np.testing.assert_allclose(actual['elements'],geo.elements,atol=2e-15,rtol=2e-15)
    assert actual['regions']==regions.tolist()
    assert actual['requests']==[dict(alpha=list(r.alpha),derivative=list(r.derivative)) for r in req]
    expected=contractions(c,queries,req,moments,R)
    for name,value in expected.items():np.testing.assert_allclose(actual['terms'][name],value,atol=2e-13,rtol=2e-13)


def test_wgsl_independent_volume_fixtures():
    base=json.loads((ROOT/'tests/fixtures/fluxbridge-v1.json').read_text())['cases']
    cases=[c for c in base if c['measure']=='volume']
    extra=json.loads((ROOT/'tests/fixtures/fluxbridge-tetquad-v1.json').read_text())['cases']
    cases+=extra
    result=run(dict(cases=cases),gpu=True)
    errors=[]
    for case,row in zip(cases,result['results']):
        a=np.array(row['values']);b=np.array(case['values'],float)
        error=np.max(abs(a-b)/(1+abs(b)));errors.append(float(error))
        assert error<1e-5,(case.get('label'),case.get('dimension',3),case['kernel'],error)
    out=Path('/tmp/fluxbridge-webgpu-evidence');out.mkdir(exist_ok=True)
    (out/'independent-core.json').write_text(json.dumps(dict(adapter=result['adapter'],cases=len(cases),channels=sum(len(c['requests']) for c in cases),worst_mixed=max(errors),errors=errors),indent=2))


@pytest.mark.parametrize('d,kernel',[(2,k) for k in ('w2','w4','w6','cubic')]+[(3,'w6')])
def test_live_browser_wall_matches_native_all_terms(d,kernel):
    config=dict(dimension=d,kernel=kernel,degree=3,subdivisions=1,angle=17,center=[.1,-.1,.1],resistance=7,
        fluid=[.3,.7,-.2],skeleton=[.1,-.2,.05],tensor_angle=33,
        fluid_gradient=[[.1,.2,.3],[-.2,.1,.4],[.3,.1,-.1]],skeleton_gradient=[[.05,0,.1],[.1,0,.05],[0,.1,.2]],
        queries=[[.1,.07]+([.13] if d==3 else []),[-.37,-.03]+([.1] if d==3 else []),[3.,3.]+([3.] if d==3 else [])])
    result=run(dict(walls=[config]),gpu=True);browser=result['walls'][0]
    native=compute_wall({**config,'precision':'float64'})
    for region,fields in native['terms'].items():
        for key,value in fields.items():np.testing.assert_allclose(browser['terms'][region][key],value,atol=2e-5,rtol=2e-5,err_msg=region+'/'+key)
    assert browser['contract']['precision']=='float32'
    assert browser['contract']['quadrature']['mixed_difference']<=1e-5
    assert browser['contract']['kernel_dimension']==d


def test_integer_reduction_survives_compiler_reassociation_and_cancellation():
    largest=float(np.finfo(np.float32).max)
    weights=[[1e30,1,-1e30,1],[-1e30,-1,1e30,-1],
             [largest]*4096+[-largest]*4095,[largest,-largest,1e-20]]
    result=run(dict(reductions=weights),gpu=True)
    np.testing.assert_array_equal(result['reductions'][:3],[2.,-2.,largest])
    np.testing.assert_allclose(result['reductions'][3],np.float32(1e-20),rtol=1e-6,atol=0)


def test_export_browser_assets_without_native_imports(tmp_path):
    module=runpy.run_path(str(ROOT/'scripts/visualizer/export_static.py'))
    site=module['export_static'](tmp_path/'repository-prefix')
    for name in ['wall.html','wall-engine.js','wall-worker.js','wall-browser.js',
                 'fluxbridge-plan.js','fluxbridge-gpu.js','fluxbridge-compute.wgsl',
                 'fluxbridge-profiles.json','index.html','volume.html','.nojekyll']:
        assert (site/name).is_file()
    assert 'warpSPHBoundaries' not in module  # exporter imports only standard library


@pytest.mark.parametrize('dimension',[2,3])
def test_browser_support_refinement_comparisons_without_native_service(dimension):
    result=run(dict(compare=True,walls=[dict(dimension=dimension,resolution=3,subdivisions=1,degree=3)]),gpu=True)['walls'][0]
    assert len(result['comparisons']['rows'])==2
    for row in result['comparisons']['rows']:
        assert row['status']=='measured'
        for fields in row['terms'].values():
            for metric in fields.values():assert metric['max_mixed']<2e-5


def test_browser_preserves_solid_fibres_and_disabled_porous_provider():
    configs=[dict(geometry=g,queries=[[.1,.07]],degree=3) for g in ['disk','fibres']]
    configs.append(dict(porous_enabled=False,queries=[[.1,.07]]))
    result=run(dict(walls=configs),gpu=True)
    for c,w in zip(configs,result['walls']):
        np.testing.assert_array_equal(w['terms']['total']['drag'],0)
        np.testing.assert_array_equal(w['terms']['total']['dissipation'],0)
        if c.get('geometry'):
            assert 'normal_response' not in w['terms']['total']
            assert set(w['regions'])=={'solid'}
        else:np.testing.assert_array_equal(w['moments']['porous'],0)
