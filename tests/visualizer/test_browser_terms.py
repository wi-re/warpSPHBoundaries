"""Numerical browser-core parity without starting any browser or Python service."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from warpSPHBoundaries.mesh3d.radial import axisymmetric, select
from warpSPHBoundaries.mesh3d.slender import ball_series
from warpSPHBoundaries.visualizer.terms3d import FIELDS3D


def test_javascript_all_channels_match_python_families():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is only needed for numerical browser-core parity')
    root=Path(__file__).parents[2]
    url=(root/'src/warpSPHBoundaries/visualizer/static/term-engine.js').as_uri()
    script=f'''
import {{FIELD_INFO,axialTerms,selectAxial,ballSeries}} from {json.dumps(url)};
const records=[];
for(const name of ['w2','w4','w6','cubic'])for(const mode of ['plane','sphere','curvature'])for(const d of [-.7,-.01,0,.001,.25,.7,1.1]){{
 const v=axialTerms(d,name,mode==='sphere'?2:null,mode==='curvature'?.25:null),n=[2/3,-2/3,1/3];
 records.push({{name,mode,d,values:FIELD_INFO.map(([id])=>selectAxial(v,n,id))}});
}}
const balls=[];for(const name of ['w2','w4','w6','cubic'])balls.push({{name,values:FIELD_INFO.map(([id])=>ballSeries([.31,.18,.11],.02,name,id))}});
console.log(JSON.stringify({{fields:FIELD_INFO.map(x=>x[0]),records,balls}}));
'''
    result=subprocess.run([node,'--input-type=module','--eval',script],capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)
    assert set(data['fields'])==set(FIELDS3D)
    normal=np.array([[2/3,-2/3,1/3]])
    for row in data['records']:
        inv=axisymmetric([row['d']],row['name'],radius=2 if row['mode']=='sphere' else None,
                         curvature=.25 if row['mode']=='curvature' else None)
        expected=[select(inv,normal,field)[0] for field in data['fields']]
        np.testing.assert_allclose(row['values'],expected,atol=3e-12,rtol=3e-12)
    for row in data['balls']:
        expected=[ball_series([.31,.18,.11],.02,row['name'],field) for field in data['fields']]
        np.testing.assert_allclose(row['values'],expected,atol=1e-14,rtol=1e-10)


def test_javascript_2d_channels_match_independent_python_references():
    from warpSPHBoundaries.visualizer.compute import geometry, validate, polygon_value, sdf, evaluate_model
    from warpSPHBoundaries.validation.atlas import CHANNELS, disk_reference
    from warpSPHBoundaries.validation.distance_only import halfplane_channels
    node=shutil.which('node')
    if not node:
        pytest.skip('Node needed for browser mathematics parity')
    url=(Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static/browser-compute2d.js').as_uri()
    script=f'''
import {{polygonChannels,planeChannels,diskChannels,compute2d}} from {json.dumps(url)};
const points=[[0,0],[.2,-.1],[.9,.1],[1,1],[1.1,-.9],[2,2]],vertices=[[-1,-1],[1,-1],[1,1],[-1,1]],records=[];
for(const kernel of ['w2','w4','w6','cubic']){{
 records.push({{kernel,polygon:points.map(p=>polygonChannels(vertices,p,kernel,32)),disk:points.map(p=>diskChannels(p,1,kernel,32)),plane:[-.9,-.3,0,.2,.95].map(d=>planeChannels(d,[.6,.8],kernel,32))}});
}}
const sampled=compute2d({{geometry:'circle',resolution:17,radius:1,h:.3,models:['sdf','curvature'],sdf_source:'sampled',sdf_spacing:.1,quadrature:32}});
const curvature=compute2d({{geometry:'circle',resolution:17,radius:1,h:.3,models:['curvature'],field:'g1xx',quadrature:32}});
console.log(JSON.stringify({{records,sampled,curvature}}));
'''
    result=subprocess.run([node,'--input-type=module','--eval',script],capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)
    points=np.array([[0,0],[.2,-.1],[.9,.1],[1,1],[1.1,-.9],[2,2]])
    for row in data['records']:
        c=validate(dict(kernel=row['kernel'],h=1))
        vertices,_=geometry(c)
        expected=np.column_stack([polygon_value(vertices,points,{**c,'field':f}) for f in CHANNELS])
        np.testing.assert_allclose(row['polygon'],expected,atol=4e-11,rtol=4e-11)
        np.testing.assert_allclose(row['disk'],disk_reference(points,1,row['kernel'],64),atol=4e-11,rtol=4e-11)
        np.testing.assert_allclose(row['plane'],halfplane_channels([-.9,-.3,0,.2,.95],[[.6,.8]]*5,row['kernel']),atol=4e-11,rtol=4e-11)
    from warpSPHBoundaries.visualizer.compute import compute
    expected=compute(data['sampled']['config'])
    for browser,python in zip(data['sampled']['results'],expected['results']):
        np.testing.assert_allclose(np.array(browser['values'],float),np.array(python['values'],float),atol=2e-6,rtol=2e-6)
    expected=compute(data['curvature']['config'])
    np.testing.assert_allclose(data['curvature']['results'][0]['values'],expected['results'][0]['values'],atol=2e-6,rtol=2e-6)
