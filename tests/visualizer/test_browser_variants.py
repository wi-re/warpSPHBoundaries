"""Independent checks of browser special forms and declared approximation errors."""
import json,subprocess,shutil
from pathlib import Path
import mpmath as mp
import numpy as np
import pytest
from warpSPHBoundaries.edge.curvature import moments
from warpSPHBoundaries.edge.tier4 import disk_series
STATIC=Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static'

def node(script):
    if not shutil.which('node'):pytest.skip('Node needed for browser special forms')
    return json.loads(subprocess.run(['node','--input-type=module','--eval',script],capture_output=True,text=True,check=True).stdout)

@pytest.mark.parametrize('name',['w2','w4','w6','cubic'])
def test_analytic_curvature_coefficients_against_arbitrary_precision(name):
    cases=[(d,order) for d in [-.3,0,.25,.8] for order in [0,1,2]]
    data=node(f'''import {{analyticPlane2d}} from {json.dumps((STATIC/'analytic2d.js').as_uri())};console.log(JSON.stringify({json.dumps(cases)}.map(([d,order])=>analyticPlane2d(d,[.6,.8],{json.dumps(name)},.25,order))));''')
    for (d,order),actual in zip(cases,data):
        zero=np.array(moments(name,d,0,normal=(.6,.8),dps=55),float)
        plus=np.array(moments(name,d,.25,normal=(.6,.8),dps=55),float)
        minus=np.array(moments(name,d,-.25,normal=(.6,.8),dps=55),float)
        expected=zero if order==0 else zero+(plus-minus)/2 if order==1 else plus
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)

@pytest.mark.parametrize('name',['w2','w4','w6','cubic'])
def test_small_disk_series_matches_independent_scalar_derivation(name):
    data=node(f'''import {{diskSeries2d}} from {json.dumps((STATIC/'analytic2d.js').as_uri())};console.log(JSON.stringify([0,2,4].map(order=>diskSeries2d([.31,.18],.02,{json.dumps(name)},0,order))));''')
    D=mp.sqrt(mp.mpf(.31)**2+mp.mpf(.18)**2)
    for order,actual in zip([0,2,4],data):
        expected=disk_series(name,.02,float(D),order//2,dps=55)
        assert abs(actual-float(expected))<1e-13

@pytest.mark.parametrize('name',['w2','w4','w6','cubic'])
def test_disk_lookup_all_channels_against_boundary_quadrature(name):
    data=node(f'''
import fs from 'node:fs';import {{diskTableChannels}} from {json.dumps((STATIC/'disk-table.js').as_uri())};import {{diskChannels,whole2d}} from {json.dumps((STATIC/'browser-compute2d.js').as_uri())};
const bytes=fs.readFileSync({json.dumps(str(STATIC/f'disk-table-{name}.bin'))}),table=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);let maximum=0;
for(const R of [.05,.2,.49,.5,.51,.9,1,1.01,2,10,40])for(let i=0;i<=24;i++){{const D=Math.max(0,R-1)+Math.min(2,R+1)*i/24,p=[D*.6,D*.8],a=diskTableChannels(table,p,R,whole2d({json.dumps(name)})),b=diskChannels(p,R,{json.dumps(name)},48);maximum=Math.max(maximum,...a.map((x,j)=>Math.abs(x-b[j])));}}
console.log(JSON.stringify({{maximum,finite:table.every(Number.isFinite)}}));
''')
    assert data['finite']
    assert data['maximum'] < (8e-5 if name=='cubic' else 3e-7)


def test_python_curvature_order_zero_matches_plane_and_validates_orders():
    from warpSPHBoundaries.visualizer.compute import compute,validate
    cfg=dict(geometry='circle',resolution=17,radius=1,h=.3,field='Gx',models=['sdf','curvature'],curvature_order=0)
    rows=compute(cfg)['results']
    np.testing.assert_allclose(np.array(rows[0]['values'],float),np.array(rows[1]['values'],float),atol=1e-12,equal_nan=True)
    with pytest.raises(ValueError,match='expansion order'):validate({'curvature_order':3})
