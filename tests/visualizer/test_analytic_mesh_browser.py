"""Closed-form browser polyhedra against the rational/mpmath analytic reference.

Independent checks of the reference (simplex polynomials, boundary fractions,
subdivision and optional Strobl overlap) are in tests/mesh3d. No quadrature is
used as the accuracy oracle here. These tolerances describe finite cases only.
"""
import json
from pathlib import Path
import shutil
import subprocess

import mpmath as mp
import numpy as np
import pytest

from warpSPHBoundaries.mesh3d import Tetrahedron

STATIC=Path(__file__).parents[2]/'src/warpSPHBoundaries/visualizer/static'
UNIT=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]


def javascript(script):
    if not shutil.which('node'):
        pytest.skip('Node needed to check the actual browser analytic module')
    return json.loads(subprocess.run(['node','--input-type=module','--eval',script],
        check=True,capture_output=True,text=True).stdout)


def reference(t,name):
    alphas=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    gradient=t.kernel_gradient(name)
    hessian=t.kernel_hessian(name)
    return [*[t.moment(name,a) for a in alphas],*gradient,
        *[x for i in range(3) for x in t.kernel_gradient(name,tuple(int(i==j) for j in range(3)))],
        *[hessian[i][j] for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]],
        t.laplacian(name),mp.sqrt(sum(x*x for x in gradient))]


@pytest.mark.parametrize('name',['w2','w4','w6','cubic'])
def test_all_thirty_fields_with_cut_support_and_boundary_queries(name):
    vertices=[[-1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1]]
    # Interior, generic partial support, vertex, edge, face, near-face, exterior.
    points=[[0,0,0],[.2,-.3,.4],[1,1,-1],[0,0,-1],[.5,-.25,-.25],
            [.5,-.25,-.25+1e-10],[.5,-.25,-.25-1e-10],[2,2,2]]
    faces=Tetrahedron(vertices).indices
    script=f'''
import {{AnalyticSolid}} from {json.dumps((STATIC/'mesh3d-analytic.js').as_uri())};
const records={json.dumps(points)}.map(q=>{{const t=new AnalyticSolid({json.dumps(vertices)},{json.dumps(faces)},q);return Array.from({{length:30}},(_,i)=>t.field({json.dumps(name)},i));}});
console.log(JSON.stringify(records));
'''
    data=javascript(script)
    with mp.workdps(65):
        for point,values in zip(points,data):
            expected=np.array(reference(Tetrahedron(vertices,point),name),float)
            assert all(v is not None for v in values)
            np.testing.assert_allclose(values,expected,atol=1e-9,rtol=1e-9)
            assert abs(sum(values[i] for i in (22,25,27))-values[28])<2e-9


def test_constant_block_overlap_and_polynomial_simplex_integrals():
    cases=[([0,0,0],.2),([.4,0,0],.1),([.2,.2,.2],.3),([.1,-.2,.3],.5),([.2,.2,.2],2),([-2,0,0],.1)]
    faces=Tetrahedron(UNIT).indices
    script=f'''
import {{AnalyticSolid}} from {json.dumps((STATIC/'mesh3d-analytic.js').as_uri())};
const overlap={json.dumps(cases)}.map(([q,R])=>new AnalyticSolid({json.dumps(UNIT)},{json.dumps(faces)},q).block(0,0,0,0,R));
const t=new AnalyticSolid({json.dumps(UNIT)},{json.dumps(faces)}),poly=[[0,0,0],[1,0,0],[2,0,0],[1,1,0]].map(a=>t.block(...a,0,2));
console.log(JSON.stringify({{overlap,poly}}));
'''
    data=javascript(script)
    with mp.workdps(65):
        expected=[Tetrahedron(UNIT,q).block_moment((0,0,0),0,mp.mpf(R)) for q,R in cases]
        np.testing.assert_allclose(data['overlap'],np.array(expected,float),atol=2e-15,rtol=2e-14)
    np.testing.assert_allclose(data['poly'],[1/6,1/24,1/60,1/120],atol=1e-15)
    assert abs(data['overlap'][0]-(4*np.pi*.2**3/3)/8)<1e-17


def test_browser_dispatch_box_orientation_whole_support_scale_and_precision_metadata():
    script=f'''
import {{prepare,pointTerm,computeCPU}} from {json.dumps((STATIC/'browser-compute.js').as_uri())};
const base={{dimension:3,geometry:'tetrahedron',radius:1,width:2,height:2,depth:2,h:.55,resolution:17,padding:1.15,kernel:'w4',field:'m00',models:['analytic'],quality:1}};
const c={{...base,geometry:'box'}},g=prepare(c),plane=pointTerm([1,0,0],c,'analytic',g),centre=Array.from({{length:30}},(_,field)=>pointTerm([0,0,0],{{...c,field}},'analytic',g));
const p=[.4,-.1,.3],scaled={{...base,width:4,height:4,depth:4,h:1.1}},a=prepare(base),b=prepare(scaled),original=Array.from({{length:30}},(_,field)=>pointTerm(p,{{...base,field}},'analytic',a)),dilated=Array.from({{length:30}},(_,field)=>pointTerm(p.map(x=>2*x),{{...scaled,field}},'analytic',b));
const grid=computeCPU(base),other=computeCPU({{...base,quality:2}});
console.log(JSON.stringify({{plane,centre,original,dilated,grid,other}}));
'''
    data=javascript(script)
    assert abs(data['plane']-.5)<1e-10
    assert data['centre'][0]==1 and data['centre'][13]==data['centre'][17]==data['centre'][21]==1
    assert data['centre'][10:13]==[0,0,0]
    np.testing.assert_allclose(data['original'],data['dilated'],atol=2e-12)
    row=data['grid']['results'][0]
    assert row['values']==data['other']['results'][0]['values']
    assert row['missing']==0 and row['min']>=-1e-7 and row['max']<=1+1e-7
    assert any('float64' in note for note in row['notes'])
    assert any('Exact antiderivative' in note for note in row['notes'])
