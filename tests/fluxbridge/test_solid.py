"""Fixed solid bundles against independently tabulated triangle/tet integrals."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import prepare_solid_geometry,solid_functions
from warpSPHBoundaries.fluxbridge.solid import solid_source

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'


def cases(d,kernel):
    names=['fluxbridge-v1.json']+(['fluxbridge-tetquad-v1.json','fluxbridge-nearplane-v1.json'] if d==3 else [])
    result=[]
    for name in names:
        for c in json.loads((FIXTURES/name).read_text())['cases']:
            if c.get('dimension',3)==d and c.get('measure','volume')=='volume' and c['kernel']==kernel and np.shape(c['vertices'])==(d+1,d):result.append(c)
    return result


def expected(c,d):
    table={(tuple(r['alpha']),tuple(r['derivative'])):float(v) for r,v in zip(c['requests'],c['values'])}
    zero=(0,)*d;keys=[(zero,())]+[(zero,(j,)) for j in range(d)]
    keys += [(tuple(int(j==a) for j in range(d)),(b,)) for a in range(d) for b in range(a,d)]
    return [table[key] for key in keys]


def launch(api,g,q,h,device):
    q=np.asarray(q);q=np.pad(q,((0,0),(0,3-api.dimension)))
    out=wp.full((api.channels,len(q)),value=float('nan'),dtype=api.real,device=device)
    args=[wp.array(q,dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(np.arange(len(q)),dtype=int,device=device),g,out]
    wp.launch(api.kernel,len(q),inputs=args,device=device,block_dim=256)
    return out.numpy().T


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('kernel',['w2','w4'])
@pytest.mark.parametrize('precision,conditioning',[('float64','raw'),('float32','protected')])
@pytest.mark.parametrize('bundle',['density','pressure','lean'])
def test_independent_solid_fixtures(device,dimension,kernel,precision,conditioning,bundle):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    cs=cases(dimension,kernel);assert len(cs)==(6 if dimension==2 else 26)
    # Test the public defaults: native fp64 and protected fp32.
    g=prepare_solid_geometry([c['vertices'] for c in cs],device,precision)
    api=solid_functions(dimension,kernel,precision,bundle)
    assert api.conditioning==conditioning
    actual=launch(api,g,[c['query'] for c in cs],[c['h'] for c in cs],device)
    reference=np.array([expected(c,dimension) for c in cs])[:,:api.channels]
    error=abs(actual-reference)/(1+abs(reference))
    assert np.isfinite(actual).all()
    assert np.max(error)<=(1e-10 if precision=='float64' else 1e-5)


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('precision,conditioning',[('float64','raw'),('float32','protected')])
def test_orientation_and_physical_scaling(dimension,precision,conditioning):
    v=np.vstack([np.zeros(dimension),2*np.eye(dimension)])
    q=np.full(dimension,.2);scale=2.
    vertices=np.array([v,v[::-1],scale*v]);queries=np.array([q,q,scale*q])
    g=prepare_solid_geometry(vertices,'cpu',precision,conditioning)
    api=solid_functions(dimension,'w4',precision,'lean',conditioning=conditioning)
    values=launch(api,g,queries,[1,1,scale],'cpu')
    tolerance=1e-10 if precision=='float64' else 1e-5
    assert np.max(abs(values[1]-values[0])/(1+abs(values[0])))<=tolerance
    wanted=values[0].copy();wanted[1:dimension+1]/=scale
    assert np.max(abs(values[2]-wanted)/(1+abs(wanted)))<=tolerance


def test_explicit_types_and_rejected_geometry():
    a=solid_functions(2,'w2','float32');b=solid_functions(3,'w4','float64')
    assert a.real==wp.float32 and b.real==wp.float64
    for bad in [np.zeros((1,3,2)),np.zeros((1,4,3)),np.full((1,3,2),np.nan)]:
        with pytest.raises(ValueError):prepare_solid_geometry(bad)
    with pytest.raises(ValueError):solid_functions(3,'w6')
    with pytest.raises(ValueError):solid_functions(2,'w2','float64',conditioning='protected')
    with pytest.raises(ValueError):solid_functions(3,'w2',registers=16)
    for d in (2,3):
        source=solid_source(d,'w4','float64','lean')
        assert 'STEPS' not in source and 'FACE_STRUCT' not in source
        assert 'field[' not in source and 'hxx' not in source
