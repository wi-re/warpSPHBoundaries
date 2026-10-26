"""The regular face primitive stays finite at the origin in native fp32/fp64."""
import mpmath as mp
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,baseline
from warpSPHBoundaries.edge.kernels import KERNELS
from warpSPHBoundaries.curvbound.kernels import KERNELS as SHAPES

DEVICES=['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision,tol',[('float64',1e-13),('float32',1e-5)])
@pytest.mark.parametrize('kernel',['w2','w4','w6','cubic'])
def test_regular_profile_limit_and_radial_values(device,precision,tol,kernel):
    g=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    req=baseline(3)+(Request((9,0,0)),Request((4,2,3),(0,2)))
    ev=Evaluator(g,req,kernel=kernel,precision=precision,device=device,method='tet_quadrature')
    p=ev.prepare([],1).integration_plan
    p.radii=wp.array([0,1e-30,1e-12,1e-4,.1,.49,.5,.9,1,2],dtype=ev.api.real,device=device)
    p.profile_kind=wp.ones(len(p.radii),dtype=int,device=device)
    p.profiles=wp.empty((len(p.radii),len(ev.host_plan.profiles)),dtype=ev.api.real,device=device)
    wp.launch(ev.api.prepare_profiles,p.profiles.shape,inputs=[p],device=device)
    expected=[]
    with mp.workdps(80):
        ratio=mp.mpf(str(SHAPES[kernel].c3_pi))/mp.mpf(str(SHAPES[kernel].c2_pi))/mp.pi
        for stored in p.radii.numpy():
            r=mp.mpf(float(stored));values=[]
            for degree,order in ev.host_plan.profiles:
                value=mp.mpf(0)
                for lo,hi,coeff in KERNELS[kernel].pieces:
                    lo=mp.mpf(str(lo));hi=mp.mpf(str(hi))
                    if not r:
                        if lo:break
                        for n,c in enumerate(coeff):
                            factor=1 if order==0 else n if order==1 else n*(n-2)
                            if c and factor and n-2*order+degree==0:value+=mp.mpf(str(c))*factor/3
                        break
                    if r<=lo:break
                    top=min(r,hi,mp.mpf(1))
                    for n,c in enumerate(coeff):
                        factor=1 if order==0 else n if order==1 else n*(n-2)
                        if not c or not factor:continue
                        power=n-2*order+degree+3
                        value+=mp.mpf(str(c))*factor*(top**power-lo**power)/power
                values.append(float(ratio*(value if not r else value/r**3)))
            expected.append(values)
    expected=np.array(expected);actual=p.profiles.numpy()
    assert np.isfinite(actual).all()
    assert np.max(abs(actual-expected)/(1+abs(expected)))<tol
