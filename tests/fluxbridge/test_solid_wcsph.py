"""WCSPH physical auxiliaries, independent face quadrature and frozen moments."""
import numpy as np
import pytest
import warp as wp
from scipy.integrate import quad
from warpSPHBoundaries.fluxbridge import prepare_solid_geometry, solid_functions
from warpSPHBoundaries.fluxbridge._solid_code import polynomial, normalizer
from test_solid import cases, expected, launch


def face_reference(vertices,q,h,kernel):
    """Independent divergence theorem with numerical face/edge integration."""
    d=len(q);phi=polynomial(kernel);C=normalizer(d,phi)
    def radial(r):
        if r>=1:return np.zeros(3)
        u=1-r
        ell=sum(float(c)*(r**(n+2)-1)/(n+2) for n,c in phi.items())
        lap=(-20*u**3 if kernel=='w2' else -56/3*u**5*(1+5*r))
        return np.array([lap,ell,r-1])
    result=np.zeros(1+2*d)
    if d==2:
        e=vertices[1]-vertices[0];f=vertices[2]-vertices[0]
        if np.linalg.det(np.array([e,f]))<0:vertices=vertices[::-1]
        faces=[(vertices[i],vertices[(i+1)%3],None) for i in range(3)]
    else:
        faces=[]
        for opp in range(4):
            a,b,c=vertices[[i for i in range(4) if i!=opp]]
            if np.dot(np.cross(b-a,c-a),vertices[opp]-a)>0:b,c=c,b
            faces.append((a,b,c))
    for a,b,c in faces:
        if d==2:
            t=(b-a)/h;length=np.linalg.norm(t);t/=length;n=np.array([t[1],-t[0]])
            rel=(a-q)/h;s0=rel@t;w=rel@n
            cut=np.sqrt(max(0,1-w*w));lo=max(s0,-cut);hi=min(s0+length,cut)
            if hi<=lo:continue
            vals=np.array([quad(lambda s:radial(np.hypot(s,w))[j],lo,hi,epsabs=2e-12,epsrel=2e-12,points=[0] if lo<0<hi else None)[0] for j in range(3)])
        else:
            a=(a-q)/h;b=(b-q)/h;c=(c-q)/h
            e=b-a;f=c-a;n=np.cross(e,f);jac=np.linalg.norm(n);n/=jac;w=n@a
            ff=f@f
            def integrate(u,j):
                base=a+u*e;mid=-(base@f)/ff
                half2=(1-base@base)/ff+mid*mid
                if half2<=0:return 0.
                half=np.sqrt(half2);lo=max(0,mid-half);hi=min(1-u,mid+half)
                if hi<=lo:return 0.
                return quad(lambda v:radial(np.linalg.norm(base+v*f))[j],lo,hi,epsabs=1e-12,epsrel=1e-12,points=[mid] if lo<mid<hi else None)[0]*jac
            vals=np.array([quad(lambda u:integrate(u,j),0,1,epsabs=2e-11,epsrel=2e-11,limit=300)[0] for j in range(3)])
        result[0]+=C/h**2*w*vals[0]
        result[1:1+d]+=C*h*n*vals[1]
        result[1+d:]-=h**d*n*vals[2]
    return result


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('kernel',['w2','w4'])
@pytest.mark.parametrize('precision',['float64','float32'])
def test_wcsph_frozen_terms(device,d,kernel,precision):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    cs=cases(d,kernel);api=solid_functions(d,kernel,precision,'wcsph_extended')
    values=launch(api,prepare_solid_geometry([c['vertices'] for c in cs],device,precision),[c['query'] for c in cs],[c['h'] for c in cs],device)
    for c,actual in zip(cs,values):
        table={(tuple(r['alpha']),tuple(r['derivative'])):float(v) for r,v in zip(c['requests'],c['values'])}
        zero=(0,)*d
        lap=sum(table[(zero,(j,j))] for j in range(d))
        m1=[table[(tuple(int(i==j) for i in range(d)),())] for j in range(d)]
        reference=np.array(expected(c,d)+[lap]+m1)
        assert np.max(abs(actual[:len(reference)]-reference)/(1+abs(reference)))<(1e-10 if precision=='float64' else 1e-5)


@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('kernel',['w2','w4'])
@pytest.mark.parametrize('precision',['float64','float32'])
def test_wcsph_independent_surface_integrals(d,kernel,precision):
    v=np.vstack([np.zeros(d),1.4*np.eye(d)])
    q=np.array([np.full(d,.15),np.full(d,-.2),np.zeros(d),np.full(d,2.)])
    vs=np.repeat(v[None],len(q),axis=0);h=np.array([.8,1.,1.2,.5])
    api=solid_functions(d,kernel,precision,'wcsph_extended')
    actual=launch(api,prepare_solid_geometry(vs,'cpu',precision),q,h,'cpu')[:,-(1+2*d):]
    reference=np.array([face_reference(v,x,s,kernel) for x,s in zip(q,h)])
    assert np.max(abs(actual-reference)/(1+abs(reference)))<(2e-9 if precision=='float64' else 1e-5)


@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('precision',['float64','float32'])
def test_wcsph_physical_scaling_and_bundle_prefix(d,precision):
    v=np.vstack([np.zeros(d),2*np.eye(d)])
    vertices=np.array([v,2*v]);q=np.array([np.full(d,.25),np.full(d,.5)])
    g=prepare_solid_geometry(vertices,'cpu',precision)
    api=solid_functions(d,'w4',precision,'wcsph_extended')
    values=launch(api,g,q,[1.,2.],'cpu')
    for bundle in ('lean','wcsph'):
        short=solid_functions(d,'w4',precision,bundle)
        a=launch(short,g,q,[1.,2.],'cpu')
        assert np.max(abs(a-values[:,:short.channels]))<(1e-10 if precision=='float64' else 1e-5)
    factors=np.ones(api.channels)
    for j,label in enumerate(api.labels):
        if label.startswith('G'):factors[j]=.5
        elif label=='lap':factors[j]=.25
        elif label.startswith('m1'):factors[j]=2
        elif label.startswith('cover'):factors[j]=2**d
    assert np.max(abs(values[1]-values[0]*factors))<(1e-10 if precision=='float64' else 1e-5)
