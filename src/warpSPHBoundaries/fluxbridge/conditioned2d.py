"""Explicit conditioned polygon boundary quadrature, ordinary fp32/fp64.

Euler's radial flux uses cumulative profiles combined in exact arithmetic
before rounding. Angle integration retains the complete exterior flux. This
is numerical integration of straight polygon geometry, not an analytic port
or a hidden precision fallback. All requested moments/derivatives are retained.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb
from types import SimpleNamespace
import numpy as np
import warp as wp
from ..edge.kernels import KERNELS
from ..edge.np2d import cheb_coeffs
from .contract import Request

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


class ConditionedPlan:
    def __init__(self,dimension,requests,kernel,measure="volume",compact=False):
        if dimension!=2 or measure!="volume":raise ValueError("conditioned boundary quadrature requires 2D polygon volume")
        if kernel not in KERNELS or kernel not in ('w2','w4','w6','cubic'):raise ValueError("unsupported kernel")
        self.requests=tuple(requests);self.kernel=kernel
        self.profiles=[];self.metadata=[]
        def term(alpha,order,sign):
            key=(sum(alpha),order)
            if key not in self.profiles:self.profiles.append(key)
            return [*alpha,self.profiles.index(key),sign]
        for req in self.requests:
            if not isinstance(req,Request) or len(req.alpha)!=2:raise ValueError("request dimension mismatch")
            a=req.alpha;ds=req.derivative
            if not ds:row=term(a,0,1)+[0,0,-1,0]
            elif len(ds)==1:
                b=tuple(v+int(j==ds[0]) for j,v in enumerate(a))
                row=term(b,1,-1)+[0,0,-1,0]
            else:
                i,j=ds;b=tuple(v+int(k==i)+int(k==j) for k,v in enumerate(a))
                row=term(b,2,1)+(term(a,1,1) if i==j else [0,0,-1,0])
            self.metadata.append(row)
        pieces=KERNELS[kernel].pieces;self.bounds=[(float(lo),float(hi)) for lo,hi,_ in pieces]
        self.coefficients=[];self.constants=[];self.amplification=0.
        for degree,order in self.profiles:
            cumulative=F(0);polys=[]
            for lo,hi,coeff in pieces:
                poly={}
                for n,c in enumerate(coeff):
                    if not c:continue
                    exponent=n;factor=1
                    if order==1:exponent=n-2;factor=n
                    if order==2:exponent=n-4;factor=n*(n-2)
                    if not factor:continue
                    power=exponent+degree+2
                    assert power>0, "all advertised weighted jet primitives are regular"
                    poly[power]=poly.get(power,F(0))+c*factor/power
                poly[0]=cumulative-sum(c*lo**n for n,c in poly.items())
                cumulative=sum(c*hi**n for n,c in poly.items())
                shifted={j:sum(c*comb(n,j)*lo**(n-j) for n,c in poly.items() if n>=j)
                         for j in range(max(poly)+1)}
                cheb=cheb_coeffs(shifted,hi-lo)
                if cumulative:self.amplification=max(self.amplification,float(sum(abs(c) for c in cheb)/abs(cumulative)))
                polys.append([float(c)/np.pi for c in cheb])
            self.coefficients.append(polys);self.constants.append(float(cumulative)/np.pi)

    def arrays(self):
        # The common host packer also serves analytic plans. No monomial DAG is
        # uploaded here: the only scratch is output sums and their residuals.
        return dict(f=np.empty((0,9),np.int32),dep=np.empty((0,9),np.int32),fr=np.empty(0),
                    b=np.empty((0,5),np.int32),br=np.empty(0),v=np.empty((0,7),np.int32),vr=np.empty(0),
                    terms=np.empty((0,3)),offsets=np.zeros(len(self.requests)+1,np.int32))

    def upload(self,plan,api,device,nodes):
        width=max((len(c) for p in self.coefficients for c in p),default=1)
        coeff=np.zeros((len(self.profiles),len(self.bounds),width));lengths=np.zeros(coeff.shape[:2],np.int32)
        for i,ps in enumerate(self.coefficients):
            for j,c in enumerate(ps):coeff[i,j,:len(c)]=c;lengths[i,j]=len(c)
        kw=dict(device=device)
        plan.requests=wp.array(np.array(self.metadata,np.int32).reshape(-1,8),dtype=int,**kw)
        plan.coefficients=wp.array(coeff,dtype=api.real,**kw);plan.lengths=wp.array(lengths,dtype=int,**kw)
        plan.bounds=wp.array(self.bounds,dtype=api.real,**kw)
        plan.constants=wp.array(self.constants,dtype=api.real,**kw)
        points,weights=np.polynomial.legendre.leggauss(nodes)
        plan.nodes=wp.array(points,dtype=api.real,**kw);plan.weights=wp.array(weights,dtype=api.real,**kw)


@lru_cache(None)
def functions(precision="float32",nodes=16):
    if precision not in ('float32','float64'):raise ValueError("conditioned quadrature supports ordinary fp32/fp64 only")
    if not isinstance(nodes,int) or not 8<=nodes<=64:raise ValueError("quadrature nodes must be an integer in [8,64]")
    real=wp.float32 if precision=='float32' else wp.float64
    vec=wp.types.vector(length=3,dtype=real)
    @wp.struct
    class Plan:
        f:wp.array2d(dtype=int)
        dep:wp.array2d(dtype=int)
        fr:wp.array(dtype=real)
        b:wp.array2d(dtype=int)
        br:wp.array(dtype=real)
        v:wp.array2d(dtype=int)
        vr:wp.array(dtype=real)
        term_kind:wp.array(dtype=int)
        term_index:wp.array(dtype=int)
        term_value:wp.array(dtype=real)
        offsets:wp.array(dtype=int)
        dimension:int
        surface:int
        compact:int
        requests:wp.array2d(dtype=int)
        coefficients:wp.array3d(dtype=real)
        lengths:wp.array2d(dtype=int)
        bounds:wp.array2d(dtype=real)
        constants:wp.array(dtype=real)
        nodes:wp.array(dtype=real)
        weights:wp.array(dtype=real)

    scalar='float' if precision=='float32' else 'double';suffix='f' if precision=='float32' else ''
    epsilon='1.1920928955078125e-7' if precision=='float32' else '2.220446049250313e-16'
    code=ELEMENT.replace('SCALAR',scalar).replace('SQRT','sqrt'+suffix).replace('ATAN2','atan2'+suffix).replace('FMA','fma'+suffix).replace('EPSILON',epsilon)
    @wp.func_native(code)
    def element(query:vec,h:real,facets:wp.array2d(dtype=vec),first:int,last:int,plan:Plan,
                work:wp.array2d(dtype=real),row:int): ...
    @wp.func
    def output(plan:Plan,work:wp.array2d(dtype=real),row:int,column:int):
        return work[row,column]+work[row,plan.requests.shape[0]+column]
    @wp.func
    def power(x:real,n:int):
        result=real(1.)
        for k in range(wp.abs(n)):result=result*x
        if n<0:result=real(1.)/result
        return result
    @wp.kernel
    def batch(queries:wp.array(dtype=vec),supports:wp.array(dtype=real),facets:wp.array2d(dtype=vec),
              starts:wp.array(dtype=int),pq:wp.array(dtype=int),pe:wp.array(dtype=int),shifts:wp.array(dtype=vec),
              plan:Plan,scales:wp.array(dtype=int),work:wp.array2d(dtype=real),out:wp.array2d(dtype=real)):
        row=wp.tid();qi=pq[row];ei=pe[row]
        element(queries[row],supports[qi],facets,starts[ei],starts[ei+1],plan,work,row)
        for col in range(out.shape[1]):out[row,col]=output(plan,work,row,col)*power(supports[qi],scales[col])
    return SimpleNamespace(real=real,vec=vec,Plan=Plan,element=element,output=output,power=power,batch=batch,
        work_dtype=real,coefficient_dtype=real,field_dtype=real,output_cache=True,cache_columns=2,local_geometry=True)


ELEMENT=r'''
using T=SCALAR;
#define FB_LOAD(...) (*wp::address(__VA_ARGS__))
int columns=plan.requests.shape[0];
for(int j=0;j<2*columns;++j)wp::array_store(work,row,j,T(0));
// Whole-element rejection is valid for the compact kernel and every jet.
// It also prevents tiny exterior roundoff from being amplified when a remote
// element's polynomial coefficients are translated by a large displacement.
T nearest=T(4);int winding=0;
for(int face=first;face<last;++face) {
    auto av=FB_LOAD(facets,face,0),bv=FB_LOAD(facets,face,1);
    T ax=(av[0]-query[0])/h,ay=(av[1]-query[1])/h;
    T bx=(bv[0]-query[0])/h,by=(bv[1]-query[1])/h;
    T det=FMA(ax,by,-ay*bx);
    if(ay<=T(0) && by>T(0) && det>T(0))++winding;
    if(ay>T(0) && by<=T(0) && det<T(0))--winding;
    T dx=bx-ax,dy=by-ay,scale=(dx<0?-dx:dx)>(dy<0?-dy:dy)?(dx<0?-dx:dx):(dy<0?-dy:dy);
    if(scale>T(0)) {
        T sx=dx/scale,sy=dy/scale,length=scale*SQRT(sx*sx+sy*sy),tx=dx/length,ty=dy/length;
        T lo=FMA(tx,ax,ty*ay),hi=FMA(tx,bx,ty*by),z=FMA(ty,ax,-tx*ay);
        T distance=z*z;
        if(lo>T(0))distance=FMA(ax,ax,ay*ay);
        if(hi<T(0))distance=FMA(bx,bx,by*by);
        T magnitude=(ax<0?-ax:ax)+(ay<0?-ay:ay)+(bx<0?-bx:bx)+(by<0?-by:by)+T(1);
        T lower=SQRT(distance)-T(64)*T(EPSILON)*magnitude;
        if(lower<T(0))lower=0;
        if(lower*lower<nearest)nearest=lower*lower;
    }
}
// Conservative margin prevents a rounded tangent distance from pruning a
// shallow supported intersection. No individual regular-flux face is dropped.
if(winding==0 && nearest>T(1))return;
auto cdf=[&](int index,T radius) {
    if(radius>=T(1))return FB_LOAD(plan.constants,index);
    int piece=0;
    for(int p=1;p<plan.bounds.shape[0];++p)if(radius>=FB_LOAD(plan.bounds,p,0))piece=p;
    T lo=FB_LOAD(plan.bounds,piece,0),hi=FB_LOAD(plan.bounds,piece,1);
    T x=T(2)*(radius-lo)/(hi-lo)-T(1),b1=0,b2=0;
    for(int k=FB_LOAD(plan.lengths,index,piece)-1;k>=1;--k) {
        T next=FMA(T(2)*x,b1,FB_LOAD(plan.coefficients,index,piece,k)-b2);b2=b1;b1=next;
    }
    return FMA(x,b1,FB_LOAD(plan.coefficients,index,piece,0)-b2);
};
// The angle is bounded by pi/2. This polynomial avoids CUDA trig routines'
// irrelevant large-argument fp64 slow path; all constants are explicitly T.
auto trig=[&](T angle,T& sn,T& cs) {
    T x2=angle*angle,st=angle,ct=1;sn=angle;cs=1;
    for(int j=1;j<=12;++j) {
        st=-st*x2/T((2*j)*(2*j+1));sn=sn+st;
        ct=-ct*x2/T((2*j-1)*(2*j));cs=cs+ct;
    }
};
for(int face=first;face<last;++face) {
    auto av=FB_LOAD(facets,face,0),bv=FB_LOAD(facets,face,1);
    T ax=(av[0]-query[0])/h,ay=(av[1]-query[1])/h;
    T bx=(bv[0]-query[0])/h,by=(bv[1]-query[1])/h;
    T dx=bx-ax,dy=by-ay;
    T scale=(dx<0?-dx:dx)>(dy<0?-dy:dy)?(dx<0?-dx:dx):(dy<0?-dy:dy);
    if(scale==T(0))continue;
    T sx=dx/scale,sy=dy/scale,length=scale*SQRT(sx*sx+sy*sy);
    T tx=dx/length,ty=dy/length,nx=ty,ny=-tx;
    T z=FMA(nx,ax,ny*ay),c=z<T(0)?-z:z;
    if(c==T(0))continue;
    T sign=z<T(0)?T(-1):T(1),s0=FMA(tx,ax,ty*ay),s1=FMA(tx,bx,ty*by);
    for(int side=-1;side<=1;side+=2) {
        T a=side==1?(s0>T(0)?s0:T(0)):(-s1>T(0)?-s1:T(0));
        T end=side==1?(s1>T(0)?s1:T(0)):(-s0>T(0)?-s0:T(0));
        while(a<end) {
            T b=T(2)*a;if(b<c)b=c;if(b>end)b=end;
            for(int p=0;p<plan.bounds.shape[0];++p) {
                T knot=FB_LOAD(plan.bounds,p,1);
                if(knot>c) {
                    T cut=SQRT((knot-c)*(knot+c));
                    if(a<cut && cut<b)b=cut;
                }
            }
            T left=ATAN2(a,c),right=ATAN2(b,c),half=(right-left)/T(2),mid=(right+left)/T(2);
            bool outside=c>=T(1) || a*a+c*c>=T(1);
            if(half>T(0))for(int node=0;node<plan.nodes.shape[0];++node) {
                T angle=mid+half*FB_LOAD(plan.nodes,node),sn,cs;trig(angle,sn,cs);sn=sn*T(side);
                T ux=FMA(tx,sn,sign*nx*cs),uy=FMA(ty,sn,sign*ny*cs);
                T radius=outside?T(1):(cs>T(0)?c/cs:T(1));
                T values[36],px[12],py[12];px[0]=1;py[0]=1;
                for(int j=1;j<12;++j){px[j]=px[j-1]*ux;py[j]=py[j-1]*uy;}
                for(int j=0;j<plan.constants.shape[0];++j)values[j]=cdf(j,radius);
                T weight=sign*half*FB_LOAD(plan.weights,node);
                for(int col=0;col<columns;++col) {
                    T v=0;
                    for(int term=0;term<2;++term) {
                        int off=4*term,index=FB_LOAD(plan.requests,col,off+2);
                        if(index>=0)v=v+T(FB_LOAD(plan.requests,col,off+3))*px[FB_LOAD(plan.requests,col,off)]*py[FB_LOAD(plan.requests,col,off+1)]*values[index];
                    }
                    T term=weight*v,old=FB_LOAD(work,row,col),next=old+term;
                    T residual=(old<0?-old:old)>=(term<0?-term:term)?(old-next)+term:(term-next)+old;
                    wp::array_store(work,row,col,next);
                    wp::array_store(work,row,columns+col,FB_LOAD(work,row,columns+col)+residual);
                }
            }
            a=b;
        }
    }
}
#undef FB_LOAD
'''
