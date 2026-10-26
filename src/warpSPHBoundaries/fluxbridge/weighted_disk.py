"""Explicit interior-volume quadrature for weighted solid disks/cross sections.

Host fp64 geometry prepares radial nodes, including a boundary-angle change of
variable at circular tangencies. Profile evaluation and moment contractions are
ordinary selected-dtype Warp functions. No porous material law is attached.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from .contract import Request,baseline,KERNELS
from .radial import factored_profiles
from .planar_shell import gauss_rule

wp.set_module_options({'enable_backward':False,'fuse_fp':False})
MAX_BATCH_BYTES=2**30
MAX_RADIAL_PANELS=128


@lru_cache(None)
def functions(precision='float64'):
    if precision not in ('float32','float64'):raise ValueError('weighted disk supports ordinary fp32/fp64 only')
    real=wp.float32 if precision=='float32' else wp.float64
    vec=wp.types.vector(length=3,dtype=real);pair=wp.types.vector(length=2,dtype=real)
    @wp.struct
    class Plan:
        requests:wp.array2d(dtype=int)
        terms:wp.array2d(dtype=int)
        coefficients:wp.array(dtype=vec)
        angular:wp.array(dtype=pair)
    @wp.struct
    class Rules:
        radial:wp.array(dtype=vec)
        offsets:wp.array(dtype=int)
        query:wp.array(dtype=int)
        normals:wp.array(dtype=vec)
    common=COMMON.replace('SCALAR','float' if precision=='float32' else 'double')
    trig='T s=wp::sin(theta),c=wp::cos(theta);'
    if precision=='float32':
        common+=BOUNDED_TRIG
        trig='T s,c;bounded_trig(theta,s,c);'
    @wp.func_native(common+SAMPLE.replace('FB_TRIG',trig))
    def sample(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec),radial:int,angular:int):...
    @wp.func_native(common+INTEGRATE)
    def integrate(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec),row:int,column:int)->real:...
    @wp.func
    def element(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec),work:wp.array2d(dtype=real),row:int):
        for ir in range(rules.offsets[row],rules.offsets[row+1]):
            for it in range(plan.angular.shape[0]):sample(plan,rules,scratch,ir,it)
        for col in range(plan.requests.shape[0]):work[row,col]=integrate(plan,rules,scratch,row,col)
    @wp.func
    def output(plan:Plan,work:wp.array2d(dtype=real),row:int,column:int):return work[row,column]
    @wp.kernel
    def prepare_samples(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec)):
        ir,it=wp.tid();sample(plan,rules,scratch,ir,it)
    @wp.kernel
    def batch(plan:Plan,rules:Rules,scratch:wp.array2d(dtype=vec),supports:wp.array(dtype=real),
              scales:wp.array(dtype=int),work:wp.array2d(dtype=real),values:wp.array2d(dtype=real)):
        row,col=wp.tid();value=integrate(plan,rules,scratch,row,col);work[row,col]=value
        scale=real(1.)
        for k in range(wp.abs(scales[col])):scale=scale*supports[row]
        if scales[col]<0:scale=real(1.)/scale
        values[row,col]=value*scale
    return SimpleNamespace(real=real,vec=vec,pair=pair,Plan=Plan,Rules=Rules,
        sample=sample,element=element,output=output,prepare_samples=prepare_samples,batch=batch)


def radial_rule(distance,radius,kernel,nodes,grade_floor):
    """Normalized fp64 r/dr/angle nodes; the actual distance is never snapped."""
    if distance>=1:return np.empty((0,3))
    centre=radius+distance;end=min(1.,2*radius+distance)
    cuts={0.,1.,abs(distance),end}
    if kernel=='cubic':cuts.add(.5)
    if distance!=0:
        cut=max(abs(distance),grade_floor)
        while cut<end:
            cuts.add(cut);cut*=2
    cuts=sorted(c for c in cuts if 0<=c<=1)
    if len(cuts)-1>MAX_RADIAL_PANELS:raise ValueError('weighted disk radial panel budget exceeded')
    rule=gauss_rule(nodes);x,w=rule[:,0],rule[:,1];rows=[]
    for left,right in zip(cuts,cuts[1:]):
        if right<=left or left>=end:continue
        half=(right-left)/2
        if right<=-distance:
            r=left+half*(1+x);weight=half*w;angle=np.full_like(r,np.pi)
        elif centre and left>=abs(distance):
            # r²=d²+4*centre*radius*sin(phi/2)² removes cap endpoint roots.
            def phi(r):
                z=((r-abs(distance))/(2*radius))*((r+abs(distance))/(2*centre))
                return 2*np.arcsin(np.sqrt(np.clip(z,0,1)))
            a,b=phi(left),phi(right)
            p=a+(b-a)*(1+x)/2;sh=np.sin(p/2);sp=np.sin(p)
            r=np.hypot(distance,2*np.sqrt(centre)*np.sqrt(radius)*sh)
            weight=(b-a)/2*w*(centre*radius/r)*sp
            angle=np.arctan2(radius*sp,distance+2*radius*sh*sh)
        else:continue
        rows.append(np.stack([r,weight,angle],axis=-1))
    return np.concatenate(rows) if rows else np.empty((0,3))


class WeightedDiskEvaluator:
    """Numerical disk volume moments through degree nine / kernel derivative two.

    `distance=|query-centre|-radius`, normal outward from centre to query. Native
    fp32/fp64 is explicit, with host fp64 radial geometry preparation. Two-order
    normalized comparisons promote or reject; this is not a global error bound.
    """
    method='disk-volume-quadrature'
    dimension=2
    def __init__(self,requests=None,*,kernel='w2',precision='float64',nodes=24,device='cpu'):
        if kernel not in KERNELS or not isinstance(nodes,int) or not 8<=nodes<=64:
            raise ValueError('supported kernel and integer nodes in 8..64 required')
        self.kernel,self.precision,self.nodes=kernel,precision,nodes
        self.device=wp.get_device(device);self.api=functions(precision)
        self.requests=baseline(2) if requests is None else tuple(requests)
        if any(not isinstance(r,Request) or len(r.alpha)!=2 for r in self.requests):raise ValueError('weighted disk requests must be 2D')
        self.angular_nodes=max(nodes,16,4*(1+max((sum(r.alpha)+len(r.derivative) for r in self.requests),default=0)))
        self.plan=self.api.Plan();kw=dict(device=self.device)
        metadata=[[*r.alpha,len(r.derivative),*(list(r.derivative)+[0,0])[:2]] for r in self.requests]
        self.plan.requests=wp.array(np.asarray(metadata,np.int32).reshape(-1,5),dtype=int,**kw)
        terms=factored_profiles(kernel)
        self.plan.terms=wp.array([[p,r,s] for p,r,s,lo,hi,c in terms],dtype=int,**kw)
        self.plan.coefficients=wp.array([[float(lo),float(hi),float(c)/np.pi] for p,r,s,lo,hi,c in terms],dtype=self.api.vec,**kw)
        self.plan.angular=wp.array(gauss_rule(self.angular_nodes),dtype=self.api.pair,**kw)
        self.scales=wp.array([r.scale for r in self.requests],dtype=int,**kw)
        self.host_plan=SimpleNamespace(requests=self.requests,kernel=kernel)
        self.accuracy_status='explicit interior volume rule; finite convergence/independent fixtures required'
        self._orders={}

    def _at_order(self,order):
        if order==self.nodes:return self
        if order not in self._orders:self._orders[order]=WeightedDiskEvaluator(self.requests,kernel=self.kernel,
            precision=self.precision,nodes=order,device=self.device)
        return self._orders[order]

    def prepare(self,distances,normals,supports,*,radii):
        d=np.atleast_1d(np.asarray(distances,dtype=float))
        if d.ndim!=1:raise ValueError('distances must be scalar or vector')
        count=len(d)
        try:
            n=np.broadcast_to(np.asarray(normals,dtype=float),(count,2)).copy()
            h=np.broadcast_to(np.asarray(supports,dtype=float),(count,)).copy()
            r=np.broadcast_to(np.asarray(radii,dtype=float),(count,)).copy()
        except ValueError as error:raise ValueError('incompatible weighted disk batch shapes') from error
        if not all(np.isfinite(a).all() for a in (d,n,h,r)) or np.any(h<=0) or np.any(r<=0) or np.any(d < -r):
            raise ValueError('finite disk inputs, positive support/radius and distance >= -radius required')
        if not np.allclose(np.linalg.norm(n,axis=1),1,rtol=0,atol=1e-12):raise ValueError('disk normals must be unit vectors')
        dtype=np.float32 if self.precision=='float32' else np.float64
        with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
            dd=np.asarray(d.astype(np.longdouble)/h.astype(np.longdouble),float)
            rr=np.asarray(r.astype(np.longdouble)/h.astype(np.longdouble),float);hh=h.astype(dtype)
            factors=np.array([hh**req.scale for req in self.requests]);intermediate=np.array([hh**abs(req.scale) for req in self.requests])
            valid=(all(np.isfinite(a).all() for a in (dd,rr,hh,factors,intermediate)) and
                np.all(hh>0) and np.all(factors!=0) and np.all(intermediate!=0))
        if not valid:raise ValueError('weighted disk support scaling exceeds finite precision storage')
        minimum=.01 if self.precision=='float32' else .0001
        if np.any(rr<minimum) or np.any(rr>1e4):raise ValueError('radius/support outside weighted disk conditioning policy')
        budget=1e-5 if self.precision=='float32' else 1e-14
        order=self.nodes;promotions=0
        while True:
            comparison=min(64,order+8)
            owner=self._at_order(order);other=self._at_order(comparison)
            b=WeightedDiskBatch(owner,dd,rr,n,h)
            if not count or not self.requests:
                b.quadrature=dict(requested_nodes=self.nodes,nodes=order,promotions=0,samples=0)
                return b
            if order==comparison:raise ValueError('weighted disk needs a higher comparison order below 64')
            reference=WeightedDiskBatch(other,dd,rr,n,h)
            b.launch();reference.launch();x=b.work.numpy();y=reference.work.numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('weighted disk produced nonfinite normalized values')
            if not np.isfinite(b.values.numpy()).all():raise ValueError('weighted disk physical output exceeds finite precision storage')
            error=float(np.max(abs(x-y)/(1+abs(y))))
            comparison_samples=reference.sample_count;comparison_bytes=reference.bytes
            del reference,x,y
            if error<=budget:
                b.quadrature=dict(requested_nodes=self.nodes,nodes=order,angular_nodes=owner.angular_nodes,
                    comparison_nodes=comparison,worst_mixed_difference=error,budget=budget,promotions=promotions,
                    samples=b.sample_count,preparation='host',geometry_precision='float64',
                    comparison_sample_count=comparison_samples,comparison_owned_batch_bytes=comparison_bytes,
                    control='two-order normalized moment comparison; not a certified global bound')
                return b
            del b
            order=comparison;promotions+=1

    def evaluate(self,distances,normals,supports,*,radii):
        # Nonempty prepare has already evaluated both rules and checked storage.
        return self.prepare(distances,normals,supports,radii=radii).numpy()


class WeightedDiskBatch:
    def __init__(self,evaluator,distances,radii,normals,supports):
        self.evaluator=evaluator;self.count=len(supports);api=evaluator.api;kw=dict(device=evaluator.device)
        floor=1e-4 if evaluator.precision=='float32' else 1e-8
        if evaluator.requests:rows=[radial_rule(d,r,evaluator.kernel,evaluator.nodes,floor) for d,r in zip(distances,radii)]
        else:rows=[np.empty((0,3)) for _ in distances]
        offsets=np.r_[0,np.cumsum([len(row) for row in rows],dtype=np.int64)]
        size=4 if evaluator.precision=='float32' else 8
        radial_count=int(offsets[-1]);self.sample_count=radial_count*evaluator.angular_nodes
        estimate=size*(6*self.sample_count+4*radial_count+4*self.count+2*self.count*len(evaluator.requests))+4*(radial_count+self.count+1)
        if estimate>MAX_BATCH_BYTES or self.sample_count>2**31-1:raise ValueError('weighted disk rule storage exceeds 1 GiB batch budget; split queries')
        radial=np.concatenate(rows) if rows else np.empty((0,3))
        self.rules=api.Rules();self.rules.radial=wp.array(radial,dtype=api.vec,**kw)
        self.rules.offsets=wp.array(offsets,dtype=int,**kw)
        self.rules.query=wp.array(np.repeat(np.arange(self.count),np.diff(offsets)),dtype=int,**kw)
        self.rules.normals=wp.array(np.pad(normals,((0,0),(0,1))),dtype=api.vec,**kw)
        self.supports=wp.array(supports,dtype=api.real,**kw)
        self.scratch=wp.empty((self.sample_count,2),dtype=api.vec,**kw)
        self.work=wp.empty((self.count,len(evaluator.requests)),dtype=api.real,**kw)
        self.values=wp.empty((self.count,len(evaluator.requests)),dtype=api.real,**kw)

    @property
    def bytes(self):
        return sum(a.capacity for a in (self.rules.radial,self.rules.offsets,self.rules.query,self.rules.normals,
            self.supports,self.scratch,self.work,self.values))

    def launch(self):
        e=self.evaluator
        if self.sample_count:wp.launch(e.api.prepare_samples,(len(self.rules.radial),e.angular_nodes),inputs=[e.plan,self.rules,self.scratch],device=e.device)
        if self.count and e.requests:wp.launch(e.api.batch,self.values.shape,
            inputs=[e.plan,self.rules,self.scratch,self.supports,e.scales,self.work,self.values],device=e.device)
        return self.values

    def numpy(self):return self.values.numpy()


COMMON=r'''
using T=SCALAR;
#define FB_LOAD(...) (*wp::address(__VA_ARGS__))
auto power=[](T x,int n){T v=1;for(int k=0;k<(n<0?-n:n);++k)v*=x;return n<0?T(1)/v:v;};
'''

SAMPLE=r'''
auto rw=FB_LOAD(rules.radial,radial);
auto aw=FB_LOAD(plan.angular,angular);
int row=FB_LOAD(rules.query,radial),index=radial*plan.angular.shape[0]+angular;
auto normal=FB_LOAD(rules.normals,row);
T r=rw[0],theta=rw[2]*aw[0];FB_TRIG
auto tangent=decltype(normal)(-normal[1],normal[0],T(0));
auto point=normal*(-r*c)+tangent*(r*s);
T values[3]={T(0),T(0),T(0)};
for(int k=0;k<plan.terms.shape[0];++k){
    auto coefficient=FB_LOAD(plan.coefficients,k);
    if(r<coefficient[0]||r>=coefficient[1])continue;
    int profile=FB_LOAD(plan.terms,k,0),radial_power=FB_LOAD(plan.terms,k,1),shifted=FB_LOAD(plan.terms,k,2);
    int out=profile,exponent=radial_power+1;
    if(profile==2)exponent=radial_power;
    if(profile==3){out=1;exponent=radial_power;}
    if(profile==4){out=2;exponent=radial_power-2;}
    values[out]+=coefficient[2]*power(r,exponent)*power(coefficient[1]-r,shifted);
}
T weight=rw[1]*rw[2]*aw[1];
wp::array_store(scratch,index,0,point);
wp::array_store(scratch,index,1,decltype(normal)(values[0]*weight,values[1]*weight,values[2]*weight));
#undef FB_LOAD
'''

# The prepared angle is in [-pi,pi]. Range reduction to [-pi/2,pi/2]
# and degree-13/14 Taylor polynomials avoid libdevice's unused large-angle
# double-precision fallback. Remainders there are below 7e-10 / 7e-11;
# floating-point range reduction/evaluation are validated independently.
BOUNDED_TRIG=r'''
auto bounded_trig=[](T x,T& sine,T& cosine){
    T sign=T(1);
    if(x>T(1.57079632679489661923)){x=T(3.14159265358979323846)-x;sign=T(-1);}
    else if(x<T(-1.57079632679489661923)){x=T(-3.14159265358979323846)-x;sign=T(-1);}
    T z=x*x;
    T s=T(1.60590438368216145994e-10);
    s=fmaf(s,z,T(-2.50521083854417187751e-8));
    s=fmaf(s,z,T(2.75573192239858906526e-6));
    s=fmaf(s,z,T(-1.98412698412698412698e-4));
    s=fmaf(s,z,T(8.33333333333333333333e-3));
    s=fmaf(s,z,T(-1.66666666666666666667e-1));
    sine=x*fmaf(s,z,T(1));
    T c=T(-1.14707455977297247139e-11);
    c=fmaf(c,z,T(2.08767569878680989792e-9));
    c=fmaf(c,z,T(-2.75573192239858906526e-7));
    c=fmaf(c,z,T(2.48015873015873015873e-5));
    c=fmaf(c,z,T(-1.38888888888888888889e-3));
    c=fmaf(c,z,T(4.16666666666666666667e-2));
    c=fmaf(c,z,T(-5.00000000000000000000e-1));
    cosine=sign*fmaf(c,z,T(1));
};
'''

INTEGRATE=r'''
int start=FB_LOAD(rules.offsets,row)*plan.angular.shape[0],stop=FB_LOAD(rules.offsets,row+1)*plan.angular.shape[0];
int ax=FB_LOAD(plan.requests,column,0),ay=FB_LOAD(plan.requests,column,1);
int order=FB_LOAD(plan.requests,column,2),i=FB_LOAD(plan.requests,column,3),j=FB_LOAD(plan.requests,column,4);
T sum=0,residual=0;
for(int k=start;k<stop;++k){
    auto point=FB_LOAD(scratch,k,0),weights=FB_LOAD(scratch,k,1);
    T value=weights[0];
    if(order==1)value=-point[i]*weights[1];
    if(order==2)value=point[i]*point[j]*weights[2]+(i==j?weights[1]:T(0));
    T term=power(point[0],ax)*power(point[1],ay)*value,next=sum+term;
    residual+=(wp::abs(sum)>=wp::abs(term))?((sum-next)+term):((term-next)+sum);sum=next;
}
return sum+residual;
#undef FB_LOAD
'''
