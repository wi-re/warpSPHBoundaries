"""Explicit spherical-shell quadrature for weighted 3D planes/bands.

Split radial intervals at material planes and kernel knots. Azimuthal trapezoid
and polar Gauss rules integrate the advertised angular polynomial exactly in
exact arithmetic. Radial Gauss is polynomial exact for Wendland profiles;
cubic's outer Hessian also contains bounded Laurent terms. Floating point
evaluation and the latter quadrature still require independent validation.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from ..curvbound.kernels import KERNELS as SHAPES
from .contract import Request
from .radial import factored_profiles

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@lru_cache(None)
def gauss_rule(count):
    """Polish NumPy's rule in extended host precision before fp64 rounding."""
    x, _ = np.polynomial.legendre.leggauss(count)
    x = x.astype(np.longdouble)
    for iteration in range(3):
        p0, p1 = np.ones_like(x), x.copy()
        for k in range(2, count + 1):
            p0, p1 = p1, ((2*k-1)*x*p1-(k-1)*p0)/k
        derivative = count*(x*p1-p0)/(x*x-1) if count > 1 else np.ones_like(x)
        x -= p1/derivative
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(2, count + 1):
        p0, p1 = p1, ((2*k-1)*x*p1-(k-1)*p0)/k
    derivative = count*(x*p1-p0)/(x*x-1) if count > 1 else np.ones_like(x)
    weights = 2/((1-x*x)*derivative*derivative)
    return np.asarray(np.stack([x, weights], axis=-1), dtype=float)


@lru_cache(None)
def functions(precision="float64", nodes=16, *, geometry="plane"):
    if geometry not in ("plane", "sphere"):
        raise ValueError("shell geometry must be plane or sphere")
    if precision not in ("float32", "float64"):
        raise ValueError("shell quadrature supports ordinary fp32/fp64 only")
    if not isinstance(nodes, int) or not 8 <= nodes <= 64:
        raise ValueError("quadrature nodes must be an integer in [8,64]")
    real = wp.float64 if precision == "float64" else wp.float32
    vec = wp.types.vector(length=3, dtype=real)
    pair = wp.types.vector(length=2, dtype=real)

    @wp.struct
    class Plan:
        requests: wp.array2d(dtype=int)
        terms: wp.array2d(dtype=int)
        coefficients: wp.array(dtype=vec)
        radial: wp.array(dtype=pair)
        polar: wp.array(dtype=pair)
        azimuth: wp.array(dtype=pair)
        cubic: int
        capacity: int

    def source(body):
        code = (COMMON + body).replace("SCALAR", "double" if precision == "float64" else "float")
        code = code.replace("SQRT", "sqrt" if precision == "float64" else "sqrtf")
        return code.replace("FMA", "fma" if precision == "float64" else "fmaf")

    build = BUILD.replace("GEOMETRY_CUTS", PLANE_CUTS if geometry == "plane" else SPHERE_CUTS)
    build = build.replace("GEOMETRY_BOUNDS", PLANE_BOUNDS if geometry == "plane" else SPHERE_BOUNDS)
    @wp.func_native(source(build))
    def sample(lo: real, hi: real, normal: vec, plan: Plan,
               facets: wp.array2d(dtype=vec), row: int, index: int): ...

    @wp.func_native(source(INTEGRATE))
    def integrate(plan: Plan, facets: wp.array2d(dtype=vec), row: int, column: int) -> real: ...

    @wp.func
    def output(plan: Plan, work: wp.array2d(dtype=real), row: int, column: int):
        return work[row, column]

    @wp.func
    def power(x: real, n: int):
        value = real(1.)
        for k in range(wp.abs(n)): value = value * x
        if n < 0: value = real(1.) / value
        return value

    @wp.func
    def element(lo: real, hi: real, normal: vec, plan: Plan, indices: wp.array2d(dtype=int),
                facets: wp.array2d(dtype=vec), work: wp.array2d(dtype=real), row: int):
        for i in range(plan.capacity): sample(lo, hi, normal, plan, facets, row, i)
        for j in range(plan.requests.shape[0]): work[row, j] = integrate(plan, facets, row, j)

    @wp.kernel
    def prepare_samples(lower: wp.array(dtype=real), upper: wp.array(dtype=real),
                        normals: wp.array(dtype=vec), plan: Plan, facets: wp.array2d(dtype=vec)):
        row, index = wp.tid()
        sample(lower[row], upper[row], normals[row], plan, facets, row, index)

    @wp.kernel
    def batch(lower: wp.array(dtype=real), upper: wp.array(dtype=real), normals: wp.array(dtype=vec),
              supports: wp.array(dtype=real), plan: Plan, indices: wp.array2d(dtype=int),
              scales: wp.array(dtype=int), facets: wp.array2d(dtype=vec),
              work: wp.array2d(dtype=real), out: wp.array2d(dtype=real)):
        row, col = wp.tid()
        value = integrate(plan, facets, row, col)
        work[row, col] = value
        out[row, col] = value * power(supports[row], scales[col])

    return SimpleNamespace(real=real, vec=vec, pair=pair, Plan=Plan, work_dtype=real,
        element=element, output=output, power=power, batch=batch,
        prepare_samples=prepare_samples, face_count=4*nodes*6*12)


def setup(evaluator, nodes, *, geometry="plane"):
    """Pack only the quadrature/profile contract, with no analytic DAG."""
    e = evaluator
    if e.kernel not in ("w2", "w4", "w6", "cubic"):
        raise ValueError("unsupported shell kernel")
    api = functions(e.precision, nodes, geometry=geometry)
    e.api = SimpleNamespace(**vars(api))  # capacity depends on this request mask
    requests = tuple(e.requests)
    if any(not isinstance(r, Request) or len(r.alpha) != 3 for r in requests):
        raise ValueError("shell request dimension must be three")
    degree = max((sum(r.alpha) + len(r.derivative) for r in requests), default=0)
    nphi, nt = degree + 1, (degree + 2) // 2
    e.api.face_count = 4 * nodes * nt * nphi
    plan = api.Plan()
    kw = dict(device=e.device)
    metadata = [[*r.alpha, len(r.derivative), *(list(r.derivative) + [0, 0])[:2]] for r in requests]
    plan.requests = wp.array(np.array(metadata, np.int32).reshape(-1, 6), dtype=int, **kw)
    ratio = SHAPES[e.kernel].c3_pi / SHAPES[e.kernel].c2_pi
    terms = factored_profiles(e.kernel)
    plan.terms = wp.array([[p, r, s] for p, r, s, lo, hi, c in terms], dtype=int, **kw)
    plan.coefficients = wp.array([[float(lo), float(hi), float(c * ratio) / np.pi]
                                  for p, r, s, lo, hi, c in terms], dtype=api.vec, **kw)
    for name, count in (("radial", nodes), ("polar", nt)):
        setattr(plan, name, wp.array(gauss_rule(count), dtype=api.pair, **kw))
    phi = np.arange(nphi, dtype=np.longdouble) * (2 * np.longdouble("3.141592653589793238462643383279502884") / nphi)
    plan.azimuth = wp.array(np.asarray(np.stack([np.cos(phi), np.sin(phi)], axis=-1), dtype=float), dtype=api.pair, **kw)
    plan.cubic, plan.capacity = int(e.kernel == "cubic"), e.api.face_count
    e.plan = plan
    e.host_plan = SimpleNamespace(requests=requests, kernel=e.kernel)
    e.width = len(requests)
    e.indices = wp.empty((0, 3), dtype=int, **kw)
    e.scales = wp.array([r.scale for r in requests], dtype=int, **kw)
    e.method = "spherical-shell-quadrature"
    e.accuracy_status = "explicit quadrature; finite roundoff/convergence gates, not a universal bound"


COMMON = r'''
using T=SCALAR;
#define FB_LOAD(...) (*wp::address(__VA_ARGS__))
auto power=[](T x,int n) {T v=1;for(int k=0;k<(n<0?-n:n);++k)v=v*x;return n<0?T(1)/v:v;};
'''

PLANE_CUTS = r'''
T cuts[5];int pieces=0;cuts[0]=0;
T candidates[3]={lo<T(0)?-lo:lo,hi<T(0)?-hi:hi,plan.cubic?T(.5):T(1)};
for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(candidates[j]<candidates[i]) {
    T temp=candidates[i];candidates[i]=candidates[j];candidates[j]=temp;
}
for(int i=0;i<3;++i)if(candidates[i]>cuts[pieces] && candidates[i]<T(1))cuts[++pieces]=candidates[i];
cuts[++pieces]=T(1);
int nr=plan.radial.shape[0],nt=plan.polar.shape[0],np=plan.azimuth.shape[0];
int size=nr*nt*np,count=pieces*size;
if(lo>=T(1)||hi<=T(-1)||lo>=hi)count=0;
'''

SPHERE_CUTS = r'''
T cuts[5];int pieces=0;cuts[0]=0;
// lo is signed outward distance/h, hi is radius/h. No sphere mesh is used.
T candidates[3]={wp::abs(lo),T(2)*hi+lo,plan.cubic?T(.5):T(1)};
for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(candidates[j]<candidates[i]) {
    T temp=candidates[i];candidates[i]=candidates[j];candidates[j]=temp;
}
for(int i=0;i<3;++i)if(candidates[i]>cuts[pieces] && candidates[i]<T(1))cuts[++pieces]=candidates[i];
cuts[++pieces]=T(1);
int nr=plan.radial.shape[0],nt=plan.polar.shape[0],np=plan.azimuth.shape[0];
int size=nr*nt*np,count=pieces*size;
if(lo>=T(1))count=0;
'''

PLANE_BOUNDS = r'''
T lower=lo/r,upper=hi/r;
if(lower<T(-1))lower=T(-1);if(upper>T(1))upper=T(1);
'''

SPHERE_BOUNDS = r'''
T lower=T(-1),upper=T(1),centre=hi+lo;
if(centre==T(0)) {if(r>hi)upper=T(-1);}
else {
    // 1-(r+d)(R+c+r)/(2cr), avoiding R^2-c^2 and r^2 cancellation.
    upper=T(1)-((r+lo)/r)*((hi/centre+T(1)+r/centre)/T(2));
    if(upper>T(1))upper=T(1);if(upper<T(-1))upper=T(-1);
}
'''

BUILD = r'''
GEOMETRY_CUTS
int base=row*plan.capacity;
if(index==0)wp::array_store(facets,base,2,decltype(normal)(T(count),T(0),T(0)));
if(index>=count)return;
int piece=index/size,ir=(index/nt/np)%nr,it=(index/np)%nt,ip=index%np;
auto rw=FB_LOAD(plan.radial,ir),tw=FB_LOAD(plan.polar,it),angle=FB_LOAD(plan.azimuth,ip);
T a=cuts[piece],b=cuts[piece+1],half=(b-a)/T(2),r=FMA(half,rw[0],a+half);
GEOMETRY_BOUNDS
if(lower>=upper) {
    wp::array_store(facets,base+index,0,decltype(normal)(T(0)));
    wp::array_store(facets,base+index,1,decltype(normal)(T(0)));return;
}
T thalf=(upper-lower)/T(2),t=FMA(thalf,tw[0],lower+thalf);
T rho=SQRT((T(1)-t)*(T(1)+t));
auto n=normal/wp::length(normal),axis=decltype(normal)(T(1),T(0),T(0));
if(wp::abs(n[1])<=wp::abs(n[0]) && wp::abs(n[1])<=wp::abs(n[2]))axis=decltype(normal)(T(0),T(1),T(0));
else if(wp::abs(n[2])<=wp::abs(n[0]))axis=decltype(normal)(T(0),T(0),T(1));
auto u=wp::normalize(wp::cross(axis,n)),v=wp::cross(n,u);
auto point=(n*t+(u*angle[0]+v*angle[1])*rho)*r;
T values[3]={T(0),T(0),T(0)};
for(int k=0;k<plan.terms.shape[0];++k) {
    auto c=FB_LOAD(plan.coefficients,k);
    if(r<c[0]||r>=c[1])continue;
    int profile=FB_LOAD(plan.terms,k,0),radial=FB_LOAD(plan.terms,k,1),shifted=FB_LOAD(plan.terms,k,2);
    int out=profile,exponent=radial+2;
    if(profile==2)exponent=radial+1;
    if(profile==3){out=1;exponent=radial+1;}
    if(profile==4){out=2;exponent=radial-1;}
    values[out]+=c[2]*power(r,exponent)*power(c[1]-r,shifted);
}
T weight=half*rw[1]*thalf*tw[1]*T(6.283185307179586476925286766559)/T(np);
wp::array_store(facets,base+index,0,point);
wp::array_store(facets,base+index,1,decltype(normal)(values[0]*weight,values[1]*weight,values[2]*weight));
#undef FB_LOAD
'''

INTEGRATE = r'''
int base=row*plan.capacity,count=wp::cast_int(FB_LOAD(facets,base,2)[0]);
int ax=FB_LOAD(plan.requests,column,0),ay=FB_LOAD(plan.requests,column,1),az=FB_LOAD(plan.requests,column,2);
int order=FB_LOAD(plan.requests,column,3),i=FB_LOAD(plan.requests,column,4),j=FB_LOAD(plan.requests,column,5);
T sum=0,residual=0;
for(int k=0;k<count;++k) {
    auto point=FB_LOAD(facets,base+k,0),weights=FB_LOAD(facets,base+k,1);
    T value=weights[0];
    if(order==1)value=-point[i]*weights[1];
    if(order==2)value=point[i]*point[j]*weights[2]+(i==j?weights[1]:T(0));
    T term=power(point[0],ax)*power(point[1],ay)*power(point[2],az)*value;
    T next=sum+term;
    residual+=(wp::abs(sum)>=wp::abs(term))?((sum-next)+term):((term-next)+sum);
    sum=next;
}
return sum+residual;
#undef FB_LOAD
'''
