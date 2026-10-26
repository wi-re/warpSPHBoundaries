"""Convergence-checked spherical boundary quadrature for tetrahedra.

Euler radial primitives are combined exactly before storage. Face/edge angular
panels split at kernel-sphere cuts and dyadic geometric scales. Frozen prepared
queries bind the sample arrays; device consumers reject mismatched query data.
There is no planar polynomial-exactness claim for these angular integrals.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb
from types import SimpleNamespace
import numpy as np
import warp as wp
from ..edge.kernels import KERNELS
from ..curvbound.kernels import KERNELS as SHAPES
from ..edge.np2d import cheb_coeffs
from .contract import Request
from .planar_shell import gauss_rule

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

STATIC = ("f", "dep", "fr", "b", "br", "v", "vr", "term_kind", "term_index",
          "term_value", "offsets", "dimension", "surface", "compact", "requests",
          "coefficients", "lengths", "bounds", "constants", "inner_scaled", "inner_lengths")


class QuadraturePlan:
    def __init__(self, dimension, requests, kernel, measure="volume", compact=False):
        if dimension != 3 or measure != "volume":
            raise ValueError("tet quadrature requires 3D volume tetrahedra")
        if kernel not in ("w2", "w4", "w6", "cubic"): raise ValueError("unsupported kernel")
        self.requests, self.kernel = tuple(requests), kernel
        self.profiles = []; self.metadata = []; self.coefficients = []; self.constants = []; self.inner_scaled=[]
        def term(alpha, order, sign):
            key = (sum(alpha), order)
            if key not in self.profiles: self.profiles.append(key)
            return [*alpha, self.profiles.index(key), sign]
        for req in self.requests:
            if not isinstance(req, Request) or len(req.alpha) != 3: raise ValueError("request dimension mismatch")
            a, ds = req.alpha, req.derivative
            if not ds: row = term(a, 0, 1) + [0,0,0,-1,0]
            elif len(ds) == 1:
                row = term(tuple(v+int(j == ds[0]) for j,v in enumerate(a)), 1, -1) + [0,0,0,-1,0]
            else:
                i,j = ds
                row = term(tuple(v+int(k == i)+int(k == j) for k,v in enumerate(a)), 2, 1)
                row += term(a, 1, 1) if i == j else [0,0,0,-1,0]
            self.metadata.append(row)
        pieces = KERNELS[kernel].pieces
        self.bounds = [(float(lo), float(hi)) for lo,hi,_ in pieces]
        ratio = SHAPES[kernel].c3_pi / SHAPES[kernel].c2_pi
        for degree, order in self.profiles:
            cumulative = F(0); polys = []
            for lo,hi,coeff in pieces:
                poly = {}
                for n,c in enumerate(coeff):
                    factor = 1 if order == 0 else n if order == 1 else n*(n-2)
                    if not c or not factor: continue
                    power = n-2*order+degree+3
                    assert power > 0
                    poly[power] = poly.get(power, F(0)) + c*factor*ratio/power
                poly[0] = cumulative-sum(c*lo**n for n,c in poly.items())
                cumulative = sum(c*hi**n for n,c in poly.items())
                if lo==0:
                    assert all(n>=3 or not c for n,c in poly.items())
                    regular={n-3:c for n,c in poly.items() if c}
                    self.inner_scaled.append([float(c)/np.pi for c in cheb_coeffs(regular,hi)])
                shifted = {j:sum(c*comb(n,j)*lo**(n-j) for n,c in poly.items() if n >= j)
                           for j in range(max(poly)+1)}
                polys.append([float(c)/np.pi for c in cheb_coeffs(shifted,hi-lo)])
            self.coefficients.append(polys); self.constants.append(float(cumulative)/np.pi)

    def arrays(self):
        return dict(f=np.empty((0,9),np.int32), dep=np.empty((0,9),np.int32), fr=np.empty(0),
            b=np.empty((0,5),np.int32), br=np.empty(0), v=np.empty((0,7),np.int32), vr=np.empty(0),
            terms=np.empty((0,3)), offsets=np.zeros(len(self.requests)+1,np.int32))

    def upload(self, plan, api, device, nodes):
        width = max((len(c) for ps in self.coefficients for c in ps), default=1)
        coeff = np.zeros((len(self.profiles),len(self.bounds),width)); lengths = np.zeros(coeff.shape[:2],np.int32)
        for i,ps in enumerate(self.coefficients):
            for j,c in enumerate(ps): coeff[i,j,:len(c)] = c; lengths[i,j] = len(c)
        kw = dict(device=device)
        plan.requests = wp.array(np.array(self.metadata,np.int32).reshape(-1,10),dtype=int,**kw)
        plan.coefficients = wp.array(coeff,dtype=api.real,**kw); plan.lengths = wp.array(lengths,dtype=int,**kw)
        plan.bounds = wp.array(self.bounds,dtype=api.real,**kw)
        plan.constants = wp.array(self.constants,dtype=api.real,**kw)
        inner=np.zeros((len(self.profiles),width));inner_lengths=[]
        for i,c in enumerate(self.inner_scaled):inner[i,:len(c)]=c;inner_lengths.append(len(c))
        plan.inner_scaled=wp.array(inner,dtype=api.real,**kw);plan.inner_lengths=wp.array(inner_lengths,dtype=int,**kw)


@lru_cache(None)
def functions(precision="float64", nodes=16):
    if precision not in ("float32", "float64"): raise ValueError("tet quadrature supports ordinary fp32/fp64 only")
    if not isinstance(nodes,int) or not 8 <= nodes <= 64: raise ValueError("quadrature nodes must be in [8,64]")
    real = wp.float64 if precision == "float64" else wp.float32
    vec = wp.types.vector(length=3,dtype=real)
    @wp.struct
    class Plan:
        f: wp.array2d(dtype=int)
        dep: wp.array2d(dtype=int)
        fr: wp.array(dtype=real)
        b: wp.array2d(dtype=int)
        br: wp.array(dtype=real)
        v: wp.array2d(dtype=int)
        vr: wp.array(dtype=real)
        term_kind: wp.array(dtype=int)
        term_index: wp.array(dtype=int)
        term_value: wp.array(dtype=real)
        offsets: wp.array(dtype=int)
        dimension: int
        surface: int
        compact: int
        requests: wp.array2d(dtype=int)
        coefficients: wp.array3d(dtype=real)
        lengths: wp.array2d(dtype=int)
        bounds: wp.array2d(dtype=real)
        constants: wp.array(dtype=real)
        inner_scaled: wp.array2d(dtype=real)
        inner_lengths: wp.array(dtype=int)
        directions: wp.array(dtype=vec)
        radii: wp.array(dtype=real)
        weights: wp.array(dtype=real)
        node_offsets: wp.array(dtype=int)
        profile_index: wp.array(dtype=int)
        profile_offsets: wp.array(dtype=int)
        profile_kind: wp.array(dtype=int)
        bound_queries: wp.array(dtype=vec)
        bound_supports: wp.array(dtype=real)
        profiles: wp.array2d(dtype=real)
        partials: wp.array2d(dtype=real)
        tile_offsets: wp.array(dtype=int)
        tile_rows: wp.array(dtype=int)
        tiles: int

    source = (COMMON+INTEGRATE).replace("SCALAR","double" if precision == "float64" else "float")
    @wp.func_native(source)
    def integral_range(plan: Plan, first: int, last: int, column: int) -> real: ...

    @wp.func_native("return "+("fma" if precision == "float64" else "fmaf")+"(x,y,z);")
    def fused(x: real, y: real, z: real) -> real: ...

    @wp.func
    def profile(plan: Plan, radius: real, index: int):
        result = plan.constants[index]
        if radius < real(1.):
            piece = int(0)
            for p in range(1,plan.bounds.shape[0]):
                if radius >= plan.bounds[p,0]: piece = p
            lo,hi = plan.bounds[piece,0],plan.bounds[piece,1]
            x = real(2.)*(radius-lo)/(hi-lo)-real(1.)
            b1,b2 = real(0.),real(0.)
            for k in range(plan.lengths[index,piece]-1,0,-1):
                value = fused(real(2.)*x,b1,plan.coefficients[index,piece,k]-b2)
                # Warp 1.17 loop phi stores alias a tuple swap here.
                b2 = b1
                b1 = value
            result = fused(x,b1,plan.coefficients[index,piece,0]-b2)
        return result

    @wp.func
    def sampled_profile(plan: Plan,radius: real,index: int,kind: int):
        result=real(0.)
        if kind!=0 and radius<plan.bounds[0,1]:
            x=real(2.)*radius/plan.bounds[0,1]-real(1.)
            b1,b2=real(0.),real(0.)
            for k in range(plan.inner_lengths[index]-1,0,-1):
                value=fused(real(2.)*x,b1,plan.inner_scaled[index,k]-b2)
                b2=b1
                b1=value
            result=fused(x,b1,plan.inner_scaled[index,0]-b2)
        else:
            result=profile(plan,radius,index)
            if kind!=0:result=result/(radius*radius*radius)
        return result

    @wp.func
    def power(x: real, n: int):
        value = real(1.)
        for k in range(wp.abs(n)): value = value*x
        if n < 0: value = real(1.)/value
        return value

    @wp.func
    def bound(query: vec, h: real, plan: Plan, row: int):
        valid = row >= 0 and row < plan.bound_supports.shape[0]
        if valid:
            valid = h == plan.bound_supports[row]
            for k in range(3): valid = valid and query[k] == plan.bound_queries[row][k]
        return valid

    @wp.func
    def output(plan: Plan, work: wp.array2d(dtype=real), row: int, column: int):
        return work[row,column]

    @wp.func
    def integral(plan: Plan, row: int, column: int):
        return integral_range(plan,plan.node_offsets[row],plan.node_offsets[row+1],column)

    @wp.func
    def element(query: vec, h: real, facets: wp.array2d(dtype=vec), first: int, last: int,
                plan: Plan, work: wp.array2d(dtype=real), row: int):
        if bound(query,h,plan,row):
            for k in range(plan.profile_offsets[row],plan.profile_offsets[row+1]):
                for j in range(plan.constants.shape[0]):
                    plan.profiles[k,j] = sampled_profile(plan,plan.radii[k],j,plan.profile_kind[k])
            for col in range(plan.requests.shape[0]): work[row,col] = integral(plan,row,col)
        else:
            for col in range(plan.requests.shape[0]): work[row,col] = real(wp.nan)

    @wp.kernel
    def prepare_profiles(plan: Plan):
        k,j = wp.tid()
        plan.profiles[k,j] = sampled_profile(plan,plan.radii[k],j,plan.profile_kind[k])

    @wp.kernel
    def integrate_tiles(plan: Plan):
        tile,col = wp.tid()
        row=plan.tile_rows[tile]
        first=plan.node_offsets[row]+256*(tile-plan.tile_offsets[row])
        last=wp.min(first+256,plan.node_offsets[row+1])
        plan.partials[tile,col]=integral_range(plan,first,last,col)

    @wp.kernel
    def integrate_rows(plan: Plan, work: wp.array2d(dtype=real)):
        row,col = wp.tid()
        total=real(0.); residual=real(0.)
        for tile in range(plan.tile_offsets[row],plan.tile_offsets[row+1]):
            value=plan.partials[tile,col]
            next_value=total+value
            if wp.abs(total)>=wp.abs(value):residual=residual+((total-next_value)+value)
            else:residual=residual+((value-next_value)+total)
            total=next_value
        work[row,col] = total+residual

    @wp.kernel
    def batch(queries: wp.array(dtype=vec),supports: wp.array(dtype=real),facets: wp.array2d(dtype=vec),
              starts: wp.array(dtype=int),pq: wp.array(dtype=int),pe: wp.array(dtype=int),shifts: wp.array(dtype=vec),
              plan: Plan,scales: wp.array(dtype=int),work: wp.array2d(dtype=real),out: wp.array2d(dtype=real)):
        row,col = wp.tid(); qi = pq[row]
        value = real(wp.nan)
        if bound(queries[row],supports[qi],plan,row): value = output(plan,work,row,col)*power(supports[qi],scales[col])
        out[row,col] = value
    return SimpleNamespace(real=real,vec=vec,Plan=Plan,element=element,output=output,power=power,batch=batch,
        work_dtype=real,coefficient_dtype=real,field_dtype=real,output_cache=True,cache_columns=1,local_geometry=True,
        prepared_integration=True,parallel_fields=True,bound=bound,prepare_profiles=prepare_profiles,
        integrate_tiles=integrate_tiles,integrate_rows=integrate_rows)


def angular_samples(facets,query,h,kernel,nodes):
    """Host panel preparation; no input-independent truncation of distant faces."""
    x,gw = gauss_rule(nodes).T
    uu=[]; rr=[]; ww=[]; total=0; knots=(.5,1.) if kernel == "cubic" else (1.,)
    for face in (facets-query)/h:
        normal=np.cross(face[1]-face[0],face[2]-face[0]); normal/=np.linalg.norm(normal)
        d=np.dot(normal,face[0]); c0=abs(d)
        if c0 == 0: continue
        sign0=np.sign(d)
        for av,bv in zip(face,np.roll(face,-1,axis=0)):
            tangent=(bv-av)/np.linalg.norm(bv-av); outward=np.cross(tangent,normal)
            z=np.dot(outward,av); c=abs(z)
            if c == 0: continue
            sign=np.sign(z); s0=np.dot(tangent,av); s1=np.dot(tangent,bv)
            for side in (-1,1):
                a=max(s0,0.) if side == 1 else max(-s1,0.)
                end=max(s1,0.) if side == 1 else max(-s0,0.)
                while a < end:
                    b=min(end,max(2*a,c))
                    for R in knots:
                        if R*R > c0*c0+c*c:
                            cut=np.sqrt(max(0.,R*R-c0*c0-c*c))
                            if a < cut < b: b=cut
                    left=np.arctan2(a,c); right=np.arctan2(b,c); half=(right-left)/2
                    phi=(left+right)/2+half*x
                    sn=np.sin(phi)*side; cs=np.cos(phi)
                    directions=sign*outward*cs[:,None]+tangent*sn[:,None]
                    maxrho=c/cs
                    # The radial breakpoints are common to every azimuth node;
                    # only the last interval is clipped by its edge radius.
                    # Vectorize the whole azimuth rule instead of a Python
                    # loop allocating one array per individual Gauss node.
                    rho=0.; last=float(max(maxrho))
                    exterior=np.sqrt((1-c0)*(1+c0)) if c0 < 1 else 0.
                    while rho < last:
                        outer=c0 >= 1 or rho >= exterior
                        high=last if outer else min(last,max(2*rho,c0))
                        for R in knots:
                            if R > c0:
                                cut=np.sqrt((R-c0)*(R+c0))
                                if rho < cut < high: high=cut
                        active=maxrho > rho
                        top=np.minimum(high,maxrho[active])
                        t0=np.arctan2(rho,c0); t1=np.arctan2(top,c0); dh=(t1-t0)/2
                        keep=dh > 0
                        if np.any(keep):
                            e=directions[active][keep]; pw=gw[active][keep]
                            dh=dh[keep]; t1=t1[keep]
                            total+=len(e)*nodes
                            if total > 2_000_000: raise ValueError("tet quadrature sample budget exceeded")
                            theta=(t0+t1[:,None])/2+dh[:,None]*x
                            st=np.sin(theta); ct=np.cos(theta)
                            uu.append((normal[None,None,:]*(sign0*ct[:,:,None])+e[:,None,:]*st[:,:,None]).reshape(-1,3))
                            rr.append((np.ones_like(ct) if outer else c0/ct).ravel())
                            ww.append((sign0*sign*half*pw[:,None]*dh[:,None]*gw[None,:]*st).ravel())
                        rho=high
                    a=b
    if not rr: return np.empty((0,3)),np.empty(0),np.empty(0)
    return np.concatenate(uu),np.concatenate(rr),np.concatenate(ww)


def prepared_samples(facets,query,h,kernel,nodes):
    """Use the regular potential directly on moderate near-plane triangles."""
    normalized=(facets-query)/h
    ab=normalized[:,1]-normalized[:,0];ac=normalized[:,2]-normalized[:,0]
    raw=np.cross(ab,ac);jacobian=np.linalg.norm(raw,axis=1);normal=raw/jacobian[:,None]
    distance=np.sum(normal*normalized[:,0],axis=1)
    span=np.maximum(np.maximum(np.linalg.norm(ab,axis=1),np.linalg.norm(ac,axis=1)),
                    np.linalg.norm(normalized[:,2]-normalized[:,1],axis=1))
    direct=(distance!=0)&(abs(distance)<=1e-12)&(span<=4)
    u,r,w=angular_samples(facets[~direct],query,h,kernel,nodes)
    if not np.any(direct):return u,r,w,np.zeros(len(r),np.int32)
    directions=[u];radii=[r];weights=[w];kinds=[np.zeros(len(r),np.int32)]
    x,gw=gauss_rule(nodes).T;a=(x[:,None]+1)/2;b=(x[None,:]+1)/2
    for i in np.flatnonzero(direct):
        y=normalized[i,0]+a[:,:,None]*ab[i]+((1-a)*b)[:,:,None]*ac[i]
        radius=np.linalg.norm(y,axis=2)
        directions.append((y/radius[:,:,None]).reshape(-1,3));radii.append(radius.ravel())
        weights.append((jacobian[i]*distance[i]*(1-a)*gw[:,None]*gw[None,:]/4).ravel())
        kinds.append(np.ones(radius.size,np.int32))
    return np.concatenate(directions),np.concatenate(radii),np.concatenate(weights),np.concatenate(kinds)


def disjoint(facets, query, h):
    vertices=facets.reshape(-1,3)
    gap=np.maximum(np.maximum(vertices.min(axis=0)-query,query-vertices.max(axis=0)),0)/h
    padding=64*np.finfo(float).eps*(np.max(abs((vertices-query)/h))+1)
    if np.linalg.norm(gap) > 1+padding: return True
    normalized=(facets-query)/h
    sides=[]; nearest=np.inf; condition=1.
    for a,b,c in normalized:
        ab,ac=b-a,c-a; matrix=np.stack([ab,ac],axis=-1)
        cond=np.linalg.cond(matrix); condition=max(condition,cond)
        n=np.cross(ab,ac); n/=np.linalg.norm(n); d=np.dot(n,a); sides.append(d)
        uv=np.linalg.lstsq(matrix,d*n-a,rcond=None)[0]
        bary_margin=64*np.finfo(float).eps*cond
        if np.min(uv) >= -bary_margin and sum(uv) <= 1+bary_margin:
            distance=abs(d)
        else:
            distance=np.inf
            for u,v in ((a,b),(b,c),(c,a)):
                edge=v-u;t=np.clip(-np.dot(u,edge)/np.dot(edge,edge),0,1)
                distance=min(distance,np.linalg.norm(u+t*edge))
        nearest=min(nearest,distance)
    padding*=condition
    if min(sides) >= -padding: return False
    return nearest > 1+padding


def run_prepared(api,plan,work,device):
    if plan.profiles.shape[0] and plan.profiles.shape[1]:
        wp.launch(api.prepare_profiles,plan.profiles.shape,inputs=[plan],device=device)
    if work.shape[0] and work.shape[1]:
        if plan.tiles:wp.launch(api.integrate_tiles,(plan.tiles,work.shape[1]),inputs=[plan],device=device)
        wp.launch(api.integrate_rows,work.shape,inputs=[plan,work],device=device)


def bind(batch, minimum_order=None):
    """Compare two rules, promote on failure, and reject unresolved convergence."""
    ev=batch.evaluator; api=ev.api; precision=ev.precision
    previous_promotions=getattr(batch,'quadrature',{}).get('promotions',0) if minimum_order is not None else 0
    tol=1e-14 if precision == "float64" else 1e-5
    pad=lambda a:np.pad(a,((0,0),(0,3-a.shape[1])))
    queries=pad(batch.queries[batch.neighbors.query])-pad(batch.neighbors.shift)-ev.element_origins[batch.neighbors.element]
    supports=batch.supports[batch.neighbors.query]
    local=ev.geometry.facets.copy()
    owners=np.repeat(np.arange(len(ev.geometry.elements)),np.diff(ev.geometry.starts))
    local-=ev.element_origins[owners,None,:]
    def build(order):
        if ev.preparation=='device':
            from .device_panels import samples
            arrays=samples(batch,local,queries,supports,order)
            plan=api.Plan()
            for name in STATIC:setattr(plan,name,getattr(ev.plan,name))
            for name,value in arrays.items():setattr(plan,name,value)
            plan.bound_queries=wp.clone(batch.evaluation_queries)
            plan.bound_supports=wp.array(supports,dtype=api.real,device=ev.device)
            plan.profiles=wp.empty((plan.radii.shape[0],len(ev.host_plan.profiles)),dtype=api.real,device=ev.device)
            plan.partials=wp.empty((plan.tiles,len(ev.host_plan.requests)),dtype=api.real,device=ev.device)
            return plan
        dirs=[]; radii=[]; weights=[]; offsets=[0]; indices=[]; profile_offsets=[0];kinds=[]
        for row,ei in enumerate(batch.neighbors.element):
            face=local[ev.geometry.starts[ei]:ev.geometry.starts[ei+1]]
            if not ev.host_plan.requests or disjoint(face,queries[row],supports[row]): u,r,w,kind=np.empty((0,3)),np.empty(0),np.empty(0),np.empty(0,np.int32)
            else: u,r,w,kind=prepared_samples(face,queries[row],supports[row],ev.host_plan.kernel,order)
            if len(r) > 2_000_000: raise ValueError("tet quadrature sample budget exceeded")
            # Full radial panels share their nodes across azimuth samples, and
            # every exterior sample uses radius one. Deduplicate only exactly
            # equal values after storage rounding, independently for each row.
            # The quadrature rule and summation order remain unchanged.
            dtype=np.float64 if precision=='float64' else np.float32
            if not np.any(kind):
                unique_radius,inverse=np.unique(r.astype(dtype),return_inverse=True);unique_kind=np.zeros(len(unique_radius),np.int32)
            else:
                stored=np.empty(len(r),dtype=[('kind',np.int32),('radius',dtype)])
                stored['kind']=kind;stored['radius']=r
                unique,inverse=np.unique(stored,return_inverse=True)
                unique_radius,unique_kind=unique['radius'],unique['kind']
            dirs.append(u); radii.append(unique_radius);kinds.append(unique_kind); weights.append(w); offsets.append(offsets[-1]+len(r))
            indices.append(inverse+profile_offsets[-1]);profile_offsets.append(profile_offsets[-1]+len(unique_radius))
        if not dirs: dirs=[np.empty((0,3))]; radii=[np.empty(0)]; weights=[np.empty(0)]
        tile_counts=(np.diff(offsets)+255)//256
        tile_offsets=np.r_[0,np.cumsum(tile_counts)]
        scratch_scalars=profile_offsets[-1]*len(ev.host_plan.profiles)+int(tile_offsets[-1])*len(ev.host_plan.requests)
        if scratch_scalars*(8 if precision == "float64" else 4) > 2**30:
            raise ValueError("tet quadrature profile/tile storage exceeds the 1 GiB per-rule budget")
        plan=api.Plan()
        for name in STATIC: setattr(plan,name,getattr(ev.plan,name))
        kw=dict(device=ev.device)
        plan.directions=wp.array(np.concatenate(dirs),dtype=api.vec,**kw)
        plan.radii=wp.array(np.concatenate(radii),dtype=api.real,**kw)
        plan.weights=wp.array(np.concatenate(weights),dtype=api.real,**kw)
        plan.node_offsets=wp.array(offsets,dtype=int,**kw)
        plan.profile_index=wp.array(np.concatenate(indices) if indices else np.empty(0),dtype=int,**kw)
        plan.profile_offsets=wp.array(profile_offsets,dtype=int,**kw)
        plan.profile_kind=wp.array(np.concatenate(kinds) if kinds else np.empty(0,np.int32),dtype=int,**kw)
        plan.bound_queries=wp.clone(batch.evaluation_queries)
        plan.bound_supports=wp.array(supports,dtype=api.real,**kw)
        plan.profiles=wp.empty((profile_offsets[-1],len(ev.host_plan.profiles)),dtype=api.real,**kw)
        plan.tiles=int(tile_offsets[-1])
        plan.tile_offsets=wp.array(tile_offsets,dtype=int,**kw)
        plan.tile_rows=wp.array(np.repeat(np.arange(batch.count),tile_counts),dtype=int,**kw)
        plan.partials=wp.empty((plan.tiles,len(ev.host_plan.requests)),dtype=api.real,**kw)
        return plan
    order=ev.nodes if minimum_order is None else minimum_order
    plan=build(order); run_prepared(api,plan,batch.work,ev.device)
    promotions=0; error=0.
    if batch.work.size:
        while True:
            other=order+8 if order < 64 else 56
            high=build(other); check=wp.empty_like(batch.work)
            run_prepared(api,high,check,ev.device)
            low_values=batch.work.numpy(); high_values=check.numpy()
            error=float(np.max(abs(low_values-high_values)/(1+abs(high_values))))
            if np.isfinite(error) and error <= tol: break
            if order >= 64: raise ValueError(f"tet quadrature did not converge: mixed difference {error}")
            plan=high; order=other; batch.work=check; promotions+=1
    batch.integration_plan=plan
    if batch.is_field_batch and batch.work.size:
        batch.comparison_plan=high
        batch.comparison_work=check
    batch.quadrature=dict(nodes=order,requested_nodes=ev.nodes,comparison_nodes=(other if batch.work.size else None),
        worst_mixed_difference=error,budget=tol,promotions=previous_promotions+promotions,samples=plan.weights.shape[0],
        cached_radii=plan.radii.shape[0],preparation=ev.preparation,geometry_precision='float64',
        control="two-rule convergence on normalized requested moments; not a certified error bound")
    batch.quadrature_bytes=sum(getattr(plan,name).capacity for name in
        ("directions","radii","weights","node_offsets","profile_index","profile_offsets","profile_kind","bound_queries","bound_supports","profiles","partials","tile_offsets","tile_rows"))
    if batch.work.size:
        batch.quadrature["comparison_sample_count"]=high.weights.shape[0]
        batch.quadrature["comparison_cached_radii"]=high.radii.shape[0]
    if ev.preparation=='host':batch.quadrature['unique_profile_arguments']=plan.radii.shape[0]


def check_field(batch):
    """Also control error after polynomial coefficient translation/contraction."""
    ev=batch.evaluator; owner=batch.owner
    if not batch.count or not batch.columns: return
    tolerance=1e-12 if ev.precision == "float64" else 1e-5
    promotions=0
    def contract(plan,work,out):
        batch._contract(plan,work,out)
    while True:
        contract(batch.integration_plan,batch.work,batch.pairs)
        other=wp.empty_like(batch.pairs)
        contract(batch.comparison_plan,batch.comparison_work,other)
        value=batch.pairs.numpy(); reference=other.numpy()
        error=float(np.max(abs(value-reference)/(1+abs(reference))))
        if np.isfinite(error) and error <= tolerance: break
        order=batch.quadrature["nodes"]
        if order >= 32:
            raise ValueError(f"tet polynomial field did not converge in {ev.precision}: mixed difference {error}; use a better conditioned field/precision")
        bind(batch,minimum_order=order+8); promotions+=1
    batch.quadrature.update(field_mixed_difference=error,field_budget=tolerance,field_promotions=promotions)
    del batch.comparison_plan,batch.comparison_work


COMMON=r'''
using T=SCALAR;
#define FB_LOAD(...) (*wp::address(__VA_ARGS__))
auto power=[](T x,int n) {T value=1;for(int k=0;k<n;++k)value=value*x;return value;};
'''
INTEGRATE=r'''
T total=0,residual=0;
for(int k=first;k<last;++k) {
    auto u=FB_LOAD(plan.directions,k);T value=0;
    for(int t=0;t<2;++t) {
        int off=5*t,index=FB_LOAD(plan.requests,column,off+3);
        if(index>=0) {
            T weighted=FB_LOAD(plan.weights,k)*FB_LOAD(plan.profiles,FB_LOAD(plan.profile_index,k),index);
            value+=T(FB_LOAD(plan.requests,column,off+4))*power(u[0],FB_LOAD(plan.requests,column,off))*
                power(u[1],FB_LOAD(plan.requests,column,off+1))*power(u[2],FB_LOAD(plan.requests,column,off+2))*weighted;
        }
    }
    T next=total+value;
    residual+=(wp::abs(total)>=wp::abs(value))?((total-next)+value):((value-next)+total);
    total=next;
}
return total+residual;
#undef FB_LOAD
'''
