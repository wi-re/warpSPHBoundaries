"""Composable native 3D plane/sphere baseline functions and reusable batches.

This family advertises exactly the degree-two combined moment/derivative
contract: the 28 baseline channels and subsets. Higher weighted products are
explicitly rejected, rather than replaced by a scalar approximation.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from ..edge.kernels import KERNELS as BLOCKS
from ..curvbound.kernels import KERNELS as SHAPES
from .contract import Request,baseline,KERNELS
from ._radial_code import arithmetic,ELEMENT

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


@lru_cache(None)
def functions(precision="float64",*,beta_early_exit=True):
    if precision not in ("float64","float32","float32_compensated"):
        raise ValueError("unknown precision")
    real=wp.float64 if precision=="float64" else wp.float32
    vec=wp.types.vector(length=3,dtype=real)
    pair=wp.types.vector(length=2,dtype=real)

    @wp.struct
    class Plan:
        powers:wp.array(dtype=int)
        radial_powers:wp.array(dtype=int)
        profiles:wp.array(dtype=int)
        bounds:wp.array(dtype=pair)
        coefficients:wp.array(dtype=pair)

    code=ELEMENT.replace("FB_BETA_EARLY_EXIT","1" if precision=="float32" and beta_early_exit else "0")
    @wp.func_native(arithmetic(precision)+code)
    def element(distance:real,radius:real,h:real,kind:int,plan:Plan,work:wp.array2d(dtype=real),row:int): ...

    @wp.func
    def select(work:wp.array2d(dtype=real),row:int,normal:vec,degree:int,i:int,j:int,order:int):
        value=real(0.)
        if degree==0 and order==0:
            value=work[row,0]
        elif degree==1 and order==0:
            value=work[row,1]*normal[i]
        elif degree==0 and order==1:
            value=work[row,4]*normal[i]
        else:
            tangent=int(2)
            axial=int(3)
            if order==1:
                tangent=5
                axial=6
            elif order==2:
                tangent=7
                axial=8
            value=(work[row,axial]-work[row,tangent])*normal[i]*normal[j]
            if i==j:
                value=value+work[row,tangent]
        return value

    @wp.kernel
    def batch(distances:wp.array(dtype=real),radii:wp.array(dtype=real),normals:wp.array(dtype=vec),
              supports:wp.array(dtype=real),kind:int,plan:Plan,requests:wp.array2d(dtype=int),
              work:wp.array2d(dtype=real),out:wp.array2d(dtype=real)):
        row=wp.tid()
        element(distances[row],radii[row],supports[row],kind,plan,work,row)
        for col in range(out.shape[1]):
            out[row,col]=select(work,row,normals[row],requests[col,0],requests[col,1],requests[col,2],requests[col,3])

    return SimpleNamespace(real=real,vec=vec,pair=pair,Plan=Plan,element=element,select=select,batch=batch)


def factored_profiles(kernel):
    """Exact c2*pi profiles as c*r**p*(piece_hi-r)**k, before rounding.

    Profile 0 is W, 1 is L=W'/r, 2 is r*Q=L'. Profiles 3/4 instead
    carry the numerators r*L and r**3*Q on cubic's outer Laurent piece.
    Wendland residual coefficients have one sign per profile, so their
    evaluation no longer cancels shifted powers with large coefficients.
    """
    from fractions import Fraction as F
    rows=[]
    for lo,hi,coeff in BLOCKS[kernel].pieces:
        laurent=len(coeff)>1 and bool(coeff[1])
        polynomials=[(0,list(coeff))]
        if laurent:
            polynomials.extend([(3,[n*c for n,c in enumerate(coeff) if n>=1]),
                                (4,[n*(n-2)*c for n,c in enumerate(coeff) if n>=1])])
        else:
            polynomials.extend([(1,[n*c for n,c in enumerate(coeff) if n>=2]),
                                (2,[n*(n-2)*c for n,c in enumerate(coeff) if n>=3])])
        for profile,poly in polynomials:
            while poly and not poly[-1]:poly.pop()
            shifted=0
            while len(poly)>1 and sum(c*hi**i for i,c in enumerate(poly))==0:
                quotient=[F(0)]*(len(poly)-1)
                quotient[-1]=-poly[-1]
                for i in range(len(quotient)-1,0,-1):
                    quotient[i-1]=hi*quotient[i]-poly[i]
                assert poly[0]==hi*quotient[0]
                poly=quotient;shifted+=1
            for radial,c in enumerate(poly):
                if c:rows.append((profile,radial,shifted,lo,hi,c))
    return tuple(rows)


class RadialEvaluator:
    """Analytic solid half-space or solid sphere, outward normals, support radius h.

    `prepare(distances, normals, supports, radii=...)` accepts per-query signed
    outward distances. For a sphere, distance=|query-centre|-radius and the normal
    points from centre to query. At the centre any unit normal is equivalent.
    Supply distances accurately; already-rounded world coordinates cannot be
    repaired by compensated arithmetic. Geometry is exact; this is not curvature.
    """
    def __init__(self,kind,requests=None,*,kernel="w2",precision="float64",device="cpu"):
        if kind not in ("plane","sphere") or kernel not in KERNELS:
            raise ValueError("kind must be plane or sphere and kernel must be supported")
        self.kind=kind
        self.kernel=kernel
        self.precision=precision
        self.device=wp.get_device(device)
        self.api=functions(precision)
        self.requests=baseline(3) if requests is None else tuple(requests)
        metadata=[]
        for r in self.requests:
            if not isinstance(r,Request) or len(r.alpha)!=3 or sum(r.alpha)+len(r.derivative)>2:
                raise ValueError("radial family supports the complete 3D baseline and subsets only")
            ids=[i for i,a in enumerate(r.alpha) for _ in range(a)]+list(r.derivative)
            metadata.append([sum(r.alpha),*(ids+[0,0])[:2],len(r.derivative)])
        powers=[];radial_powers=[];profiles=[];bounds=[];coefficients=[]
        ratio=SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi
        for profile,radial,shifted,lo,hi,c in factored_profiles(kernel):
            if profile in (3,4):assert lo>0, "singular derivative profile must be away from the origin"
            powers.append(shifted);radial_powers.append(radial);profiles.append(profile)
            bounds.append((float(lo),float(hi)))
            value=float(c*ratio)
            high=float(np.float32(value)) if precision.startswith("float32") else value
            coefficients.append((high,value-high if precision=="float32_compensated" else 0.))
        api=self.api
        self.plan=api.Plan()
        self.plan.powers=wp.array(powers,dtype=int,device=self.device)
        self.plan.radial_powers=wp.array(radial_powers,dtype=int,device=self.device)
        self.plan.profiles=wp.array(profiles,dtype=int,device=self.device)
        self.plan.bounds=wp.array(bounds,dtype=api.pair,device=self.device)
        self.plan.coefficients=wp.array(coefficients,dtype=api.pair,device=self.device)
        self.metadata=wp.array(np.array(metadata,dtype=np.int32).reshape(-1,4),dtype=int,device=self.device)
        self.accuracy_status="finite CPU baseline fixtures pass in each precision; CUDA execution unverified"

    def prepare(self,distances,normals,supports,*,radii=None):
        d=np.atleast_1d(np.asarray(distances,dtype=float))
        if d.ndim!=1:
            raise ValueError("distances must be a scalar or vector")
        count=len(d)
        try:
            h=np.broadcast_to(np.asarray(supports,dtype=float),(count,)).copy()
            n=np.broadcast_to(np.asarray(normals,dtype=float),(count,3)).copy()
            r=np.zeros(count) if radii is None else np.broadcast_to(np.asarray(radii,dtype=float),(count,)).copy()
        except ValueError as exc:
            raise ValueError("incompatible radial batch shapes") from exc
        if not all(np.isfinite(a).all() for a in (d,n,h,r)) or np.any(h<=0):
            raise ValueError("radial inputs must be finite with positive support")
        if not np.allclose(np.linalg.norm(n,axis=1),1.,rtol=0.,atol=1e-12):
            raise ValueError("normals must be unit outward vectors")
        if self.kind=="sphere":
            if radii is None or np.any(r<=0) or np.any(d < -r):
                raise ValueError("sphere needs positive radii and distance >= -radius")
            # Conservative finite acceptance domain; no hidden approximation.
            lower=1e-2 if self.precision.startswith("float32") else 1e-4
            if np.any(r/h<lower) or np.any(r/h>1e4):
                raise ValueError("radius/support outside the tested radial conditioning domain")
        elif radii is not None:
            raise ValueError("plane does not accept sphere radii")
        dtype=np.float32 if self.precision.startswith("float32") else np.float64
        with np.errstate(over="ignore",under="ignore",divide="ignore",invalid="ignore"):
            dd,nn,hh,rr=[a.astype(dtype) for a in (d,n,h,r)]
            square=hh*hh
            representable=(all(np.isfinite(a).all() for a in (dd,nn,hh,rr,dd/hh,rr/hh,square,1/square))
                           and np.all(square>0))
        if not representable:
            name="fp32" if dtype==np.float32 else "fp64"
            raise ValueError(f"radial inputs or support scaling exceed finite {name} storage")
        return RadialBatch(self,d,n,h,r)

    def evaluate(self,distances,normals,supports,**kwargs):
        batch=self.prepare(distances,normals,supports,**kwargs)
        batch.launch()
        return batch.numpy()


class RadialBatch:
    """Frozen prepared arrays; launch reuses scratch and has no allocation."""
    def __init__(self,evaluator,d,n,h,r):
        self.evaluator=evaluator
        self.count=len(d)
        api=evaluator.api
        kw=dict(device=evaluator.device)
        self.distances=wp.array(d,dtype=api.real,**kw)
        self.normals=wp.array(n,dtype=api.vec,**kw)
        self.supports=wp.array(h,dtype=api.real,**kw)
        self.radii=wp.array(r,dtype=api.real,**kw)
        self.work=wp.empty((self.count,9),dtype=api.real,**kw)
        self.values=wp.empty((self.count,len(evaluator.requests)),dtype=api.real,**kw)

    @property
    def bytes(self):
        return sum(a.capacity for a in (self.distances,self.normals,self.supports,self.radii,self.work,self.values))

    def launch(self):
        ev=self.evaluator
        if self.count and len(ev.requests):
            wp.launch(ev.api.batch,self.count,inputs=[self.distances,self.radii,self.normals,self.supports,
                int(ev.kind=="sphere"),ev.plan,ev.metadata,self.work,self.values],device=ev.device)
        return self.values

    def numpy(self):
        return self.values.numpy()
