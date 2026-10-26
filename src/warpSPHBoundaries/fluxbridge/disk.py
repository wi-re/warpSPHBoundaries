"""Complete 2D disk baseline by explicitly selected circle quadrature.

This is numerical integration of an exact circle, not an analytic disk formula,
table lookup, polygon replacement or a silent fallback of another evaluator.
"""
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from ..edge.kernels import KERNELS as BLOCKS
from ..edge.np2d import cheb_coeffs
from .contract import Request,baseline,KERNELS
from .radial import functions as radial_functions,RadialBatch
from ._disk_code import definitions,ELEMENT

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


@lru_cache(None)
def functions(precision="float64"):
    common=radial_functions(precision)
    real,vec,pair=common.real,common.vec,common.pair
    select=common.select
    @wp.struct
    class Plan:
        bounds:wp.array(dtype=pair)
        coefficients:wp.array3d(dtype=pair)
        lengths:wp.array2d(dtype=int)
        offsets:wp.array2d(dtype=pair)
        constants:wp.array(dtype=pair)
        nodes:wp.array(dtype=pair)
        weights:wp.array(dtype=pair)

    @wp.func_native(definitions(precision)+ELEMENT)
    def element(distance:real,radius:real,h:real,plan:Plan,work:wp.array2d(dtype=real),row:int): ...

    @wp.kernel
    def batch(distances:wp.array(dtype=real),radii:wp.array(dtype=real),supports:wp.array(dtype=real),normals:wp.array(dtype=vec),
              plan:Plan,requests:wp.array2d(dtype=int),work:wp.array2d(dtype=real),out:wp.array2d(dtype=real)):
        row=wp.tid()
        element(distances[row],radii[row],supports[row],plan,work,row)
        for col in range(out.shape[1]):
            out[row,col]=select(work,row,normals[row],requests[col,0],requests[col,1],requests[col,2],requests[col,3])
    return SimpleNamespace(real=real,vec=vec,pair=pair,Plan=Plan,element=element,select=select,batch=batch)


def profiles(kernel):
    """Exact pi-free circle profiles before Chebyshev conversion/rounding."""
    pieces=BLOCKS[kernel].pieces
    mass=sum(c*(hi**(n+2)-lo**(n+2))/(n+2) for lo,hi,coeff in pieces for n,c in enumerate(coeff))
    variance=sum(c*(hi**(n+4)-lo**(n+4))/(n+4) for lo,hi,coeff in pieces for n,c in enumerate(coeff))
    assert mass==F(1,2)
    macc=F(0);aacc=F(0);result=[]
    for lo,hi,coeff in pieces:
        moff=macc-sum(c*lo**(n+2)/(n+2) for n,c in enumerate(coeff))
        macc=moff+sum(c*hi**(n+2)/(n+2) for n,c in enumerate(coeff))
        phi0=moff-mass
        apoly={n+4:c/((n+2)*(n+4)) for n,c in enumerate(coeff) if c}
        apoly[2]=phi0/2
        aoff=aacc-sum(c*lo**n for n,c in apoly.items())
        aacc=aoff+sum(c*hi**n for n,c in apoly.items())
        w={n:c for n,c in enumerate(coeff) if c}
        phi={n+2:c/(n+2) for n,c in enumerate(coeff) if c};phi[0]=phi0
        qm={n:c/(n+2) for n,c in enumerate(coeff) if c}
        qa={n-2:c for n,c in apoly.items() if c}
        ell={n-2:n*c for n,c in enumerate(coeff) if n>=2 and c}
        polynomials=[cheb_coeffs(p,hi) for p in (w,phi,qm,qa,ell)]
        result.append((lo,hi,polynomials,(moff,aoff,coeff[1] if len(coeff)>1 else F(0))))
    assert 2*aacc==-variance
    return result,(mass,aacc,variance)


class DiskQuadratureEvaluator:
    """Outward disk distance/normal, all 15 baseline channels or a subset.

    Distance=|query-centre|-radius. Gauss order (8..64) is explicit; geometric
    panels split at kernel knots and refine towards the nearest circle point.
    There is no certified quadrature bound for arbitrary inputs. Use independent
    fixtures/convergence checks and the stated radius/support conditioning policy.
    """
    method="circle-quadrature"
    def __init__(self,requests=None,*,kernel="w2",precision="float64",nodes=24,device="cpu"):
        if kernel not in KERNELS or not isinstance(nodes,int) or not 8<=nodes<=64:
            raise ValueError("supported kernel and integer quadrature nodes in 8..64 required")
        self.kernel,self.precision,self.nodes=kernel,precision,nodes
        self.requests=baseline(2) if requests is None else tuple(requests)
        metadata=[]
        for req in self.requests:
            if not isinstance(req,Request) or len(req.alpha)!=2 or sum(req.alpha)+len(req.derivative)>2:
                raise ValueError("disk quadrature supports the full 2D baseline and subsets only")
            ids=[i for i,a in enumerate(req.alpha) for _ in range(a)]+list(req.derivative)
            metadata.append([sum(req.alpha),*(ids+[0,0])[:2],len(req.derivative)])
        self.device=wp.get_device(device);self.api=functions(precision);api=self.api
        def paired(values):
            values=np.asarray(values,dtype=np.float64)
            high=values.astype(np.float32).astype(np.float64) if precision.startswith('float32') else values
            low=values-high if precision=='float32_compensated' else np.zeros_like(values)
            return np.stack([high,low],axis=-1)
        pcs,constants=profiles(kernel)
        width=max(len(p) for _,_,polys,_ in pcs for p in polys)
        coefficients=np.zeros((len(pcs),5,width));lengths=np.zeros((len(pcs),5),dtype=np.int32)
        for i,(_,_,polys,_) in enumerate(pcs):
            for j,p in enumerate(polys):
                lengths[i,j]=len(p);coefficients[i,j,:len(p)]=[float(c)/np.pi for c in p]
        self.plan=api.Plan();kw=dict(device=self.device)
        self.plan.bounds=wp.array([[float(lo),float(hi)] for lo,hi,_,_ in pcs],dtype=api.pair,**kw)
        self.plan.coefficients=wp.array(paired(coefficients),dtype=api.pair,**kw)
        self.plan.lengths=wp.array(lengths,dtype=int,**kw)
        self.plan.offsets=wp.array(paired([[float(c)/np.pi for c in off] for _,_,_,off in pcs]),dtype=api.pair,**kw)
        self.plan.constants=wp.array(paired([float(constants[0])/np.pi,float(constants[1])/np.pi,float(constants[2])]),dtype=api.pair,**kw)
        points,weights=np.polynomial.legendre.leggauss(nodes)
        self.plan.nodes=wp.array(paired(points),dtype=api.pair,**kw)
        self.plan.weights=wp.array(paired(weights),dtype=api.pair,**kw)
        self.metadata=wp.array(np.array(metadata,dtype=np.int32).reshape(-1,4),dtype=int,**kw)
        self.accuracy_status="explicit circle quadrature; finite CPU/CUDA fixtures pass; no universal error bound"

    def prepare(self,distances,normals,supports,*,radii):
        d=np.atleast_1d(np.asarray(distances,dtype=float))
        if d.ndim!=1:raise ValueError("distances must be scalar or vector")
        count=len(d)
        try:
            n=np.broadcast_to(np.asarray(normals,dtype=float),(count,2)).copy()
            h=np.broadcast_to(np.asarray(supports,dtype=float),(count,)).copy()
            r=np.broadcast_to(np.asarray(radii,dtype=float),(count,)).copy()
        except ValueError as exc:raise ValueError("incompatible disk batch shapes") from exc
        if not all(np.isfinite(a).all() for a in (d,n,h,r)) or np.any(h<=0) or np.any(r<=0) or np.any(d < -r):
            raise ValueError("finite disk inputs, positive support/radius and distance >= -radius required")
        if not np.allclose(np.linalg.norm(n,axis=1),1.,rtol=0.,atol=1e-12):
            raise ValueError("disk normals must be unit outward vectors")
        lower=1e-2 if self.precision.startswith('float32') else 1e-4
        with np.errstate(over='ignore',divide='ignore',under='ignore',invalid='ignore'):
            if np.any(r/h<lower) or np.any(r/h>1e4):
                raise ValueError("radius/support outside disk conditioning policy")
            dtype=np.float32 if self.precision.startswith('float32') else np.float64
            dd,hh,rr=[a.astype(dtype) for a in (d,h,r)];square=hh*hh
            finite=all(np.isfinite(a).all() for a in (dd,hh,rr,dd/hh,rr/hh,square,1/square)) and np.all(square>0)
        if not finite:raise ValueError("disk inputs/support scaling exceed finite storage precision")
        return DiskBatch(self,d,np.pad(n,((0,0),(0,1))),h,r)

    def evaluate(self,distances,normals,supports,**kwargs):
        batch=self.prepare(distances,normals,supports,**kwargs);batch.launch();return batch.numpy()


class DiskBatch(RadialBatch):
    def launch(self):
        ev=self.evaluator
        if self.count and len(ev.requests):
            wp.launch(ev.api.batch,self.count,inputs=[self.distances,self.radii,self.supports,self.normals,
                ev.plan,ev.metadata,self.work,self.values],device=ev.device)
        return self.values
