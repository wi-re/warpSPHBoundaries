"""Exact support clipping for infinite solid half-spaces and planar bands.

The compact radial kernel makes a query-centred rectangle/box sufficient:
its artificial tangent faces lie outside support. All faces still participate
in the shared analytic flux recurrence, including its exterior corrections.
No quadrature, half-space subtraction, or curvature approximation is used.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from .contract import baseline
from .geometry import Geometry
from .evaluate import Evaluator
from .wp_func import functions as core_functions

wp.set_module_options({"enable_backward":False,"fuse_fp":False})

# Cap fans anchor exact query-on-plane cases at the cap centre. For nonzero
# offsets an eccentric fan keeps interior edges away from the query projection,
# avoiding singular intermediate primitives. Both triangulate the same region.
FACES=((8,1,0),(8,2,1),(8,3,2),(8,0,3),
       (9,4,5),(9,5,6),(9,6,7),(9,7,4),(0,1,5),(0,5,4),
       (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7))


@lru_cache(None)
def functions(dimension,precision="float64",method="analytic",nodes=16):
    """Composable device functions. Caller owns facet and recurrence scratch.

    Bounds are source-minus-query normal coordinates divided by support h.
    Supply a unit normal and bounds clipped to [-2,2]. Each row owns 4 (2D)
    or 16 (3D) facets. `output` returns normalized moments; multiply by
    h**Request.scale for physical units. Use the evaluator's reported width.
    """
    if dimension not in (2,3):raise ValueError("planar dimension must be 2 or 3")
    if method=="shell_quadrature":
        if dimension!=3:raise ValueError("shell quadrature supports 3D planes/bands only")
        from .planar_shell import functions as shell_functions
        return shell_functions(precision,nodes)
    if method=="conditioned_boundary" and dimension!=2:
        raise ValueError("conditioned boundary quadrature supports 2D planes/bands only")
    core=(core_functions(precision) if method=="analytic" else core_functions(precision,method=method,nodes=nodes))
    real,vec,Plan,work_dtype=core.real,core.vec,core.Plan,core.work_dtype
    core_element,output,power=core.element,core.output,core.power
    face_count=4 if dimension==2 else 16

    @wp.func
    def vertex(index:int,lo:real,hi:real,u:vec,v:vec,n:vec):
        z=lo
        corner=index
        if index>=4:
            z=hi
            corner=index-4
        x=real(-2.)
        y=real(-2.)
        if corner==1 or corner==2:x=real(2.)
        if corner>=2:y=real(2.)
        point=u*x+v*y+n*z
        if index==8:
            point=n*lo
            if lo!=real(0.):point=point+u*real(.25)+v*real(.5)
        if index==9:
            point=n*hi
            if hi!=real(0.):point=point+u*real(.25)+v*real(.5)
        return point

    @wp.func
    def element(lo:real,hi:real,normal:vec,plan:Plan,indices:wp.array2d(dtype=int),
                facets:wp.array2d(dtype=vec),work:wp.array2d(dtype=work_dtype),row:int):
        # Clear cached paired outputs as well as recurrence scratch on empty rows.
        if lo>=real(1.) or hi<=real(-1.) or lo>=hi:
            for j in range(work.shape[1]):work[row,j]=work_dtype(0.)
        else:
            n=wp.normalize(normal)
            u=vec(n[1],-n[0],real(0.))
            v=vec(real(0.))
            if wp.static(dimension==3):
                axis=vec(real(1.),real(0.),real(0.))
                if wp.abs(n[1])<=wp.abs(n[0]) and wp.abs(n[1])<=wp.abs(n[2]):
                    axis=vec(real(0.),real(1.),real(0.))
                elif wp.abs(n[2])<=wp.abs(n[0]):
                    axis=vec(real(0.),real(0.),real(1.))
                u=wp.normalize(wp.cross(axis,n))
                v=wp.cross(n,u)
            for f in range(wp.static(face_count)):
                for j in range(3):
                    k=indices[f,j]
                    point=vec(real(0.))
                    if wp.static(dimension==2):
                        t=real(-2.)
                        z=lo
                        if k==1 or k==2:t=real(2.)
                        if k>=2:z=hi
                        point=u*t+n*z
                    else:
                        point=vertex(k,lo,hi,u,v,n)
                    facets[row*wp.static(face_count)+f,j]=point
            core_element(vec(real(0.)),real(1.),facets,row*wp.static(face_count),
                         (row+1)*wp.static(face_count),plan,work,row)

    @wp.kernel
    def batch(lower:wp.array(dtype=real),upper:wp.array(dtype=real),normals:wp.array(dtype=vec),
              supports:wp.array(dtype=real),plan:Plan,indices:wp.array2d(dtype=int),
              scales:wp.array(dtype=int),facets:wp.array2d(dtype=vec),
              work:wp.array2d(dtype=work_dtype),out:wp.array2d(dtype=real)):
        row=wp.tid()
        element(lower[row],upper[row],normals[row],plan,indices,facets,work,row)
        for j in range(out.shape[1]):
            out[row,j]=output(plan,work,row,j)*power(supports[row],scales[j])

    return SimpleNamespace(real=real,vec=vec,Plan=Plan,work_dtype=work_dtype,
        element=element,output=output,power=power,batch=batch,face_count=face_count)


class PlanarEvaluator:
    """Infinite half-space (plane) or finite-thickness infinite band (strip).

    Plane distance is outward signed query-to-plane distance: solid points
    satisfy n dot (source-query) <= -distance. Strip distance is signed query
    distance from its centre plane: |n dot (source-query)+distance| <= half_width.
    Requests refer to global Cartesian source-minus-query monomials; derivatives
    act on the kernel only. Physical support scaling is applied once on output.
    """
    method="analytic-support-clipping"

    def __init__(self,kind,dimension,requests=None,*,kernel="w2",precision="float64",device="cpu",method="analytic",nodes=16):
        if kind not in ("plane","strip") or dimension not in (2,3):
            raise ValueError("kind must be plane/strip and dimension must be 2/3")
        self.kind,self.dimension,self.precision=kind,dimension,precision
        self.kernel=kernel
        self.requests=baseline(dimension) if requests is None else tuple(requests)
        if method=="shell_quadrature":
            if dimension!=3:raise ValueError("shell quadrature supports 3D planes/bands only")
            from .planar_shell import setup
            self.device=wp.get_device(device)
            setup(self,nodes)
            return
        self.api=functions(dimension,precision,method,nodes)
        self.method="analytic-support-clipping" if method=="analytic" else "conditioned-boundary-support-clipping"
        if dimension==2:
            template=Geometry([[[-2,-2],[2,-2],[2,2],[-2,2]]],2)
            indices=((0,1,1),(1,2,2),(2,3,3),(3,0,0))
        else:
            vertices=[[x,y,z] for z in (-2,2) for x,y in ((-2,-2),(2,-2),(2,2),(-2,2))]
            vertices.extend([[.25,.5,-2],[.25,.5,2]])
            template=Geometry.closed_surface(vertices,FACES)
            indices=FACES
        # The template supplies only geometry-independent plan packing; batch
        # facets are generated from each query's exact support-clipped solid.
        self.primitive=Evaluator(template,self.requests,kernel=kernel,precision=precision,device=device,compact=True,method=method,nodes=nodes)
        self.device=self.primitive.device
        self.plan,self.host_plan,self.width,self.scales=(self.primitive.plan,self.primitive.host_plan,
                                                       self.primitive.width,self.primitive.scales)
        self.indices=wp.array(indices,dtype=int,device=self.device)
        self.accuracy_status=("experimental raw fp32 flux recurrence; consult planar fixture errors"
                             if precision=="float32" else "finite CPU fixtures; CUDA execution unverified")
        if method!="analytic":self.accuracy_status=self.primitive.accuracy_status

    def prepare(self,distances,normals,supports,*,half_widths=None):
        d=np.atleast_1d(np.asarray(distances,dtype=float))
        if d.ndim!=1:raise ValueError("distances must be a scalar or vector")
        count=len(d)
        try:
            n=np.broadcast_to(np.asarray(normals,dtype=float),(count,self.dimension)).copy()
            h=np.broadcast_to(np.asarray(supports,dtype=float),(count,)).copy()
            a=np.zeros(count) if half_widths is None else np.broadcast_to(np.asarray(half_widths,dtype=float),(count,)).copy()
        except ValueError as exc:raise ValueError("incompatible planar batch shapes") from exc
        if not all(np.isfinite(x).all() for x in (d,n,h,a)) or np.any(h<=0):
            raise ValueError("finite planar inputs and positive support required")
        if not np.allclose(np.linalg.norm(n,axis=1),1.,rtol=0.,atol=1e-12):
            raise ValueError("normals must be unit vectors")
        with np.errstate(over="ignore",under="ignore",divide="ignore",invalid="ignore"):
            if self.kind=="strip":
                if half_widths is None or np.any(a<=0):raise ValueError("strip requires positive half widths")
                minimum=1e-2 if self.precision.startswith("float32") else 1e-4
                if np.any(a/h<minimum):raise ValueError("thin strip outside the tested conditioning domain")
                # Extended host arithmetic avoids inf-inf for finite extreme
                # inputs before clipping back to a small representable region.
                dd,aa,hh=d.astype(np.longdouble),a.astype(np.longdouble),h.astype(np.longdouble)
                lower=np.asarray(np.clip((-dd-aa)/hh,-2,2),dtype=float)
                upper=np.asarray(np.clip((-dd+aa)/hh,-2,2),dtype=float)
            else:
                if half_widths is not None:raise ValueError("plane does not accept half widths")
                lower=np.full(count,-2.)
                upper=np.clip(-d/h,-2,2)
            dtype=np.float32 if self.precision.startswith("float32") else np.float64
            hh=h.astype(dtype)
            factors=np.array([hh**r.scale for r in self.requests])
            intermediate=np.array([hh**abs(r.scale) for r in self.requests])
            valid=(np.isfinite(hh).all() and np.all(hh>0) and np.isfinite(lower).all()
                   and np.isfinite(upper).all() and np.isfinite(factors).all() and np.all(factors!=0)
                   and np.isfinite(intermediate).all() and np.all(intermediate!=0))
        if not valid:raise ValueError("planar bounds/support scaling exceed finite precision storage")
        return PlanarBatch(self,lower,upper,n,h)

    def evaluate(self,distances,normals,supports,**kwargs):
        batch=self.prepare(distances,normals,supports,**kwargs)
        batch.launch()
        return batch.numpy()


class PlanarBatch:
    """Frozen inputs; launch reuses all scratch and has no host query loop."""
    def __init__(self,evaluator,lower,upper,normals,supports):
        self.evaluator=evaluator
        self.count=len(supports)
        api=evaluator.api;device=evaluator.device
        self.lower=wp.array(lower,dtype=api.real,device=device)
        self.upper=wp.array(upper,dtype=api.real,device=device)
        self.normals=wp.array(np.pad(normals,((0,0),(0,3-evaluator.dimension))),dtype=api.vec,device=device)
        self.supports=wp.array(supports,dtype=api.real,device=device)
        self.facets=wp.empty((self.count*api.face_count,3),dtype=api.vec,device=device)
        self.work=wp.empty((self.count,evaluator.width),dtype=api.work_dtype,device=device)
        self.values=wp.empty((self.count,len(evaluator.requests)),dtype=api.real,device=device)

    @property
    def bytes(self):
        return sum(x.capacity for x in (self.lower,self.upper,self.normals,self.supports,self.facets,self.work,self.values))

    def launch(self):
        e=self.evaluator
        if self.count and self.values.shape[1]:
            dimensions=self.count
            if hasattr(e.api,"prepare_samples"):
                wp.launch(e.api.prepare_samples,(self.count,e.api.face_count),inputs=[self.lower,self.upper,
                    self.normals,e.plan,self.facets],device=e.device)
                dimensions=(self.count,self.values.shape[1])
            wp.launch(e.api.batch,dimensions,inputs=[self.lower,self.upper,self.normals,self.supports,e.plan,
                e.indices,e.scales,self.facets,self.work,self.values],device=e.device)
        return self.values

    def numpy(self):return self.values.numpy()
