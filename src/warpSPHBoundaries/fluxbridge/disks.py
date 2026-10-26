"""Standalone solid disk arrays with conservative search and periodic images.

Circle quadrature remains explicit. Contributions add, including overlaps;
there is no geometric union or porous resistance attached to resolved fibres.
"""
import numpy as np
import warp as wp
from .geometry import Geometry,adjacency,queries_and_supports,fingerprint
from .disk import DiskQuadratureEvaluator

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


class DiskGeometry:
    dimension=2
    _tree=Geometry._tree
    nearby=Geometry.nearby

    def __init__(self,centres,radii):
        c=np.array(centres,dtype=float,copy=True)
        if c.shape==(0,):c=c.reshape(0,2)
        if c.ndim==1:c=c.reshape(1,-1)
        if c.shape!=(len(c),2):raise ValueError("disk centres must have shape (E,2)")
        r=np.broadcast_to(np.asarray(radii,dtype=float),(len(c),)).copy()
        if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r<=0):
            raise ValueError("finite disk centres and positive radii required")
        self.centres,self.radii=c,r
        # Round outward: small radii at large world offsets must not collapse
        # the AABB to the centre and incorrectly reject near-tangent support.
        with np.errstate(over='ignore',invalid='ignore'):
            self.bounds=np.stack([np.nextafter(c-r[:,None],-np.inf),np.nextafter(c+r[:,None],np.inf)],axis=1)
        if not np.isfinite(self.bounds).all():raise ValueError("disk bounds exceed finite geometry range")
        # Generic adjacency needs only a stable element count, bounds and tree.
        self.elements=tuple(range(len(c)))
        self.revision=fingerprint(c,r)
        for a in (c,r,self.bounds):a.flags.writeable=False
        self.tree=self._tree(np.arange(len(c))) if len(c) else None


class DiskArrayEvaluator:
    """Additive solid disks, independently usable without porosity.

    Inputs are world-space queries/supports and immutable disk geometry.
    Preparation computes accurate relative geometry before fp32 conversion.
    Periodic images are unique, and reverse/forward adjacency is never inferred
    by truncating to a fixed slot capacity. Select search="device" for the
    reusable whole-disk BVH; its host CSR readback remains explicit.
    Default circle quadrature retains its baseline contract. Explicit
    method="volume_quadrature" supplies degree-nine moments/second derivatives.
    """
    def __init__(self,centres,radii,requests=None,*,kernel="w2",precision="float64",nodes=24,device="cpu",search="host",method="circle_quadrature"):
        if search not in ("host","device"):raise ValueError("search must be host or device")
        if method not in ('circle_quadrature','volume_quadrature'):raise ValueError('unknown disk integration method')
        self.geometry=DiskGeometry(centres,radii)
        primitive=DiskQuadratureEvaluator
        if method=='volume_quadrature':
            from .weighted_disk import WeightedDiskEvaluator
            primitive=WeightedDiskEvaluator
        self.primitive=primitive(requests,kernel=kernel,precision=precision,nodes=nodes,device=device)
        self.device,self.precision=self.primitive.device,precision
        self.kernel,self.nodes,self.method=kernel,nodes,self.primitive.method
        self.device_search=None
        if search=="device":
            from .device_search import DeviceSearch
            self.device_search=DeviceSearch(self.geometry,device=self.device)
        self.search=search
        self.requests=self.primitive.requests
        real=self.primitive.api.real
        @wp.kernel
        def reduce(offsets:wp.array(dtype=int),pairs:wp.array2d(dtype=real),out:wp.array2d(dtype=real)):
            row,col=wp.tid()
            total=real(0.);residual=real(0.)
            for j in range(offsets[row],offsets[row+1]):
                term=pairs[j,col]-residual;new=total+term
                residual=(new-total)-term;total=new
            out[row,col]=total
        self.reduce=reduce
        self.accuracy_status=self.primitive.accuracy_status

    def prepare(self,queries,supports,*,neighbors=None,periods=None,exhaustive=False):
        q,h=queries_and_supports(self.geometry,queries,supports)
        if neighbors is None:
            neighbors=(self.device_search.prepare(q,h,periods=periods) if self.device_search is not None and not exhaustive
                else adjacency(self.geometry,q,h,periods=periods,exhaustive=exhaustive))
        neighbors.validate(self.geometry,q,h)
        if periods is not None and tuple(periods)!=neighbors.periodic:
            raise ValueError("adjacency periodicity does not match request")
        e=neighbors.element;qi=neighbors.query
        delta=q[qi]-neighbors.shift-self.geometry.centres[e]
        lengths=np.hypot(delta[:,0],delta[:,1]);normals=np.zeros_like(delta)
        normals[:,0]=1.
        np.divide(delta,lengths[:,None],out=normals,where=lengths[:,None]>0)
        radii=self.geometry.radii[e]
        primitive=self.primitive.prepare(lengths-radii,normals,h[qi],radii=radii)
        batch=DiskArrayBatch(self,q,h,neighbors,primitive)
        if hasattr(primitive,'quadrature'):
            # The controlled volume route already evaluates during preparation;
            # validate the physical sum as well as every finite primitive row.
            batch._reduce()
            if not np.isfinite(batch.numpy()).all():raise ValueError('disk-array physical sum exceeds finite precision storage')
        return batch

    def evaluate(self,queries,supports,**kwargs):
        batch=self.prepare(queries,supports,**kwargs)
        if not hasattr(batch,'quadrature'):batch.launch()
        return batch.numpy()

    def iter_evaluate(self,queries,supports,**options):
        from .streaming import iter_evaluate
        return iter_evaluate(self,queries,supports,**options)

    def evaluate_streamed(self,queries,supports,**options):
        from .streaming import evaluate_streamed
        return evaluate_streamed(self,queries,supports,**options)


class DiskArrayBatch:
    def __init__(self,evaluator,queries,supports,neighbors,primitive):
        self.evaluator=evaluator;self.neighbors=neighbors;self.primitive=primitive
        self.queries=queries.copy();self.supports=supports.copy();self.count=len(neighbors.query)
        self.offsets=wp.array(neighbors.offsets,dtype=int,device=evaluator.device)
        self.values=wp.empty((len(queries),len(evaluator.requests)),dtype=evaluator.primitive.api.real,device=evaluator.device)
        if hasattr(primitive,'quadrature'):self.quadrature=dict(primitive.quadrature)

    @property
    def bytes(self):
        return self.primitive.bytes+self.offsets.capacity+self.values.capacity

    def launch(self):
        self.primitive.launch()
        return self._reduce()

    def _reduce(self):
        ev=self.evaluator
        if len(self.queries) and len(ev.requests):
            wp.launch(ev.reduce,self.values.shape,inputs=[self.offsets,self.primitive.values,self.values],device=ev.device)
        return self.values

    def numpy(self):return self.values.numpy()
