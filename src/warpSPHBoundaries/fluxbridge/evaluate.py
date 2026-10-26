"""Host preparation and thin batches around the shared analytic device function."""
import numpy as np
import warp as wp
from .geometry import adjacency, queries_and_supports
from .plan import AnalyticPlan
from .wp_func import functions

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


class Evaluator:
    def __init__(self,geometry,requests,*,kernel="w2",precision="float64",device="cpu",compact=False,method="analytic",nodes=16,preparation="host",search="host"):
        if preparation not in ('host','device') or (preparation=='device' and method!='tet_quadrature'):
            raise ValueError('device preparation is available for tetrahedral quadrature only')
        if search not in ("host","device"):
            raise ValueError("search must be host or device")
        requests=tuple(requests)
        self.geometry=geometry
        self.device=wp.get_device(device)
        self.search=search
        self.device_search=None
        if search=="device":
            from .device_search import DeviceSearch
            self.device_search=DeviceSearch(geometry,device=self.device)
        self.precision=precision
        self.method=method
        self.nodes=nodes
        self.preparation=preparation
        self.accuracy_status=("experimental: raw fp32 W4/W6 or higher weighted derivatives miss the 1e-5 fixture gate"
                              if precision=="float32" and (kernel in ("w4","w6") or any(sum(r.alpha)>2 for r in requests)) else
                              "finite baseline fixtures only; no universal error bound")
        if precision=="float32_compensated":
            self.accuracy_status="paired-fp32 analytic reduction; finite CPU/CUDA fixtures pass 1e-5; no universal error bound"
        self.coordinate_origin=(geometry.facets[0,0].copy() if len(geometry.facets) else np.zeros(3))
        self.api=functions(precision) if method=="analytic" else functions(precision,method=method,nodes=nodes)
        plan_type=AnalyticPlan
        if method=="conditioned_boundary":
            from .conditioned2d import ConditionedPlan
            plan_type=ConditionedPlan
            self.accuracy_status="explicit conditioned boundary quadrature; finite accuracy/convergence checks required"
        if method=="tet_quadrature":
            if geometry.dimension!=3 or geometry.measure!="volume" or any(v.shape!=(4,3) for v in geometry.elements):
                raise ValueError("tet quadrature requires 3D volume tetrahedra")
            from .tet_quadrature import QuadraturePlan
            plan_type=QuadraturePlan
            self.accuracy_status="convergence-checked tetrahedral quadrature; no certified global error bound"
        self.host_plan=plan_type(geometry.dimension,requests,kernel,geometry.measure,compact=compact)
        arrays=self.host_plan.arrays()
        api=self.api
        self.plan=api.Plan()
        for name in ("f","dep","fr","b","br","v","vr","offsets"):
            value=arrays[name]
            setattr(self.plan,name,wp.array(value,dtype=int if value.dtype==np.int32 else api.real,device=self.device))
        terms=arrays["terms"]
        self.plan.term_kind=wp.array(terms[:,0].astype(np.int32),dtype=int,device=self.device)
        self.plan.term_index=wp.array(terms[:,1].astype(np.int32),dtype=int,device=self.device)
        coefficients=terms[:,2]
        if api.coefficient_dtype==wp.vec2f:
            high=coefficients.astype(np.float32)
            coefficients=np.stack([high,(coefficients-high.astype(np.float64)).astype(np.float32)],axis=-1)
        self.plan.term_value=wp.array(coefficients,dtype=api.coefficient_dtype,device=self.device)
        self.plan.dimension=geometry.dimension
        self.plan.compact=int(compact)
        self.plan.surface=int(geometry.measure=="surface")
        if method in ("conditioned_boundary","tet_quadrature"):self.host_plan.upload(self.plan,api,self.device,nodes)
        self.element_origins=np.array([np.pad(v[0],(0,3-geometry.dimension)) for v in geometry.elements]).reshape(-1,3)
        local_facets=geometry.facets-self.coordinate_origin
        if api.local_geometry:
            owners=np.repeat(np.arange(len(geometry.elements)),np.diff(geometry.starts))
            local_facets=geometry.facets-self.element_origins[owners,None,:]
        if precision.startswith("float32"):
            with np.errstate(over="ignore",under="ignore"):
                representable=np.isfinite(local_facets.astype(np.float32)).all()
            if not representable:
                raise ValueError("geometry exceeds finite fp32 storage range")
        self.facets=wp.array(local_facets,dtype=api.vec,device=self.device)
        self.starts=wp.array(geometry.starts,dtype=int,device=self.device)
        # Surface uses a 3D-normalized kernel against a 2D measure.
        shift=-1 if geometry.measure=="surface" else 0
        self.scales=wp.array([r.scale+shift for r in self.host_plan.requests],dtype=int,device=self.device)
        self.width=len(arrays['f'])+len(arrays['b'])+len(arrays['v'])
        if api.output_cache:
            self.width+=getattr(api,"cache_columns",1)*len(self.host_plan.requests)

    def _check_storage(self,q,h):
        if self.method in ("conditioned_boundary","tet_quadrature"):
            dtype=np.float32 if self.precision=="float32" else np.float64
            with np.errstate(over="ignore",under="ignore",divide="ignore",invalid="ignore"):
                hh=h.astype(dtype)
                factors=np.array([hh**r.scale for r in self.host_plan.requests])
                intermediate=np.array([hh**abs(r.scale) for r in self.host_plan.requests])
            if not (np.isfinite(factors).all() and np.all(factors!=0) and np.isfinite(intermediate).all() and np.all(intermediate!=0)):
                raise ValueError("conditioned support scaling exceeds finite precision storage")
        if self.precision.startswith("float32"):
            with np.errstate(over="ignore",under="ignore"):
                supports=h.astype(np.float32)
                local=q-self.coordinate_origin[:self.geometry.dimension]
                finite=np.isfinite(local.astype(np.float32)).all() and np.isfinite(supports).all()
            if not finite or np.any(supports<=0):
                raise ValueError("queries/supports exceed positive finite fp32 storage range")
    def _check_domain(self,q,h,neighbors):
        if not len(neighbors.element):
            return
        limit=1e4 if self.precision.startswith("float32") else 1e8
        if np.any(self.geometry.condition[neighbors.element]>limit):
            raise ValueError("ill-conditioned geometry: use the arbitrary-precision reference")
        spans=np.ptp(self.geometry.bounds[neighbors.element],axis=1).max(axis=1)
        if self.method in ("conditioned_boundary","tet_quadrature") and np.any(spans/h[neighbors.query]>1e8):
            raise ValueError("conditioned boundary quadrature requires normalized element spans <= 1e8")
        tiny=1e-2 if self.precision.startswith("float32") else 1e-4
        # This conservative rejection is a documented precision policy, not a
        # certified error estimator or a silent quadrature substitution.
        if np.any(spans/h[neighbors.query]<tiny):
            raise ValueError("tiny element relative to support: use the arbitrary-precision reference")

    def _neighbors(self,q,h,*,exhaustive=False,periods=None):
        if self.device_search is not None and not exhaustive:
            return self.device_search.prepare(q,h,periods=periods)
        return adjacency(self.geometry,q,h,exhaustive=exhaustive,periods=periods)

    def prepare(self,queries,supports,*,neighbors=None,exhaustive=False,periods=None):
        q,h=queries_and_supports(self.geometry,queries,supports)
        self._check_storage(q,h)
        if neighbors is None:
            neighbors=self._neighbors(q,h,exhaustive=exhaustive,periods=periods)
        neighbors.validate(self.geometry,q,h)
        if periods is not None and tuple(periods)!=neighbors.periodic:
            raise ValueError("adjacency periodicity does not match the request")
        self._check_domain(q,h,neighbors)
        return Batch(self,q,h,neighbors)

    def evaluate(self,queries,supports,**kwargs):
        batch=self.prepare(queries,supports,**kwargs)
        batch.launch()
        return batch.numpy()

    def iter_evaluate(self,queries,supports,**options):
        """Explicit bounded chunks; yields independent NumPy EvaluationChunks."""
        from .streaming import iter_evaluate
        return iter_evaluate(self,queries,supports,**options)

    def evaluate_streamed(self,queries,supports,**options):
        from .streaming import evaluate_streamed
        return evaluate_streamed(self,queries,supports,**options)


class Batch:
    """Owns allocations. launch() reuses buffers and is graph-capture compatible.

    Geometry, adjacency, plan and material snapshots are frozen. Changes require
    a new batch. Kernels have no per-query host dispatch and no float atomics.
    """
    def __init__(self,evaluator,queries,supports,neighbors,columns=None):
        self.evaluator=evaluator
        self.neighbors=neighbors
        self.queries=queries.copy()
        self.supports=supports.copy()
        api=evaluator.api
        self.count=len(neighbors.query)
        self.columns=len(evaluator.host_plan.requests) if columns is None else columns
        self.is_field_batch=columns is not None
        pad=lambda a: np.pad(a,((0,0),(0,3-a.shape[1])))
        self.q=wp.array(pad(queries)-evaluator.coordinate_origin,dtype=api.vec,device=evaluator.device)
        self.evaluation_queries=self.q
        if api.local_geometry:
            relative=pad(queries[neighbors.query])-pad(neighbors.shift)-evaluator.element_origins[neighbors.element]
            self.evaluation_queries=wp.array(relative,dtype=api.vec,device=evaluator.device)
        self.h=wp.array(supports,dtype=api.real,device=evaluator.device)
        self.pq=wp.array(neighbors.query,dtype=int,device=evaluator.device)
        self.pe=wp.array(neighbors.element,dtype=int,device=evaluator.device)
        self.shifts=wp.array(pad(neighbors.shift),dtype=api.vec,device=evaluator.device)
        self.work=wp.empty((self.count,evaluator.width),dtype=api.work_dtype,device=evaluator.device)
        self.pairs=wp.empty((self.count,self.columns),dtype=api.real,device=evaluator.device)
        self.values=wp.zeros((len(queries),self.columns),dtype=api.real,device=evaluator.device)
        self.offsets=wp.array(neighbors.offsets,dtype=int,device=evaluator.device)
        real=api.real
        @wp.kernel
        def reduce(offsets: wp.array(dtype=int),pairs: wp.array2d(dtype=real),out: wp.array2d(dtype=real)):
            i,j=wp.tid()
            total=real(0.0)
            residual=real(0.0)
            for k in range(offsets[i],offsets[i+1]):
                term=pairs[k,j]-residual
                new=total+term
                residual=(new-total)-term
                total=new
            out[i,j]=total
        self.reduce=reduce
        self.integration_plan=evaluator.plan
        self.quadrature_bytes=0
        if getattr(api,"prepared_integration",False):
            from .tet_quadrature import bind
            bind(self)

    @property
    def bytes(self):
        arrays=(self.work,self.pairs,self.values,self.q,self.h,self.pq,self.pe,self.shifts,self.offsets)
        return sum(a.capacity for a in arrays)+(self.evaluation_queries.capacity if self.evaluation_queries is not self.q else 0)+self.quadrature_bytes

    def _prepare_integrals(self):
        if getattr(self.evaluator.api,"prepared_integration",False):
            from .tet_quadrature import run_prepared
            run_prepared(self.evaluator.api,self.integration_plan,self.work,self.evaluator.device)

    def launch(self):
        ev=self.evaluator
        if self.count and self.columns:
            self._prepare_integrals()
            dimensions=(self.count,self.columns) if getattr(ev.api,"prepared_integration",False) else self.count
            wp.launch(ev.api.batch,dimensions,inputs=[self.evaluation_queries,self.h,ev.facets,ev.starts,self.pq,self.pe,self.shifts,self.integration_plan,ev.scales,self.work,self.pairs],device=ev.device)
        if len(self.queries) and self.columns:
            wp.launch(self.reduce,self.values.shape,inputs=[self.offsets,self.pairs,self.values],device=ev.device)
        return self.values

    def numpy(self):
        return self.values.numpy()
