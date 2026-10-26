"""Element-local polynomial fields and fused device contractions.

Straight simplices support exact rational P0–P3 basis construction. Products
are algebraic through total degree nine. Coefficients are physical monomials
in source-origin; device consumers translate them relative to each query.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import comb, factorial
import numpy as np
import warp as wp
from ..mesh3d.fem import expansion, _lattice
from ..mesh3d.reference import _poly_mul
from .contract import Request,monomials
from .evaluate import Evaluator,Batch
from .geometry import queries_and_supports,adjacency
from .wp_func import functions

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


class PolynomialField:
    def __init__(self,geometry,coefficients):
        if len(coefficients)!=len(geometry.elements):
            raise ValueError("one polynomial per element required")
        self.geometry=geometry
        self.coefficients=[]
        self.shape=None
        d=geometry.dimension
        for poly in coefficients:
            packed={}
            for a,c in poly.items():
                Request(a)
                if len(a)!=d:
                    raise ValueError("field dimension mismatch")
                value=np.array(c,dtype=np.float64,copy=True)
                if self.shape is None:
                    self.shape=value.shape
                if value.shape!=self.shape or not np.isfinite(value).all():
                    raise ValueError("finite consistent field coefficient shapes required")
                value.flags.writeable=False
                packed[tuple(a)]=value
            self.coefficients.append(packed)
        if self.shape is None:
            raise ValueError("field must contain a coefficient")
        # Remove exact zeros only. This reduces the requested moment degree for
        # constant/zero fields without thresholding small physical coefficients.
        for poly in self.coefficients:
            for a in tuple(poly):
                if not np.any(poly[a]):
                    del poly[a]
            if not poly:
                zero=np.zeros(self.shape,dtype=np.float64)
                zero.flags.writeable=False
                poly[(0,)*d]=zero
        self.degree=max(sum(a) for p in self.coefficients for a in p)
        self.origins=np.array([v[0] for v in geometry.elements])

    @classmethod
    def constant(cls,geometry,values):
        values=np.asarray(values)
        if len(values)!=len(geometry.elements):
            raise ValueError("one constant value per element required")
        return cls(geometry,[{(0,)*geometry.dimension:v} for v in values])

    @classmethod
    def basis(cls,geometry,degree=1,kind="lagrange"):
        if degree not in range(4) or kind not in ("lagrange","bernstein"):
            raise ValueError("simplex basis must be P0–P3 Lagrange or Bernstein")
        polynomials=[]
        for verts in geometry.elements:
            if len(verts)!= (3 if geometry.dimension==2 or geometry.measure=="surface" else 4):
                raise ValueError("FEM basis requires straight simplices")
            v=np.pad(verts,((0,0),(0,3-geometry.dimension))).tolist()
            origin=v[0]
            _,lag=expansion(v,origin,degree)
            if kind=="bernstein":
                _,linear=expansion(v,origin,1)
                lag=[]
                for b in _lattice(degree,len(v)) if degree else [(0,)*len(v)]:
                    p={(0,0,0):Fraction(factorial(degree))}
                    for i,bi in enumerate(b):
                        for _ in range(bi):
                            p=_poly_mul(p,linear[i])
                        p={a:c/factorial(bi) for a,c in p.items()}
                    lag.append(p)
            out={}
            for i,p in enumerate(lag):
                for a,c in p.items():
                    key=a[:geometry.dimension]
                    if geometry.dimension==2 and a[2]:
                        continue
                    if key not in out:
                        out[key]=np.zeros(len(lag))
                    out[key][i]+=float(c)
            polynomials.append(out)
        return cls(geometry,polynomials)

    @classmethod
    def from_controls(cls,geometry,controls,degree=1,kind="bernstein"):
        basis=cls.basis(geometry,degree,kind)
        controls=np.asarray(controls,dtype=np.float64)
        if controls.ndim<2 or controls.shape[:2]!=(len(geometry.elements),basis.shape[0]):
            raise ValueError("controls must have shape [element,basis,...]")
        return cls(geometry,[{a:np.tensordot(c,controls[e],axes=(0,0)) for a,c in poly.items()} for e,poly in enumerate(basis.coefficients)])

    def product(self,other,action=np.multiply):
        if other.geometry is not self.geometry:
            raise ValueError("field products require the same geometry snapshot and origin")
        out=[]
        for p,q in zip(self.coefficients,other.coefficients):
            r={}
            for a,c in p.items():
                for b,d in q.items():
                    key=tuple(i+j for i,j in zip(a,b))
                    Request(key)
                    value=action(c,d)
                    r[key]=r.get(key,0)+value
            out.append(r)
        return PolynomialField(self.geometry,out)


@lru_cache(None)
def field_functions(precision,method="analytic",nodes=16):
    if precision=="float32_compensated":
        if method!="analytic":raise ValueError("paired fields require the analytic method")
        from .paired import field_functions as paired_fields
        return paired_fields()
    api=functions(precision) if method=="analytic" else functions(precision,method=method,nodes=nodes)
    real,vec,Plan=api.real,api.vec,api.Plan
    element,output,power=api.element,api.output,api.power
    parallel=bool(getattr(api,"parallel_fields",False))

    @wp.func
    def translated(plan: Plan,work: wp.array2d(dtype=real),row: int,
                   terms: wp.array2d(dtype=int),factors: wp.array(dtype=real),
                   term: int,delta: vec,h: real,scales: wp.array(dtype=int)):
        col=terms[term,1]
        value=factors[term]*output(plan,work,row,col)*power(h,scales[col])
        for j in range(3):
            value=value*power(delta[j],terms[term,2+j])
        return value

    @wp.kernel
    def fused(queries: wp.array(dtype=vec),supports: wp.array(dtype=real),facets: wp.array2d(dtype=vec),
              starts: wp.array(dtype=int),pq: wp.array(dtype=int),pe: wp.array(dtype=int),shifts: wp.array(dtype=vec),
              plan: Plan,scales: wp.array(dtype=int),work: wp.array2d(dtype=real),
              terms: wp.array2d(dtype=int),factors: wp.array(dtype=real),origins: wp.array(dtype=vec),
              coefficients: wp.array3d(dtype=real),pairs: wp.array2d(dtype=real)):
        row=int(0)
        begin=int(0)
        end=pairs.shape[1]
        if wp.static(parallel):
            row,column=wp.tid()
            begin=column
            end=column+1
        else:row=wp.tid()
        qi=pq[row]
        ei=pe[row]
        x=vec()
        if wp.static(api.local_geometry):x=queries[row]
        else:x=queries[qi]-shifts[row]
        h=supports[qi]
        if wp.static(not parallel):element(x,h,facets,starts[ei],starts[ei+1],plan,work,row)
        delta=x-origins[ei]
        term_count=terms.shape[0]
        if wp.static(parallel):
            if plan.node_offsets[row]==plan.node_offsets[row+1]:term_count=0
        for col in range(begin,end):
            total=real(0.0)
            residual=real(0.0)
            for t in range(term_count):
                coefficient=coefficients[ei,terms[t,0],col]
                if coefficient!=real(0.0):
                    value=coefficient*translated(plan,work,row,terms,factors,t,delta,h,scales)
                    y=value-residual
                    new=total+y
                    residual=(new-total)-y
                    total=new
            if wp.static(parallel):
                if not api.bound(x,h,plan,row):total=real(wp.nan)
            pairs[row,col]=total

    @wp.kernel
    def transpose(queries: wp.array(dtype=vec),supports: wp.array(dtype=real),
                  pq: wp.array(dtype=int),shifts: wp.array(dtype=vec),plan: Plan,scales: wp.array(dtype=int),
                  work: wp.array2d(dtype=real),terms: wp.array2d(dtype=int),factors: wp.array(dtype=real),
                  origins: wp.array(dtype=vec),reverse: wp.array(dtype=int),offsets: wp.array(dtype=int),
                  loads: wp.array2d(dtype=real),volumes: wp.array(dtype=real),out: wp.array3d(dtype=real)):
        ei,a,c=wp.tid()
        total=real(0.0)
        for k in range(offsets[ei],offsets[ei+1]):
            row=reverse[k]
            qi=pq[row]
            delta=vec()
            if wp.static(api.local_geometry):delta=queries[row]
            else:delta=queries[qi]-shifts[row]-origins[ei]
            weight=real(0.0)
            term_count=terms.shape[0]
            if wp.static(parallel):
                if plan.node_offsets[row]==plan.node_offsets[row+1]:term_count=0
            for t in range(term_count):
                if terms[t,0]==a:
                    weight=weight+translated(plan,work,row,terms,factors,t,delta,supports[qi],scales)
            if wp.static(parallel):
                if not api.bound(queries[row],supports[qi],plan,row):weight=real(wp.nan)
            total=total+volumes[qi]*loads[qi,c]*weight
        out[ei,a,c]=total
    return fused,transpose


@lru_cache(None)
def prepared_field_functions(precision,nodes):
    """Translate each integral once, then share it across field outputs/transpose.

    Moments and coefficients retain ordinary selected-dtype storage. Neumaier
    and FMA residuals use an explicit correction workspace in that same dtype;
    no coefficient/profile pairing, precision fallback or fp64 evaluation.
    """
    api=functions(precision,method='tet_quadrature',nodes=nodes)
    real,vec,Plan=api.real,api.vec,api.Plan

    @wp.func_native('return ::fmaf(a,b,c);' if precision=='float32' else 'return ::fma(a,b,c);')
    def multiply_add(a: real,b: real,c: real) -> real: ...

    @wp.kernel
    def weights(queries: wp.array(dtype=vec),supports: wp.array(dtype=real),
                pq: wp.array(dtype=int),plan: Plan,work: wp.array2d(dtype=real),
                scales: wp.array(dtype=int),terms: wp.array2d(dtype=int),factors: wp.array(dtype=real),
                offsets: wp.array(dtype=int),out: wp.array2d(dtype=real),errors: wp.array2d(dtype=real)):
        row,a=wp.tid()
        qi=pq[row];h=supports[qi];delta=queries[row]
        total=real(0.0);residual=real(0.0)
        if plan.node_offsets[row]!=plan.node_offsets[row+1]:
            for t in range(offsets[a],offsets[a+1]):
                col=terms[t,1]
                moment=api.output(plan,work,row,col);factor=factors[t]
                value=factor*moment
                correction=multiply_add(factor,moment,-value)
                scale=api.power(h,scales[col]);new=value*scale
                correction=multiply_add(correction,scale,multiply_add(value,scale,-new));value=new
                for j in range(3):
                    for k in range(terms[t,2+j]):
                        new=value*delta[j]
                        correction=multiply_add(correction,delta[j],multiply_add(value,delta[j],-new))
                        value=new
                residual=residual+correction
                new=total+value
                if wp.abs(total)>=wp.abs(value):residual=residual+((total-new)+value)
                else:residual=residual+((value-new)+total)
                total=new
        result=total+residual
        error=real(0.0)
        if wp.abs(total)>=wp.abs(residual):error=(total-result)+residual
        else:error=(residual-result)+total
        if not api.bound(queries[row],h,plan,row):result=real(wp.nan)
        out[row,a]=result
        errors[row,a]=error

    @wp.kernel
    def contract(weights: wp.array2d(dtype=real),errors: wp.array2d(dtype=real),pe: wp.array(dtype=int),
                 coefficients: wp.array3d(dtype=real),out: wp.array2d(dtype=real)):
        row,col=wp.tid();ei=pe[row]
        total=real(0.0);residual=real(0.0)
        for a in range(weights.shape[1]):
            coefficient=coefficients[ei,a,col];weight=weights[row,a]
            value=coefficient*weight
            # Error-free product residual for selected-dtype inputs, followed
            # by the same compensated sum. No higher-precision operands.
            residual=residual+multiply_add(coefficient,errors[row,a],multiply_add(coefficient,weight,-value))
            new=total+value
            if wp.abs(total)>=wp.abs(value):residual=residual+((total-new)+value)
            else:residual=residual+((value-new)+total)
            total=new
        out[row,col]=total+residual

    @wp.kernel
    def transpose(weights: wp.array2d(dtype=real),errors: wp.array2d(dtype=real),queries: wp.array(dtype=vec),supports: wp.array(dtype=real),
                  plan: Plan,pq: wp.array(dtype=int),reverse: wp.array(dtype=int),offsets: wp.array(dtype=int),
                  loads: wp.array2d(dtype=real),volumes: wp.array(dtype=real),out: wp.array3d(dtype=real)):
        ei,a,c=wp.tid();total=real(0.0);residual=real(0.0)
        for k in range(offsets[ei],offsets[ei+1]):
            row=reverse[k];qi=pq[row]
            value=volumes[qi]*loads[qi,c]*weights[row,a]
            residual=residual+volumes[qi]*loads[qi,c]*errors[row,a]
            if not api.bound(queries[row],supports[qi],plan,row):value=real(wp.nan)
            new=total+value
            if wp.abs(total)>=wp.abs(value):residual=residual+((total-new)+value)
            else:residual=residual+((value-new)+total)
            total=new
        out[ei,a,c]=total+residual
    return weights,contract,transpose


class FieldEvaluator:
    def __init__(self,field,*,derivative=(),kernel="w2",precision="float64",device="cpu",compact=True,method="analytic",nodes=16,preparation="host",search="host"):
        self.field=field
        self.alphas=monomials(field.geometry.dimension,field.degree)
        self.evaluator=Evaluator(field.geometry,[Request(a,derivative) for a in self.alphas],kernel=kernel,precision=precision,device=device,compact=compact,method=method,nodes=nodes,preparation=preparation,search=search)
        ev=self.evaluator
        terms=[]; factors=[]
        ids={a:i for i,a in enumerate(self.alphas)}
        for i,a in enumerate(self.alphas):
            for beta in product(*(range(t+1) for t in a)):
                terms.append([i,ids[beta]]+[x-y for x,y in zip(a,beta)]+[0]*(3-len(a)))
                factors.append(np.prod([comb(x,y) for x,y in zip(a,beta)]))
        self.terms=wp.array(np.array(terms,dtype=np.int32),dtype=int,device=ev.device)
        self.factors=wp.array(factors,dtype=ev.api.real,device=ev.device)
        origins=np.pad(field.origins,((0,0),(0,3-field.geometry.dimension)))-ev.coordinate_origin
        if ev.api.local_geometry:
            origins=np.zeros_like(origins)
        self.origins=wp.array(origins,dtype=ev.api.vec,device=ev.device)
        self.components=int(np.prod(field.shape))
        coeff=np.zeros((len(field.coefficients),len(self.alphas),self.components))
        for e,p in enumerate(field.coefficients):
            for a,c in p.items():
                coeff[e,ids[a]]=np.ravel(c)
        if ev.api.field_dtype==wp.vec2f:
            with np.errstate(over="ignore",under="ignore"):
                high=coeff.astype(np.float32)
            if not np.isfinite(high).all():
                raise ValueError("field coefficients exceed finite fp32 storage range")
            coeff=np.stack([high,(coeff-high.astype(np.float64)).astype(np.float32)],axis=-1)
        elif ev.api.field_dtype==wp.float32:
            with np.errstate(over="ignore",under="ignore"):
                coeff=coeff.astype(np.float32)
            if not np.isfinite(coeff).all():
                raise ValueError("field coefficients exceed finite fp32 storage range")
        self.coefficients=wp.array(coeff,dtype=ev.api.field_dtype,device=ev.device)
        self.fused,self.transpose_kernel=field_functions(precision,method,nodes)
        if method=='tet_quadrature':
            counts=np.bincount(np.array(terms)[:,0],minlength=len(self.alphas))
            self.term_offsets=wp.array(np.r_[0,np.cumsum(counts)].astype(np.int32),dtype=int,device=ev.device)
            self.weight_kernel,self.contract_kernel,self.prepared_transpose=prepared_field_functions(precision,nodes)

    def prepare(self,queries,supports,*,neighbors=None,exhaustive=False,periods=None):
        ev=self.evaluator
        q,h=queries_and_supports(ev.geometry,queries,supports)
        ev._check_storage(q,h)
        if neighbors is None:
            neighbors=ev._neighbors(q,h,exhaustive=exhaustive,periods=periods)
        neighbors.validate(ev.geometry,q,h)
        if periods is not None and tuple(periods)!=neighbors.periodic:
            raise ValueError("adjacency periodicity does not match the request")
        ev._check_domain(q,h,neighbors)
        return FieldBatch(self,q,h,neighbors)

    def evaluate(self,queries,supports,**kwargs):
        batch=self.prepare(queries,supports,**kwargs)
        batch.launch()
        return batch.numpy().reshape((len(batch.queries),)+self.field.shape)

    def iter_evaluate(self,queries,supports,**options):
        from .streaming import iter_evaluate
        return iter_evaluate(self,queries,supports,**options)

    def evaluate_streamed(self,queries,supports,**options):
        from .streaming import evaluate_streamed
        return evaluate_streamed(self,queries,supports,**options)


class FieldBatch(Batch):
    def __init__(self,owner,queries,supports,neighbors):
        super().__init__(owner.evaluator,queries,supports,neighbors,columns=owner.components)
        self.owner=owner
        self.launched=False
        order=np.argsort(neighbors.element,kind="stable").astype(np.int32)
        counts=np.bincount(neighbors.element,minlength=len(owner.field.geometry.elements))
        self.reverse=wp.array(order,dtype=int,device=self.evaluator.device)
        self.reverse_offsets=wp.array(np.r_[0,np.cumsum(counts)].astype(np.int32),dtype=int,device=self.evaluator.device)
        self.field_weights=None
        self.field_weight_errors=None
        if self.evaluator.method=='tet_quadrature':
            self.field_weights=wp.empty((self.count,len(owner.alphas)),dtype=self.evaluator.api.real,device=self.evaluator.device)
            self.field_weight_errors=wp.empty_like(self.field_weights)
        if getattr(self.evaluator.api,"prepared_integration",False):
            from .tet_quadrature import check_field
            check_field(self)

    @property
    def bytes(self):
        return super().bytes+self.reverse.capacity+self.reverse_offsets.capacity+(0 if self.field_weights is None else self.field_weights.capacity+self.field_weight_errors.capacity)

    def _contract(self,plan,work,out):
        ev=self.evaluator;o=self.owner
        if self.field_weights is not None:
            wp.launch(o.weight_kernel,self.field_weights.shape,inputs=[self.evaluation_queries,self.h,self.pq,
                plan,work,ev.scales,o.terms,o.factors,o.term_offsets,self.field_weights,self.field_weight_errors],device=ev.device)
            wp.launch(o.contract_kernel,(self.count,self.columns),inputs=[self.field_weights,self.field_weight_errors,self.pe,o.coefficients,out],device=ev.device)
        else:
            wp.launch(o.fused,self.count,inputs=[self.evaluation_queries,self.h,ev.facets,ev.starts,self.pq,self.pe,self.shifts,plan,ev.scales,work,
                o.terms,o.factors,o.origins,o.coefficients,out],device=ev.device)

    def launch(self):
        ev=self.evaluator; o=self.owner
        if self.count:
            self._prepare_integrals()
            self._contract(self.integration_plan,self.work,self.pairs)
        if len(self.queries):
            wp.launch(self.reduce,self.values.shape,inputs=[self.offsets,self.pairs,self.values],device=ev.device)
        self.launched=True
        return self.values

    def transpose(self,loads,volumes):
        if not self.launched:
            self.launch()
        o=self.owner; ev=self.evaluator
        loads=np.asarray(loads,dtype=np.float64).reshape(len(self.queries),o.components)
        volumes=np.broadcast_to(np.asarray(volumes,dtype=np.float64),(len(self.queries),))
        if not np.isfinite(loads).all() or not np.isfinite(volumes).all() or np.any(volumes<0):
            raise ValueError("finite loads and nonnegative particle volumes required")
        l=wp.array(loads,dtype=ev.api.real,device=ev.device)
        v=wp.array(volumes,dtype=ev.api.real,device=ev.device)
        shape=self.owner.coefficients.shape
        out=wp.zeros(shape,dtype=ev.api.real,device=ev.device)
        if np.prod(shape):
            if self.field_weights is not None:
                wp.launch(o.prepared_transpose,shape,inputs=[self.field_weights,self.field_weight_errors,self.evaluation_queries,self.h,self.integration_plan,
                    self.pq,self.reverse,self.reverse_offsets,l,v,out],device=ev.device)
            else:
                wp.launch(o.transpose_kernel,shape,inputs=[self.evaluation_queries,self.h,self.pq,self.shifts,self.integration_plan,ev.scales,self.work,o.terms,o.factors,
                    o.origins,self.reverse,self.reverse_offsets,l,v,out],device=ev.device)
        return out.numpy().reshape(shape[:2]+o.field.shape)


def differential(field,queries,supports,*,hessian=False,**options):
    """Batched interpolation and kernel-derivative contractions.

    Gradient index order is [query, field components..., derivative axis].
    Vector divergence/curl differentiate W in the query coordinate. Each
    derivative batch reuses one adjacency; this wrapper is not a fused jet.
    """
    geometry=field.geometry
    neighbors=options.pop("neighbors",None)
    compile_options={k:options.pop(k) for k in list(options) if k in ("kernel","precision","device","compact","method","nodes","preparation","search")}
    first=FieldEvaluator(field,**compile_options)
    if neighbors is None:
        q,h=queries_and_supports(geometry,queries,supports)
        neighbors=first.evaluator._neighbors(q,h,periods=options.get("periods"),exhaustive=options.get("exhaustive",False))
    def run(ds):
        owner=first if not ds else FieldEvaluator(field,derivative=ds,**{**compile_options,"search":"host"})
        return owner.evaluate(queries,supports,neighbors=neighbors,**options)
    result={"interpolation":run(())}
    d=geometry.dimension
    gradient=np.stack([run((j,)) for j in range(d)],axis=-1)
    result["gradient"]=gradient
    if field.shape==(d,):
        result["divergence"]=np.trace(gradient,axis1=-2,axis2=-1)
        result["curl"]=(gradient[:,1,0]-gradient[:,0,1] if d==2 else
                        np.stack([gradient[:,2,1]-gradient[:,1,2],gradient[:,0,2]-gradient[:,2,0],gradient[:,1,0]-gradient[:,0,1]],axis=-1))
    if hessian:
        H=np.stack([np.stack([run((i,j)) for j in range(d)],axis=-1) for i in range(d)],axis=-2)
        result["hessian"]=H
        result["laplacian"]=np.trace(H,axis1=-2,axis2=-1)
    return result
