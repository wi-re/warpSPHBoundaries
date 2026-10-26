"""Explicit bounded host drivers around reusable native CPU/CUDA providers.

Each chunk prepares, evaluates and reads back one ordinary batch, then releases
it before yielding. Kernel methods/precision and resource/convergence policies
are unchanged. This driver allocates and synchronizes; frozen batch launches
remain the separate graph-compatible interface.
"""
from dataclasses import dataclass
from math import fsum
from time import perf_counter
import numpy as np
from .geometry import Adjacency,fingerprint,queries_and_supports


@dataclass(frozen=True)
class EvaluationChunk:
    start: int
    stop: int
    values: np.ndarray
    metrics: dict


def _evaluator(provider):
    ev=provider
    while not hasattr(ev,'geometry') and hasattr(ev,'evaluator'):ev=ev.evaluator
    if not hasattr(ev,'geometry'):raise TypeError('a compiled native provider is required')
    return ev


def _size(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 1<=value<=2**31-1:
        raise ValueError('chunk_size must be a positive integer within int32 range')
    return int(value)


class _Queries:
    def __init__(self,ev,queries,supports,neighbors,periods):
        self.geometry=ev.geometry;self.revision=self.geometry.revision
        q,h=queries_and_supports(self.geometry,queries,supports)
        self.q=q.copy();self.h=h.copy()
        self.q.flags.writeable=False;self.h.flags.writeable=False
        d=self.geometry.dimension
        periodic=tuple(periods) if periods is not None else (neighbors.periodic if neighbors is not None else (0.,)*d)
        if len(periodic)!=d or any(not np.isfinite(v) or v<0 for v in periodic):
            raise ValueError('periods must be finite nonnegative lengths; zero is nonperiodic')
        if neighbors is not None:
            neighbors.validate(self.geometry,self.q,self.h)
            if periodic!=neighbors.periodic:raise ValueError('adjacency periodicity does not match request')
        self.neighbors=neighbors;self.periods=periodic

    def options(self,start,stop,exhaustive):
        if self.geometry.revision!=self.revision:
            raise ValueError('geometry changed during streaming; recompile the provider')
        sliced=None
        if self.neighbors is not None:
            a=self.neighbors;lo=int(a.offsets[start]);hi=int(a.offsets[stop])
            arrays=[a.query[lo:hi].copy()-start,a.element[lo:hi].copy(),a.shift[lo:hi].copy(),a.offsets[start:stop+1].copy()-lo]
            for array in arrays:array.flags.writeable=False
            sliced=Adjacency(*arrays,fingerprint(self.q[start:stop],self.h[start:stop]),self.revision,a.periodic)
        return dict(neighbors=sliced,periods=self.periods,exhaustive=exhaustive)


def _dtype(ev):return np.float32 if ev.precision.startswith('float32') else np.float64


def _shape(provider,ev):
    if hasattr(provider,'field'):return provider.field.shape
    return (len(ev.host_plan.requests) if hasattr(ev,'host_plan') else len(provider.requests),)


def _batch_metrics(batch,start,stop,elapsed):
    return dict(start=start,stop=stop,interactions=batch.count,owned_batch_bytes=batch.bytes,
        elapsed_ms=elapsed,quadrature=dict(batch.quadrature) if hasattr(batch,'quadrature') else None)


def _execution(ev,n,chunk_size):
    return dict(query_count=n,chunk_size=chunk_size,chunks=0,total_interactions=0,
        peak_owned_batch_bytes=0,method=getattr(ev,'method','disk_quadrature'),precision=ev.precision,
        requested_nodes=getattr(ev,'nodes',None),quadrature_orders={},quadrature_interactions_by_order={},
        quadrature_promotions=0,field_promotions=0,accepted_rule_samples=0,comparison_rule_samples=0,
        device=str(ev.device),search=getattr(ev,'search','host'),graph_compatible=False,
        static_search_array_bytes=0 if getattr(ev,'device_search',None) is None else ev.device_search.array_bytes,
        memory_scope='maximum accepted batch-owned arrays; static plans/coefficients/BVH, comparison temporaries and allocator reservation are separate',
        execution='explicit host chunk preparation, native evaluation and readback; no method/precision fallback')


def _record(execution,metrics):
    execution['chunks']+=1
    execution['total_interactions']+=metrics['interactions']
    execution['peak_owned_batch_bytes']=max(execution['peak_owned_batch_bytes'],metrics['owned_batch_bytes'])
    rule=metrics['quadrature']
    if rule is not None and metrics['interactions']:
        order=str(rule['nodes'])
        execution['quadrature_orders'][order]=execution['quadrature_orders'].get(order,0)+1
        execution['quadrature_interactions_by_order'][order]=execution['quadrature_interactions_by_order'].get(order,0)+metrics['interactions']
        execution['quadrature_promotions']+=rule['promotions']
        execution['field_promotions']+=rule.get('field_promotions',0)
        execution['accepted_rule_samples']+=rule['samples']
        execution['comparison_rule_samples']+=rule.get('comparison_sample_count',0)


def _evaluation_chunks(provider,state,chunk_size,exhaustive):
    ev=_evaluator(provider)
    for start in range(0,len(state.q),chunk_size):
        stop=min(start+chunk_size,len(state.q));began=perf_counter()
        try:
            batch=provider.prepare(state.q[start:stop],state.h[start:stop],**state.options(start,stop,exhaustive))
            batch.launch()
            # CPU .numpy() may be a view: copy so a yielded value cannot retain
            # the native batch and its much larger rule/profile allocations.
            value=np.array(batch.numpy(),copy=True).reshape((stop-start,)+_shape(provider,ev))
            metrics=_batch_metrics(batch,start,stop,1000*(perf_counter()-began))
            del batch
        except ValueError as error:
            raise ValueError(f'stream chunk [{start}:{stop}] failed: {error}') from error
        yield EvaluationChunk(start,stop,value,metrics)


def iter_evaluate(provider,queries,supports,*,chunk_size=32,neighbors=None,periods=None,exhaustive=False):
    """Snapshot queries immediately; yield owned NumPy values without native caches."""
    size=_size(chunk_size);state=_Queries(_evaluator(provider),queries,supports,neighbors,periods)
    return _evaluation_chunks(provider,state,size,exhaustive)


def evaluate_streamed(provider,queries,supports,*,chunk_size=32,neighbors=None,periods=None,exhaustive=False,return_metrics=False):
    """Collect streamed values; use iter_evaluate to avoid a full host output."""
    began=perf_counter();size=_size(chunk_size);ev=_evaluator(provider)
    state=_Queries(ev,queries,supports,neighbors,periods)
    values=np.empty((len(state.q),)+_shape(provider,ev),dtype=_dtype(ev))
    execution=_execution(ev,len(state.q),size)
    for chunk in _evaluation_chunks(provider,state,size,exhaustive):
        values[chunk.start:chunk.stop]=chunk.values;_record(execution,chunk.metrics)
    execution.update(elapsed_ms=1000*(perf_counter()-began),host_output_bytes=values.nbytes,
        host_query_snapshot_bytes=state.q.nbytes+state.h.nbytes)
    return (values,execution) if return_metrics else values


def apply_material_streamed(provider,queries,supports,fluid_velocity,masses,velocity_gradient=None,*,
                            chunk_size=32,neighbors=None,periods=None,exhaustive=False):
    """Bound native rule storage and accumulate reactions in deterministic chunks.

    Evaluation uses the selected native dtype. Host reaction/energy aggregation
    uses fp64; reactions round once to the selected output dtype. No pressure
    solve or timestep loop is introduced.
    """
    from .material import DragResult
    from .affine import AffineResistanceMaterial
    began=perf_counter();size=_size(chunk_size);ev=_evaluator(provider)
    state=_Queries(ev,queries,supports,neighbors,periods);n=len(state.q);d=ev.geometry.dimension
    affine=isinstance(provider.material,AffineResistanceMaterial)
    velocity=np.array(fluid_velocity,dtype=np.float64,copy=True)
    mass=np.broadcast_to(np.asarray(masses,dtype=np.float64),(n,)).copy()
    gradient=None if velocity_gradient is None else np.array(velocity_gradient,dtype=np.float64,copy=True)
    if velocity.shape!=(n,d) or not np.isfinite(velocity).all() or not np.isfinite(mass).all() or np.any(mass<0):
        raise ValueError('finite velocities and nonnegative masses with matching query shapes required')
    if affine and (gradient is None or gradient.shape!=(n,d,d) or not np.isfinite(gradient).all()):
        raise ValueError('supplied-affine streaming requires a finite matching velocity_gradient')
    if not affine and gradient is not None:raise ValueError('velocity_gradient requires a supplied-affine material')
    dtype=_dtype(ev)
    for array in [velocity,mass]+([gradient] if affine else []):
        with np.errstate(over='ignore',under='ignore'):
            finite=np.isfinite(array.astype(dtype)).all()
        if not finite:raise ValueError('streamed physical inputs exceed finite evaluation storage range')
    force=np.empty((n,d),dtype=dtype);resistance=np.empty((n,d,d),dtype=dtype);skeleton=np.empty((n,d),dtype=dtype)
    reaction=np.zeros((len(ev.geometry.elements),provider.material.test.shape[0],d),dtype=np.float64)
    gradient_load=np.empty((n,d,d),dtype=dtype) if affine else None
    forcing=np.empty((n,d),dtype=dtype) if affine else None
    energy=[];execution=_execution(ev,n,size)
    for start in range(0,n,size):
        stop=min(start+size,n);chunk_began=perf_counter()
        try:
            batch=provider.prepare(state.q[start:stop],state.h[start:stop],**state.options(start,stop,exhaustive))
            result=(batch.apply(velocity[start:stop],mass[start:stop],gradient[start:stop]) if affine else
                batch.apply(velocity[start:stop],mass[start:stop]))
            force[start:stop]=result.fluid_force;resistance[start:stop]=result.resistance
            skeleton[start:stop]=result.skeleton_product
            reaction+=result.reaction.astype(np.float64);energy.append(result.dissipation)
            if affine:gradient_load[start:stop]=result.gradient_load;forcing[start:stop]=result.affine_forcing
            metrics=_batch_metrics(batch.batch,start,stop,1000*(perf_counter()-chunk_began))
            metrics['owned_batch_bytes']=batch.bytes
            _record(execution,metrics)
            del result,batch
        except ValueError as error:
            raise ValueError(f'stream chunk [{start}:{stop}] failed: {error}') from error
    reaction_output=reaction.astype(dtype)
    outputs=[force,reaction_output,resistance,skeleton]+([gradient_load,forcing] if affine else [])
    execution.update(elapsed_ms=1000*(perf_counter()-began),host_output_bytes=sum(a.nbytes for a in outputs),
        host_reaction_accumulator_bytes=reaction.nbytes,
        host_query_snapshot_bytes=state.q.nbytes+state.h.nbytes+velocity.nbytes+mass.nbytes+(0 if gradient is None else gradient.nbytes),
        aggregation='deterministic fp64 host reaction/energy accumulation; reactions round once to the selected dtype')
    return DragResult(force,reaction_output,fsum(energy),resistance,skeleton,gradient_load,forcing,execution)
