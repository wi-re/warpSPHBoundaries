"""Standalone solid-simplex adjacency, evaluation and deterministic CSR assembly.

This is a library bridge for static, nonoverlapping volume meshes, not a fluid
solver or BoundaryProvider adapter. Host input/output preparation is explicit.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from .geometry import Geometry, adjacency, queries_and_supports
from .solid import solid_functions, prepare_solid_geometry


@lru_cache(None)
def _assembly(precision):
    real=wp.float32 if precision=='float32' else wp.float64
    @wp.kernel(module='unique',module_options={'enable_backward':False})
    def assemble(pair:wp.array2d(dtype=real), offsets:wp.array(dtype=wp.int64),out:wp.array2d(dtype=real)):
        channel,query=wp.tid()
        total=real(0)
        i=offsets[query]
        while i<offsets[query+1]:
            total+=pair[channel,i]
            i+=wp.int64(1)
        out[channel,query]=total
    return assemble


class SolidEvaluator:
    """C2/C4 solid bundles with reusable static geometry and whole-element search.

    ``prepare`` freezes queries/supports and returns a reusable batch with
    ``evaluate()``. The result holds device ``pairs`` and ``aggregate`` arrays in
    [channel,pair/query] order. Search never removes individual simplex faces.
    Overlapping solids are summed, not unioned; represent a solid with disjoint
    mesh interiors. Rebuild this evaluator when geometry changes.
    """
    def __init__(self,vertices,*,device='cpu',kernel='w2',precision='float64',bundle='wcsph',conditioning=None,registers=None,search='host'):
        self.api=solid_functions(np.shape(vertices)[-1],kernel,precision,bundle,registers,conditioning)
        self.device=wp.get_device(device)
        v=np.asarray(vertices,dtype=np.float64)
        if precision=='float32':v=v.astype(np.float32).astype(np.float64)
        self.prepared=prepare_solid_geometry(v,self.device,precision,self.api.conditioning)
        self.geometry=Geometry(v,self.api.dimension)
        if search not in ('host','device'):raise ValueError('search must be host or device')
        self.search=None
        if search=='device':
            from .device_search import DeviceSearch
            self.search=DeviceSearch(self.geometry,device=self.device)

    def prepare(self,queries,supports,*,periods=None,adj=None,exhaustive=False):
        q,h=queries_and_supports(self.geometry,queries,supports)
        if self.api.precision=='float32':
            with np.errstate(over='ignore'):q=q.astype(np.float32).astype(float);h=h.astype(np.float32).astype(float)
            q,h=queries_and_supports(self.geometry,q,h)
        if adj is not None:
            adj.validate(self.geometry,q,h)
            if periods is not None and tuple(periods)!=adj.periodic:raise ValueError('adjacency periodicity differs')
        elif self.search is not None and not exhaustive:adj=self.search.prepare(q,h,periods=periods)
        else:adj=adjacency(self.geometry,q,h,periods=periods,exhaustive=exhaustive)
        d=self.api.dimension;count=len(adj.query);api=self.api
        positions=wp.array(np.pad(q[adj.query]-adj.shift,((0,0),(0,3-d))),dtype=api.vec,device=self.device)
        radii=wp.array(h[adj.query],dtype=api.real,device=self.device)
        indices=wp.array(adj.element,dtype=int,device=self.device)
        offsets=wp.array(adj.offsets,dtype=wp.int64,device=self.device)
        pairs=wp.empty((api.channels,count),dtype=api.real,device=self.device)
        aggregate=wp.empty((api.channels,len(q)),dtype=api.real,device=self.device)
        def evaluate():
            if count:wp.launch(api.kernel,count,inputs=[positions,radii,indices,self.prepared,pairs],device=self.device)
            if len(q):wp.launch(_assembly(api.precision),(api.channels,len(q)),inputs=[pairs,offsets,aggregate],device=self.device)
            return batch
        batch=SimpleNamespace(evaluate=evaluate,pairs=pairs,aggregate=aggregate,adjacency=adj,labels=api.labels,
            pair_count=count,query_count=len(q),array_bytes=sum(a.capacity for a in (positions,radii,indices,offsets,pairs,aggregate)))
        return batch
