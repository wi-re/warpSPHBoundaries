"""Static geometry and conservative whole-element support search.

All facets of each selected volume are evaluated. No regular-flux face is pruned.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import numpy as np


def fingerprint(*arrays):
    h=sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str((a.shape,a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Geometry:
    def __init__(self, elements, dimension, measure="volume"):
        self.dimension=dimension
        self.measure=measure
        self.elements=tuple(np.array(v,dtype=np.float64,copy=True) for v in elements)
        facets=[]; starts=[0]; bounds=[]
        if dimension not in (2,3) or measure not in ("volume","surface"):
            raise ValueError("invalid geometry dimension/measure")
        for vertices in self.elements:
            if vertices.ndim!=2 or vertices.shape[1]!=dimension or not np.isfinite(vertices).all():
                raise ValueError("finite vertices with consistent dimension required")
            if dimension==2:
                if len(vertices)<3 or measure!="volume":
                    raise ValueError("2D elements must be polygons")
                a=vertices-vertices[0]
                area=np.sum(a[:,0]*np.roll(a[:,1],-1)-a[:,1]*np.roll(a[:,0],-1))
                if area==0:
                    raise ValueError("degenerate polygon")
                oriented=vertices if area>0 else vertices[::-1]
                pairs=zip(oriented,np.roll(oriented,-1,axis=0))
                for p,q in pairs:
                    if np.array_equal(p,q):
                        raise ValueError("duplicate edge endpoints")
                    facets.append(np.pad(np.array([p,q,q]),((0,0),(0,1))))
            elif measure=="surface":
                if vertices.shape!=(3,3) or np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]))==0:
                    raise ValueError("surface element must be a nondegenerate triangle")
                facets.append(vertices)
            else:
                if vertices.shape!=(4,3):
                    raise ValueError("volume elements must be tetrahedra; use closed_surface for a skin")
                for i in range(4):
                    f=[j for j in range(4) if j!=i]
                    p,q,r=vertices[f]
                    side=np.dot(np.cross(q-p,r-p),vertices[i]-p)
                    if side==0:
                        raise ValueError("degenerate tetrahedron")
                    if side>0:
                        f[1],f[2]=f[2],f[1]
                    facets.append(vertices[f])
            starts.append(len(facets))
            bounds.append([vertices.min(axis=0),vertices.max(axis=0)])
        self._finish(facets,starts,bounds)

    def _finish(self,facets,starts,bounds):
        self.facets=np.array(facets,dtype=np.float64).reshape(-1,3,3)
        self.starts=np.array(starts,dtype=np.int32)
        self.bounds=np.array(bounds,dtype=np.float64).reshape(-1,2,self.dimension)
        self.revision=fingerprint(self.facets,self.starts)
        self.condition=np.array([np.linalg.cond((v[1:]-v[0]).T) for v in self.elements])
        for a in (self.facets,self.starts,self.bounds,*self.elements):
            a.flags.writeable=False
        self.tree=self._tree(np.arange(len(self.bounds))) if len(self.bounds) else None

    @classmethod
    def closed_surface(cls,vertices,faces):
        # Exact incidence and orientation check belongs in static preparation.
        from ..mesh3d.reference import ClosedSurface
        ref=ClosedSurface(np.asarray(vertices,dtype=np.float64).tolist(),np.asarray(faces,dtype=np.int64).tolist())
        obj=cls.__new__(cls)
        obj.dimension=3; obj.measure="volume"
        v=np.array(vertices,dtype=np.float64)
        obj.elements=(v,)
        obj._finish([v[list(f)] for f in ref.indices],[0,len(ref.indices)],[[v.min(axis=0),v.max(axis=0)]])
        return obj

    def _tree(self,ids):
        lo=self.bounds[ids,0].min(axis=0)
        hi=self.bounds[ids,1].max(axis=0)
        if len(ids)<=8:
            return (lo,hi,None,None,ids)
        axis=np.argmax(hi-lo)
        centers=(self.bounds[ids,0,axis]+self.bounds[ids,1,axis])/2
        ordered=ids[np.argsort(centers,kind="stable")]
        mid=len(ordered)//2
        return (lo,hi,self._tree(ordered[:mid]),self._tree(ordered[mid:]),None)

    def nearby(self,query,h):
        found=[]
        def visit(node):
            if node is None:
                return
            lo,hi,left,right,ids=node
            if np.any(hi<query-h) or np.any(lo>query+h):
                return
            if ids is not None:
                for i in ids:
                    # AABB distance is a conservative ball/element test.
                    delta=np.maximum(np.maximum(self.bounds[i,0]-query,query-self.bounds[i,1]),0)
                    if np.dot(delta,delta)<=h*h:
                        found.append(int(i))
            else:
                visit(left); visit(right)
        visit(self.tree)
        return sorted(found)


def queries_and_supports(geometry,queries,supports):
    q=np.asarray(queries,dtype=np.float64)
    if q.size==0:
        q=q.reshape(0,geometry.dimension)
    h=np.broadcast_to(np.asarray(supports,dtype=np.float64),(len(q),)).copy()
    if q.shape!=(len(q),geometry.dimension) or not np.isfinite(q).all() or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError("finite queries and positive supports required")
    return q,h


@dataclass(frozen=True)
class Adjacency:
    query: np.ndarray
    element: np.ndarray
    shift: np.ndarray
    offsets: np.ndarray
    signature: str
    geometry_revision: str
    periodic: tuple

    def validate(self,geometry,queries,supports):
        q,h=queries_and_supports(geometry,queries,supports)
        if self.geometry_revision!=geometry.revision or self.signature!=fingerprint(q,h):
            raise ValueError("stale adjacency: geometry, query or support changed")


def adjacency(geometry,queries,supports,*,exhaustive=False,periods=None):
    q,h=queries_and_supports(geometry,queries,supports)
    d=geometry.dimension
    periodic=tuple(periods) if periods is not None else (0.,)*d
    if len(periodic)!=d or any(not np.isfinite(L) or L<0 for L in periodic):
        raise ValueError("periods must be finite nonnegative lengths; zero is nonperiodic")
    qs=[]; es=[]; shifts=[]; offsets=[0]
    for qi,(x,radius) in enumerate(zip(q,h)):
        ranges=[]
        for j,L in enumerate(periodic):
            if L and len(geometry.bounds):
                low=geometry.bounds[:,0,j].min(); high=geometry.bounds[:,1,j].max()
                ranges.append(range(int(np.ceil((x[j]-radius-high)/L)),int(np.floor((x[j]+radius-low)/L))+1))
            else:
                ranges.append((0,))
        pairs=[]
        for image in product(*ranges):
            shift=np.array(image)*periodic
            ids=range(len(geometry.elements)) if exhaustive else geometry.nearby(x-shift,radius)
            pairs.extend((ei,tuple(shift)) for ei in ids)
        # Each (element, image) occurs once, in deterministic order.
        for ei,shift in sorted(pairs):
            qs.append(qi); es.append(ei); shifts.append(shift)
        offsets.append(len(qs))
    arrays=[np.array(qs,dtype=np.int32),np.array(es,dtype=np.int32),np.array(shifts,dtype=np.float64).reshape(-1,d),np.array(offsets,dtype=np.int32)]
    for a in arrays:
        a.flags.writeable=False
    return Adjacency(*arrays,fingerprint(q,h),geometry.revision,periodic)
