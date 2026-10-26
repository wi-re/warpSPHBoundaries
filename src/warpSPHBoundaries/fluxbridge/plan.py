"""Exact rational preparation of a finite analytic reduction DAG.

All recurrences are algebraic. No quadrature nodes enter a native plan.
"""
from fractions import Fraction as F
import numpy as np
from ..edge.kernels import KERNELS as BLOCKS
from ..curvbound.kernels import KERNELS as SHAPES
from .contract import Request, KERNELS


def lower(a, i):
    return tuple(t-int(k==i) for k,t in enumerate(a))


def higher(a, i):
    return tuple(t+int(k==i) for k,t in enumerate(a))


class AnalyticPlan:
    def __init__(self, dimension, requests, kernel="w2", measure="volume", compact=False):
        if dimension not in (2,3) or kernel not in KERNELS or measure not in ("volume","surface"):
            raise ValueError("unsupported dimension, kernel or measure")
        if dimension == 2 and measure == "surface":
            raise ValueError("embedded surface contract requires 3D")
        self.compact=bool(compact)
        self.dimension, self.kernel, self.measure = dimension, kernel, measure
        self.requests = tuple(requests)
        if any(not isinstance(r,Request) or len(r.alpha)!=dimension for r in self.requests):
            raise ValueError("request dimension mismatch")
        self.zero=(0,)*dimension
        self.face_nodes=[]; self.face_ids={}
        self.boundaries=[]; self.boundary_ids={}
        self.volumes=[]; self.volume_ids={}
        self.outputs=[]
        ratio=F(1) if dimension==2 else SHAPES[kernel].c3_pi/SHAPES[kernel].c2_pi
        self.profile=tuple((n,b.R,c*ratio) for b in BLOCKS[kernel].blocks for n,c in enumerate(b.coeffs) if c)
        for req in self.requests:
            terms={}
            for n,R,c in self.profile:
                for key,value in self.operator(req.alpha,req.derivative,n,R).items():
                    terms[key]=terms.get(key,F(0))+c*value
            self.outputs.append({k:v for k,v in terms.items() if v})

    def face(self,a,n,R):
        key=(a,n,R)
        if key not in self.face_ids:
            deps=[]; beta=a; i=-1; base=-1
            if any(a) and self.dimension==3:
                i=next(j for j,t in enumerate(a) if t)
                beta=lower(a,i)
                base=self.face(beta,n,R)
                for j,t in enumerate(beta):
                    if t:
                        b=lower(beta,j)
                        deps.append((j,t,self.face(b,n+2,R),self.face(b,0,R)))
            idx=len(self.face_nodes)
            self.face_ids[key]=idx
            self.face_nodes.append((a,n,R,i,beta,base,deps))
        return self.face_ids[key]

    def boundary(self,a,n,R,axis):
        # axis=-1 denotes complete regular radial flux, retaining exterior tail.
        key=(a,n,R,axis)
        if key not in self.boundary_ids:
            s=self.face(a,n,R)
            ext=clip=-1
            if axis==-1:
                if not self.compact:
                    ext=self.face(self.zero,-self.dimension,F(-1))
                clip=self.face(self.zero,-self.dimension,R)
            self.boundary_ids[key]=len(self.boundaries)
            self.boundaries.append((axis,n,R,s,ext,clip))
        return self.boundary_ids[key]

    def volume(self,a,n,R):
        key=(a,n,R)
        if key not in self.volume_ids:
            if not any(a):
                row=(-1,self.boundary(a,n,R,-1),-1,-1,-1,0,n,R)
            else:
                i=next(j for j,t in enumerate(a) if t)
                beta=lower(a,i)
                b1=self.boundary(beta,n+2,R,i)
                b0=self.boundary(beta,0,R,i)
                v1=v0=-1
                if beta[i]:
                    v1=self.volume(lower(beta,i),n+2,R)
                    v0=self.volume(lower(beta,i),0,R)
                row=(i,b1,b0,v1,v0,beta[i],n,R)
            self.volume_ids[key]=len(self.volumes)
            self.volumes.append(row)
        return self.volume_ids[key]

    def moment(self,a,n,R):
        return {(0 if self.measure=="surface" else 1,
                 self.face(a,n,R) if self.measure=="surface" else self.volume(a,n,R)): F(1)}

    @staticmethod
    def add(out, src, factor):
        for k,c in src.items():
            out[k]=out.get(k,F(0))+factor*c

    def operator(self,a,ds,n,R):
        if not ds:
            return self.moment(a,n,R)
        out={}
        if self.measure=="surface":
            if len(ds)==1:
                if n:
                    self.add(out,self.moment(higher(a,ds[0]),n-2,R),-n)
            else:
                i,j=ds
                if n and i==j:
                    self.add(out,self.moment(a,n-2,R),n)
                if n and n!=2:
                    self.add(out,self.moment(higher(higher(a,i),j),n-4,R),n*(n-2))
            return out
        if len(ds)==1:
            j=ds[0]
            out[(2,self.boundary(a,n,R,j))]=F(-1)
            if a[j]:
                self.add(out,self.moment(lower(a,j),n,R),a[j])
        else:
            i,j=ds
            if n:
                out[(2,self.boundary(higher(a,j),n-2,R,i))]=F(n)
            if a[i]:
                self.add(out,self.operator(lower(a,i),(j,),n,R),a[i])
        return out

    def arrays(self):
        """Dimension-padded integer metadata; constants remain rational until here."""
        f=[]; dep=[]
        for a,n,R,i,beta,base,deps in self.face_nodes:
            f.append([n,i,base]+list(a)+(3-self.dimension)*[0]+list(beta)+(3-self.dimension)*[0])
            dep.append([x for j in range(3) for x in next(((t,s1,s0) for jj,t,s1,s0 in deps if jj==j),(0,0,0))])
        b=[[axis,n,s,ext,clip] for axis,n,R,s,ext,clip in self.boundaries]
        v=[list(row[:7]) for row in self.volumes]
        terms=[]; offsets=[0]
        for out in self.outputs:
            terms.extend((kind,index,float(c)/np.pi) for (kind,index),c in out.items())
            offsets.append(len(terms))
        return dict(f=np.array(f,dtype=np.int32).reshape(-1,9),dep=np.array(dep,dtype=np.int32).reshape(-1,9),
                    fr=np.array([float(r[2]) for r in self.face_nodes]),
                    b=np.array(b,dtype=np.int32).reshape(-1,5),br=np.array([float(r[2]) for r in self.boundaries]),
                    v=np.array(v,dtype=np.int32).reshape(-1,7),vr=np.array([float(r[7]) for r in self.volumes]),
                    terms=np.array(terms).reshape(-1,3),offsets=np.array(offsets,dtype=np.int32))
