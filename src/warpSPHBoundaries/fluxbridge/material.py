"""Polynomial resistance and work-consistent loads, independent of timestepping.

D has units 1/time; K has units length² and is deliberately a separate input
model. Polynomial K is not inverted coefficientwise. No pressure or porosity
solver is defined here.
"""
from functools import lru_cache
from dataclasses import dataclass
from copy import copy
import numpy as np
import warp as wp
from .fields import PolynomialField,FieldEvaluator
from .wp_func import functions


def _pack(fields):
    geometry=fields[0].geometry
    sizes=[int(np.prod(f.shape)) for f in fields]
    offsets=np.r_[0,np.cumsum(sizes)]
    out=[]
    for e in range(len(geometry.elements)):
        p={}
        for i,field in enumerate(fields):
            for a,c in field.coefficients[e].items():
                if a not in p:
                    p[a]=np.zeros(offsets[-1])
                p[a][offsets[i]:offsets[i+1]]=np.ravel(c)
        out.append(p)
    return PolynomialField(geometry,out),offsets


def check_psd(matrices):
    matrices=np.asarray(matrices,dtype=np.float64)
    if matrices.ndim<2 or matrices.shape[-1]!=matrices.shape[-2] or matrices.shape[-1] not in (2,3) or not np.isfinite(matrices).all():
        raise ValueError("finite 2x2 or 3x3 resistance tensors required")
    if not np.allclose(matrices,matrices.swapaxes(-1,-2),atol=1e-12,rtol=1e-12):
        raise ValueError("resistance must be symmetric")
    if np.any(np.linalg.eigvalsh(matrices)<0):
        raise ValueError("resistance must be positive semidefinite")
    return matrices


class ResistanceMaterial:
    """PSD Bernstein resistance, Lagrange skeleton velocity on straight simplices.

    Positive Bernstein control tensors imply pointwise PSD through P3. This
    guarantee does not extend to arbitrary higher-order Lagrange resistance.
    The immutable prepared field has explicit provenance and no fluid solver.
    """
    def __init__(self,geometry,resistance_controls,skeleton_controls,*,resistance_degree=0,velocity_degree=1):
        if geometry.measure!="volume":
            raise ValueError("bulk resistance requires volume elements; surface laws have different units")
        controls=check_psd(resistance_controls)
        d=geometry.dimension
        if controls.shape[-2:]!=(d,d):
            raise ValueError("resistance dimension mismatch")
        self.geometry=geometry
        self.resistance=PolynomialField.from_controls(geometry,controls,resistance_degree,"bernstein")
        self.skeleton=PolynomialField.from_controls(geometry,skeleton_controls,velocity_degree,"lagrange")
        if self.skeleton.shape!=(d,):
            raise ValueError("skeleton velocity must have one component per dimension")
        self.controls=np.array(skeleton_controls,dtype=np.float64,copy=True)
        self.test=PolynomialField.basis(geometry,velocity_degree,"lagrange")
        self.provenance=dict(contract=1,model="a=-D(u-us)",resistance_basis="Bernstein",resistance_degree=resistance_degree,
                             velocity_degree=velocity_degree,geometry=geometry.revision,porosity="not integrated",permeability="not inverted")
        D=self.resistance
        Du=D.product(self.skeleton,np.matmul)
        ND=self.test.product(D,lambda n,A:n[:,None,None]*A)
        NDu=self.test.product(Du,lambda n,b:n[:,None]*b)
        energy=self.skeleton.product(Du,np.dot)
        self.field,self.offsets=_pack([D,Du,ND,NDu,energy])

    def prepare(self,queries,supports,**kwargs):
        options={k:kwargs.pop(k) for k in list(kwargs) if k in ("kernel","precision","device","compact","method","nodes","preparation","search")}
        evaluator=FieldEvaluator(self.field,**options)
        return MaterialBatch(self,evaluator.prepare(queries,supports,**kwargs))

    def compile(self,**options):
        """Copy material fields and reuse static device preparation across queries.

        Build a new evaluator after changes to geometry/material or evaluation
        options. Each prepare() still rebuilds query adjacency, quadrature and
        convergence checks; no completed integrals are cached across queries.
        """
        return MaterialEvaluator(self,MaterialBatch,**options)


class MaterialEvaluator:
    """Explicit compiled material snapshot, independent of any fluid solver."""
    def __init__(self,material,batch_type,**options):
        self.material=copy(material)
        for name in ('field','test','resistance','skeleton'):
            field=getattr(material,name)
            setattr(self.material,name,PolynomialField(field.geometry,field.coefficients))
        for name in ('offsets','controls','affine_offsets'):
            if hasattr(material,name):
                array=np.array(getattr(material,name),copy=True);array.flags.writeable=False
                setattr(self.material,name,array)
        self.material.provenance=dict(material.provenance)
        self.evaluator=FieldEvaluator(self.material.field,**options)
        self.batch_type=batch_type

    def prepare(self,queries,supports,**options):
        return self.batch_type(self.material,self.evaluator.prepare(queries,supports,**options))

    def apply_streamed(self,queries,supports,fluid_velocity,masses,velocity_gradient=None,**options):
        from .streaming import apply_material_streamed
        return apply_material_streamed(self,queries,supports,fluid_velocity,masses,velocity_gradient,**options)


@dataclass
class DragResult:
    fluid_force: np.ndarray
    reaction: np.ndarray
    dissipation: float
    resistance: np.ndarray
    skeleton_product: np.ndarray
    gradient_load: np.ndarray | None = None
    affine_forcing: np.ndarray | None = None
    execution: dict | None = None

    def torque_defect(self,particle_positions,skeleton_nodes):
        """Torque bookkeeping for bases reproducing position (P1–P3).

        Integral drag is noncentral. A zero force balance does not imply a zero
        torque balance. Affine results include the gradient work's couple;
        pulling back that load through the supplied gradient transpose transfers
        it to particle forces. No corrective couple is invented.
        """
        q=np.asarray(particle_positions)
        nodes=np.asarray(skeleton_nodes)
        d=self.fluid_force.shape[-1]
        if q.shape!=self.fluid_force.shape or nodes.shape!=self.reaction.shape:
            raise ValueError("positions must match particle forces and nodal reactions")
        def cross(x,f):
            return x[...,0]*f[...,1]-x[...,1]*f[...,0] if d==2 else np.cross(x,f)
        total=np.sum(cross(q,self.fluid_force),axis=0)+np.sum(cross(nodes,self.reaction),axis=(0,1))
        if self.gradient_load is not None:
            M=self.gradient_load
            couple=(M[:,1,0]-M[:,0,1] if d==2 else
                    np.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1]],axis=-1))
            total=total+np.sum(couple,axis=0)
        return total


@lru_cache(None)
def material_functions(precision):
    real=functions(precision).real

    @wp.kernel
    def fluid(values: wp.array2d(dtype=real),velocity: wp.array2d(dtype=real),masses: wp.array(dtype=real),
              b_offset: int,energy_offset: int,forces: wp.array2d(dtype=real),power: wp.array(dtype=real)):
        i=wp.tid()
        d=forces.shape[1]
        energy=values[i,energy_offset]
        for j in range(d):
            force=real(0.0)
            for k in range(d):
                force=force+values[i,j*d+k]*velocity[i,k]
            energy=energy+velocity[i,j]*(force-real(2.0)*values[i,b_offset+j])
            forces[i,j]=-masses[i]*(force-values[i,b_offset+j])
        power[i]=masses[i]*energy

    @wp.kernel
    def reaction(pairs: wp.array2d(dtype=real),pq: wp.array(dtype=int),reverse: wp.array(dtype=int),offsets: wp.array(dtype=int),
                 velocity: wp.array2d(dtype=real),masses: wp.array(dtype=real),C_offset: int,c_offset: int,
                 loads: wp.array3d(dtype=real)):
        ei,a,j=wp.tid()
        d=loads.shape[2]
        total=real(0.0)
        for index in range(offsets[ei],offsets[ei+1]):
            row=reverse[index]
            qi=pq[row]
            force=-pairs[row,c_offset+a*d+j]
            for k in range(d):
                force=force+pairs[row,C_offset+a*d*d+j*d+k]*velocity[qi,k]
            total=total+masses[qi]*force
        loads[ei,a,j]=total
    return fluid,reaction


class MaterialBatch:
    def __init__(self,material,batch):
        self.material=material
        self.batch=batch
        ev=batch.evaluator
        real=ev.api.real
        d=material.geometry.dimension
        self.force=wp.empty((len(batch.queries),d),dtype=real,device=ev.device)
        self.reaction=wp.empty((len(material.geometry.elements),material.test.shape[0],d),dtype=real,device=ev.device)
        self.power=wp.empty(len(batch.queries),dtype=real,device=ev.device)
        self.fluid_kernel,self.reaction_kernel=material_functions(ev.precision)

    @property
    def bytes(self):
        """Owned field/rule/adjacency and material-action arrays, not static plans.

        Compiled coefficients/BVH, comparison temporaries and allocator scratch
        are separate. The affine subclass's extra action arrays are included.
        """
        names=('force','reaction','power','gradient_load','rhs','relative','o','ao')
        return self.batch.bytes+sum(getattr(self,name).capacity for name in names if hasattr(self,name))

    def apply_device(self,velocity,masses):
        """Allocation-free composed consumer; particle arrays are caller owned."""
        ev=self.batch.evaluator
        if velocity.shape!=self.force.shape or masses.shape!=(len(self.batch.queries),):
            raise ValueError("particle array shape mismatch")
        self.batch.launch()
        o=self.material.offsets
        if len(self.batch.queries):
            wp.launch(self.fluid_kernel,len(self.batch.queries),inputs=[self.batch.values,velocity,masses,int(o[1]),int(o[4]),self.force,self.power],device=ev.device)
        if np.prod(self.reaction.shape):
            wp.launch(self.reaction_kernel,self.reaction.shape,inputs=[self.batch.pairs,self.batch.pq,self.batch.reverse,self.batch.reverse_offsets,
                velocity,masses,int(o[2]),int(o[3]),self.reaction],device=ev.device)
        return self.force,self.reaction,self.power

    def apply(self,fluid_velocity,masses):
        d=self.material.geometry.dimension
        qcount=len(self.batch.queries)
        u=np.asarray(fluid_velocity,dtype=np.float64)
        mass=np.broadcast_to(np.asarray(masses,dtype=np.float64),(qcount,))
        if u.shape!=(qcount,d) or not np.isfinite(u).all() or not np.isfinite(mass).all() or np.any(mass<0):
            raise ValueError("finite particle velocities and nonnegative masses required")
        ev=self.batch.evaluator
        velocity=wp.array(u,dtype=ev.api.real,device=ev.device)
        weights=wp.array(mass,dtype=ev.api.real,device=ev.device)
        self.apply_device(velocity,weights)
        values=self.batch.numpy()
        o=self.material.offsets
        A=values[:,o[0]:o[1]].reshape(-1,d,d)
        b=values[:,o[1]:o[2]]
        return DragResult(self.force.numpy(),self.reaction.numpy(),float(self.power.numpy().sum()),A,b)


@lru_cache(None)
def response_function(precision):
    api=functions(precision)
    real,vec=api.real,api.vec
    mat=wp.types.matrix(shape=(3,3),dtype=real)

    @wp.func
    def mobility(D: mat,dt: real):
        return wp.inverse(wp.identity(3,dtype=real)+dt*D)

    @wp.kernel
    def response(D: wp.array(dtype=mat),u: wp.array(dtype=vec),b: wp.array(dtype=vec),dt: real,
                 out: wp.array(dtype=vec),matrices: wp.array(dtype=mat)):
        i=wp.tid()
        B=mobility(D[i],dt)
        out[i]=B*(u[i]+dt*b[i])
        matrices[i]=B
    return mobility,response,mat


def local_response(resistance,velocity,forcing,dt,*,precision="float64",device="cpu"):
    """Apply (I+dt D)^-1(u+dt forcing), returning velocity and mobility.

    forcing can include ∫D us W and a supplied acceleration. This is a local
    algebra action, not a split porous-flow solver or pressure projection.
    """
    D=check_psd(resistance)
    u=np.asarray(velocity,dtype=np.float64)
    b=np.asarray(forcing,dtype=np.float64)
    d=D.shape[-1]
    if D.ndim!=3 or u.shape!=D.shape[:1]+(d,) or b.shape!=u.shape or not np.isfinite(u).all() or not np.isfinite(b).all() or not np.isfinite(dt) or dt<0:
        raise ValueError("consistent finite batches and nonnegative dt required")
    api=functions(precision)
    _,kernel,mat=response_function(precision)
    padded=np.zeros((len(D),3,3)); padded[:,:d,:d]=D
    pad=lambda v: np.pad(v,((0,0),(0,3-d)))
    arrays=[wp.array(padded,dtype=mat,device=device),wp.array(pad(u),dtype=api.vec,device=device),wp.array(pad(b),dtype=api.vec,device=device)]
    out=wp.empty(len(D),dtype=api.vec,device=device); matrices=wp.empty(len(D),dtype=mat,device=device)
    if len(D):
        wp.launch(kernel,len(D),inputs=arrays+[api.real(dt),out,matrices],device=device)
    return out.numpy()[:,:d],matrices.numpy()[:,:d,:d]
