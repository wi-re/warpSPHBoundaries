"""Supplied-affine fluid/material work, with a gradient-load transpose contract.

No velocity reconstruction or solver is installed here. The caller supplies L
in u(s)=u_i+L_i(s-x_i), then pulls the returned gradient load back through the
same reconstruction's transpose. P1–P3 skeleton tests reproduce position.
"""
from functools import lru_cache
import numpy as np
import warp as wp
from .fields import PolynomialField,FieldEvaluator
from .material import ResistanceMaterial,MaterialBatch,MaterialEvaluator,DragResult,_pack
from .wp_func import functions

wp.set_module_options({'enable_backward':False,'fuse_fp':False})


class AffineResistanceMaterial(ResistanceMaterial):
    def __init__(self,geometry,resistance_controls,skeleton_controls,*,resistance_degree=0,velocity_degree=1):
        if velocity_degree not in (1,2,3):
            raise ValueError('affine work requires P1–P3 skeleton/test bases reproducing position')
        super().__init__(geometry,resistance_controls,skeleton_controls,resistance_degree=resistance_degree,velocity_degree=velocity_degree)
        d=geometry.dimension
        position=PolynomialField(geometry,[{tuple(int(k==j) for k in range(d)):np.eye(d)[j] for j in range(d)} for _ in geometry.elements])
        D=self.resistance
        Du=D.product(self.skeleton,np.matmul)
        ZD=D.product(position,lambda A,z:A[:,:,None]*z)
        ZZD=ZD.product(position,lambda A,z:A[:,:,:,None]*z)
        ZDu=Du.product(position,lambda b,z:b[:,None]*z)
        NZD=self.test.product(ZD,lambda n,A:n[:,None,None,None]*A)
        self.field,self.affine_offsets=_pack([self.field,ZD,ZZD,ZDu,NZD])
        self.provenance.update(fluid_reconstruction='supplied affine gradient',gradient_transpose='caller supplied; required')

    def prepare(self,queries,supports,**kwargs):
        options={k:kwargs.pop(k) for k in list(kwargs) if k in ('kernel','precision','device','compact','method','nodes','preparation','search')}
        evaluator=FieldEvaluator(self.field,**options)
        return AffineMaterialBatch(self,evaluator.prepare(queries,supports,**kwargs))

    def compile(self,**options):
        return MaterialEvaluator(self,AffineMaterialBatch,**options)


@lru_cache(None)
def affine_functions(precision):
    api=functions(precision)
    real,vec=api.real,api.vec

    @wp.func
    def first(pairs:wp.array2d(dtype=real),row:int,i:int,k:int,j:int,d:int,delta:vec,ao:wp.array(dtype=int)):
        return pairs[row,ao[1]+(i*d+k)*d+j]-delta[j]*pairs[row,i*d+k]

    @wp.func
    def second(pairs:wp.array2d(dtype=real),row:int,i:int,k:int,j:int,l:int,d:int,delta:vec,ao:wp.array(dtype=int)):
        value=pairs[row,ao[2]+((i*d+k)*d+j)*d+l]
        value=value-delta[j]*pairs[row,ao[1]+(i*d+k)*d+l]
        value=value-delta[l]*pairs[row,ao[1]+(i*d+k)*d+j]
        return value+delta[j]*delta[l]*pairs[row,i*d+k]

    @wp.kernel
    def fluid(pairs:wp.array2d(dtype=real),offsets:wp.array(dtype=int),pe:wp.array(dtype=int),queries:wp.array(dtype=vec),
              shifts:wp.array(dtype=vec),origins:wp.array(dtype=vec),u:wp.array2d(dtype=real),L:wp.array3d(dtype=real),
              masses:wp.array(dtype=real),o:wp.array(dtype=int),ao:wp.array(dtype=int),forces:wp.array2d(dtype=real),
              gradient_load:wp.array3d(dtype=real),power:wp.array(dtype=real),rhs:wp.array2d(dtype=real)):
        qi=wp.tid()
        d=forces.shape[1]
        power[qi]=real(0.)
        for i in range(d):
            forces[qi,i]=real(0.)
            rhs[qi,i]=real(0.)
            for j in range(d):
                gradient_load[qi,i,j]=real(0.)
        for row in range(offsets[qi],offsets[qi+1]):
            delta=queries[row]
            energy=pairs[row,o[4]]
            for i in range(d):
                b=pairs[row,o[1]+i]
                force=-b
                correction=real(0.)
                for k in range(d):
                    force=force+pairs[row,i*d+k]*u[qi,k]
                    for j in range(d):
                        correction=correction+first(pairs,row,i,k,j,d,delta,ao)*L[qi,k,j]
                force=force+correction
                forces[qi,i]=forces[qi,i]-masses[qi]*force
                rhs[qi,i]=rhs[qi,i]+b-correction
                energy=energy+u[qi,i]*(force-b)
                for j in range(d):
                    c=pairs[row,ao[3]+i*d+j]-delta[j]*b
                    moment=-c
                    for k in range(d):
                        moment=moment+first(pairs,row,i,k,j,d,delta,ao)*u[qi,k]
                        for l in range(d):
                            moment=moment+second(pairs,row,i,k,j,l,d,delta,ao)*L[qi,k,l]
                    gradient_load[qi,i,j]=gradient_load[qi,i,j]-masses[qi]*moment
                    energy=energy+L[qi,i,j]*(moment-c)
            power[qi]=power[qi]+masses[qi]*energy

    @wp.kernel
    def reaction(pairs:wp.array2d(dtype=real),pq:wp.array(dtype=int),reverse:wp.array(dtype=int),offsets:wp.array(dtype=int),
                 queries:wp.array(dtype=vec),shifts:wp.array(dtype=vec),origins:wp.array(dtype=vec),u:wp.array2d(dtype=real),
                 L:wp.array3d(dtype=real),masses:wp.array(dtype=real),o:wp.array(dtype=int),ao:wp.array(dtype=int),
                 loads:wp.array3d(dtype=real)):
        ei,a,i=wp.tid()
        d=loads.shape[2]
        total=real(0.)
        for index in range(offsets[ei],offsets[ei+1]):
            row=reverse[index]
            qi=pq[row]
            delta=queries[row]
            value=-pairs[row,o[3]+a*d+i]
            for k in range(d):
                C=pairs[row,o[2]+(a*d+i)*d+k]
                value=value+C*u[qi,k]
                for j in range(d):
                    moment=pairs[row,ao[4]+((a*d+i)*d+k)*d+j]-delta[j]*C
                    value=value+moment*L[qi,k,j]
            total=total+masses[qi]*value
        loads[ei,a,i]=total
    return fluid,reaction


class AffineMaterialBatch(MaterialBatch):
    def __init__(self,material,batch):
        super().__init__(material,batch)
        ev=batch.evaluator
        d=material.geometry.dimension
        self.gradient_load=wp.empty((len(batch.queries),d,d),dtype=ev.api.real,device=ev.device)
        self.rhs=wp.empty(self.force.shape,dtype=ev.api.real,device=ev.device)
        relative=batch.queries[batch.neighbors.query]-batch.neighbors.shift-material.field.origins[batch.neighbors.element]
        self.relative=wp.array(np.pad(relative,((0,0),(0,3-d))),dtype=ev.api.vec,device=ev.device)
        self.o=wp.array(material.offsets.astype(np.int32),dtype=int,device=ev.device)
        self.ao=wp.array(material.affine_offsets.astype(np.int32),dtype=int,device=ev.device)
        self.fluid_kernel,self.reaction_kernel=affine_functions(ev.precision)

    def apply_device(self,velocity,masses,velocity_gradient):
        if velocity.shape!=self.force.shape or masses.shape!=(len(self.batch.queries),) or velocity_gradient.shape!=self.gradient_load.shape:
            raise ValueError('affine particle array shape mismatch')
        b=self.batch; ev=b.evaluator
        b.launch()
        if len(b.queries):
            wp.launch(self.fluid_kernel,len(b.queries),inputs=[b.pairs,b.offsets,b.pe,self.relative,b.shifts,b.owner.origins,velocity,velocity_gradient,
                masses,self.o,self.ao,self.force,self.gradient_load,self.power,self.rhs],device=ev.device)
        if np.prod(self.reaction.shape):
            wp.launch(self.reaction_kernel,self.reaction.shape,inputs=[b.pairs,b.pq,b.reverse,b.reverse_offsets,self.relative,b.shifts,b.owner.origins,
                velocity,velocity_gradient,masses,self.o,self.ao,self.reaction],device=ev.device)
        return self.force,self.gradient_load,self.reaction,self.power

    def apply(self,fluid_velocity,masses,velocity_gradient):
        ev=self.batch.evaluator; d=self.material.geometry.dimension; count=len(self.batch.queries)
        u=np.asarray(fluid_velocity,dtype=np.float64)
        L=np.asarray(velocity_gradient,dtype=np.float64)
        mass=np.broadcast_to(np.asarray(masses,dtype=np.float64),(count,))
        if u.shape!=(count,d) or L.shape!=(count,d,d) or not np.isfinite(u).all() or not np.isfinite(L).all() or not np.isfinite(mass).all() or np.any(mass<0):
            raise ValueError('finite velocities/gradients and nonnegative masses required')
        self.apply_device(wp.array(u,dtype=ev.api.real,device=ev.device),wp.array(mass,dtype=ev.api.real,device=ev.device),
                          wp.array(L,dtype=ev.api.real,device=ev.device))
        values=self.batch.numpy(); o=self.material.offsets
        return DragResult(self.force.numpy(),self.reaction.numpy(),float(self.power.numpy().sum()),values[:,:d*d].reshape(-1,d,d),
                          values[:,o[1]:o[2]],self.gradient_load.numpy(),self.rhs.numpy())


def pullback_gradient(result,transpose_action):
    """Chain rule f=base_force+G^T gradient_load for a supplied linear G.

    The supplied G must be the one used to obtain velocity_gradient. Particle
    masses are already included in the generalized loads; do not weight twice.
    """
    if result.gradient_load is None:
        raise ValueError('an affine result with a gradient load is required')
    transferred=np.asarray(transpose_action(result.gradient_load))
    if transferred.shape!=result.fluid_force.shape or not np.isfinite(transferred).all():
        raise ValueError('gradient transpose must return finite particle force arrays')
    return result.fluid_force+transferred
