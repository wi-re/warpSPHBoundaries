"""Bounded-state geometry and caller-output experiments for the W6/J-K study.

The earlier measured kernels are kept unchanged. This module tests vertex
reloads, a packed geometry allocation, occupancy-only evaluation, and streaming
any number of P1 scalar/vector/material components through four shared moments.
There is no solver coupling and no global moment or primitive workspace.
"""
from functools import lru_cache
import re
from types import SimpleNamespace

import numpy as np
import warp as wp

import fluxbridge_shared_jk as baseline
import fluxbridge_shared_jk_registers as contracted

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})
OFFSETS=dict(vertices=0,normals=4,tangents=8,outward=14,gradients=26)


def source(precision='float64',layout='packed',reload=True,operator='p1'):
    if layout not in ('separate','packed') or operator not in ('p1','occupancy','fields'):
        raise ValueError('unknown layout or operator')
    code=(contracted.source(precision) if operator=='p1' else baseline.source(precision))
    if operator=='occupancy':
        code=code.replace('T m0=0; V m1{0,0,0};','T m0=0;')
        code=code.replace('struct Face {V normal;T d,a,a2,w,outside,apow,c_even,c_odd,p,q;};',
                          'struct Face {T d,a,a2,w,outside,apow,c_even,c_odd,p;};')
        code=code.replace('Face f{}; f.normal=cv(LD(g.normals,tet,face));',
                          'Face f{}; V normal=cv(LD(g.normals,tet,face));')
        code=code.replace('dot(f.normal,off)','dot(normal,off)')
        code=re.sub(r'f0\.q \+= [^;]+; f1\.q \+= [^;]+;','',code)
        code=code.replace('m1=add(m1,add(mul(f0.normal,f0.w*f0.q),mul(f1.normal,f1.w*f1.q)));','')
        start=code.index('V grad=add(');end=code.index('T result=',start)
        code=code[:start]+code[end:]
        code=code.replace('(A*m0+dot(mul(grad,h),m1))','m0')
    if operator=='fields':
        # Moments stay in registers; only one field/component's coefficients
        # and result are live at a time, independently of the field count.
        start=code.index('V grad=add(')
        code=code[:start]+'''\n#pragma unroll 1
for(int component=0;component<fields.shape[1];++component){
    auto field=LD(fields,tet,component);
'''+code[start:]
        code=code.replace('#undef LD\nreturn result;',
                          'LD(output,component,row)=result;\n}\n#undef LD\nreturn T(0);')
    if reload:
        for i in (1,2,3):
            code=code.replace(f'V p{i}=mul(sub(cv(LD(g.vertices,tet,{i})),cv(x)),ih);','')
        code=code.replace('auto point=[&](int i){return i==0?p0:i==1?p1:i==2?p2:p3;};','')
        code=code.replace('auto ids=LD(g.edges,edge);V v=point(ids[0]),uvec=point(ids[1]);',
            '''auto ids=LD(g.edges,edge);
    V v=mul(sub(cv(LD(g.vertices,tet,ids[0])),cv(x)),ih);
    V uvec=mul(sub(cv(LD(g.vertices,tet,ids[1])),cv(x)),ih);''')
        if operator in ('fields','occupancy'):
            declaration='V p0=mul(sub(cv(LD(g.vertices,tet,0)),cv(x)),ih);'
            code=code.replace(declaration,'')
            if operator=='fields':
                code=code.replace('V grad=add(',declaration+'\nV grad=add(')
    if layout=='packed':
        for name,offset in OFFSETS.items():
            code=code.replace(f'LD(g.{name},tet,',f'LD(g.data,tet,{offset}+')
    return code


@lru_cache(None)
def packed_type(precision='float64'):
    vec=baseline.functions(precision).vec

    @wp.struct
    class Geometry:
        data:wp.array2d(dtype=vec)
        edges:wp.array(dtype=wp.vec4i)

    return Geometry


def prepare_geometry(vertices,device='cpu',precision='float64',layout='packed'):
    if layout=='separate':
        return baseline.prepare_geometry(vertices,device,precision)
    if layout!='packed':
        raise ValueError('unknown geometry layout')
    # Reuse exactly the validated geometry preparation; pack its CPU arrays
    # before one data upload. No device->host geometry readback is introduced.
    original=baseline.prepare_geometry(vertices,'cpu',precision)
    data=np.concatenate([getattr(original,k).numpy() for k in OFFSETS],axis=1)
    g=packed_type(precision)()
    g.data=wp.array(data,dtype=baseline.functions(precision).vec,device=device)
    g.edges=wp.array(np.asarray(baseline.EDGES),dtype=wp.vec4i,device=device)
    return g


def geometry_bytes(geometry):
    names=('data','edges') if hasattr(geometry,'data') else (*OFFSETS,'edges')
    return sum(getattr(geometry,k).capacity for k in names)


@lru_cache(None)
def functions(precision='float64',layout='packed',reload=True):
    api=baseline.functions(precision)
    real,vec,field_vec=api.real,api.vec,api.field_vec
    Geometry=packed_type(precision) if layout=='packed' else api.Geometry

    @wp.func_native(source(precision,layout,reload,'p1'))
    def interaction(x:vec,h:real,tet:int,field:field_vec,g:Geometry)->real: ...

    @wp.func_native(source(precision,layout,reload,'occupancy'))
    def occupancy(x:vec,h:real,tet:int,g:Geometry)->real: ...

    @wp.func_native(source(precision,layout,reload,'fields'))
    def fields_interaction(x:vec,h:real,tet:int,fields:wp.array2d(dtype=field_vec),
                           output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel
    def p1_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
              tet:wp.array(dtype=int),field:wp.array(dtype=field_vec),
              geometry:Geometry,output:wp.array(dtype=real)):
        i=wp.tid()
        output[i]=interaction(particle[i],support[i],tet[i],field[i],geometry)

    @wp.kernel
    def occupancy_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                     tet:wp.array(dtype=int),geometry:Geometry,output:wp.array(dtype=real)):
        i=wp.tid()
        output[i]=occupancy(particle[i],support[i],tet[i],geometry)

    @wp.kernel
    def fields_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                  tet:wp.array(dtype=int),fields:wp.array2d(dtype=field_vec),
                  geometry:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=fields_interaction(particle[i],support[i],tet[i],fields,output,i,geometry)

    return SimpleNamespace(**{**vars(api),'Geometry':Geometry,'interaction':interaction,
        'kernel':p1_w6,'occupancy_kernel':occupancy_w6,'fields_kernel':fields_w6})
