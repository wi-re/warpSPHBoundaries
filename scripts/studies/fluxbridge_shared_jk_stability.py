"""Isolated all-fp32 conditioning experiments for the frozen full W6 body."""
from functools import lru_cache
import re
from types import SimpleNamespace
import numpy as np
import warp as wp
import fluxbridge_shared_jk_full_precision as baseline
import fluxbridge_shared_jk_layout as geometry
import fluxbridge_shared_jk_floatpair as floatpair

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})
VARIANTS=('raw','sums','products','floatpair','floatpair-math','floatpair-geometry','floatpair-core','floatpair-fast','floatpair-inline')
SPLIT_GEOMETRY=('floatpair-geometry','floatpair-core','floatpair-fast','floatpair-inline')


@lru_cache(None)
def split_geometry_type():
    vec=geometry.functions('float32').vec

    @wp.struct
    class Geometry:
        data:wp.array2d(dtype=vec)
        data_low:wp.array2d(dtype=vec)
        edges:wp.array(dtype=wp.vec4i)

    return Geometry


def prepare_geometry(vertices,device,variant):
    if variant not in SPLIT_GEOMETRY:return geometry.prepare_geometry(vertices,device,'float32')
    original=geometry.prepare_geometry(vertices,'cpu','float64')
    values=original.data.numpy();hi=values.astype(np.float32);lo=(values-hi.astype(float)).astype(np.float32)
    g=split_geometry_type()();vec=geometry.functions('float32').vec
    g.data=wp.array(hi,dtype=vec,device=device);g.data_low=wp.array(lo,dtype=vec,device=device)
    g.edges=wp.array(original.edges.numpy(),dtype=wp.vec4i,device=device)
    return g


def source(variant='raw'):
    if variant not in VARIANTS:raise ValueError('unknown variant')
    code=baseline.source('float32')
    if variant=='raw':return code
    if variant.startswith('floatpair'):
        code=floatpair.transform(code,variant!='floatpair',variant in ('floatpair-fast','floatpair-inline'),variant=='floatpair-inline')
        if variant in SPLIT_GEOMETRY:
            loader='''auto geo=[&](int col){
 auto hi=LD(g.data,tet,col),lo=LD(g.data_low,tet,col);
 return V{T(hi[0],lo[0]),T(hi[1],lo[1]),T(hi[2],lo[2])};
};
'''
            code=code.replace('T h_pair=T(h);',loader+'T h_pair=T(h);')
            code=re.sub(r'cv\(LD\(g.data,tet,([^\n]+?)\)\)',r'geo(\1)',code)
        if variant in ('floatpair-core','floatpair-fast','floatpair-inline'):
            # Keep higher precision in geometry, seeds and radial reductions;
            # accumulate the bounded final tensors as ordinary float scalars.
            code=code.replace('T m0=0,mL=0; V m1{0,0,0},G{0,0,0};',
                'float m0=0,mL=0;struct A{float x,y,z;};A m1{0,0,0},G{0,0,0};')
            for prefix in ('b','v','h'):code=code.replace('T '+prefix+'xx=','float '+prefix+'xx=')
            for name,value in [('m1','q'),('G','S')]:
                code=code.replace(f'{name}=add({name},mul(n,w*{value}));',
                    ''.join(f'{name}.{a}+=static_cast<float>(n.{a}*w*{value});' for a in 'xyz'))
            start=code.index('T C=');body,output=code[:start],code[start:]
            names=['m0','mL']+[p+s for p in ('b','v','h') for s in ('xx','xy','xz','yy','yz','zz')]
            names += [n+'.'+a for n in ('m1','G') for a in 'xyz']
            pattern=r'\b('+ '|'.join(re.escape(n) for n in names)+r')\b'
            output=re.sub(pattern,r'T(\1)',output)
            code=body+output
        return code
    helper='''
auto accumulate=[](T &sum,T &correction,T value){
 T next=sum+value;
 correction+=fabsf(sum)>=fabsf(value)?(sum-next)+value:(value-next)+sum;
 sum=next;
};
auto product=[&](T &sum,T &correction,T hi,T lo,T value){
 T p=hi*value;
 accumulate(sum,correction,p);
 accumulate(sum,correction,fmaf(hi,value,-p)+lo*value);
};
'''
    code=code.replace('T ih=T(1)/h;',helper+'T ih=T(1)/h;')
    code=code.replace('p,q,pL,S,R;','p,q,pL,S,R,cp,cq,cpL,cS,cR;')
    code=re.sub(r'T Ephi=([^;]+);',r'T Ephi=\1;T ephi_lo=0,el_lo=0,ep_lo=0;',code)
    def update(sum_,corr,term):
        if variant=='sums':return f'accumulate({sum_},{corr},{term});'
        match=re.fullmatch(r'T\(([^)]+)\)\*(.*)',term)
        if match is None:raise RuntimeError('unexpected coefficient product: '+term)
        value=float(match[1]);hi=np.float32(value);lo=np.float32(value-float(hi))
        return f'product({sum_},{corr},T({float(hi):.17g}),T({float(lo):.17g}),{match[2]});'
    code=re.sub(r'(f[01])\.(pL|p|q|S|R)\+=([^;\n]+);',lambda m:update(m[1]+'.'+m[2],m[1]+'.c'+m[2],m[3]),code)
    corrections={'Ephi':'ephi_lo','EL':'el_lo','EP':'ep_lo'}
    code=re.sub(r'(Ephi|EL|EP)\+=([^;\n]+);',lambda m:update(m[1],corrections[m[1]],m[2]),code)
    code=code.replace('Ephi,EL,EP);','Ephi+ephi_lo,EL+el_lo,EP+ep_lo);')
    finalize='\n'.join(f'f{f}.{name}+=f{f}.c{name};' for f in (0,1) for name in ('p','q','pL','S','R'))
    code=code.replace('m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;',finalize+'\nm0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;')
    return code


@lru_cache(None)
def functions(variant='raw'):
    api=geometry.functions('float32');vec,real,Geometry=api.vec,api.real,api.Geometry
    if variant in SPLIT_GEOMETRY:Geometry=split_geometry_type()

    @wp.func_native(source(variant))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    # Isolate each ablation's compiler unit: unrelated native bodies in one
    # module otherwise alter inlining and local allocation of this large body.
    @wp.kernel(module='unique',module_options={'enable_backward':False,'fuse_fp':True,'max_unroll':1})
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=interaction(particle[i],support[i],tet[i],output,i,g)

    return SimpleNamespace(**{**vars(api),'Geometry':Geometry,'kernel':baseline_w6,'channels':28})
