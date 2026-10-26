"""Face-state lifetime and caller-output staging experiments; no new precision.

Serial faces recompute the algebraic J stream from one shared log seed and
repeat the edge potentials. They trade arithmetic for fewer live face states.
Staging explicitly uses the caller's final output allocation for 26 raw tensor
accumulators, trading global traffic for live registers; no scratch allocation.
"""
from functools import lru_cache
import re
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk_shortmath as seeds
import fluxbridge_shared_jk_layout as geometry
import fluxbridge_shared_jk_stability as paired

BASES=('control','serial','staged','serial-staged')
VARIANTS=tuple(v+(f'@{cap}' if cap else '') for cap in (None,192,160,128,96,80,64) for v in BASES)


def source(variant='control'):
    kind=variant.split('@')[0]
    if kind not in BASES:raise ValueError('unknown variant')
    code=seeds.source('balanced')
    if kind.startswith('serial'):
        start=code.index('Face f0=face_data(');end=code.index('// C_1=',start)
        old=code[start:end];seed=old[old.index('T j_even='):]
        seed=re.sub(r'\bj_even\b','j_even_seed',seed);seed=re.sub(r'\bj_odd\b','j_odd_seed',seed)
        code=code[:start]+seed+'''for(int side=0;side<2;++side){
 Face f0=face_data(ids[2+side],2*edge+side);
 T j_even=j_even_seed,j_odd=j_odd_seed;
'''+code[end:]
        code=code.replace('m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;','m0+=f0.d*f0.w*f0.p;')
        code=code.replace('mL+=f0.d*f0.w*f0.pL+f1.d*f1.w*f1.pL;','mL+=f0.d*f0.w*f0.pL;')
        code=re.sub(r'consume\(ids\[3\],2\*edge\+1,f1\.[^;]+;','',code)
        code=code.replace('consume(ids[2],2*edge,f0.d','consume(ids[2+side],2*edge+side,f0.d')
        code=re.sub(r'T t1=[^;]+;','',code)
        code=re.sub(r'f1\.\w+(?:\*=|\+=|=)[^;]+;','',code)
        assert 'f1.' not in code
        pos=code.index('T C=');code=code[:pos]+'}\n'+code[pos:]
    if 'staged' in kind:
        names=['m0','mL']+[n+'.'+a for n in ('m1','G') for a in 'xyz']+[p+s for p in ('b','v','h') for s in ('xx','xy','xz','yy','yz','zz')]
        aliases={n:f'LD(output,{i},row)' for i,n in enumerate(names)}
        old='float m0=0,mL=0;struct A{float x,y,z;};A m1{0,0,0},G{0,0,0};'
        new='struct A{volatile float &x,&y,&z;};\n'
        new+='volatile float &m0='+aliases['m0']+',&mL='+aliases['mL']+';\n'
        for n in ('m1','G'):new+='A '+n+'{'+','.join(aliases[n+'.'+a] for a in 'xyz')+'};\n'
        assert old in code;code=code.replace(old,new)
        for prefix in ('b','v','h'):
            pattern=r'float '+prefix+r'xx=[^;]+;'
            replacement='volatile float '+','.join('&'+n+'='+aliases[n] for n in names if n.startswith(prefix) and '.' not in n)+';'
            code,n=re.subn(pattern,lambda _:replacement,code);assert n==1
        pos=code.index('auto consume=')
        code=code[:pos]+''.join(n+'=0;\n' for n in names)+code[pos:]
        pos=code.index('T C=')
        # Snapshot all raw tensor slots before writing the final, differently
        # indexed output. Volatile staging prevents accidental register caching.
        snapshot='{\nfloat m0='+aliases['m0']+',mL='+aliases['mL']+';\nstruct Final{float x,y,z;};\n'
        for n in ('m1','G'):snapshot+='Final '+n+'{'+','.join(aliases[n+'.'+a] for a in 'xyz')+'};\n'
        for prefix in ('b','v','h'):snapshot+='float '+','.join(n+'='+aliases[n] for n in names if n.startswith(prefix) and '.' not in n)+';\n'
        code=code[:pos]+snapshot+code[pos:]
        code+='\n}\n'
    return code


@lru_cache(None)
def functions(variant='control'):
    if variant not in VARIANTS:raise ValueError('unknown variant')
    cap=int(variant.split('@')[1]) if '@' in variant else None
    api=geometry.functions('float32');vec,real=api.vec,api.real;Geometry=paired.split_geometry_type()
    @wp.func_native(source(variant))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...
    @wp.kernel(module='unique',cuda_max_registers=cap,module_options={'enable_backward':False,'fuse_fp':True,'max_unroll':1})
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid();ignored=interaction(particle[i],support[i],tet[i],output,i,g)
    return SimpleNamespace(**{**vars(api),'Geometry':Geometry,'kernel':baseline_w6,'channels':28})
