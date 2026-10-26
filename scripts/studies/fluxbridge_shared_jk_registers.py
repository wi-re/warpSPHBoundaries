"""Register/lifetime experiments; the validated shared-J/K baseline is preserved.

The direct variant contracts the P1 field into a scalar while streaming radial
degrees. It does not retain two face normals, four face-moment accumulators and
the final vector moment across the degree loop. Register caps are experiments,
not defaults: lower counts may spill or slow down an otherwise correct kernel.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import warp as wp

import fluxbridge_shared_jk as baseline

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


def source(precision, variant='direct'):
    code=baseline.source(precision)
    if variant=='baseline':
        return code
    if variant!='direct':
        raise ValueError('unknown contraction variant')
    start=code.index('V grad=add(')
    end=code.index('T result=',start)
    coefficients=code[start:end].replace('T A=field[0]-dot(mul(grad,h),p0);',
        'V B=mul(grad,h); T A=field[0]-dot(B,p0);')
    code=code[:start]+code[end:]
    code=code.replace('T m0=0; V m1{0,0,0};',coefficients+'\nT total=0;')
    code=code.replace('struct Face {V normal;T d,a,a2,w,outside,apow,c_even,c_odd,p,q;};',
        'struct Face {T a,a2,outside,apow,c_even,c_odd,P,Q;bool active;};')
    code=code.replace('Face f{}; f.normal=cv(LD(g.normals,tet,face));',
        'Face f{}; V normal=cv(LD(g.normals,tet,face));')
    code=code.replace('f.d=dot(f.normal,off); f.a=fabs', 'T d=dot(normal,off); f.a=fabs')
    code=code.replace('f.d','d')
    code=code.replace('f.w=dot(cv(LD(g.outward,tet,incidence)),off);',
                      'T w=dot(cv(LD(g.outward,tet,incidence)),off);')
    code=code.replace('f.w','w')
    code=code.replace('m0-=', 'total-=A*')
    code=code.replace('f.c_odd=f.a*E;',
        'f.P=A*d*w; f.Q=dot(B,normal)*w; f.active=true;\n        f.c_odd=f.a*E;')
    code=code.replace('if(f0.w==T(0) && f1.w==T(0))continue;',
                      'if(!f0.active && !f1.active)continue;')
    for values,name in ((baseline.coefficients()[0],'P'),(baseline.coefficients()[1],'Q')):
        old_name=name.lower()
        for coefficient in set(values.values()):
            literal=f'T({float(coefficient):.17g})'
            old=f'f0.{old_name if name=="P" else "q"} += {literal}*t0; f1.{old_name if name=="P" else "q"} += {literal}*t1;'
            code=code.replace(old,f'total += {literal}*(f0.{name}*t0+f1.{name}*t1);')
    code=code.replace('m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;','')
    code=code.replace('m1=add(m1,add(mul(f0.normal,f0.w*f0.q),mul(f1.normal,f1.w*f1.q)));','')
    code=code.replace('(A*m0+dot(mul(grad,h),m1))','total')
    return code


@lru_cache(None)
def functions(precision='float64',variant='direct',registers=None):
    api=baseline.functions(precision)
    real,vec,field_vec,Geometry=api.real,api.vec,api.field_vec,api.Geometry

    @wp.func_native(source(precision,variant))
    def interaction(x:vec,h:real,tet:int,field:field_vec,g:Geometry)->real: ...

    @wp.kernel(cuda_max_registers=registers)
    def p1_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
              tet:wp.array(dtype=int),field:wp.array(dtype=field_vec),
              geometry:Geometry,output:wp.array(dtype=real)):
        i=wp.tid()
        output[i]=interaction(particle[i],support[i],tet[i],field[i],geometry)

    return SimpleNamespace(**{**vars(api),'interaction':interaction,'kernel':p1_w6})


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--variant',choices=('baseline','direct'),default='direct')
    parser.add_argument('--max-registers',type=int)
    parser.add_argument('--shapes',action='store_true')
    args,rest=parser.parse_known_args()
    if args.max_registers is not None and args.max_registers<1:
        parser.error('positive register cap required')
    if args.shapes:
        import fluxbridge_shared_jk_shapes as study
    else:
        import fluxbridge_shared_jk_study as study
        study.source=lambda precision:source(precision,args.variant)
    study.functions=lambda precision='float64':functions(precision,args.variant,args.max_registers)
    sys.argv=[sys.argv[0],*rest]
    study.main()
    output=Path(rest[rest.index('--output')+1])
    path=output if args.shapes else output/'report.json'
    report=json.loads(path.read_text())
    report.update(variant=args.variant,cuda_max_registers=args.max_registers)
    if not args.shapes:
        device='cpu' if '--device' in rest and rest[rest.index('--device')+1]=='cpu' else 'cuda:0'
        if device!='cpu':
            report['cuda_kernel_properties']=wp.get_cuda_kernel_properties(
                functions(report['precision'],args.variant,args.max_registers).kernel,device)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
