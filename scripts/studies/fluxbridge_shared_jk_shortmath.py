"""Short range-reduced two-float seed polynomials; isolated CUDA experiment.

Coefficients are frozen host-generated Chebyshev interpolants, descending power
order. No generation dependency or fp64 device arithmetic. Final acceptance is
the unchanged full interaction gate, not the approximation estimate alone.
"""
from functools import lru_cache
from types import SimpleNamespace
import re
import warp as wp
import fluxbridge_shared_jk_layout as geometry
import fluxbridge_shared_jk_stability as control
from fluxbridge_shared_jk_floatpair import literal

COEFFICIENTS = {'strict': {'atan': [0.030662440370424447, -0.05874505621307831, 0.07563703551915192, -0.09078392072943424, 0.11110385046148014, -0.14285690423839287, 0.19999999604891977, -0.33333333330803444, 0.9999999999999732], 'log': [0.09804047819876527, 0.11087124955038745, 0.14286083072937864, 0.1999999744591619, 0.3333333333978963, 0.9999999999999736]}, 'balanced': {'atan': [-0.03765510601291402, 0.06974197686595265, -0.08992552906150912, 0.11103456908947464, -0.14285386553752139, 0.19999993053555815, -0.33333333276922494, 0.9999999999992447], 'log': [0.11808179549572992, 0.14267525518908156, 0.20000192336187594, 0.3333333262374376, 1.0000000000041798]}, 'short': {'atan': [0.0470734814196802, -0.08456192886940496, 0.1104048922721824, -0.14281588772654125, 0.19999883856551304, -0.33333332097609386, 0.9999999999783987], 'log': [0.14962195239572493, 0.19987425258759525, 0.33333407669075593, 0.9999999993156591]}, 'minimal': {'atan': [-0.06026305227658459, 0.10569828806414448, -0.14239532669648974, 0.19998183041083206, -0.33333306893048514, 0.9999999993712282], 'log': [0.14962195239572493, 0.19987425258759525, 0.33333407669075593, 0.9999999993156591]}}
VARIANTS=('control',*COEFFICIENTS)


def source(variant='strict'):
    code=control.source('floatpair-inline')
    if variant=='control':return code
    cs=COEFFICIENTS[variant]
    def horner(coefficients):
        return 'T s='+literal(coefficients[0])+';\n'+''.join('s=s*z+'+literal(c)+';\n' for c in coefficients[1:])
    code,n=re.subn(r'(T u=.*?z=u\*u;\n)T s=.*?(return T\(2\)\*u\*s)',
        lambda m:m[1]+horner(cs['log'])+m[2],code,count=1,flags=re.S)
    assert n==1
    code,n=re.subn(r'(T z=t\*t;\n)T s=.*?(T a=t\*s)',
        lambda m:m[1]+horner(cs['atan'])+m[2],code,count=1,flags=re.S)
    assert n==1
    return code


@lru_cache(None)
def functions(variant='strict'):
    if variant not in VARIANTS:raise ValueError('unknown variant')
    api=geometry.functions('float32');vec,real=api.vec,api.real
    Geometry=control.split_geometry_type()
    @wp.func_native(source(variant))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...
    @wp.kernel(module='unique',module_options={'enable_backward':False,'fuse_fp':True,'max_unroll':1})
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=interaction(particle[i],support[i],tet[i],output,i,g)
    return SimpleNamespace(**{**vars(api),'Geometry':Geometry,'kernel':baseline_w6,'channels':28})
