"""Precision diagnostic for the frozen full-28 shared-edge prototype.

Raw fp32 changes runtime scalar/vector storage, arithmetic and math intrinsics
to float. Geometry is prepared on the host as before, then stored in fp32.
No mixed-precision accumulator, error correction or acceptance relaxation.
"""
from functools import lru_cache
import re
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk_full as full
import fluxbridge_shared_jk_layout as geometry

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


def source(precision='float32',family='full'):
    if precision not in ('float32','float64'):raise ValueError('unknown precision')
    code=full.source(family)
    if precision=='float32':
        code=code.replace('using T=double;','using T=float;')
        code=re.sub(r'\b(log|atan2|sqrt|fabs)\(',r'\1f(',code)
    return code


@lru_cache(None)
def functions(precision='float32',family='full'):
    api=geometry.functions(precision);vec,real,Geometry=api.vec,api.real,api.Geometry

    @wp.func_native(source(precision,family))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=interaction(particle[i],support[i],tet[i],output,i,g)

    return SimpleNamespace(**{**vars(api),'kernel':baseline_w6,'channels':len(full.FAMILIES[family])})
