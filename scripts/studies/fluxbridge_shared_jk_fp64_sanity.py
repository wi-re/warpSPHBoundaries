"""Collapse the paired core to native fp64, preserving its fp32 I/O contract."""
from functools import lru_cache
import re
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk_stability as paired
import fluxbridge_shared_jk_floatpair as arithmetic
import fluxbridge_shared_jk_layout as geometry

OPTIONS={'enable_backward':False,'fuse_fp':True,'max_unroll':1}
wp.set_module_options(OPTIONS)
VARIANTS=('paired','native64-series','native64-library')


def source(variant):
    code=paired.source('floatpair-inline')
    if variant=='paired':return code
    if variant not in VARIANTS:raise ValueError('unknown variant')
    body=code[code.index('#define LD'):]
    header='''using T=double;
auto fabsf=[](T a){return ::fabs(a);};
auto sqrtf=[](T a){return ::sqrt(a);};
'''
    if variant=='native64-series':
        # Keep the same polynomial degrees, coefficients and range reductions.
        # Extract a double mantissa instead of unpacking a two-float value.
        seeds=arithmetic.elementary_header()
        seeds=seeds[seeds.index('auto logf='):]
        tail=seeds[seeds.index('if(m<'):]
        header+='''auto logf=[](T a){
 union Bits {double d;unsigned long long u;};Bits bits;bits.d=a;
 int exponent=static_cast<int>((bits.u>>52)&2047)-1022;
 bits.u=(bits.u&0x000fffffffffffffULL)|0x3fe0000000000000ULL;
 T m=bits.d,mantissa=m;
'''+tail
    else:
        header+='''auto logf=[](T a){return ::log(a);};
auto atan2f=[](T y,T x){return ::atan2(y,x);};
'''
    code=header+body
    # Every two-word constructor in this generated source has simple scalar
    # arguments (coefficient literals or high/low geometry components).
    code=re.sub(r'T\(([^(),\n]+),([^(),\n]+)\)',r'(T(\1)+T(\2))',code)
    assert 'Pair' not in code and '.hi' not in code and '.lo' not in code
    return code


@lru_cache(None)
def functions(variant):
    if variant=='paired':return paired.functions('floatpair-inline')
    api=geometry.functions('float32');vec,real=api.vec,api.real;Geometry=paired.split_geometry_type()

    @wp.func_native(source(variant))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel(module='unique',module_options=OPTIONS)
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid();unused=interaction(particle[i],support[i],tet[i],output,i,g)

    return SimpleNamespace(**{**vars(api),'Geometry':Geometry,'kernel':baseline_w6,'channels':28})
