"""Explicit paired-fp32 arithmetic; the underlying operations stay float32."""
import warp as wp
from ._paired_code import ARITHMETIC

wp.set_module_options({"enable_backward":False,"fuse_fp":False})

@wp.func_native(ARITHMETIC+r'''
DS x(a),y(b),result;
if(op==0)result=x+y;
if(op==1)result=x-y;
if(op==2)result=x*y;
if(op==3)result=x/y;
if(op==4)result=DS::sqrt(x);
if(op==5)result=DS::log(x);
if(op==6)result=DS::atan2(x,y);
if(op==7)result=DS::asinh(x);
return result.pair();
''')
def operation(a:wp.vec2f,b:wp.vec2f,op:int)->wp.vec2f: ...

@wp.kernel
def arithmetic(a:wp.array(dtype=wp.vec2f),b:wp.array(dtype=wp.vec2f),op:int,out:wp.array(dtype=wp.vec2f)):
    i=wp.tid()
    out[i]=operation(a[i],b[i],op)

from functools import lru_cache
from types import SimpleNamespace
from ._paired_code import PRIMITIVES,ELEMENT,PROFILE,CACHE,CACHED_OUTPUT,CONTRACT,TRANSPOSE


@wp.struct
class PairedPlan:
    f:wp.array2d(dtype=int)
    dep:wp.array2d(dtype=int)
    fr:wp.array(dtype=wp.float32)
    b:wp.array2d(dtype=int)
    br:wp.array(dtype=wp.float32)
    v:wp.array2d(dtype=int)
    vr:wp.array(dtype=wp.float32)
    term_kind:wp.array(dtype=int)
    term_index:wp.array(dtype=int)
    term_value:wp.array(dtype=wp.vec2f)
    offsets:wp.array(dtype=int)
    dimension:int
    surface:int
    compact:int


@wp.func_native(ARITHMETIC+PRIMITIVES+ELEMENT+PROFILE+CACHE)
def element(query:wp.vec3f,h:wp.float32,facets:wp.array2d(dtype=wp.vec3f),first:int,last:int,
            plan:PairedPlan,work:wp.array2d(dtype=wp.vec2f),row:int): ...


@wp.func_native(CACHED_OUTPUT)
def paired_output(plan:PairedPlan,work:wp.array2d(dtype=wp.vec2f),row:int,column:int)->wp.vec2f: ...


@wp.func
def output(plan:PairedPlan,work:wp.array2d(dtype=wp.vec2f),row:int,column:int):
    value=paired_output(plan,work,row,column)
    return value[0]+value[1]


@wp.func
def power(x:wp.float32,n:int):
    value=wp.float32(1.)
    count=n
    if n<0:
        count=-n
    for k in range(count):
        value=value*x
    if n<0:
        value=wp.float32(1.)/value
    return value


@wp.kernel
def batch(queries:wp.array(dtype=wp.vec3f),supports:wp.array(dtype=wp.float32),facets:wp.array2d(dtype=wp.vec3f),
          starts:wp.array(dtype=int),pq:wp.array(dtype=int),pe:wp.array(dtype=int),shifts:wp.array(dtype=wp.vec3f),
          plan:PairedPlan,scales:wp.array(dtype=int),work:wp.array2d(dtype=wp.vec2f),out:wp.array2d(dtype=wp.float32)):
    row=wp.tid()
    qi=pq[row]
    ei=pe[row]
    h=supports[qi]
    element(queries[row],h,facets,starts[ei],starts[ei+1],plan,work,row)
    for col in range(out.shape[1]):
        out[row,col]=output(plan,work,row,col)*power(h,scales[col])


@lru_cache(None)
def functions():
    return SimpleNamespace(real=wp.float32,vec=wp.vec3f,Plan=PairedPlan,work_dtype=wp.vec2f,
                           coefficient_dtype=wp.vec2f,field_dtype=wp.vec2f,output_cache=True,local_geometry=True,element=element,output=output,
                           batch=batch,power=power,paired_output=paired_output)


@wp.func_native(ARITHMETIC+CONTRACT)
def contract(query:wp.vec3f,shift:wp.vec3f,origin:wp.vec3f,h:wp.float32,ei:int,
             plan:PairedPlan,scales:wp.array(dtype=int),work:wp.array2d(dtype=wp.vec2f),row:int,
             terms:wp.array2d(dtype=int),factors:wp.array(dtype=wp.float32),
             coefficients:wp.array3d(dtype=wp.vec2f),pairs:wp.array2d(dtype=wp.float32)): ...


@wp.kernel
def fused(queries:wp.array(dtype=wp.vec3f),supports:wp.array(dtype=wp.float32),facets:wp.array2d(dtype=wp.vec3f),
          starts:wp.array(dtype=int),pq:wp.array(dtype=int),pe:wp.array(dtype=int),shifts:wp.array(dtype=wp.vec3f),
          plan:PairedPlan,scales:wp.array(dtype=int),work:wp.array2d(dtype=wp.vec2f),
          terms:wp.array2d(dtype=int),factors:wp.array(dtype=wp.float32),origins:wp.array(dtype=wp.vec3f),
          coefficients:wp.array3d(dtype=wp.vec2f),pairs:wp.array2d(dtype=wp.float32)):
    row=wp.tid()
    qi=pq[row]
    ei=pe[row]
    h=supports[qi]
    element(queries[row],h,facets,starts[ei],starts[ei+1],plan,work,row)
    contract(queries[row],shifts[row],origins[ei],h,ei,plan,scales,work,row,terms,factors,coefficients,pairs)


@wp.func_native(ARITHMETIC+TRANSPOSE)
def transpose_entry(queries:wp.array(dtype=wp.vec3f),supports:wp.array(dtype=wp.float32),pq:wp.array(dtype=int),shifts:wp.array(dtype=wp.vec3f),
                    plan:PairedPlan,scales:wp.array(dtype=int),work:wp.array2d(dtype=wp.vec2f),terms:wp.array2d(dtype=int),
                    factors:wp.array(dtype=wp.float32),origins:wp.array(dtype=wp.vec3f),reverse:wp.array(dtype=int),offsets:wp.array(dtype=int),
                    loads:wp.array2d(dtype=wp.float32),volumes:wp.array(dtype=wp.float32),out:wp.array3d(dtype=wp.float32),ei:int,a:int,c:int): ...


@wp.kernel
def transpose(queries:wp.array(dtype=wp.vec3f),supports:wp.array(dtype=wp.float32),pq:wp.array(dtype=int),shifts:wp.array(dtype=wp.vec3f),
              plan:PairedPlan,scales:wp.array(dtype=int),work:wp.array2d(dtype=wp.vec2f),terms:wp.array2d(dtype=int),
              factors:wp.array(dtype=wp.float32),origins:wp.array(dtype=wp.vec3f),reverse:wp.array(dtype=int),offsets:wp.array(dtype=int),
              loads:wp.array2d(dtype=wp.float32),volumes:wp.array(dtype=wp.float32),out:wp.array3d(dtype=wp.float32)):
    ei,a,c=wp.tid()
    transpose_entry(queries,supports,pq,shifts,plan,scales,work,terms,factors,origins,reverse,offsets,loads,volumes,out,ei,a,c)


def field_functions():
    return fused,transpose
