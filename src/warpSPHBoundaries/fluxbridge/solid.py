"""Construction-stage solid triangle/tetrahedron kernels with explicit dtype.

No solver, porous field or general moment interpreter is involved. Outputs are
[channel,pair], in physical support units: lambda, G=int grad_x W, and symmetric
Cov=int (y-x) tensor grad_x W. ``wcsph`` appends the physical
int Laplacian(W), and ``wcsph_extended`` additionally appends m1=int (y-x)W
and cover=int unit(x-y) over the support. Tensile shifting is not included.
The default fp32 path uses protected two-float arithmetic, with no device fp64.
Explicit conditioning="raw" exposes native-fp32 diagnostics: both profiles fail
the broad 1e-5 gate on some inputs. Provider dispatch is unchanged.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from . import _solid_code as codegen

BUNDLES=('density','pressure','lean','wcsph','wcsph_extended')
LABELS={2:('lambda','Gx','Gy','Covxx','Covxy','Covyy'),
        3:('lambda','Gx','Gy','Gz','Covxx','Covxy','Covxz','Covyy','Covyz','Covzz')}
EDGES=tuple((i,j,*[k for k in range(4) if k not in (i,j)]) for i in range(4) for j in range(i+1,4))


@lru_cache(None)
def _types(precision,conditioning='raw'):
    if precision not in ('float32','float64'):raise ValueError('precision must be float32 or float64')
    if conditioning not in ('raw','protected') or (conditioning=='protected' and precision!='float32'):raise ValueError('protected arithmetic requires float32')
    real=wp.float32 if precision=='float32' else wp.float64
    vec=wp.types.vector(length=3,dtype=real)
    @wp.struct
    class Geometry:
        data:wp.array2d(dtype=vec)
        edges:wp.array(dtype=wp.vec4i)
        dimension:int
    if conditioning=='protected':
        @wp.struct
        class ProtectedGeometry:
            data:wp.array2d(dtype=vec)
            data_low:wp.array2d(dtype=vec)
            edges:wp.array(dtype=wp.vec4i)
            dimension:int
        Geometry=ProtectedGeometry
    return real,vec,Geometry


def solid_source(dimension,kernel='w2',precision='float64',bundle='lean',conditioning=None):
    """Generated native body; useful for inspection and independent cost audits."""
    if dimension not in (2,3) or bundle not in BUNDLES:raise ValueError('invalid dimension or bundle')
    conditioning=('protected' if precision=='float32' else 'raw') if conditioning is None else conditioning
    _types(precision,conditioning)
    code=(codegen.triangle if dimension==2 else codegen.tetrahedron)(kernel,bundle,precision)
    if conditioning=='protected':
        import re
        from . import _solid_pair as pair
        code=pair.transform(code,accurate_math=True,fast=True,force_inline=True)
        # Pass geometry loads by value rather than capturing the struct.
        loader='''auto cvpair=[](auto hi,auto lo){
 return V{T(hi[0],lo[0]),T(hi[1],lo[1]),T(hi[2],lo[2])};};
'''
        code=code.replace('T h_pair=T(h);',loader+'T h_pair=T(h);')
        code=re.sub(r'cv\(LD\(g.data,tet,([^\n]+?)\)\)',r'cvpair(LD(g.data,tet,\1),LD(g.data_low,tet,\1))',code)
    return code


@lru_cache(None)
def solid_functions(dimension,kernel='w2',precision='float64',bundle='lean',registers=None,conditioning=None):
    """Return reusable Warp device ``element`` and one-thread-per-pair ``kernel``.

    Inputs: vec3 positions (z=0 in 2D), positive supports, element indices,
    prepared geometry and caller-owned [channels,pairs] output. No allocations
    occur inside either device entry point. Register caps are explicit CUDA
    experiments, not implicit precision or dispatch changes. WCSPH lap is the
    physical integral (units length^-2), not the legacy provider's normalized
    auxiliary-kernel channel; do not apply lap_factor a second time. Cover has
    units length^dimension and has no wall particle number-density multiplier. Default conditioning
    is native fp64 or protected two-float fp32; conditioning="raw" is the
    explicit, currently unaccepted native-fp32 accuracy/performance probe.
    """
    if registers is not None and not 32<=registers<=255:raise ValueError('register cap must lie in [32,255]')
    conditioning=('protected' if precision=='float32' else 'raw') if conditioning is None else conditioning
    real,vec,Geometry=_types(precision,conditioning)
    source=solid_source(dimension,kernel,precision,bundle,conditioning)
    labels=LABELS[dimension][:(1 if bundle=='density' else dimension+1)] if bundle in ('density','pressure') else LABELS[dimension]
    if bundle in ('wcsph','wcsph_extended'):labels+=('lap',)
    if bundle=='wcsph_extended':labels+=tuple('m1'+a for a in 'xyz'[:dimension])+tuple('cover'+a for a in 'xyz'[:dimension])
    # Passing the 176-byte protected Geometry struct directly to the native
    # function triggers an invalid .param -> .local conversion in NVRTC 12.9
    # for some extended bodies. Separate array arguments avoid that lowering.
    native_source=source.replace('g.data_low','geometry_low').replace('g.data','geometry_data').replace('g.edges','geometry_edges')
    if conditioning=='protected':
        @wp.func_native(native_source)
        def native(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,
                   geometry_data:wp.array2d(dtype=vec),geometry_low:wp.array2d(dtype=vec),geometry_edges:wp.array(dtype=wp.vec4i))->real: ...
        @wp.func
        def element(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real:
            return native(x,h,tet,output,row,g.data,g.data_low,g.edges)
    else:
        @wp.func_native(native_source)
        def native(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,
                   geometry_data:wp.array2d(dtype=vec),geometry_edges:wp.array(dtype=wp.vec4i))->real: ...
        @wp.func
        def element(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real:
            return native(x,h,tet,output,row,g.data,g.edges)
    @wp.kernel(module='unique',cuda_max_registers=registers,module_options={'enable_backward':False,'fuse_fp':True,'max_unroll':1})
    def evaluate(positions:wp.array(dtype=vec),supports:wp.array(dtype=real),
                 indices:wp.array(dtype=int),geometry:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid();ignored=element(positions[i],supports[i],indices[i],output,i,geometry)
    return SimpleNamespace(real=real,vec=vec,Geometry=Geometry,element=element,kernel=evaluate,
        channels=len(labels),labels=labels,dimension=dimension,precision=precision,bundle=bundle,conditioning=conditioning)


def prepare_solid_geometry(vertices,device='cpu',precision='float64',conditioning=None):
    """Prepare static straight-simplex data once; rebuild when vertices change.

    Input is [element,3,2] or [element,4,3]. Geometry is computed in host fp64
    and stored in the requested dtype. No FEM gradients or per-pair data are
    prepared. Triangle winding and tet orientation do not change the solid.
    """
    v=np.asarray(vertices,dtype=np.float64)
    if v.ndim!=3 or v.shape[1:] not in ((3,2),(4,3)) or not np.isfinite(v).all():raise ValueError('finite [E,3,2] triangles or [E,4,3] tetrahedra required')
    conditioning=('protected' if precision=='float32' else 'raw') if conditioning is None else conditioning
    d=v.shape[2];real,vec,Geometry=_types(precision,conditioning);rows=[]
    if precision=='float32':
        with np.errstate(over='ignore'):v=v.astype(np.float32).astype(np.float64)
        if not np.isfinite(v).all():raise ValueError('coordinates are not representable in float32')
    for simplex in v:
        if d==2:
            simplex=simplex.copy();e0,e1=simplex[1]-simplex[0],simplex[2]-simplex[0];det=e0[0]*e1[1]-e0[1]*e1[0]
            if det==0:raise ValueError('degenerate triangle')
            if det<0:simplex[[1,2]]=simplex[[2,1]]
            tangents=np.roll(simplex,-1,axis=0)-simplex;tangents/=np.linalg.norm(tangents,axis=1)[:,None]
            normals=np.stack([tangents[:,1],-tangents[:,0]],axis=1)
            rows.append(np.pad(np.concatenate([simplex,normals,tangents]),((0,0),(0,1))))
        else:
            if np.linalg.det((simplex[1:]-simplex[0]).T)==0:raise ValueError('degenerate tetrahedron')
            normals=[]
            for opposite in range(4):
                ids=[j for j in range(4) if j!=opposite];a,b,c=simplex[ids];n=np.cross(b-a,c-a)
                if np.dot(n,simplex[opposite]-a)>0:n=-n
                normals.append(n/np.linalg.norm(n))
            tangents=[];outward=[]
            for i,j,fa,fb in EDGES:
                t=simplex[j]-simplex[i];t/=np.linalg.norm(t);tangents.append(t)
                for face in (fa,fb):
                    mu=np.cross(t,normals[face]);third=next(k for k in range(4) if k not in (i,j,face))
                    if np.dot(mu,simplex[third]-simplex[i])>0:mu=-mu
                    outward.append(mu)
            rows.append(np.concatenate([simplex,np.array(normals),np.array(tangents),np.array(outward)]))
    g=Geometry();g.dimension=d
    count=9 if d==2 else 26
    values=np.array(rows).reshape(-1,count,3)
    hi=values.astype(np.float32) if precision=='float32' else values
    g.data=wp.array(hi,dtype=vec,device=device)
    if conditioning=='protected':g.data_low=wp.array(values-hi.astype(float),dtype=vec,device=device)
    g.edges=wp.array(np.array(EDGES if d==3 else [(0,1,0,0),(1,2,0,0),(2,0,0,0)],np.int32),dtype=wp.vec4i,device=device)
    return g
