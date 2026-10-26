"""Single-P1 gradient specialization: contract before retaining a Cov tensor."""
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk_gradient as moments

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


def source():
    code=moments.source('lean')
    coefficients=r'''
V p0=mul(sub(cv(LD(g.data,tet,0)),cv(x)),T(1)/h);
V grad=add(add(mul(cv(LD(g.data,tet,27)),field[1]-field[0]),
              mul(cv(LD(g.data,tet,28)),field[2]-field[0])),
              mul(cv(LD(g.data,tet,29)),field[3]-field[0]));
V B=mul(grad,h);T A=field[0]-dot(B,p0);
'''
    code=code.replace('auto consume=',coefficients+'\nauto consume=')
    start=code.index('    V U=add(');end=code.index('\n};',start)
    code=code[:start]+r'''
    G=add(G,mul(n,(A+d*dot(B,n))*S+dot(B,mu)*L));
'''+code[end:]
    code=code.replace('G=mul(G,-C/h);','G=mul(sub(G,mul(B,m0)),-C/h);')
    # The compiler drops the unused covariance scalars. Only m0 and G remain
    # live through the six edges, rather than ten output moment accumulators.
    start=code.index('LD(output,0,row)=C*m0;');end=code.index('#undef LD',start)
    code=code[:start]+'''LD(output,0,row)=G.x;
LD(output,1,row)=G.y;
LD(output,2,row)=G.z;
'''+code[end:]
    return code


@lru_cache(None)
def functions():
    api=moments.functions();vec,real,field_vec,Geometry=api.vec,api.real,api.field_vec,api.Geometry

    @wp.func_native(source())
    def interaction(x:vec,h:real,tet:int,field:field_vec,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel
    def gradient_p1(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),fields:wp.array2d(dtype=field_vec),
                    geometry:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=interaction(particle[i],support[i],tet[i],fields[tet[i],0],output,i,geometry)

    return SimpleNamespace(**{**vars(api),'contracted_gradient_kernel':gradient_p1})
