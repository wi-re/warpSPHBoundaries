"""Experimental solid-wall gradient moments and P1 gradient contractions.

Lean output: lambda, G=int grad_x W, six symmetric Cov entries, where
Cov=int (y-x) tensor grad_x W. Field contractions retain these ten scalars
locally, not in a global moment array. No pressure/porosity solver is included.
"""
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk as baseline
import fluxbridge_shared_jk_layout as layout_impl

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


def source(operator='lean'):
    if operator not in ('lean','fields'):raise ValueError('unknown output')
    code=layout_impl.source(layout='packed',reload=True,operator='occupancy')
    code=code.replace('T m0=0;', 'T m0=0; V G{0,0,0}; T xx=0,xy=0,xz=0,yy=0,yz=0,zz=0;')
    code=code.replace('c_even,c_odd,p;','c_even,c_odd,p,S;')
    co={n+2:F(c,n+2) for n,c in enumerate(baseline.SHAPE) if c}
    a1=sum(F(c,n+2) for n,c in enumerate(baseline.SHAPE))
    # At c=0 the occupancy flux vanishes, but the tangential potential does
    # not. J_m has this elementary absolute-power limit, including s=0.
    line=[f'T L0=-T({float(a1):.17g})*(hi-lo); T pl=1,pu=1;']
    for m in range(1,14):
        line.append('pl*=fabs(lo);pu*=fabs(hi);')
        if m in co:
            line.append(f'L0+=T({float(co[m]/(m+1)):.17g})*(hi*pu-lo*pl);')
    singular='''if(c2==T(0)){
        T lo=s0>T(-1)?s0:T(-1),hi=s1<T(1)?s1:T(1);
        if(hi>lo){
'''+ '\n'.join(line)+'''
            consume(ids[2],2*edge,T(0),T(0),L0);
            consume(ids[3],2*edge+1,T(0),T(0),L0);
        }
        continue;
    }'''
    code=code.replace('if(c2==T(0))continue;',singular)
    code=code.replace('T rlpow=T(1),rupow=T(1);',
        f'T L=-T({float(a1):.17g})*(u-l); T rlpow=T(1),rupow=T(1);')
    for m,c in co.items():
        cp=baseline.coefficients()[0][m];p=f'T({float(cp):.17g})';s=f'T({float(c):.17g})'
        needle=f'f0.p += {p}*t0; f1.p += {p}*t1;'
        code=code.replace(needle,needle+f'\n f0.S += {s}*t0; f1.S += {s}*t1;')
        parity='odd' if m%2 else 'even'
        needle=f'j_{parity} = (u*rupow-l*rlpow+T({m})*c2*jm2)*T({1/(m+1):.17g});'
        if needle not in code:raise RuntimeError('J recurrence generation changed')
        code=code.replace(needle,needle+f'\n L += {s}*j_{parity};')
    consumer=r'''
auto consume=[&](int face,int incidence,T d,T S,T L){
    V n=cv(LD(g.data,tet,4+face)),mu=cv(LD(g.data,tet,14+incidence));
    V U=add(mul(n,d*S),mul(mu,L));
    G=add(G,mul(n,S));
    xx+=U.x*n.x; yy+=U.y*n.y; zz+=U.z*n.z;
    xy+=T(.5)*(U.x*n.y+U.y*n.x);
    xz+=T(.5)*(U.x*n.z+U.z*n.x);
    yz+=T(.5)*(U.y*n.z+U.z*n.y);
};
'''
    code=code.replace('for(int edge=0;edge<6;++edge){',consumer+'\nfor(int edge=0;edge<6;++edge){')
    needle='m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;'
    code=code.replace(needle,needle+'''
    consume(ids[2],2*edge,f0.d,f0.w*f0.S,L);
    consume(ids[3],2*edge+1,f1.d,f1.w*f1.S,L);
''')
    start=code.index('T result=')
    code=code[:start]+r'''
T C=T(1365.0/(64.0*3.141592653589793238462643383279502884));
G=mul(G,-C/h);
xx=C*(m0-xx);yy=C*(m0-yy);zz=C*(m0-zz);
xy=-C*xy;xz=-C*xz;yz=-C*yz;
OUTPUT
#undef LD
return T(0);
'''
    if operator=='lean':
        output='\n'.join(f'LD(output,{i},row)={value};' for i,value in enumerate(
            ('C*m0','G.x','G.y','G.z','xx','xy','xz','yy','yz','zz')))
    else:
        output=r'''
V p0=mul(sub(cv(LD(g.data,tet,0)),cv(x)),T(1)/h);
#pragma unroll 1
for(int component=0;component<fields.shape[1];++component){
    auto field=LD(fields,tet,component);
    V grad=add(add(mul(cv(LD(g.data,tet,27)),field[1]-field[0]),
                  mul(cv(LD(g.data,tet,28)),field[2]-field[0])),
                  mul(cv(LD(g.data,tet,29)),field[3]-field[0]));
    T A=field[0]-dot(mul(grad,h),p0);
    LD(output,3*component,row)=A*G.x+grad.x*xx+grad.y*xy+grad.z*xz;
    LD(output,3*component+1,row)=A*G.y+grad.x*xy+grad.y*yy+grad.z*yz;
    LD(output,3*component+2,row)=A*G.z+grad.x*xz+grad.y*yz+grad.z*zz;
}
'''
    return code.replace('OUTPUT',output)


@lru_cache(None)
def functions():
    api=layout_impl.functions();vec,real,field_vec,Geometry=api.vec,api.real,api.field_vec,api.Geometry

    @wp.func_native(source('lean'))
    def lean(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.func_native(source('fields'))
    def fields(x:vec,h:real,tet:int,fields:wp.array2d(dtype=field_vec),
               output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel
    def lean_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                tet:wp.array(dtype=int),geometry:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=lean(particle[i],support[i],tet[i],output,i,geometry)

    @wp.kernel
    def fields_gradient_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                           tet:wp.array(dtype=int),coeff:wp.array2d(dtype=field_vec),
                           geometry:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=fields(particle[i],support[i],tet[i],coeff,output,i,geometry)

    return SimpleNamespace(**{**vars(api),'lean_kernel':lean_w6,'gradient_kernel':fields_gradient_w6})
