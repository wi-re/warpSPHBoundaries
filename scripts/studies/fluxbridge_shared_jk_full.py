"""One shared edge stream for the complete 28-channel W6 tetrahedron baseline.

Local tensors are symmetric scalars, not primitive/moment scratch arrays.
Compile-time family selection lets unused output and recurrence work disappear.
Regular flux and the exact edge-line limits are retained. fp64 only for now.
"""
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
import fluxbridge_shared_jk as base
import fluxbridge_shared_jk_layout as geometry

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
LABELS=['xx','xy','xz','yy','yz','zz']
FAMILIES={'values':range(10),'gradient':range(10,22),'hessian':range(22,28),'full':range(28)}


def coeffs():
    phi={n:F(c) for n,c in enumerate(base.SHAPE) if c}
    ell={n+2:c/(n+2) for n,c in phi.items()}
    ell[0]=-sum(ell.values())
    P={n+2:c/(n+2) for n,c in ell.items()}
    P[0]=-sum(P.values())
    face={
        'p':{n+2:c/((n+3)*(n+2)) for n,c in phi.items()},
        'pL':{n+2:c/((n+3)*(n+2)) for n,c in ell.items()},
        'q':{n+2:c/(n+2) for n,c in ell.items()},
        'S':{n+2:c/(n+2) for n,c in phi.items()},
        'R':{n:c for n,c in phi.items() if n>=2}}
    return phi,ell,P,face,sum(c/(n+3) for n,c in phi.items()),sum(c/(n+3) for n,c in ell.items())


def literal(x):return f'T({float(x):.17g})'


def source(family='full'):
    if family not in FAMILIES:raise ValueError('unknown output family')
    phi,ell,P,face,a0,aL=coeffs()
    code=base.CODE.replace('SCALAR','double').replace('SUFFIX','')
    start=code.index('V grad=add(');code=code[:start]
    for i in range(4):code=code.replace(f'V p{i}=mul(sub(cv(LD(g.vertices,tet,{i})),cv(x)),ih);','')
    code=code.replace('auto point=[&](int i){return i==0?p0:i==1?p1:i==2?p2:p3;};','')
    code=code.replace('auto ids=LD(g.edges,edge);V v=point(ids[0]),uvec=point(ids[1]);',
        'auto ids=LD(g.edges,edge);\n V v=mul(sub(cv(LD(g.vertices,tet,ids[0])),cv(x)),ih),uvec=mul(sub(cv(LD(g.vertices,tet,ids[1])),cv(x)),ih);')
    declaration='T m0=0,mL=0; V m1{0,0,0},G{0,0,0};\n'
    for prefix in ('b','v','h'):declaration+='T '+','.join(prefix+s+'=0' for s in LABELS)+';\n'
    code=code.replace('T m0=0; V m1{0,0,0};',declaration)
    code=code.replace('struct Face {V normal;T d,a,a2,w,outside,apow,c_even,c_odd,p,q;};',
        'struct Face {T d,a,a2,w,outside,apow,c_even,c_odd,p,q,pL,S,R;};')
    code=code.replace('Face f{}; f.normal=cv(LD(g.normals,tet,face));','Face f{};V normal=cv(LD(g.normals,tet,face));')
    code=code.replace('dot(f.normal,off)','dot(normal,off)')
    tail='sign*f.w*(tail_theta/b-(f.a<T(1)?f.a:T(1))*f.outside)'
    code=code.replace('m0-=A0*'+tail+';',f'T tail={tail}; m0-={literal(a0)}*tail; mL-={literal(aL)}*tail;')
    # A face first moment is d*n*S plus the edge potential times mu.
    consume='''auto consume=[&](int face,int incidence,T d,T w,T q,T S,T R,T Ephi,T EL,T EP){
 V n=cv(LD(g.normals,tet,face)),mu=cv(LD(g.outward,tet,incidence));
 m1=add(m1,mul(n,w*q));G=add(G,mul(n,w*S));
'''
    for prefix,value,potential in [('b','q','EP'),('v','S','EL'),('h','R','Ephi')]:
        consume+=f'{{V U=add(mul(n,d*w*{value}),mul(mu,{potential}));\n'
        for label,(i,j) in zip(LABELS,PAIRS):
            a,b='xyz'[i],'xyz'[j]
            term=f'U.{a}*n.{b}' if i==j else f'T(.5)*(U.{a}*n.{b}+U.{b}*n.{a})'
            consume+=f'{prefix}{label}+={term};\n'
        consume+='}\n'
    consume+='};\n'
    code=code.replace('for(int edge=0;edge<6;++edge){',consume+'\nfor(int edge=0;edge<6;++edge){')
    # All three compact edge potentials use this same J stream. At c=0,
    # delta J_m is elementary; unlike occupancy their edge flux is nonzero.
    singular='''if(c2==T(0)){
 T lo=s0>T(-1)?s0:T(-1),hi=s1<T(1)?s1:T(1);
 if(hi>lo){T Ephi=0,EL=0,EP=0;T pl=1,pu=1;
'''
    for m in range(16):
        if m:singular+='pl*=fabs(lo);pu*=fabs(hi);\n'
        for name,cs in [('Ephi',phi),('EL',ell),('EP',P)]:
            if cs.get(m):singular+=f'{name}+={literal(cs[m]/(m+1))}*(hi*pu-lo*pl);\n'
    singular+='''consume(ids[2],2*edge,0,0,0,0,0,Ephi,EL,EP);
 consume(ids[3],2*edge+1,0,0,0,0,0,Ephi,EL,EP);}
 continue;
}'''
    code=code.replace('if(c2==T(0))continue;',singular)
    initial='T Ephi='+literal(phi.get(0,0))+'*j_even,EL='+literal(ell.get(0,0))+'*j_even,EP='+literal(P.get(0,0))+'*j_even;\n'
    code=code.replace('T rlpow=T(1),rupow=T(1);',initial+'T rlpow=T(1),rupow=T(1);')
    steps=[]
    for m in range(1,16):
        parity='odd' if m%2 else 'even'
        steps.append('{T jm2=j_'+parity+';')
        for f in (0,1):
            recurrence=f'jm2+f{f}.c_odd' if m==1 else f'jm2+f{f}.a2*f{f}.c_{parity}'
            steps.append(f'f{f}.c_{parity}={recurrence};f{f}.apow*=f{f}.a;')
            steps.append(f'T t{f}=f{f}.c_{parity}+(T(1)-f{f}.apow)*f{f}.outside;')
            for name,cs in face.items():
                if cs.get(m):steps.append(f'f{f}.{name}+={literal(cs[m])}*t{f};')
        steps.append('rlpow*=rl;rupow*=ru;')
        steps.append(f'j_{parity}=(u*rupow-l*rlpow+T({m})*c2*jm2)*{literal(F(1,m+1))};')
        for name,cs in [('Ephi',phi),('EL',ell),('EP',P)]:
            if cs.get(m):steps.append(f'{name}+={literal(cs[m])}*j_{parity};')
        steps.append('}')
    code=code.replace('STEPS','\n'.join(steps))
    code=code.replace('m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;', '''
 m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;
 mL+=f0.d*f0.w*f0.pL+f1.d*f1.w*f1.pL;
 consume(ids[2],2*edge,f0.d,f0.w,f0.q,f0.S,f0.R,Ephi,EL,EP);
 consume(ids[3],2*edge+1,f1.d,f1.w,f1.q,f1.S,f1.R,Ephi,EL,EP);
''')
    code=code.replace('m1=add(m1,add(mul(f0.normal,f0.w*f0.q),mul(f1.normal,f1.w*f1.q)));','')
    code+='T C=T(1365.0/(64.0*3.141592653589793238462643383279502884));\n'
    output=['C*m0']+[f'C*h*m1.{a}' for a in 'xyz']
    output += [f'C*h*h*(b{s}'+('-mL)' if i==j else ')') for s,(i,j) in zip(LABELS,PAIRS)]
    output += [f'-C*ih*G.{a}' for a in 'xyz']
    for i in range(3):
        for j in range(3):
            s='xyz'[min(i,j)]+'xyz'[max(i,j)]
            output.append(f'C*(m0-v{s})' if i==j else f'-C*v{s}')
    output += [f'C*ih*ih*h{s}' for s in LABELS]
    assert len(output)==28
    for out,index in enumerate(FAMILIES[family]):code+=f'LD(output,{out},row)={output[index]};\n'
    code+='#undef LD\nreturn T(0);\n'
    for name,offset in geometry.OFFSETS.items():code=code.replace(f'LD(g.{name},tet,',f'LD(g.data,tet,{offset}+')
    return code


@lru_cache(None)
def functions(family='full'):
    api=geometry.functions();vec,real,Geometry=api.vec,api.real,api.Geometry

    @wp.func_native(source(family))
    def interaction(x:vec,h:real,tet:int,output:wp.array2d(dtype=real),row:int,g:Geometry)->real: ...

    @wp.kernel
    def baseline_w6(particle:wp.array(dtype=vec),support:wp.array(dtype=real),
                    tet:wp.array(dtype=int),g:Geometry,output:wp.array2d(dtype=real)):
        i=wp.tid()
        ignored=interaction(particle[i],support[i],tet[i],output,i,g)

    return SimpleNamespace(**{**vars(api),'kernel':baseline_w6,'family':family,'channels':len(FAMILIES[family])})
