"""Fixed-profile solid-simplex native code generation.

The regular 3D flux seeds come from the validated shared-edge research body.
Only occupancy, gradient and first-gradient moment reductions are generated.
There is no request interpreter, FEM coefficient input or runtime moment DAG.
"""
from fractions import Fraction as F
from math import pi

TET_TEMPLATE = r'''
using T=SCALAR;
#define LD(...) (*wp::address(__VA_ARGS__))
struct V {T x,y,z;};
auto add=[](V a,V b){return V{a.x+b.x,a.y+b.y,a.z+b.z};};
auto sub=[](V a,V b){return V{a.x-b.x,a.y-b.y,a.z-b.z};};
auto mul=[](V a,T b){return V{a.x*b,a.y*b,a.z*b};};
auto dot=[](V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;};
auto cv=[](auto a){return V{a[0],a[1],a[2]};};
// Compensated log(1+z), also available in Warp's restricted CPU math runtime.
auto log_one_plus=[](T z){T y=T(1)+z;return y==T(1)?z:logSUFFIX(y)*(z/(y-T(1)));};
T ih=T(1)/h;





// Regular flux: retain the exterior r^-3 tail on every face. No winding
// classification, separately subtracted solid angle, or face pruning.
ACCUMULATORS
auto angle=[](T lo,T hi,T b){
    return hi>lo?atan2SUFFIX(b*(hi-lo),b*b+lo*hi)/b:T(0);
};
FACE_STRUCT
CONSUMER
for(int edge=0;edge<6;++edge){
    auto ids=LD(g.edges,edge);V v=mul(sub(cv(LD(g.data,tet,ids[0])),cv(x)),ih),uvec=mul(sub(cv(LD(g.data,tet,ids[1])),cv(x)),ih);
    V tangent=cv(LD(g.data,tet,8+edge));
    T s0=dot(tangent,v),s1=dot(tangent,uvec);
    V off=sub(v,mul(tangent,s0)); T c2=dot(off,off);
    SINGULAR // Particle exactly on this edge line: w=0 limits.
    T l=0,u=0,rl=T(1),ru=T(1);bool inside=false;
    if(c2<T(1)){
        T cut=sqrtSUFFIX(T(1)-c2);
        l=s0>-cut?s0:-cut;u=s1<cut?s1:cut;inside=u>l;
        // Use the same projected geometry in every seed, including vertices
        // whose exact radius is zero but whose edge projection rounded.
        if(inside){rl=sqrtSUFFIX(l*l+c2);ru=sqrtSUFFIX(u*u+c2);}
    }
    if(!inside){l=T(0);u=T(0);}
    auto theta=[&](T lo,T hi,T rlo,T rhi,T a,T b){
        if(hi<=lo || a==T(0))return T(0);
        T D=hi*rlo-lo*rhi;
        if(lo*hi>T(0))D=c2*(hi-lo)*(hi+lo)/(hi*rlo+lo*rhi);
        T c=sqrtSUFFIX(c2),an=a/c,bn=b/c;
        return atan2SUFFIX(an*bn*D,bn*bn*rhi*rlo+an*an*lo*hi);
    };
    auto face_data=[&](int face,int incidence){
        Face f{};V normal=cv(LD(g.data,tet,4+face));
        f.d=dot(normal,off); f.a=fabsSUFFIX(f.d);f.a2=f.a*f.a;
        f.w=dot(cv(LD(g.data,tet,14+incidence)),off);f.apow=T(1);
        if(f.w==T(0)){f.a=T(0);f.a2=T(0);f.d=T(0);return f;}
        T b=fabsSUFFIX(f.w),k0=inside?angle(l,u,b):T(0),E=0;
        f.outside=inside?angle(s0,l,b)+angle(u,s1,b):angle(s0,s1,b);
        T tail_theta=inside?theta(s0,l,sqrtSUFFIX(s0*s0+c2),rl,f.a,b)+theta(u,s1,ru,sqrtSUFFIX(s1*s1+c2),f.a,b):
            theta(s0,s1,sqrtSUFFIX(s0*s0+c2),sqrtSUFFIX(s1*s1+c2),f.a,b);
        T sign=f.d>T(0)?T(1):f.d<T(0)?T(-1):T(0);
        // Faces outside the ball have S_-3,1=0. Their regular full-face
        // flux therefore uses coefficient 1 here, rather than a.
        m0-=A0*sign*f.w*(tail_theta/b-(f.a<T(1)?f.a:T(1))*f.outside);
        if(f.a>=T(1)){f.w=T(0);f.a=T(0);f.a2=T(0);f.d=T(0);return f;}
        if(inside)E=theta(l,u,rl,ru,f.a,b)/b-k0;
        f.c_odd=f.a*E; // J_-1 is added after computing it once for both faces.
        return f;
    };
    Face f0=face_data(ids[2],2*edge),f1=face_data(ids[3],2*edge+1);
    if(f0.w==T(0) && f1.w==T(0))continue;
    T j_even=inside?u-l:T(0),j_odd=0;
    if(inside){
        T c=sqrtSUFFIX(c2);
        if(l>=T(0))j_odd=log_one_plus(((u-l)+(u-l)*(u+l)/(ru+rl))/(l+rl));
        else if(u<=T(0))j_odd=log_one_plus(((u-l)-(u-l)*(u+l)/(ru+rl))/(-u+ru));
        else j_odd=log_one_plus((u+u*u/(ru+c))/c)+log_one_plus((-l+l*l/(rl+c))/c);
    }
    // C_1=J_-1+a E; begin the parity recurrence with C_-1=E/a.
    // Its first update is handled explicitly to avoid the division by a.
    POTENTIAL
T rlpow=T(1),rupow=T(1);
    STEPS
    m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;
    CONSUME
}
'''


def polynomial(kernel):
    if kernel not in ('w2','w4'):raise ValueError('solid kernels support w2/w4')
    power,poly=(4,[F(1),F(4)]) if kernel=='w2' else (6,[F(1),F(6),F(35,3)])
    from math import comb
    out=[F(0)]*(power+len(poly))
    for i in range(power+1):
        for j,c in enumerate(poly):out[i+j]+=(-1)**i*comb(power,i)*c
    return {i:c for i,c in enumerate(out) if c}


def normalizer(dimension,phi):
    radial=sum(c/F(n+dimension) for n,c in phi.items())
    return float(1/((2 if dimension==2 else 4)*radial))/pi


def literal(value):return f'T({float(value):.17g})'


def common(precision):
    real,suffix=('float','f') if precision=='float32' else ('double','')
    return f'''using T={real};
#define LD(...) (*wp::address(__VA_ARGS__))
struct V {{T x,y,z;}};
auto add=[](V a,V b){{return V{{a.x+b.x,a.y+b.y,a.z+b.z}};}};
auto sub=[](V a,V b){{return V{{a.x-b.x,a.y-b.y,a.z-b.z}};}};
auto mul=[](V a,T b){{return V{{a.x*b,a.y*b,a.z*b}};}};
auto dot=[](V a,V b){{return a.x*b.x+a.y*b.y+a.z*b.z;}};
auto cv=[](auto a){{return V{{a[0],a[1],a[2]}};}};
auto log_one_plus=[](T z){{T y=T(1)+z;return y==T(1)?z:log{suffix}(y)*(z/(y-T(1)));}};
'''


def outputs(dimension,bundle,C):
    values=['C*m0']
    if bundle!='density':values += ['-C*ih*G.'+a for a in 'xyz'[:dimension]]
    if bundle=='lean':
        values += ['C*(m0-vxx)','-C*vxy']
        if dimension==2:values+=['C*(m0-vyy)']
        else:values+=['-C*vxz','C*(m0-vyy)','-C*vyz','C*(m0-vzz)']
    return f'T C={literal(C)};\n'+''.join(f'LD(output,{i},row)={v};\n' for i,v in enumerate(values))+'#undef LD\nreturn T(0);\n'


def tetrahedron(kernel,bundle,precision):
    phi=polynomial(kernel);last=max(phi)+2
    p={n+2:c/F((n+3)*(n+2)) for n,c in phi.items()}
    S={n+2:c/F(n+2) for n,c in phi.items()}
    A0=sum(c/F(n+3) for n,c in phi.items());A1=sum(S.values())
    code=TET_TEMPLATE.replace('SCALAR','float' if precision=='float32' else 'double').replace('SUFFIX','f' if precision=='float32' else '')
    acc='T m0=0;'
    if bundle!='density':acc+='V G{0,0,0};'
    if bundle=='lean':acc+='T vxx=0,vxy=0,vxz=0,vyy=0,vyz=0,vzz=0;'
    code=code.replace('ACCUMULATORS',acc)
    code=code.replace('FACE_STRUCT','struct Face {T d,a,a2,w,outside,apow,c_even,c_odd,p'+(',S' if bundle!='density' else '')+';};')
    code=code.replace('A0',literal(A0));consumer='';consume=''
    if bundle!='density':
        consumer='''auto consume=[&](int face,int incidence,T d,T S,T L){
 V n=cv(LD(g.data,tet,4+face));G=add(G,mul(n,S));
'''
        if bundle=='lean':consumer+='''V mu=cv(LD(g.data,tet,14+incidence));V U=add(mul(n,d*S),mul(mu,L));
 vxx+=U.x*n.x;vyy+=U.y*n.y;vzz+=U.z*n.z;
 vxy+=T(.5)*(U.x*n.y+U.y*n.x);vxz+=T(.5)*(U.x*n.z+U.z*n.x);vyz+=T(.5)*(U.y*n.z+U.z*n.y);
'''
        consumer+='};\n'
        consume='\n'.join(f'consume(ids[{2+i}],2*edge+{i},f{i}.d,f{i}.w*f{i}.S,'+('L' if bundle=='lean' else 'T(0)')+');' for i in (0,1))
    code=code.replace('CONSUMER',consumer).replace('CONSUME',consume)
    if bundle=='lean':
        singular='''if(c2==T(0)){
 T lo=s0>T(-1)?s0:T(-1),hi=s1<T(1)?s1:T(1);
 if(hi>lo){T L=-'''+literal(A1)+'''*(hi-lo);T pl=1,pu=1;
'''
        for m in range(1,last+1):
            singular+='pl*=fabs'+('f' if precision=='float32' else '')+'(lo);pu*=fabs'+('f' if precision=='float32' else '')+'(hi);\n'
            if m in S:singular+='L+='+literal(S[m]/F(m+1))+'*(hi*pu-lo*pl);\n'
        singular+='consume(ids[2],2*edge,0,0,L);consume(ids[3],2*edge+1,0,0,L);}\ncontinue;} '
    else:singular='if(c2==T(0))continue;'
    code=code.replace('SINGULAR',singular)
    code=code.replace('POTENTIAL','T L=-'+literal(A1)+'*j_even;' if bundle=='lean' else '')
    steps=[]
    for m in range(1,last+1):
        parity='odd' if m%2 else 'even';steps.append('{T jm2=j_'+parity+';')
        for f in (0,1):
            recurrence=f'jm2+f{f}.c_odd' if m==1 else f'jm2+f{f}.a2*f{f}.c_{parity}'
            steps.append(f'f{f}.c_{parity}={recurrence};f{f}.apow*=f{f}.a;T t{f}=f{f}.c_{parity}+(T(1)-f{f}.apow)*f{f}.outside;')
            if m in p:steps.append(f'f{f}.p+={literal(p[m])}*t{f};')
            if bundle!='density' and m in S:steps.append(f'f{f}.S+={literal(S[m])}*t{f};')
        if m<last or bundle=='lean':
            steps.append(f'rlpow*=rl;rupow*=ru;j_{parity}=(u*rupow-l*rlpow+T({m})*c2*jm2)*{literal(F(1,m+1))};')
        if bundle=='lean' and m in S:steps.append(f'L+={literal(S[m])}*j_{parity};')
        steps.append('}')
    return code.replace('STEPS','\n'.join(steps))+outputs(3,bundle,normalizer(3,phi))


def triangle(kernel,bundle,precision):
    phi=polynomial(kernel);suffix='f' if precision=='float32' else '';last=max(phi)
    code=common(precision)+'''T ih=T(1)/h,m0=0;
 V G{0,0,0};T vxx=0,vxy=0,vyy=0;
 for(int edge=0;edge<3;++edge){
 V v=mul(sub(cv(LD(g.data,tet,edge)),cv(x)),ih);
 V end=mul(sub(cv(LD(g.data,tet,(edge+1)%3)),cv(x)),ih);
 V n=cv(LD(g.data,tet,3+edge)),t=cv(LD(g.data,tet,6+edge));
 T s0=dot(v,t),s1=dot(end,t),w=dot(v,n),c2=w*w;
 T l=0,u=0,rl=1,ru=1;bool inside=false;
 if(c2<T(1)){T cut=SQRT(T(1)-c2);l=s0>-cut?s0:-cut;u=s1<cut?s1:cut;inside=u>l;
 if(inside){rl=SQRT(l*l+c2);ru=SQRT(u*u+c2);}}
 auto angle=[](T lo,T hi,T b){return hi>lo?ATAN2(b*(hi-lo),b*b+lo*hi)/b:T(0);};
 T E=0,F=0,B=0;
 if(inside){
  if(c2==T(0)){
'''
    code+='T pl0=1,pu0=1;\n'
    for n in range(last+1):
        if n:code+='pl0*=ABS(l);pu0*=ABS(u);\n'
        if n not in phi:continue
        j=f'(u*pu0-l*pl0)*{literal(F(1,n+1))}'
        if bundle!='density':code+=f'E+={literal(phi[n])}*{j};\n'
        code+=f'F+={literal(phi[n]/F(n+2))}*{j};\n'
    code+='''}else{
 T j_even=u-l,j_odd=0,c=SQRT(c2);
 if(l>=T(0))j_odd=log_one_plus(((u-l)+(u-l)*(u+l)/(ru+rl))/(l+rl));
 else if(u<=T(0))j_odd=log_one_plus(((u-l)-(u-l)*(u+l)/(ru+rl))/(-u+ru));
 else j_odd=log_one_plus((u+u*u/(ru+c))/c)+log_one_plus((-l+l*l/(rl+c))/c);
 T pl=1,pu=1;
'''
    for m in range(last+1):
        parity='even' if m%2==0 else 'odd'
        if m:
            code+='pl*=rl;pu*=ru;\n'
            code+=f'j_{parity}=(u*pu-l*pl+T({m})*c2*j_{parity})*{literal(F(1,m+1))};\n'
        if m in phi:
            if bundle!='density':code+=f'E+={literal(phi[m])}*j_{parity};\n'
            code+=f'F+={literal(phi[m]/F(m+2))}*j_{parity};\n'
    code+='}\n'
    if bundle=='lean':
        code+='T bl=rl,bu=ru;\n'
        for m in range(2,last+3):
            code+='bl*=rl;bu*=ru;\n'
            if m-2 in phi:code+=f'B+={literal(phi[m-2]/F(m))}*(bu-bl);\n'
    code+='}\n'
    A1=sum(c/F(n+2) for n,c in phi.items())
    code+='''if(w!=T(0)){
 T b=ABS(w),outside=inside?angle(s0,l,b)+angle(u,s1,b):angle(s0,s1,b);
 m0+=w*(F+'''+literal(A1)+'''*outside);
 }
'''
    if bundle!='density':code+='G=add(G,mul(n,E));\n'
    if bundle=='lean':code+='''V U=add(mul(n,w*E),mul(t,B));
 vxx+=U.x*n.x;vyy+=U.y*n.y;vxy+=T(.5)*(U.x*n.y+U.y*n.x);
'''
    code+='}\n'
    for name,fun in [('SQRT','sqrt'),('ABS','fabs'),('ATAN2','atan2'),('POW','pow')]:code=code.replace(name,fun+suffix)
    return code+outputs(2,bundle,normalizer(2,phi))
