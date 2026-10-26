"""Experimental fused, closed-form W6/P1 particle–tetrahedron interaction.

Six physical edges share J interval recurrences across their two incident faces.
Each face streams C_m = delta K_m - a**m delta K_0, the combination required by
the flux formula. No quadrature, dynamic analytic DAG or per-pair scratch array.
Static tet geometry is prepared once, independently of particles and supports.
"""
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'max_unroll': 1})

SHAPE = [1, 0, -11, 0, 66, 0, -462, 1056, -1155, 704, -231, 32]
EDGES = [(i, j, *[k for k in range(4) if k not in (i, j)])
         for i in range(4) for j in range(i+1, 4)]


def coefficients():
    p, q = {}, {}
    a0 = sum(F(c, n+3) for n, c in enumerate(SHAPE))
    a1 = sum(F(c, n+2) for n, c in enumerate(SHAPE))
    for n, c in enumerate(SHAPE):
        if c:
            p[n+2] = F(c, (n+3)*(n+2))
            q[n+4] = F(c, (n+2)*(n+4))
    q[2] = q.get(2, F(0))-a1/2
    return p, q, a0


def source(precision):
    p, q, a0 = coefficients()
    def literal(value):
        return f'T({float(value):.17g})'
    # Emit fixed-degree algebra: zero coefficients disappear before compilation.
    # J and C keep only parity states. Radial coefficients are combined locally
    # into the occupancy flux and first-moment potential, not stored as S_n.
    steps = []
    for m in range(1, 16):
        parity = 'odd' if m % 2 else 'even'
        recurrence = 'jm2 + f{f}.c_odd' if m == 1 else 'jm2 + f{f}.a2*f{f}.c_'+parity
        steps.append(f'''
        T jm2 = j_{parity};
        f0.c_{parity} = {recurrence.format(f=0)};
        f1.c_{parity} = {recurrence.format(f=1)};
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_{parity} + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_{parity} + (T(1)-f1.apow)*f1.outside;
        ''')
        if p.get(m):
            steps.append(f'        f0.p += {literal(p[m])}*t0; f1.p += {literal(p[m])}*t1;')
        if q.get(m):
            steps.append(f'        f0.q += {literal(q[m])}*t0; f1.q += {literal(q[m])}*t1;')
        if m <= 13:
            steps.append(f'''
        rlpow *= rl; rupow *= ru;
        j_{parity} = (u*rupow-l*rlpow+T({m})*c2*jm2)*T({1/(m+1):.17g});
            ''')
    # Braces bound transient values to one degree, aiding register liveness.
    blocks = []
    current = ''
    for text in steps:
        if 'T jm2' in text:
            if current:
                blocks.append('{'+current+'\n}')
            current = text
        else:
            current += '\n'+text
    if current:
        blocks.append('{'+current+'\n}')
    return (CODE.replace('SCALAR', 'double' if precision == 'float64' else 'float')
        .replace('SUFFIX', '' if precision == 'float64' else 'f')
        .replace('A0', literal(a0)).replace('STEPS', '\n'.join(blocks)))


@lru_cache(None)
def functions(precision='float64'):
    if precision not in ('float64', 'float32'):
        raise ValueError('ordinary fp64 or experimental raw fp32 required')
    real = wp.float64 if precision == 'float64' else wp.float32
    vec = wp.types.vector(length=3, dtype=real)
    field_vec = wp.types.vector(length=4, dtype=real)

    @wp.struct
    class Geometry:
        vertices: wp.array2d(dtype=vec)
        normals: wp.array2d(dtype=vec)
        tangents: wp.array2d(dtype=vec)
        outward: wp.array2d(dtype=vec)
        gradients: wp.array2d(dtype=vec)
        edges: wp.array(dtype=wp.vec4i)

    @wp.func_native(source(precision))
    def interaction(x: vec, h: real, tet: int, field: field_vec, g: Geometry) -> real: ...

    @wp.kernel
    def p1_w6(particle: wp.array(dtype=vec), support: wp.array(dtype=real),
              tet: wp.array(dtype=int), field: wp.array(dtype=field_vec),
              geometry: Geometry, output: wp.array(dtype=real)):
        i = wp.tid()
        output[i] = interaction(particle[i], support[i], tet[i], field[i], geometry)

    return SimpleNamespace(real=real, vec=vec, field_vec=field_vec, Geometry=Geometry,
                           interaction=interaction, kernel=p1_w6)


def prepare_geometry(vertices, device='cpu', precision='float64'):
    """Particle-independent data for nondegenerate, straight tetrahedra."""
    api = functions(precision)
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 3 or v.shape[1:] != (4, 3) or not np.isfinite(v).all():
        raise ValueError('finite [tet,4,3] coordinates required')
    normals, tangents, outward, gradients = [], [], [], []
    for tet in v:
        inv = np.linalg.inv((tet[1:]-tet[0]).T)
        gradients.append(np.vstack([-inv.sum(axis=0), inv]))
        nf = []
        for opposite in range(4):
            ids = [j for j in range(4) if j != opposite]
            a, b, c = tet[ids]
            n = np.cross(b-a, c-a)
            if np.dot(n, tet[opposite]-a) > 0:
                ids[1], ids[2] = ids[2], ids[1]; n = -n
            nf.append(n/np.linalg.norm(n))
        normals.append(nf)
        ts, mus = [], []
        for i, j, fa, fb in EDGES:
            t = tet[j]-tet[i]; t /= np.linalg.norm(t); ts.append(t)
            for f in (fa, fb):
                mu = np.cross(t, nf[f])
                third = next(k for k in range(4) if k not in (i, j, f))
                if np.dot(mu, tet[third]-tet[i]) > 0:
                    mu = -mu
                mus.append(mu)
        tangents.append(ts); outward.append(mus)
    g = api.Geometry()
    for name, data in dict(vertices=v, normals=normals, tangents=tangents,
                           outward=outward, gradients=gradients).items():
        setattr(g, name, wp.array(np.asarray(data), dtype=api.vec, device=device))
    g.edges = wp.array(np.asarray(EDGES), dtype=wp.vec4i, device=device)
    return g


CODE = r'''
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
V p0=mul(sub(cv(LD(g.vertices,tet,0)),cv(x)),ih);
V p1=mul(sub(cv(LD(g.vertices,tet,1)),cv(x)),ih);
V p2=mul(sub(cv(LD(g.vertices,tet,2)),cv(x)),ih);
V p3=mul(sub(cv(LD(g.vertices,tet,3)),cv(x)),ih);
auto point=[&](int i){return i==0?p0:i==1?p1:i==2?p2:p3;};
// Regular flux: retain the exterior r^-3 tail on every face. No winding
// classification, separately subtracted solid angle, or face pruning.
T m0=0; V m1{0,0,0};
auto angle=[](T lo,T hi,T b){
    return hi>lo?atan2SUFFIX(b*(hi-lo),b*b+lo*hi)/b:T(0);
};
struct Face {V normal;T d,a,a2,w,outside,apow,c_even,c_odd,p,q;};
for(int edge=0;edge<6;++edge){
    auto ids=LD(g.edges,edge);V v=point(ids[0]),uvec=point(ids[1]);
    V tangent=cv(LD(g.tangents,tet,edge));
    T s0=dot(tangent,v),s1=dot(tangent,uvec);
    V off=sub(v,mul(tangent,s0)); T c2=dot(off,off);
    if(c2==T(0))continue; // Particle exactly on this edge line: w=0 limits.
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
        Face f{}; f.normal=cv(LD(g.normals,tet,face));
        f.d=dot(f.normal,off); f.a=fabsSUFFIX(f.d);f.a2=f.a*f.a;
        f.w=dot(cv(LD(g.outward,tet,incidence)),off);f.apow=T(1);
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
    T rlpow=T(1),rupow=T(1);
    STEPS
    m0+=f0.d*f0.w*f0.p+f1.d*f1.w*f1.p;
    m1=add(m1,add(mul(f0.normal,f0.w*f0.q),mul(f1.normal,f1.w*f1.q)));
}
V grad=add(add(mul(cv(LD(g.gradients,tet,1)),field[1]-field[0]),
              mul(cv(LD(g.gradients,tet,2)),field[2]-field[0])),
              mul(cv(LD(g.gradients,tet,3)),field[3]-field[0]));
T A=field[0]-dot(mul(grad,h),p0);
T result=T(1365.0/(64.0*3.141592653589793238462643383279502884))*
    (A*m0+dot(mul(grad,h),m1));
#undef LD
return result;
'''
