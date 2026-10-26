
using T=double;
#define LD(...) (*wp::address(__VA_ARGS__))
struct V {T x,y,z;};
auto add=[](V a,V b){return V{a.x+b.x,a.y+b.y,a.z+b.z};};
auto sub=[](V a,V b){return V{a.x-b.x,a.y-b.y,a.z-b.z};};
auto mul=[](V a,T b){return V{a.x*b,a.y*b,a.z*b};};
auto dot=[](V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;};
auto cv=[](auto a){return V{a[0],a[1],a[2]};};
// Compensated log(1+z), also available in Warp's restricted CPU math runtime.
auto log_one_plus=[](T z){T y=T(1)+z;return y==T(1)?z:log(y)*(z/(y-T(1)));};
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
    return hi>lo?atan2(b*(hi-lo),b*b+lo*hi)/b:T(0);
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
        T cut=sqrt(T(1)-c2);
        l=s0>-cut?s0:-cut;u=s1<cut?s1:cut;inside=u>l;
        // Use the same projected geometry in every seed, including vertices
        // whose exact radius is zero but whose edge projection rounded.
        if(inside){rl=sqrt(l*l+c2);ru=sqrt(u*u+c2);}
    }
    if(!inside){l=T(0);u=T(0);}
    auto theta=[&](T lo,T hi,T rlo,T rhi,T a,T b){
        if(hi<=lo || a==T(0))return T(0);
        T D=hi*rlo-lo*rhi;
        if(lo*hi>T(0))D=c2*(hi-lo)*(hi+lo)/(hi*rlo+lo*rhi);
        T c=sqrt(c2),an=a/c,bn=b/c;
        return atan2(an*bn*D,bn*bn*rhi*rlo+an*an*lo*hi);
    };
    auto face_data=[&](int face,int incidence){
        Face f{}; f.normal=cv(LD(g.normals,tet,face));
        f.d=dot(f.normal,off); f.a=fabs(f.d);f.a2=f.a*f.a;
        f.w=dot(cv(LD(g.outward,tet,incidence)),off);f.apow=T(1);
        if(f.w==T(0)){f.a=T(0);f.a2=T(0);f.d=T(0);return f;}
        T b=fabs(f.w),k0=inside?angle(l,u,b):T(0),E=0;
        f.outside=inside?angle(s0,l,b)+angle(u,s1,b):angle(s0,s1,b);
        T tail_theta=inside?theta(s0,l,sqrt(s0*s0+c2),rl,f.a,b)+theta(u,s1,ru,sqrt(s1*s1+c2),f.a,b):
            theta(s0,s1,sqrt(s0*s0+c2),sqrt(s1*s1+c2),f.a,b);
        T sign=f.d>T(0)?T(1):f.d<T(0)?T(-1):T(0);
        // Faces outside the ball have S_-3,1=0. Their regular full-face
        // flux therefore uses coefficient 1 here, rather than a.
        m0-=T(0.011721611721611722)*sign*f.w*(tail_theta/b-(f.a<T(1)?f.a:T(1))*f.outside);
        if(f.a>=T(1)){f.w=T(0);f.a=T(0);f.a2=T(0);f.d=T(0);return f;}
        if(inside)E=theta(l,u,rl,ru,f.a,b)/b-k0;
        f.c_odd=f.a*E; // J_-1 is added after computing it once for both faces.
        return f;
    };
    Face f0=face_data(ids[2],2*edge),f1=face_data(ids[3],2*edge+1);
    if(f0.w==T(0) && f1.w==T(0))continue;
    T j_even=inside?u-l:T(0),j_odd=0;
    if(inside){
        T c=sqrt(c2);
        if(l>=T(0))j_odd=log_one_plus(((u-l)+(u-l)*(u+l)/(ru+rl))/(l+rl));
        else if(u<=T(0))j_odd=log_one_plus(((u-l)-(u-l)*(u+l)/(ru+rl))/(-u+ru));
        else j_odd=log_one_plus((u+u*u/(ru+c))/c)+log_one_plus((-l+l*l/(rl+c))/c);
    }
    // C_1=J_-1+a E; begin the parity recurrence with C_-1=E/a.
    // Its first update is handled explicitly to avoid the division by a.
    T rlpow=T(1),rupow=T(1);
    {
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.c_odd;
        f1.c_odd = jm2 + f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(1)*c2*jm2)*T(0.5);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(0.16666666666666666)*t0; f1.p += T(0.16666666666666666)*t1;
        f0.q += T(-0.022435897435897436)*t0; f1.q += T(-0.022435897435897436)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(2)*c2*jm2)*T(0.33333333333333331);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(3)*c2*jm2)*T(0.25);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(-0.55000000000000004)*t0; f1.p += T(-0.55000000000000004)*t1;
        f0.q += T(0.125)*t0; f1.q += T(0.125)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(4)*c2*jm2)*T(0.20000000000000001);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(5)*c2*jm2)*T(0.16666666666666666);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(1.5714285714285714)*t0; f1.p += T(1.5714285714285714)*t1;
        f0.q += T(-0.45833333333333331)*t0; f1.q += T(-0.45833333333333331)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(6)*c2*jm2)*T(0.14285714285714285);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(7)*c2*jm2)*T(0.125);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(-6.416666666666667)*t0; f1.p += T(-6.416666666666667)*t1;
        f0.q += T(1.375)*t0; f1.q += T(1.375)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(8)*c2*jm2)*T(0.1111111111111111);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(11.733333333333333)*t0; f1.p += T(11.733333333333333)*t1;

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(9)*c2*jm2)*T(0.10000000000000001);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(-10.5)*t0; f1.p += T(-10.5)*t1;
        f0.q += T(-5.7750000000000004)*t0; f1.q += T(-5.7750000000000004)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(10)*c2*jm2)*T(0.090909090909090912);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(5.333333333333333)*t0; f1.p += T(5.333333333333333)*t1;
        f0.q += T(10.666666666666666)*t0; f1.q += T(10.666666666666666)*t1;

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(11)*c2*jm2)*T(0.083333333333333329);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(-1.4807692307692308)*t0; f1.p += T(-1.4807692307692308)*t1;
        f0.q += T(-9.625)*t0; f1.q += T(-9.625)*t1;

        rlpow *= rl; rupow *= ru;
        j_even = (u*rupow-l*rlpow+T(12)*c2*jm2)*T(0.076923076923076927);
            
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        
        f0.p += T(0.17582417582417584)*t0; f1.p += T(0.17582417582417584)*t1;
        f0.q += T(4.9230769230769234)*t0; f1.q += T(4.9230769230769234)*t1;

        rlpow *= rl; rupow *= ru;
        j_odd = (u*rupow-l*rlpow+T(13)*c2*jm2)*T(0.071428571428571425);
            
}
{
        T jm2 = j_even;
        f0.c_even = jm2 + f0.a2*f0.c_even;
        f1.c_even = jm2 + f1.a2*f1.c_even;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_even + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_even + (T(1)-f1.apow)*f1.outside;
        
        f0.q += T(-1.375)*t0; f1.q += T(-1.375)*t1;
}
{
        T jm2 = j_odd;
        f0.c_odd = jm2 + f0.a2*f0.c_odd;
        f1.c_odd = jm2 + f1.a2*f1.c_odd;
        f0.apow *= f0.a; f1.apow *= f1.a;
        T t0 = f0.c_odd + (T(1)-f0.apow)*f0.outside;
        T t1 = f1.c_odd + (T(1)-f1.apow)*f1.outside;
        
        f0.q += T(0.1641025641025641)*t0; f1.q += T(0.1641025641025641)*t1;
}
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
