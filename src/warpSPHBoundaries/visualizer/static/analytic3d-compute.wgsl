// Closed-form analytic polyhedra. All instructions are f32. Paired f32 values
// retain cancellation residuals in high-order kernel/primitive combinations.
// No f64 feature, quadrature, sampled occupancy, or CPU integration is used.
// ids: grid size, field, face/series count, profile count/cubic flag.
// grid: paired axis origin and step. values: h, whole value/R, dimension, mode.
// mode: >=0 solid/polygon, -1 series, -2 half-plane, -3 curvature.
// origin: paired y/z starts; local modes reuse z/w for W-count and order.
struct Settings { ids:vec4u, grid:vec4f, values:vec4f, origin:vec4f }
@group(0) @binding(0) var<uniform> cfg:Settings;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
alias D=vec2f;
struct V { x:D,y:D,z:D }
struct Primitive { I:array<D,17>, K:array<D,17>, minus:D }
fn d(x:f32)->D{return D(x,0);}
// frexp/ldexp round intermediate results explicitly; algebraic reassociation
// otherwise erases TwoSum/Dekker residuals on some valid WGSL implementations.
fn rounded(x:f32)->f32{let f=frexp(x);return ldexp(f.fract,f.exp);}
fn norm(a:D)->D{let s=rounded(a.x+a.y);return D(s,rounded(a.y-rounded(s-a.x)));}
fn add(a:D,b:D)->D{let s=rounded(a.x+b.x);let v=rounded(s-a.x);let e=rounded(rounded(rounded(a.x-rounded(s-v))+rounded(b.x-v))+rounded(a.y+b.y));return norm(D(s,e));}
fn neg(a:D)->D{return -a;}
fn sub(a:D,b:D)->D{return add(a,-b);}
fn mul(a:D,b:D)->D{
 let p=rounded(a.x*b.x);let ca=rounded(4097.0*a.x);let ah=rounded(ca-rounded(ca-a.x));let al=rounded(a.x-ah);
 let cb=rounded(4097.0*b.x);let bh=rounded(cb-rounded(cb-b.x));let bl=rounded(b.x-bh);
 let error=rounded(rounded(rounded(rounded(ah*bh)-p)+rounded(ah*bl))+rounded(al*bh));
 let e=rounded(rounded(rounded(error+rounded(al*bl))+rounded(a.x*b.y))+rounded(a.y*b.x));return norm(D(p,e));
}
fn div(a:D,b:D)->D{let q=a.x/b.x;let rem=sub(a,mul(b,d(q)));return add(d(q),d((rem.x+rem.y)/b.x));}
fn sqrtD(a:D)->D{if a.x<=0.0{return d(0);}let x=sqrt(a.x);return add(d(x),div(sub(a,mul(d(x),d(x))),d(2.0*x)));}
fn absD(a:D)->D{if a.x<0.0{return -a;}return a;}
fn power(a:D,n:i32)->D{var v=d(1);for(var j=0;j<abs(n);j++){v=mul(v,a);}if n<0{return div(d(1),v);}return v;}
const DPI=D(3.1415927410125732,-8.742278000372485e-8);
fn atanD(a:D)->D{
 var x=absD(a);let inverse=x.x>1.0;if inverse{x=div(d(1),x);}
 // Two half-angle reductions bound the alternating series by |x|<0.2.
 for(var i=0;i<2;i++){x=div(x,add(d(1),sqrtD(add(d(1),mul(x,x)))));}
 let xx=mul(x,x);var term=x;var sum=x;
 for(var i=1;i<15;i++){term=neg(mul(term,xx));sum=add(sum,div(term,d(f32(2*i+1))));}
 sum=mul(d(4),sum);if inverse{sum=sub(mul(d(0.5),DPI),sum);}if a.x<0.0{return -sum;}return sum;
}
fn atan2D(y:D,x:D)->D{if x.x==0.0{if y.x==0.0{return d(0);}return mul(d(select(-0.5,0.5,y.x>0.0)),DPI);}var a=atanD(div(y,x));if x.x<0.0{a=add(a,mul(d(select(-1.0,1.0,y.x>=0.0)),DPI));}return a;}
fn logD(a:D)->D{
 let f=frexp(a.x);let scale=exp2(-f32(f.exp));let u=mul(a,d(scale));let y=div(sub(u,d(1)),add(u,d(1)));let yy=mul(y,y);var term=y;var sum=y;
 for(var j=1;j<22;j++){term=mul(term,yy);sum=add(sum,div(term,d(f32(2*j+1))));}
 return add(mul(d(2),sum),mul(d(f32(f.exp)),D(0.6931471824645996,-1.904654323148236e-9)));
}
fn asinhD(a:D)->D{if a.x==0.0{return d(0);}let x=absD(a);let v=logD(add(x,sqrtD(add(d(1),mul(x,x)))));if a.x<0.0{return -v;}return v;}
fn vzero()->V{return V(d(0),d(0),d(0));}
fn vadd(a:V,b:V)->V{return V(add(a.x,b.x),add(a.y,b.y),add(a.z,b.z));}
fn vsub(a:V,b:V)->V{return V(sub(a.x,b.x),sub(a.y,b.y),sub(a.z,b.z));}
fn scale(a:V,b:D)->V{return V(mul(a.x,b),mul(a.y,b),mul(a.z,b));}
fn dotD(a:V,b:V)->D{return add(add(mul(a.x,b.x),mul(a.y,b.y)),mul(a.z,b.z));}
fn crossD(a:V,b:V)->V{return V(sub(mul(a.y,b.z),mul(a.z,b.y)),sub(mul(a.z,b.x),mul(a.x,b.z)),sub(mul(a.x,b.y),mul(a.y,b.x)));}
fn component(a:V,i:u32)->D{if i==0u{return a.x;}if i==1u{return a.y;}return a.z;}
fn native(a:V)->vec3f{return vec3f(a.x.x,a.y.x,a.z.x);}
fn vertex(f:u32,i:u32)->V{let j=cfg.ids.w*6u+f*18u+i*6u;return V(D(table[j],table[j+1u]),D(table[j+2u],table[j+3u]),D(table[j+4u],table[j+5u]));}
fn primitive(s:D,w:D,a:D)->Primitive{
 let b=absD(w);let cc=add(mul(w,w),mul(a,a));let c=sqrtD(cc);let r=sqrtD(add(mul(s,s),cc));var P:Primitive;
 P.I[0]=s;var tail=d(0);if c.x!=0.0{tail=mul(cc,asinhD(div(s,c)));}P.I[1]=mul(d(0.5),add(mul(s,r),tail));
 var rp=r;for(var m=2;m<17;m++){rp=mul(rp,r);P.I[m]=div(add(mul(s,rp),mul(mul(d(f32(m)),cc),P.I[m-2])),d(f32(m+1)));}
 if w.x==0.0{return P;}
 if a.x==0.0{P.minus=div(s,mul(mul(b,b),r));}else{P.minus=div(atanD(div(mul(a,s),mul(b,r))),mul(a,b));}
 P.K[0]=div(atanD(div(s,b)),b);P.K[1]=add(asinhD(div(s,c)),mul(mul(a,a),P.minus));
 for(var m=2;m<17;m++){P.K[m]=add(P.I[m-2],mul(mul(a,a),P.K[m-2]));}return P;
}
// Radial profiles: regular volume flux (0), compact potential (1), W/L (2).
// Area models use (1) for W/L and (2) for the second-moment compact potential.
struct Profiles { scalar:array<D,3>,first:array<V,3>, second:array<V,3> }
fn integrateFace(vertices:array<V,3>,normal:V,z:D)->Profiles{
 let a=absD(z);var S:Profiles;
 for(var edge=0u;edge<3u;edge++){
  let start=vertices[edge];let delta=vsub(vertices[(edge+1u)%3u],start);let len=sqrtD(dotD(delta,delta));let t=scale(delta,div(d(1),len));let mu=crossD(t,normal);
  let w=dotD(start,mu);let lo=dotD(start,t);let hi=add(lo,len);let c2=add(mul(w,w),mul(a,a));var cuts:array<D,6>;cuts[0]=lo;var count=1u;
  for(var knot=0u;knot<2u;knot++){let R=d(select(1.0,0.5,knot==0u));if R.x*R.x>c2.x{let L=sqrtD(sub(mul(R,R),c2));for(var sign=0u;sign<2u;sign++){let x=mul(d(select(-1.0,1.0,sign==1u)),L);if x.x>lo.x && x.x<hi.x{var j=count;loop{if j==0u{break;}if cuts[j-1u].x<x.x{break;}cuts[j]=cuts[j-1u];j--;}cuts[j]=x;count++;}}}}
  cuts[count]=hi;count++;
  for(var panel=1u;panel<count;panel++){
   let l=cuts[panel-1u];let u=cuts[panel];let mid=mul(d(0.5),add(l,u));let r2=add(mul(mid,mid),c2);let A=primitive(l,w,a);let B=primitive(u,w,a);var k0=d(0);
   if w.x!=0.0{k0=div(atan2D(mul(absD(w),sub(u,l)),add(mul(w,w),mul(l,u))),absD(w));}
   for(var k=0u;k<cfg.ids.w;k++){
    let n=i32(table[k*6u]);let R=d(table[k*6u+1u]);let coeff=D(table[k*6u+2u],table[k*6u+3u]);let kind=u32(table[k*6u+4u]);
    if kind==0u && z.x==0.0{continue;}if R.x>0.0 && a.x>=R.x{continue;}
    let inside=R.x==0.0 || r2.x<R.x*R.x;let exponent=n+2;var value=d(0);
    if w.x!=0.0{
     if inside{var basis=d(0);if n== -3{basis=sub(B.minus,A.minus);}else{basis=sub(B.K[exponent],A.K[exponent]);}value=sub(basis,mul(power(a,exponent),k0));}
     else{value=mul(sub(power(R,exponent),power(a,exponent)),k0);}
     S.scalar[kind]=add(S.scalar[kind],mul(coeff,div(mul(w,value),d(f32(exponent)))));
    }
    if kind>0u && inside && (cfg.values.z>2.5 || kind==1u){
     let line=sub(sub(B.I[exponent],A.I[exponent]),mul(power(R,exponent),sub(u,l)));
     S.first[kind]=vadd(S.first[kind],scale(mu,mul(coeff,div(line,d(f32(exponent))))));
     if cfg.values.z<2.5 && ((cfg.ids.y>=3u && cfg.ids.y<=5u) || (cfg.ids.y>=8u && cfg.ids.y<=11u)){
      let ru=sqrtD(add(mul(u,u),c2));let rl=sqrtD(add(mul(l,l),c2));let J=sub(div(sub(power(ru,n+4),power(rl,n+4)),d(f32(n+4))),mul(power(R,exponent),mul(d(0.5),sub(mul(u,u),mul(l,l)))));
      let moment=vadd(scale(mu,mul(w,line)),scale(t,J));let factor=div(coeff,d(f32(exponent)));
      S.second[kind]=vadd(S.second[kind],V(mul(factor,mul(mu.x,moment.x)),mul(factor,mul(mu.x,moment.y)),mul(factor,mul(mu.y,moment.y))));
     }
    }
   }
  }
 }
 for(var kind=1u;kind<3u;kind++){S.first[kind]=vadd(S.first[kind],scale(normal,mul(z,S.scalar[kind])));}return S;
}
fn pair(i:u32)->vec2u{return array<vec2u,6>(vec2u(0,0),vec2u(0,1),vec2u(0,2),vec2u(1,1),vec2u(1,2),vec2u(2,2))[i];}
fn distancePoint(p:vec3f)->f32{
 if cfg.values.z<2.5{
  let base=cfg.ids.w*6u+cfg.ids.z*18u;var inside=false;var best=1e30;
  for(var j=0u;j<u32(cfg.values.w);j++){let k=base+j*12u;let a=vec2f(table[k],table[k+2u]);let b=vec2f(table[k+6u],table[k+8u]);let e=b-a;let t=clamp(dot(p.xy-a,e)/dot(e,e),0.0,1.0);best=min(best,length(p.xy-a-t*e));if (a.y>p.y)!=(b.y>p.y){if p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x{inside=!inside;}}}return select(best,-best,inside);
 }
 var contained=true;var best=1e30;
 for(var f=0u;f<cfg.ids.z;f++){
  let a=native(vertex(f,0u));let b=native(vertex(f,1u));let c=native(vertex(f,2u));let u=b-a;let v=c-a;let n=normalize(cross(u,v));let q=p-a;let z=dot(q,n);contained=contained && z<=0.0;
  let uu=dot(u,u);let uv=dot(u,v);let vv=dot(v,v);let ru=dot(q,u);let rv=dot(q,v);let alpha=(vv*ru-uv*rv)/(uu*vv-uv*uv);let beta=(uu*rv-uv*ru)/(uu*vv-uv*uv);
  if alpha>=0.0 && beta>=0.0 && alpha+beta<=1.0{best=min(best,z*z);}
  let starts=array<vec3f,3>(a,b,c);let ends=array<vec3f,3>(b,c,a);
  for(var e=0u;e<3u;e++){let d=ends[e]-starts[e];let t=clamp(dot(p-starts[e],d)/dot(d,d),0.0,1.0);let y=p-starts[e]-t*d;best=min(best,dot(y,y));}
 }return select(sqrt(best),-sqrt(best),contained);
}
struct Local { distance:D, normal:V, curvature:D, normalValid:bool,curvatureValid:bool,geometryDistance:f32 }
fn localAt(index:u32)->Local{let k=cfg.ids.w*6u+u32(cfg.origin.z)*4u+index*12u;return Local(D(table[k],table[k+1u]),V(D(table[k+2u],table[k+3u]),D(table[k+4u],table[k+5u]),d(0)),D(table[k+6u],table[k+7u]),table[k+8u]>.5,table[k+9u]>.5,table[k+10u]);}
fn localCorrection(j:Local)->V{
 let a=absD(j.distance);var B:array<D,3>;
 for(var q=0u;q<u32(cfg.origin.z);q++){let k=cfg.ids.w*6u+q*4u;let n=i32(table[k]);let R=d(table[k+1u]);if a.x>=R.x{continue;}let coefficient=D(table[k+2u],table[k+3u]);let top=sqrtD(sub(mul(R,R),mul(a,a)));let P=primitive(top,d(0),a);let a2=mul(a,a);let factor=mul(d(2),coefficient);B[0]=add(B[0],mul(factor,P.I[n]));B[1]=add(B[1],mul(factor,sub(P.I[n+2],mul(a2,P.I[n]))));B[2]=add(B[2],mul(factor,add(sub(P.I[n+4],mul(mul(d(2),a2),P.I[n+2])),mul(mul(a2,a2),P.I[n]))));}
 var k=j.curvature;if cfg.origin.w<1.0{k=d(0);}var k2=d(0);if cfg.origin.w>=2.0{k2=mul(k,k);}let z=j.distance;let z2=mul(z,z);let z3=mul(z2,z);
 let mass=add(mul(d(-.5),mul(k,B[1])),mul(d(.375),mul(mul(z,k2),B[1])));
 let mn=add(mul(d(.5),mul(mul(k,z),B[1])),mul(d(.125),mul(k2,sub(B[2],mul(mul(d(3),z2),B[1])))));
 let T=add(mul(d(-.5),mul(k,B[2])),mul(d(.625),mul(mul(z,k2),B[2])));
 let N=add(mul(d(-.5),mul(mul(k,z2),B[1])),mul(d(.125),mul(k2,add(mul(d(-2),mul(z,B[2])),mul(d(3),mul(z3,B[1]))))));
 let G=add(mul(d(.5),mul(mul(k,z),B[0])),mul(d(.375),mul(k2,sub(B[1],mul(z2,B[0])))));
 let Ct=add(mul(d(-1.5),mul(k,B[1])),mul(d(1.875),mul(mul(z,k2),B[1])));
 let Cn=add(mul(d(-.5),mul(mul(k,z2),B[0])),mul(d(.125),mul(k2,add(mul(d(-6),mul(z,B[1])),mul(d(3),mul(z3,B[0]))))));
 let id=cfg.ids.y;let normal=j.normal;
 if id==0u{return V(mass,d(0),d(0));}if id<=2u{return V(mul(mn,component(normal,id-1u)),d(0),d(0));}
 if id<=5u{let ij=array<vec2u,3>(vec2u(0,0),vec2u(0,1),vec2u(1,1))[id-3u];var v=mul(sub(N,T),mul(component(normal,ij.x),component(normal,ij.y)));if ij.x==ij.y{v=add(v,T);}return V(v,d(0),d(0));}
 if id<=7u{return V(mul(G,component(normal,id-6u)),d(0),d(0));}
 if id<=11u{let i=(id-8u)/2u;let n=(id-8u)%2u;var v=mul(sub(Cn,Ct),mul(component(normal,i),component(normal,n)));if i==n{v=add(v,Ct);}return V(v,d(0),d(0));}
 // For the norm return the complete axial gradient, including its plane term.
 return scale(normal,sub(G,B[0]));
}
@compute @workgroup_size(64)
fn compute_analytic3d(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x;let n=cfg.ids.x;let area=cfg.values.z<2.5;let count=select(n*n*n,n*n,area);if index>=count{return;}let ids=vec3u(index%n,(index/n)%n,index/(n*n));let start=D(cfg.grid.x,cfg.grid.y);let step=D(cfg.grid.z,cfg.grid.w);var p=V(add(start,mul(step,d(f32(ids.x)))),add(D(cfg.origin.x,cfg.origin.y),mul(step,d(f32(ids.y)))),add(D(cfg.origin.z,cfg.origin.w),mul(step,d(f32(ids.z)))));
 if cfg.values.w== -1.0{
  let q=sqrtD(dotD(p,p));let a=cfg.values.y;let distance=(q.x-a)*cfg.values.x;
  if q.x-a>=1.0{output[index]=vec4f(0,distance,1,0);return;}
  if a>0.3 || q.x<=a || q.x+a>=1.0 || (cfg.ids.w==1u && q.x-a<=0.5 && q.x+a>=0.5){output[index]=vec4f(0,distance,0,0);return;}
  var sum:array<D,3>;for(var j=0u;j<cfg.ids.z;j++){let k=j*8u;let R=table[k+4u];if q.x>=R{continue;}var value=power(q,i32(table[k+3u]));value=mul(value,power(neg(p.x),i32(table[k])));value=mul(value,power(neg(p.y),i32(table[k+1u])));value=mul(value,power(neg(p.z),i32(table[k+2u])));let component=u32(table[k+7u]);sum[component]=add(sum[component],mul(value,D(table[k+5u],table[k+6u])));}
  var value=sum[0].x+sum[0].y;if (area && cfg.ids.y==12u) || (!area && cfg.ids.y==29u){value=length(vec3f(sum[0].x,sum[1].x,sum[2].x));}output[index]=vec4f(value,distance,1,0);return;
 }
 let local=cfg.values.w<= -2.0;var j=Local(d(0),vzero(),d(0),true,true,0);var distance=0.0;
 if local{j=localAt(index);distance=j.geometryDistance;if j.distance.x>=1.0{output[index]=vec4f(0,distance*cfg.values.x,1,0);return;}if j.distance.x<= -1.0{output[index]=vec4f(cfg.values.y,distance*cfg.values.x,1,0);return;}if (!j.normalValid && cfg.ids.y!=0u) || (!j.curvatureValid && cfg.values.w== -3.0){output[index]=vec4f(0,distance*cfg.values.x,0,0);return;}p=vzero();}else{distance=distancePoint(native(p));}if !local && distance>1.000002{output[index]=vec4f(0,distance*cfg.values.x,1,0);return;}if !local && distance< -1.000002{output[index]=vec4f(cfg.values.y,distance*cfg.values.x,1,0);return;}
 let id=cfg.ids.y;var total=d(0);var gradient=vzero();
 for(var f=0u;f<cfg.ids.z;f++){
  var vertices:array<V,3>;
  if local{let t=V(j.normal.y,neg(j.normal.x),d(0));let a=vadd(scale(t,d(-4)),scale(j.normal,d(-4)));let b=vadd(scale(t,d(4)),scale(j.normal,d(-4)));let c=vadd(scale(t,d(4)),scale(j.normal,neg(j.distance)));let e=vadd(scale(t,d(-4)),scale(j.normal,neg(j.distance)));if f==0u{vertices=array<V,3>(a,b,c);}else{vertices=array<V,3>(a,c,e);}}
  else{vertices=array<V,3>(vsub(vertex(f,0u),p),vsub(vertex(f,1u),p),vsub(vertex(f,2u),p));}let N=crossD(vsub(vertices[1],vertices[0]),vsub(vertices[2],vertices[0]));let normal=scale(N,div(d(1),sqrtD(dotD(N,N))));let z=dotD(vertices[0],normal);let S=integrateFace(vertices,normal,z);var value=d(0);
  if area{
   let Q=V(sub(S.second[1].x,S.scalar[2]),S.second[1].y,sub(S.second[1].z,S.scalar[2]));
   if id==0u{value=S.scalar[1];}else if id<=2u{value=component(S.first[1],id-1u);}else if id<=5u{value=component(Q,id-3u);}else if id<=7u{value=neg(component(S.first[1],id-6u));}else if id<=11u{value=neg(component(Q,select(select(1u,0u,id==8u),2u,id==11u)));}else{gradient=vsub(gradient,S.first[1]);}
  }else if id==0u{value=mul(z,S.scalar[0]);}
  else if id<=3u{value=mul(component(normal,id-1u),S.scalar[1]);}
  else if id<=9u{let ij=pair(id-4u);value=mul(component(normal,ij.x),component(S.first[1],ij.y));if ij.x==ij.y{value=add(value,mul(z,S.scalar[0]));}}
  else if id<=12u{value=neg(mul(component(normal,id-10u),S.scalar[2]));}
  else if id<=21u{let i=(id-13u)/3u;let j=(id-13u)%3u;value=neg(mul(component(normal,j),component(S.first[2],i)));if i==j{value=add(value,mul(z,S.scalar[0]));}}
  else if id<=27u{let ij=pair(id-22u);value=mul(component(normal,ij.x),component(S.first[2],ij.y));}
  else if id==28u{value=mul(z,S.scalar[2]);}
  else{gradient=vadd(gradient,scale(normal,neg(S.scalar[2])));}total=add(total,value);
 }
 if local && cfg.values.w== -3.0{let correction=localCorrection(j);if cfg.ids.y==12u{gradient=correction;}else{total=add(total,correction.x);}}
 var value=total.x+total.y;if (id==29u && !area) || (id==12u && area){value=length(native(gradient));}output[index]=vec4f(value,distance*cfg.values.x,1,0);
}
// This check rejects drivers that reassociate away paired-f32 residuals.
@compute @workgroup_size(1)
fn check_analytic_arithmetic(){
 let a=D(table[0],table[1]);let b=D(table[2],table[3]);
 output[0]=vec4f(add(a,b),mul(a,b));output[1]=vec4f(atanD(a),asinhD(a));
}
