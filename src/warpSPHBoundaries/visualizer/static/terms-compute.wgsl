// Direct computation of 3D moments, not the volume ray caster.
// One invocation per query. See radial-terms-3d.md for the equations/limits.
struct Config { dims: vec4u, shape: vec4f, sizes: vec4f, polynomial: vec4f }
struct Axial { v: array<f32,9> }
@group(0) @binding(0) var<uniform> cfg: Config;
// 12 near-origin coefficients, 16 (node,weight) pairs, cubature records (8 floats).
@group(0) @binding(1) var<storage,read> table: array<f32>;
@group(0) @binding(2) var<storage,read_write> output: array<vec2f>;
const PI: f32 = 3.141592653589793;
fn ipow(x:f32,p:i32)->f32 { var value=1.;for(var j=0;j<abs(p);j++){value*=x;}if(p<0){return 1./value;}return value; }
fn falling(n:u32,k:u32)->f32 { var v=1.;for(var i=0u;i<k;i++){v*=f32(n-i);}return v; }
fn choose(n:u32,k:u32)->f32 { var v=1.;for(var i=1u;i<=k;i++){v*=f32(n-i+1u)/f32(i);}return v; }
fn jet(q:f32)->vec4f {
 if(q<0.||q>=1.){return vec4f(0.);}
 if(q<.1){
  var W=0.;var L=0.;var Q=0.;var R=0.;
  for(var p=0u;p<12u;p++){
   let c=table[p];if(c==0.){continue;}let n=i32(p);W+=c*ipow(q,n);
   if(p>=2u){L+=f32(p)*c*ipow(q,n-2);}
   if(q>0.&&p>=3u){Q+=f32(n*(n-2))*c*ipow(q,n-4);R+=f32(n*(n-2)*(n-4))*c*ipow(q,n-6);}
  }
  return vec4f(W,L,Q,R);
 }
 var derivatives:array<f32,4>;
 let cubic=u32(cfg.sizes.z)==0u;
 let exponent=u32(cfg.polynomial.x);
 let normalization=select(table[0],2.*table[0],cubic);
 for(var order=0u;order<4u;order++){
  var v=0.;
  if(cubic){
   var inner=0.;if(q<.5){inner=4.*ipow(.5-q,3-i32(order));}
   v=falling(3u,order)*select(1.,-1.,order%2u==1u)*(ipow(1.-q,3-i32(order))-inner);
  }else{
   let P=array<f32,4>(1.,cfg.polynomial.y,cfg.polynomial.z,cfg.polynomial.w);
   for(var j=0u;j<=order;j++){
    var poly=0.;for(var p=order-j;p<4u;p++){poly+=P[p]*falling(p,order-j)*ipow(q,i32(p-order+j));}
    v+=choose(order,j)*select(1.,-1.,j%2u==1u)*falling(exponent,j)*ipow(1.-q,i32(exponent-j))*poly;
   }
  }
  derivatives[order]=normalization*v;
 }
 let L=derivatives[1]/q;let Q=(derivatives[2]-L)/(q*q);
 let R=(derivatives[3]-3.*derivatives[2]/q+3.*derivatives[1]/(q*q))/(q*q*q);
 return vec4f(derivatives[0],L,Q,R);
}
fn pair(index:u32)->vec2u {
 let pairs=array<vec2u,6>(vec2u(0,0),vec2u(0,1),vec2u(0,2),vec2u(1,1),vec2u(1,2),vec2u(2,2));return pairs[index];
}
fn select_axial(v:Axial,n:vec3f)->f32 {
 let id=cfg.dims.y;
 if(id==0u){return v.v[0];}
 if(id<=3u){return v.v[1]*n[id-1u];}
 if(id<=9u){let ij=pair(id-4u);return select(0.,v.v[2],ij.x==ij.y)+(v.v[3]-v.v[2])*n[ij.x]*n[ij.y];}
 if(id<=12u){return v.v[4]*n[id-10u];}
 if(id<=21u){let i=(id-13u)/3u;let j=(id-13u)%3u;return select(0.,v.v[5],i==j)+(v.v[6]-v.v[5])*n[i]*n[j];}
 if(id<=27u){let ij=pair(id-22u);return select(0.,v.v[7],ij.x==ij.y)+(v.v[8]-v.v[7])*n[ij.x]*n[ij.y];}
 if(id==28u){return 2.*v.v[7]+v.v[8];}return v.v[4];
}
fn shell(q:f32,cap:f32)->Axial {
 let j=jet(q);let c=clamp(cap,-1.,1.);let A0=2.*PI*(c+1.);let A1=PI*(c*c-1.);let An=2.*PI*(c*c*c+1.)/3.;let At=PI*(c-c*c*c/3.+2./3.);let q2=q*q;let q3=q2*q;let q4=q2*q2;let base=q2*j.y*A0;
 return Axial(array<f32,9>(q2*j.x*A0,q3*j.x*A1,q4*j.x*At,q4*j.x*An,-q3*j.y*A1,-q4*j.y*At,-q4*j.y*An,base+q4*j.z*At,base+q4*j.z*An));
}
fn correction(t:u32,p:u32,f:f32,df:f32,q:f32,d:f32,k:f32)->f32 {
 let z=-d;let rho=max(q*q-d*d,0.);let first=-.5*k*ipow(z,i32(p))*ipow(rho,i32((t+2u)/2u))*f;
 var derivative=ipow(z,i32(p+1u))*df;if(p>0u){derivative+=f32(p)*ipow(z,i32(p-1u))*f;}
 return select(2.*PI,PI,t>0u)*q*(first+select(0.,k*k/8.*ipow(rho,i32((t+4u)/2u))*derivative,cfg.sizes.w>=2.));
}
fn curvature(q:f32,d:f32,k:f32)->Axial {
 let j=jet(q);let base=correction(0u,0u,j.y,j.z,q,d,k);
 return Axial(array<f32,9>(correction(0u,0u,j.x,j.y,q,d,k),correction(0u,1u,j.x,j.y,q,d,k),correction(2u,0u,j.x,j.y,q,d,k),correction(0u,2u,j.x,j.y,q,d,k),-correction(0u,1u,j.y,j.z,q,d,k),-correction(2u,0u,j.y,j.z,q,d,k),-correction(0u,2u,j.y,j.z,q,d,k),base+correction(2u,0u,j.z,j.w,q,d,k),base+correction(0u,2u,j.z,j.w,q,d,k)));
}
fn axial(d:f32,n:vec3f)->f32 {
 var result=0.;let R=cfg.shape.y/cfg.shape.x;let sphere=cfg.dims.w==0u;let knots=select(1u,2u,u32(cfg.sizes.z)==0u);
 for(var range=0u;range<2u;range++){
  let full=range==0u;let lo=select(abs(d),0.,full);var hi=select(1.,clamp(-d,0.,1.),full);
  if(sphere&&!full){hi=min(1.,2.*R+d);}
  for(var cut=0u;cut<knots;cut++){
   let l=max(lo,f32(cut)/f32(knots));let r=min(hi,f32(cut+1u)/f32(knots));if(r<=l){continue;}
   for(var j=0u;j<16u;j++){
    let q=(l+r)/2.+(r-l)/2.*table[12u+j*2u];var cap=1.;
    if(!full){cap=-d/q;if(sphere){let D=R+d;cap=(R*R-D*D-q*q)/(2.*select(1.,D,D>0.)*q);}}
    result+=(r-l)/2.*table[13u+j*2u]*select_axial(shell(q,cap),n);
   }
  }
 }
 if(cfg.dims.w==2u&&cfg.sizes.w>=1.&&abs(d)<1.){
  let k=cfg.shape.x/cfg.shape.y;
  for(var cut=0u;cut<knots;cut++){
   let l=max(abs(d),f32(cut)/f32(knots));let r=f32(cut+1u)/f32(knots);if(r<=l){continue;}
   for(var j=0u;j<16u;j++){let q=(l+r)/2.+(r-l)/2.*table[12u+j*2u];result+=(r-l)/2.*table[13u+j*2u]*select_axial(curvature(q,d,k),n);}
  }
 }
 if(cfg.dims.y==29u){return abs(result);}
 return result;
}
fn vertex(i:u32)->vec3f {
 let v=array<vec3f,4>(vec3f(-1.,-1.,-1.),vec3f(1.,1.,-1.),vec3f(1.,-1.,1.),vec3f(-1.,1.,1.));return v[i]*vec3f(cfg.shape.z,cfg.shape.w,cfg.sizes.x)/2.;
}
fn face(i:u32)->vec3u {let f=array<vec3u,4>(vec3u(0,1,2),vec3u(0,3,1),vec3u(0,2,3),vec3u(1,3,2));return f[i];}
fn inside(p:vec3f)->bool {
 if(cfg.dims.z==0u){return dot(p,p)<=cfg.shape.y*cfg.shape.y;}
 if(cfg.dims.z==1u){return all(abs(p)<=vec3f(cfg.shape.z,cfg.shape.w,cfg.sizes.x)/2.);}
 for(var i=0u;i<4u;i++){let f=face(i);let a=vertex(f.x);var n=normalize(cross(vertex(f.y)-a,vertex(f.z)-a));if(dot(n,a)<0.){n=-n;}if(dot(p-a,n)>0.){return false;}}return true;
}
fn distance_normal(p:vec3f)->vec4f {
 if(cfg.dims.z==0u){let r=length(p);var n=vec3f(1.,0.,0.);if(r>0.){n=p/r;}return vec4f(n,r-cfg.shape.y);}
 if(cfg.dims.z==1u){
  let q=abs(p)-vec3f(cfg.shape.z,cfg.shape.w,cfg.sizes.x)/2.;let signs=select(vec3f(-1.),vec3f(1.),p>=vec3f(0.));let v=max(q,vec3f(0.))*signs;let r=length(v);
  if(r>0.){return vec4f(v/r,r);}var index=0u;if(q.y>q[index]){index=1u;}if(q.z>q[index]){index=2u;}var n=vec3f(0.);n[index]=signs[index];return vec4f(n,q[index]);
 }
 var contained=true;var best=1e30;var closest_point=vec3f(0.);var nearest=1e30;var nearest_normal=vec3f(1.,0.,0.);
 for(var i=0u;i<4u;i++){
  let ids=face(i);let a=vertex(ids.x);let b=vertex(ids.y);let c=vertex(ids.z);let u=b-a;let v=c-a;var n=normalize(cross(u,v));if(dot(n,a)<0.){n=-n;}
  let relative=p-a;let plane=dot(relative,n);contained=contained&&plane<=0.;if(abs(plane)<nearest){nearest=abs(plane);nearest_normal=n;}
  let uu=dot(u,u);let uv=dot(u,v);let vv=dot(v,v);let ru=dot(relative,u);let rv=dot(relative,v);let alpha=(vv*ru-uv*rv)/(uu*vv-uv*uv);let beta=(uu*rv-uv*ru)/(uu*vv-uv*uv);
  if(alpha>=0.&&beta>=0.&&alpha+beta<=1.&&plane*plane<best){best=plane*plane;closest_point=p-plane*n;}
  let starts=array<vec3f,3>(a,b,c);let ends=array<vec3f,3>(b,c,a);
  for(var edge=0u;edge<3u;edge++){let delta=ends[edge]-starts[edge];let t=clamp(dot(p-starts[edge],delta)/dot(delta,delta),0.,1.);let candidate=starts[edge]+t*delta;let dist=dot(p-candidate,p-candidate);if(dist<best){best=dist;closest_point=candidate;}}
 }
 let r=sqrt(best);var n=nearest_normal;if(!contained&&r>0.){n=(p-closest_point)/r;}return vec4f(n,select(r,-r,contained));
}
@compute @workgroup_size(64)
fn compute_terms(@builtin(global_invocation_id) gid:vec3u) {
 let id=gid.x;let n=cfg.dims.x;if(id>=n*n*n){return;}
 let xyz=vec3u(id%n,(id/n)%n,id/(n*n));let p=-vec3f(cfg.sizes.y)+2.*cfg.sizes.y*vec3f(xyz)/f32(n-1u);let dn=distance_normal(p);var value=0.;
 if(cfg.dims.w==3u){
  // Sixteen partial sums and a balanced reduction limit long f32 cancellation.
  var partial:array<f32,16>;var partial_gradient:array<vec3f,16>;
  for(var j=0u;j<u32(cfg.sizes.w);j++){
   let offset=44u+j*8u;let y=vec3f(table[offset],table[offset+1u],table[offset+2u]);
   if(inside(p+cfg.shape.x*y)){
    let bin=j%16u;partial[bin]+=table[offset+3u];
    if(cfg.dims.y==29u){partial_gradient[bin]+=vec3f(table[offset+4u],table[offset+5u],table[offset+6u]);}
   }
  }
  for(var stride=8u;stride>0u;stride/=2u){for(var j=0u;j<stride;j++){partial[j]+=partial[j+stride];partial_gradient[j]+=partial_gradient[j+stride];}}
  value=partial[0];if(cfg.dims.y==29u){value=length(partial_gradient[0]);}
 }else{value=axial(dn.w/cfg.shape.x,dn.xyz);}
 output[id]=vec2f(value,dn.w);
}
