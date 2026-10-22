// Same regular radial flux and split Gauss rules as the float64 reference.
// One invocation per query. Only the selected term is accumulated; shared mesh
// edges are cancelled on the host before evaluation. All arithmetic here is f32.
struct Settings {
 ids:vec4u, // resolution, field, geometry (rectangle/triangle/circle), model
 counts:vec4u, // polygon vertices, assembled mesh edges, Gauss nodes, kernel ID
 grid:vec4f, // x0/h, y0/h, grid step/h, h
 params:vec4f, // R/h, SDF spacing/h, sampled SDF flag, C2
 shape:vec4f, // exponent n, P1, P2, P3
 lattice:vec4f, // sampled SDF origin/h
}
@group(0) @binding(0) var<uniform> s:Settings;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
struct Sum { value:vec3f, correction:vec3f }
fn add(sum:ptr<function,Sum>,v:vec3f){let y=v-(*sum).correction;let t=(*sum).value+y;(*sum).correction=(t-(*sum).value)-y;(*sum).value=t;}
// Invalid samples carry explicit flags. WGSL constant expressions cannot
// construct NaN portably, and nonfinite arithmetic is not a validity protocol.
struct GeometryJet { d:f32, normal:vec2f, normalValid:bool }
struct LocalJet { d:f32, normal:vec2f, curvature:f32, normalValid:bool, curvatureValid:bool }
struct Evaluation { value:vec3f, valid:bool }
fn cross2(a:vec2f,b:vec2f)->f32{return a.x*b.y-a.y*b.x;}
fn vertex(i:u32)->vec2f{return vec2f(table[236u+2u*i],table[237u+2u*i]);}
fn kernel(r:f32)->f32{
 if r>=1.0{return 0.0;}
 let u=1.0-r;
 if s.counts.w==0u{return s.params.w*(u*u*u-4.0*pow(max(0.5-r,0.0),3.0));}
 return s.params.w*pow(u,s.shape.x)*(1.0+r*(s.shape.y+r*(s.shape.z+r*s.shape.w)));
}
fn jet(r:f32)->vec3f{
 if r>=1.0{return vec3f(0);}
 if r<0.1 {
  var W=0.0;var L=0.0;var Q=0.0;
  for(var p=0u;p<12u;p++){let c=table[224u+p];if c==0.0{continue;}W+=c*pow(r,f32(p));if p>=2u{L+=f32(p)*c*pow(r,f32(p)-2.0);}if r>0.0 && p>=3u{Q+=f32(p)*f32(p-2u)*c*pow(r,f32(p)-4.0);}}
  return vec3f(W,L,Q);
 }
 let u=1.0-r;var W=0.0;var dW=0.0;var ddW=0.0;
 if s.counts.w==0u {
  let v=max(0.5-r,0.0);W=s.params.w*(u*u*u-4.0*v*v*v);dW=s.params.w*(-3.0*u*u+12.0*v*v);ddW=s.params.w*(6.0*u-24.0*v);
 }else{
  let n=s.shape.x;let P=1.0+r*(s.shape.y+r*(s.shape.z+r*s.shape.w));let dP=s.shape.y+r*(2.0*s.shape.z+3.0*r*s.shape.w);let ddP=2.0*s.shape.z+6.0*r*s.shape.w;
  W=s.params.w*pow(u,n)*P;dW=s.params.w*(pow(u,n)*dP-n*pow(u,n-1.0)*P);
  ddW=s.params.w*(pow(u,n)*ddP-2.0*n*pow(u,n-1.0)*dP+n*(n-1.0)*pow(u,n-2.0)*P);
 }
 let L=dW/r;return vec3f(W,L,(ddW-L)/(r*r));
}
fn flux(r:f32,degree:u32)->f32{
 let base=128u+degree*32u;var value=0.0;
 if r<=0.5 {for(var p=12u;p>0u;p--){value=value*r+table[base+p];}return value;}
 if r<1.0 {let u=1.0-r;for(var p=16u;p>0u;p--){value=value*u+table[base+12u+p];}value=table[base]-pow(u,s.shape.x+1.0)*value;}else{value=table[base];}
 return value/pow(r,f32(degree)+2.0);
}
fn monomial(y:vec2f)->f32{
 switch s.ids.y {case 1u:{return y.x;}case 2u:{return y.y;}case 3u:{return y.x*y.x;}case 4u:{return y.x*y.y;}case 5u:{return y.y*y.y;}default:{return 1.0;}}
}
fn whole()->vec3f{
 let id=s.ids.y;if id==0u || id==8u || id==11u{return vec3f(1,0,0);}
 if id==3u || id==5u{return vec3f(3.141592653589793*table[192u],0,0);}return vec3f(0);
}
fn boundary(y:vec2f,n:vec2f)->vec3f{
 let id=s.ids.y;let r=length(y);
 if id<=5u {let degree=select(select(0u,1u,id>0u),2u,id>=3u);return vec3f(monomial(y)*dot(y,n)*flux(r,degree),0,0);}
 let W=kernel(r);
 if id==12u{return vec3f(0,-n.x*W,-n.y*W);}
 if id==6u || id==7u{return vec3f(-n[id-6u]*W,0,0);}
 let i=(id-8u)/2u;let j=(id-8u)%2u;var v=-y[i]*n[j]*W;
 if i==j {v+=dot(y,n)*flux(r,0u);}return vec3f(v,0,0);
}
fn insert(cuts:ptr<function,array<f32,48>>,count:ptr<function,u32>,x:f32,lo:f32,hi:f32){
 if x<=lo || x>=hi || *count>=48u{return;}
 for(var i=0u;i<*count;i++){if (*cuts)[i]==x{return;}}
 var j=*count;loop {if j==0u{break;}if (*cuts)[j-1u]<x{break;}(*cuts)[j]=(*cuts)[j-1u];j--;}
 (*cuts)[j]=x;*count+=1u;
}
fn edge(sum:ptr<function,Sum>,a:vec2f,b:vec2f){
 let e=b-a;let t=e/length(e);let n=vec2f(t.y,-t.x);
 // cross(a,b)/length(e) uses the same endpoints on reversed shared edges.
 let z=cross2(a,b)/length(e);let lo=dot(a,t);let hi=dot(b,t);
 var cuts:array<f32,48>;cuts[0]=lo;cuts[1]=hi;var count=2u;
 insert(&cuts,&count,0.0,lo,hi);
 for(var knot=0u;knot<2u;knot++){
  let R=select(1.0,0.5,knot==0u);if knot==0u && s.counts.w!=0u{continue;}
  if abs(z)<R{let q=sqrt(max(0.0,(R-z)*(R+z)));insert(&cuts,&count,-q,lo,hi);insert(&cuts,&count,q,lo,hi);}
 }
 var q=max(0.125,abs(z));for(var level=0u;level<24u;level++){if q>=max(abs(lo),abs(hi)){break;}insert(&cuts,&count,-q,lo,hi);insert(&cuts,&count,q,lo,hi);q*=2.0;}
 for(var panel=1u;panel<count;panel++){
  let half=(cuts[panel]-cuts[panel-1u])*0.5;let mid=(cuts[panel]+cuts[panel-1u])*0.5;
  for(var node=0u;node<s.counts.z;node++){let v=mid+half*table[2u*node];add(sum,boundary(z*n+v*t,n)*(half*table[2u*node+1u]));}
 }
}
fn signedPolygon(p:vec2f)->GeometryJet{
 var best=1e30;var closest=vec2f(0);var face=vec2f(0);var onEdge=false;var inside=false;var ambiguous=false;
 for(var i=0u;i<s.counts.x;i++){
  let a=vertex(i);let b=vertex((i+1u)%s.counts.x);let e=b-a;let t=clamp(dot(p-a,e)/dot(e,e),0.0,1.0);var targetPoint=a+t*e;if t==0.0{targetPoint=a;}if t==1.0{targetPoint=b;}
  let n=vec2f(e.y,-e.x)/length(e);let interior=t>0.0 && t<1.0;var r=length(p-targetPoint);if interior{r=abs(dot(p-a,n));}
  if (a.y>p.y)!=(b.y>p.y){if p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x{inside=!inside;}}
  let eps=0.000002*max(1.0,r);
  if r<best-eps{ambiguous=false;}
  else if abs(r-best)<=eps && (length(targetPoint-closest)>eps || (r<=eps && length(n-face)>eps)){ambiguous=true;}
  if r<best{best=r;closest=targetPoint;face=n;onEdge=interior;}
 }
 // Convex polygon projection is unique outside, even at edge/vertex transitions.
 // Preserve genuine interior medial ties and undefined boundary-corner normals.
 let d=select(best,-best,inside);var n=face;if !onEdge && best>0.000002{n=(p-closest)/d;}
 return GeometryJet(d,n,!(ambiguous && (inside || best<=0.000002)));
}
fn signedGeometry(p:vec2f)->GeometryJet{
 if s.ids.z!=2u{return signedPolygon(p);}let r=length(p);if r==0.0{return GeometryJet(-s.params.x,vec2f(0),false);}return GeometryJet(r-s.params.x,p/r,true);
}
fn polygon(p:vec2f,mesh:bool,d:f32)->vec3f{
 if d>=1.0{return vec3f(0);}if d<=-1.0{return whole();}var sum=Sum(vec3f(0),vec3f(0));
 if mesh{
  let base=236u+2u*s.counts.x;
  for(var i=0u;i<s.counts.y;i++){let j=base+4u*i;edge(&sum,vec2f(table[j],table[j+1u])-p,vec2f(table[j+2u],table[j+3u])-p);}
 }else{for(var i=0u;i<s.counts.x;i++){edge(&sum,vertex(i)-p,vertex((i+1u)%s.counts.x)-p);}}
 return sum.value;
}
fn disk(p:vec2f)->vec3f{
 let D=length(p);let R=s.params.x;if D>=R+1.0{return vec3f(0);}if D+1.0<=R{return whole();}
 var angle=0.0;if D>0.0{angle=atan2(p.y,p.x);}var cuts:array<f32,48>;cuts[0]=0.0;cuts[1]=3.141592653589793;var count=2u;
 for(var knot=0u;knot<2u;knot++){if knot==0u && s.counts.w!=0u{continue;}let r=select(1.0,0.5,knot==0u);if D>0.0{let v=(D*D+R*R-r*r)/(2.0*D*R);if abs(v)<1.0{insert(&cuts,&count,acos(v),0.0,3.141592653589793);}}}
 var a=1.0/max(max(R,D),1.0);for(var level=0u;level<16u;level++){if a>=3.141592653589793{break;}insert(&cuts,&count,a,0.0,3.141592653589793);a*=2.0;}
 var sum=Sum(vec3f(0),vec3f(0));
 for(var panel=1u;panel<count;panel++){let half=(cuts[panel]-cuts[panel-1u])*0.5;let mid=(cuts[panel]+cuts[panel-1u])*0.5;
  for(var node=0u;node<s.counts.z;node++){let theta=mid+half*table[2u*node];let weight=half*table[2u*node+1u]*R;
   for(var sign=0u;sign<2u;sign++){let alpha=angle+select(-theta,theta,sign==1u);let n=vec2f(cos(alpha),sin(alpha));add(&sum,boundary(R*n-p,n)*weight);}
  }
 }return sum.value;
}
fn latticeValue(p:vec2f)->f32{return signedGeometry(p).d;}
fn jets(p:vec2f,analytic:GeometryJet)->LocalJet{
 if s.params.z<0.5{return LocalJet(analytic.d,analytic.normal,1.0/s.params.x,analytic.normalValid,true);}
 let spacing=s.params.y;let uv=(p-s.lattice.xy)/spacing;let index=floor(uv);let f=uv-index;
 var d=0.0;var g=vec2f(0);var H=vec3f(0);let ex=vec2f(spacing,0);let ey=vec2f(0,spacing);
 for(var a=0u;a<2u;a++){for(var b=0u;b<2u;b++){
  let q=s.lattice.xy+(index+vec2f(f32(a),f32(b)))*spacing;let weight=select(1.0-f.x,f.x,a==1u)*select(1.0-f.y,f.y,b==1u);let S=latticeValue(q);
  d+=weight*S;g+=weight*vec2f(latticeValue(q+ex)-latticeValue(q-ex),latticeValue(q+ey)-latticeValue(q-ey))/(2.0*spacing);
  H+=weight*vec3f(latticeValue(q+2.0*ex)-2.0*S+latticeValue(q-2.0*ex),latticeValue(q+ex+ey)-latticeValue(q+ex-ey)-latticeValue(q-ex+ey)+latticeValue(q-ex-ey),latticeValue(q+2.0*ey)-2.0*S+latticeValue(q-2.0*ey))/(4.0*spacing*spacing);
 }}
 let lengthG=length(g);let normalValid=lengthG>=0.000001;var n=vec2f(0);if normalValid{n=g/lengthG;}
 let level=(g.y*g.y*H.x-2.0*g.x*g.y*H.y+g.x*g.x*H.z)/max(lengthG*lengthG*lengthG,1e-30);let den=1.0-d*level;let curvatureValid=abs(den)>0.05;var k=0.0;if curvatureValid{k=level/den;}return LocalJet(d,n,k,normalValid,curvatureValid);
}
fn term(y:vec2f,n:vec2f,derivative:bool)->vec3f{
 let k=jet(length(y));let W=k.x;let L=k.y;let Q=k.z;let z=dot(y,n);let id=s.ids.y;
 if id<=5u{
  let m=monomial(y);if !derivative{return vec3f(m*W,0,0);}var dm=0.0;
  switch id{case 1u:{dm=n.x;}case 2u:{dm=n.y;}case 3u:{dm=2.0*y.x*n.x;}case 4u:{dm=n.x*y.y+n.y*y.x;}case 5u:{dm=2.0*y.y*n.y;}default:{}}
  return vec3f(dm*W+m*z*L,0,0);
 }
 if id==12u{if derivative{return vec3f(0,-n*L-y*z*Q);}return vec3f(0,-y*L);}
 if id==6u || id==7u{let i=id-6u;if derivative{return vec3f(-n[i]*L-y[i]*z*Q,0,0);}return vec3f(-y[i]*L,0,0);}
 let i=(id-8u)/2u;let j=(id-8u)%2u;if derivative{return vec3f(-(n[i]*y[j]+y[i]*n[j])*L-y[i]*y[j]*z*Q,0,0);}return vec3f(-y[i]*y[j]*L,0,0);
}
fn plane(j:LocalJet,curvature:bool)->Evaluation{
 let d=j.d;if d>=1.0{return Evaluation(vec3f(0),true);}if d<=-1.0{return Evaluation(whole(),true);}
 // A missing normal still permits a scalar mass integral.
 var n=vec2f(0,1);if j.normalValid{n=j.normal;}else if s.ids.y!=0u{return Evaluation(vec3f(0),false);}
 if curvature && !j.curvatureValid{return Evaluation(vec3f(0),false);}
 let t=vec2f(n.y,-n.x);let a=-4.0*t-4.0*n;let b=4.0*t-4.0*n;let c=4.0*t-d*n;let e=-4.0*t-d*n;
 var sum=Sum(vec3f(0),vec3f(0));edge(&sum,a,b);edge(&sum,b,c);edge(&sum,c,e);edge(&sum,e,a);
 if curvature{
  let R=sqrt(max(0.0,1.0-d*d));var cuts:array<f32,48>;cuts[0]=-R;cuts[1]=R;var count=2u;insert(&cuts,&count,0.0,-R,R);
  if s.counts.w==0u && abs(d)<0.5{let r=sqrt(0.25-d*d);insert(&cuts,&count,-r,-R,R);insert(&cuts,&count,r,-R,R);}
  for(var panel=1u;panel<count;panel++){let half=(cuts[panel]-cuts[panel-1u])*0.5;let mid=(cuts[panel]+cuts[panel-1u])*0.5;
   for(var node=0u;node<s.counts.z;node++){let x=mid+half*table[2u*node];let y=t*x-d*n;let corr=-0.5*j.curvature*x*x*term(y,n,false)+0.125*j.curvature*j.curvature*x*x*x*x*term(y,n,true);add(&sum,corr*(half*table[2u*node+1u]));}
  }
 }return Evaluation(sum.value,true);
}
@compute @workgroup_size(64)
fn compute_terms2d(@builtin(global_invocation_id) invocation:vec3u){
 let index=invocation.x;let n=s.ids.x;if index>=n*n{return;}
 let p=s.grid.xy+s.grid.z*vec2f(f32(index%n),f32(index/n));let geometry=signedGeometry(p);var v=vec3f(0);var valid=true;
 switch s.ids.w{
  case 0u,1u,2u:{var d=geometry.d;if s.ids.z==2u{d=signedPolygon(p).d;}v=polygon(p,s.ids.w==0u,d);}
  case 3u,4u:{let evaluated=plane(jets(p,geometry),s.ids.w==4u);v=evaluated.value;valid=evaluated.valid;}
  default:{v=disk(p);}
 }
 var value=v.x;if s.ids.y==12u{value=length(v.yz);}output[index]=vec4f(value,geometry.d*s.grid.w,select(0.0,1.0,valid),0.0);
}
