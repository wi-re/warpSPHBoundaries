// Float64 browser evaluation of the existing 2D family. No Python or GPU needed.
// Polygon moments use the regular divergence flux A_k(r)=r^(-k-2)
// integral_0^r t^(k+1) W(t)dt. Edge/arc integrals use split Gauss quadrature.
// Query gradients follow integration by parts: G_i=-oint W n_i and
// g1_ij=delta_ij M0-oint y_i W n_j. These are not mass-grid differences.
import {SHAPES,radialJet,gauss} from './term-engine.js?v=20261022-4';
export const FIELDS2D=['m00','m10','m01','m20','m11','m02','Gx','Gy','g1xx','g1xy','g1yx','g1yy','grad_norm'];
export const DEFAULT2D={geometry:'rectangle',width:2,height:2,radius:2,angle:0,vertices:[[-1,-1],[1,-1],[1,1],[-1,1]],segments:16,kernel:'w4',h:.65,resolution:81,padding:1.15,field:'m00',models:['mesh','rectangle','sdf'],sdf_source:'analytic',sdf_spacing:.12,quadrature:24};
export const MODELS2D={mesh:['Triangle mesh','Split Gauss integration of regular radial flux, summed over triangles.'],exact:['Polygon · radial flux','Split Gauss integration over polygon edges; numerical quadrature.'],rectangle:['Rectangle · radial flux','Four-edge integration of the same regular flux; rectangle only.'],sdf:['SDF · distance only','Local half-plane integrals with analytic or sampled SDF jets.'],curvature:['SDF · curvature O(2)','Quadratic-graph expansion of each integrand; circle only, R/h ≥ 2.'],disk:['Circle · boundary quadrature','Circle arcs split at kernel knots; refine Gauss nodes to assess convergence.']};
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1],cross=(a,b)=>a[0]*b[1]-a[1]*b[0],sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const norm2={w2:7/Math.PI,w4:9/Math.PI,w6:78/(7*Math.PI),cubic:80/(7*Math.PI)};
const choose=(n,k)=>{let v=1;for(let i=1;i<=k;i++)v=v*(n-i+1)/i;return v;};
const horner=(p,x)=>{let v=0;for(let i=p.length-1;i>=0;i--)v=v*x+p[i];return v;};
const fluxCache=new Map();
export function value2d(r,name){if(r>=1)return 0;const k=SHAPES[name];return norm2[name]*(name==='cubic'?(1-r)**3-4*Math.max(.5-r,0)**3:(1-r)**k.n*k.P.reduce((s,c,i)=>s+c*r**i,0));}
export function kernel2d(r,name){return radialJet(r,name).map(v=>v*norm2[name]/SHAPES[name].C);}
export function fluxPlan(name){
 if(fluxCache.has(name))return fluxCache.get(name);
 const shape=SHAPES[name],ratio=norm2[name]/shape.C,coeffs=shape.coeffs.map(c=>c*ratio),plans=[];
 for(let degree=0;degree<=2;degree++){
  const tail=Array(degree+shape.P.length+3).fill(0),power=shape.n;
  // u=1-r makes the outer tail well conditioned near the support boundary.
  for(let i=0;i<shape.P.length;i++)for(let j=0;j<=degree+1+i;j++)tail[j]+=norm2[name]*shape.P[i]*choose(degree+1+i,j)*(-1)**j/(power+j+1);
  const cuts=name==='cubic'?[0,.5,1]:[0,1];let total=0;
  for(let k=1;k<cuts.length;k++)for(const [x,w]of gauss(16)){const r=(cuts[k]+cuts[k-1]+(cuts[k]-cuts[k-1])*x)/2;total+=w*(cuts[k]-cuts[k-1])/2*r**(degree+1)*kernel2d(r,name)[0];}
  plans.push({total,inner:coeffs.map((c,p)=>c/(degree+2+p)),tail,power});
 }
 fluxCache.set(name,plans);return plans;
}
export function flux(r,name,degree){
 const p=fluxPlan(name)[degree];if(r<=.5)return horner(p.inner,r);
 const integral=r>=1?p.total:p.total-(1-r)**(p.power+1)*horner(p.tail,1-r);
 return integral/r**(degree+2);
}
export function whole2d(name){const v=Array(12).fill(0);v[0]=1;v[3]=v[5]=Math.PI*fluxPlan(name)[2].total;v[8]=v[11]=1;return v;}
export function geometry2d(c){
 let v;if(c.geometry==='circle')v=Array.from({length:c.segments},(_,i)=>[c.radius*Math.cos(2*Math.PI*i/c.segments),c.radius*Math.sin(2*Math.PI*i/c.segments)]);
 else {v=c.geometry==='rectangle'?[[-c.width/2,-c.height/2],[c.width/2,-c.height/2],[c.width/2,c.height/2],[-c.width/2,c.height/2]]:c.vertices.slice(0,3).map(p=>[...p]);const a=c.angle*Math.PI/180;v=v.map(([x,y])=>[Math.cos(a)*x-Math.sin(a)*y,Math.sin(a)*x+Math.cos(a)*y]);}
 if(c.geometry==='triangle'&&Math.abs(cross(sub(v[1],v[0]),sub(v[2],v[0])))<1e-7)throw new Error('Triangle vertices must not be collinear');
 if(v.reduce((s,p,i)=>s+cross(p,v[(i+1)%v.length]),0)<0)v.reverse();
 const triangles=v.length===3?[v]:c.geometry==='circle'?v.map((p,i)=>[[0,0],p,v[(i+1)%v.length]]):[[v[0],v[1],v[2]],[v[0],v[2],v[3]]];
 return {vertices:v,triangles};
}
export function signed2d(p,c,vertices){
 if(c.geometry==='circle'){const r=Math.hypot(...p);return [r-c.radius,r?p.map(x=>x/r):[NaN,NaN]];}
 let best=Infinity,closest,face,inside=false,ambiguous=false;
 for(let i=0;i<vertices.length;i++){
  const a=vertices[i],b=vertices[(i+1)%vertices.length],edge=sub(b,a),len=Math.hypot(...edge),t=Math.max(0,Math.min(1,dot(sub(p,a),edge)/(len*len))),q=[a[0]+t*edge[0],a[1]+t*edge[1]],delta=sub(p,q),r=Math.hypot(...delta),n=[edge[1]/len,-edge[0]/len];
  if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;
  const eps=2e-14*Math.max(1,r);
  if(r<best-eps){best=r;closest=q;face=n;ambiguous=false;}
  else if(Math.abs(r-best)<=eps&&(Math.hypot(...sub(q,closest))>eps||(r<=eps&&Math.hypot(...sub(n,face))>eps)))ambiguous=true;
 }
 const d=inside?-best:best,n=best>2e-14?sub(p,closest).map(x=>x/(d||1)):face;return [d,ambiguous?[NaN,NaN]:n];
}
// Accumulate all 12 channels together. Moments and query gradients have
// different fluxes; first gradient moments are completed with delta_ij M0.
function addBoundary(out,x,y,nx,ny,weight,name){
 const r=Math.hypot(x,y),d=x*nx+y*ny,W=value2d(r,name),f0=flux(r,name,0)*d*weight,f1=flux(r,name,1)*d*weight,f2=flux(r,name,2)*d*weight,w=W*weight;
 out[0]+=f0;out[1]+=x*f1;out[2]+=y*f1;out[3]+=x*x*f2;out[4]+=x*y*f2;out[5]+=y*y*f2;
 out[6]-=nx*w;out[7]-=ny*w;out[8]-=x*nx*w;out[9]-=x*ny*w;out[10]-=y*nx*w;out[11]-=y*ny*w;
}
function finish(out){out[8]+=out[0];out[11]+=out[0];return out;}
export function polygonChannels(vertices,point,name,nodes=24){
 const out=Array(12).fill(0),v=vertices.map(p=>sub(p,point));
 for(let i=0;i<v.length;i++){
  const a=v[i],b=v[(i+1)%v.length],e=sub(b,a),length=Math.hypot(...e),t=e.map(x=>x/length),normal=[t[1],-t[0]],z=dot(a,normal),lo=dot(a,t),hi=dot(b,t),cuts=[lo,hi];
  // Smooth panels at closest approach, radial knots, and geometric intervals
  // outside support. The regular flux remains nonzero outside the kernel.
  for(const R of name==='cubic'?[.5,1]:[1])if(Math.abs(z)<R){const s=Math.sqrt((R-z)*(R+z));cuts.push(-s,s);}
  for(let s=Math.max(.125,Math.abs(z));s<Math.max(Math.abs(lo),Math.abs(hi));s*=2)cuts.push(-s,s);
  cuts.push(0);const sorted=[...new Set(cuts.filter(x=>x>=lo&&x<=hi))].sort((a,b)=>a-b);
  for(let k=1;k<sorted.length;k++){const mid=(sorted[k]+sorted[k-1])/2,half=(sorted[k]-sorted[k-1])/2;for(const [node,w]of gauss(nodes)){const s=mid+half*node;addBoundary(out,z*normal[0]+s*t[0],z*normal[1]+s*t[1],...normal,w*half,name);}}
 }
 return finish(out);
}
export function diskChannels(p,R,name,nodes=24){
 const D=Math.hypot(...p);if(D>=R+1)return Array(12).fill(0);if(D+1<=R)return whole2d(name);
 const angle=D?Math.atan2(p[1],p[0]):0,cuts=[0,Math.PI];
 for(const r of name==='cubic'?[.5,1]:[1])if(D){const v=(D*D+R*R-r*r)/(2*D*R);if(Math.abs(v)<1)cuts.push(Math.acos(v));}
 for(let a=1/Math.max(R,D,1);a<Math.PI;a*=2)cuts.push(a);
 cuts.sort((a,b)=>a-b);const out=Array(12).fill(0);
 for(let k=1;k<cuts.length;k++)for(const [node,w]of gauss(nodes))for(const sign of [-1,1]){const theta=angle+sign*(cuts[k]+cuts[k-1]+(cuts[k]-cuts[k-1])*node)/2,n=[Math.cos(theta),Math.sin(theta)];addBoundary(out,R*n[0]-p[0],R*n[1]-p[1],...n,w*(cuts[k]-cuts[k-1])/2*R,name);}
 return finish(out);
}
export function planeChannels(d,n,name,nodes=24,curvature=0){
 if(d>=1)return Array(12).fill(0);if(d<=-1)return whole2d(name);if(!Number.isFinite(d)||!Number.isFinite(curvature))return Array(12).fill(NaN);
 const local=polygonChannels([[-4,-4],[4,-4],[4,-d],[-4,-d]],[0,0],name,nodes);
 // Enforce exact reflection symmetries instead of quadrature residuals.
 local[1]=local[4]=local[6]=local[9]=local[10]=0;
 if(curvature){
  const a=Math.sqrt(Math.max(0,1-d*d)),cuts=[-a,0,a];if(name==='cubic'&&Math.abs(d)<.5){const b=Math.sqrt(.25-d*d);cuts.push(-b,b);}cuts.sort((a,b)=>a-b);
  for(let k=1;k<cuts.length;k++)for(const [node,w]of gauss(nodes)){
   const x=(cuts[k]+cuts[k-1]+(cuts[k]-cuts[k-1])*node)/2,z=-d,r=Math.hypot(x,z),[W,L,Q]=kernel2d(r,name),weight=w*(cuts[k]-cuts[k-1])/2;
   const F=[W,x*W,z*W,x*x*W,x*z*W,z*z*W,-x*L,-z*L,-x*x*L,-x*z*L,-x*z*L,-z*z*L];
   const dz=[z*L,x*z*L,W+z*z*L,x*x*z*L,x*W+x*z*z*L,2*z*W+z**3*L,-x*z*Q,-L-z*z*Q,-x*x*z*Q,-x*L-x*z*z*Q,-x*L-x*z*z*Q,-2*z*L-z**3*Q];
   for(let j=0;j<12;j++)local[j]+=weight*(-curvature/2*x*x*F[j]+curvature*curvature/8*x**4*dz[j]);
  }
 }
 if(!n.every(Number.isFinite)){const out=Array(12).fill(NaN);out[0]=local[0];return out;}
 const [nx,ny]=n,t=[-ny,nx],out=Array(12).fill(0);out[0]=local[0];out[1]=local[2]*nx;out[2]=local[2]*ny;out[6]=local[7]*nx;out[7]=local[7]*ny;
 out[3]=local[3]*t[0]**2+local[5]*nx*nx;out[4]=local[3]*t[0]*t[1]+local[5]*nx*ny;out[5]=local[3]*t[1]**2+local[5]*ny*ny;
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)out[8+2*i+j]=local[8]*t[i]*t[j]+local[11]*n[i]*n[j];return out;
}
export function select2d(v,field){return field==='grad_norm'?Math.hypot(v[6],v[7]):v[FIELDS2D.indexOf(field)];}
export function validate2d(payload){
 const c={...structuredClone(DEFAULT2D),...payload};
 for(const [key,choices]of [['geometry',['rectangle','triangle','circle']],['kernel',Object.keys(SHAPES)],['field',FIELDS2D],['sdf_source',['analytic','sampled']]])if(!choices.includes(c[key]))throw new Error(`Unsupported ${key}`);
 for(const [key,lo,hi,integer]of [['width',.1,8],['height',.1,8],['radius',.15,8],['h',.1,3],['angle',-180,180],['resolution',17,513,true],['padding',1,2],['segments',8,64,true],['quadrature',8,64,true],['sdf_spacing',.03,.5]])if(!Number.isFinite(c[key])||c[key]<lo||c[key]>hi||(integer&&!Number.isInteger(c[key])))throw new Error(`Invalid ${key}`);
 if(!Array.isArray(c.models)||c.models.length<1||c.models.length>4||new Set(c.models).size!==c.models.length||c.models.some(m=>!MODELS2D[m]))throw new Error('Choose one to four distinct 2D models');
 if(!Array.isArray(c.vertices)||c.vertices.length!==4||c.vertices.some(p=>!Array.isArray(p)||p.length!==2||p.some(x=>!Number.isFinite(x)||Math.abs(x)>8)))throw new Error('Invalid triangle vertices');return c;
}
function sampledJets(c,vertices,x,y){
 const spacing=c.sdf_spacing*c.h,origin=[x[0]-3*spacing,y[0]-3*spacing],nx=Math.ceil((x.at(-1)-origin[0])/spacing)+4,ny=Math.ceil((y.at(-1)-origin[1])/spacing)+4;
 if(nx*ny>2_000_000)throw new Error('SDF lattice too large; increase SDF spacing');
 const field=Float64Array.from({length:nx*ny},(_,i)=>signed2d([origin[0]+Math.floor(i/ny)*spacing,origin[1]+i%ny*spacing],c,vertices)[0]);
 const derivative=(v,axis)=>Float64Array.from(v,(_,index)=>{const i=axis?index%ny:Math.floor(index/ny),size=axis?ny:nx,stride=axis?1:ny;if(i===0)return (-3*v[index]+4*v[index+stride]-v[index+2*stride])/(2*spacing);if(i===size-1)return (3*v[index]-4*v[index-stride]+v[index-2*stride])/(2*spacing);return (v[index+stride]-v[index-stride])/(2*spacing);});
 const gx=derivative(field,0),gy=derivative(field,1),gxx=derivative(gx,0),gxy=derivative(gx,1),gyy=derivative(gy,1);
 return p=>{const u=p.map((x,i)=>(x-origin[i])/spacing),i=u.map((x,k)=>Math.max(0,Math.min((k?ny:nx)-2,Math.floor(x)))),t=u.map((x,k)=>x-i[k]);
  const sample=v=>{let value=0;for(let a=0;a<2;a++)for(let b=0;b<2;b++)value+=(a?t[0]:1-t[0])*(b?t[1]:1-t[1])*v[(i[0]+a)*ny+i[1]+b];return value;};
  const d=sample(field),vx=sample(gx),vy=sample(gy),length=Math.hypot(vx,vy),n=length>=1e-6?[vx/length,vy/length]:[NaN,NaN],level=(vy*vy*sample(gxx)-2*vx*vy*sample(gxy)+vx*vx*sample(gyy))/Math.max(length**3,1e-30),den=1-d*level;
  return [d/c.h,n,Math.abs(den)>.05?level/den*c.h:NaN];};
}
export function compute2d(payload,progress=()=>{}){
 const start=performance.now(),c=validate2d(payload),g=geometry2d(c),n=c.resolution,lo=[0,1].map(i=>c.geometry==='circle'?-c.radius:Math.min(...g.vertices.map(p=>p[i]))),hi=[0,1].map(i=>c.geometry==='circle'?c.radius:Math.max(...g.vertices.map(p=>p[i]))),half=Math.max(...hi.map((x,i)=>x-lo[i]))/2+c.padding*c.h;
 const axes=lo.map((v,i)=>Array.from({length:n},(_,j)=>(v+hi[i])/2-half+2*half*j/(n-1))),[x,y]=axes,points=Array.from({length:n*n},(_,i)=>[x[i%n],y[Math.floor(i/n)]]),signed=points.map(p=>signed2d(p,c,g.vertices)),distance=signed.map(s=>s[0]),results=[];
 let sampled=null,sampledError;
 if(c.sdf_source==='sampled'&&c.models.some(m=>m==='sdf'||m==='curvature'))try{sampled=sampledJets(c,g.vertices,x,y);}catch(e){sampledError=e.message;}
 const v=g.vertices.map(p=>p.map(x=>x/c.h)),triangles=g.triangles.map(t=>t.map(p=>p.map(x=>x/c.h)));
 for(const model of c.models){
  const tick=performance.now(),error=model==='rectangle'&&c.geometry!=='rectangle'?'Built-in rectangle requires rectangle geometry':model==='disk'&&c.geometry!=='circle'?'Circle reference requires circle geometry':model==='curvature'&&(c.geometry!=='circle'||c.radius/c.h<2)?'Curvature model requires circle geometry and R/h ≥ 2':(model==='sdf'||model==='curvature')?sampledError:null;
  if(error){results.push({model,error});continue;}
  const values=[];
  for(let i=0;i<points.length;i++){
   const p=points[i].map(x=>x/c.h);let channels;
   if(['mesh','exact','rectangle'].includes(model)){const d=c.geometry==='circle'?signed2d(p,{geometry:'polygon'},v)[0]:distance[i]/c.h;if(d>=1){values.push(0);continue;}if(d<=-1){values.push(select2d(whole2d(c.kernel),c.field));continue;}}
   if(model==='exact'||model==='rectangle'){channels=polygonChannels(v,p,c.kernel,c.quadrature);}
   else if(model==='mesh'){channels=Array(12).fill(0);for(const triangle of triangles){const q=polygonChannels(triangle,p,c.kernel,c.quadrature);channels=channels.map((v,j)=>v+q[j]);}}
   else if(model==='disk')channels=diskChannels(p,c.radius/c.h,c.kernel,c.quadrature);
   else{const [d,normal,kappa]=sampled?sampled(points[i]):[signed[i][0]/c.h,signed[i][1],c.h/c.radius];channels=planeChannels(d,normal,c.kernel,c.quadrature,model==='curvature'?kappa:0);}
   values.push(select2d(channels,c.field));if(i%n===n-1)progress(`${MODELS2D[model][0]} · ${Math.round((i+1)/points.length*100)}%`);
  }
  const finite=values.filter(Number.isFinite),notes=['Browser JavaScript float64 worker; radial flux with split Gauss quadrature. Direct kernel-gradient integrals, no grid finite differences.',MODELS2D[model][1],`${c.quadrature} Gauss nodes per smooth panel. Refine to assess numerical convergence; no exactness or certified tolerance claim.`];
  if(model==='sdf'||model==='curvature')notes.push(c.sdf_source==='sampled'?`Bilinear sampled SDF; finite-difference jets at spacing ${c.sdf_spacing} h.`:'Analytic signed distance and normals; ambiguous polygon normals remain missing.');
  if(c.geometry==='circle'&&['mesh','exact'].includes(model))notes.push(`Inscribed ${c.segments}-gon approximates the circle.`);
  results.push({model,label:MODELS2D[model][0],values:values.map(v=>Number.isFinite(v)?v:null),min:null,max:null,missing:values.length-finite.length,notes,milliseconds:performance.now()-tick});
  // Avoid spreading large manually requested grids into the argument stack.
  const row=results.at(-1);if(finite.length){row.min=finite.reduce((a,b)=>Math.min(a,b),Infinity);row.max=finite.reduce((a,b)=>Math.max(a,b),-Infinity);}
 }
 return {schema_version:1,dimension:2,config:c,x,y,...g,distance,results,milliseconds:performance.now()-start};
}
