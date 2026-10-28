// Portable FluxBridge numerical plans. y = source-query, unit support.
// Host geometry is JavaScript Number; uploaded/evaluated profiles are fp32.
import {gauss} from './term-engine.js?v=20261022-7';
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const mul=(a,s)=>a.map(v=>v*s);
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/Math.hypot(...a));
export function monomials(d,degree){
 const lattice=(d,n)=>d===1?[[n]]:Array.from({length:n+1},(_,i)=>n-i).flatMap(i=>lattice(d-1,n-i).map(b=>[i,...b]));
 return Array.from({length:degree+1},(_,n)=>lattice(d,n)).flat();
}
export function baseline(d){
 const zero=Array(d).fill(0);
 return [...monomials(d,2).map(alpha=>({alpha,derivative:[]})),
  ...monomials(d,1).flatMap(alpha=>Array.from({length:d},(_,j)=>({alpha,derivative:[j]}))),
  ...Array.from({length:d},(_,i)=>Array.from({length:d-i},(_,j)=>({alpha:zero,derivative:[i,i+j]}))).flat()];
}
export function validateRequests(d,requests){
 if(![2,3].includes(d)||!Array.isArray(requests)||!requests.length||requests.length>2200)throw Error('Invalid FluxBridge request count/dimension');
 return requests.map(({alpha,derivative=[]})=>{
  if(!Array.isArray(alpha)||alpha.length!==d||alpha.some(a=>!Number.isInteger(a)||a<0)||alpha.reduce((a,b)=>a+b,0)>9||!Array.isArray(derivative)||derivative.length>2||derivative.some(j=>!Number.isInteger(j)||j<0||j>=d))throw Error('Requests require degree ≤ 9 and kernel derivative order ≤ 2');
  return {alpha:[...alpha],derivative:[...derivative].sort()};
 });
}
export function compileProfiles(source,d,kernel,requests){
 const req=validateRequests(d,requests),p=source.dimensions[d]?.[kernel];if(!p)throw Error('Unknown FluxBridge kernel');
 const table=Array(p.profiles.length*12).fill(0),metadata=new Int32Array(req.length*12);
 p.profiles.forEach((_,i)=>{
  table[i*12]=p.constants[i];table[i*12+1]=p.bounds[0][1];
  p.coefficients[i].forEach((cs,j)=>{table[i*12+2+j*2]=cs.length;table[i*12+3+j*2]=table.length;table.push(...cs);});
  if(p.inner_scaled[i]){table[i*12+6]=p.inner_scaled[i].length;table[i*12+7]=table.length;table.push(...p.inner_scaled[i]);}
 });
 const term=(alpha,order,sign)=>{
  const index=p.profiles.findIndex(([degree,k])=>degree===alpha.reduce((s,a)=>s+a,0)&&k===order);
  if(index<0)throw Error('Missing generated radial profile');
  return {a:[...alpha,...Array(3-d).fill(0),index],sign};
 };
 req.forEach(({alpha,derivative:ds},i)=>{
  const raised=[...alpha];for(const j of ds)raised[j]++;
  const a=term(raised,ds.length,ds.length===1?-1:1);
  metadata.set(a.a,i*12);metadata[i*12+8]=a.sign;
  metadata[i*12+7]=-1;
  if(ds.length===2&&ds[0]===ds[1]){const b=term(alpha,1,1);metadata.set(b.a,i*12+4);metadata[i*12+9]=b.sign;}
  metadata[i*12+10]=alpha.reduce((s,a)=>s+a,0)-ds.length;
 });
 const profileIndices=[...new Set(req.flatMap((_,i)=>[metadata[i*12+3],metadata[i*12+7]]).filter(i=>i>=0))];
 const mapSize=Math.ceil(profileIndices.length/4)*4,masks=new Int32Array(Math.ceil(req.length/64));
 const sharedMetadata=new Int32Array(metadata.length+mapSize+Math.ceil(masks.length/4)*4);sharedMetadata.set(metadata);
 for(let i=0;i<req.length;i++)for(const offset of [3,7])if(metadata[i*12+offset]>=0)sharedMetadata[i*12+offset]=profileIndices.indexOf(metadata[i*12+offset]);
 for(let i=0;i<req.length;i++)for(const offset of [3,7]){const profile=sharedMetadata[i*12+offset];if(profile>=0)masks[Math.floor(i/64)]|=1<<profile;}
 sharedMetadata.set(profileIndices,metadata.length);
 sharedMetadata.set(masks,metadata.length+mapSize);
 return {table:new Float32Array(table),metadata,sharedMetadata,profileCount:profileIndices.length,requests:req,dimension:d,kernel};
}
// All edges/faces of selected volumes participate: exterior regular flux stays.
export function facets(vertices,d){
 if(d===2){
  const v=vertices.map(p=>[...p]);const origin=v[0];
  const area=v.reduce((s,a,i)=>s+(a[0]-origin[0])*(v[(i+1)%v.length][1]-origin[1])-(a[1]-origin[1])*(v[(i+1)%v.length][0]-origin[0]),0);
  if(!Number.isFinite(area)||area===0)throw Error('Degenerate polygon');if(area<0)v.reverse();
  return v.map((a,i)=>[a,v[(i+1)%v.length]]);
 }
 if(vertices.length!==4)throw Error('3D volume elements must be tetrahedra');
 return vertices.map((opposite,i)=>{
  const face=vertices.filter((_,j)=>j!==i);const side=dot(cross(sub(face[1],face[0]),sub(face[2],face[0])),sub(opposite,face[0]));
  if(!Number.isFinite(side)||side===0)throw Error('Degenerate tetrahedron');if(side>0)[face[1],face[2]]=[face[2],face[1]];return face;
 });
}
export function selectedElements(elements,q,h,exhaustive=false){
 return elements.flatMap((v,i)=>{
  const gap=q.map((x,j)=>Math.max(0,Math.min(...v.map(p=>p[j]))-x,x-Math.max(...v.map(p=>p[j]))));
  // Conservative host-f64 margin, never an individual regular-flux face test.
  const margin=64*Number.EPSILON*(1+Math.max(...v.flat().map(Math.abs),...q.map(Math.abs))/h);
  return exhaustive||Math.hypot(...gap)<=h*(1+margin)?[i]:[];
 });
}
export function prepareSamples(geometry,q,h,kernel,nodes=16){
 const d=q.length,vertices=Array.isArray(geometry)?geometry:geometry.faces.flat();if(![2,3].includes(d)||q.some(v=>!Number.isFinite(v))||!Number.isFinite(h)||h<=0||!Number.isInteger(nodes)||nodes<8||nodes>64||vertices.some(v=>v.length!==d||v.some(x=>!Number.isFinite(x))))throw Error('Invalid integration geometry/rule');
 const rawFaces=Array.isArray(geometry)?facets(vertices,d):geometry.faces;
 const fs=rawFaces.map(face=>face.map(v=>mul(sub(v,q),1/h))),rule=[...gauss(nodes)].sort((a,b)=>a[0]-b[0]),knots=kernel==='cubic'?[.5,1]:[1];
 let samples=new Float32Array(4096),size=0;
 const pushXYZ=(x,y,z,r,w,kind=0)=>{
  if(size>=8*2_000_000)throw Error('FluxBridge sample budget exceeded; reduce geometry/grid size');
  if(size+8>samples.length){const grown=new Float32Array(Math.min(8*2_000_000,samples.length*2));grown.set(samples);samples=grown;}
  samples[size++]=x;samples[size++]=y;samples[size++]=z;samples[size++]=r;
  samples[size++]=w;samples[size++]=kind;samples[size++]=0;samples[size++]=0;
 };
 const push=(u,r,w,kind=0)=>pushXYZ(u[0],u[1],u[2]??0,r,w,kind);
 // Split each signed edge at closest approach, kernel knots and geometric scales.
 const panels=(s0,s1,c,c0,visit)=>{
  for(const side of [-1,1]){
   let a=side===1?Math.max(s0,0):Math.max(-s1,0);const end=side===1?Math.max(s1,0):Math.max(-s0,0);
   let guard=0;
   while(a<end){
    let b=Math.min(end,Math.max(2*a,c));
    for(const R of knots)if(R*R>c*c+c0*c0){const cut=Math.sqrt(Math.max(0,R*R-c*c-c0*c0));if(a<cut&&cut<b)b=cut;}
    if(!(b>a)||++guard>2048)throw Error('Unresolved angular geometry; use the native reference');
    const left=Math.atan2(a,c),right=Math.atan2(b,c);visit(side,(left+right)/2,(right-left)/2,a);a=b;
   }
  }
 };
 if(d===2){
  for(const [a,b]of fs){
   const t=unit(sub(b,a)),n=[t[1],-t[0]],z=dot(n,a),c=Math.abs(z);if(c===0)continue;
   const sign=Math.sign(z);
   panels(dot(t,a),dot(t,b),c,0,(side,mid,half,start)=>{
    const outside=c>=1||start*start+c*c>=1;
    for(const [x,w]of rule){const angle=mid+half*x,sn=Math.sin(angle)*side,cs=Math.cos(angle),axial=sign*cs;pushXYZ(t[0]*sn+n[0]*axial,t[1]*sn+n[1]*axial,0,outside?1:c/cs,sign*half*w);}
   });
  }
 }else{
  for(const face of fs){
   const ab=sub(face[1],face[0]),ac=sub(face[2],face[0]),raw=cross(ab,ac),jacobian=Math.hypot(...raw),n=mul(raw,1/jacobian),distance=dot(n,face[0]),c0=Math.abs(distance);
   if(c0===0)continue;
   const span=Math.max(Math.hypot(...ab),Math.hypot(...ac),Math.hypot(...sub(face[2],face[1])));
   if(c0<=1e-12&&span<=4){
    // Same regular F/r^3 Duffy treatment as native near-plane preparation.
    for(const [x,wx]of rule)for(const [y,wy]of rule){const a=(x+1)/2,b=(y+1)/2,v=add(face[0],add(mul(ab,a),mul(ac,(1-a)*b))),r=Math.hypot(...v);push(mul(v,1/r),r,jacobian*distance*(1-a)*wx*wy/4,1);}
    continue;
   }
   const sign0=Math.sign(distance);
   for(let edge=0;edge<3;edge++){
    const av=face[edge],bv=face[(edge+1)%3],t=unit(sub(bv,av)),outward=cross(t,n),z=dot(outward,av),c=Math.abs(z);if(c===0)continue;
    const sign=Math.sign(z);
    panels(dot(t,av),dot(t,bv),c,c0,(side,mid,half)=>{
     const azimuth=rule.map(([x,w])=>{const phi=mid+half*x,cs=Math.cos(phi);return {e:add(mul(outward,sign*cs),mul(t,side*Math.sin(phi))),maxrho:c/cs,w};});
     const last=Math.max(...azimuth.map(a=>a.maxrho)),exterior=c0<1?Math.sqrt((1-c0)*(1+c0)):0;
     let rho=0,guard=0;
     while(rho<last){
      const outer=c0>=1||rho>=exterior;let high=outer?last:Math.min(last,Math.max(2*rho,c0));
      for(const R of knots)if(R>c0){const cut=Math.sqrt((R-c0)*(R+c0));if(rho<cut&&cut<high)high=cut;}
      if(!(high>rho)||++guard>2048)throw Error('Unresolved radial geometry');
      for(const {e,maxrho,w:pw}of azimuth){
       if(maxrho<=rho)continue;const t0=Math.atan2(rho,c0),t1=Math.atan2(Math.min(high,maxrho),c0),dh=(t1-t0)/2;
       if(dh<=0)continue;
       for(const [x,w]of rule){const theta=(t0+t1)/2+dh*x,st=Math.sin(theta),ct=Math.cos(theta),axial=sign0*ct;pushXYZ(n[0]*axial+e[0]*st,n[1]*axial+e[1]*st,n[2]*axial+e[2]*st,outer?1:c0/ct,sign0*sign*half*pw*dh*w*st);}
      }
      rho=high;
     }
    });
   }
  }
 }
 const result=size===samples.length?samples:samples.slice(0,size);
 if(!result.every(Number.isFinite))throw Error('Nonfinite prepared geometry');
 return result;
}

// Exact incidence cancellation is legal here only for a common volume integrand.
// Call separately for each material trace/region. No geometric tolerance merging.
export function mergeVolumeFacets(elements,indices,d){
 const map=new Map();
 for(const ei of indices)for(const face of facets(elements[ei],d)){
  const names=face.map(v=>JSON.stringify(v)),sorted=[...names].sort(),id=sorted.join('|');let inversions=0;
  for(let i=0;i<names.length;i++)for(let j=i+1;j<names.length;j++)if(names[i]>names[j])inversions++;
  const sign=inversions%2?-1:1,row=map.get(id)??{face,sign,count:0};row.count+=sign;map.set(id,row);
 }
 const faces=[];
 for(const row of map.values()){
  if(Math.abs(row.count)>1)throw Error('Overlapping/nonmanifold regional elements');
  if(row.count)faces.push(row.count===row.sign?row.face:[...row.face].reverse());
 }
 return {faces};
}
