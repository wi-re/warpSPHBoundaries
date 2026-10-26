// Elementary face-to-edge integrals, unit support, y = source - query.
// Port of mesh3d/reference.py; no quadrature or sampled solid occupancy.
// See docs/derivations/tet-face-edge-chain.md. Exact in real arithmetic;
// JavaScript arithmetic/transcendentals remain float64 (not certified intervals).
import {FIELD_IDS,SHAPES} from './term-engine.js?v=20261022-7';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),sub=(a,b)=>a.map((x,i)=>x-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const choose=(n,k)=>{let v=1;for(let i=1;i<=k;i++)v=v*(n-i+1)/i;return v;};
export function compensated(values){let s=0,c=0;for(const x of values){const t=s+x;c+=Math.abs(s)>=Math.abs(x)?(s-t)+x:(x-t)+s;s=t;}return s+c;}
const memo=(fn)=>{const cache=new Map();return (...args)=>{const key=args.join(',');if(!cache.has(key))cache.set(key,fn(...args));return cache.get(key);};};
const zero=[0,0,0],unit=i=>zero.map((_,j)=>+(i===j));
const lower=(a,i)=>a.map((x,j)=>x-+(i===j));
const pairs=[[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]];
const profileCache=new Map();
export function analyticProfile(name){
 if(profileCache.has(name))return profileCache.get(name);
 const k=SHAPES[name],blocks=[];
 if(name==='cubic'){
  blocks.push([1,[1,-3,3,-1]],[.5,[-.5,3,-6,4]]);
 }else{
  // Form the rational shape before multiplying by its common normalization.
  const denominator=name==='w4'?3:1,P=k.P.map(x=>Math.round(x*denominator)),coeffs=Array(k.n+P.length).fill(0);
  for(let j=0;j<=k.n;j++)for(let i=0;i<P.length;i++)coeffs[i+j]+=choose(k.n,j)*(-1)**j*P[i];
  blocks.push([1,coeffs.map(c=>c/denominator)]);
 }
 const profile=blocks.flatMap(([R,cs])=>cs.flatMap((c,n)=>c?[[n,R,c*k.C]]:[]));profileCache.set(name,profile);return profile;
}
export function analyticWhole(name,field){
 const id=typeof field==='number'?field:FIELD_IDS[field];
 if(id===0||id===13||id===17||id===21)return 1;
 if([4,7,9].includes(id))return 4*Math.PI/3*compensated(analyticProfile(name).map(([n,R,c])=>c*R**(n+5)/(n+5)));
 return 0;
}
// All I_m and K_m at an edge endpoint. The only negative I needed is I_-1.
function primitives(s,w,a){
 const b=Math.abs(w),c=Math.hypot(w,a),r=Math.hypot(s,c),I=Array(17),K=Array(17);
 I[0]=s;I[1]=(s*r+(c?c*c*Math.asinh(s/c):0))/2;
 for(let m=2;m<I.length;m++)I[m]=(s*r**m+m*c*c*I[m-2])/(m+1);
 const km1=a?Math.atan2(a*s,b*r)/(a*b):s/(b*b*r);
 K[0]=Math.atan2(s,b)/b;K[1]=Math.asinh(s/c)+a*a*km1;
 for(let m=2;m<K.length;m++)K[m]=I[m-2]+a*a*K[m-2];
 return {I,K,km1,r,c};
}
export function linePrimitive(j,n,s,c){
 const r=Math.hypot(s,c);
 if(j===0){if(n===-1)return Math.asinh(s/c);let prev=s;if(!n)return prev;let curr=(s*r+(c?c*c*Math.asinh(s/c):0))/2;if(n===1)return curr;const values=[prev,curr];for(let m=2;m<=n;m++)values[m]=(s*r**m+m*c*c*values[m-2])/(m+1);return values[n];}
 if(j%2===1){const h=(j-1)/2;return compensated(Array.from({length:h+1},(_,i)=>choose(h,i)*(-c*c)**(h-i)*r**(n+2*i+2)/(n+2*i+2)));}
 const h=j/2;return compensated(Array.from({length:h+1},(_,i)=>choose(h,i)*(-c*c)**(h-i)*linePrimitive(0,n+2*i,s,c)));
}
function multiply(p,q){const out=new Map();for(const [a,c]of p)for(const [b,d]of q){const k=a.split(',').map((v,i)=>+v+(+b.split(',')[i])).join(',');out.set(k,(out.get(k)??0)+c*d);}return out;}
export class AnalyticTriangle {
 constructor(vertices,query=[0,0,0]){
  const p=vertices.map(v=>sub(v,query)),D=sub(p[1],p[0]),N=cross(D,sub(p[2],p[0])),length=Math.hypot(...N),ell=Math.hypot(...D);
  if(!length)throw new Error('Degenerate analytic triangle');
  this.normal=N.map(x=>x/length);this.z=dot(N,p[0])/length;this.a=Math.abs(this.z);this.e1=D.map(x=>x/ell);this.e2=cross(this.normal,this.e1);
  this.edges=p.map((v,j)=>{
   const q=p[(j+1)%3],d=sub(q,v),L=Math.hypot(...d),t=d.map(x=>x/L),mu=cross(t,this.normal);
   const w=dot(cross(d,N),v)/(L*length),lo=dot(d,v)/L,hi=dot(d,q)/L;
   return {w,lo,hi,mu:[dot(mu,this.e1),dot(mu,this.e2)],t:[dot(t,this.e1),dot(t,this.e2)],primitive:memo(s=>primitives(s,w,this.a))};
  });
  this.scalar=memo(this.scalar.bind(this));this.local=memo(this.local.bind(this));this.block=memo(this.block.bind(this));this.line=memo(this.line.bind(this));
 }
 scalar(n,R=Infinity){
  const a=this.a;if(a>=R)return 0;
  if(n===-3&&!a)throw new Error('Singular auxiliary face profile at zero plane offset');
  const contributions=[];
  for(const edge of this.edges){const {w,lo,hi}=edge;if(w===0)continue;const cuts=[lo,hi],c2=w*w+a*a;
   if(Number.isFinite(R)&&R*R>c2){const L=Math.sqrt(R*R-c2);for(const x of [-L,L])if(lo<x&&x<hi)cuts.push(x);}cuts.sort((x,y)=>x-y);
   for(let j=1;j<cuts.length;j++){
    const l=cuts[j-1],u=cuts[j],A=edge.primitive(l),B=edge.primitive(u),inside=R===Infinity||((l+u)/2)**2+c2<R*R;
    // A finite endpoint angle difference, including chords crossing s=0.
    const dK0=Math.atan2(Math.abs(w)*(u-l),w*w+l*u)/Math.abs(w);
    const primitive=n+2===-1?B.km1-A.km1:B.K[n+2]-A.K[n+2];
    const value=inside?primitive-a**(n+2)*dK0:(R**(n+2)-a**(n+2))*dK0;
    contributions.push(w*value/(n+2));
   }
  }return compensated(contributions);
 }
 line(edgeIndex,b0,b1,n,R){
  const e=this.edges[edgeIndex],c2=e.w*e.w+this.a*this.a;if(c2>=R*R)return 0;
  const L=Math.sqrt(R*R-c2),lo=Math.max(e.lo,-L),hi=Math.min(e.hi,L);if(lo>=hi)return 0;
  let poly=[1];for(const [b,mu,t]of [[b0,e.mu[0],e.t[0]],[b1,e.mu[1],e.t[1]]]){
   const q=Array.from({length:b+1},(_,j)=>choose(b,j)*(e.w*mu)**(b-j)*t**j),out=Array(poly.length+b).fill(0);
   poly.forEach((v,i)=>q.forEach((u,j)=>out[i+j]+=v*u));poly=out;
  }
  const c=Math.sqrt(c2);return compensated(poly.map((v,j)=>v?(v*(linePrimitive(j,n,hi,c)-linePrimitive(j,n,lo,c))):0));
 }
 local(b0,b1,n,R){
  if(b0===0&&b1===0)return this.scalar(n,R);
  const i=b0?0:1,b=[b0,b1];b[i]--;
  let result=compensated(this.edges.map((e,j)=>e.mu[i]*(this.line(j,...b,n+2,R)-R**(n+2)*this.line(j,...b,0,R))))/(n+2);
  if(b[i]){const factor=b[i];b[i]--;result-=factor*(this.local(...b,n+2,R)-R**(n+2)*this.local(...b,0,R))/(n+2);}return result;
 }
 block(a0,a1,a2,n,R){
  const alpha=[a0,a1,a2];let poly=new Map([['0,0',1]]);
  for(let i=0;i<3;i++)for(let j=0;j<alpha[i];j++)poly=multiply(poly,new Map([['0,0',this.z*this.normal[i]],['1,0',this.e1[i]],['0,1',this.e2[i]]]));
  return compensated([...poly].map(([b,c])=>c?c*this.local(...b.split(',').map(Number),n,R):0));
 }
 integrate(profile,alpha=zero){return compensated(profile.map(([n,R,c])=>c*this.block(...alpha,n,R)));}
}
export class AnalyticSolid {
 constructor(vertices,faces,query=[0,0,0]){this.faces=faces.map(ids=>new AnalyticTriangle(ids.map(i=>vertices[i]),query));this.block=memo(this.block.bind(this));}
 block(a0,a1,a2,n,R){
  const alpha=[a0,a1,a2];if(!alpha.some(Boolean))return compensated(this.faces.map(f=>f.z===0?0:f.z*(f.scalar(n,R)+R**(n+3)*(f.scalar(-3)-f.scalar(-3,R)))/(n+3)));
  const i=alpha.findIndex(a=>a>0),beta=lower(alpha,i);
  let result=compensated(this.faces.map(f=>f.normal[i]*(f.block(...beta,n+2,R)-R**(n+2)*f.block(...beta,0,R))))/(n+2);
  if(beta[i]){const a=lower(beta,i);result-=beta[i]*(this.block(...a,n+2,R)-R**(n+2)*this.block(...a,0,R))/(n+2);}return result;
 }
 integrate(profile,alpha=zero){return compensated(profile.map(([n,R,c])=>c*this.block(...alpha,n,R)));}
 field(name,field){
  const id=typeof field==='number'?field:FIELD_IDS[field],W=analyticProfile(name),L=W.filter(([n])=>n).map(([n,R,c])=>[n-2,R,n*c]);
  if(id===0)return this.integrate(W);
  if(id>=1&&id<=9){const alpha=id<=3?unit(id-1):zero.map((_,i)=>+(i===pairs[id-4][0])+(i===pairs[id-4][1]));return this.integrate(W,alpha);}
  if(id>=10&&id<=12)return -compensated(this.faces.map(f=>f.normal[id-10]*f.integrate(W)));
  if(id>=13&&id<=21){const i=Math.floor((id-13)/3),j=(id-13)%3;return (i===j?this.integrate(W):0)-compensated(this.faces.map(f=>f.normal[j]*f.integrate(W,unit(i))));}
  if(id>=22&&id<=27){const [i,j]=pairs[id-22];return compensated(this.faces.map(f=>f.normal[i]*f.integrate(L,unit(j))));}
  if(id===28)return compensated(this.faces.map(f=>f.z===0?0:f.z*f.integrate(L)));
  if(id===29)return Math.hypot(...[10,11,12].map(f=>this.field(name,f)));
  throw new Error('Unknown analytic solid term');
 }
}
