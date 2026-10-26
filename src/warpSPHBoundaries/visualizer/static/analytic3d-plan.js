// Production layout shared by WebGPU and native shader checks.
import {analyticProfile,analyticWhole,compensated} from './mesh3d-analytic.js?v=20261022-7';
import {FIELD_IDS} from './term-engine.js?v=20261022-7';
import {prepare,grid,validate} from './browser-compute.js?v=20261022-7';
export function split32(x){const hi=Math.fround(x);return [hi,Math.fround(x-hi)];}
function compact(profile){const terms=new Map();for(const [n,R,c]of profile){const key=n+','+R;const a=terms.get(key)??[];a.push(c);terms.set(key,a);}return [...terms].map(([key,cs])=>[...key.split(',').map(Number),compensated(cs)]).filter(t=>t[2]!==0);}
export function potentialProfile(profile){return compact(profile.flatMap(([n,R,c])=>[[n+2,R,c/(n+2)],[0,R,-c*R**(n+2)/(n+2)]]));}
function flux(profile){const tails=new Map();const terms=profile.map(([n,R,c])=>{const a=tails.get(R)??[];a.push(c*R**(n+3)/(n+3));tails.set(R,a);return [n,R,c/(n+3)];});const cs=[...tails].map(([R,a])=>[R,compensated(a)]);return compact([...terms,[-3,0,compensated(cs.map(t=>t[1]))],...cs.map(([R,c])=>[-3,R,-c])]);}
export function selectedProfiles(name,id){
 const W=analyticProfile(name),L=W.filter(([n])=>n).map(([n,R,c])=>[n-2,R,n*c]);let A=[],B=[],C=[];
 if(id===0||id>=13&&id<=21)A=flux(W);
 if(id>=1&&id<=9){B=potentialProfile(W);if(id>=4)A=flux(B).map(([n,R,c])=>[n,R,-c]);}
 if(id>=10&&id<=21||id===29)C=W;if(id>=22&&id<=28)C=L;
 return [A,B,C].flatMap((ps,kind)=>ps.map(([n,R,c])=>[n,R,c,kind]));
}
export function compileAnalytic3d(config){
 const c=validate(config);if(c.geometry==='sphere')throw new Error('Analytic polyhedra require tetrahedron or box geometry');
 const g=prepare(c),{n,half,axis}=grid(c),id=typeof c.field==='number'?c.field:FIELD_IDS[c.field],profile=selectedProfiles(c.kernel,id),rows=profile.flatMap(([p,R,k,kind])=>[p,R,...split32(k),kind,0]);
 const vertices=g.vertices.map(v=>v.map(x=>x/c.h)),centre=vertices[0].map((_,i)=>vertices.reduce((s,v)=>s+v[i],0)/vertices.length);
 const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 for(const face of g.faces){let v=face.map(i=>vertices[i]);if(dot(cross(sub(v[1],v[0]),sub(v[2],v[0])),sub(v[0],centre))<0)v=[v[0],v[2],v[1]];rows.push(...v.flat().flatMap(split32));}
 const uniform=new ArrayBuffer(64),u=new Uint32Array(uniform),f=new Float32Array(uniform);
 u.set([n,typeof c.field==='number'?c.field:FIELD_IDS[c.field],g.faces.length,profile.length]);
 f.set([...split32(-half/c.h),...split32((axis[1]-axis[0])/c.h),c.h,analyticWhole(c.kernel,c.field),3,0,...split32(-half/c.h),...split32(-half/c.h)],4);
 return {c,axis,table:new Float32Array(rows),uniform:new Uint8Array(uniform)};
}
