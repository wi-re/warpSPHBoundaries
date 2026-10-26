import {analyticProfile,compensated} from './mesh3d-analytic.js?v=20261022-7';
import {profile2d} from './analytic2d.js?v=20261022-7';
import {grid2d} from './compute2d-plan.js?v=20261022-7';
import {FIELDS2D} from './browser-compute2d.js?v=20261022-7';
import {FIELD_IDS} from './term-engine.js?v=20261022-7';
import {validate,grid} from './browser-compute.js?v=20261022-7';
import {split32} from './analytic3d-plan.js?v=20261022-7';
const pairs=[[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]];
export function compileSeries(config,dimension){
 const base=dimension===2?grid2d(config):(()=>{const c=validate(config),g=grid(c);return {c,x:g.axis,y:g.axis};})(),{c,x,y}=base,id=dimension===2?FIELDS2D.indexOf(c.field):FIELD_IDS[c.field],a=c.radius/c.h,profile=dimension===2?profile2d(c.kernel):analyticProfile(c.kernel),normal=id===(dimension===2?12:29),terms=[];
 const alpha=(i,j=-1)=>[0,1,2].map(k=>+(i===k)+ +(j===k));
 const add=(n,R,k,al,p,component=0)=>{if(k)terms.push([al,p,R,k,component]);};
 for(const [n,R,k]of profile){
  if(dimension===2){if(id<=5){const al=[[0,0,0],[1,0,0],[0,1,0],[2,0,0],[1,1,0],[0,2,0]][id];add(n,R,k,al,n);}else if(id<=7)add(n,R,-n*k,alpha(id-6),n-2);else if(id<=11){const ij=[[0,0],[0,1],[0,1],[1,1]][id-8];add(n,R,-n*k,alpha(...ij),n-2);}else for(let i=0;i<2;i++)add(n,R,-n*k,alpha(i),n-2,i);}
  else{if(id===0)add(n,R,k,[0,0,0],n);else if(id<=9)add(n,R,k,id<=3?alpha(id-1):alpha(...pairs[id-4]),n);else if(id<=12)add(n,R,-n*k,alpha(id-10),n-2);else if(id<=21)add(n,R,-n*k,alpha(Math.floor((id-13)/3),(id-13)%3),n-2);else if(id<=27){const [i,j]=pairs[id-22];if(i===j)add(n,R,n*k,[0,0,0],n-2);add(n,R,n*(n-2)*k,alpha(i,j),n-4);}else if(id===28)add(n,R,n*(n+1)*k,[0,0,0],n-2);else for(let i=0;i<3;i++)add(n,R,-n*k,alpha(i),n-2,i);}
 }
 const lap=ts=>ts.flatMap(([al,p,R,k,j])=>{const out=[],r=p*(p+dimension-2+2*al.reduce((a,b)=>a+b,0));if(r)out.push([al,p-2,R,k*r,j]);al.forEach((n,i)=>{if(n>=2)out.push([al.map((v,l)=>v-2*+(i===l)),p,R,k*n*(n-1),j]);});return out;});
 const first=lap(terms),second=lap(first),order=c.series_order??4,retained=[...terms];
 if(order>=2)retained.push(...first.map(([al,p,R,k,j])=>[al,p,R,k*a*a/(dimension===2?8:10),j]));if(order>=4)retained.push(...second.map(([al,p,R,k,j])=>[al,p,R,k*a**4/(dimension===2?192:280),j]));
 const grouped=new Map();for(const [al,p,R,k,j]of retained){const key=[...al,p,R,j].join(',');const v=grouped.get(key)??[];v.push(k);grouped.set(key,v);}
 const volume=dimension===2?Math.PI*a*a:4*Math.PI*a**3/3,rows=[];for(const [key,values]of grouped){const [ax,ay,az,p,R,j]=key.split(',').map(Number),k=volume*compensated(values);if(k)rows.push(ax,ay,az,p,R,...split32(k),j);}
 const uniform=new ArrayBuffer(64),u=new Uint32Array(uniform),f=new Float32Array(uniform);u.set([c.resolution,id,rows.length/8,c.kernel==='cubic'?1:0]);f.set([...split32(x[0]/c.h),...split32((x[1]-x[0])/c.h),c.h,a,dimension,-1,...split32(y[0]/c.h),...split32(dimension===2?0:x[0]/c.h)],4);
 return {...base,table:new Float32Array(rows),uniform:new Uint8Array(uniform),allowMissing:true};
}
