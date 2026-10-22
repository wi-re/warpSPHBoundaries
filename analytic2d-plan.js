import {grid2d,meshBoundary2d} from './compute2d-plan.js?v=20261022-7';
import {profile2d} from './analytic2d.js?v=20261022-7';
import {FIELDS2D,whole2d} from './browser-compute2d.js?v=20261022-7';
import {potentialProfile,split32} from './analytic3d-plan.js?v=20261022-7';
export function areaProfiles(name,id){const W=profile2d(name),L=W.filter(([n])=>n).map(([n,R,v])=>[n-2,R,n*v]),selected=id<=5?W:L;return [...selected.map(t=>[...t,1]),...([3,4,5,8,9,10,11].includes(id)?potentialProfile(selected).map(t=>[...t,2]):[])];}
export function compileAnalytic2d(config){
 const {c,geometry,x,y}=grid2d(config),id=FIELDS2D.indexOf(c.field),W=profile2d(c.kernel),L=W.filter(([n])=>n).map(([n,R,v])=>[n-2,R,n*v]),selected=id<=5?W:L;
 const profiles=areaProfiles(c.kernel,id);
 const rows=profiles.flatMap(([n,R,v,kind])=>[n,R,...split32(v),kind,0]);
 for(const triangle of geometry.triangles)rows.push(...triangle.flatMap(p=>[p[0]/c.h,p[1]/c.h,0]).flatMap(split32));
 const edges=meshBoundary2d(geometry.triangles);for(const edge of edges)rows.push(...edge.flatMap(p=>[p[0]/c.h,p[1]/c.h,0]).flatMap(split32));
 const uniform=new ArrayBuffer(64),u=new Uint32Array(uniform),f=new Float32Array(uniform);u.set([c.resolution,id,geometry.triangles.length,profiles.length]);
 f.set([...split32(x[0]/c.h),...split32((x[1]-x[0])/c.h),c.h,id===12?0:whole2d(c.kernel)[id],2,edges.length,...split32(y[0]/c.h),0,0],4);
 return {c,x,y,geometry,table:new Float32Array(rows),uniform:new Uint8Array(uniform)};
}

export function compileStrip2d(config){
 const plan=grid2d(config),{c,x,y}=plan,id=FIELDS2D.indexOf(c.field),profiles=areaProfiles(c.kernel,id),angle=c.angle*Math.PI/180,t=[Math.cos(angle),Math.sin(angle)],n=[-t[1],t[0]],half=c.height/(2*c.h),extent=Math.hypot(Math.max(...x.map(Math.abs))/c.h,Math.max(...y.map(Math.abs))/c.h)+2;
 const v=[[-extent,-half],[extent,-half],[extent,half],[-extent,half]].map(([a,b])=>t.map((v,i)=>a*v+b*n[i]));
 const rows=profiles.flatMap(([p,R,k,kind])=>[p,R,...split32(k),kind,0]);for(const ids of [[0,1,2],[0,2,3]])rows.push(...ids.flatMap(i=>[...v[i],0]).flatMap(split32));for(let i=0;i<4;i++)rows.push(...[v[i],v[(i+1)%4]].flatMap(p=>[...p,0]).flatMap(split32));
 const uniform=new ArrayBuffer(64),u=new Uint32Array(uniform),f=new Float32Array(uniform);u.set([c.resolution,id,2,profiles.length]);f.set([...split32(x[0]/c.h),...split32((x[1]-x[0])/c.h),c.h,id===12?0:whole2d(c.kernel)[id],2,4,...split32(y[0]/c.h),0,0],4);
 return {...plan,table:new Float32Array(rows),uniform:new Uint8Array(uniform)};
}
