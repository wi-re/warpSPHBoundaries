import {grid2d} from './compute2d-plan.js?v=20261022-7';
import {areaProfiles} from './analytic2d-plan.js?v=20261022-7';
import {profile2d} from './analytic2d.js?v=20261022-7';
import {FIELDS2D,signed2d,sampledJets,whole2d} from './browser-compute2d.js?v=20261022-7';
import {split32} from './analytic3d-plan.js?v=20261022-7';
export function compileLocal2d(config,curvature=false){
 const plan=grid2d(config),{c,geometry,x,y}=plan,id=FIELDS2D.indexOf(c.field),profiles=areaProfiles(c.kernel,id),W=profile2d(c.kernel),rows=profiles.flatMap(([p,R,k,kind])=>[p,R,...split32(k),kind,0]);
 rows.push(...W.flatMap(([p,R,k])=>[p,R,...split32(k)]));const sampled=c.sdf_source==='sampled'?sampledJets(c,geometry.vertices,x,y):null;
 for(const py of y)for(const px of x){const exact=signed2d([px,py],c,geometry.vertices),[d,n,k]=sampled?sampled([px,py]):[exact[0]/c.h,exact[1],c.h/c.radius],valid=n.every(Number.isFinite);rows.push(...split32(d),...split32(valid?n[0]:0),...split32(valid?n[1]:1),...split32(Number.isFinite(k)?k:0),+valid,+Number.isFinite(k),exact[0]/c.h,0);}
 const uniform=new ArrayBuffer(64),u=new Uint32Array(uniform),f=new Float32Array(uniform);u.set([c.resolution,id,2,profiles.length]);f.set([...split32(x[0]/c.h),...split32((x[1]-x[0])/c.h),c.h,id===12?0:whole2d(c.kernel)[id],2,curvature?-3:-2,...split32(y[0]/c.h),W.length,c.curvature_order],4);
 return {...plan,table:new Float32Array(rows),uniform:new Uint8Array(uniform),allowMissing:true};
}
