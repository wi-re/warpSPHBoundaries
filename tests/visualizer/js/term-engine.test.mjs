import test from 'node:test';
import assert from 'node:assert/strict';
import {FIELD_INFO,FIELD_IDS,axialTerms,selectAxial,ballSeries,radialJet,gauss} from '../../../src/warpSPHBoundaries/visualizer/static/term-engine.js';
import {computeCPU,distanceNormal,prepare,samples,pointTerm} from '../../../src/warpSPHBoundaries/visualizer/static/browser-compute.js';
const near=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const config={dimension:3,geometry:'sphere',radius:1,width:2,height:2,depth:2,h:.4,resolution:17,padding:1.15,kernel:'w4',field:'m00',models:['sphere','sdf'],quality:1};
test('complete field catalogue, quadrature and origin jets',()=>{
 assert.equal(FIELD_INFO.length,30);assert.equal(new Set(FIELD_INFO.map(x=>x[0])).size,30);
 near(gauss(16).reduce((s,[x,w])=>s+w,0),2);near(gauss(16).reduce((s,[x,w])=>s+w*x**30,0),2/31);
 for(const k of ['w2','w4','w6','cubic'])assert.ok(radialJet(0,k).every(Number.isFinite));
});
test('half-space whole-support signs and tensor normalization',()=>{
 for(const name of ['w2','w4','w6','cubic']){
  const v=axialTerms(-2,name);near(v[0],1);near(v[1],0);near(v[4],0);near(v[5],1);near(v[6],1);near(v[7],0);near(v[8],0);
  const boundary=axialTerms(0,name);near(boundary[0],.5);near(boundary[5],.5);near(boundary[6],.5);
  const n=[2/3,-2/3,1/3];near(selectAxial(boundary,n,'laplacian'),0);near(selectAxial(boundary,n,'Gx'),boundary[4]*2/3);
 }
});
test('curvature expansion converges toward true sphere for all nine invariants',()=>{
 for(const name of ['w2','w4','w6','cubic']){
  const errors=[4,8,16].map(R=>Math.max(...axialTerms(.27,name,R).map((v,i)=>Math.abs(v-axialTerms(.27,name,null,1/R)[i]))));
  assert.ok(errors[1]<errors[0]/6&&errors[2]<errors[1]/6);
 }
});
test('small-ball validity and whole exterior zeros',()=>{
 assert.ok(Number.isNaN(ballSeries([.49,0,0],.03,'cubic','m00')));assert.ok(Number.isNaN(ballSeries([0,0,0],.1,'w4','m00')));near(ballSeries([2,0,0],.1,'w4','m00'),0);
 const p=[.4,0,0],a=.02,exact=selectAxial(axialTerms(.4-a,'w4',a),[1,0,0],'m00');near(ballSeries(p,a,'w4','m00'),exact,1e-11);
});
test('static browser provider computes new derivative grids without a service',()=>{
 const r=computeCPU({...config,field:'laplacian'});assert.equal(r.results.length,2);assert.equal(r.shape[0],17);assert.ok(r.results.every(v=>v.min<0&&v.max>0));
 const centre=8*17*17+8*17+8,raw=atob(r.results[0].values),view=new DataView(Uint8Array.from(raw,c=>c.charCodeAt(0)).buffer);near(view.getFloat32(centre*4,true),0,1e-5);
});
test('tetrahedron geometry preserves Euclidean distances and cubature integrands',()=>{
 const c={...config,geometry:'tetrahedron',width:2,height:2,depth:2,field:'g1xy',models:['quadrature']},g=prepare(c);
 near(distanceNormal([0,0,0],c,g)[0],-1/Math.sqrt(3));near(distanceNormal([-2,-2,-2],c,g)[0],Math.sqrt(3));
 const q=samples(c);assert.equal(q.length/8,576);assert.ok(Number.isFinite(pointTerm([.3,.2,.1],c,'quadrature',g,q)));
});

test('selectable curvature and ball orders expose the individual corrections',()=>{
 const plane=axialTerms(.25,'w4'),order0=axialTerms(.25,'w4',null,.25,0);
 plane.forEach((v,i)=>near(v,order0[i]));
 const first=axialTerms(.25,'w4',null,.25,1),second=axialTerms(.25,'w4',null,.25,2);
 assert.ok(first[0]<plane[0]&&second[0]>first[0]);
 const p=[.31,.18,.11],a=.02,zero=ballSeries(p,a,'w4','m00',0),two=ballSeries(p,a,'w4','m00',2),four=ballSeries(p,a,'w4','m00',4);
 near(zero,4*Math.PI*a**3/3*radialJet(Math.hypot(...p),'w4')[0]);assert.notEqual(zero,two);assert.notEqual(two,four);
});
test('GPU readback distinguishes a valid zero from a finite missing-value placeholder',async()=>{
 const {decodeSamples2d,resultRow2d}=await import('../../../src/warpSPHBoundaries/visualizer/static/compute2d-plan.js');
 const raw=new Float32Array([0,.25,1,0,42,-.5,0,0]),decoded=decodeSamples2d(raw);
 assert.ok(raw.every(Number.isFinite));assert.equal(decoded.values[0],0);assert.ok(Number.isNaN(decoded.values[1]));assert.deepEqual(decoded.distance,[.25,-.5]);
 const row=resultRow2d('sdf',decoded.values,0,{quadrature:24,sdf_source:'analytic'},'WebGPU fp32');assert.deepEqual(row.values,[0,null]);assert.equal(row.missing,1);
});

test('convex triangle projection stays defined outside across resolutions and vertex transitions',async()=>{
 const {signed2d}=await import('../../../src/warpSPHBoundaries/visualizer/static/browser-compute2d.js');
 const v=[[-1,-.9],[1.1,-.7],[-.3,1.1]],c={geometry:'triangle'};
 for(const size of [81,129,193,257])for(let j=0;j<size;j++)for(let i=0;i<size;i++){
  const [d,n]=signed2d([-1.7475+3.595*i/(size-1),-1.6975+3.595*j/(size-1)],c,v);
  if(d>1e-10){assert.ok(n.every(Number.isFinite));near(Math.hypot(...n),1,2e-14);}
 }
 for(let k=0;k<3;k++){
  const a=v[k],b=v[(k+1)%3],e=b.map((x,i)=>x-a[i]),L=Math.hypot(...e),normal=[e[1]/L,-e[0]/L];
  for(const t of [1e-9,.25,.999999999])for(const d of [1e-12,.4]){
   const p=a.map((x,i)=>x+t*e[i]+d*normal[i]),jet=signed2d(p,c,v);
   near(jet[0],d,2e-15);jet[1].forEach((x,i)=>near(x,normal[i],t===.25?2e-14:3e-8));
  }
 }
 const square=[[-1,-1],[1,-1],[1,1],[-1,1]];
 assert.ok(signed2d([0,0],c,square)[1].every(Number.isNaN));
 assert.ok(signed2d([1,1],c,square)[1].every(Number.isNaN));
});
