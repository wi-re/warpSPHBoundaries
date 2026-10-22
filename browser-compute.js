import {FIELD_IDS,axialTerms,selectAxial,integrand,gauss,ballSeries} from './term-engine.js';
export const BROWSER_MODELS={
 sphere:['Sphere · polynomial integral','Exact-degree radial integration of each moment/kernel derivative; sphere only.'],
 sdf:['SDF · local plane','Local half-space moment family using geometry distance and normal.'],
 curvature:['SDF · curvature expansion','Quadratic-graph expansion applied to each integrand; sphere R/h ≥ 2.'],
 ball_series:['Small ball · mean-value series','Ball mean-value expansion; R/h ≤ 0.3, single smooth kernel piece.'],
 quadrature:['Solid · support cubature','Positive support-centred spherical quadrature. Approximate; refine quadrature quality.']
};
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],sub=(a,b)=>a.map((x,i)=>x-b[i]),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function geometry(c){
 if(c.geometry==='sphere')return {kind:'sphere',radius:c.radius,vertices:[],faces:[]};
 if(c.geometry==='tetrahedron')return {kind:'tetrahedron',vertices:[[-1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1]].map(v=>v.map((x,i)=>x*[c.width,c.height,c.depth][i]/2)),faces:[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]};
 const vertices=[];for(const z of [-1,1])for(const y of [-1,1])for(const x of [-1,1])vertices.push([x*c.width/2,y*c.height/2,z*c.depth/2]);
 return {kind:'box',vertices,faces:[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]};
}
export function prepare(c){
 const g=geometry(c);g.planes=[];
 if(c.geometry==='tetrahedron')for(const ids of g.faces){
  const [a,b,d]=ids.map(i=>g.vertices[i]),u=sub(b,a),v=sub(d,a);let n=cross(u,v);const length=Math.hypot(...n);n=n.map(x=>x/length);if(dot(n,a)<0)n=n.map(x=>-x);
  g.planes.push({a,b,d,u,v,n,uu:dot(u,u),uv:dot(u,v),vv:dot(v,v)});
 }
 return g;
}
export function distanceNormal(p,c,g=prepare(c)){
 if(c.geometry==='sphere'){const length=Math.hypot(...p);return [length-c.radius,length?p.map(x=>x/length):[1,0,0]];}
 if(c.geometry==='box'){
  const q=p.map((x,i)=>Math.abs(x)-[c.width,c.height,c.depth][i]/2),v=q.map((x,i)=>Math.max(x,0)*(p[i]>=0?1:-1)),length=Math.hypot(...v);
  if(length)return [length,v.map(x=>x/length)];let j=q.indexOf(Math.max(...q));return [q[j],[0,0,0].map((_,i)=>i===j?(p[i]>=0?1:-1):0)];
 }
 let inside=true,best=Infinity,target=[0,0,0],bestPlane=Infinity,insideNormal=[1,0,0];
 for(const f of g.planes){
  const {a,b,d,u,v,n,uu,uv,vv}=f,relative=sub(p,a),plane=dot(relative,n);inside&&=plane<=0;
  if(Math.abs(plane)<bestPlane){bestPlane=Math.abs(plane);insideNormal=n;}
  const ru=dot(relative,u),rv=dot(relative,v),alpha=(vv*ru-uv*rv)/(uu*vv-uv*uv),beta=(uu*rv-uv*ru)/(uu*vv-uv*uv);
  if(alpha>=0&&beta>=0&&alpha+beta<=1&&plane*plane<best){best=plane*plane;target=p.map((x,i)=>x-plane*n[i]);}
  for(const [start,end] of [[a,b],[b,d],[d,a]]){
   const edge=sub(end,start),t=Math.max(0,Math.min(1,dot(sub(p,start),edge)/dot(edge,edge))),candidate=start.map((x,i)=>x+t*edge[i]),delta=sub(p,candidate),dist=dot(delta,delta);
   if(dist<best){best=dist;target=candidate;}
  }
 }
 const length=Math.sqrt(best);return [inside?-length:length,inside?insideNormal:length?sub(p,target).map(x=>x/length):insideNormal];
}
export function inside(p,c,g){
 if(c.geometry==='sphere')return dot(p,p)<=c.radius*c.radius;
 if(c.geometry==='box')return Math.abs(p[0])<=c.width/2&&Math.abs(p[1])<=c.height/2&&Math.abs(p[2])<=c.depth/2;
 return g.planes.every(({a,n})=>dot(sub(p,a),n)<=0);
}
export function validate(c){
 if(!['sphere','box','tetrahedron'].includes(c.geometry)||!['w2','w4','w6','cubic'].includes(c.kernel)||!Object.hasOwn(FIELD_IDS,c.field))throw new Error('Unsupported geometry, kernel or term');
 for(const [key,lo,hi]of [['radius',.2,5],['width',.2,6],['height',.2,6],['depth',.2,6],['h',.15,2],['padding',1,2],['resolution',17,81]])if(!Number.isFinite(c[key])||c[key]<lo||c[key]>hi)throw new Error(`Invalid ${key}`);
 if(!Number.isInteger(c.resolution)||!Array.isArray(c.models)||c.models.length<1||c.models.length>3||new Set(c.models).size!==c.models.length||c.models.some(m=>!BROWSER_MODELS[m]))throw new Error('Choose one to three distinct browser models');
 if(![0,1,2].includes(c.curvature_order??2)||![0,2,4].includes(c.series_order??4))throw new Error('Unknown expansion order');
 if(![1,2].includes(c.quality??1))throw new Error('Unknown quadrature quality');return c;
}
export function samples(c){
 const quality=c.quality??1,nr=quality===1?8:12,nmu=quality===1?6:10,nphi=quality===1?12:20,knots=c.kernel==='cubic'?[0,.5,1]:[0,1],rows=[];
 for(let cut=0;cut<knots.length-1;cut++)for(const [node,weight] of gauss(nr)){
  const lo=knots[cut],hi=knots[cut+1],r=(lo+hi)/2+(hi-lo)/2*node;
  for(const [mu,wm]of gauss(nmu))for(let j=0;j<nphi;j++){
   const phi=2*Math.PI*(j+.5)/nphi,t=Math.sqrt(Math.max(0,1-mu*mu)),y=[r*t*Math.cos(phi),r*t*Math.sin(phi),r*mu],volume=weight*(hi-lo)/2*r*r*wm*2*Math.PI/nphi;
   const value=c.field==='grad_norm'?0:integrand(y,c.kernel,c.field)*volume;
   rows.push(...y,value,...[10,11,12].map(id=>integrand(y,c.kernel,id)*volume),0);
  }
 }
 return Float64Array.from(rows);
}
export function cubature(p,c,g,quadrature){
 let value=0,gx=0,gy=0,gz=0;
 for(let i=0;i<quadrature.length;i+=8){
  const source=[p[0]+c.h*quadrature[i],p[1]+c.h*quadrature[i+1],p[2]+c.h*quadrature[i+2]];
  if(inside(source,c,g)){value+=quadrature[i+3];gx+=quadrature[i+4];gy+=quadrature[i+5];gz+=quadrature[i+6];}
 }
 return c.field==='grad_norm'?Math.hypot(gx,gy,gz):value;
}
export function pointTerm(p,c,model,g=prepare(c),quadrature=null){
 const [distance,n]=distanceNormal(p,c,g),d=distance/c.h;
 if(model==='sphere')return selectAxial(axialTerms(d,c.kernel,c.radius/c.h),n,c.field);
 if(model==='sdf')return selectAxial(axialTerms(d,c.kernel),n,c.field);
 if(model==='curvature')return selectAxial(axialTerms(d,c.kernel,null,c.h/c.radius,c.curvature_order??2),n,c.field);
 if(model==='ball_series')return ballSeries(p.map(x=>x/c.h),c.radius/c.h,c.kernel,c.field,c.series_order??4);
 return cubature(p,c,g,quadrature??samples(c));
}
export function modelError(model,c){
 if(['sphere','curvature','ball_series'].includes(model)&&c.geometry!=='sphere')return 'This model requires sphere geometry';
 if(model==='curvature'&&c.radius/c.h<2)return 'Curvature model requires R/h ≥ 2; the cutoff is not an accuracy guarantee';
 if(model==='ball_series'&&c.radius/c.h>.3)return 'Small-ball model requires R/h ≤ 0.3';
 return null;
}
export function encode(values){
 const bytes=new Uint8Array(values.length*4),view=new DataView(bytes.buffer);values.forEach((x,i)=>view.setFloat32(i*4,x,true));let s='';
 for(let i=0;i<bytes.length;i+=16384)s+=String.fromCharCode(...bytes.subarray(i,i+16384));return btoa(s);
}
export function grid(c){
 const n=c.resolution,half=(c.geometry==='sphere'?c.radius:Math.max(c.width,c.height,c.depth)/2)+c.padding*c.h;
 return {n,half,axis:Array.from({length:n},(_,i)=>-half+2*half*i/(n-1))};
}
export function packageResult(c,axis,distance,results,milliseconds){
 return {dimension:3,schema_version:1,encoding:'base64-f32-le',order:'zyx',shape:[axis.length,axis.length,axis.length],config:{...c,dimension:3},x:axis,y:axis,z:axis,geometry:geometry(c),distance:encode(distance),results,milliseconds};
}
export function resultRow(model,values,milliseconds,engine,c){
 let min=Infinity,max=-Infinity,missing=0;for(const x of values){if(Number.isFinite(x)){min=Math.min(min,x);max=Math.max(max,x);}else missing++;}
 if(missing===values.length)return {model,label:BROWSER_MODELS[model][0],error:'No samples lie in the validity domain'};
 const notes=[`${engine}; direct selected-integrand evaluation, with no mass-grid finite differences.`,BROWSER_MODELS[model][1],'Dimensionless fields: moments divided by h^degree; gradients/Hessians multiplied by h/h². Display textures use float16; downloaded/probed grids use float32.'];
 if(model==='quadrature')notes.push(`Support cubature quality ${c.quality??1}; finite sampling and sharp solid-occupancy error remain. Compare quality levels; no renormalization, clamp or exactness claim.`);
 if(model==='curvature')notes.push(`Selected curvature order ${c.curvature_order??2}.`);
 if(model==='ball_series')notes.push(`Selected mean-value order ${c.series_order??4}.`);
 if(model==='ball_series')notes.push('Origin/knot/support-crossing samples remain missing. Order four is asymptotic; missing values are not replaced with zero.');
 return {model,label:BROWSER_MODELS[model][0],values:encode(values),min,max,missing,notes,milliseconds};
}
export function computeCPU(config,progress=()=>{}){
 const c=validate(config),{n,axis}=grid(c),g=prepare(c),distance=new Float32Array(n**3),results=[],quad=c.models.includes('quadrature')?samples(c):null,start=performance.now();
 for(let z=0;z<n;z++)for(let y=0;y<n;y++)for(let x=0;x<n;x++)distance[(z*n+y)*n+x]=distanceNormal([axis[x],axis[y],axis[z]],c,g)[0];
 for(const model of c.models){
  const error=modelError(model,c);if(error){results.push({model,label:BROWSER_MODELS[model][0],error});continue;}
  const tick=performance.now(),values=new Float32Array(n**3);
  for(let z=0;z<n;z++){for(let y=0;y<n;y++)for(let x=0;x<n;x++)values[(z*n+y)*n+x]=pointTerm([axis[x],axis[y],axis[z]],c,model,g,quad);progress(`${BROWSER_MODELS[model][0]} · ${Math.round((z+1)/n*100)}%`);}
  results.push(resultRow(model,values,performance.now()-tick,'Browser JavaScript float64 worker',c));
 }
 return packageResult(c,axis,distance,results,performance.now()-start);
}
