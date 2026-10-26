// Shared host-side layout for browser WebGPU dispatch and native WGSL tests.
import {FIELDS2D,MODELS2D,validate2d,geometry2d,fluxPlan} from './browser-compute2d.js?v=20261022-7';
import {SHAPES,gauss} from './term-engine.js?v=20261022-7';
export function grid2d(payload){
 const c=validate2d(payload),geometry=geometry2d(c),n=c.resolution;
 const lo=[0,1].map(i=>c.geometry==='circle'?-c.radius:Math.min(...geometry.vertices.map(p=>p[i]))),hi=[0,1].map(i=>c.geometry==='circle'?c.radius:Math.max(...geometry.vertices.map(p=>p[i]))),half=Math.max(...hi.map((x,i)=>x-lo[i]))/2+c.padding*c.h;
 const [x,y]=lo.map((v,i)=>Array.from({length:n},(_,j)=>(v+hi[i])/2-half+2*half*j/(n-1)));
 return {c,geometry,x,y};
}
export function meshBoundary2d(triangles){
 const edges=new Map();
 for(const triangle of triangles)for(let i=0;i<3;i++){
  const a=triangle[i],b=triangle[(i+1)%3],ka=JSON.stringify(a),kb=JSON.stringify(b),forward=ka<kb,key=forward?ka+'|'+kb:kb+'|'+ka;
  const row=edges.get(key)??{a:forward?a:b,b:forward?b:a,count:0};row.count+=forward?1:-1;edges.set(key,row);
 }
 const boundary=[];for(const {a,b,count}of edges.values())for(let i=0;i<Math.abs(count);i++)boundary.push(count>0?[a,b]:[b,a]);return boundary;
}
export function modelError2d(model,c,x,y){
 if(['rectangle','strip'].includes(model)&&c.geometry!=='rectangle')return 'Built-in rectangle requires rectangle geometry';
 if(['disk','disk_table','disk_series'].includes(model)&&c.geometry!=='circle')return 'Circle reference requires circle geometry';
 if(['curvature','curvature_analytic'].includes(model)&&(c.geometry!=='circle'||c.radius/c.h<2))return 'Curvature model requires circle geometry and R/h ≥ 2';
 if(model==='disk_series'&&c.radius/c.h>.3)return 'Small-disk series requires R/h ≤ 0.3';
 if(c.sdf_source==='sampled'&&['sdf','curvature','sdf_analytic','curvature_analytic'].includes(model)){
  const spacing=c.sdf_spacing*c.h,nx=Math.ceil((x.at(-1)-x[0])/spacing+3)+4,ny=Math.ceil((y.at(-1)-y[0])/spacing+3)+4;
  if(nx*ny>2_000_000)return 'SDF lattice too large; increase SDF spacing';
 }
 return null;
}
export function compile2d(payload,diskTable=null){
 const plan=grid2d(payload),{c,geometry,x,y}=plan,edges=meshBoundary2d(geometry.triangles),table=new Float32Array(236+geometry.vertices.length*2+edges.length*4+(diskTable?.length??0)),shape=SHAPES[c.kernel];
 table.set(gauss(c.quadrature).flat());
 const flux=fluxPlan(c.kernel);
 for(let degree=0;degree<=2;degree++){const base=128+degree*32;table[base]=flux[degree].total;table.set(flux[degree].inner,base+1);table.set(flux[degree].tail,base+13);}
 table.set(flux[0].inner.map((v,i)=>v*(i+2)),224);
 table.set(geometry.vertices.flat().map(x=>x/c.h),236);table.set(edges.flat(2).map(x=>x/c.h),236+geometry.vertices.length*2);
 const uniform=new ArrayBuffer(96),u=new Uint32Array(uniform),f=new Float32Array(uniform);
 u.set([c.resolution,FIELDS2D.indexOf(c.field),['rectangle','triangle','circle'].indexOf(c.geometry),0,geometry.vertices.length,edges.length,c.quadrature,['cubic','w2','w4','w6'].indexOf(c.kernel)]);
 const tableOffset=236+geometry.vertices.length*2+edges.length*4;if(diskTable)table.set(diskTable,tableOffset);
 f.set([x[0]/c.h,y[0]/c.h,(x[1]-x[0])/c.h,c.h,c.radius/c.h,c.sdf_spacing,c.sdf_source==='sampled'?1:0,flux[0].inner[0]*(c.kernel==='cubic'?4:2),shape.n,shape.P[1]??0,shape.P[2]??0,shape.P[3]??0,x[0]/c.h-3*c.sdf_spacing,y[0]/c.h-3*c.sdf_spacing,tableOffset,c.curvature_order],8);
 return {...plan,table,uniform:new Uint8Array(uniform)};
}
export function resultRow2d(model,values,milliseconds,c,engine){
 let min=Infinity,max=-Infinity,missing=0;
 const data=Array.from(values,v=>{if(!Number.isFinite(v)){missing++;return null;}min=Math.min(min,v);max=Math.max(max,v);return v;});
 const notes=[`Browser ${engine}; selected-integrand evaluation without mass-grid finite differences.`,MODELS2D[model][1],`${c.quadrature} Gauss nodes per smooth panel. Numerical quadrature approximates the represented model; arithmetic and geometry/model errors are separate.`];
 if(['analytic','analytic_mesh','sdf_analytic','curvature_analytic','strip'].includes(model))notes.splice(0,notes.length,`Browser ${engine}; elementary antiderivatives of the represented model or retained coefficients. No quadrature; geometry-model, expansion and arithmetic errors remain as applicable.`);
 if(model==='disk_table')notes.splice(0,notes.length,`Browser ${engine}; disk Chebyshev table, all moments. Reference uses split circle-flux quadrature. Interpolation and coefficient errors remain.`);
 if(model==='disk_series')notes.splice(0,notes.length,`Browser ${engine}; disk mean-value expansion, order ${c.series_order}. Truncation error remains. Origin/knot/support crossings stay missing.`);
 if(model==='mesh')notes.push('Opposite shared triangle edges cancel algebraically before fp32 boundary evaluation; the represented mesh is unchanged.');
 if(c.geometry==='circle'&&['mesh','exact'].includes(model))notes.push(`The inscribed ${c.segments}-gon approximates the circle.`);
 if(['sdf','curvature','sdf_analytic','curvature_analytic'].includes(model))notes.push(c.sdf_source==='sampled'?'Bilinearly sampled SDF with finite-difference normals and curvature; fp32 cancellation in small lattice spacings can affect jets.':'Analytic signed distance and normal; ambiguous normal samples remain missing.');
 return {model,engine,label:MODELS2D[model][0],values:data,min:Number.isFinite(min)?min:null,max:Number.isFinite(max)?max:null,missing,notes,milliseconds};
}

// Shader buffers contain only finite values plus an explicit 0/1 validity flag.
// Missing values are introduced on the JS side, after GPU readback.
export function decodeSamples2d(raw){
 if(raw.length%4)throw new Error('Invalid 2D GPU sample stride');
 const values=new Float32Array(raw.length/4),distance=new Array(values.length);
 for(let i=0;i<values.length;i++){values[i]=raw[i*4+2]>.5?raw[i*4]:NaN;distance[i]=raw[i*4+1];}
 return {values,distance};
}
