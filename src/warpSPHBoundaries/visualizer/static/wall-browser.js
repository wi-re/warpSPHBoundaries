// Prescribed-state wall adapter. No pressure/porosity solver or native imports.
import {add,sub,mul,dot,monomials,baseline,compileProfiles,prepareSamples,selectedElements,mergeVolumeFacets} from './fluxbridge-plan.js';
import {fluxBridgeGPU,dispatchFluxBridge} from './fluxbridge-gpu.js';
export const DEFAULT_WALL={dimension:2,geometry:'wall',width:2,thickness:.4,depth:1,fraction:.5,subdivisions:2,grading:1,angle:0,center:[0,0,0],h:.6,kernel:'w2',precision:'float32',backend:'numerical',device:'cpu',preparation:'host',resolution:9,slice:0,slice_plane:'xy',degree:0,resistance:4,anisotropy:4,tensor_angle:0,dt:.1,fluid:[0,1,0],skeleton:[0,0,0],fluid_gradient:[[0,0,0],[0,0,0],[0,0,0]],skeleton_gradient:[[0,0,0],[0,0,0],[0,0,0]],porous_enabled:true,exhaustive:false,segments:16,fibres:3,search:'host'};
const identity=d=>Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>+(i===j)));
const mv=(A,v)=>A.map(row=>dot(row,v));
const transpose=A=>A[0].map((_,i)=>A.map(row=>row[i]));
const mm=(A,B)=>A.map(row=>transpose(B).map(col=>dot(row,col)));
const matrixAdd=(A,B)=>A.map((row,i)=>add(row,B[i]));
const matrixScale=(A,s)=>A.map(row=>mul(row,s));
const factorial=n=>n<2?1:n*factorial(n-1);
const key=(alpha,ds=[])=>alpha.join(',')+'|'+ds.join(',');
export function validateWall(payload={}){
 const c={...structuredClone(DEFAULT_WALL),...payload};
 for(const [name,lo,hi,integer]of [['dimension',2,3,true],['width',.2,6],['thickness',.05,2],['depth',.1,4],['fraction',0,1],['subdivisions',1,12,true],['grading',.5,3],['angle',-180,180],['h',.05,3],['resolution',3,33,true],['slice',-4,4],['degree',0,3,true],['resistance',0,1000],['anisotropy',1,100],['tensor_angle',-180,180],['dt',0,10],['segments',8,64,true],['fibres',1,8,true]]){
  if(!Number.isFinite(c[name])||c[name]<lo||c[name]>hi||(integer&&!Number.isInteger(c[name])))throw Error(`Invalid ${name}`);
 }
 for(const [name,choices]of [['geometry',['wall','disk','fibres']],['kernel',['w2','w4','w6','cubic']],['slice_plane',['xy','xz','yz']]])if(!choices.includes(c[name]))throw Error(`Unknown ${name}`);
 if(c.dimension===3&&c.geometry!=='wall')throw Error('3D browser explorer currently requires a tetrahedral wall');
 for(const name of ['porous_enabled','exhaustive'])if(typeof c[name]!=='boolean')throw Error(`Invalid ${name}`);
 for(const name of ['center','fluid','skeleton','fluid_gradient','skeleton_gradient']){
  const a=c[name],gradient=name.endsWith('gradient');if(!Array.isArray(a)||a.length!==3||(gradient&&a.some(r=>!Array.isArray(r)||r.length!==3))||a.flat().some(v=>!Number.isFinite(v)||Math.abs(v)>100))throw Error(`Invalid ${name}`);
 }
 if(c.precision!=='float32'||c.backend!=='numerical'||c.preparation!=='host'||c.search!=='host')throw Error('Browser provider requires numerical fp32 with JavaScript host preparation/search');
 delete c.queries;return c;
}
export function generateWall(c){
 const d=c.dimension,a=c.angle*Math.PI/180,R=identity(d);R[0][0]=R[1][1]=Math.cos(a);R[0][1]=-Math.sin(a);R[1][0]=Math.sin(a);
 const center=c.center.slice(0,d),elements=[],regions=[];
 const append=(v,label)=>{elements.push(v.map(p=>add(mv(R,p),center)));regions.push(label);};
 if(c.geometry==='wall'){
  const cut=c.width*(c.fraction-.5);
  for(const [left,right,label]of [[-c.width/2,cut,'porous'],[cut,c.width/2,'solid']]){
   if(left===right)continue;
   const xs=Array.from({length:c.subdivisions+1},(_,i)=>left+(right-left)*(i/c.subdivisions)**c.grading);
   for(let i=0;i<c.subdivisions;i++){
    const lo=[xs[i],-c.thickness/2,...(d===3?[-c.depth/2]:[])],hi=[xs[i+1],c.thickness/2,...(d===3?[c.depth/2]:[])];
    if(d===2){const v=[lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]];append([v[0],v[1],v[2]],label);append([v[0],v[2],v[3]],label);}
    else for(const order of [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]){let q=[...lo];const v=[q];for(const axis of order){q=[...q];q[axis]=hi[axis];v.push(q);}append(v,label);}
   }
  }
 }else{
  const count=c.geometry==='fibres'?c.fibres:1,radius=count>1?Math.min(c.thickness/2,c.width/(3*count)):c.width/2;
  for(let i=0;i<count;i++){
   const x=count===1?0:-c.width/2+radius+i*(c.width-2*radius)/(count-1);
   const ring=Array.from({length:c.segments},(_,j)=>[x+radius*Math.cos(2*Math.PI*j/c.segments),radius*Math.sin(2*Math.PI*j/c.segments)]);
   ring.forEach((p,j)=>append([[x,0],p,ring[(j+1)%ring.length]],'solid'));
  }
 }
 return {elements,regions,rotation:R};
}
export function wallRequests(d,degree){const found=new Map();for(const r of [...baseline(d),...monomials(d,degree+2).map(alpha=>({alpha,derivative:[]}))])found.set(key(r.alpha,r.derivative),r);return [...found.values()];}
function inverse(A){
 const d=A.length,rows=A.map((r,i)=>[...r,...identity(d)[i]]);
 for(let j=0;j<d;j++){
  let best=j;for(let i=j+1;i<d;i++)if(Math.abs(rows[i][j])>Math.abs(rows[best][j]))best=i;[rows[j],rows[best]]=[rows[best],rows[j]];
  const pivot=rows[j][j];if(!Number.isFinite(pivot)||Math.abs(pivot)<1e-14)throw Error('Singular local resistance response');
  rows[j]=mul(rows[j],1/pivot);for(let i=0;i<d;i++)if(i!==j)rows[i]=sub(rows[i],mul(rows[j],rows[i][j]));
 }return rows.map(r=>r.slice(d));
}
export function contractWall(c,queries,requests,moments,rotation){
 const d=c.dimension,lookup=new Map(requests.map((r,i)=>[key(r.alpha,r.derivative),i])),z=Array(d).fill(0),eye=identity(d),L=c.fluid_gradient.slice(0,d).map(row=>row.slice(0,d)),S=c.skeleton_gradient.slice(0,d).map(row=>row.slice(0,d)),A=L.map((r,i)=>sub(r,S[i]));
 const angle=c.tensor_angle*Math.PI/180,axis=[Math.cos(angle),Math.sin(angle),...(d===3?[0]:[])],tensor=mm(mm(rotation,eye.map((r,i)=>r.map((v,j)=>c.resistance*(v+(c.anisotropy-1)*axis[i]*axis[j])))),transpose(rotation));
 const normal=rotation.map(r=>r[1]),t=rotation.map(r=>r[0]),center=c.center.slice(0,d),out={};
 for(let qi=0;qi<queries.length;qi++){
  const get=(alpha,ds=[])=>moments[qi][lookup.get(key(alpha,ds))];
  const m0=get(z),m1=eye.map(e=>get(e)),m2=eye.map(a=>eye.map(b=>get(add(a,b)))),grad=eye.map((_,j)=>get(z,[j])),g1=eye.map(e=>eye.map((_,j)=>get(e,[j]))),H=eye.map((_,i)=>eye.map((_,j)=>get(z,[Math.min(i,j),Math.max(i,j)])));
  const offset=sub(queries[qi],center),rel=add(sub(c.fluid.slice(0,d),c.skeleton.slice(0,d)),mv(A,offset));
  const compression=dot(rel,grad)-S.reduce((sum,row,i)=>sum+row.reduce((s,v,j)=>s+v*g1[j][i],0),0),xi=dot(offset,t)/c.width+.5;
  const coefficients=monomials(d,c.degree).map(alpha=>{
   const k=alpha.reduce((a,b)=>a+b,0);let value=factorial(c.degree)/factorial(c.degree-k)*(1+.25*xi)**(c.degree-k);
   alpha.forEach((a,j)=>{value*= (.25*t[j]/c.width)**a/factorial(a);});return {alpha,value};
  });
  const weighted=beta=>coefficients.reduce((sum,{alpha,value})=>sum+value*get(add(alpha,beta)),0),wm0=weighted(z),wm1=eye.map(weighted),wm2=eye.map(a=>eye.map(b=>weighted(add(a,b))));
  const v=add(mul(rel,wm0),mv(A,wm1)),drag=mul(mv(tensor,v),-1),integrated=matrixScale(tensor,wm0),mobility=inverse(matrixAdd(eye,matrixScale(integrated,c.dt)));
  let diss=wm0*dot(rel,mv(tensor,rel))+2*dot(rel,mv(tensor,mv(A,wm1)));
  const ATA=mm(mm(transpose(A),tensor),A);for(let i=0;i<d;i++)for(let j=0;j<d;j++)diss+=ATA[i][j]*wm2[j][i];
  const row={occupancy:m0,compression,gradient:grad,first:m1,second:m2,hessian:H,laplacian:H.reduce((s,r,i)=>s+r[i],0),resistance:integrated,
   skeleton_product:mv(tensor,add(mul(c.skeleton.slice(0,d),wm0),mv(S,add(mul(offset,wm0),wm1)))),drag,reaction:mul(drag,-1),dissipation:diss,mobility,normal_relative:dot(rel,normal),normal_response:dot(mv(mobility,rel),normal)};
  for(const [k,v]of Object.entries(row))(out[k]??=[]).push(v);
 }return out;
}
const deepMap=(v,fn)=>Array.isArray(v)?v.map(x=>deepMap(x,fn)):fn(v);
const deepAdd=(a,b)=>Array.isArray(a)?a.map((v,i)=>deepAdd(v,b[i])):a+b;
export function assembleWall(c,queries,requests,regional,rotation,profiles){
 const d=c.dimension,terms={solid:contractWall(c,queries,requests,regional.solid,rotation),porous:contractWall(c,queries,requests,regional.porous,rotation)};
 for(const k of ['resistance','skeleton_product','drag','reaction','dissipation'])terms.solid[k]=deepMap(terms.solid[k],()=>0);
 terms.solid.mobility=queries.map(()=>identity(d));terms.solid.normal_response=[...terms.solid.normal_relative];
 terms.total={};for(const k of ['occupancy','compression','gradient','first','second','hessian','laplacian','resistance','skeleton_product','drag','reaction','dissipation'])terms.total[k]=deepAdd(terms.solid[k],terms.porous[k]);
 for(const k of ['mobility','normal_relative','normal_response'])terms.total[k]=terms.porous[k];
 terms.free={occupancy:terms.total.occupancy.map(v=>1-v)};
 for(const k of ['gradient','first','hessian','laplacian'])terms.free[k]=deepMap(terms.total[k],v=>-v);
 const p=profiles.dimensions[d][c.kernel],index=p.profiles.findIndex(([degree,order])=>degree===2&&order===0),fullSecond=p.constants[index]*(d===2?2*Math.PI:4*Math.PI)/d*c.h*c.h;
 terms.free.second=terms.total.second.map(M=>M.map((r,i)=>r.map((v,j)=>+(i===j)*fullSecond-v)));
 if(c.geometry!=='wall')for(const r of ['solid','porous','total']){delete terms[r].normal_relative;delete terms[r].normal_response;}
 return terms;
}
async function integrateRule(resource,plan,elements,regions,q,h,kernel,nodes,exhaustive,progress){
 const columns=plan.requests.length,regional={solid:q.map(()=>Array(columns).fill(0)),porous:q.map(()=>Array(columns).fill(0))},selected=q.map(x=>selectedElements(elements,x,h,exhaustive));
 let peak=0,totalSamples=0,prepare_ms=0,action_readback_ms=0,dispatches=0,parts=[],ranges=[],owners=[],sampleCount=0;
 const flush=async()=>{
  if(!owners.length)return;const started=performance.now(),samples=new Float32Array(sampleCount*8);let offset=0;for(const p of parts){samples.set(p,offset);offset+=p.length;}
  peak=Math.max(peak,Math.max(4,samples.byteLength)+2*owners.length*columns*4+plan.table.byteLength+plan.metadata.byteLength+ranges.length*4+32);totalSamples+=sampleCount;prepare_ms+=performance.now()-started;
  const before=performance.now(),values=await dispatchFluxBridge(resource,plan,samples,new Uint32Array(ranges),h);action_readback_ms+=performance.now()-before;dispatches++;
  owners.forEach(([qi,region],row)=>{const output=regional[region][qi];for(let j=0;j<columns;j++)output[j]+=values[row*columns+j];});
  parts=[];ranges=[];owners=[];sampleCount=0;
 };
 for(let qi=0;qi<q.length;qi++){
  for(const region of ['solid','porous']){
   const ids=selected[qi].filter(ei=>regions[ei]===region);if(!ids.length)continue;
   const beforeMerge=performance.now(),surface=mergeVolumeFacets(elements,ids,plan.dimension);prepare_ms+=performance.now()-beforeMerge;
   // Each bounded face group contributes to the same regional integral. Groups
   // need not be closed individually; all regular-flux tails are retained.
   const faceChunk=plan.dimension===3?4:16;
   for(let first=0;first<surface.faces.length;first+=faceChunk){
    const before=performance.now(),samples=prepareSamples({faces:surface.faces.slice(first,first+faceChunk)},q[qi],h,kernel,nodes);prepare_ms+=performance.now()-before;
    if(sampleCount+samples.length/8>250_000)await flush();
    if(samples.byteLength>8*1024*1024)throw Error('One face group exceeds the 8 MiB browser sample budget; reduce geometry or support disparity');
    ranges.push(sampleCount,sampleCount+samples.length/8,qi,0);parts.push(samples);owners.push([qi,region]);sampleCount+=samples.length/8;
   }
  }
  if(qi%4===3)await flush();
  if(qi%4===0)progress(`Order ${nodes}: preparing ${qi+1}/${q.length} probes…`);
 }
 await flush();return {regional,selected,timing:{prepare_ms,action_readback_ms,samples:totalSamples,dispatches,owned_gpu_peak_bytes:peak,method:plan.dimension===2?'browser_conditioned_boundary':'browser_tet_quadrature',nodes}};
}
export async function computeBrowserWall(payload,progress=()=>{},dependencies={}){
 const start=performance.now(),c=validateWall(payload),d=c.dimension,geo=generateWall(c),requests=wallRequests(d,c.degree),plot_axes=d===2?[0,1]:{xy:[0,1],xz:[0,2],yz:[1,2]}[c.slice_plane];
 const vertices=geo.elements.flat(),axes=plot_axes.map(j=>{const lo=Math.min(...vertices.map(p=>p[j]))-c.h,hi=Math.max(...vertices.map(p=>p[j]))+c.h;return Array.from({length:c.resolution},(_,i)=>lo+(hi-lo)*i/(c.resolution-1));});
 const queries=payload.queries??axes[1].flatMap(y=>axes[0].map(x=>{const q=Array(d).fill(c.slice);q[plot_axes[0]]=x;q[plot_axes[1]]=y;return q;}));
 if(!Array.isArray(queries)||!queries.length||queries.length>1089||queries.some(q=>!Array.isArray(q)||q.length!==d||q.some(v=>!Number.isFinite(v))))throw Error('Invalid query array');
 progress('Loading WebGPU numerical fp32…');
 const resource=dependencies.resource??await fluxBridgeGPU();
 const profiles=dependencies.profiles??await fetch(new URL('./fluxbridge-profiles.json',import.meta.url)).then(r=>{if(!r.ok)throw Error('Could not load FluxBridge profiles');return r.json();});
 const plan=compileProfiles(profiles,d,c.kernel,requests),static_prepare_ms=performance.now()-start;
 let nodes=16,low=await integrateRule(resource,plan,geo.elements,geo.regions,queries,c.h,c.kernel,nodes,c.exhaustive,progress),difference=Infinity,high;
 const allTiming=[low.timing];
 while(nodes<64){
  high=await integrateRule(resource,plan,geo.elements,geo.regions,queries,c.h,c.kernel,nodes+8,c.exhaustive,progress);allTiming.push(high.timing);difference=0;
  for(const r of ['solid','porous'])for(let i=0;i<queries.length;i++)for(let j=0;j<requests.length;j++){
   const scale=c.h**(requests[j].alpha.reduce((a,b)=>a+b,0)-requests[j].derivative.length),a=low.regional[r][i][j]/scale,b=high.regional[r][i][j]/scale;difference=Math.max(difference,Math.abs(a-b)/(1+Math.abs(b)));
  }
  if(difference<=1e-5)break;nodes+=8;low=high;
 }
 if(difference>1e-5)throw Error(`Browser quadrature did not converge at order 64 (${difference.toExponential(2)}); change geometry or use the native reference`);
 if(!c.porous_enabled)for(const row of low.regional.porous)row.fill(0);
 const terms=assembleWall(c,queries,requests,low.regional,geo.rotation,profiles);
 if(Object.values(terms).some(fields=>Object.values(fields).some(a=>a.flat(3).some(v=>!Number.isFinite(v)))))throw Error('Nonfinite browser material contraction');
 return {schema_version:1,kind:'fluxbridge-wall',dimension:d,config:c,axes,plot_axes,queries,shape:payload.queries?[queries.length]:[c.resolution,c.resolution],elements:geo.elements,regions:geo.regions,selected:low.selected,terms,requests,moments:low.regional,timing:allTiming,static_prepare_ms,elapsed_ms:performance.now()-start,
  contract:{backend:d===2?'WebGPU conditioned polygon quadrature':'WebGPU tetrahedral angular quadrature',precision:'float32',device:'WebGPU',adapter:resource.adapter??null,preparation:'JavaScript Number',search:'host whole-element AABB',geometry_precision:'JavaScript Number (float64)',contraction_precision:'JavaScript Number (float64)',measure:d===2?'area':'volume',kernel_dimension:d,support_radius:c.h,
   quadrature:{nodes,comparison_nodes:nodes+8,mixed_difference:difference,budget:1e-5,scope:'normalized regional requested moments; two-order agreement is not an error certificate'},
   occupancy:'geometric bulk-region measure, not solid fraction or porosity; free complement excludes represented porous bulk',compression:'supplied boundary continuity contribution, not a pressure residual',permeation:'supplied-state local resistance response; no pressure solve',fibre_flow:'transverse 2D cylinder sections; longitudinal response requires axial formulation or 3D',reference:'Böttcher et al. (2025), Implicit Incompressible Porous Flow using SPH, arXiv:2504.07739v2',
   units:{occupancy:'1',first:'length',second:'length^2',gradient:'1/length',hessian:'1/length^2',laplacian:'1/length^2',compression:'1/time',resistance:'1/time',drag:'length/time^2 per unit fluid mass',reaction:'length/time^2 per unit fluid mass',dissipation:'length^2/time^3 per unit fluid mass',mobility:'1',skeleton_product:'length/time^2',normal_relative:'length/time',normal_response:'length/time'}}};
}
export async function compareBrowserWall(payload,progress=()=>{},dependencies={}){
 const base=await computeBrowserWall(payload,progress,dependencies),c=base.config,rows=[];
 const variants=[['Whole-element support localization',{...c,exhaustive:!c.exhaustive}]];
 if(c.geometry==='wall'){
  if(c.subdivisions<12)variants.push(['Fixed-geometry mesh refinement',{...c,subdivisions:Math.min(12,c.subdivisions*2)}]);
 }else if(c.segments<64)variants.push(['Chord geometry refinement (geometry changes)',{...c,segments:Math.min(64,c.segments*2)}]);
 for(const [label,config]of variants){
  progress(label+'…');const other=await computeBrowserWall({...config,queries:base.queries},progress,dependencies),metrics={};
  for(const [region,fields]of Object.entries(base.terms)){
   metrics[region]={};for(const [key,values]of Object.entries(fields)){
    const a=values.flat(3),b=other.terms[region][key].flat(3);let absolute=0,mixed=0;
    a.forEach((v,i)=>{absolute=Math.max(absolute,Math.abs(v-b[i]));mixed=Math.max(mixed,Math.abs(v-b[i])/(1+Math.abs(b[i])));});
    metrics[region][key]={max_absolute:absolute,max_mixed:mixed};
   }
  }
  rows.push({label,status:'measured',elapsed_ms:other.elapsed_ms,terms:metrics});
 }
 base.comparisons={rows,convention:'Mixed difference = |a-b|/(1+|b|).',scope:'Same displayed probes and supplied fields, WebGPU fp32 only. Order comparison is recorded separately in the execution contract; fp64 is available only through the explicit native provider.'};return base;
}
