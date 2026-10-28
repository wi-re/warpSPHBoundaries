// Matched actual WebGPU and complete wall evaluations, no CPU/GPU emulation.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {compileProfiles,prepareSamples,mergeVolumeFacets,selectedElements} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-plan.js';
import {computeBrowserWall,generateWall,validateWall,wallRequests} from '../../../src/warpSPHBoundaries/visualizer/static/wall-browser.js';
import {createFluxBridgeGPU,dispatchFluxBridge} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-gpu.js';
import {selectComputeAdapter,adapterSummary} from '../../../src/warpSPHBoundaries/visualizer/static/gpu-adapter.js';
const payload=JSON.parse(fs.readFileSync(0,'utf8')),baseline=await import(pathToFileURL(payload.baselineWall)),oldPlan=await import(pathToFileURL(payload.baselinePlan));
const binding=process.env.FLUXBRIDGE_WEBGPU_MODULE??new URL('../../../.tmp/dawn-check/node_modules/webgpu/index.js',import.meta.url).pathname;
const {create,globals}=await import(pathToFileURL(binding));Object.assign(globalThis,globals);
const gpu=create([`backend=${process.env.FLUXBRIDGE_WEBGPU_BACKEND??'vulkan'}`]),{adapter}=await selectComputeAdapter(gpu),device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:5}});
const profiles=JSON.parse(fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-profiles.json',import.meta.url)));
const median=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
const mixed=(a,b)=>Array.isArray(a)?Math.max(0,...a.map((v,i)=>mixed(v,b[i]))):Math.abs(a-b)/(1+Math.abs(a));
const sum=(rows,key)=>rows.reduce((s,r)=>s+r[key],0);
try{
 const resource=await createFluxBridgeGPU(device,fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-compute.wgsl',import.meta.url),'utf8'));
 const generic={...resource,strategy:'generic'},core=[],walls=[];
 for(const dimension of [2,3]){
  const c=validateWall({dimension,degree:3,subdivisions:1,angle:17}),q=[.1,.07,...(dimension===3?[.13]:[])],geo=generateWall(c);
  const ids=selectedElements(geo.elements,q,c.h).filter(i=>geo.regions[i]==='porous'),surface=mergeVolumeFacets(geo.elements,ids,dimension),requests=wallRequests(dimension,c.degree),plan=compileProfiles(profiles,dimension,c.kernel,requests);
  const prep={generic:[],shared:[]};let samples;
  for(let repeat=0;repeat<5;repeat++){
   const t=performance.now(),before=oldPlan.prepareSamples(surface,q,c.h,c.kernel,24);prep.generic.push(performance.now()-t);
   const tick=performance.now();samples=prepareSamples(surface,q,c.h,c.kernel,24);prep.shared.push(performance.now()-tick);
   const a=new Uint32Array(before.buffer),b=new Uint32Array(samples.buffer);
   if(a.length!==b.length||a.some((v,i)=>v!==b[i]))throw Error('Prepared sample bits changed');
  }
  const range=new Uint32Array([0,samples.length/8,0,0]),times={generic:[],shared:[]};let error=0;
  for(let repeat=0;repeat<=payload.repeats;repeat++){
   const values={};for(const strategy of repeat%2?['shared','generic']:['generic','shared']){
    const t=performance.now();values[strategy]=[...await dispatchFluxBridge(resource,plan,samples,range,c.h,{strategy:strategy==='shared'?'auto':strategy})];times[strategy].push(performance.now()-t);
   }
   error=Math.max(error,mixed(values.generic,values.shared));
  }
  if(error>1e-6)throw Error(`Core mixed error ${error}`);
  core.push({dimension,columns:requests.length,profiles:plan.profileCount,samples:samples.length/8,preparation_ms:prep,dispatch_readback_ms:times,warm_speedup:median(times.generic.slice(1))/median(times.shared.slice(1)),worst_mixed:error});
 }
 for(const dimension of [2,3])for(const degree of [0,3]){
  const config={dimension,degree,resolution:dimension===2?9:5,subdivisions:1,angle:17},times={generic:[],shared:[]},costs={generic:[],shared:[]};let error=0,orders;
  for(let repeat=0;repeat<=payload.repeats;repeat++){
   const results={};for(const strategy of repeat%2?['shared','generic']:['generic','shared']){
    const fn=strategy==='generic'?baseline.computeBrowserWall:computeBrowserWall,r=strategy==='generic'?generic:resource,t=performance.now();
    const result=await fn(config,()=>{},{resource:r,profiles});times[strategy].push(performance.now()-t);results[strategy]=result;
    costs[strategy].push({static_prepare_ms:result.static_prepare_ms,prepare_ms:sum(result.timing,'prepare_ms'),dispatch_readback_ms:sum(result.timing,'action_readback_ms'),samples:sum(result.timing,'samples'),dispatches:sum(result.timing,'dispatches')});
   }
   for(const region of Object.keys(results.generic.terms))for(const name of Object.keys(results.generic.terms[region]))error=Math.max(error,mixed(results.generic.terms[region][name],results.shared.terms[region][name]));
   orders={generic:results.generic.contract.quadrature,shared:results.shared.contract.quadrature};
  }
  if(error>2e-5)throw Error(`Wall mixed error ${error}`);
  walls.push({config,elapsed_ms:times,costs,orders,warm_speedup:median(times.generic.slice(1))/median(times.shared.slice(1)),worst_mixed:error});
 }
 console.log(JSON.stringify({adapter:adapterSummary(adapter),repeats:payload.repeats,core,walls}));
}finally{device.destroy();}
