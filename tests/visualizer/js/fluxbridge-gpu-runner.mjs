// Executes the production browser shader/dispatch through Dawn WebGPU.
// The adapter description is evidence: software Vulkan != hardware GPU timing.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {compileProfiles,prepareSamples} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-plan.js';
import {selectComputeAdapter} from '../../../src/warpSPHBoundaries/visualizer/static/gpu-adapter.js';
import {createFluxBridgeGPU,dispatchFluxBridge} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-gpu.js';
import {computeBrowserWall,compareBrowserWall,generateWall,validateWall,wallRequests,contractWall} from '../../../src/warpSPHBoundaries/visualizer/static/wall-browser.js';
const payload=JSON.parse(fs.readFileSync(0,'utf8'));
if(payload.host){
 const c=validateWall(payload.host.config),geo=generateWall(c),requests=wallRequests(c.dimension,c.degree);
 console.log(JSON.stringify({...geo,requests,terms:payload.host.moments?contractWall(c,payload.host.queries,requests,payload.host.moments,geo.rotation):null}));
}else{
 const module=process.env.FLUXBRIDGE_WEBGPU_MODULE??new URL('../../../.tmp/dawn-check/node_modules/webgpu/index.js',import.meta.url).pathname;
 const {create,globals}=await import(pathToFileURL(module));Object.assign(globalThis,globals);
 const gpu=create([`backend=${process.env.FLUXBRIDGE_WEBGPU_BACKEND??'vulkan'}`]),selection=await selectComputeAdapter(gpu),{adapter}=selection;const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:5}});
 const profiles=JSON.parse(fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-profiles.json',import.meta.url)));
 try{
  const resource=await createFluxBridgeGPU(device,fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-compute.wgsl',import.meta.url),'utf8'));
  const results=[],walls=[],reductions=[];
  for(const weights of payload.reductions??[]){
   const plan=compileProfiles(profiles,2,'w2',[{alpha:[0,0],derivative:[]}]);plan.table[0]=1;
   const samples=new Float32Array(weights.length*8);weights.forEach((w,i)=>{samples.set([1,0,0,1,w,0,0,0],i*8);});
   reductions.push([...(await dispatchFluxBridge(resource,plan,samples,new Uint32Array([0,weights.length,0,0]),1))][0]);
  }
  for(const c of payload.cases??[]){
   const d=c.dimension??3,vertices=c.vertices.map(p=>p.map(Number)),q=c.query.map(Number),h=Number(c.h),requests=c.requests;
   const plan=compileProfiles(profiles,d,c.kernel,requests),samples=prepareSamples(vertices,q,h,c.kernel,payload.nodes??24),start=performance.now();
   const values=await dispatchFluxBridge(resource,plan,samples,new Uint32Array([0,samples.length/8,0,0]),h);
   results.push({values:[...values],samples:samples.length/8,dispatch_readback_ms:performance.now()-start});
  }
  for(const c of payload.walls??[])walls.push(await (payload.compare?compareBrowserWall:computeBrowserWall)(c,()=>{},{resource,profiles}));
  console.log(JSON.stringify({adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description,feature_level:selection.featureLevel},results,walls,reductions}));
 }finally{device.destroy();}
}
