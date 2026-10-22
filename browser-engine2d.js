import {compile2d,modelError2d,resultRow2d,decodeSamples2d} from './compute2d-plan.js?v=20261022-6';
import {MODELS2D} from './browser-compute2d.js?v=20261022-6';
let worker,job=0,resourcePromise;const pending=new Map();
function cpu(config,progress){
 if(!worker){worker=new Worker(new URL('./compute-worker2d.js?v=20261022-6',import.meta.url),{type:'module'});worker.onmessage=event=>{const {id,result,error,progress:note}=event.data,item=pending.get(id);if(!item)return;if(note){item.progress(note);return;}pending.delete(id);if(error)item.reject(new Error(error));else{result.compute_engine='JavaScript float64';item.resolve(result);}};worker.onerror=event=>{for(const item of pending.values())item.reject(new Error(event.message));pending.clear();worker.terminate();worker=null;};}
 return new Promise((resolve,reject)=>{const id=++job;pending.set(id,{resolve,reject,progress});worker.postMessage({id,config});});
}
async function resources(){
 if(!navigator.gpu)throw new Error('WebGPU is not exposed by this browser');
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw new Error('No WebGPU compute adapter');
 const device=await adapter.requestDevice();
 try{
  const response=await fetch(new URL('./terms2d-compute.wgsl?v=20261022-6',import.meta.url));if(!response.ok)throw new Error('Could not load the 2D compute shader');
  const module=device.createShaderModule({label:'2D boundary integrals · fp32',code:await response.text()}),info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');
  if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join('\n'));
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compute_terms2d'}});
  device.lost.then(()=>{resourcePromise=undefined;});return {device,pipeline};
 }catch(error){device.destroy();throw error;}
}
export async function computeGPU2d(config,progress=()=>{}){
 const started=performance.now(),plan=compile2d(config),{c,x,y,geometry,table}=plan;
 progress('Preparing WebGPU fp32 · first use compiles the 2D shader…');
 const {device,pipeline}=await(resourcePromise??=resources()),ready=performance.now();
 const uniform=device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),lookup=device.createBuffer({size:table.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:c.resolution**2*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:c.resolution**2*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const results=[];let distance;
 try{
  device.queue.writeBuffer(lookup,0,table);
  const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:lookup}},{binding:2,resource:{buffer:output}}]});
  for(const model of c.models){
   const error=modelError2d(model,c,x,y);if(error){results.push({model,error});continue;}
   progress(`WebGPU fp32 · ${MODELS2D[model][0]}`);const tick=performance.now(),raw=plan.uniform.slice();new Uint32Array(raw.buffer)[3]=['mesh','exact','rectangle','sdf','curvature','disk'].indexOf(model);device.queue.writeBuffer(uniform,0,raw);
   device.pushErrorScope('validation');let validation;
   try{
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(c.resolution**2/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,c.resolution**2*16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const decoded=decodeSamples2d(new Float32Array(read.getMappedRange())),values=decoded.values;distance??=decoded.distance;read.unmap();
    results.push(resultRow2d(model,values,performance.now()-tick,c,'WebGPU fp32'));
   }finally{validation=await device.popErrorScope();}if(validation)throw new Error(validation.message);
  }
  // Incompatible model choices can all be rejected before dispatch.
  if(!distance){const subset=await cpu({...c,models:['sdf']},progress);distance=subset.distance;}
  return {schema_version:1,dimension:2,config:c,x,y,...geometry,distance,results,compute_engine:'WebGPU fp32',gpu_initialization_ms:ready-started,milliseconds:performance.now()-started};
 }finally{uniform.destroy();lookup.destroy();output.destroy();read.destroy();}
}
export async function computeBrowser2d(config,progress=()=>{},preference='auto'){
 if(preference==='worker')return cpu(config,progress);
 try{return await computeGPU2d(config,progress);}catch(error){const failed=resourcePromise;resourcePromise=undefined;failed?.then(r=>r.device.destroy(),()=>{});if(preference==='gpu')throw error;progress(`WebGPU unavailable: ${error.message}. Using float64 worker.`);const result=await cpu(config,progress);result.compute_fallback=error.message;return result;}
}
export function disposeCompute2d(){worker?.terminate();worker=null;for(const item of pending.values())item.reject(new Error('Compute worker closed'));pending.clear();const old=resourcePromise;resourcePromise=undefined;old?.then(r=>r.device.destroy(),()=>{});}
