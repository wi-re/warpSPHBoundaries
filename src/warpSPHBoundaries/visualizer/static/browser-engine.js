import {compileSeries} from './series-plan.js?v=20261022-7';
import {compileAnalytic3d} from './analytic3d-plan.js?v=20261022-7';
import {dispatchAnalytic,disposeAnalyticGPU} from './analytic-gpu.js?v=20261028-webgpu1';
import {FIELD_IDS,SHAPES,gauss} from './term-engine.js?v=20261022-7';
import {BROWSER_MODELS,validate,grid,samples,modelError,resultRow,packageResult} from './browser-compute.js?v=20261022-7';
let resourcesPromise;
async function resources(){
 if(!navigator.gpu)throw new Error('WebGPU is not available');
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU compute adapter');
 const device=await adapter.requestDevice(),response=await fetch(new URL('./terms-compute.wgsl?v=20261022-7',import.meta.url));if(!response.ok)throw new Error('Could not load compute shader');
 const module=device.createShaderModule({code:await response.text(),label:'Boundary moment evaluation'}),info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');
 if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join('\n'));
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compute_terms'}});return {device,pipeline};
}
let worker,job=0;
const pending=new Map();
function cpu(config,progress){
 if(!worker){worker=new Worker(new URL('./compute-worker.js?v=20261022-7',import.meta.url),{type:'module'});worker.onmessage=event=>{const {id,result,error,progress:note}=event.data,item=pending.get(id);if(!item)return;if(note){item.progress(note);return;}pending.delete(id);if(error)item.reject(new Error(error));else item.resolve(result);};worker.onerror=event=>{for(const item of pending.values())item.reject(new Error(event.message));pending.clear();worker.terminate();worker=null;};}
 return new Promise((resolve,reject)=>{const id=++job;pending.set(id,{resolve,reject,progress});worker.postMessage({id,config});});
}
async function gpu(config,progress){
 const c=validate(config),{n,half,axis}=grid(c),{device,pipeline}=await(resourcesPromise??=resources()),quad=c.models.includes('quadrature')?samples(c):new Float64Array(),table=new Float32Array(44+quad.length),k=SHAPES[c.kernel];table.set(k.coeffs);table.set(gauss(16).flat(),12);table.set(quad,44);
 const uniform=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),lookup=device.createBuffer({size:table.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:n**3*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:n**3*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),start=performance.now(),results=[];let distance;
 device.queue.writeBuffer(lookup,0,table);
 const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:lookup}},{binding:2,resource:{buffer:output}}]});
 try{
  for(const model of c.models){
   const error=modelError(model,c);if(error){results.push({model,label:BROWSER_MODELS[model][0],error});continue;}
   if(model==='analytic'){
    const tick=performance.now(),sample=await dispatchAnalytic(compileAnalytic3d(c),progress);results.push(resultRow(model,sample.values,performance.now()-tick,'WebGPU compensated fp32',c));distance??=sample.distance;continue;
   }
   if(model==='ball_series'){const tick=performance.now(),sample=await dispatchAnalytic(compileSeries(c,3),progress);results.push(resultRow(model,sample.values,performance.now()-tick,'WebGPU compensated fp32',c));distance??=sample.distance;continue;}
   progress(`WebGPU compute · ${BROWSER_MODELS[model][0]}`);
   const tick=performance.now(),buffer=new ArrayBuffer(64),u=new Uint32Array(buffer),f=new Float32Array(buffer);
   u.set([n,FIELD_IDS[c.field],['sphere','box','tetrahedron'].indexOf(c.geometry),{sphere:0,sdf:1,curvature:2,quadrature:3}[model]]);
   f.set([c.h,c.radius,c.width,c.height,c.depth,half,['cubic','w2','w4','w6'].indexOf(c.kernel),model==='curvature'?(c.curvature_order??2):quad.length/8,k.n,k.P[1]??0,k.P[2]??0,k.P[3]??0],4);
   device.queue.writeBuffer(uniform,0,buffer);device.pushErrorScope('validation');
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(n**3/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,n**3*8);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const pairs=new Float32Array(read.getMappedRange()),values=new Float32Array(n**3);distance??=new Float32Array(n**3);for(let i=0;i<values.length;i++){values[i]=pairs[i*2];distance[i]=pairs[i*2+1];}read.unmap();const validation=await device.popErrorScope();if(validation)throw new Error(validation.message);
   results.push(resultRow(model,values,performance.now()-tick,'WebGPU compute f32',c));
  }
  if(!distance){const subset=await cpu({...c,models:['sdf']},progress);const raw=atob(subset.distance),view=new DataView(Uint8Array.from(raw,ch=>ch.charCodeAt(0)).buffer);distance=Float32Array.from({length:n**3},(_,i)=>view.getFloat32(i*4,true));}
  return packageResult(c,axis,distance,results,performance.now()-start);
 }finally{uniform.destroy();lookup.destroy();output.destroy();read.destroy();}
}
export async function computeBrowser(config,progress=()=>{},preference='auto'){
 if(preference==='worker')return cpu(config,progress);
 try{return await gpu(config,progress);}catch(error){const failed=resourcesPromise;resourcesPromise=undefined;failed?.then(r=>r.device.destroy(),()=>{});if(preference==='gpu')throw error;progress(`Compute fallback: ${error.message}. Using JavaScript worker.`);const result=await cpu(config,progress);result.compute_fallback=error.message;return result;}
}
export function disposeCompute(){const old=resourcesPromise;resourcesPromise=undefined;old?.then(r=>r.device.destroy(),()=>{});disposeAnalyticGPU();worker?.terminate();worker=null;for(const p of pending.values())p.reject(new Error('Compute worker closed'));pending.clear();}
