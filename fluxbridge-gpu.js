import {selectComputeAdapter,adapterSummary} from './gpu-adapter.js';
// Follows the existing browser dispatch lifecycle; no DOM/native dependencies.
let ready;
export async function createFluxBridgeGPU(device,code){
 const module=device.createShaderModule({code,label:'FluxBridge numerical fp32'});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');
 if(errors.length)throw Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join('\n'));
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compute_fluxbridge'}});
 return {device,pipeline};
}
async function resources(){
 if(!navigator.gpu)throw Error('WebGPU is unavailable. Select the native provider or open a saved snapshot.');
 const selection=await selectComputeAdapter(navigator.gpu),{adapter}=selection;
 if(adapter.limits.maxStorageBuffersPerShaderStage<5)throw Error('This WebGPU adapter exposes fewer than the five compute storage buffers required by FluxBridge. See Browser GPU diagnostics.');
 const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:5}});
 try{
  const response=await fetch(new URL('./fluxbridge-compute.wgsl',import.meta.url));if(!response.ok)throw Error('Could not load FluxBridge shader');
  const resource=await createFluxBridgeGPU(device,await response.text());device.lost.then(()=>{ready=undefined;});
  return {...resource,adapter:{...adapterSummary(adapter),feature_level:selection.featureLevel,request_options:selection.options,discovery_attempts:selection.attempts}};
 }catch(e){device.destroy();throw e;}
}
export async function fluxBridgeGPU(){try{return await(ready??=resources());}catch(e){ready=undefined;throw e;}}
export async function dispatchFluxBridge(resource,plan,samples,ranges,h){
 const {device,pipeline}=resource,columns=plan.requests.length,rows=ranges.length/4,count=columns*rows;
 if(!rows)return new Float32Array();
 const limits=device.limits;
 for(const size of [samples.byteLength,plan.table.byteLength,plan.metadata.byteLength,ranges.byteLength,count*4])if(size>Math.min(limits.maxStorageBufferBindingSize,limits.maxBufferSize,64*1024*1024))throw Error('FluxBridge buffer budget exceeded; reduce grid or use smaller batches');
 if(Math.ceil(count/64)>limits.maxComputeWorkgroupsPerDimension)throw Error('FluxBridge dispatch limit exceeded');
 const owned=[];
 const buffer=(data,usage)=>{const b=device.createBuffer({size:Math.max(4,data.byteLength),usage:usage|GPUBufferUsage.COPY_DST});owned.push(b);if(data.byteLength)device.queue.writeBuffer(b,0,data);return b;};
 device.pushErrorScope('validation');
 try{
  const uniform=new ArrayBuffer(32);new Uint32Array(uniform).set([columns,rows,plan.dimension,0]);new Float32Array(uniform)[4]=h;
  const inputs=[buffer(new Uint8Array(uniform),GPUBufferUsage.UNIFORM),buffer(samples,GPUBufferUsage.STORAGE),buffer(plan.table,GPUBufferUsage.STORAGE),buffer(plan.metadata,GPUBufferUsage.STORAGE),buffer(ranges,GPUBufferUsage.STORAGE)];
  const output=buffer(new Uint8Array(count*4),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=device.createBuffer({size:count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});owned.push(read);
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...inputs,output].map((b,binding)=>({binding,resource:{buffer:b}}))});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,count*4);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();
  if(!values.every(Number.isFinite))throw Error('Nonfinite fp32 FluxBridge result; reduce physical scaling');return values;
 }finally{for(const b of owned)b.destroy();const error=await device.popErrorScope();if(error)throw Error(error.message);}
}
