import {selectComputeAdapter,adapterSummary} from './gpu-adapter.js';
// Follows the existing browser dispatch lifecycle; no DOM/native dependencies.
let ready;
export async function createFluxBridgeGPU(device,code){
 const module=device.createShaderModule({code,label:'FluxBridge numerical fp32'});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');
 if(errors.length)throw Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join('\n'));
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compute_fluxbridge'}});
 return {device,pipeline,module};
}
async function resources(){
 if(!navigator.gpu)throw Error('WebGPU is unavailable. Select the native provider or open a saved snapshot.');
 const selection=await selectComputeAdapter(navigator.gpu),{adapter}=selection;
 if(adapter.limits.maxStorageBuffersPerShaderStage<5)throw Error('This WebGPU adapter exposes fewer than the five compute storage buffers required by FluxBridge. See Browser GPU diagnostics.');
 const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:5}});
 try{
  const response=await fetch(new URL('./fluxbridge-compute.wgsl?v=20261028-webgpu1',import.meta.url));if(!response.ok)throw Error('Could not load FluxBridge shader');
  const resource=await createFluxBridgeGPU(device,await response.text());device.lost.then(()=>{ready=undefined;});
  return {...resource,adapter:{...adapterSummary(adapter),feature_level:selection.featureLevel,request_options:selection.options,discovery_attempts:selection.attempts}};
 }catch(e){device.destroy();throw e;}
}
export async function fluxBridgeGPU(){try{return await(ready??=resources());}catch(e){ready=undefined;throw e;}}
export function fluxBridgeDispatchLayout(plan,ranges,limits,{strategy='auto',chunkSamples=4096}={}){
 if(!['auto','generic','shared','chunked'].includes(strategy))throw Error('Unknown FluxBridge dispatch strategy');
 if(!Number.isInteger(chunkSamples)||chunkSamples<64||chunkSamples%64)throw Error('FluxBridge chunks require a multiple of 64 samples');
 const columns=plan.requests.length,rows=ranges.length/4,groups=Math.ceil(columns/64),budget=Math.min(limits.maxStorageBufferBindingSize,limits.maxBufferSize,64*1024*1024);
 const shared=strategy!=='generic'&&!!plan.sharedMetadata&&plan.profileCount<=32&&(strategy!=='auto'||columns>=8);
 const metadata=shared?plan.sharedMetadata:plan.metadata;
 const plain={shared,chunked:false,metadata,ranges,scratchBytes:0,chunkCount:0,workgroups:shared?groups*rows:Math.ceil(columns*rows/64)};
 let longest=0;for(let i=0;i<ranges.length;i+=4)longest=Math.max(longest,ranges[i+1]-ranges[i]);
 if(!shared||strategy==='shared'||(strategy==='auto'&&longest<4096))return plain;
 const maxChunks=Math.min(Math.floor((budget-metadata.byteLength)/(columns*80)),Math.floor(limits.maxComputeWorkgroupsPerDimension/groups));
 let chunkCount=0;
 for(;;){
  chunkCount=0;for(let i=0;i<ranges.length;i+=4)chunkCount+=Math.ceil((ranges[i+1]-ranges[i])/chunkSamples);
  if(chunkCount<=maxChunks)break;
  if(chunkSamples>=longest)return plain;
  chunkSamples*=2;
 }
 const packed=new Uint32Array((rows+chunkCount)*4);packed.set(ranges);let chunk=0;
 for(let row=0;row<rows;row++){
  packed[row*4+2]=chunk;
  for(let start=ranges[row*4];start<ranges[row*4+1];start+=chunkSamples){packed.set([start,Math.min(start+chunkSamples,ranges[row*4+1]),row,0],(rows+chunk)*4);chunk++;}
  packed[row*4+3]=chunk;
 }
 return {shared:true,chunked:true,metadata,ranges:packed,scratchBytes:chunkCount*columns*80,chunkCount,workgroups:chunkCount*groups};
}
async function cachedPipeline(resource,key,entryPoint){
 try{return await(resource[key]??=resource.device.createComputePipelineAsync({layout:'auto',compute:{module:resource.module,entryPoint}}));}
 catch(error){resource[key]=undefined;throw error;}
}
export async function dispatchFluxBridge(resource,plan,samples,ranges,h,options={}){
 const {device}=resource,columns=plan.requests.length,rows=ranges.length/4,count=columns*rows;
 if(!rows)return new Float32Array();
 const layout=fluxBridgeDispatchLayout(plan,ranges,device.limits,{strategy:resource.strategy??'auto',...options});
 const {shared,chunked,metadata,scratchBytes,chunkCount,workgroups}=layout;ranges=layout.ranges;
 const limits=device.limits;
 for(const size of [samples.byteLength,plan.table.byteLength,metadata.byteLength+scratchBytes,ranges.byteLength,count*4])if(size>Math.min(limits.maxStorageBufferBindingSize,limits.maxBufferSize,64*1024*1024))throw Error('FluxBridge buffer budget exceeded; reduce grid or use smaller batches');
 if(Math.max(workgroups,Math.ceil(count/64))>limits.maxComputeWorkgroupsPerDimension)throw Error('FluxBridge dispatch limit exceeded');
 const pipeline=chunked?await cachedPipeline(resource,'chunkPipeline','compute_fluxbridge_chunked'):shared?await cachedPipeline(resource,'sharedPipeline','compute_fluxbridge_shared'):resource.pipeline;
 const reducer=chunked?await cachedPipeline(resource,'reducePipeline','reduce_fluxbridge_chunks'):null;
 const owned=[];
 const buffer=(data,usage,extra=0)=>{const b=device.createBuffer({size:Math.max(4,data.byteLength+extra),usage:usage|GPUBufferUsage.COPY_DST});owned.push(b);if(data.byteLength)device.queue.writeBuffer(b,0,data);return b;};
 device.pushErrorScope('validation');
 try{
  const uniform=new ArrayBuffer(48);new Uint32Array(uniform).set([columns,rows,plan.dimension,plan.profileCount??0]);new Float32Array(uniform)[4]=h;new Uint32Array(uniform).set([chunkCount,metadata.byteLength/16,0,0],8);
  const inputs=[buffer(new Uint8Array(uniform),GPUBufferUsage.UNIFORM),buffer(samples,GPUBufferUsage.STORAGE),buffer(plan.table,GPUBufferUsage.STORAGE),buffer(metadata,GPUBufferUsage.STORAGE,scratchBytes),buffer(ranges,GPUBufferUsage.STORAGE)];
  const output=buffer(new Uint8Array(count*4),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=device.createBuffer({size:count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});owned.push(read);
  const entries=[...inputs,output].map((b,binding)=>({binding,resource:{buffer:b}})),encoder=device.createCommandEncoder();
  const compute=(pipeline,groups,bindings=entries)=>{if(!groups)return;const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings}),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(groups);pass.end();};
  compute(pipeline,workgroups);
  // The reducer only reads the header/partials and ranges. Separate passes
  // make all storage writes visible without atomics or float partial sums.
  if(reducer)compute(reducer,Math.ceil(count/64),entries.filter(e=>[0,3,4,5].includes(e.binding)));
  encoder.copyBufferToBuffer(output,0,read,0,count*4);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();
  if(!values.every(Number.isFinite))throw Error('Nonfinite fp32 FluxBridge result; reduce physical scaling');
  resource.lastDispatch={strategy:chunked?'chunked':shared?'shared':'generic',chunks:chunkCount,compute_passes:(workgroups?1:0)+(reducer?1:0),owned_gpu_bytes:owned.reduce((sum,b)=>sum+b.size,0)};
  return values;
 }finally{for(const b of owned)b.destroy();const error=await device.popErrorScope();if(error)throw Error(error.message);}
}
