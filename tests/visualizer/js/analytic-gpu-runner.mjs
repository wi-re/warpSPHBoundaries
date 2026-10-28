// Execute production pipeline selection through Dawn, with optional matched timings.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {analyticPipeline} from '../../../src/warpSPHBoundaries/visualizer/static/analytic-gpu.js';
const payload=JSON.parse(fs.readFileSync(0,'utf8'));
const module=process.env.FLUXBRIDGE_WEBGPU_MODULE??new URL('../../../.tmp/dawn-check/node_modules/webgpu/index.js',import.meta.url).pathname;
const {create,globals}=await import(pathToFileURL(module));Object.assign(globalThis,globals);
const gpu=create([`backend=${process.env.FLUXBRIDGE_WEBGPU_BACKEND??'vulkan'}`]);
const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();
const source=fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/analytic3d-compute.wgsl',import.meta.url),'utf8');
const resource={device,source,module:device.createShaderModule({code:source}),pipelines:new Map()};
const results=[];
try{
 for(const row of payload.rows){
  const plan={uniform:new Uint8Array(row.uniform),table:new Float32Array(row.table)};
  const u=new Uint32Array(plan.uniform.buffer),f=new Float32Array(plan.uniform.buffer),count=u[0]**f[10];
  const uniform=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),table=device.createBuffer({size:plan.table.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:count*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   device.queue.writeBuffer(uniform,0,plan.uniform);device.queue.writeBuffer(table,0,plan.table);
   const result={field:row.field};
   for(const variant of payload.benchmark?['generic','rolling']:['rolling']){
    const start=performance.now();
    const pipeline=variant==='generic'?await(resource.generic??=device.createComputePipelineAsync({layout:'auto',compute:{module:resource.module,entryPoint:'compute_analytic3d'}})):await analyticPipeline(resource,plan);
    const compile_ms=performance.now()-start;
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,table,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const times=[];let values;
    for(let i=0;i<(payload.benchmark?(payload.repeats??3)+1:1);i++){
     const tick=performance.now(),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
     pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,count*16);device.queue.submit([encoder.finish()]);
     await read.mapAsync(GPUMapMode.READ);values=new Float32Array(read.getMappedRange().slice(0));read.unmap();times.push(performance.now()-tick);
    }
    result[variant]={values:[...values],compile_ms,cold_dispatch_readback_ms:times[0],warm_dispatch_readback_ms:times.slice(1)};
   }
   results.push(result);
  }finally{uniform.destroy();table.destroy();output.destroy();read.destroy();}
 }
 const {vendor,architecture,device:deviceName,description}=adapter.info;
 console.log(JSON.stringify({adapter:{vendor,architecture,device:deviceName,description},results}));
}finally{device.destroy();}
