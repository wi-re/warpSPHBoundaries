// Optional second WGSL implementation: npm install --prefix .tmp/dawn-check
// --no-save --ignore-scripts webgpu. Uses Dawn/Tint's null adapter for compiler
// validation; wgpu-native executes the numerical and rendering checks separately.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const binding=process.env.DAWN_NODE_MODULE?pathToFileURL(process.env.DAWN_NODE_MODULE):new URL('../../../.tmp/dawn-check/node_modules/webgpu/index.js',import.meta.url);
const available=fs.existsSync(binding);
test('Dawn/Tint reproduces the rejected NaN constant and compiles all production shaders',{skip:!available},async()=>{
 const {create,globals}=await import(binding.href);Object.assign(globalThis,globals);
 const gpu=create(['backend=null']),adapter=await gpu.requestAdapter();assert.ok(adapter);const device=await adapter.requestDevice();
 try{
  device.pushErrorScope('validation');
  const broken=device.createShaderModule({code:'fn nan()->f32{return bitcast<f32>(0x7fc00000u);}\n@compute @workgroup_size(1) fn main(){}'});
  const originalErrors=(await broken.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.ok(originalErrors.length);assert.match(originalErrors[0].message,/nan.*cannot be represented/i);await device.popErrorScope();
  for(const [name,entryPoint]of [['fluxbridge-compute.wgsl','compute_fluxbridge'],['analytic3d-compute.wgsl','compute_analytic3d'],['terms2d-compute.wgsl','compute_terms2d'],['terms-compute.wgsl','compute_terms'],['volume.wgsl',null]]){
   device.pushErrorScope('validation');
   const code=fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/'+name,import.meta.url),'utf8'),module=device.createShaderModule({label:name,code});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[],name);
   if(entryPoint)await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
   else await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertexMain'},fragment:{module,entryPoint:'fragmentMain',targets:[{format:'rgba8unorm'}]},primitive:{topology:'triangle-list'}});
   assert.equal(await device.popErrorScope(),null,name);
  }
 }finally{device.destroy();}
});
