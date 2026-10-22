import {cameraFrame,clamp} from './volume-math.js?v=20261022-2';
let gpuPromise;
async function gpuResources() {
  if(!navigator.gpu)throw new Error('WebGPU is not exposed by this browser');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw new Error('No WebGPU adapter is available');
  const device=await adapter.requestDevice();
  const shader=await(await fetch(new URL('./volume.wgsl?v=20261022-2',import.meta.url))).text();
  const module=device.createShaderModule({label:'Boundary integral volume ray caster',code:shader});
  const info=await module.getCompilationInfo();
  const errors=info.messages.filter(m=>m.type==='error');
  if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join('\n'));
  const format=navigator.gpu.getPreferredCanvasFormat();
  const pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertexMain'},fragment:{module,entryPoint:'fragmentMain',targets:[{format}]},primitive:{topology:'triangle-list'}});
  return {device,pipeline,format};
}
function half(value) {
  if(!Number.isFinite(value))return 0;
  const sign=value<0?0x8000:0;value=Math.min(Math.abs(value),65504);
  if(value<2**-14)return sign|Math.min(1023,Math.round(value*2**24));
  let exponent=Math.floor(Math.log2(value)),mantissa=Math.round((value/2**exponent-1)*1024);
  if(mantissa===1024){exponent++;mantissa=0;}
  return sign|((exponent+15)<<10)|mantissa;
}
export class VolumeRenderer {
  static async create(canvas,onStatus,onFailure) {
    try{
      gpuPromise??=gpuResources();const resources=await gpuPromise;
      const result=new VolumeRenderer(canvas,resources,onFailure);onStatus('WebGPU · ray casting');return result;
    }catch(error){onStatus(`CPU fallback · unlit, grid-aware ray steps · ${error.message}`);return new CPUVolume(canvas,onFailure);}
  }
  constructor(canvas,{device,pipeline,format},onFailure) {
    this.canvas=canvas;this.device=device;this.pipeline=pipeline;this.format=format;
    this.context=canvas.getContext('webgpu');if(!this.context)throw new Error('WebGPU canvas context is unavailable');
    this.context.configure({device,format,alphaMode:'opaque',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    this.uniform=device.createBuffer({size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'clamp-to-edge',addressModeV:'clamp-to-edge',addressModeW:'clamp-to-edge'});
    this.transfer=device.createTexture({size:[256,1],format:'rgba8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    this.errorListener=event=>{if(!this.disposed)onFailure(`WebGPU validation: ${event.error.message}`);};device.addEventListener('uncapturederror',this.errorListener);
    device.lost.then(info=>{if(!this.disposed)onFailure(`WebGPU device lost: ${info.message}. Reload to recreate the renderer.`);});
  }
  setVolume(values,distance,n,h) {
    this.texture?.destroy();
    this.texture=this.device.createTexture({dimension:'3d',size:[n,n,n],format:'rgba16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const packed=new Uint16Array(n*n*n*4);
    for(let i=0;i<values.length;i++){
      const valid=Number.isFinite(values[i])&&Number.isFinite(distance[i]);
      packed[i*4]=half(values[i]);packed[i*4+1]=half(distance[i]/h);packed[i*4+2]=valid?0x3c00:0;
    }
    this.device.queue.writeTexture({texture:this.texture},packed,{bytesPerRow:n*8,rowsPerImage:n},[n,n,n]);
    this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:this.texture.createView()},{binding:2,resource:this.sampler},{binding:3,resource:this.transfer.createView()}]});
  }
  setTransfer(lut){this.device.queue.writeTexture({texture:this.transfer},lut,{bytesPerRow:1024},[256,1]);}
  draw(state,capture=false) {
    if(this.disposed||!this.bindGroup)return;
    const box=this.canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,1.5)*state.renderScale;
    const width=Math.max(1,Math.round(box.width*ratio)),height=Math.max(1,Math.round(box.height*ratio));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    const camera=cameraFrame(state.yaw,state.pitch,state.zoom),uniform=new Float32Array(32);
    uniform.set([...camera.eye,width/height],0);uniform.set(camera.right,4);uniform.set(camera.up,8);uniform.set(camera.forward,12);
    uniform.set([state.min,state.max,state.density,state.steps],16);
    uniform.set([state.mode!=='slices'?1:0,state.sliceOpacity,state.band?1:0,state.lighting?1:0],20);
    uniform.set(state.offsets,24);uniform.set(state.planes.map(x=>state.mode==='volume'?0:Number(x)),28);
    this.device.queue.writeBuffer(this.uniform,0,uniform);
    const encoder=this.device.createCommandEncoder(),target=this.context.getCurrentTexture();
    const pass=encoder.beginRenderPass({colorAttachments:[{view:target.createView(),clearValue:{r:.04,g:.06,b:.08,a:1},loadOp:'clear',storeOp:'store'}]});
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindGroup);pass.draw(3);pass.end();
    let buffer,bytesPerRow;
    if(capture){bytesPerRow=Math.ceil(width*4/256)*256;buffer=this.device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyTextureToBuffer({texture:target},{buffer,bytesPerRow,rowsPerImage:height},[width,height]);}
    this.device.queue.submit([encoder.finish()]);
    if(capture)return this.readPNG(buffer,width,height,bytesPerRow);
  }
  async readPNG(buffer,width,height,stride) {
    await buffer.mapAsync(GPUMapMode.READ);const bytes=new Uint8Array(buffer.getMappedRange()),rgba=new Uint8ClampedArray(width*height*4),bgra=this.format.startsWith('bgra');
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const src=y*stride+x*4,dst=(y*width+x)*4;rgba.set([bytes[src+(bgra?2:0)],bytes[src+1],bytes[src+(bgra?0:2)],255],dst);}
    buffer.unmap();buffer.destroy();const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;canvas.getContext('2d').putImageData(new ImageData(rgba,width,height),0,0);return new Promise(resolve=>canvas.toBlob(resolve));
  }
  dispose(){this.disposed=true;this.device.removeEventListener('uncapturederror',this.errorListener);this.texture?.destroy();this.transfer.destroy();this.uniform.destroy();this.context.unconfigure();}
}
class CPUVolume {
  constructor(canvas,onFailure) {
    // A failed WebGPU context cannot be changed into 2D; replace that canvas.
    this.canvas=canvas;
    if(!canvas.getContext('2d')){const replacement=canvas.cloneNode();canvas.replaceWith(replacement);this.canvas=replacement;}
    this.ctx=this.canvas.getContext('2d');this.worker=new Worker(new URL('./volume-worker.js?v=20261022-2',import.meta.url),{type:'module'});this.id=0;this.waiters=[];
    this.worker.onerror=event=>onFailure(`CPU volume renderer failed: ${event.message}`);
    this.worker.onmessage=event=>{
      const {id,pixels,width,height}=event.data;
      if(id===this.id&&!this.pending){this.canvas.width=width;this.canvas.height=height;this.ctx.putImageData(new ImageData(pixels,width,height),0,0);for(const resolve of this.waiters.splice(0))this.canvas.toBlob(resolve);}
      this.busy=false;if(this.pending){const state=this.pending;this.pending=null;this.draw(state);}
    };
  }
  setVolume(values,distance,n,h){this.worker.postMessage({type:'volume',values,distance,n});this.h=h;}
  setTransfer(lut){this.lut=lut;}
  draw(state,capture=false){if(this.disposed||!this.lut)return;const promise=capture?new Promise(resolve=>this.waiters.push(resolve)):undefined;if(this.busy){this.pending=state;return promise;}this.busy=true;const width=Math.round(180*state.renderScale/.8);this.worker.postMessage({id:++this.id,width,height:Math.round(width*this.canvas.clientHeight/Math.max(1,this.canvas.clientWidth)),state,lut:this.lut});return promise;}
  async readPNG(){return new Promise(resolve=>this.canvas.toBlob(resolve));}
  dispose(){this.disposed=true;this.worker.terminate();}
}
