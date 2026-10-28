// Reuse the worker/device/pipeline between successful evaluations. Termination
// cancels CPU preparation immediately and releases the worker's WebGPU context.
let worker=null,active=null;
export function cancelWall(){
 if(!active)return;const pending=active;active=null;worker.terminate();worker=null;
 pending.reject(new DOMException('Evaluation cancelled','AbortError'));
}
export function computeWallInBrowser(config,progress=()=>{},comparison=false){
 cancelWall();return new Promise((resolve,reject)=>{
  if(!worker){
   worker=new Worker(new URL('./wall-worker.js?v=20261028-webgpu1',import.meta.url),{type:'module'});
   worker.onmessage=({data})=>{
    if(!active)return;
    if(data.type==='progress'){active.progress(data.detail);return;}
    const pending=active;active=null;
    if(data.type==='result')pending.resolve(data.result);else pending.reject(Error(data.error||'Browser wall worker failed'));
   };
   worker.onerror=event=>{const pending=active;active=null;worker.terminate();worker=null;pending?.reject(Error(event.message||'Browser wall worker failed'));};
  }
  active={resolve,reject,progress};worker.postMessage({config,comparison});
 });
}
