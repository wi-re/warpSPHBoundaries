let worker,job=0;const pending=new Map();
export function computeBrowser2d(config,progress=()=>{}){
 if(!worker){worker=new Worker(new URL('./compute-worker2d.js',import.meta.url),{type:'module'});worker.onmessage=event=>{const {id,result,error,progress:note}=event.data,item=pending.get(id);if(!item)return;if(note){item.progress(note);return;}pending.delete(id);error?item.reject(new Error(error)):item.resolve(result);};worker.onerror=event=>{for(const item of pending.values())item.reject(new Error(event.message));pending.clear();worker.terminate();worker=null;};}
 return new Promise((resolve,reject)=>{const id=++job;pending.set(id,{resolve,reject,progress});worker.postMessage({id,config});});
}
export function disposeCompute2d(){worker?.terminate();worker=null;for(const item of pending.values())item.reject(new Error('Compute worker closed'));pending.clear();}
