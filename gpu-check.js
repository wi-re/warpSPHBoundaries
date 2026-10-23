import {inspectComputeAdapters} from './gpu-adapter.js';
const output=document.getElementById('report'),button=document.getElementById('check');
async function workerReport(){
 return new Promise(resolve=>{
  const worker=new Worker(new URL('./gpu-check-worker.js',import.meta.url),{type:'module'});
  const finish=report=>{clearTimeout(timer);worker.terminate();resolve(report);};
  const timer=setTimeout(()=>finish({context:'worker',error:'Adapter discovery timed out after 20 seconds'}),20000);
  worker.onmessage=({data})=>finish(data);worker.onerror=e=>finish({context:'worker',error:e.message});worker.postMessage({});
 });
}
function pageReport(){
 return new Promise(resolve=>{
  const timer=setTimeout(()=>resolve({context:'page',error:'Adapter discovery timed out after 20 seconds'}),20000);
  inspectComputeAdapters().then(report=>{clearTimeout(timer);resolve(report);},error=>{clearTimeout(timer);resolve({context:'page',error:error.message});});
 });
}
async function check(){
 button.disabled=true;output.textContent='Checking adapter discovery in the page and in a dedicated worker…';
 try{
  const [page,worker]=await Promise.all([pageReport(),workerReport()]);
  const report={url:location.origin+location.pathname,browser:navigator.userAgent,page,worker};output.textContent=JSON.stringify(report,null,2);
 }catch(error){output.textContent=error.message;}
 finally{button.disabled=false;}
}
button.addEventListener('click',check);check();
