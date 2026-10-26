// Discovery is shared by the wall worker and the page/worker diagnostic.
// Compatibility mode still executes WGSL fp32 on WebGPU; no CPU/native fallback.
export const ADAPTER_REQUESTS=[
 {powerPreference:'high-performance'},
 {},
 {featureLevel:'compatibility',powerPreference:'high-performance'},
 {featureLevel:'compatibility'}
];
export function adapterSummary(adapter){
 const info=adapter.info??{},limits=adapter.limits??{};
 return {vendor:info.vendor??'',architecture:info.architecture??'',device:info.device??'',description:info.description??'',
  core_features_and_limits:adapter.features?.has('core-features-and-limits')??null,
  limits:Object.fromEntries(['maxStorageBuffersPerShaderStage','maxStorageBufferBindingSize','maxBufferSize','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupsPerDimension'].map(k=>[k,limits[k]??null]))};
}
async function attempt(gpu,options){
 try{const adapter=await gpu.requestAdapter(options);return {adapter,record:{options,status:adapter?'available':'unavailable',...(adapter?{adapter:adapterSummary(adapter)}:{})}};}
 catch(error){return {adapter:null,record:{options,status:'error',error:error.message}};}
}
export async function selectComputeAdapter(gpu){
 if(!gpu)throw Error('WebGPU is not exposed in this context. Open the HTTPS site in a WebGPU-capable browser.');
 const attempts=[];
 for(const options of ADAPTER_REQUESTS){
  const {adapter,record}=await attempt(gpu,options);attempts.push(record);
  if(adapter)return {adapter,featureLevel:options.featureLevel??'core',options,attempts};
 }
 const error=Error('Chrome/browser returned no WebGPU adapter for core or compatibility requests. Open Browser GPU diagnostics below; graphics acceleration status alone does not establish adapter availability.');
 error.attempts=attempts;throw error;
}
export async function inspectComputeAdapters(gpu=globalThis.navigator?.gpu){
 const report={context:typeof document==='undefined'?'worker':'page',secure_context:globalThis.isSecureContext??null,api_exposed:!!gpu,attempts:[]};
 if(gpu)for(const options of ADAPTER_REQUESTS)report.attempts.push((await attempt(gpu,options)).record);
 return report;
}
