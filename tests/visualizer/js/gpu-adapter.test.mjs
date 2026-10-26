import test from 'node:test';
import assert from 'node:assert/strict';
import {selectComputeAdapter,inspectComputeAdapters,ADAPTER_REQUESTS} from '../../../src/warpSPHBoundaries/visualizer/static/gpu-adapter.js';
const adapter={info:{vendor:'test'},features:new Set(),limits:{maxStorageBuffersPerShaderStage:8}};
test('core adapter returns immediately',async()=>{
 const calls=[];const result=await selectComputeAdapter({requestAdapter:async options=>{calls.push(options);return adapter;}});
 assert.equal(result.adapter,adapter);assert.equal(result.featureLevel,'core');assert.deepEqual(calls,[ADAPTER_REQUESTS[0]]);
});
test('default request handles a rejected high-performance preference',async()=>{
 const calls=[];const result=await selectComputeAdapter({requestAdapter:async options=>{calls.push(options);return options.powerPreference?null:adapter;}});
 assert.equal(result.featureLevel,'core');assert.deepEqual(calls,ADAPTER_REQUESTS.slice(0,2));
});
test('compatibility-only adapter stays a GPU path and records discovery',async()=>{
 const result=await selectComputeAdapter({requestAdapter:async options=>options.featureLevel==='compatibility'?adapter:null});
 assert.equal(result.adapter,adapter);assert.equal(result.featureLevel,'compatibility');assert.deepEqual(result.attempts.map(a=>a.status),['unavailable','unavailable','available']);
});
test('unsupported options are recoverable; total failure preserves diagnostics',async()=>{
 const gpu={requestAdapter:async options=>{if(options.powerPreference)throw Error('unsupported preference');return null;}};
 await assert.rejects(selectComputeAdapter(gpu),error=>error.attempts.length===4&&error.attempts[0].error==='unsupported preference');
 const report=await inspectComputeAdapters(gpu);assert.equal(report.api_exposed,true);assert.equal(report.attempts.length,4);
});
test('missing API is distinct from unavailable adapters',async()=>{
 await assert.rejects(selectComputeAdapter(null),/not exposed/);
 assert.deepEqual((await inspectComputeAdapters(null)).attempts,[]);
});
