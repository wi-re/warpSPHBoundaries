import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {prepareWallRegions} from '../../../src/warpSPHBoundaries/visualizer/static/wall-browser.js';
import {wallRequests} from '../../../src/warpSPHBoundaries/visualizer/static/wall-browser.js';
import {compileProfiles} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-plan.js';
import {fluxBridgeDispatchLayout} from '../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-gpu.js';

test('geometry reuse keeps both traces of a material interface',()=>{
 const elements=[[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]];
 const result=prepareWallRegions(elements,['solid','porous'],[[.5,.4],[.4,.5]],2,2);
 assert.deepEqual(result.selected,[[0,1],[0,1]]);
 const [solid,porous]=result.surfaces[0];
 assert.equal(solid.surface.faces.length,3);assert.equal(porous.surface.faces.length,3);
 assert.notEqual(solid.surface,porous.surface);
 assert.equal(solid.surface,result.surfaces[1][0].surface);
 assert.equal(porous.surface,result.surfaces[1][1].surface);
 const diagonal=face=>face.every(v=>v[0]===v[1]);
 assert.ok(solid.surface.faces.some(diagonal));assert.ok(porous.surface.faces.some(diagonal));
});

test('chunk planning preserves offset/empty rows and obeys scratch/dispatch budgets',()=>{
 const profiles=JSON.parse(fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/fluxbridge-profiles.json',import.meta.url)));
 const plan=compileProfiles(profiles,3,'w4',wallRequests(3,3)),ranges=new Uint32Array([7,8200,99,0,8200,8200,100,0]);
 const limits={maxStorageBufferBindingSize:64*1024*1024,maxBufferSize:64*1024*1024,maxComputeWorkgroupsPerDimension:65535};
 const layout=fluxBridgeDispatchLayout(plan,ranges,limits);
 assert.equal(layout.chunkCount,3);assert.equal(layout.scratchBytes,3*74*80);
 assert.deepEqual([...layout.ranges.slice(0,8)],[7,8200,0,3,8200,8200,3,3]);
 assert.deepEqual([...layout.ranges.slice(8)],[7,4103,0,0,4103,8199,0,0,8199,8200,0,0]);
 assert.equal(ranges[2],99);
 const small=fluxBridgeDispatchLayout(plan,ranges,{...limits,maxStorageBufferBindingSize:plan.sharedMetadata.byteLength+74*80*2});
 assert.equal(small.chunkCount,2);assert.ok(small.scratchBytes+small.metadata.byteLength<=plan.sharedMetadata.byteLength+74*80*2);
 const bounded=fluxBridgeDispatchLayout(plan,ranges,{...limits,maxComputeWorkgroupsPerDimension:4});
 assert.equal(bounded.chunkCount,2);assert.equal(bounded.workgroups,4);
 const fallback=fluxBridgeDispatchLayout(plan,ranges,{...limits,maxStorageBufferBindingSize:plan.sharedMetadata.byteLength});
 assert.equal(fallback.chunked,false);assert.equal(fallback.shared,true);assert.equal(fallback.scratchBytes,0);
});
