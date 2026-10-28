import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {compileAnalytic2d} from '../../../src/warpSPHBoundaries/visualizer/static/analytic2d-plan.js';
import {compileAnalytic3d} from '../../../src/warpSPHBoundaries/visualizer/static/analytic3d-plan.js';
import {compileSeries} from '../../../src/warpSPHBoundaries/visualizer/static/series-plan.js';
import {specializeAnalyticShader} from '../../../src/warpSPHBoundaries/visualizer/static/analytic-specialize.js';
import {analyticPipeline} from '../../../src/warpSPHBoundaries/visualizer/static/analytic-gpu.js';
const source=fs.readFileSync(new URL('../../../src/warpSPHBoundaries/visualizer/static/analytic3d-compute.wgsl',import.meta.url),'utf8');
const config={radius:1,width:2,height:2,depth:2,h:1,padding:1,quality:1,geometry:'triangle',kernel:'w2',field:'m00',models:['analytic'],resolution:17};
const plan=compileAnalytic2d(config);

test('variants reuse across grids, geometry and axes, but distinguish radial contracts',()=>{
 const key=c=>specializeAnalyticShader(source,compileAnalytic2d({...config,...c})).key;
 assert.equal(key({}),key({resolution:33,h:.7,geometry:'rectangle'}));
 assert.notEqual(key({}),key({field:'Gx'}));
 assert.equal(key({field:'Gx'}),key({field:'Gy'}));
 assert.notEqual(key({}),key({kernel:'w4'}));
 assert.notEqual(key({}),specializeAnalyticShader(source,compileAnalytic3d({...config,geometry:'tetrahedron'})).key);
 const series=compileSeries({...config,geometry:'circle',models:['disk_series']},2);
 assert.equal(specializeAnalyticShader(source,series).code,source);
});

test('concurrent calls compile once; a failed compilation can be retried',async()=>{
 let compiles=0,fail=true;
 const device={createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),createComputePipelineAsync:async()=>{compiles++;if(fail)throw Error('compiler failed');return {compiled:true};}};
 const resource={source,device,pipelines:new Map()};
 await assert.rejects(analyticPipeline(resource,plan),/compiler failed/);
 fail=false;
 const [a,b]=await Promise.all([analyticPipeline(resource,plan),analyticPipeline(resource,plan)]);
 assert.equal(a,b);assert.equal(compiles,2);
});
