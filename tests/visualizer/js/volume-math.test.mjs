import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleVolume,sliceValues,makeTransfer,decodeVolume,cameraFrame,difference} from '../../../src/warpSPHBoundaries/visualizer/static/volume-math.js';
const near=(a,b,t=1e-6)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const n=5,values=Float32Array.from({length:n**3},(_,i)=>{const x=i%n,y=Math.floor(i/n)%n,z=Math.floor(i/n/n);return x+10*y+100*z;});
test('trilinear sampling uses x-fastest z/y/x order',()=>{near(sampleVolume(values,n,[.25,.5,.75]),321);near(sampleVolume(values,n,[.1,.2,.3]),128.4);assert.ok(Number.isNaN(sampleVolume(values,n,[-1,0,0])));});
test('all orthogonal slice orientations and fractional offsets',()=>{const xy=sliceValues(values,n,2,.375),xz=sliceValues(values,n,1,.375),yz=sliceValues(values,n,0,.375);near(xy[3*n+2],182);near(xz[3*n+2],317);near(yz[3*n+2],321.5);});
test('missing samples remain missing but zero-weight neighbours do not poison a node',()=>{const a=new Float32Array(values);a[1]=NaN;near(sampleVolume(a,n,[0,0,0]),0);assert.ok(Number.isNaN(sampleVolume(a,n,[.125,0,0])));});
test('transfer RGB is colormap-based and opacity interpolates independently',()=>{const palette=Array.from({length:256},(_,i)=>[i,0,255-i]);const lut=makeTransfer(palette,[[0,0],[.5,1],[1,0]]);assert.equal(lut[0],0);assert.equal(lut[3],0);assert.ok(lut[127*4+3]>253);assert.equal(lut[255*4+3],0);assert.equal(lut[255*4],255);});
test('float32/base64 volume round trip rejects malformed payloads',()=>{const bytes=new Uint8Array(new Float32Array([1,2.5,-3]).buffer);const encoded=btoa(String.fromCharCode(...bytes));assert.deepEqual([...decodeVolume(encoded,3)],[1,2.5,-3]);assert.throws(()=>decodeVolume(encoded,4));});
test('camera basis is orthonormal',()=>{const {eye,right,up,forward}=cameraFrame(.7,.4,2.3);const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);near(Math.hypot(...eye),2.3);for(const v of [right,up,forward])near(dot(v,v),1);near(dot(right,up),0);near(dot(right,forward),0);near(dot(up,forward),0);});
test('differences retain sign and unavailable values',()=>{assert.deepEqual([...difference(new Float32Array([1,2]),new Float32Array([3,1]))],[-2,1]);assert.deepEqual([...difference(new Float32Array([1,2]),new Float32Array([3,1]),true)],[2,1]);});
test('signed-error transfer leaves zero transparent on both sides of the LUT midpoint',()=>{const palette=Array.from({length:256},()=>[100,150,200]);const lut=makeTransfer(palette,[[0,.38],[.3,.1],[.5,0],[.7,.1],[1,.38]]);assert.equal(lut[127*4+3],0);assert.equal(lut[128*4+3],0);assert.ok(lut[3]>90);assert.ok(lut[255*4+3]>90);});
test('comparison retains the raw reference and only subtracts later fields',async()=>{
 const {comparisonFields}=await import('../../../src/warpSPHBoundaries/visualizer/static/volume-math.js');
 const ref=new Float32Array([1,2,NaN]),other=new Float32Array([.5,3,4]);
 for(const mode of ['difference','absolute']){const fields=comparisonFields([ref,other],mode);assert.equal(fields[0],ref);assert.equal(fields[1][0],mode==='absolute'?.5:-.5);assert.equal(fields[1][1],1);assert.ok(Number.isNaN(fields[1][2]));}
});
test('parallel and negative ray directions intersect the same cube slabs',async()=>{
 const {rayBox}=await import('../../../src/warpSPHBoundaries/visualizer/static/volume-ray.js');
 assert.deepEqual(rayBox([0,0,2],[0,0,-1]),[1.5,2.5]);assert.deepEqual(rayBox([0,0,-2],[0,0,1]),[1.5,2.5]);assert.equal(rayBox([1,0,2],[0,0,-1]),null);
});
test('CPU volumes ignore slice flags and offsets from older saved views',async()=>{
 const {renderPixels}=await import('../../../src/warpSPHBoundaries/visualizer/static/volume-ray.js');
 const field=new Float32Array(5**3).fill(.8),distance=new Float32Array(5**3).fill(-1),lut=makeTransfer(Array.from({length:256},()=>[80,140,210]),[[0,.1],[1,.1]]);
 const state={yaw:.6,pitch:.3,zoom:2.3,steps:64,min:0,max:1,density:.5,band:false,h:.3};
 const volume=renderPixels(field,distance,5,24,24,state,lut),legacy=renderPixels(field,distance,5,24,24,{...state,mode:'slices',planes:[true,true,true],offsets:[.2,0,-.1],sliceOpacity:1},lut);
 assert.deepEqual(legacy,volume);
});
