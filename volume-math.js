// Shared by UI, CPU fallback worker, and numerical renderer checks.
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function cameraFrame(yaw,pitch,distance) {
  const eye=[distance*Math.cos(pitch)*Math.sin(yaw),distance*Math.sin(pitch),distance*Math.cos(pitch)*Math.cos(yaw)];
  const forward=eye.map(x=>-x/distance),right=[Math.cos(yaw),0,-Math.sin(yaw)];
  const up=[-Math.sin(pitch)*Math.sin(yaw),Math.cos(pitch),-Math.sin(pitch)*Math.cos(yaw)];
  return {eye,forward,right,up};
}
export function sampleVolume(values,n,uv) {
  if(uv.some(x=>x<0||x>1))return NaN;
  const p=uv.map(x=>x*(n-1)),i=p.map(x=>Math.min(n-2,Math.floor(x))),t=p.map((x,j)=>x-i[j]);
  let result=0;
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){
    const weight=(x?t[0]:1-t[0])*(y?t[1]:1-t[1])*(z?t[2]:1-t[2]);
    if(weight<=1e-12)continue;
    const value=values[((i[2]+z)*n+i[1]+y)*n+i[0]+x];
    if(!Number.isFinite(value))return NaN;
    result+=value*weight;
  }
  return result;
}
export function makeTransfer(palette,points) {
  const sorted=points.map(p=>[clamp(p[0],0,1),clamp(p[1],0,1)]).sort((a,b)=>a[0]-b[0]);
  const lut=new Uint8Array(256*4);
  for(let i=0;i<256;i++){
    const t=i/255;let j=0;while(j<sorted.length-2&&t>sorted[j+1][0])j++;
    const a=sorted[j],b=sorted[j+1],u=clamp((t-a[0])/Math.max(b[0]-a[0],1e-8),0,1);
    lut.set([...palette[i],Math.round(255*(a[1]*(1-u)+b[1]*u))],i*4);
  }
  return lut;
}
export function decodeVolume(encoded,count) {
  if(typeof encoded!=='string'||encoded.length!==Math.ceil(count*4/3)*4)throw new Error('Invalid encoded volume length');
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  if(bytes.length!==count*4)throw new Error('Invalid volume byte count');
  // Explicit endian decode: exported data is always little endian.
  const view=new DataView(bytes.buffer),values=new Float32Array(count);
  for(let i=0;i<count;i++)values[i]=view.getFloat32(i*4,true);
  return values;
}
export function difference(values,reference,absolute=false) {
  return Float32Array.from(values,(v,i)=>absolute?Math.abs(v-reference[i]):v-reference[i]);
}
export function sliceValues(values,n,axis,offset) {
  const out=new Float32Array(n*n),axes=axis===0?[1,2]:axis===1?[0,2]:[0,1];
  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const uv=[0,0,0];uv[axis]=offset;uv[axes[0]]=i/(n-1);uv[axes[1]]=j/(n-1);
    out[j*n+i]=sampleVolume(values,n,uv);
  }
  return out;
}
// A comparison keeps the original first field; only subsequent views subtract.
export function comparisonFields(arrays,display){
 return arrays.map((values,i)=>!values?null:display==='value'||i===0?values:arrays[0]?difference(values,arrays[0],display==='absolute'):Float32Array.from(values,()=>NaN));
}
