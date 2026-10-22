// Pure volume-only CPU ray caster, also used by regression tests. Slice
// offsets affect the separate 2D plots, never these rendered pixels.
import {cameraFrame,sampleVolume,clamp} from './volume-math.js?v=20261022-4';
export function rayBox(origin,direction){
 let near=0,far=Infinity;
 for(let j=0;j<3;j++){
  if(Math.abs(direction[j])<1e-8){if(Math.abs(origin[j])>.5)return null;continue;}
  const a=(-.5-origin[j])/direction[j],b=(.5-origin[j])/direction[j];near=Math.max(near,Math.min(a,b));far=Math.min(far,Math.max(a,b));
 }
 return far>near?[near,far]:null;
}
export function renderPixels(values,distance,n,width,height,state,lut){
 const pixels=new Uint8ClampedArray(width*height*4),frame=cameraFrame(state.yaw,state.pitch,state.zoom);
 for(let py=0;py<height;py++)for(let px=0;px<width;px++){
  const sx=(2*(px+.5)/width-1)*width/height*.48,sy=(1-2*(py+.5)/height)*.48;
  let rd=frame.forward.map((x,j)=>x+sx*frame.right[j]+sy*frame.up[j]);const length=Math.hypot(...rd);rd=rd.map(x=>x/length);
  const interval=rayBox(frame.eye,rd);let r=0,g=0,b=0,a=0;const bg=[12,20,27];
  if(interval){
   const [near,far]=interval,steps=Math.max(Math.min(768,state.steps),Math.ceil((far-near)*(n-1)*2)),dt=(far-near)/steps;
   const composite=t=>{
    const uv=rd.map((x,j)=>clamp(frame.eye[j]+x*t+.5,0,1)),v=sampleVolume(values,n,uv);
    if(!Number.isFinite(v)||(state.band&&Math.abs(sampleVolume(distance,n,uv))>state.h))return;
    const index=clamp(Math.round((v-state.min)/(state.max-state.min)*255),0,255)*4;
    const opacity=lut[index+3]/255,alpha=1-(1-opacity)**(state.density*dt*128),factor=(1-a)*alpha;
    r+=factor*lut[index];g+=factor*lut[index+1];b+=factor*lut[index+2];a+=factor;
   };
   for(let k=0;k<steps&&a<.995;k++)composite(near+(k+.5)*dt);
  }
  pixels.set([r+(1-a)*bg[0],g+(1-a)*bg[1],b+(1-a)*bg[2],255],(py*width+px)*4);
 }
 return pixels;
}
