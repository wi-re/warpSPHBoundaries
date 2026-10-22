// Pure CPU ray caster, also used by regression tests. Integrate segments in
// depth order, splitting at slice intersections rather than overpainting them.
import {cameraFrame,sampleVolume,clamp} from './volume-math.js?v=20261022-2';
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
   const events=state.mode==='volume'?[]:state.offsets.map((offset,j)=>state.planes[j]&&Math.abs(rd[j])>1e-8?(offset-frame.eye[j])/rd[j]:Infinity).filter(t=>t>=near&&t<=far).sort((a,b)=>a-b);
   const composite=(t,segment,isSlice)=>{
    const uv=rd.map((x,j)=>clamp(frame.eye[j]+x*t+.5,0,1)),v=sampleVolume(values,n,uv);
    if(!Number.isFinite(v)||(state.band&&Math.abs(sampleVolume(distance,n,uv))>state.h))return;
    const index=clamp(Math.round((v-state.min)/(state.max-state.min)*255),0,255)*4;
    // A zero-opacity slice sample must not hide volume behind the plane.
    const opacity=lut[index+3]/255,alpha=isSlice?state.sliceOpacity*Math.min(1,opacity*8):1-(1-opacity)**(state.density*segment*128),factor=(1-a)*alpha;
    r+=factor*lut[index];g+=factor*lut[index+1];b+=factor*lut[index+2];a+=factor;
   };
   let cursor=near,event=0;
   for(let k=0;k<steps&&a<.995;k++){
    const end=k===steps-1?far:near+(k+1)*dt;
    while(event<events.length&&events[event]<=end){const t=events[event++];if(state.mode!=='slices'&&t>cursor)composite((cursor+t)/2,t-cursor,false);composite(t,0,true);cursor=t;}
    if(state.mode!=='slices'&&end>cursor)composite((cursor+end)/2,end-cursor,false);cursor=end;
   }
  }
  pixels.set([r+(1-a)*bg[0],g+(1-a)*bg[1],b+(1-a)*bg[2],255],(py*width+px)*4);
 }
 return pixels;
}
