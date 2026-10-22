import {cameraFrame,sampleVolume,clamp} from './volume-math.js';
let values,distance,n;
onmessage=event=>{
  if(event.data.type==='volume'){({values,distance,n}=event.data);return;}
  const {id,width,height,state,lut}=event.data;
  const pixels=new Uint8ClampedArray(width*height*4),frame=cameraFrame(state.yaw,state.pitch,state.zoom),steps=Math.min(160,state.steps);
  for(let py=0;py<height;py++)for(let px=0;px<width;px++){
    const sx=(2*(px+.5)/width-1)*width/height*.48,sy=(1-2*(py+.5)/height)*.48;
    let rd=frame.forward.map((x,j)=>x+sx*frame.right[j]+sy*frame.up[j]);const length=Math.hypot(...rd);rd=rd.map(x=>x/length);
    let near=0,far=Infinity;
    for(let j=0;j<3;j++){const inv=1/(Math.abs(rd[j])>1e-6?rd[j]:1e-6),a=(-.5-frame.eye[j])*inv,b=(.5-frame.eye[j])*inv;near=Math.max(near,Math.min(a,b));far=Math.min(far,Math.max(a,b));}
    let r=0,g=0,b=0,a=0;
    const bg=[12,20,27];
    if(far>near){
      const dt=(far-near)/steps,planeT=state.offsets.map((x,j)=>(x-frame.eye[j])/rd[j]);
      const composite=(t,isSlice)=>{
        const uv=rd.map((x,j)=>frame.eye[j]+x*t+.5),v=sampleVolume(values,n,uv);
        if(!Number.isFinite(v)||(state.band&&Math.abs(sampleVolume(distance,n,uv))>state.h))return;
        const index=clamp(Math.round((v-state.min)/(state.max-state.min)*255),0,255)*4;
        const alpha=isSlice?state.sliceOpacity:1-(1-lut[index+3]/255)**(state.density*dt*128),factor=(1-a)*alpha;
        r+=factor*lut[index];g+=factor*lut[index+1];b+=factor*lut[index+2];a+=factor;
      };
      for(let k=0;k<steps&&a<.995;k++){
        const t=near+k*dt;if(state.mode!=='slices')composite(t+dt*.5,false);
        if(state.mode!=='volume')for(let axis=0;axis<3;axis++)if(state.planes[axis]&&planeT[axis]>=t&&planeT[axis]<t+dt)composite(planeT[axis],true);
      }
    }
    const i=(py*width+px)*4;pixels.set([r+(1-a)*bg[0],g+(1-a)*bg[1],b+(1-a)*bg[2],255],i);
  }
  postMessage({id,width,height,pixels},[pixels.buffer]);
};
