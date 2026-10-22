import {renderPixels} from './volume-ray.js';
let values,distance,n;
onmessage=event=>{
 if(event.data.type==='volume'){({values,distance,n}=event.data);return;}
 const {id,width,height,state,lut}=event.data,pixels=renderPixels(values,distance,n,width,height,state,lut);
 postMessage({id,width,height,pixels},[pixels.buffer]);
};
