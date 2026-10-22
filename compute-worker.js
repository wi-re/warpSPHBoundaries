import {computeCPU} from './browser-compute.js?v=20261022-2';
onmessage=event=>{const {id,config}=event.data;try{const result=computeCPU(config,progress=>postMessage({id,progress}));postMessage({id,result});}catch(error){postMessage({id,error:error.message});}};
