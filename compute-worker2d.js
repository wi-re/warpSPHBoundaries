import {compute2d} from './browser-compute2d.js';
onmessage=event=>{const {id,config}=event.data;try{postMessage({id,result:compute2d(config,progress=>postMessage({id,progress}))});}catch(error){postMessage({id,error:error.message});}};
