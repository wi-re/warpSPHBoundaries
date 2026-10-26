import {inspectComputeAdapters} from './gpu-adapter.js';
self.onmessage=async()=>{
 try{self.postMessage(await inspectComputeAdapters());}
 catch(error){self.postMessage({context:'worker',error:error.message});}
};
