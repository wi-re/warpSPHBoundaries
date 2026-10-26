import {computeBrowserWall,compareBrowserWall} from './wall-browser.js';
self.onmessage=async ({data})=>{
 try{const result=await (data.comparison?compareBrowserWall:computeBrowserWall)(data.config,detail=>self.postMessage({type:'progress',detail}));self.postMessage({type:'result',result});}
 catch(error){self.postMessage({type:'error',error:error.message});}
};
