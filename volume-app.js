import {renderMethods} from './method-catalogue.js?v=20261022-7';
import {contractElement} from './model-contract.js?v=20261022-7';
import {gridControl} from './grid-control.js?v=20261022-7';
import {FIELD_INFO,FIELD_IDS} from './term-engine.js?v=20261022-7';
import {BROWSER_MODELS} from './browser-compute.js?v=20261022-7';
import {computeBrowser,disposeCompute} from './browser-engine.js?v=20261022-7';
import {renderTermPanel} from './term-panel.js?v=20261022-7';
import {VolumeRenderer} from './volume-renderer.js?v=20261022-7';
import {cameraFrame,clamp,decodeVolume,comparisonFields,makeTransfer,sampleVolume,sliceValues} from './volume-math.js?v=20261022-7';
const $=id=>document.getElementById(id),create=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
const PYTHON_MODELS={sphere:['Sphere · polynomial integral','Float64 exact-degree radial shell integration. Sphere geometry only.'],sdf:['SDF · local plane','Exact 3D half-space formula using the geometry distance; includes model error.'],voxel:['Voxel · FFT quadrature','Sampled solid occupancy convolved with the 3D kernel; grid-dependent quadrature.']};
const MODELS={...PYTHON_MODELS,...BROWSER_MODELS};
const DEFAULT={dimension:3,geometry:'sphere',radius:1,width:2,height:2,depth:2,h:.6,resolution:33,padding:1.15,kernel:'w4',field:'m00',models:['sphere','sdf','quadrature'],supersampling:2,quality:1,curvature_order:2,series_order:4};
const PRESETS={boundary:[[0,0],[.15,.02],[.5,.28],[.85,.04],[1,0]],difference:[[0,.38],[.3,.1],[.5,0],[.7,.1],[1,.38]],solid:[[0,0],[.15,0],[.5,.1],[.85,.3],[1,.5]],linear:[[0,0],[.25,.05],[.5,.1],[.75,.2],[1,.4]],constant:[[0,.08],[1,.08]]};
const fmt=x=>!Number.isFinite(x)?'—':(Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(3):x.toFixed(4));
const short=x=>Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(2):Number(x.toPrecision(4)).toString();
let config=structuredClone(DEFAULT),data,arrays=[],distance,fields=[],palettes,lut,online=false,busy=false,revision=0,timer,queued=false,generation=0;
let renderers=[],renderStatus=[],framePending=false,selectedModel=0,importRevision=-1;
let state={yaw:.68,pitch:.36,zoom:2.3,mode:'volume',steps:256,density:.65,renderScale:.8,lighting:true,band:false,box:true,
  colormap:'viridis',opacity:'boundary',knots:structuredClone(PRESETS.boundary),min:0,max:1,offsets:[0,0,0],planes:[false,false,false],sliceOpacity:0,display:'value'};
let sliceOffsets=[0,0,0],provider='browser';
function canCompute(){return provider!=='python'||online;}
$('field').replaceChildren(...FIELD_INFO.map(([id,label])=>{const o=create('option','',label);o.value=id;return o;}));
function message(text){$('message').hidden=!text;$('message').textContent=text;}
function status(text,stale=false){$('status').textContent=text;$('status').classList.toggle('stale',stale);}
function changed(){renderMethods($('method-catalogue'),3,config);revision++;status('Configuration changed · volume awaits evaluation',true);$('compute-note').textContent=canCompute()?'Awaiting new integral grid':'Local Python service is unavailable';renderTermPanel($('term-equations'),config);clearTimeout(timer);if(canCompute()&&$('auto').checked)timer=setTimeout(evaluate,450);}
function rangeControl(parent,key,label,min,max,step,target,onChange,unit=''){
  const labelElement=create('label','range'),top=create('span','range-top'),output=create('output'),input=create('input');input.type='range';input.min=min;input.max=max;input.step=step;input.value=target[key];input.setAttribute('aria-label',label);
  const update=()=>{output.value=`${Number(input.value).toFixed(step<.1?2:step<1?1:0)}${unit}`;};update();top.append(create('span','',label),output);labelElement.append(top,input);parent.append(labelElement);
  input.oninput=()=>{target[key]=+input.value;update();onChange();};
}
function syncControls(){
  renderMethods($('method-catalogue'),3,config);
  for(const key of ['geometry','kernel','field','supersampling','quality','curvature_order','series_order'])$(key).value=config[key];$('occupancy-control').hidden=provider!=='python';$('quality').disabled=provider==='python';
  const shape=$('shape-controls');shape.replaceChildren();
  if(config.geometry==='sphere')rangeControl(shape,'radius','Sphere radius R',.2,5,.05,config,changed);
  else for(const [key,label]of [['width','Width x'],['height','Height y'],['depth','Depth z']])rangeControl(shape,key,label,.2,6,.05,config,changed);
  const sampling=$('grid-controls');sampling.replaceChildren();rangeControl(sampling,'h','Support radius h',.15,2,.05,config,changed);
  gridControl(sampling,config,{label:'Volume resolution',max:129,sliderMax:81,unit:'³',changed});rangeControl(sampling,'padding','View padding',1,2,.05,config,changed,'h');
  const models=$('models');models.replaceChildren();
  Object.entries(provider==='python'?{...PYTHON_MODELS,curvature:MODELS.curvature,ball_series:MODELS.ball_series}:BROWSER_MODELS).forEach(([key,[title,note]])=>{
    const item=create('div','volume-model'),label=create('label','check'),input=create('input');input.type='checkbox';input.checked=config.models.includes(key);input.setAttribute('aria-label',title);
    input.onchange=()=>{if(!input.checked&&config.models.length===1){input.checked=true;return;}if(input.checked&&config.models.length>=3){input.checked=false;message('Choose at most three models.');return;}config.models=input.checked?[...config.models,key]:config.models.filter(m=>m!==key);changed();};
    label.append(input,document.createTextNode(title));item.append(label,create('small','',note));models.append(item);
  });
}
function preset(kind){config={...structuredClone(DEFAULT),kernel:config.kernel,resolution:config.resolution,geometry:['curvature','small-ball'].includes(kind)?'sphere':kind};if(kind==='curvature'){config.models=['sphere','sdf','curvature'];config.h=.3;}if(kind==='small-ball'){config.radius=.2;config.h=1.2;config.models=['sphere','ball_series','quadrature'];}if(!['sphere','curvature','small-ball'].includes(kind))config.models=provider==='python'?['voxel','sdf']:['analytic','quadrature','sdf'];if(provider==='python')config.models=config.models.map(m=>m==='quadrature'?'voxel':m);if(kind==='tetrahedron'){config.h=.55;config.width=2;config.height=2;config.depth=2;}if(kind==='box'){config.width=2.6;config.height=1.6;config.depth=1.2;config.h=.65;}sliceOffsets=[0,0,0];state.offsets=[0,0,0];syncControls();changed();}
function connect(value){online=value;$('connection').textContent=provider==='python'?(value?'Python · local':'Python unavailable'):'Browser · live compute';$('connection').classList.toggle('offline',provider==='python'&&!value);$('compute').disabled=!canCompute();}
async function evaluate(){
  clearTimeout(timer);if(!canCompute())return;if(busy){queued=true;return;}busy=true;queued=false;const version=revision;
  $('compute').disabled=true;$('compute').textContent='Computing volume…';$('volumes').classList.add('pending-volume');message('');status(provider==='python'?'Computing integral grid in local Python…':'Computing moment integrals in the browser…');
  try{
    let result;if(provider==='python'){const response=await fetch('./api/compute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});result=await response.json();if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);}else result=await computeBrowser(structuredClone(config),note=>{if(version===revision)status(note);},provider==='worker'?'worker':provider==='gpu'?'gpu':'auto');
    if(version===revision){await loadVolume(result,false);status(`Computed ${(result.shape[0]**3).toLocaleString()} samples/model in ${(result.milliseconds/1000).toFixed(2)} s`);$('compute-note').textContent=provider==='python'?'Up to date · local Python':'Up to date · browser computation';if(result.compute_fallback)message('WebGPU compute unavailable: '+result.compute_fallback+'. Evaluated with the JavaScript float64 worker.');}
  }catch(e){message(e.message);status('Volume evaluation failed',true);$('compute-note').textContent='Evaluation failed · previous volume retained';if(provider==='gpu'){$('connection').textContent='WebGPU · evaluation failed';$('connection').classList.add('offline');}}
  finally{busy=false;$('compute').disabled=!canCompute();$('compute').innerHTML='Evaluate volume <span>↗</span>';$('volumes').classList.remove('pending-volume');if(queued||(version!==revision&&revision!==importRevision&&$('auto').checked))evaluate();}
}
function validateResult(result){
  if(result?.dimension!==3||result.schema_version!==1||result.encoding!=='base64-f32-le'||result.order!=='zyx'||!Array.isArray(result.shape)||result.shape.length!==3)throw new Error('Expected a dimension-3 volume with base64-f32-le encoding and zyx storage');
  const n=result.shape[0];if(!Number.isInteger(n)||n<17||n>129||result.shape.some(v=>v!==n))throw new Error('Volume dimensions must be equal, with 17–129 nodes per axis');
  for(const axis of ['x','y','z']){if(!Array.isArray(result[axis])||result[axis].length!==n||result[axis].some((v,i)=>!Number.isFinite(v)||(i&&v<=result[axis][i-1])))throw new Error('Invalid volume coordinate axes');}
  // Ray casting uses one uniformly sampled cube. Reject alternate layouts rather
  // than displaying stretched or incorrectly indexed imported data.
  const step=(result.x.at(-1)-result.x[0])/(n-1);
  for(const axis of ['x','y','z'])for(let i=0;i<n;i++)if(Math.abs(result[axis][i]-(result.x[0]+step*i))>1e-5*Math.max(1,Math.abs(result.x.at(-1))))throw new Error('Current renderer requires identical, uniform x/y/z axes');
  if(!result.config||!Number.isFinite(result.config.h)||result.config.h<=0||!['sphere','box','tetrahedron'].includes(result.config.geometry)||!Object.hasOwn(FIELD_IDS,result.config.field))throw new Error('Invalid 3D geometry/field configuration');
  if(!Array.isArray(result.results)||result.results.length<1||result.results.length>3||new Set(result.results.map(r=>r.model)).size!==result.results.length)throw new Error('Invalid volume model list');
  const decoded=result.results.map(r=>{if(!MODELS[r.model])throw new Error('Unknown volume model');if(r.error)return null;if(!Array.isArray(r.notes)||!Number.isFinite(r.milliseconds))throw new Error('Invalid model metadata');return decodeVolume(r.values,n**3);});
  return {decoded,distance:decodeVolume(result.distance,n**3)};
}
async function loadVolume(result,imported=true){
  const previousField=data?.config.field,decoded=validateResult(result);data=result;arrays=decoded.decoded;distance=decoded.distance;
  if(previousField!==result.config.field&&state.opacity!=='custom')sensibleOpacity();
  if(imported){revision++;importRevision=revision;queued=false;clearTimeout(timer);config={...structuredClone(DEFAULT),...result.config,models:result.results.map(r=>provider==='python'?(r.model==='quadrature'?'voxel':r.model):(r.model==='voxel'?'quadrature':r.model))};syncControls();}
  selectedModel=Math.max(0,result.results.findIndex(r=>!r.error));
  for(let j=0;j<3;j++)sliceOffsets[j]=clamp(sliceOffsets[j],data[['x','y','z'][j]][0],data[['x','y','z'][j]].at(-1));
  state.h=data.config.h;updateOffsets();fitWindow(false);buildSliceControls();buildSlices();buildSliceSelector();await buildVolumes();showNotes();renderTermPanel($('term-equations'),result.config);$('save').disabled=false;
  $('quantity-note').textContent=result.config.field==='grad_norm'?'Gradient magnitude combines vector components before taking the norm. Geometry/model error and integration error are described separately beneath each view.':/^(G|H|g1)|laplacian/.test(result.config.field)?'Signed kernel-derivative fields can cross zero inside the volume. The signed opacity preset makes those zero sets transparent; use “Constant opacity” or the full-color slices to distinguish a zero from missing data. Local SDF normals can change across nearest-face/medial boundaries.':'';
  $('grid-status').textContent=`${result.shape.join(' × ')} · h = ${result.config.h} · ${result.config.kernel} · ${result.config.field} · z/y/x storage`;
  if(imported)status('Saved 3D volume loaded · display controls remain interactive');
  message('');drawEverything();
}
function viewState(index){
 if(index!==0||state.display==='value'||!arrays[0])return {...state};
 let min=Infinity,max=-Infinity;for(const v of arrays[0])if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}
 if(!Number.isFinite(min)){min=0;max=1;}if(data.config.field==='m00'&&min>-1e-6)min=0;if(min<0&&max>0){max=Math.max(-min,max);min=-max;}if(max===min)max=min+1e-6;
 return {...state,min,max,knots:['constant','linear','solid','custom'].includes(state.opacity)?state.knots:PRESETS[min < -1e-6?'difference':'boundary']};
}
function viewLUT(index){const v=viewState(index);return index===0&&state.display!=='value'?makeTransfer(palettes[state.colormap],v.knots):lut;}
function updateReferenceCaption(){const el=$('volumes').querySelector('.reference-scale');if(el){const v=viewState(0);el.textContent=state.display==='value'?'':`Raw reference · own scale ${short(v.min)} to ${short(v.max)}. Other views and the main legend show differences.`;}}
function displayedFields(){return comparisonFields(arrays,state.display);}
function fitWindow(upload=true){
  if(!data)return;$('legend-caption').textContent=state.display==='value'?'Transfer window · shared by all views':'Difference window · views 2+ · view 1 uses its raw scale';fields=displayedFields();let low=Infinity,high=-Infinity;
  for(const volume of (state.display==='value'?fields:fields.slice(1)))if(volume)for(const value of volume)if(Number.isFinite(value)){low=Math.min(low,value);high=Math.max(high,value);}
  if(!Number.isFinite(low)){low=0;high=1;}
  if(state.display==='value'&&data.config.field==='m00'){low=Math.min(0,low);if(low>-1e-6)low=0;}
  else if(state.display==='difference'||(low<0&&high>0)){high=Math.max(Math.abs(low),Math.abs(high));low=-high;}
  if(state.display==='absolute')low=0;
  if(high===low){high=low+Math.max(Math.abs(low)*.01,1e-6);}
  state.min=low;state.max=high;$('tf-min').value=short(low);$('tf-max').value=short(high);
  if(upload){uploadFields();drawEverything();}
}
async function buildVolumes(){
  const current=++generation;renderers.forEach(v=>v?.renderer.dispose());renderers=[];renderStatus=[];const parent=$('volumes');parent.replaceChildren();parent.classList.toggle('single',data.results.length===1);
  const pending=data.results.map(async(result,i)=>{
    const card=create('article','volume-card'),heading=create('div','viewport-title'),label=create('span','view-label');label.append(create('span','view-index',String(i+1).padStart(2,'0')),document.createTextNode(MODELS[result.model][0]));
    const save=create('button','','↓');save.setAttribute('aria-label',`Export 3D viewport ${i+1} PNG`);heading.append(label,save);card.append(heading);parent.append(card);
    if(result.error){card.append(create('div','error-plot',result.error));save.disabled=true;return;}
    const stage=create('div','volume-stage'),canvas=create('canvas','volume-image'),overlay=create('canvas','volume-overlay');canvas.setAttribute('aria-label',`${MODELS[result.model][0]} 3D volume; drag to orbit`);stage.append(canvas,overlay);card.append(stage);
    const info=create('div','viewport-footer');info.append(create('span','',`${Math.round(result.milliseconds)} ms compute · ${data.shape[0]}³${result.engine?' · '+result.engine:''}`),create('span','',i===0?'REFERENCE · RAW FIELD':`VIEW ${i+1}`));card.append(info);card.append(contractElement(3,result.model,data.config));if(i===0)card.append(create('div','reference-scale'));
    const renderer=await VolumeRenderer.create(canvas,text=>{if(current===generation){renderStatus[i]=text;updateGPUStatus();}},text=>{if(current===generation)message(text);});
    if(current!==generation){renderer.dispose();return;}
    renderer.setVolume(fields[i],distance,data.shape[0],data.config.h);if(lut)renderer.setTransfer(viewLUT(i));
    renderers[i]={renderer,overlay,stage};installOrbit(stage);
    save.onclick=async()=>{try{const capture=renderer.draw(viewState(i),true);const blob=capture?await capture:await renderer.readPNG();download(blob,`boundary-3d-${result.model}-${data.config.field}.png`);}catch(e){message(`PNG export failed: ${e.message}`);}};
    requestDraw();
  });await Promise.all(pending);if(!renderers.some(Boolean))$('gpu-status').textContent='No compatible volume models selected';
}
function updateGPUStatus(){const statuses=[...new Set(renderStatus.filter(Boolean))];$('gpu-status').textContent=statuses.join(' · ');const fallback=statuses.length>0&&statuses.every(s=>s.startsWith('CPU'));$('gpu-status').classList.toggle('fallback',fallback);$('lighting').disabled=fallback;}
function uploadFields(){if(!data)return;fields=displayedFields();renderers.forEach((view,i)=>view?.renderer.setVolume(fields[i],distance,data.shape[0],data.config.h));}
function updateTransfer(){if(!palettes)return;lut=makeTransfer(palettes[state.colormap],state.knots);renderers.forEach((view,i)=>view?.renderer.setTransfer(viewLUT(i)));updateReferenceCaption();$('legend-min').textContent=short(state.min);$('legend-max').textContent=short(state.max);$('colorbar').style.background=`linear-gradient(to right,${palettes[state.colormap].filter((_,i)=>i%8===0).map(rgb=>`rgb(${rgb.join(',')})`).join(',')})`;}
function requestDraw(){if(framePending)return;framePending=true;requestAnimationFrame(()=>{framePending=false;renderers.forEach((view,i)=>{if(view){view.renderer.draw(viewState(i));drawOverlay(view);}});});}
function drawEverything(){updateTransfer();drawTransfer();drawSlices();requestDraw();}
function installOrbit(stage){let drag;
  stage.onpointerdown=e=>{drag=[e.clientX,e.clientY];stage.setPointerCapture(e.pointerId);};stage.onpointermove=e=>{if(!drag)return;state.yaw-=(e.clientX-drag[0])*.008;state.pitch=clamp(state.pitch+(e.clientY-drag[1])*.008,-1.45,1.45);drag=[e.clientX,e.clientY];requestDraw();};
  stage.onpointerup=stage.onpointercancel=()=>drag=null;stage.onwheel=e=>{e.preventDefault();state.zoom=clamp(state.zoom*Math.exp(e.deltaY*.001),1.1,5);requestDraw();};
}
function drawOverlay(view){
  const canvas=view.overlay,rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,dpr=Math.min(devicePixelRatio||1,2);canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);if(!state.box)return;
  const camera=cameraFrame(state.yaw,state.pitch,state.zoom),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),project=p=>{const d=p.map((x,i)=>x-camera.eye[i]),z=dot(d,camera.forward);return [w/2+dot(d,camera.right)/(.48*z)*h/2,h/2-dot(d,camera.up)/(.48*z)*h/2];};
  const corners=Array.from({length:8},(_,i)=>[(i&1)? .5:-.5,(i&2)? .5:-.5,(i&4)? .5:-.5]);ctx.strokeStyle='#a7c6d144';ctx.lineWidth=.7;
  for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit)){const a=project(corners[i]),b=project(corners[i|bit]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
  const origin=[36,h-35],colors=['#e59b8d','#92d5ab','#94b8ed'];ctx.font='10px Consolas,monospace';
  for(let axis=0;axis<3;axis++){const dx=camera.right[axis]*24,dy=-camera.up[axis]*24;ctx.strokeStyle=colors[axis];ctx.fillStyle=colors[axis];ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(origin[0]+dx,origin[1]+dy);ctx.stroke();ctx.fillText('xyz'[axis],origin[0]+dx*1.3-3,origin[1]+dy*1.3+3);}
}
function updateOffsets(){if(!data)return;state.offsets=sliceOffsets.map((v,j)=>(v-data[['x','y','z'][j]][0])/(data[['x','y','z'][j]].at(-1)-data[['x','y','z'][j]][0])-.5);}
function buildSliceControls(){const parent=$('slice-controls');parent.replaceChildren();
  for(let axis=0;axis<3;axis++){
    const label=['YZ · x offset','XZ · y offset','XY · z offset'][axis],wrapper=create('div','slice-offset'),top=create('div','range-top'),labelElement=create('span','',label),numeric=create('input'),slider=create('input');
    const grid=data[['x','y','z'][axis]],lo=grid[0],hi=grid.at(-1),step=(hi-lo)/(grid.length-1)/8;
    for(const input of [numeric,slider]){input.type=input===numeric?'number':'range';input.min=lo;input.max=hi;input.step=input===numeric?'any':step;input.value=sliceOffsets[axis];input.setAttribute('aria-label',`${label}${input===numeric?' value':''}`);}
    const change=(value,editing=false)=>{if(value.trim()==='')return;const parsed=+value;if(!Number.isFinite(parsed))return;sliceOffsets[axis]=clamp(parsed,lo,hi);if(!editing)numeric.value=sliceOffsets[axis].toFixed(4);slider.value=sliceOffsets[axis];updateOffsets();drawSlices();};
    slider.oninput=()=>change(slider.value);numeric.oninput=()=>change(numeric.value,true);numeric.onchange=()=>change(numeric.value);
    top.append(labelElement,numeric);wrapper.append(top,slider);parent.append(wrapper);
  }
}
function buildSlices(){const parent=$('slices');parent.replaceChildren();for(const axis of [2,1,0]){const card=create('div','slice-card'),title=create('div','slice-title');title.append(create('b','',['YZ','XZ','XY'][axis]),create('span',''));const canvas=create('canvas');canvas.dataset.axis=axis;canvas.setAttribute('aria-label',`${['YZ','XZ','XY'][axis]} integral slice; click to position crosshair`);card.append(title,canvas);parent.append(card);
    canvas.onpointerdown=e=>{const rect=canvas.getBoundingClientRect(),axes=axis===0?[1,2]:axis===1?[0,2]:[0,1],uv=[clamp((e.clientX-rect.left)/rect.width,0,1),clamp(1-(e.clientY-rect.top)/rect.height,0,1)];
      axes.forEach((j,k)=>{const grid=data[['x','y','z'][j]];sliceOffsets[j]=grid[0]+uv[k]*(grid.at(-1)-grid[0]);});updateOffsets();buildSliceControls();drawSlices();};
  }}
function buildSliceSelector(){const select=$('slice-model');select.replaceChildren();data.results.forEach((r,i)=>{const opt=new Option(MODELS[r.model][0],i);opt.disabled=!!r.error;select.add(opt);});select.value=selectedModel;}
function drawSlices(){if(!data||!lut)return;const n=data.shape[0],values=fields[selectedModel];if(!values)return;
  $('slices').querySelectorAll('canvas').forEach(canvas=>{
    const axis=+canvas.dataset.axis,grid=data[['x','y','z'][axis]],offset=(sliceOffsets[axis]-grid[0])/(grid.at(-1)-grid[0]),slice=sliceValues(values,n,axis,offset),axes=axis===0?[1,2]:axis===1?[0,2]:[0,1];
    const displayState=viewState(selectedModel),sliceLUT=viewLUT(selectedModel);
    const imageCanvas=document.createElement('canvas');imageCanvas.width=n;imageCanvas.height=n;const image=imageCanvas.getContext('2d').createImageData(n,n);
    for(let y=0;y<n;y++)for(let x=0;x<n;x++){const value=slice[y*n+x],index=clamp(Math.round((value-displayState.min)/(displayState.max-displayState.min)*255),0,255)*4,target=((n-1-y)*n+x)*4;image.data.set(Number.isFinite(value)?[sliceLUT[index],sliceLUT[index+1],sliceLUT[index+2],255]:[38,48,58,255],target);}
    imageCanvas.getContext('2d').putImageData(image,0,0);const w=Math.max(120,Math.round(canvas.clientWidth*1.5));canvas.width=w;canvas.height=w;const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=true;const cell=w/(n-1);ctx.drawImage(imageCanvas,-cell/2,-cell/2,w+cell,w+cell);
    ctx.strokeStyle='#ffffff99';ctx.lineWidth=1;ctx.setLineDash([4,5]);ctx.beginPath();const u=state.offsets[axes[0]]+.5,v=state.offsets[axes[1]]+.5;ctx.moveTo(u*w,0);ctx.lineTo(u*w,w);ctx.moveTo(0,(1-v)*w);ctx.lineTo(w,(1-v)*w);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#fff';ctx.font='13px Consolas,monospace';ctx.fillText('xyz'[axes[0]],w-17,w-8);ctx.fillText('xyz'[axes[1]],8,17);
    canvas.parentElement.querySelector('.slice-title span').textContent=`${'xyz'[axis]} = ${sliceOffsets[axis].toFixed(3)}`;
  });
  const uv=state.offsets.map(x=>x+.5),parts=fields.map((f,i)=>f?`${i===0&&state.display!=='value'?'Reference (raw) · ':''}${MODELS[data.results[i].model][0]}: ${fmt(sampleVolume(f,data.shape[0],uv))}`:null).filter(Boolean);
  $('probe').textContent=`x ${sliceOffsets[0].toFixed(3)} · y ${sliceOffsets[1].toFixed(3)} · z ${sliceOffsets[2].toFixed(3)} | ${parts.join(' · ')}`;
}
function showNotes(){const parent=$('model-notes');parent.replaceChildren();data.results.forEach((r,i)=>{const details=create('details'),summary=create('summary','',MODELS[r.model][0]);details.append(summary);(r.error?[r.error]:r.notes).forEach(note=>details.append(create('p','',note)));if(i&&arrays[0]&&arrays[i]){let max=0,sum=0,count=0;for(let j=0;j<arrays[i].length;j++){const d=arrays[i][j]-arrays[0][j];if(Number.isFinite(d)){max=Math.max(max,Math.abs(d));sum+=d*d;count++;}}if(count)details.append(create('p','',`Across raw float32 grid: max |Δ to view 1| ${short(max)}; RMS ${short(Math.sqrt(sum/count))}. These are model differences, not certified numerical errors.`));}parent.append(details);});}
function transferBox(){const canvas=$('transfer-editor'),rect=canvas.getBoundingClientRect();return {canvas,w:rect.width,h:rect.height,left:32,right:rect.width-12,top:10,bottom:rect.height-24};}
function drawTransfer(){if(!palettes)return;const b=transferBox(),dpr=Math.min(devicePixelRatio||1,2);b.canvas.width=b.w*dpr;b.canvas.height=b.h*dpr;const ctx=b.canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);const gradient=ctx.createLinearGradient(b.left,0,b.right,0);for(let i=0;i<=8;i++)gradient.addColorStop(i/8,`rgb(${palettes[state.colormap][Math.min(255,Math.round(i*255/8))].join(',')})`);ctx.globalAlpha=.25;ctx.fillStyle=gradient;ctx.fillRect(b.left,b.top,b.right-b.left,b.bottom-b.top);ctx.globalAlpha=1;ctx.font='9px Consolas,monospace';ctx.fillStyle='#839dad';ctx.strokeStyle='#47617066';ctx.lineWidth=.5;
  for(let k=0;k<=4;k++){const y=b.bottom-(b.bottom-b.top)*k/4;ctx.beginPath();ctx.moveTo(b.left,y);ctx.lineTo(b.right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText((k/4).toFixed(2),b.left-6,y+3);ctx.textAlign='center';ctx.fillText(short(state.min+(state.max-state.min)*k/4),b.left+(b.right-b.left)*k/4,b.h-7);}
  const points=state.knots.map(([x,y])=>[b.left+x*(b.right-b.left),b.bottom-y*(b.bottom-b.top)]);ctx.strokeStyle='#83e5c0';ctx.lineWidth=1.8;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();points.forEach(([x,y])=>{ctx.beginPath();ctx.fillStyle='#d4fff0';ctx.arc(x,y,4,0,2*Math.PI);ctx.fill();});
}
function buildKnots(){const parent=$('transfer-knots');parent.replaceChildren();state.knots.forEach(([position,opacity],i)=>{const group=create('div','knot-fields');for(const [column,title,value]of [[0,'t',position],[1,'α',opacity]]){const label=create('label','',title),input=create('input');input.type='number';input.step=.01;input.min=0;input.max=1;input.value=value.toFixed(2);input.setAttribute('aria-label',`${column?'Opacity':'Position'} knot ${i+1}`);input.disabled=column===0&&(i===0||i===state.knots.length-1);const commit=editing=>{if(!input.value.trim())return;const parsed=+input.value;if(!Number.isFinite(parsed))return;state.knots[i][column]=column===0?clamp(parsed,state.knots[i-1][0]+.005,state.knots[i+1][0]-.005):clamp(parsed,0,1);customTransfer();if(!editing)buildKnots();drawEverything();};input.oninput=()=>commit(true);input.onchange=()=>commit(false);label.append(input);group.append(label);}if(i>0&&i<state.knots.length-1){const remove=create('button','','×');remove.setAttribute('aria-label',`Remove transfer knot ${i+1}`);remove.onclick=()=>{state.knots.splice(i,1);customTransfer();buildKnots();drawEverything();};group.append(remove);}parent.append(group);});}
function customTransfer(){state.opacity='custom';$('opacity-preset').value='custom';}
function sensibleOpacity(){
  const signed=state.display==='difference'||(state.display==='value'&&data&&data.results.some(r=>Number.isFinite(r.min)&&r.min < -1e-6));
  const key=signed?'difference':state.display==='absolute'?'linear':'boundary';state.opacity=key;state.knots=structuredClone(PRESETS[key]);$('opacity-preset').value=key;
  if(signed&&state.colormap==='viridis'){state.colormap='coolwarm';$('colormap').value='coolwarm';}buildKnots();
}
function installTransferEditor(){const canvas=$('transfer-editor');let dragging=-1;const point=e=>{const b=transferBox(),r=canvas.getBoundingClientRect();return [clamp((e.clientX-r.left-b.left)/(b.right-b.left),0,1),clamp((b.bottom-(e.clientY-r.top))/(b.bottom-b.top),0,1)];};const nearest=uv=>{const b=transferBox();return state.knots.findIndex(([x,y])=>Math.hypot((x-uv[0])*(b.right-b.left),(y-uv[1])*(b.bottom-b.top))<12);};
  canvas.onpointerdown=e=>{if(e.button!==0)return;const uv=point(e);dragging=nearest(uv);if(dragging<0){if(state.knots.length>=12)return;if(uv[0]<.005||uv[0]>.995)return;state.knots.push(uv);state.knots.sort((a,b)=>a[0]-b[0]);dragging=state.knots.indexOf(uv);}canvas.setPointerCapture(e.pointerId);customTransfer();drawEverything();};
  canvas.onpointermove=e=>{if(dragging<0)return;const uv=point(e),last=state.knots.length-1;state.knots[dragging]=[dragging===0?0:dragging===last?1:clamp(uv[0],state.knots[dragging-1][0]+.005,state.knots[dragging+1][0]-.005),uv[1]];drawEverything();};
  canvas.onpointerup=canvas.onpointercancel=()=>{dragging=-1;buildKnots();};canvas.oncontextmenu=e=>{e.preventDefault();const i=nearest(point(e));if(i>0&&i<state.knots.length-1){state.knots.splice(i,1);customTransfer();buildKnots();drawEverything();}};
}
function download(blob,name){if(!blob)throw new Error('No rendered image is available');const a=create('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportResult(){return {...data,view:{...state,offsets:undefined,sliceOffsets,selectedModel}};}
function restoreView(view){if(!view||typeof view!=='object')return;const numeric={yaw:[-100,100],pitch:[-1.45,1.45],zoom:[1.1,5],steps:[64,768],density:[.01,4],renderScale:[.4,1]};for(const [key,[lo,hi]]of Object.entries(numeric))if(Number.isFinite(view[key]))state[key]=clamp(view[key],lo,hi);if(Number.isFinite(view.min)&&Number.isFinite(view.max)&&view.max>view.min){state.min=view.min;state.max=view.max;}
  for(const key of ['lighting','band','box'])if(typeof view[key]==='boolean')state[key]=view[key];state.mode='volume';state.planes=[false,false,false];state.sliceOpacity=0;if(palettes[view.colormap])state.colormap=view.colormap;if(['value','difference','absolute'].includes(view.display))state.display=view.display;if(PRESETS[view.opacity]||view.opacity==='custom')state.opacity=view.opacity;
  if(Array.isArray(view.knots)&&view.knots.length>=2&&view.knots.length<=12&&view.knots.every(p=>Array.isArray(p)&&p.length===2&&p.every(x=>Number.isFinite(x)&&x>=0&&x<=1))&&view.knots[0][0]===0&&view.knots.at(-1)[0]===1&&view.knots.every((p,i)=>!i||p[0]>view.knots[i-1][0]))state.knots=structuredClone(view.knots);
  if(Array.isArray(view.sliceOffsets)&&view.sliceOffsets.length===3&&view.sliceOffsets.every(Number.isFinite))sliceOffsets=[...view.sliceOffsets];if(data&&Number.isInteger(view.selectedModel)&&arrays[view.selectedModel])selectedModel=view.selectedModel;
}
function renderingControls(){
  const parent=$('ray-controls');parent.replaceChildren();rangeControl(parent,'steps','Ray samples',64,768,32,state,requestDraw);rangeControl(parent,'density','Optical density',.05,4,.05,state,requestDraw);rangeControl(parent,'renderScale','Render resolution',.4,1,.05,state,requestDraw);
  $('colormap').value=state.colormap;$('display').value=state.display;$('opacity-preset').value=state.opacity;for(const key of ['lighting','band','box'])$(key).checked=state[key];$('tf-min').value=short(state.min);$('tf-max').value=short(state.max);buildKnots();
}
function applySavedView(view){
  if(!view)return;restoreView(view);
  for(let j=0;j<3;j++){const axis=data[['x','y','z'][j]];sliceOffsets[j]=clamp(sliceOffsets[j],axis[0],axis.at(-1));}
  fields=displayedFields();updateOffsets();uploadFields();buildSliceControls();buildSliceSelector();renderingControls();drawEverything();
}
for(const key of ['kernel','field','supersampling','quality','curvature_order','series_order'])$(key).onchange=()=>{config[key]=['supersampling','quality','curvature_order','series_order'].includes(key)?+$(key).value:$(key).value;changed();};
$('geometry').onchange=()=>preset($('geometry').value);$('preset').onchange=()=>preset($('preset').value);$('compute').onclick=evaluate;$('auto').onchange=()=>{clearTimeout(timer);if($('auto').checked)changed();};
$('colormap').onchange=()=>{state.colormap=$('colormap').value;drawEverything();};
$('display').onchange=()=>{state.display=$('display').value;if(data){if(state.opacity!=='custom')sensibleOpacity();fitWindow();if(!arrays[0]&&state.display!=='value')message('A difference volume needs a valid first model as its reference.');}};
for(const key of ['lighting','band','box'])$(key).onchange=()=>{state[key]=$(key).checked;requestDraw();};
$('slice-model').onchange=()=>{selectedModel=+$('slice-model').value;drawSlices();};$('auto-window').onclick=()=>fitWindow();
for(const id of ['tf-min','tf-max'])$(id).onchange=()=>{const min=+$('tf-min').value,max=+$('tf-max').value;if(!Number.isFinite(min)||!Number.isFinite(max)||max<=min){message('Transfer window maximum must be greater than its minimum.');return;}state.min=min;state.max=max;message('');drawEverything();};
function resetTransfer(){state.knots=structuredClone(PRESETS.boundary);state.opacity='boundary';$('opacity-preset').value='boundary';buildKnots();drawEverything();}
$('opacity-preset').onchange=()=>{const key=$('opacity-preset').value;if(PRESETS[key]){state.knots=structuredClone(PRESETS[key]);state.opacity=key;buildKnots();drawEverything();}};$('reset-transfer').onclick=resetTransfer;
$('reset-camera').onclick=()=>{Object.assign(state,{yaw:.68,pitch:.36,zoom:2.3});requestDraw();};
$('save').onclick=()=>download(new Blob([JSON.stringify(exportResult())],{type:'application/json'}),'boundary-volume.json');$('open').onclick=()=>$('file').click();
$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;try{if(file.size>100_000_000)throw new Error('Volume file exceeds 100 MB');const result=JSON.parse(await file.text());validateResult(result);restoreView(result.view);await loadVolume(result);applySavedView(result.view);}catch(e){message(`Could not open volume: ${e.message}`);}finally{$('file').value='';}};
window.addEventListener('resize',()=>{drawEverything();});window.addEventListener('pagehide',()=>{renderers.forEach(v=>v?.renderer.dispose());disposeCompute();});
async function start(){
  try{const response=await fetch('./palettes.json');if(!response.ok)throw new Error();palettes=await response.json();}catch{message('Could not load colormaps. Serve this folder over HTTP.');return;}
  syncControls();renderingControls();installTransferEditor();drawEverything();
  renderTermPanel($('term-equations'),config);
  try{const response=await fetch('./api/info');if(response.ok){const info=await response.json();online=!!info.models3d;}}catch{online=false;}
  connect(online);
  const query=new URLSearchParams(location.search),initialPreset=query.get('preset');
  if(['sphere','box','tetrahedron','curvature','small-ball'].includes(initialPreset)){
    $('preset').value=initialPreset;preset(initialPreset);
    if(Object.hasOwn(FIELD_IDS,query.get('field')))config.field=query.get('field');
    syncControls();await evaluate();return;
  }
  try{const response=await fetch('./example.json');if(response.ok){const result=await response.json();if(result.dimension===3){restoreView(result.view);await loadVolume(result);applySavedView(result.view);status('Saved volume loaded · browser recomputation available');return;}}}catch{}
  await evaluate();
}
$('compute-provider').onchange=()=>{
 if($('compute-provider').value==='python'&&config.models.includes('analytic')){$('compute-provider').value=provider;message('Analytic face-to-edge grids run in the browser float64 worker. Keep a browser provider, or deselect that model to use the Python grid service. The arbitrary-precision Python reference is available separately in mesh3d.');return;}
 provider=$('compute-provider').value;
 config.models=config.models.map(m=>provider==='python'?(m==='quadrature'?'voxel':m):(m==='voxel'?'quadrature':m));
 syncControls();connect(online);changed();
};
start();
