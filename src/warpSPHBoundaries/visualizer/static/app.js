import {renderMethods} from './method-catalogue.js?v=20261022-7';
import {renderTermPanel2d} from './term-panel.js?v=20261022-7';
import {contractElement} from './model-contract.js?v=20261022-7';
import {gridControl} from './grid-control.js?v=20261022-7';
import {computeBrowser2d,disposeCompute2d} from './browser-engine2d.js?v=20261028-webgpu1';
import {MODELS2D} from './browser-compute2d.js?v=20261022-7';
const $ = id => document.getElementById(id);
const COLORS = ['#83e5c0', '#e6b575', '#8ca7ef', '#df94cb'];
const MODELS = {
  strip:MODELS2D.strip,sdf_analytic:MODELS2D.sdf_analytic,curvature_analytic:MODELS2D.curvature_analytic,disk_table:MODELS2D.disk_table,analytic:MODELS2D.analytic,analytic_mesh:MODELS2D.analytic_mesh,disk_series:MODELS2D.disk_series,
  mesh: ['Triangle mesh', 'Exact edge integrals summed over the mesh triangles.'],
  exact: ['Exact polygon', 'Closed-form edge reduction of the represented polygon.'],
  rectangle: ['Built-in rectangle', 'The same exact edge engine, evaluated over four rectangle edges.'],
  sdf: ['SDF · distance only', 'Local half-plane model; geometry error remains near corners and curves.'],
  curvature: ['SDF · curvature table', 'Order-two smooth expansion. Circle only; R/h ≥ 2.'],
  disk: ['Circle reference', 'Angular quadrature with analytic radial integrals; refine to check convergence.'],
};
const modelInfo=model=>(provider!=='python'?MODELS2D:MODELS)[model];
const isBrowserResult=()=>data?.compute_provider==='browser'||['WebGPU fp32','JavaScript float64'].includes(data?.compute_engine)||data?.results.some(r=>r.notes?.some(n=>n.startsWith('Browser ')));
const resultModelInfo=model=>(isBrowserResult()?MODELS2D:MODELS)[model];
const DEFAULT = {geometry:'rectangle',width:2,height:2,radius:2,angle:0,vertices:[[-1,-1],[1,-1],[1,1],[-1,1]],segments:16,kernel:'w4',h:.65,resolution:81,padding:1.15,field:'m00',models:['mesh','rectangle','sdf'],sdf_source:'analytic',sdf_spacing:.12,quadrature:24,curvature_order:2,series_order:4};
let config = structuredClone(DEFAULT), data = null, online = false, busy = false, revision = 0, timer, palettes;
let provider='browser',fieldRanges=[];
const canCompute=()=>provider!=='python'||online;
let probe = [0,0], hover = null, range = [0,1], fields = [], heatmaps = [], requested = false;
let rendering = {colormap:'viridis',display:'value',scale:'shared',vmin:0,vmax:1,cut:'horizontal',boundary:true,mesh:true,band:true,grid:false,onlyband:false,smooth:false};
const create = (tag, cls, text) => {const el=document.createElement(tag); if(cls)el.className=cls; if(text!==undefined)el.textContent=text; return el;};
const fmt = v => v === null || !Number.isFinite(v) ? '—' : (Math.abs(v)>0 && Math.abs(v)<.001 ? v.toExponential(3) : v.toFixed(4));
const short = v => Math.abs(v)<.001 && v!==0 ? v.toExponential(1) : Number(v.toPrecision(3)).toString();
function message(text) {$('message').hidden=!text; $('message').textContent=text;}
function status(text, stale=false) {$('status').textContent=text; $('status').classList.toggle('stale',stale);}
function changed() {
  renderMethods($('method-catalogue'),2,config);
  renderTermPanel2d($('term-equations'),config);
  if($('curvature-label'))$('curvature-label').textContent=`Curvature κ = ${(1/config.radius).toFixed(3)} · κh = ${(config.h/config.radius).toFixed(3)}. Mesh uses an inscribed polygon.`;
  revision++; status('Configuration changed · plot awaits evaluation', true);
  $('compute-note').textContent=canCompute()?'Plot awaits evaluation':'Local Python service is unavailable';
  clearTimeout(timer);
  if (canCompute() && $('auto').checked) timer=setTimeout(evaluate,350);
}
function slider(parent, key, label, min, max, step, unit='', callback) {
  const wrapper=create('label','range'), top=create('span','range-top'), output=create('output');
  const text=create('span','',label); top.append(text,output);
  const input=create('input'); input.type='range'; input.min=min;input.max=max;input.step=step;
  input.value=callback?callback():config[key];input.setAttribute('aria-label',label);
  output.value=`${Number(input.value).toFixed(step<.1?2:step<1?1:0)}${unit}`;
  input.oninput=()=>{const value=+input.value;output.value=`${value.toFixed(step<.1?2:step<1?1:0)}${unit}`;if(callback)callback(value);else config[key]=value;changed();};
  wrapper.append(top,input);parent.append(wrapper);
}
function syncControls() {
  renderMethods($('method-catalogue'),2,config);
  for(const key of ['geometry','kernel','field','sdf_source','curvature_order','series_order']) $(key).value=config[key];
  const g=$('geometry-controls');g.replaceChildren(); $('vertex-controls').replaceChildren();
  if(config.geometry==='rectangle') {slider(g,'width','Width',.2,5,.05);slider(g,'height','Height',.2,5,.05);}
  if(config.geometry==='circle') {
    slider(g,'radius','Radius R',.2,5,.05);slider(g,'segments','Mesh segments',8,64,1);
    const hint=create('p','hint',`Curvature κ = ${(1/config.radius).toFixed(3)} · κh = ${(config.h/config.radius).toFixed(3)}. Mesh uses an inscribed polygon.`);hint.id='curvature-label';g.append(hint);
  } else slider(g,'angle','Rotation',-180,180,1,'°');
  if(config.geometry==='triangle') {
    const parent=$('vertex-controls');
    for(let i=0;i<3;i++) {
      parent.append(create('div','vertex-label',`Vertex ${i+1}`));const row=create('div','vertex-grid');
      for(let j=0;j<2;j++)slider(row,'',`${j?'y':'x'}${i+1}`,-3,3,.05,'',v=>v===undefined?config.vertices[i][j]:(config.vertices[i][j]=v));
      parent.append(row);
    }
    parent.append(create('p','hint','Move each vertex with its sliders. Coordinates rotate around the origin.'));
  }
  const s=$('sampling-controls');s.replaceChildren();
  slider(s,'h','Support radius h',.1,2,.05);gridControl(s,config,{label:'Grid resolution',max:513,sliderMax:193,changed});slider(s,'padding','View padding',1,2,.05,'h');
  const a=$('sdf-controls');a.replaceChildren();slider(a,'sdf_spacing','SDF lattice spacing',.03,.5,.01,'h');slider(a,'quadrature','Reference Gauss nodes',8,64,8);
  modelControls();
}
function modelControls() {
  const parent=$('model-controls');parent.replaceChildren();
  config.models.forEach((model,i)=>{
    const card=create('div','model-control'), heading=create('div','eyebrow'), label=create('span');
    const marker=create('i','model-marker');marker.style.background=COLORS[i];label.append(marker,document.createTextNode(`VIEW ${String(i+1).padStart(2,'0')}`));
    const remove=create('button','','×');remove.title='Remove viewport';remove.setAttribute('aria-label',`Remove viewport ${i+1}`);remove.disabled=config.models.length===1;
    remove.onclick=()=>{config.models.splice(i,1);modelControls();changed();};heading.append(label,remove);
    const select=create('select');select.setAttribute('aria-label',`Model in viewport ${i+1}`);
    Object.entries(provider==='python'?Object.fromEntries(Object.entries(MODELS).filter(([k])=>!['analytic','analytic_mesh','disk_series','disk_table','sdf_analytic','curvature_analytic','strip'].includes(k))):MODELS2D).forEach(([key,[title]])=>{const opt=new Option(modelInfo(key)[0],key);opt.disabled=config.models.includes(key)&&key!==model;select.add(opt);});select.value=model;
    select.onchange=()=>{config.models[i]=select.value;modelControls();changed();};
    card.append(heading,select,create('p','',(provider!=='python'?MODELS2D:MODELS)[model][1]));parent.append(card);
  });$('add').disabled=config.models.length>=4;
}
function preset(name) {
  const keep={kernel:config.kernel,resolution:config.resolution};config={...structuredClone(DEFAULT),...keep};probe=[0,0];
  const notes={square:'One square, three ways to evaluate the solid integral.',rectangle:'Thin walls expose interactions missed by a single local plane.',triangle:'Adjust the vertices to compare an exact triangle with its SDF model.',circle:'Compare polygonization, a local plane, and smooth curvature against a circle reference.'};
  if(name==='rectangle'){config.width=3;config.height=.65;config.h=.8;}
  if(name==='triangle'){config.geometry='triangle';config.vertices=[[-1,-.9],[1.1,-.7],[-.3,1.1],[-1,1]];config.models=['analytic','exact','sdf'];}
  if(name==='circle'){config.geometry='circle';config.radius=2;config.h=.65;config.models=['disk','disk_table','sdf','curvature'];probe=[2,0];}
  if(name==='small-disk'){config.geometry='circle';config.radius=.2;config.h=1.2;config.models=['disk','disk_table','disk_series'];probe=[.6,0];}
  $('preset-note').textContent=notes[name]??'Compare the original disk, its lookup approximation and mean-value orders.';syncControls();changed();
}
function setConnection(value) {
  online=value;const available=canCompute();$('connection').textContent=provider==='browser'?'Automatic · GPU fp32 / CPU float64':provider==='gpu'?'WebGPU · fp32':provider==='worker'?'Browser · float64':value?'Python · local':'Python unavailable';$('connection').classList.toggle('offline',!available);$('compute').disabled=!available;
}
async function evaluate() {
  clearTimeout(timer);
  if(!canCompute())return;
  if(busy){requested=true;return;}
  busy=true;requested=false;const version=revision;
  $('compute').disabled=true;$('compute').textContent='Computing…';$('viewports').classList.add('busy');$('compute-note').textContent=provider==='python'?'Working · local Python':provider==='worker'?'Working · float64 worker':'Working · WebGPU fp32 / float64';status(provider!=='python'?'Evaluating integrals in the browser…':'Evaluating on local Python backend…');message('');
  try{
    let result;if(provider!=='python')result=await computeBrowser2d(structuredClone(config),note=>{if(version===revision)status(note);},provider==='browser'?'auto':provider);else{const response=await fetch('./api/compute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});result=await response.json();if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);}
    if(version===revision){data=result;data.compute_provider=provider==='python'?'python':'browser';showData();status(`Computed in ${(result.milliseconds/1000).toFixed(2)} s · ${result.results.filter(r=>!r.error).length} models`);$('compute-note').textContent=provider!=='python'?`Up to date · ${result.compute_engine}`:'Up to date · local Python · CPU';if(provider!=='python'){$('connection').textContent=result.compute_engine==='WebGPU fp32'?'WebGPU · fp32':'Browser · float64';$('connection').classList.remove('offline');}if(result.compute_fallback)message(`WebGPU unavailable: ${result.compute_fallback}. Evaluated with the JavaScript float64 worker.`);}
    else status('Configuration changed · waiting for latest evaluation',true);
  }catch(error){message(error.message);status('Evaluation failed',true);$('compute-note').textContent='Evaluation failed · previous plot retained';if(provider==='gpu'){$('connection').textContent='WebGPU · evaluation failed';$('connection').classList.add('offline');}}
  finally{
    busy=false;$('compute').disabled=!canCompute();$('compute').innerHTML='Evaluate integrals <span>↗</span>';$('viewports').classList.remove('busy');
    if(requested || (version!==revision && $('auto').checked))evaluate();
  }
}
function color(t) {const lut=palettes[rendering.colormap];return lut[Math.max(0,Math.min(255,Math.round(t*255)))];}
function makeFields() {
  const reference=data.results[0].values;
  fields=data.results.map((result,i)=>{
    if(!result.values)return null;
    if(rendering.display==='value'||i===0)return result.values;
    return result.values.map((v,j)=>{if(v===null||!reference||reference[j]===null)return null;const d=v-reference[j];return rendering.display==='absolute'?Math.abs(d):d;});
  });
  let low=Infinity,high=-Infinity;
  (rendering.display==='value'?fields:fields.slice(1)).forEach(f=>f?.forEach((v,j)=>{if(v!==null&&(!rendering.onlyband||Math.abs(data.distance[j])<=data.config.h)){low=Math.min(low,v);high=Math.max(high,v);}}));
  if(!Number.isFinite(low)){low=0;high=1;}
  if(rendering.display==='value'&&['m00','m20','m02','grad_norm'].includes(data.config.field)&&low>=-1e-9*Math.max(1,Math.abs(high)))low=0;
  if(rendering.display==='difference'||(rendering.display==='value'&&!['m00','grad_norm'].includes(data.config.field)&&low<0&&high>0)){high=Math.max(Math.abs(low),Math.abs(high));low=-high;}
  if(rendering.display==='absolute')low=0;
  if(rendering.display==='value'&&data.config.field==='m00'){low=Math.min(0,low);high=Math.max(1,high);}
  if(rendering.scale==='fixed'&&Number.isFinite(rendering.vmin)&&Number.isFinite(rendering.vmax)&&rendering.vmax>rendering.vmin){low=rendering.vmin;high=rendering.vmax;}
  if(high===low){const pad=Math.max(1e-15,Math.abs(low)*.01);low-=pad;high+=pad;}
  range=[low,high];$('scale-min').textContent=short(low);$('scale-max').textContent=short(high);
  $('scale-kind').textContent=`· ${rendering.display==='value'?data.config.field:'Δ '+data.config.field}`;
  const gradient=palettes[rendering.colormap].filter((_,i)=>i%8===0).map(rgb=>`rgb(${rgb.join(',')})`);$('colorbar').style.background=`linear-gradient(to right,${gradient.join(',')})`;
  fieldRanges=fields.map((f,i)=>{if(i!==0||rendering.display==='value'||!f)return range;let lo=Infinity,hi=-Infinity;for(const v of f)if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}if(!Number.isFinite(lo))return [0,1];if(hi===lo)hi=lo+1e-6;return [lo,hi];});
  $('reference-hint').textContent=rendering.display==='value'?'Shared color scale':'Differences · reference view keeps its raw field and own scale';
  heatmaps=fields.map((f,modelIndex)=>{
    if(!f)return null;const n=data.x.length, canvas=document.createElement('canvas');canvas.width=n;canvas.height=n;const ctx=canvas.getContext('2d'),img=ctx.createImageData(n,n);
    for(let j=0;j<n;j++)for(let i=0;i<n;i++){
      const source=j*n+i,target=((n-1-j)*n+i)*4,v=f[source];
      const rgb=(v===null||(rendering.onlyband&&Math.abs(data.distance[source])>data.config.h))?[27,37,46]:color((v-fieldRanges[modelIndex][0])/(fieldRanges[modelIndex][1]-fieldRanges[modelIndex][0]));
      img.data.set([...rgb,255],target);
    }ctx.putImageData(img,0,0);return canvas;
  });
}
function showData() {
  renderTermPanel2d($('term-equations'),data.config);
  makeFields();const parent=$('viewports');parent.replaceChildren();parent.className='viewports '+(['','one','two','','four'][data.results.length]||'');
  const legend=$('line-legend');legend.replaceChildren();
  data.results.forEach((result,i)=>{
    const card=create('article','viewport'),title=create('div','viewport-title'),name=create('span','view-label');
    name.append(create('span','view-index',String(i+1).padStart(2,'0')),document.createTextNode(resultModelInfo(result.model)[0]));title.append(name);
    const download=create('button','','↓');download.title='Export viewport PNG';download.setAttribute('aria-label',`Export viewport ${i+1} as PNG`);download.disabled=!!result.error;title.append(download);card.append(title);
    if(result.error)card.append(create('div','error-plot',result.error));
    else{
      const canvas=create('canvas');canvas.setAttribute('aria-label',`${resultModelInfo(result.model)[0]} integral field; click to place probe`);canvas.dataset.index=i;
      canvas.onpointermove=e=>{hover=canvasPoint(e,canvas);drawAll();};canvas.onpointerleave=()=>{hover=null;drawAll();};
      canvas.onpointerdown=e=>{probe=canvasPoint(e,canvas);hover=null;drawAll();};card.append(canvas);
      download.onclick=()=>canvas.toBlob(blob=>downloadBlob(blob,`boundary-${result.model}-${data.config.field}.png`));
      const item=create('span'),line=create('i');line.style.background=COLORS[i];item.append(line,document.createTextNode(resultModelInfo(result.model)[0]));legend.append(item);
    }
    const bottom=create('div','viewport-footer');bottom.append(create('span','',result.error?'Unavailable':`${result.milliseconds.toFixed(0)} ms · ${data.x.length} × ${data.y.length}${result.engine?' · '+result.engine:''}`),create('span','',i===0?'REFERENCE · RAW FIELD':`VIEW ${i+1}`));card.append(bottom);card.append(contractElement(2,result.model,data.config,isBrowserResult()));if(i===0&&rendering.display!=='value')card.append(create('div','reference-scale',`Raw reference · ${short(fieldRanges[0][0])} to ${short(fieldRanges[0][1])}`));parent.append(card);
  });
  $('export').disabled=false;$('footer-grid').textContent=`${data.x.length} × ${data.y.length} grid · h = ${data.config.h} · ${data.config.kernel} · ${data.config.field}`;
  const notes=$('model-notes');notes.replaceChildren();
  data.results.forEach((r,i)=>{notes.append(create('h3','',resultModelInfo(r.model)[0]));(r.error?[r.error]:r.notes).forEach(note=>notes.append(create('p','',note)));if(r.missing)notes.append(create('p','',`${r.missing} unavailable samples; shown in gray.`));
    if(i&&r.values&&data.results[0].values){let max=0,sum=0,count=0;r.values.forEach((v,j)=>{const ref=data.results[0].values[j];if(v!==null&&ref!==null){const d=v-ref;max=Math.max(max,Math.abs(d));sum+=d*d;count++;}});if(count)notes.append(create('p','',`To view 1: max |Δ| ${short(max)} · RMS ${short(Math.sqrt(sum/count))}`));}
  });
  if(rendering.display!=='value'&&!data.results[0].values)message('Difference plots require a valid first viewport. Select a compatible reference model.');
  drawAll();
}
function canvasSize(canvas) {
  const box=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),w=box.width,h=box.height;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,w,h};
}
function plotBox(w,h){const size=Math.min(w-45,h-45);return {left:34,top:12,size};}
function canvasPoint(event,canvas) {
  const r=canvas.getBoundingClientRect(),b=plotBox(r.width,r.height),tx=Math.max(0,Math.min(1,(event.clientX-r.left-b.left)/b.size)),ty=Math.max(0,Math.min(1,(event.clientY-r.top-b.top)/b.size));
  return [data.x[0]+tx*(data.x.at(-1)-data.x[0]),data.y.at(-1)-ty*(data.y.at(-1)-data.y[0])];
}
function trace(ctx,points,tx,ty,close=false){ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(tx(x),ty(y)):ctx.moveTo(tx(x),ty(y)));if(close)ctx.closePath();ctx.stroke();}
function contour(ctx,level,tx,ty) {
  // Marching triangles keeps d=±h correct for both curved and cornered boundaries.
  const n=data.x.length,d=data.distance;ctx.beginPath();
  for(let j=0;j<n-1;j++)for(let i=0;i<n-1;i++){
    const indices=[j*n+i,j*n+i+1,(j+1)*n+i+1,(j+1)*n+i];
    for(const tri of [[0,1,2],[0,2,3]]){
      const hits=[];
      for(let k=0;k<3;k++){
        const a=indices[tri[k]],b=indices[tri[(k+1)%3]],va=d[a],vb=d[b];
        if(va===null||vb===null||(va<level)===(vb<level))continue;
        const t=(level-va)/(vb-va);hits.push([data.x[a%n]+t*(data.x[b%n]-data.x[a%n]),data.y[Math.floor(a/n)]+t*(data.y[Math.floor(b/n)]-data.y[Math.floor(a/n)])]);
      }
      if(hits.length===2){ctx.moveTo(tx(hits[0][0]),ty(hits[0][1]));ctx.lineTo(tx(hits[1][0]),ty(hits[1][1]));}
    }
  }ctx.stroke();
}
function drawViewport(canvas,i) {
  const {ctx,w,h}=canvasSize(canvas),b=plotBox(w,h),x0=data.x[0],x1=data.x.at(-1),y0=data.y[0],y1=data.y.at(-1);
  const tx=x=>b.left+(x-x0)/(x1-x0)*b.size,ty=y=>b.top+(y1-y)/(y1-y0)*b.size;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#0f171e';ctx.fillRect(0,0,w,h);
  ctx.save();ctx.beginPath();ctx.rect(b.left,b.top,b.size,b.size);ctx.clip();
  const cell=b.size/(data.x.length-1);
  ctx.imageSmoothingEnabled=rendering.smooth;ctx.drawImage(heatmaps[i],b.left-cell/2,b.top-cell/2,b.size+cell,b.size+cell);
  if(rendering.grid){ctx.strokeStyle='#ffffff19';ctx.lineWidth=.6;for(let t=0;t<=4;t++){const x=b.left+t*b.size/4,y=b.top+t*b.size/4;ctx.beginPath();ctx.moveTo(x,b.top);ctx.lineTo(x,b.top+b.size);ctx.moveTo(b.left,y);ctx.lineTo(b.left+b.size,y);ctx.stroke();}}
  if(rendering.band){ctx.strokeStyle='#f1f5ef9c';ctx.lineWidth=.8;ctx.setLineDash([3,4]);contour(ctx,data.config.h,tx,ty);contour(ctx,-data.config.h,tx,ty);ctx.setLineDash([]);}
  if(rendering.mesh&&data.results[i].model==='mesh'){ctx.strokeStyle='#f9f3d284';ctx.lineWidth=.75;data.triangles.forEach(triangle=>trace(ctx,triangle,tx,ty,true));}
  if(rendering.boundary){ctx.strokeStyle='#f1f7f2';ctx.lineWidth=1.3;
    const vertices=data.config.geometry==='circle'?Array.from({length:181},(_,j)=>[data.config.radius*Math.cos(j*Math.PI/90),data.config.radius*Math.sin(j*Math.PI/90)]):data.vertices;trace(ctx,vertices,tx,ty,true);
  }
  ctx.strokeStyle='#fff9';ctx.lineWidth=.75;ctx.setLineDash([4,4]);ctx.beginPath();if(rendering.cut==='horizontal'){ctx.moveTo(b.left,ty(probe[1]));ctx.lineTo(b.left+b.size,ty(probe[1]));}else{ctx.moveTo(tx(probe[0]),b.top);ctx.lineTo(tx(probe[0]),b.top+b.size);}ctx.stroke();ctx.setLineDash([]);
  const point=hover||probe;ctx.beginPath();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.arc(tx(point[0]),ty(point[1]),4,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.fillStyle='#fff';ctx.arc(tx(point[0]),ty(point[1]),1,0,Math.PI*2);ctx.fill();ctx.restore();
  ctx.strokeStyle='#536b7b';ctx.lineWidth=.5;ctx.strokeRect(b.left,b.top,b.size,b.size);ctx.fillStyle='#7890a2';ctx.font='9px Consolas,monospace';ctx.textAlign='center';
  for(let t=0;t<=2;t++){ctx.fillText(short(x0+(x1-x0)*t/2),b.left+b.size*t/2,b.top+b.size+14);ctx.textAlign='right';ctx.fillText(short(y1-(y1-y0)*t/2),b.left-6,b.top+b.size*t/2+3);ctx.textAlign='center';}
  ctx.fillText('x',b.left+b.size/2,h-3);ctx.save();ctx.translate(9,b.top+b.size/2);ctx.rotate(-Math.PI/2);ctx.fillText('y',0,0);ctx.restore();
}
function interpolate(f,point) {
  if(!f)return null;const n=data.x.length,u=(point[0]-data.x[0])/(data.x.at(-1)-data.x[0])*(n-1),v=(point[1]-data.y[0])/(data.y.at(-1)-data.y[0])*(n-1);
  if(u<0||u>n-1||v<0||v>n-1)return null;
  const i=Math.min(n-2,Math.floor(u)),j=Math.min(n-2,Math.floor(v)),a=u-i,b=v-j;
  const values=[f[j*n+i],f[j*n+i+1],f[(j+1)*n+i],f[(j+1)*n+i+1]],weights=[(1-a)*(1-b),a*(1-b),(1-a)*b,a*b];
  if(values.some((x,k)=>x===null&&weights[k]>1e-12))return null;return values.reduce((sum,x,k)=>sum+(x??0)*weights[k],0);
}
function drawCross() {
  const {ctx,w,h}=canvasSize($('cross')),b={left:47,right:w-14,top:10,bottom:h-28},vertical=rendering.cut==='vertical',axis=vertical?data.y:data.x;
  const lines=fields.map(f=>axis.map(t=>interpolate(f,vertical?[probe[0],t]:[t,probe[1]])));
  let lo=Infinity,hi=-Infinity;lines.forEach(line=>line.forEach(v=>{if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
  if(!Number.isFinite(lo)){lo=0;hi=1;}const pad=Math.max((hi-lo)*.12,Math.max(Math.abs(hi),1e-10)*.06);lo-=pad;hi+=pad;
  const tx=x=>b.left+(x-axis[0])/(axis.at(-1)-axis[0])*(b.right-b.left),ty=y=>b.bottom-(y-lo)/(hi-lo)*(b.bottom-b.top);
  ctx.clearRect(0,0,w,h);ctx.font='9px Consolas,monospace';ctx.lineWidth=.5;
  for(let k=0;k<=4;k++){const y=b.top+(b.bottom-b.top)*k/4;ctx.strokeStyle='#2b3b47';ctx.beginPath();ctx.moveTo(b.left,y);ctx.lineTo(b.right,y);ctx.stroke();ctx.fillStyle='#7a93a5';ctx.textAlign='right';ctx.fillText(short(hi-(hi-lo)*k/4),b.left-8,y+3);const x=b.left+(b.right-b.left)*k/4;ctx.textAlign='center';ctx.fillText(short(axis[0]+(axis.at(-1)-axis[0])*k/4),x,h-10);}
  lines.forEach((line,i)=>{ctx.strokeStyle=COLORS[i];ctx.lineWidth=1.5;ctx.setLineDash(i===0?[]:i===1?[6,3]:i===2?[2,3]:[8,3,2,3]);ctx.beginPath();let open=false;line.forEach((v,j)=>{if(v===null){open=false;return;}if(open)ctx.lineTo(tx(axis[j]),ty(v));else ctx.moveTo(tx(axis[j]),ty(v));open=true;});ctx.stroke();});ctx.setLineDash([]);
  $('slice-label').textContent=`${vertical?'x':'y'} = ${probe[vertical?0:1].toFixed(3)} · ${rendering.display==='value'?data.config.field:'Δ '+data.config.field+' · view 1 raw'} · physical coordinates`;
}
function updateProbe() {
  const point=hover||probe;$('probe-coordinates').textContent=`x ${point[0].toFixed(3)}   y ${point[1].toFixed(3)}`;
  const parent=$('probe-values');parent.replaceChildren();data.results.forEach((result,i)=>{const row=create('div','probe-row'),label=create('span','',resultModelInfo(result.model)[0]),value=create('b','',fmt(interpolate(fields[i],point)));label.style.color=COLORS[i];row.append(label,value);parent.append(row);});
}
function drawAll(){if(!data)return;$('viewports').querySelectorAll('canvas').forEach(canvas=>drawViewport(canvas,+canvas.dataset.index));drawCross();updateProbe();}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob),a=create('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function loadResult(result) {
  if(result.schema_version!==1||result.dimension!==2||!Array.isArray(result.x)||!Array.isArray(result.y)||result.x.length!==result.y.length||result.x.length<17||result.x.length>513)throw new Error('Unsupported result format or grid size');
  const n=result.x.length,finite=x=>typeof x==='number'&&Number.isFinite(x),nullable=x=>x===null||finite(x);
  if(!result.x.every(finite)||!result.y.every(finite)||result.x.some((v,i)=>i&&v<=result.x[i-1])||result.y.some((v,i)=>i&&v<=result.y[i-1]))throw new Error('Invalid coordinate grid');
  if(!Array.isArray(result.results)||result.results.length<1||result.results.length>4||!Array.isArray(result.distance)||result.distance.length!==n*n||!result.distance.every(nullable))throw new Error('Invalid result arrays');
  result.results.forEach(r=>{if(!MODELS[r.model]||(!r.error&&(!Array.isArray(r.values)||r.values.length!==n*n||!r.values.every(nullable)||!Array.isArray(r.notes)||!finite(r.milliseconds))))throw new Error('Invalid model result');});
  if(!result.config||!finite(result.config.h)||result.config.h<=0||!['rectangle','triangle','circle'].includes(result.config.geometry)||!Array.isArray(result.vertices)||!Array.isArray(result.triangles))throw new Error('Missing geometry configuration');
  const validPoint=p=>Array.isArray(p)&&p.length===2&&p.every(finite);
  if(!result.vertices.every(validPoint)||result.vertices.length>64||result.triangles.length>64||!result.triangles.every(t=>Array.isArray(t)&&t.length===3&&t.every(validPoint)))throw new Error('Invalid geometry arrays');
  revision++;data=result;config={...structuredClone(DEFAULT),...result.config,models:result.results.map(r=>r.model)};probe=[(result.x[0]+result.x.at(-1))/2,(result.y[0]+result.y.at(-1))/2];syncControls();showData();message('');status('Saved result loaded · browser recomputation available');
}
for(const key of ['geometry','kernel','field','sdf_source','curvature_order','series_order'])$(key).onchange=()=>{config[key]=['curvature_order','series_order'].includes(key)?+$(key).value:$(key).value;if(key==='geometry')syncControls();changed();};
$('preset').onchange=()=>preset($('preset').value);$('compute').onclick=evaluate;
$('auto').onchange=()=>{if($('auto').checked&&canCompute())changed();};
$('add').onclick=()=>{const available=Object.keys(provider==='python'?MODELS:MODELS2D).find(key=>!config.models.includes(key));if(available&&config.models.length<4){config.models.push(available);modelControls();changed();}};
$('compute-provider').onchange=()=>{if($('compute-provider').value==='python'&&config.models.some(m=>['analytic','analytic_mesh','disk_series','disk_table','sdf_analytic','curvature_analytic','strip'].includes(m))){$('compute-provider').value=provider;message('The selected analytic, lookup and series models run in the browser. Deselect those models to use the local Python grid service.');return;}provider=$('compute-provider').value;setConnection(online);modelControls();changed();};
window.addEventListener('pagehide',disposeCompute2d);
for(const key of Object.keys(rendering))$(key).onchange=()=>{rendering[key]=$(key).type==='checkbox'?$(key).checked:$(key).type==='number'?+$(key).value:$(key).value;$('limits').hidden=rendering.scale!=='fixed';if(data){showData();}if(key==='scale'&&rendering.vmax<=rendering.vmin)message('Maximum color limit must be greater than minimum.');};
$('export').onclick=()=>downloadBlob(new Blob([JSON.stringify(data)],{type:'application/json'}),'boundary-experiment.json');
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;try{if(file.size>100_000_000)throw new Error('Result file exceeds 100 MB');loadResult(JSON.parse(await file.text()));}catch(e){message(`Could not open result: ${e.message}`);}finally{$('file').value='';}};
$('share').onclick=async()=>{const url=new URL(location.href);url.hash=encodeURIComponent(JSON.stringify(config));try{await navigator.clipboard.writeText(url.toString());$('share').textContent='Setup link copied';setTimeout(()=>$('share').textContent='Copy setup link ↗',2000);}catch{history.replaceState(null,'',url);message('The address bar now contains this setup. Copy its URL to share the configuration.');}};
new ResizeObserver(()=>requestAnimationFrame(drawAll)).observe($('viewports'));window.addEventListener('resize',drawAll);
async function start(){
  try{palettes=await(await fetch('./palettes.json')).json();}catch{message('Could not load colormaps. Serve this folder over HTTP instead of opening a file URL.');return;}
  if(location.hash){try{const parsed=JSON.parse(decodeURIComponent(location.hash.slice(1)));if(parsed&&typeof parsed==='object'&&!Array.isArray(parsed)){config={...structuredClone(DEFAULT),...parsed};if(!Array.isArray(config.models)||!config.models.length||config.models.length>4||config.models.some(m=>!MODELS[m]))throw new Error();}}catch{config=structuredClone(DEFAULT);message('Invalid setup link; using the square preset.');}}
  const initialPreset=new URLSearchParams(location.search).get('preset');if(['square','rectangle','triangle','circle','small-disk'].includes(initialPreset)){$('preset').value=initialPreset;if(!location.hash)preset(initialPreset);}
  syncControls();
  try{const response=await fetch('./api/info');if(response.ok){const info=await response.json();online=!!info.models;}}catch{online=false;}
  setConnection(online);
  if(!location.hash&&!new URLSearchParams(location.search).has('2d'))try{const response=await fetch('./example.json');if(response.ok){const result=await response.json();if(result.dimension===3){location.replace('./volume.html');return;}if(result.dimension===2){loadResult(result);return;}}}catch{}
  await evaluate();
}
start();
