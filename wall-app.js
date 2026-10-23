import {computeWallInBrowser,cancelWall} from './wall-engine.js';
const $ = id => document.getElementById(id);
const form = $('controls');
const vectorKeys = new Set(['center', 'fluid', 'skeleton', 'fluid_gradient', 'skeleton_gradient']);
const regionNames = {solid:'Impermeable', porous:'Permeable', total:'Solid + porous', free:'Free-fluid complement'};
let data = null, probe = 0, geometry = 'wall', source = 'snapshot';

function config() {
  const c = {geometry};
  for (const el of form.elements) {
    if (!el.name) continue;
    let value = el.type === 'checkbox' ? el.checked : el.value;
    if (vectorKeys.has(el.name)) {
      value = value.split(',').map(Number);
      if (el.name.endsWith('gradient')) value = [value.slice(0,3), value.slice(3,6), value.slice(6,9)];
    } else if (el.type === 'number' || ['dimension','degree'].includes(el.name)) value = Number(value);
    c[el.name] = value;
  }
  if ($('provider').value === 'webgpu') Object.assign(c,{precision:'float32',backend:'numerical',preparation:'host',search:'host'});
  return c;
}

function loadConfig(c) {
  geometry = c.geometry || 'wall';
  for (const el of form.elements) {
    if (!el.name || c[el.name] === undefined) continue;
    const value = c[el.name];
    if (el.type === 'checkbox') el.checked = value;
    else el.value = Array.isArray(value) ? value.flat().join(',') : value;
  }
}

function field(region) {
  const [key,...indices] = $('quantity').value.split(':');
  return data.terms[region]?.[key]?.map(v => indices.reduce((a,i) => a?.[Number(i)],v));
}

function frame() {
  const canvas = $('map'), [xs,ys] = data.axes;
  const dx = xs.at(-1)-xs[0], dy = ys.at(-1)-ys[0];
  const scale = Math.min((canvas.width-65)/dx, (canvas.height-55)/dy);
  const width = dx*scale, height = dy*scale;
  const left = 40+(canvas.width-65-width)/2, top = 25+(canvas.height-55-height)/2;
  return {left, top, right:left+width, bottom:top+height, width, height, scale};
}

function project(q) {
  const f = frame(), axes = data.plot_axes || [0,1];
  return [f.left+(q[axes[0]]-data.axes[0][0])*f.scale,
          f.bottom-(q[axes[1]]-data.axes[1][0])*f.scale];
}

const finite = x => typeof x === 'number' && Number.isFinite(x);
const pretty = x => finite(x) ? x.toExponential(6) : 'Unavailable';
function color(t) {
  t = Math.max(0,Math.min(1,t));
  const a = t < .5 ? [48,71,109] : [244,244,233];
  const b = t < .5 ? [244,244,233] : [187,98,71], s = t < .5 ? 2*t : 2*t-1;
  return `rgb(${a.map((v,i) => Math.round(v+s*(b[i]-v))).join(',')})`;
}

function drawGeometry(ctx) {
  const selected = new Set(data.selected[probe]);
  ctx.save();
  if ($('overlay').checked) data.elements.forEach((vertices,i) => {
    ctx.strokeStyle = data.regions[i] === 'solid' ? '#a35431' : '#137b7f';
    ctx.lineWidth = selected.has(i) ? 1.7 : .6;
    ctx.globalAlpha = selected.has(i) ? .9 : .35;
    const edges = vertices.length === 3 ? [[0,1],[1,2],[2,0]] : [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
    for (const [a,b] of edges) {
      ctx.beginPath(); ctx.moveTo(...project(vertices[a])); ctx.lineTo(...project(vertices[b])); ctx.stroke();
    }
  });
  ctx.restore();
  const [x,y] = project(data.queries[probe]);
  ctx.strokeStyle = '#243b44'; ctx.lineWidth = 1.4; ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.arc(x,y,data.config.h*frame().scale,0,Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle = '#243b44';
  ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill();
}

function drawAxes(ctx) {
  const f = frame(), axes = data.plot_axes || [0,1];
  ctx.strokeStyle = '#c7d5cc'; ctx.lineWidth = .7;
  ctx.strokeRect(f.left,f.top,f.width,f.height);
  ctx.fillStyle = '#718178'; ctx.font = '11px system-ui';
  for (let i=0;i<=4;i++) {
    const t = i/4, x = f.left+t*f.width, y = f.bottom-t*f.height;
    ctx.fillText((data.axes[0][0]+t*(data.axes[0].at(-1)-data.axes[0][0])).toFixed(2),x-10,f.bottom+17);
    ctx.fillText((data.axes[1][0]+t*(data.axes[1].at(-1)-data.axes[1][0])).toFixed(2),f.left-37,y+3);
  }
  ctx.fillText('xyz'[axes[0]],f.right+8,f.bottom+4);
  ctx.fillText('xyz'[axes[1]],f.left-4,f.top-10);
  if (data.dimension === 3) ctx.fillText(`${data.config.slice_plane.toUpperCase()} slice offset=${data.config.slice}; tetrahedron edges projected`,f.left,f.top-10);
}

function render() {
  if (!data) return;
  const canvas = $('map'), ctx = canvas.getContext('2d'), f = frame();
  const values = field($('region').value), valid = values?.filter(finite) || [];
  const limit = Math.max(...valid.map(Math.abs),1e-12);
  let lo = -limit, hi = limit;
  if ($('quantity').value === 'occupancy') {lo=0; hi=1;}
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const cw = f.width/(data.shape[1]-1), ch = f.height/(data.shape[0]-1);
  if (values) values.forEach((value,i) => {
    if (!finite(value)) return;
    const [x,y] = project(data.queries[i]);
    const left = Math.max(f.left,x-cw/2), right = Math.min(f.right,x+cw/2);
    const top = Math.max(f.top,y-ch/2), bottom = Math.min(f.bottom,y+ch/2);
    ctx.fillStyle = color((value-lo)/(hi-lo)); ctx.fillRect(left,top,right-left,bottom-top);
  });
  drawGeometry(ctx); drawAxes(ctx);
  if (!valid.length) {
    ctx.fillStyle = '#526b60'; ctx.font = '13px system-ui';
    ctx.fillText('This quantity is unavailable for the selected region.',f.left+12,f.top+22);
  }
  $('min').textContent = valid.length ? pretty(lo) : '—';
  $('max').textContent = valid.length ? pretty(hi) : 'Unavailable';
  const q = data.queries[probe], selected = new Set(data.selected[probe]);
  $('probe').textContent = `q = (${q.map(v=>v.toFixed(4)).join(', ')}) · ${selected.size} selected elements · h = ${data.config.h}`;
  $('values').replaceChildren(...['solid','porous','total','free'].map(r => {
    const tr = document.createElement('tr'), label = document.createElement('td'), value = document.createElement('td');
    label.textContent = regionNames[r]; value.textContent = pretty(field(r)?.[probe]);
    tr.append(label,value); return tr;
  }));
  const closure = ['solid','porous','free'].reduce((s,r)=>s+data.terms[r].occupancy[probe],0);
  $('closure').textContent = `Occupancy closure: solid + porous + free = ${closure.toPrecision(12)}. ${data.dimension===2?'Area measure.':'Volume measure; this is a slice of a 3D integral.'}`;
  const quantity = $('quantity').value.split(':')[0];
  $('quantity-note').textContent = `Units: ${data.contract.units?.[quantity]??'see saved contract'}. Moments: ${data.contract.precision}; contractions: ${data.contract.contraction_precision??'host float64'}; geometry: ${data.contract.geometry_precision??'float64'}. Equal physical scale on both plot axes.`;
  $('backend').textContent = `${source==='snapshot'?'Snapshot · ':''}${data.contract.backend} · ${data.contract.precision} · ${data.contract.device}`;
  linePlot(); comparisonTable();
}

function comparisonTable() {
  const key=$('quantity').value.split(':')[0], region=$('region').value;
  const comparisons=data.comparisons;
  $('comparison-values').replaceChildren(...(comparisons?.rows||[]).map(row=>{
    const tr=document.createElement('tr'), label=document.createElement('td'), absolute=document.createElement('td'), mixed=document.createElement('td');
    const metric=row.terms?.[region]?.[key];
    label.textContent=row.label+(row.status==='measured'?` · ${row.elapsed_ms.toFixed(0)} ms`: ` · ${row.reason}`);
    absolute.textContent=metric?pretty(metric.max_absolute):'Unavailable';
    mixed.textContent=metric?pretty(metric.max_mixed):'Unavailable';
    tr.append(label,absolute,mixed);return tr;
  }));
  $('comparison-scope').textContent=comparisons?`${regionNames[region]} · ${key}, all components and displayed probes. ${comparisons.convention} ${comparisons.scope}`:'No comparisons saved for this snapshot.';
}

function linePlot() {
  const canvas = $('line'), ctx = canvas.getContext('2d'), n = data.shape[1], row = Math.floor(probe/n);
  const colors = {solid:'#ad623f', porous:'#218e91', total:'#243b44'};
  const series = Object.keys(colors).map(r=>({r,v:field(r)?.slice(row*n,(row+1)*n)})).filter(s=>s.v);
  const limit = Math.max(1e-12,...series.flatMap(s=>s.v.filter(finite).map(Math.abs)));
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#c7d5cc'; ctx.beginPath(); ctx.moveTo(20,100); ctx.lineTo(535,100); ctx.stroke();
  for (const s of series) {
    ctx.beginPath(); ctx.strokeStyle = colors[s.r];
    s.v.forEach((v,i)=>{const x=20+515*i/(n-1),y=100-85*v/limit; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
    ctx.stroke();
  }
  ctx.fillStyle = '#718178'; ctx.font = '11px system-ui';
  ctx.fillText(`±${limit.toExponential(2)}`,22,14);
  ctx.fillText(`${'xyz'[(data.plot_axes||[0,1])[0]]} →`,505,195);
}

function validateSnapshot(result) {
  if (result?.kind !== 'fluxbridge-wall' || ![2,3].includes(result.dimension) ||
      !Array.isArray(result.shape) || result.shape.length !== 2 ||
      result.shape.some(n=>!Number.isInteger(n)||n<3||n>33) ||
      !Array.isArray(result.queries) || result.queries.length !== result.shape[0]*result.shape[1] ||
      !Array.isArray(result.axes) || result.axes.length !== 2 || !result.config || !result.contract ||
      !result.terms?.solid || !result.terms?.porous || !result.terms?.free) throw Error('Choose a valid wall grid snapshot');
  for (let j=0;j<2;j++) {
    const axis = result.axes[j];
    if (!Array.isArray(axis) || axis.length !== result.shape[1-j] ||
        axis.some((v,i)=>!finite(v)||(i>0&&v<=axis[i-1]))) throw Error('Snapshot axes must increase in physical coordinates');
  }
  if (result.queries.some(q=>!Array.isArray(q)||q.length!==result.dimension||q.some(v=>!finite(v)))) throw Error('Invalid snapshot query coordinates');
}

function setData(result, mode='snapshot') {
  validateSnapshot(result);
  data=result; source=mode; loadConfig(result.config); probe=Math.floor(data.queries.length/2);
  $('save').disabled=false; $('compare').disabled=false;
  $('provenance').textContent=JSON.stringify({contract:data.contract,execution:data.timing,comparisons:data.comparisons},null,2);
  updateProvider(); render();
}

function updateProvider() {
  const browser=$('provider').value==='webgpu';
  if(browser) for(const [key,value]of Object.entries({precision:'float32',backend:'numerical',preparation:'host',search:'host'}))form.elements[key].value=value;
  for(const key of ['precision','backend','preparation','search','device']){form.elements[key].disabled=browser;form.elements[key].closest('label').hidden=browser;}
}
$('provider').addEventListener('change',updateProvider);
$('cancel').addEventListener('click',()=>cancelWall());

async function compute(event, comparison=false) {
  event?.preventDefault(); $('compute').disabled=true; $('compare').disabled=true; $('status').className='';
  const browser=$('provider').value==='webgpu';$('cancel').disabled=!browser;
  $('status').textContent=browser?'Preparing browser WebGPU fp32 terms…':'Preparing native Warp terms…';
  try {
    let result;
    if(browser) result=await computeWallInBrowser(comparison?{...data.config,precision:'float32',backend:'numerical',preparation:'host',search:'host'}:config(),detail=>{$('status').textContent=detail;},comparison);
    else {
      const res=await fetch(comparison?'./api/wall/compare':'./api/wall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(comparison?data.config:config())});
      result=await res.json();if(!res.ok)throw Error(result.error||`HTTP ${res.status}`);
    }
    setData(result,'live');
    $('status').textContent=`Evaluated ${result.queries.length} queries and ${result.elements.length} elements. Total ${result.elapsed_ms.toFixed(0)} ms; static setup ${result.static_prepare_ms.toFixed(0)} ms; preparation ${result.timing.reduce((s,t)=>s+t.prepare_ms,0).toFixed(0)} ms; action/readback ${result.timing.reduce((s,t)=>s+t.action_readback_ms,0).toFixed(0)} ms. ${browser?'Browser fp32; accepted/comparison orders '+result.contract.quadrature.nodes+'/'+result.contract.quadrature.comparison_nodes+'.':''}`;
  } catch (error) {
    $('status').className=error.name==='AbortError'?'':'error';
    $('status').textContent=error.message+(browser?'':' · Start the local boundary-lab service for native evaluation.');
  } finally {$('compute').disabled=false; $('compare').disabled=!data;$('cancel').disabled=true;}
}

form.addEventListener('submit',compute);
$('compare').addEventListener('click',event=>compute(event,true));
for (const key of ['dimension','backend']) form.elements[key].addEventListener('change',()=>{
  if(form.elements.dimension.value==='2'||form.elements.backend.value==='analytic') form.elements.preparation.value='host';
});
for (const id of ['quantity','region','overlay']) $(id).addEventListener('change',render);
$('map').addEventListener('click',e=>{
  if (!data) return;
  const rect=e.currentTarget.getBoundingClientRect();
  const x=(e.clientX-rect.left)*e.currentTarget.width/rect.width,y=(e.clientY-rect.top)*e.currentTarget.height/rect.height;
  let best=Infinity;
  data.queries.forEach((q,i)=>{const [px,py]=project(q),dist=(x-px)**2+(y-py)**2;if(dist<best){best=dist;probe=i;}});
  render();
});
$('preset').addEventListener('change',()=>{
  const p=$('preset').value;
  geometry=p==='fibres'?'fibres':p==='disk'?'disk':'wall';
  loadConfig({geometry,fraction:p==='solid'?0:p==='porous'||p==='zero'?1:.5,resistance:p==='zero'?0:4,
    tensor_angle:p==='rotated'?45:0,angle:p==='rotated'?25:0,subdivisions:p==='refined'?4:2});
  if(geometry!=='wall'){form.elements.dimension.value='2';form.elements.preparation.value='host';}
});
$('save').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify(data)],{type:'application/json'}),a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='fluxbridge-wall.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
$('file').addEventListener('change',async e=>{
  try {
    setData(JSON.parse(await e.target.files[0].text()));
    $('status').textContent='Snapshot loaded. Select Browser · WebGPU fp32 to recompute without the native service.';
  } catch(error){$('status').textContent=error.message;$('status').className='error';}
});
fetch('./wall-example.json').then(r=>r.ok?r.json():null).then(result=>{
  if(result){setData(result);$('status').textContent='Snapshot loaded. Browser · WebGPU fp32 can recompute on this static site.';}
}).catch(()=>{});

updateProvider();
