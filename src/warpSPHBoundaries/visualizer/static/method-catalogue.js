import {MODELS2D} from './browser-compute2d.js?v=20261022-7';
import {BROWSER_MODELS,modelError} from './browser-compute.js?v=20261022-7';
import {modelError2d} from './compute2d-plan.js?v=20261022-7';
export function renderMethods(parent,dimension,c){
 if(!parent)return;parent.replaceChildren();const table=document.createElement('table'),head=document.createElement('tr');for(const text of ['Method','Applies here']){const th=document.createElement('th');th.textContent=text;head.append(th);}table.append(head);
 for(const [key,[title]]of Object.entries(dimension===2?MODELS2D:BROWSER_MODELS)){
  const error=dimension===2?modelError2d(key,{...c,sdf_source:'analytic'},[],[]):modelError(key,c),row=document.createElement('tr');for(const text of [title,error??'Yes']){const td=document.createElement('td');td.textContent=text;row.append(td);}table.append(row);
 }parent.append(table);const note=document.createElement('p');note.className='hint';note.textContent=dimension===2?'All listed browser methods support every displayed 2D quantity. Curvature orders 0/1/2 and small-disk orders 0/2/4 are selected in SDF & reference settings.':'All listed browser methods support all 30 quantities. Curvature orders 0/1/2 and small-ball orders 0/2/4 are selected in Expansion orders.';parent.append(note);
}
