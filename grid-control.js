// Keep the convenient slider and an exact integer override in sync.
export function gridControl(parent,config,{label='Grid resolution',max=513,sliderMax=193,unit='²',changed}){
 const group=document.createElement('div');group.className='manual-grid';
 const top=document.createElement('label');top.className='range-top';top.textContent=label;
 const numeric=document.createElement('input');numeric.type='number';numeric.min=17;numeric.max=max;numeric.step=1;numeric.value=config.resolution;numeric.setAttribute('aria-label',`${label} override`);
 const slider=document.createElement('input');slider.type='range';slider.min=17;slider.max=Math.max(sliderMax,config.resolution);slider.step=1;slider.value=config.resolution;slider.setAttribute('aria-label',label);
 const hint=document.createElement('p');hint.className='hint';hint.textContent=`${config.resolution}${unit} samples · enter any integer from 17 to ${max}.`;
 const commit=value=>{if(!Number.isInteger(value)||value<17||value>max){numeric.setCustomValidity(`Enter an integer from 17 to ${max}`);numeric.reportValidity();return;}numeric.setCustomValidity('');config.resolution=value;slider.max=Math.max(sliderMax,value);slider.value=value;numeric.value=value;hint.textContent=`${value}${unit} samples · enter any integer from 17 to ${max}.`;changed();};
 numeric.onchange=()=>commit(numeric.valueAsNumber);numeric.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();commit(numeric.valueAsNumber);}};
 slider.oninput=()=>commit(+slider.value);top.append(numeric);group.append(top,slider,hint);parent.append(group);
}
