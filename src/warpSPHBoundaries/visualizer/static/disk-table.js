// Seven disk invariants, same radius/offset maps as edge.disktables.
// Coefficients come from circle-flux quadrature. This is an approximation.
const cached=new Map();
export async function loadDiskTable(name){
 if(!cached.has(name))cached.set(name,(async()=>{const response=await fetch(new URL(`./disk-table-${name}.bin?v=20261022-7`,import.meta.url));if(!response.ok)throw new Error('Could not load the disk lookup table');const a=new Float32Array(await response.arrayBuffer());if(a.length!==72576||!a.every(Number.isFinite))throw new Error('Invalid disk lookup table');return a;})());
 return cached.get(name);
}
function panel(table,base,pa,pb,a,b){
 a=Math.max(0,Math.min(1,a));b=Math.max(0,Math.min(1,b));const i=Math.min(pa-1,Math.floor(a*pa)),j=Math.min(pb-1,Math.floor(b*pb)),x=2*(a*pa-i)-1,y=2*(b*pb-j)-1,Tx=[1,x],Ty=[1,y],out=Array(7).fill(0);
 for(let k=2;k<12;k++){Tx[k]=2*x*Tx[k-1]-Tx[k-2];Ty[k]=2*y*Ty[k-1]-Ty[k-2];}
 const start=base+(i*pb+j)*12*12*7;
 for(let u=0;u<12;u++)for(let v=0;v<12;v++)for(let c=0;c<7;c++)out[c]+=Tx[u]*Ty[v]*table[start+(u*12+v)*7+c];return out;
}
export function diskTableChannels(table,p,R,whole){
 const D=Math.hypot(...p);if(D>=R+1)return Array(12).fill(0);if(D+1<=R)return [...whole];let v;
 if(R>1)v=panel(table,6*3*3*12*12*7,6,3,(D-R+1)/2,1/R);
 else{
  const branch=R<=.5?0:1,delta=D-R,intervals=branch===0?[[-R,0],[0,1-2*R],[1-2*R,1]]:[[-R,1-2*R],[1-2*R,0],[0,1]],k=intervals.findIndex(([lo,hi],i)=>delta>=lo&&(delta<hi||i===2&&delta<=hi)),[lo,hi]=intervals[Math.max(k,0)],u=Math.max(0,Math.min(1,(delta-lo)/Math.max(hi-lo,1e-30)));
  v=panel(table,(branch*3+Math.max(k,0))*3*3*12*12*7,3,3,2/Math.PI*Math.asin(Math.sqrt(u)),branch===0?2*R:2*R-1).map(x=>x*R*R);
 }
 if(D===0){v[1]=v[4]=0;v[2]=v[3]=(v[2]+v[3])/2;v[5]=v[6]=(v[5]+v[6])/2;}
 const n=D?p.map(x=>x/D):[1,0],out=[v[0],v[1]*n[0],v[1]*n[1]];
 out.push(v[2]+(v[3]-v[2])*n[0]**2,(v[3]-v[2])*n[0]*n[1],v[2]+(v[3]-v[2])*n[1]**2,v[4]*n[0],v[4]*n[1],v[5]+(v[6]-v[5])*n[0]**2,(v[6]-v[5])*n[0]*n[1],(v[6]-v[5])*n[0]*n[1],v[5]+(v[6]-v[5])*n[1]**2);return out;
}
