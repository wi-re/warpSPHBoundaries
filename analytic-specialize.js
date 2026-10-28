// Fixed-profile rolling edge reductions, following the native solid kernels.
// Keep the generic shader for series/local models and as an independent check.
// Constants are split on the host; every shader instruction remains fp32.
const literal=x=>{const hi=Math.fround(x),lo=Math.fround(x-hi);return `D(${hi},${lo})`;};

export function specializeAnalyticShader(source,plan){
 const f=new Float32Array(plan.uniform.buffer,plan.uniform.byteOffset,16),u=new Uint32Array(plan.uniform.buffer,plan.uniform.byteOffset,16);
 if(f[11]<0)return {key:'generic',code:source};
 const area=f[10]<2.5,id=u[1],profiles=[];
 for(let k=0;k<u[3];k++){
  const [n,R,hi,lo,kind]=plan.table.slice(k*6,k*6+5);
  profiles.push({m:n+2,R,c:(hi+lo)/(n+2),kind});
 }
 const first=area?[1,2,6,7,12].includes(id):(id>=4&&id<=9||id>=13&&id<=27);
 const second=area&&[3,4,5,8,9,10,11].includes(id);
 const max=Math.max(...profiles.map(p=>p.m));
 const terms=m=>profiles.filter(p=>p.m===m).map(({R,c,kind})=>{
  const inside=R===0?'true':`r2.x<${R*R}`;
  const radial=R===0?'d(0)':literal(R**m);
  const basis=m===-1?'cm':`c${m%2}`;
  return `{
   let coefficient=${literal(c)};let inside=${inside};
   if w.x!=0.0 && ${R===0?'true':`a.x<${R}`} ${kind===0?'&& z.x!=0.0':''}{
    var value=mul(sub(${radial},ap),k0);
    if inside{value=sub(${basis},mul(ap,k0));}
    S.scalar[${kind}]=add(S.scalar[${kind}],mul(coefficient,mul(w,value)));
   }
   ${kind>0&&(first||second)&&(!area||kind===1)?`if inside{
    let line=sub(j${m%2},mul(${radial},span));
    ${first?`S.first[${kind}]=vadd(S.first[${kind}],scale(mu,mul(coefficient,line)));`:''}
    ${second?`let J=sub(div(sub(mul(up,mul(ru,ru)),mul(lp,mul(rl,rl))),d(${m+2})),mul(${radial},mul(d(0.5),sub(mul(u,u),mul(l,l)))));
    let moment=vadd(scale(mu,mul(w,line)),scale(t,J));
    S.second[${kind}]=vadd(S.second[${kind}],V(mul(coefficient,mul(mu.x,moment.x)),mul(coefficient,mul(mu.x,moment.y)),mul(coefficient,mul(mu.y,moment.y))));`:''}
   }`:''}
  }`;
 }).join('\n');
 let steps='';
 // C_m = J_(m-2) + a² C_(m-2). Consume each degree immediately.
 // Only two parity states, two endpoint powers and a^m remain live.
 for(let m=0;m<=max;m++){
  if(m===0)steps+='var ap=d(1);\n';
  else steps+='ap=mul(ap,a);lp=mul(lp,rl);up=mul(up,ru);\n';
  if(m>=2)steps+=`c${m%2}=add(j${m%2},mul(a2,c${m%2}));\n`;
  if(m>=1)steps+=`j${m%2}=div(add(sub(mul(u,up),mul(l,lp)),mul(d(${m}),mul(c2,j${m%2}))),d(${m+1}));\n`;
  steps+=terms(m);
 }
 const body=`fn integrateFace(vertices:array<V,3>,normal:V,z:D)->Profiles{
 let a=absD(z);let a2=mul(a,a);var S:Profiles;
 for(var edge=0u;edge<3u;edge++){
  let start=vertices[edge];let delta=vsub(vertices[(edge+1u)%3u],start);let len=sqrtD(dotD(delta,delta));let t=scale(delta,div(d(1),len));let mu=crossD(t,normal);
  let w=dotD(start,mu);let b=absD(w);let lo=dotD(start,t);let hi=add(lo,len);let c2=add(mul(w,w),a2);
  var cuts:array<D,6>;cuts[0]=lo;var count=1u;
  ${[...new Set(profiles.map(p=>p.R).filter(R=>R>0))].map(R=>`{
   if c2.x<${R*R}{let cut=sqrtD(sub(${literal(R*R)},c2));
    for(var sign=0u;sign<2u;sign++){let x=mul(d(select(-1.0,1.0,sign==1u)),cut);
     if x.x>lo.x && x.x<hi.x{var j=count;loop{if j==0u{break;}if cuts[j-1u].x<x.x{break;}cuts[j]=cuts[j-1u];j--;}cuts[j]=x;count++;}
    }
   }
  }`).join('\n')}
  cuts[count]=hi;count++;
  for(var panel=1u;panel<count;panel++){
   let l=cuts[panel-1u];let u=cuts[panel];let span=sub(u,l);let mid=mul(d(0.5),add(l,u));let r2=add(mul(mid,mid),c2);
   let rl=sqrtD(add(mul(l,l),c2));let ru=sqrtD(add(mul(u,u),c2));
   var k0=d(0);var cm=d(0);var c1=d(0);var j1=d(0);
   if c2.x>0.0{let c=sqrtD(c2);j1=sub(asinhD(div(u,c)),asinhD(div(l,c)));}
   if w.x!=0.0{
    k0=div(atan2D(mul(b,span),add(mul(w,w),mul(l,u))),b);
    if a.x!=0.0{
     var delta=sub(mul(u,rl),mul(l,ru));
     if l.x*u.x>0.0{delta=div(mul(c2,mul(span,add(u,l))),add(mul(u,rl),mul(l,ru)));}
     let theta=atan2D(mul(mul(a,b),delta),add(mul(mul(b,b),mul(ru,rl)),mul(a2,mul(l,u))));
     cm=div(theta,mul(a,b));
    }else{cm=div(sub(div(u,ru),div(l,rl)),mul(b,b));}
   }
   c1=add(j1,mul(a2,cm));var c0=k0;var j0=span;var lp=d(1);var up=d(1);
   ${profiles.some(p=>p.m===-1)?`if a.x!=0.0{let ap=div(d(1),a);${terms(-1)}}`:''}
   ${steps}
  }
 }
 ${first?'for(var kind=1u;kind<3u;kind++){S.first[kind]=vadd(S.first[kind],scale(normal,mul(z,S.scalar[kind])));}':''}
 return S;
}
`;
 const begin=source.indexOf('fn integrateFace('),end=source.indexOf('fn pair(',begin);
 if(begin<0||end<0)throw Error('Analytic shader template is missing the face reduction');
 // Axis components with the same profiles share a pipeline. Field selection
 // remains a uniform, so changing Gx to Gy does not trigger another compile.
 const code=(source.slice(0,begin)+body+source.slice(end))
  .replaceAll('cfg.values.z',area?'2.0':'3.0')
  .replaceAll('cfg.values.w== -1.0','false').replaceAll('cfg.values.w<= -2.0','false');
 return {key:JSON.stringify([area,first,second,profiles]),code};
}
