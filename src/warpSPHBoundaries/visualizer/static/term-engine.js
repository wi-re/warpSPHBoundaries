// Browser-independent mathematics; no DOM, Python, Warp, CDN, or network imports.
// Unit support; y=source-query. See docs/derivations/radial-terms-3d.md.
export const FIELD_INFO=[
 ['m00','λ · kernel mass',0,'∫Ω Wₕ dV'],
 ['m100','Mx / h',1,'∫Ω yₓ Wₕ dV / h'],['m010','My / h',1,'∫Ω yᵧ Wₕ dV / h'],['m001','Mz / h',1,'∫Ω y_z Wₕ dV / h'],
 ['m200','Mxx / h²',2,'∫Ω yₓ² Wₕ dV / h²'],['m110','Mxy / h²',2,'∫Ω yₓyᵧ Wₕ dV / h²'],['m101','Mxz / h²',2,'∫Ω yₓy_z Wₕ dV / h²'],['m020','Myy / h²',2,'∫Ω yᵧ² Wₕ dV / h²'],['m011','Myz / h²',2,'∫Ω yᵧy_z Wₕ dV / h²'],['m002','Mzz / h²',2,'∫Ω y_z² Wₕ dV / h²'],
 ...[...'xyz'].map(a=>['G'+a,'h G'+a+' · kernel gradient',-1,'h ∫Ω ∂query,'+a+' Wₕ dV']),
 ...[...'xyz'].flatMap(a=>[...'xyz'].map(b=>['g1'+a+b,'g₁ '+a+b+' · first gradient moment',0,'∫Ω y'+a+' ∂query,'+b+' Wₕ dV'])),
 ...['xx','xy','xz','yy','yz','zz'].map(ab=>['H'+ab,'h² H'+ab+' · kernel Hessian',-2,'h² ∫Ω ∂query,'+ab+' Wₕ dV']),
 ['laplacian','h² Δλ · wall Laplacian',-2,'h² ∫Ω ΔWₕ dV = h² tr H'],['grad_norm','|h G| · kernel gradient magnitude',-1,'h |∫Ω ∇query Wₕ dV|']
];
export const FIELD_IDS=Object.fromEntries(FIELD_INFO.map((f,i)=>[f[0],i]));
export const SHAPES={cubic:{n:3,P:[1],C:16/Math.PI},w2:{n:4,P:[1,4],C:21/(2*Math.PI)},w4:{n:6,P:[1,6,35/3],C:495/(32*Math.PI)},w6:{n:8,P:[1,8,25,32],C:1365/(64*Math.PI)}};
const choose=(n,k)=>{let r=1;for(let i=1;i<=k;i++)r=r*(n-i+1)/i;return r;};
const falling=(n,k)=>{let r=1;for(let i=0;i<k;i++)r*=n-i;return r;};
for(const [name,k] of Object.entries(SHAPES)){
 k.coeffs=Array(12).fill(0);
 if(name==='cubic'){k.coeffs[0]=.5*k.C;k.coeffs[2]=-3*k.C;k.coeffs[3]=3*k.C;}
 else for(let j=0;j<=k.n;j++)for(let i=0;i<k.P.length;i++)k.coeffs[i+j]+=k.C*choose(k.n,j)*(-1)**j*k.P[i];
 // All assembled kernels have W'(0)=0 exactly.
 k.coeffs[1]=0;
}
export function radialJet(r,name){
 if(r<0||r>=1)return [0,0,0,0];
 const k=SHAPES[name];let W=0,L=0,Q=0,R=0;
 if(r<.1){
  for(let p=0;p<12;p++){const c=k.coeffs[p];if(!c)continue;W+=c*r**p;if(p>=2)L+=p*c*r**(p-2);if(r&&p>=3)Q+=p*(p-2)*c*r**(p-4);if(r&&p>=3)R+=p*(p-2)*(p-4)*c*r**(p-6);}
  return [W,L,Q,R];
 }
 const jet=[];
 for(let order=0;order<4;order++){
  let v=0;
  if(name==='cubic')v=falling(3,order)*(-1)**order*((1-r)**(3-order)-(r<.5?4*(.5-r)**(3-order):0));
  else for(let j=0;j<=order;j++){
   let poly=0;for(let p=order-j;p<k.P.length;p++)poly+=k.P[p]*falling(p,order-j)*r**(p-order+j);
   v+=choose(order,j)*(-1)**j*falling(k.n,j)*(1-r)**(k.n-j)*poly;
  }
  jet.push(v*k.C);
 }
 W=jet[0];L=jet[1]/r;Q=(jet[2]-L)/(r*r);R=(jet[3]-3*jet[2]/r+3*jet[1]/(r*r))/r**3;
 return [W,L,Q,R];
}
const gaussCache=new Map();
export function gauss(n){
 if(gaussCache.has(n))return gaussCache.get(n);
 const out=[];
 for(let i=1;i<=n;i++){
  let z=Math.cos(Math.PI*(i-.25)/(n+.5)),derivative=0;
  for(let iteration=0;iteration<30;iteration++){
   let p0=1,p1=z;for(let j=2;j<=n;j++){const p=((2*j-1)*z*p1-(j-1)*p0)/j;p0=p1;p1=p;}
   derivative=n*(z*p1-p0)/(z*z-1);const delta=p1/derivative;z-=delta;if(Math.abs(delta)<1e-15)break;
  }
  out.push([z,2/((1-z*z)*derivative*derivative)]);
 }
 gaussCache.set(n,out);return out;
}
export function shell(q,cap,name){
 const [W,L,Q]=radialJet(q,name),c=Math.max(-1,Math.min(1,cap));
 const A0=2*Math.PI*(c+1),A1=Math.PI*(c*c-1),An=2*Math.PI*(c**3+1)/3,At=Math.PI*(c-c**3/3+2/3),base=q*q*L*A0;
 return [q*q*W*A0,q**3*W*A1,q**4*W*At,q**4*W*An,-(q**3)*L*A1,-(q**4)*L*At,-(q**4)*L*An,base+q**4*Q*At,base+q**4*Q*An];
}
export function curvatureIntegrand(q,d,kappa,name,order=2){
 const [W,L,Q,R]=radialJet(q,name),z=-d,rho2=Math.max(q*q-d*d,0);
 const corr=(t,p,f,df)=>{
  const first=-.5*kappa*z**p*rho2**((t+2)/2)*f;
  const derivative=(p?p*z**(p-1)*f:0)+z**(p+1)*df;
  return (t?Math.PI:2*Math.PI)*q*(first+(order>=2?kappa*kappa/8*rho2**((t+4)/2)*derivative:0));
 };
 const base=corr(0,0,L,Q);
 return [corr(0,0,W,L),corr(0,1,W,L),corr(2,0,W,L),corr(0,2,W,L),-corr(0,1,L,Q),-corr(2,0,L,Q),-corr(0,2,L,Q),base+corr(2,0,Q,R),base+corr(0,2,Q,R)];
}
export function axialTerms(d,name,radius=null,curvature=null,curvatureOrder=2){
 const out=Array(9).fill(0),knots=name==='cubic'?[0,.5,1]:[0,1];
 const ranges=[[0,Math.min(1,Math.max(-d,0)),true],[Math.abs(d),radius===null?1:Math.min(1,2*radius+d),false]];
 for(const [lo,hi,full] of ranges)for(let j=0;j<knots.length-1;j++){
  const l=Math.max(lo,knots[j]),r=Math.min(hi,knots[j+1]);if(r<=l)continue;
  for(const [node,weight] of gauss(16)){
   const q=(l+r)/2+(r-l)/2*node,D=radius===null?0:radius+d;
   const cap=full?1:radius===null?-d/q:(radius*radius-D*D-q*q)/(2*(D||1)*q);
   const v=shell(q,cap,name);for(let i=0;i<9;i++)out[i]+=(r-l)/2*weight*v[i];
  }
 }
 if(curvature!==null&&curvatureOrder>=1&&Math.abs(d)<1)for(let j=0;j<knots.length-1;j++){
  const l=Math.max(Math.abs(d),knots[j]),r=knots[j+1];if(r<=l)continue;
  for(const [node,weight]of gauss(16)){const q=(l+r)/2+(r-l)/2*node,v=curvatureIntegrand(q,d,curvature,name,curvatureOrder);for(let i=0;i<9;i++)out[i]+=(r-l)/2*weight*v[i];}
 }
 return out;
}
const pairs=[[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]];
export function selectAxial(v,n,field){
 const id=typeof field==='number'?field:FIELD_IDS[field];
 if(id===0)return v[0];
 if(id>=1&&id<=3)return v[1]*n[id-1];
 if(id>=4&&id<=9){const [i,j]=pairs[id-4];return (i===j?v[2]:0)+(v[3]-v[2])*n[i]*n[j];}
 if(id>=10&&id<=12)return v[4]*n[id-10];
 if(id>=13&&id<=21){const i=Math.floor((id-13)/3),j=(id-13)%3;return (i===j?v[5]:0)+(v[6]-v[5])*n[i]*n[j];}
 if(id>=22&&id<=27){const [i,j]=pairs[id-22];return (i===j?v[7]:0)+(v[8]-v[7])*n[i]*n[j];}
 if(id===28)return 2*v[7]+v[8];
 if(id===29)return Math.abs(v[4]);
 throw new Error('Unknown 3D term');
}
export function integrand(y,name,field){
 const id=typeof field==='number'?field:FIELD_IDS[field],r=Math.hypot(...y),[W,L,Q]=radialJet(r,name);
 if(id===0)return W;
 if(id>=1&&id<=3)return y[id-1]*W;
 if(id>=4&&id<=9){const [i,j]=pairs[id-4];return y[i]*y[j]*W;}
 if(id>=10&&id<=12)return -y[id-10]*L;
 if(id>=13&&id<=21)return -y[Math.floor((id-13)/3)]*y[(id-13)%3]*L;
 if(id>=22&&id<=27){const [i,j]=pairs[id-22];return (i===j?L:0)+y[i]*y[j]*Q;}
 if(id===28)return 3*L+r*r*Q;
 throw new Error('Integrate vector components before taking a norm');
}
export function ballSeries(point,a,name,field,order=4){
 const q=Math.hypot(...point);if(q-a>=1)return 0;
 if(a>.3||q<=a||(name==='cubic'&&q-a<=.5&&q+a>=.5)||q+a>=1)return NaN;
 const id=typeof field==='number'?field:FIELD_IDS[field];
 if(id===29)return Math.hypot(...[10,11,12].map(f=>ballSeries(point,a,name,f,order)));
 let coeffs=SHAPES[name].coeffs;
 if(name==='cubic'&&q>.5)coeffs=[1,-3,3,-1].map(c=>c*SHAPES.cubic.C);
 let terms=[];
 const add=(alpha,p,c)=>{if(c)terms.push([alpha,p,c]);},zero=[0,0,0];
 coeffs.forEach((c,p)=>{
  if(id===0)add(zero,p,c);
  else if(id>=1&&id<=9){const alpha=id<=3?zero.map((_,i)=>+(i===id-1)):zero.map((_,i)=>+(i===pairs[id-4][0])+(i===pairs[id-4][1]));add(alpha,p,c);}
  else if(id>=10&&id<=12)add(zero.map((_,i)=>+(i===id-10)),p-2,-p*c);
  else if(id>=13&&id<=21)add(zero.map((_,i)=>+(i===Math.floor((id-13)/3))+(i===(id-13)%3)),p-2,-p*c);
  else if(id>=22&&id<=27){const [i,j]=pairs[id-22];if(i===j)add(zero,p-2,p*c);add(zero.map((_,k)=>+(k===i)+(k===j)),p-4,p*(p-2)*c);}
  else if(id===28)add(zero,p-2,p*(p+1)*c);
 });
 const evaluate=ts=>ts.reduce((sum,[alpha,p,c])=>sum+c*q**p*alpha.reduce((s,k,i)=>s*(-point[i])**k,1),0);
 const lap=ts=>ts.flatMap(([alpha,p,c])=>{
  const out=[],radial=p*(p+1+2*alpha.reduce((s,a)=>s+a,0))*c;
  if(radial)out.push([alpha,p-2,radial]);alpha.forEach((a,j)=>{if(a>=2)out.push([alpha.map((b,k)=>b-2*+(k===j)),p,a*(a-1)*c]);});return out;
 });
 let value=evaluate(terms);if(order>=2){terms=lap(terms);value+=a*a/10*evaluate(terms);}if(order>=4){terms=lap(terms);value+=a**4/280*evaluate(terms);}
 return 4*Math.PI*a**3/3*value;
}
