// Exact elementary polygon integration and disk mean-value expansions.
import {AnalyticTriangle,analyticProfile,compensated,linePrimitive} from './mesh3d-analytic.js?v=20261022-7';
const norm={w2:7/Math.PI,w4:9/Math.PI,w6:78/(7*Math.PI),cubic:80/(7*Math.PI)};
const norm3={w2:21/(2*Math.PI),w4:495/(32*Math.PI),w6:1365/(64*Math.PI),cubic:16/Math.PI};
const cache=new Map();
export function profile2d(name){if(!cache.has(name))cache.set(name,analyticProfile(name).map(([p,R,c])=>[p,R,c*norm[name]/norm3[name]]));return cache.get(name);}
export function analyticPolygon2d(vertices,point,name){
 const faces=[];for(let i=1;i<vertices.length-1;i++)faces.push(new AnalyticTriangle([vertices[0],vertices[i],vertices[i+1]].map(p=>[p[0],p[1],0]),[...point,0]));
 const W=profile2d(name),L=W.filter(([p])=>p).map(([p,R,c])=>[p-2,R,p*c]);
 const integral=(profile,a,b)=>compensated(faces.map(f=>f.integrate(profile,[a,b,0])));
 return [integral(W,0,0),integral(W,1,0),integral(W,0,1),integral(W,2,0),integral(W,1,1),integral(W,0,2),-integral(L,1,0),-integral(L,0,1),-integral(L,2,0),-integral(L,1,1),-integral(L,1,1),-integral(L,0,2)];
}
export function diskSeries2d(point,a,name,field,order=4){
 const q=Math.hypot(...point);if(q-a>=1)return 0;
 if(a>.3||q<=a||q+a>=1||(name==='cubic'&&q-a<=.5&&q+a>=.5))return NaN;
 if(field===12)return Math.hypot(diskSeries2d(point,a,name,6,order),diskSeries2d(point,a,name,7,order));
 const terms=[];for(const [p,R,c]of profile2d(name)){
  if(q>=R)continue;
  if(field<=5){const alpha=[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]][field];terms.push([alpha,p,c]);}
  else if(p){const alpha=field<=7?[[1,0],[0,1]][field-6]:[[2,0],[1,1],[1,1],[0,2]][field-8];terms.push([alpha,p-2,-p*c]);}
 }
 const value=ts=>compensated(ts.map(([alpha,p,c])=>c*q**p*(-point[0])**alpha[0]*(-point[1])**alpha[1]));
 const lap=ts=>ts.flatMap(([a,p,c])=>{const out=[],radial=p*(p+2*(a[0]+a[1]))*c;if(radial)out.push([a,p-2,radial]);for(let j=0;j<2;j++)if(a[j]>=2)out.push([a.map((n,k)=>n-2*+(j===k)),p,a[j]*(a[j]-1)*c]);return out;});
 let result=value(terms);if(order>=2){const first=lap(terms);result+=a*a/8*value(first);if(order>=4)result+=a**4/192*value(lap(first));}return Math.PI*a*a*result;
}

export function analyticPlane2d(d,n,name,k=0,order=2){
 const W=profile2d(name),variance=Math.PI*compensated(W.map(([p,R,c])=>c*R**(p+4)/(p+4))),whole=[1,0,0,variance,0,variance,0,0,1,0,0,1];
 if(d>=1)return Array(12).fill(0);if(d<=-1)return whole;
 const local=analyticPolygon2d([[-4,-4],[4,-4],[4,0],[-4,0]],[0,d],name),L=local[0];let B0=0,B2=0,B4=0;
 for(const [p,R,c]of W){if(Math.abs(d)>=R)continue;const top=Math.sqrt((R-Math.abs(d))*(R+Math.abs(d)));B0+=2*c*linePrimitive(0,p,top,Math.abs(d));B2+=2*c*linePrimitive(2,p,top,Math.abs(d));B4+=2*c*linePrimitive(4,p,top,Math.abs(d));}
 if(order===0)k=0;const k2=order>=2?k*k:0;
 const mass=L-k*B2/2+3*d*k2*B2/8,mn=-B2+k*d*B2/2+k2*(B4-3*d*d*B2)/8,T=local[3]-k*B4/2+5*d*k2*B4/8,N=local[5]-k*d*d*B2/2+k2*(-2*d*B4+3*d**3*B2)/8,G=-B0+k*d*B0/2+3*k2*(B2-d*d*B0)/8,Ct=L-3*k*B2/2+15*d*k2*B2/8,Cn=L+d*B0-k*d*d*B0/2+k2*(-6*d*B2+3*d**3*B0)/8;
 if(!n.every(Number.isFinite))return [mass,...Array(11).fill(NaN)];const [nx,ny]=n;
 return [mass,mn*nx,mn*ny,T+(N-T)*nx*nx,(N-T)*nx*ny,T+(N-T)*ny*ny,G*nx,G*ny,Ct+(Cn-Ct)*nx*nx,(Cn-Ct)*nx*ny,(Cn-Ct)*nx*ny,Ct+(Cn-Ct)*ny*ny];
}
