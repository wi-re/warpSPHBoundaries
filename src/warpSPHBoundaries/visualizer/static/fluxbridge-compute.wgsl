// Generic degree-nine volume moments / kernel-only derivatives, ordinary fp32.
// Directions/radii/weights are host-prepared. Each invocation owns one output.
struct Params { counts:vec4u, physical:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> samples:array<vec4f>;
@group(0) @binding(2) var<storage,read> profiles:array<f32>;
@group(0) @binding(3) var<storage,read> requests:array<vec4i>;
@group(0) @binding(4) var<storage,read> ranges:array<vec4u>;
@group(0) @binding(5) var<storage,read_write> output:array<f32>;
fn chebyshev(x:f32,offset:u32,length:u32)->f32 {
 var b1=0.0;var b2=0.0;
 for(var k=i32(length)-1;k>=1;k--){let next=fma(2.0*x,b1,profiles[offset+u32(k)]-b2);b2=b1;b1=next;}
 return fma(x,b1,profiles[offset]-b2);
}
fn cumulative(index:u32,r:f32,kind:u32)->f32 {
 let base=12u*index;let knot=profiles[base+1u];
 if(kind!=0u&&r<knot){return chebyshev(2.0*r/knot-1.0,u32(profiles[base+7u]),u32(profiles[base+6u]));}
 var value=profiles[base];
 if(r<1.0){
  let outer=r>=knot;let lo=select(0.0,knot,outer);let hi=select(knot,1.0,outer);let slot=base+2u+select(0u,2u,outer);
  value=chebyshev(2.0*(r-lo)/(hi-lo)-1.0,u32(profiles[slot+1u]),u32(profiles[slot]));
 }
 if(kind!=0u){value/=r*r*r;}return value;
}
fn ipow(x:f32,n:i32)->f32 {var v=1.0;for(var k=0;k<abs(n);k++){v*=x;}return select(v,1.0/v,n<0);}
fn term(request:vec4i,u:vec3f,r:f32,kind:u32)->f32 {
 if(request.w<0){return 0.0;}
 return ipow(u.x,request.x)*ipow(u.y,request.y)*ipow(u.z,request.z)*cumulative(u32(request.w),r,kind);
}
// Integer superaccumulator: a finite fp32 number is an integer multiple of
// 2^-149. Ten 32-bit limbs cover every finite fp32 exponent plus >2M samples.
// Compiler reassociation cannot erase a floating-point compensation residual.
fn accumulate(limbs:ptr<function,array<u32,10>>,value:f32){
 let bits=bitcast<u32>(abs(value));let biased=(bits>>23u)&255u;
 var mantissa=bits&0x7fffffu;var shift=0u;
 if(biased>0u){mantissa|=0x800000u;shift=biased-1u;}
 let word=shift/32u;let offset=shift%32u;
 var index=word;var carry=mantissa<<offset;
 var high=0u;if(offset>0u){high=mantissa>>(32u-offset);}
 loop{
  if(index>=10u){break;}
  let old=(*limbs)[index];let next=old+carry;(*limbs)[index]=next;
  carry=select(0u,1u,next<old);
  if(index==word){carry+=high;}
  if(carry==0u){break;}index++;
 }
}
fn finish(positive:array<u32,10>,negative:array<u32,10>)->f32 {
 var sign=1.0;var greater=false;var different=false;
 for(var j=9;j>=0;j--){if(positive[u32(j)]!=negative[u32(j)]){greater=positive[u32(j)]>negative[u32(j)];different=true;break;}}
 if(!different){return 0.0;}if(!greater){sign=-1.0;}
 var words:array<u32,10>;var borrow=0u;
 for(var j=0u;j<10u;j++){
  let a=select(negative[j],positive[j],greater);let b=select(positive[j],negative[j],greater);
  let delta=a-b;words[j]=delta-borrow;
  borrow=select(0u,1u,a<b||(borrow!=0u&&delta==0u));
 }
 // Sum positive powers only, after exact cancellation. Each conversion/addition
 // rounds in fp32; this is not an fp64/paired-float evaluation path.
 var result=0.0;
 for(var j=0u;j<10u;j++){if(words[j]!=0u){result+=ldexp(f32(words[j]),i32(32u*j)-149);}}
 return sign*result;
}
@compute @workgroup_size(64)
fn compute_fluxbridge(@builtin(global_invocation_id) id:vec3u){
 let columns=params.counts.x;let index=id.x;if(index>=columns*params.counts.y){return;}
 let column=index%columns;let row=index/columns;let range=ranges[row];
 let a=requests[column*3u];let b=requests[column*3u+1u];let signs=requests[column*3u+2u];
 var positive:array<u32,10>;var negative:array<u32,10>;
 for(var s=range.x;s<range.y;s++){
  let v=samples[2u*s];let w=samples[2u*s+1u];
  let value=w.x*(f32(signs.x)*term(a,v.xyz,v.w,u32(w.y))+f32(signs.y)*term(b,v.xyz,v.w,u32(w.y)));
  if(value>=0.0){accumulate(&positive,value);}else{accumulate(&negative,value);}
 }
 output[index]=finish(positive,negative)*ipow(params.physical.x,signs.z);
}
