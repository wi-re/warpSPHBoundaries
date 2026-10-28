// Generic degree-nine volume moments / kernel-only derivatives, ordinary fp32.
// Directions/radii/weights are host-prepared. Each invocation owns one output.
struct Params { counts:vec4u, physical:vec4f, work:vec4u }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> samples:array<vec4f>;
@group(0) @binding(2) var<storage,read> profiles:array<f32>;
// Header: requests/profile map/masks. Chunked evaluation writes exact integer
// partials after that header, keeping the same five-storage-buffer contract.
@group(0) @binding(3) var<storage,read_write> requests:array<vec4i>;
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

// Each radial profile is evaluated once per sample/column group. The sample
// products and exact integer summation are the same as the scalar entry point.
// 64 samples * 32 profiles plus samples/range use 10,256 shared bytes.
var<workgroup> tileSamples:array<vec4f,128>;
var<workgroup> tileProfiles:array<f32,2048>;
var<workgroup> sharedRange:vec4u;
fn cachedTerm(request:vec4i,u:vec3f,sample:u32)->f32 {
 if(request.w<0){return 0.0;}
 return ipow(u.x,request.x)*ipow(u.y,request.y)*ipow(u.z,request.z)*tileProfiles[u32(request.w)*64u+sample];
}
fn sharedEvaluation(group:vec3u,local:vec3u,chunked:bool){
 let columns=params.counts.x;let groups=(columns+63u)/64u;
 let unit=group.x/groups;let columnGroup=group.x%groups;let column=columnGroup*64u+local.x;
 let mask=bitcast<u32>(requests[columns*3u+(params.counts.w+3u)/4u+columnGroup/4u][columnGroup%4u]);
 if(local.x==0u){sharedRange=ranges[select(unit,params.counts.y+unit,chunked)];}
 workgroupBarrier();let range=workgroupUniformLoad(&sharedRange);
 var a=vec4i(0);var b=vec4i(0);var signs=vec4i(0);
 if(column<columns){a=requests[column*3u];b=requests[column*3u+1u];signs=requests[column*3u+2u];}
 var positive:array<u32,10>;var negative:array<u32,10>;
 for(var start=range.x;start<range.y;start+=64u){
  let count=min(64u,range.y-start);
  if(local.x<count){tileSamples[2u*local.x]=samples[2u*(start+local.x)];tileSamples[2u*local.x+1u]=samples[2u*(start+local.x)+1u];}
  workgroupBarrier();
  for(var slot=local.x;slot<params.counts.w*64u;slot+=64u){
   let sample=slot%64u;let profile=slot/64u;
   if(sample<count && (mask&(1u<<profile))!=0u){
    let original=u32(requests[columns*3u+profile/4u][profile%4u]);
    tileProfiles[slot]=cumulative(original,tileSamples[2u*sample].w,u32(tileSamples[2u*sample+1u].y));
   }
  }
  workgroupBarrier();
  if(column<columns){for(var sample=0u;sample<count;sample++){
   let v=tileSamples[2u*sample];let w=tileSamples[2u*sample+1u];
   let value=w.x*(f32(signs.x)*cachedTerm(a,v.xyz,sample)+f32(signs.y)*cachedTerm(b,v.xyz,sample));
   if(value>=0.0){accumulate(&positive,value);}else{accumulate(&negative,value);}
  }}
  workgroupBarrier();
 }
 if(column<columns){
  if(chunked){
   let base=params.work.y+(unit*columns+column)*5u;
   requests[base]=bitcast<vec4i>(vec4u(positive[0],positive[1],positive[2],positive[3]));
   requests[base+1u]=bitcast<vec4i>(vec4u(positive[4],positive[5],positive[6],positive[7]));
   requests[base+2u]=bitcast<vec4i>(vec4u(positive[8],positive[9],negative[0],negative[1]));
   requests[base+3u]=bitcast<vec4i>(vec4u(negative[2],negative[3],negative[4],negative[5]));
   requests[base+4u]=bitcast<vec4i>(vec4u(negative[6],negative[7],negative[8],negative[9]));
  }else{output[unit*columns+column]=finish(positive,negative)*ipow(params.physical.x,signs.z);}
 }
}
@compute @workgroup_size(64)
fn compute_fluxbridge_shared(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u){sharedEvaluation(group,local,false);}
@compute @workgroup_size(64)
fn compute_fluxbridge_chunked(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u){sharedEvaluation(group,local,true);}
@compute @workgroup_size(64)
fn reduce_fluxbridge_chunks(@builtin(global_invocation_id) id:vec3u){
 let columns=params.counts.x;let index=id.x;if(index>=columns*params.counts.y){return;}
 let column=index%columns;let row=index/columns;let range=ranges[row];
 var positive:array<u32,10>;var negative:array<u32,10>;
 for(var chunk=range.z;chunk<range.w;chunk++){
  let base=params.work.y+(chunk*columns+column)*5u;var pc=0u;var nc=0u;
  for(var word=0u;word<10u;word++){
   let p=bitcast<u32>(requests[base+word/4u][word%4u]);let nword=word+10u;
   let n=bitcast<u32>(requests[base+nword/4u][nword%4u]);
   let ps=positive[word]+p;let pn=ps+pc;pc=select(0u,1u,ps<positive[word]||pn<ps);positive[word]=pn;
   let ns=negative[word]+n;let nn=ns+nc;nc=select(0u,1u,ns<negative[word]||nn<ns);negative[word]=nn;
  }
 }
 output[index]=finish(positive,negative)*ipow(params.physical.x,requests[column*3u+2u].z);
}
