// Front-to-back emission/absorption ray casting. rgba16float stores
// (field, signed_distance/h, validity, 0); CPU probe data remains float32.
struct Settings {
  eye: vec4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  transfer: vec4f, // window min/max, density, ray steps
  mode: vec4f, // volume enabled, slice opacity, support band mask, lighting
  offsets: vec4f, // plane positions in centred unit cube
  planes: vec4f, // YZ, XZ, XY enable flags
}
@group(0) @binding(0) var<uniform> s: Settings;
@group(0) @binding(1) var field: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var transferLUT: texture_2d<f32>;

struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) id: u32) -> VertexOutput {
  var vertices = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var out: VertexOutput;
  out.position = vec4f(vertices[id],0,1);
  out.uv = vertices[id];
  return out;
}
fn volumeAt(p: vec3f) -> vec4f {
  let dims = vec3f(textureDimensions(field));
  let uv = ((p+0.5)*(dims-1.0)+0.5)/dims;
  return textureSampleLevel(field,linearSampler,uv,0.0);
}
fn mapValue(value: f32) -> vec4f {
  let u = clamp((value-s.transfer.x)/(s.transfer.y-s.transfer.x),0.0,1.0);
  return textureSampleLevel(transferLUT,linearSampler,vec2f((u*255.0+0.5)/256.0,0.5),0.0);
}
fn acceptable(sample: vec4f) -> bool {
  return sample.b > 0.999 && (s.mode.z < 0.5 || abs(sample.g) <= 1.0);
}
fn illumination(p: vec3f, base: vec3f) -> vec3f {
  let e = 1.0 / vec3f(textureDimensions(field));
  let gradient = vec3f(volumeAt(p+vec3f(e.x,0,0)).r-volumeAt(p-vec3f(e.x,0,0)).r,
                       volumeAt(p+vec3f(0,e.y,0)).r-volumeAt(p-vec3f(0,e.y,0)).r,
                       volumeAt(p+vec3f(0,0,e.z)).r-volumeAt(p-vec3f(0,0,e.z)).r);
  let normal = gradient / max(length(gradient),0.000001);
  let light = 0.5+0.5*abs(dot(normal,normalize(vec3f(0.4,0.7,1.0))));
  return base*light;
}
@fragment fn fragmentMain(vout: VertexOutput) -> @location(0) vec4f {
  let background = mix(vec3f(0.033,0.053,0.071),vec3f(0.075,0.11,0.137),(vout.uv.y+1.0)*0.5);
  let origin = s.eye.xyz;
  let direction = normalize(s.forward.xyz + 0.48*vout.uv.x*s.eye.w*s.right.xyz + 0.48*vout.uv.y*s.up.xyz);
  let safeDirection = select(vec3f(0.000001),direction,abs(direction)>vec3f(0.000001));
  let a = (-vec3f(0.5)-origin)/safeDirection;
  let b = (vec3f(0.5)-origin)/safeDirection;
  let near = min(a,b);
  let far = max(a,b);
  let begin = max(max(near.x,max(near.y,near.z)),0.0);
  let end = min(far.x,min(far.y,far.z));
  if end <= begin { return vec4f(background,1); }
  // At least two samples per voxel along the ray, including diagonal views.
  let steps = max(u32(s.transfer.w),u32(ceil((end-begin)*f32(textureDimensions(field).x-1u)*2.0)));
  let dt = (end-begin)/f32(steps);
  var planeT = vec3f(end+1.0);
  for(var axis=0u;axis<3u;axis++) {
    if s.planes[axis] > 0.5 && abs(direction[axis])>0.000001 {
      let t = (s.offsets[axis]-origin[axis])/direction[axis];
      if t >= begin && t <= end { planeT[axis]=t; }
    }
  }
  var accumulated = vec4f(0);
  for(var i=0u;i<1024u;i++) {
    if i >= steps || accumulated.a > 0.995 { break; }
    let left = begin+f32(i)*dt;
    let right = min(end,begin+f32(i+1u)*dt);
    var cursor=left;
    // Three planes can intersect the same step. Process nearest first and
    // integrate the volume on each side, so a plane never erases a nearer shell.
    for(var event=0u;event<4u;event++) {
      let next=min(planeT.x,min(planeT.y,planeT.z));
      let stop=min(right,next);
      if s.mode.x > 0.5 && stop > cursor {
        let p=origin+direction*((cursor+stop)*0.5);
        let sample=volumeAt(p);
        if acceptable(sample) {
          let mapped=mapValue(sample.r);
          let alpha=1.0-pow(max(1.0-mapped.a,0.000001),s.transfer.z*(stop-cursor)*128.0);
          var rgb=mapped.rgb;
          if s.mode.w > 0.5 && alpha > 0.001 { rgb=illumination(p,rgb); }
          accumulated=accumulated+(1.0-accumulated.a)*vec4f(rgb*alpha,alpha);
        }
      }
      cursor=stop;
      if next > right { break; }
      let sample=volumeAt(origin+direction*next);
      if acceptable(sample) {
        let mapped=mapValue(sample.r);
        let alpha=s.mode.y*min(1.0,mapped.a*8.0);
        accumulated=accumulated+(1.0-accumulated.a)*vec4f(mapped.rgb*alpha,alpha);
      }
      if planeT.x==next { planeT.x=end+1.0; }
      else if planeT.y==next { planeT.y=end+1.0; }
      else { planeT.z=end+1.0; }
    }
  }
  return vec4f(accumulated.rgb+(1.0-accumulated.a)*background,1);
}
