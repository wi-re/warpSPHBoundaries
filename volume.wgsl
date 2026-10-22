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
  let steps = u32(s.transfer.w);
  let dt = (end-begin)/f32(steps);
  let planeT = (s.offsets.xyz-origin)/safeDirection;
  var accumulated = vec4f(0);
  for(var i=0u;i<768u;i++) {
    if i >= steps || accumulated.a > 0.995 { break; }
    let t = begin + f32(i)*dt;
    let p = origin+direction*(t+dt*0.5);
    if s.mode.x > 0.5 {
      let sample = volumeAt(p);
      if acceptable(sample) {
        let mapped = mapValue(sample.r);
        let alpha = 1.0-pow(max(1.0-mapped.a,0.000001),s.transfer.z*dt*128.0);
        var rgb = mapped.rgb;
        if s.mode.w > 0.5 && alpha > 0.001 { rgb = illumination(p,rgb); }
        accumulated = accumulated + (1.0-accumulated.a)*vec4f(rgb*alpha,alpha);
      }
    }
    for(var axis=0u;axis<3u;axis++) {
      if s.planes[axis] > 0.5 && planeT[axis] >= t && planeT[axis] < t+dt && abs(direction[axis])>0.000001 {
        let sample = volumeAt(origin+direction*planeT[axis]);
        if acceptable(sample) {
          let mapped = mapValue(sample.r);
          let alpha = s.mode.y;
          accumulated = accumulated + (1.0-accumulated.a)*vec4f(mapped.rgb*alpha,alpha);
        }
      }
    }
  }
  return vec4f(accumulated.rgb+(1.0-accumulated.a)*background,1);
}
