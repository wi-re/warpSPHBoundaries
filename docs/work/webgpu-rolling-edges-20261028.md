# WebGPU analytic rolling edge reductions, 28 October 2026

The browser analytic polygon/polyhedron route now applies the same recurrence
organization used by the dedicated native solid triangle/tetrahedron kernels:
consume fixed radial profiles immediately from two rolling parity states,
rather than materializing endpoint primitive arrays and interpreting each
profile at runtime. The host splits constants into two fp32 components and
specializes the shader for the selected dimension and required profiles/outputs.
Axis components sharing those profiles use the same pipeline. Variants
are cached per device (up to 32); geometry, support and grid changes reuse them.
The general pipeline is compiled lazily for local and series models.

`analytic-specialize.js` replaces only the face reduction in the existing WGSL
source. The retained source is also the before-change benchmark implementation.
For each clipped edge panel, the generated body uses definite integrals

- J_m = [u r_u^m - l r_l^m + m c² J_(m-2)] / (m+1),
- C_m = J_(m-2) + a² C_(m-2),

and immediately adds the needed scalar/first/second moment contributions.
Endpoint and plane-offset powers are updated by multiplication. The negative
regular-flux tail remains on all faces; its angle difference is evaluated with
atan2 and a rationalized numerator for endpoints on the same side of zero.
Only actual profile knots split panels (Wendland: support; cubic: half support
and support).

All arithmetic still uses the existing protected paired-fp32 operations and
compiler arithmetic check. This is not a switch to the inaccurate raw-fp32
native diagnostic. Output layouts, physical scaling, boundary validity, and
kernel families C2/C4/C6/cubic are preserved. Local curvature and small-body
series retain their general implementation, as does the separate numerical
FluxBridge degree-nine/porous-moment path.

This ports the recurrence storage, immediate-consumption and specialization
steps. It does not yet share a physical edge's recurrence between both incident
3D faces, prepack all face geometry, or cancel polygon interior edges in the
analytic shader. Those remain possible further optimizations.

## Measured result

[Matched measurements and source hashes](webgpu-rolling-edges-20261028.json)
on Dawn/Vulkan report Mesa llvmpipe, architecture `software` (LLVM 20.1.2).
Three warm repetitions per workload, at resolution 17, give:

| Geometry / kernel | Mass, generic → rolling | Gx, generic → rolling |
|---|---:|---:|
| 2D triangle C2 | 14.36 → 5.30 ms (2.71×) | 13.30 → 4.77 ms (2.79×) |
| 2D triangle C4 | 14.17 → 4.76 ms (2.98×) | 13.97 → 5.28 ms (2.64×) |
| 3D tetrahedron C2 | 139.47 → 68.45 ms (2.04×) | 142.29 → 68.76 ms (2.07×) |
| 3D tetrahedron C4 | 166.71 → 64.02 ms (2.60×) | 147.92 → 74.89 ms (1.98×) |

There are 289 / 4,913 queries per 2D / 3D workload. The worst mixed difference
from the original shader across every output lane is 5.818e-11. These timings
include readback and exclude pipeline compilation and the first dispatch.
New variants take roughly 2–3 seconds to compile on this software stack, with
additional first-dispatch driver compilation; the raw measurements retain both
costs. Caching avoids this cost on reuse and between equivalent axis fields.
Hardware GPU timing remains unverified; no order-of-magnitude claim is made.

Verification on this adapter: all 30 tetrahedron and 13 polygon outputs pass
the existing float64 absolute/relative 3e-6 gates for C2/C4/C6/cubic. The box
second-moment/gradient/Hessian selection and all strip outputs also pass
(`10 passed` in the focused analytic suite). Three local/series fallback cases,
the static exporter, and all five JavaScript test files pass separately.

## Reproduce

Numerical checks use the production host plans and WGSL pipeline selection,
against the existing independent JavaScript float64 reference for every field.
An optional Dawn runtime is preferred; wgpu-native remains a fallback:

```bash
python -m pytest tests/visualizer/test_analytic_gpu.py \
  -k 'polyhedron_every_field or polygon_every_channel or box_analytic or infinite_strip' -q
node --test tests/visualizer/js/*.test.mjs
python scripts/bench/webgpu_analytic_bench.py \
  --resolution 17 --repeats 3 --output /tmp/webgpu-analytic-benchmark.json
```

`FLUXBRIDGE_WEBGPU_MODULE` selects an installed Dawn Node module; the repository
convention is `.tmp/dawn-check/node_modules/webgpu/index.js`. The benchmark
serializes device work with `/tmp/fluxbridge-gpu.lock`. It reports the concrete
adapter, pipeline compilation, cold execution and warm dispatch-plus-readback
with preallocated buffers, and compares all four output lanes against the old
shader. These are not shader-only timestamp measurements. Software Vulkan
results cannot establish hardware GPU throughput or a hardware speedup.

The deployment versions the page/module import chains to refresh browser caches;
the measured analytic shader and specialization source are unchanged.
