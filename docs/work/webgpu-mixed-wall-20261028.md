# Mixed wall WebGPU optimization, 28 October 2026

The Browser/WebGPU provider on `wall.html` now shares radial profiles between
moments and parallelizes long sample ranges. This is enabled automatically.
The numerical integration contract and the displayed material terms remain the
same. The deployment uses versioned entry points, imports and shader URLs so browsers
load a consistent updated asset set. The timing record preserves measured source
hashes and separately records the deployment hashes after URL versioning.

## Implementation

- Each 64-column workgroup loads tiles of 64 samples and evaluates each needed
  radial profile once per sample. The wall's degree-three requests use ten
  profiles for 30 / 74 outputs in 2D / 3D. A mask skips profiles unused by the
  current column group. The original scalar entry point remains available for
  small request sets and matched validation.
- Long ranges are split into jobs of initially 4,096 samples. Jobs write the
  positive/negative ten-limb integer accumulators, and a second compute pass
  merges those integers before the existing fp32 conversion and h scaling.
  No floating-point partial sums or float atomics are introduced. The same
  five-storage-buffer requirement is preserved: partials occupy a checked tail
  of the metadata buffer. Shared workgroup memory is 10,256 bytes.
- The planner increases job size to fit the adapter's buffer and dispatch
  limits, with a 64 MiB per-binding ceiling. If partial storage cannot fit, it
  uses the shared, unchunked kernel. `owned_gpu_peak_bytes` includes actual
  partial storage, uniforms, output and readback; `compute_passes` distinguishes
  two-stage jobs from one-stage batches. Driver allocations are excluded.
- Host preparation writes into a growing Float32Array, with scalar direction
  assembly in the innermost loop. Geometry and intermediate calculations remain
  JavaScript Number; the final sample bits match the previous preparation.
  Whole-element selections and regional merged surfaces are reused between
  convergence rules and between queries with the same element selection.
  Solid and porous interface traces remain separate.

Degree-nine moments, kernel derivatives through order two, all four kernel
families, regular exterior flux, and near-plane Duffy samples are preserved.
The accepted/comparison rules still start at 16/24, increase by eight to 64,
and use the existing 1e-5 normalized regional-moment agreement gate. This does
not turn numerical quadrature into an analytic or certified integration path.

## Matched measurements

[Raw timings, adapter and source hashes](webgpu-mixed-wall-20261028.json) compare
against the original source at Git revision
`c624db745629b274589b0be6f19fbda2098a1c3b`. Dawn/Vulkan selected Mesa llvmpipe,
architecture **software**, LLVM 20.1.2. Three warm repetitions, alternating
before/after order, give these complete wall-evaluation medians:

| Dimension / material degree | Probes | Original | Optimized | Ratio |
|---|---:|---:|---:|---:|
| 2D / 0 | 81 | 51.50 ms | 30.50 ms | 1.69× |
| 2D / 3 | 81 | 82.07 ms | 35.31 ms | 2.32× |
| 3D / 0 | 25 | 5,620.29 ms | 1,034.72 ms | 5.43× |
| 3D / 3 | 25 | 8,322.89 ms | 1,783.35 ms | 4.67× |

These use C2, a 17-degree wall rotation, one subdivision per material region,
and both convergence rules. Allocation, upload, readback, host preparation and
material assembly are included. Device creation is excluded; first-use shader
costs are retained separately in the raw first repetition. All measured
material terms and chosen convergence orders match the original exactly.

For a single 3D rule with 218,616 samples, 74 columns and ten profiles, matched
warm dispatch/upload/readback improves from 572.38 to 60.51 ms (9.46×), and
median host preparation from 67.60 to 22.40 ms (3.02×). This is actual software
Vulkan execution, not hardware GPU throughput. Hardware timing remains open.

## Verification and reproduction

`tests/visualizer/test_fluxbridge_browser.py`: **21 passed, 2 skipped**. The
skips are the existing unsupported 3D disk/fibre geometry contracts. Coverage
includes independent volume fixtures for automatic/shared/chunked dispatch,
all displayed material terms against native fp64, support/refinement checks,
solid fibres and disabled porous response. Additional degree-nine requests use
all 31 generated profiles, multiple column groups, offset/empty rows, partial
sample tiles and mixed ordinary/Duffy samples. Shared and chunked results are
bit-for-bit equal to the scalar shader. Cancellation/overflow cases likewise
pass all three strategies. The six JavaScript test files also pass, including
material-interface geometry reuse and partial-storage/dispatch budgeting.

```bash
WARP_CACHE_PATH=/tmp/curvatureBoundaries-warp-cache \
  python -m pytest tests/visualizer/test_fluxbridge_browser.py -q
node --test tests/visualizer/js/*.test.mjs
python scripts/bench/webgpu_mixed_wall_bench.py \
  --repeats 3 --output /tmp/mixed-wall-benchmark.json
```

The benchmark reconstructs the baseline JavaScript from Git in a temporary
directory, checks identical prepared sample bits, and compares all displayed
terms. It uses the optional Dawn module at
`.tmp/dawn-check/node_modules/webgpu/index.js` or `FLUXBRIDGE_WEBGPU_MODULE`.
GPU measurements serialize with `/tmp/fluxbridge-gpu.lock`.
