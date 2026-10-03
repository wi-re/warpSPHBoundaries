# Profile of one Δ⁺-SPH step (`DeltaSPH2D.step`)

Measured, not estimated (WORK-001 T0.2). `python/edgebound/deltasph_profile.py` monkey-patches the
listed callables with a stack-based timer (`torch.cuda.synchronize()` before and after each call) so
that **inclusive** and **self** times are both reported and nested calls are counted correctly (a
parent's self time excludes its tracked children). The patched callables are restored afterwards; the
solver is not modified. Dam break: 300 warm-up + 200 timed steps. Sloshing: 100 warm-up + 100 timed.
Plus a `torch.profiler` run of 20 dam-break steps.

Command:
```
cd python && python -m edgebound.deltasph_profile
```
Run on 2026-10-03, RTX PRO 6000 (torch 2.13.0+cu130, warp 1.17.0), no other GPU job of ours running
(~28 GiB of the 96 GiB free; the local LLM's ~69 GiB is untouched). Total wall 57 s.

The 13 tracked names: `DeltaSPH2D.{step,rhs,shift,no_penetration,_wall_data,_surface_state,
_detect_surface,_solid_samples}`, `deltasph2d.{neighbor_pairs,sceneOperation}`, `Scene.{buildAdjacency,
inside,signed_distance}`. **Sum of the self times of the tracked names = 100.0 % of the measured step
time in both cases** (the tracked names' bodies cover the whole step, since `step` itself is tracked).

## Dam break — `marrone_dambreak(nx=67)`, `shifting=True`, `noPen="impulse"`

N particles = 3240, dx = 0.01493, H = 0.05970, 1 body. **44.383 ms/step** (timer; wall 44.387).

| name | calls/step | inclusive ms/step | self ms/step | % of step |
|---|---:|---:|---:|---:|
| `DeltaSPH2D.step` | 1.000 | 44.3826 | 0.5388 | 100.00 |
| `DeltaSPH2D.rhs` | 2.000 | 24.6485 | 2.0994 | 55.54 |
| `DeltaSPH2D.shift` | 1.000 | 13.6269 | 2.1522 | 30.70 |
| `DeltaSPH2D.no_penetration` | 1.000 | 5.5684 | 0.1883 | 12.55 |
| `DeltaSPH2D._wall_data` | 3.000 | 14.0763 | 0.4277 | 31.72 |
| `DeltaSPH2D._surface_state` | 1.000 | 11.4748 | 0.3986 | 25.85 |
| `DeltaSPH2D._detect_surface` | 3.000 | 1.3601 | 1.3601 | 3.06 |
| `DeltaSPH2D._solid_samples` | 3.000 | 16.3131 | 0.3188 | 36.76 |
| `deltasph2d.neighbor_pairs` | 3.000 | 2.0918 | 2.0918 | 4.71 |
| `deltasph2d.sceneOperation` | 12.000 | 1.4691 | 1.4691 | 3.31 |
| `Scene.buildAdjacency` | 4.000 | 16.6334 | 16.6334 | 37.48 |
| `Scene.inside` | 3.000 | 15.9943 | 15.9943 | 36.04 |
| `Scene.signed_distance` | 1.000 | 0.7101 | 0.7101 | 1.60 |

("% of step" = inclusive / step-inclusive; nested names overlap, so the column does not sum to 100. The **self** column is the non-overlapping cost and sums to 100 %.)

## Sloshing — `sloshing_tank(nx=200)`, `shifting=True`, `noPen="impulse"`

N particles = 4200, dx = 0.00450, H = 0.01800, 1 body. **72.095 ms/step** (timer; wall 72.100).

| name | calls/step | inclusive ms/step | self ms/step | % of step |
|---|---:|---:|---:|---:|
| `DeltaSPH2D.step` | 1.000 | 72.0946 | 1.1122 | 100.00 |
| `DeltaSPH2D.rhs` | 2.000 | 41.6893 | 2.6050 | 57.83 |
| `DeltaSPH2D.shift` | 1.000 | 22.2852 | 2.6327 | 30.91 |
| `DeltaSPH2D.no_penetration` | 1.000 | 7.0079 | 0.1957 | 9.72 |
| `DeltaSPH2D._wall_data` | 3.000 | 18.1502 | 0.4433 | 25.18 |
| `DeltaSPH2D._surface_state` | 1.000 | 19.6525 | 0.4495 | 27.26 |
| `DeltaSPH2D._detect_surface` | 3.000 | 1.6811 | 1.6811 | 2.33 |
| `DeltaSPH2D._solid_samples` | 3.000 | 34.9237 | 0.3656 | 48.44 |
| `deltasph2d.neighbor_pairs` | 3.000 | 3.7884 | 3.7884 | 5.25 |
| `deltasph2d.sceneOperation` | 12.000 | 1.4818 | 1.4818 | 2.06 |
| `Scene.buildAdjacency` | 4.000 | 22.0634 | 22.0634 | 30.60 |
| `Scene.inside` | 3.000 | 34.5581 | 34.5581 | 47.93 |
| `Scene.signed_distance` | 1.000 | 0.7176 | 0.7176 | 1.00 |

## Answers (from the measurements)

**(a) `buildAdjacency` and scene operations per step.** Both cases: **4** `buildAdjacency` calls/step
and **12** `sceneOperation` calls/step. This **confirms the plan's claim** (4 and ~12). The four
`buildAdjacency` = 2 in `rhs` (once per RHS call, inside `_wall_data`) + 1 in `shift` (inside
`_surface_state`) + 1 in `no_penetration` (inside `_wall_data`); the 12 `sceneOperation` = 3 per
`_wall_data` (λ, G, A) × 3 calls + 3 per `_surface_state` (λ, G, Mw).

**(b) `Scene.inside` traffic.** Dam break: **3,234,816** particle positions through `Scene.inside`
per step (3 calls × ~468 near-wall particles × 24×96 polar samples), **36.04 %** of the step.
Sloshing: **6,469,632** positions/step, **47.93 %** of the step. This polar `inside` evaluation is a
single large cost in both cases (it is the `24×96` `_solid_samples` sampling of the solid that T0.2 is
measuring, not to be touched in this work document).

**(c) Fluid pair sums** (`rhs` minus its scene/detector children, i.e. `neighbor_pairs` + the torch
pair reductions). Dam break: **3.4910 ms/step = 7.87 %** of the step. Sloshing: **5.1402 ms/step =
7.13 %**. The fluid–fluid pair work is a small fraction; the step is dominated by the scene layer.

**(d) Against the stored series.** `dambreak_B_nx67_series.npz`: `wall = 1222.02 s`, `steps = 19533`
→ **62.562 ms/step**. Profiled dam-break step = **44.383 ms/step**. **Ratio profiled / series =
0.709.** (Reviewer's note: the reason for the gap — the validation run also evaluates probes every 10 steps, and
the machine load differed — was not measured; the reviewer's `deltasph_regress check` re-run took 342 s for the
same 6683 steps that took 857 s at record time, so wall time on this machine varies by more than 2x with GPU load.
Compare ms/step only between runs made back to back.) The "sum of self times = 100 %" statement above holds by construction (`step` is tracked), it is not a check; the check
is timer 44.383 ms vs wall 44.387 ms.

**(e) Three largest single costs** (self ms/step). Dam break: `Scene.buildAdjacency` = 16.6334, `Scene.inside`
= 15.9943, `DeltaSPH2D.shift` = 2.1522. Sloshing: `Scene.inside` = 34.5581, `Scene.buildAdjacency` =
22.0634, `deltasph2d.neighbor_pairs` = 3.7884. In both cases the two scene costs (`buildAdjacency`,
`Scene.inside`) are by far the largest; together they are ~73 % (dam break) / ~79 % (sloshing) of the step.

## `torch.profiler` — 20 dam-break steps

Top 15 by **CUDA** time (`key_averages().table(sort_by="cuda_time_total")`):

| name | Self CUDA | Self CUDA % | CUDA total | # calls |
|---|---:|---:|---:|---:|
| `aten::index` | 275.275 ms | 42.22 % | 277.304 ms | 5660 |
| `vectorized_gather_kernel<16,long>` | 262.246 ms | 40.23 % | 262.246 ms | 1940 |
| `_edge_channels_kernel_..._cuda_kernel_forward` | 152.765 ms | 23.43 % | 152.765 ms | 80 |
| `aten::matmul` | 0 | 0 % | 69.967 ms | 620 |
| `aten::mm` | 64.376 ms | 9.87 % | 64.381 ms | 480 |
| `cutlass_80_tensorop_d884gemm_3...` | 46.702 ms | 7.16 % | 46.702 ms | 220 |
| `aten::cdist` | 0 | 0 % | 24.444 ms | 60 |
| `aten::_euclidean_dist` | 0 | 0 % | 24.444 ms | 60 |
| `aten::repeat_interleave` | 6.736 ms | 1.03 % | 22.947 ms | 960 |
| `aten::linalg_pinv` | 0 | 0 % | 19.943 ms | 20 |
| `aten::sub` | 18.785 ms | 2.88 % | 18.785 ms | 4140 |
| `aten::svd` | 0 | 0 % | 18.156 ms | 20 |
| `aten::linalg_svd` | 0 | 0 % | 18.156 ms | 20 |
| `aten::_linalg_svd` | 18.049 ms | 2.77 % | 18.156 ms | 20 |
| `batched_svd_parallel_jacobi_32x16<...>` | 17.993 ms | 2.76 % | 17.993 ms | 20 |

Self CPU total 742.431 ms; Self CUDA total 651.933 ms (20 steps → ~32.6 ms CUDA + 37.1 ms CPU/step).

Top 15 by **CPU** time (`key_averages().table(sort_by="cpu_time_total")`):

| name | Self CPU | Self CPU % | CPU total | # calls |
|---|---:|---:|---:|---:|
| `cudaStreamSynchronize` | 335.153 ms | 45.14 % | 335.153 ms | 2700 |
| `aten::nonzero` | 9.114 ms | 1.23 % | 293.634 ms | 720 |
| `aten::index` | 23.658 ms | 3.19 % | 290.187 ms | 5660 |
| `aten::repeat_interleave` | 4.189 ms | 0.56 % | 138.605 ms | 960 |
| `cudaLaunchKernel` | 103.419 ms | 13.93 % | 108.620 ms | 48460 |
| `aten::item` | 938.194 us | 0.13 % | 65.334 ms | 1480 |
| `aten::_local_scalar_dense` | 2.496 ms | 0.34 % | 64.396 ms | 1480 |
| `aten::to` | 1.836 ms | 0.25 % | 41.174 ms | 6440 |
| `aten::_to_copy` | 3.231 ms | 0.44 % | 39.338 ms | 2500 |
| `aten::nonzero_numpy` | 80.582 us | 0.01 % | 35.138 ms | 60 |
| `aten::copy_` | 5.460 ms | 0.74 % | 34.402 ms | 3140 |
| `cudaMemcpyAsync` | 33.194 ms | 4.47 % | 33.194 ms | 5380 |
| `aten::mul` | 17.684 ms | 2.38 % | 32.246 ms | 6420 |
| `aten::linalg_pinv` | 299.221 us | 0.04 % | 21.885 ms | 20 |
| `aten::sub` | 11.933 ms | 1.61 % | 20.478 ms | 4140 |

The `cudaStreamSynchronize` (45 %) / `cudaLaunchKernel` (14 %) CPU entries are the launch/sync
overhead of the many small kernels (48,460 kernel launches over 20 steps ≈ 2423/step); the heavy CUDA
work is the `aten::index`/gather pair reductions and the scene `_edge_channels_kernel`.
