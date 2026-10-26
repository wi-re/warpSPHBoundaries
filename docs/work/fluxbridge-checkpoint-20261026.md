# FluxBridge native checkpoint, 26 October 2026

This checkpoint saves the independent native FluxBridge operators, fixtures,
material/field contracts and shared-edge performance studies before further
analytic-fp32 stability work. It includes the existing `mesh3d` reference as a
required import dependency. Pre-existing paper, scene, browser and research
changes remain outside this commit. No warpSPH/Core/Integrator or solver code
is included; pressure/porosity integration remains Stage 5.

The detailed execution record is [the plan](../plan-porous-coupling.md). Stages
0–4 are not universally accepted: coverage, production dispatch, geometry
conditioning and precision gates remain explicit there. Ordinary solid disks,
cylinder sections and periodic fibres remain independent of porous coupling;
transverse 2D and longitudinal flow are distinct. Böttcher et al. (2025),
arXiv:2504.07739v2, remains the principal porous-flow reference.

## Complete shared-edge analytic W6 prototype

The experimental tetrahedron kernel computes the actual 28-output baseline:
ten value moments through degree two, twelve first-gradient moments, and six
symmetric Hessian entries. Six physical edges share J/C recurrences through
degree 15. There is no quadrature or global primitive/moment scratch. fp64
passes 26 fixtures plus 90 transformed independent cases on CPU/CUDA at mixed
error `abs(actual-reference)/(1+abs(reference)) <= 1e-10`; worst CPU/CUDA error
is 6.624e-12 / 3.002e-12. Register count is 188 with no local allocation/traffic.

This is an isolated prototype, not a replacement for native production dispatch.
The native compact rotated-boundary failures still need a guarded production
fix. Other radial profiles, surface triangles and higher polynomial contracts
are not certified by this W6 baseline experiment.

## Raw fp32: useful speed, failed accuracy

The raw-fp32 full kernel fails the existing 1e-5 gate on CPU and CUDA. CUDA
fails 799/3,248 scalar outputs across 87/116 independent pairs. Worst mixed
errors are 2.750e-5 for occupancy, 2.949e-4 for the gradient vector and
2.103e-3 for Hessians. All outputs are finite. The larger mixed diagnostic
includes a confirmed Hessian sign reversal (+0.00264979 to -0.01040694).

At 1,048,576 pairs, 64-thread blocks, warm CUDA-event medians are:

| Workload | Raw fp32 | Pairs/s | Matched fp64 | Ratio |
|---|---:|---:|---:|---:|
| Canonical right tetrahedron | 0.215648 ms | 4.862 billion | 8.546464 ms | 39.63x |
| Mixed 256 tetrahedra | 0.441472 ms | 2.375 billion | 16.234144 ms | 36.77x |

Graphs yield 5.058 / 2.435 billion pairs/s. These are preallocated device-array
measurements writing all 28 outputs. Allocation/validation/upload/readback cost
about 54 ms for one million fp32 pairs; search, reduction and material assembly
are absent. The raw kernel uses 108 registers, no local allocation/traffic,
about 32% achieved occupancy and zero executed fp64 arithmetic. Serialized GPU
measurements use unlocked clocks/caches and share the desktop device.

On canonical inputs, replaying the same stored fp32 geometry/query buffers in
fp64 reduces worst mixed error from 9.108e-4 to 4.139e-8, indicating substantial
loss of precision during evaluation. Rounded geometry also causes errors in
some transformed boundary cases. The stable fp32 evaluation of radial
potentials/face reductions and coherent geometry are the next investigation.
Local fp64 operations are a last-resort candidate, not the default remedy.

Accepted production policy remains analytic fp64 / numerical fp32. The
reorganization establishes a speed target; it does not close analytic fp32.

## Evidence and reproduction

- [Full fp64 validation and profiling](fluxbridge-full-20261026.json)
- [Independent full-contract references](fluxbridge-full-reference-20261026.json)
- [Raw fp32 errors, matched speed and profiling](fluxbridge-full-fp32-20261026.json)

With the project environment installed, the isolated studies can be rerun:

```bash
python scripts/studies/fluxbridge_shared_jk_full_study.py \
  --reference docs/work/fluxbridge-full-reference-20261026.json --output /tmp/fluxbridge-full-repeat
python scripts/studies/fluxbridge_shared_jk_full_precision_study.py \
  --output /tmp/fluxbridge-fp32-repeat
python scripts/studies/fluxbridge_shared_jk_full_precision_perf.py \
  --output /tmp/fluxbridge-fp32-speed-repeat
```

Use `--device cpu` for CPU-only accuracy studies. Performance studies require
CUDA; GPU drivers serialize using `/tmp/fluxbridge-gpu.lock`. Source hashes and
preservation checks are recorded in each evidence file; raw profiles/logs remain
under the corresponding `/tmp/fluxbridge-*20261026/` directories.

Fresh checkpoint smoke check:
`python -m pytest tests/fluxbridge/test_regular_potential.py tests/fluxbridge/test_native.py -q`
on CPU: **37 passed, 1 CUDA-only skip, 4 expected raw-fp32 failures**. The complete
new full-contract CPU/CUDA checks and measured fp32 failures are separately
recorded in the linked evidence; this smoke check does not replace those gates.
