# Complete 2D surface moments and conditioned device integration

The explicit `SurfaceRep` path exposes all twelve quantities in the current
2D audit on CPU and CUDA. The [external acceptance matrix](../../warpSPHDevelopment/curvatureBoundaries/progress/plan-2d-accuracy.md) records
finite tested coverage. Scene SDF/half-plane coverage is described below; this
does not complete the moment coverage of
lookup, disk or volume representations.

## Contract

Use y = source − query and W_h(y) = h⁻² W(|y|/h). Gradients act on the query
coordinate of W. All public tensors are in physical units; `RepMoments.toWorld`
and `FusedWall.evaluate` rotate vectors/tensors into the world frame.

| Quantity | Integral | RepMoments field | Fused WallOutput kind | Physical support scaling | Surface channel IDs |
|---|---|---|---|---|---|
| m00 | ∫ W_h | `lam` | `lam` | 1 | 0 |
| m10, m01 | ∫ y W_h | `m1[P,2]` | `m1[B,N,2]` | h | 1,2 |
| Gx, Gy | ∫ ∇query W_h | `g0[P,2]` | `g0[B,N,2]` | 1/h | 3,4 |
| g1xx, g1xy, g1yx, g1yy | ∫ y ⊗ ∇query W_h | `g1[P,2,2]` | `cov[B,N,2,2]` | 1 | 5,6,7,8 |
| m20, m11, m02 | ∫ y ⊗ y W_h | `m2[P,2,2]` | `m2[B,N,2,2]` | h² | 9,10,11 |

IDs 0–8 preserve the existing device layout. The symmetric off-diagonal m11
occupies both off-diagonal entries of `m2`. Reports retain their original order:
m00,m10,m01,m20,m11,m02,Gx,Gy,g1xx,g1xy,g1yx,g1yy.

`scene.precompute(adjacency, props)` requests all twelve surface channels.
An explicit `channels` subset zeros unrequested components; `m2` is None when
none of its three channels is requested. Other representations may have `m2=None`.
The existing body fields and scene operations remain linear in position; exposing
m2 does not add a quadratic-field consumer or g2 gradient moments.

```python
pm = scene.precompute(adjacency, props)
for entry in pm.entries:
    for moments in entry['surface']:
        second = moments.m2              # physical, world-frame tensor

wall = FusedWall(scene, fixed_adjacency(scene, particles, props, support_max),
                 (FusedGroup('w4'),))
result = wall.evaluate((WallOutput('second', 0, 'm2'),))['second']
```

Fused `m2` currently requires explicit SurfaceRep geometry throughout the scene.
Unsupported representation coverage raises rather than substituting zero.

## Analytic and quadrature hierarchy

Resolved surfaces retain the existing exact polynomial edge identities (or the
selected Chebyshev edge quadrature). The added scalar second moments reuse
`compile_moment` with α=(2,0),(1,1),(0,2), through a separate device plan; the
legacy nine-channel buffers retain their layout. Fused second moments follow the
requested monomial or Chebyshev route, including its node/panel configuration. Whole-support second moments
are accumulated from exact rational kernel coefficients. For w4, m20=m02=7/132
at h=1, and m11=0.

A single closed triangle uses the conditioned area rule when its longest edge
is ≤0.02h, or twice-area/longest-edge² is ≤0.01. The device gate is more
conservative than the CPU FEM gate (0.001), covering difficult acute/obtuse edge
quadrature cases as well. These are tested routing thresholds, not certificates.
Multi-loop surfaces, periodic copies and other polygons do not acquire a general
sliver decomposition from this change.

The current device rule supports kernel profiles with at most two radii.
The rule preserves the original geometry. It forms edge differences before
query translation, uses the short edge transversely, splits at analytic
line/circle intersections, and evaluates the regular origin polynomial for
−W′(r)/r. Transverse Gauss order is 32, with a 16-node comparison; outer panels
compare 16 versus 8 nodes and subdivide adaptively. Keeping the inner comparison
out of the outer convergence decision avoids treating shared inner-order bias
as an outer convergence failure. That bias remains visible in the estimate.

`SurfaceRep.polygon(points, conditioned=False)` keeps the edge route available
explicitly, including for difficult triangles. `warpbc.edge_channels` and
`warpbc.second_channels` also expose the separate edge expressions. The CPU
`np_fem.weights` closed form and `weights_hybrid(..., adaptive=False)` remain
available.

## Diagnostics and execution

`RepMoments.quadrature_status` is per query: 0 = edge route/unassessed;
1 = outer area quadrature converged; −1 = panel budget or refinement floor
exhausted. Failed values are NaN, never an accepted zero. Estimates combine
outer and inner-order differences and are empirical, not rigorous error bounds.
`converged` does not establish few-eps accuracy of the inner rule.

`RepMoments.quadrature_estimate[P,12]` uses the legacy nine + three channel
order in physical units. World-frame conversion uses absolute rotation
coefficients to propagate componentwise estimates conservatively. Analytic
rows have status 0; zero stored estimates there make no accuracy claim.
The audit masks those estimates as unavailable.

Fused items expose `qstatus[rows,groups]` and `estimates[group][rows,12]` in
local, normalized support units. Values, estimates, statuses, validity and
query supports stay on the device. Static triangle orientation and exact
float-geometry area are prepared once, like the existing cell structures;
recreate the representation after changing its local geometry. Moving poses
and query positions/supports remain runtime inputs.

Warm static plans and Python-valued kinematics constants before CUDA capture
(or provide device scalar tensors), and use the same stream for Torch and
Warp (`wp.ScopedStream(wp.stream_from_torch(stream))`). Tests cover both capturing
the complete adjacency/evaluation path and replaying a captured contraction
after `FusedWall.copy_from` updates its input buffers, including second moments
and quadrature diagnostics.

Unbounded exterior solids retain far-away queries in both adjacency paths, even
when no boundary edge lies inside the support. The convention is shared by
surfaces, boxes, lowered implicit cavities and SDF exterior fallbacks. Their full-plane
mass and g1 diagonals share the exact rational normalization, preserving the
zero whole-support Laplacian.

## Evidence

- [CUDA Markdown report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/device-conditioning-cuda/report.md)
  and [CPU report](../../warpSPHDevelopment/curvatureBoundaries/results/figures/device-conditioning-cpu/report.md):
  all three executed paths pass 21/21 original fixtures, all twelve channels,
  maximized over four base kernels and the original query catalogue.
- Worst signal-relative error: 1.02e−9 for Scene/fused monomial and 3.95e−9
  for fused Chebyshev on CPU/CUDA. No quadrature failures in these sweeps.
- References use the preserved 70/45-digit targets and finite independent polar
  spot checks. This meets the finite ε32 gate; universal few-ε64 accuracy is open.
- Maple: `maple/18_surface_second_moments.mpl` verifies the compact-potential
  product identities, P2 reproduction and whole-support normalization.
- Audit timings include adjacency, allocations, synchronization and compilation.
  Chebyshev timings include the extra estimator evaluation. They are not
  simulation-throughput benchmarks; performance tuning remains open.

Reproduce with `python scripts/studies/device_conditioning.py --device cuda:0
--output "$EVIDENCE/device-conditioning-cuda"` (one shell command), or use
`--report-only` to redraw the saved measurements. The reports and figure links
are files; they do not require a localhost server.

Generated reports live outside the repository. See [output configuration and
reference prerequisites](validation.md) before running these commands; `EVIDENCE`
denotes the external figures directory. Test logs and progress history are archived there.

Scene also exposes all twelve moments for the corrected SDF local model and
implicit half-planes. See the [curvature-moment contract](derivations/curvature-moments-2d.md).
FusedWall still lowers an SDF to its explicit polygon; direct curved local-model
execution is a Scene path. Other representation coverage remains separate.
