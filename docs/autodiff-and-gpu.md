# Stages 4 and 5: torch autodiff, explicit adjoints, Warp kernels (2D)

**Status:** [V] — `src/warpSPHBoundaries/edge/torch2d.py`, `warp2d.py`, `np2d.shape_gradient`; tests `tests/edge/test_torch2d.py`, `test_warp2d.py`, `test_shape_gradient.py`

## 1. The analytic derivatives (the primary implementation)

| derivative | formula | primitive cost | check |
|---|---|---|---|
| `∂_x m_α` | `−Σ_e n_e ∫_chord y^α W ds` (edge-gradient identity) | the chord primitives, no atan / indicator | stage-1 FD, polar oracle (`edge-gradient-identity.md`) |
| `∂m_α/∂(vertex)` — **shape derivative** | Reynolds: only the boundary moves, `dm = Σ_e ∫_e y^α W (v·n_e) ds`, edge velocity `v(τ) = (1−τ)δp + τδq`, `τ = (s−s_0)/ℓ`; hence `∂m/∂p_e = n_e ∫(1−τ) y^α W ds`, `∂m/∂q_e = n_e ∫ τ y^α W ds` | the same `S_{j,n}` primitives with one extra power of `s`; **no extra transcendental function** | exact 4th-order FD of the 40-digit stage 1 (`test_shape_gradient.py`, all `α ≤ (2,1)`, cubic & w4, 1e-11), translation identity `Σ_k ∂/∂v_k = −∇_x` (value, 1e-11), orientation/`h` scaling, float32 stable (5e-5) |

The shape derivative is edge-local (only the 2 edges adjacent to a vertex contribute), so no mesh-wide coupling and no differentiation through
clipping logic — exactly the "derive by hand, implement as custom adjoint" plan of `backends-and-verification.md`.

## 2. torch (stage 4) — autograd as an independent check

`torch2d` is the same algorithm with the same exact compiled plans (float64/float32, CPU/CUDA, differentiable). Findings:

* **Autograd equals the analytic gradient and shape adjoint** (cubic 1e-15, w6 ~1e-12; Hessian vs FD of the analytic gradient 1e-7). The reason is the
  structure of the formula: every discontinuous piece (clipping, indicator, atan jump) cancels in the total, and chord end-point terms are multiplied by
  integrands that vanish there (the potentials vanish at `R`).
* **NaN safety** needs three guards: `sqrt` at 0 (safe-sqrt with double `where`), `z = 0` in `asinh(s/|z|)` (double `where`), `atan2(0, 0)`.
* **The one real trap — exactly `z = 0`:** the angle term `atan(hi/z) − atan(lo/z)` is a jump; forcing it to 0 at `z = 0` makes plain autograd drop an `O(1)`
  term `−1/hi + 1/lo` of `d(total)/dz`. Fixed by a custom backward (`_AngleDiff`) with the exact partials `∂_u atan(u/z) = z/(z²+u²)`,
  `∂_z = −u/(z²+u²)`, valid at `z = 0` too. With it, autograd equals the analytic derivative at: `x` on an edge interior, `x` on an edge line outside the segment,
  `z = ±1e-12`, an edge grazing the support circle, an edge tangent to the cubic knot circle, a tiny element with `x` inside, a far element, an engulfing element
  (finite, 1e-9).
* **Known limitation (documented, tested):** with `x` exactly **at a vertex**, the value *is* differentiable w.r.t. that vertex (the swept sliver has bounded `W`), but
  the indicator (`angle/2π`) and the angle terms are constant for autograd, so autograd misses the vertex derivative (the `x`-derivative is right). The analytic adjoint is
  correct there (verified against exact FD, 1e-7). Production code uses the analytic adjoint; autograd is the check.
* float32 autograd is finite and agrees to the float32 closed-form accuracy (5e-3 for gradients, stage 3).

## 3. Warp (stage 5)

`warp2d`: triangles, float64, CPU and CUDA. One generic kernel interprets the exact compiled plan (`np2d.compile_moment`, truncated + inner variants, indicator and atan
weights as exact rationals rounded once) from constant arrays; every thread = one (triangle, point) pair; the primitives are evaluated on the fly with the parity-chain recurrence
(`I_m` needs only `I_{m−2}`; no local arrays). Explicit adjoint kernels: `shape_gradient` (d/d vertices) and `moment_gradient` (d/dx) in one pass.
`TorchMoment` is a `torch.autograd.Function` with forward = Warp and backward = those kernels (equals `torch2d` autograd to 1e-10).

* Accuracy: all 82 convex-triangle fixtures, all four kernels, moments `k ≤ 4` on CPU and CUDA: value 1e-12, gradient 1e-11, moments 1e-12 (same tolerances as numpy f64).
* **Open question resolved:** the installed Warp (1.17) has reverse-mode `wp.Tape` and `wp.autograd.jacobian` (built from backward passes) / `jacobian_fd`; there is **no forward mode**
  (`jvp`) API. Together with the fact that Warp's reverse mode does not handle loop-carried accumulations in dynamic loops robustly, this confirms the design choice:
  explicit analytic adjoint kernels rather than taping the loops.
* Throughput (RTX PRO 6000 Blackwell, float64, `python scripts/bench/warp_bench.py`; 1e6 triangle/point pairs, kernel time, data resident):

  | α | Warp CUDA | numpy (1 thread, extrapolated) | torch CUDA eager | Warp vs numpy |
  |---|---|---|---|---|
  | (0,0) value | 0.032 s | 3.90 s | 0.058 s | 123× |
  | (1,1) | 0.065 s | 2.73 s | 0.056 s | 42× |
  | (2,2) | 0.33 s | 10.4 s | 0.19 s | 32× |

  (≈ 3·10⁷ value evaluations/s in float64 on a consumer-class FP64 rate.) The plan interpreter re-runs the `I_m` recurrence per `(j, n)` term, so for `k ≥ 2` it is
  *slower* than torch eager; a generated, unrolled kernel per `(kernel, α)` or sharing `I_m` across terms is the obvious optimisation, not done here.

## 4. Not done / next

* float32 on the GPU (needs the Chebyshev stable path of stage 3 ported to Warp; the float32 closed form is limited to ~1e-4…1e-3).
* Warp kernels for the FEM nodal weights (needs the Gauss far-field branch) and for non-triangular polygons.
* Shape derivative of the FEM weights (the nodal basis depends on the vertices too).
