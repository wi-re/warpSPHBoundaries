# Edge reduction of the value integral (2D)

**Status:** [V] identity checked numerically (Wendland C4, 20 random triangle/point pairs, max abs diff 3e-13, quadrature-limited); Maple proof of the divergence step [P]
**Tier(s):** 1, 2 · **Dimension:** 2D (3D version: `tet-face-edge-chain.md`)
**Depends on:** `../notation.md`
**Implemented in:** `edge_identity_check.py` (quadrature edge integrals); closed-form edge integrals: `monomial_edge_forms.py` [D]
**Verified by:** `edge_identity_check.py`; planned: `tests/test_edge_value.py`

## 1. Statement

For a triangle (or any polygon) `T` and normalised radial kernel `W` with
support 1, `M(r) = ∫_0^r t W(t) dt`, `2π M(1) = 1`:

$$\int_T W(|x-x'|)\,dA' \;=\; \mathbb 1[x\in T] \;+\; \sum_e z_e \int_{\text{chord}_e}\frac{M(r)-M(1)}{r^2}\,ds .$$

## 2. Assumptions and validity

- `W` radial, compactly supported on `r <= 1`, `M` continuous.
- Polygon edges are straight; `T` simple and consistently oriented.
- `x` not exactly on an edge line with `z_e = 0` and chord non-empty
  (degenerate: half-plane limit, see §4).

## 3. Derivation

1. `∇·( y M(r)/r² ) = W(r)` in 2D (`y = x'-x`).
2. Divergence theorem on `T` (the singularity of the field at `x` is
   integrable because `M(r) ~ r²` as `r → 0`, so no small-disk term appears):
   `∫_T W = ∮_{∂T} n·y M(r)/r² ds = Σ_e z_e ∫_edge M(r)/r² ds`, using
   `n_e·y = z_e` on the edge line.
3. Split `M = (M - M(1)) + M(1)`. The first part is compact (zero for
   `r >= 1`), so its edge integrals are the clipped chord integrals. The
   second part gives `M(1) Σ_e z_e ∫_edge ds/r²`, and `z_e ∫ ds/r² = arctan`
   is the angle subtended by the edge at `x`; summed over a closed polygon this
   is `2π` for `x ∈ T` and `0` otherwise, so the term equals
   `2π M(1) 1[x ∈ T] = 1[x ∈ T]` (normalisation). Edges beyond the support
   therefore contribute nothing explicit: they enter only through the
   indicator.

Steps 2–3 are the part to re-derive carefully in Maple (task 1 of
`HANDOFF.md §12`); the identity itself is numerically verified.

## 4. Special cases and limits

- **Half-plane:** one infinite edge at distance `z = d`: the formula must equal
  the 2D planar closed form `λ_2(d)` of `docs/derivation.md` §3 (all four
  kernels). This is the cross-check linking `PLAN.md`'s planar results to the
  edge machinery. Expected agreement: 1e-30 or better in mpmath.
- `x` on an edge: half-plane limit gives exactly 1/2 of the disk contribution
  (convention to be fixed; see `HANDOFF.md §13`).
- Mesh covering the whole support: sum over all elements = 1.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| random triangles vs area quadrature | `edge_identity_check.py` | 1e-12 | [V] |
| half-plane equals PLAN.md `λ_2(d)` | mpmath, 4 kernels × d grid | 1e-30 | [P] |
| tiled mesh covering support sums to 1 | exact Fractions where possible | 1e-15 (f64) | [P] |
| `x` crossing an edge: total continuous | sweep through edge, no jump | 1e-12 | [P] |
| sub-triangle splitting invariance | split one triangle in 2/4 | 1e-30 mp | [P] |

## 6. Known failure modes

`z_e → 0` (`asinh` primitive diverges like `log|z|` but enters multiplied by
`z²`); tiny chords with both ends of the same sign (difference of nearby
values); orientation predicate mismatch between indicator and `z_e` signs.

## 7. Open questions

Convention exactly on an edge or vertex; best evaluation of short chords.
