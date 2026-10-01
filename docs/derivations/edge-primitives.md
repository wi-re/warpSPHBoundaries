# Closed-form edge primitives

**Status:** [V] Maple (symbolic, `maple/10_edge_primitives.mpl`) + mpmath (`tests/edge/test_primitives.py`)
**Tier(s):** 1, 2, 4 · **Dimension:** 2D (3D adds a new family: `tet-face-edge-chain.md`)
**Depends on:** `../notation.md`
**Implemented in:** `python/edgebound/primitives.py` (mpmath stage 1); the unverified `monomial_edge_forms.py` is superseded
**Verified by:** `maple/10_edge_primitives.mpl`, `tests/edge/test_primitives.py`

## 1. Statement

With `r = sqrt(s² + z²)`:

- `I_m(s) = ∫ r^m ds` — the **odd** antiderivative (`I_m(0) = 0`):
  `I_0 = s`, `I_{-1} = asinh(s/|z|)`, `I_{-2} = arctan(s/z)/z` (only ever appears as
  `z I_{-2} = arctan(s/z)`), recurrence `(m+1) I_m = s r^m + m z² I_{m-2}`.
  - even `m = 2k ≥ 0`: polynomial, `I_{2k} = Σ_i C(k,i) z^{2k-2i} s^{2i+1}/(2i+1)`;
  - odd `m ≥ 1`: `I_m = (m!!/(m+1)!!) z^{m+1} asinh(s/|z|) + s·r·poly(s², z²)`.
- `J_m(s) = ∫ s r^m ds = r^{m+2}/(m+2)`, `J_{-2} = ln r` (**even** in `s`, see §6).
- `S_{j,m}(s) = ∫ s^j r^m ds` via `s² = r² − z²`:
  `j` even → `Σ_i C(j/2,i) (−z²)^{j/2−i} I_{m+2i}`, `j` odd → same with `J_{m+2i}`.
- Angle difference over a chord (no cancellation): `arctan(hi/z) − arctan(lo/z) = atan2(z(hi−lo), z² + hi·lo)`
  (valid for either sign of `z`; convention `0` at `z = 0`).

## 2. Assumptions and validity

`m ≥ 0` is all the production path needs (value, gradient, recursion (a), cubic/Wendland
kernels). `z = 0` is legal for every `m ≥ 0` (`z² asinh(s/|z|) := 0`, `I_m(z=0) = s|s|^m/(m+1)`).
`m < 0` (`Ig`) is only used by the far-field cross-check (b) and needs `z ≠ 0`.

## 3. Derivation

Differentiate each antiderivative and compare with the integrand (`diff` in Maple, derivative
taken *outside* `assuming`, see gotchas in `PLAN.md`). The recurrence is the identity
`d/ds[s r^m] = (m+1) r^m − m z² r^{m-2}`. The odd-in-`s` property follows by induction from
the base cases. No step is unchecked.

## 4. Special cases and limits

- `z → 0`: `z² I_{-1} → 0`, `z I_{-2} = arctan(s/z) → ±π/2` (sign `s·z`), odd-`m` `I_m → s^{m+1}/(m+1)`
  (`s > 0`), all verified symbolically (`limit`).
- chords spanning `s = 0`: odd antiderivatives add, they do not cancel.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| `d/ds I_m = r^m`, `m = −12…12`, `z > 0` and `z = −w < 0` (with `|z|`) | Maple symbolic | exact | [V] |
| recurrence `(m+1)I_m = s r^m + m z² I_{m−2}` (`m = −1…12`), constant = 0 | Maple symbolic | exact | [V] |
| `I_m` odd in `s`, `m = −10…12` | Maple symbolic | exact | [V] |
| closed forms: even `m` polynomial, odd `m` asinh tail with `m!!/(m+1)!!` | Maple symbolic | exact | [V] |
| `d/ds J_m = s r^m`, `m = −2…12` | Maple symbolic | exact | [V] |
| `d/ds S_{j,m} = s^j r^m`, `j = 0…7`, `m = −8…10` (152 cases) | Maple symbolic | exact | [V] |
| `S_{j,m}` definite integrals, chords above / spanning / below 0 | Maple `evalf(Int)` 40 digits | 1e-30 | [V] |
| `z → 0` limits (above) | Maple `limit` | exact | [V] |
| `I_m` derivative / definite integral vs mpmath `quad`, random `z, s0, s1`, `z` down to 1e-9 | `tests/edge/test_primitives.py` | 1e-28…1e-30 | [V] |
| `S_{j,m}` (also `m < 0`, `z = 3/2`) vs `quad` | same | 1e-30 | [V] |
| `dangle` equals `atan` difference and keeps *relative* accuracy for a 1e-15 chord at `z = 1e-20` | same | 1e-25 rel | [V] |
| `z = 0` guard (`I_m(s,0) = s|s|^m/(m+1)`) | same | 1e-38 | [V] |

## 6. Known failure modes

- **Short chords.** `F(hi) − F(lo)` of a closed form cancels when `hi ≈ lo`. The angle term is
  fixed exactly by `atan2` (above). For the polynomial / `asinh` parts the 53-bit emulation
  (`python -m edgebound.precision_probe`, table in `../backends-and-verification.md`) shows the
  *absolute* error stays at the 1e-14 level for chords down to 1e-13 (the integrand there is
  tiny, so the *absolute* error is what matters). The mpmath stage simply carries 20 guard digits.
- **Downward recurrence** (`m < −2`, only for (b)) divides by `z²` at every step: at `z = 1e-9`,
  `m = −8` it returns garbage at 40 digits (measured: derivative test returns 0 instead of 6561).
  Never use it in production; (b) is a cross-check at moderate `z`.
- `J_m` is *even* in `s` (the HANDOFF text says "odd antiderivatives" — true for `I_m` only).
  `J(hi) − J(lo)` over a chord spanning `s = 0` is an honest difference of same-sign numbers;
  for `m = 2k` it can be written exactly as `(hi² − lo²)·poly`, which is what a float
  implementation should do.
- float32 loses digits earlier (stage 3).

## 7. Open questions

Best *float* evaluation of `I_m` differences for odd `m` and large `m` (the atan2 trick is
only for the angle term); the per-edge cost model for GPU.
