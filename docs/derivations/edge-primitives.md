# Closed-form edge primitives

**Status:** [D] implemented in `monomial_edge_forms.py`, not yet verified in Maple
**Tier(s):** 1, 2, 4 · **Dimension:** 2D (3D adds a new family: `tet-face-edge-chain.md`)
**Depends on:** `../notation.md`
**Implemented in:** `monomial_edge_forms.py`
**Verified by:** not yet (task 1 of `HANDOFF.md §12`)

## 1. Statement

With `r = sqrt(s² + z²)` use **odd** antiderivatives (chords spanning `s = 0`
then add, not cancel):

- `I_m(s) = ∫ r^m ds`: `I_0 = s`, `I_{-1} = asinh(s/|z|)`,
  `I_{-2} = arctan(s/z)/z` (only appears as `z I_{-2} = arctan(s/z)`),
  recurrence `(m+1) I_m = s r^m + m z² I_{m-2}`.
- `J_m(s) = ∫ s r^m ds = r^{m+2}/(m+2)`, `J_{-2} = log r`.
- `∫ s^j r^m ds`: substitute `s² = r² - z²` → combination of `I` (j even) or
  `J` (j odd).

Even `m >= 0` is a polynomial in `s`; odd `m >= 1` ends in `I_{-1}`.

## 2. Assumptions and validity

`z ≠ 0` for the `asinh` and `arctan` forms; `m >= -2` only (downward
recurrence for `m < -2` divides by `z²` and is **avoided**).

## 3. Derivation

Differentiate each antiderivative; verify the recurrence by
`diff` in Maple (`simplify(diff(I_m) - r^m) = 0`). The truncated-monomial value
formula (`HANDOFF.md §4`) follows by combining with the value identity: one
`arctan` per edge per support radius after summing the normalised kernel.

## 4. Special cases and limits

`z → 0`: `I_{-1}` diverges like `log|z|` but enters multiplied by `z²`, so guard
the exact `z = 0` case. Same-sign short chords: use a series or short Gauss.

## 5. Checks

| check | how | tolerance | status |
|---|---|---|---|
| `d/ds I_m = r^m`, `m = -2…12` | Maple symbolic | exact | [P] |
| `d/ds (s^j r^m)` combinations | Maple symbolic | exact | [P] |
| definite integrals vs mpmath `quad` | random `z, s_0, s_1` | 1e-30 | [P] |
| chords spanning `s = 0`, `z → 0` | sweep in mpmath vs float64 | see backends page | [P] |

## 6. Known failure modes

Cancellation for short chords; `float32` loses digits earlier (stage-3 test).

## 7. Open questions

Best stable evaluation for odd `m` and large `m` (`HANDOFF.md §13`).
