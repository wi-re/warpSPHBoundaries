# <Result name>

**Status:** [P] plan | [D] derived, not verified | [V] verified
**Tier(s):** 1 / 2 / 3 / 4 · **Dimension:** 2D / 3D
**Depends on:** <other derivation files>
**Implemented in:** <file::function> (or "not yet")
**Verified by:** <test file / Maple script / notebook> (or "not yet")

## 1. Statement

The result in one displayed formula, with every symbol defined or pointing to
`../notation.md`.

## 2. Assumptions and validity

Bullet list: smoothness, kernel class, orientation, geometry restrictions
(reach, single sheet in support, non-overlapping elements), what happens at
the excluded cases.

## 3. Derivation

Numbered steps. Each step is either an identity used with a citation, or an
algebraic step the reader can follow. Mark any step not independently checked.

## 4. Special cases and limits

Limits that must hold (planar limit, `z → 0`, `d → 0`, `d → h`, flat limit…).

## 5. Checks

A table: check · how · tolerance · where it lives (test / Maple script /
notebook cell) · status. These are the same checks the golden fixtures
(`../backends-and-verification.md`) must contain.

## 6. Known failure modes

Cancellation, singularities, conventions at degenerate configurations.

## 7. Open questions
