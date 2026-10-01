# Documentation index

Status tags: **[V]** verified numerically · **[D]** derived, not verified · **[P]** plan.

| file | content | status |
|---|---|---|
| `notation.md` | conventions shared by all derivations and code | draft |
| `derivation.md` | exact closed forms: 2D/3D planar, 3D sphere (the `PLAN.md` track; Maple + mpmath validated) | [V] |
| `backends-and-verification.md` | Maple → mpmath → numpy → torch → warp hierarchy, golden fixtures, tolerances | plan |
| `derivations/TEMPLATE.md` | template every derivation file follows | — |
| `derivations/edge-value-identity.md` | 2D value integral as edge integrals | [V] numerics |
| `derivations/edge-gradient-identity.md` | 2D gradient and first moment | [V] numerics |
| `derivations/edge-primitives.md` | `I_m`, `J_m` antiderivatives | [D] |
| `derivations/truncated-monomials.md` | kernels as `r^n 1[r<=R]` blocks | [D] |
| `derivations/moments-recursion.md` | higher moments (compact-potential recursion) | [D] |
| `derivations/fem-nodal-weights.md` | P0–P3 nodal weights, order-recovery checks | [D] |
| `derivations/tier3-curvature-2d.md` | 2D curvature expansion (closest point) | [P] |
| `derivations/tier4-slender-series.md` | fibres: slender series, ball primitive | [D]/[P] |
| `derivations/tet-face-edge-chain.md` | 3D tetrahedra | [P] |

Two tracks meet here: `PLAN.md` (curvature-aware exact/series results; tier 3 and
the oracles for the others) and `HANDOFF.md` (edge reductions, tiers 1/2/4, FEM
fields). See `PLAN.md` "Relation to HANDOFF.md" for how they fit.

Rule for derivation files: each lists the checks it needs; those checks are the
golden-fixture cases. Do not mark a file [V] until the check table is green.
