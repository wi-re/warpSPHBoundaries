# Documentation index

Status tags: **[V]** verified numerically · **[D]** derived, not verified · **[P]** plan.

| file | content | status |
|---|---|---|
| `notation.md` | conventions shared by all derivations and code | draft |
| `derivation.md` | exact closed forms: 2D/3D planar, 3D sphere (the `PLAN.md` track; Maple + mpmath validated) | [V] |
| `boundary-operations.md` | warpSPH-style boundary operations (Interpolate/Gradient/Divergence/Curl/Density), Warp P1 pair engine, kernels, throughput, tier 3/4 implicit bodies (disk, half-plane) with hard tier switches | [V] tiers 1–4 |
| `scene-architecture.md` | scene layer: bodies with pose + OBB, surface / volume / implicit / SDF representations, per-type adjacency and operations, reactions, DFSPH mapping | [V] 2D |
| `dfsph-validation.md` | omniSPH-style DFSPH on the scene layer, validation against the compiled omniSPH (tank, dam break), domain body as tier 1 / 2 / 3, findings | [V] 2D |
| `deltasph-porting-notes.md` | living notes for porting the analytic boundaries into warpSPH: where each δ⁺ term lives there, what replaces it, what changed | notes |
| `deltasph-validation.md` | δ⁺-SPH on analytic walls vs warpSPH + mDBC: flat tank (10× better profile), English wedge (passes, rings), Marrone 3.1 dam break (matches P1 / KE), SPHERIC sloshing (flow and impact times match warpSPH), pair-list bug | [V] 2D |
| `deltasph-plan.md` | plan: δ⁺-SPH (Marrone 3.1) on the scene boundary layer — term inventory, new scene operations, phases, open decisions | plan |
| `tier-selection-2d.md` | which tier for which obstacle size (2D disk), jumps at switches | [V] |
| `autodiff-and-gpu.md` | torch autodiff vs analytic, shape adjoint, Warp float64 kernels, throughput | [V] |
| `exactness-and-approximations.md` | stage-2 numpy results: what is exact, accuracy/cost of cheap approximations (float32, Gauss, tiny elements) | [V] |
| `backends-and-verification.md` | Maple → mpmath → numpy → torch → warp hierarchy, golden fixtures, tolerances; stage-1 results | stage 0–1 done |
| `derivations/TEMPLATE.md` | template every derivation file follows | — |
| `derivations/edge-value-identity.md` | 2D value integral as edge integrals | [V] Maple + mpmath |
| `derivations/edge-gradient-identity.md` | 2D gradient and first moment | [V] Maple + mpmath |
| `derivations/edge-primitives.md` | `I_m`, `J_m`, `S_{j,m}` antiderivatives | [V] Maple + mpmath |
| `derivations/truncated-monomials.md` | kernels as `r^n 1[r<=R]` blocks | [V] Maple + mpmath |
| `derivations/moments-recursion.md` | higher moments (compact-potential recursion) | (a) [V] k≤4; (b) [V], z≠0 only |
| `derivations/fem-nodal-weights.md` | P0–P3 nodal weights, order-recovery checks, conditioning study and remedies | [V] checks 1–7 |
| `derivations/tier2-closed-boundary.md` | closed polylines, boundary data only, polygon→disk convergence | [V] |
| `derivations/tier3-curvature-2d.md` | 2D curvature expansion (closest point), F0/F1/F2 closed forms | [V] |
| `derivations/tier4-slender-series.md` | fibres: slender series (2D strip/disk [V]; 3D ball/strand [D]/[P]), circle edge identity | 2D [V] |
| `derivations/tet-face-edge-chain.md` | 3D tetrahedra | [P] |

Two tracks meet here: `PLAN.md` (curvature-aware exact/series results; tier 3 and
the oracles for the others) and `HANDOFF.md` (edge reductions, tiers 1/2/4, FEM
fields). See `PLAN.md` "Relation to HANDOFF.md" for how they fit.

Rule for derivation files: each lists the checks it needs; those checks are the
golden-fixture cases. Do not mark a file [V] until the check table is green.

## Phase 1 artefacts (HANDOFF track, 2D tier 1/2 verification)

| what | where |
|---|---|
| Maple proofs/checks | `maple/10_edge_primitives.mpl`, `11_truncated_monomials.mpl`, `12_moments_recursion.mpl`, `13_halfplane.mpl` (`maple/run_edge.sh` runs all four) |
| stage-1 reference implementation (mpmath) + polar oracle | `python/edgebound/` (`core`, `geometry`, `primitives`, `kernels`, `oracle`) |
| tests | `tests/edge/` (`pytest tests/edge`, ≈2.5 min) |
| golden fixtures | `tests/fixtures/edge2d_golden.json` (92 cases; `python -m edgebound.fixtures` regenerates; schema in `python/edgebound/fixtures.py`) |
| float64 behaviour of the guard-free algorithm | `python -m edgebound.precision_probe` (table in `backends-and-verification.md`) |

## Stage 2 (numpy, vectorised, branch-free)

`python/edgebound/np2d.py` (batched convex polygons; exact `Fraction` kernel compilation; dtype and quadrature options),
`tests/edge/test_np2d.py` (all 82 convex-triangle fixtures in one vectorised call per kernel, covering meshes, stage-1
agreement, tiny elements, longdouble/Gauss options, continuity across an edge), `python -m edgebound.np2d_study`
(accuracy/cost tables in `exactness-and-approximations.md`).

## Stage 3 (float32)

Exact Chebyshev compilation of the edge profiles (`np2d.cheb_coeffs`) + stable quadrature mode (`stable=(nodes, panels)`):
float32 at ≈ machine epsilon on all representable fixtures; tolerance table per class in `exactness-and-approximations.md`;
tests `tests/edge/test_np2d_stable.py`.

## FEM nodal weights (2D)

`python/edgebound/fem.py` (exact stage 1), `np_fem.py` (numpy: edge reduction near `x`, Gauss on far elements), `fem_fixtures.py` →
`tests/fixtures/edge2d_fem_golden.json`, `fem_study.py` (conditioning table), tests `tests/edge/test_fem.py`, `test_fem_np.py`.

External benchmark: `tests/edge/test_paper2025.py` (Winchenbach & Kolb 2025 validation problems; results in `derivations/fem-nodal-weights.md` §9).

Notebook: `notebooks/edge2d_demo.ipynb` (executed; rebuilt by `python notebooks/build_edge2d_demo.py`): identities, paper benchmark, FEM conditioning, tier map, throughput.
