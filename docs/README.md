# Documentation index

> **2026-10-11:** the solver docs (`deltasph-*`, `dfsph-validation`, `plan-upstreaming`, `plan-next-steps`, `audit-warpsph-boundary-hooks`) were removed with the retired `sim/` solvers; they are at the git tag `oracle-solvers-2026-10-11` and the validation records are archived in warpSPH `docs/analytic_boundaries/oracle_docs/`. Docs that mention `sim/`, `DeltaSPH2D` or `DFSPH2D` describe the retired solvers.

Status tags: **[V]** verified numerically · **[D]** derived, not verified · **[P]** plan.

| file | content | status |
|---|---|---|
| [validation.md](validation.md) | accuracy scope, limitations, external evidence storage and reproduction commands | development progress and generated artifacts are outside the repo |
| [plan-porous-coupling.md](plan-porous-coupling.md) | flux boundary Warp operators, solid disk/cylinder cases, resolved fibre to effective permeability comparisons and later porous coupling; includes Böttcher et al. 2025 | [P] main warpSPH integration deferred until incompressible interfaces settle |
| [handoffs/solid-simplex-wcsph.md](handoffs/solid-simplex-wcsph.md) | dedicated C2/C4 triangle/tet solid bundles, adjacency/CSR bridge, physical viscosity conversion and first fluid-trial prerequisites | CPU/CUDA library gates passed; solver adapter pending |
| [surface-moments.md](surface-moments.md) | twelve-moment SurfaceRep contract, device conditioning, diagnostics and validation | CPU/CUDA finite ε32 gate passed; other representations remain partial |
| [sdf-guard.md](sdf-guard.md) | sampled-cell interval guard, curvature-pole rejection, explicit fallback and accuracy limits | conservative domain guard; local-model integral tolerance remains open |
| [sdf-distance-only.md](sdf-distance-only.md) | curvature-free semi-analytic half-plane reference, twelve moments and separated error budgets | reference complete; geometric bias explicitly measured |
| [derivations/curvature-moments-2d.md](derivations/curvature-moments-2d.md) | consistent order-two curvature/variation corrections, twelve moments, sampled-jet guard and differential identities | implemented in Scene; higher-order accuracy limits remain |
| [sdf-accepted-accuracy.md](sdf-accepted-accuracy.md) | baseline accepted smooth-SDF audit and reference construction | historical mixed model; superseded by the shared curvature-moment family |
| `notation.md` | conventions shared by all derivations and code | draft |
| `derivation.md` | exact closed forms: 2D/3D planar, 3D sphere (the `PLAN.md` track; Maple + mpmath validated) | [V] |
| `boundary-operations.md` | warpSPH-style boundary operations (Interpolate/Gradient/Divergence/Curl/Density), Warp P1 pair engine, kernels, throughput, tier 3/4 implicit bodies (disk, half-plane) with hard tier switches | [V] tiers 1–4 |
| `scene-architecture.md` | scene layer: bodies with pose + OBB, surface / volume / implicit / SDF representations, per-type adjacency and operations, reactions, DFSPH mapping | [V] 2D |
| `q2-conditioning.md` | Q2 (WORK-002 T2.4): the shifting tensile term T = (1/5)∇∫W⁵ as the W⁵ g(0,0) gradient channel — Warp plan vs np2d (plain/stable) vs mpmath reference, classification at k = 5, flat-floor sign, stable-route vertex finding | [V] |
| `work/` | local-model work packages: `KICKOFF.md` (protocol), `WORK-NNN.md`, `REVIEW.md`, logs and reports | process |
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
| [derivations/tet-face-edge-chain.md](derivations/tet-face-edge-chain.md) | analytic 3D tetrahedral meshes: regular face/edge flux, moments, derivatives and P0–P3 FEM; independent sphere overlap comparison | [V] arbitrary-precision reference; device conditioning open |
| [derivations/tri-mesh-surfaces-3d.md](derivations/tri-mesh-surfaces-3d.md) | closed triangular boundaries and standalone area elements, surface FEM and ambient gradients | [V] reference and finite tests; GPU backend open |
| [derivations/radial-terms-3d.md](derivations/radial-terms-3d.md) | 3D sphere/plane moments, Hessian/Laplacian, curvature graph terms and small-ball series; browser equations and compute | [V] finite reference and browser/WGSL cases; model limits explicit |
| `derivations/laplacian-wall.md` | wall Laplacian Δλ = ∫_solid ∇²W dA′ = 2λ[L] − tr Cov[L] (the viscosity wall term, Q1): scene route through the registered L = W′/r, ν_eff = α c0 H/(8ξ) | [V] tests |
| `derivations/noslip-wall.md` | no-slip wall viscosity (WORK-007, Q1): the Chiron et al. 2019 Eq. 91–92 one-sided finite-difference wall flux `-2 ν_eff (v − v_w)|G|/(ρ d)`, `ν_eff = α c0 H/(8ξ)`, `d_min = 0.25 dx`, all-components relative velocity (the third `wallViscosityForm`) | [V] tests |
| `plan-wall-evaluation.md` | **the working plan for the speed work**: adjacency / precompute / evaluation terminology, steps 0–6 with their results (idle-GPU measurements, float32 kernels, term-parallel and fused wall evaluation, `BoxRep`), the "Resume here" block, open decisions | plan, steps 0–3 done |
| `box-domain-primitive.md` | direct treatment of axis-aligned boxes (tanks, channels, baffles): all nine moment channels from four corner lookups of two 2D tables per kernel, no edge loop / pair list; accuracy table, 3D, scene-path and fused-evaluation fit | [V] 2D table accuracy, [P] BoxRep |
| `plan-exact-wall-laplacian.md` | exact wall Laplacian from a reconstructed ghost field: closed-form moments of ∇²W (Green's identity), free-slip mirror hierarchy [V], no-slip as BC-constrained polynomial continuation [V noise-free], experiments E1–E7 (E4 pairwise-operator moments and E5 corner / circular wall done [V]; noise vs order, constrained MLS still to run) | [V] identity, E4, E5, [P] experiments |

Two tracks meet here: `PLAN.md` (curvature-aware exact/series results; tier 3 and
the oracles for the others) and `HANDOFF.md` (Part A: state and the work plan towards the warpSPH port; Part B: edge reductions, tiers 1/2/4, FEM
fields). See `PLAN.md` "Relation to HANDOFF.md" for how they fit.

Rule for derivation files: each lists the checks it needs; those checks are the
golden-fixture cases. Do not mark a file [V] until the check table is green.

## Phase 1 artefacts (HANDOFF track, 2D tier 1/2 verification)

Reusable device evaluation and derivative contracts:
[Warp boundary functions](warp-boundary-functions.md) documents the component
folder layout, direct batches, fused-consumer calls and checked local-model
adjoints, including the remaining sampled-SDF/Scene derivative gaps.

| what | where |
|---|---|
| Maple proofs/checks | `maple/10_edge_primitives.mpl`, `11_truncated_monomials.mpl`, `12_moments_recursion.mpl`, `13_halfplane.mpl` (`maple/run_edge.sh` runs all four) |
| stage-1 reference implementation (mpmath) + polar oracle | `src/warpSPHBoundaries/` (`core`, `geometry`, `primitives`, `kernels`, `oracle`) |
| tests | `tests/edge/` (`pytest tests/edge`, ≈2.5 min) |
| golden fixtures | `tests/fixtures/edge2d_golden.json` (92 cases; `python scripts/make_fixtures.py edge` regenerates; schema in `src/warpSPHBoundaries/edge/fixtures.py`) |
| float64 behaviour of the guard-free algorithm | `python scripts/studies/precision_probe.py` (table in `backends-and-verification.md`) |

## Stage 2 (numpy, vectorised, branch-free)

`src/warpSPHBoundaries/edge/np2d.py` (batched convex polygons; exact `Fraction` kernel compilation; dtype and quadrature options),
`tests/edge/test_np2d.py` (all 82 convex-triangle fixtures in one vectorised call per kernel, covering meshes, stage-1
agreement, tiny elements, longdouble/Gauss options, continuity across an edge), `python scripts/studies/np2d_study.py`
(accuracy/cost tables in `exactness-and-approximations.md`).

## Stage 3 (float32)

Exact Chebyshev compilation of the edge profiles (`np2d.cheb_coeffs`) + stable quadrature mode (`stable=(nodes, panels)`):
float32 at ≈ machine epsilon on all representable fixtures; tolerance table per class in `exactness-and-approximations.md`;
tests `tests/edge/test_np2d_stable.py`.

## FEM nodal weights (2D)

`src/warpSPHBoundaries/edge/fem.py` (exact stage 1), `np_fem.py` (numpy: edge reduction near `x`, Gauss on far elements), `fem_fixtures.py` →
`tests/fixtures/edge2d_fem_golden.json`, `fem_study.py` (conditioning table), tests `tests/edge/test_fem.py`, `test_fem_np.py`.

External benchmark: `tests/edge/test_paper2025.py` (Winchenbach & Kolb 2025 validation problems; results in `derivations/fem-nodal-weights.md` §9).

Notebook: `notebooks/edge2d_demo.ipynb` (executed; rebuilt by `python notebooks/build_edge2d_demo.py`): identities, paper benchmark, FEM conditioning, tier map, throughput.
