# δ⁺-SPH on the scene boundary layer — plan

**Status:** plan, nothing implemented. Follows `dfsph-validation.md` (DFSPH2D vs omniSPH). Same pattern: a self-contained 2D torch solver `deltasph2d.py` whose boundary terms are exact kernel integrals over the scene
(`scene-architecture.md`), validated against a live reference with identical initial particles. Reference here: warpSPH's `sun2017DeltaSPH` (δ⁺, PST on) + `fourtakas2019` DDT + `symplecticEuler` + `english2025` mDBC + free-slip, the
Marrone 3.1 dam break (`warpSPH/scripts/probe_deltaSPHMarrone.py`: column 2H × H, H = 0.6 m, tank 5.366 H, ceiling at 1.0 m, probes P1/P2/P3 on the impact wall, c0 = 40 √(gH), H/dx = 40/80/320).

## 1. What changes against DFSPH

DFSPH had **three** wall quantities (λ, ∇λ, a hydrostatic pressure extrapolation). δ⁺ is an explicit weakly-compressible scheme with many more wall-dependent terms, and the wall is a *particle field* in the reference
(mDBC ghost-node density clone, free-slip mirrored velocity, no-penetration shift), so every term needs a decision about what the analytic wall supplies. Boundary-particle quantities that must become fields over the wall:

| quantity at the wall | reference (mDBC, english2025) | analytic version |
|---|---|---|
| density ρ_b, pressure P_b | ghost value, `P_b = P_g + ρ0 (g − a_b)·relPos`, `ρ_b = ρ0 + P_b/c0²` | per-query linear field `P_b(x') = P_i + ρ(g − a_w)·(x' − x_i)`, **clamped ≥ 0 (DFSPH §7: otherwise ceiling suction)**; ρ_b from the EOS |
| velocity v_b | free-slip: Shepard fluid velocity at the ghost with the normal part reflected | per-query field `v_i − 2 (v_i·n(x')) n(x')`; constant n for polygons/half planes, varying for curved bodies |
| position constraint | no-penetration shift of particles that crossed the surface | closest point + normal of the scene (does not exist yet) |

## 2. Term inventory (one δ⁺ RHS call) and what each needs from the scene

| # | term (reference module) | wall contribution, exact form | scene operation | status |
|---|---|---|---|---|
| 1 | density / kernel sum, continuity `dρ/dt = −ρ Σ V (v_j − v_i)·∇W` (`momentum`) | `+ρ_i (v_i·∇λ − ∫ v_b·∇W)` with the mirrored `v_b` | `Gradient` (∇λ), `Divergence` with a per-query vector field | have (per-query vector fields: check the vector case) |
| 2 | Tait EOS | none | — | — |
| 3 | pressure force, Antuono surface-aware (`pressure/surfaceAware`) | `−(1/ρ_i)(P_i ∇λ + ∫ P_b ∇W)` for surface rows; wall rows follow the reference's switch | `Gradient`, Symmetric, per-query `P_b` | have (this is the DFSPH wall term) |
| 4 | density diffusion `fourtakas2019`: `δ h c ρ0 Σ V ψ_ij (r_ij·∇W)/(r² + η)`, ψ_ij = (ρ_j − ρ_i) − ½(∇ρ^L_i + ∇ρ^L_j)·r_ij − hydrostatic part | `∫ ψ(x') (y·∇W)/(y² + η) dA`: a **radial kernel `W'(r)/r`** applied to a per-query scalar field | **new kernel** `W'(r)/r` (polynomial for Wendland: no singularity) + `Interpolate`; ∇ρ^L needs 5 | new kernel |
| 5 | renormalised gradients `L_i = (Σ V r⊗∇W)⁻¹`, `∇ρ^L` (`density/gradRhoL`) | wall adds the first moments `∫ y⊗∇W` (g1) and `∫ (ρ_b − ρ_i) ∇W` | `Covariance`, `Gradient` per-query | have (DFSPH §6) |
| 6 | velocity dissipation (`deltaSPH/velocityDissipation`): `α h c/ρ Σ m (v_ij·r_ij)/(r² + η) ∇W` | free-slip: `v_ij = 2 (v_i·n) n`, so `2 (v_i·n) ∫ (n·y)/(y² + η) ∇W` — a **second moment `∫ y⊗y g(r)`** | **new**: p = 2 weights (`y_a y_b` times a radial kernel; the same missing item as the torque, `dfsph-validation §5`) | new |
| 7 | surface detection (`surfaceDetection`, Maronne/Barecasco/λ_min of `L⁻¹`) and normals | needs the wall in `Σ V ∇W`, in the covariance eigenvalue and in the completeness `Σ V W + λ` | `Density`, `Gradient`, `Covariance` | have; the *detector's* thresholds were tuned for mDBC particle walls — re-check |
| 8 | PST shift δ⁺ (Sun 2017 Eq. 7): `δr = −Ma·16 h² Σ (1 + R (W/W_Δp)⁴) V ∇W` with R = 0.2 (`sun2017Eq7Shift`; the historical default is 1/8 of it), normal part removed at the free surface | `∫ (1 + R (W/W_Δp)⁴) ∇W dA` = `Gradient` of the kernels `W` and `W⁵/5` (degree-25 polynomial for Wendland C2: exact Chebyshev compile, `stable` mode) | **new kernel** `W⁵` | new kernel |
| 9 | no-penetration shift, free-slip reflection of the normal velocity (`mdbc/wp_nopenshift`) | closest point, normal, signed distance | **new**: `Scene.closestPoint(x)` per representation (surface: nearest edge; SDF/implicit: from the field / half plane / disk; volume: boundary of the union) | new |
| 10 | gravity, forcing, symplectic Euler (midpoint x update) | none | — | — |
| 11 | body forces / per-body momentum bookkeeping | `perBody` ops (have); with the DDT and viscosity terms the wall exchange is no longer pairwise antisymmetric by construction — the exact-bookkeeping property of DFSPH has to be re-derived per term | `perBody` | open question |

The pair sums (fluid–fluid) are plain torch (`neighbor_pairs`), checked against `warpSPHCore.warpOperation` as in `test_pair_sums_match_warpoperation`.

## 3. Phases (each ends with a test and a number)

**P0 — reference capture (no new solver).** Run warpSPH's Marrone 3.1 (default combination, H/dx = 40 and 80) with snapshot export; keep the initial particles, pressure probes P1/P2 and video. Acceptance bands are in the
probe's docstring (P1 arrival 2.5 < t* < 3.0, plateau P* 0.45–0.68, P2 peak 0.22–0.40 at 5.2 < t* < 6.1). Also record how the reference behaves at the ceiling: its own `CEILING_STICKING_PLAN.md` documents ceiling riders and
kicked clusters with mDBC, so the analytic wall may well *differ* from it there (that would be a result, not a mismatch).

**P1 — scene-layer additions** (kernels first, they are table entries): `W'(r)/r`, `W⁵` (and `W⁵/5` gradient form); p = 2 weights `∫ y_a y_b g(r)`; `Scene.closestPoint` / normal; per-query vector fields. Each verified against a dense
boundary-particle lattice (the continuum limit) and, for the kernels, against the polar disk oracle (`oracle.py`) as before.

**P2 — solver, one term at a time**, always against the particle-wall limit so that a sign or factor error shows as a mismatch with a *dense wall particle layer* rather than with the reference scheme:
hydrostatic tank at rest (pressure profile, spurious velocity) → add DDT (ψ with the hydrostatic correction) → viscosity → PST → free surface switch → Marrone 3.1.

**P3 — validation and videos**: Marrone 3.1 probes P1/P2/P3 and the front against warpSPH, 2k–33k particles, tier 1/2/3 domain representations, the hexagon dam break, the same `dfsph_runcase` / `dfsph_video` tooling.

## 4. Decisions needed

1. **Where does the solver live?** (a) self-contained `deltasph2d.py` here, like DFSPH2D (fast to iterate, reference is warpSPH run separately), or (b) a boundary provider inside warpSPH replacing the `kinds == 1` mDBC particles
   (reuses its CUDA graphs, integrators and the surface detector, but is intrusive and the scene layer is torch/2D today).
2. **Free-slip semantics for the viscous and DDT wall terms**: the reference mirrors the *Shepard-interpolated* fluid velocity at the ghost node, the analytic version mirrors the particle's own velocity. Matching the reference term by term is
   impossible; the plan matches macroscopic behaviour (probes, front). Is that the right standard here?
3. **Wall bookkeeping**: keep exact momentum bookkeeping (as in DFSPH) at the price of dropping non-antisymmetric wall terms, or accept approximate wall forces?
