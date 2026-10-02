# δ⁺-SPH on the scene boundary layer — plan

**Status:** tank, wedge and Marrone 3.1 dam break done (`deltasph-validation.md`: tank 10× more accurate than mDBC; wedge passes all probe checks but rings; dam break P1 and KE match warpSPH, P2 / ceiling probe qualitatively); sloshing done (tank-fixed frame, rotating gravity as in warpSPH: flow and impact times match warpSPH, probe model weak). Term-by-term status: `deltasph-porting-notes.md` §4b. Follows `dfsph-validation.md` (DFSPH2D vs omniSPH). Same pattern: a self-contained 2D torch solver `deltasph2d.py` whose boundary terms are exact kernel integrals over the scene
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
| 4 | density diffusion `fourtakas2019` (**fluid–fluid only in warpSPH: no wall term, no new kernel**; was planned as a W'(r)/r wall integral): `δ h c ρ0 Σ V ψ_ij (r_ij·∇W)/(r² + η)`, ψ_ij = (ρ_j − ρ_i) − ½(∇ρ^L_i + ∇ρ^L_j)·r_ij − hydrostatic part | `∫ ψ(x') (y·∇W)/(y² + η) dA`: a **radial kernel `W'(r)/r`** applied to a per-query scalar field | **new kernel** `W'(r)/r` (polynomial for Wendland: no singularity) + `Interpolate`; ∇ρ^L needs 5 | new kernel |
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

**P0 result (2026-10-02, done).** `cd ~/dev/warpSPH && python scripts/probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40 --video --out <dir> --no-show` (warpSPH branch `dev`, unmodified): H/dx = 40.2, 10272 particles incl. wall layers,
c0 = 97.04, M = 0.049, 19551 steps, 147 s on the RTX PRO 6000, not diverged, ρ ∈ [0.983, 1.028], max|v| 9.6 m/s, max penetration 0.096 dx. Stored in `<dir>/sun2017DeltaSPH_nx67_c40.npz` (probe series, no particle snapshots) and the
video / frames in `<dir>/*_run/`. Probe numbers of the stored series (t* = t √(g/H), P* = P/(ρ0 g H); estimators `Star` / `In1Star` / `ShepStar` agree within 3 %):

| probe | first P* > 0.05 | mean P*, t* ∈ [3.2, 4.8] | mean P*, t* ∈ [5.2, 6.1] | maximum |
|---|---|---|---|---|
| P1 (z = 0.16 m, impact wall) | t* = 2.48 | 0.353 | 0.478 | 1.08 at t* = 6.37 |
| P2 (z = 0.584 m) | t* = 4.31 | 0.035 | 0.211 | 1.28 at t* = 6.92 |
| P3 (z = 1.0 m, **ceiling height**) | t* = 3.25 | 0.596 | 0.010 | **15.1 at t* = 3.29** |

These are the curves the analytic-wall run is compared with. Two observations to keep honest: (i) P1's plateau (0.35) and P2's late peak are *outside* the acceptance bands quoted in the probe's docstring (plateau 0.45–0.68, P2 peak 0.22–0.40 at
5.2 < t* < 6.1) — I have not audited whether the stored series is what the docstring scores, so the reference's own curves, not the bands, are the target; (ii) the P3 spike (P* = 15 at t* = 3.29) is the first ceiling impact of the right-wall run-up,
exactly where the ceiling-sticking work in warpSPH (`CEILING_STICKING_PLAN.md`) finds riders and kicked clusters. The matched initial particles are not stored: they are rebuilt from the case geometry (regular lattice, dx = 0.014925, box 3.2196 × 1.0, column 2H × H).

**Order decided 2026-10-02 (user): hydrostatic tank → English wedge → dam break → sloshing.** Tank and wedge need none of the planned new scene operations (only `Scene.inside`, added); the viscous wall term, the
`W⁵` shifting kernel and the closest-point query come with the dam break.

**P1 — scene-layer additions: done in a reduced form.** Added `Scene.inside`, `Scene.signed_distance`, the C4 kernel path, per-query / covariance use; the planned `W'(r)/r` kernel was not needed (DDT is fluid-to-fluid in warpSPH) and the `W⁵` and `p = 2` weights are replaced by a polar quadrature of the solid
(`DeltaSPH2D._solid_samples`, 24 × 96, ≤ 1 % on a half plane). Still open: a Laplacian operation (`Δλ`, first moments of `∇²W`) for the viscous wall term, the exact `W⁵` weight, per-query vector fields for curved walls, volume representations in `signed_distance`.

**P2 — solver, one term at a time: done through the dam break.** Order followed: tank (EOS, continuity, Antuono wall, DDT, viscosity, detector) → wedge (sloped faces and corners; staircase control, ramp reproducer) → dam break (viscous wall term, time-centred continuity,
dilated mask, shifting, no-penetration, probes) → sloshing (C4 kernel, rotating gravity, fixed dt, sensor probes). Each stage has unit tests (`tests/edge/test_deltasph.py`, 10) and a validation table in `deltasph-validation.md`.

**P3 — validation and videos: tank, wedge, dam break, sloshing done; not done:** a wall-pressure sensor for analytic walls (the MLS probe is unusable at impacts), sloshing at nx = 100 / 400 and with Michel shifting, the dam break at H/dx = 80 and 320, tiers 1 and 3 on the dam break, the hexagon obstacle in δ⁺, a body-fitted packing for smooth sloped walls (wedge KE; deferred by the user as a general problem for airfoils / complex geometry).

**Results so far.** Tank: profile error 10× below mDBC. Wedge: all probe checks pass, profile error below mDBC, settled KE 400× the reference (layout inconsistency). Dam break: P1 and KE match warpSPH, P2 and ceiling probe qualitatively.
Found on the way: the pair-list bug (2 % of reverse pairs lost for lattices commensurate with the support), warpSPH's dilated surface mask for the Antuono switch, the reference's own failure without shifting / no-penetration at the first ceiling impact.

**Open decisions for the port** (decision 1 is made; these follow from it): whether the analytic obstacle needs a packing pass before the run; whether the viscous / shifting / no-penetration wall terms should be exact (new kernels) or stay quadrature-based; how the scene adjacency becomes fixed-capacity and graph-capturable.

## 3b. From the 2D tiers to warpSPH, and what 3D changes (2026-10-02, with the user)

**Strategy (user).** Pull the 2D tiers into warpSPH generally and make them stable everywhere first (most uses of the solver are 2D; 3D needs a proper plotting / visualisation framework and fibre-bundle experiments to demonstrate the
solutions); 3D terms are research (derivation, then integration). Built on a solid 2D foundation (adjacency reuse, fixed capacity, graph capture, smart time stepping) the 3D runs become computationally feasible.

**What carries over to 3D unchanged:** the scene interface (bodies with pose, representation lists, per-type adjacency, per-body reactions, `a0` / `a1` fields), the structure of every δ⁺ / DFSPH term (pair sums + one wall integral), the tier ladder.
**What changes:** `Pose` (scalar angle, 2 × 2 rotation → rotation matrix / quaternion, vector ω, `ω×(ω×s) + α×s`, inertia tensor, 3-vector torque); edges → triangles (face → edge chain), triangles → tetrahedra, winding number / `inside` in 3D;
tier 3 needs mean and Gaussian curvature (Tube Maps K sign open), trilinear SDF; tier 4: ball / capsule / slender-body series for fibres; `a1` is 3 × 3 and second moments are 3 × 3 tensors; every hard-coded 2 (polar sampling, 2 × 2 inverses and eigenvalues, `[-y, x]` rotations) must become dimension-generic.

**2D shortcuts that do not scale** (the quadrature, in detail): `DeltaSPH2D._solid_samples` samples the solid on a 24 × 96 polar grid around every wall-adjacent particle and `Scene.inside` classifies the samples. It stands in for three integrals:
(1) the viscous wall moment `M₂ = ∫_solid y⊗y W'(r)/r³ dA` (= `∫W' dr ∫ŷ⊗ŷ 1[solid] dφ`), (2) the shifting tensile control `∫_solid W⁴ ∇W dA` (the gradient of the kernel `W⁵/5`),
(3) the Barecasco wall counts (cover vector `∫ unit(x_i − x') 1[solid]`, exact as the gradient of the kernel `K = r`; the cone count `area(solid ∩ sector)` is a clipping problem, only `> 0.5` matters).
**Correction (user, 2026-10-02): (1) does not need p = 2 weights with negative powers.** The `/r³` only reproduces warpSPH's pairwise Monaghan-Gingold / Brookshaw form (direction `x̂`, one `1/r` from the difference quotient, velocity difference projected on `x̂`), an approximation made to avoid `∇²W` on disordered particles.
An exact boundary samples its volume exactly, so the naive Laplacian applies: for the (constant) mirrored difference the wall term is `ν_eff (v_b − v_i) Δλ`, because `∫_solid ∇²_x W dA = ∇²_x λ` (tier 3: second derivative of the planar table; tier 2: an edge integral of `∂_n W`);
for a linear mirror field (the renormalised velocity gradient mirrored: the image field of free slip, odd normal / even tangential components, is C¹) the first moments of `∇²W` enter, also edge integrals (the per-query `a1` machinery). So the scene needs a **Laplacian operation** (the old "Laplacian: not done"), not negative-power potentials.
Caveats: it changes the operator relative to the reference (small effect expected, α ≈ 0.01-0.02, to be checked on the dam break), `ν_eff` must be calibrated against the fluid pair form (`α c0 H/(ξ 2(d+2))` in warpSPH's `alphaToNu` convention, unverified).
(1) and (2) otherwise are the same type of integral as `λ`, `∇λ` and the covariance, which the edge machinery does exactly (risks: `W⁵` is a degree-25 polynomial for C2, more for C4, and needs the exact Chebyshev compile).
Why exact: **cost** (3D: 24 × 48 × 96 ≈ 10⁵ samples and a 3D `inside` per wall-adjacent particle, 10¹⁰–10¹¹ per step at 10⁵–10⁶ such particles, versus a few elements exactly); **consistency** (at the first row the shift is a small difference of large terms,
+2.24 fluid, −2.07 exact `G`, −0.14 tensile, net +0.03, so a 1 % sampling error in one term is a few percent of the net; the viscous term was 5.6 % off at 12 radii, ~1 % at 24); **smoothness** (the sampled force jumps as the solid edge crosses a sample, the exact integral is smooth in the position and differentiable).
Other non-scaling items: flat-wall mirror normals (`∇λ/|∇λ|`, needs a per-query vector field at curved walls), brute-force `signed_distance` over edges (needs a spatial structure).

**The main design point for fibre bundles: many small bodies.** The scene layer costs ~3 ms of launch overhead per body in a Python loop. A bundle needs one flat array of primitive instances (pose, radius or skeleton index) handled by one adjacency and one kernel launch,
with reactions (force, torque) accumulated per instance. Build and test that in 2D first: a bundle of disks in cross-flow (tier 4 disks exist) is the natural first demo.

**Cost of 3D.** Neighbours ~50 → ~270 at support 4 dx; wall integrals stay affordable only if tiers 3 / 4 give O(1) closed forms per particle away from sharp features, exact elements near them. The weakly-compressible acoustic CFL is the same per step as in 2D but each step is far more expensive,
so the incompressible path (DFSPH, already on this scene layer) is likely preferable for many 3D cases; both need a fixed-capacity adjacency reused across steps (Verlet-style, graph-capturable).

**Order.** (1) Freeze a dimension-generic interface for bodies, representations and operations (implement 2D). (2) Run the 2D tiers on Warp with reusable fixed-capacity adjacency and batched primitive instances; exact weights replace the quadrature one use at a time (quadrature stays as a checked fallback).
(3) Integrate in warpSPH as a boundary provider replacing the `kinds == 1` wall particles (`deltasph-porting-notes.md` term map). (4) Regression suite over schemes and integrators with this session's cases (tank, wedge, dam break, sloshing, hexagon, DFSPH).
(5) Fibre-bundle demos in 2D, in parallel with the 3D derivations (Maple: face → edge chain, tetrahedra, curvature series). **Open in 2D before the port:** the wedge layout (general packing problem, deferred), exact weights, moving bodies in δ⁺, exact force bookkeeping.

## 3c. To investigate: openMaelstrom (user, 2026-10-02)

How `~/dev/openMaelstrom` implements the surface detection (reported: cover-vector based, works with an SDF boundary), boundary friction (reported: supported) and its shifting / surface handling near boundaries (pointers and the questions are in `deltasph-resume.md`).
Expected consequence to confirm from the code: if the cover vector and the friction term need only SDF quantities (`λ`, `∇λ`, distance), then **shifting is the only term whose wall part has no existing exact weight**. Our accounting: the viscous wall term becomes the exact Laplacian operation (§3b), the detector's cover vector is already exact
(gradient of the kernel `K = r`) and only the cone count is a clipping problem, and the shifting tensile control is `∫W⁴∇W dA = ∇∫W⁵/5 dA`: exact in principle with a new kernel table entry (degree-25 polynomial for C2, more for C4: conditioning to be checked), so quadrature is not needed even for shifting, only a heavier kernel entry. It also depends on whether openMaelstrom's shifting has a wall term at all.

## 4. Decisions

1. **Where does the solver live? — decided (2026-10-02): self-contained here** (`deltasph2d.py`, like DFSPH2D), warpSPH's `sun2017DeltaSPH` run separately as the live reference. The long-term goal is to integrate the boundary code into
   warpSPH's core and add front-end support for such bodies; that needs a better understanding of where the boundaries touch each part of the solver, which is easier to build up outside. **`deltasph-porting-notes.md`** records,
   term by term, where the corresponding operation lives in warpSPH, how the wall enters there, what replaces it here and what changed; it is updated with every change. (Correction of an earlier statement: the scene layer's
   kernels are Warp float64 on torch memory already; what limits a port is data-dependent shapes / per-step adjacency rebuild / host-side branching, see the notes §3.)
2. **Free-slip semantics for the viscous and DDT wall terms**: the reference mirrors the *Shepard-interpolated* fluid velocity at the ghost node, the analytic version mirrors the particle's own velocity. Matching the reference term
   by term is impossible; the plan matches macroscopic behaviour (probes, front). Default taken unless told otherwise.
3. **Wall bookkeeping**: keep exact momentum bookkeeping as in DFSPH where a term allows it, and record where it does not (the DDT and the viscous terms are not pairwise antisymmetric with a wall). Default taken unless told otherwise.
