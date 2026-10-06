# Next steps inside this repo (2026-10-06)

Everything here stays in `curvatureBoundaries` / `warpSPHBoundaries`; nothing edits the warpSPH repo.  Order: 1, 4, 2, then the rest as time allows.
Status of the base: `main` 746f37f, 941 tests, bit-level harness PASS.

| # | item | size | depends on |
|---|---|---|---|
| 1 | periodic domain, prescribed-velocity (Dirichlet) frame, body force, drag / lift validation ladder | large | – |
| 2 | DFSPH on the fused path (m1, force, torque outputs of the fused contraction; disk element + graph step) | medium | – |
| 3 | vectorise the per-body torch loops of the wall terms (~0.3 ms per body) | small | – |
| 4 | `BCType` closures in `DeltaSPH2D` (constant / zeros, freeSlip, noSlip; relative no-penetration law for the pinned ones) | medium | – |
| 5 | remaining non-Galilean pieces (shifting Mach number from the absolute max speed; free-slip mirror and Laplacian `un` with the particle velocity instead of the wall-point velocity) | small | 4 |
| 6 | f32 baselines for the regression harness | small | – |
| 7 | other integrator schemes (`velocityVerlet`, `leapFrog`, RK) on `DeltaSPHSystem`; adaptive dt and `gravityFn` into the library hooks | medium | – |
| 8 | disk-identity derivation note with a symbolic / multi-precision check (identities, table construction, the ε⁴ ln ε term) | small | – |
| 9 | CI-style test that installs the wheel into a clean venv and runs the scene layer | small | – |
| 10 | warpSPH-side change set prepared as a patch series in a scratch clone (provider hooks in `_deltaSPH_rhs`, `RigidBody.representation`, `filterRegion`, new modules) | medium | 1, 4 |

Not doing (agreed): 3D, slot-resolved per-fibre kinematics.

## 1. Periodic flow past a body: drag and lift

Goal: a reusable setup to extract drag and lift on cylinders / fibres / general `SurfaceRep` and `SdfRep` bodies, with literature or exact reference values.
This is also the case the fibre-permeability studies need.

### Setup (the warpSPH `movingObstacle` style, kept)

* Domain periodic in x and y (or only x for a channel with walls); the particle positions are raw, unwrapped trajectories (see 1a).
* The fluid is a full periodic box with the body cut out.  A band of fluid particles at the outer border of the box (the wrapping region) has its velocity **prescribed**
  (Dirichlet, free stream U∞).  No inlets or outlets: the particle count and the particle identities are constant, so a trajectory can be exported and the loads recovered
  post-hoc with `loadsAt` (verified, 5e-4 frame mean).
* Second driving mode for permeability: a uniform body force (no frame), steady state for a periodic array.  This is a separate driver, not gravity: gravity enters the
  hydrostatic term of the density diffusion and the no-penetration law, and a periodic pressure has no hydrostatic part.  `cfg.bodyForce` is applied to the momentum
  only and booked as such.

### Work packages

1a. **Periodic fluid pairs, raw positions untouched (user, 2026-10-06).**  The stored positions are never wrapped, clipped or modified: `x` is the integrated state, so
    dx/dt along a trajectory stays exact (ML training on trajectories, exported frames, `loadsAt`).  Periodicity lives only in the pair geometry:
    * the displacement is minimum-imaged where it is formed, `d = x_i - x_j - L * round((x_i - x_j) / L)` per periodic axis, for any drift (a particle may be many box
      lengths away from the box);
    * the neighbour search hashes a **temporary wrapped copy** of the positions for the cell lists (cell ids only) and returns pairs; the distances and vectors of the pairs are
      then computed from the raw positions with the minimum image; the copy is discarded;
    * every site that forms a pair vector uses the one helper (`deltasph2d.py:272`, `:356`, the `sim/modules` kernels, which already take a periodic term in the generated
      code, to be checked), so there is no second convention.
    Test: a translation of the whole configuration by an arbitrary vector (not a multiple of L), and by several box lengths, leaves all fluid sums unchanged to round-off; a
    particle that crosses the border keeps its raw coordinate continuous (x(t + dt) - x(t) = v dt exactly, no jump of L) and its sums are continuous; the stored positions are
    bit-identical before and after `rhs` (the arrays are not written).

    **DONE (2026-10-06, `main`).**  `sim/pairs.py`: `Periodic(lo, hi, flags)`, `min_image`, `pair_delta`, `neighbor_pairs(pos, h, periodic)` (hash on a temporary wrapped copy, images as shifted queries, pairs de-duplicated);
    `DeltaSPHConfig.periodic`; the four torch pair sites of `DeltaSPH2D` use `pair_delta`; `FluidWarp` hands the box to warpSPHCore as a periodic `DomainDescription`.  Finding: warpSPHCore's Verlet list and the warpSPH modules
    (`computeDistanceVec`, `wrapCellComponentPeriodic`) already take the minimum image of raw positions for any integer box offset per particle (probe: sums equal to 1e-12 for offsets up to +-3 box lengths and
    translations up to 5 L; the solver's own `wp_*` modules use the same `computeDistanceVec`), so the Warp path needed only the flags.  `tests/sim/test_periodic_fluid.py` (10 tests: pairs against brute force incl. a < 3-cell box,
    invariance under translation / per-particle box offsets, positions not written, warp modules = torch oracle, steps keep the raw trajectory, uniform stream through the seam); suite 952 passed, harness PASS unchanged.
    The DFSPH solver does not take `periodic` yet (item 2).

1b. **Periodic wall integrals, one real body (user, 2026-10-06).**  Per-slot shift, no image bodies.  The slot builder of `fixedadj.py` (polygon and disk) takes the raw
    query position, forms the integer image shift of the body (`round((x - c_body) / L)`, valid for any drift), tests the shifted query against the static cell list of the
    body, and stores the shift in the slot; the contraction evaluates the wall geometry with the shifted query position of that slot (a local value; the raw position is never
    written).  There is one body: its poses, loads, mass and kinematics are the real ones; the periodic images exist only as shifted queries.  A disk needs R + support < L/2;
    an arbitrary body needs the support below half the free gap (checked at build, with a clear error).  Fibre bundles: the shift is per fibre slot (each fibre of the bundle has its own
    nearest image).
    Test: a periodic array evaluated with the shift equals the same body replicated 3x3 (a test-only construction) with the box clipped, per particle and channel, to the table
    accuracy; the loads of the single body equal the per-image loads summed over the replicas divided by the number of images.
    **DONE (2026-10-06, `main`), with a simpler mechanism than per-slot kernel shifts.**  The body-frame conversion is the one place the image enters: `Body.toLocal(x) = pose.toLocal(c + min_image(x - c))`
    (`Body.image` / `relative`, set by `Scene.setPeriodic(box, support)`, which `DeltaSPH2D` calls from `cfg.periodic`), used by `fixed_adjacency`, `Scene.candidates`, `signed_distance`, `inside`,
    `velocityAt` / `accelerationAt` and the lever arms of the loads.  So every wall consumer (fused kernels, cover, cone area, Laplacian, tensile, no-penetration) sees the nearest image and no Warp kernel
    changed.  A compact body (extent + support < L/2, checked at build) needs nothing else; a bundle that spans the box carries the tiled copies of its disks inside the one body
    (`DiskArrayRep.tiled`: images k in [-2, 2]^2 within the reach of the box around the centre; the broadphase box of such a body is the whole box), so a query meets each fibre once and the loads of the single body
    are the sum over its rows.  Limits: a bundle with images cannot rotate (offsets are in the body frame), the torque of a bundle is about its centre with image arms (meaningful for one compact body only),
    the wall-particle provider (`ParticleBoundary`) and the oracle scene path (`cone_area_scene`) are not periodic (the periodic solver needs the fused wall).
    Tests: `tests/scene/test_periodic_bodies.py` (fused lam / G / Cov / A of raw positions with integer box offsets per particle equal the explicit 3x3 / 5x5 image bodies at the wrapped positions to 1e-9,
    one fibre, a six-fibre bundle with fibres at the seam, a channel periodic in x only, misuse refused, signed distance / inside), `tests/sim/test_periodic_wall.py` (`rhs` and loads invariant under per-particle
    box offsets, a symmetric fluid at rest exerts no force on the fibre, steps eager and graph keep the offsets, translating problem + body together is a symmetry).

1c. **Prescribed-velocity frame.**  `cfg.pinned` (bool mask or region callback on the initial positions): velocity set to U∞ at the end of every stage, excluded from
    shifting and no-penetration, still counted as fluid in all sums (density, pressure); the same mask is what the `zeros` / `constant` BC closure of item 4 uses.
    Test: a uniform stream in an empty periodic box is stationary (all rates zero).
1d. **Load extraction.**  Drag and lift coefficients from `wallLoads` (pressure + wall viscous + impulse), averaged over a window; a helper that computes C_D, C_L,
    Strouhal number (FFT of the lift) and the booked load history.  `loadsAt` on an exported trajectory gives the same numbers (test).

**1c DONE (2026-10-06, `main`).**  `sim/pinned.py` `Pinned(slabs, velocity, ramp)`: the band is slabs `|min_image(x_axis - c)| < half` of the periodic box (a slab at the seam wraps); inside the momentum equation is replaced
by the prescription (acceleration weighted away, velocity set to the stream at the end of every step in `finalize`, shift off); the particles remain ordinary fluid particles in every sum.  `cfg.bodyForce` is the momentum-only driver
(not gravity: outside the hydrostatic density-diffusion term, the wall pressure condition and the no-penetration law).  `tests/sim/test_pinned_frame.py`: uniform stream stationary through the seam, band holds the stream and drives
the rest, body force gives v = f t with no pressure response, fibre + band + body force graph = eager and offsets do not matter.
**1d DONE.**  `sim/loads.py`: `LoadHistory` (books `wallLoads` per step), `loads_from_frames` (pressure + viscous from exported raw frames via `loadsAt`, interval means), `coefficients` (C_D along the stream, C_L 90 deg
counter-clockwise, 1/2 rho U^2 D), `strouhal`, `window_mean`; `tests/sim/test_loads_helpers.py`.

### Validation ladder (each rung is a script in `scripts/` and a short test)

1. **Rest and invariance.**  Fluid at rest in the periodic box with an obstacle: loads are the hydrostatic buoyancy only (no spurious drag from the periodic wall sums);
   uniform stream without obstacle stays uniform.
2. **Momentum balance (exact, no literature).**  Body-force driven steady state in a periodic cell: total wall drag = body force × fluid mass.  Checks that the wall
   loads, the periodic pair sums and the booked loads are consistent with each other; error is the residual of the unsteadiness.
3. **Plane Poiseuille / Couette, periodic in x** with the fused tank walls: velocity profile against the exact parabola, wall shear against the exact value (tests the
   wall viscosity under periodicity without the fibre element).
4. **Stokes drag of a periodic array of cylinders**, body-force driven, Re ≪ 1, solid fractions c = 0.05 … 0.4: drag per length against the Hasimoto /
   Sangani–Acrivos square-array result F = 4πμU / (−½ ln c − 0.738 + c − 0.887 c² + 2.04 c³) (constants to be verified against the paper before the number is used as a gate), and
   permeability K = μU/(ρ g) from the superficial velocity.  Includes a non-circular body (SurfaceRep ellipse) against the circular result at equal area as a smoke test, and a
   fibre bundle of several radii in one cell.
5. **Flow past a cylinder with the Dirichlet frame**, Re = 20, 40 steady: C_D against the literature for the blockage of the box (Dennis & Chang; Tritton), wake length;
   Re = 100: Strouhal number ≈ 0.16–0.17, mean C_D, RMS C_L.  The box size and frame width are a parameter study (blockage correction), since the Dirichlet frame is not the
   free-stream boundary condition.
6. **Sensitivity of the loads** to the closure choices (free-slip vs no-slip, `constant` vs `zeros`), to the resolution (R/H), and to the representation (disk element vs
   polygon vs SdfRep of the same disk): the representation-independence of the loads is a test of the package itself.

### Results so far (2026-10-06, `scripts/studies/periodic_cylinder_array.py`, `periodic_channel.py`, `stokes_array_ref.py`)

Setup of every run: Wendland C2, H = 4 dx, c0 = 10, alpha chosen for nu = 0.0185 (the shear-wave decay of the discretisation measures nu_eff = 0.955 alpha c0 H / (8 xi)), body force f, Re ~ 0.6, no-slip wall.

* **Rung 1 (invariance, rest)**: tests (`test_periodic_fluid`, `test_periodic_wall`).  **Rung 2 (momentum balance)**: PASS.  Steady-state plate / cylinder loads equal the body force on the fluid particles present to
  0.2-0.6 % at every resolution (cylinder n = 32 / 64 / 96: 1.006, 0.999, 1.002; channel 0.996-1.001).  Against the geometric fluid area (L^2 - pi R^2) the balance looks 1-2 % off only because the cut lattice holds ~0.8 % fewer
  particles than the area (a half-spacing gap at the wall).  Lift of the symmetric array: zero to noise (1e-5 of the drag).
* **Viscosity under periodicity** (shear-wave decay, no walls): nu_eff = 0.955 +- 0.01 of alpha c0 H / (8 xi), linear in alpha, independent of n at fixed H/dx.
* **Reference for rung 4**: the Sangani-Acrivos square-array expansion K = 4 pi / (-1/2 ln c - 0.738 + c - 0.887 c^2 + 2.038 c^3) = 30.30 at c = 0.1257 is confirmed by an independent Fourier volume-penalisation Stokes
  solve (K = 31.9, 30.7, 30.4 at 128, 256, 384 points, converging to 29.8-30.3; the momentum balance of the solve is exact).
* **Rung 4, the drag is 8 % low and does not converge away**: K (flux-form `noslip` wall, fixed nu) = 28.26, 27.80, 27.71 at n = 32, 64, 96 (R/dx = 6.4 ... 19), i.e. 0.915 of the reference; with the exact antisymmetric-mirror wall
  Laplacian (`wallViscosityForm="noslipMirror"`, new: the `laplacian` integral with the whole relative velocity flipped) 28.6 at n = 48 (0.943).  Resolution does not remove it, so it is a wall-model offset, not sampling.
* **Rung 3, plane Poiseuille between periodic-spanning plates** (new: `Body._checkCompact` accepts a body that spans the box): the profile amplitude over the parabola of the bulk viscosity is 0.898 / 0.913 / 0.927 (mirror,
  n = 32 / 48 / 64) and 0.950 / 0.961 / 0.966 (flux form `noslip`): first order or slower in dx, equivalent to a no-slip plane 0.4-0.6 dx inside the fluid.  The channel is too slow while the cylinder array is too permeable,
  so the two errors do not have one sign: the wall viscosity closure is not yet accurate to better than ~5-10 % in either geometry.
* **Conclusion**: the periodic machinery (1a-1d) is consistent (invariance to round-off, exact momentum balance, loads recoverable); the quantity that limits permeability / drag accuracy is the no-slip wall viscosity
  model of the solver (a model question, independent of periodicity).  Candidate next studies, in order: wall-closure calibration on the channel + a Couette annulus (a convex curved wall with an exact solution) with the
  mirror and flux forms, the dependence on H / dx and on shifting, and a sampling check of the half-spacing wall gap (fluid mass).

### Wall viscosity closure investigation (2026-10-06, after the first ladder results)

* **Literature (read: Mayrhofer, Ferrand, Kassiotis, Violeau, Morel 2013, arXiv:1304.3692, sections 2-3, 6-7).**  The semi-analytical wall boundary conditions of Ferrand et al. use the Laplacian with boundary term
  `-(2/gamma) sum_s f_s (grad B)_s . grad gamma_as` where `mu (grad v) . n = tau` is the wall shear stress and, for laminar flow, `tau = nu v(z) / z` with `z` a short distance from the wall: this is the solver's
  `wallViscosityForm="noslip"` (first order: v(z)/z estimates the wall gradient).  Mayrhofer et al. section 6 generalise the wall boundary condition to arbitrary order by a weighted local least-squares polynomial of the field along the wall
  normal with the wall value constrained (Robin condition), and show on a wave problem that m = 2 reduces the error by up to 30 % relative to m = 1; section 7.2 removes the hydrostatic / body-force part of the
  extrapolated pressure for the wall pressure condition (consistent with `bodyForceAtWall`).  `noslipCurv` (this session, derived locally: gradient `v_rel / d - (d/2) lap v` with `lap v` from the particle's viscous acceleration) is the
  second-order member of the same family obtained from the momentum balance instead of a least-squares fit; no citation for this exact form was found.
* **Planar wall**: `noslipCurv` brings the plane Poiseuille amplitude to 1.011 / 1.006 / 1.001 at n = 32 / 48 / 64 (flux form 0.95-0.97, mirror 0.90-0.93), momentum balance 0.999.
* **Curved walls (Taylor-Couette, r1 = 0.2, r2 = 0.5, `scripts/studies/taylor_couette.py`)**:
  * *Torque bookkeeping, FIXED*: the tangential wall friction was booked at the contact point while the fluid loses the momentum at the particle position, so the torque was not conserved between the two walls (outer / -inner
    1.13 at n = 48, 1.21 at n = 32).  `wallFrictionLever="particle"` (now the default) books it where the fluid loses it: outer / -inner = 1.0004, and `tests/sim/test_wall_friction_torque.py` checks
    `sum torque + sum m x cross acc = 0` to 1e-9 (the contact lever fails it by > 1 %).  The force on a body is unchanged.
  * *The flow is still not right* (n = 32 / 48, after the lever fix, torque conserved): flux form `noslip`: profile amplitude 0.931 / 0.949, inner torque 0.774 / 0.77 of the exact; `noslipCurv`: 0.908 at n = 48 (the full Laplacian is
    not the wall-normal second derivative on a curved wall: for Couette flow lap u = 0 but u'' = 2B/r^3 != 0, so the correction must carry the curvature terms; it is exact on flat walls only); `noslipMirror` (exact
    wall integral): torque 1.031 of the exact and balanced 1.001, but the profile is non-monotone-wrong (u_theta 0.159, 0.163, 0.140 vs exact 0.175, 0.141, 0.114 in the three inner bins) and the outer bin does not vanish.
    So no closure is right on both walls; the errors are 5-25 % at r1 / H = 2.4 and r2 / H = 6.
  * Periodic cylinder array: K = 27.5 vs 30 (the body force in the wall pressure condition changes it by < 0.5 %).
  * Candidate next steps (decision pending): (a) the weighted local least-squares polynomial along the wall normal with the wall value constrained (Mayrhofer et al. section 6; second order for flat and curved
    walls, needs the fluid neighbour sums of each near-wall particle along its own normal); (b) a curvature-aware version of `noslipCurv` (normal second derivative from the vector Laplacian with the metric terms);
    (c) first check that the FLUID operator transmits torque in rotational shear correctly (the alpha viscosity is a central pair form; a Couette flow in a periodic annulus-free setting, e.g. rigid-body-subtracted shear, would
    separate the wall closures from the bulk operator).

### Risks

* The Dirichlet frame pins a region that is also a shifting / density-diffusion neighbour; edge effects at the frame feed the wake through the periodic image.  Rung 5 quantifies this.
* The wall viscosity under the relative no-penetration law is not yet exercised at Re ≪ 1; rung 4 is the first test (a stronger gate than the dam break).
* Periodic + `fixedAdjacency`: the slot capacity K is unchanged (one image per fibre per query within the support); the slot builder reports overflow in the existing check.
* Unwrapped positions grow with time, which costs float32 resolution in the minimum image.  Not a risk in practice (user, 2026-10-06): runs last a few domain-passing times; only very long runs or fast flow in a small domain would see it, and neither is used.  No mitigation planned.
* Time step at Re ≪ 1 is viscous-limited; the graph step handles a constant dt, an adaptive dt is item 7.

## Remaining items (short notes)

2. **DFSPH fused.**  The fused contraction already produces lam, G, Cov, A; DFSPH additionally needs the boundary mass-flux terms (m1), the force and the torque on the body.
   Add the outputs to `FusedWall` for the DFSPH kernel groups, move `dfsph2d.py` onto `AnalyticBoundary`, keep `dfsph_validation.py` as the gate.
3. **Per-body loops.**  `_wall_terms` and `_body_pack` loop over bodies in torch (~0.3 ms each).  Segment-reduce with `index_add_` / a Warp kernel over (body, row); gate: bit-level
   harness unchanged, 16 separate bodies < 2× a bundle.
4. **BC closures.**  One function `wallVelocity(policy, x, v, normal, bodyVelocity)` returning the wall-side velocity for the viscous / continuity / Laplacian terms, for
   `constant` (the value is the prescribed velocity, no relative law), `zeros`, `freeSlip`, `noSlip`; no-penetration remains the impulse form only.  Tests per policy: static fluid, uniform stream, Couette.
5. Shifting Mach scaling from the relative speed to the nearest wall; free-slip mirror from the wall-point velocity; Galilean-boost test (uniform boost leaves the relative dynamics unchanged).
6. `scripts/deltasph_regress.py` gains `--dtype f32` with its own baselines and margins (f32 vs f64 sets the margin, ~1e-6 on the aggregates).
7. `velocityVerlet`, `leapFrog` and an RK scheme through `DeltaSPHSystem`; the exact-pose body integration is the check (rotating body, constant angular rate); adaptive dt and `gravityFn` become hooks of the system.
8. `docs/derivations/disk-element.md` with a sympy / mpmath check script in `scripts/derivation_checks/`; the ε⁴ ln ε panel boundary and the two radius branches are the non-obvious parts.
9. `tests/test_wheel_install.py` (marked slow): build the wheel, install into a temp venv with `--no-deps` + warpSPHCore from the checkout, import `warpSPHBoundaries.scene` and evaluate one disk.
10. A patch series in a scratch clone of warpSPH (nothing pushed): provider hook in `_deltaSPH_rhs`, `RigidBody.representation`, `filterRegion`, detector / shifting partial-sum hooks, the generated `wp_*` modules.
