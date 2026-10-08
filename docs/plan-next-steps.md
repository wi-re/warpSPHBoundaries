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
* **[CORRECTED 2026-10-07, see "Reference correction" below: the reference here was the per-cell drag; for the fluid-only body force it is (1 - c) K_SA = 26.49, so every K in this section is 4-8 % TOO HIGH, not low.]** **Rung 4, the drag is 8 % low and does not converge away**: K (flux-form `noslip` wall, fixed nu) = 28.26, 27.80, 27.71 at n = 32, 64, 96 (R/dx = 6.4 ... 19), i.e. 0.915 of the reference; with the exact antisymmetric-mirror wall
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
  extrapolated pressure for the wall pressure condition (consistent with `bodyForceAtWall`).  `noslipCurv` is NOT that: it was meant as the second-order wall gradient `v_rel / d - (d/2) lap v` with `lap v` from the viscous acceleration, but the consistent algebra
  (`term = (term0 + kappa viscf) / (1 - kappa)`) gives a plane-channel amplitude of 0.920 / 0.958 (n = 32 / 64), worse than the flux form (0.950 / 0.966); the implemented `(1 + kappa)` (a sign slip found later) fits the channel (1.011 / 1.001)
  but has no derivation.  It stays as an EXPERIMENTAL flag, not a method; the first-order wall error is not the missing (d/2) lap v of the gradient alone: the linear continuation of the wall extension misses the (lap v / 2) s^2 part
  of the profile in the whole wall region of the Laplacian integral, a moment of the wall region that needs its own geometric integral.
* **Planar wall**: the empirical `noslipCurv` gives the plane Poiseuille amplitude 1.011 / 1.006 / 1.001 at n = 32 / 48 / 64 (flux form 0.95-0.97, mirror 0.90-0.93), momentum balance 0.999 (see the caveat above).
* **Curved walls (Taylor-Couette, r1 = 0.2, r2 = 0.5, `scripts/studies/taylor_couette.py`)**:
  * *Torque bookkeeping, FIXED*: the tangential wall friction was booked at the contact point while the fluid loses the momentum at the particle position, so the torque was not conserved between the two walls (outer / -inner
    1.13 at n = 48, 1.21 at n = 32).  `wallFrictionLever="particle"` (now the default) books it where the fluid loses it: outer / -inner = 1.0004, and `tests/sim/test_wall_friction_torque.py` checks
    `sum torque + sum m x cross acc = 0` to 1e-9 (the contact lever fails it by > 1 %).  The force on a body is unchanged.
  * *The flow is still not right* (n = 32 / 48, after the lever fix, torque conserved): flux form `noslip`: profile amplitude 0.931 / 0.949, inner torque 0.774 / 0.77 of the exact; `noslipCurv`: 0.908 at n = 48 (the full Laplacian is
    not the wall-normal second derivative on a curved wall: for Couette flow lap u = 0 but u'' = 2B/r^3 != 0, so the correction must carry the curvature terms; it is exact on flat walls only); `noslipMirror` (exact
    wall integral): torque 1.031 of the exact and balanced 1.001, but the profile is non-monotone-wrong (u_theta 0.159, 0.163, 0.140 vs exact 0.175, 0.141, 0.114 in the three inner bins) and the outer bin does not vanish.
    So no closure is right on both walls; the errors are 5-25 % at r1 / H = 2.4 and r2 / H = 6.
  * *Bulk operator in curved shear* (new check, exact Taylor-Couette profile, interior particles, fluid-only operator, exact value 0): the azimuthal viscous acceleration is 28 / 10 / 4.4 / 2.3 / 0.7 % of nu u'' at H/dx = 3 / 4 / 5 / 6 / 8, independent of the resolution at fixed H/dx
    (lattice anisotropy of the alpha-viscosity pair sum: u'' and u'/r cancel only to that accuracy).  The default H = 4 dx therefore carries a ~10 % consistency error in curved shear; at H = 6 dx it is 2 %, and the annulus
    profile (0.907) and torque (0.783) did NOT improve with H = 6 dx, so it is not the dominant error there.
  * Periodic cylinder array: K = 27.5 vs 30 (the body force in the wall pressure condition changes it by < 0.5 %).
  * Candidate next steps (decision pending): (a) the weighted local least-squares polynomial along the wall normal with the wall value constrained (Mayrhofer et al. section 6; second order for flat and curved
    walls, needs the fluid neighbour sums of each near-wall particle along its own normal); (b) a curvature-aware version of `noslipCurv` (normal second derivative from the vector Laplacian with the metric terms);
    (c) first check that the FLUID operator transmits torque in rotational shear correctly (the alpha viscosity is a central pair form; a Couette flow in a periodic annulus-free setting, e.g. rigid-body-subtracted shear, would
    separate the wall closures from the bulk operator).

### Exact-moment wall closure `noslipMoment` (2026-10-06, user request: try the exact geometric moment integral first, then Mayrhofer)

* **Derivation** (`sim/wallmoments.py`).  The fluid sum is the pair form `acc_i = fac sum_j V_j / rho_ij P(r_ij) (v_i - v_j)`, `P = (W'/r^3) r (x) r`.  Truncated at the wall, the missing part is the same integral over the solid S with a continuation of the
  velocity field; for a continuation that is a polynomial of the wall distance, relative to the wall, `w_ext = a s + (L/2) s^2`, it is `A_w = -beta [T1 a + (T2/2) L - T0 w]`, `beta = fac wm / rho`, `Tk = int_S P s'^k dA'` (tabulated
  for the half-plane, diagonal in (n, t), checked against direct quadrature; a rigid wall motion adds nothing: P annihilates every field with `u_ij . r_ij = 0`).  `a`, `L` from `w(d) = w_i` and the viscous balance of the particle (fluid
  pair sum + wall term = `nu_p (lap w + 2 grad div w)`, `nu_p = fac / 8`; normal component factor 3; tangential curvature terms `lap w_t = w'' + kappa w' - kappa^2 w`, `kappa = div n` by a central difference of the wall normal).
* **Plane wall**: operator on the exact Poiseuille parabola: rows next to the wall within 3-16 % (flux form: -0.15 to 2.7 times); channel amplitude 0.967 / 0.981 at n = 32 / 64 (flux 0.950 / 0.966, mirror 0.90 / 0.93), momentum balance 1.001.
  The remainder is lattice quadrature of the FLUID sum (continuum-exact closure, discrete particle sum): measured fluid sum / continuum prediction for the same quadratic field = 1.00, 0.79-0.87, 1.0-1.06, then the bulk 0.96 for rows 0..4
  (row 1 depends on n through the linear part `a` of the profile).  Larger H/dx lowers it (0.98 / 1.08 / 1.0 at H/dx = 6) but the channel amplitude falls with H (0.967, 0.967, 0.960, 0.935 at H/dx = 4, 5, 6, 8 for n = 32)
  because a particle then sees both walls and each wall solves its own profile independently (the double counting needs a joint solve).
* **Curved walls**: cylinder array K = 28.34 (0.935 of the reference; flux form 0.91); Taylor-Couette (r1 = 0.2, r2 = 0.5, n = 32): profile 0.943 and UNIFORM in radius (every earlier closure had a radius-dependent profile), torque balanced
  0.9966, inner torque 0.766.  Wide annulus (r2 = 0.7, gap 16 dx so that one particle never sees both walls), H/dx = 4 / 6 / 8: profile 0.939 / 0.934 / 0.939, torque 0.858 / 0.858 / 0.870: independent of H, so NOT the lattice error of the bulk
  operator (which falls from 10 % to 0.7 % over that range).  Fit of the measured profile `u = A' r + B' / r`: u at the INNER wall 0.1875 for a wall speed 0.2 (6 % slip at the convex rotating wall), at the outer wall +0.0013 (no slip).
* **Torque accounting** (`tc_balance`, n = 24): the pair-viscous torque into the inner shell and the wall torque on it cancel to 1e-3 at three radii (the discrete system conserves angular momentum), but the torque the discrete operator transmits
  (0.0077) is 0.81 of the continuum value for the fitted profile, `-4 pi nu B' = 0.0096` (nu = the shear-wave viscosity): the central pair form transmits less torque in rotational shear than in rectilinear shear at H/dx = 4 (and 0.91-0.93 for
  H/dx = 4 - 8 in the wide annulus).  Together with the slip at the convex wall these two account for the torque ratio 0.77-0.86.
* **Monaghan switch** (user, 2026-10-06): the solver passes `approachOnly=False` (all pairs, `fluidwarp.py`), which is why the measured bulk viscosity is 0.955 of the nominal `alpha c0 H / (8 xi)`; warpSPH's default applies viscosity to approaching
  pairs only, which lowers the effective bulk viscosity by roughly 40 % at the same alpha.  Port consequence: the wall closure here (tables, `nu_p = fac / 8`) is the all-pairs form; with the switch the pair weight becomes one-sided and
  field-dependent, so the wall term would need the same switch (not covered).
* **Open**: (1) the 6 % slip at the convex rotating wall, independent of H; the tables are for a flat half-plane, a convex wall has less solid (the true moments of S need the curved region); (2) the double wall counting; (3) the neighbour-based fit
  (Mayrhofer et al. section 6) to absorb the lattice quadrature of the fluid sum.

### Curved-wall closure results after the fix (2026-10-06): the tangential closure is right; the pressure level and the layout consistency are the limit

* **Curved moment tables** (`CurvedWallMoments`, `curved_moments`): the moments `T_k` of the pair weight over the actual circular solid (convex disk / cavity, any signed curvature `kappa_w H` in [-0.7, 0.7]) by a polar quadrature of the real region,
  with the (n, t) frame turning with the position as in a flow along the wall; tabulated once over (d / H, kappa_w H), bilinear lookup at run time (no quadrature per step, nothing that grows in 3D beyond the table dimension); verified against a
  brute-force 2D integral to 5-6 digits for convex and concave walls and against the plane table at kappa = 0.  Curvature matters: tt, k = 0 at d = 0.125 H: -154 plane, -122 convex (R = H/0.6), -192 concave.
* **Bug found and fixed**: the closure needs the velocity relative to the RIGID motion of the wall AT THE PARTICLE, `v - b.velocityAt(x)`, not at the contact point (the difference omega d is ~45 % of the relative velocity at the first row); only the
  rotating wall was affected.  This was the 6 % slip at the convex rotating wall.
* **Operator test on the exact Couette flow** (true viscous acceleration 0, units of nu u''): `noslipMoment` with wall-fitted ring packing: inner rows 0-1 -0.23, +0.10, outer +0.21, -0.50 (n = 32); the flux form: -3.0, -0.1 / +3.0, -2.0.
* **Taylor-Couette dynamics** (r1 = 0.2, r2 = 0.5, n = 32, H = 4 dx): square lattice cut at dx/2: profile amplitude 0.995, wall speed reached 0.1988 / 0.2000 (slip gone), outer 0.0002; ring packing (concentric rings tiling both walls, `--ring`): profile 0.9993,
  wall speed 0.2000, outer -0.0001.  The booked torque is 0.81 (lattice) and 1.09 (ring) of the exact value although the profile is exact: the discrete pair operator transmits a packing-dependent torque (the bulk viscosity of the pair form is
  anisotropic, below).  Ring packing is the case-specific optimum for Taylor-Couette (user: acceptable).
* **Bulk anisotropy of the pair viscosity** (shear-wave decay, n = 48): nu_eff / nominal = 0.957 for waves along the lattice axes and 1.031 along the diagonal at H = 4 dx; 0.984 / 0.994 at H = 6 dx.  Calibrating nu on one direction is good to ~4 % at H = 4 dx.
* **Cylinder array (n = 48, nu = 0.0185, noslipMoment)**: K = 27.9 at the natural mean pressure (0.92 of the reference), and K DEPENDS ON THE MEAN PRESSURE LEVEL: initial density 1.00 / 1.02 / 1.05 gives K = 27.9 / 30.2 / 32.8 (n = 32, 64 at 1.02: 30.9 / 29.9).
  The physical answer cannot depend on the absolute pressure.  Cause (measured: uniform density 1 + delta at rest around the cylinder, `uniform_p`): a uniform pressure P exerts the force -2 P S_i / rho_i with S_i = sum_j V_j grad W_ij + mu grad lambda,
  the static wall-consistency residual of the layout; with the cut lattice the spurious acceleration is 18.7 at P = 2 (|a_phys| ~ 0.03), packed (`pack`, 400-4000 iterations, which stalls at |a| ~ 1.25) it is 1.2.  At the natural P ~ 0.03 the spurious force
  is ~0.3, 10x the physical one, and the sign of the pressure fluctuations switches the Antuono switch and the clamp of the wall pressure (negative pressure: no wall pressure term), which is why the unbiased run (27.9) and the biased runs differ.
  K = 30.2 at rho = 1.02 matches the reference (0.997, 1.021, 0.986 at n = 48, 32, 64) but that is two errors cancelling, not a result.
* **Open decision** (model): make the pressure force exact for a uniform pressure near a wall.  The difference form of the pressure gradient (`sum V_j (P_j - P_i) grad W + (P_w - P_i) G`) is exact for uniform P and loses the exact pairwise
  momentum conservation by O(S_i); a corrected symmetric form `... - 2 P_i S_i` is the same thing.  Alternatively a particle layout / volumes with S_i = 0 (not reachable for a curved wall by `pack`), or per-particle volumes.
* Also tried and dropped: a normal-component closure with `a_n = 0` (unstable: the polynomial over-extrapolates), the viscous term in the wall pressure condition (`wallPressureViscous`: -0.4 % on K, left as an option).

### `pressureConsistent` (option 1, 2026-10-07)

* **What it does**: the force a uniform pressure would exert through the layout residual, `[(1 + s) P S_f + (pp + s P) G] / rho` (`S_f` = the fluid-pair kernel-gradient sum, one extra call of the Antuono module with P = 1), is removed from the particle
  acceleration: the pressure force becomes the difference form `sum V (P_j - P_i) grad W + (P_w - P_i) G` with the hydrostatic extension A; a uniform pressure of any level and sign gives zero acceleration to round-off (`tests/sim/test_pressure_consistent.py`).  The
  body loads use `2 (P - mean P) G + A` (a uniform pressure exerts no net force on a closed body; without this the booked force follows the level: F / balance 0.97 / 0.85 / 0.73 at rho = 1 / 1.02 / 1.05).  Not applied at the free surface (the truncation
  there is physical): with the mask the tank and the dam break run.  `backgroundPressure` (P = c0^2 (rho - rho0) + P_b, density untouched) is the clean way to test the level.
* **Result (cylinder array, n = 48, noslipMoment)**: plain: K = 27.9 / 32.1 / 39.4 at P_b = 0 / 2 / 5 (the physical answer cannot depend on it); consistent: K = 27.17 and 27.17 at P_b = 0 and 5, F / balance 0.997 at both: the flow and the loads no longer depend on the pressure level.
  The remaining deficit against the Sangani-Acrivos / Fourier reference (30.3, 29.8-30.3) is 10 % (0.897), now a real discretisation or model error, not the pressure level.
* **Costs**: the difference form is not exactly momentum conserving (O(S_i P)); in the free-surface cases, with the interior mask, the bit baselines change (as expected) and the hydrostatic tank gets noisier: tank KE at t = 1: 5.4e-5 vs 7.3e-7, rmseBulk 1.9e-3 vs 1.5e-3;
  dam break KE vs the reference B 2.0 %, P1 arrival 2.478 t* (2.49 reference; the plain baseline 2.474).  Default off.
* **Wall renormalisation (Kulasegaram gamma, Ferrand / Mayrhofer)** does not remove this inconsistency: gamma is the ZEROTH-order partition of unity (`gamma_i = sum V W + lambda_wall`) and divides the operators; the uniform-pressure force is the FIRST-order
  consistency `S_i = sum V grad W + mu grad lambda`, which gamma does not enforce.  What does is a renormalised gradient in difference form (`L_i sum V (P_j - P_i) grad W`, `L = (sum V (x_j - x_i) (x) grad W)^{-1}`, the matrix `Mf + Mw` the shifting already assembles): exact for uniform AND linear fields (hydrostatics included),
  at the cost of non-conservative forces.  Ferrand's gamma form keeps symmetric forces through its boundary term, but needs a layout that satisfies the discrete identity (what `pack` cannot reach on a curved wall).

### Reference correction (2026-10-07): the drag of a fluid-only body force

* **The comparison above used the wrong reference.**  Sangani-Acrivos (and the Fourier solve of `stokes_array_ref.py`, which applied the force to the whole cell) define the drag as F = G L^2: a mean pressure gradient acts on the fluid AND on the solid (its
  surface integral over the cylinder adds G c L^2).  The SPH driver `cfg.bodyForce` acts on the fluid only; the flow is identical (same U) but the drag is F = f (1 - c) L^2.  Checked with the Fourier solve, force on the fluid only: U unchanged to 1e-5,
  K ratio 0.8743 = 1 - c exactly (N = 256 / 384).  Reference for the SPH driver: K_ref = (1 - c) K_SA = 26.49 (c = 0.1257).  `periodic_cylinder_array.py` now prints that; `stokes_array_ref.py --fluid-only` computes it.
* **Earlier results against the corrected reference** (n = 48 unless noted; K / K_ref): flux `noslip` 1.067 / 1.049 / 1.046 (n = 32 / 64 / 96), `noslipMirror` 1.079, `noslipMoment` plane tables 1.070, curved tables 1.053, curved at rho = 1.02 1.140,
  `noslipMoment` + `pressureConsistent` 1.025.  Every closure has TOO MUCH drag (flow too slow): the same sign as the plane channel (amplitude 0.97-0.98 with `noslipMoment`).  The earlier statements "8-10 % low", "the two errors do not have one sign" and
  "K = 30.2 at rho = 1.02 matches the reference" are void.
* **Inertia**: `noslipMoment` + `pressureConsistent`, f = 0.03 / 0.015 / 0.0075 (Re_D ~ 1.2 / 0.6 / 0.3): K = 27.13 / 27.07 / 27.12, ratio 1.022-1.024: no Re dependence, the residual is not inertia.
* **Grid-aligned square** (`--shape square --a 1/3`: walls half a spacing from the lattice rows, no curvature, no cut lattice, the particles fill the fluid area exactly; four convex corners).  Reference from the Fourier solve (`stokes_array_ref.py --square --fluid-only`):
  K = 26.60 / 26.10 / 25.98 at N = 192 / 384 / 576, extrapolated 25.84, used 25.91 (+-0.3 %).  n = 48: flux `noslip` 1.076, `noslipMoment` 1.092 (SQUARE_RESULTS).  The square is FURTHER off than the disk although its layout is ideal: candidate cause the corners
  (both closures assume one wall per particle; the moment tables are half-plane / circle moments, a convex 90 deg corner has a quarter-plane of solid).

### Lattice anisotropy of the alpha viscosity: discrete Fourier analysis (2026-10-07, `scripts/studies/viscosity_lattice_fourier.py`)

* **Analysis**: the alpha form (all pairs) on a lattice has the symbol `A(k) = -fac sum_r V W'/r^3 (r (x) r)(1 - cos k.r)`; its long-wave limit is a FOURTH-rank lattice moment.  A square lattice makes second-rank tensors isotropic but not fourth-rank ones,
  so the alpha form has two shear viscosities on a square lattice, nu(theta) = A + B cos 4 theta (axis minimum, diagonal maximum, the angular mean = the continuum value to < 0.1 %).  The Morris form (second-rank moment) is isotropic
  at long waves (0.985 of nu at every H/dx, the eta^2 regulariser); a hexagonal lattice makes the alpha form isotropic too (0.998 at H = 4 dx).
* **Numbers** (Wendland C2, long waves, nu / continuum, axis / diagonal): H/dx = 3: 0.886 / 1.109; 4: 0.959 / 1.039; 5: 0.983 / 1.017; 6: 0.991 / 1.009; 8: 0.997 / 1.003.  Wendland C4 is about 1.5x more anisotropic (4: 0.939 / 1.058).
* **Validation**: at the wave numbers of the shear-wave measurements (n = 48, |k| dx = 0.131 axis, 0.185 diagonal) the prediction is 0.9560 / 1.0328 (H = 4 dx; measured 0.9579 / 1.0313) and 0.9840 / 0.9945 (H = 6 dx; measured
  0.9844 / 0.9942): the lattice sum explains the measured anisotropy to 0.2 %.  The calibration "nu_eff = 0.955 alpha c0 H / (8 xi)" used in this section is the AXIS value, not a mean viscosity.
* **Stokes array with the long-wave anisotropic operator** (Fourier penalisation, body force on the fluid only; what a PERFECT wall would give with this bulk operator), reported the way `periodic_cylinder_array.py` reports
  (divided by the axis nu): disk 1.019 (H = 4 dx) / 1.004 (6 dx), square 1.0125 / 1.003.  Against the measurements at H = 4 dx: the disk (noslipMoment + pressureConsistent 1.022-1.024) is the bulk anisotropy plus 0.3-0.5 % from the
  wall; the square (1.064-1.067 at n = 48 / 96) has a wall error of about 5 % that the bulk does not explain (corners, slow convergence).
* **All standard kernels** (`scripts/studies/kernel_viscosity_table.py`, warpSPHCore's own kernel functions; kernels compared at equal resolution sigma / dx, sigma = the kernel's standard deviation): the anisotropy is set by the
  resolution, hardly by the kernel.  Spread diag / axis - 1 at the sigma of Wendland C2 with H = 3 / 4 / 5 / 6 dx: cubic 0.21 / 0.063 / 0.029 / 0.010, quartic 0.21 / 0.069 / 0.025 / 0.012, quintic 0.23 / 0.065 / 0.027 / 0.013,
  B7 0.23 / 0.068 / 0.027 / 0.013, B8 0.24 / 0.070 / 0.028 / 0.013, Wendland C2 0.25 / 0.083 / 0.035 / 0.018, C4 0.24 / 0.074 / 0.030 / 0.015, C6 0.24 / 0.073 / 0.029 / 0.014, HOCT4 0.26 / 0.090 / 0.042 / 0.028 (worst; also the
  largest lattice density error E0 = 2e-2 at the H = 4 dx resolution), poly6 0.16 / 0.007 / -0.001 / 0.010 (non-monotone: a lattice-sum coincidence, not isotropy).  Morris form: isotropic, 0.97-0.99 of nu for every kernel (the
  eta^2 regulariser).  At warpSPH's default packing (H / dx = xi) the spread is large for the low-order kernels: cubic 0.55, quartic 0.84, Wendland C2 0.31, C4 0.21, quintic / B7 / B8 / C6 0.12-0.13, HOCT4 0.08; poly6 has
  kernelScale = packing = 1 in warpSPHCore (H / dx = 1, 3 neighbours: placeholder constants, not usable as configured).  2D Fourier transform: negative lobes for the B-splines (cubic -6e-4 ... B8 -1e-6 of W^(0), pairing instability
  at large neighbour numbers, Dehnen & Aly 2012) and poly6 (-1.6e-2); Wendland C2 / C4 / C6 and HOCT4 non-negative to the quadrature accuracy (|min| < 1e-7 for Wendland).
  Usable for viscous flow on near-lattice layouts: the Wendland family (stable at any neighbour number) at H >= 6 dx (C2: spread 1.8 %, 113 neighbours) or the Morris form at the usual H; the cubic is cheapest per spread but pairs at
  the neighbour numbers needed.
* **Literature**: no WCSPH paper found that quantifies the direction dependence; the fixed-h/dx quadrature-error floor is Quinlan, Basa & Lastiwka (2006, IJNME); Maciá et al. (2011, IJNMF) see the non-converging floor in vortex flows;
  Violeau & Leroy (2014) use the continuous kernel transform (no lattice), Chaussonnet et al. (arXiv:1807.02315) a 1D lattice.

### Morris viscosity instead of the alpha form (2026-10-07, `kernel_viscosity_table.py --morris`)

* **Isotropic at every wave number** on a square lattice (axis / diagonal equal to ~1e-4 up to |k| dx = 1 for all kernels except poly6); the alpha form at |k| dx = 0.5, H = 4 dx: 0.914 / 0.994.
* **Calibratable**: nu_Morris / nu = (continuum bias of eta^2 = 0.0025 H^2) x (lattice factor).  Wendland C2: 0.9714 x 1.0139 = 0.985, constant to 0.06 % over H = 3-6 dx (the lattice does not resolve r < eta, hence the factor);
  Wendland C4 / C6 constant to 0.1-0.2 %; the B-splines and HOCT4 vary by 0.3-2 % with H / dx (cubic 1.006-1.029).  So one constant per kernel, no large support needed.
* **Model costs**: the pair force is along v_ij, not r_ij: linear momentum is conserved, angular momentum is NOT conserved exactly (the alpha form is central); no bulk part (the alpha form is nu (lap v + 2 grad div v), Morris
  nu lap v: less damping of compressive / acoustic modes); every wall closure here is built for the alpha weight (exact wall Laplacian, `noslipMoment` planar and curved tables, fused channels) and needs a Morris version
  (the Morris weight is a scalar: one table per k instead of nn / tt).
* warpSPH has the fluid operator (`wp_viscosityDelta`, `inviscid=False, morris=True`).
* **Implemented (2026-10-07)**: `cfg.fluidViscosity = "morris"` (Warp module and torch oracle, nu = alpha c0 H / (8 xi) / `cfg.morrisCalibration`), the `noslipMoment` closure with the Morris weight (`curved_moments(weight="morris")`,
  planar + curved tables, no normal factor 3, nu_p = the calibrated nu), the alpha-specific wall forms refuse it; `tests/sim/test_morris_viscosity.py` (operator = lattice symbol to 1e-9 for both forms, Warp and torch;
  isotropy of the symbols; tables = brute-force solid integral; refusals).  Shear wave n = 48, H = 4 dx: axis 0.9803 / diagonal 0.9748 measured vs 0.9802 / 0.9755 predicted (the difference is the larger |k| of the diagonal wave).
* **Results, H = 4 dx, cal = 0.985, noslipMoment (+ pressureConsistent for the arrays)**: disk array n = 48 K / K_ref = 0.995 (alpha 1.024); square array n = 48 1.022 (alpha 1.064); plane channel amplitude 0.979 / 0.993 at
  n = 32 / 64 (alpha 0.967 / 0.981), plate loads / body force 1.000 / 0.999.  The remaining square error (2 %) is the corners.
* **Rotating walls (bug found and fixed)**: the Morris weight annihilates a rigid TRANSLATION of the wall but not a rigid ROTATION (the alpha weight annihilates both): over the truncated solid,
  int_S K(y) (Omega x y) dA = Omega x M1, M1 = the first moment of the solid.  Without it the Taylor-Couette fluid overtook the rotating wall (amplitude 1.136, u at the inner wall 0.227 for a wall speed 0.2).  Fix: slot k = 3 of the
  Morris tables holds M1_n = int_S K (y . n) dA, the closure adds -beta Omega M1_n t and removes the opposite artefact from the fluid sum before the balance of w.  After the fix (n = 32, cut lattice): amplitude 0.9935, uniform in
  radius, u at the inner wall 0.1987.
* **Torque bookkeeping (model cost, open)**: Morris is the non-symmetric stress nu grad v.  For incompressible flow its force density equals that of the symmetric stress nu (grad v + grad v^T), so the flow is right, but the wall
  traction differs by nu (grad v^T) n, on a curved no-slip wall -nu kappa v_wall,t: the booked torque on a ROTATING body misses 2 pi nu Omega R^2 per disk (Taylor-Couette n = 32: inner 0.51 of the exact torque, 0.93 with that
  correction added; the static outer wall needs none, 0.76 = the packing-dependent family of the alpha form, 0.81 / 1.09).  Ring packing: amplitude 0.993, outer torque 1.015, inner 0.675 (1.095 with the correction).  Forces (drag,
  lift) are not affected; torques of rotating bodies need the traction correction.
* **Torque correction implemented**: on a rigid no-slip wall the traction difference is mu (grad v^T) n = -mu Omega t for ANY wall shape (the tangential derivatives are those of the rigid motion), so no net force and a torque
  -2 mu Omega A, A = the signed area enclosed by the solid's boundary (`DeltaSPH2D._solid_area`: disks, polygons, boxes, implicit disks, SDF fallback; + obstacle, - cavity), added to the booked wall-viscous torque when
  fluidViscosity = "morris".  Taylor-Couette n = 32: inner torque 0.513 -> 0.938 (cut lattice), 0.675 -> 1.100 (ring); outer 0.76 / 1.015; the packing dependence remains (alpha form: 0.81 / 1.09).
* **Cost**: cylinder array n = 48, graph step: alpha 6.01 / 3.80 ms/step (f64 / f32), Morris 6.06 / 3.86 (+1 %).
* **Verification 2026-10-07**: full suite 994 passed (before the torque correction; Morris + torque tests 14 passed after), `deltasph_regress.py check --cases tank,dambreak` OVERALL PASS with margin 0.000 (default alpha path bit-identical).
* **H = 6 dx diagnostics (alpha form, noslipMoment + pressureConsistent, n = 72)**: disk 1.0115 (perfect-wall prediction 1.004; H 4 -> 6 dx moved it by 1.25 points, predicted 1.5), square 1.049 (flux `noslip` 1.037; prediction 1.003): the
  square's wall error (~4.6 %) does not shrink with H, the corners are the open item.

### Decisions and order after the viscosity study (user, 2026-10-07)

* `fluidViscosity = "morris"` stays opt-in (not the default; free-surface cases stay on the alpha form).
* Corners and sharp features (e.g. an airfoil trailing edge) matter as much as the fibre application: the corner wall error (square array: Morris ~2 %, alpha 4-5 %, not shrinking with H) is the next model item.
  Diagnostic (square, n = 48): 52 of 308 near-wall particles (17 %) lie in a vertex quadrant (nearest point = the vertex); their curvature estimate is kappa_w H = 80-120 (the distance field is radial there), all clamped to the
  table edge 0.7 (= a disk of radius 1.4 H instead of a 90 deg wedge); face particles within H of a vertex get kappa = 0 (a full half-plane although the solid ends at the vertex).
* The wake rung (Re 20 / 40 / 100, blockage study, verified square-cylinder references) runs after the warpSPH port in its harness (numbers and video from one run).  Partial numbers from here (8.3 % blockage, D / dx = 20, f32):
  disk Re 100 alpha C_D 1.40, C_L amplitude 0.333, St 0.171; disk Re 40 alpha C_D 1.717, L_w / D 2.06, Morris C_D 1.694, L_w / D 2.13; square Re 100 alpha C_D 1.60, C_L amplitude 0.255, St 0.150.
* Order: verify (full suite, regression harness, Morris timing) and commit; Morris torque (traction) correction; corner closure; then the plan items (BC closures, DFSPH fused) and the warpSPH patch series.

### Sharp-corner rig (2026-10-07, `scripts/studies/corner_rig.py`)

* **Cases**: a periodic array of one polygon (square, isosceles triangle of apex angle beta pointing downstream = a trailing-edge stand-in, rhombus with two sharp edges, any rotation), Stokes flow driven by a body force on the fluid.
  **Reference**: Fourier volume penalisation with a polygon indicator (`fourier`, cached in .tmp): K converges at order ~1.5 in 1 / N (30 deg triangle, c = 0.08: 18.75 / 18.38 / 18.30 at N = 192 / 384 / 576), N = 576 is the default.
  Two fixes of the reference were needed for the near-wall check: (1) the spectral Laplacian rings from the kink of u at the wall (-0.81 instead of -1.11 half a spacing from a face) -> 4th-order finite differences, |FD4 - FD2| as
  the local uncertainty; (2) the penalised flow vanishes 0.26 grid cells inside the smoothed indicator (0.021 dx at N = 576, n = 48) -> the indicator is shifted outward by 0.26 cells (measured offset after: 0.002 dx).
* **`operator` (static)**: the reference field on the particles (uniform density: no pressure force), SPH viscous acceleration (fluid sum + wall closure) vs nu lap u, medians per class (face, face within H of a vertex,
  vertex quadrant) and wall row, with the partition-of-unity deficit 1 - (sum V W + lambda) as the sampling check.  **`array` (dynamic)**: superficial velocity (both components), K / K_ref.
* **Finding 1, sampling dominates on a cut lattice**: a wall not on a half-spacing line leaves an unsampled layer (square of side 13.6 dx: first row at 0.71 dx, PU deficit +5.9 %, first-row error 1.5 x |exact|; the aligned square
  of side 16 dx: 0.50 dx, -0.8 %, 0.55 x).  The closure assumes a sampled fluid side, so the static check needs body-fitted sampling: `--layers K` puts K offset curves of the (convex) polygon at (k + 1/2) dx, arc spacing dx, the
  lattice beyond (the seam lies outside the support of the wall particles for K dx > H).  This is the sampling open problem below in its simplest form (convex polygons, static check).
* **Finding 2, with layers (n = 48, H = 4 dx, Morris + noslipMoment)**: interior error 1e-4 f; faces: first row 0.57-0.66 x |exact| (also on the flat faces, not corner-specific; rows 1-3: 0.07-0.28); vertex quadrant rows 0 / 1:
  0.33-0.43 x |exact| with |exact| = 10-44 f (absolute 4-15 f): square and 30 deg triangle alike.  These are the targets of the corner closure (and the first-row face error a separate item).
* **Finding 3, integrated effect (`array`, Morris + noslipMoment + pressureConsistent, n = 48, c = 0.08)**: superficial velocity U_x / U_ref (the consistent metric; K mixes in the particle fill): 64-gon (corner-free baseline)
  0.993 (cut lattice and layers), square 0.986 / 0.986, 30 deg triangle 0.988 / 0.989, aligned square (c = 1/9) 0.988.  The dynamic result does not depend on the initial sampling (shifting relaxes it); the corners cost
  0.5-0.7 % of the flow on top of a general wall error of ~0.7 %.  The square reference used earlier (25.91, unshifted indicator) was ~1 % too permeable: the aligned square is K / K_ref = 1.010 against the corrected one.

### Wedge tables at corners (2026-10-07, `cfg.cornerWedgeTables`, experimental, default off)

* **What**: `wedge_moments` / `WedgeWallMoments` (sim/wallmoments.py): T_k (k = 0, 1, 2) as FULL 2 x 2 tensors in the particle frame (a wedge couples n and t) and M1 of the pair weight over the wedge solid of angle beta (convex and
  re-entrant), the turning-frame continuation of the curved tables, polar quadrature split at every line through the vertex where the integrand kinks; indexed by (rho, |phi|) about the vertex, mirrored particles flip the
  n-t couplings and M1_t.  Checked against a brute-force solid integral (30 / 90 / 270 deg, alpha and Morris: 1e-4 ... 5e-3, the latter the brute grid) and against the planar table for beta = 180 deg (1e-7, couplings 0).
  The closure solves a 2 x 2 system per particle (Cramer's rule: `torch.linalg.solve` is not capturable); diagonal T = the former per-component solve (results identical).  Corners from `SurfaceRep` loops / `BoxRep` (cached,
  tolerance `cornerAngleTol` = 20 deg), the distance-field curvature analytic (1 / rho in the vertex quadrant, 0 on faces).
* **Result (corner rig, Morris, n = 48)**: static, with layers: rows 1-2 near corners much better (vertex quadrant row 1 0.36 -> 0.25, row 2 0.13 -> 0.017; triangle alike) but row 0 worse (vertex quadrant 0.33 -> 0.93,
  near-vertex face 0.38 -> 0.66); integrated: U_x / U_ref square 0.986 -> 1.038, 30 deg triangle 0.989 -> 1.049 (the corners now drag too little).  Kept as an option, not the default.
* **Diagnosis of the first row (item 2)**: not ill-conditioning (the closure amplifies a fluid-sum error by 1 / (c + bn h) = 1.48 at d = dx / 2, 1.09 at 1.5 dx), and not resolution (square, n = 72: face row 0 still 0.46,
  rows 1-3 0.13 / 0.09 / 0.006).  Cancellation: next to the wall the fluid pair sum and the wall term are each O(nu tau / d) ~ 20 f, their sum O(f); the wall term is continuum-exact, the discrete fluid sum carries a lattice
  quadrature error of a few % (rows 0-1: 0.79-0.87 of the continuum value for a quadratic field, measured earlier), i.e. O(f) at the first row.  The same mechanism limits the corners' first row.
* **Proposed next (first row and corners together)**: discrete-complement moments: the wall moments as (full-plane continuum moment) - (the particle's own DISCRETE fluid moment sum_j V_j K(r_ij) (y . e)^k ...), so that fluid
  + wall is exact for linear and quadratic fields on the actual particle neighbourhood (the partition-of-unity idea of lambda = 1 - sum V W, one order up); geometry (corners, curvature, sampling gaps) then enters through the
  real neighbours.  Open choice: the continuation frame (fixed particle frame: complement-exact; turning frame: needed the curved tables for Taylor-Couette).

### Complement moments, prototype results (2026-10-07, `cfg.complementMoments`, Morris + noslipMoment)

* **Implementation**: `_complement_moments` (fixed particle frame; I_0 cancels in the closure, I_2 = -2 cal), the discrete sums as Warp modules on the fluid adjacency (`sim/modules/wallComplement`, generated with
  docs/work/refs/gen_wp_modules.py; = the torch oracle to 2e-15), graph-capturable (replay = eager bit for bit; 6.7 ms/step vs 6.1 for Morris with tables, cylinder array n = 48 f64).  The complement belongs to the particle's
  nearest body within the support only (found by the exactness test: a far plate got a copy of the near plate's term).  `tests/sim/test_complement_moments.py`: fluid + wall = nu u'' to 1e-9 on a wall-quadratic profile
  (the tables miss it by > 1 %).  Also fixed: the torch fluid path (fluidWarp = False) never set `viscf`, which `noslipMoment` needs.
* **Static (corner rig, layers)**: face first row 0.66 -> 0.11 (square), 0.57 -> 0.17 (triangle); vertex quadrant row 1 0.36 -> 0.17 / 0.42 -> 0.21; vertex quadrant row 0 (4 particles at the singularity, |exact| ~ 40 f,
  reference uncertainty 0.2-0.7 f) 0.33 -> 0.67.
* **Integrated (tables -> complement)**: 64-gon U_x / U_ref 0.993 -> 0.990; square 0.986 -> 1.014; 30 deg triangle 0.989 -> 1.023; disk array K / K_ref 0.995 -> 0.998; channel amplitude unchanged 0.979 / 0.993 (n = 32 / 64:
  the remaining channel deficit is not the wall closure; candidate: the finite-k shear-wave nu used for the reference parabola, ~0.5 %); Taylor-Couette n = 32 profile 0.9935 -> 1.0001 (cut lattice), 0.993 -> 0.998 (ring),
  torques unchanged (inner 0.943 / 1.103, outer 0.77 / 1.02).
* **Reading**: the fixed-frame continuation is fine for curved walls down to R / H = 1.6 (Taylor-Couette improves), the convex corners now drag too little (+1.4 % / +2.3 %): behind a convex vertex part of the solid lies on the fluid
  side of the particle's tangent plane, where the fixed-frame continuation assigns fluid-like velocities.  Candidate fix: at corner particles only, the turning-frame wedge continuation with the discrete quadrature correction,
  T = T_wedge,turning + (T_complement - T_wedge,fixed) (needs fixed-frame wedge moments from the same quadrature).

### Hybrid corner closure (2026-10-07, `complementMoments` + `cornerWedgeTables`)

* **What**: at corner particles T_k = T_k,complement + (T_k,wedge,turning - T_k,wedge,fixed) for k = 1, 2 (k = 0 and M1 do not depend on the continuation), the turning-frame curvature; `wedge_moments(fixed=True)` returns the
  fixed-frame moments (s~ = y . n + d) from the same quadrature (brute force <= 1.4e-3; beta = pi: T = Tf exactly).
* **Result**: static (layers) worse than the plain complement near corners (near-vertex face row 0 0.26 -> 0.35 square, 0.42 -> 0.77 triangle; vertex quadrant row 1 0.17 -> 0.29); integrated U_x / U_ref square 1.038,
  triangle 1.048 = the wedge tables alone (1.038 / 1.049).  The integrated corner drag is set by the continuation model at the corner: curved-clamped 0.986 / 0.989, fixed-frame complement 1.014 / 1.023, turning-frame wedge
  1.038 / 1.048 (target: the 64-gon baseline ~0.990).  The hybrid does not fix the corners; kept in the code path as an option.
* **Pressure side checked**: `wallPressureViscous` (the viscous term in the wall pressure condition, nu lap u = 10-40 f at corners) with the complement: square 1.014 -> 1.016, triangle 1.023 -> 1.023, 64-gon 0.990 -> 0.993: not the corner deficit.  
* **Convergence (complement, U_x / U_ref, n = 48 -> 72)**: 64-gon 0.990 -> 0.994, square 1.014 -> 1.009, 30 deg triangle 1.023 -> 1.014: first order for every shape (error ratios 1.5-1.7 for a refinement of 1.5); the corner
  excess over the 64-gon falls from +2.4 / +3.3 % to +1.5 / +2.0 %.  The corner error is a local first-order error with a larger constant, not a structural defect of the complement; no corner-specific model found that
  improves it (curved-clamped tables, wedge tables, hybrid, viscous wall pressure all tried).  Remedies with a known effect would be local resolution at sharp corners (not in this plan).
* **Plane channel deficit explained (2026-10-07)**: (1) the reference parabola used the shear-wave nu (k = 2 pi: 0.989 / 0.997 of the long-wave value at n = 32 / 64); a parabola sees the long-wave nu (= nominal with the
  Morris calibration): amplitudes 0.979 / 0.993 -> 0.989 / 0.995 (`periodic_channel.py` now prints both).  (2) On the steady DYNAMIC state (n = 32) the wall closure is exact: the exact parabola on the final positions /
  densities gives nu u'' to 1e-4 in rows 0-3 (those that see the wall), the bulk rows 4-7 are 0.07-0.55 % too viscous (on the initial lattice all rows are exact to 1e-4; row offsets of a sheared lattice change nu by
  < 0.05 %: the excess comes from the in-row rearrangement by the flow and shifting); steady residual <= 0.1 % of f; the steady profile fits A y (W - y) + B with A = 0.997 of exact and B = -0.6 % of u_max (an
  effective no-slip plane 0.024 dx inside the fluid).  Shifting off: 0.989 -> 0.991.  The remaining ~1 % (n = 32) / 0.5 % (n = 64) is the bulk operator on the flow-rearranged particles (the open sampling problem below),
  not the wall closure.
* **Decision (user, 2026-10-07)**: the complement is the default `noslipMoment` closure with the Morris viscosity (`complementMoments = None` = automatic; True / False force it, the study scripts take `--tables`); the
  alpha form keeps the curved tables; Morris itself stays opt-in; the wedge tables / hybrid remain experimental options.

### After the warpSPH port (user, 2026-10-07): application cases

Run in the warpSPH harness (numbers and video from one run), not here:
* **Water entry / slamming at prescribed motion**: constant-velocity wedge entry (Zhao & Faltinsen 1993 similarity solution, 30 deg deadrise: pressure distribution and slamming force), cylinder entry.  Needs a wall-pressure
  sampler built from the wall pressure condition (the MLS probe is erratic at impacts).  Caveats: the keel apex is a sharp corner (sampling, spray-jet resolution), single phase (no air cushion: deadrise >~ 10 deg),
  the weakly compressible first-contact spike ~ rho c0 V.
* **Floating and two-way coupled bodies**: free-fall entry (drop tests), floating bodies: the body acceleration from `wallLoads` instead of the prescribed `bodyAccelerations` (the extension point in sim/system.py), with the
  added-mass stability of light bodies; time-dependent prescribed motion uses the same hook.
* **Jet impingement**: transient (a finite slug onto a plate) as an initial condition; a steady jet needs inlets / outlets (below).
* The wake rung (Re 20 / 40 / 100, blockage, square references), see "Decisions and order".

### Open problem (not part of this plan): inlets and outlets

Open-boundary inflow / outflow is a general open problem of the solver, independent of the solid boundaries (this code keeps a constant particle count; steady jets, channel flows with outflow and wakes without a periodic
frame need it).

### Open problem (not part of this plan): sampling quality of relaxed / glass particle distributions

* The viscosity anisotropy above is a property of the regular lattice; in a flow the particles disorder, and shifting drives them towards a glass-like relaxed state.  What that does to the effective viscosity (anisotropy
  -> noise), to E0 / the first-order consistency residual, and to the wall-consistency residual S_i near bodies is not measured.  Needed: a noise measure for relaxed distributions (e.g. the spread of the operator response to
  smooth fields over an ensemble of shifted states) as a function of H / dx and kernel.
* Sampling around bodies is the outstanding part: a cut lattice (S_i != 0 on curved walls, ~1 % mass deficit), ring packing (case-specific), `pack` (stalls on curved walls).  Hexagonal lattices fix only the initial
  long-wave isotropy of the alpha form, not the sampling at the body.
* Reference for generating initial conditions: Diehl, Rockefeller, Fryer, Riethmiller & Statler, "Generating Optimal Initial Conditions for Smoothed Particle Hydrodynamics Simulations", PASA (arXiv:1211.0525): iterative
  placement by neighbour proximity and target resolution, inspired by weighted-Voronoi adaptive binning; written for compressible astrophysics, possibly applicable to obstacle-conforming initial conditions here.

### Risks

* The Dirichlet frame pins a region that is also a shifting / density-diffusion neighbour; edge effects at the frame feed the wake through the periodic image.  Rung 5 quantifies this.
* The wall viscosity under the relative no-penetration law is not yet exercised at Re ≪ 1; rung 4 is the first test (a stronger gate than the dam break).
* Periodic + `fixedAdjacency`: the slot capacity K is unchanged (one image per fibre per query within the support); the slot builder reports overflow in the existing check.
* Unwrapped positions grow with time, which costs float32 resolution in the minimum image.  Not a risk in practice (user, 2026-10-06): runs last a few domain-passing times; only very long runs or fast flow in a small domain would see it, and neither is used.  No mitigation planned.
* Time step at Re ≪ 1 is viscous-limited; the graph step handles a constant dt, an adaptive dt is item 7.

## DFSPH track (user, 2026-10-07): omniSPH-style DFSPH as the incompressible reference, scheme-agnostic boundary operators

The omniSPH-style solver (`sim/dfsph2d.py`, validated in docs/dfsph-validation.md: front 0.3 % / mean height 0.5 % of omniSPH at 8k particles, exact momentum bookkeeping) is the reference incompressible scheme for now
(omniSPH uses an integral boundary and works; warpSPH's own DFSPH variants are in troubleshooting).  Improve, expand and validate it; the boundary operators must serve every scheme (delta+, DFSPH, later ones) through
one provider interface.

| step | content | gate |
|---|---|---|
| D1 | wall terms through the provider (`AnalyticBoundary` / `FusedWall`) instead of the oracle `sceneOperation`: lam, grad lam (g0), the first-moment tensor int y (x) grad W (cov: the wall pressure p_b = p_i + a1 . y and the divergence of a rigid wall velocity, once per step for all iterates), a new `m1` output (int y W: the wall velocity of the friction); per body; the oracle stays the fallback (VolumeRep) and the check | dfsph tests + validation tables unchanged within the solve tolerance; fused = oracle per term |
| D1 status | **DONE 2026-10-07**: `DFSPHConfig.wallBackend` ('auto' = fused where every body is supported, else the oracle; 'fused'; 'scene'); new fused output `m1` (channels 1-2, all representations: polygon, disk element, box tables; = brute force to 1e-6 abs); DFSPH fused = oracle to 1e-13 in positions / 1e-10 in velocities after 30 steps (fixed and 3 rad/s hexagon), forces identical, momentum balance exact; `tests/sim/test_dfsph.py` 12 + `test_dfsph_fused.py` 8 pass; `dfsph_validation.py obstacle`: fixed hexagon 0.09190 / 1.00008 unchanged, spinning within the chaos of repeated runs; `reps`: the SDF domain now on the fused exact polygon (8e-13 from the surface run at t = 0.1, was 5e-6 on tier 3) | — |
| D2 | fluid sums on the Warp adjacency (warpSPH modules / generated modules), float32-capable; CUDA graph with chunked pressure iterations (residual checked between chunks) | same trajectories (tolerance), ms/step |
| D2 status | **first stage DONE 2026-10-07**: lean provider layout for DFSPH (only the scheme kernel's group: wall aggregate 4 -> 2-3 ms); fluid pairs from warpSPHCore's Verlet list (`fluidPairs`, reused while valid; = the cell list to 1e-12); the pressure iterates as CUDA graphs (`graphIterations`, default on): persistent buffers, pair arrays padded to a fixed capacity (25 % headroom; padding adds exactly zero) so a Verlet rebuild does not recapture (2 captures per run), dt^2 a device scalar, the host keeps omniSPH's convergence test; the hydrostatic / mirror iterate is two fused Warp kernels over the CSR adjacency (`sim/dfsph_kernels.py`), the MLS one the torch iterate (graphed with the fused wall backend only); the MLS moment matrix only for wallPressure = 'linear'.  Graphed = eager (positions 1e-13 / 4e-11 after 30 steps at 6k / 24k particles, identical iteration counts, exact momentum balance).  ms/step (tank with a 3 rad/s hexagon, f64): N = 2.1k 11.0 -> 8.7, 5.9k 18.1 -> 13.1, 23.7k 69.3 -> 18.6.  Remaining per-step host work (eager wall aggregate for the friction, Verlet check, alpha / source in torch): the next stage would capture those as well | tests 12 + 10 |
| D3 | viscosity: Morris + the complement no-slip closure (shared with delta+), torque bookkeeping, periodic domains | channel, Taylor-Couette, periodic arrays (disk, square), corner rig |
| D3 status | **in progress 2026-10-08**: D3a (shared closure) committed; D3b/c: `cfg.viscosity` (Morris, `morris_calibration` of the DFSPH lattice 0.943 at H/dx 2.5) + `NoSlipClosure` with complement moments from the DFSPH pairs, viscous force / torque booking, `cfg.periodic`, `cfg.bodyForce`, viscous dt <= 0.1 dx^2 / nu.  Shear wave nu_eff / nu 1.004; plane channel amplitude 0.9974 (n=48) / 0.9978 (n=96), plate force 1.000.  Periodic arrays: see below | |
| D3 next (resume here, 2026-10-08) | **State:** closed / periodic flows use the converged compact projection (`densitySolve=False, projection='compact', projectionTol=1e-8, shifting='fixed', shiftA=0.5, divergenceGauge='min'`, projectionWall 'robin'): Stokes arrays n=48 square K 0.998 / disk 1.005 (DFSPH), delta+ `pressureSolver='projection', ddt=False` square 1.011 / disk 0.999; TGV 1.00-1.01x analytic; shared `sim/projection.py` (Warp CG, CUDA graph blocks, ACSPH check schedule).  Free surfaces stay on the omniSPH-style default (user decision; the mirror projection has the right hydrostatic pressure but surface noise / splash clusters).  **Done 2026-10-08 (session 3):** (1) channel n=48 amplitude 0.9926 / plate force 0.9990 (omni-style 0.9968 / 0.9993); Taylor-Couette (r1 0.2, r2 0.5) profile 1.0002 / inner torque 0.974 at n=48 t=30, 0.9990 / 0.987 at n=96 t=20 (omni-style profile 0.815 at t=10: the density-solve pressure again); corner rig (30 deg triangle, c 0.0806, H/dx 4, `corner_rig.py array --set "pressureSolver='projection', ddt=False"`) delta+ U_x/ref 1.0009, K/K_ref 0.968 (EOS 0.992 / 0.977); (3) `tests/sim/test_deltasph_projection.py`.  (2) n=96 arrays with the wall row (DFSPH compact, tol 1e-6, t = 15, 505 s each): square K 1.0224 / U 0.959, disk 1.0158 / U 0.968, F / balance 0.995-0.997, rho [0.98, 1.02]; **not steady at t = 15** (U_x / U_ref still rising, increments 0.010 -> 0.005 per 1.76 time units, asymptote ~0.964 -> K ~1.017): the wall row did NOT remove the ~2 % n=96 excess (n=48: 0.9996 / 1.005), so K is not monotone in n; open: tol 1e-8 at n=96, t = 30, H/dx.  **Done (user decision 2026-10-08):** (4) `cfg.closedPreset` (None: on iff fully periodic; True forces it for walled closed domains such as Taylor-Couette; False = omniSPH-style): fields still at their library default take `CLOSED_PRESET` (projection 'compact', densitySolve False, shifting 'fixed' shiftA 0.5, divergenceGauge 'min'), explicit values win; (5) multigrid / preconditioner dropped: orthogonal to the boundary integration (user).  **Next:** (2b) n=96 arrays at tol 1e-8, t = 30 are steady (U_x / U_ref 0.9627 square / 0.9713 disk) with **K 1.029 square / 1.014 disk** (n=48: 0.9996 / 1.005; F / balance 1.005 / 0.999): the n=96 excess is real, not time or tolerance; open: H/dx, wall-row coefficient, n=144; (6) free-surface projection only if wanted later (per-cluster solvability, surface detection, surface-row noise; see "Mirror projection wall"); then D4 validation expansion | details in "DFSPH in closed periodic domains" |
| D4 status | **2026-10-08**: regression vs omniSPH / representations / hexagon unchanged (docs/dfsph-validation.md s.9); hydrostatic column with corners (omni path 6.7 % RMSE), dam break DFSPH vs delta+ (agree to 1-3 % before the impact, DFSPH dissipates more after; `scripts/studies/dambreak_dfsph_vs_delta.py`); `wallPressure='linear'` found intermittently unstable (about 1/3 of runs, pre-existing); open: buoyancy of a free body vs delta+ | |
| D4 | validation expansion: tank / dam break / hexagon vs omniSPH (existing), hydrostatic column with corners, dam break vs the delta+ path, buoyancy, Stokes arrays (no c0: no pressure-level issue) | tables in docs/dfsph-validation.md |

### DFSPH in closed periodic domains (2026-10-08, `scripts/studies/dfsph_periodic.py`, `dfsph_viscous.py array [--shape square]`)

Ladder (user): TGV (periodic, no walls) -> a static obstacle without forcing -> the forced array.
- TGV: stable with and without the p >= 0 clamp, KE decay 1.05x analytic (0.64-0.69x with a 0.1 dx jitter).
- The forced disk array exploded at t ~ 7: an **initial-condition** defect, not the closure.  `carve(lamRow0)` keeps only lattice particles beyond the flat-wall first-row distance; around a staircased disk this leaves
  near-wall gaps (rho 0.76-0.88) and a 2 % volume deficit (N dx^2 / (1 - c) = 0.979) that an incompressible p >= 0 solve can never close.  `lattice_calibration()['lamCell']` (keep a particle iff its lattice cell
  starts beyond d_wall - dx/2) is volume-consistent; with it the unforced disk heals (rho_min 0.83 -> 0.98) and the forced array is stable (rho [0.97, 1]).  The signed pressure (`densityClamp=False`) closes
  the gaps but blows up (tension / particles drawn into the walls).
- Drag excess: disk U_x / U_ref 0.76 (n=48), worse at n=96; a lattice-aligned square (faces at the rest distance, no cut cells, Fourier reference K = 25.65 for c = 0.10968) 0.71.  delta+ at the same support
  (H/dx 2.5) is itself K 1.17x (disk and square): the small-support closure; DFSPH adds 5-25 %.
- Not the cause: hydrostatic vs mirror wall pressure, 60 pressure iterations, the wall in the divergence solve, a smaller viscous dt, the viscous operator (static test on the exact Fourier field: bulk to 0.3 %,
  total viscous force 4 % low).
- Mechanism found: in a closed domain the density source's mean is in the null space, the density pressure floats up (0.3-0.4 against a Stokes pressure ~5e-3), and the DFSPH wall coupling is not force-free for
  a uniform pressure (p = 0.1 uniform -> near-wall |a| 0.32 at the square, 1.3 at the staircased disk, against f = 0.03; `gradientCorrection='wall'` partial).  Started from the exact Stokes field the run holds
  the field until the pressure lifts off zero, then the slot between periodic squares stagnates.
- Added (options): `closedDomain` (zero-mean density source; auto iff fully periodic), `pressureGauge` ('min': p -= min p after the density solve; auto with closedDomain), `divergenceGauge`,
  `divergenceWarmStart`, `densityShift` (VD+PS: the density correction moves positions, the velocity and the loads carry only the divergence solve), `wallPressureFactor` (0.5 = SPlisHSPlasH's p_i / rho_i^2
  form), `densityClamp`; the divergence solve's wall force booked separately (`forcePressureDiv`).
- Results (square, n=48, t=15): baseline 0.71; gauge + zero-mean 0.77-0.80 (disk 0.78); VD+PS (divergence pressure physical, converged, warm-started, min gauge) 0.79; factor 0.5 no gain.  From the exact field
  the gauged run converges to the same 0.78: one (wrong) steady state.
- SPlisHSPlasH (`~/dev/SPlisHSPlasH`, density maps / volume maps): the same structure (wall term (p_i / rho_i^2) grad rho_b), both solves clamped, compression-only sources, capped warm starts, no closed-domain
  treatment; nothing that addresses this.
- Next: the force budget at the steady state against the exact Stokes pressure; the openMaelstrom MLS wall-pressure extrapolation for comparison; a uniform-pressure-consistent wall coupling.
- openMaelstrom (`~/dev/openMaelstrom/SPH/DFSPH/dfsph.cu`): p_b by a linear MLS fit of the fluid pressures around the closest wall point, per iterate, p_b >= 0; wall acceleration -(p_i / rho_i^2 + p_b) grad rho_b.
  Ours `wallPressure='linear'` is the integrated variant; on the square 0.77 (no change): the extrapolation is not the limit.
- **Pressure noise is the limit (2026-10-08).**  Exact Stokes pressure (Fourier, Brinkman, div(nu lap u) = 0) spans +-0.028; from the exact field the DFSPH divergence pressure has a fitted amplitude
  0.13 / 1.47 / 0.65 (steps 1 / 5 / 20) and residual 16-52 % (near walls 54-137 %); the density pressure is noise 1.6-6.7x the exact spread.
- **The converged projection is dissipative**: TGV decay / analytic 1.02 / 1.06 / 1.15 / 1.38 at 4 / 10 / 30 / 100 divergence iterations (nu = 0.0185; 2.2 / 6.7 at nu = 0.002): an exact discrete projection
  removes the discrete divergence of the moving-particle field every step.  The divergence warm start with omniSPH's one-sided residual injects energy (stale pressure): not usable as is.
- **Divergence-only variant** (`densitySolve=False`, summation density, `shifting='fixed'` shiftA 0.5 (D = A h_s^2 per step, grad C = sum V grad W + mu grad lambda), 4 divergence iterations, wall in the
  divergence solve without clamp, min gauge): square n=48 U 0.970 / K 1.007; disk n=48 0.968 / 1.009; square n=96 0.956 / 1.024 (K_ref 25.80 for c = 0.1104); rho [0.98, 1.02]; TGV 1.011 (jitter 1.015),
  monotone.  The Lind shift (D proportional to |u|) leaves rho 0.86-1.16 (stagnant regions unshifted); continuity density drifts with the incomplete projection.
- **delta+ with the projection** (`DeltaSPHConfig.pressureSolver='projection'`: Jacobi on the scheme's own symmetric pressure force + continuity rate, 4 iterations, min gauge, torch path; delta+ shifting,
  Morris + complement closure, no DDT): square n=48 H/dx 2.5 K 26.15 / K_ref 25.91 = 1.009 (U 0.978, F / balance 0.986) at t = 13.2 (run cut by its time limit; 1.03 at t = 10.7), against 1.17 with the EOS.  The small-support closure was not the limit:
  the pressure was (EOS noise / density-solve noise).
- **Converged projection (user: a result that relies on an unconverged solver is not reliable), 2026-10-08.**  `projection='compact'`: approximate projection (Cummins & Rudman 1999) with the compact
  Morris / Brookshaw Laplacian, lattice-calibrated like the Morris viscosity, Neumann (mirror) walls, rows weighted by V/rho so the operator is exactly symmetric with the constant as null vector (the
  unweighted rows have the left null vector V/rho: removing the plain mean left an inconsistent component and the CG diverged), the source normalised (a near-solenoidal field underflows the CG inner
  products), Jacobi-preconditioned CG to `projectionTol`.  TGV converged to 1e-8: decay 1.003 (no shift), 1.006 (fixed shift), 1.010 at nu = 0.002 (DFSPH's operator at 100 iterations: 1.38 / 6.7),
  1.008 with a 0.1 dx jitter; without the shift nu = 0.002 disorders and blows up at t = 2.45.  Arrays (summation density, fixed shift 0.5, min gauge, tol 1e-8): square n=48 K 1.013 / U 0.969, disk n=48
  1.014 / 0.964, square n=96 1.031 / 0.954 (tol 1e-6, ~190 CG iterations / step); tolerance sweep 1e-3 / 1e-5 / 1e-8 / 1e-11 on the square: K 1.016 / 1.012 / 1.013 / 1.012 (converged, insensitive).
  Cost: n=48 75-110 s for t = 15 (50-160 iterations / step, eager torch CG) vs 45 s for 4 Jacobi iterations.  tests/sim/test_dfsph_projection.py.
- **Graph path + shared module (2026-10-08).**  `sim/projection.py` `CompactProjection` (scheme-agnostic: CSR pairs + weights + right-hand side): Warp CG kernels (dfsph_kernels.py; matvec, update, direction,
  scalars; the dot products as reductions, no atomics on the scalars: atomics cost 151 us / iteration at 37k particles), `projectionBlock` iterations per CUDA graph replay, a device flag makes iterations past
  convergence no-ops, the host reads it only at warpSPH ACSPH's schedule (0.50 / 0.75 / 0.85 of the previous first-converged count, then every block: ~4 reads / step), warm start from the previous pressure
  (80 -> 60 iterations).  Graphed = eager to 1e-13 after 200 steps.  ~27 us / CG iteration at 2k-37k particles (launch floor); step: N = 2.3k 3.5 ms (4-Jacobi path 1.8), 9.2k 6.7 (3.4), 37k 17.4 (10.5) with
  64 / 121 / 229 iterations (Jacobi-preconditioned CG: ~ n).  Square n=48 unchanged (K 1.0128), 53 s for t = 15 (eager CG 114 s).
- **delta+ on the shared projection**: `pressureSolver='projection'` now uses `CompactProjection` (the Warp fluid path, the step eager, the CG graphed): square n=48 H/dx 2.5 K 1.015 (U 0.987 U_ref, F / balance
  1.0005), 427 s for t = 15 (EOS 1.17).
- **Wall row (Robin) in the compact operator** (`diag` of `CompactProjection.solve`): with Neumann walls the delta+ disk grew 1.4x per step (CG converged every step): delta+ corrects a near-wall particle by
  -2 P G / rho and its continuity counts the wall flux with the mirror factor 2, so P_i moves the next divergence by -4 P |G|^2, which the fluid-only Laplacian does not see.  The scheme-consistent diagonal
  D_i = -4 V_i |G_i|^2 (delta+), -dt^2 V_i (1 / rho_i^2 + 1) |g_k,i|^2 (DFSPH) keeps the operator symmetric (definite with walls: no mean removal then).  delta+ projection: the density held at rho0 (integrated it
  drifted 2.5 % in 1100 steps without the EOS feedback).  Results n=48 H/dx 2.5: **DFSPH square K 0.9996 / disk 1.0048; delta+ square 1.0114 / disk 0.9994** (delta+ EOS 1.17, omniSPH-style DFSPH 1.25-1.45).
  delta+ projection ~6 min for t = 15 (eager step, graphed CG), DFSPH ~50 s.
- **Free surfaces (2026-10-08, `scripts/studies/dfsph_freesurface.py column|dam`, in progress).**  Added: `CompactProjection.solve(dirichlet=)` (P = 0 rows, eliminated symmetrically), DFSPH `freeSurface` /
  `surfaceRho` (surface = summation density with lambda < 0.85; P = 0 there; the shift tangential-only in the surface layer, Lind 2012), the pressure-independent wall part (hydrostatic extrapolation) moved into
  v*, `projectionDensity` (compression-only density term), `projectionGradient` ('symmetric' + wall row | 'difference' | 'renormalised'), `projectionWallFlux`.  Hydrostatic column 0.4 x 0.3, dx 0.01, t = 2:
  omniSPH-style default stable, KE tail 6e-5, but its density-solve pressure is 12 % off rho g (H - y) (RMSE / rho g H, bulk and walls).  Compact projection: 'symmetric' + wall row gives 5 % of the
  hydrostatic pressure (the near-wall particle's own wall force absorbs the floor flux: the column free-falls); 'difference' (Neumann walls) 66 %; with the wall flux doubled (the source's mu grad lambda term
  sums to lambda(0) ~ 1/2 of the flux into the wall; delta+'s continuity has the mirror factor 2) 93-105 %, but the surface / top-corner particles (P = 0, one-sided gradient) are kicked and it blows up at
  t ~ 0.7; the renormalised gradient fixes the surface and moves the hot spot to the bottom corners (the wall's hydrostatic push now double counts with an exact fluid gradient).  Open: a consistent
  Neumann wall in the compact projection (the wall's contribution to the compact Laplacian and to the gradient from the same extrapolated p_b, instead of the DFSPH flux term with an ad hoc factor).
- **Mirror projection wall (`projectionWall='mirror'`, 2026-10-08).**  The consistent Neumann wall of a mirror (ghost) continuation: velocity mirrored (odd normal part) -> the source's wall flux twice the
  integral term (the factor 2 is the mirror, not ad hoc; delta+'s continuity has it); pressure mirrored (even) -> the walls add nothing to the compact Laplacian; correction = the renormalised fluid gradient
  (exact for a linear p with a one-sided support); no pressure push in v* (the Neumann condition makes the hydrostatic pressure from v* = v + dt g); Dirichlet P = 0 on the surface; loads = the wall integral
  of p.  Hydrostatic column (nu 1e-3): **the pressure is hydrostatic from step 2 on, slope 0.999, scatter 0.1-0.4 % of rho g H** (omniSPH-style default: 10 % RMSE).  But the surface row carries a growing noise
  (vmax ~0.1 m/s at y ~ H; inviscid it grows 10x / 100 steps; nu 1e-2 / XSPH 0.1 damp it to ~0.04 m/s; plain difference gradient and stricter renormalisation guards blow up; the shift and the threshold
  0.9-0.98 do not matter; 0.7 misses the corner particles and blows up at once), and runs blow up between 1000 and 2000 steps.  Dam break (0.2 x 0.8, dx 0.01, nu 1e-3, XSPH 0.1): the mirror variant diverges at
  t ~ 0.2 (CG stops converging: detached clusters above the surface threshold are pure-Neumann components with an inconsistent source; compression rho 1.25 at impact); the omniSPH-style default runs (one flier
  at 50 m/s at this resolution).  Open for free surfaces: per-cluster solvability (a cluster without a Dirichlet / wall point), robust surface detection in splashes, the surface-row noise.
- **Decision (user, 2026-10-08): split by case.**  Free surfaces keep the omniSPH-style DFSPH (density solve, p >= 0); the converged compact projection (projectionWall 'robin') is the closed / periodic path.
- **The dam-break flier was ours, not omniSPH's**: inviscid the omni-style path is clean (vmax 1.8-6.3 m/s, as omniSPH with triangle walls); with nu = 1e-3 a near-wall particle under the collapsing column (d = 0.74 dx,
  rho 0.95, 10 neighbours) got a viscous acceleration of 9e4 m/s^2 (19 m/s in one step) from the complement no-slip closure: the complement books the missing FREE-SURFACE support as wall and its 2 x 2
  solve goes near singular.  Fix: `NoSlipClosure.term(coverage=)` blends the complement into the geometric tables with clamp((coverage - 0.92) / 0.05, 0, 1), coverage = sum V W + mu lambda (DFSPH passes
  its summation density; delta+ passes nothing: unchanged).  Dam break nu 1e-3: vmax 1.8-4.1 m/s to t = 0.5 (8 m/s at the far-wall impact, inviscid 6.3); arrays unchanged (square 0.998, disk 1.005).

## Remaining items (short notes)

2. **DFSPH fused.**  Superseded by the DFSPH track above.  The fused contraction already produces lam, G, Cov, A; DFSPH additionally needs the boundary mass-flux terms (m1), the force and the torque on the body.
   Add the outputs to `FusedWall` for the DFSPH kernel groups, move `dfsph2d.py` onto `AnalyticBoundary`, keep `dfsph_validation.py` as the gate.
3. **Per-body loops.**  `_wall_terms` and `_body_pack` loop over bodies in torch (~0.3 ms each).  Segment-reduce with `index_add_` / a Warp kernel over (body, row); gate: bit-level
   harness unchanged, 16 separate bodies < 2× a bundle.
   **Status 2026-10-08 (partly done):** `Scene.kinematics(x)` (`BodyKinematics`: relative positions, wall velocity, acceleration and body-frame positions of ALL bodies in one batched evaluation, guarded one-slot cache, bit-for-bit the per-body values, tests/scene/test_body_kinematics.py), the batched wall-continuity term, shared index arrays of the slot lists.  Kernel-launching torch ops per extra separate fibre per step 261 -> 103, exactly identical results on a moving multi-body scene (eager and graph); ms/step at 16 fibres (N ~ 5000, `scripts/studies/fibre_bodies_bench.py`) f64 37 -> 16 (bundle 7.6), f32 30 -> 11 (bundle 2.7): separate/bundle 3.3 -> 2.1 (f64), 3.4 -> 3.9 (f32, the bundle is faster). Gate NOT met in f32.  Remaining per fibre (device time 0.58 ms): the closed-form `_disk_cone_area_kernel`, 96 us per launch and one launch per body per stage (4.3 of the extra 9.2 ms at 16 fibres), then ~100 small elementwise launches in `fixed_adjacency`, `cone_area`, the viscous loop.  The remedy is the slot-resolved instance kinematics of docs/disk-element.md (one launch for all fibres), not more per-body batching.
4. **BC closures.**  One function `wallVelocity(policy, x, v, normal, bodyVelocity)` returning the wall-side velocity for the viscous / continuity / Laplacian terms, for
   `constant` (the value is the prescribed velocity, no relative law), `zeros`, `freeSlip`, `noSlip`; no-penetration remains the impulse form only.  Tests per policy: static fluid, uniform stream, Couette.
   **Status 2026-10-08 (done):** `sim/bcclosures.py` `wallVelocity(policy, v, n, u_body, value)` = warpSPH's ghost velocities (checked against warpSPH's own `_slipVelocity`), `Body.bc` / `Body.bcValue`: `zeros` / `constant` pin the wall velocity in the continuity, viscous, friction and relative no-penetration terms, `freeSlip` selects the exact symmetric-mirror wall Laplacian per body (alpha viscosity only; Morris refused), `noSlip` the configured form; tests/sim/test_bc_closures.py, test_bc_policies.py (static fluid; uniform stream: freeSlip and pinned-at-U keep U to 4e-16, noSlip 0.178; Couette: pinned = rigid exactly, zeros / freeSlip leave the fluid at rest to 2e-12).
5. Shifting Mach scaling from the relative speed to the nearest wall; free-slip mirror from the wall-point velocity; Galilean-boost test (uniform boost leaves the relative dynamics unchanged).
   **Status 2026-10-08:** Galilean boost measured: the wall closures are exactly invariant (4e-15 without shifting); the only break is the shifting's absolute-speed Mach number (8e-5 of U).  `cfg.shiftMachFrame='wall'` (speeds relative to the contact-weighted wall velocity; = 'absolute' for walls at rest, bit-for-bit) removes it (< 1e-9), tests/sim/test_bc_policies.py.  The nearest-wall-per-particle reference and the free-slip mirror from the wall-point velocity (needs a contact-point query per body) are not done.
6. `scripts/deltasph_regress.py` gains `--dtype f32` with its own baselines and margins (f32 vs f64 sets the margin, ~1e-6 on the aggregates).
   **Status 2026-10-08 (done):** `--dtype f32`, baseline `results/deltasph/regress_baseline_f32.json` (relative floor 1e-4; f32 is bit-reproducible across processes, margins 0.000; negative control `--perturb` fails with margins 1-87); the recorded f32 values differ from f64 by 3e-6 (dam break KE) to 7e-4 (tank rmseBulk).
7. `velocityVerlet`, `leapFrog` and an RK scheme through `DeltaSPHSystem`; the exact-pose body integration is the check (rotating body, constant angular rate); adaptive dt and `gravityFn` become hooks of the system.
   **Status 2026-10-08 (done except `gravityFn`):** `cfg.integrator` (warpSPHIntegrators display name; default 'Symplectic Euler' bit-identical to before), the stage loads weighted with the scheme's weights, the adaptive dt a hook of `DeltaSPHSystem.finalize`; tests/sim/test_integrators.py: exact body poses to 1e-12 for Symplectic Euler, Velocity Verlet, Leap Frog, Midpoint, RK4 (constant rotation rate, constant acceleration), the schemes agree on the buoyancy of a submerged bundle to 2 %.  Findings: (1) only the symplectic Euler is CUDA-graph capturable (the library's other schemes read `float(state.t + dt)`, `graphstep.GRAPH_SCHEMES`; the rest run eagerly) -- a one-line `hostTime` fix per scheme in warpSPHIntegrators would lift that; (2) the Verlet pair needs the drift fields (their end-of-step evaluation sees an old density otherwise: rho [0.79, 1.32] without, [0.9998, 1.0027] with), switched on automatically; (3) `gravityFn` stays host bookkeeping after the step (a host function of time cannot be captured).
8. `docs/derivations/disk-element.md` with a sympy / mpmath check script in `scripts/derivation_checks/`; the ε⁴ ln ε panel boundary and the two radius branches are the non-obvious parts.  **Done 2026-10-08:** docs/derivations/disk-element.md, scripts/derivation_checks/disk_element_checks.py (the -105/pi eps^4 ln|eps| coefficient fitted to 5 digits; Chebyshev decay n^-5 / n^-10 / geometric; exact-surface failure mode).
9. `tests/test_wheel_install.py` (marked slow): build the wheel, install into a temp venv with `--no-deps` + warpSPHCore from the checkout, import `warpSPHBoundaries.scene` and evaluate one disk.  **Done 2026-10-08** (builds from a source copy, throw-away venv, nothing installed into the test environment; 9 s).
10. A patch series in a scratch clone of warpSPH (nothing pushed): provider hook in `_deltaSPH_rhs`, `RigidBody.representation`, `filterRegion`, detector / shifting partial-sum hooks, the generated `wp_*` modules.
