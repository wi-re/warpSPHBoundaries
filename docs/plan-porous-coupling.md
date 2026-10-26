# Flux boundary Warp operators and porous material coupling

**Status:** [WIP] independent native operators and mixed-wall web view, 23 October 2026.
Browser fp32 port is now the next web milestone; see the final checkpoint below.
Stages 0–4 are **not yet fully accepted**. Remaining gates include retained
local-model/special-form ports, independent fixtures for those contracts, and
their complete CPU/CUDA acceptance. Pressure and porosity remain Stage 5.

The core polygon, tetrahedron, closed-skin and open-triangle analytic families,
explicit conditioned 2D and tetrahedral numerical routes, weighted planes/bands,
sphere-shell integration and weighted disk volume integration,
P0–P3 material contractions, optional CUDA whole-element search and explicit
bounded host streaming are implemented. The requested native-backed mixed-wall
explorer works in 2D/3D and its live/snapshot comparisons have been inspected;
additional model/editor extensions remain separate open items.

Tetrahedral fp32 numerical evaluation uses explicit same-dtype correction
workspace, with fp64 angular geometry preparation. The weighted sphere route
uses selected-dtype geometry/evaluation and compensated scalar summation.
Analytic raw-fp32
conditioning is deferred by user agreement; the 32 expected failures remain
recorded. Method and precision selection remain explicit. Actual CUDA
execution, not just compilation, passes the accepted finite gates. The latest
broad regression checkpoint has 896 passes, four optional legacy-WGSL
skips and 32 expected analytic-fp32 failures. The final sphere storage guard has
separate focused CPU/CUDA coverage; source snapshots distinguish both checks.

Current performance evidence includes complete preparation, static reuse,
readback, scaling and separate allocator high-water measurements. Historical
analytic/numerical speed ratios compare current implementations: the numerical
field path now shares translated moments and parallelizes outputs, whereas the
analytic consumer retains older scheduling. Those ratios do not isolate the
integration formula's cost. Host synchronization, closed-skin facet traversal
and query-specific sample storage remain practical limits.

The latest lower-order fp32 work preserves order comparison/promotion and the
original precision budgets. Default tetrahedral order remains 16; explicit
weighted sphere-shell integration starts at 24. The checkpoints below
distinguish accepted evidence from
preliminary probes. No warpSPH or shared Core/Integrator change is part of this
work.

**Precision priority:** ordinary fp32 is the primary production target on the
intended GPU, whose fp64 arithmetic throughput the user reports as 1/64 of fp32.
Keep fp64 as an accepted analytic route and a reference/verification route.
The current numerical fp32 accuracy tradeoff is accepted and passes CUDA at the
same error budgets. Measure GPU cost; CPU fp64 timing is
not a production-backend selection criterion. Paired fp32 remains explicit and
its GPU cost must be measured separately.

**Working name:** FluxBridge — analytic flux boundaries and porous coupling for SPH.

Build efficient, independently tested NVIDIA Warp operators in
`warpSPHBoundaries`. We use **flux boundaries** as the name for the kernel-integral
geometry representations; their physical wall conditions are specified separately.
The incompressible handoff must support solid obstacles, including 2D disks
representing cylinder cross sections, as well as mesh-defined porous materials.
First validate the resolved solid cases, then a stationary, saturated material
with prescribed resistance. Porosity-aware pressure coupling follows after the
incompressible solver interfaces in warpSPH have settled.

The operator work can proceed without changing or importing the main warpSPH
repository. A complete porous-flow solver cannot: timestep orchestration,
pressure projection, density constraints and material evolution belong to the
solver. Completion of the independent library stages is the handoff point,
not a claim that adding a drag callback completes porous flow.

## Resolved fibres and effective porous materials

Keep two complementary descriptions in scope. A resolved fibre array has
impermeable individual fibres and fluid in the gaps; the geometry generates the
resistance. It needs no prescribed permeability law or porous density model.
An effective material replaces many fibres by a continuum with a permeability
tensor. In the creeping-flow regime, with a representative averaging volume and
adequate separation from the macroscopic scale, aligned fibres can produce
anisotropic permeability. The tensor also depends on packing and connectivity,
not orientation or porosity alone. See the cylinder-array and aligned-fibre
references below.

For a transversely isotropic aligned bundle with unit fibre direction `a`, use
the comparison form

```text
K = K_parallel a a^T + K_perp (I - a a^T).
```

General packings require a full symmetric permeability tensor. A square or
hexagonal array of circular cross sections can be isotropic within its transverse
plane even though the three-dimensional bundle distinguishes axial and
transverse flow. A two-component 2D disk simulation measures transverse response;
an additional axial velocity formulation on the cross section or a 3D case is
required to measure longitudinal response.

The resolved and effective cases form a validation bridge: measure permeability
from converged flux-boundary cell problems, then test the porous implementation
against that response. Do not apply an additional effective resistance for the
same resolved fibres, which would count their drag twice. Low Reynolds number
and averaging assumptions must be checked before making a Darcy comparison.

## Repository ownership and dependencies

| Work | Location | Dependency on main warpSPH development |
|---|---|---|
| Analytic moments, derivatives, FEM weights and material contractions | `warpSPHBoundaries` | None; use a pinned compatible `warpSPHCore` for shared conventions where needed |
| Warp device functions, batch wrappers, adjacency, reductions and benchmarks | `warpSPHBoundaries` | None; inputs are explicit arrays and structs |
| Small matrix, constitutive and supplied-operator algebra tests | `warpSPHBoundaries` | None; these are verification harnesses, not another SPH solver |
| Resistance-aware projection and porous density/continuity changes | warpSPH, later integration branch | Requires the maintained incompressible solver contracts |
| Solid disk/cylinder arrays, channel/insert cases and solver regression comparisons | warpSPH, after handoff | Uses the settled case runner and timestep pipeline; solid cases do not depend on a porous model |
| Deformable skeleton, saturation and capillary models | Later application work | Requires additional physical models and solver integration |

Do not edit the main warpSPH checkout or shared Core/Integrator installations
during the independent stages. Do not copy its pressure solvers into this
repository. Preserve the existing `edge` to `scene` layering; new mesh and
coupling modules must also remain importable without warpSPH. Extend the
layering checks recursively when those modules are introduced.

## References and their role

1. **Böttcher, Jeske, Westhofen and Bender (2025), Implicit Incompressible
   Porous Flow using SPH.** [arXiv](https://arxiv.org/abs/2504.07739),
   [version 2 full text](https://arxiv.org/html/2504.07739v2),
   [publication DOI](https://doi.org/10.1145/3763325).
   Use version 2, revised 5 December 2025, as the pinned reference. Its porous
   density estimates, pressure treatment and implicit interaction forces are
   the principal comparison for a later porous solver. It represents both
   phases with particles; replacing solid sums by mesh integrals is a new
   discretization to derive and test, not an equivalence supplied by the paper.
   Its drag law is not assumed to be identical to the permeability-based law
   below. Reproduce the reference's parameter conventions before comparing it.
2. **Thirumalaisamy, Khedkar, Ghysels and Bhalla (2023), An effective
   preconditioning strategy for volume penalized incompressible/low Mach
   multiphase flow solvers.** [Paper](https://doi.org/10.1016/j.jcp.2023.112325).
   Reference for incorporating implicit resistance into both the pressure
   equation and velocity correction. Its discretization is not SPH; derive the
   corresponding discrete response for the chosen warpSPH solver.
3. **Kostorz and Esmail-Yakas (2021), Semi-analytical smoothed-particle
   hydrodynamics correction factors for polynomial kernels and piecewise-planar
   boundaries.** [Paper](https://doi.org/10.1002/nme.6771).
   Prior exact radial-kernel integration and SPH normalization/gradient work.
4. **Petkova, Laibe and Bonnell (2018), Fast and accurate Voronoi density
   gridding from Lagrangian hydrodynamics data.**
   [Paper](https://doi.org/10.1016/j.jcp.2017.10.024).
   Prior analytic polyhedral integration for conservative SPH-to-cell transfer.
5. **Winchenbach and Kolb (2025), Solving Boundary Handling Analytically in
   Two Dimensions for Smoothed Particle Hydrodynamics.**
   [Paper](https://arxiv.org/abs/2507.21686).
   Reference for higher-order polynomial triangle fields and FEM coupling.
6. **Sangani and Acrivos (1982), Slow flow past periodic arrays of cylinders
   with application to heat transfer.**
   [Paper](https://doi.org/10.1016/0301-9322(82)90029-5).
   Reference for transverse periodic cylinder-array flow and drag. Use its
   specified averaging and driving conventions when relating drag to permeability.
7. **An analytical model for the longitudinal permeability of aligned fibrous
   media (2012).**
   [Paper](https://www.sciencedirect.com/science/article/pii/S0266353812002102).
   Reference for longitudinal flow and the distinction between axial/transverse
   permeability and bundle orientation.

Exact integration itself is established prior work. Assess any contribution
around the weighted operators, numerical conditioning, efficient evaluation
and verified porous coupling, rather than claiming the first analytic SPH
mesh integral.

## Operator and material contract

Freeze this contract before implementing new device APIs:

- Use `y = source - query`, with `h` the support radius. State physical units,
  support scaling and kernel dimension at the API boundary. Start with Wendland
  C2/C4/C6 and the cubic spline, matching the current reference.
- Distinguish 2D area elements, 3D tetrahedral volumes, closed 3D boundary meshes
  and open 3D surface triangles. Surface measures need their own units and
  membrane constitutive laws. An open surface does not define porous bulk volume.
- Support straight-sided geometry with polynomial fields. Higher-order fields
  do not imply curved isoparametric geometry support.
- Supply interpolation, density/occupancy, kernel gradient, divergence, curl,
  first/second moments, kernel Hessian and Laplacian, and their FEM-weighted
  contractions. Specify index ordering and which field is differentiated.
  The 2D twelve stored moments and 3D twenty-two stored moments are baseline
  contracts; the additional Hessian entries and derived display quantities must
  not be confused with independent stored moments.
- Map actual viscosity, pressure and shifting consumers to their radial profiles,
  including `W'/r`, Laplacian profiles and requested kernel powers. A generic
  Laplacian is not automatically equivalent to every SPH viscous operator.
  Regularized rational profiles need their own derivation or an explicitly
  approximate contract; do not advertise polynomial closure for those by default.
- Keep derivatives of the complete weighted integral distinct from integrals
  of kernel derivatives. Basis coefficients in query-centred coordinates also
  depend on the query. Add derivative and transpose actions required by each
  matrix-free consumer; do not imply arbitrary shape adjoints are already done.
- Implement scalar, vector and tensor polynomial coefficients, including
  discontinuous fields across elements. Start with P0/P1, then P2/P3.
  Derive the degree needed by each contraction: resistance degree `q`, velocity
  degree `p`, and test degree `r` require moments through `p+q+r`. Three cubic
  factors can require degree nine, beyond the current viewer's channels and
  finite high-order reference checks.
- Keep exact geometry integrals, analytic integrals of approximate models,
  truncated expansions, coefficient tables and numerical quadrature separately
  identified. Browser implementations are comparison tools, not Warp acceptance.
- Use an explicit coverage matrix. Every exposed method must support every
  quantity in its advertised contract within stated geometry/validity limits;
  otherwise withhold that method from that contract. Do not silently substitute
  a different model or drop an output.

For the first simplified constant-density resistance model, define

```text
a_drag = -D (u - u_s),       D = nu K^(-1)
```

Here `K` has units of length squared and `D` has units of inverse time.
Porosity factors and the distinction between intrinsic and superficial velocity
must be derived for the later porous model, rather than inserted into this
baseline by analogy. Polynomial `K` generally has a nonpolynomial inverse;
store polynomial resistance for exact drag contractions, or identify the
inverse approximation explicitly. Averaging `K` and then inverting it is a
different model from averaging resistance.

Require symmetric positive semidefinite resistance and admissible porosity.
Nodal positivity alone does not guarantee positivity of higher-order Lagrange
fields. Choose a bounded/positive representation, such as suitable Bernstein
control coefficients, or provide a verified admissibility check. Keep free
fluid (`D=0`) and impermeable-wall treatment explicit.

## Stage 0 Inventory and fixtures

- [x] Map the required operations and representations to existing
  `edge`, `scene`, `mesh3d` and browser implementations. Record exactness,
  precision, missing outputs, derivative support and independent tests.
- [x] Define the versioned array/struct contract, field basis, requested output
  masks, validity flags, units and reduction ordering. Follow the device-function
  pattern in [Warp boundary functions](warp-boundary-functions.md).
- [ ] Extend arbitrary-precision references to every requested weighted product
  and degree. Build fixtures before porting the corresponding device kernels.
- [x] Define per-family error budgets and adversarial cases before tuning speed.

**Exit:** a finite operator/representation/degree coverage matrix and fixtures
with reproducible independent references. No completed box is inferred from
the existence of a browser method or a CPU implementation.

## Stage 1 Complete the Warp evaluation families

- [x] Extract reusable `wp.func` evaluators with thin host preparation and batch
  wrappers. Batch, fused and material consumers call the same arithmetic.
- [ ] Finish 2D polygon/triangle and analytic strip/local-plane paths, plus
  full-contract disk/box special forms, disk lookup, disk series and curvature
  variants. Include analytic and sampled-SDF jet inputs with separate derivative
  and reconstruction contracts. Retain their
  approximation labels and validity domains; do not port scalar-only research
  prototypes as apparently complete methods.
  - [x] Exact rectangle adapter to the native polygon evaluator, including
    Hessians and degree-nine Cartesian moments/derivatives, checked in fp64 and
    paired fp32. The older box lookup is a separate, uncertified method.
  - [x] Explicit native circle quadrature with every 15-channel baseline output,
    independent volume-shell fixtures and finite CPU/CUDA checks in all three
    arithmetic modes. Full disk lookup/series remain open.
  - [x] Explicit interior-volume disk quadrature supplies every degree-nine /
    second-kernel-derivative request in native fp32/fp64, including ordinary
    additive disk/cylinder cross sections and unique periodic fibre images.
    Independent whole-circle flux fixtures, complete polynomial/derivative
    checks, CPU/CUDA graphs, actual end-to-end costs and bounded streaming
    memory pass. Curved FEM bases/shape adjoints are not implied.
  - [x] Infinite plane and finite-thickness strip/band adapters in 2D and 3D,
    using exact support clipping and the shared analytic flux functions. All
    Cartesian monomials through degree nine and kernel derivatives through order
    two are exposed; no subtraction of nearly equal half-space values is used.
    CPU/CUDA fp64/paired checks and deferred raw-fp32 failures are recorded below.
  - [x] Explicit conditioned **numerical** boundary integration for 2D polygons,
    rectangles and support-clipped planes/bands. Combined cumulative profiles
    are prepared exactly before Chebyshev conversion and ordinary fp32 storage.
    Full degree-nine/kernel-derivative-two fixtures and P0–P3 material/transpose
    checks pass on CPU/CUDA. This does not certify the analytic raw recurrence.
  - [x] Explicit spherical-shell quadrature for 3D infinite planes/bands, with
    every degree-nine moment and second kernel derivative. CPU/CUDA references
    pass a tighter 1e-14 fp64 budget, with unchanged 1e-5 raw-fp32 acceptance.
    Matched GPU costs and exactness limits are recorded in the latest checkpoint.
  - [x] Explicit convergence-checked tetrahedral boundary quadrature, through
    degree nine and kernel derivative order two. CPU/CUDA fp64/raw-fp32 moments,
    P0–P3 fields/transpose, real P3 material work and graph replay pass. It uses
    exact combined radial primitives and numerical face-angle integration;
    no planar polynomial-exactness claim is inherited.
- [ ] Retain disk/cylinder cross sections as required solid flux-boundary
  representations. Compare disk reference evaluation, lookup, polygon geometry,
  local curvature and small-disk series across their radius/support domains for
  all advertised quantities. Keep radius/support limitations and geometric,
  integration and truncation errors separate; do not force a special form
  outside its domain to complete a solid-obstacle case.
- [x] Port the 3D analytic triangle-face/edge machinery, tetrahedra, closed
  surfaces and tetrahedral meshes. Complete the advertised surface contract,
  including ambient normal derivatives, not just in-plane gradients. Native
  fp64 and paired-fp32 CPU/CUDA fixture and composed-consumer gates pass.
- [ ] Port full-contract sphere/plane, retained curvature and small-ball variants
  needed by the general comparison suite. These are separate models and do not
  replace analytic mesh evaluation.
  - [x] Native 3D exact plane/sphere **28-channel baseline and subsets**, finite
    CPU/CUDA fp64/raw-fp32/paired-fp32 acceptance, composable functions and reusable batches.
    Higher weighted requests remain withheld in that analytic API.
    The separate support-clipped planar provider supplies weighted planes/bands.
  - [x] Explicit sphere-shell quadrature supplies the full degree-nine /
    second-kernel-derivative request mask, independent 70-digit cap fixtures,
    native fp32/fp64 CPU/CUDA, composed consumers and frozen CUDA graphs.
    The analytic baseline remains separate; curved FEM/shape adjoints and
    sphere-array search are not advertised by the new provider.
- [x] Implement P0/P1 weighted operations first; add P2/P3 and the mixed-product
  degrees from the coverage matrix before advertising higher-order coupling.
- [x] Provide fp32 and fp64 routes. Port compensated arithmetic only where its
  measured accuracy/cost warrants it; test Warp compiler behaviour independently
  of WGSL. Flag ill-conditioned configurations and specify any precision fallback.
  Actual CUDA tests pass the accepted routes; analytic raw-fp32 repair is deferred
  by the user. Method/precision selection is explicit, with no automatic fallback.

**Checkpoint scope:** checked implementation items apply to the new native
polygon/tetrahedron/closed-surface/open-triangle family. CPU/CUDA fp64 and paired
evidence is recorded below; analytic raw-fp32 conditioning is deferred. Existing special forms
are retained, not newly certified by these checks.

**Exit:** every advertised operation passes direct and composed-consumer checks
on Warp CPU and CUDA separately. Analytic routes retain no numerical integration
rule; difficult cases have an explicit tested status, not hidden clamps.

## Stage 2 Local support and efficient assembly

- [x] Keep the all-boundary regular-flux evaluation as the reference.
- [x] Derive compact-support value and weighted-moment corrections in 2D/3D,
  including containment, winding/solid angle, and face/edge/vertex limits.
  Derive each moment correction; a scalar inside flag alone is not a recipe
  for every weighted term. Far-face pruning of regular flux is invalid.
- [x] Build element/face acceleration structures and query-support adjacency.
  Test against exhaustive enumeration, including full containment with no
  boundary intersections. Handle periodic images without duplicates.
- [x] Use element-local support tests for heterogeneous volume fields. Internal
  face cancellation is valid only for matching integrands/traces, not arbitrary
  discontinuous material coefficients.
- [x] Separate static geometry/profile preparation, adjacency and evaluation.
  Reuse adjacency across compatible requests, invalidate it on motion, shifting,
  support changes or material/geometry revisions, and define graph-capture rules.
  - [x] Explicit static `MaterialEvaluator` snapshots reuse field plans across
    changing queries without caching completed integrals; source mutation and
    independent batch checks pass on CPU/CUDA.
  - [x] Explicit fp64 device geometry and selected-dtype node/profile generation
    for tetrahedral quadrature, with integer count/scan/fill, convergence/resource
    checks and measured full/reused-static preparation costs.
  - [x] Optional reusable CUDA whole-element BVH traversal, count/scan, periodic
    enumeration and deterministic sorting, with explicit frozen host CSR readback.
    CPU/CUDA random/brute-force, containment/cavity, trace and consumer checks pass.
  - [x] Shared translated monomial integrals and same-dtype FMA correction
    workspace, reused across field outputs and the transpose within each action.
- [ ] Compare per-query and per-interaction execution, shared primitives,
  requested-output specialization, fixed-order reductions and fused contractions.
  Avoid per-query host dispatch and materializing every moment when a consumer
  only needs a contracted field.

**Checkpoint scope:** the host whole-element AABB tree remains the default;
`search="device"` selects the reusable Warp CPU/CUDA BVH. It returns frozen host
CSR data to the present batch interface, so readback/reupload remains explicit.
Angle enumeration still visits every facet of a selected closed skin; there is
no hierarchical solid-angle search yet. Device-side tetrahedral panel generation
is a separate accepted finite route and retains host input snapshots,
allocation/convergence decisions and transfers. Explicit bounded host streaming is accepted below. It prepares and reads back
each chunk; a GPU-only query/assembly pipeline remains separate work.

**Exit:** localized and exhaustive results agree within the numerical budget;
cost tracks nearby elements rather than total mesh size apart from explicitly
measured containment/search work. No query silently loses a flux contribution.

## Stage 3 Material operators independent of a fluid solver

- [x] Add a material-field provider separate from the impermeable boundary
  provider. Return weighted resistance, skeleton velocity products, bulk-region
  occupancy, required derivatives and provenance. Bulk occupancy is not porosity.
  Solid-fraction/porosity and pressure integration remain Stage 5, following the
  user's explicit scope boundary.
- [x] Implement matrix-free interpolation, gradient/divergence contractions,
  transpose load transfer and local implicit resistance actions with supplied
  particle volumes/masses. Do not embed a pressure iteration or timestep loop.
- [x] Derive the discrete coupling from a common work/dissipation statement.
  Check partition of unity, equal-and-opposite forces, torque where applicable,
  weighted adjoint identities and nonnegative dissipation. A fixed skeleton
  receives a reported reaction load even when it is not dynamically advanced.
- [x] Exercise the local response `(I + dt D_i)^(-1)` with scalar and tensor
  resistance. Expose the response needed by a future pressure operator; do not
  hard-code its use in a currently changing warpSPH solver.
The earlier density/porosity-identity derivation item is retained under Stage 5
below, following the user's instruction that pressure and porosity integration
belong to that handoff. The current supplied boundary-compression diagnostic is
not such a formulation.

**Checkpoint scope:** `AffineResistanceMaterial` supplies the fluid-gradient
load and requires the same supplied reconstruction's transpose. Common work,
force, torque and frame covariance are checked for this standalone contract.
The original point-only model remains noncentral. Porosity/pressure integration
and the actual solver reconstruction are deferred to Stage 5 by user request.

**Exit:** a standalone install can evaluate all agreed coupling terms and apply
their algebraic actions without importing warpSPH. Resistance, pore-volume
terms and permeability-to-resistance approximations remain distinguishable.

## Visual term explorer: mixed impermeable and permeable wall

Added 22 October 2026. Build this alongside the independent operator validation:
first a **2D section**, then its **3D counterpart**. Extend the existing webpage
laboratory where practical; retain the native Warp/reference comparison route.
The explorer is an operator diagnostic before Stage 5, and can gain a solver
comparison view after the maintained incompressible interfaces settle.

### Geometry and controls

- Generate a finite-thickness wall/insert with adjacent impermeable and permeable
  portions. Let the user move their interface, vary the permeable fraction,
  thickness, shape, position and orientation, and inspect queries on either side.
  In 2D include a straight wall, disk/cylinder cross sections and resolved fibre
  sections. In 3D begin with planar/tetrahedral inserts, then compatible fibre
  geometries. Label transverse 2D flow separately from longitudinal flow.
- Configure element count, mesh subdivision and grading independently of the
  geometric dimensions. Offer triangles/polygons in 2D and tetrahedra/closed
  skins in 3D, with a surface-only view whose units are explicitly different.
  Discontinuous material traces stay element-local at the interface.
- Configure support radius, query spacing, integration kernel (Wendland C2/C4/C6
  and cubic spline), field degree, reference/native backend and precision. Draw
  the actual support footprint and selected contributing elements at each probe.
- Configure prescribed fluid/skeleton velocities and resistance tensor magnitude,
  anisotropy and orientation. If permeability is entered, show viscosity, units
  and the explicit `D=nu K^-1` conversion/model; retain a resistance-field input
  for exact polynomial contractions. Porosity controls can initially show
  candidate occupancy terms, with pressure integration reserved for Stage 5.

### Linked views and physical interpretation

- Show geometry/material labels, support overlaps, field maps, query probes and
  cross-interface line plots. Decompose each selected quantity by impermeable
  wall, porous element and free-fluid contribution. Keep the solid boundary
  provider independently selectable with porous coupling disabled.
- Show kernel occupancy/density contributions, first/second moments, gradients,
  Hessian/Laplacian and FEM-weighted terms. For compression, show the relevant
  relative-velocity/continuity contractions and the signs of their contributions;
  label candidate porous-density terms against Böttcher et al. (2025), v2.
- For permeation, show weighted resistance, skeleton-velocity products, drag,
  reaction, dissipation and local implicit mobility. Display normal relative
  velocity and an explicitly prescribed or solver-computed normal flux. Kernel
  integrals and local damping alone do not determine pressure-driven permeation.
  Until Stage 5, any driven local response is a supplied-state diagnostic.
- Compare exhaustive versus localized evaluation and mesh refinement at fixed
  geometry, support and fields. Separate geometry error, integration error,
  precision error and model approximation; show unavailable quantities explicitly.
  Include finite-face/edge limits and support entirely contained in a region.
- Keep resolved impermeable fibres and an effective porous replacement as separate
  scenarios. Do not add effective drag for the same resolved fibres.

### Delivery gates

- [x] 2D mixed-wall generator and controls, support/probe overlay and linked maps
  and line plots for the accepted native/reference terms.
- [x] Reproducible 2D presets: fully impermeable, fully permeable, mixed interface,
  zero resistance, rotated anisotropy, coarse/refined elements and a fibre section.
- [x] Cross-check browser displays and per-region sums against saved native Warp
  and arbitrary-precision fixtures; browser agreement is not native acceptance.
- [x] 3D version with tetrahedral insert, slices and probes, preserving volume
  versus surface units and testing interface/containment cases.
- [ ] Stage 5 overlay of actual compression residuals, pressure response and
  permeation flux from the maintained solver, after the solid regression gates.

**Exit:** the same prescribed physical fields produce converged maps under mesh
refinement, per-region contributions sum to the reported total, support/kernel
changes are visible, and diagnostic local response is clearly distinguished from
an incompressible flow solution.

## Stage 4 Accuracy and performance acceptance

| Check family | Required evidence |
|---|---|
| Independent values | Arbitrary precision, exact simplex polynomial moments, planar/whole-support limits, and independent refined quadrature where appropriate |
| Geometry | Tangency, exact face/edge/vertex queries, slivers, large translations, scale/rotation, reversed winding, cavities, subdivision and internal-face cancellation |
| Fields | Polynomial reproduction through each advertised degree, discontinuous coefficients, constant-field reductions, tensor rotations and admissibility |
| Operators | Kernel versus complete query derivatives, Hessian trace, finite differences away from branch changes, adjoint/work identities, momentum and torque bookkeeping |
| Locality | Exhaustive versus localized queries, inside/outside/boundary limits, empty/full support, variable supports and periodic images |
| Runtime | CPU/CUDA, fp32/fp64, empty batches, sparse output masks, repeated launches, graph replay, topology changes and unsupported-input reporting |
| Constitutive algebra | Scalar/tensor relaxation, driven terminal response, positive dissipation, zero resistance and stiff limits |

Use normalized mixed absolute/relative errors; relative error alone is
meaningless near zero. Initial targets for well-conditioned baseline moments
are `1e-10` in fp64 and `1e-5` in fp32, subject to explicit family/degree budgets
registered at Stage 0. These are engineering targets, not certified global
bounds. Adversarial cases must either pass their declared budget or trigger a
documented validity/precision policy. Report fallback frequency as well as error.

Benchmark cold preparation/compilation separately from warm kernels, adjacency,
readback, complete evaluation and transpose actions. Report GPU model, precision,
requested degree/outputs, query/interaction counts and memory. Measure small
batches and scaling, graph replay and fused consumers; compare with the existing
Warp paths and a stated quadrature baseline at comparable measured accuracy.
Do not substitute software Vulkan/browser timings for NVIDIA CUDA throughput.
Serialize GPU measurements on a shared machine and store reports through the
repository's [external evidence convention](validation.md).

**Exit:** published coverage and accuracy tables, hardware benchmark evidence,
known failure cases and reproducible commands. An efficient operator is accepted
on measured end-to-end cost and accuracy, not kernel time alone.

## Stage 5 Handoff after incompressible solver work settles

- [ ] Translate the selected density/continuity identities from Böttcher et al.
  (2025), v2, into candidate mesh-weighted solid-fraction/porosity terms. Derive
  and test their Jacobian and transpose with supplied velocities before entering
  a maintained solver. Replacing sums by integrals does not itself establish the
  full pressure formulation. This item is explicitly deferred from Stage 3 by
  the user's pressure/porosity scope instruction.

Prepare an integration note naming the tested library revision, compatible Core
version, fixtures, supported degrees, performance results, and unresolved cases.
Agree an adapter contract against the then-current warpSPH interfaces. This
stage may require revising the candidate algebra from Stage 3.

- [ ] Choose one maintained incompressible route for the first integration.
- [ ] Pass the ordinary solid flux-boundary cases first: planar channel,
  confined cylinder/disk, periodic cylinder array and a moving/rotating disk
  where the solver supports it. Include pressure, viscosity, continuity,
  shifting/near-wall handling and force/torque reactions. These cases must work
  with porous coupling disabled and must not wait for pore-volume accounting.
  Reuse the maintained `stokesArray` and `periodicChannel` cases as seeds after
  the interfaces settle; existing results do not validate the new operators.
- [ ] Include implicit resistance mobility in the pressure operator, final
  velocity correction, diagonal/preconditioner and wall/source terms as required
  by that route. Re-derive timestep factors and pressure units consistently.
- [ ] Retain an explicit/split baseline for convergence comparisons; do not
  equate stable local damping with an accurate pressure-coupled scheme.
- [ ] After the solid gates, run fixed, saturated resistance cases. Then add pore-volume-aware
  density and continuity using the derived model and the 2025 paper as a
  comparison. Account for material interfaces and shifting consistently.
- [ ] Run the maintained incompressible regression suite before extending to
  other schemes. Put all simulation cases and timestep orchestration in warpSPH.

**Exit:** stable, converged pressure/flux results and solver regressions, not
merely a plausible animation. This is the first stage that modifies warpSPH.

## Application validation sequence

1. Solid flux boundaries: planar channel, confined disk/cylinder and periodic
   arrays, including wall loads and mesh/special-form comparisons. Sweep radius
   relative to support and particle spacing, inter-fibre gap resolution and
   periodic images. Use confined or periodic references for 2D creeping flow;
   do not use an unbounded single-cylinder Stokes drag as a finite reference.
2. Uniform stationary material: exact scalar and tensor velocity relaxation,
   driven terminal response, drag reaction and dissipation.
3. Fully periodic uniform resistance: isolate pressure/drag coupling and sweep
   timestep relative to the resistance relaxation time.
4. Channel with a porous insert: pressure drop, flux, interface behaviour and
   convergence against an independent solution of the same continuum model.
5. Layered and rotated anisotropic material: series/parallel transport limits
   where applicable, continuous normal flux and discontinuous coefficients.
6. Manufactured polynomial material fields: preserve the same field under
   retriangulation/subdivision; compare analytic, quadrature and point sampling
   at matched fluid resolution, separating model and discretization errors.
7. Resolved-to-effective fibre comparison: obtain transverse permeability from
   periodic flux-boundary disk arrays using independent driving directions,
   measure off-diagonal response and test tensor rotation. Add axial cell flow
   or 3D fibres for longitudinal permeability. Preserve the distinction between
   superficial cell-average and intrinsic fluid-average velocity, and between
   a mean pressure gradient and a body force applied only to fluid. Convert the
   existing case's dimensionless drag coefficient to dimensional permeability
   with those conventions; its variable named `K` is not itself permeability.
   Repeat across packing fraction and anisotropic arrangements, then compare
   the coarse material's pressure drop and flux against the resolved reference.
8. Repeat the accepted 2D sequence in 3D with tetrahedra. Compare different
   volume decompositions and compatible closed-surface representations only
   where the field assumptions permit boundary-only evaluation.
   Three-dimensional cylinder/fibre geometry needs its own integration model
   or a converged mesh representation; the 2D disk evaluator is not a 3D
   cylinder integral. Keep any cylinder/tube approximation explicit.

Use a small independent continuum reference before interpreting material
parameters physically. Permeability, porosity, effective viscosity and velocity
conventions must match that reference. Saturation fronts, capillary rise,
deformable sponges, nonlinear Forchheimer drag and evolving permeability are
later extensions; each adds operators and validation beyond the fixed saturated
material contract.

## Resume sequence

The checkpoints below implement portions of Stages 0–4. Preserve the inventory,
fixtures, raw/exhaustive evaluator and existing solid Scene applications.

The explicit `float32_compensated` route passes the current finite CPU gate;
raw `float32` remains experimental in the analytic mesh/box/clipped-planar recurrences.
The opt-in 2D `method="conditioned_boundary"` numerical route passes its finite
raw-fp32 gates, including degree-nine moments and material work checks.
The factored radial plane/sphere baseline now passes the raw-fp32 CPU gate. The new
conditioned circle-quadrature route passes its finite raw-fp32 baseline gate.
CPU fp64 remains the current API default; this is not a GPU performance recommendation. CUDA
compilation has been audited. Actual CUDA execution now passes the current
regression gates; see the latest checkpoint for accuracy and measured costs.
The affine material action has common-work, force/torque and frame checks when
its supplied gradient and matching transpose are used. Do not integrate it into
a solver with a different reconstruction or drop the gradient load.

The 2D disk/box and 3D plane/sphere baselines are now implemented with explicit
method labels. Support-clipped planes and bands add the degree-nine weighted
contract in both dimensions. Higher weighted circle/sphere products remain
withheld; old lookup/series methods are not certified by the new evaluators.
Continue the independent library Stages 0–4, as explicitly requested in the
latest steering. The user now accepts the analytic-fp64/numerical-fp32 tradeoff
and defers the analytic raw-fp32 basis problem. **The existing operators now
pass actual CUDA correctness gates**, including independent fixtures, fused
material/transpose actions, localization/periodicity, repeated launches and
graph replay. Serialized GPU measurements expose costly high-degree small
batches. The new 3D shell quadrature now supplies a much faster plane/band route
at near-roundoff fp64 accuracy. A separately validated tetrahedral quadrature
now accelerates the real degree-nine material workload, with explicit checked
preparation cost and higher storage. Later shared-radius caches, device panel
generation and static material reuse now have measured batch-scaling evidence.
Next address CUDA-native adjacency, larger streamed batches and the remaining
representation/web gates. Keep the existing raw analytic failure
record; do not lower error budgets or hide paired arithmetic.
Direct weighted planar primitives should be assessed for fp32 conditioning
and eventual GPU cost, rather than chosen from CPU fp64 timings.
Weighted-circle fixtures/ports follow before accepting P1–P3 disk material fields.
The mixed-wall generator, term export and initial 2D/3D webpage now run against
native Warp moments; finish their delivery gaps alongside library acceptance.
Host search dominates complete circle-array CPU costs;
CUDA-native search, general streaming and isolated GPU-throughput qualification remain open.
Pressure/porosity and the actual SPH reconstruction adapter remain Stage 5.

## Native baseline checkpoint — 22 October 2026

See [coverage and versioned contract](fluxbridge-coverage.md) and
[operator derivation, API and limitations](fluxbridge-operators.md).

Implemented source is `src/warpSPHBoundaries/fluxbridge/`. No main warpSPH
checkout or shared Core/Integrator source was edited; no solver was copied.
The pre-existing dirty source/browser/research files were preserved, with an
initial content snapshot at `/tmp/fluxbridge-kickoff/dirty-tree.tar.gz` and its
SHA256 manifest. Only this plan was intentionally extended among the original
97 dirty files; the other originals remain byte-for-byte unchanged.

### Implemented library work

- A finite request/array contract, coverage inventory, 70-digit reference fixture
  generator and 52 fixtures containing **11,172 channels**. Complete volume
  monomials through degree nine and kernel derivatives through order two are
  checked at an interior query, with additional baseline face/edge/vertex,
  exterior and open-surface cases. This exceeds the browser selection count.
- Shared analytic `wp.func` reductions for 2D polygons and 3D tetrahedra, closed
  triangular skins and open triangles, with explicit fp32/fp64 factories.
  These supply interpolation/occupancy, moments, kernel gradients, divergence,
  curl, Hessians/Laplacians, P0–P3 FEM and polynomial field products through
  degree nine. Open surfaces include ambient normal Hessians and derivatives.
- Retained exhaustive regular flux; compact angle/solid-angle scalar corrections
  propagated through **every** weighted-moment recurrence. Native checks include
  full support with no facet hits, cavity orientation, exact face/edge/vertex
  limits and tangency. Host element AABB search, unique periodic images,
  adjacency revision checks and deterministic forward/reverse reductions are
  separate from evaluation. Angle enumeration on a selected skin is still global.
- Fused polynomial contractions, weighted transpose transfer, PSD Bernstein
  resistance, skeleton products, native deterministic material reactions and
  reusable local implicit mobility. Cubic resistance/velocity/test work and
  dissipation exercise degree-nine products. The work form conserves total
  force but is noncentral: torque/objectivity is an explicit unresolved gate.
- A 2D-first, then 3D mixed impermeable/permeable wall visualization milestone,
  described above. It distinguishes compression/occupancy diagnostics, local
  resistance response and later solver-computed permeation flux.

### Evidence and acceptance limits

Evidence is written outside the source tree using the explicit sandbox-safe
output override `/tmp/fluxbridge-evidence/`. `report.json` records source SHA256
hashes, actual device/precision, raw accuracy failures, timings and allocations;
`pytest.log` records **143 passed, 17 skipped, 4 expected fp32 failures** in
67.27 seconds for the combined native and regression command. The skips are
CUDA graph and body-kinematics checks without hardware. Temporary
storage should be copied to the development archive for long-term retention.

The CPU is **Intel Core Ultra 9 285K**, Warp **1.17.0**. CUDA initialization
reports no CUDA-capable device; CUDA correctness, throughput and graph replay
are **unverified**, not passed. In the complete finite fixture sweep, worst
mixed fp64 error is about **5.1e-12**, within `1e-10`. Raw fp32 reaches about
**3.15e-3**, failing `1e-5`: all four kernel families fail some higher weighted
derivatives, and W4/W6 also fail baseline cases. Four expected test failures
retain those failed acceptance gates; they are not accuracy approvals. No
silent fp64 fallback or quadrature substitution is used. Tiny/sliver cases
outside documented conditioning thresholds are explicitly rejected.

Representative CPU fp64 timing for 128 queries and 64 separated elements
(128 local interactions versus 8192 exhaustive interactions):

| Measurement | 2D | 3D |
|---|---:|---:|
| Warm baseline evaluation + deterministic reduction | about 0.74 ms | about 5.54 ms |
| Complete local evaluation, including search/allocation/upload | about 5.2 ms | about 10.1 ms |
| Exhaustive warm evaluation | about 5.3 ms | about 13.4 ms |
| Fused P1 interpolation (smaller output contract) | about 0.33 ms | about 1.20 ms |
| Transpose including upload/readback | about 0.16 ms | about 0.24 ms |
| Supplied-array material action including reactions | about 2.42 ms | about 12.49 ms |

These are native CPU microbenchmarks, not CUDA or solver throughput. P1 and
baseline requests differ and must not be treated as equal-work speedups. The
report includes single-query and fp32 timings, search, allocations, first
execution, readback, a matched scalar quadrature baseline and source provenance.
The fresh-cache first operator preparation took about 7.35 ms and its first
compilation/execution about 1317 ms; warm costs exclude that compilation.
Local/exhaustive differences are zero in the separated-element timing cases.

Reproduction (existing environment; no shared install changes):

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge tests/test_package_layering.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest \
  tests/edge/test_disk.py tests/edge/test_warpbc.py tests/scene/test_disk_array.py \
  tests/scene/test_periodic_bodies.py tests/scene/test_body_kinematics.py \
  tests/mesh3d/test_reference.py -q
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_acceptance.py \
  --device cpu --fresh-cache --output /tmp/fluxbridge-evidence
# On an NVIDIA host repeat the acceptance run with --device cuda:0.
```

At this initial checkpoint, Stages 0–4 were **partially complete**. The
continuations below supersede its conditioned-fp32 and affine-work limitations.
The then-remaining work included
conditioned fp32, CUDA validation, native disk/box/plane/sphere/curvature/series
full-contract ports, sampled-SDF derivative acceptance, additional radial
consumer profiles, CUDA-native search, matched existing-path performance
comparisons, and torque/objectivity treatment. Ordinary disk/cylinder and
periodic-fibre Scene APIs remain usable without the material provider. Resolved
transverse fibre flow remains the future effective-permeability reference;
longitudinal response still requires axial velocity or 3D geometry. Böttcher
et al. (2025), arXiv:2504.07739v2 remains the principal porous-flow reference.

## Paired-fp32 and affine-work continuation — 22 October 2026

This checkpoint supersedes the raw-fp32-only limitation of the preceding native
baseline, without removing its retained raw evaluator or historical evidence.

- Added `precision="float32_compensated"`: paired single-precision arithmetic
  covers elementary functions, geometry normalization, face/volume recurrences,
  profile sums, field contractions and transpose transfer. No fp64 device math
  or quadrature fallback is substituted. Rational/float64 host preparation splits
  coefficient high/low parts. Device arrays use paired scratch where needed and
  single-precision results. The mode is explicit because its accuracy and cost
  differ from ordinary 24-bit arithmetic.
- Packed facets in element-local coordinates and queries per interaction for
  this mode, including periodic image shifts. This avoids loss of local distances
  across widely separated elements. Cached profile outputs are shared by fused
  consumers and transpose. Raw/fp64 layouts remain separate; callers use the
  reported scratch width and prepared coordinate arrays.
- Corrected the Warp 1.17 contraction option to `fuse_fp=False`; the earlier
  `enable_fma` spelling was ineffective. Explicit single-precision FMA remains
  essential for multiplication residuals. Legacy Scene files were preserved.
- Added `AffineResistanceMaterial`: supplied `L_i` reconstructs
  `u_i(s)=u_i+L_i(s-x_i)`. The common work returns particle force, nodal reaction
  **and** the conjugate gradient load. `pullback_gradient` applies a supplied
  matching transpose. No arbitrary corrective couple or pressure solver is added.
  Tests cover common rigid rotation, tensor covariance, torque, virtual-work
  directional derivatives, affine-reproducing transpose and cubic degree-nine
  products. Point-only drag remains available with its noncentral limitation.
- Fixed empty compact open-triangle evaluation: face scratch is initialized
  before support rejection, including exact tangency. Poisoned-scratch tests
  cover raw fp32, paired fp32 and fp64 and prevent stale values leaking into
  ambient gradients/Hessians or later launches.
- Reduced field work by omitting exact-zero coefficient terms and exact-zero
  high monomials. No tolerance threshold deletes small physical coefficients.
  This particularly helps constant and zero-skeleton material consumers.

The 2D/3D mixed-wall explorer remains in the plan. Its material view should offer
point-only versus supplied-affine work, show the gradient load and torque
bookkeeping, and never display an unassembled local response as a complete
pressure/permeation solution.

Evidence and final timings for this continuation are in
`/tmp/fluxbridge-compensated-evidence/` (`report.json`, `pytest.log`, and
`cuda-compile/compile-report.json`). The original evidence was not overwritten.
The source tree at continuation start was separately snapshotted under
`/tmp/fluxbridge-continuation/`. Main warpSPH and shared Core/Integrator sources
remain unchanged.

CPU fixtures: paired fp32 has zero failing channels at `1e-5`, worst mixed error
about **4.7e-8** over all 52 fixtures (11,172 channels), in exhaustive and compact
forms. CPU fp64 remains within `1e-10`. Raw fp32 retains four expected test
failures; paired checks are ordinary passing tests, not expected failures.
No fixture is rejected or sent to a fallback. The combined regression run passed
182 tests with 17 CUDA skips and four expected raw-fp32 failures. After the final
empty-surface audit, the focused native/layering run passed **101 tests**, with
one CUDA skip and the same four expected raw failures (`final-native.log`). The
84 existing disk/Warp/periodic/reference regressions passed in the combined run;
no legacy Scene or reference source was changed by the final surface fix.
This is finite-case acceptance,
not a bound for arbitrary slivers or near-singular configurations.

CUDA compiler audit: NVRTC 12.9 accepts the paired evaluator/field kernels for
SM80. PTX has no fp64 instructions; the two fp64 declarations are unused Warp
header constants. This does **not** establish GPU execution, runtime accuracy,
graph replay or throughput; no CUDA device was available.

Continuation CPU timings (128 queries, 64 separated elements, 128 local
interactions; cached compilation, medians in milliseconds):

| Supplied-array action | fp64 2D | fp64 3D | paired fp32 2D | paired fp32 3D |
|---|---:|---:|---:|---:|
| Baseline moments and reduction | 0.69 | 5.35 | 6.44 | 46.37 |
| Point resistance and reaction | 0.42 | 1.24 | 2.73 | 18.62 |
| Supplied-affine force, gradient load and reaction | 1.56 | 7.74 | 5.29 | 37.83 |

Use `report.json` for exact samples and preparation/upload/readback costs. The
supplied-affine timing excludes an application's gradient and transpose actions.
Paired arithmetic remains opt-in: it is slower for these baseline consumers.
For one cubic resistance/velocity/test interaction (degree nine), fp64 versus
paired action times were 2.11/1.69 ms in 2D and 43.10/22.34 ms in 3D, with
mixed action differences below 7.8e-8. This comparison also includes different
profile caching strategies; it does not isolate arithmetic throughput. The
final report's source hashes match the tested implementation. It was run with
the existing cache (`fresh_cache=false`); cold compilation is not included.

Reproduce the added work with the same existing Python environment:

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest \
  tests/fluxbridge tests/test_package_layering.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_compile_cuda.py \
  --output /tmp/fluxbridge-compensated-evidence/cuda-compile
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_acceptance.py \
  --device cpu --fresh-cache --output /tmp/fluxbridge-compensated-evidence
```

The checkpoint below records the native 3D plane/sphere baseline port. Actual
CUDA acceptance, higher weighted special forms and Stage 5 integration remain
separate gates.

## Native 3D plane/sphere baseline — 22 October 2026

Added `fluxbridge.radial.RadialEvaluator` with shared `element`/`select` device
functions and an allocation-free prepared launch. Exact solid half-spaces and
spheres support arbitrary outward axes, physical support scaling and all
**28 baseline moment/gradient/Hessian channels**. A subset request is allowed;
higher combined moment/derivative degree is explicitly rejected. The API does
not advertise degree-nine sphere/plane FEM fields or an approximate curvature
method as a complete exact representation.

Independent 70-digit numerical shell fixtures were established before the port:
112 cases, **3,136 channels**, four kernels, signed inside/outside distances,
centre and near-centre queries, small spheres, large-radius limits and exact
tangencies. Differential and closed-mesh checks additionally caught a cubic
outer-piece linear term omitted by the initial shell/native derivative setup;
both were corrected and fixtures regenerated. All kernel pieces, including
the resulting cubic Laurent derivatives, are retained.

The stable implementation factors the cap in local radial intervals and uses
Bernstein products and endpoint-shifted kernel profiles. This avoids cancellation
of large powers near a concentric sphere and at support tangency. Cubic outer
negative-power primitives use a documented 64-term analytic beta series: the
scalar tail is below 8e-18 before explicit scale/weight factors. This numerical
primitive approximation is separate from any geometric or physical model.

CPU fixture mixed errors: fp64 **4.62e-14**, paired fp32 **1.50e-7**, with zero
failures, rejections or fallbacks in those modes. Raw fp32 W6 reaches
**1.36e-5** (16 failing channels), recorded as two expected family failures.
Other raw radial kernel families pass the finite gate; general raw mesh failures
remain separately recorded. Derivative checks use fixed geometry, including
normal rotation for a moving query around a sphere. Independent mesh/radial
comparisons, centre isotropy, scaling, subsets, poisoned scratch reuse, empty
batches and a custom fused consumer also pass.

Evidence lives in `/tmp/fluxbridge-radial-evidence/`: `report.json`,
`pytest.log`, `combined-pytest.log`, and `cuda-compile/compile-*.json`.
The combined native, disk, periodic-body and reference run passed **218 tests**,
with 18 CUDA skips and six expected raw-fp32 failures, in 67.76 seconds. After
the final support-scaling guard, the focused suite passed **134 tests**, with
two unavailable CUDA graph tests and the same six expected raw-fp32 failures
(8.36 seconds). The separate plane/sphere suite has
33 passing tests, one CUDA skip and two of those expected failures.

CPU medians for **128 queries**, one solid, all 28 outputs (Intel Core Ultra 9
285K, Warp 1.17.0, existing compilation cache):

| Representation/kernel | fp64 warm | fp64 complete | paired fp32 warm | matched mesh fp64 warm |
|---|---:|---:|---:|---:|
| Plane W2 | 0.25 ms | 0.35 ms | 1.77 ms | 1.53 ms |
| Plane cubic | 0.75 ms | 0.87 ms | 6.94 ms | 2.46 ms |
| Sphere W2 | 0.33 ms | 0.44 ms | 2.57 ms | — |
| Sphere cubic | 1.29 ms | 1.41 ms | 12.09 ms | — |

The matched plane uses a closed box whose other faces miss support; both paths
request exactly the same channels. Complete radial costs include host validation,
allocation and upload, but expect signed distances/normals already supplied;
they exclude geometry search or constructing those radial inputs. The report
also records one-query costs, readback, allocations and the retained NumPy
radial/rotation comparison. Paired CPU arithmetic is slower than fp64 and can
be slower than that retained reference, especially for cubic's rational primitive;
these measurements do not imply CUDA throughput. All final report/compiler
source hashes match the implementation. No timing run overlaps the test or
compiler runs.

NVRTC 12.9 compiles all three radial precision modes for SM80; raw and paired
fp32 PTX contain **zero fp64 instructions**. Compilation does not validate GPU
runtime accuracy or performance. The existing CPU-only environment has no CUDA
device; Stage 0–4 acceptance remains incomplete for that and the remaining
representation/profile/search gates. Main warpSPH and shared installations
remain unchanged. Ordinary solid Scene disk/cylinder and fibre APIs are retained.

Reproduce this checkpoint (same Python environment as above):

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_radial_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge tests/test_package_layering.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache WARPSPH_DEVELOPMENT_ROOT=/tmp/fluxbridge-radial-evidence \
  $PY -m pytest tests/fluxbridge tests/test_package_layering.py \
  tests/edge/test_disk.py tests/edge/test_warpbc.py tests/scene/test_disk_array.py \
  tests/scene/test_periodic_bodies.py tests/scene/test_body_kinematics.py \
  tests/mesh3d/test_reference.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --device cpu --output /tmp/fluxbridge-radial-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --compile-only float32_compensated --output /tmp/fluxbridge-radial-evidence/cuda-compile
# Repeat --compile-only with float32 and float64 for the other compiler audits.
# On hardware, repeat runtime acceptance with --device cuda:0.
```

The following checkpoint records the 2D disk/box port. Higher weighted special
forms, lookup/series methods and CUDA runtime acceptance remain separate work.

## 2D disk/box and periodic-array checkpoint — 22 October 2026

The dirty tree at the start of this continuation (128 files) was archived under
`/tmp/fluxbridge-diskbox-kickoff/`, with a SHA256 manifest and tracked diff.
No legacy Scene, browser or research files were edited. Among the original 97
dirty files, only this plan has changed; all 96 unrelated originals remain
byte-for-byte preserved. No main warpSPH or shared Core/Integrator source was
changed, and no solver was imported or copied.

Implemented in `fluxbridge/box.py`, `disk.py`, `_disk_code.py` and `disks.py`:

- `BoxEvaluator` constructs exact rotated/additive rectangles and reuses the
  native analytic polygon plan, device functions, adjacency and contractions.
  All 15 baseline channels and requested Cartesian moments/kernel derivatives
  through degree nine are retained. This is a geometry adapter, not a new closed
  form or validation of the old box lookup. Quad FEM bases remain unsupported.
- `DiskQuadratureEvaluator` explicitly integrates an exact circle numerically.
  It supplies nine axial invariants, hence all 15 Cartesian baseline channels,
  including second moments, Hessians and their derived contractions. The
  default is 24 Gauss nodes per panel, selectable from 8 to 64; knot splitting
  and near-point refinement are explicit. Higher combined requests are rejected.
- Second moments use a second regular radial potential; Hessians use a direct
  circle identity involving W'/r. Exact rational preparation precedes conditioned
  Chebyshev evaluation. The **entire exterior regular-flux tail** is retained
  as an analytic angle correction while the supported arc uses quadrature.
  Exact centre, full support containment and disjoint/tangent limits are handled
  separately. No table, polygon or truncated geometric model is substituted.
- `DiskArrayEvaluator` makes the full baseline usable for standalone solid
  disk/cylinder sections and periodic fibre arrays. Host AABB search and unique
  periodic images prepare explicit interactions; native launches reuse buffers
  and reduce in deterministic order without a fixed-capacity truncation. Bounds
  round outward to prevent small radii disappearing at large world coordinates.
  Overlaps add rather than forming a union. No porous resistance is attached to
  these resolved solids; ordinary 2D response remains transverse.
- CUDA's generic raw-fp32 sine/cosine routine emitted fp64 slow-path instructions
  despite this problem's bounded angles. A bounded half-angle polynomial removes
  that irrelevant path. Paired fp32 uses its explicit higher-precision polynomial.
  All disk and reduction precision modes compile for SM80; neither fp32 PTX has
  fp64 instructions. GPU runtime correctness and speed remain untested.

Independent references were established before the ports. The new 70-digit
fixture set has **128 cases / 3,180 channels**: 100 disk cases / 1,500 channels
and 28 rectangular cases / 1,680 channels, including complete degree-nine
moment/derivative requests. Disk fixtures use volume-shell integration rather
than the new boundary algorithm; their five common channels also match the
retained 48-node circle reference within 4.9e-14. Tests additionally check fixed
geometry derivatives, moving-query normal rotation, 24/48-node convergence over
random radii, periodic/exhaustive sums, translations, support containment,
poisoned scratch reuse, exact tangency, empty batches and a custom fused Warp call.

Worst mixed fixture errors (`abs(error)/(1+abs(reference))`, unit support):

| Family | fp64 | raw fp32 | paired fp32 |
|---|---:|---:|---:|
| Circle quadrature, all 15 channels | 2.77e-15 | 3.95e-7 | 1.49e-7 |
| Exact rectangles, baseline and degree nine | 8.47e-12 | 3.79e-3 (fails) | 1.49e-7 |

There are no rejected fixtures or hidden fallbacks. The circle method passes
both precision budgets; raw rectangular recurrences have 463 failing channels,
retained as four additional expected family test failures. A mixed absolute
budget does not establish relative accuracy or exact floating-point positivity
for vanishing intersections. There is no universal quadrature error bound.

The combined native/solid/reference run passed **253 tests**, with 19 CUDA skips
and ten expected raw-recurrence failures, in 75.65 seconds. After the final
bounded-trig change, the focused native/layering suite passed **169 tests**, with
three CUDA skips and the same ten expected failures, in 11.17 seconds. The
disk/box suite has 35 passes independently, with one CUDA skip and four expected
raw-box failures. Legacy solid APIs remain unchanged and usable without porous
coupling.

Evidence: `/tmp/fluxbridge-diskbox-evidence/report.json`, `pytest.log`,
`final-native.log` and `cuda-compile/compile-*.json`. Source hashes in all final
benchmark/compiler reports match the implementation. The original evidence sets
were preserved. CPU: Intel Core Ultra 9 285K, Warp 1.17.0, existing cache;
timings were serialized after compiler/test work.

For 128 queries, 64 separated elements, 128 selected versus 8,192 exhaustive
interactions, and all 15 baseline outputs (W2, CPU medians):

| Supplied geometry/queries | fp64 warm evaluation/reduction | fp64 complete | paired fp32 warm |
|---|---:|---:|---:|
| Solid disks, circle quadrature | 0.32 ms | 4.32 ms | 6.33 ms |
| Exact rectangles | 0.85 ms | 5.09 ms | 4.97 ms |

The geometries differ, so these rows are costs rather than an equal-operator
speedup. Complete costs include host search, local geometry construction,
allocation and upload. Circle localization matches exhaustive evaluation exactly
in this benchmark; rectangular differences are below 3e-17 in fp64. Host search
dominates the circle's complete cost. Raw-fp32 circle warm cost is 0.42 ms;
compensated arithmetic remains opt-in. Given precomputed radial inputs, 128-query
fp64 circle W2/cubic costs are 0.25/0.27 ms warm and 0.39/0.41 ms complete.
Single-query, 48-node, allocation, readback and full-array samples are in the report.

A separate one-query polygon sequence with 32/128/512 vertices gives mixed
geometry differences of 3.16e-3 / 1.97e-4 / 1.23e-5 relative to the circle.
These are geometry convergence results, not matched-accuracy speed comparisons.
The older five-invariant disk tables still omit the new second-moment/Hessian
contract and cannot be used as a full-contract speed baseline.

Reproduce with the existing Python environment:

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_diskbox_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge tests/test_package_layering.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache WARPSPH_DEVELOPMENT_ROOT=/tmp/fluxbridge-diskbox-evidence \
  $PY -m pytest tests/fluxbridge tests/test_package_layering.py \
  tests/edge/test_disk.py tests/edge/test_warpbc.py tests/scene/test_disk_array.py \
  tests/scene/test_periodic_bodies.py tests/scene/test_body_kinematics.py \
  tests/mesh3d/test_reference.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_diskbox_acceptance.py \
  --device cpu --output /tmp/fluxbridge-diskbox-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_diskbox_acceptance.py \
  --compile-only float32 --output /tmp/fluxbridge-diskbox-evidence/cuda-compile
# Repeat compiler audit for float64 and float32_compensated.
# On an NVIDIA host, repeat runtime acceptance with --device cuda:0.
```

The later steering keeps the next executable step within **Stages 0–4**; see
the planar checkpoint below and the updated Resume sequence. The mixed-wall
generator and 2D/3D webpage remain planned diagnostic work. Pressure, porosity
and the actual SPH solver/reconstruction remain Stage 5, with Böttcher et al.
(2025), arXiv:2504.07739v2 retained as the principal porous-flow reference.

## Weighted plane/band checkpoint — 22 October 2026

The continuation started from 137 dirty files, archived with a SHA256 manifest
and tracked diff in `/tmp/fluxbridge-planar-kickoff/`. Of those files, only the
three FluxBridge coverage/operator/plan documents changed. All 96 unrelated
files from the original kickoff remain byte-for-byte preserved; main warpSPH,
shared packages, legacy Scene and browser sources were not edited.

Implemented `fluxbridge/planar.py`, an exact support-clipping adapter for infinite
half-spaces and finite-thickness infinite bands in **both 2D and 3D**. It exposes
all Cartesian monomials through degree nine with kernel-only derivatives through
order two, for W2/W4/W6/cubic. In particular, the older radial provider's
combined-degree-two restriction no longer limits the separate **planar** route.
The sphere/circle contracts are unchanged. This is a reusable analytic geometry
adapter, not a new axial closed form, lookup, sampled-SDF model or quadrature.

For normalized support, tangent extents ±2 and normal bounds clipped to ±2
contain the entire unit ball. The clipped rectangle/box and infinite solid
therefore agree wherever the kernel or its derivatives are nonzero, for every
polynomial weight. The shared flux recurrence retains all required exterior
angle corrections. Bands are integrated directly, without subtracting nearly
equal half-space results. Device functions generate per-query geometry into
caller-owned scratch; launches reuse allocations and perform no host query loop.
Empty/disjoint rows clear both recurrence and paired output caches.

Testing found and fixed a precision-specific cap triangulation issue. Exact
query-on-plane caps now contain an exact origin vertex; nonzero-offset caps use
an eccentric interior fan so radial primitives do not see nearly singular
interior edges. Both triangulations represent the same cap. No input offset is
clamped to zero, no precision fallback is hidden, and no approximate shape is
substituted. Tests cover zero and offsets down to ±1e-18, including support
rescaling that places a band face within rounding distance of the query.

Independent references preceded the port: **136 cases / 22,820 channels** at
70 digits in `tests/fixtures/fluxbridge-planar-v1.json`, generated by
`scripts/make_fluxbridge_planar_fixtures.py`. The 3D axis-aligned high-degree
oracle (`planar_reference.py`) integrates exact rational angular polynomials
and radial powers/logarithms, independently of mesh flux. Its plane baseline
also agrees with the retained adaptive shell oracle to better than 1e-43.
Rotated baseline fixtures use the retained arbitrary-precision mesh reference;
additional rotated degree-nine requests are checked independently at runtime.

Worst mixed unit-support error `abs(value-reference)/(1+abs(reference))`:

| Planar family | fp64 | raw fp32 | paired fp32 |
|---|---:|---:|---:|
| 2D plane/band, 3,540 channels | 5.26e-12 | 2.56e-3 (413 failures) | 2.79e-7 |
| 3D plane/band, 19,280 channels | 8.29e-12 | 2.30e-3 (1,549 failures) | 1.62e-7 |

No fixture was rejected and no fallback was used. fp64 passes 1e-10 and paired
fp32 passes 1e-5. Raw fp32 misses each dimension/kernel family gate, retained as
eight additional expected failures. Finite mixed-error acceptance does not
certify relative error for tiny values, exact positivity or a universal bound.
Preparation explicitly rejects band half-width/support below 1e-4 (fp64) or
1e-2 (either fp32 mode), invalid normals, and unrepresentable support scaling.

The new suite passed **35 tests**, with one CUDA skip and eight expected raw
failures. The combined native/layering/solid/reference regression passed
**288 tests**, with 20 CUDA skips and 18 expected raw failures, in 96.73 seconds.
Checks include complete moving-query derivatives, direct-band versus half-space
identities, support units, masks/repeated requests, containment/tangency, empty
batches, poisoned scratch reuse, and a custom Warp consumer contracting the
product of three cubic fields from the degree-nine moments. The latter fuses
integration and contraction without materializing the complete output matrix.
Automatic infinite-domain FEM/material packing is not advertised.

NVRTC 12.9 compiles the 2D and 3D functions for SM80 in all three modes. Raw
and paired fp32 PTX contain **zero fp64 instructions**. There is still no CUDA
hardware; neither runtime accuracy, graph replay nor GPU performance is claimed.
CPU benchmark and compiler reports record actual precision, masks, allocations,
preparation/readback costs and source SHA256 hashes. All recorded hashes match
the final implementation. Evidence is in `/tmp/fluxbridge-planar-evidence/`:
`report.json`, `pytest.log`, `compile.log`, and `cuda-compile/compile-*.json`.
Timings were serialized after tests and compilation, on Intel Core Ultra 9 285K,
Warp 1.17.0, with an existing JIT cache.

For 128 queries, W2, supplied signed distances/unit normals and every baseline
channel, CPU median milliseconds (axis-aligned geometry):

| Provider | fp64 warm | fp64 complete | paired fp32 warm |
|---|---:|---:|---:|
| 2D plane, 15 channels | 0.27 | 0.44 | 2.15 |
| 2D band, 15 channels | 0.44 | 0.61 | 3.77 |
| 3D plane, 28 channels | 4.63 | 5.22 | 64.49 |
| 3D band, 28 channels | 8.56 | 8.78 | 124.12 |

Complete costs include host validation, bound construction, allocation and
upload, but exclude static plan construction and readback, reported separately.
They expect distances/normals already supplied and do not include world-geometry
search. Rotated baseline warm fp64 costs are 0.39/0.52 ms (2D plane/band) and
5.58/8.82 ms (3D). The report also includes raw fp32 costs, which must not be
interpreted as accuracy-accepted timings.

The matched 3D axial plane baseline remains **0.23 ms fp64 / 1.75 ms paired**
for 128 queries, with mixed differences from the clipped provider of 6.7e-14 /
7.0e-9. Prefer that existing provider when only its baseline contract is needed.
The clipped provider adds weighted operations rather than improving that cost.
For a single band query requesting every degree-nine moment and derivative,
2D/3D fp64 warm costs are **0.18 / 19.80 ms** (330 / 2,200 outputs), versus
**0.82 / 213.06 ms** paired. Static plan construction samples are 9.47 / 64.84 ms
fp64 and 9.95 / 226.72 ms paired; these are single samples, not cold-JIT timings.

Output masks reduce work: the fp64 2D/3D occupancy plans have scratch width 16
versus 106/176 for baseline and 1,965/9,156 for degree nine. For 128 band queries,
occupancy alone costs 0.13 / 1.08 ms fp64 warm. Paired scratch includes cached
outputs (17, 121/204 and 2,295/11,356 respectively). The general 3D triangle
recurrences, especially paired arithmetic, are a measured bottleneck. This
checkpoint establishes correctness and composition; it does **not** close the
Stage 4 efficiency gate for a large weighted planar workload.

Reproduce using the existing Warp environment:

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_planar_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache WARPSPH_DEVELOPMENT_ROOT=/tmp/fluxbridge-planar-evidence \
  $PY -m pytest tests/fluxbridge tests/test_package_layering.py \
  tests/edge/test_disk.py tests/edge/test_warpbc.py tests/scene/test_disk_array.py \
  tests/scene/test_periodic_bodies.py tests/scene/test_body_kinematics.py \
  tests/mesh3d/test_reference.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_planar_acceptance.py \
  --device cpu --output /tmp/fluxbridge-planar-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_planar_acceptance.py \
  --compile-only float32_compensated --output /tmp/fluxbridge-planar-evidence/cuda-compile
# Repeat the compiler audit with float32 and float64.
# Repeat runtime acceptance with --device cuda:0 on an NVIDIA host.
```

The precision-priority checkpoint below supersedes this checkpoint's proposed
next step: prioritize raw-fp32 conditioning of the remaining general recurrences,
then compare any direct weighted planar primitive against the frozen channels
and complete query derivatives. CPU fp64 timings alone do not select the GPU
backend. Then extend independent circle fixtures
and native evaluation beyond the combined-degree-two baseline. Full-contract
lookup/series, sampled-jet contracts and CUDA-native preparation/search remain
open. The mixed-wall 2D/3D explorer stays in the plan but is deferred. Pressure,
porosity and the maintained SPH reconstruction/solver stay at Stage 5. Böttcher
et al. (2025), *Implicit Incompressible Porous Flow using SPH*, arXiv:2504.07739v2,
remains the principal porous-flow reference. Ordinary solid disk/cylinder and
periodic-fibre applications remain usable without porous coupling; their 2D
flow measures transverse rather than longitudinal response.

## Raw-fp32 priority and radial conditioning — 22 October 2026

The user clarified that the target GPU has 1/64 as much fp64 arithmetic
throughput as fp32. **Ordinary fp32 accuracy and GPU suitability take priority
over optimizing CPU fp64 cost.** fp64 remains the independent reference route;
paired fp32 remains explicit, with its own accuracy/cost measurements. Neither
CPU timings nor nominal arithmetic ratios alone establish a GPU kernel speedup.
The Resume sequence above now targets the remaining raw-fp32 recurrence failures.

The 143-file dirty tree was archived in `/tmp/fluxbridge-fp32-kickoff/` before
editing. Only seven FluxBridge code/test/study/document files changed. The 96
unrelated files from the original kickoff remain byte-for-byte preserved. No
main warpSPH, shared packages, legacy Scene or browser sources were modified.

Implemented exact radial-profile factoring in `fluxbridge/radial.py` and
`_radial_code.py`. Preserve `r**p * (piece_end-r)**k` through the existing
Bernstein-product integration, instead of expanding the remaining low-degree
polynomial into alternating shifted powers. Wendland residual coefficients have
one sign per profile. For example, before the common normalization, W6 uses

```text
W  = (1-r)^8 (1+8r+25r²+32r³)
L  = W'/r = -22 (1-r)^7 (1+7r+16r²)
rQ = L'   = 528 r (1-r)^6 (1+6r).
```

Cubic's outer Laurent profiles use factored numerators r*L and r³*Q with explicit
denominators. The broader randomized gate catches a cubic error missed by the
original fixtures, so its conditioning is retained too. This is the same
analytic operator and geometry, without quadrature, altered error budgets,
paired arithmetic hidden inside the raw mode, or an fp64 device fallback.
The prepared Plan adds `radial_powers`; callers should rebuild it through
`RadialEvaluator` before custom device calls.

Cubic's rational primitive still has a 64-term positive beta expansion. In raw
fp32 only, stop when `series + term == series`: subsequent positive terms are
nonincreasing and would all round away too. This preserves the fixed-64-step
**floating result**, rather than choosing a looser numerical tolerance. fp64
and paired fp32 retain 64 steps. Tests compare all nine invariant scratch values
and all Cartesian outputs bit for bit over 325 plane/sphere queries; the runtime
study also measures the fixed-step variant with identical inputs.

All 112 independent 70-digit radial fixtures / 3,136 baseline channels now pass
strictly in all three modes. The two old raw-W6 expected-failure waivers were
removed. Worst mixed unit-support errors:

| 3D plane/sphere baseline | fp64 | raw fp32 | paired fp32 |
|---|---:|---:|---:|
| All four kernels | 4.35e-15 | 2.11e-6 | 1.50e-7 |
| W6 raw-fp32 improvement | — | 1.36e-5 → 2.11e-6 | — |

A further raw-fp32 gate compares 7,000 plane/sphere queries (63,000 invariant
channels) with the independently retained shell integrator: randomized signed
distances, exact/tiny support contacts and 13 radius/support ratios from 1e-2
to 1e4. All pass the original 1e-5 mixed budget. These remain finite tests, not
a universal rounding or relative-error bound. The conditioning/units policy and
the combined-degree-two advertised radial contract are unchanged.

The final radial suite passed **41 tests**, with one CUDA skip. The focused
native/library/layering suite passed **212 tests**, with four CUDA skips and
**16 expected failures** confined to general mesh, box and clipped-planar raw
recurrences, in 35.51 seconds. Degree-nine weighted raw-fp32 acceptance remains
open in those families. Solid Scene sources are unchanged; their broader
regression evidence remains in the preceding checkpoint.

NVRTC 12.9 compiles the updated radial functions for SM80 in all three modes;
raw and paired fp32 PTX contain **zero fp64 instructions**. CUDA hardware/runtime
validation is still unavailable in this execution environment. Final reports
and source SHA256 hashes are in `/tmp/fluxbridge-fp32-evidence/report.json`,
`pytest.log`, `compile.log`, `cuda-compile/compile-*.json`, and `preservation.json`.
The fixed-64-step factored baseline is retained separately under
`fixed-64-beta-baseline/`. Final report/compiler hashes match the implementation.

CPU measurements were serialized after tests and compiler work (Intel Core
Ultra 9 285K, Warp 1.17.0, existing JIT cache). For 128 supplied signed-distance/
normal queries and all 28 outputs, raw-fp32 median milliseconds:

| Kernel | Plane warm / complete | Sphere warm / complete |
|---|---:|---:|
| W2 | 0.18 / 0.30 | 0.24 / 0.35 |
| W4 | 0.41 / 0.53 | 0.54 / 0.69 |
| W6 | 0.87 / 0.98 | 1.09 / 1.22 |
| Cubic, fp32 early exit | 0.33 / 0.44 | 0.49 / 0.61 |

The matched fixed-64-step cubic warm costs are 1.44 / 2.26 ms, with bitwise-equal
outputs. This demonstrates reduced CPU work for the raw path, not a measured
GPU speedup. Complete costs include validation, allocation and upload; static
plan preparation, compilation, world-geometry search and readback are separate.
The report includes one-query, fp64 and paired-fp32 measurements without using
them to select the intended GPU production backend.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge tests/test_package_layering.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --device cpu --output /tmp/fluxbridge-fp32-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_radial_acceptance.py \
  --compile-only float32 --output /tmp/fluxbridge-fp32-evidence/cuda-compile
# Repeat compiler audit with float64 and float32_compensated.
# On the target NVIDIA host, repeat runtime acceptance with --device cuda:0.
```

**Next executable step:** isolate the failing raw-fp32 W4/W6 Hessian and weighted
channels in the general mesh/box/clipped-planar family, separate primitive error
from profile-coefficient amplification, and derive a coupled conditioned profile
reduction against the frozen degree-nine fixtures. Preserve exhaustive/local
parity and strict error budgets. A future direct weighted planar provider should
be chosen for raw-fp32 accuracy and measured GPU cost. Pressure, porosity and
solver integration remain Stage 5; the mixed-wall visualization stays planned
but deferred. Ordinary solid disks/cylinder sections and periodic fibres remain
independent of porous coupling, with transverse and longitudinal response kept
distinct. Böttcher et al. (2025), arXiv:2504.07739v2, remains the principal porous
reference.

## Conditioned raw-fp32 2D polygon integration — 22 October 2026

The 143-file dirty tree was archived before edits under
`/tmp/fluxbridge-fp32-general-kickoff/`, with a SHA256 manifest and tracked diff.
Only FluxBridge library, test, study and coverage/operator/plan files changed.
All 96 unrelated original kickoff files remain byte-for-byte preserved. Main
warpSPH, shared packages, legacy Scene, browser work and solver code were not
modified or imported.

The numerical diagnosis (`scripts/studies/fluxbridge_fp32_diagnose.py`) separates
primitive, profile-coefficient and reduction error by reconstructing the final
sum offline in fp64 against the frozen arbitrary-precision fixtures. This
fp64 reconstruction is diagnostic only, not device arithmetic. In the worst
3D W6 case, native mixed error is 3.15e-3, with approximately 3.18e-3 from
primitive error and 7.1e-5 / 1.0e-4 from coefficient/reduction effects. A final
summation improvement alone cannot pass the gate. The signed term L1 norm,
normalized by 1+|reference|, reaches 4.35e4 in this inventory. The report retains
all failing regular/compact channels rather than deleting adverse cases.

Implemented an explicitly selected **numerical** integration family in
`fluxbridge/conditioned2d.py`, exposed as `method="conditioned_boundary"` on
Evaluator, BoxEvaluator, 2D PlanarEvaluator, FieldEvaluator, differential and
material prepare methods. Ordinary fp32 and fp64 are supported; paired fp32,
3D volume and surface requests are rejected. The default analytic path and its
16 expected raw-fp32 failures remain unchanged. This is not a silent fallback,
an analytic certification, or a replacement of straight geometry by a model.

The new route uses Euler's radial flux for a homogeneous Cartesian monomial,
with cumulative profiles `C[k,f](r)=integral_0^min(r,1) t^(k+1) f(t) dt` for W,
L=W'/r and Q=L'/r. All weighted jet primitives are piecewise polynomials, even
for cubic's Laurent derivatives. Kernel/profile combinations are formed with
exact rational arithmetic, then converted to Chebyshev coefficients before
precision conversion. The prepared-profile coefficient L1 norm relative to its
endpoint maximum is at most **1.5486** in the full tested profile set. This is
a different, per-profile basis diagnostic from the whole-channel term norm
above; neither is a global numerical error certificate.

Boundary integration uses angle coordinates and dyadic panels split at kernel
knots. For r>=1, the nonzero cumulative constant supplies the complete exterior
regular flux. No face of an intersecting polygon is pruned. There is no fixed
panel-slot truncation or per-query host dispatch. Allocation-free device entry
points, unit-support outputs and physical scaling match the existing consumer
interface; scratch is two ordinary-fp32 entries per output for the sum and its
Neumaier residual. Bounded-angle trig avoids irrelevant fp64 slow paths.

New locality tests caught far-field flux roundoff being amplified by large
query-to-element polynomial translations in exhaustive field evaluation. A
native whole-element winding/point-to-segment test now clears disjoint rows
before contraction. Its distance padding scales with coordinate magnitude and
machine epsilon to retain shallow contacts. This is conservative finite-precision
conditioning, not a certified exact predicate for arbitrary geometry. Fused
field and transpose consumers now handle the new per-interaction local-coordinate
layout. Existing analytic and paired layouts are preserved.

Every monomial through degree nine and kernel derivative through order two is
retained. Independent fixtures were already frozen before the port: **120 cases /
6,880 channels per precision**, drawn from polygon, disk/box (rectangle cases)
and planar fixture sets. No fixture was rejected and no fallback was used.
Worst mixed unit-support errors at the default 16 nodes:

| Geometry / requested contract | fp64 | raw fp32 |
|---|---:|---:|
| Polygons, baseline and degree nine | 3.38e-15 | 7.07e-7 |
| Rectangles, baseline and degree nine | 2.77e-15 | 5.79e-7 |
| Planes/bands, baseline and degree nine | 3.82e-15 | 2.99e-6 |

Every case passes 1e-10 / 1e-5. Comparing 16 with 32 nodes gives worst mixed
differences below 1.41e-15 in fp64 and 4.79e-7 in fp32. These are finite
convergence results, not a universal quadrature bound or relative-error claim
for tiny intersections. Existing geometry policies apply; normalized element
spans above 1e8 and unrepresentable support powers are additionally rejected.

The new suite has **36 passing tests** and one CUDA skip. Tests cover complete
query derivatives, support units, masks/empty batches, exact and near contacts,
nonconvex polygons, containment, large translations, periodic arrays, poisoned
scratch reuse, and exhaustive/localized P0–P3 fields and transpose work. Cubic
resistance/velocity/test products through degree nine pass in raw fp32 for both
point and supplied-affine material work, with force/reaction/dissipation checked
against the analytic fp64 reference. A far disjoint degree-nine field is exactly
zero before translation. The combined native/layering/solid/reference run passed
**332 tests**, with 21 CUDA skips and 16 existing analytic-raw expected failures,
in 111.42 seconds. Ordinary solid APIs remain available without porous coupling.
After final low-level factory guards, the focused native/layering suite passed
248 tests, with five CUDA skips and the same 16 expected failures, in 38.17 seconds.

NVRTC 12.9 compiles the new primitive, fused field/transpose, plane/band wrapper
and deterministic reduction modules for SM80 in fp32 and fp64. All four fp32
modules contain **zero fp64 instructions**. This includes instantiated reduction
kernels, not just an empty module audit. CUDA hardware remains unavailable here;
no GPU runtime accuracy, graph replay or performance is claimed.

Final evidence: `/tmp/fluxbridge-fp32-general-evidence/diagnosis.json`,
`report.json`, `pytest.log`, `compile.log`, `cuda-compile/compile-*.json`, and
`preservation.json`. Reports record source SHA256 hashes, actual method/precision,
quad order, reference/convergence errors, allocations, search/preparation/readback
and complete costs. All final hashes match. Original evidence was preserved.
CPU timings were serialized after tests/compiler work, on Intel Core Ultra 9
285K, Warp 1.17.0, existing JIT cache.

Matched W6 triangle workload, requested channels and supplied queries, CPU
median milliseconds:

| Method | 128 queries / 15 outputs warm | Complete | One query / 330 outputs warm | Complete |
|---|---:|---:|---:|---:|
| Conditioned boundary quadrature, raw fp32 | 3.86 | 5.41 | 0.19 | 1.09 |
| Analytic paired fp32 | 8.40 | 10.13 | 2.52 | 3.10 |
| Analytic raw fp32 (fails accuracy) | 0.82 | 2.30 | 0.33 | 0.88 |

The conditioned/paired rows meet the same 1e-5 workload budget, with different
actual errors recorded in the report. Static preparation is separate (single
samples: conditioned raw 3.75 / 21.99 ms for baseline / degree nine). Complete
costs include host validation/search, allocation and upload; readback and JIT
compilation are separate. The high-degree row has one query, not 128. Cubic
conditioned raw costs are 3.83 ms warm for 128 baseline queries and 0.19 ms for
one degree-nine query. These CPU measurements do not select the target GPU
backend; hardware throughput and occupancy still require measurement.

For 128 queries against 64 rectangles, W6, all 15 baseline outputs, host search
selects 128 of 8,192 exhaustive interactions. Conditioned raw-fp32 warm/complete
costs are **1.45 / 5.66 ms**, versus 2.21 ms warm for exhaustive enumeration.
Localized and exhaustive outputs match exactly in this workload; mixed error
against analytic fp64 is below 3.40e-7. Host preparation dominates complete cost.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_fp32_diagnose.py \
  --output /tmp/fluxbridge-fp32-general-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache WARPSPH_DEVELOPMENT_ROOT=/tmp/fluxbridge-fp32-general-evidence \
  $PY -m pytest tests/fluxbridge tests/test_package_layering.py \
  tests/edge/test_disk.py tests/edge/test_warpbc.py tests/scene/test_disk_array.py \
  tests/scene/test_periodic_bodies.py tests/scene/test_body_kinematics.py \
  tests/mesh3d/test_reference.py -q -rx
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_conditioned_acceptance.py \
  --device cpu --output /tmp/fluxbridge-fp32-general-evidence
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_conditioned_acceptance.py \
  --compile-only float32 --output /tmp/fluxbridge-fp32-general-evidence/cuda-compile
# Repeat compiler audit with float64; use --device cuda:0 for runtime hardware gates.
```

**Next executable library step (superseded by the GPU checkpoint below):**
execute the existing analytic fp64 and supported numerical/factored fp32 routes
on CUDA at unchanged accuracy budgets, including fused consumers and graph
replay, then measure serialized GPU costs. The user accepts this precision/method
tradeoff and defers analytic raw-fp32 recurrence repair. Retain its failure
evidence. Native search remains open.
The mixed-wall explorer stays planned but deferred. Pressure, porosity and
solver integration remain Stage 5; Böttcher et al. (2025), arXiv:2504.07739v2,
remains the principal porous-flow reference. Standalone disks/cylinder sections
and periodic fibre arrays remain solid applications; 2D transverse flow does
not establish longitudinal permeability.

## GPU execution and accepted precision tradeoff — 22 October 2026

The user accepts the current **analytic fp64 / explicitly numerical fp32**
tradeoff and asks to defer analytic raw-fp32 conditioning. Keep its 16 precision
family failures and diagnostics visible; do not change tolerances, silently
substitute a method, or spend further work on an analytic-fp32 recurrence now.
fp64 remains available for the full analytic contract; the numerical fp32
alternative remains limited to its advertised geometry/degree domains.

**Actual hardware is available.** The restricted filesystem sandbox hides
`/dev/nvidia*`, causing Warp CUDA error 100 and `nvidia-smi` driver errors. The
same read-only probe outside that sandbox detects the NVIDIA RTX PRO 6000
Blackwell Workstation Edition, 95 GiB, SM120, driver 610.43.02 (CUDA 13.3),
with Warp 1.17.0 / toolkit 12.9. Earlier statements about absent hardware are
superseded by this execution evidence. No driver installation or device
configuration change was needed.

The combined CPU/CUDA/native/solid/layering regression run completed with
**522 passed, 32 expected failures, zero skips or unexpected failures**, in
660.24 seconds including JIT compilation. Of these, 174 passing tests have an
explicit `cuda:0` parameter and five additional CUDA-only tests validate graph
replay. The 32 expected failures are the same 16 analytic raw-fp32 misses on
each device. The accepted fp64, numerical/factored fp32 and explicit paired-fp32
routes retain their `1e-10` / `1e-5` reference budgets. Tests exercise independent
weighted fixtures, reusable/composed device functions, empty/reused buffers,
local versus exhaustive support, periodic images, P0–P3 fields and transpose,
degree-nine material products, supplied-affine force/torque/work identities,
and legacy solid disks/body kinematics. No new expected-failure waiver or
operator change was needed for CUDA.

The 2D numerical route passes all 120 cases / **6,880 channels per precision**
on CUDA: worst raw-fp32 mixed error **2.9845e-6**, fp64 **3.814e-15**, with no
rejections or fallbacks. The raw-fp32 radial plane/sphere baseline also passes
all 3,136 channels, worst **2.107e-6**; circle quadrature passes all 1,500,
worst **3.947e-7**. Cubic radial early-exit and fixed-64-step buffers remain
bitwise equal on CUDA. These are finite fixture results, not universal bounds.

The analytic planar provider passes all **22,820 channels per precision** in
fp64 (worst **8.286e-12**) and explicit paired fp32 (worst **2.783e-7**).
Ordinary analytic fp32 retains 1,942 failing channels, worst **2.556e-3**;
that deferred result is retained, not waived or folded into the numerical route.

All five study reports completed on `cuda:0`. Their accepted precision/family
rows have **zero failing channels, rejected cases or automatic fallbacks**:

| CUDA fixture family | Channels per precision | fp64 worst mixed error | accepted raw-fp32 worst | paired-fp32 worst |
|---|---:|---:|---:|---:|
| General analytic 2D/3D volume/surface, regular + compact | 22,344 | 6.090e-12 | deferred | 4.652e-8 |
| Analytic rectangles | 1,680 | 8.469e-12 | deferred | 1.482e-7 |
| Analytic clipped planes/bands | 22,820 | 8.286e-12 | deferred | 2.783e-7 |
| Numerical 2D polygons/rectangles/planes/bands | 6,880 | 3.814e-15 | 2.985e-6 | not offered |
| Factored 3D radial plane/sphere baseline | 3,136 | 4.343e-15 | 2.107e-6 | 1.496e-7 |
| Numerical circle baseline | 1,500 | 2.885e-15 | 3.947e-7 | 1.489e-7 |

Repeated geometry/channels across fixture families and regular/compact modes
are separate acceptance checks, not extra unique operators. `summary.json`
checks every report's source hashes and accepted accuracy rows; raw analytic
failures are grouped separately. The full regression suite has no skips.

Evidence is in `/tmp/fluxbridge-gpu-evidence/`: `pytest.log`, machine-readable
`pytest.xml`, `hardware.csv`, per-family `report.json` and logs. The 147-file
dirty tree was preserved first in `/tmp/fluxbridge-gpu-kickoff/` with an archive,
SHA256 manifest and tracked diff. The main warpSPH checkout, shared Core and
Integrator sources remain unchanged.

The five acceptance studies run **serially after regression testing**, with
`--device cuda:0`; the existing scripts synchronize around each timed call.
Warm timings therefore include Python/driver launch and synchronization cost;
they are not CUDA-event measurements of kernel arithmetic alone. Preparation,
readback, complete evaluation and graph replay are reported separately. Other
CUDA contexts and desktop graphics were present, so these measurements are
provisional shared-machine costs, not isolated GPU-throughput claims.

For W6 integration over one triangle (all requests evaluated at unit support):

| Method / precision | 128 queries × 15 baseline outputs, warm ms | complete ms | 1 query × 330 degree-nine outputs, warm ms | complete ms |
|---|---:|---:|---:|---:|
| Conditioned boundary quadrature, raw fp32 | 2.78 | 6.21 | 16.48 | 18.93 |
| Conditioned boundary quadrature, fp64 | 14.37 | 19.21 | 81.90 | 85.98 |
| Analytic paired fp32 | 1.57 | 3.04 | 33.60 | 34.48 |

The raw analytic timing is still recorded in the report but fails accuracy and
is not an accepted speed comparison. Small-batch degree-nine CUDA cost is a
real limitation: the numerical raw-fp32 route took about 0.19 ms on CPU in the
preceding checkpoint, versus 16.48 ms here on CUDA. Different machine activity
and host overhead preclude a clean speed ratio, but GPU execution alone does
not make this workload efficient. No production-backend choice follows solely
from the GPU's nominal fp64/fp32 arithmetic ratio.

The full 3D analytic W2 degree-nine mask contains 2,200 outputs. For one query,
warm plane/band costs were **2,603 / 5,470 ms in fp64**, and **1,944 / 4,350 ms
in paired fp32**. These are warm execution costs after compilation. The broad
planar acceptance study completed in 408.47 seconds; its repeated evaluations
confirm that this full-mask path is correct but currently unsuitable for a
production GPU workload. Sparse requested masks and representative batch sizes
must be profiled before deciding how to parallelize or reduce its work.

The fused 3D cubic-affine material action (P3 resistance, skeleton and test
fields, degree-nine product, 220 requested moments / 910 contracted outputs)
takes **4,890 ms in fp64 / 2,064 ms in paired fp32** for one interaction.
Paired output differs from analytic fp64 by 7.71e-8 in mixed error. Thus the
high-degree cost also affects a real consumer, not only an all-output diagnostic.
For comparison, the analytic W2 baseline over 64 tetrahedra selects 128
interactions for 128 queries: fp64 warm/complete medians are 19.31 / 20.74 ms
for all 28 baseline outputs. No accepted performance gate is inferred from
these correctness results.

The 64-rectangle / 128-query numerical-fp32 W6 assembly selects 128 of 8,192
interactions and agrees exactly with exhaustive output; warm/complete medians
were 1.05 / 5.97 ms. Its fp64 analytic comparison differs by 2.90e-7 in mixed
error. The five graph replay regressions passed with identical saved outputs.

Reproduction requires actual GPU device access; do not accept a CPU-only run
with CUDA tests silently omitted. Use the Warp Python environment above and
`XDG_CACHE_HOME=/tmp/fluxbridge-cache`:

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -c \
  'import warp as wp; wp.init(); assert wp.is_cuda_available(); print(wp.get_device("cuda:0"))'
XDG_CACHE_HOME=/tmp/fluxbridge-cache WARPSPH_DEVELOPMENT_ROOT=/tmp/fluxbridge-gpu-evidence \
  flock /tmp/fluxbridge-gpu.lock $PY -m pytest tests/fluxbridge \
  tests/test_package_layering.py tests/edge/test_disk.py tests/edge/test_warpbc.py \
  tests/scene/test_disk_array.py tests/scene/test_periodic_bodies.py \
  tests/scene/test_body_kinematics.py tests/mesh3d/test_reference.py -q -rx
# Execute these separately, serially, with --device cuda:0 and distinct --output directories:
# scripts/studies/fluxbridge_conditioned_acceptance.py
# scripts/studies/fluxbridge_radial_acceptance.py
# scripts/studies/fluxbridge_diskbox_acceptance.py
# scripts/studies/fluxbridge_planar_acceptance.py
# scripts/studies/fluxbridge_acceptance.py
```

**Next executable library step after this GPU gate:** profile raw-fp32 CUDA
execution at representative batch sizes and sparse/full output masks, separating
device execution from host launch/search/upload. Investigate the high-degree
small-batch cost before expanding the numerical geometry family. Analytic raw
fp32 repair is deferred; the 3D numerical extension, remaining special forms,
native search and mixed-wall explorer stay planned. Pressure, porosity and
solver reconstruction remain Stage 5. Böttcher et al. (2025),
arXiv:2504.07739v2, remains the principal porous-flow reference; ordinary solid
disks/cylinder sections and periodic fibres retain independent application paths.

## Practical near-roundoff 3D planar quadrature — 22 October 2026

The user asks whether an analytic definition/reference plus a practical
quadrature evaluating the same integrals to double precision is a defensible
performance route. **Yes, for the demonstrated 3D plane/band family:** an
explicit `method="shell_quadrature"` now replaces the slow clipped-face
recurrence for that geometry when selected. It does not change the operator,
kernel, requested degree, Cartesian axes or exact planar geometry. Keep the
analytic implementation as a separate verification route and keep analytic
raw-fp32 repair deferred. Do not describe this as an exact numerical replacement
for every geometry, or as a proof that an optimized analytic implementation
could never compete.

Before editing, the current 147-file dirty tree was archived with a manifest
and tracked diff in `/tmp/fluxbridge-quadrature-kickoff/`. No changes to the main
warpSPH checkout, shared Core/Integrator or existing solid/browser paths are
needed. The new implementation is `fluxbridge/planar_shell.py`, selected only
through `PlanarEvaluator` or its composable function factory. The general
polygon/tetrahedral `Evaluator` and `FieldEvaluator` do not acquire this method.

### Exactness argument and method contract

Use spherical coordinates about the query, with polar axis the wall normal.
The material region on a shell is the exact interval
`max(-1,lo/r) <= cos(theta) <= min(1,hi/r)`. Split the radial domain at kernel
knots and the absolute plane/band bounds so its clipping branches stay fixed.
For maximum combined moment/jet degree d (at most 11), `d+1` azimuth nodes
integrate every required Fourier harmonic; `ceil((d+1)/2)` polar Gauss nodes
then integrate the resulting polynomial exactly in exact arithmetic. For
Wendland C2/C4/C6 the remaining radial integrands are polynomials of degree at
most 22, so the default 16 radial Gauss nodes exceed polynomial exactness needs.
See the API derivation and [NIST's Gauss quadrature property](https://dlmf.nist.gov/3.5.v).

Cubic's outer Hessian also has bounded Laurent terms at `r >= 1/2`; there the
radial rule is numerical, checked independently and against 24 nodes. Device
evaluation is ordinary fp32/fp64: factored profiles incorporate the spherical
Jacobian before singular factors, and output sums use compensation. Static
Gauss rules are polished with extended host precision before fp64 storage.
This gives polynomial-exact quadrature in the stated Wendland class and
**near-roundoff floating-point evidence**, not guaranteed correct rounding or
uniform relative accuracy at zero. A two-order convergence check alone would
not establish accuracy; the frozen arbitrary-precision fixtures supply it here.

The new batch prepares query-dependent samples on the device and parallelizes
over outputs. Each repeated launch rebuilds those samples and reuses all
allocations. A composed `wp.func element` uses the same arithmetic with caller
scratch, evaluated serially for that consumer; its performance is not the
parallel-batch timing. The speedup includes this changed GPU work distribution,
not just a change of integration formula. The previous 2D quadrature's serial
output loop is not evidence against this implementation.

### Independent CPU/CUDA acceptance

All 68 existing 3D plane/band cases / **19,280 channels per precision** pass
on CPU and the RTX PRO 6000 Blackwell, including every degree-nine moment and
second kernel derivative. Both devices give these worst mixed errors:

| Kernel | fp64, budget 1e-14 | raw fp32, budget 1e-5 |
|---|---:|---:|
| Wendland C2 | 2.001e-15 | 1.436e-6 |
| Wendland C4 | 4.241e-15 | 1.327e-6 |
| Wendland C6 | 3.892e-15 | 1.198e-6 |
| Cubic | 3.304e-15 | 1.062e-6 |

There are no rejected cases, automatic fallbacks or new expected failures.
16/24-node mixed differences are at most 2.520e-15 in fp64 and 1.834e-6 in
raw fp32. Extra rotated degree-nine requests agree with the independent
70-digit mesh/flux reference, not merely with analytic fp64. Unit scaling,
complete query derivatives, subset/repeated requests, empty batches, contacts,
poisoned/reused scratch, polynomial/transpose work, composed device calls and
graph replay pass. The existing planar/2D-quadrature/layering run passed
**168 tests**, with 16 existing analytic-raw expected failures; after adding
the rotated independent reference, the final shell-only CPU/CUDA run passed
**28 tests**, with zero skips or expected failures. These runs overlap and
their counts must not be added as distinct tests.

NVRTC builds both precisions for **SM120**. The raw-fp32 module contains **zero
fp64 instructions**, separately from actual runtime testing. Reports include
current source/fixture/test SHA256 hashes. Evidence is under
`/tmp/fluxbridge-quadrature-evidence/`: `cpu/report.json`, `cuda/report.json`,
`pytest.log`, `pytest.xml`, `pytest-shell.log`, `pytest-shell.xml`, and
`cuda-compile/compile-*.json`. Existing analytic-fp32 failure evidence remains.

### Matched GPU costs

Measurements ran serially after tests, on the same GPU as the prior checkpoint.
The table compares the current analytic and numerical implementations on the
same W2 geometry, requested outputs, support and precision. One-query workloads
use distance -0.35 and +z normal; bands have half-width 0.2. Baseline batches
use 128 distances spanning -0.35 to 0.9. Warm values are medians of three
synchronized launches, including sample generation for quadrature:

| Workload, fp64 | Analytic warm ms | Shell quadrature warm ms | Current implementation speedup |
|---|---:|---:|---:|
| Plane, 1 query × 2,200 degree-nine/jet outputs | 2,650.60 | 13.01 | 204× |
| Band, 1 query × 2,200 degree-nine/jet outputs | 2,661.63 | 9.43 | 282× |
| Plane, 1 query × 220 degree-nine moments | 1,172.55 | 4.41 | 266× |
| Band, 1 query × 220 degree-nine moments | 1,190.61 | 6.69 | 178× |
| Plane, 128 queries × 28 baseline outputs | 12.03 | 2.65 | 4.5× |
| Band, 128 queries × 28 baseline outputs | 10.49 | 0.627 | 16.7× |

For 32 full-mask queries, numerical fp64 takes 24.15 ms (plane) / 18.50 ms
(band) total; raw fp32 takes 3.90 / 2.41 ms. Raw-fp32 one-query full-mask costs
are 3.56 / 2.25 ms, with its own looser precision gate. The fp64 full-mask
complete evaluation samples, including allocation/upload/readback, are 15.41 /
12.14 ms, compared with analytic 2,667.58 / 2,676.31 ms. Complete times are
single samples, not three-sample medians. GPU graph outputs are bitwise equal
to the corresponding uncaptured outputs in every runtime row.

CUDA-event intervals are recorded separately: full-mask fp64 plane/band
intervals are 10.71 / 9.43 ms versus analytic 2,648.34 / 2,661.59 ms. Events
span both preparation and integration kernels and can include inter-launch
gaps; they are not instruction-level arithmetic measurements. Other desktop
and CUDA contexts remain present, so these are shared-machine measurements.
The improvement is real for these matched runs; isolated throughput remains
a separate qualification. Sample caching costs memory: a one-query fp64
full-mask batch uses 367,024 bytes versus analytic 92,048; the 128-query
baseline uses 3,602,432 bytes versus analytic 362,496.

This validates a practical evaluation argument for the **planar family**.
The 4.89-second tetrahedral cubic-affine material action from the previous
checkpoint is **not replaced** by this provider. Moment-only timings show the
cost of obtaining the planar polynomial moments; they are not a measured
tetrahedral material speedup or an infinite-domain FEM implementation.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_shell_acceptance.py \
  --device cpu --output /tmp/fluxbridge-quadrature-evidence/cpu
# Requires actual GPU access; run serially after tests:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_shell_acceptance.py \
  --device cuda:0 --output /tmp/fluxbridge-quadrature-evidence/cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_shell_acceptance.py \
  --compile-only float32 --arch 120 --output /tmp/fluxbridge-quadrature-evidence/cuda-compile
# Repeat compiler audit with float64; runtime execution is checked separately.
```

**Next executable step:** establish support-aware tetrahedral quadrature fixtures
and an error-controlled evaluation rule for the real degree-nine material
consumer, including near faces/edges, kernel-sphere intersections, containment
and slivers. Compare at the same independent fp64 accuracy as the analytic
operator and report complete fused-action cost; a naive fixed Duffy rule must
not inherit the planar exactness claim. Retain the analytic route for validation
and keep all current method labels explicit. The mixed-wall explorer stays
planned, and pressure/porosity and maintained solver integration remain Stage 5.
Böttcher et al. (2025), arXiv:2504.07739v2, remains the principal porous-flow
reference; standalone disks/cylinder sections and periodic transverse fibre
flow remain usable without porous coupling.

## Near-roundoff tetrahedral quadrature and real P3 material action — 22 October 2026

Continued the requested practical quadrature route beyond infinite planes.
`method="tet_quadrature"` now evaluates **tetrahedral volume moments through
degree nine and kernel derivatives through order two**, in ordinary fp32/fp64.
The general evaluator, P0–P3 polynomial fields/transpose, point resistance and
supplied-affine material consumers use it through the existing `method` option.
Defaults remain analytic; method selection never silently changes precision.
Surfaces, 2D geometry, general closed skins and paired precision are rejected
by this numerical family. Analytic raw-fp32 conditioning remains deferred.

The 150-file current dirty tree was preserved first in
`/tmp/fluxbridge-tetquad-kickoff/`, with archive, SHA256 manifest and tracked
diff. The main warpSPH checkout, shared Core/Integrator, existing solid Scene
and browser implementations were not modified. The independent 70-digit flux
oracle remains the source of truth; no incompressible solver is imported.

### Formulation, support and accuracy control

For the combined radial profiles W, L=W'/r and Q=L'/r, prepare
`F[k,f](r)=int_0^min(r,1) t^(k+2)*f(t) dt` exactly before floating conversion.
For `k=|beta|`, the potential
`y*y^beta*F[k,f](|y|)/|y|^(k+3)` has divergence `y^beta*f`.
Its radial primitive is regular at the origin for every advertised request.
The primitive's exterior constant is retained; individual distant faces are
not dropped. Kernel-only gradients/Hessians use the same exact W/L/Q identities
as the analytic operator.

Each oriented face is decomposed into signed sectors from its plane foot to
the edges. With `rho=|d| tan(theta)`, the flux integrand is a unit-direction
monomial times the combined radial primitive, with angular measure
`sign(d)*sign(edge)*sin(theta) dtheta dphi`. Split both angular rules where the
kernel spheres intersect the sectors, and subdivide at dyadic geometric scales
near faces/edges. This integrates the exact straight tetrahedral geometry; it
introduces numerical angular integration, not a polyhedral approximation to
the support sphere. Unlike the planar shell rule, **these angular rules are
not polynomial exact**. Exact query-on-face contributions with d=0 vanish for
the regular potential.

`prepare()` compares the default 16-node angular rule against 24 nodes on
every normalized requested moment, at mixed-difference budgets 1e-14 (fp64)
and 1e-5 (fp32). It promotes by eight nodes when needed and rejects unresolved
convergence at 64. Polynomial fields also compare the actual translated and
contracted per-element outputs at 1e-12 / 1e-5; these can promote through 32
before explicit rejection. This is convergence control, **not a certified
global error bound**. Independent references are essential: two orders can
share an implementation error. Tests exercise real promotion and unresolved
convergence rejection; no new expected-failure waiver is used.

Host panel preparation uses fp64 and vectorizes all azimuth nodes of each
radial interval. Stored sample data and every device operation use the requested
precision. A padded whole-tetrahedron distance test prunes disjoint rows while
retaining containment; empty rows are cleared before polynomial translation.
No internal-face cancellation across material traces or floating-point atomics
are introduced. Existing conditioning/tiny-element/scaling policies remain;
normalized spans above 1e8, more than two million samples per interaction or
more than 1 GiB of profile/tile scratch per rule produce explicit errors.

### Independent accuracy and regression evidence

The existing tetrahedral fixtures contribute 9,400 channels per precision.
The new `fluxbridge-tetquad-v1.json` adds 40 independently generated 70-digit
cases / 1,240 channels: exact and near faces, face/edge/vertex tangency,
disjoint support inside the bounding box, full-support containment, a 2,000:1
sliver and a scalene tetrahedron. Selected additional moments/jets reach degree
nine. Both CPU and actual CUDA pass all **10,640 channels per precision**:

| Kernel | fp64 worst mixed error, budget 1e-14 | raw fp32 worst, budget 1e-5 |
|---|---:|---:|
| Wendland C2 | 1.543e-15 | 4.573e-7 |
| Wendland C4 | 2.113e-15 | 7.684e-7 |
| Wendland C6 | 2.634e-15 | 4.415e-7 |
| Cubic | 1.478e-15 | 1.807e-7 |

All report rows have zero rejected cases, zero failing channels, no promotions
at the default order, and zero precision/method fallbacks. Worst 16/24 mixed
differences are 5.211e-16 in fp64 and 2.337e-7 in fp32. The largest accepted
fixture rule has 232,544 samples; resource limits were not bypassed.

The full CPU/CUDA/native/solid/layering run passed **589 tests**, with the same
32 expected analytic-fp32 misses and no skips/unexpected failures, in 255.88 s.
After vectorizing host preparation and adding direct `wp.func` checks, the
final tetrahedral-only CPU/CUDA run passed **43 tests** with no skips or
expected failures, in 52.02 s. The runs overlap; their counts are not additive.
Tests include P0–P3 field reproduction and transpose work, local/exhaustive and
periodic comparisons, complete query derivatives, support scaling, independent
batches, poisoned/reused storage, large disjoint polynomial translations,
stale bindings, the composed prepared device function, degree-nine material
work/force/torque and graph replay of the complete material action in both
precisions. Ordinary solid disks, periodic bodies and body kinematics still pass.

NVRTC compilation for **SM120** passes for tetrahedral evaluation, fields,
deterministic reduction and affine-material kernels. All four raw-fp32 modules
contain **zero fp64 instructions**. CPU and actual CUDA runtime evidence are
separate from those compiler audits. Source, fixture and test SHA256 hashes
are saved with each report.

### Fast device pipeline and prepared-state contract

A launch recomputes the combined profiles at the frozen sample locations,
integrates **256-node tiles**, reduces them with compensated summation and
contracts the polynomial fields in parallel by component. Tile arrays are
ragged by interaction; one large rule does not pad every row to its size.
The same integral-range arithmetic is exposed by the prepared `wp.func`
consumer. That direct consumer is serial for its row; the multi-pass path is
the measured fast route. The performance improvement includes the changed
GPU parallelization, not merely a different integral formula.

Every batch owns its sample arrays and **`batch.integration_plan`**. A low-level
composed call must use that prepared plan and its matching local query/support;
the static `evaluator.plan` does not bind quadrature. Changed query/support
arguments produce NaNs as a stale-binding diagnostic. Changing queries,
supports, geometry, adjacency or material fields requires a new prepared
batch. `launch()`/`apply_device()` allocate nothing and are capture-compatible.
They recompute the integrals on each launch; the benchmark does not obtain its
speed by caching completed material integrals across actions. The supplied
gradient and its matching transpose remain the affine model's contract.

### Matched real material performance

The benchmark now uses the **real P3 Bernstein resistance, P3 skeleton velocity
and P3 test material action**, with supplied affine fluid velocity. One
tetrahedron `[(0,0,0),(2,0,0),(0,2,0),(0,0,2)]`, query `(0.2,0.2,0.2)`, unit
support, 220 degree-nine moments and 910 packed outputs match across methods.
Each action includes moment evaluation, field contraction, fluid force,
gradient load, skeleton reaction and dissipation. Material coefficients and
queries are frozen for warm calls. Three synchronized launches give the warm
median; preparation/complete columns below are single samples. Timings ran
serially after tests on the RTX PRO 6000 Blackwell, Warp 1.17.0.

| Kernel / method / precision | Warm full action ms | Reprepare, check and apply ms |
|---|---:|---:|
| W2 analytic fp64 | 2,460.09 | 2,482.93 |
| W2 tetrahedral quadrature fp64 | 7.44 | 50.99 |
| W2 tetrahedral quadrature raw fp32 | 2.81 | 36.41 |
| W6 analytic fp64 | 3,216.37 | 3,257.25 |
| W6 tetrahedral quadrature fp64 | 7.44 | 54.94 |
| W6 tetrahedral quadrature raw fp32 | 2.67 | 40.20 |

At matched fp64 precision this is **330× / 433×** faster for the frozen action,
and **48.7× / 59.3×** for reprepare-plus-apply. The reported CUDA-event intervals
also include the multi-kernel pipeline: 7.441 / 7.432 ms for numerical fp64,
versus 2,460.114 / 3,216.405 ms analytic. Complete material graph replay is
bitwise equal to ordinary launches. These are current-implementation,
shared-machine measurements, not an inherent lower bound on analytic methods
or isolated GPU-throughput qualification. Other desktop/CUDA contexts existed.

The stored packed fields are additionally contracted against the independent
70-digit frozen moments, using high-precision coefficient translation. The
quadrature's worst packed mixed errors for W2/W6 are **2.145e-15 / 1.492e-15**
in fp64 and **2.122e-6 / 8.882e-7** in fp32. This measures integration of the
stored polynomial field; it does not remove error from converting controls
to that polynomial representation. Normalized material work defects are at
most 2.757e-15 in fp64 and 1.020e-6 in fp32; force/torque checks pass and the
measured dissipation is positive. No negative-eigenvalue or energy clamp is
inserted. Signed quadrature has no universal PSD/positivity guarantee.

The tradeoffs matter. The accepted numerical batch uses **6,327,804 bytes in
fp64 / 3,164,308 bytes in fp32**, versus analytic 50,400 bytes (W2) / 69,760
bytes (W6), excluding shared static field/plan buffers in both methods. Static
220-by-910 field coefficients add 1,601,600 / 800,800 bytes separately. The
16-node accepted rule has 49,728 samples and its 24-node comparison has 112,032;
comparison buffers require extra temporary memory. Checked preparation takes
about 37–51 ms in these GPU runs and dominates moving-query complete cost.
The query arrays start on the host; readback of live GPU particle queries is
an additional integration cost until GPU-native preparation exists.

CPU costs remain separate: numerical fp64 frozen action is 37.97 / 40.27 ms
versus analytic 45.24 / 50.98 ms, but checked reprepare is substantially more
expensive than the CPU analytic path. This is not a reason to change the API
default automatically or infer GPU performance from CPU timings.

Evidence: `/tmp/fluxbridge-tetquad-evidence/` contains `cpu/report.json`,
`cuda/report.json`, both runtime logs, `pytest.log`/`pytest.xml`, the final
tetrahedral `pytest-final.log`/`pytest-final.xml`, the direct consumer log,
`fixtures.log`, compiler JSON/source/PTX, and the preservation audit.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_tetquad_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_tetquad_acceptance.py \
  --device cpu --output /tmp/fluxbridge-tetquad-evidence/cpu
# Requires actual GPU access; run serially after tests:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_tetquad_acceptance.py \
  --device cuda:0 --output /tmp/fluxbridge-tetquad-evidence/cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_tetquad_acceptance.py \
  --compile-only float32 --arch 120 --output /tmp/fluxbridge-tetquad-evidence/cuda-compile
# Repeat compiler audit with float64; it does not replace runtime execution.
```

**Next executable step:** measure preparation, memory and action cost across
realistic interaction counts, then stream/chunk the quadrature caches and move
query-dependent panel/sample preparation and convergence scheduling onto CUDA.
Keep the independent near-roundoff and work/transpose gates while reducing
sample counts; do not drop the reprepare column or hide resource rejections.
General closed-skin/surface numerical rules, native search and remaining
special forms stay separate. The mixed-wall explorer remains planned; pressure,
porosity and maintained solver reconstruction remain Stage 5. Böttcher et al.
(2025), arXiv:2504.07739v2, remains the principal porous-flow reference.
Standalone solid disks/cylinder sections and periodic fibre applications
remain independent; transverse 2D flow does not establish longitudinal response.

## Native mixed-wall web explorer checkpoint — 23 October 2026

The user authorized continued overnight work until the independent Stages 0–4
and the web view are complete, with no warpSPH changes. A persistent goal and
hourly thread heartbeat `finish-fluxbridge-library-and-web-explorer` retain
that objective. Stop/disable continuation when every in-scope gate passes;
never use it to begin Stage 5. The snapshot at
`/tmp/fluxbridge-explorer-kickoff/` preserves the 155-file dirty tree before
this milestone. This checkpoint is **partial delivery**, not acceptance of all
remaining library/web work.

Implemented `visualizer/wall.py`, `/api/wall` and `static/wall.html` with links
from the retained 2D/3D viewers. The local native provider uses explicit
`conditioned_boundary` in 2D and `tet_quadrature` in 3D, or caller-selected
analytic fp64. It exposes CPU/CUDA and raw-fp32/fp64 choices; analytic raw-fp32
is withheld under the accepted policy. The displayed moment dtype is explicit;
polynomial/material diagnostics currently contract those moments in **host
float64**, separately reported. These UI costs are not the fused GPU material
benchmarks from the preceding checkpoint.

A conforming generator splits the porous/impermeable interface before meshing.
Wall dimensions, permeable fraction, mesh count/grading, position/orientation,
query resolution, support/kernel, affine fluid/skeleton velocities, resistance
magnitude, anisotropy and polynomial degree P0–P3 are configurable. The 3D wall
is tetrahedral, with XY/XZ/YZ coordinate slices and projected element edges.
Circle/cylinder and finite fibre-section presets remain impermeable chord
geometry with no added effective drag. This retains a standalone solid mode;
periodic fibre cells remain a separate UI gap.

The page shows regional heatmaps, selected whole-element candidates, support
footprints, sampled-query probes and cross-interface line plots. Native
per-region moment arrays and requested-channel metadata are exported. Solid,
porous and free occupancy sum to one; exact full-support second moments define
the free complement. Physical units are used without the older viewer's h
normalization. Unsupported free-fluid constitutive quantities show unavailable.
The compression diagnostic is the supplied boundary continuity contraction
`int (u_fluid(query)-u_skeleton(source)) dot grad_query W`, not a pressure
residual or full porous density model. Permeation is a prescribed normal
velocity/local resistance response. Drag/reaction, weighted D and skeleton
products, dissipation and `(I+dt*int D W)^-1` use moments through p+2. The
positive resistance profile `(1+.25*xi(source))^p` is explicit; it is not an
arbitrary FEM-control editor. Böttcher et al. v2 and the transverse/longitudinal
fibre distinction are visible in the page.

**Validation:** 74 tests passed with actual CUDA enabled in 18.44 s, with no
skips or expected failures. The suite includes the new explorer, existing HTTP
service and recursive package layering. Four kernels × two dimensions × both
precisions are compared with independently frozen high-precision fixtures on
CPU/CUDA. Dense polynomial algebra independently checks P0–P3 drag/energy
contractions, force reaction and positive local mobility. Tests also cover
exact regional mesh volume/interface, rotated/graded geometry, localized versus
exhaustive evaluation, mesh refinement at fixed geometry, disabled porous and
zero resistance modes, all coordinate slice axes, finite JSON, same-origin
rejection, invalid contracts and static export/reopen data. No existing solid
implementation or browser numerical engine was replaced.

Browser checks executed both 2D CPU and 3D CUDA requests, switched compression
and dissipation maps, and verified displayed regional values/occupancy closure.
The final exported 5×5 XZ slice uses 12 tetrahedra, raw-fp32 CUDA moments and
host-fp64 contractions. Its central dissipation is 1.590071 and occupancy
closure is one. It prepared in about 778 ms with a 2 ms action/readback sample;
these shared-machine UI timings are illustrative, not throughput qualification.
Export produced a finite native-data JSON snapshot. The CLI exports a static
site whose `wall.html` reopens the saved data; live recomputation requires the
loopback native service. The rendered browser view and native snapshot are
saved in `/tmp/fluxbridge-explorer-evidence/`, along with JUnit, logs and hashes.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest \
  tests/visualizer/test_wall.py tests/visualizer/test_server.py \
  tests/test_package_layering.py -q
$PY -m warpSPHBoundaries.visualizer --port 8765
# Open http://127.0.0.1:8765/wall.html
$PY -m warpSPHBoundaries.visualizer --wall --export /tmp/fluxbridge-wall-site
```

Remaining web gates: explicit browser/native fixture reports, automated browser
interaction/export/reopen coverage, arbitrary FEM controls, error/refinement
comparison panels, periodic fibre cells and a surface-only mode with distinct
units. Arbitrary imported meshes and 3D fibre geometry remain outside this
initial generator. The Stage 5 solver overlay is intentionally excluded.
Library-wide GPU streaming, CUDA-native search/panel preparation and retained
special-form ports also remain open. Query groups of four in this 3D explorer
bound individual caches; they do **not** close library GPU-preparation/scaling
requirements. **Next executable library step:** record material preparation,
action, memory and rejection costs across interaction counts, then implement
bounded reusable caches/GPU preparation while retaining accuracy/work gates.

## Real material batch scaling and shared-radius cache — 23 October 2026

Measured the existing implementation before changing its cache, using the new
`scripts/studies/fluxbridge_batch_scaling.py`. Each batch uses a real P3
resistance/P3 skeleton/P3 test supplied-affine action, 220 degree-nine moments
and 910 packed outputs. Queries vary within one tetrahedron, with one
interaction each; no repeated-query completed-integral cache is used. W2/W6,
1/8/32/128 queries, explicit precisions, three warm calls, CUDA graph replay and
a fresh checked preparation-plus-apply sample are recorded separately. GPU work
was serialized. The baseline report is in
`/tmp/fluxbridge-scaling-evidence/cuda/`, and its exact source was preserved in
`/tmp/fluxbridge-cache-kickoff/` before editing the library.

The baseline raw-fp32 128-interaction action takes 9.89 / 10.35 ms, but complete
reprepare takes 3,460 / 2,193 ms. Baseline fp64 rejects 128 rows because its
comparison rule exceeds the existing 1 GiB profile/tile scratch budget. Accepted
raw-fp32 batches use 411.4 MB at that count. Thus the preceding single-query
speedup does not establish practical moving-query throughput.

Implemented exact sharing of radial-profile cache entries within each row.
Full radial panels repeat radii across azimuth nodes and exterior samples use
radius one. `np.unique` merges only exactly equal values after selected-dtype
conversion. The device stores unweighted profile values once and applies the
original node weight at integration time. Row-local cache offsets and node
indices retain independent composed calls. This changes storage/evaluation
reuse, **not the quadrature nodes, weights, order, summation order or budgets**.
The existing scratch cap now accounts for compressed storage; no resource cap
was increased. This is not GPU-native panel preparation or streaming.

**Validation:** 117 tests passed on CPU/actual CUDA in 71.03 s, covering the
10,640 independent channels per precision, P0–P3 fields, material work/torque,
composed consumers, stale data, graph replay, explorer, HTTP and layering.
Four additional CPU/CUDA/precision tests poison caches and compose two rows
concurrently to check row ownership; all pass. The counts overlap earlier suites
and are not additive. Source hashes accompany the new scaling report in
`/tmp/fluxbridge-radius-cache-evidence/`.

| Varied interactions | fp64 batch MB before → after | raw fp32 batch MB before → after | New W2/W6 warm action ms, fp64 / fp32 |
|---:|---:|---:|---|
| 1 | 6.44 → 2.88 | 3.22 → 1.54 | 8.08 / 2.71; 7.91 / 2.69 |
| 8 | 51.38 → 23.01 | 25.70 → 12.30 | 12.61 / 3.03; 13.04 / 3.13 |
| 32 | 205.74 → 92.12 | 102.88 → 49.24 | 27.10 / 4.32; 26.96 / 4.13 |
| 128 | rejected → 368.42 | 411.42 → 196.93 | 92.25 / 9.72; 93.02 / 9.75 |

Memory falls about 55% in fp64 / 52% in fp32 at matched accepted sizes. Static
field coefficients, comparison-rule temporaries and allocator reservation are
separate from these accepted batch byte counts. At 128 interactions the new
rule has 6,466,240 samples and 1,009,792 fp64 / 1,003,705 raw-fp32 cached radii.
Both kernels and precisions now pass that count and graph replay. Raw packed
mixed differences against numerical fp64 stay below 3.605e-6; normalized work
and force balance pass their declared precision budgets, with positive measured
dissipation. Independent high-precision integration gates remain separate.

Complete preparation-plus-apply still costs 2,425 / 2,435 ms in fp64 and
2,185 / 2,204 ms in fp32 at 128 rows. Differences in individual complete samples
are not isolated speedup qualification; host preparation remains dominant and
sharing/sorting radii adds host work. The GPU action is approximately unchanged.
Do not label this a completed preparation-performance gate.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_batch_scaling.py \
  --device cuda:0 --counts 1,8,32,128 --kernels w2,w6 --analytic-max 0 \
  --output /tmp/fluxbridge-radius-cache-evidence/cuda
```

**Next executable step:** implement and validate device-side panel/sample
preparation, then bound/cache its working storage across larger moving-query
batches. Geometry preparation precision must remain explicit: current panels
are prepared in host fp64 even for raw-fp32 evaluation. Preserve the exact
support geometry, near-face tests and finite convergence/resource rejection
policies. Remaining web delivery gates and special forms stay on the same
active goal; warpSPH, pressure and porosity integration remain untouched.

## Device panel generation and reusable material compilation — 23 October 2026

Implemented the next preparation step instead of changing precision/error
budgets. `method="tet_quadrature", preparation="device"` now generates the
angular rule on Warp CPU/CUDA. Default host preparation remains available and
other families reject this option. Local fp64 geometry/query snapshots are
uploaded after existing host adjacency/validity checks. Device whole-element
distance rejection is conservative; uncertain Gram classification retains the
element. All regular-flux faces of a selected tetrahedron remain represented.

Each interaction has 24 oriented sectors. Integer counting and int64 scans
allocate exact storage; sectors emit angular-interval descriptors, then a
separate kernel generates all quadrature nodes in parallel. Shared radial
panels/exterior constants provide cache indices without reading back or sorting
node/radius arrays. Per-row nodes/caches remain independent. Geometry generation
is explicitly **fp64**, with directions/weights/radii rounded to the selected
fp32/fp64 evaluation dtype. The host still owns scalar allocation/convergence
decisions and requested-moment/field comparisons. This is not a GPU-only
particle pipeline. Device rules retain per-interaction sample limits, int32
index guards and a stricter 1 GiB per-rule estimate for node/cache/work storage
or interval-generation peak. Comparison temporaries and allocator reservation
are additional costs. Host-only preparation retains its documented guard.

`ResistanceMaterial.compile(...)` and the supplied-affine variant return a
`MaterialEvaluator` with copied polynomial fields, basis and constitutive
metadata plus a reusable static `FieldEvaluator`. New query batches retain
separate adjacency/rules/storage. Source mutation, offsets and controls cannot
change an existing compiled snapshot. Reconstruct material fields when changing
constitutive inputs, and recompile for changed geometry/material/evaluation
options. Existing `material.prepare()` keeps its full static-plus-query setup.
No completed moments are cached across actions or changing queries.

### Accuracy, execution and precision

The complete FluxBridge suite plus native web/HTTP/layering passed **600 tests**
in 258.67 s on CPU/actual CUDA, with the same **32 deferred analytic raw-fp32
expected failures**, no unexpected failures and no hardware skips. Separate
focused web checks after static-export isolation and evaluator reuse also pass;
the counts overlap, not add. Independent device-preparation fixture reports
cover all **10,640 channels per precision** with no rejected cases, failing
channels, promotions at the default order or method/precision fallbacks:

| Backend | Worst fp64 mixed error, budget 1e-14 | Worst raw fp32, budget 1e-5 |
|---|---:|---:|
| Warp CPU device generation | 2.634e-15 | 7.684e-7 |
| Actual CUDA device generation | 3.092e-15 | 7.684e-7 |

The finite adversarial set includes exact/near faces, edge/vertex tangencies,
slivers, full containment, disjoint support inside bounding boxes, all four
kernels, degree-nine moments and second kernel derivatives. Reuse, periodicity,
P3 field transpose, supplied-affine material work, reaction, graph replay,
unsupported options and explicit resource rejection pass. The new static
material snapshot tests also compare moving queries and deliberate source
mutation against independent compiled snapshots.

SM120 NVRTC audits pass for both precisions. The raw-fp32 evaluation, field,
reduction and affine-material modules each contain **zero fp64 instructions**.
The distinct `device_panels` geometry module intentionally contains fp64
arithmetic (1,468 instructions in the current raw-evaluation build). This
separation is explicit; calling the entire preparation pipeline raw fp32 would
be incorrect. Actual runtime checks are separate from compilation.

### Matched real P3 material scaling

The scene and query/control seeds match the preceding varied-query scaling
study: one tetrahedron, one interaction per query, P3 resistance/skeleton/tests,
220 moments and 910 outputs. W2/W6 and both precisions pass the declared work,
force, packed-action and graph gates at 1/8/32/128 interactions. Three warm
calls and one fresh complete-cost sample are recorded serially on the RTX PRO
6000 Blackwell. Static material compilation is separate, around 18 ms for
these raw-fp32 providers; full static rebuild costs are retained below.

| Interactions | Raw fp32 warm action ms W2 / W6 | Checked reprepare + apply, static reused ms W2 / W6 | Full static rebuild + apply ms W2 / W6 |
|---:|---:|---:|---:|
| 1 | 2.79 / 2.68 | 16.94 / 16.60 | 30.08 / 33.52 |
| 8 | 3.15 / 3.04 | 20.55 / 21.01 | 35.20 / 38.23 |
| 32 | 4.14 / 4.26 | 29.78 / 29.74 | 42.61 / 46.82 |
| 128 | 9.89 / 9.92 | 55.53 / 56.34 | 67.86 / 72.51 |

At 128 rows, the previous host-prepared/shared-radius path cost 2,185 / 2,204 ms
including complete static reconstruction. Device preparation with full static
rebuild costs about 68 / 73 ms here; explicit static reuse reduces that to
56 ms. These are measured current-implementation improvements, not isolated
GPU throughput or a universal error/performance guarantee. Timing includes the
convergence controls and recomputes material integrals on every action.

Accepted 128-row device batches use **401.0 MB fp64 / 213.5 MB fp32**, excluding
static coefficients and comparison-rule temporaries. Panel sharing keeps more
cache entries than host exact-deduplication, a measured storage tradeoff.
Fp64 warm action is about 93 ms at this count and its complete checked cost
remains around 0.4 s; the GPU's slow fp64 arithmetic still matters. Raw-fp32
packed differences against numerical fp64 remain within 1e-5, with positive
measured dissipation and passing common-work/force checks.

The web provider exposes the same explicit preparation option, physical units
and provenance. It now reuses one static evaluator across its bounded query
groups, reports total/static/query/action costs separately, and labels saved
snapshots. Static wall exports keep a separate ordinary viewer example, so
opening an existing 2D/3D page cannot misinterpret a 3D wall slice as a volume.
Live and exported views retain the native data and the Stage 5 diagnostic
labels; they do not add pressure, porosity or a fluid solver.

Evidence: `/tmp/fluxbridge-deviceprep-evidence/` contains independent CPU/CUDA
accuracy JSON/logs, the full JUnit/log, focused tests, both compiler audits and
PTX, initial serial-sector and parallel preparation reports, reused-static
scaling reports, exact measured-source archives, and the native/static web
artifacts. Earlier cache/scaling snapshots remain intact. The main warpSPH
checkout and shared Core/Integrator remain unchanged.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_batch_scaling.py \
  --device cuda:0 --counts 1,8,32,128 --kernels w2,w6 --analytic-max 0 \
  --preparation device --reuse-material --output /tmp/fluxbridge-deviceprep-evidence/reused-cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_tetquad_acceptance.py \
  --device cuda:0 --accuracy-only --preparation device \
  --output /tmp/fluxbridge-deviceprep-evidence/accuracy-cuda
# Compile separately with --compile-only float32/float64 --preparation device.
```

**Next executable step:** add and benchmark CUDA-native whole-element adjacency
against the exhaustive/host reference, preserving deterministic element/image
order, periodicity and containment. Then address larger streamed caches and the
remaining special-form/web gates. This checkpoint does not complete Stages 0–4
or the entire web plan. Pressure, porosity and maintained solver reconstruction
stay Stage 5; Böttcher et al. v2, ordinary solid cases and the transverse/axial
fibre distinction remain unchanged.


## Checkpoint — regular near-plane potential and physical wall axes (23 October)

Equal-scale physical plotting exposed a real numerical preparation defect:
normal 9×9 wall-grid probes can lie within floating-point roundoff of a face
and edge. The polar rule then exceeded its sample cap. The fix evaluates the
same face potential through the regular `F(r)/r^3` representation with a Duffy
triangle rule on moderate near-plane faces. It does not move queries, snap
geometry, drop faces, increase caps or substitute a lower precision/method.
Inner radial cancellation is exact before polynomial conversion. Cache kinds
keep the two representations distinct. Other faces retain the existing rule.

The new frozen 70-digit fixture contributes 1,240 channels. Host CPU preparation,
Warp CPU device preparation and actual CUDA device preparation each pass all
**11,880 channels per precision**, with no failures, rejected cases, default-order
promotions or fallbacks. Worst normalized mixed errors are **3.248e-15** on CPU
and **3.552e-15** on CUDA in fp64 (budget 1e-14), and **7.684e-7** in raw fp32
(budget 1e-5). Errors are normalized by the request's support-radius scaling.
The complete regression passes **621 tests**, with the same **32 explicitly
deferred analytic raw-fp32 expected failures**, in 294.14 s. This includes 16
primitive checks against independent 80-digit radial integration down to zero.
SM120 compilation passes; all four raw-fp32 evaluation/field/reduction/material
modules contain zero fp64 instructions. The distinct geometry module contains
1,626 expected fp64 instructions.

Serialized matched P3 CUDA scaling after this fix retains the earlier speedup:

| Interactions | Raw fp32 warm ms W2 / W6 | Checked reprepare + apply, static reused ms W2 / W6 | Full static rebuild + apply ms W2 / W6 |
|---:|---:|---:|---:|
| 1 | 2.71 / 2.79 | 16.95 / 16.53 | 31.79 / 38.31 |
| 8 | 3.12 / 3.04 | 20.99 / 20.81 | 37.38 / 43.28 |
| 32 | 4.12 / 4.14 | 30.10 / 29.62 | 45.39 / 51.97 |
| 128 | 9.63 / 9.64 | 55.76 / 57.94 | 70.69 / 82.66 |

All work/force/packed-action/graph gates pass. Accepted 128-row storage is
**219.03 MB raw fp32 / 406.57 MB fp64**; representation-kind arrays account for
the increase. Static coefficients, comparison temporaries and allocator
reservation remain additional. These are finite real-material measurements,
not a full moving-particle solver benchmark or universal accuracy guarantee.

The physical 9×9 P3 raw-fp32 CUDA wall export now succeeds. The browser reopens
that snapshot, renders a round support footprint with equal scale on both axes,
and shows the native solid/porous/total compression at `(0,0.4,0)` as
`-0.5852562 / -0.5852562 / -1.170512`. Unavailable free-fluid dissipation is
explicitly labelled. Plot bounds follow geometry plus support instead of a
stretched square. Bulk porous occupancy remains distinct from porosity.

Evidence: `/tmp/fluxbridge-nearplane-evidence/` contains independent host/CPU/CUDA
accuracy reports, SM120 audits/PTX, complete JUnit/log, serialized scaling,
working static export, browser/native value comparison and the inspected
`wall-compression.jpg`. Sources are hashed in the accuracy/scaling reports.
The prior evidence and original unrelated dirty work remain preserved.

**Next executable step:** implement reusable CUDA whole-element BVH search,
compare deterministic CSR/periodic image output against exhaustive enumeration,
and measure build, query/readback and complete consumer cost separately.
Stages 0–4 and the wider web gates remain in progress. No warpSPH or shared
Core/Integrator changes; pressure and porosity integration remain Stage 5.


## Checkpoint — CUDA search and shared fp32 field contractions (23 October)

### Implemented and retained contracts

`DeviceSearch` supplies an optional reusable whole-element Warp BVH to ordinary
moment, polynomial, supplied-affine material and solid disk-array providers via
`search="device"`. Host search remains the default and exhaustive evaluation
remains available. Outward-rounded recentered fp32 boxes drive traversal;
fp64 geometric filtering and bounded periodic enumeration keep conservative
contacts. Integer scans and sorting restore deterministic query/element/image
order. The present interface explicitly reads CSR back and uploads it into
prepared batches. It is not a GPU-only moving-particle pipeline. No individual
regular-flux face is removed, and differing material traces never cancel.
Containment, oriented cavities, surfaces, exact/near contacts, shifted/sliver
geometry and periodic disks/fibres pass CPU/CUDA checks.

Prepared tetrahedral fields now translate each monomial integral once per
interaction, sharing it across the 910 P3 outputs and transpose. The previous
repeated translation contributed substantial work and rounding error. A larger
16-element P3 scene exposed packed-output errors above 1e-5; these were fixed
without changing that precision target. Ordinary selected-dtype moments and
coefficients now use Neumaier sums, explicit FMA product residuals and a small
**same-dtype correction workspace** through projection. This is not the analytic
paired-profile/coefficient backend and no method/precision fallback occurs.
Weights and corrections are refreshed every action, including graph replay.
The worst packed fp32 differences in the larger scene are 7.416e-6 W2 and
4.227e-6 W6 against numerical fp64. Work, force, energy and graph gates pass.

The larger-scene force gate now uses `absolute defect / (1 + sum fluid-force
norms + sum nodal-reaction norms)`, with the absolute defect also retained.
A first absolute 1e-5 threshold failed at 1.190e-5 in a force scale of 8.675;
its normalized defect was 1.372e-6. This dimensional scaling change is explicit,
not an undeclared looser packed-output threshold. Saved initial-failure and
rounding-attribution evidence remains in the milestone directory.

`MaterialBatch.bytes` counts its owned field/rule/forward/reverse adjacency and
material-action arrays. At 128 rows the raw-fp32 batch owns **219.27 MB**, including
**225,280 bytes** of weight/correction workspace and **13,616 bytes** of material
action arrays in the 16-element case. Static field coefficients add **12.81 MB**.
Fp64 accepted storage is about **407 MB**. Comparison temporaries, static plans,
opaque BVH storage and allocator reservation are separate. Reported host RSS
is a process high-water mark (about 1.22 GB in this study), not batch memory.
Fp32 coefficient overflow is rejected before launch.

### Validation and instruction audit

Actual CUDA and CPU targeted checks pass for all newly added search and field
consumers. Independent degree-nine values and mixed Hessians are reconstructed
from the frozen 70-digit moment fixture and checked at two physical scales.
Transpose work, same-dtype operands, correction refresh and stale queries pass.
The new oracle initially needed a Python-int conversion for mpmath; the corrected
focused run passes **120 tests**, overlapping the broader regression. The final
full regression passes **797 tests** in 302.83 s, with **32 deferred analytic
raw-fp32 expected failures** and **four optional legacy WGSL module skips**
because Python `wgpu` is absent. No native CUDA checks were skipped.

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache /home/lu26029/miniconda3/envs/warp/bin/python \
  -m pytest tests/fluxbridge tests/visualizer tests/test_package_layering.py -q
```

The independent CUDA acceptance study still passes all **11,880 channels per
precision**, with worst normalized mixed error **3.552e-15 fp64 / 7.684e-7 fp32**,
no rejected cases, promotions or fallbacks. The real W2 P3 packed material action
against independent 70-digit moments has error **1.390e-16 fp64 / 1.544e-6 fp32**.
SM120 NVRTC audits pass. Raw-fp32 `tet_quadrature`, `fields`, `evaluate` and
`affine` modules each contain **zero fp64 instructions**, including the new FMA
contractions. Separate geometry modules contain 1,626 (`device_panels`) and 254
(`device_search`) expected fp64 instructions. The analytic raw-fp32 deferral
and its 32 expected failures remain unchanged.

### Serialized CUDA costs

The sparse search scene has one interaction per varied probe. Current Python
host-tree search is compared with native device search including upload,
allocation, scans/sorting, host synchronization and CSR readback. This is not
an isolated GPU traversal benchmark or a comparison with an optimized C++ CPU
search. Static Geometry construction and BVH build are separate.

| Elements | Probes | Host search ms | Device search with readback ms |
|---:|---:|---:|---:|
| 64 | 128 | 3.79 | 0.448 |
| 64 | 16,384 | 452.78 | 0.949 |
| 1,024 | 16,384 | 1,001.39 | 0.864 |
| 16,384 | 128 | 10.52 | 0.478 |
| 16,384 | 4,096 | 324.22 | 0.628 |
| 16,384 | 16,384 | 1,292.96 | 0.873 |

For 16,384 elements, host Geometry construction costs about **840 ms** once;
the reusable GPU BVH build/upload costs **1.31 ms**. Owned static search arrays
use **1.18 MB**. The largest query case owns **2.88 MB** of search arrays and
reads back **655,372 bytes**; opaque BVH/scan/sort storage is additional. Limits,
including 4,096 periodic images per query and the 1 GiB owned-array budget,
remain explicit resource rejections. Large overlap/periodic scaling is not
established by this sparse scene.

A matched 16-element scene cycles varied queries among separated tetrahedra,
with P3 resistance/skeleton/test fields, 220 moments and 910 outputs:

| Interactions | fp32 warm action ms W2 / W6 | Checked reprepare + apply, static reused ms W2 / W6 | Full static rebuild + apply ms W2 / W6 |
|---:|---:|---:|---:|
| 1 | 0.30 / 0.30 | 9.35 / 9.23 | 33.41 / 40.22 |
| 8 | 0.59 / 0.71 | 13.28 / 13.58 | 37.94 / 44.21 |
| 32 | 1.66 / 1.66 | 22.88 / 21.78 | 50.14 / 53.83 |
| 128 | 6.99 / 6.99 | 45.75 / 46.16 | 70.63 / 79.28 |

The same scene with host adjacency costs about 50 / 49 ms reused-static at
128 rows; search is a small fraction of this material workload. Initial
one-element timings before the final overflow check/memory-accounting update
remain separately source-archived, rather than relabelled as the final run.

The independent matched W2 single-query P3 action costs **2,619.98 ms analytic
fp64 versus 1.177 ms numerical fp64**, about **2,225×** faster for a warm action.
Fp32 costs **0.301 ms**. Complete static-rebuild-plus-apply costs are about
2,671 / 27.80 / 24.90 ms respectively. The fp64 numerical result passes the
70-digit check near machine roundoff; this finite evidence is not a global
error certificate or a polynomial-exactness claim for every clipped face.

### Web delivery and preservation

The native-backed 2D/3D wall view now compares exhaustive/localized support,
wall element refinement and numerical fp32/fp64 at the same physical probes.
Circle/fibre refinement is labelled as changing chord geometry. Comparisons
save with snapshots and reopen offline. Browser values for the live 2D panel
and exported 9×9 P3 CUDA 3D panel match native JSON, including compression at
`(0,0.4,0)` and all displayed comparison errors. Volume/area units, supplied-state
response, unavailable quantities and bulk occupancy versus porosity remain
explicit. A circular region has no single planar normal, so its planar normal
response is withheld. The wall fraction is geometric, not intrinsic porosity.
The local preview service is `http://127.0.0.1:8767/wall.html`; the exported site
is `/tmp/fluxbridge-search-evidence/wall-site/`.

Evidence: `/tmp/fluxbridge-search-evidence/` contains serialized search/material
reports, exact measured-source archives (`initial-passing/source` and
`accepted-source`), independent CUDA/70-digit acceptance, SM120/PTX audits,
JUnit/logs, initial failure diagnostics, saved comparisons and browser proof.
Earlier milestone evidence is retained. Original dirty-tree audit still finds
only the same five authorized changes among its 97 files; the other 92 are
preserved. No main warpSPH/shared Core/Integrator source changes or solver copies.
Böttcher et al. (2025), arXiv:2504.07739v2, remains the principal porous reference;
2D disk/fibre cases remain ordinary solid and transverse.

**Next executable step:** implement explicit bounded chunk streaming with reused
static providers, preserving deterministic reactions/common work and reporting
complete cost and peak owned storage beyond the single-rule cache limit.
Then finish the still-open special-form/model ports and comparisons. Stages 0–4
are not yet fully accepted. The main wall delivery gates are met; arbitrary FEM
editing, periodic-cell UI, imported geometries and a distinct surface-only view
remain broader extensions. Pressure/porosity identities have been retained under
Stage 5, respecting the user's explicit scope boundary; they are not silently
claimed as implemented independent material operators.


## Checkpoint — bounded streamed native batches (23 October)

`Evaluator`, `FieldEvaluator` and ordinary solid `DiskArrayEvaluator` now expose
`iter_evaluate(...)` and `evaluate_streamed(..., return_metrics=True)` with an
explicit `chunk_size` (default 32). The iterator snapshots queries/supports when
created, remaps compatible supplied CSR into each chunk, evaluates one native
batch and releases it before yielding an independent NumPy result. It preserves
scalar/tensor field shapes, periodic shifts and the selected method/precision.
Geometry revisions during a stream and stale adjacency are rejected. Resource
or convergence failures identify the failing range; no fallback or skipped row.

Compiled `MaterialEvaluator.apply_streamed(q,h,u,m,L,chunk_size=...)` also handles
ordinary and supplied-affine material work. Reactions/energy accumulate in fixed
chunk order in host fp64, with reactions rounded once to the selected output
dtype; native evaluation remains in that dtype. `DragResult.execution` records
complete driver time, interaction/chunk counts, maximum accepted native-batch
storage, host input/output/aggregation storage and explicit synchronization.
No pressure, density, porosity or timestep loop was added.

This is a bounded **host streaming driver**, with allocations, convergence
checks and readback per chunk. It cannot be captured as one GPU graph; frozen
prepared batch actions retain their separate graph interface. Output collection
needs a full host array; use the iterator to consume values incrementally.
Memory metrics do not silently include static plans, comparison rules, opaque
BVH allocations or total CUDA reservation in the accepted-batch byte count.

The targeted CPU/actual-CUDA suite passes **37 checks** (31 streaming cases plus
six layering checks). It covers 2D/3D values, scalar/tensor fields, mixed kernel
derivatives, periodic CSR, ordinary disks/fibres, copied inputs, batch lifetime,
material force/gradient/reaction/common work, empty inputs, invalid physical
storage and explicit rule-cap rejection. Tests lower the rule cap, verify a full
batch fails, then verify explicit streaming agrees with the original result.
The prior full 797-test checkpoint remains source-preserved.

### Complete P3 CUDA costs and bounded storage

The serialized study uses the same 16-element varied-query P3 scene as the prior
checkpoint: 220 moments, 910 packed outputs, P3 resistance/skeleton/tests. Static
material construction/compilation and initial runtime warmup are separate.
Every measured driver call prepares rules and recomputes integrals; the three
wall samples include allocation, convergence, evaluation, transfers and host
aggregation. W2 and W6 pass common work/force/energy and agree with a separately
streamed fp64 reference within **6.637e-7 mixed error** over the displayed material
outputs. These are finite data, not a global accuracy/positivity certificate.

| Interactions | Chunk size | Complete fp32 ms W2 / W6 | Max accepted batch MB | Single batch |
|---:|---:|---:|---:|---|
| 128 | 32 | 91.59 / 92.33 | 54.83 | accepted |
| 128 | 64 | 64.12 / 62.74 | 109.65 | accepted |
| 512 | 32 | 363.79 / 367.43 | 54.89 | explicit rule-storage rejection |
| 512 | 64 | 255.69 / 253.82 | 109.65 | explicit rule-storage rejection |
| 2,048 | 32 | 1,458.73 / 1,463.68 | 54.89 | explicit rule-storage rejection |
| 2,048 | 64 | 1,019.42 / 1,022.43 | 109.66 | explicit rule-storage rejection |

The 1 GiB **per-rule** cap is unchanged. Bounded execution costs more than an
accepted single batch and remains expensive for large moving-query workloads.
It makes those workloads executable without retaining every rule at once; it is
not a GPU-only production particle pipeline or a claim of real-time throughput.

Fresh-process CUDA-pool measurements exclude fp64 references and failed full
batches. For chunk 32, application-pool high-water usage is **187.85 MB at 128
queries / 188.00 MB at 2,048**. Chunk 64 at 2,048 uses **362.75 MB**. These include
comparison-rule and other pooled live allocations beyond the accepted batch.
After each driver, current pool usage returns to **12.95 MB** of static/runtime
arrays. Pool usage is not total context/driver/opaque storage or GPU reservation.
Host output/input snapshots and aggregation are reported separately.

Evidence: `/tmp/fluxbridge-stream-evidence/` contains JUnit/log, complete scaling,
separate fresh-process memory reports and exact verified `measured-source/`.
Source and unrelated dirty work were preserved first in
`/tmp/fluxbridge-stream-kickoff/`. Main warpSPH/shared Core/Integrator remain
unchanged; ordinary resolved fibres retain no effective drag, and transverse 2D
versus longitudinal flow stays distinct. Böttcher et al. v2 remains the principal
porous reference, with its integration deferred to Stage 5.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_stream_scaling.py \
  --device cuda:0 --output /tmp/fluxbridge-stream-evidence/scaling-cuda
# Run each memory case in its own process, e.g.:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_stream_scaling.py \
  --device cuda:0 --counts 2048 --chunk-sizes 32 --kernels w2 --repeats 1 \
  --memory-only --output /tmp/fluxbridge-stream-evidence/memory-2048-32
```

**Next executable step:** validate and measure the already supported explicit
`nodes=8` start order for the primary fp32 target, retaining comparison/promotion
and the original precision budgets. A preliminary 128-row P3 CUDA probe accepts
order 8, reduces warm action from 7.22 to 1.89 ms and checked preparation/application
from 50.49 to 23.02 ms, with 2.22e-7 packed difference from order 16. This probe
is not broad acceptance and does not change the default order. Complete the
CPU/CUDA independent fixtures, material checks and order/promotion provenance
before recommending it. Remaining special-form/model ports stay open; this
checkpoint does not complete all independent Stages 0–4.


## Checkpoint — explicit order-8 fp32 tetrahedral rules (23 October)

**Accepted finite option:** `nodes=8` is now validated for the primary fp32
numerical tetrahedral target. The API default remains 16. Two-rule moment and
field convergence, promotion and storage rejection are unchanged, and the
user-accepted analytic-fp64 / numerical-fp32 distinction is retained. Evaluation
uses ordinary fp32 moments/coefficients with explicit fp32 FMA correction
workspace; angular geometry uses fp64. There is no hidden method/precision
fallback and no claim of universal accuracy or positivity.

The accuracy-only study checks **11,880 independent 70-digit channels per
precision**, for W2/W4/W6/cubic, on host-prepared CPU, device-prepared CPU and
actual CUDA. Every case passes with zero rejected/failing channels. Worst
normalized mixed errors are **7.153e-7 fp32** and **3.552e-15 fp64**. All 104 fp32
cases accept order 8; fp64 promotes 96 of 104 to order 16. Thus order 8 is an
fp32 cost option, not a claim that eight nodes meet double precision everywhere.
The targeted degree-nine/jet/field/composed/graph suite passes **153 checks**;
streaming/provenance/layering passes **41 checks** including empty chunks and
explicit requested versus accepted orders. The prior 797-pass broad suite is
source-preserved and is not relabelled as a fresh full regression.

The real one-interaction P3 material study independently contracts frozen
70-digit moments. Numerical packed errors are at most **1.390e-16 fp64** and
**1.544e-6 fp32** for W2/W6; work, force, torque and actual CUDA graph replay
pass. Fp64 promotes to 16, fp32 retains 8. Its launch-bound warm fp32 action
remains about **0.288 ms**, so a smaller rule does not imply a uniform speedup.
The study also fixes an argument-shadowing bug (`nodes` versus FEM nodal
coordinates) exposed by exercising multiple arithmetic modes with `--nodes`.

### Matched P3 batch and complete streamed costs

The scene has 16 tetrahedra, varied queries, one interaction per query, P3
resistance/skeleton/tests, 220 moments and 910 packed outputs. Every timed call
recomputes integrals. GPU measurements are serialized; module/runtime warmup and
static material work are separate. Prepared 128-row fp32 actions take **1.741 /
1.719 ms W2/W6**, checked prepare-plus-apply **22.76 / 22.47 ms**, and complete
static-rebuild-plus-apply **47.97 / 55.28 ms**. Accepted native batch ownership is
**59.44 MB**. All force/work/action/packed gates pass against the same scene's
fp64 reference; worst packed error **7.460e-6**, below the unchanged 1e-5 gate.
At order 16 the earlier matched action was about 7 ms and checked action 46 ms;
these shared-machine measurements are finite implementation costs, not an
isolated quadrature-formula speedup.

| Queries | Chunk | Complete fp32 ms W2 / W6 | Max accepted batch MB |
|---:|---:|---:|---:|
| 128 | 32 | 52.32 / 53.19 | 14.87 |
| 128 | 64 | 31.90 / 32.24 | 29.72 |
| 128 | 128 | 22.05 / 22.26 | 59.44 |
| 512 | 32 | 207.54 / 211.50 | 14.88 |
| 512 | 64 | 129.53 / 131.60 | 29.73 |
| 512 | 128 | 88.57 / 91.53 | 59.44 |
| 2,048 | 32 | 857.25 / 908.33 | 14.88 |
| 2,048 | 64 | 533.99 / 582.89 | 29.73 |
| 2,048 | 128 | 360.64 / 369.98 | 59.45 |

All streamed physical outputs pass against an independently prepared fp64
stream, worst mixed error **6.583e-7**. All chunks accept order 8, with zero
moment/field promotions in this finite scaling scene. A 512-row single batch
now fits; 2,048 rows still trigger the unchanged 1 GiB per-rule cap. Streaming
makes them executable, with explicit host preparation/readback per chunk.

Fresh-process actual CUDA pool high-water usage is **82.47 MB at 128 queries /
chunk 32**, **82.53 MB at 2,048 / chunk 32**, and **290.95 MB at 2,048 / chunk
128**. Current pool usage returns to **12.95 MB** after each driver. These
include comparison allocations and static/runtime arrays; they are not total
CUDA reservation/context/opaque memory. Accepted-batch byte counts exclude
those temporaries. Source hashes and exact measured source are archived for
all 12 accuracy/material/scaling/memory/compiler reports.

The SM120 PTX audit finds **zero fp64 instructions** in fp32 `tet_quadrature`,
`fields`, `evaluate` and `affine`; explicit geometry `device_panels` and
`device_search` retain 1,626 and 254 respectively. Actual CUDA execution and
capture evidence are separate from that compile audit.

Evidence: `/tmp/fluxbridge-order8-evidence/`, including per-report
`measured-source/`, `source-verification.json`, accuracy reports, batch/stream
costs, fresh memory workers, PTX, JUnit and logs. Before changes, the dirty tree
was preserved in `/tmp/fluxbridge-order8-kickoff/` and the resume snapshot
`/tmp/fluxbridge-order8-resume/`. Earlier source-preserved measurements remain
unchanged. Main warpSPH/shared Core/Integrator and unrelated dirty work remain
untouched. Ordinary disks/fibres remain independent of porous material coupling;
2D transverse and longitudinal response remain distinct. Böttcher et al.
(2025), arXiv:2504.07739v2 remains the principal porous-flow reference.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_tetquad_acceptance.py \
  --device cuda:0 --preparation device --nodes 8 --accuracy-only \
  --output /tmp/fluxbridge-order8-evidence/accuracy-cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_batch_scaling.py \
  --device cuda:0 --nodes 8 --counts 32,128 --elements 16 --analytic-max 0 \
  --preparation device --search device --reuse-material \
  --output /tmp/fluxbridge-order8-evidence/batch-cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_stream_scaling.py \
  --device cuda:0 --nodes 8 --chunk-sizes 32,64,128 \
  --output /tmp/fluxbridge-order8-evidence/scaling-cuda
# Fresh process per pool measurement; no reference/full-batch run in this mode:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_stream_scaling.py \
  --device cuda:0 --nodes 8 --counts 2048 --chunk-sizes 128 --kernels w2 \
  --repeats 1 --memory-only --output /tmp/fluxbridge-order8-evidence/memory-2048-128
```

**Next executable gate:** build independent higher-weighted sphere fixtures,
then implement a full request-mask native sphere provider with explicit
quadrature/exactness limits, CPU/CUDA fp32/fp64 and composed consumer checks.
The existing analytic sphere baseline stays available. Retained disk lookup/
series/curvature/SDF ports and remaining assembly comparisons stay open; this
checkpoint does not complete all Stages 0–4.


## Checkpoint — full weighted sphere request masks (23 October)

`SphereShellEvaluator` now exposes exact solid-sphere geometry with an explicit
numerical shell rule, all degree-nine Cartesian moments and kernel-only
derivatives through order two, in native ordinary fp32/fp64. It shares profile,
angular rule and contraction primitives with the planar shell provider, with
separate sphere cuts/cap limits. The existing analytic `RadialEvaluator` stays
baseline-only and unchanged. No curvature approximation or mesh geometry is
substituted; no pressure/porosity integration is introduced.

### Independent fixtures and precision

The reference was implemented **before** the native path. Exact rational
Cartesian angular-cap coefficients use tangent-projector Wick contractions;
radial Laurent coefficients are integrated independently of native quadrature.
Rational sums are combined exactly before 70-digit conversion, with extra
precision for potential near-centre logarithmic cancellation. It cross-checks
**784 baseline channels to 1e-55 mixed error** against the existing independent
radial-shell fixture. See [weighted sphere derivation](derivations/weighted-sphere-shell.md).

`fluxbridge-sphere-v1.json` has 44 cases and **12,320 channels per precision**:
one full 2,200-output degree-nine / second-derivative mask per kernel plus
baseline/high-order sparse masks at centre, near-centre, interior/exterior,
small/large spheres, containment, exact and near tangency, and empty support.
This is finite coverage, not every conditioning/geometry combination.
CPU and actual RTX PRO 6000 CUDA pass W2/W4/W6/cubic at unchanged **1e-5 fp32 /
1e-14 fp64** gates. Worst normalized mixed errors on both devices are
**4.094e-6 / 5.499e-15**. Every case accepts requested order 24, compared to 32,
without promotion, failure or fallback.

Default radial order 24, exactly sufficient angular rules in real arithmetic,
selected-dtype geometry/evaluation and explicit Neumaier accumulation are
reported separately. Wendland's exact angular reduction leaves radial
polynomials of degree at most 32; cubic outer Hessian Laurent terms retain
numerical radial error. Two-order comparison is not a certified global error
or positivity bound. Unresolved rules promote by eight up to 64, then reject.
A requested 64 cannot provide its higher comparison and rejects nonempty work.
Both accepted/comparison rules obey a 1 GiB per-batch cap. Invalid inputs,
unrepresentable scaling and nonfinite physical outputs reject explicitly.
In particular finite normalized values do not permit an overflowing physical
Hessian after multiplying by h^-2.

Focused initial native tests: **24 passed**. The shared-code full regression:
**896 passed, 4 optional legacy-WGSL skips, 32 unchanged analytic-fp32 expected
failures**, 322.64 s. That exact pre-storage-guard source is archived. After the
final physical-output guard, the sphere/plane/layering suite passes **62 checks**
on CPU/actual CUDA, including both-precision overflow regressions, complete
translated-moment derivatives, polynomial work, subsets/repeats, scaled units,
centre covariance, composed `wp.func` calls and frozen CUDA graph replay.
The SM120 pure-fp32 shell module has **zero fp64 instructions**; its fp64 module
has 294. The Python storage guard does not change generated native arithmetic.
Actual CUDA execution and replay are independently tested.

### Complete CUDA costs, with final storage validation

W2 uses requested order 24, comparison 32, arbitrary unit axes and varied query
distances around R/h=0.5. Three warm/complete samples are serialized. Static
profile/rule packing and first checked preparation are separate. Frozen actions
recompute every integral; no completed-integral cache is used.

| Precision | Mask | Queries | Warm ms | Complete ms | Owned native MB |
|---|---|---:|---:|---:|---:|
| fp32 | 28 baseline | 1 | 0.264 | 1.231 | 0.021 |
| fp32 | 28 baseline | 128 | 0.339 | 1.394 | 2.686 |
| fp32 | 2,200 degree-nine jets | 1 | 3.742 | 14.276 | 0.266 |
| fp32 | 2,200 degree-nine jets | 8 | 3.304 | 15.144 | 2.132 |
| fp64 | 28 baseline | 1 | 0.678 | 2.266 | 0.042 |
| fp64 | 28 baseline | 128 | 0.937 | 3.505 | 5.372 |
| fp64 | 2,200 degree-nine jets | 1 | 14.995 | 55.048 | 0.533 |
| fp64 | 2,200 degree-nine jets | 8 | 15.136 | 55.100 | 4.263 |

Complete costs include allocation, rule comparison, normalized and physical
storage checks, native evaluation and readback. In particular preparation
already evaluates for convergence; explicit `evaluate()` also launches the
returned rule before readback. Warm actions exclude that host driver. Owned
bytes exclude the separate higher-order comparison batch, host convergence
arrays, static profiles and allocator reservation. Reports include actual CUDA
pool current usage; those values are not clean-process per-batch high-water or
total GPU memory. The earlier tetrahedral streaming checkpoint supplies separate
fresh-process allocator high-water scaling. This finite sphere benchmark is not
a solver throughput result or an equal-cost comparison to the analytic baseline.

Evidence: `/tmp/fluxbridge-sphere-evidence/` holds CPU/CUDA independent accuracy,
initial and final runtime reports, pure-precision PTX, 24/62-test JUnit, full
regression and exact verified per-report `measured-source/`. The broad suite's
source is `before-storage-guard/source.tar.gz`. Original dirty work is preserved
in `/tmp/fluxbridge-sphere-kickoff/`; the preservation audit still finds all
92 unrelated original dirty files unchanged and only the same five authorized
original files changed. Main warpSPH/shared Core/Integrator remain untouched.
Ordinary solid fibre applications and the native 2D/3D wall explorer remain
independent of this sphere extension. Böttcher et al. v2 remains the principal
porous-flow reference, with pressure/porosity integration reserved for Stage 5.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_sphere_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_sphere_acceptance.py \
  --device cuda:0 --output /tmp/fluxbridge-sphere-evidence/cuda
# Final host-storage validation affects driver timing, not native arithmetic:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_sphere_acceptance.py \
  --device cuda:0 --runtime-only --output /tmp/fluxbridge-sphere-evidence/final-runtime-cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY -m pytest tests/fluxbridge/test_sphere_shell.py \
  tests/fluxbridge/test_planar_shell.py tests/test_package_layering.py -q
```

**Next executable gate:** establish independent higher-weighted 2D disk
fixtures and extend the circle provider beyond its current 15 baseline outputs.
Preserve solid cylinder cross sections/periodic fibres and separate geometric,
integration and series-truncation errors. Then validate the retained small-disk/
small-ball and curvature/SDF models with explicit domains and complete advertised
request masks. Remaining assembly comparisons and model gates keep Stages 0–4
open; Stage 5 is not started.


## Checkpoint — full weighted 2D disks and periodic solid fibres (23 October)

`WeightedDiskEvaluator` now supplies 2D exact-circle **volume geometry** with
explicit numerical integration, every degree-nine Cartesian moment and
kernel-only derivative through order two. `DiskArrayEvaluator(...,
method="volume_quadrature")` adds whole-disk host/CUDA search, deterministic
periodic images/reduction and the existing bounded streaming interface. The
baseline-only circle evaluator and array default remain unchanged. Resolved
solid disks have no porous material law; 2D transverse fibre flow is distinct
from longitudinal flow and from an effective anisotropic permeability model.

### Reference and native arithmetic

Before the native port, the independent 70-digit reference integrated regular
flux over **all** circle arcs, including exterior tails, using exact rational
cumulative radial primitives. It cross-checks **540 baseline channels at 1e-50
mixed error** against the existing independent volume-shell reference. Native
evaluation instead uses interior radial/angular volume nodes. See the
[weighted disk derivation](derivations/weighted-disk-volume.md).

The fixture set has **2,724 channels per precision**, including a full 330-output
degree-nine/second-derivative mask per kernel, sparse higher products, centres,
near-centres, near/exact boundary queries, small/large circles, full support and
near/exact tangency. Additional exact whole-support rational polynomial moments
and kernel integration-by-parts identities check **every** request through
 degree nine, at physical support 0.7. These are finite fixtures, not every
possible radius/conditioning combination.

The default start order 24 passes CPU and actual RTX PRO 6000 CUDA for
W2/W4/W6/cubic at **1e-5 fp32 / 1e-14 fp64** budgets. Worst normalized mixed errors
are **1.094e-6 fp32 / 3.402e-15 fp64**, without promotions/rejections. Explicit
`nodes=8` passes the same fp32 CPU/CUDA fixture gate, worst **1.898e-6**;
37 of 40 cases accept order 8 and three promote to 16. No precision or method
fallback occurs; the default remains 24. Control compares normalized requested
moments, promotes by eight to 64 or rejects; two-order agreement is not a global
error/positivity certificate. A nonempty request starting at 64 cannot provide
a higher comparison rule and rejects.

Geometry preparation is **host fp64**, then normalized radial/angle data is
stored and evaluated in the selected precision. Full shells use radial Gauss;
partial shells use a circle-boundary-angle change of variable to remove cap
endpoint roots. Geometric grading adds integration panels without moving or
snapping the actual circle/distance. Native sampling and contraction are
reusable `wp.func` calls with caller-owned Plan/Rules/scratch. Batches parallelize
samples and requested outputs, with fixed-order selected-dtype Neumaier sums.
The explicit conditioning policy is R/h in [1e-2,1e4] fp32 / [1e-4,1e4] fp64;
reference cases span 0.02..100. Each rule has a 1 GiB owned-batch cap; invalid
geometry, unsupported scaling, nonfinite physical primitives/sums and unresolved
convergence reject. Streaming is explicit, not automatic fallback.

The first PTX audit exposed six fp64 instructions in libdevice's generic float
sin/cos large-angle reduction, even though prepared angles stay within +/-pi.
The final fp32 path uses bounded degree-13/14 sine/cosine polynomials and
explicit fp32 FMAs after reducing to +/-pi/2. Its real-arithmetic remainders are
below 7e-10 / 7e-11; actual floating-point error is checked by the fixtures.
The final SM120 fp32 evaluator has **zero fp64 instructions**; fp64 has 209.
The initial compile/audit failures and initial measured source remain archived.
No numerical budget was loosened. This generated-code audit is separate from
actual CUDA execution and graph replay.

### Consumers and complete measured CUDA costs

The final dependent suite passes **179 tests**, with the same **eight expected
analytic raw-fp32 rectangle misses** from the user-deferred family. It covers
weighted disks, old circle/box paths, streaming, device search and layering.
This is a scoped regression, not a relabelling of the older 896-pass broad
checkpoint. After correcting comparison-sample provenance, all **40 weighted
checks** pass again on CPU/CUDA, including units, copied inputs, repeated/subset
outputs, exact polynomial reproduction, complete translated-moment derivatives,
composed calls, periodic host/device search versus explicit image sums and
exhaustive enumeration, physical overflow rejection and frozen CUDA graphs.

W2 workloads have varied query distances/normals, h=1, R=0.5. The array scene
uses 16 separated disks and one interaction per query. Static profile/BVH
construction and first preparation/module loading are separate; each timed
call recomputes integrals. Three complete samples include host geometry
preparation, both convergence rules, allocation, storage validation and readback;
array costs also include adjacency and deterministic reduction.

| Precision / start order | Workload | Queries | Warm ms | Complete ms | Accepted native MB |
|---|---|---:|---:|---:|---:|
| fp32 / 24 | 15 baseline outputs | 128 | 2.358 | 19.394 | 7.775 |
| fp32 / 8 | 15 baseline outputs | 128 | 0.498 | 14.313 | 1.765 |
| fp32 / 24 | 330 weighted outputs | 1 | 1.787 | 5.041 | 0.087 |
| fp32 / 8 | 330 weighted outputs | 1 | 0.574 | 2.735 | 0.031 |
| fp32 / 24 | 330 weighted outputs, array | 128 | 7.269 | 32.446 | 15.815 |
| fp32 / 8 | 330 weighted outputs, array | 128 | 2.181 | 21.493 | 5.612 |
| fp32 / 24 | 330 weighted outputs, array | 512 | 8.423 | 74.155 | 63.514 |
| fp32 / 8 | 330 weighted outputs, array | 512 | 2.700 | 60.972 | 22.531 |
| fp64 / 24 | 330 weighted outputs, array | 128 | 35.142 | 98.296 | 31.577 |
| fp64 / 24 | 330 weighted outputs, array | 512 | 68.892 | 220.062 | 126.813 |

Host geometry preparation and synchronization remain material costs, especially
for changing small masks. These costs are not production particle throughput
and do not certify equal-accuracy superiority over the existing baseline circle
path. Static fields/curved FEM interpolation, shape derivatives and a solved
flow response are separate contracts.

Fresh-process fp32 order-8 array workers retain no accepted batch while timing
complete calls, and run no fp64 reference in that process. Full-batch CUDA pool
high-water is **16.01 MB at 128 / 256.98 MB at 2,048**. Explicit chunk-128
streaming holds it to **16.01 / 16.87 MB** respectively, with **5.61 / 5.90 MB**
maximum accepted native batch. The complete streamed costs are **23.88 / 379.09
ms**, versus **21.58 / 225.07 ms** for the full batches. Current pool usage
returns to **19,373 bytes** of static/runtime arrays after each driver. These
are application-pool usage, not total GPU context/reservation/opaque memory.
Owned batch bytes exclude comparison storage, static profiles/BVH and host
geometry/convergence/output arrays. Both accepted and comparison rule sample
counts now propagate through array/stream provenance; the initial zero comparison
count was a reporting omission, not a skipped convergence rule.

Evidence is in `/tmp/fluxbridge-weighted-disk-evidence/`: exact per-report
`measured-source/`, source verification, 28/40/179-test JUnit/log, independent
accuracy, direct/array CUDA timings, fresh full/stream pool workers and PTX.
The pre-port dirty snapshot is `/tmp/fluxbridge-weighted-disk-kickoff/`.
Preservation still verifies the same 92 unrelated original dirty files and only
five authorized original-file changes. Main warpSPH/shared Core/Integrator and
the inspected native 2D/3D wall explorer remain unchanged. Böttcher et al.
(2025), arXiv:2504.07739v2 remains the principal porous-flow reference; pressure
and porosity integration remain Stage 5.

```bash
PY=/home/lu26029/miniconda3/envs/warp/bin/python
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/make_fluxbridge_weighted_disk_fixtures.py
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_weighted_disk_acceptance.py \
  --device cuda:0 --output /tmp/fluxbridge-weighted-disk-evidence/final-cuda
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_weighted_disk_acceptance.py \
  --device cuda:0 --nodes 8 --precisions float32 \
  --output /tmp/fluxbridge-weighted-disk-evidence/final-cuda-order8
# Run memory cases in separate fresh processes, e.g.:
XDG_CACHE_HOME=/tmp/fluxbridge-cache $PY scripts/studies/fluxbridge_weighted_disk_acceptance.py \
  --device cuda:0 --nodes 8 --precisions float32 --memory-only \
  --memory-count 2048 --memory-scope array --memory-stream-chunk 128 \
  --output /tmp/fluxbridge-weighted-disk-evidence/provenance-memory-2048-stream
```

**Next library model gate:** implement and validate full-contract small-disk/
small-ball mean-value series in their smooth single-kernel-piece domain, starting
with independent circle/sphere references. Keep truncation separate from
geometry and arithmetic error, reject out-of-domain inputs and retain the
ordinary exact-circle/mesh routes. Disk/box lookup and retained curvature/SDF
models still need their own full advertised contracts. Stages 0–4 remain open;
Stage 5 has not started.

## Browser fp32 port — authorized 23 October

The mixed-wall explorer must recompute on static GitHub Pages without a local
Python/Warp service. Browser shader fp64 is outside this port's scope. Reuse the
existing 2D/3D browser workers, Gauss rules, dispatch/resource patterns and static
asset export. Preserve the native provider as an explicitly selected reference.
This web milestone now precedes the next library model gate above.

1. [x] Establish a reusable browser request/profile contract: Cartesian moments
   through degree nine, kernel-only derivatives through order two, explicit
   physical scaling; generate conditioned coefficients from the existing exact
   native plans rather than inventing separate kernel formulas.
2. [x] Port conditioned 2D polygon integration to WGSL fp32, with JavaScript
   Number geometry/panel preparation, conservative whole-element support search,
   retained exterior regular flux, deterministic reductions and bounded batches.
   Validate all four kernels against native and independent fixtures before
   calling the 2D core accepted.
3. [x] Connect live 2D mixed-wall geometry and prescribed-state contractions to
   that core, retaining ordinary disks/fibre chord geometry, native snapshots,
   imports/exports and a clearly identified backend. Contractions may initially
   use JavaScript Number; no claim that these host operations run in fp32.
4. [x] Port 3D tetrahedral angular preparation/integration and the same material
   terms. Preserve near-plane regular-potential treatment, explicit quadrature
   refinement/rejection and correct area versus volume units.
5. [x] Move preparation off the UI thread; add cancellation/device-loss handling,
   bounded adapter-aware buffers and static relative asset loading. No silent
   native/precision fallback. Keep old browser viewers usable.
6. [ ] Validate actual browser GPU execution in 2D/3D, fixture/native parity,
   displayed terms, static deployment under a repository URL prefix, complete
   preparation/readback timings and memory/scaling. Hardware/software adapters
   and shader compilation checks must be reported separately.

The first implementation gate is the reusable 2D numerical core and its native
fixture comparisons, followed immediately by the wall frontend adapter. Full
parity with every native assembly/curved-body API is a later explicit scope;
browser pressure/porosity solvers remain excluded (Stage 5). Retain Böttcher
et al. (2025), arXiv:2504.07739v2, and transverse/longitudinal fibre distinctions.


## Checkpoint — browser fp32 core and wall adapter (23 October)

The first browser port is implemented, with a shared degree-nine / DS≤2 volume
request/profile core, polygon and tetrahedron preparation, static exact-rational
profile export, worker-based fp32 WebGPU dispatch, host Number regional/material
contractions, 2D/3D mixed walls and existing solid chord disks/fibres. Native
provider selection and snapshots remain explicit. Browser support/refinement
comparisons are wired; static exports use relative worker/shader/profile URLs
and require no native compute API. Assets-only export does not import Warp.

`tests/visualizer/test_fluxbridge_browser.py` passes **16 checks**, with two
explicitly skipped unsupported 3D disk/fibre geometry cases. The production
WGSL/JS layout executes through Dawn/Vulkan on **NVIDIA RTX PRO 6000 Blackwell**:
**88 independent cases / 12,300 values**, all four kernels, both dimensions,
full degree-nine masks, kernel derivatives through order two, face/near-face/
edge/sliver/tangent cases, worst mixed difference **7.277e-7** at the unchanged
**1e-5** budget. Five live wall cases (four 2D kernels, 3D W6) match every native
fp64 displayed term at 2e-5 absolute/relative tolerances, including degree-three
resistance, affine velocities and rotated tensors. Separate host-only algebra
checks compare identical arbitrary moment arrays to native contractions.

The first software Vulkan run passed, but the initial NVIDIA implementation
failed one 3D fixture at 2.44e-5 and then failed convergence in a wall case.
NVIDIA shader optimization invalidated floating-point Neumaier compensation.
The final shader uses ten 32-bit integer limbs per sign to accumulate rounded
fp32 products, then cancels exactly and converts back to fp32. This is ordinary
fp32 profile/product evaluation with integer summation, not shader fp64 or a
hidden precision/method fallback. Adversarial large cancellation and positive/
negative sums pass on the hardware adapter. The failed run and original shader
are retained in the evidence directory; no tolerance was relaxed.

Existing native wall/server/layering regressions pass **62 tests**. All three
existing JS test files pass, including Dawn/Tint compilation of the additional
shader. No solver or shared Core/Integrator source changed. Original dirty
files were archived before edits in `/tmp/fluxbridge-webgpu-kickoff/`.

**Not yet a completed browser delivery gate:** the Codex in-app browser's worker
reports **no WebGPU compute adapter**. This is separate from successful actual
NVIDIA execution through Dawn. The static site loaded under `/fluxbridge/` and
correctly reported the unavailable adapter without calling a native API. An
actual WebGPU-capable browser still needs live worker/UI checks, repeated edits,
export/reopen, cancellation/device loss and compatibility checks. Other remaining
work includes end-to-end scaling/performance acceptance, reducing repeated
profile evaluation and query-specific host preparation, and broader material/
geometry adversarial cases. The implemented comparison UI needs its own live
browser verification. Do not mark Stage 4 or the whole web milestone complete
on shader compilation, software-adapter tests or native-node execution alone.

Evidence: `/tmp/fluxbridge-webgpu-evidence/` contains initial software/core
reports, initial hardware failures, final hardware JUnit/log, independent error
report, native regressions and legacy JS shader compilation checks. The native
library's special-form/retained-model gates above remain open; Stage 5 is still
excluded. Böttcher et al. v2 and transverse/longitudinal flow distinctions remain.


### First end-to-end browser-core timing and regional assembly

The initial complete fp32 degree-three W2 runs on NVIDIA measured **115.6 ms
for 81 2D probes / 8 triangles**, but **20,116 ms for 25 3D probes / 24 tets**.
These measurements include both order-16 and order-24 rules, host preparation,
allocation/upload, GPU evaluation/readback and material/result assembly. Device/
pipeline creation is excluded from these injected-device Node/Dawn timings;
actual cold browser startup remains unmeasured.

The wall adapter now cancels exactly shared oriented edges/triangles **within
each region**, where the global polynomial integrand has identical traces.
It never cancels across the solid/permeable interface. This optimization is
specific to this adapter's common regional integrand, not a license to cancel
faces of arbitrary discontinuous native material fields. Whole-element support
selection and displayed selected element IDs remain unchanged. Every remaining
face participates, including regular-flux exterior tails; bounded face groups
are integrated and reduced deterministically. The first attempt to treat an
entire region as one upload exceeded the buffer cap and was corrected by face
batching; its failed test is retained.

The same workloads then measured **107.6 ms (2D)** and **6,697.5 ms (3D)**.
3D sample counts fell from **18.20 million to 6.76 million** across both rules.
The merged 3D run spends approximately **3,121 ms preparing** and **3,566 ms
in GPU dispatch/readback**; its largest owned GPU buffer set is **7.82 MB**.
The merged 2D peak is **0.39 MB**. These are single complete runs, not robust
throughput distributions or CPU/CUDA formula comparisons. The GPU figures are
logical application buffer totals, excluding driver/context memory and host
arrays. Accepted mixed order differences are 1.14e-7 (2D) / 1.16e-7 (3D).

The expanded 16-pass hardware suite also covers browser-only exhaustive/support
and fixed-geometry refinement comparisons in both dimensions, resolved solid
disks/fibres, and disabling the porous provider. Exact region closure and the
absence of effective drag for resolved fibres remain intact. The static site
under `/fluxbridge/wall.html` was visually inspected with 2D compression and
3D local-response snapshots computed by the production WGSL on NVIDIA. The
in-app browser displays the correct probe terms, volume units and provenance;
this snapshot inspection is **not** live browser GPU execution.

**Next executable web gate:** cache/deduplicate shared radial profile evaluation
and benchmark controlled lower starting quadrature orders against the existing
independent fixtures; preserve convergence checks and the 1e-5 accuracy budget.
Then validate live worker computation and interaction in a WebGPU-capable
browser. Current 3D preparation/readback cost is still too high to declare the
web performance gate complete. Broader scaling/memory distributions, browser
compatibility and cold-start measurements remain open. Native special-form
work resumes after this browser milestone; Stage 5 remains excluded.


## Publishing checkpoint — GitHub Pages (23 October)

At the user's explicit request, the static mixed-wall page and browser compute
assets were committed and pushed to the existing `wi-re/warpSPHBoundaries`
`gh-pages` branch, **ed350f22a9c00465d1bec27c79e1cae7c2dde1ba**. The original
`curvatureBoundaries` repository URL redirects to that canonical repository.
An isolated checkout `/tmp/fluxbridge-pages-publish-20261023` kept the research
checkout and its unrelated dirty files unstaged and unchanged.

The publishing diff contains 13 files: the new wall/browser-core assets, a
verified NVIDIA-computed 2D default snapshot, and navigation links in the
existing 2D/3D viewer HTML. Existing viewer computation assets are byte-identical
to the previous deployment. Relative asset links, JSON and JS syntax were
checked before pushing. The deployment uses the existing Pages source and HTTPS
configuration; no permissions/settings were expanded. URL:
https://fluids.dev/warpSPHBoundaries/wall.html .

The user-facing page defaults to browser WebGPU fp32; the explicit native
provider still needs a local compute service. Static snapshots remain usable
when an adapter is unavailable. Publishing does not close the live-browser GPU
validation or 3D performance gates above, and does not enter Stage 5.
Publishing hashes are in `/tmp/fluxbridge-webgpu-evidence/publish-manifest.json`.


GitHub Pages build/deploy workflow **39967389458** completed successfully.
Public HTTPS checks returned 200 for **all 13 publishing files**, each matching
its pushed SHA-256 hash. The live page was opened and visually inspected; its
initial snapshot, controls, region occupancy and closure rendered correctly.
Evidence: `publish-live-verification.json` and `published-wall.png` in the same
local evidence directory. Live WebGPU execution in this in-app browser remains
unavailable, as documented above; publication does not imply that gate passed.


### Browser adapter discovery follow-up

The user also reports no available adapter in Chrome. Read-only host inspection
finds installed Chrome 156.0.8078.47, X11 session and NVIDIA driver 610.43.02.
GPUWeb currently lists NVIDIA Linux default support on Wayland (Chrome 147+),
with X11 behind the explicit experimental Vulkan/WebGPU launch flags. This is a
likely platform gate, not yet a confirmed diagnosis from `chrome://gpu`.
The user's Chrome is not connected to the browser tools; its WebGPU/Vulkan and
Problems Detected report was requested. An isolated-profile launch recipe and
primary sources are recorded in `docs/visualizer.md`; no Chrome settings,
blocklist/security flags or desktop session were changed automatically. Live
browser GPU acceptance remains open until adapter discovery and worker compute
are actually verified in that browser.


### Adapter follow-up — successful user Chrome run and compatibility discovery

The user supplied a Chrome feature report with WebGPU/interoperability hardware
accelerated but Vulkan disabled, then confirmed that the isolated Chrome test
instance with the documented Vulkan/WebGPU flags **works**. This is direct user
evidence of successful operation in Chrome with that launch configuration. It
does not establish complete 2D/3D browser accuracy/performance or that normal
Chrome needs any particular one of those flags; their individual effects have
not been isolated. No normal Chrome profile/settings were changed by the agent.

The application now tries core requests with high-performance/default selection,
then compatibility requests with both preferences. Selection and feature level
are recorded in result provenance; fp32 stays explicit and no CPU/native fallback
is introduced. A static `gpu-check.html` diagnoses all four requests separately
in the page and a dedicated worker, with bounded diagnostic waits and no settings
changes or external reporting. The wall page links to it. This avoids treating
Chrome's feature-status summary as proof of usable page/worker adapter discovery.

Independent native Dawn/OpenGL ES probing on this machine reproduces an important
case: core requests return null, compatibility requests expose the NVIDIA RTX
PRO 6000 (OpenGL ES 3.2, driver 610.43.02), and the production shader compiles.
This is a native backend comparison, not proof that normal Chrome exposes that
same adapter. In the in-app browser all four requests remain unavailable in both
page and worker. This distinction is visible in its diagnostic report.


The discovery update passes both NVIDIA **OpenGL ES compatibility** and **Vulkan
core** suites: **16 checks / two unsupported-geometry skips per backend**, each
including all **12,300 independent values**, worst mixed difference **7.277e-7**
at the unchanged 1e-5 budget, material diagnostics, support/refinement comparisons
and exact integer-reduction cancellation checks. Selector regressions cover
preference rejection, compatibility-only devices, unsupported options and absent
APIs. The existing JS shader compilation/math suites pass. Arithmetic/shader
source is unchanged by this adapter-discovery update. Evidence lives in
`/tmp/fluxbridge-adapter-evidence/`; the pre-change source was archived in
`/tmp/fluxbridge-adapter-kickoff/`.


The adapter-selection/diagnostics update was published as Pages commit
**a9a0b3a9a04c0b7c6a4231b1219c4710de2cd49b**. Workflow **39971795236** succeeded;
all seven changed/new HTTPS assets returned 200 and matched their published
hashes. Diagnostics URL: https://fluids.dev/warpSPHBoundaries/gpu-check.html .
This complements the user's confirmed working isolated Chrome profile; it does
not claim their normal Chrome profile is fixed. Unrelated dirty files remain
preserved, and no warpSPH/shared Core/Integrator changes occurred.


## Checkpoint — native per-term interaction throughput (26 October)

In response to the user's performance question, a new serialized fp32 CUDA
study measures seven separate requested-term bundles for W2/W6. The benchmark
is `scripts/studies/fluxbridge_term_throughput.py`; it changes no operator,
solver, browser or precision policy. GPU: RTX PRO 6000 Blackwell Workstation,
95 GiB, SM120, Warp 1.17.0, toolkit 12.9 / driver CUDA 13.3.

One interaction is **one particle–volume-element pair**, independently of how
many requested outputs are produced. The scene has 16 separated triangles or
tetrahedra (vertices 0 and twice the unit axes, translated by four units),
varied queries around local coordinate .2 and h=1; every query selects exactly
one volume. The query scene represents interior material/boundary queries;
ordinary exterior solid-wall queries, slivers, grazing support and overlaps may
have different cost. This is an operator benchmark, not a solved flow timestep.

The first run covers 56 rows: 2D batches of 128/2,048 at order 16 and 3D batches
of 128 at orders 16/8, seven masks, W2/W6. A second 14-row run measures 512-pair
3D batches at explicit order 8. Every lower-order 3D batch accepts eight nodes
under the existing normalized two-rule 1e-5 budget; the API default remains 16.
All subset masks agree exactly with the corresponding fp32 fused baseline;
sampled fp64 numerical references agree within **1.648e-7** mixed error. These
finite sanity checks supplement the earlier independent native fixtures; they
are not new independent proofs of all geometry/precision combinations.

Prepared CUDA rates below use five-sample median CUDA-event elapsed time for
the whole multi-kernel native action, including pair-to-particle reduction.
Every action recomputes profiles/integrals; there is no completed-integral cache.
Buffers, adjacency and query-dependent angular preparation are reused. Event
intervals can contain host-submission gaps; they are not isolated instruction
throughput. Separately recorded host wall timings include submission/sync.
Complete rates use three-sample median wall time: static geometry/plan/BVH are
reused, while host validation, support search with readback, allocation/upload,
3D convergence preparation/evaluations, native action, reduction and readback
are included. Static construction/JIT are separate. Measurements are serialized
under `/tmp/fluxbridge-gpu.lock`; other desktop/context activity is not isolated.

The table uses **W6 fp32**, **2,048 2D pairs / order 16**, and **512 3D pairs /
explicit order 8**. M=millions, k=thousands of interactions per second. Different
batch sizes/rules mean these columns are not an intrinsic dimensional-cost ratio.

| Term bundle | 2D prepared CUDA /s | 2D complete /s | 3D prepared CUDA /s | 3D complete /s |
|---|---:|---:|---:|---:|
| Occupancy / kernel density | 5.214 M | 1.215 M | 1.828 M | 28.1 k |
| Kernel-gradient vector | 4.786 M | 1.277 M | 1.836 M | 28.5 k |
| First + second moments | 1.845 M | 989.2 k | 1.518 M | 28.2 k |
| First gradient moments | 3.566 M | 1.036 M | 1.816 M | 28.1 k |
| Symmetric Hessian | 3.694 M | 1.021 M | 1.507 M | 28.0 k |
| Diagonal Hessian (Laplacian inputs) | 3.994 M | 1.205 M | 1.526 M | 26.9 k |
| All baseline outputs (15 / 28) | 690.7 k | 438.3 k | 596.4 k | 23.2 k |

The Laplacian-input row returns only diagonal Hessian entries; the final trace
addition is not timed. Gradient rows produce all vector components, and moment/
Hessian rows produce every component in the stated bundle. Rates are not scalar
output counts per second, and separately timed masks are not additive costs of
a fused consumer. Compression/drag/dissipation/local mobility are not each
separately isolated by this study; they contract combinations of these moments.
The earlier real 3D P3 material action supplies a separate complete consumer
benchmark (220 moments / 910 outputs): approximately **74k prepared host-observed
interactions/s / 5.7k complete interactions/s** at 128 pairs and order 8, W2/W6.

For the table's full baseline, host-observed prepared rates are **672k/s (2D)**
and **501k/s (3D)**, versus the event rates above. At the **default 3D order 16**
and 128 pairs, W6 measures **147k/s CUDA-event**, **146k/s host-observed prepared**
and **6.33k/s complete**. At the same 128 pairs, explicit order 8 gives **507k/s
event / 495k/s host-observed / 11.0k/s complete**. Large 2D batches substantially
improve utilization: its 128-pair full baseline is only 88.7k/s host-observed,
versus 672k/s at 2,048. Do not extrapolate the small-batch numbers to saturation.

For changing particles, **3D preparation/convergence remains the main bottleneck**.
The 512-pair order-8 full baseline takes 1.023 ms host-observed prepared versus
22.046 ms complete, owning 192.95 MB accepted native storage (the higher-order
comparison rule, static arrays, allocator reservation and host data are separate).
2D full-baseline costs are 3.049 ms prepared / 4.673 ms complete at 2,048 pairs.
No solver, warpSPH/shared Core/Integrator or Stage 5 integration was changed.

Evidence: `/tmp/fluxbridge-term-throughput-20261026/report.json` and
`large-3d/report.json`, logs and exact source verification. The first report's
benchmark source predates the optional large-batch CLI argument and is retained
as `measured-script.py`; all native source hashes match. Both timed runs passed.
Reproduce from the Warp environment:

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/studies/fluxbridge_term_throughput.py --output /tmp/fluxbridge-term-throughput-repeat
XDG_CACHE_HOME=/tmp/fluxbridge-cache /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/studies/fluxbridge_term_throughput.py --large-3d --output /tmp/fluxbridge-term-throughput-large-repeat
```

The next performance step remains reducing query-dependent preparation and
storage and comparing fused/specialized schedules. Browser performance is a
separate backend and is not used for these native Warp rates. Remaining model/
fixture gates and Stage 4 acceptance remain open; Stage 5 is excluded.

## Checkpoint — measured CUDA occupancy, FLOPs and large-scale behavior (26 October)

The user's requested hardware investigation is complete for the seven native
W6 fp32 volume-term bundles. This checkpoint supersedes small-batch throughput
as a saturation estimate. No operator implementation, precision budget, memory
cap, solver or browser backend was changed. The dirty tree was inspected and
the pre-investigation tracked patch, plan and previous benchmark were preserved
in `/tmp/fluxbridge-hardware-20261026/`. All timed GPU actions and profiler
targets serialize on `/tmp/fluxbridge-gpu.lock`.

Reproducible drivers:
`scripts/studies/fluxbridge_hardware_scaling.py` and
`scripts/studies/fluxbridge_hardware_report.py`. Durable selected counters,
complete scaling samples, source hashes and timeline summaries are saved in
`docs/work/fluxbridge-hardware-20261026.json`. Raw Nsight reports, CSV exports,
SQLite timelines, logs, exact profiler commands and separate host profiles are
in `/tmp/fluxbridge-hardware-20261026/`.

### Hardware-counter method and scope

- GPU: RTX PRO 6000 Blackwell Workstation, 188 SMs, SM120, driver 610.43.02;
  Warp 1.17.0. Nsight Compute **2026.3.1** collected actual hardware counters;
  Nsight Systems **2026.3.1** recorded CUDA timelines. No administrator-setting
  change or root execution was needed, despite the driver's restricted-counter
  setting. Counter acquisition succeeded; values below are not estimates from
  source or `nvidia-smi` activity percentages.
- Same separated-simplex scene as the previous checkpoint: 16 volumes, varied
  interior queries, h=1, exactly one selected particle–element pair per query.
  2D uses conditioned boundary order 16; 3D uses explicit tetrahedral order 8
  with the existing order-16 comparison and unchanged 1e-5 acceptance budget.
  These are integrated volume interactions, not sampled point-wall interactions.
- Unprofiled timing covers **49 2D cases**, N=128 through 524,288, five samples
  each; **42 3D cases**, N=128 through 32,768, three samples each. Graph replay,
  instrumented individual-operation events, and complete host-driven evaluation
  are recorded separately. Static geometry/plan compilation and JIT are excluded
  from steady timings. Complete calls include all ordinary validation, search,
  preparation, convergence evaluations, allocation, reductions and readback.
  Scaling uses the explicit streaming driver: one chunk equal to N in 2D and
  chunks of at most 512 in 3D, including its host snapshots/output allocation.
  Nsight's complete profile captures one direct ordinary `evaluate` call;
  its duty fraction uses that profiled region's own duration.
- Nsight profiles all seven bundles at N=131,072 (2D) / 512 (3D), the small
  N=128 baseline in each dimension, and the full 3D preparation/evaluation action.
  Additional memory and sustained-peak profiles supplement those 17 cases.
  Collection is restricted by CUDA profiler start/stop after warmup. Kernel replay
  takes multiple passes; its duration is not substituted for unprofiled timing.
  GPU clocks/caches are left uncontrolled to avoid changing user/device state;
  unrelated desktop activity is a measurement limitation.
- Counted FLOPs are predicated-on **FADD + FMUL + 2*FFMA**, or their double
  equivalents. Integer/address work, division, sqrt and transcendental operations
  are excluded, not treated as free work in the bottleneck interpretation.
  All prepared evaluation kernels report zero counted fp64 arithmetic; the
  separate 3D geometry preparation explicitly uses fp64.
- Achieved occupancy uses Nsight's active-warps metric. FP32 utilization below
  is counted FLOP/s divided by sustained FMA peak at each kernel's measured clock.
  The measured peak basis is 24,064 FP32 FMA instructions/cycle versus 376 FP64,
  confirming the 64:1 ratio. Nsight SM throughput is a different metric: the
  busiest contributing SM pipeline, not percentage of fp32 peak.

### Achieved occupancy and arithmetic throughput by term

The table selects the main integration kernel: the complete 2D pair kernel and
the 3D `integrate_tiles` kernel. Additional profile preparation, row reductions,
output scaling and particle reduction remain in the stored report and full
prepared timings. TF/s counts actual executed fp32 arithmetic including the
accuracy-preserving product/summation work.

| Bundle | 2D occupancy | 2D TF/s | 2D fp32 peak | 3D occupancy | 3D TF/s | 3D fp32 peak |
|---|---:|---:|---:|---:|---:|---:|
| Occupancy / kernel density | 31.92% | 8.870 | 6.51% | 16.23% | 0.228 | 0.17% |
| Gradient vector | 31.66% | 7.606 | 5.59% | 25.82% | 0.709 | 0.52% |
| First + second moments | 30.90% | 3.640 | 2.66% | 74.98% | 2.139 | 1.59% |
| First gradient moments | 31.03% | 3.863 | 2.82% | 74.70% | 2.252 | 1.67% |
| Symmetric Hessian | 30.83% | 6.112 | 4.48% | 50.37% | 1.879 | 1.40% |
| Diagonal Hessian inputs | 31.00% | 7.907 | 5.82% | 25.99% | 1.061 | 0.78% |
| Full baseline, 15 / 28 outputs | 31.54% | 2.316 | 1.69% | 71.80% | 3.212 | 2.46% |

SM throughput / maximum memory-system throughput, for the same kernels:

| Bundle | 2D SM | 2D memory | 3D SM | 3D memory |
|---|---:|---:|---:|---:|
| Occupancy | 33.16% | 71.33% | 2.68% | 36.71% |
| Gradient | 30.28% | 76.48% | 12.09% | 32.56% |
| First + second | 17.61% | 78.11% | 35.13% | 31.48% |
| First gradient | 17.19% | 78.21% | 36.04% | 31.36% |
| Hessian | 26.61% | 75.70% | 33.86% | 38.56% |
| Diagonal Hessian | 32.86% | 74.00% | 15.62% | 30.63% |
| Baseline | 15.86% | 76.78% | 70.87% | 50.63% |

The memory column is the busiest contributing memory pipeline/cache throughput,
not DRAM bandwidth utilization. Separate baseline memory profiles measure about
**9.26 GB/s DRAM (2D)** and **273.9 GB/s (3D)**. Missing/unsupported metric names
are not interpreted as zero bandwidth.

### Measured scaling

Host-observed prepared timings, with query-dependent angular preparation reused
and profiles/integrals recomputed on every action, for the full baseline:

| Dimension | Interactions | Prepared ms | Prepared pairs/s | Complete ms | Complete pairs/s |
|---|---:|---:|---:|---:|---:|
| 2D | 128 | 1.544 | 82.9 k | 2.408 | 53.1 k |
| 2D | 2,048 | 3.152 | 649.7 k | 4.143 | 494.3 k |
| 2D | 8,192 | 3.271 | 2.504 M | 5.590 | 1.465 M |
| 2D | 32,768 | 3.529 | 9.285 M | 11.248 | 2.913 M |
| 2D | 131,072 | 11.579 | 11.320 M | 45.284 | 2.894 M |
| 2D | 524,288 | 42.203 | 12.423 M | 206.343 | 2.541 M |
| 3D | 128 | 0.261 | 490.9 k | 11.627 | 11.0 k |
| 3D | 512 | 0.882 | 580.4 k | 21.063 | 24.3 k |
| 3D | 2,048 | — | — | 81.193 | 25.2 k |
| 3D | 8,192 | — | — | 321.694 | 25.5 k |
| 3D | 32,768 | — | — | 1,295.249 | 25.3 k |

3D single-batch preparation rejects **1,024 and larger** at the existing 1 GiB
per-rule cap during higher-order comparison. Large complete runs explicitly use
512-query chunks, with no relaxed limit or silent precision/method fallback.
There is no measured large single-batch prepared rate for those rows. A 512-pair
accepted baseline owns **192,952,112 bytes** and uses **6,462,472 angular samples**;
its convergence comparison uses **25,859,392 samples**. Chunking bounds accepted
storage but does not improve simultaneous work occupancy beyond the chunk size.

Log-log least-squares fits of time versus pair count give empirical exponents,
not asymptotic proofs. From 32,768 to 524,288, 2D baseline prepared time scales as
N^0.895 and complete time as N^1.049. Across all seven bundles the ranges are
0.826–0.955 prepared and 1.023–1.073 complete. From 512 to 32,768, 3D complete
baseline scales as N^0.993 (all masks 0.981–1.117), consistent with approximately
linear fixed-chunk work. Geometry count and neighbors per query remain fixed;
scaling with increasing local wall valence/tessellation is not established by
this experiment.

At 524,288 pairs the 2D occupancy-only prepared kernel action reaches **118.3 M
pairs/s**, gradient **92.5 M/s**, Hessian **65.4 M/s**, and baseline **12.4 M/s**.
Complete rates are only **4.26 M/s**, **4.34 M/s**, **4.15 M/s** and **2.54 M/s**,
respectively. At 32,768 pairs, 3D complete rates remain roughly 21–31 k/s across
all masks. More particles fix the small 2D launch underfill; they do not remove
the 3D preparation bottleneck.

### Bottlenecks established by the measurements

1. **2D register pressure and local-array access.** The kernel uses 96 registers
   per thread, allowing only two 256-thread blocks per SM, a 33.3% theoretical
   occupancy ceiling. Achieved occupancy is about 31%. The 128-pair case launches
   only one block and achieves 8.32% occupancy / 0.12% SM throughput. At the large
   baseline, sampled non-issued stall reasons are about 64.9% long scoreboard
   and 19.9% local/global instruction throttle. A separate memory profile records
   240.5 M local-load sectors and 145.7 M local-store sectors, each 32 bytes, with
   96.49% L1 and 99.84% L2 hit rates. Nsight's register-spilling count is **zero**:
   this is local-array traffic, not evidence of compiler register spills.
2. **3D scalar scheduling underfill versus full-bundle instruction cost.** The
   scalar `integrate_tiles` launch has only 100 blocks on 188 SMs, with one thread
   serially integrating up to 256 samples for one output. The baseline expands to
   2,800 blocks and achieves 71.8% occupancy, so low occupancy alone cannot explain
   its cost. It still delivers only 2.46% of sustained fp32 arithmetic peak. Its
   sampled non-issued stalls are 47.8% long scoreboard, 29.8% execution dependency
   wait and 11.3% branch resolution. Counter totals for the whole prepared baseline
   are about **180,301 fp32 FLOPs/pair in 2D** and **6,085,249 in 3D**. These are
   physically executed arithmetic counts, not a lower bound for the operator.
3. **3D preparation is substantial GPU work, including fp64.** The complete
   baseline timeline has 42 kernel launches totaling **13.986 ms** kernel time
   within a **22.841 ms** profiled region. Count/fill take about **6.473 ms**, node
   generation **1.966 ms**, and the low/comparison/final integrations **4.607 ms**.
   Nsight Compute sees count/fill at approximately 123–131 GFLOP/s fp64 and 16.6%
   occupancy; node generation reaches approximately **1.05–1.10 TFLOP/s fp64**
   and **84.9–85.7% fp64-pipeline activity**. The primary fp32 evaluation path is
   not secretly fp64, but query-dependent geometric preparation is explicitly
   fp64 and costly on this device. This is separate from the accepted pinned
   analytic-fp64/raw-numerical-fp32 tradeoff.
4. **Host validation/readback also matters.** CUDA kernel/copy/memset busy-time
   unions span **25.88%** of the profiled complete 2D region and **61.40%** of 3D;
   prepared regions are **97.94%** / **77.74%**. These are process CUDA duty
   fractions, not occupancy or device-wide utilization. Separate warmed cProfile
   records show `_check_domain` about 14 ms and `_check_storage` about 7 ms in a
   46 ms 2D baseline call. Copy-wrapper time includes waiting for preceding GPU
   work; it must not all be attributed to transfer bandwidth. For 3D, binding and
   preparing the two convergence rules dominate the 20 ms host-profile call.
   CUDA graph replay changes baseline prepared event time only modestly at scale
   (2D 41.79 vs 41.63 ms at N=524,288; 3D 0.875 vs 0.867 ms at N=512), so graph
   launch reduction alone cannot fix the principal costs.

### Validation, limitations and next executable performance gate

All 91 complete scaling cases finish with finite outputs and exactly one selected
pair per query. Prepared output prefixes reproduce the warmed deterministic
16-query reference exactly. Existing two-rule convergence checks remain active
for every ordinary or streamed 3D preparation. This is a scheduling/profiling
study, not new independent accuracy evidence; prior high-precision fixtures and
the earlier sampled fp64 checks remain the numerical evidence. All native source
hashes are retained for verification. This does not establish solver throughput,
browser speed, arbitrary geometry/sliver behavior, or a fair comparison with
sampled SPH point-boundary interactions. No warpSPH/shared Core/Integrator or
Stage 5 work was entered.

The initially proposed next executable performance gate was a controlled 3D scheduling experiment:
replace the serial 256-sample tile traversal with a cooperative sample reduction
or smaller tiles, preserving the fp32 correction and error budgets; measure all
seven bundles at identical geometry/rules and include added scratch storage and
complete cost. In parallel as separate implementation steps, specialize the 2D
request/local-array work and cache immutable per-element domain quantities;
then reduce repeated 3D count/fill/node preparation and its fp64 cost without
weakening acceptance checks. Compare against the counters and 91-case baseline
above. These are remaining performance gates, not claims of achieved speedup.

**Priority correction after the user's work-count objection:** algorithmic work
and storage reduction precede that scheduling experiment. The 6.1 M FLOPs/pair
figure measures the current numerical implementation, not the necessary cost of
the integrated operators. The accepted 512-pair W6/order-8 baseline expands each
pair into an average of **12,622.016 angular samples**, **3,283.625 stored radial
arguments**, seven radial profiles (**22,985.375 profile evaluations**), and 28
output traversals (**353,416.438 output–sample visits**). Its batch-owned accepted
storage is **376,859.594 bytes/pair**, excluding comparison and temporary storage;
this is capacity, not a measurement of bytes fetched per execution. The comparison
rule adds **50,506.625 angular samples/pair** during preparation.

Counter accounting for the accepted prepared rule is **1,351,968.75 fp32
FLOPs/pair** in `prepare_profiles`, **4,727,405.748** in `integrate_tiles`,
**5,621** in tile-to-row reduction, **198** in output scaling/binding, and **56**
in particle reduction: total **6,085,249.498**. The stored JSON now includes this
reproducible work-expansion breakdown. It excludes geometric preparation and the
higher-order comparison, so those do not explain away the large warm count.
The profile kernel uses coefficient lists of length 13–17; the integrator reads
dynamic request metadata, forms monomial powers/weights and reduces the same
sample list independently for each output. In 2D, dynamically indexed local
arrays and per-node updates to global sum/residual storage similarly manufacture
memory work from an operator whose geometry inputs are small.

The next executable algorithmic gate is therefore to instrument panel/sample
counts and rule-order convergence on the existing fixtures and representative
supported queries, then implement a compact, fused baseline evaluation: compute
common monomials and radial quantities once per node, retain small reductions
locally, and generate/consume nodes without materializing the present expanded
per-pair cache where possible. Compare reduced-panel rules and supported analytic
angular/exterior contributions against the existing high-precision/analytic
references; preserve complete regular flux and every existing accuracy and
geometry rejection budget. Establish the sample, FLOP and storage reduction
before treating tile/block tuning as the main speed improvement. Accuracy alone
has been demonstrated for the current quadrature route; its practical-efficiency
gate remains open. No new quadrature count, speedup or small arithmetic-cost
bound is claimed until that experiment is measured.

Example reproduction (in the CUDA-enabled Warp environment):

```bash
XDG_CACHE_HOME=/tmp/fluxbridge-cache /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/studies/fluxbridge_hardware_scaling.py --dimension 2 \
  --counts 128,512,2048,8192,32768,131072,524288 --output /tmp/fluxbridge-hw-2d-repeat
XDG_CACHE_HOME=/tmp/fluxbridge-cache /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/studies/fluxbridge_hardware_scaling.py --dimension 3 --repeats 3 \
  --counts 128,512,1024,2048,8192,32768 --output /tmp/fluxbridge-hw-3d-repeat
XDG_CACHE_HOME=/tmp/fluxbridge-cache /usr/local/NVIDIA-Nsight-Compute-2026.3/ncu \
  --profile-from-start off --clock-control none --cache-control none \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section SchedulerStats --section WarpStateStats \
  --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
  -o /tmp/fluxbridge-hw-profile-repeat /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/studies/fluxbridge_hardware_scaling.py --dimension 3 --terms baseline \
  --counts 512 --profile warm --output /tmp/fluxbridge-hw-profile-driver-repeat
```

## Checkpoint — reduced single-output, single-kernel toy (26 October)

At the user's request, a separate toy now evaluates one complete particle–simplex
interaction inside one Warp kernel. Source: `scripts/studies/fluxbridge_single_term.py`;
accuracy/scaling driver: `fluxbridge_single_term_study.py`; matched native and
metamorphic driver: `fluxbridge_single_term_compare.py`. Existing production
operators and the original tracked dirty patch are unchanged. Raw evidence is in
`/tmp/fluxbridge-toy-20261026/`; durable selected profiler counters, accuracy
entries, timings and source hashes are in
`docs/work/fluxbridge-single-term-20261026.json`. Generated CUDA and SM120 PTX are
archived with the raw evidence.

The selected scalar is

\[
 I(x,h,T,f)=\int_T W_6(\lVert y-x\rVert,h)
             \left(\sum_{a=0}^{d}\lambda_a(y)f_a\right)\,dy .
\]

Here T is a 2D triangle or 3D tetrahedral volume, f contains the nodal values of
one P1 FEM scalar, and h is the support radius. Each thread takes particle
position, support, all simplex vertices and nodal values directly, normalizes
the geometry by h, computes its Jacobian, constructs Duffy quadrature points,
interpolates the field, evaluates factored W6, sums with a local Neumaier
correction, and writes one scalar. No per-pair panel/profile/sample cache or
intermediate GPU arrays exist. The host uploads an input-independent n-point
Gauss rule on [0,1]; all per-interaction preparation is inside the kernel.

Complexity is O(n^2) for triangles / O(n^3) for tetrahedra, O(1) per-thread local
state and an 8n-byte shared read-only Gauss table. With vec3-padded coordinates,
input/output capacity is **68 bytes/pair in 2D** and **84 bytes/pair in 3D**.
These are data capacity counts, not total bus traffic; the small Gauss arrays
are read in the loops and cache across threads. Loop unrolling is limited to
one to keep the nested rule compact. Nsight confirms one kernel launch for the
entire interaction, zero local-load/store sectors, and zero fp64 arithmetic in
the toy profiles.

| Toy | Samples/pair | Counted fp32 FLOPs/pair | Registers/thread |
|---|---:|---:|---:|
| Triangle, n=16 | 256 | 5,692 | 40 |
| Tetrahedron, n=8 | 512 | 10,971 | 61 |
| Tetrahedron, n=16 | 4,096 | 83,894 | 53 |

FLOPs are actual predicated FADD + FMUL + 2*FFMA counts in the varied local-.2,
twice-unit-simplex scene; counts vary with support intersection. Square roots,
division, address/control and integer operations are separate. At 1,048,576
tetrahedral pairs / n=16 the profiler measures **62.45% achieved occupancy**,
**80.34% SM throughput**, **35.59 TFLOP/s fp32** and 2.472 ms instrumented kernel
time. Separate seven-sample unprofiled measurements give **2.928 ms CUDA-event**
and **3.250 ms host-observed** median, approximately 358 M / 323 M pairs/s. New
host upload, allocation, kernel and readback cost **40.56 ms** at that count;
this includes the Python/NumPy packing route and is not the resident-array rate.
GPU actions/profiles are serialized under the shared lock; desktop activity and
uncontrolled clocks/caches remain limitations.

### Accuracy and matching the equation

Orders 4, 8, 12, 16, 24 and 32 were tested on CPU and CUDA. Independent 70-digit
fixture occupancy/first moments determine the P1 reference through an exact
affine field expansion on the host. At n=16, **all six 2D fixture entries** meet
1e-5 mixed error (worst 3.27e-6), and **25/26 3D entries** meet it (worst among
those 25 is 1.546e-6). The large full-support tetrahedron fails badly: mixed error
0.08658 at n=16 and 0.11204 at n=32. A fixed volume rule can miss a small compact
support inside a large element. That failure is retained and visible in the
report; this is a reduced toy, not a production backend or a passed general
accuracy gate. CPU/CUDA results differ by at most 4.793e-8 (2D) / 3.804e-8 (3D).
At n=8 the toy fails many fixture entries; its faster throughput is not accepted
as an accuracy-equivalent replacement.

Matched comparisons use precisely the same scalar P1 field and varied queries,
with the existing native field contraction producing one output. For 512 3D
pairs, existing prepared order-8 boundary quadrature takes **0.326 ms host /
0.320 ms CUDA-event**, while the toy n=16 takes **0.297 / 0.292 ms**, with 7.133e-8
mixed difference. Existing complete native preparation/evaluation/readback is
16.764 ms; that cost includes host work and convergence checks that the fixed
toy does not provide. At toy n=12, prepared time is 0.142 ms and mixed difference
1.503e-6 in this particular scene, but the broader fixture failures remain.
At 8,192 2D pairs, existing prepared single-output P1 evaluation is **0.685 ms**
versus toy n=16 **0.0414 ms**, mixed difference 9.396e-8. Vertex permutation,
uniform length/support scaling, translation and FEM partition-of-unity checks
pass at n=16 on 16 varied queries per dimension, worst defect below 9.46e-8.

A matched native 3D P1 Nsight profile counts **996,866 fp32 FLOPs/pair** over six
kernels, versus the toy's approximately 83,894. The previous **6,085,249** figure
was for the 28-output baseline; it is not a fair single-output denominator.
The toy demonstrates that compact local state and higher arithmetic throughput
are achievable, while its small 512-pair workload underfills the GPU and has
similar latency to prepared native evaluation. Its large-batch rate is not a
claim of hundreds-fold improvement at the same small batch size.

The reduced-problem gate requested by the user is implemented and measured.
The next algorithmic step is to handle support/element scale disparity within
this scalar toy, first testing an analytic full-support P1 identity
I=f(x) for an entirely enclosed normalized radial kernel, then a compact
geometry-aware subdivision or integration rule for partial intersections.
Compare error, sample/FLOP count and one-kernel latency on the retained failing
case and near-containment cases before expanding back to more outputs or FEM
orders. Existing accuracy budgets and production behavior remain unchanged;
Stage 5 remains excluded.

## Checkpoint — return to the closed-form single P1 term (26 October)

The user explicitly reasserted the original closed-form goal. The previous
quadrature toy is a reduced comparison experiment, not the intended solution;
the proposed next quadrature/subdivision work above is superseded by specializing
the existing analytic scalar evaluator first. The accepted analytic-fp64 versus
numerical-fp32 policy and pinned raw-fp32 analytic failures remain unchanged.

`scripts/studies/fluxbridge_single_term_analytic.py` now measures exactly the same
P1-weighted W6 scalar as that toy, using `method='analytic'`, compact boundary
flux and ordinary fp64. There are no quadrature nodes or integration-rule
preparation. The scalar is assembled as f(x) M0 + grad(f) dot M1, using the
closed-form occupancy and three first moments in 3D (two in 2D). Static rational
plans and geometry are reused; every timed launch recomputes the integrals.

Actual Nsight predicated add/multiply/FMA counts and scratch capacity:

| Single P1 scalar | fp64 FLOPs/pair | Secondary fp32 FLOPs/pair | Analytic scratch/pair |
|---|---:|---:|---:|
| 2D triangle | 18,921 | 984 | 544 bytes |
| 3D tetrahedron | 111,890 | 5,144 | 696 bytes |

The primary fused kernel contributes 18,919 / 111,888 fp64 FLOPs and the particle
reduction two more. Counts include arithmetic instructions inside CUDA math
implementations; division, square-root and transcendental operations themselves
are not assigned additional FLOPs. The secondary fp32 instructions occur in the
fp64 evaluator's generated math implementation; this is an explicitly fp64
analytic evaluation, not a precision fallback. Per-pair scratch is 68 / 87 double
scalars; total batch-owned capacity is approximately 632 / 784 bytes per pair.
The 3D plan has 12 scalar face nodes, 39 boundary nodes and 36 volume nodes for
four moment outputs, combined into one field scalar. It traverses the tetrahedron's
four faces and their three edges, with clipping and finite radial recurrences.
The stencil has no sample-order parameter; the amount of algebra depends on
kernel polynomial degree and requested moments.

Seven-sample median unprofiled CUDA-event elapsed times for the prepared action
(fused analytic evaluation/contraction plus particle reduction):

| Dimension | Pairs | GPU ms | Pairs/s | Complete host driver ms |
|---|---:|---:|---:|---:|
| 2D | 128 | 0.441 | 290 k | 1.876 |
| 2D | 512 | 0.752 | 681 k | 2.205 |
| 2D | 8,192 | 0.760 | 10.78 M | 3.408 |
| 2D | 131,072 | 2.926 | 44.79 M | 25.303 |
| 3D | 128 | 2.760 | 46.4 k | 4.356 |
| 3D | 512 | 5.261 | 97.3 k | 6.878 |
| 3D | 8,192 | 5.081 | 1.61 M | 7.937 |
| 3D | 131,072 | 16.204 | 8.09 M | 38.567 |

Host-observed prepared medians at the largest count are 3.095 / 16.404 ms.
Complete driver time includes query/storage/domain validation, support search,
allocation/upload, evaluation/reduction and readback, with three samples;
static plan construction and JIT are separate. The scene and P1 values match
the previous toy: vertices 0 and twice the unit axes, varied queries near .2,
h=1, nodal values [.35,1.1,-.2,.7] in 3D. GPU actions serialize on the shared
lock; desktop activity, clocks and caches are not isolated. Exhaustive regular
flux at 512 3D pairs measures 5.135 ms GPU versus compact 5.261 ms, within this
run's variation; no compact/exhaustive performance advantage is claimed.

The current **generic analytic interpreter is not a specialized closed-form
kernel**. The fused evaluator uses **242 registers/thread**, limiting a
256-thread block to one resident block per SM and **16.67% achieved occupancy**.
Its dynamic local arrays generate local-memory traffic, while Nsight's register
spill counter is zero. At 131,072 3D pairs the instrumented kernel reaches
**1.082 TFLOP/s fp64**, **76.78% fp64-pipeline activity**, and 13.557 ms isolated
kernel time; replay timing remains separate from the ordinary timing above.
The nested K/primitive/power recurrences recompute related endpoint quantities
across radial degrees, and dynamic plan indexing prevents normal compiler
constant-folding and broad sharing of those recurrences. These are implementation
costs, not a measured lower bound for the closed-form expression. A specialized
formula's operation count and runtime have not yet been measured.

Analytic fp64 passes **all 26 3D P1 fixture entries** on CPU and CUDA: worst mixed
errors 9.168e-12 / 9.448e-12. The enclosing-tetrahedron case returns
0.4875000000004971 versus 0.4875, mixed error 3.342e-13. All six 2D entries pass on
CUDA with worst 8.688e-14. Thus the toy's full-support failure is absent from
the analytic calculation. These are finite fixture results, not a new universal
roundoff guarantee or a claim of machine-epsilon fp64 accuracy.

A separate **timing-only raw-fp32 analytic diagnostic** measures 0.474 ms for
512 pairs and 1.154 ms for 131,072, but **20/26 fixture entries fail 1e-5**, including
two nonfinite outputs and worst finite mixed error 0.08865. Those timings are not
an accuracy-equivalent replacement; the pinned raw-fp32 issue is preserved.

Evidence: `docs/work/fluxbridge-analytic-term-20261026.json` and
`/tmp/fluxbridge-analytic-term-20261026/` with profiler reports, CSV, logs and
source verification. The exact initial fp64 study source is archived as
`fp64-measured-script.py`; the current study adds JSON-safe recording of raw-fp32
nonfinite results, without changing operators. All measurement source hashes,
the pre-existing tracked patch and all 32 native FluxBridge source files verify.

The next executable implementation step is a dedicated closed-form W6/P1
tetrahedral kernel: fix the moment/radial-degree set, share endpoint geometry
and radial recurrences, keep compact local state, and fuse the scalar field
contraction. Retain fp64 as the accepted reference precision and validate against
the same 26 entries, especially support cuts and full containment, before
measuring FLOPs, registers, local-memory traffic and small/large batch latency.
This is the next cost to establish; neither a hypothetical speedup nor a fixed
small FLOP count is asserted in advance. No warpSPH or Stage 5 work was entered.

### Equation/implementation audit prompted by the user

For z=(y-x)/h, A=f(x), B=h grad(f), the measured scalar is exactly
`1365/(64*pi) * integral_(normalized tet intersect unit ball)
(A+B dot z) (1-r)^8 (1+8r+25r^2+32r^3) dz`, r=|z|. The P1 field needs only
occupancy and three first moments. The current analytic route expands the radial
shape into `1-11r^2+66r^4-462r^6+1056r^7-1155r^8+704r^9-231r^10+32r^11`,
evaluates its monomial moments separately, then recombines them. This expansion
also introduces strong cancellation near r=1.

The 696-byte scratch row is **12 face values + 39 boundary accumulators + 36
volume values = 87 doubles**, not intrinsic input geometry storage. The generic
plan shares some nodes, but only above the terminal face/edge primitive layer.
That layer repeatedly calls K_m, which repeatedly calls J_(m-2), where
`J_m=integral r^m ds` and `K_m=integral r^m/(s^2+w^2) ds`. K obeys
`K_m=J_(m-2)+a^2 K_(m-2)`; J has its own finite recurrence and recomputes powers
with multiplication loops. Existing `K` restarts the recurrence for every
requested degree, and each `primitive` call restarts J as well. For the P1/W6
face power set [-3,0,2,4,6,7,8,9,10,11,12,13], one common interior endpoint
requests **12 K evaluations and 54 J evaluations, although only 15 distinct J
indices occur**. This is source-level call accounting for an interior branch,
not a new measured count of transcendental instructions or every clipped case.
Face/edge geometry and clipping are also rebuilt inside the per-degree loop.

Thus the measured 112k arithmetic count includes substantial repeated evaluation
inside an insufficiently specialized recurrence interpreter. Generality through
degree nine/derivatives remains in its generated code and register requirements
even though this P1 plan needs only scalar face terminals. Exact sphere-clipped
tetrahedral integration has real geometric/elementary-function work, but neither
112k operations nor the 87-value buffer has been established as necessary for
this scalar. The specialization must share endpoint recurrence tables, powers,
geometry and clipping, combine radial coefficients with conditioning in mind,
and accumulate the needed final moments/scalar with compact live state.

### Primitive sharing across operators and particles

A static audit of the current compact 3D W6 plans is saved in
`docs/work/fluxbridge-primitive-sharing-20261026.json`. Including parity-chain
intermediates, the required endpoint families are:

| Requested quantities | J indices | K indices |
|---|---|---|
| Occupancy only | -1 through 11 | -1 through 13 |
| P1 scalar's four moments | -1 through 13 | -1 through 15 |
| Full 28-output baseline | -1 through 15 | -1 through 15 |
| P3 interpolation's 20 moments | -1 through 15 | -1 through 17 |
| Degree-nine moments and jets, 2,200 outputs | -1 through 21 | -1 through 23 |

These are unions of primitive indices required by the current reductions, not
operation counts or a claim that every branch uses every entry. The complete
baseline needs the same K family as the one P1 scalar, and only two additional
J indices. Coordinate factors, derivative directions, FEM basis functions and
field components change polynomial coefficients and contractions; they do not
require new evaluations of an identical primitive with identical geometry and
endpoint. Higher weighted moments additionally use ordinary endpoint powers.

For a fixed particle/support and a canonical physical edge tangent t, define
`s=t dot (v-x)/h`, `a=n dot (v-x)/h`, and
`w=(t cross n) dot (v-x)/h`. Then `c^2=a^2+w^2` is the squared distance from the
particle to the infinite edge line divided by h^2. It is identical for the two
faces incident to that edge. Thus J_m(s,c), vertex radii and edge/sphere cut
parameters can be shared across those faces: six physical edges rather than
twelve face-edge incidences. The identity verifies numerically on 720 varied
face-edge/particle/support combinations with mixed defect below 1e-14.
K_m(s,w,a) generally remains face-specific because its denominator includes w^2
and its seeds distinguish a and w; the two faces have different a/w decompositions
even though their c is identical. Each face's K parity chains can consume the
shared edge J table. Canonical edge orientation is necessary when sharing signed
endpoint quantities. Stable interval differences and limiting branches must be
preserved rather than replaced by unstable endpoint subtraction.

Across different particles interacting with the same fixed tet, cache geometry
(vertices, edge tangents/lengths, normals, plane constants, FEM inverse map and
basis gradients) once. Actual s/a/w, endpoint radii, c, support cuts, J and K vary
with particle position and support. For example
`s0=(t dot v0-t dot x)/h`, `s1=s0+edge_length/h`, and
`a=(plane_constant-n dot x)/h`. Shared geometry makes these inexpensive projections;
it does not make distinct particles' integral values equal. When x and h stay
fixed and only velocities/material/FEM coefficients change, the primitive tables
and geometric moments can be reused exactly. No cross-particle numerical reuse
or additional GPU speedup is asserted by this index/geometric audit.

## Shared-J/K fused analytic prototype checkpoint (2026-10-26)

Implemented the next executable step from the analytic audit, rather than
another plan: `scripts/studies/fluxbridge_shared_jk.py` provides an isolated Warp
**device function and one kernel for one scalar W6/P1 particle–tet interaction**.
It consumes particle position, support radius, a tet index, four independently
supplied nodal coefficients and reusable geometry. This is the same scalar as
the earlier generic analytic benchmark. It does **not** claim the full 28-output
baseline, degree-nine/2,200-output stress contract, or solver coupling.

The prototype traverses six canonical physical edges. Both incident faces share
one clipped interval and J parity recurrences through J_13. Each face streams
`C_m = delta K_m - a^m delta K_0` through m=15, consuming that shared J stream;
radial coefficients accumulate directly into occupancy and three first moments,
then the P1 scalar is contracted in the same kernel. There is no quadrature,
dynamic recurrence interpreter, terminal-moment workspace or global pair scratch.
The degree loop is generated with exact rational coefficients and omitted zeros;
work is O(6D) per pair at fixed D=15, with constant recurrence state. Static
vertices, tangents, normals, outward in-face vectors and FEM basis gradients are
prepared once per tet in host fp64 and stored in the requested device dtype.

The final formulation retains the **regular exterior flux on every face**,
including faces whose distance exceeds the support radius. It avoids a separate
containment/solid-angle classification. Edge radii and each incident face's
normal/outward offsets use a coherent projected edge geometry, preserving
finite vertex limits despite rounded tangents. The stable interval seeds use
angle differences and compensated log(1+z), with normalized angular arguments to
avoid small-distance underflow. No production source was changed.

### Input and accuracy evidence

- **26/26 existing independent P1 fixture entries pass fp64** on CPU and CUDA at
  mixed-error budget 1e-10. Worst errors are **3.000e-12 / 3.044e-12**; worst
  CPU/CUDA difference is 6.044e-12. Support cuts, disjoint/AABB cases, full-ball
  containment, slivers and near-plane/edge/vertex cases are included.
- All 24 vertex orderings, translation/unit scaling, basis partition, field
  linearity, zero fields, empty batches and 64 varied positions/supports/fields
  pass. Eight malformed-input cases reject before kernel execution, including
  invalid supports, nonfinite positions, malformed fields and invalid tet indices.
- **90 additional queries on 18 rotated/stretched/sheared tets** cover interior,
  two vertices, an edge and a face. Independent CPU integration uses the
  **factored** W6 profile with Gauss/Duffy orders 48, 72, 96, solely for validation.
  Worst 72-to-96 mixed difference is **5.450e-12**; prototype CUDA vs independent
  order-96 reference is **2.466e-13**. This is a finite convergence check, not a
  rigorous universal quadrature error bound or a quadrature production backend.
- This broader check reveals a **pre-existing native compact boundary-limit
  failure in 38/90 cases**, worst mixed error 0.8642. The independent integration
  agrees with the new prototype; comparisons to the old evaluator at exact
  rotated vertices/edges/faces cannot serve as ground truth. This gate is now
  explicitly open for the native implementation, and the affected inputs and
  references are retained. Do not promote the prototype solely on agreement
  with the old compact evaluator.
- Raw device fp32 remains **diagnostic**: all 26 entries are finite, but **3/26
  entries fail 1e-5**, worst 0.001118 on the sliver; the 64 varied-input comparison
  reaches 5.720e-5. The two other failed entries duplicate the same geometry
  in the existing fixture set. The accepted analytic-fp64 / numerical raw-fp32
  tradeoff and existing pinned raw-fp32 failures remain unchanged.

Reproduce with `scripts/studies/fluxbridge_shared_jk_study.py` (CPU/CUDA, precision,
accuracy-only/counts/profile switches), `fluxbridge_shared_jk_shapes.py` for the
independent boundary audit, and `fluxbridge_shared_jk_mixed.py` for varying-input
throughput. The generated fp64 native body is saved as
`docs/work/fluxbridge-shared-jk-native-fp64-20261026.cpp` for inspection.

### Actual CUDA cost and scaling

Measurements use the same RTX PRO 6000 Blackwell, serialized with the existing
GPU lock. Default launch is 256 threads/block. GPU times below are medians of
seven warm CUDA-event samples, excluding JIT, geometry preparation and transfers.
The canonical scene matches the previous P1 analytic comparison; the mixed
scene varies particle position, support, tet index and four nodal field values
independently across 256 rotated/stretched/sheared tets.

| Pairs | Canonical GPU ms | Canonical pairs/s | Mixed GPU ms | Mixed pairs/s |
|---:|---:|---:|---:|---:|
| 512 | 0.272576 | 1.88 million | 0.596128 | 0.86 million |
| 8,192 | 0.275872 | 29.7 million | 0.597120 | 13.7 million |
| 131,072 | 0.817984 | 160 million | 1.785056 | 73.4 million |
| 1,048,576 | 7.883744 | 133 million | 16.706656 | 62.8 million |

At 512 / 131,072 canonical pairs, the generic analytic predecessor took
5.261376 / 16.203680 GPU ms: measured speedups are **19.3x / 19.8x**. These are
separate runs with uncontrolled desktop load/clocks, not a locked-clock guarantee.
A launch sweep of 32/64/128/256 threads gives bitwise-identical outputs. Smaller
blocks improve 512-pair latency to about 0.134 ms, but give little large-batch
improvement (131,072 stays about 0.817 ms). The final two large sizes give local
runtime exponents **1.09 canonical / 1.08 mixed** for an 8x input increase;
small batches underfill the device. This does not establish asymptotic scaling
from two points.

Nsight Compute directly captures one warm canonical kernel, at 512 and 131,072
pairs. It measures **7,671 fp64 FLOPs/pair** plus 264 fp32 FLOPs in math routines,
versus 111,890 fp64 FLOPs previously: a **14.6x arithmetic reduction**. These
counts include add/multiply/FMA (FMA=2); division, sqrt, log, atan, control and
integer work are not assigned equivalent FLOP costs. The final kernel uses
**134 registers/thread (136 allocated), 16.65% achieved occupancy**, and has
**zero measured local load/store sectors and spill requests**. The compiler
still reports a 1,024-byte stack reservation. At 131,072 pairs, the profiling
replay takes 0.825856 ms, achieves **1.217 TFLOP/s fp64**, with fp64 pipeline
activity **78.25%** and DRAM activity **0.67%** of their respective sustained
peaks. This workload is primarily limited by device fp64 compute; occupancy alone
is not the achieved arithmetic fraction.

For completeness, the accuracy-failing raw-fp32 variant measures 5,614 fp32
FLOPs/pair, 79 registers/thread, 42.54% achieved occupancy and zero local traffic.
Its 1,048,576-pair GPU time is 0.154848 ms, about 6.77 billion pairs/s. This is
**not an accuracy-equivalent replacement** and does not close the raw-fp32 gate.

Device allocation is **76 bytes/pair fp64 / 40 bytes/pair fp32**, consisting of
positions, supports, indices, fields and output; there is no pair integration
scratch. One cached tet uses **816 / 456 bytes**, including shared edge topology;
256 fp64 tets use 184,416 bytes. These are actual allocated device array sizes,
not peak process/driver memory measurements. Separately timed host-fp64 static
preparation is 0.410 ms for one tet and 42.925 ms for 256. For 1,048,576 fp64
pairs, prepared-buffer host wall time is 8.688 / 16.906 ms canonical/mixed;
**validation + allocation + upload + kernel + readback** takes 24.101 / 31.544 ms.
Static geometry and JIT are excluded from those transfer timings. Moving-mesh
geometry preparation remains unoptimized, and transfer/host overhead must remain
visible when assessing application cost.

Evidence: `docs/work/fluxbridge-shared-jk-20261026.json`, the full independent
input/reference audit `docs/work/fluxbridge-shared-jk-shapes-20261026.json`, and
`/tmp/fluxbridge-shared-jk-20261026/` for Nsight reports, CSV, logs and generated
sources. Final measured-source hashes verify, as do the original tracked dirty
patch, previous plan content and all **32 native FluxBridge source files**.
Earlier compact/intermediate results in the temporary evidence directory are
superseded by `final-*` reports and final profiler captures.

**Next executable step:** extend this shared-edge engine to all 28 baseline
outputs in one pass, validate every component (including the new independent
rotated boundary cases), and compare directly with the existing full-contract
measurements. Correct the native compact boundary-limit failure before promoting
any replacement. Higher-order FEM, 2D specialization and production dispatch are
not implemented by this prototype. No warpSPH, Core/Integrator, ordinary solid
application or Stage 5 pressure/porosity code was changed; Böttcher et al. (2025),
arXiv:2504.07739v2 remains the principal porous-flow reference.

## Existing solver applicability and register-lifetime checkpoint (2026-10-26)

Read-only inspection of warpSPH confirms that its WCSPH analytic wall stages and
DFSPH projection geometry call the existing `AnalyticBoundary.aggregate()`
provider in `scene/provider.py`. They consume solid-domain integrals `lam`, `G`
and `Cov`, with hydrostatic contraction and optional Laplacian/cover/tensile
channels on the WCSPH side. DFSPH requests the lean layout through the shared
projection-geometry builder; that fixed geometry is assembled once and reused
by pressure iterations. The current provider is the existing **2D** edge/box/
disk/conditioned-triangle path, already fused across channel/kernel groups and
cached at fixed positions. The new **3D W6/P1 tetrahedral study is not wired into
that provider**. Its measured 20x scalar speedup is not a measured speedup for
current WCSPH/DFSPH boundaries or whole timesteps.

The common solid-domain moments make the shared primitive/contraction approach
applicable to ordinary solid boundaries as well as later porous material fields.
Constant P1 nodal data gives the occupancy integral; the solver gradient and
first-gradient moment outputs require the planned derivative extension and
validation of signs, support scaling, kernel family and provider conventions.
The full current 2D wall layout serves Wendland C2/C4, whereas this prototype
specializes 3D W6. A 2D adaptation and/or 3D provider is needed before those
solvers benefit. Keep disk/cylinder cross sections and periodic fibre arrays as
ordinary solid applications. The previously reported 38/90 failures belong to
the **3D FluxBridge compact evaluator**; they are not evidence of a failure in
this separate existing 2D solver boundary path. No solver code was imported or
modified during this inspection.

Before adding outputs, implemented an isolated early-P1-contraction variant in
`scripts/studies/fluxbridge_shared_jk_registers.py`. It contracts radial/edge
terms into the final scalar while they are live, avoiding two persistent face
normals, four face-moment accumulators and a final vector-moment accumulator.
The original validated prototype and production operators remain unchanged.

Direct contraction passes all 26 fp64 entries on CPU/CUDA, worst mixed errors
**2.495e-12 / 2.141e-12**, all existing input/property checks, and all 90 independent
rotated/stretched/sheared cases, worst **2.370e-13**. Device register queries and
Nsight Compute verify the following at 131,072 pairs and 256 threads/block:

| Variant | Registers/thread | Achieved occupancy | Allocated local bytes/thread | Warm GPU ms |
|---|---:|---:|---:|---:|
| Original shared-J/K | 134 | 16.65% | 0 | 0.817984 |
| Early scalar contraction | 128 | 29.10% | 0 | 0.819616 |
| Original, forced cap 128 | 128 | 29.08% | 24 | 0.818400 |
| Original, forced cap 112 | 112 | not profiled | 80 | 0.827680 |
| Original, forced cap 96 | 96 | 29.04% | 144 | 0.830048 |
| Early contraction, forced cap 112 | 112 | not profiled | 40 | 0.823200 |
| Early contraction, forced cap 96 | 96 | not profiled | 96 | 0.843872 |

The original occupancy/profile row is the preceding checkpoint; variant times
are seven-sample CUDA-event medians in separate serialized runs. Clocks/caches
and desktop activity are uncontrolled, so small differences are not established
speedups. The source-level reduction crosses a block-residency threshold: 128
registers permits two resident 256-thread blocks in 65,536 registers per SM;
134 (136 allocated) permits one. Reducing 128 to 96 does not admit a third block
at this launch size. Most importantly, **higher occupancy leaves throughput about
the same**: direct contraction's fp64 pipeline activity is 78.90%, close to the
original 78.25%. Its measured arithmetic rises slightly to 7,719 fp64 FLOPs/pair,
and its profiling replay is 0.822272 ms. Register headroom and avoiding future
spills are the benefit demonstrated here, not a new large speedup.

The uncapped direct variant retains **zero measured local LD/ST sectors**.
Caps add actual local traffic: original/cap128 has 98,304 load and 98,396 store
sectors; cap96 has 2,883,584 load and 1,113,172 store sectors across 131,072 pairs.
The raw counters are retained because Nsight's derived spilling-request metric
reports zero even in these capped kernels. Do not use that derived metric alone
to infer an absence of local traffic. No cap is promoted to a default.

Adding 28 fp64 accumulators naively would itself require at least 56 32-bit
registers before geometry/recurrence temporaries. The extension must preserve
shared J/K streams while compiling the actual requested contractions and
bounding output liveness; it must not simply append all output arrays to the
current thread state. Further candidates are reloading cached vertices instead
of retaining all four normalized vertices, packing geometry to shorten address/
stride lifetimes, and distributing output ownership across lanes if a full
moment tensor is required. These are candidates, not measured improvements yet.
The earlier block-size sweep remains relevant: smaller blocks helped tiny
batches but did not materially improve large-batch fp64 throughput.

Evidence: `docs/work/fluxbridge-registers-20261026.json` and
`/tmp/fluxbridge-registers-20261026/` (raw Nsight reports, CSV, CPU/CUDA/property
results, independent geometry audit, generated variant sources and logs). All
pre-existing shared-J/K/native source hashes, the tracked dirty patch and prior
plan content verify unchanged. Next: implement bounded full-contract output
scheduling on this shared engine, retain the independent boundary-limit gate,
and measure register allocation, local traffic and throughput per added output
family before promoting a replacement. warpSPH, Core/Integrator and Stage 5
pressure/porosity integration remain untouched.

## Bounded geometry, caller contractions and first-gradient family (2026-10-26)

Implemented and measured the next priorities. New experimental device functions
are in `scripts/studies/fluxbridge_shared_jk_layout.py`,
`fluxbridge_shared_jk_gradient.py` and `fluxbridge_shared_jk_gradient_contract.py`.
The earlier measured prototypes and all production sources remain unchanged.

### Geometry lifetime and requested-output specialization

Four layouts isolate the effects of retaining/reloading normalized vertices and
using separate/packed geometry arrays. Packed storage puts the same 30 vec3
records per tet into one allocation: vertices, normals, tangents, outward
in-face vectors and FEM basis gradients. It reduces pointer/stride liveness,
not the number of geometry bytes. Vertex reload evaluates only the current
edge's two normalized vertices. Host geometry preparation remains the validated
fp64 implementation; packing precedes a single data upload, with no device
readback. Rebuild this explicit static data when vertices change.

Added two requested-output paths alongside the per-pair P1 scalar:

- Occupancy-only omits first-moment/field work and has no nodal-field input.
- Multicomponent P1 evaluation computes four moments once per pair, then
  contracts/stores **one component at a time**. Nodal coefficients are shared
  `[tet,component,4]` FEM data; the original scalar still accepts per-pair nodal
  values. The output is `[component,pair]` for coalesced stores. No primitive or
  moment tensor is materialized globally. Register use is independent of the
  tested component count (1, 3, 12, 28).

Actual CUDA properties and seven-sample warm CUDA-event medians at 131,072 pairs,
256 threads/block, canonical scene:

| Geometry | Scalar P1 registers / GPU ms | Occupancy registers / GPU ms | 28 P1 fields registers / GPU ms |
|---|---:|---:|---:|
| Separate, retained vertices | 128 / 0.815584 | 112 / 0.745536 | 128 / 0.905856 |
| Separate, reloaded vertices | 118 / 0.823776 | 96 / 0.753280 | 122 / 0.940864 |
| Packed, retained vertices | 126 / 0.815136 | 110 / 0.745472 | 128 / 0.906144 |
| Packed, reloaded vertices | **112 / 0.824384** | **94 / 0.758336** | **120 / 0.940384** |

Every variant reports **zero allocated local bytes/thread**. The scalar drops
from the original 134 registers through early contraction's 128 to 112; this is
headroom, not a new throughput improvement. Reloading slightly increases work;
keep the cached variants available. The packed/reloaded 28-field kernel remains
at 120 registers for every tested component count. No forced register cap is used.

Nsight captures one warm kernel at 131,072 pairs. Packed/reloaded scalar,
occupancy, 1-field and 28-field paths respectively measure **7,773 / 7,115 /
7,686 / 8,793 fp64 FLOPs per pair**. Their achieved occupancy is about 29%, fp64
pipeline activity about 79%, and **local load/store sectors are zero**. Packing
and reloads should not be described as a demonstrated speedup: retained packed
vertices use 8,580 FLOPs for 28 fields and can be slightly faster. FLOP counts
mean add/multiply/FMA (FMA=2); transcendental, division/root, control and integer
costs are not assigned equivalent FLOP costs.

### Matched consumer comparison and application costs

`scripts/studies/fluxbridge_shared_jk_consumers.py` compares shared moments with
one scalar-kernel launch per field component, using **identical nodal, query,
geometry and output buffers**. The repeated path also reads shared per-tet
nodal data, so it is not penalized by duplicating field input per pair. CUDA
graph replay removes Python gaps between repeated launches; graph results are
bitwise equal to the corresponding direct launches. Shared vs repeated outputs
have worst mixed difference 1.061e-12.

| Scene / pairs | Components | Shared graph GPU ms | Repeated graph GPU ms | Ratio |
|---|---:|---:|---:|---:|
| Canonical / 131,072 | 3 | 0.823008 | 2.678112 | 3.25x |
| Canonical / 131,072 | 28 | 0.939392 | 25.967169 | 27.64x |
| Canonical / 1,048,576 | 3 | 6.460640 | 20.426880 | 3.16x |
| Canonical / 1,048,576 | 28 | 8.022208 | 196.905884 | 24.55x |
| Mixed / 131,072 | 3 | 1.775040 | 5.759136 | 3.24x |
| Mixed / 131,072 | 28 | 1.888224 | 55.677246 | 29.49x |
| Mixed / 1,048,576 | 3 | 15.259680 | 44.012798 | 2.88x |
| Mixed / 1,048,576 | 28 | 15.394240 | 423.571289 | 27.51x |

The mixed scene contains 256 rotated/stretched/sheared tets and independently
changing positions, supports, tet indices and nodal data across tets. These are
**P1 field components, not the full 28 distinct monomial/derivative operators**.
The work scales as shared geometric integration plus component contractions,
rather than repeating integration for each component. All GPU runs serialize on
the existing lock; clocks/caches and desktop activity remain uncontrolled.

Geometry allocation is still 816 bytes for one fp64 tet (720 data + 96 common
edge topology), or 184,416 bytes for 256. The mixed consumer setup takes
43.981 ms in the host-fp64 geometry loop; moving/deforming-mesh preparation is
not optimized. P1 scalar buffers use 76 bytes/pair, occupancy 44, and shared
F-field buffers `36+8F` bytes/pair plus `32*E*F` nodal bytes. There is no global
pair integration scratch. For 1,048,576 shared 28-field values, allocation,
validation, upload, launch and readback take **112.224 ms canonical / 121.410 ms
mixed**, versus roughly 8 / 15 ms GPU time; output readback matters. JIT and
separately timed static geometry are excluded. These are allocated array sizes,
not peak process/driver memory measurements or native search/assembly timings.

### Solid-wall gradient family using the same edge stream

Extended the analytic engine to the actual first-gradient quantities needed by
solid-wall consumers: **lambda, three G entries and six symmetric Cov entries**.
Coordinates z=(y-x)/h and phi(r)=W6's unnormalized radial profile give
`G=-(C/h) sum_F n_F S_F` and
`Cov=C [m0 I - sum_F U_F tensor n_F]`, where C=1365/(64*pi),
`m0=integral phi dz`, `S_F=integral_(F intersect B1) phi dS` and
`U_F=integral_(F intersect B1) z phi dS`.
For each face,
`U_F=d_F n_F S_F + sum_edges mu_e L_e`, with the compact potential edge integral
`L_e=sum_n c_n/(n+2) [delta J_(n+2) - delta J_0]`.
The S_F coefficients consume the existing C_m/K stream; L_e consumes the same J
stream. No quadrature or derivative interpreter enters the device kernel.

The occupancy-only `c=0` edge-line skip cannot be copied into this family:
the tangential potential L_e remains nonzero. Implemented its exact absolute-
power J_m limit, clipped to [-1,1], including particle-on-vertex cases. This
restores the off-diagonal Cov contributions at exact edge/vertex queries.

For a P1 field with physical gradient g and A=f(x), the requested vector is
`integral f(y) grad_x W dy = A G + g dot Cov`. Multicomponent callers stream this
contraction from ten local scalars, producing `[3*component+axis,pair]` outputs.
The specialized single-field variant contracts during edge consumption, retaining
only its requested vector and occupancy correction instead of a Cov tensor.
It needs 132 registers versus the moment-based path's 136. That modest reduction
does not establish a meaningful new throughput gain.

Canonical fp64 measurements (one pair is one whole particle/tet interaction):

| Requested output | Registers/thread | GPU ms, 131,072 pairs | GPU ms, 1,048,576 pairs |
|---|---:|---:|---:|
| lambda + G + symmetric Cov (10 scalars) | 136 | 0.844576 | 6.681856 |
| One P1 field's gradient (3 outputs), moment contraction | 136 | 0.846688 | 6.939296 |
| Three P1 fields' gradients (9 outputs) | 136 | 0.857248 | 7.412800 |
| 28 P1 fields' gradients (84 outputs) | 136 | 0.976576 | 8.296928 |
| One P1 field's gradient, early contraction | 132 | 0.834368 | 6.304832 |

The 10-scalar solid-wall family reaches about **155 million pairs/s** at 131,072
pairs. Nsight measures **8,051 fp64 FLOPs/pair** for that family, **9,379** for
28 field gradients, and **7,846** for early single-field gradient contraction.
All three have **zero local LD/ST sectors and zero runtime local allocation**.
Achieved occupancy is about 16.66% at this launch size, with fp64 pipeline
activity 78.3–78.6%. The compiler's 1,024-byte stack reservation is distinct from
actual measured local traffic. Fewer resident blocks do not imply idle fp64
execution resources. Gradients' mixed-geometry performance has not been measured
in this checkpoint; the mixed speedup table above concerns P1 values.

Output traffic remains explicit: 84 gradient outputs use 672 MiB at 1,048,576
pairs. The 28-field gradient allocation/upload/kernel/readback path takes
**289.718 ms**, compared with 8.297 ms GPU time. It excludes geometry/JIT/search/
scatter and uses the study's direct array preparation, not a complete native
provider/application step. Such tensors should remain device-resident when the
caller can consume them there; that integration is not claimed here.

### Acceptance, preserved work and next executable gate

All four value layouts pass the **26 independent fixture entries and 90 rotated/
stretched/sheared queries**, including basis coefficients and arbitrary signed
1/3/12/28-component fields, at the unchanged **1e-10 mixed-error budget**. Worst
value error across layouts is 2.501e-11, dominated by sliver-weighted fields.
Independent four-basis CPU references use the factored W6 profile and converged
orders 72/96; they are validation-only.

The gradient family and early contraction pass the same fixtures on **CPU and
CUDA**, with arbitrary fields, unit/zero fields and FEM basis fields. The
10-moment fixture errors are 2.732e-13 CPU / 2.125e-13 CUDA. Worst weighted-
gradient fixture errors are **6.441e-11 CPU / 4.821e-11 CUDA**, on the sliver;
these finite checks do not certify arbitrary ill-conditioning. The independent
90-case gradient reference was tightened to worst successive-order mixed change
**9.738e-13**, refining only cases needing orders 128/160/192 after 72/96. Worst
contracted gradient error against those references is **1.136e-12**. No accuracy
budget was relaxed. The analytic-fp64 / numerical raw-fp32 tradeoff stays pinned;
this checkpoint validates the new first-gradient kernels in fp64 only.

Evidence: `docs/work/fluxbridge-layout-20261026.json`,
`docs/work/fluxbridge-layout-reference-20261026.json`,
`docs/work/fluxbridge-gradient-reference-20261026.json`, and
`/tmp/fluxbridge-layout-20261026/` for nine Nsight captures, CSV, generated native
bodies, logs and intermediate/final reports. The durable evidence retains the
reference refinement source and exact hashes. Final measurement source hashes,
all prior prototype/native hashes, all prior fixture hashes, the tracked dirty
patch and previous plan content verify unchanged. Intermediate failed gradient
experiments are superseded by the final hash-verified tight-reference reports.

The priorities now have concrete results: shorter geometry liveness, bounded
caller-specific value/gradient outputs, coalesced stores, actual CUDA occupancy/
arithmetic/traffic evidence and matched consumer comparisons. Cooperative lane
ownership remains a candidate for a full tensor contract, not a demonstrated
requirement or implemented optimization in this step.

**Next executable gate:** add the six quadratic value moments and six symmetric
kernel-Hessian entries with bounded output scheduling. Combine them with the
validated first-moment/gradient families and map the three redundant Cov entries
into the exact 28-output baseline ordering; test every channel against independent
fixtures and rotated boundary limits before production promotion. Fix the native
compact boundary-limit issue before replacing its dispatch. The new device
functions remain isolated experiments; no existing WCSPH/DFSPH provider, 2D solid
disk/cylinder/fibre path, warpSPH/Core/Integrator or Stage 5 pressure/porosity code
was changed. Böttcher et al. (2025), arXiv:2504.07739v2 remains the principal porous
reference, and transverse 2D fibre flow remains distinct from longitudinal flow.

## Complete shared-stream 28-output prototype (2026-10-26)

**Completed the preceding executable gate:** the analytic fp64 W6 tetrahedron
prototype now computes all **28 distinct baseline outputs in one kernel**:
10 value moments (degrees 0–2), 12 first-gradient moments (degrees 0–1), and six
symmetric kernel-Hessian entries. This is the actual `baseline(3)` ordering,
not 28 components of one P1 field. The six quadratic values and six Hessians
are implemented and validated, with the three redundant Cov entries expanded
only when storing the output.

Implementation: `scripts/studies/fluxbridge_shared_jk_full.py`; acceptance and
scaling: `fluxbridge_shared_jk_full_study.py`; independent reference:
`fluxbridge_shared_jk_full_reference.py`; matched launch experiment:
`fluxbridge_shared_jk_full_launch.py`. These are isolated experimental device
functions, not a production-provider replacement. The preceding prototype and
all native library source hashes remain unchanged.

### Shared work and bounded state

Each of the six physical edges supplies one J stream to its two adjacent face
C streams, through fixed degree 15. The compact potentials `ell` and `P`, with
`ell'(r)=r*phi(r)` and `P'(r)=r*ell(r)` and both zero at unit support, supply
quadratic moments by boundary reduction. Gradient and Hessian surface terms
use the same primitives and geometry. The full kernel performs no quadrature,
primitive-table interpretation or global per-pair moment allocation. Regular
exterior flux and the exact edge-line tangential limits apply to all families.

There are 26 named double accumulators: two scalar volume potentials, two
three-vectors, and three symmetric six-component tensors. Family selection is
at generation time; compiler elimination removes unused accumulation and
recurrence work. Measured registers/thread are **144 values-only, 136 gradient,
128 Hessian, and 188 full** (192 allocated for full). All have **zero runtime
local allocation and zero measured local LD/ST sectors**, without forced
register caps. A compiler stack reservation is not evidence of local traffic.

### Independent accuracy gate

The full 28-output body and all three specializations pass **CPU and CUDA**:

- 26 established fixtures, including tangent/disjoint/full-support, sliver,
  face/edge/vertex limits, switch and roundoff cases: 728 full outputs/device.
- 90 independently referenced rotated/stretched/sheared queries at interior,
  face, edge and vertex positions: 2,520 full outputs/device.
- All 24 vertex orders, translation, unit scaling, tensor rotation, empty
  batches, and subdivision into four tetrahedra with the query exactly at their
  common internal vertex. Internal-face contributions cancel.

The unchanged criterion is mixed error `abs(actual-reference)/(1+abs(reference))
<= 1e-10`. Worst across fixtures and independent cases is **6.624e-12 CPU /
3.002e-12 CUDA**. Graph replay is bitwise stable; launches with 32/64/128/256
threads also produce bitwise-identical outputs.

The independent reference integrates signed cones from the query to the four
faces, using exact radial polynomial integrals and a 2D Gauss face rule in
`np.longdouble`. It shares neither J/K recurrences nor device geometry code.
All 90 cases meet the **2e-12 successive-order target**, worst 1.804e-12,
with orders 32–192; the reference itself agrees with the established fixtures
to 9.403e-13. These are empirical convergence checks, not a rigorous universal
bound or a machine-epsilon claim.

The initial Cartesian 3D Duffy oracle did **not** meet its tighter 2e-12 target
on five face queries at order 256 (worst change 3.838e-11), so it was rejected
as the acceptance oracle. Its 90 results nevertheless cross-check the signed-
cone reference to 1.015e-11. The failed convergence gate is retained explicitly
in the evidence; the implementation's acceptance budget was not relaxed.

### Actual CUDA cost, memory and scaling

Serialized RTX PRO 6000 Blackwell runs, warm CUDA-event medians, default
256-thread blocks; each row is the entire indicated output family:

| Outputs/family | 131,072 pairs (ms), canonical | 1,048,576 pairs (ms), canonical | 1,048,576 pairs (ms), mixed |
|---|---:|---:|---:|
| 10 values | 0.897344 | 7.960288 | 15.843840 |
| 12 gradient moments | 0.839936 | 6.746528 | 16.251425 |
| 6 Hessian entries | 0.557600 | 4.578688 | 9.129184 |
| Full 28 outputs | **1.048576** | **9.049760** | **17.664928** |

The full family reaches **125.0 million pairs/s** at 131,072 canonical pairs,
**115.9 million** at 1,048,576, and **59.36 million** for the larger mixed
sample. Mixed means 256 differing tetrahedra with random indices, queries and
supports, including rejected/support-clipped work; it is not a localized
accepted-neighbor distribution. From 131,072 to 1,048,576 pairs, the empirical
full-family scaling order is **1.036 canonical / 1.042 mixed**, with linear
input/output storage. Small batches have a latency plateau.

Nsight measures **10,342 fp64 arithmetic FLOPs/pair** for the canonical full
contract, versus 8,620 values-only, 8,015 gradient-only and 5,265 Hessian-only.
The full capture achieves about **1.285 TFLOP/s**, **16.64% achieved occupancy**,
**78.72% fp64 pipeline activity** and 0.94% DRAM throughput. Mixed full executes
9,418.66 fp64 FLOPs/pair, about 0.609 TFLOP/s, with 16.58% occupancy and 79.03%
fp64 pipeline activity. Its lower arithmetic count and longer runtime are
consistent with divergent paths and partial active lanes. Arithmetic counts
include add/mul/FMA (FMA=2); roots, division, transcendental and control/integer
work are not assigned artificial FLOP equivalents.

A matched launch sweep keeps the same kernel body, allocations and inputs.
At 8,192 pairs, 64 versus 256 threads reduces full-kernel time from **0.351264
to 0.168288 ms** canonical and **0.676864 to 0.324064 ms** mixed. At large
counts the differences are small/noisy; profiling 64-thread blocks still gives
only 15.59–15.67% achieved occupancy and no spills. Smaller blocks help small
launch latency; they do not solve the large-batch fp64 cost. No universally
preferred large-batch launch size or forced cap is promoted.

Full input/output storage is **260 bytes/pair**, including 224 output bytes,
with **zero global per-pair integration scratch**. At 1,048,576 pairs this is
260 MiB total, 224 MiB outputs. Static packed geometry is 720 bytes/tet plus
96 bytes common topology: 816 bytes for one tet, 184,416 bytes for 256.
Measured one-shot static host preparation/upload is 0.561 ms and 43.092 ms
respectively. Allocation, host input validation, upload, kernel and readback
for the full million-pair output take **109.257 ms canonical / 117.706 ms
mixed**, versus the 9.050/17.665 ms device kernels. Queries/supports already
exist at that timer's entry. JIT, search, scatter, material contraction and
complete application/provider preparation remain outside these measurements.
Keeping outputs on device is still necessary for practical consumers.

Durable evidence: `docs/work/fluxbridge-full-20261026.json` and
`docs/work/fluxbridge-full-reference-20261026.json`; raw seven Nsight reports,
CSV, generated native bodies, logs and source-preservation snapshots are in
`/tmp/fluxbridge-full-20261026/`. All measurement source hashes, 44 pre-existing
prototype/native hashes, prior fixture hashes, the original tracked dirty
patch and preceding plan content verify unchanged. GPU clocks/caches and
desktop activity were uncontrolled; small timing differences are not promoted
as optimizations.

**Next executable gate:** extract this validated regular-flux/edge-line stream
into a supported **production W6 fp64 device path**, with explicit request and
output-order mapping. Reproduce the native compact rotated-boundary failures
as regressions and validate the replacement's supported dispatch on CPU/CUDA
before promotion. Other kernels, arbitrary higher-degree requests and surface
triangle interactions need separate validation; this result does not certify
them. Cooperative ownership remains optional future work if profiling justifies
it; the present full contract already has zero local traffic.

The independent Stages 0–4 are not declared universally complete by this
prototype checkpoint. The accepted analytic-fp64 / numerical raw-fp32 policy
stays pinned. Existing 2D solid disk/cylinder and periodic-fibre applications,
web implementations, warpSPH and Core/Integrator are unchanged. No Stage 5
pressure/porosity integration occurred. Böttcher et al. (2025), arXiv:2504.07739v2
remains the principal porous-flow reference; transverse 2D fibre flow and
longitudinal flow remain distinct.

## Full-contract raw-fp32 accuracy and speed diagnostic (2026-10-26)

Following the full fp64 result, the user requested quantified fp32 errors and
then a best-case speed estimate despite those errors. Added isolated
`fluxbridge_shared_jk_full_precision.py`, `_precision_study.py` and
`_precision_perf.py` studies. The frozen full-fp64 source and earlier measured
prototypes/native library files remain unchanged. The new wrapper changes
runtime scalar/vector arrays, accumulators and math intrinsics to fp32;
geometry preparation remains host fp64 with coefficients stored as fp32.
There are no promoted device accumulators, clamped answers or relaxed budgets.

**Raw fp32 still fails on CPU and actual CUDA**, at the existing **1e-5 mixed
error** criterion `abs(actual-reference)/(1+abs(reference))`. On the 26 fixtures
plus 90 independent transformed queries, CUDA has **799 / 3,248 failed scalar
outputs**, affecting **87 / 116 pairs**. All outputs are finite. CPU fails
839 outputs on 88 pairs, with worst mixed error 2.204e-3.

The following CUDA rows select channels from the full compiled body, rather
than conflating separately specialized kernels with its output rounding:

| Full-kernel term group | Failed outputs / tested | Worst mixed error |
|---|---:|---:|
| Occupancy | 8 / 116 | 2.750e-5 |
| First and second value moments | 10 / 1,044 | 1.565e-5 |
| Kernel-gradient vector | 149 / 348 | 2.949e-4 |
| First gradient moments / Cov | 245 / 1,044 | 9.666e-5 |
| Symmetric Hessian | 387 / 696 | 2.103e-3 |

The worst independent CUDA Hessian example returns -0.08905935 instead of
-0.08677409: absolute error 0.00228526, about 2.63% relative error. Its mixed
error exceeds the gate by roughly 210 times. This is not double-epsilon-like
rounding. Separately generated values/gradient/Hessian specializations were
also measured and fail; their complete statistics are retained in the evidence.

Two additional 4,096-query samples compare to the validated fp64 body. Every
canonical pair fails at least one output; its worst mixed error is 9.108e-4.
The mixed sample has 3,678 failing pairs and worst error **1.302e-2**, including
a Hessian component whose true value **+0.00264979** becomes **-0.01040694**.
These are initially fp64-comparison diagnostics, not 8,192 new independent
fixtures. The worst canonical and mixed cases for **both CPU and CUDA** were
then checked independently using the signed-cone oracle: successive-order
changes meet 2e-12, and fp64 device results agree within 1.774e-11. The CUDA
mixed sign reversal is independently confirmed.

A replay promotes the identical stored fp32 geometry and query/support buffers
to fp64. On the canonical sample its worst error is only **4.139e-8**, versus
9.108e-4 for raw fp32, pointing to substantial loss of precision in evaluation.
On transformed boundary fixtures, quantized-geometry replay itself reaches
1.835e-4, so geometry coherence/conditioning also needs attention. These
replays diagnose arithmetic effects; they are not physical reference integrals
for an exact tetrahedron defined by the rounded vertices. Cancellation in the
expanded radial potentials/face reductions is a candidate to isolate, not yet
a proven single failing operation or a completed stability fix.

### Speed of the numerically failing kernel

The requested rough performance target is now measured, with the same logical
inputs and full 28-output contract for fp32/fp64. Values below use seven-sample
warm CUDA-event medians at **1,048,576 pairs, 64-thread blocks**:

| Workload | Raw fp32 (ms) | Raw fp32 billion pairs/s | Matched fp64 (ms) | Ratio |
|---|---:|---:|---:|---:|
| Canonical | **0.215648** | **4.862** | 8.546464 | 39.63x |
| Mixed, 256 tetrahedra | **0.441472** | **2.375** | 16.234144 | 36.77x |

CUDA-graph medians are **0.207296 ms / 5.058 billion pairs/s** canonical and
**0.430560 ms / 2.435 billion pairs/s** mixed; matched graph ratios are 40.17x
and 36.85x. At 131,072 pairs, raw-fp32 warm times are 0.047104/0.085024 ms
canonical/mixed; small batches retain launch/underutilization costs. The driver
also records 512/8,192 pairs and 256-thread blocks. No sub-percent block-size
preference is promoted from unlocked-clock measurements.

Nsight verifies **zero executed fp64 arithmetic** in the raw-fp32 kernel,
**108 registers/thread**, zero runtime local allocation and zero local LD/ST
sectors. At one million pairs, canonical/mixed achieved occupancy is
**32.09% / 32.32%**, SM throughput **69.17% / 64.27%**, and DRAM throughput
**23.40% / 11.67%**. Executed fp32 add/mul/FMA work is **8,042 / 7,249.85
FLOPs per pair**, yielding **39.13 / 17.63 TFLOP/s** in the profiled captures.
FMA counts as two; no artificial FLOP equivalence is assigned to roots,
transcendentals, control or integer work. These counts differ from fp64 because
the precision-specific compiler lowering differs. Pipeline FMA activity is
43.04% / 34.15%, with FMA-heavy activity 49.40% / 38.21%.

These rates are **device-resident interaction evaluation**, with every output
written. They exclude query localization, support search, convergence checking,
reduction, material assembly and transfers. Raw-fp32 arrays use 132 bytes/pair
(112 output + 20 input), or **132 MiB** at 1,048,576 pairs, with zero global
integration scratch. Allocation, host validation, upload, evaluation and
readback at 256-thread blocks cost **53.899 ms canonical / 52.475 ms mixed**;
static geometry is reused. This transfer timer is not the earlier complete
native evaluator path. GPU measurements were serialized; clocks, cache state
and desktop activity were uncontrolled.

Evidence: `docs/work/fluxbridge-full-fp32-20261026.json` contains both accuracy
reports, independent raw outputs, the four independent worst-sample checks,
matched performance reports and both Nsight summaries. Full output arrays,
profiles, generated native code and logs remain under
`/tmp/fluxbridge-fp32-full-20261026/`. All pre-existing prototype/native hashes,
measurement hashes, tracked dirty changes and preceding plan content verify
preserved. The accepted analytic-fp64 / numerical-fp32 policy and production
promotion gate remain unchanged. If analytic fp32 work resumes, the executable
stability gate is to isolate radial-potential/face-reduction cancellation and
rounded-geometry consistency while retaining the independent 1e-5 checks.
No 2D solid/fibre, browser, warpSPH/Core/Integrator or Stage 5 code changed.

## Precision investigation checkpoint — fp32 instructions, protected core (2026-10-26)

At the user's request, **commit `b2f4cf4`** checkpoints the independent native
FluxBridge implementation, required mesh3d reference dependency, fixtures and
all preceding shared-edge studies/findings. See
`docs/work/fluxbridge-checkpoint-20261026.md`. Unrelated dirty paper, scene and
browser work remains outside the commit. Fresh CPU native/profile smoke tests
passed **37 tests**, with one CUDA-only skip and four expected raw-fp32 failures.
This section records the subsequent precision research; it does not promote a
new production method or enter Stage 5.

### What the fp32 ablations show

Added isolated `fluxbridge_shared_jk_floatpair.py`, `_floatpair_study.py`,
`_stability.py`, `_stability_study.py` and `_stability_perf.py`. The committed raw
fp32 and accepted fp64 bodies remain byte-for-byte unchanged. Nine variants
are checked on CPU/CUDA against the same 26 fixtures and 90 independent
transformed queries, with the unchanged **1e-5 mixed-error gate**:

| Variant | Failed independent outputs, CPU / CUDA | Worst mixed error, CUDA |
|---|---:|---:|
| Raw fp32 control | 839 / 799 | 2.103e-3 |
| Compensated coefficient sums | 869 / 769 | 3.262e-3 |
| Compensated products, coefficient residuals and sums | 711 / 695 | 2.066e-3 |
| Two-float arithmetic, ordinary fp32 log/atan seeds | 96 / 90 | 2.554e-4 |
| Two-float arithmetic and range-reduced log/atan | 31 / 31 | 1.835e-4 |
| Above plus coherent split-float static geometry | **0 / 0** | **1.680e-6** |
| Protected core, ordinary fp32 final tensors | **0 / 0** | **1.673e-6** |
| Cheaper residual formulas; explicit inlining | **0 / 0** | **1.673e-6** |

The final row summarizes the separately recorded `floatpair-fast` and
`floatpair-inline` variants; both pass. Each total covers 3,248 scalar outputs
per device. These ablations identify contributions from **recurrence/reduction
arithmetic, elementary-function seeds, and coherence of rounded geometry**.
Compensating only the final coefficient sums does not repair the implementation.

The protected core uses about 48-bit working precision represented by **two
fp32 words**, including generation-time coefficient residuals, error-free sum/
product components, corrected division/square-root residuals, and range-reduced
log/atan series. Static host-fp64 geometry is split into high/low fp32 arrays,
so the cached edge/face frames retain their near-orthogonality. Query and support
arrays and final 28-output tensors remain ordinary single-word fp32. The final
moment accumulators also use ordinary fp32. **No device fp64 arithmetic is
introduced.** This is a paired-fp32 precision experiment, not a claim that the
original raw single-word arithmetic now passes.

Pointwise arithmetic and elementary functions have separate CPU/CUDA checks:
8,192 inputs, seven operations, three helper variants, all passing a **2e-12
mixed-error check on the reconstructed two-word values**. Worst observed error
is about 1.02e-13. Inputs cover varied finite normal magnitudes/quadrants; the
log helper is private to positive normal `1+z` inputs. These helpers are not
advertised as a generic IEEE arithmetic library or a proof over all ranges.

### Distinguish arithmetic error from input information loss

The two 4,096-pair samples retain both reference interpretations. Against the
original fp64-valued queries/supports, the final variant passes the complete
canonical sample but still exceeds 1e-5 in **10 scalar outputs on eight mixed
pairs**, worst 2.003e-5. Those inputs were rounded when copied into the fp32
query/support arrays; the original-input differences remain explicitly recorded.

Against reference integrals at the **actual fp32 query/support values**, every
output in both samples passes on CPU and CUDA. Worst mixed error is
**1.797e-7 canonical / 1.400e-6 mixed** for the narrower final core. This is a
separate diagnostic of the provided fp32 input values, not a relaxed budget or
silent replacement of the original 116 independent fixtures. Those fixtures
already pass against their unchanged original references.

Five difficult cases at the rounded inputs also have independent signed-cone
references, converged to the existing **2e-12 successive-order target**, including
the worst remaining mixed arithmetic error (index 2655) and the worst original-
input quantization error (index 1131). The independent result confirms the
1.400e-6 arithmetic error at index 2655. Four cases use orders through 128/160;
the difficult index 1131 required order 512. These are validation-only rules,
not quadrature in the device implementation.

### Performance and code-generation evidence

All following warm timings use the **same full 28 outputs**, 1,048,576 pairs,
256-thread blocks and seven-sample CUDA-event medians, with serialized execution:

| Workload | Raw fp32, failing (ms) | Protected fp32 core (ms) | Matched fp64 (ms) | Core million pairs/s |
|---|---:|---:|---:|---:|
| Canonical | 0.215200 | **3.826176** | 10.114208 | **274.1** |
| Mixed 256 tetrahedra | 0.430016 | **5.889856** | 16.818913 | **178.0** |

The protected implementation is **2.64x / 2.86x faster than the matched fp64
controls**, but **17.78x / 13.70x slower than failing raw fp32**. This is a
precision-control baseline with a substantial remaining speed gap. At 131,072
pairs, all variants are also measured in the evidence. Timings use unlocked
clocks/caches and share the desktop GPU; do not compare minor differences with
previous checkpoints as isolated speedups.

A compiler issue was also isolated: grouping these large experimental native
bodies in one Warp module changes inlining and local allocation depending on
which other variants are loaded. Each ablation now has its own explicit unique
kernel module and the same precision/compiler options. Explicitly inlining the
two-float operators then gives a consistent **254 registers/thread, zero runtime
local allocation and zero measured local LD/ST sectors**. The non-inlined fast
variant still has 253 registers and 464 bytes runtime local allocation. Earlier
non-isolated captures/timings remain in `/tmp` as intermediate evidence and are
superseded by the final unique-module runs.

Nsight confirms **zero executed fp64 arithmetic** in the final protected core.
Canonical/mixed achieved occupancy is **16.67% / 16.67%**. Executed fp32
add/mul/FMA counts are **75,366 / 67,763 FLOPs per pair** (FMA=2), compared with
8,042 for the canonical raw-fp32 control. About 47,178 / 42,268 are float adds.
The pair representation's arithmetic and dependency/register cost is now a
measured limitation, even after eliminating local traffic. Profile replays take
3.010/4.573 ms, separately from the warm medians above; their SM throughput is
37.91%/43.45%, with FMA activity 34.31%/37.09%. No equivalence is assigned to
transcendental, division, control or integer work in these FLOP counts.

Per-pair inputs/output still occupy **132 bytes**, or 132 MiB at 1,048,576 pairs,
with zero global integration scratch. Split static geometry uses **816 bytes
for one tetrahedron / 184,416 bytes for 256**, equal to the fp64 geometry
footprint. Separately measured static preparation/upload is 1.747/43.879 ms.
Fresh allocation, input validation, upload, evaluation and readback take
**59.733/60.560 ms** canonical/mixed; static geometry/JIT/search/reduction/material
assembly are excluded. CUDA-graph output is bitwise identical to the ordinary
launch for both million-pair workloads.

Evidence: `docs/work/fluxbridge-fp32-stability-20261026.json`, with final CPU/CUDA
ablations, arithmetic checks, unchanged-reference and actual-input statistics,
five independent hard-case checks, matched timing reports, two final profiles,
static/transfer memory and graph evidence. Exact generated native sources,
output arrays, compiler variants and raw profiles are retained in
`/tmp/fluxbridge-fp32-stability-20261026/`. All prior committed source hashes,
measurement hashes, unrelated tracked changes and previous plan content verify
preserved. The frozen analytic-fp64/numerical-fp32 production policy is unchanged.

**Next executable precision gate:** replace cancellation-prone expanded radial
potential/face divided differences with well-conditioned local or polynomial-
basis evaluations. Use this paired implementation as a precision control,
protect fewer live quantities, and retain the same independent 1e-5 checks and
matched CUDA counters/timings. Selected device-fp64 operations remain a last-
resort candidate after these fp32 alternatives; none was added here. Production
promotion and the native compact boundary regression fix remain separate gates.
Existing 2D solid/fibre and browser implementations, warpSPH/Core/Integrator and
Stage 5 integration remain untouched.

## Quick native-fp64 sanity check of the paired core (2026-10-26)

Before further fp32 reformulation, the user requested a sanity check of the
paired code using fp64. Added isolated `fluxbridge_shared_jk_fp64_sanity.py`
and `_fp64_sanity_study.py`. **The two-word arithmetic is collapsed to native
double operations**, preserving the same split-fp32 geometry, fp32 queries and
supports, fp32 final tensor accumulation/output, identical device buffers and
28-output contract. This isolates the arithmetic core instead of conflating it
with a different geometry, output footprint or provider. The frozen paired and
accepted fp64 implementations remain unchanged.

Two controls distinguish the arithmetic representation from elementary seeds:
`native64-series` retains the same log/atan range reductions, polynomial degrees
and collapsed high/low coefficients; `native64-library` uses standard fp64
log/atan2. The former unpacks an IEEE-double mantissa instead of a two-float
mantissa for log range reduction. Neither is a paired-double/double-double test.

Actual CUDA, 1,048,576 pairs, 256-thread blocks, seven-sample warm event medians:

| Core | Canonical (ms) | Mixed (ms) | Registers/thread |
|---|---:|---:|---:|
| Paired fp32 control | **2.978784** | **5.425056** | 254 |
| Native fp64, same seed formulas | 8.763968 | 18.806463 | 162 |
| Native fp64, standard math | 10.495168 | 17.796448 | 160 |

The same-seed native-fp64 replacement is **2.94x / 3.47x slower**. All variants
pass the unchanged 116 independent CPU/CUDA cases at 1e-5. Million-pair outputs
agree with the paired control within **2.008e-7 canonical / 4.177e-6 mixed**;
all graph replays are bitwise stable. Final fp32 accumulation/output is retained,
so these checks do not certify a double-output 1e-10 contract.

Canonical Nsight captures confirm **zero local LD/ST sectors** for both native
fp64 variants, about **16.6% achieved occupancy**, and **85.8–86.0% fp64 pipeline
activity**. Executed fp64 arithmetic is **10,225 FLOPs/pair** for the same-series
variant and 10,306 for library math (FMA=2), versus the paired core's approximately
75,366 fp32 arithmetic FLOPs/pair. Thus fewer registers and substantially fewer
arithmetic instructions still do not compensate for this device's fp64 compute
throughput. The canonical profile captures are 8.781/8.788 ms; the standard-math
warm timings fluctuate, so no strong ranking between the two fp64 seed choices
is inferred. Clocks/caches/desktop activity remain uncontrolled and GPU runs
serialized. All timings exclude preparation, search, assembly and transfers.

**Result:** replacing the whole paired core with native fp64 is not a speed
shortcut here. This does not rule out a few carefully selected fp64 operations
later, and does not establish which individual paired operation dominates.
The next precision gate remains narrowing/reformulating the cancellation-prone
radial reductions. No production precision policy or dispatch changed.

Evidence: `docs/work/fluxbridge-fp64-sanity-20261026.json`, with exact generated
native bodies, logs and profiles under `/tmp/fluxbridge-fp64-sanity-20261026/`.
All previous source hashes, unrelated tracked changes and preceding plan content
verify preserved; warpSPH/Core/Integrator and Stage 5 remain untouched.

## Paired/native-fp64 rerun after reported GPU contention (2026-10-26)

The user reported contention during the preceding measurements. **The timing
and utilization evidence in this checkpoint supersedes the preceding sanity
check's measurements.** Kernels, source hashes, geometry, seeds, buffers,
1,048,576-pair workloads and 256-thread blocks are unchanged. No precision or
production dispatch changes were made.

Ran three serialized rounds, canonical then mixed, with seven ordinary-launch
CUDA-event samples and seven graph samples per variant per round. The table
uses the median of the three ordinary-launch medians; parentheses give their
observed range. Timings exclude preparation, search, assembly and transfers.

| Core | Canonical ms (range) | Mixed ms (range) | Canonical / mixed million pairs/s |
|---|---:|---:|---:|
| Paired fp32 | **2.967 (2.966–2.971)** | **4.527 (4.522–4.550)** | **353.4 / 231.6** |
| Native fp64, same seed formulas | 8.791 (8.772–9.178) | 17.512 (17.506–17.512) | 119.3 / 59.9 |
| Native fp64, standard math | 8.794 (8.793–8.795) | 16.401 (16.400–16.617) | 119.2 / 63.9 |

Graph medians of round medians are respectively 2.953/4.520 ms,
8.801/17.492 ms and 9.158/16.617 ms canonical/mixed. Paired mixed throughput
improves from the preceding 193.3 to **231.6 million pairs/s**; the canonical
paired result is essentially unchanged. Same-seed native fp64 is now
**2.96x / 3.87x slower** than paired, so the wholesale replacement conclusion
still holds. The standard-math graph measurements vary more than ordinary
launches; retain both rather than selecting whichever is faster.

Six new Nsight captures, taken only after timing runs exited, confirm:

- Paired canonical/mixed: 254 registers/thread, 16.67% achieved occupancy,
  zero fp64 arithmetic, zero local LD/ST sectors; 75,366/67,763 executed fp32
  FLOPs/pair and **26.22/15.59 TFLOP/s** in the separate profiler captures.
- Same-seed native fp64: 162 registers/thread, 16.63%/16.55% occupancy,
  **85.84%/86.01% fp64 pipeline activity**, zero local LD/ST sectors;
  10,225/9,397 fp64 FLOPs/pair and **1.221/0.590 TFLOP/s** in the captures.
- Library native fp64: 160 registers/thread, 16.65%/16.60% occupancy,
  86.04%/86.15% fp64 pipeline activity and zero local LD/ST sectors.

FMA counts as two FLOPs; these counts do not assign FLOP equivalents to
transcendentals, integer/control operations or division. Nsight replay durations
remain separate from the event timings. No active compute processes appeared
in the pre/post GPU checks; 200 ms GPU telemetry is saved. There was no exclusive
GPU reservation or clock locking, and desktop activity remains possible.

Every million-pair run repeats finite-output, paired-output agreement at 1e-5
and bitwise graph-stability checks; all pass. The prior independent accuracy
suite is retained without rerunning unchanged numerical code. All source hashes
and the entire pre-existing dirty patch verified unchanged before appending
this checkpoint. Evidence: `docs/work/fluxbridge-fp64-rerun-20261026.json`;
raw logs, event samples, telemetry, generated bodies and six profiles:
`/tmp/fluxbridge-fp64-rerun-20261026/`.

**Next executable step remains unchanged:** reformulate the cancellation-prone
radial reductions and narrow protected precision, using this paired core as the
control. Do not promote a whole-core native-fp64 replacement or enter Stage 5.

## Shorter protected fp32 elementary seeds (2026-10-26)

Implemented four new seed candidates in
`scripts/studies/fluxbridge_shared_jk_shortmath.py`, with reproducible accuracy,
additional stress and sustained graph-batch drivers (`_shortmath_study.py`,
`_shortmath_stress.py`, `_shortmath_batch.py`). These are isolated study kernels;
production dispatch and the frozen paired reference remain unchanged.

The existing range reductions bound atan(t)/t to z=t^2 in
[0,(sqrt(2)-1)^2], and atanh(u)/u to z=u^2 in
[0,((sqrt(2)-1)/(sqrt(2)+1))^2]. Generated Chebyshev interpolants with mpmath
at 80 decimal digits and froze their coefficients as paired-fp32 constants.
The evidence includes the generator, full high-precision coefficients, estimated
approximation errors and 10,001-point high-precision grids for each polynomial;
these are numerical checks, not rigorous uniform-error certificates.

- `strict`: atan/log use 9/6 coefficients instead of the original 17/10.
  Device elementary mixed errors are at most 8.1e-15 on the 8,192 seed samples.
- `balanced`: 8/5 coefficients. Worst seed errors are 1.05e-12 for log and
  1.91e-13 for atan. This candidate also passes the interaction gates.
- `short` and `minimal`: 7/4 and 6/4 coefficients. Both pass the original
  116 independent cases, but **fail** the broad mixed sample at actual fp32
  input values: 3/6 failing outputs, worst 2.385e-5/2.132e-5. They are rejected
  and excluded from performance comparisons. Passing only the fixtures would
  have missed these regressions.

Both accepted candidates and the unchanged control pass CPU and actual CUDA:
116 independent cases, worst mixed error **1.674e-6** at the unchanged 1e-5
budget, plus the 4,096 canonical and 4,096 mixed samples using references at
actual fp32 query/support values. The existing ten failures when comparing the
mixed sample to *unrounded* input values remain input-representation error;
no such failure occurs against the actual-input reference.

An additional deterministic seed (28491) supplies **65,536 random + 16,384
near-face interactions** on the 256 transformed tetrahedra. All 28 channels
pass against the analytic fp64 reference at actual fp32 inputs, worst **2.347e-6**.
Independently checked the worst random and near-face cases using signed-cone
volume integration with exact long-double radial integration and successive
96/192/384-point face rules. Convergence is **3.147e-13 / 1.492e-12**;
all three kernels pass 1e-5 against those independent values.

Ordinary seven-sample launches were measured in three serialized processes per
workload, but varied enough to warrant a sustained comparison. The additional
protocol captures **32 sequential launches per graph**, warms every graph, then
rotates variant order through three rounds of seven CUDA-event batch timings.
The following are median per-launch times of the three round medians, at
1,048,576 interactions, 256-thread blocks and identical preallocated buffers:

| Seed core | Canonical ms | Mixed ms | Canonical / mixed million pairs/s |
|---|---:|---:|---:|
| Frozen paired control | 3.311 | 5.053 | 316.7 / 207.5 |
| Strict shorter seeds | 3.075 | 4.530 | 341.0 / 231.5 |
| Balanced shorter seeds | **3.048** | **4.432** | **344.0 / 236.6** |

Balanced throughput increases **8.63% / 14.01%** against the matched control.
Its round-median ranges are 3.020–3.051 / 4.409–4.436 ms; control ranges are
3.300–3.318 / 5.038–5.087 ms. These sustained batches are a different timing
protocol from the preceding individual-launch rerun; retain the matched control
and do not infer a speedup by mixing absolute rates across protocols. Clocks,
power and desktop activity remain uncontrolled. Every ordinary run repeats
finite/output-agreement checks at 1e-5; all ordinary and batch graph outputs are
bitwise stable. Preparation, search, assembly and transfers are excluded.

Six serialized Nsight captures confirm zero fp64 arithmetic and zero local
LD/ST sectors for all three variants. Balanced reduces executed fp32 arithmetic
from **75,366 to 67,932 FLOPs/pair canonical**, and **67,763 to 61,061 mixed**
(about 9.9%); strict uses 68,814/61,851. FMA counts as two, with no FLOP equivalents
assigned to transcendentals, division or integer/control work. Balanced profiler
throughput is 25.70/15.94 TFLOP/s, with SM throughput 37.31%/42.20%.
**Registers remain 254/thread and achieved occupancy 16.67%**: this does not
solve register pressure or establish a generally accurate raw-fp32 analytic path.

The final counter captures followed a corrected temporary diagnostic helper:
`profile.py` accidentally shadowed Python's standard-library module and spawned
two stalled profiler trees during independent-oracle startup. Only those owned
jobs were stopped; renaming the helper to `collect_counters.py` restored normal
operation. No interrupted capture is used. All final GPU measurement jobs ran
serially under the existing GPU lock.

Evidence: `docs/work/fluxbridge-shortmath-20261026.json`; raw arrays, generated
native bodies, ordinary and sustained timing samples, independent checks and
profiles: `/tmp/fluxbridge-shortmath-20261026/`. All previously existing source
hashes, the pre-existing dirty patch and earlier plan content verify preserved.
No warpSPH/Core/Integrator, ordinary solid/fibre application or browser code
changed; Stage 5 remains deferred.

**Next executable precision step:** isolate the Hessian R face reduction before
changing all five radial reductions. Its polynomial is R(r)=phi(r)-1, with
phi(r)=(1-r)^8(32r^3+25r^2+8r+1) and
phi'(r)=-22r(1-r)^7(16r^2+7r+1). Use this factorization to derive a conditioned
divided-difference/shifted-basis form for the same analytic edge reduction,
first as a local fp32 precision probe, then inside the shared six-edge kernel.
The short/strict seed controls bound the elementary-function contribution;
retain independent and broad actual-input gates before narrowing more paired
arithmetic. Selected fp64 operations remain a last resort.

## Register pressure, occupancy and lifetime experiments (2026-10-26)

The user requested continued investigation, explicitly questioning the paired
core's register count and occupancy. **Yes: 254 registers/thread exceeds the
160–162 registers in the matched native-fp64 controls** (the earlier full-fp64
body uses 188). Registers are counted as 32-bit words; native double takes two
words, but paired arithmetic also carries correction/normalization temporaries.
The baseline native-fp64 and paired kernels nevertheless occupy the same block
allocation step at the previously used launch size.

Nsight's measured hardware/launch limits resolve this: **65,536 registers,
1,536 threads / 48 warps per SM**. Paired 254 registers round to 256 allocated,
while native-fp64-series 162 rounds to 168. At 256-thread blocks both permit
only one resident block: 8/48 warps = **16.67% theoretical occupancy**. Lowering
register use from 254 to 192 alone at that launch size does not add a block.
With 128-thread blocks the fp64 control reaches **24.59% achieved occupancy**
instead of 16.63%, but its canonical profile remains 8.787/8.784 ms: the fp64
pipeline is already the limiting resource. More occupancy is not universally
useful across the arithmetic implementations.

Added isolated experiments in `fluxbridge_shared_jk_lifetime.py` and accuracy,
launch-sweep, sustained-batch and stress drivers. Every variant starts from the
accepted `balanced` paired seed body; no device-fp64 arithmetic is introduced.

- `serial`: consume one incident face at a time. The log seed is shared, while
  the algebraic edge stream/potentials may be recomputed. This shortens face-state
  lifetimes; it retains all 28 final channels and the exact edge-line limit.
  **Zero-w faces must still consume compact edge potentials**: pruning their
  whole contribution is incorrect, as the initial prototype's fixtures exposed.
  The corrected implementation consumes both faces.
- `staged`: place the 26 raw tensor accumulators in the existing caller-output
  buffer through volatile references, then snapshot before writing final channel
  layout/scaling. No extra scratch array is allocated. This lowers some register
  lifetimes but loses performance; the combined serial/staged variant also loses
  to moderate-cap controls. Neither is recommended for dispatch.
- Explicit caps on the unchanged paired body: 192, 160, 128, 96, 80 and 64
  registers. The cap changes compiler allocation/scheduling, not precision.
  Tested selected serial/staged caps as additional controls.

Four distinct mathematical bodies pass the unchanged CPU gates. Actual CUDA
validates **14 body/cap combinations** against the 116 independent cases and
both 4,096-case actual-input samples, with no failures at 1e-5. The original
unrounded-input mixed discrepancies remain separately reported. Selected
control, serial and 160/128/96/64-cap bodies also pass **81,920 extra random and
near-face cases**, worst actual-input error 2.347e-6. Their two worst-case inputs
match the earlier independently converged signed-cone oracle inputs exactly;
reused those oracle values and confirmed every selected body passes, without
repeating completed quadrature. Million-pair launch sweeps check finite output,
1e-5 agreement with control, bitwise stability across block sizes, and bitwise
graph replay. All checks pass. The 128-register body is bitwise identical to
control in the million-pair sweeps.

Swept 32/64/128/256-thread blocks, 12 initial configurations and three lower caps,
serially, for both million-pair workloads. Then warmed 32-launch graphs and
rotated four configurations through three rounds of seven CUDA-event batches.
Times below are medians of round medians, **per full-28 interaction launch at
1,048,576 pairs**, with identical preallocated buffers:

| Configuration | Registers | Block | Canonical ms / million pairs/s | Mixed ms / million pairs/s |
|---|---:|---:|---:|---:|
| Balanced paired control | 254 | 256 | 2.854 / 367.4 | 4.150 / 252.7 |
| Same body, cap 160 | 160 | 32 | 1.924 / 545.0 | **2.924 / 358.6** |
| Same body, cap 128 | 128 | 256 | **1.591 / 658.9** | 3.016 / 347.7 |
| Serial faces, uncapped | 239 | 256 | 2.112 / 496.4 | 3.826 / 274.1 |

The 128-register cap gives **1.79x canonical / 1.38x mixed** sustained throughput;
cap 160 gives 1.48x / 1.42x. Mixed timing varies: control round medians span
4.142–4.776 ms, cap 128 2.892–3.187 ms, cap 160 2.920–3.452 ms. Matched-round
speedups retain the gain (cap 128 1.38–1.50x mixed; cap 160 1.38–1.42x mixed).
Do not mix absolute rates from different runs or timing protocols. Clocks,
power and desktop activity are uncontrolled. No automatic cap/launch dispatch
or production precision policy changed.

Fourteen new Nsight captures quantify the occupancy/spill tradeoff. The table
uses canonical captures; mixed occupancy is similar:

| Paired configuration | Achieved occupancy | Local allocation bytes/thread | Local L1 sector bytes/pair (LD+ST) | fp32 TFLOP/s | SM issue utilization |
|---|---:|---:|---:|---:|---:|
| Control, block 256 | 16.67% | 0 | 0 | 25.55 | 37.21% |
| Cap 160, block 32 | 24.89% | 40 | 72 | 37.83 | 54.99% |
| Cap 128, block 256 | **32.99%** | 160 | 1,508 | **46.34** | **68.24%** |
| Cap 96, block 128 | 39.75% | 344 | 4,349 | 41.86 | 64.78% |
| Cap 64, block 256 | 60.64% | 784 | 13,885 | 19.87 | 30.71% |

Local allocation metadata is **not** transferred bytes/pair; local L1 request
sectors are **not** DRAM bytes. Cap 128's measured DRAM throughput is only 4.18%
of peak canonical, versus 68.37% for cap 64. The latter is slower than control
despite its much higher occupancy. Eligible warps/scheduler rise from **0.50 to
1.71** with cap 128; active issue utilization rises from 38.59% to 70.88% on the
per-scheduler active-cycle metric. Cap 128's executed arithmetic barely changes:
**67,932 to 67,938 fp32 FLOPs/pair** canonical, 61,061 to 61,066 mixed; FMA=2.
No fp64 instructions execute. This directly demonstrates a latency-hiding gain
with moderate cached spill traffic, rather than a reduction in numerical work.
Serial faces use 239 registers and 16 bytes local allocation, remain at 16.66%
occupancy and improve canonical throughput despite almost unchanged executed
FLOPs (68,187/pair); shorter live ranges/scheduling also matter.

A five-size sweep (8,192 to 1,048,576 pairs, fixed block 256) checks applicability.
At 8k–32k, cap 128 is about 1% slower than control; it has no useful occupancy
advantage on the underfilled grid. Graph speedup at 131k is **1.38x canonical /
1.24x mixed**, growing to **1.84x / 1.46x at 1M** in that separate sweep. Doubling
524k to 1M multiplies time by about 1.83–1.97 across control/capped workloads,
consistent with the expected large-grid linear work. Serial faces are the
faster tested choice at the two small sizes. These measurements motivate a
future workload-aware choice, not an unconditional cap default.

Evidence: `docs/work/fluxbridge-register-lifetime-20261026.json`, preserving all
round samples, scaling, 14 selected profiles, precision results, independent
hard-case rechecks and exact source versions. Raw evidence:
`/tmp/fluxbridge-registers-20261026/`. The initial cap-list extension and a batch
metadata initialization fix changed only study-source versions; the earlier
successful versions are archived and hash-checked. Initial failed prototype
logs are retained separately and are excluded from accepted results. All
pre-existing source hashes, earlier plan content and unrelated dirty work verify
preserved. warpSPH/Core/Integrator, browser and ordinary solid/fibre applications
remain unchanged; no Stage 5 work was performed.

**Next executable step:** use the 128/160-register capped controls as performance
baselines while conditioning the Hessian R face reduction identified above.
Narrow protected radial state at the same 1e-5 independent/actual-input gates,
then check whether the source reformulation retains the added active warps with
less spill traffic. Keep serial faces as a small-grid/lifetime control; do not
pursue output staging or lower caps simply to maximize occupancy.

## Priority adjustment: ordinary solid-boundary kernel set (2026-10-26)

The user asked about fp64/fp32 triangle/tetrahedron kernels limited to ordinary
warpSPH boundary applications. A read-only consumer audit confirms this is a
better next specialization target than carrying all 15/28 FluxBridge outputs
through ordinary walls. **No new kernel implementation or timing is claimed by
this contract audit.** Existing precision/dispatch and solver sources are intact.

Actual sources: warpSPH `boundary/provider.py`,
`modules/analyticBoundary/wallTerms.py`, `modules/iisph/boundaryIntegrals.py`,
and this repository's `scene/provider.py`, `scene/fused.py`, `edge/precision.py`
and `edge/warpfused.py`. The wired solver provider is currently **2D**; 3D
particle/tetrahedron kernels are the matching independent-library extension,
not an already wired 3D solver provider. Here triangles mean 2D area elements
and tetrahedra mean 3D volume elements; 3D surface triangles have a separate
surface-integral contract.

The current analytic wall provider uses **Wendland C2/C4**, rather than the W6
profile used for the recent full-28 performance/precision research. Its existing
`lean=True` contract supplies lambda, G and Cov. Cov's full interface duplicates
symmetric entries; its independent content is only three scalars in 2D or six
in 3D. The core bundles should be explicit:

| Bundle | Quantities | Independent 2D / 3D scalars |
|---|---|---:|
| Density | lambda = integral_solid W | 1 / 1 |
| Constant-field pressure/continuity | lambda, G = integral_solid grad_x W | 3 / 4 |
| Lean solid wall | lambda, G, Cov = integral_solid (y-x) tensor grad_x W | **6 / 10** |
| Optional first value moment | m1 = integral_solid (y-x) W | +2 / +3 |

Keep query-relative coordinates, physical support scaling, solid orientation,
normal sign, complement/background terms, exact edge/vertex limits and the
current provider conventions explicit. Reconstruct duplicate Cov entries only
at the legacy interface. Hydrostatic/linear-gradient extrapolation contracts
with Cov; it does not require a new general FEM/second-value-moment/Hessian
basis. Optional m1 remains available for the existing XSPH request. Normalized
versus physical units must be cross-checked against the current provider, rather
than inferred from similarly named channels.

The full WCSPH provider additionally requests **separate radial groups** for
wall-viscosity weight, cover vector and tensile/shifting gradient. These remain
explicit opt-in kernels/bundles, not mandatory pressure/continuity work. The
wall-viscosity weight must not be replaced by a generic kernel-Hessian trace.
No porosity, resistance/material fields, degree-nine products or solver state
belongs in this set.

Existing foundations: the production 2D fused provider already selects a lean
group and its fp32 route uses stable Chebyshev plans; monomial fp32 is explicitly
rejected. The experimental `fluxbridge_shared_jk_gradient.py` already provides
the 10-independent-scalar 3D W6 family in fp64. Neither fact certifies a new
raw-fp32 C2/C4 lean kernel, nor transfers the recent full-28 timings to it.

**Next executable implementation priority:** explicit dtype factories for
C2 first, then C4, producing the density and lean solid bundles for 2D triangles
and 3D tetrahedra. Keep fp64 analytic reference kernels and fp32 conditioned
paths independently testable without changing Core's process-global types.
Generate only the selected radial reductions, recurrence degrees and local
accumulators. Share edge geometry/J-K seeds within each requested bundle and
avoid the general term interpreter and per-pair primitive scratch. Try ordinary
fp32 at the unchanged actual-input error gate; use the accepted conditioned
numerical route or narrowly protected arithmetic where required. Do not assume
that fewer outputs alone cures monomial cancellation.

Validate CPU/CUDA, fp32/fp64, random/rotated/sliver and support/edge/vertex limits,
physical scaling and legacy-provider agreement. Benchmark each bundle against
the existing triangle/tet paths with matched buffers, static preparation and
end-to-end costs, then register/launch sweeps and actual Nsight counters. Retain
ordinary solid disk/cylinder and periodic-fibre regressions. Defer optional WCSPH
radial groups until the lean set is independently correct and measured. Further
general full-28 tuning is secondary to this application-specific gate.
warpSPH/Core/Integrator and Stage 5 pressure/porosity integration stay untouched.

Audit evidence/source hashes: `docs/work/fluxbridge-solid-contract-20261026.json`.

## First dedicated C2/C4 solid-simplex kernels (2026-10-26)

Committed the preceding precision/register research and consumer contract audit
as **617b4c5** before starting this implementation. Unrelated dirty work was
excluded. The first independent-library kernel set is now implemented and tested;
this checkpoint does not claim solver/provider integration or completed WCSPH
auxiliary groups.

### Implemented entry points and scope

New `fluxbridge/solid.py` exports `solid_functions` and
`prepare_solid_geometry` through `warpSPHBoundaries.fluxbridge`. Explicit factories
cover **2D triangles / 3D tetrahedra, Wendland C2 / C4, fp64 / fp32**, with three
separate bundles:

- `density`: lambda only, one scalar.
- `pressure`: lambda and G, 3/4 scalars in 2D/3D.
- `lean`: lambda, G and symmetric Cov, 6/10 independent scalars.

Outputs are caller-owned `[channel,pair]` arrays, in physical support units.
Both a reusable Warp device `element` and a one-thread-per-pair launch kernel are
available. No FEM fields/gradients, second value moments, Hessians, material
fields, interpreter or per-pair primitive scratch is present. The 3D generator
shares its J stream across two incident faces, generates only p/S face reductions
and the lean compact edge potential, and stops at the C2/C4 degree required by
the selected bundle. Exact edge-line limits and regular exterior flux are retained.
The 2D generator directly integrates the triangle's three edges, sharing radial
powers, seeds and reductions. General `pow` calls were replaced by shared integer-
power recurrences; this materially improved fp64 2D timings.

Static geometry is prepared once on the host, then uploaded: 9 vec3 records per
triangle or 26 per tet, plus common edge topology. No four FEM basis gradients
are stored. fp32 preparation first rounds the actual vertex coordinates to fp32,
then computes host-fp64 geometry from those coordinates. The protected path
stores hi/lo residuals for static geometry; queries/supports/output remain fp32.
Preparation rejects degenerate geometry, including degeneration after input
rounding, and rejects non-representable fp32 coordinates. Existing moving-body,
complement/background and Scene adapters are unchanged and are not newly wired.

Example of the tested fp32 lean tetrahedron entry point:

```python
from warpSPHBoundaries.fluxbridge import solid_functions, prepare_solid_geometry

api = solid_functions(3, kernel="w4", precision="float32", bundle="lean")
g = prepare_solid_geometry(vertices, device="cuda:0", precision="float32")
# q: api.vec array (vec3), h: api.real array, ids: int array
# out: caller-owned api.real array of shape [10, pair_count]
wp.launch(api.kernel, pair_count, inputs=[q, h, ids, g, out], device="cuda:0")
```

**Precision selection is explicit and deterministic:** fp64 defaults to ordinary
analytic fp64; fp32 defaults to protected two-float arithmetic with no device fp64.
`conditioning="raw"` explicitly selects the native-fp32 diagnostic. It is not an
accepted broad-range precision path. Protected arithmetic is migrated from the
validated research helper; its normalization is generated for the actual profile
and dimension, with no W6 normalization assumption.

### Numerical evidence and tests

The public defaults pass **59 tests with actual CUDA available**, including
package layering, plus the CPU-only invocation's **35 passes / 24 CUDA skips**.
Independent fixture coverage is 6 triangle and 26 tetrahedron cases per C2/C4
profile (64 case/profile combinations), across all three bundles and both dtypes
on CPU/CUDA. Budgets remain **1e-10 fp64 / 1e-5 fp32 mixed error**. Tests cover
orientation changes, physical support scaling, dtype coexistence, exact edge/
vertex and near-plane cases, every output being written, and invalid geometry.

Additional 8,192 transformed queries per dimension/profile compare against the
analytic fp64 kernel at the **actual fp32 vertex/query/support values**, on CPU
and CUDA. Protected paths pass throughout. The million-pair canonical/mixed
runs repeat those actual-input checks; protected worst errors stay below
**4.8e-8**. These broad comparisons are not independent quadrature for each point;
independent tabulated fixtures establish the separate reference gate.

**Raw native fp32 remains open even for C2.** On the mixed million-pair workloads,
lean worst mixed errors are approximately 2.15e-5 / 2.46e-5 for C2 in 2D/3D, and
2.79e-4 / 2.26e-4 for C4. C2 passes the small fixtures but not the broad gate.
Fast raw timings are retained as diagnostic targets, never substituted for the
accepted protected results or selected by provider dispatch. No tolerance was
relaxed and no device-fp64 fallback was added.

### First CUDA performance measurements

All rates below are **million complete particle/simplex interactions per second**
for the lean bundle (6/10 independent scalars), 1,048,576 pairs, 256-thread blocks,
seven-sample warm CUDA-event medians and preallocated buffers. Geometry, input
values, IDs and output contracts are matched; static preparation and transfer/
search/assembly costs are excluded. These are first timings, not a multi-round
performance acceptance claim; clocks/power/desktop activity are uncontrolled.

| Profile / arithmetic | 2D canonical | 2D mixed | 3D canonical | 3D mixed |
|---|---:|---:|---:|---:|
| C2 native fp64 | 1,245 | 551 | 186 | 81 |
| C2 protected fp32 | **6,047** | **3,118** | **536** | **272** |
| C4 native fp64 | 1,125 | 526 | 176 | 75 |
| C4 protected fp32 | **5,666** | **2,879** | **451** | **264** |

Every graph replay is bitwise stable. Density/pressure timings, exact per-bundle
allocation sizes, raw-fp32 diagnostic timings/errors and initial geometry
preparation/upload costs are recorded separately in the evidence. Complete
preparation/evaluation/readback, peak driver-memory and provider/application
measurements remain future gates for this new family.

Eight serialized C4 lean Nsight captures show:

- 2D protected: **96 registers**, 32.37%/32.17% achieved occupancy canonical/mixed,
  8,158/7,871 executed fp32 FLOPs per interaction, zero fp64 operations and zero
  local LD/ST sectors. Native fp64 uses 94 registers and 1,177 fp64 FLOPs in the
  canonical capture. Raw fp32 has 47 registers / 881 fp32 FLOPs, but fails accuracy.
- 3D protected: **179 registers**, 16.67% occupancy, 51,670/46,990 fp32 FLOPs,
  zero fp64 operations and zero local LD/ST sectors. Native fp64 uses 134
  registers / 7,542 fp64 FLOPs canonical. Raw fp32 uses 77 registers / 5,488 fp32
  FLOPs and fails accuracy. The 3D protected seed/correction cost remains large.

FMA counts as two FLOPs; no equivalent costs are assigned to division/root,
transcendental or integer/control work. Register caps are available as explicit
factory experiments but are not yet selected/tuned for this family. The preceding
full-28 register-cap results must not be confused with these uncapped lean timings.

Evidence: `docs/work/fluxbridge-solid-kernels-20261026.json`; native bodies,
arrays, logs and profiles under the raw directory recorded there. The evidence
archives successful source versions and verifies exact native-body identity
for the retained 3D measurements across the 2D-only power refactor and for
explicit arithmetic modes across the default-selection/API export change.
All prior research hashes, earlier plan content and unrelated dirty work are
preserved. No warpSPH/Core/Integrator, browser or existing ordinary solid/fibre
implementation was changed, and no Stage 5 work was performed.

**Next executable step:** condition the raw-fp32 C2/C4 radial/face reductions using
these dedicated protected kernels as the reference and the raw kernels as cost
bounds. First target G/Cov, where the broad failures remain; retain actual-input
and independent gates. Apply the already motivated 128/160-register experiment
to the new 3D protected family as a separate performance control. Provider
agreement/assembly and optional m1, wall-viscosity, cover and tensile bundles
remain explicit subsequent gates; solver wiring stays deferred.
