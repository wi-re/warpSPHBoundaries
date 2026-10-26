# FluxBridge coverage and v1 operator contract

Inventory recorded 22 October 2026, before the native implementation. The
pre-existing dirty tree is preserved in `/tmp/fluxbridge-kickoff/dirty-tree.tar.gz`
and its SHA256 manifest. No changes in the main warpSPH checkout are required.

| Representation | Existing arithmetic | Existing advertised outputs | FluxBridge work needed |
|---|---|---|---|
| 2D polygons, triangles | arbitrary precision and native analytic plans; new conditioned boundary quadrature | numerical route supports degree-nine moments/kernel derivatives and raw-fp32 material/transpose checks on CPU/CUDA | analytic raw conditioning deferred; universal quadrature bounds remain open |
| 2D strip / plane | tier3 tables, Scene curvature functions; support-clipped analytic and explicit numerical adapters | degree-nine Cartesian moments and kernel derivatives through order two on CPU/CUDA | old tables/curvature remain separate; analytic raw fp32 conditioning deferred |
| 2D disks / cylinder cross sections | `edge/disk.py` circle reference; `warpdisk`, tables, tier4 series; native `fluxbridge.disk` circle quadrature | baseline circle plus explicit interior-volume route with degree-nine/second-derivative requests, periodic arrays and streaming on CPU/CUDA; old lookup/series contracts are restricted | full-contract lookup/series parity and retained curvature comparisons |
| 2D boxes | `scene/box.py` lookup; `fluxbridge.box` exact rectangle adapter | exact polygon route supports baseline and degree-nine moments/derivatives | old lookup still needs full-contract acceptance; raw recurrence fp32 remains experimental |
| SDF / curvature | `scene/curvature/wp_func.py`, sampled guard | twelve channels, fp64; local-model VJP | sampled reconstruction derivatives and fp32 acceptance |
| 3D tetrahedra, closed skins | arbitrary precision, browser and native analytic plans; explicit tetrahedral boundary quadrature | degree-nine moments/second kernel derivatives and P0–P3 fields on CPU/CUDA; tetrahedral numerical route passes fp64/raw-fp32 material and transpose gates | ordinary analytic fp32 conditioning deferred; general closed skins have no new numerical route |
| 3D open triangles | arbitrary precision and native analytic plans | area moments, ambient gradient/Hessian, P0–P3 FEM on CPU/CUDA | ordinary analytic fp32 conditioning deferred |
| 3D sphere / plane / ball series | `mesh3d/radial.py`, browser; native radial and planar providers, explicit planar shell quadrature | radial sphere/plane baseline on CPU/CUDA; analytic/numerical planes/bands and explicit sphere-shell quadrature add degree-nine weighted derivatives; finite shell gates pass near-roundoff fp64 | ball series and retained curvature/SDF model ports; analytic sphere remains baseline-only |
| material operators | independent polynomial resistance provider plus existing solid Scene reactions | common work, fused contractions, transpose, supplied-affine response, dissipation and local implicit mobility on CPU/CUDA | pressure/porosity and solver reconstruction remain Stage 5 |

Browser fp64/WGSL results establish neither native Warp accuracy nor CUDA speed.
Existing disk/solid Scene APIs remain available independently of porous materials.

## v1 contract

`fluxbridge.Request(alpha, derivative=())` requests
`∫ y^alpha ∂x_derivative W_h(source-query) dμ`, with **kernel-only**
derivatives of order 0–2. `y=source-query`; `h` is support radius, not smoothing
length. `μ` is 2D area, 3D volume, or explicitly 3D surface area. Units are
`length^(measure_dimension-kernel_dimension+|alpha|-len(derivative))`.
Index tuples specify derivative order; Hessian trace defines the kernel
Laplacian. Complete translated-moment derivatives subtract
`alpha[j] M_(alpha-e_j)`; complete global-field contractions also differentiate
the query-centred coefficients. No shape adjoint is implied.

The finite requested set is an output mask: omitted channels are never silently
substituted. Coefficients use Cartesian monomials, degree ≤9; straight-sided
geometry only. Four normalized kernels: cubic spline, Wendland C2/C4/C6.
No regularized rational viscosity profile or kernel power is implied by W's
Hessian. These require independent profiles and acceptance.

Separate fp32 and fp64 factories avoid process-global Core precision changes.
Plan and geometry preparation are outside launches/capture. Connectivity and
arrays are explicit. Exhaustive regular flux retains every boundary edge/face.
Localization may prune **whole elements** with disjoint support AABBs; never
individual regular-flux faces. This conservative strategy covers containment
without a winding classification and preserves discontinuous element fields.
Per-query interactions and reductions have fixed ascending element order.

Initial mixed error is `abs(value-reference)/(1+abs(reference))`, after support
scaling to unit support: fp64 ≤1e-10; fp32 ≤1e-5 for well-conditioned baselines.
Adversarial failures remain recorded limitations, not hidden clamps or a claim
of universal accuracy. Field expansion conditioning is separate from geometry
integration. Runtime validity checks reject nonfinite inputs, nonpositive h,
degenerate simplices, invalid requests and stale adjacency.

## Acceptance boundaries

The inventory is finite; implementation is not completion of every row. Native
special forms, sampled SDF adjoints, hierarchical solid-angle acceleration, robust predicates,
CUDA acceptance, shape adjoints and a complete pressure discretization require
separate evidence. See the main plan's dated evidence section for actual results.

## Paired-fp32 and affine-work checkpoint

The explicit `float32_compensated` mode passes the current CPU fixture gate;
raw `float32` remains experimental. It uses fp32 pairs, not fp64 device math.
Element-local coordinates, high/low coefficients, paired scratch and cached
profile outputs are a distinct prepared low-level layout; callers should use
`Evaluator`/`FieldEvaluator` and their reported scratch width. See the operator
document for the per-interaction coordinate contract and CPU cost.

`AffineResistanceMaterial` adds supplied fluid-gradient work and its transpose
load. Its common-work/rigid-rotation/tensor-covariance checks pass independently
of a fluid solver. The point-only model is still noncentral. These checks do
not validate an unspecified SPH gradient, pressure coupling or timestep scheme.

## Native 3D special-form baseline checkpoint

`fluxbridge.radial.RadialEvaluator` now supplies every one of the **28 3D
baseline channels** for an exact solid plane/half-space or sphere, in fp64 and
explicit paired fp32. Subsets and arbitrary outward normals are supported.
Higher combined moment/derivative degrees are rejected; this does not certify
degree-nine fields, curvature, small-ball series or sampled-SDF methods.

The independent 70-digit shell-quadrature fixture set has 112 cases and 3,136
channels. Additional checks compare the plane with a closed mesh, sphere with
the retained radial oracle, and complete derivatives with finite differences.
Near-centre and tangency conditioning use interval Bernstein products. Cubic's
outer Laurent derivatives use a documented 64-term beta primitive with a
bounded numerical tail; the geometry and kernel are not replaced by a model.
The later factored-profile update removes the initial raw-fp32 W6 gate misses;
those tests now assert the original budget without expected-failure waivers.
The later GPU checkpoint supplies actual CUDA runtime acceptance separately
from compilation. See the latest main-plan checkpoint for measurements and
reproducible commands.

## Native 2D disk/box checkpoint

`fluxbridge.box.BoxEvaluator` constructs exact, optionally rotated rectangles
and reuses the analytic polygon functions, adjacency and field contractions.
It is a geometry adapter, not a new closed form or certification of the old
box lookup. Cartesian polynomial fields retain the degree-nine request contract;
quadrilateral FEM shape functions are not supplied.

`fluxbridge.disk.DiskQuadratureEvaluator` is an explicitly numerical circle
method with all **15 baseline channels**, including second moments and Hessian.
`fluxbridge.disks.DiskArrayEvaluator` adds standalone additive disk arrays,
optional host or native CPU/CUDA BVH search, unique periodic images and deterministic reductions.
Disks remain ordinary solid geometry without any porous resistance law. An
independent 70-digit volume-shell oracle checks the circle boundary algorithm;
the combined disk/box fixtures contain 128 cases and 3,180 channels.

The circle route passes the finite fp64, raw-fp32 and paired-fp32 gates. The box
route passes fp64 and paired fp32; four additional tests preserve its raw-fp32
failures. Neither a full disk lookup nor higher-order disk field products are
advertised. CUDA compilation of the circle and reduction functions is separate
from unavailable runtime acceptance. See the main plan's latest checkpoint.

## Infinite planar geometry checkpoint

`fluxbridge.planar.PlanarEvaluator(kind, dimension, requests, ...)` supplies
infinite half-spaces (`plane`) and finite-thickness infinite bands (`strip`)
in both 2D and 3D. It uses exact support clipping and the shared analytic
recurrences, including exterior flux corrections. The weighted contract covers
every monomial through degree nine and kernel-only derivatives through order
two for W2/W4/W6/cubic; no axial-baseline restriction applies to this provider.
This is a geometry adapter, not a new table, axial primitive or SDF model.

Independent 70-digit references cover 136 cases / 22,820 channels. The 3D
axis-aligned high-degree oracle integrates shell angular polynomials and radial
powers, independently of mesh flux. It also agrees with the earlier adaptive
plane-shell baseline oracle. Rotated baseline cases use the retained arbitrary
precision mesh reference; extra selected rotated degree-nine requests are tested.

Fused consumers may evaluate and contract the reported plan and scratch directly.
No infinite-domain FEM basis, automatic polynomial material packing or pressure
law is supplied. fp64 and paired-fp32 CPU/CUDA fixture gates pass; the analytic
raw-fp32 misses remain explicit and deferred. Extremely thin bands remain an
acceptance gap. The explicit numerical 2D route passes raw-fp32 CPU/CUDA gates.

## Conditioned raw-fp32 polygon route

`method="conditioned_boundary"` is an explicit numerical alternative for 2D
polygon volumes, rectangles and support-clipped planes/bands, retaining every
requested degree-nine moment and second kernel derivative. It combines radial
cumulative profiles exactly before ordinary-fp32 Chebyshev evaluation, then
integrates complete boundary fluxes in angle coordinates. The tested default
is 16 Gauss nodes per dyadic panel; 16/32 convergence is recorded separately.
It does not silently replace or certify the analytic raw recurrence.

The same device function supplies fused P0–P3 field contractions and their
transpose, degree-nine material products and supplied-affine work. Per-element
local coordinates and native whole-element disjoint rejection keep exhaustive
and localized polynomial assembly consistent. Intersecting elements retain all
regular-flux faces, including exterior tails. 120 independent frozen cases /
6,880 channels per precision pass on CPU and CUDA; the 3D numerical extension
and universal error bounds remain open. See the API derivation and latest
plan checkpoint for exact limits, evidence and costs.

## GPU execution checkpoint

Actual NVIDIA RTX PRO 6000 Blackwell execution passes the combined regression
suite: **522 passed, 32 expected analytic-fp32 failures, no skips**. This includes
174 passing CUDA-parameterized tests and five passing CUDA-only graph tests;
the expected failures are the same 16 precision-family misses on each device.
The user accepts analytic fp64 and the explicitly numerical fp32 alternative
for now; repairing the analytic raw-fp32 basis is deferred. Neither error budget
nor method defaults changed. Device reports and timing limitations are recorded
in the plan's latest checkpoint and `/tmp/fluxbridge-gpu-evidence/`.

## Practical 3D planar quadrature checkpoint

`PlanarEvaluator(..., dimension=3, method="shell_quadrature")` supports exact
infinite-plane/band geometry and the full degree-nine / derivative-order-two
contract in ordinary fp32/fp64. It integrates spherical shells split at kernel
knots and material planes, with polynomial-exact angular rules. The default
16-node radial rule is polynomial exact for the Wendland family in exact
arithmetic; cubic outer Hessian terms also require numerical radial integration.
Independent CPU/CUDA evidence covers 19,280 channels per precision, worst
fp64 error 4.241e-15 and raw-fp32 1.436e-6, with 16/24 convergence checks and
extra rotated degree-nine mesh references. Exactness and measured speed apply
to this planar provider, not arbitrary tetrahedra or general material actions.

Matched fp64 GPU full-mask plane/band costs drop from 2,651 / 2,662 ms to
13.01 / 9.43 ms for one query × 2,200 outputs. Device output parallelism and
sample storage are part of that implementation tradeoff. See the latest plan
checkpoint and `/tmp/fluxbridge-quadrature-evidence/` for host/event timings,
complete evaluation, memory, regression results and remaining limits.

## Tetrahedral quadrature and real material checkpoint

`method="tet_quadrature"` now supplies the full 3D volume moment/jet contract
to the general evaluator, polynomial fields, transpose and point/affine material
consumers. It uses combined analytic radial primitives and numerical angular
integration over oriented face sectors, split at kernel-sphere intersections.
CPU/CUDA fixtures cover 10,640 channels per precision: worst fp64 mixed error
2.634e-15 and raw-fp32 7.684e-7. Query-specific prepared rules compare two
orders and reject unresolved convergence; this is not a universal error proof.

For the matched one-interaction P3 material workload, fp64 CUDA action drops
from 2,460 to 7.44 ms (W2) and 3,216 to 7.44 ms (W6). Rebuilding, checking and
applying the quadrature batch costs 51.0 / 54.9 ms, versus analytic 2,483 /
3,257 ms. The numerical prepared batch uses about 6.33 MB in fp64, and its
comparison rule needs additional temporary storage. FP64 host panel preparation
and memory therefore remain production scaling issues. This route does not
cover surfaces, non-tetrahedral skins or paired precision. See the latest plan
checkpoint and `/tmp/fluxbridge-tetquad-evidence/` for complete evidence.


## Search, shared contractions and bounded fp32 checkpoint

Optional `search="device"` supplies whole-element Warp CPU/CUDA BVH count/scan/
fill and deterministic periodic CSR, verified against exhaustive enumeration.
It currently returns host CSR for the batch interface; a GPU-only query pipeline
is not advertised. Field/material actions share translated monomial integrals
and retain explicit same-dtype FMA correction workspace. Frozen batch actions
remain graph-compatible; `iter_evaluate`, `evaluate_streamed` and compiled
material `apply_streamed` are separate bounded host drivers with per-chunk
preparation/readback and owned-host outputs.

The explicit `nodes=8` tetrahedral option passes 11,880 independent channels per
precision on CPU/CUDA; raw fp32 accepts 8 throughout those fixtures, while fp64
usually promotes to 16. Default remains 16. The real P3 packed fp32 gate stays
1e-5, and actual CUDA action/graphs, complete scaling, fresh-process allocator
usage and generated-code audits are recorded in the latest plan checkpoint.
The requested mixed-wall web delivery in 2D/3D has native/independent and
live/offline comparison evidence; pressure/porosity coupling remains Stage 5.


## Weighted sphere checkpoint

`SphereShellEvaluator` now supplies the full Cartesian degree-nine / second
kernel-derivative request mask for exact solid-sphere geometry, using explicit
numerical shell integration. Native fp32/fp64 CPU/CUDA checks cover 12,320
independent reference channels per precision, worst mixed errors 4.094e-6 /
5.499e-15. The reference's rational angular-cap calculation cross-checks 784
baseline channels against the older radial-shell oracle. Composed calls,
complete translated-moment derivatives, physical units, graph replay and
physical-overflow rejection pass. Default order 24 compares with 32, may promote
and rejects unresolved/resource-limited inputs; no universal bound is implied.
Analytic `RadialEvaluator` is still baseline-only. Sphere arrays/curved FEM/shape
adjoints are not added. See the plan for complete measured CUDA preparation,
allocation, readback and degree-mask/query-count costs and the distinct source
snapshots for the broad regression versus final storage guard.


## Weighted disk/cylinder cross-section checkpoint

`WeightedDiskEvaluator` and explicit `DiskArrayEvaluator(method="volume_quadrature")`
now expose every degree-nine / second-kernel-derivative request in 2D. The
existing circle/default array contracts remain baseline-only. Independent
whole-circle regular-flux fixtures precede the native interior-volume port and
cross-check 540 baseline channels against a separate volume-shell oracle.
The 2,724-channel CPU/CUDA finite gate passes at 1e-14 fp64 / 1e-5 fp32;
exact full-support polynomial/IBP checks cover all requests. Optional order-8
fp32 retains convergence/promotion and passes below 1.898e-6. The final pure-fp32
native module has no fp64 instructions, using bounded-angle fp32 trig and
Neumaier sums after explicit host-fp64 geometry preparation.

Periodic host/device search, exhaustive image sums, composed calls, CUDA graphs
and bounded streaming pass. Complete preparation/adjacency/readback, actual
CUDA allocator scaling and comparison sample provenance are recorded in the
plan checkpoint. No porous law, curved FEM basis, effective permeability or
longitudinal flow model is attached to resolved 2D solid fibres.
