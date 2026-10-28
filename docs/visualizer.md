# Boundary laboratory

A browser-based comparison workspace with an optional local Python provider for boundary integrals. The visual
language follows the sibling `warpSPHRemote` project: central linked viewports,
configuration at the left, model and display controls at the right. No frontend
build, Node runtime, CDN, external fonts, or additional Python dependencies are
required beyond this package's existing dependencies.

## Run locally

From an environment where `warpSPHBoundaries` is installed (this checkout uses
the `warp` conda environment):

```bash
conda activate warp
python -m warpSPHBoundaries.visualizer
```

Open <http://127.0.0.1:8765>. Use `--port 8766` to change the port. After an
editable reinstall, `boundary-lab` is an equivalent entry point. The server binds
only to loopback and serves only the bundled static directory. Live computation
is same-origin, with bounded request size and serialized evaluations to bound
working memory. Evaluations write no result files; native runtime/JIT caches
may be created.

## Explore and compare

- Start with square/two triangles, thin rectangle, editable triangle, or circle.
- Choose one to four distinct models. All views share geometry, kernel, support
  radius, sample coordinates, quantity, probe, and color scale.
- Change rectangle width/height/rotation, each triangle vertex, or circle radius
  (constant curvature `κ = 1/R`) and polygon segment count.
- Set kernel (Wendland C2/C4/C6 or cubic spline), support radius `h`, grid
  resolution (17–513 samples per axis, slider or integer override), and padding in support radii.
- Select mass, first/second scalar moments, kernel gradient components/magnitude,
  or the four first gradient moments. The twelve-channel ordering and units
  match the existing validation atlas. Coordinates themselves are physical;
  moments are normalized by powers of `h`, gradients multiplied by `h`.
- Click any viewport to place the linked probe and horizontal/vertical cut.
  Probe and cut values are bilinear interpolations of computed samples, not
  fresh point evaluations. Undefined samples remain missing.
- Select signed or absolute difference to the **first viewport**. This is a total
  difference between represented models, not a separated numerical/model
  error budget. View 1 retains its **raw field and its own scale**; other views
  and the main legend show differences. Put the desired reference in view 1. Raw results are never clipped to [0,1].
- Toggle triangle edges, geometry, coordinate grid, `d=±h` contours, support-band
  masking, and optional image smoothing. Contours interpolate the exact geometry
  distance, even when a model uses sampled SDF inputs. The circle's polygon mesh
  therefore has a separate approximate boundary inside the exact circle.
- Export raw computed results and configuration to JSON; reopen that JSON without
  a backend. Each field viewport can also export a PNG. Setup links encode the
  configuration in the URL hash and recompute with the selected provider.

Display-only changes do not recompute integrals. Numerical changes debounce live
computation; disable “Recompute on change” to stage expensive edits and evaluate
manually. A result uses its own configuration while a new request is pending,
so changing `h` or geometry never relabels old fields as new calculations.

## Model adapters and accuracy

`visualizer/compute.py` is independent of HTTP and rendering. The following table
describes the optional **Python** provider. Browser evaluation differs as described below.

| Model | Evaluation | Limits |
| --- | --- | --- |
| Triangle mesh | Sum `edge.np2d` closed-form triangle moments | Rectangle has exactly two triangles; circle uses an inscribed polygon fan |
| Exact polygon | `edge.np2d` moments of the entire represented polygon | “Exact” describes the formula; floating-point error remains; circle polygonization is model error |
| Built-in rectangle | Four-edge `edge.np2d` rectangle | Same numerical engine as mesh, **not** the scene's tabulated `BoxRep` backend; allows checking internal-edge cancellation |
| SDF distance only | Exact half-plane moment formulas with analytic or sampled signed distance and normals | Local model misses corners, thin-wall interactions, and curvature; ambiguous normals leave oriented moments missing |
| SDF curvature O(2) | Existing CPU `scene.implicitBodies.Tier3.curvature_moments` | Circle only and `R/h ≥ 2`; asymptotic order-two model, not an error certificate |
| Circle reference | `validation.atlas.disk_reference`, analytic radial integrals and angular Gauss quadrature | Configurable 8–64 nodes; refinement is empirical and not a certified bound |

The geometry distance is negative inside the **solid** and positive outside.
The planar solid at the boundary has mass 1/2, including at polygon corners in
the **local-plane model**. The actual rectangle mass at a corner is 1/4. That
discrepancy is expected geometry/model error and is useful to visualize.

Polygon adapters normalize in NumPy `longdouble` **before** division by `h`.
This avoids inconsistent float64 winding/atan branches near shared triangle
rays on this Linux platform. Platforms where `longdouble` is float64 show an
explicit note. API output rounds to JSON float64; Tier3 uses CPU float64.
Near-degenerate elements and known numerical failures of the underlying library
remain subject to the library's documented limitations. The viewer is exploratory,
not a replacement for the accuracy audits.

Sampled SDF mode creates a Cartesian lattice with spacing `sdf_spacing*h`,
bilinearly interpolates distance, and obtains normals/Hessians by finite
differences. For a circle, level-set curvature is transported back to the
boundary using `κ_boundary = κ_level / (1 - d κ_level)`. Near singular
reconstruction, the curvature/normal remains undefined. Circle curvature
variation is zero. The model does not yet implement arbitrary user SDFs,
nonconstant boundary curvature, or the scene production guard/fallback dispatch.

Colormap lookup tables were sampled from Matplotlib's Viridis, Magma, Inferno,
Coolwarm, and Cividis; its license is included with the static files.

## Static export and GitHub Pages

Export a self-contained static viewer and one precomputed experiment:

```bash
python -m warpSPHBoundaries.visualizer --export .tmp/boundary-site
# An exported result JSON or a plain configuration JSON can seed the example:
python -m warpSPHBoundaries.visualizer --export .tmp/boundary-site --config experiment.json
python -m http.server 8080 --directory .tmp/boundary-site
```

The export contains HTML, CSS, JS, WGSL, disk lookup tables, palettes/license, and `example.json`.
It uses relative asset URLs and works beneath a GitHub Pages repository prefix.
The published viewer is [Boundary laboratory](https://fluids.dev/warpSPHBoundaries/volume.html),
served from the repository’s `gh-pages` branch. The [2D interface](https://fluids.dev/warpSPHBoundaries/index.html?2d=1)
also computes directly in the browser.

Static mode loads an exported example when present. Both 2D and 3D geometry,
kernel, field and grid changes compute in browser workers without Python or Warp.
The 3D provider also uses WebGPU when available. All imports, probes, cuts, PNG
exports and comparison controls work on GitHub Pages.

## 2D browser integrals and the meaning of “exact”

The default 2D provider tries **WebGPU fp32**, with an explicit JavaScript
float64 fallback when no adapter or working pipeline is available. Select
**WebGPU · fp32** to require GPU computation, or **Browser · JavaScript float64**
for reference arithmetic. Explicit fp32 mode reports an error if unavailable;
it does not silently use float64. Both providers support mesh,
polygon, rectangle, local half-plane, curvature expansion and circle models,
all twelve stored channels and gradient magnitude, four kernels, and analytic
or sampled SDF inputs. The Python provider remains available locally. The connection badge, compute
note and exported results identify the engine actually used.

Every viewport states **Geometry / model** and **Integral evaluation** separately:
an analytically integrated half-plane is exact for its surrogate geometry but
still approximates the original solid. Quadrature over the original polygon or
circle introduces integration error without replacing its geometry. A circle’s
inscribed polygon adds a separate geometric approximation. A curvature series
is a truncated integral model, not the exact integral over a quadratic graph.
Python’s curvature tables also add interpolation error.

For the browser polygon path, the regular divergence flux for a degree-k
homogeneous moment is

```text
A_k(r) = r^(-k-2) ∫₀ʳ t^(k+1) W(t) dt
M_α = ∮∂Ω y^α (y·n) A_|α|(|y|) ds
G_i = −∮∂Ω W n_i ds
g₁,ij = δ_ij M₀ − ∮∂Ω y_i W n_j ds
```

Radial antiderivatives are polynomial. Edge and circle-arc integrals use Gauss
quadrature split at kernel knots, closest approach and geometric panels. The
outer radial tail is evaluated in powers of `1-r` to avoid cancellation near
support. Triangle sums and whole polygons use the same flux; gradient magnitude
is taken after summing vector components. These browser quadrature methods are separate from the analytic polygon/mesh
methods described below. Python polygon adapters also use closed-form edge
integrals. Refine the reference Gauss-node setting to assess quadrature convergence.

WebGPU assigns one invocation to each query and accumulates only the selected
quantity. Gradient magnitude accumulates the vector first. Flux primitives and
Gauss weights are prepared in float64, then uploaded in fp32. Compensated
summation reduces fp32 cancellation. Shared, oppositely oriented mesh edges
cancel algebraically before evaluation; this leaves the mesh integral unchanged.
Every circle/plane/curvature model and every field runs on the GPU, including
sampled-SDF finite-difference jets. It uses the same geometric models and Gauss
rules as the reference, rather than substituting occupancy samples or a raster
convolution. These quadrature methods remain numerical boundary quadrature; selecting an
analytic method instead uses the compensated fp32 shader described below.

The compute pipeline/device are cached across interactions. First use compiles
the shader and may take longer; per-model times include dispatch and readback.
Use float64 for very small differences and finely spaced sampled-SDF derivative
jets, where fp32 subtraction can amplify roundoff. The grid and sampled-SDF
bounds are identical in both browser providers.

In an 81² mass benchmark on Mesa llvmpipe/Vulkan (a **software CPU adapter**, not
a hardware GPU measurement), warm shader dispatch plus readback took about
8 ms for all three square models and 8 ms for the four circle models. These
measurements exclude JS result packaging and canvas rendering; first dispatch
also incurred about 1.9 seconds of driver compilation. Separately, Node float64
end-to-end compute time improved from 2.17 s to 0.77 s for the square and from
4.85 s to 1.56 s for the circle after skipping empty/full polygon supports and
avoiding unused kernel derivatives. These are local example timings, not
promises for another browser/device.

The actual `terms2d-compute.wgsl` is checked with the production JS uniform/table
layout through wgpu-native for all 13 fields, all four kernels, rotated polygons,
clockwise triangles, circle references, curvature and sampled SDFs. Tested
analytic-input comparisons use absolute/relative tolerances of 3–5×10⁻⁵ against
the float64 reference; sampled jets use 5×10⁻⁴. These are regression tolerances,
not certified bounds across all geometry and spacing choices.

The curvature correction to any selected integrand F is
`−κ/2 ∫t² F(t,−d)dt + κ²/8 ∫t⁴ ∂ₙF(t,−d)dt`, added to the local-plane integral.
The curvature quadrature method integrates these coefficients numerically; the
analytic curvature method evaluates the retained coefficients in closed form.
Both truncate the curvature expansion. The circle reference integrates the
original circle boundary numerically and has no certified numerical bound.

An integer override beside each resolution slider accepts 17–513 nodes per axis
in 2D and 17–129 in 3D, including even sizes and values beyond the slider’s usual
range. Invalid values are rejected; preset changes preserve a chosen resolution.
Grids grow quadratically/cubically, so large overrides increase compute time and
memory. Result imports allow up to 100 MB.

## 3D terms, browser computation and WebGPU

Open `volume.html`, including directly from a static HTTPS/GitHub Pages host.
The default compute provider is **Browser · WebGPU / JavaScript**. Geometry,
kernel, support, grid, model and term changes recompute locally in the browser;
Python, Warp and warpSPH are not needed. WebGPU computes on the device when an
adapter is available. A float64 JavaScript worker handles unavailable adapters,
explicit float64 requests. Analytic polyhedra and the small-ball series also
have compensated fp32 GPU implementations. The compute provider is
separate from the ray-casting renderer: either can fall back independently.
A simple static HTTP server is sufficient for development; no compute API is
required. WebGPU needs HTTPS or localhost and a supported adapter.

The optional **Local Python service** provider retains NumPy/SciPy evaluation,
including the existing voxel FFT model. Both dimension tabs can also run
without that service.

Curvature order (0/1/2) and small-ball order (0/2/4) are selectable; dedicated
presets put both models inside their stated radius domains. The term selector
contains 30 scalar selections: the 22-entry moment
layout (`M0`, `M1`, symmetric `M2`, `G0`, full `g1`), six symmetric kernel
Hessian entries, their trace (Laplacian), and gradient magnitude. These are
direct **kernel-integrand derivatives**, replacing finite differences of a
sampled mass grid. For radial scalar kernels `g1` is symmetric, so its full
stored matrix retains redundant off-diagonal entries. Fields use the same
normalized units in Python, JavaScript, WGSL, exported files and probes:
`M_alpha/h^degree`, `h G0`, `g1`, `h² H`, `h² Delta lambda`.

The **Equation** section follows the selected term and documents its integral,
normalization, kernel, derivative convention, model formulas and validity.
The derivation is [radial terms in 3D](derivations/radial-terms-3d.md).

| Model | Browser | Python | Mathematical / accuracy contract |
| --- | --- | --- | --- |
| Sphere polynomial integral | WebGPU f32 / JS f64 | f64 | Complete/partial shell angular moments; split at shell boundaries and cubic knots; 16-node Gauss exactly integrates the polynomial pieces in exact arithmetic |
| SDF local plane | WebGPU f32 / JS f64 | f64 | Half-space moment family from geometry distance and closest normal; local geometry error remains |
| SDF curvature O(2) | WebGPU f32 / JS f64 | f64 | Quadratic-graph expansion applied to each integrand; umbilic sphere only, R/h ≥ 2; higher geometry and finite radial quadrature errors remain |
| Small ball orders 0/2/4 | Compensated WebGPU f32 / JS f64 | f64 | Mean-value expansion `Va[F+a² Delta F/10+a⁴ Delta²F/280]`; R/h ≤ .3; origin, knot and support crossings stay missing |
| Solid analytic face-to-edge | Compensated WebGPU f32 / JS f64 | — | Exact integral for the represented tetrahedron/box in real arithmetic; floating-point roundoff remains |
| Solid support cubature | WebGPU f32 / JS f64 | — | Preserves sphere, box or tetrahedron geometry; support-centred spherical Gauss sampling, approximate sharp occupancy; quality 1/2 refines sampling |
| Voxel FFT quadrature | — | f64 | Linear zero-padded convolution with selected moment/derivative stencil; grid and occupancy sampling error; no renormalization or clipping |

Standard cubature uses 8 radial × 6 polar × 12 azimuthal nodes (576 per support);
refined uses 12 × 10 × 20 (2400). The cubic spline splits radial integration at
its knot, doubling those sample counts. Sampling refinement is empirical and
is not a certified error bound. The browser's cubature is distinct from the
Python voxel FFT discretization and from the arbitrary-precision analytic
`mesh3d` face-to-edge reference. No exact-mesh label is attached to cubature.

Sphere and box signed distances and normals are analytic. Tetrahedron distance
uses triangle and segment projections with an oriented face inside test.
Nearest-face ties use a deterministic normal; local-plane model error and
medial ambiguities are not repaired by this choice. Presets support independent
x/y/z tetrahedron dimensions. Arbitrary imported meshes, non-umbilic or varying
curvature controls, curved strands and plates are not implemented.

3D grids have 17–129 nodes per axis; the browser defaults to 33³. Up to three
models share geometry, support, axes and selected scalar field. Invalid models
show their domain restriction; they do not silently clamp curvature/radius or
substitute another representation. The small-ball valid domain may have holes,
which remain NaN in float32 buffers and transparent in rendering.

### Ray casting and transfer functions

The browser uploads field and signed distance to a filterable `rgba16float`
3D texture, then runs the bundled `volume.wgsl` full-screen ray caster. It uses
front-to-back emission/absorption integration, early termination, explicit
trilinear sampling, and ray-step-corrected opacity. Optional lighting estimates
the field gradient in the shader. No JavaScript graphics library, CDN, build
step, or `float32-filterable` optional GPU feature is required.

Drag any model viewport to orbit all cameras; scroll to zoom; reset the camera
with the toolbar button. Ray samples (64–768), optical density, render resolution,
gradient lighting, bounding-box/axis overlay, and support-band clipping are display
controls; they do not make Python requests.

Color is sampled from the five existing colormaps. Opacity is an independent
piecewise-linear curve with presets for boundary shell, signed/zero-transparent
differences, solid interior, linear ramp, and constant opacity. Drag knots or edit their numeric fields; click to
add a knot and right-click an interior knot to remove it (up to twelve knots).
Window min/max apply to color and normalized opacity coordinates. “Fit window”
uses the difference fields in comparison mode. The first viewport retains its
raw field with an independent scale and opacity mapping. The boundary preset makes full-support and zero-field
regions transparent. Signed quantities/differences default to a curve with zero
opacity at the centre of the symmetric window, so zero-error space stays
transparent; custom curves are retained. Changing ray-step count preserves optical depth rather
than multiplying opacity by the number of samples.

The 3D renderers show **volume only**. Slice-plane compositing has been removed
from both the WGSL and CPU ray casters, rather than hidden behind a default
setting. Old saved-view plane flags, slice opacity and “slices only” render modes
are ignored on import, so reopening a file cannot restore overlays. There are
no plane visibility or plane-opacity controls in the 3D rendering interface.

Three separate orthogonal **2D slice plots** retain physical axis offsets and
full colormap values independently of volume opacity. Clicking a slice moves
the linked crosshair and the other two cross-sections. Their offsets do not
change rendered volume pixels. Slice probes are trilinear interpolations of
raw float32 volume values. Ray spacing still resolves at least two samples per
voxel along the ray, including oblique views.

For browsers without a WebGPU adapter, the app explicitly reports a CPU worker
fallback. It uses lower image resolution and disables lighting, with the same
grid-aware ray spacing; data and slice sampling remain the same. WebGPU requires a supported
browser/device and a secure context (localhost or HTTPS). The preview browser
used during development exposed no WebGPU adapter. Its CPU fallback was tested
in the browser, and the **actual WGSL** was compiled and render-tested separately
through wgpu-native on Mesa llvmpipe/Vulkan; this is not a hardware performance
benchmark.

### Payloads, precision, and saved views

Dimension-3 responses use `schema_version=1`, `dimension=3`, `shape=[nz,ny,nx]`,
`order="zyx"`, and `encoding="base64-f32-le"`. The x index varies fastest. Every
model and the shared distance grid encode little-endian float32 buffers. This
gives a maximum three-model payload of about 46 MB at 129³, before JSON metadata. Axes are physical
coordinates; kernels use support radius h and 3D normalization.

Python/JavaScript evaluation is float64; WebGPU computation is float32; raw exported/probed data is float32; the GPU display
texture is float16. The viewer is therefore appropriate for model/geometry
comparisons, not studying differences at float64 epsilon. Camera, opacity knots,
colormap, display window, selected slice model, and slice offsets are saved
in volume JSON and restored on import. PNG export reads WebGPU pixels back from
the rendered texture. No volume computation is performed when reopening a file.

Export a 3D static site, optionally seeded by an exported volume (including its
saved view):

```bash
python -m warpSPHBoundaries.visualizer --export .tmp/boundary-3d-site --dimension 3
python -m warpSPHBoundaries.visualizer --export .tmp/boundary-3d-site --config boundary-volume.json
python -m http.server 8080 --directory .tmp/boundary-3d-site
```

A 3D export redirects the default entry point to `volume.html`. Relative asset
and worker URLs support a GitHub Pages repository prefix. Static 3D mode retains all camera/transfer/slice/comparison controls and
**live browser recomputation** for geometry, kernels and terms. The local Python
provider is optional. An exported Python voxel model switches to browser
support cubature for the next browser calculation; its saved values remain
unchanged until a new calculation succeeds.

Implementation follows the [WebGPU specification](https://www.w3.org/TR/webgpu/)
and [WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/).

## Verify

```bash
python -m pytest tests/visualizer tests/test_package_layering.py -q
node --test tests/visualizer/js/volume-math.test.mjs tests/visualizer/js/term-engine.test.mjs
node --check src/warpSPHBoundaries/visualizer/static/volume-app.js
```

Tests cover the 2D moment adapters, 3D normalization and scale invariance, sphere
closed forms, voxel convergence, Euclidean tetrahedron distances, gradient-axis
ordering, encoded payloads, static exports, trilinear sampling, all slice
orientations, transfer interpolation, camera bases, and missing data.

Optional offscreen WebGPU checks use `wgpu` (a development verification tool,
**not** an app dependency):

```bash
python -m pip install wgpu
python -m pytest tests/visualizer/test_gpu3d.py tests/visualizer/test_terms_gpu.py -q
```

These compile and execute the actual WGSL with a filterable 3D texture and test
opacity changes, ray-step stability, slice-offset changes, and gradient lighting.
The module skips when the optional library/adapter is unavailable.

The compute-shader tests execute the actual bundled WGSL against float64
reference values for all 30 selections, four kernels, and sphere/plane/curvature
models. Software Vulkan execution tests correctness, not GPU throughput.
The browser smoke check uses a plain static server with no Python API and
verifies new fields, live recomputation, equations, and the worker fallback.

Signed gradient components can cross zero or change sign. Zero-transparent
transfer functions can expose those zero sets, whereas missing/invalid samples
remain unavailable independently of the transfer function. Gradient magnitude
combines components before taking the norm. Local SDF normals can change across
nearest-face or medial boundaries; these are properties of the surrogate model,
not quadrature holes in the original geometry. The UI recommends full-color
slices or constant opacity to inspect this distinction.

### WGSL compiler portability

The 2D shader uses explicit validity flags, not NaN constants or NaN comparisons.
A finite placeholder stays separate from the validity bit through GPU readback;
JavaScript then restores missing samples as NaN internally and null in JSON.
Valid numerical zeros remain valid. This avoids Dawn/Tint’s rejection of a
constant NaN bitcast while preserving ambiguous-normal and invalid-curvature
samples. Numerical tests execute through wgpu-native/Naga; a second optional
Dawn/Tint test reproduces the old parser error and compiles all production
shaders and pipelines. Run it with:

```bash
npm install --prefix .tmp/dawn-check --no-save --ignore-scripts webgpu
node tests/visualizer/js/dawn-shaders.test.mjs
```

The test uses Dawn’s null adapter for compiler validation; numerical execution
is checked separately with the software Vulkan tests. The browser app needs no
Node package or additional runtime dependency.

### Analytic tetrahedra and boxes

The browser's `Solid · analytic face-to-edge` model evaluates the original
polyhedron using elementary edge antiderivatives, with analytic splits at
kernel support intersections. It covers all 30 quantities and uses the
[volume-to-face-to-edge derivation](derivations/tet-face-edge-chain.md).
The tetrahedron and box presets place this model first, followed by cubature
and the SDF approximation. Cubature quality affects only cubature, while grid
resolution controls field sampling for every model.

The analytic computation runs in compensated WebGPU fp32, or in a JavaScript
float64 worker when selected or when automatic GPU initialization fails. It has no
numerical integration rule. It is exact for the represented polyhedron in
real arithmetic; floating-point roundoff and cancellation remain. Export/probe grids
are float32 and renderer textures are float16. The independent Python mesh3d
reference remains the path for caller-controlled precision and rational
geometry predicates. The local Python visualization grid service does not
expose this analytic model; select a browser provider for live analytic grids.

The browser regression suite compares all channels and kernels against the
65-digit reference, plus constant-block overlap, simplex polynomial moments,
boundary queries, scale invariance, full/empty support and cubature-quality
independence. These are finite validation cases, not certified global error bounds.

### Browser variants and compensated fp32

Analytic polygon/strip and polyhedron WebGPU computations use fixed-profile
rolling edge recurrences, consuming each degree immediately instead of storing
endpoint primitive arrays. Equivalent axis fields share a cached pipeline;
new profile/output variants incur compilation on first use. Paired fp32 and
the arithmetic self-test remain active. Matched software-Vulkan measurements
show about 2–3× faster warm dispatch/readback for C2/C4 mass and gradient fields;
hardware GPU timing remains unverified. See the
[implementation and benchmark record](work/webgpu-rolling-edges-20261028.md).

`Available browser methods` lists each viewer's methods and geometry limits.
The 2D viewer exposes analytic polygon/mesh edges and boundary-quadrature
comparisons; analytic, quadrature and sampled-SDF local models; an exact
infinite-strip surrogate; and disk boundary quadrature, seven-invariant
Chebyshev lookup, curvature orders 0/1/2, and small-disk orders 0/2/4.
All listed 2D methods cover the thirteen displayed quantities. The disk lookup
includes second moments as well as mass, first moments and gradient moments.
3D exposes analytic tetrahedra/boxes, sphere integrals, local planes, curvature
orders, support cubature and small-ball orders for all thirty quantities.
Methods retain their geometry and expansion validity restrictions; missing
normals or invalid series samples are not replaced by zero.

The new analytic shader (`analytic3d-compute.wgsl`) serves analytic 2D polygon,
strip, half-plane and curvature-coefficient evaluation, analytic 3D polyhedra,
and 2D/3D mean-value series. It uses paired fp32 arithmetic throughout and
requires no float64 device feature. An explicit rounding step preserves the
compensated residuals against reassociation, and the arithmetic self-test runs
before the pipeline is accepted. High-order coefficients and elementary
functions use that arithmetic; the host combines radial profiles before
upload, and edge primitives are shared across their terms. This removes
quadrature error for analytic integrals/retained coefficients, not model or
series error. Backend labels on every view report the engine actually used.
The explicit WebGPU provider reports failures; automatic mode can use float64.

Disk lookup coefficients are generated offline by
`scripts/visualizer/build_disk_tables.py`: same radius/offset panel maps as
`edge.disktables`, extended to seven rotational invariants and four kernels.
The source is 32-node split circle-flux quadrature and the stored coefficients
are float32. Tables are approximations. A finite regression sweep against
48-node boundary quadrature has max absolute differences below 3e-7 for the
Wendland kernels and 8e-5 for cubic; these are not certified global bounds.
The shader avoids the native `asin` for the radius-panel mapping: the tested
software adapter's allowed inverse-trig error otherwise displaced the panel
coordinate enough to corrupt the lookup. A range-reduced convergent `atan`
series gives a more accurate map.

Native shader regressions use the actual production host layouts and execute
all fields/kernels, aligned boundary queries, box geometry, sampled local
curvature, disk lookup and expansion orders. Independent Python checks audit
analytic coefficients with arbitrary precision and scalar disk series with
`edge.tier4`; table approximation is checked separately from GPU arithmetic.
Dawn/Tint compilation is checked as a second WGSL implementation. Software
Vulkan timings are not hardware GPU performance measurements. The in-app
preview may expose no GPU adapter, in which case its UI check covers the
float64 fallback and the numerical shader checks remain native offscreen.

```bash
PYTHONPATH=.tmp/wgpu-check python -m pytest tests/visualizer -q
node tests/visualizer/js/dawn-shaders.test.mjs
```

## Mixed impermeable/permeable wall explorer

Open <http://127.0.0.1:8765/wall.html> or choose **Mixed wall** from either
existing viewer. The local service's `/api/wall` route evaluates the native
FluxBridge numerical or analytic operators, using an explicit CPU/CUDA and
fp32/fp64 choice. Analytic raw fp32 is withheld under the accepted conditioning
policy. Warp moments use the selected dtype; displayed polynomial/material
contractions currently use host float64 and report that provenance separately.
This is not a performance benchmark of the existing fused device material path.

The generator splits a finite-thickness wall exactly at the porous/solid
interface, then triangulates it in 2D or tetrahedralizes it in 3D. Geometry,
porous fraction, subdivisions, grading, orientation, position, support radius,
kernel, query grid, supplied affine velocities and anisotropic resistance are
independent controls. Higher resistance degrees use the explicit positive
profile `(1 + .25*xi(source))**p`, p=0..3, times a constant PSD tensor, not
arbitrary FEM controls. Circle/fibre presets retain impermeable polygonal
cross sections with explicit geometric approximation and no added effective drag.

Heatmaps, selected whole-element candidates, support footprints, sampled-query
probes and horizontal line plots share the same native data. The 3D data are
volume integrals sampled on XY/XZ/YZ coordinate slices; drawn tetrahedron edges
are projections. These are not surface integrals. Solid/porous/free occupancy
sums to one. First/second moments, gradients, Hessians and Laplacian use physical
units, without the older viewer's support normalization. Free-fluid complements
use exact full-support moments and preserve signed numerical values.

Compression displays the prescribed boundary continuity contraction
`int (u_fluid(query)-u_skeleton(source)) dot grad_query W`. Permeation displays
a supplied normal velocity and local implicit resistance response. Neither
computes pressure-driven flux, an incompressibility residual or the complete
porous density model. Drag, reaction, dissipation, weighted D and skeleton
products use polynomial contractions through degree p+2. Mobility is
`(I+dt*int D W)^-1`. Pressure/porosity and maintained solver integration stay
in Stage 5. Böttcher et al. (2025), arXiv:2504.07739v2 remains the porous reference;
ordinary two-component 2D fibre flow describes transverse response only.

Native queries are evaluated in bounded groups (64 in 2D, four in 3D), reusing
one static evaluator per configuration. In 3D, explicit host/device panel
preparation is selectable; both use fp64 geometry and the selected evaluation
dtype. This bounds each explorer batch without claiming library-wide GPU
streaming or CUDA-native search. Total/static setup, query preparation and
action/readback costs are exported separately. The display reports physical
units and per-unit-fluid-mass acceleration/reaction/dissipation. Saved snapshots
are explicitly labelled; their recorded CUDA backend does not imply live GPU
execution in the browser.
Live controls require the loopback service; snapshots reopen on a static site:

```bash
python -m warpSPHBoundaries.visualizer --wall --export /tmp/fluxbridge-wall-site
# Supply --dimension 3 and an optional --config file for a 3D slice.
python -m http.server 8080 --directory /tmp/fluxbridge-wall-site
# Open http://127.0.0.1:8080/wall.html
```

Current limits: no arbitrary imported wall mesh, periodic fibre cell UI,
membrane/surface-only mode, arbitrary FEM control editor, automatic error
separation panel, or maintained fluid solver overlay. Existing viewers and their
browser-only providers are retained independently.

The mixed-wall canvas uses equal physical scale on both axes and bounds the
actual generated geometry plus kernel support. A 3D slice depicts volume
integrals, not a surface measure. The 9×9 P3 CUDA export, including probes within
roundoff of wall faces, is verified in `/tmp/fluxbridge-nearplane-evidence/`;
its displayed compression matches the native JSON and unavailable region/term
combinations are labelled explicitly.

The mixed-wall view now offers host versus device whole-element search and a
**Compare support, refinement and precision** action. It compares exactly the
displayed probes and prescribed fields, showing maximum physical absolute and
mixed differences for the selected region/term. Straight-wall subdivision keeps
geometry fixed; circle/fibre segment refinement is explicitly a geometry-error
comparison. The fp64 comparison uses the same numerical method, while independent
high-precision native fixtures remain separate acceptance evidence. Comparison
metadata is saved with snapshots and reopens offline.

Use `--wall --compare --export DIR --config CONFIG.json` to produce a static
snapshot with these reports. `/api/wall/compare` uses the same same-origin and
serialized native-computation rules as `/api/wall`. Example evidence and browser
value checks are in `/tmp/fluxbridge-search-evidence/`. The live 2D panel and
exported 3D P3 CUDA panel have been inspected and their displayed values matched
to native JSON. Four optional legacy WGSL test modules require Python `wgpu`;
these are distinct from actual CUDA execution and the native-backed wall view.

The fraction control describes the **permeable portion of the wall**, not its
intrinsic porosity. A circular/fibre region has no single planar normal, so its
planar normal-response diagnostic is unavailable. The supplied local porous
response does not impose the impermeable wall condition; pressure/porosity
coupling remains the Stage 5 handoff. Arbitrary FEM-control editing, periodic
fibre-cell UI, imported skins/fibres and a distinct surface-only view remain
broader explorer extensions.


## FluxBridge browser-only mixed-wall port (23 October 2026)

`wall.html` now defaults to **Browser · WebGPU fp32**. It uses the same static
asset directory/export as the existing 2D and 3D laboratories. Live browser
computation requires a WebGPU-capable browser over HTTPS or localhost, but no
Python/Warp compute server. The explicit **Local Python / Warp** provider and
saved native/browser snapshots remain available. Unsupported adapters produce
an error; there is no silent native or precision fallback.

Export the complete static site without importing the native package:

```bash
python scripts/visualizer/export_static.py /tmp/fluxbridge-site
python -m http.server 8080 --directory /tmp/fluxbridge-site
# Open http://localhost:8080/wall.html
```

Upload that directory's contents to the existing GitHub Pages publishing source;
all imports, workers, shaders and profiles use relative URLs and support a
repository URL prefix. `.nojekyll` is included. The static browser assets were pushed to the existing `gh-pages` branch on
23 October, commit `ed350f2`: [Mixed-wall explorer](https://fluids.dev/warpSPHBoundaries/wall.html).
Only the static viewer and navigation links were published; the research working
checkout remains uncommitted. Deployment evidence is recorded in the plan.
The older `--export` / `--wall --export` commands still include native examples.
An assets-only export starts empty and computes when **Evaluate terms** is used.

The reusable `fluxbridge-plan.js` / `fluxbridge-compute.wgsl` core accepts
Cartesian volume moments through degree nine and kernel-only derivatives
through order two, with physical h scaling. The wall adapter requests baseline
moments plus degree p+2 scalar moments for its degree-0–3 resistance profile.
`fluxbridge-profiles.json` is generated from the exact native rational plans by
`scripts/visualizer/build_fluxbridge_profiles.py`, before conversion to fp32.
It introduces no separate hand-maintained kernel coefficients.

Geometry, whole-element AABB search and angular panel preparation run in a
JavaScript worker using Number arithmetic. The GPU evaluates the profiles,
monomials and quadrature products in fp32. A ten-limb integer accumulator sums
these rounded fp32 sample values before conversion back to fp32. This avoids
compiler reassociation invalidating Neumaier compensation; it does not remove
profile/product or quadrature error. No GPU fp64 or paired-float integration is
used. Regional reductions and the small material contractions use JavaScript
Number, explicitly identified in exported provenance.

The mixed wall GPU path now shares profile evaluations between requested
moments. Long sample ranges run as parallel jobs, whose integer accumulators
are merged exactly in a second compute pass. Host preparation avoids per-sample
vector allocations and reuses regional geometry between convergence rules.
These optimizations are automatic in Browser/WebGPU mode.

Both polygon edges and tetrahedral face/edge angular rules retain exterior
regular flux. The 3D near-plane path uses the native regular-potential Duffy
rule. Accepted/comparison orders start at 16/24, increasing by eight up to 64,
then reject unresolved agreement at a 1e-5 normalized regional-moment budget.
Two-order agreement is not an independent error certificate or a guarantee for
all material parameters. Refinement and exhaustive/localized comparisons are
available without the native service; browser fp64 comparison is outside scope.

Buffers are adapter-checked and batched: a prepared face group is capped at 8 MiB
of uploaded samples, with roughly 250,000 samples per dispatch batch. Reported
owned GPU peak bytes are logical buffer sizes (including output and readback),
plus exact-integer partial storage when jobs are split, not driver allocations/
context or total host memory. Job sizes adapt to the existing buffer/dispatch
limits; partial storage retains the five-storage-buffer requirement. Timings
also report the actual compute-pass count. Query-specific host arrays and
preparation can dominate 3D cost. The worker/device/pipeline persist between
successful requests; **Cancel evaluation** terminates the worker and cancels CPU
preparation, and a later request creates a fresh context.

Independent fixtures, host/native material parity, actual adapter execution and
remaining browser compatibility gates are recorded in `docs/plan-porous-coupling.md`.
This prescribed-state tool does not solve pressure, porosity or longitudinal
fibre flow. Böttcher et al. (2025), arXiv:2504.07739v2 remains its porous reference.


For the wall adapter, identical internal faces cancel within each material
region before preparation. Solid/permeable interfaces remain distinct. This
requires the common global polynomial integrand used by this adapter and does
not apply across arbitrary discontinuous material traces. The first hardware
measurements are about 108 ms for an 81-probe 2D example and 6.7 s for a 25-probe
3D example (degree three, both convergence rules, warm device/pipeline). These
single runs include host preparation/readback and are not general performance
promises. The subsequent
[mixed wall optimization record](work/webgpu-mixed-wall-20261028.md) reports
matched software-Vulkan evaluations at roughly 1.7–2.3× faster in 2D and
4.7–5.4× in 3D, with identical measured material terms. Those measurements
include preparation/upload/readback and do not establish hardware GPU speedups.


### Chrome/Linux adapter discovery

A supported GPU in native Dawn/Vulkan does not guarantee Chrome will expose a
WebGPU adapter. On this workstation, the installed Chrome is 156.0.8078.47,
the session reports X11 (`DISPLAY=:1`, no Wayland display), and the NVIDIA
Blackwell driver is 610.43.02. GPUWeb's current implementation status enables
NVIDIA Linux by default on Wayland from Chrome 147; the other Linux path remains
experimental. This makes the platform gate a likely explanation for Chrome's
null adapter, pending its own `chrome://gpu` report.

First inspect `chrome://gpu` (WebGPU, Vulkan and Problems Detected) and
`chrome://settings/system` (graphics acceleration). For an explicit development
test of the documented X11 path, use a separate Chrome profile:

```bash
google-chrome --user-data-dir=/tmp/fluxbridge-chrome-webgpu \
  --enable-unsafe-webgpu --ozone-platform=x11 --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE \
  https://fluids.dev/warpSPHBoundaries/wall.html
```

`--enable-unsafe-webgpu` bypasses the WebGPU adapter blocklist; this is an
isolated development configuration, not a change to the user's normal profile.
This command has not been launched automatically. It is a diagnostic candidate,
not a claim that live Chrome execution has passed. A page cannot turn on this
browser/platform capability itself. No `--no-sandbox` flag is needed here.

Sources: [GPUWeb implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
and [Chrome WebGPU troubleshooting](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips).


The user subsequently confirmed the isolated Chrome test instance works. The
wall provider additionally tries both core and compatibility adapter discovery,
with high-performance/default preferences. This remains GPU fp32 computation;
the selected feature level appears in provenance. Use `gpu-check.html` beside
the wall page to compare page/worker discovery without DevTools. A positive
Chrome feature-status summary does not substitute for those actual API calls.
Native OpenGL ES tests expose a compatibility-only NVIDIA adapter on this
workstation; whether normal Chrome exposes it must be checked in that browser.
