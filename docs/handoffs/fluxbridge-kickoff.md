# FluxBridge session kickoff

Copy the prompt below into a new session opened in
`/home/lu26029/dev/curvatureBoundaries`.

```text
Start work on FluxBridge: analytic flux boundaries and porous coupling for SPH.

Repository: /home/lu26029/dev/curvatureBoundaries
Implementation plan: docs/plan-porous-coupling.md

Read that plan, applicable AGENTS.md instructions, README.md, docs/README.md,
docs/warp-boundary-functions.md, and the 3D tetrahedral/surface derivations.
Inspect the working tree first: it contains ongoing research and browser work.
Preserve unrelated changes and distinguish existing implementation from plans.

The immediate objective is to complete efficient, tested NVIDIA Warp operators
in warpSPHBoundaries. Work through the independent library stages 0–4. Begin
with the coverage inventory and independent fixtures, then implement the first
missing 2D P0/P1 and 3D analytic baseline operations. Continue with concrete
implementation and validation, rather than stopping after another plan.

Keep the main /home/lu26029/dev/warpSPH checkout unchanged. It is undergoing
incompressible solver improvements. Do not import or copy its solvers into this
repository or modify shared Core/Integrator installations. Use explicit arrays,
structs and composable wp.func evaluators with batch/fused wrappers; preserve
package layering. The later solver handoff is Stage 5, after interfaces settle.

Cover the agreed interpolation, density/occupancy, gradient, divergence, curl,
moment, Hessian/Laplacian and FEM-weighted operations in 2D and 3D. Audit their
actual radial profiles and conventions. Expose only methods supporting every
quantity in their advertised contract. Include disk/cylinder cross sections,
polygons, tetrahedra, closed triangle meshes and standalone surface triangles,
plus the applicable full-contract special forms and approximate models.

Use arbitrary-precision references and independent checks. Browser WGSL results
are useful comparisons but do not establish native Warp CPU/CUDA correctness
or performance. Validate fp32/fp64 separately, including cancellation, slivers,
tangency and exact face/edge/vertex queries. Report unavailable devices honestly.
Higher-order material/velocity/test products may require moments through degree
nine; the current viewer's 30 selections do not establish this coverage.

Retain the exhaustive regular-flux reference. Its exterior tail prevents simply
dropping distant faces/edges. Derive and test compact-support corrections,
containment and angle/solid-angle limits before accepting localized evaluation.
Test acceleration structures against exhaustive enumeration. Do not cancel
internal faces across differing material traces.

Build standalone material contractions, transpose/reaction transfer, dissipation
checks and local implicit resistance responses. Keep resistance, permeability
and porosity distinct; polynomial permeability generally has a nonpolynomial
inverse. Do not treat a damping callback as a complete porous-flow solver.

Primary application reference: Böttcher, Jeske, Westhofen and Bender (2025),
Implicit Incompressible Porous Flow using SPH, arXiv:2504.07739v2,
https://arxiv.org/html/2504.07739v2 . Use its porous density, pressure and implicit
coupling treatment as a comparison; derive the mesh discretization explicitly.
The plan also includes prior exact integration and resistance-aware projection
references. Exact SPH mesh integration itself is established prior work.

Preserve ordinary solid flux-boundary applications in the eventual handoff:
channels, confined disks/cylinders, periodic fibre arrays, and supported moving
or rotating disks, with pressure/viscous loads and torque. These must work with
porous coupling disabled. Resolved impermeable fibres generate resistance from
geometry; an effective anisotropic porous model replaces many fibres. Do not
double-count both. Ordinary 2D disk flow measures transverse permeability;
longitudinal response needs an axial-velocity formulation or 3D geometry.

Update the plan with evidence and remaining gaps after each completed milestone.
Use the existing validation/output conventions, separate accuracy from speed,
and benchmark real CUDA consumers including preparation, adjacency and reduction.
Keep GPU measurements serialized on a shared machine. At the end of the session,
report implemented operators, exact validation commands/results, limitations and
the next executable step.
```
