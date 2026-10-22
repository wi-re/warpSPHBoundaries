Boundary laboratory: static browser application

Open volume.html for 3D, or index.html?2d=1 for 2D.
Serve this directory over HTTP/HTTPS; do not use a file: URL.
Both dimension tabs compute in the browser without Python or Warp.
3D uses WebGPU when available and a float64 JavaScript fallback.
2D uses WebGPU fp32 by default, with a float64 reference/fallback worker.
Both use the same split boundary Gauss rules; explicit fp32 requires a GPU adapter.
First use compiles the compute pipeline; later interactions reuse it.
Each view distinguishes its geometry/model from its integration method.
Comparison view 1 retains its raw field and its own scale.
Manual resolutions: 17–513 nodes/axis in 2D, 17–129 in 3D.
The Local Python provider requires the package's loopback service.

Live site: https://fluids.dev/warpSPHBoundaries/volume.html

3D renderings are volume-only. Separate 2D cross-sections remain available.
Old saved slice-plane settings do not enable volume overlays.
