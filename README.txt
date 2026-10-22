Boundary laboratory — portable static viewer

Serve this folder on an HTTPS static host (including GitHub Pages), or a
localhost HTTP server, and open volume.html. No Python compute service,
Warp or warpSPH installation is required for the 3D browser providers.
WebGPU is optional: the JavaScript float64 worker remains available.

The 2D tab can reopen saved results; fresh 2D computation uses the optional
local Python service. The 3D mesh preset uses approximate support cubature;
the analytic mesh reference is not compiled into the browser. Equations,
normalization and model validity are documented in the 3D interface.
