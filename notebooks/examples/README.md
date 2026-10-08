# Example cases: integration and evaluation against reference solutions

Nine notebooks that run a flow with this package's two fluid solvers and compare it with a closed-form or literature reference. Every notebook starts with the same three switches:

```python
SCHEME = "delta"      # "delta": weakly compressible delta+-SPH (DeltaSPH2D)   |  "dfsph": divergence-free SPH (DFSPH2D)
QUALITY = "quick"     # "quick": minutes   |  "full": finer lattice, longer run
DEVICE = "cuda:0"
```

and builds either solver from one description of the flow through `warpSPHBoundaries.sim.solvers.make_solver` (lattice, kinematic viscosity, walls, driving), so the two schemes are compared on identical set-ups.
The walls are always **exact analytic walls** (the kernel integrals over the solid, no boundary particles): tank walls, plates, cylinders, polygons. The closed-form references live in
`warpSPHBoundaries.sim.references`.

| notebook | flow | reference | what it tests | cost (quick) |
|---|---|---|---|---|
| `01_taylor_green_vortex` | periodic vortex array, no walls | decay `exp(-4 nu k^2 t)` | viscous operator, pressure, shifting | seconds |
| `02_poiseuille_channel` | channel between plates, body force | parabola, plate load = body force | flat no-slip wall, momentum balance | 1 min |
| `03_taylor_couette` | rotating inner cylinder | `u = A r + B/r`, torque `-4 pi nu B` | curved moving no-slip walls, torque | 1 min |
| `04_stokes_cylinder_array` | periodic array of cylinders, Stokes | Sangani-Acrivos / Hasimoto drag | curved-wall load, superficial velocity | 2-3 min |
| `05_sharp_corner_array` | periodic array of triangles | penalised Fourier Stokes solve | corners of an analytic polygon | 3-5 min |
| `06_still_water_and_sloshing` | free-surface tank | `p = rho g h`, `omega^2 = g k tanh(k d)` | free surface, hydrostatic wall pressure | 1 min |
| `07_dam_break` | Marrone dam break | Ritter front, warpSPH run (stored) | violent free-surface flow | 2 min |
| `08_archimedes_submerged_cylinder` | fixed cylinder in still water | buoyancy `rho g pi R^2`, weight balance | pressure load on a curved wall | 1 min |
| `09_flow_past_cylinder` | cylinder in a stream, Re = 20 / 100 | `C_D`, `St` of the literature | drag, lift, Strouhal number | 10 min (Re 20); hours (Re 100, stored) |

How to read the results (and which numbers are known limitations rather than bugs) is summarised at the end of each notebook. The notebooks store the series of a run in `results/examples/<case>_<scheme>.npz`;
running the same notebook with the other `SCHEME` overlays it (dam break) or lets you compare the printed numbers.

Dependencies beyond the package: `matplotlib`, `scipy` (notebook 06), and for notebook 05 the Fourier reference of `scripts/studies/corner_rig.py` (computed once, cached in `.tmp`). The notebooks need a CUDA device.
The saved outputs in the repository are those of `SCHEME = "delta"`, `QUALITY = "quick"`; the measured numbers of both schemes are in the table below.

## Measured numbers (quick resolution, both solvers)

| notebook | quantity (ideal) | `delta` | `dfsph` |
|---|---|---|---|
| 01 Taylor-Green | KE decay rate / analytic (1) | 0.976 | 1.004 |
| 02 Poiseuille | profile amplitude / parabola (1) | 0.988 | 0.983 |
| 03 Taylor-Couette (n = 48) | profile amplitude (1); torque / exact (1) | 0.999; 0.974 | 0.83; 1.03-1.04 |
| 04 cylinder array (c = 0.126) | K / K_SA (1); load / body force (1) | 0.999; 1.002 | 1.012; 1.001 |
| 05 triangle array (30 deg, c = 0.04) | K / K_ref (1); U_x / U_ref (1) | 0.930; 1.049 | 0.955; 1.031 |
| 06 sloshing | omega / omega_linear (1); dp/d(depth) / rho g (1) | 0.998; 1.011 | 1.077; 1.099 |
| 07 dam break | KE / warpSPH at t = 0.2, 0.4, 0.6 s | 0.95, 0.94, 0.94 | 0.95, 0.90, 0.85 |
| 08 Archimedes | F_y / buoyancy (1); all walls / weight (1) | 1.077; 1.004 | 1.011; 0.997 |
| 09 cylinder, Re = 20, 16 x 12 D box | C_D (unconfined literature 2.0-2.09) | 2.523 | 2.521 |

Known limits behind the entries that are not close to 1: the sharp 30-degree corners (05) are the hardest case for the no-slip closure; `dfsph` keeps the omniSPH-style free-surface treatment in the tank (06, 07: about 10 % pressure error at the
surface, more dissipation after the impact); the cylinder drag of 09 is raised by the confinement of the periodic box with its velocity frame (blockage 0.11; it falls steeply as the box widens) and by the coarse resolution D/dx = 12.
The Re = 100 run (Strouhal number) needs D/dx of 40 or more for the boundary layer; the stored run at D/dx = 14 gives a drag and a lift that are only indicative.

**Open items found while building these (not investigated further):** `dfsph` Taylor-Couette: the profile amplitude is 0.83 at n = 48 (reproduced by `scripts/studies/dfsph_viscous.py couette`), and at n = 32 the moving wall gives run-to-run scatter and an occasional CUDA device-side assert at the impulsive start;
the Re = 100 cylinder run (delta, D/dx = 14, box 22 x 11 D, T = 120, stored in `results/examples/flow_past_cylinder_full_delta.npz`) gives C_D 1.54, C_L rms 0.29, St 0.18 against 1.33-1.35, 0.32-0.34 (amplitude) and 0.164-0.166: under-resolved (boundary layer 1.4 dx), indicative only;
the `dfsph` Re = 100 run was not made.
