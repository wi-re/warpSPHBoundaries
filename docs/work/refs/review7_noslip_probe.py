"""REVIEW-007 probes (throw-away, shares no code with tests/sim/test_deltasph_noslip.py):
(1) rigid-rotation Galilean check: wall rotating about its centre, particle with v = omega x (x - c) -> no wall term; at rest -> non-zero
(2) factor sensitivity of the Couette tolerance: scale the flux contribution by 0.5 / 1 / 2 / 4 (printed numbers of the test)
(3) scale of nu_eff in the dam break (nx=67) against the physical viscosity and the boundary-layer thickness sqrt(nu t)"""
import math, sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
from warpSPHBoundaries.sim import cases, deltasph2d as D
from warpSPHBoundaries.sim.deltasph2d import DeltaSPHConfig
from warpSPHBoundaries.sim.cases import hydrostatic_tank
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
F64 = torch.float64
sim, info = hydrostatic_tank(dp=0.04, domain="surface", device=dev, cfg=DeltaSPHConfig())
sim.cfg.wallViscosityForm = "noslip"; sim.cfg.ddt = False; sim.cfg.delta = 0.0
dp = sim.dx
x = torch.tensor([[0.1, info["bed"] + 0.5 * dp]], dtype=F64, device=dev)
rho = torch.ones(1, dtype=F64, device=dev)
sim.Hvec = torch.full((1,), sim.H, dtype=F64, device=dev); sim.kinds = torch.zeros(1, dtype=torch.int32, device=dev)
def term(v):
    sim.cfg.wallViscosity = True;  a1 = sim.rhs(x, v, rho)[0]
    sim.cfg.wallViscosity = False; a0 = sim.rhs(x, v, rho)[0]
    return (a1 - a0)[0]
b = sim.scene.bodies[0]
print("bodies:", len(sim.scene.bodies), "centres:", [tuple(bb.center.tolist()) for bb in sim.scene.bodies])
w = 0.8
try:
    b.angularVelocity = w
    vw = b.velocityAt(x)
    print("wall velocity at particle", vw.tolist())
    print("(1) co-rotating: |term| = %.3e" % float(term(vw).norm()))
    print("(1) at rest, wall rotating: |term| = %.3f (expected |2 nu |G|/d| * |vw| > 0)" % float(term(torch.zeros_like(vw)).norm()))
finally:
    b.angularVelocity = 0.0
# (2) factor sensitivity (printed numbers of the test: a_bulk row0 .06014, flux total .00815/-.00855/-.00280/.00001; bulk .06014/.01405/.00084/.00004)
bulk = [0.06014, 0.01405, 0.00084, 0.00004]; tot = [0.00815, -0.00855, -0.00280, 0.00001]
for f in (0.5, 1.0, 2.0, 4.0):
    rows = [bulk[k] + f / 2.0 * (tot[k] - bulk[k]) for k in range(4)]
    print("(2) prefactor %.1f (implemented: 2): Couette rows %s  max|.|/0.06014 = %.3f (tol 0.2)" % (f, ["%+.4f" % r for r in rows], max(abs(r) for r in rows) / 0.06014))
# (3) nu_eff scale in the dam break
sim2, info2 = cases.hydrostatic_tank(dp=0.04, device=dev)[0:2]
c = sim2.cfg
print("(3) tank dp=0.04: alpha=%g c0=%g H=%g xi=%g nu_eff=%.3e m2/s  (water 1.0e-6)  ratio %.0f" % (c.alpha, c.c0, sim2.H, sim2.xi, c.alpha * c.c0 * sim2.H / sim2.xi / 8, c.alpha * c.c0 * sim2.H / sim2.xi / 8 / 1e-6))
print("(3) boundary layer sqrt(nu_eff t) at t=0.5 s: %.4f m = %.2f dp; physical sqrt(1e-6*0.5) = %.2e m" % (math.sqrt(c.alpha * c.c0 * sim2.H / sim2.xi / 8 * 0.5), math.sqrt(c.alpha * c.c0 * sim2.H / sim2.xi / 8 * 0.5) / dp, math.sqrt(5e-7)))
