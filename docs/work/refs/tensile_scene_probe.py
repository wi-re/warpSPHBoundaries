"""Reviewer's probe for WORK-003 T3.3 (throw-away; write your own code).  Q2 for Wendland C2 through the UNMODIFIED scene path:
register W^5 as kernel `w2p5` (terms as in q2_conditioning.terms), Gradient operation (Naive, BoundaryToFluid) of the scene with a unit body field gives
g0 = grad_x ∫_solid W5reg dA'  with W5reg = c25/(pi H^2) shape^5.  The solver's tensile integral is
T = ∫_solid W^4 grad_i W dA' = (1/5) grad_x ∫ W^5 dA' = (1/5) * c2^5 / (pi^4 c25 H^8) * g0      (c2 = 7 for w2, c25 = c2_pi of w2p5).
Compared with the polar quadrature of DeltaSPH2D.shift (24 x 96) on the tank initial state."""
import sys, math
sys.path.insert(0, "python")
import torch
from edgebound.edge import kernels
from edgebound.q2_conditioning import terms
from edgebound.sim.cases import hydrostatic_tank
from edgebound.sim.pairs import F64, neighbor_pairs
from edgebound.scene.scene import BodyField, sceneOperation
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

kernels.KERNELS["w2p5"] = kernels._from_terms("w2p5", terms(5, "w2"))
c2, c25 = float(kernels.KERNELS["w2"].c2_pi), float(kernels.KERNELS["w2p5"].c2_pi)
sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0")
x, H, dev = sim.x, sim.H, sim.dev
lam, G, A = sim._wall_data(x, sim.rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
ins, u, rk, dr, dphi = sim._solid_samples(x, near)
Fr = sim.W(rk, H) ** 4 * sim.dW(rk, H) * rk * dr
Tq = -torch.einsum("bqrp,r,pa->qa", ins.to(F64), Fr, u) * dphi                       # exactly as in DeltaSPH2D.shift (without the wallMass shiftR / w0^4 prefactor)
n = len(near)
ps = ParticleState(positions=x[near], supports=torch.full((n,), H, dtype=F64, device=dev), masses=torch.ones(n, dtype=F64, device=dev),
                   kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=torch.ones(n, dtype=F64, device=dev))
pr = OperationProperties(kernel="w2p5", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
one = BodyField(torch.tensor(1.0, dtype=F64, device=dev))
g0 = sceneOperation(ps, pr, sim.scene, None, None, [one] * len(sim.scene.bodies), perBody=True).sum(0)
Te = (1.0 / 5.0) * c2 ** 5 / (math.pi ** 4 * c25 * H ** 8) * g0
w0 = float(sim.W(torch.tensor([sim.dx], dtype=F64, device=dev), H)[0])
print("N =", len(x), "near =", n, " H =", H, " dx =", sim.dx, " c2 =", c2, " c25 =", c25, " W(dx) =", w0)
print("max|T_quad| = %.6e   max|T_exact| = %.6e   max|T_quad - T_exact| = %.3e   relative to max|T_exact|: %.3e" % (
    Tq.abs().max(), Te.abs().max(), (Tq - Te).abs().max(), (Tq - Te).abs().max() / Te.abs().max()))
print("sign control: max|T_quad + T_exact| / max|T_exact| = %.3f" % ((Tq + Te).abs().max() / Te.abs().max()))
print("prefactor in shift: wallMass*shiftR/w0^4 =", sim.cfg.wallMass * sim.cfg.shiftR / w0 ** 4, " (shiftR =", sim.cfg.shiftR, ")")

# ---- (2) Warp scene route vs the independent np2d stable=(16,8) route on a rotated, translated L-shaped body (T2.4: route D is GOOD, 5.9e-16 at k=5)
import numpy as np
from edgebound.edge import np2d
from edgebound.scene.scene import Body, Scene, SurfaceRep
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float); C0, ANG, Hh = (0.3, -0.2), 0.7, 0.6
R = np.array([[math.cos(ANG), -math.sin(ANG)], [math.sin(ANG), math.cos(ANG)]]); Wp = LS @ R.T + np.array(C0)
sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=C0, angle=ANG)], "cuda:0")
pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
n = len(pos)
ps = ParticleState(positions=torch.tensor(pos, dtype=F64, device="cuda:0"), supports=torch.full((n,), Hh, dtype=F64, device="cuda:0"), masses=torch.ones(n, dtype=F64, device="cuda:0"),
                   kinds=torch.zeros(n, dtype=torch.int32, device="cuda:0"), densities=torch.ones(n, dtype=F64, device="cuda:0"))
g0w = sceneOperation(ps, pr, sc, None, None, [BodyField(torch.tensor(1.0, dtype=F64, device="cuda:0"))], perBody=True).sum(0).cpu().numpy()
g0n = np2d.gradient(np.repeat(Wp[None], n, 0), pos, "w2p5", h=Hh, dtype=np.float64, stable=(16, 8))
print("(2) L-shape rotated, H=0.6: max|g0_warp - g0_np2d(16,8)| / max|g0_np2d| = %.3e   (max|g0| = %.3e)" % (np.abs(g0w - g0n).max() / np.abs(g0n).max(), np.abs(g0n).max()))

# ---- (3) effect on shift(): patched copy of DeltaSPH2D whose wall tensile integral is the exact one
import types, importlib.util
src = open("src/edgebound/sim/deltasph2d.py").read()
old = src[src.index("                Fr = self.W(rk, H) ** 4"):src.index("                wall = wall.index_add(0, near, cfg.wallMass * cfg.shiftR / w0 ** 4 * T)")]
new = "                T = _TE(self, near)\n"
assert old in src
mod = types.ModuleType("edgebound.deltasph2d_probe"); mod.__package__ = "edgebound"
def _TE(self, near):
    n_ = len(near)
    ps_ = ParticleState(positions=self.x[near], supports=torch.full((n_,), self.H, dtype=F64, device=self.dev), masses=torch.ones(n_, dtype=F64, device=self.dev),
                        kinds=torch.zeros(n_, dtype=torch.int32, device=self.dev), densities=torch.ones(n_, dtype=F64, device=self.dev))
    g = sceneOperation(ps_, pr, self.scene, None, None, [BodyField(torch.tensor(1.0, dtype=F64, device=self.dev))] * len(self.scene.bodies), perBody=True).sum(0)
    return (1.0 / 5.0) * c2 ** 5 / (math.pi ** 4 * c25 * self.H ** 8) * g
mod.__dict__["_TE"] = _TE
exec(compile(src.replace(old, new), "deltasph2d_probe", "exec"), mod.__dict__)
import edgebound.sim.deltasph2d as orig
def shift_effect(sim, label):
    dt = sim.dt
    u_q = sim.shift(dt)
    sim.__class__ = mod.DeltaSPH2D
    u_e = sim.shift(dt)
    sim.__class__ = orig.DeltaSPH2D
    print("(3) %s: max|upd_quad| = %.3e  max|upd_quad - upd_exact| = %.3e  ratio %.3e;  relative to the dx clamp (0.5 dx*shiftThreshold): %.3e" % (
        label, u_q.norm(dim=1).max(), (u_q - u_e).abs().max(), (u_q - u_e).abs().max() / u_q.norm(dim=1).max(), (u_q - u_e).abs().max() / sim.dx))
sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0")
shift_effect(sim, "tank initial")
for _ in range(50): sim.step()
shift_effect(sim, "tank after 50 steps")
