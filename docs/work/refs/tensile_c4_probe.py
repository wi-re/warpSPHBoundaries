"""Reviewer's probe for WORK-004 (throw-away; write your own code).  Q2 for Wendland C4 through the scene path with the STABLE edge kernel of
`stable_plan_probe.py` monkeypatched into `warpbc.edge_channels` for the kernel `w4p5` (= W^5 of Wendland C4, degree 40).  Same recipe as tensile_scene_probe.py
(w2): T = (1/5) c4^5 / (pi^4 c45 H^8) * g0, c4 = KERNELS['w4'].c2_pi = 9, c45 = KERNELS['w4p5'].c2_pi.  Prints the numbers quoted in WORK-004.md."""
import sys, math
sys.path.insert(0, "python"); sys.path.insert(0, "docs/work/refs")
import numpy as np, torch
from edgebound import kernels, warpbc, np2d
from edgebound.q2_conditioning import terms
from edgebound.deltasph2d import hydrostatic_tank, DeltaSPHConfig, sloshing_tank
from edgebound.dfsph2d import F64
from edgebound.scene import Body, BodyField, Scene, SurfaceRep, sceneOperation
from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
import stable_plan_probe as sp

kernels.KERNELS["w4p5"] = kernels._from_terms("w4p5", terms(5, "w4"))
_orig = warpbc.edge_channels
_plans = {}
def patched(pair_q, pair_e, positions, supports, vertices, edges, kernel, device="cuda:0", plan=None, as_torch=True):
    if kernel == "w4p5":
        pl = _plans.setdefault(device, sp.StablePlan("w4p5", device))
        return sp.stable_edge_channels(pair_q, pair_e, positions, supports, vertices, edges, kernel, device=device, plan=pl)
    return _orig(pair_q, pair_e, positions, supports, vertices, edges, kernel, device=device, plan=plan, as_torch=as_torch)
warpbc.edge_channels = patched

c4, c45 = float(kernels.KERNELS["w4"].c2_pi), float(kernels.KERNELS["w4p5"].c2_pi)
print("c4 =", c4, " c45 =", c45)
pr = OperationProperties(kernel="w4p5", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)

def T_scene(scene, pos, H, dev="cuda:0"):
    pos = (pos.to(dev, F64) if isinstance(pos, torch.Tensor) else torch.as_tensor(np.asarray(pos), dtype=F64, device=dev)); n = len(pos)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=F64, device=dev), masses=torch.ones(n, dtype=F64, device=dev),
                       kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=torch.ones(n, dtype=F64, device=dev))
    g0 = sceneOperation(ps, pr, scene, None, None, [BodyField(torch.tensor(1.0, dtype=F64, device=dev))] * len(scene.bodies), perBody=True).sum(0)
    return (1.0 / 5.0) * c4 ** 5 / (math.pi ** 4 * c45 * float(H) ** 8) * g0

# (0) the solver's Wendland4 W is c4/(pi H^2)(1-q)^6 (1+6q+35/3 q^2) ?
sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0", cfg=DeltaSPHConfig(kernel=KernelFunctions.Wendland4))
H = sim.H
r = torch.linspace(0, 1, 7, dtype=F64, device="cuda:0")[:-1] * H
q = r / H
print("(0) solver W4 vs formula: max rel diff = %.2e" % float(((sim.W(r, H) - c4 / (math.pi * H * H) * (1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q * q)) / sim.W(r, H)).abs().max()))

# (1) flat floor absolute check, w4 (grid)
FLOOR = [(-2.0, -2.0), (2.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]
scf = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])], "cuda:0")
def dense_T(px, py, H, n=2000):
    xp = -H + (np.arange(n) + 0.5) / n * (2 * H); yp = -H + (np.arange(n) + 0.5) / n * H
    X, Y = np.meshgrid(xp, yp, indexing="ij"); rx, ry = px - X, py - Y; rr = np.sqrt(rx * rx + ry * ry); m = rr < H; qq = rr[m] / H
    c = 9.0 / (math.pi * H * H); pol = 1 + 6 * qq + 35 / 3 * qq ** 2
    W = c * (1 - qq) ** 6 * pol
    dWdq = c * (-6 * (1 - qq) ** 5 * pol + (1 - qq) ** 6 * (6 + 70 / 3 * qq))
    dA = (2 * H / n) * (H / n)
    return np.array([np.sum(W ** 4 * dWdq / H * rx[m] / rr[m]), np.sum(W ** 4 * dWdq / H * ry[m] / rr[m])]) * dA
for (px, py, Hh) in [(0.0, 0.3, 1.0), (0.0, 0.15, 0.5)]:
    T = T_scene(scf, [[px, py]], Hh).cpu().numpy()[0]; gd = dense_T(px, py, Hh)
    print("(1) floor w4 (%.2f, %.2f) H=%.1f: T = (%.9e, %.9e)  grid = (%.9e, %.9e)  rel diff = %.2e" % (px, py, Hh, T[0], T[1], gd[0], gd[1], np.abs(T - gd).max() / abs(gd[1])))

# (2) L-shape vs np2d stable (16,8)
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float); C0, ANG, Hh = (0.3, -0.2), 0.7, 0.6
R = np.array([[math.cos(ANG), -math.sin(ANG)], [math.sin(ANG), math.cos(ANG)]]); Wp = LS @ R.T + np.array(C0)
sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=C0, angle=ANG)], "cuda:0")
pos = np.random.default_rng(3).uniform(-2, 3, (200, 2))
Tw = T_scene(sc, pos, Hh).cpu().numpy()
fac = (1.0 / 5.0) * c4 ** 5 / (math.pi ** 4 * c45 * Hh ** 8)
Tn = fac * np2d.gradient(np.repeat(Wp[None], 200, 0), pos, "w4p5", h=Hh, dtype=np.float64, stable=(16, 8))
print("(2) L-shape w4 H=0.6: max|T_warp - T_np2d(16,8)| / max|T| = %.2e   (max|T| = %.4e)" % (np.abs(Tw - Tn).max() / np.abs(Tn).max(), np.abs(Tn).max()))
Tm = fac * np2d.gradient(np.repeat(Wp[None], 200, 0), pos, "w4p5", h=Hh, dtype=np.float64)
print("    (monomial float64 np2d route for comparison: %.2e)" % (np.abs(Tm - Tn).max() / np.abs(Tn).max()))

# (3) tank C4: polar quadrature vs exact
st = sim._surface_state(sim.x, sim.rho)
near, (ins, u, rk, dr, dphi) = st["samples"]
Fr = sim.W(rk, H) ** 4 * sim.dW(rk, H) * rk * dr
Tq = -torch.einsum("bqrp,r,pa->qa", ins.to(F64), Fr, u) * dphi
Te = T_scene(sim.scene, sim.x[near], H)
print("(3) C4 tank dp=0.04: near=%d  max|T_exact| = %.4e  max|Tq-Te|/max|Te| = %.4e   sign control max|Tq+Te|/max|Te| = %.3f" % (
    len(near), Te.abs().max(), (Tq - Te).abs().max() / Te.abs().max(), (Tq + Te).abs().max() / Te.abs().max()))
w0 = float(sim.W(torch.tensor([sim.dx], dtype=F64, device="cuda:0"), H)[0])
print("    prefactor wallMass*shiftR/w0^4 = %.4e" % (sim.cfg.wallMass * sim.cfg.shiftR / w0 ** 4))

# (4) effect on shift() of the exact C4 tensile term: a patched copy of deltasph2d (guard removed, T from the scene route)
import types
src = open("python/edgebound/deltasph2d.py").read()
src = src.replace('                    if cfg.kernel != KernelFunctions.Wendland2:  raise NotImplementedError("tensileExact: Wendland C2 only (C4 needs the Chebyshev plan)")\n', "")
src = src.replace("T = tensile_vector_scene(self.scene, x[near], H)", "T = _TE(self.scene, x[near], H)")
assert "_TE(" in src and "Wendland2:  raise" not in src
mod = types.ModuleType("edgebound.deltasph2d_probe"); mod.__package__ = "edgebound"; mod.__dict__["_TE"] = T_scene
exec(compile(src, "deltasph2d_probe", "exec"), mod.__dict__)
import edgebound.deltasph2d as orig
def shift_effect(sim, label):
    dt = sim.dt
    u_q = sim.shift(dt)
    sim.cfg.tensileExact = True; sim.__class__ = mod.DeltaSPH2D
    u_e = sim.shift(dt)
    sim.__class__ = orig.DeltaSPH2D; sim.cfg.tensileExact = False
    print("(4) %s: max|u_q - u_e| / max||u_q|| = %.4e   (max||u_q|| = %.3e)" % (label, (u_q - u_e).norm(dim=1).max() / u_q.norm(dim=1).max(), u_q.norm(dim=1).max()))
shift_effect(sim, "C4 tank dp=0.04 initial")
