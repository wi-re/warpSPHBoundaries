# reviewer probe for WORK-006: can ONE adjacency (built for all N, one kernel) serve the exact ops of other kernels?  cost of op vs adjacency.
import time, math, torch
from edgebound.sim.deltasph2d import marrone_dambreak
from edgebound.scene.cover import cover_vector_scene
from edgebound.scene.tensile import tensile_vector_scene, tensile_factor
from edgebound.scene.viscosity import lap_lambda_scene
from edgebound.scene import scene as S
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
sim,_ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho, H = sim.x, sim.rho, sim.H
lam, G, A = sim._wall_data(x, rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
def t(f, n=20):
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
lap_lambda_scene(sim.scene, x[near][:2], H, "w2"); tensile_vector_scene(sim.scene, x[near][:2], H, "w2")   # registers lw2 / w2p5
ps = ParticleState(positions=x, supports=sim.Hvec, masses=torch.full_like(rho, sim.m), kinds=sim.kinds, densities=rho)
for kern in ("cone", "w2p5", "lw2", sim.cfg.kernel):
    pr = OperationProperties(kernel=kern, operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    print("buildAdjacency(all N, kernel=%s): %.2f ms" % (kern, t(lambda: sim.scene.buildAdjacency(ps, pr))))
pr = OperationProperties(kernel="cone", operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
adj = sim.scene.buildAdjacency(ps, pr)
# (1) cover with a shared adjacency (built for ALL particles with the kernel `cone`, positions = all x) vs the near-only call
ref = cover_vector_scene(sim.scene, x[near], H)
got = cover_vector_scene(sim.scene, x, H, adjacency=adj)[near]
print("cover shared-vs-own max|diff| %.2e (scale %.3e)   time own %.2f ms, shared(all N) %.2f ms" % (float((ref-got).abs().max()), float(ref.abs().max()),
      t(lambda: cover_vector_scene(sim.scene, x[near], H)), t(lambda: cover_vector_scene(sim.scene, x, H, adjacency=adj))))
# (2) can a `cone` adjacency be reused by another kernel's op?  (lw2 Density through sceneOperation with the cone adjacency)
from edgebound.scene.scene import BodyField, sceneOperation
for kern in ("lw2", "w2p5"):
    lap_lambda_scene(sim.scene, x[near][:2], H, "w2"); tensile_vector_scene(sim.scene, x[near][:2], H, "w2")       # registers kernels
    p2 = OperationProperties(kernel=kern, operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    own = sceneOperation(ps, p2, sim.scene, None, None, [BodyField(rho=1.0)]*len(sim.scene.bodies), perBody=True)
    shr = sceneOperation(ps, p2, sim.scene, adj, None, [BodyField(rho=1.0)]*len(sim.scene.bodies), perBody=True)
    print("kernel %s Density: own-adjacency vs cone-adjacency max|diff| %.2e (scale %.3e)   op time own(incl. build) %.2f / shared %.2f ms" % (kern, float((own-shr).abs().max()), float(own.abs().max()),
          t(lambda: sceneOperation(ps, p2, sim.scene, None, None, [BodyField(rho=1.0)]*len(sim.scene.bodies), perBody=True)),
          t(lambda: sceneOperation(ps, p2, sim.scene, adj, None, [BodyField(rho=1.0)]*len(sim.scene.bodies), perBody=True))))
    p3 = OperationProperties(kernel=kern, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    one = BodyField(torch.tensor(1.0, dtype=torch.float64, device=x.device))
    print("   Gradient op time with a shared adjacency, all N: %.2f ms" % t(lambda: sceneOperation(ps, p3, sim.scene, adj, None, [one]*len(sim.scene.bodies), perBody=True)))
