# reviewer probe for WORK-006 T6.2: lap_lambda_scene with ONE adjacency (kernel l+family) for Density and Covariance
import time, torch
from edgebound.sim.deltasph2d import marrone_dambreak
from edgebound.scene.viscosity import lap_lambda_scene, lap_factor
from edgebound.scene.scene import BodyField, sceneOperation
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
sim,_ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho, H = sim.x, sim.rho, sim.H
lam, G, A = sim._wall_data(x, rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten(); pts = x[near]
def shared(scene, pos, H, fam):
    dev = scene.device; n = pos.shape[0]; B = len(scene.bodies)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=dev), masses=torch.ones(n, dtype=torch.float64, device=dev),
                       kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=torch.ones(n, dtype=torch.float64, device=dev))
    pr = lambda op: OperationProperties(kernel="l"+fam, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    adj = scene.buildAdjacency(ps, pr(WarpOperation.Density))
    l = sceneOperation(ps, pr(WarpOperation.Density), scene, adj, None, [BodyField(rho=1.0)]*B, perBody=True).reshape(B, n)
    c = sceneOperation(ps, pr(WarpOperation.Covariance), scene, adj, None, [BodyField(rho=1.0)]*B, perBody=True).reshape(B, n, 2, 2)
    return lap_factor(H, fam) * (2*l - c[:,:,0,0] - c[:,:,1,1])
def t(f, n=20):
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
for fam in ("w2","w4"):
    a = lap_lambda_scene(sim.scene, pts, H, fam); b = shared(sim.scene, pts, H, fam)
    print(fam, "max|own - shared| = %.2e (scale %.3e)   own %.2f ms  shared %.2f ms" % (float((a-b).abs().max()), float(a.abs().max()), t(lambda: lap_lambda_scene(sim.scene, pts, H, fam)), t(lambda: shared(sim.scene, pts, H, fam))))
