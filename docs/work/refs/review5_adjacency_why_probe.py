# why a single adjacency "breaks": (1) it stores kernel moments, the operation only contracts them; (2) cost split of buildAdjacency; (3) multi-kernel evaluation on one pair list
import time, torch
from edgebound.deltasph2d import marrone_dambreak
from edgebound.viscosity import lap_lambda_scene
from edgebound.tensile import tensile_vector_scene
from edgebound import scene as S, warpbc
from edgebound.scene import BodyField, sceneOperation, queryCellList, _segment_distance
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
sim,_ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho, H = sim.x, sim.rho, sim.H
lam,_,_ = sim._wall_data(x, rho); near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
lap_lambda_scene(sim.scene, x[near][:2], H, "w2"); tensile_vector_scene(sim.scene, x[near][:2], H, "w2")
ps = ParticleState(positions=x, supports=sim.Hvec, masses=torch.full_like(rho, sim.m), kinds=sim.kinds, densities=rho)
P = lambda k, op=WarpOperation.Density: OperationProperties(kernel=k, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
F = [BodyField(rho=1.0)]
def t(f, n=15):
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
adj = sim.scene.buildAdjacency(ps, P("cone"))
print("adjacency.kernel =", adj.kernel)
a = sceneOperation(ps, P("lw2"), sim.scene, adj, None, F, perBody=True)          # lw2 requested, cone adjacency
b = sceneOperation(ps, P("cone"), sim.scene, None, None, F, perBody=True)        # cone, own adjacency
c = sceneOperation(ps, P("lw2"), sim.scene, None, None, F, perBody=True)         # lw2, own adjacency
print("(1) 'lw2 with cone adjacency' vs 'cone with its own adjacency': max|diff| = %.2e (scale %.3e);  vs 'lw2 own': %.2e" % (float((a-b).abs().max()), float(b.abs().max()), float((a-c).abs().max())))
# (2) split of buildAdjacency for the surface rep
body = sim.scene.bodies[0]; rep = body.reps[0]
pos, sup = ps.positions, ps.supports
cand = torch.arange(len(pos), device=pos.device)
lpos = pos  # world == local for an unposed wall (dam break); only used for timing
cl = rep._celllist(float(sup.max()))
def pairlist():
    qi, e = queryCellList(cl, lpos)
    a_, b_ = rep.vertices[rep.edges[e, 0].long()], rep.vertices[rep.edges[e, 1].long()]
    keep = _segment_distance(lpos[qi], a_, b_) < sup[qi]
    return qi[keep].to(torch.int32), e[keep].to(torch.int32)
qi, e = pairlist()
print("(2) pairs (query,edge) in support: %d" % len(qi))
print("    kernel-independent  pair list (cell list + segment-distance cull): %.2f ms" % t(pairlist))
print("    kernel-independent  indicator (indicatorFast): %.2f ms" % t(lambda: rep.indicatorFast(lpos, float(sup.max()))))
for k in ("cone", "lw2", "w2", "w2p5"):
    print("    kernel-DEPENDENT    edge_channels(%-5s): %.2f ms" % (k, t(lambda: warpbc.edge_channels(qi, e, lpos, sup, rep.vertices, rep.edges, k, device=str(pos.device)))))
print("    whole buildAdjacency cone / w2p5: %.2f / %.2f ms" % (t(lambda: sim.scene.buildAdjacency(ps, P("cone"))), t(lambda: sim.scene.buildAdjacency(ps, P("w2p5")))))
