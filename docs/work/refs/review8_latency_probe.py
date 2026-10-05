# is edge_channels latency-bound? time vs number of (query, edge) pairs, dam-break state
import time, torch
from edgebound.sim.deltasph2d import marrone_dambreak
from edgebound.edge import warpbc
from edgebound.scene.scene import queryCellList, _segment_distance
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x = sim.x; H = sim.H; dev = str(x.device); body = sim.scene.bodies[0]; rep = body.reps[0]
lam, G, A = sim._wall_data(x, sim.rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
lpos = ((x[near] - body.center) @ body.pose.R); lsup = torch.full((len(near),), H, dtype=torch.float64, device=x.device)
qi, e = queryCellList(rep._celllist(H), lpos)
a, b = rep.vertices[rep.edges[e, 0].long()], rep.vertices[rep.edges[e, 1].long()]
keep = _segment_distance(lpos[qi], a, b) < lsup[qi]; qi, e = qi[keep].to(torch.int32), e[keep].to(torch.int32)
def T(f, n=30):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3
for kern, ch in (("w2", None), ("w2", (3, 4)), ("cone", (3, 4))):
    row = []
    for rep_n in (1, 10, 100, 1000):
        q2, e2 = qi.repeat(rep_n), e.repeat(rep_n)
        row.append("%d pairs: %.2f ms" % (len(q2), T(lambda: warpbc.edge_channels(q2, e2, lpos, lsup, rep.vertices, rep.edges, kern, device=dev, channels=ch), 10)))
    print(kern, ch, " | ".join(row))
