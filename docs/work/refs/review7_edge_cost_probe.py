# NOTE: run on HEAD (1090f5a) WITHOUT review7_work008_proto.diff (its `pruned_plan` rebuilds a plan by hand; the diff makes `edge_channels(channels=...)` do it).
# reviewer probe for WORK-008: where one SurfaceRep adjacency build spends its time (dam-break state, 300 steps), per kernel:
# plan construction (DevicePlan / ChebPlan host->device) vs the launch, and the effect of dropping the terms of unused channels
import time, torch, numpy as np
from edgebound.sim.cases import marrone_dambreak
from edgebound.edge import warpbc
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x = sim.x; H = sim.H; dev = str(x.device)
body = sim.scene.bodies[0]; rep = body.reps[0]
lam, G, A = sim._wall_data(x, sim.rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
lpos = ((x[near] - body.center) @ body.pose.R)
lsup = torch.full((len(near),), H, dtype=torch.float64, device=x.device)
cl = rep._celllist(H)
from edgebound.scene.scene import queryCellList, _segment_distance
qi, e = queryCellList(cl, lpos)
a, b = rep.vertices[rep.edges[e, 0].long()], rep.vertices[rep.edges[e, 1].long()]
keep = _segment_distance(lpos[qi], a, b) < lsup[qi]
qi, e = qi[keep].to(torch.int32), e[keep].to(torch.int32)
print("near", len(near), "pairs", len(qi), "STABLE_KERNELS", dict(warpbc.STABLE_KERNELS))
def T(f, n=30):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3
for kern in ("lw2", "cone", "w2", "w4", "w2p5", "w4p5"):
    try:
        full = lambda: warpbc.edge_channels(qi, e, lpos, lsup, rep.vertices, rep.edges, kern, device=dev)
        t_full = T(full)
        if kern in warpbc.STABLE_KERNELS or warpbc.STABLE_KERNELS.get(kern):
            plan_t = T(lambda: warpbc.ChebPlan(kern, dev, *warpbc.STABLE_KERNELS[kern]))
        else:
            plan_t = T(lambda: warpbc.DevicePlan(kern, dev))
        pb = (warpbc.build_cheb_plan if kern in warpbc.STABLE_KERNELS else warpbc.build_plan)(kern)[0]
        print("%-6s edge_channels %7.2f ms   plan construction alone %6.2f ms   terms E %d V %d" % (kern, t_full, plan_t, len(pb.E), len(pb.V)))
    except Exception as ex:
        print(kern, "ERR", repr(ex)[:120])

# ---- channel pruning prototype: keep only the terms of the wanted channels (rows of pb.E / pb.V filtered; plan arrays rebuilt)
import copy
def pruned_plan(kern, dev, wanted, cheb):
    pb, rin = (warpbc.build_cheb_plan if cheb else warpbc.build_plan)(kern)
    q = copy.copy(pb); q.E = [r for r in pb.E if r[0] in wanted]; q.V = [r for r in pb.V if r[0] in wanted]
    warpbc.build_cheb_plan.__wrapped__  # (lru_cache original exists)
    P = warpbc.ChebPlan.__new__(warpbc.ChebPlan) if cheb else warpbc.DevicePlan.__new__(warpbc.DevicePlan)
    i32 = lambda rows, k: warpbc.wp.array(np.array([r[k] for r in rows] or [0], dtype=np.int32), dtype=int, device=dev)
    P.rin_idx = rin; P.nE, P.nV = len(q.E), len(q.V)
    P.radii = warpbc.wp.array(np.array(pb.radii, dtype=np.float64), dtype=warpbc.f64, device=dev)
    P.cc = warpbc.wp.array(np.array(pb.cc or [0.0], dtype=np.float64), dtype=warpbc.f64, device=dev)
    P.e = [i32(q.E, k) for k in range(9)]; P.v = [i32(q.V, k) for k in range(6)]
    P.v_mR = warpbc.wp.array(np.array([r[6] for r in q.V] or [0.0], dtype=np.float64), dtype=warpbc.f64, device=dev)
    if cheb:
        gx, gw = np.polynomial.legendre.leggauss(16); P.nodes, P.panels = 16, 8
        P.gx = warpbc.wp.array(gx, dtype=warpbc.f64, device=dev); P.gw = warpbc.wp.array(gw, dtype=warpbc.f64, device=dev)
    else:
        P.cn = warpbc.wp.array(np.array(pb.cn or [0], dtype=np.int32), dtype=int, device=dev)
        P.ubR = None
    return P
print("--- pruned to the gradient channels {3,4}")
for kern, cheb in (("cone", False), ("w2p5", True), ("lw2", False)):
    full = warpbc.edge_channels(qi, e, lpos, lsup, rep.vertices, rep.edges, kern, device=dev)
    P = pruned_plan(kern, dev, {3, 4}, cheb)
    run = lambda: warpbc.edge_channels(qi, e, lpos, lsup, rep.vertices, rep.edges, kern, device=dev, plan=P)
    out = run()
    print("%-5s pruned {3,4}: %6.2f ms (full %6.2f)  terms E %d V %d  channels 3,4 bit-identical: %s   others zero: %s" % (
        kern, T(run), T(lambda: warpbc.edge_channels(qi, e, lpos, lsup, rep.vertices, rep.edges, kern, device=dev)),
        P.nE, P.nV, bool(torch.equal(out[:, 3:5], full[:, 3:5])), bool((out[:, [0, 1, 2, 5, 6, 7, 8]] == 0).all())))
