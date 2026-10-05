# step 1a of docs/plan-wall-evaluation.md: is edge_channels latency-bound in the KERNEL, or in the host path around it?
# per pair count: end-to-end edge_channels (as review8_latency_probe.py), the host-side launch cost (n launches without sync), and the GPU time of the kernel alone (Warp events around n
# back-to-back launches of pre-wrapped arrays; the GPU is idle between nothing, so this is the kernel duration), monomial f64 kernel and Chebyshev stable=(8, 6).   GPU idle: the numbers depend on load.
import time, sys
import numpy as np, torch, warp as wp
from edgebound.sim.cases import marrone_dambreak
from edgebound.edge import warpbc as W
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
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

def arrays(q2, e2):
    return (W._wp_from(q2, wp.int32, dev, torch.int32)[0], W._wp_from(e2, wp.int32, dev, torch.int32)[0], W._wp_from(lpos, wp.vec2d, dev, torch.float64)[0],
            W._wp_from(lsup, W.f64, dev, torch.float64)[0], W._wp_from(rep.vertices, wp.vec2d, dev, torch.float64)[0], W._wp_from(rep.edges, wp.vec2i, dev, torch.int32)[0])

def launcher(route, kern, channels, q2, e2):
    wq, we, wpos, wsup, wv, wed = arrays(q2, e2)
    P = len(q2); cout = wp.from_torch(torch.zeros((P, 9), dtype=torch.float64, device=dev), dtype=W.f64)
    if route == "mono":
        plan = W._device_plan(kern, dev, channels)
        args = [wq, we, wpos, wsup, wv, wed, plan.radii, W.f64(1 / np.pi), *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cn, plan.cc, cout]
        return lambda: wp.launch(W._edge_channels_kernel, dim=P, device=dev, inputs=args)
    plan = W._cheb_plan_for(kern, dev, 8, channels)
    args = [wq, we, wpos, wsup, wv, wed, plan.radii, W.f64(1 / np.pi), *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cc, plan.gx, plan.gw, 8, 6, cout]
    return lambda: wp.launch(W._edge_channels_cheb_kernel, dim=P, device=dev, inputs=args)

def kernel_ms(launch, n=50):
    launch(); wp.synchronize_device(dev)
    e0, e1 = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(e0)
    for _ in range(n): launch()
    wp.record_event(e1); wp.synchronize_event(e1)
    gpu = wp.get_event_elapsed_time(e0, e1) / n
    t0 = time.perf_counter()
    for _ in range(n): launch()
    host = (time.perf_counter() - t0) / n * 1e3                      # CPU time to enqueue (no sync)
    wp.synchronize_device(dev)
    return gpu, host

print("pairs        | route, kernel, channels        | end-to-end ms | kernel (GPU) ms | enqueue (host) ms")
for kern, ch, route in (("w2", None, "mono"), ("w2", (3, 4), "mono"), ("cone", (3, 4), "mono"), ("w2", None, "cheb"), ("w2p5", (3, 4), "cheb")):
    for rep_n in (1, 10, 100, 1000):
        q2, e2 = qi.repeat(rep_n), e.repeat(rep_n)
        if route == "mono":
            e2e = T(lambda: W.edge_channels(q2, e2, lpos, lsup, rep.vertices, rep.edges, kern, device=dev, channels=ch, stable=False), 10)
        else:
            e2e = T(lambda: W.edge_channels(q2, e2, lpos, lsup, rep.vertices, rep.edges, kern, device=dev, channels=ch, stable=(8, 6)), 10)
        gpu, host = kernel_ms(launcher(route, kern, ch, q2, e2))
        print("%9d    | %-4s %-5s %-9s | %8.3f      | %8.3f        | %8.3f" % (len(q2), route, kern, ch, e2e, gpu, host))
