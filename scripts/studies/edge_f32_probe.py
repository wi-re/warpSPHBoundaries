# step 1a (ii) of docs/plan-wall-evaluation.md: the edge kernels in the precision of warpSPHCore (edge/precision.py), Chebyshev stable plans, dam-break wall pairs.
# Run once per precision (the precision is fixed per process, as in warpSPH):
#     warpSPHCore_PRECISION=float64 python scripts/studies/edge_f32_probe.py
#     warpSPHCore_PRECISION=float32 python scripts/studies/edge_f32_probe.py        (the second run also prints the error against the saved float64 result)
# Per pair count: GPU time of the kernel alone (Warp events around back-to-back launches), and the channel values are saved to .tmp/ for the comparison.   GPU idle: timings depend on load.
import os, sys
import numpy as np, torch, warp as wp
import warpSPHBoundaries                                                   # noqa: F401  (float64 default unless warpSPHCore_PRECISION is set)
from warpSPHBoundaries import paths
from warpSPHBoundaries.edge import precision as PR, warpbc as W
from warpSPHBoundaries.sim.cases import marrone_dambreak
from warpSPHBoundaries.scene.scene import queryCellList, _segment_distance

tag = PR.real.__name__
sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x = sim.x; H = sim.H; dev = str(x.device); body = sim.scene.bodies[0]; rep = body.reps[0]
lam, Gw, A = sim._wall_data(x, sim.rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
lpos = ((x[near] - body.center) @ body.pose.R); lsup = torch.full((len(near),), H, dtype=torch.float64, device=x.device)
qi, e = queryCellList(rep._celllist(H), lpos)
a, b = rep.vertices[rep.edges[e, 0].long()], rep.vertices[rep.edges[e, 1].long()]
keep = _segment_distance(lpos[qi], a, b) < lsup[qi]; qi, e = qi[keep].to(torch.int32), e[keep].to(torch.int32)


def gpu_ms(launch, n=50):
    launch(); wp.synchronize_device(dev)
    e0, e1 = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(e0)
    for _ in range(n): launch()
    wp.record_event(e1); wp.synchronize_event(e1)
    return wp.get_event_elapsed_time(e0, e1) / n


def launcher(kern, channels, q2, e2, nodes=8, panels=6):
    P = len(q2)
    w = lambda t, dt, tdt: W._wp_from(t, dt, dev, tdt)[0]
    plan = W._cheb_plan_for(kern, dev, nodes, channels)
    cout = wp.from_torch(torch.zeros((P, 9), dtype=PR.torch_real, device=dev), dtype=PR.real)
    args = [w(q2, wp.int32, torch.int32), w(e2, wp.int32, torch.int32), w(lpos, PR.vec2_t, PR.torch_real), w(lsup, PR.real, PR.torch_real), w(rep.vertices, PR.vec2_t, PR.torch_real),
            w(rep.edges, wp.vec2i, torch.int32), plan.radii, PR.real(1 / np.pi), *plan.e, plan.nE, *plan.v, plan.v_mR, plan.nV, plan.cc, plan.gx, plan.gw, nodes, panels, cout]
    return lambda: wp.launch(W._edge_channels_cheb_kernel, dim=P, device=dev, inputs=args)


TMPD = paths.tmp_dir(); os.makedirs(TMPD, exist_ok=True)
print("precision", tag, "- Chebyshev stable=(8, 6) plans; kernel GPU ms (alone), and error against the float64 run when its file exists")
print("pairs     | kernel, channels | kernel ms | max |err| (units of h) | max rel err (|c| > 1e-3)")
for kern, ch in (("w2", None), ("w2", (3, 4)), ("w2p5", (3, 4)), ("cone", (3, 4))):
    for rep_n in (1, 10, 100, 1000):
        q2, e2 = qi.repeat(rep_n), e.repeat(rep_n)
        t = gpu_ms(launcher(kern, ch, q2, e2))
        c = W.edge_channels(qi, e, lpos, lsup, rep.vertices, rep.edges, kern, device=dev, channels=ch, stable=(8, 6)).cpu().numpy()
        fn = os.path.join(TMPD, "edge_probe_%s_%s_%s.npy" % (kern, "all" if ch is None else "".join(map(str, ch)), "float64"))
        if PR.IS_F64:
            np.save(fn, c); err = rel = float("nan")
        elif os.path.exists(fn):
            ref = np.load(fn); err = float(np.abs(c - ref).max()); big = np.abs(ref) > 1e-3; rel = float((np.abs(c - ref)[big] / np.abs(ref)[big]).max())
        else:
            err = rel = float("nan")
        print("%9d | %-4s %-9s | %8.3f  | %.2e               | %.2e" % (len(q2), kern, ch, t, err, rel))
