"""Reviewer's probe for WORK-003 T3.3 (throw-away).  On real solver states: the wall part of the free-surface detector's cone count (polar 24x96 quadrature,
DeltaSPH2D._detect_surface) vs the closed form of cone_area_probe.py.  Reports the quadrature's own error (in counts: n_w * area) and how many near particles sit
so close to the threshold 0.5 that the quadrature error could flip their flag."""
import sys, math
sys.path.insert(0, "python"); sys.path.insert(0, "docs/work/refs")
import numpy as np, torch
from cone_area_probe import cone_area
from edgebound.sim.deltasph2d import hydrostatic_tank, marrone_dambreak
from edgebound.sim.dfsph2d import F64, neighbor_pairs
from edgebound.scene.scene import SurfaceRep

def world_loops(sim):
    out = []
    for b in sim.scene.bodies:
        for rep in b.reps:
            assert isinstance(rep, SurfaceRep)
            V = b.pose.toWorld(rep.vertices.to(sim.dev)).cpu().numpy(); E = rep.edges.cpu().numpy()
            out.append((V, E, rep.background))
    return out

def exact_area(p, th, al, H, loops):
    tot = 0.0
    for V, E, bg in loops:
        tot += bg * 0.5 * H * H * (2 * al if al < math.pi else 2 * math.pi)
        # edge list may not be a single loop ordering; sum over edges directly (formula is per edge)
        from cone_area_probe import edge_sector_area
        for e0, e1 in E:
            tot += edge_sector_area(p, V[e0], V[e1], H, th, al)
    return tot

def run(sim, label):
    x, H = sim.x, sim.H
    i, j, r = neighbor_pairs(x, sim.Hvec)
    lam, G, A = sim._wall_data(x, sim.rho)
    near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
    ins, u, rk, dr, dphi = sim._solid_samples(x, near)
    nz = i != j; ii, jj, rr = i[nz], j[nz], r[nz]
    unit = (x[ii] - x[jj]) / rr.clamp(min=1e-300)[:, None]
    C = sim._sum(unit, ii); nw = sim.cfg.wallMass / sim.dx ** 2
    area = (rk * dr * dphi)[:, None]; wt = ins.any(0).to(F64) * area[None]
    C = C.index_add(0, near, -nw * (wt[..., None] * u[None, None]).sum((1, 2)))
    norm = C.norm(dim=1); c = C / norm.clamp(min=1e-300)[:, None]
    cosang = -(unit * c[ii]).sum(1); inCone = torch.acos(cosang.clamp(-1, 1)) <= sim.cfg.barecascoThreshold / 2
    fcount = sim._sum(inCone.to(F64), ii)
    cn = (u[None] * c[near][:, None, :]).sum(2); cone = (torch.acos(cn.clamp(-1, 1)) <= sim.cfg.barecascoThreshold / 2).to(F64)
    wc_q = nw * (wt * cone[:, None, :]).sum((1, 2)); wa_q = nw * wt.sum((1, 2))
    loops = world_loops(sim); nl = near.cpu().numpy(); xs = x[near].cpu().numpy(); cs = c[near].cpu().numpy()
    wc_e = np.array([nw * exact_area(xs[k], math.atan2(cs[k, 1], cs[k, 0]), sim.cfg.barecascoThreshold / 2, H, loops) for k in range(len(nl))])
    wa_e = np.array([nw * exact_area(xs[k], 0.0, math.pi, H, loops) for k in range(len(nl))])
    wc_q, wa_q = wc_q.cpu().numpy(), wa_q.cpu().numpy()
    ok = (norm[near] > 1e-12).cpu().numpy()                      # axis defined
    tot_q = (fcount[near].cpu().numpy() + wc_q); tot_e = (fcount[near].cpu().numpy() + wc_e)
    flips = int(((tot_q < 0.5) != (tot_e < 0.5))[ok].sum())
    print("%s: N=%d near=%d (|C|>0: %d)  cone count: max|polar-exact| = %.3f (max exact %.2f)  all-neighbour area count: max|polar-exact| = %.3f (max %.2f)   flag flips from the cone count alone: %d" % (
        label, len(x), len(nl), ok.sum(), np.abs(wc_q - wc_e)[ok].max(), wc_e.max(), np.abs(wa_q - wa_e).max(), wa_e.max(), flips))
    print("    relative to n_w H^2 = %.2f : cone %.3e   all %.3e" % (nw * H * H, np.abs(wc_q - wc_e)[ok].max() / (nw * H * H), np.abs(wa_q - wa_e).max() / (nw * H * H)))

sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0"); run(sim, "tank dp=0.04 initial")
sim, _ = marrone_dambreak(nx=67, device="cuda:0") if isinstance(marrone_dambreak(nx=67, device="cuda:0"), tuple) else (marrone_dambreak(nx=67, device="cuda:0"), None)
run(sim, "dambreak nx=67 initial")

# ---- negative control claimed in WORK-003 T3.3 (c): the exact cone count about the axis -c differs from the polar count (axis +c) by > 1 count somewhere
def neg_control(sim, label):
    x, H = sim.x, sim.H
    i, j, r = neighbor_pairs(x, sim.Hvec); lam, G, A = sim._wall_data(x, sim.rho)
    near = torch.nonzero(lam.sum(0) > 1e-9).flatten(); ins, u, rk, dr, dphi = sim._solid_samples(x, near)
    nz = i != j; ii, jj, rr = i[nz], j[nz], r[nz]; unit = (x[ii] - x[jj]) / rr.clamp(min=1e-300)[:, None]
    C = sim._sum(unit, ii); nw = sim.cfg.wallMass / sim.dx ** 2
    wt = ins.any(0).to(F64) * (rk * dr * dphi)[:, None][None]
    C = C.index_add(0, near, -nw * (wt[..., None] * u[None, None]).sum((1, 2)))
    c = C / C.norm(dim=1).clamp(min=1e-300)[:, None]
    cn = (u[None] * c[near][:, None, :]).sum(2); cone = (torch.acos(cn.clamp(-1, 1)) <= sim.cfg.barecascoThreshold / 2).to(F64)
    wc_q = (nw * (wt * cone[:, None, :]).sum((1, 2))).cpu().numpy()
    loops = world_loops(sim); xs = x[near].cpu().numpy(); cs = c[near].cpu().numpy()
    wc_neg = np.array([nw * exact_area(xs[k], math.atan2(-cs[k, 1], -cs[k, 0]), sim.cfg.barecascoThreshold / 2, H, loops) for k in range(len(xs))])
    print("%s: axis -c: max|polar(+c) - exact(-c)| = %.3f counts" % (label, np.abs(wc_q - wc_neg).max()))
sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0"); neg_control(sim, "tank dp=0.04 initial")
