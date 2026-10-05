import sys; sys.path.insert(0, "python")
import numpy as np, torch
from warpSPHCore import KernelFunctions
from edgebound.sim.deltasph2d import DeltaSPHConfig
from edgebound.sim.cases import hydrostatic_tank
from edgebound.sim.pairs import F64
dev = "cuda:0"
def brute(fam, p, H, L=2.4, Ht=1.2, nr=600, nt=1200):
    # solid = outside the box; lap W polar midpoint grid
    c = {"w2": 7.0, "w4": 9.0}[fam]
    r = (np.arange(nr) + .5) / nr * H; th = (np.arange(nt) + .5) / nt * 2 * np.pi
    R, T = np.meshgrid(r, th, indexing="ij"); q = R / H
    if fam == "w2": lap = c / (np.pi * H ** 4) * 20 * (1 - q) ** 2 * (5 * q - 2)
    else:           lap = c / (np.pi * H ** 4) * (-(112 / 3)) * (1 - q) ** 4 * (1 + 4 * q - 20 * q * q)
    out = []
    for (x, y) in p:
        X = x + R * np.cos(T); Y = y + R * np.sin(T)
        solid = (np.abs(X) > L / 2) | (np.abs(Y) > Ht / 2)
        out.append(np.sum(lap * solid * R) * (H / nr) * (2 * np.pi / nt))
    return np.array(out)
for kern, fam in ((KernelFunctions.Wendland2, "w2"), (KernelFunctions.Wendland4, "w4")):
    cfg = DeltaSPHConfig(kernel=kern, viscosityExact=True)
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=dev, cfg=cfg)
    x = sim.x
    v = torch.stack([0.3 * torch.sin(2 * x[:, 0]) + 0.1, -0.2 * torch.cos(3 * x[:, 1]) + 0.05], 1)
    sim.v = v
    a_ex, _, _ = sim.rhs(sim.x, v, sim.rho)
    sim.cfg.wallViscosity = False
    a_no, _, _ = sim.rhs(sim.x, v, sim.rho)
    sim.cfg.wallViscosity = True; sim.cfg.viscosityExact = False
    a_pair, _, _ = sim.rhs(sim.x, v, sim.rho)
    d_ex = (a_ex - a_no).cpu().numpy(); d_pair = (a_pair - a_no).cpu().numpy()
    near = np.nonzero(np.abs(d_ex).max(1) > 0)[0]
    # independent prediction: -2 nu wallMass u_n / rho * dl * n   with n, u_n from the solver's own wall gradient (not under test)
    st = sim._surface_state(sim.x, sim.rho)
    G = st["G"] if "G" in st else None
    print(fam, "near(d_ex!=0):", len(near), " max|d_ex|", np.abs(d_ex).max(), " max|d_pair|", np.abs(d_pair).max(), " max|d_ex-d_pair|/max|d_pair|", np.abs(d_ex - d_pair).max() / np.abs(d_pair).max())
    H = sim.H; nu = sim.cfg.alpha * sim.cfg.c0 * H / sim.xi / 8
    n_ = (G[0] / G[0].norm(dim=1).clamp(min=1e-300)[:, None]).cpu().numpy() if G is not None else None
    if G is not None:
        un = ((v.cpu().numpy()) * n_).sum(1)
        dlb = brute(fam, x.cpu().numpy()[near], H)
        pred = (-2 * nu * sim.cfg.wallMass * un[near] / sim.rho.cpu().numpy()[near] * dlb)[:, None] * n_[near]
        err = np.abs(pred - d_ex[near]).max() / np.abs(pred).max()
        print("   brute-vs-solver wall-lap term: max rel err", err, " scale", np.abs(pred).max())
