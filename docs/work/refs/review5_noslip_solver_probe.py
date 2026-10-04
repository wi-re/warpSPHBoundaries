# reviewer probe: no-slip wall viscosity at the solver level.  Couette u = a (y - y_floor), floor at rest; Poiseuille u = (y - y_f)(Y - (y - y_f)).
# bulk = solver's own pairwise term; wall = candidates.  Rows z = 0.5, 1.5, 2.5, 3.5 dp above the floor, central columns of the tank.
import torch, numpy as np
from edgebound.deltasph2d import DeltaSPHConfig, hydrostatic_tank
sim, info = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0", cfg=DeltaSPHConfig())
x = sim.x; dp, H = sim.dx, sim.H; fac = sim.cfg.alpha*sim.cfg.c0*H/sim.xi; nu = fac/8
yf = float(x[:,1].min()) - 0.5*dp                       # floor surface
print("dp %.3f H %.3f floor y = %.3f  fluid y in [%.3f, %.3f]  nu_eff = %.5f" % (dp, H, yf, float(x[:,1].min()), float(x[:,1].max()), nu))
lam, G, A = sim._wall_data(x, sim.rho)
d, n, hit = sim.scene.signed_distance(x)
print("signed distance of the first row: %.4f (expect dp/2 = %.4f)  |G| = %.4f (expect wm * int W ds = ?)" % (abs(float(d[x[:,1].argmin()])), 0.5*dp, float(G[0].norm(dim=1)[x[:,1].argmin()])))
def total(v, wall):
    sim.cfg.wallViscosity = False
    a1,_,_ = sim.rhs(x, v, sim.rho); sim.cfg.viscosity = False
    a0,_,_ = sim.rhs(x, v, sim.rho); sim.cfg.viscosity = True; sim.cfg.wallViscosity = True
    return (a1 - a0) + wall
cols = (x[:,0].abs() < 0.3)
for name, prof, tgt in (("Couette  u=a y", lambda s: 1.0*s, lambda s: 0*s), ("Poiseuille u=s(Y-s), Y=0.36", lambda s: s*(0.36 - s), lambda s: -2.0*nu + 0*s)):
    s = x[:,1] - yf
    v = torch.stack([prof(s), torch.zeros_like(s)], 1)
    vrel = v                                                                  # wall at rest
    gmag = G[0].norm(dim=1); dd = d.abs().clamp(min=1e-6)
    near = (lam.sum(0) > 1e-9)
    ns = torch.zeros_like(v); ns[near] = (-2.0*nu*vrel*(gmag/dd/sim.rho)[:, None])[near]         # Chiron-style flux term (G = wm * int W ds)
    a_bulk = total(v, 0*v); a_ns = total(v, ns)
    print("\n%s: target (nu lap u_x) per row; columns |x|<0.3" % name)
    print("  %-6s %-14s %-14s %-14s" % ("z/dp", "bulk only", "bulk + flux", "target"))
    for k in range(4):
        row = cols & ((s - (k+0.5)*dp).abs() < 1e-9)
        t_ = float(tgt(torch.tensor((k+0.5)*dp)))
        print("  %-6.1f %-14.5f %-14.5f %-14.5f   (n=%d)" % (k+0.5, float(a_bulk[row,0].mean()), float(a_ns[row,0].mean()), t_, int(row.sum())))
