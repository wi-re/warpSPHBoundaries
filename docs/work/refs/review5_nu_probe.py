# reviewer probe: does the SOLVER's own bulk viscous term equal (fac/8)(lap v + 2 grad div v)?
import torch, numpy as np
from edgebound.sim.deltasph2d import DeltaSPHConfig
from edgebound.sim.cases import hydrostatic_tank
sim,_ = hydrostatic_tank(dp=0.04, domain="surface", device="cuda:0", cfg=DeltaSPHConfig())
x = sim.x; fac = sim.cfg.alpha*sim.cfg.c0*sim.H/sim.xi
def visc(v):
    a1,_,_ = sim.rhs(sim.x, v, sim.rho)
    sim.cfg.viscosity=False
    try: a0,_,_ = sim.rhs(sim.x, v, sim.rho)
    finally: sim.cfg.viscosity=True
    return (a1-a0)
xn = x.cpu().numpy()
print("H %.3f dp %.3f x [%.3f,%.3f] y [%.3f,%.3f] N %d" % (sim.H, sim.dx, xn[:,0].min(), xn[:,0].max(), xn[:,1].min(), xn[:,1].max(), len(xn)))
cx, cy = 0.0, -0.36
inner = (np.abs(xn[:,0]) < 1.18-1.1*sim.H) & (xn[:,1] > -0.58+1.05*sim.H) & (xn[:,1] < -0.14-1.05*sim.H)
print("interior", int(inner.sum()))
xc = torch.as_tensor(x[:,0]-cx); yc = torch.as_tensor(x[:,1]-cy)
z = torch.zeros_like(xc)
for name, v, exp in (("v=(yc^2,0): lap=(2,0) graddiv=0", torch.stack([yc**2, z],1), 2.0),
                     ("v=(xc^2,0): lap=(2,0) graddiv=(2,0)", torch.stack([xc**2, z],1), 6.0)):
    d = visc(v).cpu().numpy()[inner]
    print(name, "mean a_x/fac = %.5f   expected (lap+2 graddiv)_x/8 = %.5f   spread %.1e   max|a_y|/fac %.1e" % (d[:,0].mean()/fac, exp/8, d[:,0].std()/fac, np.abs(d[:,1]).max()/fac))
