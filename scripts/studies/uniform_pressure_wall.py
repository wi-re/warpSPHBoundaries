"""Spurious force of a UNIFORM pressure near a curved wall (the static wall-consistency residual S_i): a fluid at rest with density 1 + delta around a cylinder in a periodic box must stay at rest; the acceleration printed is the error
(cut lattice, and after `DeltaSPH2D.pack`).  See docs/plan-next-steps.md (curved-wall closure results)."""
import numpy as np, torch, math, warpSPHBoundaries
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
import sys
CONS = len(sys.argv) > 1 and sys.argv[1] == "consistent"
dev="cuda:0"; n=48; R=0.2; dx=1/n
X,Y=np.meshgrid(dx*(np.arange(n)+.5),dx*(np.arange(n)+.5),indexing="ij"); pos=np.stack([X.ravel(),Y.ravel()],1)
pos=pos[np.linalg.norm(pos-0.5,axis=1)>=R+0.5*dx]
for pack in (0,400):
  for delta in (-0.02,0.0,0.02,0.05):
    sc=Scene([Body(bodyId=0,center=(0.5,0.5),reps=[DiskArrayRep([(0,0)],[R])])],dev)
    cfg=DeltaSPHConfig(gravity=(0,0),c0=10.0,alpha=0.5,periodic=Periodic((0,0),(1,1)),graphStep=False,shifting=False,wallViscosityForm="noslipMoment",pressureConsistent=CONS)
    sim=DeltaSPH2D(pos,np.zeros_like(pos),np.ones(len(pos)),dx,sc,cfg,dev,support=4*dx)
    if pack: sim.pack(np.ones(len(pos),dtype=bool),iters=pack)
    rho=torch.full_like(sim.rho,1.0+delta)
    acc,drho,f=sim.rhs(sim.x,sim.v,rho,want_forces=True)
    d=(sim.x-0.5).norm(dim=1)-R
    near=d<2*dx
    print(f"pack={pack} delta={delta}: max|acc| near wall {float(acc[near].norm(dim=1).max()):.4f} (bulk {float(acc[~near].norm(dim=1).max()):.1e}); mean radial acc near wall {float(((acc[near]*((sim.x[near]-0.5)/ (sim.x[near]-0.5).norm(dim=1,keepdim=True))).sum(1)).mean()):+.4f}; net pressure force on the cylinder {f[0,0,:2].tolist()}")
