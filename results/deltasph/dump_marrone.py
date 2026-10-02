import sys, numpy as np
sys.path.insert(0, "/home/lu26029/dev/warpSPH/scripts")
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run
nx = int(sys.argv[1])
H, TANK_L, TANK_W, COL_W, G = 0.60, 1.00, 5.366*0.6, 2.0*0.6, 9.81
params = dict(W=TANK_W, fillRatio=H/TANK_L, fluidWidth=COL_W/TANK_W, gravityMagnitude=G, referenceVelocity=1.95*(G*H)**0.5, machTarget=1.95/40.0,
              pressureProbeHeights=[0.16, 0.584, 1.0], pressureProbeInset=0.0, pressureProbeDiscRadius=0.045)
r = run(dambreakCase, scheme='sun2017DeltaSPH', L=TANK_L, nx=nx, nSteps=1, tLimit=1e9, quiet=True, store=False, progress=False, params=params)
st = r.state.state
pos = st.positions.detach().cpu().numpy(); kinds = st.kinds.detach().cpu().numpy()
it = r.ctx.scratch.get('interiorDomain')
dom = r.ctx.config.domain
print("nx", nx, "dx", float(r.ctx.config.dx), "fluid", int((kinds==0).sum()), "interior", it.min.tolist(), it.max.tolist(), "domain", dom.min.tolist(), dom.max.tolist())
np.savez(sys.argv[2], x=pos[kinds==0], interior_min=it.min.cpu().numpy(), interior_max=it.max.cpu().numpy(), dx=float(r.ctx.config.dx), c0=float(r.ctx.schemeConfig.fluid.fixedSoundSpeed))
