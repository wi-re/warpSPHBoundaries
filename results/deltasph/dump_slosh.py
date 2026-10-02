import sys, numpy as np
sys.path.insert(0, "/home/lu26029/dev/warpSPH/scripts")
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
importAll()
from warpSPH.runner import getCase, run, CaseSpec
case = getCase('sloshingTank')
spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params)).merged(**case.defaults).merged(nSteps=1, tLimit=1e9, quiet=True, store=False, progress=False, plot=False, video=False)
r = run(case, spec)
st = r.state.state
pos = st.positions.detach().cpu().numpy(); kinds = st.kinds.detach().cpu().numpy()
it = r.ctx.scratch.get('interiorDomain')
sc = r.ctx.schemeConfig
print("dx", float(r.ctx.config.dx), "fluid", int((kinds==0).sum()), "interior", it.min.tolist(), it.max.tolist(), "c0", float(sc.fluid.fixedSoundSpeed), "dt", float(r.ctx.config.dt),
      "kernel", r.ctx.config.kernel, "support", float(st.supports[0]), "mass", float(st.masses[0]), "eos", sc.fluid.eosType, "ddt", sc.diffusionParams.densityDiffusionTerm, "alpha", sc.diffusionParams.inviscidAlpha,
      "delta", sc.diffusionParams.densityDelta, "shift", sc.shiftProperties.active, sc.shiftProperties.scheme, sc.shiftProperties.projectionScheme, "noPen", sc.mdbcNoPenShiftMode, "tcc", getattr(sc,'timeCentredContinuity',None))
f = pos[kinds==0]
print("fluid x", f[:,0].min(), f[:,0].max(), "y", f[:,1].min(), f[:,1].max())
np.savez(sys.argv[1], x=f, interior_min=it.min.cpu().numpy(), interior_max=it.max.cpu().numpy(), dx=float(r.ctx.config.dx), c0=float(sc.fluid.fixedSoundSpeed), dt=float(r.ctx.config.dt))
