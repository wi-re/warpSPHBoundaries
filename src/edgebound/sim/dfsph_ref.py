"""omniSPH as a live reference: build the SAME case (positions, V, h, domain) in the compiled `omnySPH` module and in `DFSPH2D`.

omniSPH walls are thick triangle slabs of thickness `eps_adj` around the box [domain.min, domain.max]; the inner faces are placed half a lattice spacing outside the
outermost particles, so the lattice continues into the wall.  Needs the working directory of omniSPH's `cfg/` (a symlink in the scratch directory)."""
import math
import sys

import numpy as np

from .. import paths

sys.path.insert(0, str(paths.OMNISPH_HOME / "omnySPH" / "src"))
PACKING = 0.399200743165053487


def _yaml(r, fmin, fmax, dmin, dmax, eps, gravity=9.81, maxDt=1e-3, minDt=1e-4, incompressible=True, divergence=True):
    return f"""
fluids:
     - min: [{fmin[0]}, {fmin[1]}]
       max: [{fmax[0]}, {fmax[1]}]
       radius: {r}
       type: once
       velocity: [0, 0]
gravity:
    - pointSource: false
      direction: [0., -1.]
      magnitude: {gravity}
props:
    maxnumptcls: 512000
    backgroundPressure: false
dfsph:
  divergenceSolve: {str(divergence).lower()}
domain:
    min: [{dmin[0]}, {dmin[1]}]
    max: [{dmax[0]}, {dmax[1]}]
    epsilon: {eps}
sim:
    maxDt: {maxDt}
    minDt: {minDt}
    incompressible: {str(incompressible).lower()}
"""


def adjusted_epsilon(r, eps):
    h = r * math.sqrt(TARGET)
    packing = PACKING * h
    rows = math.ceil(eps / packing)
    return rows * packing - 0.99 * 0.23999418487168855 * h


TARGET = 20


def omni_case(r, fmin, fmax, eps=0.06, top=None, right=None, **kw):
    """returns dict(sim, x, V, h, lo, hi (inner faces), epsAdj).  `top` / `right`: inner face coordinates of the top / right wall (default: half a spacing outside the fluid)."""
    import omnySPH
    dummy = omnySPH.SPHSimulation(_yaml(r, fmin, fmax, (-5, -5), (5, 5), eps))
    n = dummy.getInteger("props.numPtcls")
    x = np.array(dummy.fluidPosition)[:n].copy()
    V = np.array(dummy.fluidArea)[:n].copy()
    h = np.array(dummy.fluidSupport)[:n].copy()
    ux, uy = np.unique(np.round(x[:, 0], 12)), np.unique(np.round(x[:, 1], 12))
    dx, dy = np.median(np.diff(ux)), np.median(np.diff(uy))
    lo = np.array([x[:, 0].min() - dx / 2, x[:, 1].min() - dy / 2])
    hi = np.array([x[:, 0].max() + dx / 2 if right is None else right, x[:, 1].max() + dy / 2 if top is None else top])
    ea = adjusted_epsilon(r, eps)
    sim = omnySPH.SPHSimulation(_yaml(r, fmin, fmax, lo - ea, hi + ea, eps, **kw))
    assert abs(sim.getScalar("domain.epsilon") - ea) < 1e-12, (sim.getScalar("domain.epsilon"), ea)
    n2 = sim.getInteger("props.numPtcls")
    assert n2 == n
    return dict(sim=sim, x=x, V=V, h=h, lo=lo, hi=hi, epsAdj=ea, dx=dx, dy=dy, n=n)


def omni_state(sim, n):
    return dict(x=np.array(sim.fluidPosition)[:n].copy(), v=np.array(sim.fluidVelocity)[:n].copy(), rho=np.array(sim.fluidDensity)[:n].copy(),
                p=np.array(sim.fluidPressure1)[:n].copy())
