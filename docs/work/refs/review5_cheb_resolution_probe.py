# reviewer probe for WORK-006: cost/accuracy of the Chebyshev edge plan resolution (nodes, panels) for the W^5 tensile kernels, dam-break state
import time, torch
from edgebound.sim.cases import marrone_dambreak
from edgebound.scene.tensile import tensile_vector_scene
from edgebound.edge import warpbc
from edgebound.scene import tensile as TN
sim,_ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
for _ in range(300): sim.step()
x, rho, H = sim.x, sim.rho, sim.H
lam, G, A = sim._wall_data(x, rho)
near = torch.nonzero(lam.sum(0) > 1e-9).flatten(); xn = x[near]
def t(f, n=15):
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
for f_ in ("w2", "w4"): tensile_vector_scene(sim.scene, x[near][:2], H, f_)
TN._register = lambda family: None          # the real one resets STABLE_KERNELS to (16, 8) on every call
for fam in ("w2", "w4"):
    name = fam + "p5"
    tensile_vector_scene(sim.scene, xn[:2], H, fam)                    # register
    ref = tensile_vector_scene(sim.scene, xn, H, fam).clone()           # (16, 8)
    sc = float(ref.abs().max())
    for res in ((16, 8), (12, 6), (8, 8), (8, 6), (8, 4), (6, 4), None):
        if res is None: warpbc.STABLE_KERNELS.pop(name, None)
        else: warpbc.STABLE_KERNELS[name] = res
        try:
            T = tensile_vector_scene(sim.scene, xn, H, fam)
            err = float((T-ref).abs().max())/sc
            ms = t(lambda: tensile_vector_scene(sim.scene, xn, H, fam))
            print("%s %-9s max|T - T(16,8)|/max|T| = %.2e   %.2f ms/call" % (fam, "monomial" if res is None else str(res), err, ms), flush=True)
        except Exception as e:
            print(fam, res, "ERR", repr(e)[:120])
    warpbc.STABLE_KERNELS[name] = (16, 8)
