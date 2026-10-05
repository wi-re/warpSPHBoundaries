import sys; sys.path.insert(0, "/home/lu26029/dev/curvatureBoundaries/docs/work/refs")
import numpy as np, torch, math
from lap_lshape_probe import *
import lap_scene_probe as P
from edgebound.edge import kernels
from warpSPHCore import *
from edgebound.scene.scene import sceneOperation, BodyField
CEN, ANG, H = (0.3, -0.2), 0.7, 0.6
sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CEN, angle=ANG)], dev)
wp_ = LS @ rot(ANG).T + np.asarray(CEN)
pts = np.random.default_rng(3).uniform(-2, 3, (200, 2)); pt = torch.as_tensor(pts, device=dev)
for fam in ("w2", "w4"):
    full = P.delta_lambda_scene(sc, pt, H, fam)[0].cpu().numpy()
    ref = brute(fam, pts, H, wp_); s = np.abs(ref).max()
    # control: 2 lambda only (trace dropped)
    name = P.register(fam); n = len(pts)
    ps = ParticleState(positions=pt, supports=torch.full((n,), H, dtype=torch.float64, device=dev), masses=torch.ones(n, dtype=torch.float64, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=torch.ones(n, dtype=torch.float64, device=dev))
    pr = OperationProperties(kernel=name, operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    lam = sceneOperation(ps, pr, sc, None, None, [BodyField(rho=1.0)], perBody=True).reshape(n).cpu().numpy()
    f = float(P.PRE[fam]) * float(kernels.KERNELS[fam].c2_pi) / (float(kernels.KERNELS[name].c2_pi) * H**2)
    print(fam, "scale", s, " scene-vs-brute", np.abs(full-ref).max()/s, " | control 2*lam only:", np.abs(f*2*lam-ref).max()/s, " | control -Delta:", np.abs(-full-ref).max()/s,
          " | control lam only:", np.abs(f*lam-ref).max()/s)
# nu identity numbers
from scipy import integrate
for fam in ("w2","w4"):
    from lap_calib_probe import kern
    W,dW,_ = kern(fam)
    I = 2*np.pi*integrate.quad(lambda r: r*dW(r)*r, 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
    # quadratic field v=(y^2,0): pair acc_x/fac = -int W'/r^3 x^2 y^2 dA = -(pi/4) int W' r^2 dr
    a = -(np.pi/4)*integrate.quad(lambda r: dW(r)*r*r, 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
    print(fam, "int r W' dA =", I, "  pair acc_x/fac for v=(y^2,0) =", a, " (expect -2, 0.25)")
