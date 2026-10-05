"""Reviewer probe for WORK-005 T5.1 (throw-away): Delta-lambda = int_solid lap W dA' through the UNMODIFIED Density and Covariance scene operations on the registered, ORDINARY (normalised)
kernel L(r) = W'(r)/r  (a polynomial for Wendland: no negative powers).   lap W = div(L y) = 2 L + r L'  =>  Delta-lambda = 2 lambda[L] - tr Cov[L]   (Cov = int y (x) grad_x L; tr Cov = -int r L').
A direct registration of lap W as a kernel does NOT work: the scene adds the body-indicator pseudo-pair with lambda = 1 (the integral of a normalised kernel over a full disk), but int lap W = 0 over a
disk -> a spurious constant c/H^2 for every particle INSIDE a body (found by the reviewer: 28.0 = 7/0.5^2 at the centre of a square).  With L the indicator enters as 2*ind - tr(ind*I) = 0.
Conversion: W = c/(pi H^2) s(q), c = KERNELS[fam].c2_pi;  L = c/(pi H^4) l(q),  l = s'/q:  w2: l = -20 (1-q)^3;  w4: l = -(56/3) (1-q)^5 (5q+1) = -(56/3)(6u^5 - 5u^6), u = 1-q.
Register K_l = normalised kernel with shape (1-q)^3 (w2) / (6u^5 - 5u^6) (w4), normalisation C_l = KERNELS[..].c2_pi;  K_l(r;h) = C_l shape/(pi h^2);
L = f K_l with f = -20 c/(C_l H^2) (w2),  -(56/3) c/(C_l H^2) (w4).     Delta-lambda = f (2 lambda[K_l] - tr Cov[K_l])."""
import sys; sys.path.insert(0, "/home/lu26029/dev/curvatureBoundaries/python"); sys.path.insert(0, "/home/lu26029/dev/curvatureBoundaries/docs/work/refs")
from fractions import Fraction as F
import numpy as np, torch
from edgebound.edge import kernels
from edgebound.scene.scene import Body, Scene, SurfaceRep, BodyField, sceneOperation
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
from lap_calib_probe import AB

TERMS = {"w2": [(1, 1, 3)], "w4": [(6, 1, 5), (-5, 1, 6)]}
PRE = {"w2": F(-20), "w4": F(-56, 3)}
def register(fam):
    name = "l" + fam
    if name not in kernels.KERNELS: kernels.KERNELS[name] = kernels._from_terms(name, TERMS[fam])
    return name
dev = "cuda:0"
def delta_lambda_scene(scene, pos, H, fam):
    """per-body Delta-lambda [B, N] for an arbitrary SurfaceRep scene."""
    name = register(fam); n = len(pos); B = len(scene.bodies)
    ps = ParticleState(positions=pos, supports=torch.full((n,), float(H), dtype=torch.float64, device=pos.device), masses=torch.ones(n, dtype=torch.float64, device=pos.device),
                       kinds=torch.zeros(n, dtype=torch.int32, device=pos.device), densities=torch.ones(n, dtype=torch.float64, device=pos.device))
    res = {}
    for op in (WarpOperation.Density, WarpOperation.Covariance):
        pr = OperationProperties(kernel=name, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
        res[op] = sceneOperation(ps, pr, scene, None, None, [BodyField(rho=1.0)] * B, perBody=True)
    lam = res[WarpOperation.Density].reshape(B, n); cov = res[WarpOperation.Covariance].reshape(B, n, 2, 2)
    c = float(F(kernels.KERNELS[fam].c2_pi)); Cl = float(F(kernels.KERNELS[name].c2_pi))
    f = float(PRE[fam]) * c / (Cl * H ** 2)
    return f * (2 * lam - cov[:, :, 0, 0] - cov[:, :, 1, 1])

FLOOR = [(-5.0, -2.0), (5.0, -2.0), (5.0, 0.0), (-5.0, 0.0)]
sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])], dev)
def delta_lambda(fam, pts, H):
    return delta_lambda_scene(sc, torch.as_tensor(np.asarray(pts), dtype=torch.float64, device=dev), H, fam)[0].cpu().numpy()
if __name__ == "__main__":
    zs = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9]
    for fam in ("w2", "w4"):
        for H in (1.0, 0.7):
            got = delta_lambda(fam, [(0.0, z * H) for z in zs], H)
            ref = [abs(AB(fam, z)[2]) / H ** 2 for z in zs]          # B(z; H) = B(z/H; 1)/H^2
            print(fam, f"H={H}  max rel err vs independent B(z):", max(abs(g - r) / r for g, r in zip(got, ref)), " abs err / max", max(abs(g - r) for g, r in zip(got, ref)) / max(ref))
