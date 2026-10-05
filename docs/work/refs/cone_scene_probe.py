"""Reviewer's probe for WORK-002 T2.1 (throw-away, NOT part of the deliverable; copy to .tmp/ to run, do not import in tests).
Shows: (1) a degree-1 'cone' EdgeKernel registered in edgebound.edge.kernels.KERNELS runs through the UNMODIFIED Warp edge kernel;
(2) the scene path accepts a kernel *name* if kernelName passes strings through;  both reproduce cover.cover_vector_np.
Conversion: W_cone = 3 (1-q)/(pi h^2)  (c2_pi = 3), K(r) = (r-H) 1[r<=H] = -H (1-q) at h = H, so
    grad_x int_solid K dA = -H * (pi h^2 / 3) * (grad_x int W_cone dA) = -(pi H^3 / 3) * g0     (h = H).
Reviewer's result 2026-10-03: max |scene - cover_vector_np| = 3.9e-16 (rotated + translated L-shaped body, 200 random points, H = 0.6);
raw edge_channels on a unit square, H = 0.7: 1.4e-16.   Run from the repo root:  python docs/work/refs/cone_scene_probe.py"""
import math, sys
from fractions import Fraction as F
sys.path.insert(0, "python")
import numpy as np, torch
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation
from edgebound.scene import cover, scene as S
from edgebound.edge import kernels
from edgebound.scene.scene import Body, BodyField, Scene, SurfaceRep, sceneOperation

kernels.KERNELS["cone"] = kernels._from_terms("cone", [(1, F(1), 1)])        # (coef, knot, power): shape (1 - q)^1, normalisation 3
_orig = S.kernelName
S.kernelName = lambda k: k if isinstance(k, str) else _orig(k)               # what the WORK-002 change in boundaryOps.kernelName has to do properly
dev, TD, H = "cuda:0", torch.float64, 0.6
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=(0.3, -0.2), angle=0.7, linearVelocity=(0, 0), angularVelocity=0)], dev)
pos = np.random.default_rng(3).uniform(-2, 3, (200, 2)); n = len(pos)
t = lambda a: torch.as_tensor(a, dtype=TD, device=dev)
ps = ParticleState(positions=t(pos), supports=t(np.full(n, H)), masses=t(np.ones(n)), kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=t(np.ones(n)))
pr = OperationProperties(kernel="cone", operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
g0 = sceneOperation(ps, pr, sc, None, None, [BodyField(torch.tensor(1.0, dtype=TD, device=dev))], perBody=True)[0].cpu().numpy()
res = -(math.pi * H ** 3 / 3) * g0
R = np.array([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
ref = cover.cover_vector_np(pos, LS @ R.T + np.array([0.3, -0.2]), H)         # world polygon: rotate by angle, then translate by center
print("max |scene - cover_vector_np| =", np.abs(res - ref).max(), "  max|ref| =", np.abs(ref).max())
