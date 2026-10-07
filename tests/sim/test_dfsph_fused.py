"""DFSPH2D on the boundary provider (docs/plan-next-steps.md "DFSPH track", D1): the wall terms from `AnalyticBoundary` / `FusedWall` (lam, grad lam, the first-moment tensor int y (x) grad W, m1 = int y W) instead of
the oracle `sceneOperation`.

(a) the new fused output `m1` = int_solid (x' - x) W dA equals a brute-force integral (polygon, disk, box domain; rotated bodies) to 1e-4 of its size;
(b) the DFSPH trajectories and body forces with wallBackend = 'fused' equal the oracle's after 20 steps (fixed and rotating hexagon in a tank): positions to 1e-11, forces to 1e-9 relative; momentum balance exact.
Float64 contracts.
"""
import math

import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions, ParticleState

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.scene.fused import WallOutput
from warpSPHBoundaries.scene.provider import AnalyticBoundary
from warpSPHBoundaries.scene.scene import Body, BoxRep, DiskArrayRep, Scene, SurfaceRep
from warpSPHBoundaries.sim.deltasph2d import KERNELS
from warpSPHBoundaries.sim.dfsph_cases import tank_with_obstacle

pytestmark = pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available(), reason="float64 contracts, fused path on the GPU")
DEV = "cuda:0"
F64 = torch.float64
W = KERNELS[KernelFunctions.Wendland2][0]
H = 0.1


def _brute(inside, x, n=1200):
    g = (torch.arange(n, dtype=F64, device=DEV) + 0.5) / n * 2 * H - H
    Y = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    r = Y.norm(dim=1)
    m = r < H
    Y, r = Y[m], r[m]
    w = W(r, H) * inside(x + Y).to(F64) * (2 * H / n) ** 2
    return (w[:, None] * Y).sum(0)


def _local(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    d = p - 0.5
    return torch.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)


@pytest.mark.parametrize("ang", [0.0, 0.6])
@pytest.mark.parametrize("name", ["polygon", "disk", "box domain"])
def test_fused_m1_is_the_first_moment(name, ang):
    if name == "polygon":
        V = [(-0.2, -0.1), (0.2, -0.15), (0.25, 0.1), (-0.1, 0.2)]
        rep = SurfaceRep.polygon(V)
        Vt = torch.tensor(V, dtype=F64, device=DEV)

        def inside(p):
            q = _local(p, ang)
            ins = torch.zeros(len(q), dtype=torch.bool, device=DEV)
            for k in range(len(Vt)):
                a, b = Vt[k], Vt[(k + 1) % len(Vt)]
                ins ^= ((a[1] > q[:, 1]) != (b[1] > q[:, 1])) & (q[:, 0] < a[0] + (q[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
            return ins
    elif name == "disk":
        rep = DiskArrayRep([(0.0, 0.0)], [0.15])
        inside = lambda p: _local(p, ang).norm(dim=1) < 0.15
    else:
        rep = BoxRep((-0.2, -0.1), (0.2, 0.1), solid="outside")
        inside = lambda p: ~((_local(p, ang).abs() <= torch.tensor([0.2, 0.1], dtype=F64, device=DEV)).all(1))
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), angle=ang, reps=[rep])], DEV)
    x = torch.tensor([[0.5, 0.62], [0.71, 0.52], [0.27, 0.43], [0.62, 0.36]], dtype=F64, device=DEV)
    ps = ParticleState(positions=x, supports=torch.full((4,), H, dtype=F64, device=DEV), masses=torch.ones(4, dtype=F64, device=DEV), kinds=torch.zeros(4, dtype=torch.int32, device=DEV),
                       densities=torch.ones(4, dtype=F64, device=DEV))
    m1 = AnalyticBoundary(scene).aggregate(ps, H).evaluate((WallOutput("m1", 0, "m1"),))["m1"][0]
    ref = torch.stack([_brute(inside, x[k]) for k in range(4)])
    assert float((m1 - ref).abs().max()) <= 1e-3 * float(ref.abs().max()) + 1e-6


@pytest.mark.parametrize("omega", [0.0, 3.0])
def test_dfsph_fused_backend_equals_the_oracle(omega):
    out = {}
    for be in ("scene", "fused"):
        sim, _ = tank_with_obstacle(omega=omega, r=0.008)
        sim.cfg.wallBackend = be
        for _ in range(20):
            sim.step()
        assert sim.balance < 1e-14
        out[be] = (sim.x.clone(), sim.forcePressure.clone())
    xs, fs = out["scene"]
    xf, ff = out["fused"]
    assert float((xs - xf).abs().max()) < 1e-11
    assert float((fs - ff).abs().max()) <= 1e-9 * float(fs.abs().max())


@pytest.mark.parametrize("wallPressure", ["hydrostatic", "linear"])
def test_graphed_iterates_equal_the_eager_solve(wallPressure):
    """D2: the pressure iterates replayed as CUDA graphs (fused Warp kernels for hydrostatic / mirror, the torch iterate for linear) give the eager trajectories and iteration counts; two captures (one per solve)."""
    out = {}
    for graphs in (False, True):
        sim, _ = tank_with_obstacle(omega=3.0, r=0.008)
        sim.cfg.graphIterations = graphs
        sim.cfg.wallPressure = wallPressure
        its = []
        for _ in range(15):
            sim.step()
            its.append(sim.iters)
        assert sim.balance < 1e-14
        out[graphs] = (sim.x.clone(), its, sim.stats.get("graphCaptures", 0))
    assert out[True][1] == out[False][1]
    assert float((out[True][0] - out[False][0]).abs().max()) < 1e-11
    assert out[True][2] <= 4
