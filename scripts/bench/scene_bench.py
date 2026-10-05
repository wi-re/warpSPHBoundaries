"""Scene pipeline cost: a tank wall as ONE closed surface loop plus many small rotating polygon bodies and implicit disks among 1e6 particles.
python scripts/bench/scene_bench.py"""
import time

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
from edgebound.scene.implicitBodies import DiskBody
from edgebound.scene.scene import Body, BodyField, ImplicitRep, Scene, SurfaceRep, sceneOperation


def run(N=1_000_000, nBodies=100, dev="cuda:0", h_over_dx=3.0):
    L = 1.0
    dx = L / np.sqrt(N)
    h = h_over_dx * dx
    n = int(np.sqrt(N))
    g = (np.arange(n) + 0.5) / n
    pos = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    ps = ParticleState(positions=t(pos), supports=t(np.full(len(pos), h)), masses=t(np.full(len(pos), dx * dx)),
                       kinds=torch.zeros(len(pos), dtype=torch.int32, device=dev), densities=t(np.ones(len(pos))))
    rng = np.random.default_rng(0)
    # tank: the fluid box as a hole in an infinite solid: ONE clockwise loop with 2000 edges (subdivided only for the edge count; one edge per wall would do)
    m = 500
    s = np.linspace(0, 1, m, endpoint=False)
    loop = np.concatenate([np.stack([s, 0 * s], 1), np.stack([1 + 0 * s, s], 1), np.stack([1 - s, 1 + 0 * s], 1), np.stack([0 * s, 1 - s], 1)])
    bodies = [Body(bodyId=0, reps=[SurfaceRep.polygon(loop, solid="outside")])]
    R = 0.012
    for i in range(nBodies):
        c = rng.uniform(0.1, 0.9, 2)
        k = int(rng.integers(3, 9))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        P = R * np.stack([np.cos(ang) * rng.uniform(0.6, 1.0, k), np.sin(ang) * rng.uniform(0.6, 1.0, k)], 1)
        if i % 2:
            bodies.append(Body(bodyId=i + 1, center=tuple(c), angle=rng.uniform(0, 6), angularVelocity=1.0, reps=[SurfaceRep.polygon(np.concatenate([P, P[:1] * 0 + P[:1]])[:k]) if k > 2 else None]))
        else:
            bodies.append(Body(bodyId=i + 1, center=tuple(c), reps=[ImplicitRep(DiskBody((0.0, 0.0), R * 1.2))]))
    scene = Scene(bodies, dev)
    pr = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                             operationMode=OperationDirection.BoundaryToFluid)
    fields = [BodyField(torch.tensor(1.0, dtype=torch.float64, device=dev), None) for _ in bodies]
    sceneOperation(ps, pr, scene, bodyFields=fields)                  # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    adj = scene.buildAdjacency(ps, pr)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    sceneOperation(ps, pr, scene, adj, bodyFields=fields)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    for b in bodies[1:]:
        b.move(1e-3)
    adj = scene.buildAdjacency(ps, pr)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    print(f"N = {len(pos)}, h = {h:.4f}, bodies = {len(bodies)} (tank loop 2000 edges + {nBodies} small bodies)")
    print(f"adjacency {1e3 * (t1 - t0):.0f} ms ({adj.stats['pairs']} pair terms, candidates per body: max {max(adj.stats['candidates'])}), operation {1e3 * (t2 - t1):.0f} ms, "
          f"re-adjacency after moving all bodies {1e3 * (t3 - t2):.0f} ms")
    return adj


if __name__ == "__main__":
    run()
