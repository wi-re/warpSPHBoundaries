"""Throughput of the boundary pipeline (grid -> adjacency -> Warp P1 weights -> operation) for a tank wall.   python scripts/bench/boundary_bench.py"""
import time

import numpy as np
import torch

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
from warpSPHBoundaries.scene import boundaryOps as B


def tank_mesh(L=1.0, thick=0.2, ds=0.05, dev="cuda:0"):
    """thick frame around [0,L]^2 as a regular triangulation of 4 slabs, edge length ds."""
    verts, elems = [], []
    def slab(x0, y0, x1, y1):
        nx, ny = max(1, int(round((x1 - x0) / ds))), max(1, int(round((y1 - y0) / ds)))
        base = len(verts)
        for j in range(ny + 1):
            for i in range(nx + 1):
                verts.append((x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny))
        for j in range(ny):
            for i in range(nx):
                a = base + j * (nx + 1) + i
                b, c, d = a + 1, a + nx + 1, a + nx + 2
                elems.append((a, b, d)); elems.append((a, d, c))
    slab(-thick, -thick, L + thick, 0.0); slab(-thick, L, L + thick, L + thick); slab(-thick, 0.0, 0.0, L); slab(L, 0.0, L + thick, L)
    return B.BoundaryMesh(torch.tensor(verts, dtype=torch.float64, device=dev), torch.tensor(elems, dtype=torch.int32, device=dev))


def run(N=1_000_000, h_over_dx=3.0, dev="cuda:0"):
    L = 1.0
    dx = L / np.sqrt(N)
    h = h_over_dx * dx
    mesh = tank_mesh(L, thick=h * 1.2, ds=max(h / 2, 0.01), dev=dev)
    n = int(np.sqrt(N))
    g = (np.arange(n) + 0.5) / n
    pos = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=dev)
    ps = ParticleState(positions=t(pos), supports=t(np.full(len(pos), h)), masses=t(np.full(len(pos), dx * dx)), kinds=torch.zeros(len(pos), dtype=torch.int32, device=dev), densities=t(np.ones(len(pos))))
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density, operationMode=OperationDirection.BoundaryToFluid)
    B.buildBoundaryAdjacency(ps, props, mesh)             # warm up (compile)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    grid = B.buildElementGrid(mesh, h)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    adj = B.buildBoundaryAdjacency(ps, props, mesh, grid=grid)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    P = len(adj.pairQuery)
    pr = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.BoundaryToFluid)
    q = t(np.ones(len(pos))); ref = t(np.ones(mesh.vertices.shape[0]))
    B.boundaryOperation(ps, pr, mesh, q, ref, 1.0, adjacency=adj)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    for _ in range(5):
        B.boundaryOperation(ps, pr, mesh, q, ref, 1.0, adjacency=adj)
    torch.cuda.synchronize(); t4 = time.perf_counter()
    print(f"\n{len(pos)} fluid particles, {mesh.elements.shape[0]} wall elements (edge h/2), h = {h_over_dx} dx, Wendland C2, {dev}\n")
    print(f"| stage | time |\n|---|---|\n| element grid | {(t1-t0)*1e3:.1f} ms |\n| adjacency incl. P1 weights ({P} pairs, {len(torch.unique(adj.pairQuery))} particles) | {(t2-t1)*1e3:.1f} ms |\n"
          f"| Gradient/Symmetric operation on cached adjacency | {(t4-t3)/5*1e3:.2f} ms |")
    return P


if __name__ == "__main__":
    run()
