"""Throughput of the Warp kernels (float64) vs numpy float64 / torch float64 (kernel time only, data resident on the device).

    python -m edgebound.warp_bench
"""
import time

import numpy as np
import torch
import warp as wp

from . import np2d, torch2d, warp2d


def run(N=1_000_000, kernel="w4"):
    rng = np.random.default_rng(1)
    V = rng.uniform(-1, 1, (N, 3, 2))
    X = rng.uniform(-0.4, 0.4, (N, 2))
    dev = "cuda:0"
    wv = wp.array(V, dtype=wp.vec2d, device=dev)
    wx = wp.array(X, dtype=wp.vec2d, device=dev)
    wh = wp.array(np.ones(N), dtype=wp.float64, device=dev)
    out = wp.zeros(N, dtype=wp.float64, device=dev)
    rows = []
    for al in [(0, 0), (1, 1), (2, 2)]:
        L = warp2d.MomentLauncher(kernel, al, dev)
        L.run(wv, wx, wh, out)
        wp.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            L.run(wv, wx, wh, out)
        wp.synchronize()
        tw = (time.perf_counter() - t) / 5
        n = 100_000
        t = time.perf_counter()
        np2d.moment(V[:n], X[:n], kernel, al)
        tn = (time.perf_counter() - t) / n * N
        Vt, Xt = torch.tensor(V[:n]).cuda(), torch.tensor(X[:n]).cuda()
        torch2d.moment(Vt, Xt, kernel, al)
        torch.cuda.synchronize()
        t = time.perf_counter()
        torch2d.moment(Vt, Xt, kernel, al)
        torch.cuda.synchronize()
        tt = (time.perf_counter() - t) / n * N
        rows.append((al, tw, tn, tt))
    print(f"\n{N} triangle/point pairs, kernel {kernel}, float64 (seconds per {N} pairs; numpy extrapolated from 1e5)\n")
    print("| alpha | warp CUDA | numpy (1 thread) | torch CUDA (eager) | warp speed-up vs numpy |\n|---|---|---|---|---|")
    for al, tw, tn, tt in rows:
        print(f"| {al} | {tw:.4f} | {tn:.2f} | {tt:.3f} | {tn / tw:.0f}x |")
    return rows


if __name__ == "__main__":
    run()
