"""ms/step of a periodic box with an M x M array of fibres (disks), as ONE bundle body (`DiskArrayRep`) or as M^2 separate one-disk bodies (docs/disk-element.md "Per-fibre kinematics"), CUDA graph replay.
The gate of docs/plan-next-steps.md item 3: 16 separate bodies < 2x the bundle.

    python scripts/studies/fibre_bodies_bench.py [--n 80] [--M 2,4,6] [--steps 60] [--profile]     (warpSPHCore_PRECISION=float32 for f32)
"""
import argparse
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge import precision as PR
from warpSPHBoundaries.scene.implicitBodies import DiskBody
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig


def make(n, M, bundle, dev, **cfgkw):
    dx = 1.0 / n
    cs = [((i + 0.5) / M, (j + 0.5) / M) for i in range(M) for j in range(M)]
    R = 0.25 / M
    X, Y = np.meshgrid(dx * (np.arange(n) + 0.5), dx * (np.arange(n) + 0.5), indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    keep = np.ones(len(pos), dtype=bool)
    for c in cs:
        d = pos - np.array(c)
        keep &= np.linalg.norm(d, axis=1) > R + 0.5 * dx
    pos = pos[keep]
    if bundle:
        bodies = [Body(bodyId=0, reps=[DiskArrayRep(cs, [R] * len(cs))])]
    else:
        bodies = [Body(bodyId=i, center=c, reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R))]) for i, c in enumerate(cs)]
    cfg = DeltaSPHConfig(gravity=(0.0, 0.0), c0=10.0, alpha=0.1, periodic=Periodic((0, 0), (1, 1)), bodyForce=(0.03, 0.0), shifting=True, **{"graphStep": True, **cfgkw})
    return DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, Scene(bodies, dev), cfg, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--M", default="2,4,6")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--profile", action="store_true", help="count the CUDA kernel launches of one step")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    print(PR.real.__name__, flush=True)
    for M in [int(m) for m in a.M.split(",")]:
        row = []
        for bundle in (True, False):
            sim = make(a.n, M, bundle, a.device)
            for _ in range(8):
                sim.step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(a.steps):
                sim.step()
            torch.cuda.synchronize()
            ms = (time.time() - t0) / a.steps * 1e3
            row.append(ms)
            extra = ""
            if a.profile:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as pr:
                    sim.step()
                    torch.cuda.synchronize()
                extra = f"  kernels/step {sum(e.count for e in pr.key_averages() if e.device_type.name == 'CUDA')}"
            print(f"  M = {M} ({M * M} fibres, N = {len(sim.x)}): {'bundle  ' if bundle else 'separate'} {ms:7.2f} ms/step{extra}   graphed {bool(getattr(sim, '_graphed', False))}", flush=True)
        print(f"  M = {M}: separate / bundle = {row[1] / row[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
