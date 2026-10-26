"""Plane slices of analytic ball overlap and kernel mass/gradient around a tet.

Plots the true section of the tetrahedron, not its projected silhouette.
All samples use the analytic reference; display interpolation adds no geometric
approximation to stored data. This is a CPU research viewer, not a GPU path.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from warpSPHBoundaries.mesh3d import Tetrahedron
from warpSPHBoundaries.validation.paths import evidence_path


VERTICES = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])


def section(vertices, fixed, offset, axes):
    points = []
    for i in range(4):
        for j in range(i):
            a, b = vertices[i], vertices[j]
            if a[fixed] == offset:
                points.append(a[list(axes)])
            if (a[fixed]-offset)*(b[fixed]-offset) < 0:
                p = a+(b-a)*(offset-a[fixed])/(b[fixed]-a[fixed])
                points.append(p[list(axes)])
            if b[fixed] == offset:
                points.append(b[list(axes)])
    if not points:
        return np.empty((0, 2))
    points = np.unique(points, axis=0)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    return np.concatenate([points, points[:1]])


def run(output, grid=41, support=.3, offset=.15, name="w2", dps=35):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    span = np.linspace(-support*1.1, 1+support*1.1, grid)
    xx, yy = np.meshgrid(span, span)
    values, overlaps, gradients = [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for column, (a, b, fixed) in enumerate([(0, 1, 2), (0, 2, 1), (1, 2, 0)]):
        mass = np.zeros((grid, grid))
        overlap = np.zeros_like(mass)
        grad = np.zeros((grid, grid, 3))
        with mp.workdps(dps):
            r = mp.mpf(support)
            ball_volume = 4*mp.pi*r**3/3
            for i, j in np.ndindex(grid, grid):
                q = [offset]*3
                q[a], q[b] = float(xx[i, j]), float(yy[i, j])
                t = Tetrahedron(VERTICES, q)
                mass[i, j] = float(t.moment(name, h=support))
                overlap[i, j] = float(t.block_moment((0, 0, 0), 0, r)/ball_volume)
                grad[i, j] = np.array(t.kernel_gradient(name, h=support), float)
        values.append(mass)
        overlaps.append(overlap)
        gradients.append(grad)
        polygon = section(VERTICES, fixed, offset, (a, b))
        for row, field in enumerate((overlap, mass)):
            ax = axes[row, column]
            im = ax.pcolormesh(xx, yy, field, cmap="viridis", vmin=0, vmax=1, shading="nearest", rasterized=True)
            if len(polygon):
                ax.plot(polygon[:, 0], polygon[:, 1], color="white", linewidth=2)
                ax.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=.7)
            ax.set(xlabel=f"{'xyz'[a]} / L", ylabel=f"{'xyz'[b]} / L", aspect="equal",
                   title=f"{'Ball overlap fraction' if row == 0 else name+' kernel mass'} · {'xyz'[fixed]}={offset:g} L")
            if row == 1:
                step = max(1, grid//10)
                sl = (slice(None, None, step), slice(None, None, step))
                ax.quiver(xx[sl], yy[sl], grad[sl][..., a], grad[sl][..., b],
                          color="white", scale=30/support, width=.004, pivot="mid")
        print(f"slice {'xyz'[a]}{'xyz'[b]} complete ({grid}x{grid})", flush=True)
    fig.colorbar(im, ax=axes, label="Fraction / normalized kernel mass", shrink=.85)
    fig.suptitle(f"Tetrahedron boundary integrals · support h={support:g} L\nWhite arrows: query gradient of kernel mass; outline: solid section")
    fig.savefig(output/"slices.png", dpi=180)
    fig.savefig(output/"slices.svg")
    plt.close(fig)
    np.savez_compressed(output/"slices.npz", coordinate=span, vertices=VERTICES, offset=offset,
                        support=support, kernel=name, dps=dps, mass=values, ball_fraction=overlaps,
                        gradient=gradients, planes=np.array(["xy", "xz", "yz"]))
    print(output/"slices.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=evidence_path("mesh3d-slices"))
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--support", type=float, default=.3)
    parser.add_argument("--offset", type=float, default=.15)
    parser.add_argument("--kernel", choices=["cubic", "w2", "w4", "w6"], default="w2")
    parser.add_argument("--dps", type=int, default=35)
    args = parser.parse_args()
    run(args.output, args.grid, args.support, args.offset, args.kernel, args.dps)
