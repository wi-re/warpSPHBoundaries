"""Render dam-break snapshots (npz from `.tmp/omni/runcase.py`-style runs: t, x [F,N,2], v, rho, p, lo, hi, r) as an mp4, one panel per run, particles coloured by speed.

    python -m edgebound.dfsph_video out.mp4 "omniSPH=runs/omni.npz" "ours=runs/ours.npz" [--vmax 5] [--fps 30]
"""
import subprocess
import sys

import numpy as np


def render(out, runs, vmax=5.0, fps=30, width=None, color="speed", t_max=None):
    """`color`: "speed" | "p", or one per panel; `vmax`: one value or one per panel.  A `poly` array in the npz (vertices [K,2]) is drawn as a static solid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    data = [(name, np.load(f)) for name, f in runs]
    colors = color if isinstance(color, (list, tuple)) else [color] * len(data)
    vmaxs = vmax if isinstance(vmax, (list, tuple)) else [vmax] * len(data)
    nF = min(len(d["t"]) for _, d in data)
    if t_max is not None:
        nF = min(nF, int(np.searchsorted(data[0][1]["t"], t_max)) + 1)
    lo, hi = data[0][1]["lo"], data[0][1]["hi"]
    width = width or (7.0 if data[0][1]["x"].shape[1] < 5000 else 11.0)          # more pixels per particle at high counts
    pad, padTop = 0.05, 0.14
    ar = (hi[1] - lo[1] + pad + padTop) / (hi[0] - lo[0] + 2 * pad)
    k = len(data)
    fig, axs = plt.subplots(k, 1, figsize=(width, width * ar * k + 0.5 * k), dpi=100, squeeze=False)
    axs = axs[:, 0]
    r = float(data[0][1]["r"])
    # marker size in points^2 for a disc of radius ~ r (the lattice spacing is 0.89 h / sqrt(20) ... ~ 1.79 r): s = (2 r px)^2
    fig.canvas.draw()
    w_px = axs[0].get_window_extent().width
    pt = (2 * r * 1.0) / (hi[0] - lo[0] + 2 * pad) * w_px * 72 / fig.dpi
    sc = []
    for ax, (name, d), vm in zip(axs, data, vmaxs):
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + padTop)
        ax.set_aspect("equal")
        ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], "k-", lw=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        s = ax.scatter(d["x"][0][:, 0], d["x"][0][:, 1], s=pt ** 2, c=np.zeros(len(d["x"][0])), cmap="viridis", vmin=0, vmax=vm, marker="o", linewidths=0)
        lab = ax.text(0.005, 0.995, name, transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold")
        tt = ax.text(0.995, 0.995, "", transform=ax.transAxes, va="top", ha="right", fontsize=10)
        lab.set_in_layout(False)
        tt.set_in_layout(False)
        sc.append((s, lab))
        if "poly" in d.files:
            ax.add_patch(Polygon(d["poly"], closed=True, facecolor="0.55", edgecolor="k", lw=1.0, zorder=3))
        if "body" in d.files:                                                       # hexagonal obstacle (circumradius 0.06, pose [cx, cy, angle]) as a patch
            ax.add_patch(Polygon(np.zeros((6, 2)), closed=True, facecolor="0.55", edgecolor="k", lw=1.0, zorder=3))
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    W, H = W - W % 2, H - H % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{fig.canvas.get_width_height()[0]}x{fig.canvas.get_width_height()[1]}", "-r", str(fps), "-i", "-",
           "-vf", f"crop={W}:{H}:0:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(nF):
        for (s, _), ax, (name, d), col in zip(sc, axs, data, colors):
            s.set_offsets(d["x"][f])
            val = np.linalg.norm(d["v"][f], axis=1) if col == "speed" else d["p"][f]
            s.set_array(val)
            ax.texts[-1].set_text(f"t = {d['t'][f]:.3f} s")
            if "body" in d.files:
                cx, cy, a = d["body"][f]
                k6 = a + np.arange(6) * np.pi / 3
                ax.patches[-1].set_xy(np.stack([cx + 0.06 * np.cos(k6), cy + 0.06 * np.sin(k6)], 1))
        fig.canvas.draw()
        p.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    p.stdin.close()
    p.wait()
    plt.close(fig)


def ceiling_stats(npz, top=1.0, band=None):
    """sticking diagnostics of a snapshot file: (t, number of particles within `band` (1.5 h) of the ceiling per snapshot, longest uninterrupted residence of each particle there [s]).
    A splash touches the ceiling for ~0.2-0.3 s (omniSPH: 0.22-0.29 s); a wall suction holds particles there for 0.7 s and more."""
    d = np.load(npz)
    t, x, band = d["t"], d["x"], band or 1.5 * float(d["h"])
    near = x[:, :, 1] > top - band
    run = np.zeros(x.shape[1], int)
    best = np.zeros(x.shape[1], int)
    for i in range(len(t)):
        run = np.where(near[i], run + 1, 0)
        best = np.maximum(best, run)
    return t, near.sum(1), best * (t[1] - t[0])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    runs = [tuple(a.rsplit("=", 1)) for a in args[1:]]
    render(args[0], runs, vmax=float(opts.get("vmax", 5.0)), fps=int(opts.get("fps", 30)), t_max=float(opts["tmax"]) if "tmax" in opts else None)
