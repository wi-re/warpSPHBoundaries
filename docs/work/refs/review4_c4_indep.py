"""REVIEW-004 independent check: C4 tensile T = int_{y'<0} W^4 grad_x W dA' (flat floor y=0, H) on a plain-numpy polar-in-x'/midpoint grid,
shares no code with warpSPHBoundaries.scene.tensile.  Point (0.2, 0.45), H=1 and H=0.7 (scaling T ~ H^-9... checked directly)."""
import sys, numpy as np
sys.path.insert(0, "python")
def make(H):
    # Wendland C4 2D, support H: W = c/H^2 (1-q)^6 (35/3 q^2 + 6 q + 1), normalised numerically
    qq = (np.arange(400000) + .5) / 400000
    c = 1.0 / (2 * np.pi * np.sum((1 - qq) ** 6 * (35 / 3 * qq ** 2 + 6 * qq + 1) * qq) / 400000)
    W  = lambda r: c / H**2 * np.clip(1 - r / H, 0, None) ** 6 * (35 / 3 * (r / H) ** 2 + 6 * r / H + 1)
    dW = lambda r: c / H**3 * (-6 * np.clip(1 - r / H, 0, None) ** 5 * (35 / 3 * (r / H) ** 2 + 6 * r / H + 1)
                              + np.clip(1 - r / H, 0, None) ** 6 * (70 / 3 * r / H + 6))
    return W, dW
def grid(H, x0, y0, n=3000):
    W, dW = make(H)
    # solid y' in [-H, 0], x' in [x0-H, x0+H]; the Wendland shape has a (1-q)^5 factor at the edge so the grid converges fast
    xs = x0 + (np.arange(n) + .5) / n * 2 * H - H
    ys = -(np.arange(n) + .5) / n * H
    X, Y = np.meshgrid(xs, ys)
    dx, dy = x0 - X, y0 - Y
    r = np.hypot(dx, dy)
    f = np.where(r < H, W(r) ** 4 * dW(r) / np.where(r > 0, r, 1), 0.0)
    cell = (2 * H / n) * (H / n)
    return np.array([np.sum(f * dx), np.sum(f * dy)]) * cell
if __name__ == "__main__":
    import torch
    from warpSPHBoundaries.scene.scene import Scene, Body, SurfaceRep
    from warpSPHBoundaries.scene.tensile import tensile_vector_scene
    FLOOR = [(-5.0, -2.0), (5.0, -2.0), (5.0, 0.0), (-5.0, 0.0)]   # counter-clockwise, solid y < 0
    dev = "cuda:0"
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(FLOOR)])], dev)
    for H, (x0, y0) in [(1.0, (0.2, 0.45)), (0.7, (-0.3, 0.1)), (1.0, (0.0, 0.9))]:
        T = tensile_vector_scene(sc, np.array([[x0, y0]]), H, "w4")[0].cpu().numpy()
        G = grid(H, x0, y0)
        print(f"H={H} x=({x0},{y0})  scene T={T}  grid T={G}  rel err={np.abs(T-G).max()/np.abs(G).max():.2e}")
