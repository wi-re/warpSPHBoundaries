"""Reviewer probe for WORK-005 T5.1 (throw-away): Delta-lambda of the scene route (lap_scene_probe.delta_lambda_scene) vs an independent polar midpoint brute force with an even-odd
point-in-polygon test, on (a) the rotated/translated L body of test_cover_scene.py (centre (0.3,-0.2), angle 0.7), H = 0.6, 200 points seed 3 in [-2,3]^2, w2 and w4;
(b) a cavity (SurfaceRep.polygon(box, solid='outside')) H = 0.5, 40 points in [-0.1,2.1]x[-0.1,1.1] seed 5; (c) the tangent case z == H (particle at the centre of a square of side 2H -> 0).
Prints worst |scene - brute| / max|brute|."""
import sys, math; sys.path.insert(0, "/home/lu26029/dev/curvatureBoundaries/docs/work/refs")
import numpy as np, torch
from lap_scene_probe import delta_lambda_scene, Scene, Body, SurfaceRep, dev
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
def inside_eo(P, poly):
    x, y = P[:, 0], P[:, 1]; ins = np.zeros(len(P), bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        cr = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-300) + a[0])
        ins ^= cr
    return ins
def lapW(fam, q, H):
    c = {"w2": 7.0, "w4": 9.0}[fam]
    if fam == "w2": return c / (np.pi * H ** 4) * 20 * (1 - q) ** 2 * (5 * q - 2)
    return c / (np.pi * H ** 4) * (-(112 / 3)) * (1 - q) ** 4 * (1 + 4 * q - 20 * q * q)
def brute(fam, p, H, poly, solid_inside=True, nr=500, nt=1000):
    r = (np.arange(nr) + .5) / nr * H; th = (np.arange(nt) + .5) / nt * 2 * np.pi
    R, T = np.meshgrid(r, th, indexing="ij"); lap = lapW(fam, R / H, H); out = []
    for (x, y) in p:
        P = np.stack([(x + R * np.cos(T)).ravel(), (y + R * np.sin(T)).ravel()], 1)
        s = inside_eo(P, poly) if solid_inside else ~inside_eo(P, poly)
        out.append(np.sum(lap.ravel() * s * R.ravel()) * (H / nr) * (2 * np.pi / nt))
    return np.array(out)
if __name__ == "__main__":
    CEN, ANG, H = (0.3, -0.2), 0.7, 0.6
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(LS)], center=CEN, angle=ANG)], dev)
    wp_ = LS @ rot(ANG).T + np.asarray(CEN)
    pts = np.random.default_rng(3).uniform(-2, 3, (200, 2))
    for fam in ("w2", "w4"):
        got = delta_lambda_scene(sc, torch.as_tensor(pts, device=dev), H, fam)[0].cpu().numpy()
        ref = brute(fam, pts, H, wp_)
        print(f"(a) L body {fam}: max|brute|={np.abs(ref).max():.4f}  worst |scene-brute|/max|brute| = {np.abs(got-ref).max()/np.abs(ref).max():.2e}  (#nonzero {int((np.abs(ref)>1e-9).sum())})")
    BOX = [(0, 0), (2, 0), (2, 1), (0, 1)]
    sc2 = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(BOX, solid="outside")])], dev)
    pts2 = np.random.default_rng(5).uniform([-0.1, -0.1], [2.1, 1.1], (40, 2))
    for fam in ("w2", "w4"):
        got = delta_lambda_scene(sc2, torch.as_tensor(pts2, device=dev), 0.5, fam)[0].cpu().numpy()
        ref = brute(fam, pts2, 0.5, np.array(BOX, float), solid_inside=False)
        print(f"(b) cavity {fam}: max|brute|={np.abs(ref).max():.4f}  worst rel = {np.abs(got-ref).max()/np.abs(ref).max():.2e}")
    sc3 = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])], dev)
    for fam in ("w2", "w4"):
        print(f"(c) tangent z==H centre of square side 2H, {fam}:", delta_lambda_scene(sc3, torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=dev), 0.5, fam).cpu().numpy().ravel(),
              " inside the solid (support fully inside):", delta_lambda_scene(sc3, torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=dev), 0.4, fam).cpu().numpy().ravel())
