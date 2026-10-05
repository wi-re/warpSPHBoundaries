"""`BoxRep`: an axis-aligned box (obstacle or fluid domain) evaluated by four corner lookups of two 2D tables per kernel (docs/box-domain-primitive.md), against the exact
polygon path (`SurfaceRep.box`, the library's own edge kernel) on the same body.

Scene: a box [0.1, 2.3] x [-0.2, 1.1] in a body frame rotated by 0.7 and translated; 600 queries (inside the box, within a support of the walls / corners, far away; the rows
also include points exactly on the wall lines and exactly on the corners: BoxRep is the continuous value there, the polygon path agrees on the large box) with per-query supports 0.3-0.7; a thin plate (0.05 < support) as a second box.

Tolerances (stated before looking; the table is a controlled-error model of an exact quantity, measured at the default grid before choosing): relative to the largest value of the
channel, lam <= 1e-8 for every tabulated kernel; gradient channel g0 and Cov: w2 3e-7, w4 1e-8, lw2 3e-6, w2p5 1e-8; the kinked kernel `cone` (BOX_EXACT_KERNELS) takes the exact polygon
path and agrees to round-off (1e-12).  Measured: w2 3.8e-11 / 1e-7 / 1.5e-8, w4 6e-13 / 1.7e-9 / 2.4e-10, lw2 3e-10 / 8e-7 / 1.2e-7, w2p5 1e-14 / 4e-11 / 8e-14 (lam / g0 / Cov, scale 1-7).
"""
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

from edgebound.scene import tensile
from edgebound.scene.cover import cover_vector_scene
from edgebound.scene.scene import Body, BodyField, BoxRep, Scene, SurfaceRep, sceneOperation
from edgebound.scene.tensile import tensile_vector_scene
from edgebound.scene.viscosity import lap_factor, lap_lambda_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
LO, HI = (0.1, -0.2), (2.3, 1.1)
CENTER, ANGLE = (0.3, -0.2), 0.7
TOL = {"w2": (1e-8, 3e-7), "w4": (1e-8, 1e-8), "lw2": (1e-8, 3e-6), "w2p5": (1e-8, 1e-8), "cone": (1e-12, 1e-12)}


def register():
    for fam in ("w2", "w4"):
        lap_factor(1.0, fam)                                                 # registers 'lw2' / 'lw4' (lazy registration)
        tensile._register(fam)                                               # 'w2p5' / 'w4p5'


def props(op, kernel):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)


def scenes(device, solid, lo=LO, hi=HI, angle=ANGLE):
    mk = lambda rep: Scene([Body(bodyId=0, center=CENTER, angle=angle, reps=[rep])], device)
    return mk(SurfaceRep.box(lo, hi, solid)), mk(BoxRep(lo, hi, solid))


def queries(device, lo=LO, hi=HI, n=600, seed=7, exact=True):
    """world positions: random around the box, plus points exactly on the wall lines and on the corners (body frame), mapped to the world frame."""
    rng = np.random.default_rng(seed)
    loc = rng.uniform(np.array(lo) - 1.2, np.array(hi) + 1.2, (n, 2))
    k = 3
    loc[:k, 1], loc[:k, 0] = lo[1], rng.uniform(lo[0], hi[0], k)             # exactly on the bottom wall line
    loc[k:2 * k, 0], loc[k:2 * k, 1] = hi[0], rng.uniform(lo[1], hi[1], k)   # exactly on the right wall line
    loc[2 * k] = lo; loc[2 * k + 1] = hi; loc[2 * k + 2] = (lo[0], hi[1]); loc[2 * k + 3] = (hi[0], lo[1])   # exactly on the four corners
    if not exact:                                                            # nudge them 1e-6 off the boundary (the polygon path's winding rule is ambiguous exactly ON a thin plate)
        loc[:2 * k + 4] += 1e-6 * rng.choice([-1.0, 1.0], (2 * k + 4, 2))
    c, s = np.cos(ANGLE), np.sin(ANGLE)
    R = np.array([[c, -s], [s, c]])
    world = loc @ R.T + np.array(CENTER)
    sup = rng.uniform(0.3, 0.7, n)
    return torch.as_tensor(world, dtype=F64, device=device), torch.as_tensor(sup, dtype=F64, device=device)


def state(pos, sup, device):
    n = len(pos)
    return ParticleState(positions=pos, supports=sup, masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                         densities=torch.ones(n, dtype=F64, device=device))


def ops(sc, ps, kernel, **kw):
    out = {}
    for name, op in (("lam", WarpOperation.Density), ("G", WarpOperation.Gradient), ("Cov", WarpOperation.Covariance)):
        fld = BodyField(rho=1.0) if op == WarpOperation.Density else BodyField(torch.tensor(1.0, dtype=F64, device=sc.device))
        out[name] = sceneOperation(ps, props(op, kernel), sc, None, None, [fld], **kw).cpu().numpy()
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solid", ["inside", "outside"])
def test_box_channels_equal_the_polygon_path(device, solid):
    """(a) Density, Gradient of a constant, Covariance of BoxRep vs SurfaceRep.box, every registered kernel, both orientations, rotated pose, per-query supports."""
    register()
    sS, sB = scenes(device, solid)
    pos, sup = queries(device)
    ps = state(pos, sup, device)
    for kernel, (tl, tg) in TOL.items():
        a, b = ops(sS, ps, kernel), ops(sB, ps, kernel)
        for name, tol in (("lam", tl), ("G", tg), ("Cov", tg)):
            d = float(np.abs(a[name] - b[name]).max()); sc = float(np.abs(a[name]).max())
            assert sc > 0.5, (kernel, name, sc)
            assert d <= tol * sc, (solid, kernel, name, d, sc)
        print(f"(a) {solid} {kernel}: lam {np.abs(a['lam'] - b['lam']).max():.1e} G {np.abs(a['G'] - b['G']).max():.1e} Cov {np.abs(a['Cov'] - b['Cov']).max():.1e}")


@pytest.mark.parametrize("device", DEVICES)
def test_thin_plate_and_reaction(device):
    """(b) a plate thinner than the support (0.05 < 0.3): four corners of a thin rectangle, nothing is clamped away; Gradient of a constant with returnReaction=True gives the
    same force on the body as the polygon path (the force is minus the summed contribution, a sum of exactly the checked channels)."""
    register()
    sS, sB = scenes(device, "inside", lo=(0.0, 0.0), hi=(2.0, 0.05))
    pos, sup = queries(device, lo=(0.0, 0.0), hi=(2.0, 0.05), seed=3, exact=False)
    ps = state(pos, sup, device)
    a, b = ops(sS, ps, "w2"), ops(sB, ps, "w2")
    for name, tol in (("lam", 1e-8), ("G", 3e-7), ("Cov", 3e-7)):
        d = float(np.abs(a[name] - b[name]).max()); sc = float(np.abs(a[name]).max())
        assert d <= tol * sc, (name, d, sc)
    fld = BodyField(torch.tensor(1.0, dtype=F64, device=device))
    ra = sceneOperation(ps, props(WarpOperation.Gradient, "w2"), sS, None, None, [fld], returnReaction=True)[1].force.cpu().numpy()
    rb = sceneOperation(ps, props(WarpOperation.Gradient, "w2"), sB, None, None, [fld], returnReaction=True)[1].force.cpu().numpy()
    assert float(np.abs(ra - rb).max()) <= 3e-7 * max(float(np.abs(ra).max()), 1.0), (ra, rb)


@pytest.mark.parametrize("device", DEVICES)
def test_channel_pruning_and_exact_kernel_fallback(device):
    """(c) `channels=(3, 4)`: g0 equals the unpruned g0 (torch.equal on the box rows), every other channel is exactly 0, the rows without gradient are dropped; the pruned moments
    serve the Naive gradient of a constant and the guard of sceneOperation rejects anything else (same contract as the polygon path)."""
    register()
    sS, sB = scenes(device, "outside")
    pos, sup = queries(device)
    ps = state(pos, sup, device)
    pr = props(WarpOperation.Gradient, "w2")
    full, pruned = sB.pairMoments(ps, pr), sB.pairMoments(ps, pr, channels=(3, 4))
    f, p = full.entries[0]["implicit"][0], pruned.entries[0]["implicit"][0]
    assert (p.lam == 0).all() and (p.m1 == 0).all() and (p.g1 == 0).all()
    common = {int(q): i for i, q in enumerate(f.q.tolist())}
    idx = torch.tensor([common[int(q)] for q in p.q.tolist()], device=device)
    assert torch.equal(p.g0, f.g0[idx])
    one = [BodyField(torch.tensor(1.0, dtype=F64, device=device))]
    got = sceneOperation(ps, pr, sB, pruned, None, one).cpu().numpy()
    ref = sceneOperation(ps, pr, sB, full, None, one).cpu().numpy()
    assert np.abs(got - ref).max() <= 1e-12
    with pytest.raises(ValueError, match="built with the channels"):
        sceneOperation(ps, props(WarpOperation.Density, "w2"), sB, pruned, None, [BodyField(rho=1.0)])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solid", ["inside", "outside"])
def test_geometry_queries_and_wall_operations(device, solid):
    """(d) Scene.inside / signed_distance agree with the polygon (away from ties between two faces), and the three wall operations of the solver (exact cover vector with `cone`,
    wall Laplacian with `lw2`, tensile term with `w2p5`, each also through `restrict` of the box adjacency) agree with the polygon path."""
    register()
    sS, sB = scenes(device, solid)
    pos, sup = queries(device, n=400)
    pts = pos[12:]                                                           # drop the exact wall / corner points: the nearest-face normal is a tie there
    assert torch.equal(sS.inside(pts), sB.inside(pts))
    dS, nS, _ = sS.signed_distance(pts); dB, nB, _ = sB.signed_distance(pts)
    assert float((dS - dB).abs().max()) <= 1e-12
    fl = (dS > 0) & (dS < 0.5)                                               # fluid side near the wall (normals are defined by the nearest face; corners of a solid box differ only in the solid)
    assert float((nS[fl] - nB[fl]).abs().max()) <= 1e-9
    H = 0.5
    near = pts[torch.nonzero(dS.abs() < H).flatten()]
    a = cover_vector_scene(sS, near, H); b = cover_vector_scene(sB, near, H)
    assert float((a - b).abs().max()) <= 1e-12 * max(float(a.abs().max()), 1.0)                          # cone: exact fallback
    a = lap_lambda_scene(sS, near, H, "w2"); b = lap_lambda_scene(sB, near, H, "w2")
    assert float((a - b).abs().max()) <= 3e-5 * max(float(a.abs().max()), 1.0), float((a - b).abs().max())
    a = tensile_vector_scene(sS, near, H, "w2"); b = tensile_vector_scene(sB, near, H, "w2")
    assert float((a - b).abs().max()) <= 1e-7 * max(float(a.abs().max()), 1.0), float((a - b).abs().max())
    # restrict of the box adjacency, evaluated for the tensile term, equals the box path on the subset
    ps = state(pts, torch.full((len(pts),), H, dtype=F64, device=device), device)
    adj = sB.adjacency(ps, props(WarpOperation.Density, "w2"))
    idx = torch.nonzero(dS.abs() < H).flatten()
    t_sub = tensile_vector_scene(sB, pts[idx], H, "w2", adj.restrict(idx))
    t_ref = tensile_vector_scene(sB, pts[idx], H, "w2")
    assert float((t_sub - t_ref).abs().max()) <= 1e-12 * max(float(t_ref.abs().max()), 1.0)
