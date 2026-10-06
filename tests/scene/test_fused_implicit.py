"""`ImplicitRep` (disks) and `SdfRep` bodies on the fused wall path (`Body.fusedReps`, scene/fused.py): the fixed-capacity adjacency builds the exact tier-2 polygon of the body (a disk: area-preserving polygon
with edges <= h/16; an SDF: the marching-squares contour of the sampled distance, or its `fallback` surface) and the fused kernels integrate that.

(a) the lowered disk equals the high-resolution polygon `SurfaceRep` on the same queries: lam, G, Cov, the hydrostatic A: a solid disk is the disk element (`DiskArrayRep`, table lookup, any radius) to 5e-6, a cavity the polygon itself to 1e-12, with a rotated / translated body;
(b) against the tier-3 / tier-4 models of the same disk (`sceneOperation`, curvature expansion R/h >= 2, small-obstacle series R/h <= 0.2) lam and G agree to the tier accuracy (3e-4 of the scale; 5e-4 for the series, whose polygon has the minimum of 24 edges);
(c) the contour of a sampled distance: the loops are oriented (solid on the left) and closed -- the winding number of random points equals `d < 0` except within one grid spacing of the surface -- for an obstacle
    (all border nodes fluid, `background = 0`), a tank (all border solid, `background = 1`), two merging disks and a ring (a hole inside the solid) including the saddle cells; a grid with a mixed border is refused;
(d) an `SdfRep` of a disk through the fused path agrees with the analytic disk polygon (the contour error O(spacing^2 kappa), 2e-3 of the scale at spacing h/16), and the build is sync-free;
(e) a body without a polygon form (half plane) is not fusable and `FusedWall` refuses it with the oracle adjacency.
"""
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

from warpSPHBoundaries.edge.warpfused import FusedGroup
from warpSPHBoundaries.scene.fixedadj import fixed_adjacency
from warpSPHBoundaries.scene.fused import FusedWall, WallOutput
from warpSPHBoundaries.scene.implicitBodies import DiskBody, HalfPlaneBody
from warpSPHBoundaries.scene.scene import Body, ImplicitRep, Scene, SdfRep, SurfaceRep, sceneOperation, BodyField

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
H = 0.3


def props(op=WarpOperation.Density, kernel="w2"):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)


def particles(device, pos, h=H):
    n = len(pos)
    return ParticleState(positions=pos, supports=torch.full((n,), h, dtype=F64, device=device), masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                         densities=torch.ones(n, dtype=F64, device=device))


def queries(device, n=400, seed=3, lo=-1.3, hi=1.3):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.uniform(lo, hi, (n, 2)), dtype=F64, device=device)


def fused_outputs(scene, ps, a1=None):
    adj = fixed_adjacency(scene, ps, props(), H)
    fw = FusedWall(scene, adj, (FusedGroup("w2"),))
    outs = (WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("A", 0, "a1g1"))
    return fw.evaluate(outs, a1=torch.zeros((len(scene.bodies), len(ps.positions), 2), dtype=F64, device=scene.device) if a1 is None else a1)


def scale(*a):
    return max(float(x.abs().max()) for x in a)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solid", ["inside", "outside"])
def test_lowered_disk_equals_the_polygon_surface(device, solid):
    c, R = (0.2, -0.1), 0.8
    rep = ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R, solid=solid))
    ref = SurfaceRep.regularPolygon((0.0, 0.0), R, max(24, int(np.ceil(2 * np.pi * R / (H / 16)))), solid=solid)
    pos = queries(device)
    ps = particles(device, pos)
    a1 = torch.as_tensor(np.random.default_rng(1).normal(size=(1, len(pos), 2)), dtype=F64, device=device)
    got = fused_outputs(Scene([Body(bodyId=0, center=c, angle=0.6, reps=[rep])], device), ps, a1)
    exp = fused_outputs(Scene([Body(bodyId=0, center=c, angle=0.6, reps=[ref])], device), ps, a1)
    tol = 5e-6 if solid == "inside" else 1e-12                                              # a solid disk is the disk element (tables, ~1e-7, against a polygon with its own 1e-6); a cavity is the polygon itself
    for k in got:
        assert float((got[k] - exp[k]).abs().max()) <= tol * scale(exp[k]), k
    assert scale(got["lam"]) > 0.1 and scale(got["Cov"]) > 0.1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("R,tier", [(0.8, 3), (0.04, 4)])
def test_lowered_disk_agrees_with_the_tier_models(device, R, tier):
    rep = ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R))
    sc = Scene([Body(bodyId=0, center=(0.1, 0.05), reps=[rep])], device)
    pos = queries(device, n=600, seed=5, lo=-0.7, hi=0.8)
    ps = particles(device, pos)
    got = fused_outputs(sc, ps)
    adj = sc.adjacency(ps, props())
    pm = sc.precompute(adj, props())
    lam = sceneOperation(ps, props(), sc, pm, None, [BodyField(rho=1.0)], perBody=True)[0]
    G = sceneOperation(ps, props(WarpOperation.Gradient), sc, pm, None, [BodyField(torch.tensor(1.0, dtype=F64, device=device))], perBody=True)[0]
    tol = 3e-4 if tier == 3 else 5e-4                                                      # tier 4: the polygon of a disk of R/h = 0.13 has the minimum of 24 edges (the same polygon as the oracle's tier-2 fallback)
    assert float((got["lam"][0] - lam).abs().max()) <= tol * max(scale(lam), 1e-3)
    assert float((got["G"][0] - G).abs().max()) <= tol * max(scale(G), 1e-3)


def disks_sdf(device, centres, radii, op="min", spacing=0.02, pad=0.5):
    cs = np.asarray(centres, float)
    lo, hi = cs.min(0) - max(radii) - pad, cs.max(0) + max(radii) + pad

    def fn(p):
        p = p.numpy() if isinstance(p, torch.Tensor) else p
        ds = [np.linalg.norm(p - c, axis=1) - r for c, r in zip(cs, radii)]
        d = ds[0]
        for x in ds[1:]:
            d = np.minimum(d, x) if op == "min" else np.maximum(d, -x)
        return torch.as_tensor(d, dtype=F64)
    return SdfRep.fromFunction(fn, lo, hi, spacing).to(device), fn


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["disk", "merged", "ring", "tank"])
def test_contour_is_oriented_and_closed(device, name):
    sp = 0.02
    if name == "disk":
        rep, fn = disks_sdf(device, [(0, 0)], [0.6], spacing=sp)
    elif name == "merged":                                                                  # two overlapping disks: a concave union with two reflex corners (saddle cells appear on the grid)
        rep, fn = disks_sdf(device, [(0, 0), (0.7, 0.2)], [0.5, 0.4], spacing=sp)
    elif name == "ring":                                                                    # solid disk with a fluid hole: loops of both orientations
        cs, rs = [(0, 0), (0.1, 0.0)], [0.8, 0.3]
        rep, fn = disks_sdf(device, cs, rs, op="ring", spacing=sp)
        fn0 = fn
        fn = lambda p: torch.maximum(torch.as_tensor(np.linalg.norm(np.asarray(p) - np.array(cs[0]), axis=1) - rs[0], dtype=F64), -torch.as_tensor(np.linalg.norm(np.asarray(p) - np.array(cs[1]), axis=1) - rs[1], dtype=F64))
        lo, hi = np.array([-1.3, -1.3]), np.array([1.3, 1.3])
        rep = SdfRep.fromFunction(lambda p: fn(p.numpy()), lo, hi, sp).to(device)
    else:                                                                                   # a tank: the fluid is the box, the solid the unbounded outside
        lo, hi = np.array([-1.0, -0.6]), np.array([1.0, 0.6])
        fn = lambda p: -torch.as_tensor(np.maximum(np.abs(np.asarray(p)[:, 0]) - 0.8, np.abs(np.asarray(p)[:, 1]) - 0.4), dtype=F64)
        rep = SdfRep.fromFunction(lambda p: fn(p.numpy()), lo, hi, sp).to(device)
    surf = rep.contour()
    assert surf.background == (1 if name == "tank" else 0)
    pts = queries(device, n=3000, seed=9, lo=-1.1 if name != "tank" else -0.95, hi=1.1 if name != "tank" else 0.95)
    d = torch.as_tensor(fn(pts.cpu().numpy()), dtype=F64, device=device)
    ins = (surf.indicator(pts) > 0.5)
    far = d.abs() > 1.5 * sp
    assert int(far.sum()) > 1000 and bool((ins[far] == (d[far] < 0)).all()), (name, int((ins[far] != (d[far] < 0)).sum()))


def test_contour_refuses_a_grid_that_cuts_the_body():
    d = torch.as_tensor(np.linspace(-1, 1, 21)[:, None] * np.ones((1, 21)), dtype=F64)       # solid on one side of the grid, fluid on the other: the zero set leaves the grid
    with pytest.raises(ValueError, match="margin"):
        SdfRep(d, (0.0, 0.0), 0.1).contour()


@pytest.mark.parametrize("device", DEVICES)
def test_sdf_disk_through_the_fused_path(device):
    R, sp = 0.8, H / 16
    rep, _ = disks_sdf(device, [(0, 0)], [R], spacing=sp)
    ref = SurfaceRep.regularPolygon((0.0, 0.0), R, 400)
    pos = queries(device, seed=7)
    ps = particles(device, pos)
    a1 = torch.as_tensor(np.random.default_rng(2).normal(size=(1, len(pos), 2)), dtype=F64, device=device)
    got = fused_outputs(Scene([Body(bodyId=0, center=(0.1, 0.2), angle=0.4, reps=[rep])], device), ps, a1)
    exp = fused_outputs(Scene([Body(bodyId=0, center=(0.1, 0.2), angle=0.4, reps=[ref])], device), ps, a1)
    for k in ("lam", "G", "Cov", "A"):
        assert float((got[k] - exp[k]).abs().max()) <= 2e-3 * scale(exp[k]), k
    if device != "cpu":                                                                    # the adjacency and the evaluation of a lowered body do not synchronise
        sc = Scene([Body(bodyId=0, center=(0.1, 0.2), angle=0.4, reps=[rep])], device)
        fused_outputs(sc, ps, a1)                                                          # warm: kernels, tables, the contour
        torch.cuda.set_sync_debug_mode("error")
        try:
            fused_outputs(sc, ps, a1)
        finally:
            torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("device", DEVICES)
def test_half_plane_has_no_polygon_form(device):
    sc = Scene([Body(bodyId=0, reps=[ImplicitRep(HalfPlaneBody(point=(0.0, 0.0), normal=(0.0, 1.0)))])], device)
    assert not FusedWall.supported(sc)
    assert FusedWall.supported(Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0, 0), radius=0.2))])], device))
    assert not FusedWall.supported(Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0, 0), radius=0.2))])], device), lowering=False)
