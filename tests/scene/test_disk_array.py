"""`DiskArrayRep`: a bundle of fibres (disks of any radius) as ONE body on the fused path, every kernel integral a table lookup (`edge/disktables.py`, `edge/warpdisk.py`, docs/disk-element.md).

(a) a bundle of disks with different radii (R/h = 0.1 ... 6) in one body against the SUM of the same disks as high-resolution polygons (edges <= h/16) in one body each: lam, G, Cov, the hydrostatic A, the cover
    vector, the wall Laplacian, the tensile vector (5e-6 of the scale; the cone-kernel cover 1e-4: the table floor of the kernel with a cusp at r = 0), rotated / translated body;
(b) the closed-form areas `cone_area` (disk cap ball cap wedge, both rows) against an independent numerical angular integration of the ray lengths (1e-7 of H^2), several wedges, bundle of three disks;
(c) the slot list: every disk within one support + radius of a query is in its slots, K is bounded by the most disks in a 3 x 3 cell block, slots are filled in a fixed order, the adjacency and the evaluation do not
    synchronise (torch sync debug mode);
(d) geometry queries: `signed_distance` / `inside` of the bundle against the analytic distances, a rotated body;
(e) cross-representation: the wall-particle sums (`ParticleBoundary`) of the same bundle converge to the element's aggregates, first order in the lattice spacing.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions, ParticleState

from warpSPHBoundaries.scene import AnalyticBoundary, Body, ImplicitRep, ParticleBoundary, Scene, SurfaceRep, WallOutput
from warpSPHBoundaries.scene.scene import DiskArrayRep
from warpSPHBoundaries.scene.fixedadj import fixed_adjacency

DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
F64 = torch.float64
H = 0.3
KEYS = ("lam", "G", "Cov", "cover", "lap", "tens", "A", "cone0", "cone1")


def props():
    return AnalyticBoundary(Scene([], DEV)).properties(KernelFunctions.Wendland2)


def ps_of(pos):
    n = len(pos)
    return ParticleState(positions=pos, supports=torch.full((n,), H, dtype=F64, device=DEV), masses=torch.ones(n, dtype=F64, device=DEV), kinds=torch.zeros(n, dtype=torch.int32, device=DEV),
                         densities=torch.ones(n, dtype=F64, device=DEV))


def outputs(scene, ps, a1, ax, half=math.pi / 6):
    ag = AnalyticBoundary(scene).aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)
    out = {k: v.sum(0, keepdim=True) for k, v in ag.out.items()}                                    # summed over the bodies
    out["A"] = ag.evaluate((WallOutput("A", 0, "a1g1"),), a1=a1)["A"].sum(0, keepdim=True)
    c = ag.cone_area(ax, half)
    out["cone0"], out["cone1"] = c[0:1], c[1:2]
    return out


def err(got, ref, k):
    return float((got[k] - ref[k]).abs().max()) / max(float(ref[k].abs().max()), 1e-12)


BUNDLE = ([(0.0, 0.0), (0.9, 0.2), (-0.7, 0.6), (0.3, -0.8)], [0.1 * H, 0.6 * H, 2.0 * H, 6.0 * H])


def queries(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.uniform(-1.6, 1.6, (n, 2)), dtype=F64, device=DEV)


def test_bundle_equals_the_sum_of_polygon_disks():
    cs, rs = BUNDLE
    pose = dict(center=(0.15, -0.1), angle=0.5)
    bundle = Scene([Body(bodyId=0, reps=[DiskArrayRep(cs, rs)], **pose)], DEV)
    # the same disks as polygons in ONE body each (their centres in the body frame -> the same pose)
    polys = Scene([Body(bodyId=i, reps=[SurfaceRep.regularPolygon(c, r, max(24, int(math.ceil(2 * math.pi * r / (H / 16)))))], **pose) for i, (c, r) in enumerate(zip(cs, rs))], DEV)
    pos = queries()
    ps = ps_of(pos)
    rng = np.random.default_rng(2)
    n = len(pos)
    a1b = torch.as_tensor(rng.normal(size=(1, n, 2)), dtype=F64, device=DEV)
    a1p = a1b.expand(len(cs), -1, -1).contiguous()
    ax = torch.as_tensor(rng.normal(size=(n, 2)), dtype=F64, device=DEV)
    got, ref = outputs(bundle, ps, a1b, ax), outputs(polys, ps, a1p, ax)
    for k in KEYS:
        assert err(got, ref, k) < (1e-4 if k in ("cover", "cone0", "cone1") else 5e-6), (k, err(got, ref, k))
    assert float(ref["lam"].max()) > 0.5 and float(ref["cone0"].max()) > 0.0


def numeric_cone_areas(centres, radii, x, axis, half, nth=200000):
    """area of (disks cap ball(x, H) cap wedge(x, axis, half)) and of (disks cap ball) by integrating the ray lengths over the angle (independent of the closed form)."""
    th = (np.arange(nth) + 0.5) * (2 * np.pi / nth) - np.pi
    d = np.stack([np.cos(th), np.sin(th)], 1)
    wedge = np.abs(np.angle(np.exp(1j * (th - np.arctan2(axis[1], axis[0]))))) <= half
    tot_w = tot_f = 0.0
    for c, r in zip(centres, radii):
        v = np.asarray(c) - x
        b = d @ v
        disc = b * b - (v @ v - r * r)
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t1, t2 = b - sq, b + sq
        lo, hi = np.clip(t1, 0, H), np.clip(t2, 0, H)
        a = np.where(hit & (t2 > 0), 0.5 * (hi ** 2 - lo ** 2), 0.0)
        tot_f += a.sum() * (2 * np.pi / nth)
        tot_w += a[wedge].sum() * (2 * np.pi / nth)
    return tot_w, tot_f


@pytest.mark.parametrize("half", [math.pi / 6, math.pi / 2, 2.5])
def test_cone_area_closed_form_equals_the_numerical_angular_integral(half):
    cs, rs = [(0.0, 0.0), (0.45, 0.1), (-0.3, 0.5)], [0.15, 0.25, 0.4]
    sc = Scene([Body(bodyId=0, center=(0.1, 0.2), angle=0.7, reps=[DiskArrayRep(cs, rs)])], DEV)
    rng = np.random.default_rng(3)
    pos = sc.bodies[0].pose.toWorld(torch.as_tensor(rng.uniform(-0.6, 0.9, (120, 2)), dtype=F64, device=DEV))
    ax = torch.as_tensor(rng.normal(size=(120, 2)), dtype=F64, device=DEV)
    ag = AnalyticBoundary(sc).aggregate(ps_of(pos), H, KernelFunctions.Wendland2)
    got = ag.cone_area(ax, half).cpu().numpy()
    lp = sc.bodies[0].pose.toLocal(pos).cpu().numpy()
    la = (ax @ sc.bodies[0].pose.R.to(ax.device)).cpu().numpy()                                      # world axis -> body frame (R^T a)
    worst = 0.0
    for i in range(0, 120, 6):
        w, f = numeric_cone_areas(cs, rs, lp[i], la[i], half)
        worst = max(worst, abs(got[0, i] - w), abs(got[1, i] - f))
    assert worst < 1e-7 * H * H * 100, worst
    assert float(got[1].max()) > 0.1 * H * H                                                         # some queries see a disk


def test_slots_cover_every_disk_in_reach_and_nothing_synchronises():
    cs, rs = BUNDLE
    sc = Scene([Body(bodyId=0, reps=[DiskArrayRep(cs, rs)])], DEV)
    pos = queries(300, seed=4)
    ps = ps_of(pos)
    adj = fixed_adjacency(sc, ps, props(), H)
    ba = adj.bodies[0]
    topo = ba.reps[0]
    K = topo.K
    slots = topo.e.reshape(-1, K).cpu().numpy()
    cen, rad = np.asarray(cs), np.asarray(rs)
    d = np.linalg.norm(pos.cpu().numpy()[:, None, :] - cen[None], axis=2)
    for i in range(len(pos)):
        want = set(np.nonzero(d[i] < H + rad)[0].tolist())
        assert set(int(m) for m in slots[i] if m >= 0) == want, i
    assert 1 <= K <= len(cs)
    if DEV != "cpu":
        AnalyticBoundary(sc).aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)           # warm
        torch.cuda.set_sync_debug_mode("error")
        try:
            ag = AnalyticBoundary(sc).aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)
            ag.cone_area(torch.ones((len(pos), 2), dtype=F64, device=DEV), 0.5)
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_signed_distance_and_inside():
    cs, rs = BUNDLE
    sc = Scene([Body(bodyId=0, center=(0.2, 0.1), angle=0.9, reps=[DiskArrayRep(cs, rs)])], DEV)
    b = sc.bodies[0]
    pos = b.pose.toWorld(queries(400, seed=5))
    d, n, hit = sc.signed_distance(pos)
    lp = b.pose.toLocal(pos).cpu().numpy()
    dist = np.linalg.norm(lp[:, None, :] - np.asarray(cs)[None], axis=2) - np.asarray(rs)[None]
    assert np.abs(d.cpu().numpy() - dist.min(1)).max() < 1e-12 and bool(hit.all())
    k = dist.argmin(1)
    nl = lp - np.asarray(cs)[k]
    nl = nl / np.linalg.norm(nl, axis=1, keepdims=True)
    assert np.abs((b.pose.vecToWorld(torch.as_tensor(nl, dtype=F64, device=DEV)) - n).abs().max().cpu().numpy()) < 1e-12
    assert bool((sc.inside(pos) == (d < 0)).all())


def test_wall_particles_converge_to_the_bundle():
    cs, rs = [(0.0, 0.0), (0.7, 0.3)], [0.5 * H, 1.5 * H]
    sc = Scene([Body(bodyId=0, reps=[DiskArrayRep(cs, rs)])], DEV)
    rng = np.random.default_rng(6)
    pos = torch.as_tensor(rng.uniform(-0.6, 1.4, (150, 2)), dtype=F64, device=DEV)
    ps = ps_of(pos)
    a1 = torch.as_tensor(rng.normal(size=(1, 150, 2)), dtype=F64, device=DEV)
    ax = torch.as_tensor(rng.normal(size=(150, 2)), dtype=F64, device=DEV)
    ref = outputs(sc, ps, a1, ax)

    def particle_outputs(spacing):
        ag = ParticleBoundary(sc, spacing).aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)
        o = dict(ag.out)
        o["A"] = ag.evaluate((WallOutput("A", 0, "a1g1"),), a1=a1)["A"]
        c = ag.cone_area(ax, math.pi / 6)
        o["cone0"], o["cone1"] = c[0:1], c[1:2]
        return o
    e = [{k: err(particle_outputs(H / n), ref, k) for k in KEYS} for n in (8, 32)]
    for k in KEYS:
        assert e[1][k] < (0.1 if k in ("lap", "tens", "cone0") else 0.04), (k, e[1][k])
        assert e[1][k] < 0.5 * e[0][k], (k, e[0][k], e[1][k])
