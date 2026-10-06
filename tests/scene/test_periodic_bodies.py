"""Step 1b of docs/plan-next-steps.md: periodic wall integrals.  ONE real body; the particles are seen at their nearest image of the body centre (`Body.toLocal`, positions never written) and a bundle that
spans the box carries the tiled copies of its disks (`DiskArrayRep.tiled`).

(a) one fibre in a box (`R + H < L / 2`): lam, G, Cov, A of raw positions (a different integer number of box lengths out per particle) equal those of the non-periodic scene with the explicit 3 x 3 images of
    the disk at the particles wrapped into the box around the centre -- the reference construction of the test only;
(b) a bundle of fibres of different radii inside the box, several of them within a support of the seam: the same, with the explicit 5 x 5 images; the rows of the bundle are the single body's rows (one launch);
(c) a body that does not fit its cell is refused, a rotating bundle with images is refused, a periodic axis alone (channel) works with images along that axis;
(d) `signed_distance` / `inside` of the periodic scene equal the images'.
"""
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.warpfused import FusedGroup
from warpSPHBoundaries.scene.fixedadj import fixed_adjacency
from warpSPHBoundaries.scene.fused import FusedWall, WallOutput
from warpSPHBoundaries.scene.periodic import Periodic, min_image
from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, Scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
H = 0.12


def props():
    return OperationProperties(kernel="w2", operation=WarpOperation.Density, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)


def particles(device, pos, h=H):
    n = len(pos)
    return ParticleState(positions=pos, supports=torch.full((n,), h, dtype=F64, device=device), masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device),
                         densities=torch.ones(n, dtype=F64, device=device))


def evaluate(scene, pos, device):
    ps = particles(device, pos)
    adj = fixed_adjacency(scene, ps, props(), H)
    fw = FusedWall(scene, adj, (FusedGroup("w2"),))
    outs = (WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("A", 0, "a1g1"))
    a1 = torch.as_tensor(np.random.default_rng(3).normal(size=(len(scene.bodies), len(pos), 2)), dtype=F64, device=device)
    return fw.evaluate(outs, a1=a1)


def scale(a):
    return max(float(a.abs().max()), 1e-12)


def reference(centres, radii, c, per, device, nimg):
    """the non-periodic body with the explicit images k in [-nimg, nimg]^2 of every disk."""
    L = torch.tensor([per.hi[0] - per.lo[0], per.hi[1] - per.lo[1]], dtype=F64)
    cen, rad = [], []
    for i in range(-nimg, nimg + 1):
        for j in range(-nimg, nimg + 1):
            cen.append(torch.as_tensor(centres, dtype=F64) + torch.stack([i * L[0], j * L[1]]))
            rad.append(torch.as_tensor(radii, dtype=F64))
    return Scene([Body(bodyId=0, center=c, reps=[DiskArrayRep(torch.cat(cen), torch.cat(rad))])], device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["one", "bundle"])
def test_periodic_body_equals_the_explicit_images(device, case):
    per = Periodic((0.0, 0.0), (1.0, 1.0))
    c = (0.5, 0.5)
    if case == "one":
        centres, radii, nimg = [(0.05, -0.03)], [0.15], 1
    else:
        centres, radii, nimg = [(0.0, 0.0), (0.38, 0.1), (-0.4, 0.3), (0.1, -0.42), (-0.3, -0.35), (0.42, -0.4)], [0.12, 0.07, 0.09, 0.05, 0.1, 0.06], 2
    scene = Scene([Body(bodyId=0, center=c, reps=[DiskArrayRep(centres, radii)])], device).setPeriodic(per, H)
    rng = np.random.default_rng(2)
    n = 1500
    base = torch.as_tensor(rng.uniform(0.0, 1.0, (n, 2)), dtype=F64, device=device)
    raw = base + torch.as_tensor(rng.integers(-3, 4, (n, 2)), dtype=F64, device=device)        # raw positions any number of box lengths out
    raw0 = raw.clone()
    got = evaluate(scene, raw, device)
    assert torch.equal(raw, raw0)                                                                # not written
    cc = torch.tensor(c, dtype=F64, device=device)
    wrapped = cc + min_image(raw - cc, per)
    ref = evaluate(reference(centres, radii, c, per, device, nimg), wrapped, device)
    for k in got:
        assert float((got[k] - ref[k]).abs().max()) <= 1e-9 * scale(ref[k]), (k, float((got[k] - ref[k]).abs().max()), scale(ref[k]))
    assert scale(got["lam"]) > 0.3
    near = ((wrapped - cc).abs() > 0.5 - H).any(1)                                                # rows within a support of the seam see a fibre across it
    if case == "bundle":
        assert int(near.sum()) > 100 and float(got["lam"][0][near].abs().max()) > 0.1


@pytest.mark.parametrize("device", DEVICES)
def test_a_channel_periodic_in_one_axis(device):
    per = Periodic((0.0, -9.0), (1.0, 9.0), (True, False))
    centres, radii = [(0.0, 0.0), (0.4, 0.3)], [0.1, 0.07]
    scene = Scene([Body(bodyId=0, center=(0.5, 0.0), reps=[DiskArrayRep(centres, radii)])], device).setPeriodic(per, H)
    rng = np.random.default_rng(4)
    pos = torch.as_tensor(rng.uniform((0.0, -0.6), (1.0, 0.6), (800, 2)), dtype=F64, device=device) + torch.tensor([3.0, 0.0], dtype=F64, device=device)
    got = evaluate(scene, pos, device)
    cc = torch.tensor([0.5, 0.0], dtype=F64, device=device)
    ref = evaluate(reference(centres, radii, (0.5, 0.0), Periodic((0.0, -9.0), (1.0, 9.0)), device, 0) if False else _x_images(centres, radii, device), cc + min_image(pos - cc, per), device)
    for k in got:
        assert float((got[k] - ref[k]).abs().max()) <= 1e-9 * scale(ref[k]), k


def _x_images(centres, radii, device):
    cen = torch.cat([torch.as_tensor(centres, dtype=F64) + torch.tensor([i * 1.0, 0.0], dtype=F64) for i in (-2, -1, 0, 1, 2)])
    rad = torch.cat([torch.as_tensor(radii, dtype=F64)] * 5)
    return Scene([Body(bodyId=0, center=(0.5, 0.0), reps=[DiskArrayRep(cen, rad)])], device)


@pytest.mark.parametrize("device", DEVICES)
def test_misuse_is_refused(device):
    per = Periodic((0.0, 0.0), (1.0, 1.0))
    big = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep([(0.0, 0.0)], [0.45])])], device).setPeriodic(per, H)
    with pytest.raises(ValueError, match="fit its cell"):
        big.bodies[0].fusedReps(H, device)
    rot = Scene([Body(bodyId=0, center=(0.5, 0.5), angle=0.3, reps=[DiskArrayRep([(0.0, 0.0), (0.4, 0.4)], [0.1, 0.1])])], device).setPeriodic(per, H)
    with pytest.raises(NotImplementedError, match="rotate"):
        rot.bodies[0].fusedReps(H, device)
    with pytest.raises(ValueError, match="twice the support"):
        per.checkSupport(0.6)


@pytest.mark.parametrize("device", DEVICES)
def test_signed_distance_and_inside_see_the_images(device):
    per = Periodic((0.0, 0.0), (1.0, 1.0))
    centres, radii = [(0.0, 0.0), (0.4, 0.3)], [0.1, 0.07]
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[DiskArrayRep(centres, radii)])], device).setPeriodic(per, H)
    rng = np.random.default_rng(6)
    base = torch.as_tensor(rng.uniform(0.0, 1.0, (500, 2)), dtype=F64, device=device)
    raw = base + torch.as_tensor(rng.integers(-3, 4, (500, 2)), dtype=F64, device=device)
    d, nrm, hit = scene.signed_distance(raw)
    d0, n0, _ = scene.signed_distance(base)
    assert float((d - d0).abs().max()) < 1e-12 and float((nrm - n0).abs().max()) < 1e-12
    cc = torch.tensor([0.5, 0.5], dtype=F64, device=device)
    ref = reference(centres, radii, (0.5, 0.5), per, device, 2)
    dr, nr, _ = ref.signed_distance(cc + min_image(raw - cc, per))
    near = dr < H                                                                                 # the images are tiled for the reach of one support: the distance is exact within it
    assert int(near.sum()) > 50 and float((d - dr)[near].abs().max()) < 1e-12
    assert torch.equal(scene.inside(raw)[near], ref.inside(cc + min_image(raw - cc, per))[near])
