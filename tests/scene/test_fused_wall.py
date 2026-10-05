"""Fused wall evaluation (`scene/fused.py`, docs/plan-wall-evaluation.md step 3) against the `sceneOperation` path on the same adjacency.

Scene: a rotated, translated L-shape (reflex corner) and a second, thin rotated box body (two bodies: per-body outputs, per-body rotation), 500 random queries (inside the bodies, within a
support of the edges and the corner, far away) with per-query supports 0.35 - 0.7, a per-body per-query a1 field.  Outputs: lam, G (Gradient of a constant), Cov, A (Gradient with a per-query a1)
for `w2`; the wall Laplacian of `lw2` (x lap_factor), the cover vector (`cone`, channels 3, 4) and the tensile vector (`w2p5`, 3, 4) on a constant support (their library functions take one).
Tolerance stated before looking: 1e-12 of the largest value (float64 contract; the two paths differ in summation order and the FMA contraction only); two evaluations are torch.equal
(one thread per row, fixed order, no atomics).
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation

from edgebound.edge.warpfused import FusedGroup
from edgebound.scene import tensile
from edgebound.scene.cover import cover_vector_scene
from edgebound.scene.fused import FusedWall, WallOutput
from edgebound.scene.scene import Body, BodyField, BoxRep, Scene, SurfaceRep, sceneOperation
from edgebound.scene.tensile import tensile_factor, tensile_vector_scene
from edgebound.scene.viscosity import lap_factor, lap_lambda_scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64
LS = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)


def props(op, kernel):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)


def make(device, n=500, seed=2, constant_support=None):
    lap_factor(1.0, "w2"); tensile._register("w2")
    sc = Scene([Body(bodyId=0, center=(0.3, -0.2), angle=0.7, reps=[SurfaceRep.polygon(LS)]),
                Body(bodyId=1, center=(3.0, 1.0), angle=-0.4, reps=[SurfaceRep.box((-0.6, -0.05), (0.6, 0.05))])], device)
    rng = np.random.default_rng(seed)
    pos = torch.as_tensor(rng.uniform(-1.2, 4.2, (n, 2)), dtype=F64, device=device)
    sup = torch.full((n,), float(constant_support), dtype=F64, device=device) if constant_support else torch.as_tensor(rng.uniform(0.35, 0.7, n), dtype=F64, device=device)
    ps = ParticleState(positions=pos, supports=sup, masses=torch.ones(n, dtype=F64, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device), densities=torch.ones(n, dtype=F64, device=device))
    a1 = torch.as_tensor(rng.normal(size=(2, n, 2)), dtype=F64, device=device)
    return sc, ps, pos, a1


def close(a, b, tol=1e-12):
    sc = max(float(b.abs().max()), 1e-300)
    d = float((a - b).abs().max())
    assert d <= tol * sc, (d, sc)
    return d, sc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constant", [None, 0.5])
def test_fused_outputs_equal_scene_operations(device, constant):
    sc, ps, pos, a1 = make(device, constant_support=constant)
    B, N = len(sc.bodies), len(pos)
    adj = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    groups = (FusedGroup("w2"), FusedGroup("lw2"), FusedGroup("cone", (3, 4)), FusedGroup("w2p5", (3, 4)))
    fw = FusedWall(sc, adj, groups)
    outs = (WallOutput("lam", 0, "lam"), WallOutput("G", 0, "g0"), WallOutput("Cov", 0, "cov"), WallOutput("A", 0, "a1g1"),
            WallOutput("lap", 1, "lap"), WallOutput("cover", 2, "g0"), WallOutput("tens", 3, "g0"))
    got = fw.evaluate(outs, a1=a1)
    pm = sc.precompute(adj, props(WarpOperation.Density, "w2"))
    one = BodyField(torch.tensor(1.0, dtype=F64, device=device))
    ref_lam = sceneOperation(ps, props(WarpOperation.Density, "w2"), sc, pm, None, [BodyField(rho=1.0)] * B, perBody=True)
    ref_G = sceneOperation(ps, props(WarpOperation.Gradient, "w2"), sc, pm, None, [one] * B, perBody=True)
    ref_C = sceneOperation(ps, props(WarpOperation.Covariance, "w2"), sc, pm, None, [one] * B, perBody=True).reshape(B, N, 2, 2)
    flds = [BodyField(torch.zeros(N, dtype=F64, device=device), a1[b], rho=1.0, perQuery=True) for b in range(B)]
    ref_A = sceneOperation(ps, props(WarpOperation.Gradient, "w2"), sc, pm, None, flds, perBody=True)
    for name, ref in (("lam", ref_lam), ("G", ref_G), ("Cov", ref_C), ("A", ref_A)):
        d, s = close(got[name], ref)
        assert s > 0.5, (name, s)
        print(f"constant={constant} {name}: max|fused - sceneOperation| {d:.1e} (scale {s:.2f})")
    if constant:                                                                    # the library functions take one support
        H = constant
        ref_lap = lap_lambda_scene(sc, pos, H, "w2")
        close(lap_factor(H, "w2") * got["lap"], ref_lap)
        ref_cov = cover_vector_scene(sc, pos, H)
        close(-(math.pi * H ** 3 / 3) * got["cover"].sum(0), ref_cov)
        ref_t = tensile_vector_scene(sc, pos, H, "w2")
        close(tensile_factor(H, "w2") * got["tens"].sum(0), ref_t)
    again = fw.evaluate(outs, a1=a1)
    for o in outs:
        assert torch.equal(got[o.name], again[o.name]), o.name


@pytest.mark.parametrize("device", DEVICES)
def test_fused_rejects_other_representations(device):
    sc = Scene([Body(bodyId=0, reps=[BoxRep((0.0, 0.0), (1.0, 1.0))])], device)
    assert not FusedWall.supported(sc)
    pos = torch.tensor([[0.5, 0.5]], dtype=F64, device=device)
    ps = ParticleState(positions=pos, supports=torch.full((1,), 0.3, dtype=F64, device=device), masses=torch.ones(1, dtype=F64, device=device), kinds=torch.zeros(1, dtype=torch.int32, device=device),
                       densities=torch.ones(1, dtype=F64, device=device))
    with pytest.raises(NotImplementedError, match="surface representations only"):
        FusedWall(sc, sc.adjacency(ps, props(WarpOperation.Density, "w2")), (FusedGroup("w2"),))


@pytest.mark.parametrize("device", DEVICES)
def test_fused_evaluate_order_independence(device):
    """(c) one FusedWall, several `evaluate` calls in different order and with different output sets (a gradient-only call first, then the indicator-carrying outputs, then a1 changed):
    each result equals the one of a fresh FusedWall (the cached indicator / output specification must not leak between calls)."""
    sc, ps, pos, a1 = make(device)
    adj = sc.adjacency(ps, props(WarpOperation.Density, "w2"))
    groups = (FusedGroup("w2"), FusedGroup("cone", (3, 4)))
    fw = FusedWall(sc, adj, groups)
    g_only = (WallOutput("cover", 1, "g0"),)
    full = (WallOutput("lam", 0, "lam"), WallOutput("Cov", 0, "cov"), WallOutput("A", 0, "a1g1"))
    first = fw.evaluate(g_only)                                          # no indicator needed: must not freeze a zero indicator
    second = fw.evaluate(full, a1=a1)
    third = fw.evaluate(full, a1=2.0 * a1)
    fresh = FusedWall(sc, adj, groups)
    for name in ("lam", "Cov", "A"):
        assert torch.equal(second[name], fresh.evaluate(full, a1=a1)[name]), name
    assert float(second["lam"].abs().max()) > 0.5 and torch.equal(first["cover"], fresh.evaluate(g_only)["cover"])
    assert torch.equal(third["A"], 2.0 * second["A"]) or float((third["A"] - 2.0 * second["A"]).abs().max()) <= 1e-12 * float(second["A"].abs().max())
    assert torch.equal(third["lam"], second["lam"])
