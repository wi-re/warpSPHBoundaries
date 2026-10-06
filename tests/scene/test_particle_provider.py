"""Cross-representation test: the PARTICLE provider (`ParticleBoundary`, wall particles on a lattice of spacing s, aggregates as pair sums) against the ANALYTIC provider (`AnalyticBoundary`, exact edge integrals)
on the same bodies.  The analytic integrals are the continuum limit of the wall-particle sums, so every output of the provider contract -- lam, G, Cov, cover, lap, tens, the hydrostatic term A and the cone areas --
must converge with the lattice spacing; the indicator of the solid is sampled once per lattice cell, so the rate is FIRST order in s / H (observed: the error halves with s, for all outputs and geometries).

Bodies: a rotated, translated floor slab (queries on both sides of its surface), a disk obstacle (`ImplicitRep`), the corner of a tank (`SurfaceRep` loop, the solid is the unbounded outside).  s = H/8, H/16, H/32.
Thresholds (stated from the observed behaviour, with margin): error at s = H/32 below 3 % of the scale for lam, G, Cov, A, cover and the full-disk area, below 10 % for the second-derivative / W^4 / wedge outputs
(lap, tens, cone area), and err(H/32) < 0.45 err(H/8) for every output.  Negative control: the particle aggregate follows the pose of the body, and compared with the analytic one of a displaced body it is far off.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions, ParticleState

from edgebound.scene import AnalyticBoundary, Body, BoundaryProvider, DiskBody, ImplicitRep, ParticleBoundary, Scene, SurfaceRep, WallOutput

DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
F64 = torch.float64
H = 0.3
KEYS = ("lam", "G", "Cov", "cover", "lap", "tens", "A", "cone0", "cone1")
LOOSE = ("lap", "tens", "cone0")


def ps_of(pos):
    n = len(pos)
    return ParticleState(positions=pos, supports=torch.full((n,), H, dtype=F64, device=DEV), masses=torch.ones(n, dtype=F64, device=DEV), kinds=torch.zeros(n, dtype=torch.int32, device=DEV),
                         densities=torch.ones(n, dtype=F64, device=DEV))


def outputs(provider, ps, a1, ax):
    ag = provider.aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)
    out = dict(ag.out)
    out["A"] = ag.evaluate((WallOutput("A", 0, "a1g1"),), a1=a1)["A"]
    c = ag.cone_area(ax, math.pi / 6)
    out["cone0"], out["cone1"] = c[0:1], c[1:2]
    return out


def err(got, ref):
    return {k: float((got[k] - ref[k]).abs().max()) / max(float(ref[k].abs().max()), 1e-12) for k in KEYS}


def make(name):
    rng = np.random.default_rng(0)
    if name == "floor":
        sc = Scene([Body(bodyId=0, center=(0.2, -0.1), angle=0.5, reps=[SurfaceRep.box((-3, -1), (3, 0), solid="inside")])], DEV)
        loc = torch.stack([torch.as_tensor(rng.uniform(-0.5, 0.5, 150), dtype=F64, device=DEV), torch.as_tensor(rng.uniform(-0.15, 0.35, 150), dtype=F64, device=DEV)], 1)
        pos = sc.bodies[0].pose.toWorld(loc)
    elif name == "disk":
        sc = Scene([Body(bodyId=0, center=(0.1, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.4))])], DEV)
        r, a = torch.as_tensor(rng.uniform(0.1, 0.7, 150), dtype=F64, device=DEV), torch.as_tensor(rng.uniform(0, 6.28, 150), dtype=F64, device=DEV)
        pos = torch.stack([0.1 + r * a.cos(), r * a.sin()], 1)
    else:
        sc = Scene([Body(bodyId=0, reps=[SurfaceRep.box((-1, -0.6), (1, 0.6), solid="outside")])], DEV)
        q = torch.as_tensor(rng.uniform(0, 1, (150, 2)), dtype=F64, device=DEV)
        pos = torch.stack([-1 + 0.4 * q[:, 0], -0.6 + 0.4 * q[:, 1]], 1)                 # within 0.4 of the corner
    a1 = torch.as_tensor(rng.normal(size=(1, 150, 2)), dtype=F64, device=DEV)
    ax = torch.as_tensor(rng.normal(size=(150, 2)), dtype=F64, device=DEV)
    return sc, ps_of(pos), a1, ax


@pytest.mark.parametrize("name", ["floor", "disk", "tank"])
def test_particle_sums_converge_to_the_analytic_integrals(name):
    sc, ps, a1, ax = make(name)
    ref = outputs(AnalyticBoundary(sc), ps, a1, ax)
    errs = [err(outputs(ParticleBoundary(sc, H / n), ps, a1, ax), ref) for n in (8, 16, 32)]
    for k in KEYS:
        assert errs[2][k] < (0.1 if k in LOOSE else 0.03), (name, k, [e[k] for e in errs])
        assert errs[2][k] < 0.45 * errs[0][k], (name, k, [e[k] for e in errs])
    assert max(errs[0].values()) > 0.03                                                    # the coarse lattice is visibly off: the test can fail


def test_particle_provider_follows_the_body_and_is_a_provider():
    sc, ps, a1, ax = make("floor")
    pb = ParticleBoundary(sc, H / 16)
    assert isinstance(pb, BoundaryProvider) and isinstance(AnalyticBoundary(sc), BoundaryProvider)
    before = outputs(pb, ps, a1, ax)
    ref0 = outputs(AnalyticBoundary(sc), ps, a1, ax)
    sc.bodies[0].center = sc.bodies[0].center + torch.tensor([0.0, 0.2 * H], dtype=F64, device=DEV)         # the floor moves up by 0.2 H
    after = outputs(pb, ps, a1, ax)
    ref1 = outputs(AnalyticBoundary(sc), ps, a1, ax)
    assert max(err(after, ref1).values()) < 0.15                                           # follows: the same convergence-level error at the new pose (no rebuild of the lattice)
    assert err(after, ref0)["lam"] > 0.15 and err(after, ref0)["lam"] > 4 * err(after, ref1)["lam"] and float((after["lam"] - before["lam"]).abs().max()) > 0.05      # negative control: against the old pose it is far off
