"""`Scene.kinematics(x)`: the rigid-body fields of all bodies in one batched evaluation (`BodyKinematics`).

(a) bit-for-bit equal to the per-body `Body.relative` / `velocityAt` / `accelerationAt` (torch.equal, not a tolerance: the arithmetic is the same element by element), with and without a periodic box, with python
    floats and with 0-d device tensors as angular rates (what a graph capture holds), several bodies with different poses and rates;
(b) the cache: the same positions and body states return the same object; moving the positions (new storage or an in-place write), replacing or writing in place any body tensor, or changing a python-float rate
    returns a fresh one with the new values;
(c) a scene whose bodies do not share one periodic box returns None (the per-body methods stay the fallback);
(d) the launch count of the batched evaluation does not grow with the number of bodies.
Float64 contracts.
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, BoxRep, Scene

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
DEV = "cuda:0"
F64 = torch.float64


def make(nb=4, periodic=None, tensors=False, seed=0):
    rng = np.random.default_rng(seed)
    bodies = []
    for i in range(nb):
        b = Body(bodyId=i, center=tuple(rng.uniform(0.1, 0.9, 2)), angle=float(rng.uniform(-1, 1)), linearVelocity=tuple(rng.normal(size=2)), angularVelocity=float(rng.normal()),
                 linearAcceleration=tuple(rng.normal(size=2)), angularAcceleration=float(rng.normal()), reps=[BoxRep((-0.05, -0.05), (0.05, 0.05))])
        bodies.append(b)
    sc = Scene(bodies, DEV)
    if periodic is not None:
        sc.setPeriodic(periodic, 0.1)
    if tensors:
        for b in bodies:
            b.angularVelocity = torch.tensor(b.angularVelocity, dtype=F64, device=DEV)
            b.angularAcceleration = torch.tensor(b.angularAcceleration, dtype=F64, device=DEV)
    x = torch.as_tensor(rng.uniform(-1.5, 2.5, (300, 2)), dtype=F64, device=DEV)
    return sc, x


@pytest.mark.parametrize("periodic", [None, Periodic((0.0, 0.0), (1.0, 1.0)), Periodic((0.0, 0.0), (1.0, 1.0), (True, False))])
@pytest.mark.parametrize("tensors", [False, True])
def test_batched_fields_are_the_per_body_fields_bit_for_bit(periodic, tensors):
    sc, x = make(periodic=periodic, tensors=tensors)
    kin = sc.kinematics(x)
    for i, b in enumerate(sc.bodies):
        assert torch.equal(kin.rel[i], b.relative(x))
        assert torch.equal(kin.velocity[i], b.velocityAt(x))
        assert torch.equal(kin.acceleration[i], b.accelerationAt(x))


def test_cache_hits_and_invalidations():
    sc, x = make(periodic=Periodic((0.0, 0.0), (1.0, 1.0)))
    k0 = sc.kinematics(x)
    assert sc.kinematics(x) is k0
    b = sc.bodies[1]
    # positions: an in-place write, another tensor
    x2 = x.clone()
    assert sc.kinematics(x2) is not k0
    k1 = sc.kinematics(x)
    x[0, 0] += 0.1
    k2 = sc.kinematics(x)
    assert k2 is not k1 and torch.equal(k2.velocity[1], b.velocityAt(x))
    # body tensors: in place, replaced; python-float rates
    b.linearVelocity.add_(0.5)
    k3 = sc.kinematics(x)
    assert k3 is not k2 and torch.equal(k3.velocity[1], b.velocityAt(x))
    b.center = b.center + 0.01
    k4 = sc.kinematics(x)
    assert k4 is not k3 and torch.equal(k4.rel[1], b.relative(x))
    b.angularVelocity = 3.0
    k5 = sc.kinematics(x)
    assert k5 is not k4 and torch.equal(k5.velocity[1], b.velocityAt(x))
    b.angularAcceleration = -2.0
    k6 = sc.kinematics(x)
    assert k6 is not k5 and torch.equal(k6.acceleration[1], b.accelerationAt(x))
    b.linearAcceleration[0] = 7.0
    assert torch.equal(sc.kinematics(x).acceleration[1], b.accelerationAt(x))


def test_bodies_with_different_boxes_have_no_batched_form():
    sc, x = make(periodic=Periodic((0.0, 0.0), (1.0, 1.0)))
    sc.bodies[0].periodic = Periodic((0.0, 0.0), (2.0, 2.0))
    assert sc.kinematics(x) is None
    assert Scene([], DEV).kinematics(x) is None


def test_launches_do_not_grow_with_the_number_of_bodies():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            sch = getattr(func, "_schema", None)
            if not (sch is not None and any(r.alias_info is not None for r in sch.returns)):
                Count.n += 1
            return func(*args, **(kwargs or {}))

    counts = []
    for nb in (2, 16):
        sc, x = make(nb=nb, periodic=Periodic((0.0, 0.0), (1.0, 1.0)))
        Count.n = 0
        with Count():
            k = sc.kinematics(x)
            k.velocity, k.acceleration
        counts.append(Count.n)
    assert counts[0] == counts[1], counts


@pytest.mark.parametrize("periodic", [None, Periodic((0.0, 0.0), (1.0, 1.0)), Periodic((0.0, 0.0), (1.0, 1.0), (True, False))])
@pytest.mark.parametrize("tensors", [False, True])
def test_batched_local_frames_are_toLocal_bit_for_bit(periodic, tensors):
    sc, x = make(periodic=periodic, tensors=tensors)
    if tensors:                                                             # a captured pose: (cos, sin) as 0-d device tensors
        for b in sc.bodies:
            a = float(b.angle)
            b._cs = (torch.tensor(np.cos(a), dtype=F64, device=DEV), torch.tensor(np.sin(a), dtype=F64, device=DEV))
    kin = sc.kinematics(x)
    for i, b in enumerate(sc.bodies):
        assert torch.equal(kin.local[i], b.toLocal(x))


def test_a_rotated_body_invalidates_the_local_frames():
    sc, x = make(periodic=Periodic((0.0, 0.0), (1.0, 1.0)))
    k0 = sc.kinematics(x)
    l0 = k0.local[2].clone()
    b = sc.bodies[2]
    b.angle = float(b.angle) + 1e-3
    k1 = sc.kinematics(x)
    assert k1 is not k0 and torch.equal(k1.local[2], b.toLocal(x)) and not torch.equal(k1.local[2], l0)
    b._cs = (torch.tensor(1.0, dtype=F64, device=DEV), torch.tensor(0.0, dtype=F64, device=DEV))             # a captured pose replaces the angle
    k2 = sc.kinematics(x)
    assert k2 is not k1 and torch.equal(k2.local[2], b.toLocal(x))
