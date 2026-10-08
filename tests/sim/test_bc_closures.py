"""`wallVelocity` (sim/bcclosures.py): the BC policies as wall-velocity closures.

(a) the definitions: noSlip reverses the tangential relative velocity and drops the normal one (u_g . n = u_body . n), freeSlip keeps the tangential and reflects the normal part, zeros / constant pin the wall;
    properties on random data (idempotence of the pinned forms, a wall at rest, the normal sign does not matter);
(b) against warpSPH's own `_slipVelocity` (the ghost velocity of its mDBC conditions), when warpSPH is importable: equal to 1e-6 (its 1e-7 regularisations);
(c) unknown policies are refused.
"""
import numpy as np
import pytest
import torch

from warpSPHBoundaries.sim.bcclosures import POLICIES, pinned, wallVelocity

F64 = torch.float64


def data(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    v, ub = torch.randn(n, 2, generator=g, dtype=F64), torch.randn(n, 2, generator=g, dtype=F64)
    a = torch.rand(n, generator=g, dtype=F64) * 6.28
    return v, torch.stack([torch.cos(a), torch.sin(a)], 1), ub


def test_definitions_on_random_data():
    v, n, ub = data()
    w = v - ub
    wn = (w * n).sum(1, keepdim=True) * n
    wt = w - wn
    ns, fs = wallVelocity("noSlip", v, n, ub), wallVelocity("freeSlip", v, n, ub)
    assert torch.allclose(((ns - ub) * n).sum(1), torch.zeros(len(v), dtype=F64), atol=1e-14)             # normal component = u_body . n
    assert torch.allclose(ns - ub, -wt, atol=1e-14)
    assert torch.allclose(fs - ub, wt - wn, atol=1e-14)
    assert torch.allclose(((fs - ub) * n).sum(1), -(w * n).sum(1), atol=1e-14)                              # reflected normal part
    assert torch.equal(wallVelocity("zeros", v, n, ub), torch.zeros_like(v))
    val = torch.tensor([0.3, -0.2], dtype=F64)
    assert torch.equal(wallVelocity("constant", v, n, ub, val), val.expand_as(v))
    assert torch.equal(wallVelocity("constant", v, n, ub), ub)                                              # unchanged: the body's own velocity
    assert torch.allclose(wallVelocity("noSlip", v, -n, ub), ns, atol=1e-14) and torch.allclose(wallVelocity("freeSlip", v, -n, ub), fs, atol=1e-14)


def test_wall_at_rest_and_fluid_at_the_wall_velocity():
    v, n, _ = data()
    z = torch.zeros_like(v)
    assert torch.allclose(wallVelocity("noSlip", v, n, z), -(v - (v * n).sum(1, keepdim=True) * n), atol=1e-14)      # -tangential(v)
    ub = torch.tensor([0.4, 0.1], dtype=F64).expand_as(v)
    for p in ("noSlip", "freeSlip"):                                                                          # a fluid moving with the wall sees the wall's velocity
        assert torch.allclose(wallVelocity(p, ub, n, ub), ub, atol=1e-14)


def test_matches_warpsph_ghost_velocity():
    mod = pytest.importorskip("warpSPH.modules.mdbc.velocity")
    slip = mod._slipVelocity
    v, n, ub = data()
    r = 0.37
    for reflect, policy in ((False, "noSlip"), (True, "freeSlip")):
        ref = slip(v * 1.0, torch.ones(len(v), dtype=F64), ub, n * r, reflect)               # qVelSum = v with Shepard weight 1; ghost offset = r n
        assert torch.allclose(ref, wallVelocity(policy, v, n, ub), atol=1e-6)


def test_unknown_policy_is_refused_and_pinned_flags():
    v, n, ub = data(4)
    with pytest.raises(ValueError):
        wallVelocity("extended", v, n, ub)
    assert [pinned(p) for p in POLICIES] == [False, False, True, True]
