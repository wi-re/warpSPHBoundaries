"""Tests for the cfg.coverExact switch (Q3a in the free-surface detector).

The wall part of the cover vector is computed two ways:
  - polar quadrature (the default, Q3b):  Cw = -n_w * sum_samples u * dA
  - exact edge reduction (cover.cover_vector_scene, kernel `cone`):
    Cw = n_w * cover_vector_scene(scene, x[near], H)
The polar sum carries a minus because u = unit(x_i -> p) while the cover vector
integrates unit(x_i - x'); cover_vector_scene already returns the integral of
unit(x - x'), so it needs no minus.
"""
import pytest
import torch
import warp as wp

from edgebound.cover import cover_vector_scene
from edgebound.deltasph2d import DeltaSPHConfig, hydrostatic_tank
from edgebound.dfsph2d import F64, neighbor_pairs

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


@pytest.mark.parametrize("device", DEVICES)
def test_cover_exact_switch(device):
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    N, H = len(sim.x), sim.H
    n_w = sim.cfg.wallMass / sim.dx ** 2                       # the detector's nw (wallMass = 1: a wall of the same lattice)

    # rhs prelude (as in DeltaSPH.rhs)
    x = sim.x
    i, j, r = neighbor_pairs(x, sim.Hvec)
    lam, G, A = sim._wall_data(x, sim.rho)
    near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
    samples = (near, sim._solid_samples(x, near))
    ins, u, rk, dr, dphi = samples[1]
    area = (rk * dr * dphi)[:, None]
    wt = ins.any(0).to(F64) * area[None]
    Cw_quad = -n_w * (wt[..., None] * u[None, None]).sum((1, 2))          # as in _detect_surface
    Cw_exact = n_w * cover_vector_scene(sim.scene, x[near], H)

    # (a) default off
    assert DeltaSPHConfig().coverExact is False

    # (b) the two routes agree within the quadrature error; the magnitude is O(n_w H^2)
    scale = n_w * H * H
    assert (Cw_quad - Cw_exact).abs().max().item() <= 1e-2 * scale
    assert Cw_exact.abs().max().item() > 0.5 * scale

    # (c) sign control: a flipped exact vector must NOT agree with the quadrature
    assert (-Cw_exact - Cw_quad).abs().max().item() > 0.5 * scale

    # (d) coverExact=True: the detector runs end-to-end and returns a bool mask of length N
    sim.cfg.coverExact = True
    try:
        surf = sim._detect_surface(x, i, j, r, lam, samples)
    finally:
        sim.cfg.coverExact = False
    assert surf.dtype == torch.bool and surf.shape[0] == N
