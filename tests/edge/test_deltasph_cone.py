"""Tests for the cfg.coneExact switch (Q3b in the free-surface detector).

The wall part of the Barecasco cone count (and of the all-neighbour count) is computed two ways:
  - polar quadrature (the default, Q3b):  nw * sum_samples wt * 1[angle(p - x_i, c) <= threshold/2]
  - closed-form area (cone_area.cone_area_scene):  nw * area(solid ∩ disk ∩ wedge)
The polar sum is a quadrature of the same area; the difference is the quadrature's own error.

Tolerances (stated with their reason):
  * (b)  the wall cone count  max |polar - exact| <= 3e-2 * n_w H^2 (the reviewer measured 1.398e-2 on the tank, the
    quadrature's own error, so a tighter bound would be false and a looser one vacuous) over the near particles with
    |C| > 1e-12 (the axis defined); max(exact) > 3 counts (not two zeros).  The all-neighbour count <= 1.5e-2 * n_w H^2
    (measured 5.69e-3); max(exact) > 30.
  * (c)  negative control: the exact cone count about the axis -c differs from the polar (axis +c) count by > 1 count
    (the reviewer measured 8.23); a correct axis dependence must be visible.
"""
import math

import pytest
import torch
import warp as wp

from edgebound.cone_area import cone_area_scene
from edgebound.deltasph2d import DeltaSPHConfig, hydrostatic_tank
from edgebound.dfsph2d import F64, neighbor_pairs

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


def _prelude(sim):
    """reproduce the prelude of DeltaSPH2D.rhs up to the unit axis c (as cone_count_probe.py): the fluid sum C, the polar wall part of C, c = C/|C|. Returns (x, i, j, r, lam, samples, near, wt, c, norm, nw, H)."""
    x, H = sim.x, sim.H
    i, j, r = neighbor_pairs(x, sim.Hvec)
    lam, G, A = sim._wall_data(x, sim.rho)
    near = torch.nonzero(lam.sum(0) > 1e-9).flatten()
    ins, u, rk, dr, dphi = sim._solid_samples(x, near)
    area = (rk * dr * dphi)[:, None]
    wt = ins.any(0).to(F64) * area[None]
    nz = i != j
    ii, jj, rr = i[nz], j[nz], r[nz]
    unit = (x[ii] - x[jj]) / rr.clamp(min=1e-300)[:, None]
    C = sim._sum(unit, ii)
    nw = sim.cfg.wallMass / sim.dx ** 2
    C = C.index_add(0, near, -nw * (wt[..., None] * u[None, None]).sum((1, 2)))
    norm = C.norm(dim=1)
    c = C / norm.clamp(min=1e-300)[:, None]
    samples = (near, (ins, u, rk, dr, dphi))
    return x, i, j, r, lam, samples, near, wt, c, norm, nw, H


@pytest.mark.parametrize("device", DEVICES)
def test_cone_exact_switch(device):
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    x, i, j, r, lam, samples, near, wt, c, norm, nw, H = _prelude(sim)
    N = len(x)
    scale = nw * H * H
    ok = norm[near] > 1e-12                                     # the axis c is defined
    ins, u, rk, dr, dphi = samples[1]

    # (a) default off
    assert DeltaSPHConfig().coneExact is False

    # (b) the wall cone count and the all-neighbour count: polar quadrature vs the closed-form area (the quadrature's own error)
    cn = (u[None] * c[near][:, None, :]).sum(2)
    cone = (torch.acos(cn.clamp(-1.0, 1.0)) <= sim.cfg.barecascoThreshold / 2).to(F64)
    wc_q = nw * (wt * cone[:, None, :]).sum((1, 2))
    wc_e = nw * cone_area_scene(sim.scene, x[near], c[near], sim.cfg.barecascoThreshold / 2, H)
    wa_q = nw * wt.sum((1, 2))
    wa_e = nw * cone_area_scene(sim.scene, x[near], c[near], math.pi, H)
    assert (wc_q - wc_e).abs()[ok].max().item() <= 3e-2 * scale
    assert wc_e[ok].max().item() > 3
    assert (wa_q - wa_e).abs().max().item() <= 1.5e-2 * scale
    assert wa_e.max().item() > 30

    # (c) negative control: the exact cone count about the axis -c differs from the polar (axis +c) count by > 1 count
    wc_neg = nw * cone_area_scene(sim.scene, x[near], -c[near], sim.cfg.barecascoThreshold / 2, H)
    assert (wc_q - wc_neg).abs().max().item() > 1.0

    # (d) coneExact=True (and, separately, coneExact=coverExact=True): the detector runs end to end and returns a bool mask of length N
    for flags in (dict(coneExact=True), dict(coneExact=True, coverExact=True)):
        for k, v in flags.items():
            setattr(sim.cfg, k, v)
        try:
            surf = sim._detect_surface(x, i, j, r, lam, samples)
        finally:
            sim.cfg.coneExact = False
            sim.cfg.coverExact = False
        assert surf.dtype == torch.bool and surf.shape[0] == N
