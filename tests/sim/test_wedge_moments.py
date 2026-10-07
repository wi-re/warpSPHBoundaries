"""Wedge (corner) moment tables of the no-slip closure (sim/wallmoments.py `wedge_moments`, `WedgeWallMoments`; cfg.cornerWedgeTables, experimental).

(a) beta = pi (a straight wall) reproduces the planar table at the same distance, with vanishing n-t couplings;
(b) the quadrature equals a brute-force integral over the wedge solid (convex 30 deg, re-entrant 270 deg; alpha and Morris weights) to 5e-3 of the largest entry;
(c) mirror symmetry: a particle reflected about the bisector gets the same diagonal entries and negated couplings / M1_t from the table lookup.
"""
import math

import pytest
import torch
import warp as wp
from warpSPHCore import KernelFunctions

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.edge.precision import real
from warpSPHBoundaries.sim.deltasph2d import KERNELS
from warpSPHBoundaries.sim.wallmoments import MORRIS_ETA2, WedgeWallMoments, _wedge_nearest, curved_moments, wedge_moments

pytestmark = pytest.mark.skipif(real != wp.float64, reason="float64 contracts")
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
F64 = torch.float64
_, dW, _, _ = KERNELS[KernelFunctions.Wendland2]


def t(v):
    return torch.tensor([v], dtype=F64, device=DEV)


@pytest.mark.parametrize("weight", ["alpha", "morris"])
def test_straight_wedge_is_the_plane(weight):
    for rho, f in ((0.3, 0.2), (0.6, 0.7)):
        phi = f * math.pi / 2
        T, _ = wedge_moments(dW, 1.0, t(rho), t(phi), math.pi, weight)
        C = curved_moments(dW, 1.0, t(rho * math.cos(phi)), t(0.0), weight=weight)[0]
        diag = torch.stack([T[0, :3, 0, 0], T[0, :3, 1, 1]], 1)
        assert float((diag - C[:3]).abs().max()) <= 1e-6 * float(C[:3].abs().max())
        assert float(T[0, :, 0, 1].abs().max()) < 1e-12 and float(T[0, :, 1, 0].abs().max()) < 1e-12


def brute(rho, phi, beta, weight, n=1600):
    P0 = torch.tensor([[rho * math.cos(phi), rho * math.sin(phi)]], dtype=F64, device=DEV)
    _, n0, _ = _wedge_nearest(P0, beta)
    n0 = n0[0]
    t0 = torch.stack([-n0[1], n0[0]])
    g = (torch.arange(n, dtype=F64, device=DEV) + 0.5) / n * 2 - 1
    Y = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    r = Y.norm(dim=1)
    m = (r < 1) & (r > 1e-9)
    Y, r = Y[m], r[m]
    s, nq, solid = _wedge_nearest(P0 + Y, beta)
    tq = torch.stack([-nq[:, 1], nq[:, 0]], 1)
    L = dW(r, 1.0) * r / (r * r + MORRIS_ETA2) if weight == "morris" else dW(r, 1.0) / r
    w = L * solid.to(F64) * (2.0 / n) ** 2
    yh = Y / r[:, None]
    fr, fq = [n0, t0], [nq, tq]
    T = torch.zeros((3, 2, 2), dtype=F64)
    for a in range(2):
        for b in range(2):
            if weight == "morris":
                g0, gg = torch.full_like(L, float(a == b)), (fr[a] * fq[b]).sum(-1)
            else:
                g0, gg = (yh @ fr[a]) * (yh @ fr[b]), (yh @ fr[a]) * (yh * fq[b]).sum(-1)
            T[0, a, b], T[1, a, b], T[2, a, b] = (w * g0).sum(), (w * gg * s).sum(), (w * gg * s * s).sum()
    return T


@pytest.mark.parametrize("beta_deg", [30, 270])
@pytest.mark.parametrize("weight", ["alpha", "morris"])
def test_wedge_moments_match_the_solid_integral(beta_deg, weight):
    beta = math.radians(beta_deg)
    for rho, f in ((0.3, 0.5), (0.5, 0.9)):
        phi = f * (math.pi - beta / 2)
        T, _ = wedge_moments(dW, 1.0, t(rho), t(phi), beta, weight)
        Tb = brute(rho, phi, beta, weight)
        assert float((T[0].cpu() - Tb).abs().max()) <= 5e-3 * float(Tb.abs().max())


def test_mirror_symmetry_of_the_table():
    tab = WedgeWallMoments(dW, 1.0, DEV, math.radians(90), weight="morris", nr=9, nphi=9)
    rho, phi = torch.tensor([0.4, 0.4], dtype=F64, device=DEV), torch.tensor([0.7, -0.7], dtype=F64, device=DEV)
    T, M = tab.eval(rho, phi)
    assert torch.allclose(T[0, :, 0, 0], T[1, :, 0, 0]) and torch.allclose(T[0, :, 1, 1], T[1, :, 1, 1])
    assert torch.allclose(T[0, :, 0, 1], -T[1, :, 0, 1]) and torch.allclose(T[0, :, 1, 0], -T[1, :, 1, 0])
    assert float(T[0, 1, 0, 1].abs()) > 1e-6                                       # the coupling is not trivially zero off the bisector
    assert torch.allclose(M[0, 0], M[1, 0]) and torch.allclose(M[0, 1], -M[1, 1])
