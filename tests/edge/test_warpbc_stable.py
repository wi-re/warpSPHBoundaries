"""Tests for the opt-in Chebyshev-quadrature edge plan of warpbc (WORK-004 T4.1).

The `stable=(nodes, panels)` route of `warpbc.edge_channels` is the algorithm of the float64 numpy route
`np2d.gradient(..., stable=(nodes, panels))` (route D = (16, 8) of docs/q2-conditioning.md) moved to Warp: each
edge-profile block is compiled EXACTLY (Fractions) into its Chebyshev series on [0, R] (coefficient amplification
~1.2 instead of 1e2..2e4 of the monomial basis) and integrated by Gauss-Legendre on dyadic panels around the foot
point; the angle (atan) term stays closed form.  It is OPT-IN: the monomial plan is unchanged and remains the
default of `edge_channels`.

Purpose (Q2): the monomial plan loses digits with the degree of the kernel; the exact W^5 of Wendland C4 (the
tensile term, T4.2/T4.3) is degree 40, and its g(0,0) gradient channel is ~1e-3 (relative) off in float64 monomial
arithmetic.  The Chebyshev-quadrature route is ~1e-16 for the same kernel.

Tolerances (fixed in the WORK-004 T4.1 spec, stated BEFORE looking at the results):
  (a)  gradient channel g(0,0) of w2p5 / w4p5, stable=(16, 8), vs the exact mpmath reference (150 + GUARD = 170 dps),
       unit square, 214 points: 206 generic points max|err| <= 1e-13 * max|ref|;  all 214 (incl. the 4 vertices and
       4 edge midpoints) max|err| <= 5e-11 * max|ref|.
  (b)  negative controls, w4p5 (degree 40), 206 generic points:  the monomial path (stable=False) max|err| > 1e-4 *
       max|ref|  and  the under-resolved stable=(8, 4) max|err| > 1e-6 * max|ref|  (both must FAIL the (a) bound).
  (c)  all 9 channels, low-degree kernels (w2, w4, quintic, b7, cone, quartic), L polygon, 3000 uniform (seed 7) +
       12 boundary points, supports 0.4 and 1.0, every (point, edge) pair: max|c_stable - c_mono| <= 5e-9;  negative
       control: swapping the x/y gradient channels of b7/support 1 differs by > 1e-2 somewhere.
  (d)  independent route: the stable Warp gradient channel vs the float64 numpy route np2d.gradient(stable=(16, 8))
       for b7 (the (c) point set, support 1.0): max|diff| <= 1e-13.
  (e)  registry semantics (STABLE_KERNELS["w4p5"] = (16, 8), restored in finally): stable=None (no plan) torch.equal
       to stable=(16, 8); stable=False bit-identical to the monomial path; a passed DevicePlan + stable=None ->
       monomial even if the kernel is registered; an unregistered kernel with stable=None bit-identical to
       stable=False; and the registry entry actually changed the route (default != monomial for w4p5).
  (f)  finiteness: the z == 0 (point on an edge's chord line) and |z| = 1e-9 (a hair off it) pairs are finite for
       w4p5 through the stable route (no NaN / inf).
"""
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.edge import geometry as G
from warpSPHBoundaries.edge import kernels, np2d, warpbc
from warpSPHBoundaries.edge.core import GUARD, block_grad
from warpSPHBoundaries.edge.kernels import power_monomials as ref_coeffs, power_terms as terms

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64

# (a)/(b)/(e) the unit square, support h = 1
POLY = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
DEGMAX = 40                                # w4 k = 5 (the deepest reference degree needed)
PTS6 = [(0.3, 0.4), (1.02, 0.5), (0.5, -0.3), (0.97, 0.03), (1.3, 0.5), (0.5, -0.9)]

# (c)/(d) the L polygon (6 edges) and its boundary / near-boundary probe points
LS = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
LE = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
L_SPECIAL = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.5, 0.0), (1.5, 0.0), (2.0, 0.5),
             (0.5, 2.0), (1.0, 1.5), (1.5, 1.0), (0.0, 1.0), (-0.001, 1.0), (1.0, -0.0005)]


def _points_square():
    """214 points: 6 probe + 200 random (seed 5) in [-0.5, 1.5]^2 + the 4 vertices + 4 edge midpoints of the unit square."""
    rng = np.random.default_rng(5)
    return np.vstack([
        np.asarray(PTS6, dtype=np.float64),
        rng.uniform(-0.5, 1.5, (200, 2)),
        np.asarray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float64),
        np.asarray([(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)], dtype=np.float64),
    ])


def _mpf(x):                               # Fraction -> mpf (the mpmath constructor rejects a Fraction)
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)


def _register_wkp5():
    """register w2p5 / w4p5 in kernels.KERNELS (in memory only) if absent (the truncated-power construction of q2_conditioning)."""
    for fam in ("w2", "w4"):
        nm = fam + "p5"
        if nm not in kernels.KERNELS:
            kernels.KERNELS[nm] = kernels._from_terms(nm, terms(5, fam))


def _pairs(n, nedges, device):
    qi = torch.arange(n).repeat_interleave(nedges).to(torch.int32).to(device)
    ee = torch.arange(nedges).repeat(n).to(torch.int32).to(device)
    return qi, ee


def _grad_channel(c, n, nedges):
    """sum of the g(0,0) channel (columns 3:5) over the nedges edges of each of the n points (rows are contiguous)."""
    return c.reshape(n, nedges, 9)[:, :, 3:5].sum(dim=1)


@pytest.fixture(scope="module")
def square_refs():
    """(X, raw): the 214 square points and the shared mpmath block_grad reference (degree 0..DEGMAX) at 150 + GUARD dps.
    Computed once (module scope), shared by (a) and (b)."""
    X = _points_square()
    with mp.workdps(150 + GUARD):
        raw = []
        for (px, py) in X:
            P = G.prepare(POLY, (px, py), h=1)
            raw.append([block_grad(P, j, 1) for j in range(DEGMAX + 1)])
    return X, raw


def _ref_g0(nm, raw):
    """the exact mpmath gradient channel  (C/pi) * grad int_solid (pi W^k) dA'   (C = the registered kernel's c2_pi)."""
    fam = "w2" if nm.startswith("w2") else "w4"
    a = ref_coeffs(5, fam)
    out = np.zeros((len(raw), 2), dtype=np.float64)
    with mp.workdps(150 + GUARD):
        C = _mpf(F(kernels.KERNELS[nm].c2_pi)) / mp.pi
        for i, gs in enumerate(raw):
            tot = [mp.mpf(0), mp.mpf(0)]
            for j, aj in enumerate(a):
                if aj:
                    tot[0] += aj * gs[j][0]
                    tot[1] += aj * gs[j][1]
            out[i, 0] = float(C * tot[0])
            out[i, 1] = float(C * tot[1])
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_a_accuracy_mpmath(device, square_refs):
    """(a) the g(0,0) gradient channel of w2p5 / w4p5, stable=(16, 8), vs the exact mpmath reference (170 dps), the 214-point
    unit-square set: 206 generic points max|err| <= 1e-13 * max|ref|; all 214 max|err| <= 5e-11 * max|ref|."""
    _register_wkp5()
    X, raw = square_refs
    n = len(X)
    P = torch.tensor(X, dtype=TD, device=device)
    V = torch.tensor(POLY, dtype=TD, device=device)
    E = torch.tensor(EDGES, dtype=torch.int32, device=device)
    sup = torch.ones(n, dtype=TD, device=device)
    qi, ee = _pairs(n, 4, device)
    generic = np.arange(206)                     # the 6 probe + 200 random points (not a vertex / edge midpoint)
    for nm in ("w2p5", "w4p5"):
        c = warpbc.edge_channels(qi, ee, P, sup, V, E, nm, device=device, stable=(16, 8))
        g = _grad_channel(c, n, 4).cpu().numpy()
        ref = _ref_g0(nm, raw)
        scale = float(np.abs(ref).max())
        err = np.abs(g - ref).max(axis=1)
        gerr = float(err[generic].max())
        aerr = float(err.max())
        assert gerr <= 1e-13 * scale, (nm, gerr, scale)
        assert aerr <= 5e-11 * scale, (nm, aerr, scale)
        print(f"(a) {nm}: generic(206) worst {gerr / scale:.2e}   all(214) worst {aerr / scale:.2e}   scale {scale:.3e}")


@pytest.mark.parametrize("device", DEVICES)
def test_b_negative_controls(device, square_refs):
    """(b) w4p5 (degree 40) must FAIL the (a) bound through the monomial path (stable=False, > 1e-4 * scale) and the
    under-resolved stable=(8, 4) (> 1e-6 * scale), on the 206 generic points."""
    _register_wkp5()
    X, raw = square_refs
    n = len(X)
    P = torch.tensor(X, dtype=TD, device=device)
    V = torch.tensor(POLY, dtype=TD, device=device)
    E = torch.tensor(EDGES, dtype=torch.int32, device=device)
    sup = torch.ones(n, dtype=TD, device=device)
    qi, ee = _pairs(n, 4, device)
    ref = _ref_g0("w4p5", raw)
    scale = float(np.abs(ref).max())
    generic = np.arange(206)
    c_mono = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device, stable=False)
    emono = float(np.abs(_grad_channel(c_mono, n, 4).cpu().numpy() - ref)[generic].max())
    c_84 = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device, stable=(8, 4))
    e84 = float(np.abs(_grad_channel(c_84, n, 4).cpu().numpy() - ref)[generic].max())
    assert emono > 1e-4 * scale, (emono, scale)
    assert e84 > 1e-6 * scale, (e84, scale)
    print(f"(b) w4p5 negative controls: monomial {emono / scale:.2e} (> 1e-4)   stable(8,4) {e84 / scale:.2e} (> 1e-6)")


@pytest.mark.parametrize("device", DEVICES)
def test_c_all_channels_vs_monomial(device):
    """(c) all 9 channels, the low-degree kernels, the L polygon, 3000 uniform (seed 7) + 12 boundary points, supports 0.4
    and 1.0, every (point, edge) pair: max|c_stable - c_mono| <= 5e-9;  negative control: swapping the x/y gradient
    channels of b7/support 1 differs by > 1e-2 somewhere (the channels are not degenerate)."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.uniform(-1.2, 3.2, (3000, 2)), np.asarray(L_SPECIAL, dtype=np.float64)])
    n = len(X)
    nedges = len(LE)
    P = torch.tensor(X, dtype=TD, device=device)
    V = torch.tensor(LS, dtype=TD, device=device)
    E = torch.tensor(LE, dtype=torch.int32, device=device)
    qi, ee = _pairs(n, nedges, device)
    worst = 0.0
    for sup_val in (0.4, 1.0):
        sup = torch.full((n,), sup_val, dtype=TD, device=device)
        for nm in ("w2", "w4", "quintic", "b7", "cone", "quartic"):
            c_st = warpbc.edge_channels(qi, ee, P, sup, V, E, nm, device=device, stable=(16, 8))
            c_mo = warpbc.edge_channels(qi, ee, P, sup, V, E, nm, device=device, stable=False)
            d = float((c_st - c_mo).abs().max())
            worst = max(worst, d)
            assert d <= 5e-9, (nm, sup_val, d)
    sup = torch.full((n,), 1.0, dtype=TD, device=device)
    c_st = warpbc.edge_channels(qi, ee, P, sup, V, E, "b7", device=device, stable=(16, 8))
    dswap = float((c_st[:, [0, 1, 2, 4, 3, 6, 5, 8, 7]] - c_st).abs().max())
    assert dswap > 1e-2, dswap
    print(f"(c) all-9ch max|stable - mono| {worst:.2e} (<= 5e-9, 6 kernels x 2 supports); b7 x/y-swap {dswap:.2e} (> 1e-2)")


@pytest.mark.parametrize("device", DEVICES)
def test_d_independent_np2d_route(device):
    """(d) independent route: the stable Warp g(0,0) channel vs the float64 numpy route np2d.gradient(stable=(16, 8)) for b7
    (the (c) point set, support 1.0): max|diff| <= 1e-13."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.uniform(-1.2, 3.2, (3000, 2)), np.asarray(L_SPECIAL, dtype=np.float64)])
    n = len(X)
    nedges = len(LE)
    P = torch.tensor(X, dtype=TD, device=device)
    V = torch.tensor(LS, dtype=TD, device=device)
    E = torch.tensor(LE, dtype=torch.int32, device=device)
    sup = torch.ones(n, dtype=TD, device=device)
    qi, ee = _pairs(n, nedges, device)
    c_st = warpbc.edge_channels(qi, ee, P, sup, V, E, "b7", device=device, stable=(16, 8))
    g_warp = _grad_channel(c_st, n, nedges).cpu().numpy()
    verts_rep = np.repeat(np.asarray(LS, dtype=np.float64)[None], n, axis=0)
    g_np = np2d.gradient(verts_rep, X, "b7", h=1.0, dtype=np.float64, stable=(16, 8))
    d = float(np.abs(g_warp - g_np).max())
    assert d <= 1e-13, d
    print(f"(d) stable Warp vs np2d.gradient(stable=(16,8)) b7: max|diff| {d:.2e} (<= 1e-13)")


@pytest.mark.parametrize("device", DEVICES)
def test_e_registry_semantics(device):
    """(e) registry semantics (STABLE_KERNELS["w4p5"] = (16, 8), restored in finally): stable=None (no plan) torch.equal to
    stable=(16, 8); stable=False bit-identical to the monomial path; a passed DevicePlan + stable=None -> monomial even if
    the kernel is registered; an unregistered kernel with stable=None bit-identical to stable=False; and the registry entry
    actually changed the route (default != monomial for w4p5)."""
    _register_wkp5()
    rng = np.random.default_rng(11)
    X = rng.uniform(-0.5, 1.5, (30, 2))
    n = len(X)
    P = torch.tensor(X, dtype=TD, device=device)
    V = torch.tensor(POLY, dtype=TD, device=device)
    E = torch.tensor(EDGES, dtype=torch.int32, device=device)
    sup = torch.ones(n, dtype=TD, device=device)
    qi, ee = _pairs(n, 4, device)
    saved = dict(warpbc.STABLE_KERNELS)
    try:
        warpbc.STABLE_KERNELS["w4p5"] = (16, 8)
        c_default = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device)
        c_explicit = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device, stable=(16, 8))
        assert torch.equal(c_default, c_explicit)
        c_false = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device, stable=False)
        c_dp = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device,
                                    plan=warpbc.DevicePlan("w4p5", device), stable=None)
        assert torch.equal(c_false, c_dp)
        c_unreg_none = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4", device=device)
        c_unreg_false = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4", device=device, stable=False)
        assert torch.equal(c_unreg_none, c_unreg_false)
        assert not torch.equal(c_default, c_false)
        print("(e) registry semantics OK: default == explicit(16,8);  stable=False == DevicePlan;  unreg None == False;  default != mono")
    finally:
        warpbc.STABLE_KERNELS.clear()
        warpbc.STABLE_KERNELS.update(saved)


@pytest.mark.parametrize("device", DEVICES)
def test_f_finiteness_z_zero_and_tiny(device):
    """(f) finiteness: for w4p5 the z == 0 (point exactly on an edge's chord line) and |z| = 1e-9 (a hair off it) pairs are
    finite through the stable route (no NaN / inf)."""
    _register_wkp5()
    V = torch.tensor(POLY, dtype=TD, device=device)
    E = torch.tensor(EDGES, dtype=torch.int32, device=device)
    X = np.array([[0.5, 0.0], [0.5, 1e-9], [0.0, 0.5], [-1e-9, 0.5]], dtype=np.float64)
    n = len(X)
    P = torch.tensor(X, dtype=TD, device=device)
    sup = torch.ones(n, dtype=TD, device=device)
    qi, ee = _pairs(n, 4, device)
    c = warpbc.edge_channels(qi, ee, P, sup, V, E, "w4p5", device=device, stable=(16, 8))
    assert torch.isfinite(c).all(), (float("nan") if not torch.isfinite(c).all() else 0.0, c)
    print("(f) w4p5 z==0 / |z|=1e-9 pairs finite: OK")
