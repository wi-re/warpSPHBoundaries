"""Tests for cone_area (Q3b geometry): the closed-form area of solid ∩ disk ∩ wedge vs an independent brute-force polar integral.

Tolerances (stated with their reason, before looking at results):
  * (a) smoke rows: atol = 1e-9 against the work-document closed forms (the reviewer verified them against the brute force to <= 3e-6); the tangent row (the mandatory z == H pitfall) atol = 1e-12 (the area is continuous in H; a chord-vs-sector bug would be O(H^2) = O(0.25)).
  * (b) vs the own brute-force midpoint polar grid (1000 x 2000, even-odd, no winding, no edge formula): max |diff| <= 1e-3 * H^2 over the 10 rows + 120 random (seed 11). Reason: the brute-force grid error is the dominant term, measured 2.95e-4 H^2 by the reviewer on this recipe (1000 x 2000, 120 random); 1e-3 leaves ~3x margin, while a formula error (a wrong sign, a missed breakpoint) is >= 1e-2 H^2 (10x above).
  * (c) cone_area (vectorised) vs cone_area_scalar (loop): max |diff| <= 1e-11 * H^2 over 300 random (seed 12). Reason: both implement the same closed form in float64; the difference is a few ulps of an O(H^2) number (round-off of tan/atan2 sums); (b) is the independent check.
  * (d) degenerate cases vs the brute force: 1e-3 * H^2.
  * (e) negative controls: the closed form with the half-angle x 1.1 must differ from the brute force (original half-angle) by > 5e-3 * H^2 on at least one of the ten rows; the TK rows computed with background = 0 must differ from the brute force by > 0.05 * H^2.
"""
import math

import numpy as np
import pytest
import torch
import warp as wp

from edgebound.cone_area import cone_area, cone_area_scalar

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
TD = torch.float64

SQ = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)            # CCW, solid inside
LS = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], dtype=float)
TK = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=float)            # CW around the fluid, solid outside


def edges_of(pts):
    n = len(pts)
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)


SQE, LSE, TKE = edges_of(SQ), edges_of(LS), edges_of(TK)


def brute(p, V, E, H, th, al, background=0, nr=1000, nphi=2000):
    """independent oracle: midpoint polar grid over disk ∩ wedge (full circle if al >= pi), even-odd point-in-polygon (solid = inside, or outside if background = 1); no winding numbers, no edge formula."""
    p = np.asarray(p, float)
    V = np.asarray(V, float)
    E = np.asarray(E, int)
    a0, a1 = (th - al, th + al) if al < math.pi else (0.0, 2.0 * math.pi)
    ph = a0 + (np.arange(nphi) + 0.5) / nphi * (a1 - a0)
    rr = (np.arange(nr) + 0.5) / nr * H
    X = p[0] + rr[:, None] * np.cos(ph)[None]
    Y = p[1] + rr[:, None] * np.sin(ph)[None]
    inside = np.zeros(X.shape, bool)
    for e0, e1 in E:
        (x1, y1), (x2, y2) = V[int(e0)], V[int(e1)]
        cond = (y1 > Y) != (y2 > Y)
        xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        inside ^= cond & (X < xi)
    solid = inside if background == 0 else ~inside
    w = rr[:, None] * (H / nr) * ((a1 - a0) / nphi)
    return float((w * solid).sum())


# the ten smoke rows of the work document: (V, E, bg, p, H, th, al, expected); the last is the tangent row (z == H)
ROWS = [
    (SQ, SQE, 0, (1.3, 0.5), 1.0, math.pi, math.pi / 6, 0.4716372514),
    (SQ, SQE, 0, (0.8, 0.5), 1.0, 0.0, math.pi / 6, 0.0230940108),
    (SQ, SQE, 0, (0.3, 0.4), 1.0, 0.0, math.pi, 1.0),
    (SQ, SQE, 0, (-0.2, -0.1), 0.8, math.pi / 4, math.pi / 6, 0.2618019462),
    (LS, LSE, 0, (1.4, 1.4), 1.0, 1.25 * math.pi, math.pi / 6, 0.4064706464),
    (LS, LSE, 0, (2.3, 0.5), 1.0, math.pi, math.pi / 6, 0.4716372514),
    (TK, TKE, 1, (0.1, 0.5), 0.5, math.pi, math.pi / 6, 0.1251261912),
    (TK, TKE, 1, (0.1, 0.1), 0.5, 1.25 * math.pi, math.pi / 6, 0.1235791858),
    (TK, TKE, 1, (0.1, 0.1), 0.5, 0.0, math.pi, 0.4797193475),
    (SQ, SQE, 0, (0.5, 0.5), 0.5, 0.0, math.pi, math.pi / 4),
]
TANGENT_ATOL = 1e-12
SMOKE_ATOL = 1e-9


def _random_cases(seed, n):
    """the work-document recipe: loops SQ / LS / TK in turn; p uniform in [-0.6, 2.4]^2 ([-0.1, 1.1]^2 for TK), H uniform in [0.2, 1.2], th uniform in [-pi, pi], al = pi/6 (every 4th case full)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        V, E, bg = (SQ, SQE, 0) if t % 3 == 0 else (LS, LSE, 0) if t % 3 == 1 else (TK, TKE, 1)
        p = rng.uniform(-0.6, 2.4, 2) if bg == 0 else rng.uniform(-0.1, 1.1, 2)
        H = float(rng.uniform(0.2, 1.2))
        th = float(rng.uniform(-math.pi, math.pi))
        al = math.pi if t % 4 == 0 else math.pi / 6
        out.append((V, E, bg, tuple(p), H, th, al))
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_smoke_rows(device):
    for idx, (V, E, bg, p, H, th, al, exp) in enumerate(ROWS):
        atol = TANGENT_ATOL if idx == len(ROWS) - 1 else SMOKE_ATOL
        got_s = cone_area_scalar(p, th, al, H, V, E, bg)
        assert abs(got_s - exp) <= atol, ("scalar", p, H, th, al, got_s, exp)
        pts = torch.tensor([list(p)], dtype=TD, device=device)
        ax = torch.tensor([[math.cos(th), math.sin(th)]], dtype=TD, device=device)
        got_v = cone_area(pts, ax, al, H, V, E, bg)[0].item()
        assert abs(got_v - exp) <= atol, ("vectorised", p, H, th, al, got_v, exp)


@pytest.mark.parametrize("device", DEVICES)
def test_vs_bruteforce_rows_and_random(device):
    cases = [(V, E, bg, p, H, th, al) for (V, E, bg, p, H, th, al, _exp) in ROWS] + _random_cases(11, 120)
    worst = 0.0
    for (V, E, bg, p, H, th, al) in cases:
        got = cone_area_scalar(p, th, al, H, V, E, bg)
        ref = brute(p, V, E, H, th, al, bg)
        worst = max(worst, abs(got - ref) / (H * H))
    assert worst <= 1e-3, worst


@pytest.mark.parametrize("device", DEVICES)
def test_vectorised_matches_scalar(device):
    cases = _random_cases(12, 300)
    worst = 0.0
    for (V, E, bg, p, H, th, al) in cases:
        pts = torch.tensor([list(p)], dtype=TD, device=device)
        ax = torch.tensor([[math.cos(th), math.sin(th)]], dtype=TD, device=device)
        got_v = cone_area(pts, ax, al, H, V, E, bg)[0].item()
        got_s = cone_area_scalar(p, th, al, H, V, E, bg)
        worst = max(worst, abs(got_v - got_s) / (H * H))
    assert worst <= 1e-11, worst


@pytest.mark.parametrize("device", DEVICES)
def test_degenerate_cases(device):
    cases = [
        (SQ, SQE, 0, (0.0, 0.0), 0.3, math.pi / 4, math.pi / 6),   # p at the vertex (0,0) of SQ
        (SQ, SQE, 0, (0.5, 0.0), 0.3, math.pi / 2, math.pi / 6),   # p on the edge
        (SQ, SQE, 0, (0.5, 0.5), 0.5, 0.0, math.pi / 6),           # p at distance H from an edge (tangent), narrow wedge
        (SQ, SQE, 0, (0.5, 0.5), 0.5, 0.0, math.pi),               # ... full wedge
    ]
    worst = 0.0
    for (V, E, bg, p, H, th, al) in cases:
        got = cone_area_scalar(p, th, al, H, V, E, bg)
        ref = brute(p, V, E, H, th, al, bg)
        worst = max(worst, abs(got - ref) / (H * H))
    assert worst <= 1e-3, worst
    pts = torch.tensor([[0.5, 0.5]], dtype=TD, device=device)
    ax0 = torch.tensor([[0.0, 0.0]], dtype=TD, device=device)
    z = cone_area(pts, ax0, math.pi / 6, 0.5, SQ, SQE, 0)[0].item()
    assert math.isfinite(z)


@pytest.mark.parametrize("device", DEVICES)
def test_negative_controls(device):
    maxdiff = 0.0
    for (V, E, bg, p, H, th, al, _exp) in ROWS:
        got = cone_area_scalar(p, th, al * 1.1, H, V, E, bg)      # wrong (x 1.1) half-angle
        ref = brute(p, V, E, H, th, al, bg)                        # the brute force with the original half-angle
        maxdiff = max(maxdiff, abs(got - ref) / (H * H))
    assert maxdiff > 5e-3, maxdiff
    maxdiff2 = 0.0
    for (V, E, bg, p, H, th, al, _exp) in ROWS:
        if V is not TK:
            continue
        got = cone_area_scalar(p, th, al, H, V, E, 0)              # wrong background = 0
        ref = brute(p, V, E, H, th, al, bg)                        # the brute force with the correct background (1)
        maxdiff2 = max(maxdiff2, abs(got - ref) / (H * H))
    assert maxdiff2 > 0.05, maxdiff2
