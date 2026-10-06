"""The no-penetration law for moving walls (`DeltaSPH2D.no_penetration`; warpSPH `computeMdbcNoPenShift` works on vel_i - vel_j of the boundary particle).

(a) a particle 0.1 dx above the floor of a tank that moves with velocity u (and rotates with omega): v_rel . n < 0 -> the RELATIVE normal velocity becomes (1 - f) vn_rel with f = 3 - 4 clip(1/2 + d/dp, 1/4, 1),
    the tangential relative velocity is untouched; a particle that moves with the wall (v = u_w) is not touched, a particle that moves away from it is not touched;
(b) Galilean invariance of the law: (v + U, wall u + U) gives the corrected velocity of (v, u) plus U (round-off);
(c) the solver: a dam break in a tank that moves at constant velocity U with the fluid initially moving at U (no shifting: its Mach scaling uses the absolute speed) is the static dam break in the moving
    frame: |x - U t - x_static| and |v - U - v_static| <= 1e-9 after 80 steps, eager and as a graph; the same run with the absolute-velocity law (wall velocity 0) is not (negative control, > 1e-3).
"""
import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.sim import cases
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
F64 = torch.float64


def tank(device, u=(0.0, 0.0), omega=0.0):
    sim = cases.hydrostatic_tank(dp=0.04, domain="surface", device=device, cfg=DeltaSPHConfig(noPen="impulse", graphStep=False))[0]
    b = sim.scene.bodies[0]
    b.linearVelocity = torch.tensor(u, dtype=F64, device=device)
    b.angularVelocity = omega
    return sim


def floor_point(sim, dx_frac=0.1, xoff=0.0):
    """a world point dx_frac * dx above the floor of the tank (the wall point under it is found with signed_distance)."""
    lo = sim.x.amin(0)
    p = torch.tensor([[float(lo[0]) + 0.5 * (float(sim.x[:, 0].max()) - float(lo[0])) + xoff, float(lo[1])]], dtype=F64, device=sim.dev)
    d, n, _ = sim.scene.signed_distance(p)
    return p - (d[:, None] - dx_frac * sim.dx) * n * 0 + (dx_frac * sim.dx - d[:, None]) * n          # move along the normal to the requested distance


def place(sim, p, v):
    """all other particles far from the walls and at rest: only particle 0 is near a wall."""
    sim.x = sim.x.clone()
    sim.x[0] = p[0]
    sim.v = torch.zeros_like(sim.v)
    sim.v[0] = v
    return sim


@pytest.mark.parametrize("device", DEVICES)
def test_relative_normal_velocity_of_a_moving_wall(device):
    for u, omega in (((0.0, 0.0), 0.0), ((0.3, 0.5), 0.0), ((0.3, 0.5), 0.8)):
        sim = tank(device, u, omega)
        p = floor_point(sim, 0.1)
        d, n, hit, bidx = sim.scene.signed_distance(p, want_body=True)
        assert abs(float(d) - 0.1 * sim.dx) < 1e-12 and abs(float(n[0, 1]) - 1.0) < 1e-12 and int(bidx) == 0
        uw = sim._wall_velocity(p, d, n, bidx)[0]
        f = 3.0 - 4.0 * (0.5 + 0.1)
        vrel = torch.tensor([0.2, -1.0], dtype=F64, device=device)                                   # closing at 1, sliding at 0.2 relative to the wall
        place(sim, p, uw + vrel)
        before = sim.v[0].clone()
        sim.no_penetration()
        after = sim.v[0]
        rel = after - uw
        assert abs(float(rel[1]) - (1.0 - f) * -1.0) < 1e-12 and abs(float(rel[0]) - 0.2) < 1e-12, (u, omega, rel.tolist())
        # moving with the wall / away from it: untouched
        for vr in (torch.zeros(2, dtype=F64, device=device), torch.tensor([0.1, 0.7], dtype=F64, device=device)):
            sim2 = place(tank(device, u, omega), p, uw + vr)
            v0 = sim2.v.clone()
            sim2.no_penetration()
            assert torch.equal(sim2.v, v0)
    # (b) Galilean: shift both velocities by U
    U = torch.tensor([1.7, -0.9], dtype=F64, device=device)
    a, b = tank(device, (0.3, 0.5), 0.0), tank(device, (0.3 + 1.7, 0.5 - 0.9), 0.0)
    pa, pb = floor_point(a), floor_point(b)
    place(a, pa, torch.tensor([0.5, -0.8], dtype=F64, device=device))
    place(b, pb, torch.tensor([0.5, -0.8], dtype=F64, device=device) + U)
    a.no_penetration(); b.no_penetration()
    assert float((b.v[0] - U - a.v[0]).abs().max()) <= 1e-14


@pytest.mark.parametrize("graph", [False, True])
def test_galilean_dam_break(graph):
    if graph and not wp.is_cuda_available():
        pytest.skip("CUDA graphs")
    U = torch.tensor([0.4, 0.1], dtype=F64, device=DEVICES[0])
    W = torch.tensor([0.0, -8.0], dtype=F64, device=DEVICES[0])                                   # the fluid starts moving into the floor: the law acts

    def run(moving, absolute_law=False):
        sim = cases.marrone_dambreak(nx=24, shifting=False, noPen="impulse", device=DEVICES[0], graphStep=graph)[0]
        sim.v = sim.v + W
        if moving:
            sim.scene.bodies[0].linearVelocity = U.clone()
            sim.v = sim.v + U
        if absolute_law:
            sim._wall_velocity = lambda x, d, n, bidx: torch.zeros_like(x)
        active = 0
        for _ in range(80):
            sim.step()
            active += int(sim.nopen_count)
        return sim, active
    (s, n_active), (m, _) = run(False), run(True)
    assert n_active > 0                                                                           # the law was exercised
    t = m.time
    assert abs(t - s.time) < 1e-15
    dx = float((m.x - U * t - s.x).abs().max())
    dv = float((m.v - U - s.v).abs().max())
    print(f"graph={graph}: Galilean |dx| {dx:.1e} |dv| {dv:.1e}, corrections {n_active}")
    assert dx <= 1e-9 and dv <= 1e-9, (dx, dv)
    if not graph:
        (w, _) = run(True, absolute_law=True)
        assert float((w.v - U - s.v).abs().max()) > 1e-3                                              # negative control: the absolute-velocity law is not Galilean
