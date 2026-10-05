"""Pressure probes of the delta+-SPH cases: first-order MLS at a point, the Marrone wall probes P1-P3, the SPHERIC sensor."""
import torch

from .pairs import F64


# ---------------------------------------------------------------------------------------------------------------------------- Marrone 3.1 dam break
_GAUSS7_NODES = (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585)
_GAUSS7_WEIGHTS = (0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697)
_DISC_CHORD_WEIGHTS = tuple(w * max(0.0, 1.0 - x * x) ** 0.5 for x, w in zip(_GAUSS7_NODES, _GAUSS7_WEIGHTS))


def mls_pressure(sim, q, neighbor_threshold=4):
    """first-order MLS (Liu-Liu) fit of the fluid pressure at query points q [M,2] (warpSPH `_mlsPressureDevice`): weights V_j W, a + b.(x_j - q), Shepard fallback where the 3x3 system is ill-conditioned
    or has fewer than `neighbor_threshold` neighbours, clamped >= 0.  Returns (value [M], neighbours [M])."""
    x, H = sim.x, sim.H
    P = sim.pressure()
    V = sim.m / sim.rho
    d = x[None, :, :] - q[:, None, :]                                           # [M,N,2]
    r = d.norm(dim=2)
    inside = r < H
    w = torch.where(inside, V[None] * sim.W(r, H), torch.zeros_like(r))     # [M,N]
    nn = inside.sum(1)
    y = d / H
    B = torch.stack([torch.ones_like(r), y[..., 0], y[..., 1]], 2)              # [M,N,3]
    A = torch.einsum("mn,mni,mnj->mij", w, B, B)
    b = torch.einsum("mn,mni,n->mi", w, B, P)
    ev = torch.linalg.eigvalsh(A)
    wc = (nn >= neighbor_threshold) & (ev[:, 0] > 1e-6 * ev[:, 2].clamp(min=1e-300))
    sol = torch.linalg.solve(A + torch.where(wc, 0.0, 1.0)[:, None, None] * torch.eye(3, dtype=F64, device=x.device)[None], b[:, :, None])[:, :, 0]
    shep = torch.where(A[:, 0, 0] > 0, b[:, 0] / A[:, 0, 0].clamp(min=1e-300), torch.zeros_like(r[:, 0]))
    val = torch.where(wc, sol[:, 0], shep).clamp(min=0.0)
    return val, nn


def wall_probes(sim, info, heights=(0.16, 0.584, 1.0), disc=0.045, inset=0.0):
    """P* = P / (rho0 g H) of the three impact-wall probes (disc-averaged with the 7-point chord quadrature, samples with <= 1 neighbour dry) at the wall and one dx into the fluid."""
    xw = info["xr"] - inset
    out = []
    for xq in (xw, xw - info["dx"]):
        q = torch.tensor([[xq, info["yb"] + z + disc * s] for z in heights for s in _GAUSS7_NODES], dtype=F64, device=sim.dev)
        val, nn = mls_pressure(sim, q)
        val, nn = val.reshape(len(heights), 7), nn.reshape(len(heights), 7)
        w = torch.tensor(_DISC_CHORD_WEIGHTS, dtype=F64, device=sim.dev)[None] * (nn > 1).to(F64)
        ws = w.sum(1)
        out.append(torch.where(ws > 0, (val * w).sum(1) / ws.clamp(min=1e-12), torch.zeros_like(ws)) / (sim.cfg.rho0 * info["g"] * info["H"]))
    return out[0].cpu().numpy(), out[1].cpu().numpy()


def sloshing_probes(sim, info, radius=0.02):
    """Sensor-1 pressure in Pa: (a) warpSPH's `sensorPressureProbe`: Gaussian Shepard average (exp(-(r / (radius / 2))^2), fluid particles within `radius`) of the Tait pressure rho0 c0^2 / 7 ((rho / rho0)^7 - 1) x rho0Phys;
    (b) first-order MLS of the (linear) EOS pressure at the wall point of the sensor.  NaN where fewer than 3 neighbours."""
    c0, rho0, rp = sim.cfg.c0, sim.cfg.rho0, info["rho0Phys"]
    q = torch.tensor(info["sensor"], dtype=F64, device=sim.dev)
    r = (sim.x - q[None]).norm(dim=1)
    near = r < radius
    if int(near.sum()) >= 3:
        w = torch.exp(-(r[near] / (0.5 * radius)) ** 2)
        tait = rp * rho0 * c0 ** 2 / 7.0 * ((sim.rho[near] / rho0) ** 7 - 1.0)
        pg = float((w * tait).sum() / w.sum())
    else:
        pg = float("nan")
    val, nn = mls_pressure(sim, q[None])
    return pg, float(val[0]) * rp
