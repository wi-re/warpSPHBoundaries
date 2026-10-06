"""Warp pair kernels of the fluid side that have no warpSPH module (yet): the sums of the free-surface detector and of the particle shifting over the CSR neighbour list of a warpSPHCore Verlet
adjacency (`AdjacencyList`: pairs sorted by query, `edgeOffsets` / `numNeighbors` / `j`, self pair included).  One thread per particle, a fixed loop over its neighbours (deterministic: no atomics,
the order of the list), precision `real` (warpSPHCore's `scalar_t`).  The torch pair sums in `deltasph2d.py` (cfg.fluidWarp = False) are the oracle.

A pair counts when j != i (except the dilation, which includes i) and r <= H: a Verlet list with a skin carries pairs beyond the support, which must not count.  Quantities:

    pass 1   C_i = sum unit(x_i - x_j)         (Barecasco cover vector, fluid part)      nAll_i = number of neighbours
             Mf_i = sum V_j (-d) (x) gradW     (fluid renormalisation matrix)            raw_i = sum m / (2 (rho_i + rho_j)) [1 + R (W / W0)^4] gradW   (shift)
    cone     number of neighbours with acos(-unit . c_i) <= half angle
    dilate   sum_j surface_j (j = i included)
    normal   sum V_j (lam_j - lam_i) gradW       (grad lambda_min of the shift)
    mindot   min over neighbours j != i with F_i and F_j of n_i . n_j
"""
import math

import torch
import warp as wp

from ..edge.precision import real, torch_real, vec2_t

F64 = torch.float64


@wp.func
def _w_dw(kern: int, r: real, h: real, cf: wp.array(dtype=real)):
    """Wendland C2 (kern 0) / C4 (kern 1) in 2D: W(r) and W'(r), zero for q >= 1 (`pairs.py`)."""
    q = r / h
    w = real(0.0)
    dw = real(0.0)
    if q < real(1.0):
        a = real(1.0) - q
        if kern == 0:
            w = cf[0] / (h * h) * a * a * a * a * (real(1.0) + real(4.0) * q)
            dw = -cf[1] / (h * h * h) * q * a * a * a
        else:
            w = cf[2] / (h * h) * a * a * a * a * a * a * (real(1.0) + real(6.0) * q + real(35.0 / 3.0) * q * q)
            dw = -cf[3] / (h * h * h) * q * (real(1.0) + real(5.0) * q) * a * a * a * a * a
    return w, dw


@wp.kernel
def _pass1_kernel(x: wp.array(dtype=vec2_t), rho: wp.array(dtype=real), off: wp.array(dtype=int), nn: wp.array(dtype=int), jl: wp.array(dtype=wp.int64),
                  h: real, m: real, kern: int, cf: wp.array(dtype=real), w0: real, shiftR: real,
                  C: wp.array(dtype=vec2_t), nAll: wp.array(dtype=real), Mf: wp.array(dtype=real), raw: wp.array(dtype=vec2_t)):
    i = wp.tid()
    xi = x[i]
    c = vec2_t(real(0.0), real(0.0))
    rw = vec2_t(real(0.0), real(0.0))
    cnt = real(0.0)
    m00 = real(0.0)
    m01 = real(0.0)
    m10 = real(0.0)
    m11 = real(0.0)
    for k in range(off[i], off[i] + nn[i]):
        j = int(jl[k])
        if j == i:
            continue
        d = xi - x[j]
        r = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        if r > h:
            continue
        rr = wp.max(r, real(1.0e-300))
        c += d / rr
        cnt += real(1.0)
        w, dw = _w_dw(kern, r, h, cf)
        g = d * (dw / rr)
        vj = m / rho[j]
        m00 += vj * (-d[0]) * g[0]
        m01 += vj * (-d[0]) * g[1]
        m10 += vj * (-d[1]) * g[0]
        m11 += vj * (-d[1]) * g[1]
        t = w / w0
        coef = real(0.5) * m / (rho[i] + rho[j]) * (real(1.0) + shiftR * t * t * t * t)
        rw += coef * g
    C[i] = c
    nAll[i] = cnt
    raw[i] = rw
    Mf[4 * i] = m00
    Mf[4 * i + 1] = m01
    Mf[4 * i + 2] = m10
    Mf[4 * i + 3] = m11


@wp.kernel
def _cone_kernel(x: wp.array(dtype=vec2_t), c: wp.array(dtype=vec2_t), off: wp.array(dtype=int), nn: wp.array(dtype=int), jl: wp.array(dtype=wp.int64), h: real, half: real,
                 out: wp.array(dtype=real)):
    i = wp.tid()
    xi = x[i]
    ci = c[i]
    cnt = real(0.0)
    for k in range(off[i], off[i] + nn[i]):
        j = int(jl[k])
        if j == i:
            continue
        d = xi - x[j]
        r = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        if r > h:
            continue
        u = d / wp.max(r, real(1.0e-300))
        cosang = -(u[0] * ci[0] + u[1] * ci[1])
        if wp.acos(wp.clamp(cosang, real(-1.0), real(1.0))) <= half:
            cnt += real(1.0)
    out[i] = cnt


@wp.kernel
def _dilate_kernel(x: wp.array(dtype=vec2_t), surf: wp.array(dtype=int), off: wp.array(dtype=int), nn: wp.array(dtype=int), jl: wp.array(dtype=wp.int64), h: real,
                   out: wp.array(dtype=int)):
    i = wp.tid()
    xi = x[i]
    s = int(0)
    for k in range(off[i], off[i] + nn[i]):
        j = int(jl[k])
        d = xi - x[j]
        if wp.sqrt(d[0] * d[0] + d[1] * d[1]) <= h:
            s += surf[j]
    out[i] = s


@wp.kernel
def _normal_kernel(x: wp.array(dtype=vec2_t), rho: wp.array(dtype=real), lam: wp.array(dtype=real), off: wp.array(dtype=int), nn: wp.array(dtype=int), jl: wp.array(dtype=wp.int64),
                   h: real, m: real, kern: int, cf: wp.array(dtype=real), out: wp.array(dtype=vec2_t)):
    i = wp.tid()
    xi = x[i]
    s = vec2_t(real(0.0), real(0.0))
    for k in range(off[i], off[i] + nn[i]):
        j = int(jl[k])
        if j == i:
            continue
        d = xi - x[j]
        r = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        if r > h:
            continue
        w, dw = _w_dw(kern, r, h, cf)
        s += (m / rho[j]) * (lam[j] - lam[i]) * (d * (dw / wp.max(r, real(1.0e-300))))
    out[i] = s


@wp.kernel
def _mindot_kernel(x: wp.array(dtype=vec2_t), F: wp.array(dtype=int), nrm: wp.array(dtype=vec2_t), off: wp.array(dtype=int), nn: wp.array(dtype=int), jl: wp.array(dtype=wp.int64), h: real,
                   out: wp.array(dtype=real)):
    i = wp.tid()
    xi = x[i]
    mn = real(1.0e300)
    if F[i] != 0:
        for k in range(off[i], off[i] + nn[i]):
            j = int(jl[k])
            if j == i or F[j] == 0:
                continue
            d = xi - x[j]
            if wp.sqrt(d[0] * d[0] + d[1] * d[1]) > h:
                continue
            mn = wp.min(mn, nrm[i][0] * nrm[j][0] + nrm[i][1] * nrm[j][1])
    out[i] = mn


_CF = {}


def _kernel_constants(device):
    """the four normalisation constants of the Wendland kernels as a device array (built once per device: a host-to-device copy synchronises)."""
    if device not in _CF:
        cf = torch.tensor([7.0 / math.pi, 7.0 * 20.0 / math.pi, 9.0 / math.pi, 9.0 * 56.0 / 3.0 / math.pi], dtype=torch_real, device=device)
        _CF[device] = (cf, wp.from_torch(cf, dtype=real))
    return _CF[device][1]


def _wa(t, dt):
    return wp.from_torch(t.contiguous(), dtype=dt)


class FluidKernels:
    """host side of the kernels above for one `AdjacencyList` (`adj`) at the positions `x` [N,2] (float64 torch): every method takes / returns float64 torch tensors."""

    def __init__(self, sim, adj, x, rho):
        self.sim = sim
        self.n = len(x)
        self.dev = str(x.device)
        self.kern = 0 if sim.cfg.kernel.name == "Wendland2" else 1
        self.x = x.to(torch_real).contiguous()
        self.rho = rho.to(torch_real).contiguous()
        self.off, self.nn = adj.edgeOffsets, adj.numNeighbors
        self.jl = adj.j
        self.wx = wp.from_torch(self.x, dtype=vec2_t)
        self.woff, self.wnn, self.wj = _wa(self.off, wp.int32), _wa(self.nn, wp.int32), _wa(self.jl, wp.int64)
        self.H = sim.H
        self.cf = _kernel_constants(x.device)
        self._p1 = None

    def pass1(self):
        if self._p1 is None:
            s, n = self.sim, self.n
            C = torch.empty((n, 2), dtype=torch_real, device=self.x.device)
            nAll = torch.empty(n, dtype=torch_real, device=self.x.device)
            Mf = torch.empty((n, 2, 2), dtype=torch_real, device=self.x.device)
            raw = torch.empty((n, 2), dtype=torch_real, device=self.x.device)
            wp.launch(_pass1_kernel, dim=n, device=self.dev, inputs=[self.wx, _wa(self.rho, real), self.woff, self.wnn, self.wj, real(self.H), real(s.m), self.kern, self.cf, real(s._w0), real(s.cfg.shiftR),
                                                                       wp.from_torch(C, dtype=vec2_t), wp.from_torch(nAll, dtype=real), wp.from_torch(Mf.reshape(-1), dtype=real), wp.from_torch(raw, dtype=vec2_t)])
            self._p1 = dict(C=C.to(F64), nAll=nAll.to(F64), Mf=Mf.to(F64), raw=raw.to(F64))
        return self._p1

    def cone_count(self, c, half):
        out = torch.empty(self.n, dtype=torch_real, device=self.x.device)
        wp.launch(_cone_kernel, dim=self.n, device=self.dev, inputs=[self.wx, _wa(c.to(torch_real), vec2_t), self.woff, self.wnn, self.wj, real(self.H), real(half), wp.from_torch(out, dtype=real)])
        return out.to(F64)

    def dilate(self, surface):
        out = torch.empty(self.n, dtype=torch.int32, device=self.x.device)
        wp.launch(_dilate_kernel, dim=self.n, device=self.dev, inputs=[self.wx, _wa(surface.to(torch.int32), wp.int32), self.woff, self.wnn, self.wj, real(self.H), wp.from_torch(out, dtype=wp.int32)])
        return out > 0

    def lam_gradient(self, lam):
        out = torch.empty((self.n, 2), dtype=torch_real, device=self.x.device)
        wp.launch(_normal_kernel, dim=self.n, device=self.dev, inputs=[self.wx, _wa(self.rho, real), _wa(lam.to(torch_real), real), self.woff, self.wnn, self.wj, real(self.H), real(self.sim.m), self.kern, self.cf,
                                                                         wp.from_torch(out, dtype=vec2_t)])
        return out.to(F64)

    def min_dot(self, F, nrm):
        out = torch.empty(self.n, dtype=torch_real, device=self.x.device)
        wp.launch(_mindot_kernel, dim=self.n, device=self.dev, inputs=[self.wx, _wa(F.to(torch.int32), wp.int32), _wa(nrm.to(torch_real), vec2_t), self.woff, self.wnn, self.wj, real(self.H), wp.from_torch(out, dtype=real)])
        return out.to(F64)

