"""Usability table of the standard SPH kernels for viscous flow on a regular square lattice in 2D: the anisotropy of the alpha (Monaghan) viscosity, the Morris viscosity, the lattice density error and the pairing criterion.

The kernels are evaluated through warpSPHCore's own functions (`eval_k`, `eval_dkdq`, `eval_C_d`; support 1, q = r / H, 2D), so the definitions are the ones the solvers use.  To compare kernels at the same RESOLUTION the width
is measured by the kernel's own standard deviation, sigma^2 = int r^2 W dA / 2 (per axis); the rows are at equal sigma / dx, labelled by the H / dx that Wendland C2 has at that sigma (`W2-equivalent H/dx`), and at warpSPH's default
packing (H / dx = xi = packing * kernelScale) for reference.  The quantities (long waves, see viscosity_lattice_fourier.py for the derivation):

  alpha axis / diag   shear viscosity of the alpha form for a wave along the lattice axis / the diagonal, over its continuum value (nu(theta) = A + B cos 4 theta on a square lattice)
  spread              diag / axis - 1: the viscosity anisotropy
  Morris              shear viscosity of the Morris form (eta^2 = 0.0025 H^2, isotropic at long waves) over the continuum Laplacian (eta -> 0)
  E0                  sum_j V W(x_i - x_j) - 1 on the lattice (density / partition-of-unity error)
  min FT              min over k of the 2D Fourier transform W^(k) / W^(0): negative values allow the pairing instability at large neighbour numbers (Dehnen & Aly 2012)
  N                   neighbours within the support, pi (H / dx)^2 (cost)

    python scripts/studies/kernel_viscosity_table.py
"""
import math

import numpy as np
import warp as wp

import warpSPHBoundaries  # noqa: F401  (float64)
from warpSPHCore.enumTypes import KernelFunctions as KF
from warpSPHCore.kernels.eval_kernel import eval_C_d, eval_dkdq, eval_k, eval_kernelScale, eval_packing
from warpSPHCore.type_config import scalar_t

KERNELS = [("cubic (M4)", KF.CubicSpline), ("quartic (M5)", KF.QuarticSpline), ("quintic (M6)", KF.QuinticSpline), ("B7 (deg 6)", KF.B7), ("B8 (deg 7)", KF.B8),
           ("Wendland C2", KF.Wendland2), ("Wendland C4", KF.Wendland4), ("Wendland C6", KF.Wendland6), ("HOCT4", KF.HOCT4), ("poly6", KF.Poly6)]
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.kernel
def _eval(q: wp.array(dtype=scalar_t), kern: wp.int32, w: wp.array(dtype=scalar_t), dw: wp.array(dtype=scalar_t)):
    i = wp.tid()
    c = eval_C_d(2, kern)
    w[i] = eval_k(q[i], 2, kern) * c
    dw[i] = eval_dkdq(q[i], 2, kern) * c


@wp.kernel
def _props(kern: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_kernelScale(kern, 2)
    out[1] = eval_packing(kern)


class Kernel:
    def __init__(self, kind):
        self.kind = kind
        p = wp.zeros(2, dtype=scalar_t, device=DEV)
        wp.launch(_props, dim=1, inputs=[kind.value, p], device=DEV)
        self.kernelScale, self.packing = (float(v) for v in p.numpy())
        q = np.linspace(0.0, 1.0, 40001)
        w, _ = self(q)
        self.sigma = math.sqrt(np.trapezoid(w * q ** 2 * 2 * math.pi * q, q) / 2)     # per-axis standard deviation in units of H

    def __call__(self, q):
        q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
        qa = wp.array(q, dtype=scalar_t, device=DEV)
        w = wp.zeros(len(q), dtype=scalar_t, device=DEV)
        dw = wp.zeros(len(q), dtype=scalar_t, device=DEV)
        wp.launch(_eval, dim=len(q), inputs=[qa, self.kind.value, w, dw], device=DEV)
        return w.numpy(), dw.numpy()


def lattice(Hdx):
    """square-lattice offsets in units of H (origin included), particle volume (dx = 1 / Hdx)."""
    dx = 1.0 / Hdx
    m = int(math.ceil(Hdx)) + 1
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    r = np.stack([i.ravel(), j.ravel()], 1) * dx
    d = np.linalg.norm(r, axis=1)
    return r[d < 1.0], d[d < 1.0], dx * dx


def row(K, Hdx):
    r, d, V = lattice(Hdx)
    w, dw = K(d)
    E0 = float(np.sum(V * w)) - 1.0
    nz = d > 1e-12
    r, d, dw = r[nz], d[nz], dw[nz]
    # alpha form: long-wave 4th-rank moments S = -1/2 sum V W'/r^3 r_a r_b r_c r_d; continuum S_xxxx = 3 S_xxyy = -(3/16) int W' r dA
    a = -0.5 * V * dw / d ** 3
    Sx4, Sx2y2 = float(np.sum(a * r[:, 0] ** 4)), float(np.sum(a * r[:, 0] ** 2 * r[:, 1] ** 2))
    q = np.linspace(0.0, 1.0, 40001)
    _, dwq = K(q)
    Icont = np.trapezoid(dwq * q * 2 * math.pi * q, q)                               # int W' r dA (= -2 for a normalised kernel)
    nu_cont = -Icont / 16.0                                                          # shear viscosity of the continuum alpha form per unit fac
    nu_axis = Sx2y2                                                                 # shear wave along x, polarisation y: e_a e_b k_c k_d S_abcd = S_yyxx
    nu_diag = 0.5 * (Sx4 - Sx2y2)                                                   # k = (1,1)/sqrt2, e = (-1,1)/sqrt2: (S_xxxx + S_yyyy - 2 S_xxyy) / 4 + ... = (Sx4 - Sx2y2) / 2
    # Morris form: long-wave rate -sum 2 V W' r / (r^2 + eta^2) (k.r)^2 / 2 per nu; continuum Laplacian per nu (eta -> 0): -int W' r dA / 2 = 1
    eta2 = 0.0025
    m = -np.sum(V * dw * d / (d * d + eta2) * r[:, 0] ** 2)
    morris = float(m / (-Icont / 2))
    return dict(axis=nu_axis / nu_cont, diag=nu_diag / nu_cont, E0=E0, morris=morris)


def min_ft(K):
    q = np.linspace(0.0, 1.0, 8001)
    w, _ = K(q)
    from scipy.special import j0
    ks = np.linspace(0.0, 120.0, 2401)                                               # k H
    ft = np.array([np.trapezoid(w * j0(k * q) * 2 * math.pi * q, q) for k in ks])
    return float(ft.min() / ft[0])


def main():
    Ks = [(name, Kernel(kind)) for name, kind in KERNELS]
    w2 = dict(Ks)["Wendland C2"]
    print("kernel constants: H / sigma (own width), warpSPH kernelScale H/h, packing h/dx, default xi = H/dx, min of the 2D Fourier transform")
    for name, K in Ks:
        print(f"  {name:13s} H/sigma {1 / K.sigma:6.3f}  kernelScale {K.kernelScale:6.4f}  packing {K.packing:6.4f}  xi {K.kernelScale * K.packing:6.3f}  min FT {min_ft(K):+.2e}", flush=True)
    for Heq in (3.0, 4.0, 5.0, 6.0):
        sdx = w2.sigma * Heq                                                         # sigma / dx of Wendland C2 at H = Heq dx
        print(f"\nequal resolution: sigma / dx = {sdx:.3f} (Wendland C2 at H = {Heq:g} dx)")
        print(f"  {'kernel':13s} {'H/dx':>6s} {'N':>5s} | {'alpha axis':>10s} {'diag':>7s} {'spread':>7s} | {'Morris':>7s} | {'E0':>9s}")
        for name, K in Ks:
            Hdx = sdx / K.sigma
            R = row(K, Hdx)
            print(f"  {name:13s} {Hdx:6.2f} {math.pi * Hdx ** 2:5.0f} | {R['axis']:10.4f} {R['diag']:7.4f} {R['diag'] / R['axis'] - 1:+7.3f} | {R['morris']:7.4f} | {R['E0']:+9.2e}", flush=True)
    print("\nwarpSPH default packing (H / dx = xi):")
    print(f"  {'kernel':13s} {'H/dx':>6s} {'sigma/dx':>8s} {'N':>5s} | {'alpha axis':>10s} {'diag':>7s} {'spread':>7s} | {'Morris':>7s} | {'E0':>9s}")
    for name, K in Ks:
        Hdx = K.kernelScale * K.packing
        R = row(K, Hdx)
        print(f"  {name:13s} {Hdx:6.2f} {K.sigma * Hdx:8.3f} {math.pi * Hdx ** 2:5.0f} | {R['axis']:10.4f} {R['diag']:7.4f} {R['diag'] / R['axis'] - 1:+7.3f} | {R['morris']:7.4f} | {R['E0']:+9.2e}", flush=True)


def shear_finite_k(K, Hdx, kdx, theta, form, eta2=0.0025):
    """shear viscosity at |k| dx = kdx along theta over the continuum value (alpha form: continuum alpha form; Morris: continuum Laplacian, eta -> 0)."""
    r, d, V = lattice(Hdx)
    nz = d > 1e-12
    r, d = r[nz], d[nz]
    _, dw = K(d)
    q = np.linspace(0.0, 1.0, 40001)
    _, dwq = K(q)
    Icont = np.trapezoid(dwq * q * 2 * math.pi * q, q)
    kk = kdx * Hdx
    k = kk * np.array([math.cos(theta), math.sin(theta)])
    c = 1 - np.cos(r @ k)
    if form == "alpha":
        e = np.array([-math.sin(theta), math.cos(theta)])
        rate = float(np.sum(-V * dw / d ** 3 * (r @ e) ** 2 * c))
        return rate / kk ** 2 / (-Icont / 16)
    rate = float(np.sum(-2 * V * dw * d / (d * d + eta2) * c))
    return rate / kk ** 2 / (-Icont / 2)


def morris_eta_bias(K, eta2=0.0025):
    """continuum Morris viscosity with the regulariser over the eta -> 0 value: int W' r^3 / (r^2 + eta^2) dr / int W' r dr."""
    q = np.linspace(0.0, 1.0, 40001)
    _, dwq = K(q)
    return float(np.trapezoid(dwq * q ** 3 / (q * q + eta2), q) / np.trapezoid(dwq * q, q))


def morris():
    Ks = [(name, Kernel(kind)) for name, kind in KERNELS]
    w2 = dict(Ks)["Wendland C2"]
    print("Morris form: continuum bias of eta^2 = 0.0025 H^2 (calibration constant), discrete long-wave value / that constant, and the axis / diagonal values at finite |k| dx (alpha form alongside)")
    for Heq in (3.0, 4.0, 5.0, 6.0):
        sdx = w2.sigma * Heq
        print(f"\nsigma / dx = {sdx:.3f} (Wendland C2 at H = {Heq:g} dx)")
        print(f"  {'kernel':13s} {'H/dx':>5s} {'eta bias':>8s} {'M lw/bias':>9s} | {'M kdx=0.5 ax':>12s} {'diag':>6s} {'kdx=1 ax':>8s} {'diag':>6s} | {'alpha kdx=0.5 ax':>16s} {'diag':>6s}")
        for name, K in Ks:
            Hdx = sdx / K.sigma
            b = morris_eta_bias(K)
            lw = shear_finite_k(K, Hdx, 1e-3, 0.0, "morris") / b
            m5a, m5d = shear_finite_k(K, Hdx, 0.5, 0.0, "morris") / b, shear_finite_k(K, Hdx, 0.5, math.pi / 4, "morris") / b
            m1a, m1d = shear_finite_k(K, Hdx, 1.0, 0.0, "morris") / b, shear_finite_k(K, Hdx, 1.0, math.pi / 4, "morris") / b
            a5a, a5d = shear_finite_k(K, Hdx, 0.5, 0.0, "alpha"), shear_finite_k(K, Hdx, 0.5, math.pi / 4, "alpha")
            print(f"  {name:13s} {Hdx:5.2f} {b:8.4f} {lw:9.5f} | {m5a:12.4f} {m5d:6.4f} {m1a:8.4f} {m1d:6.4f} | {a5a:16.4f} {a5d:6.4f}", flush=True)


if __name__ == "__main__":
    import sys
    if "--morris" in sys.argv:
        morris()
    else:
        main()
