"""Discrete Fourier (von Neumann) analysis of the SPH viscous operators on a regular particle lattice in 2D: the effective viscosity as a function of the wave direction, the wave number and H / dx.

Operators (warpSPH `wp_viscosityDelta`, unit density, V = dx^2):
  alpha form (inviscid, approachOnly = False):  a_i = fac sum_j V W'(r) / r^3 (x_ij (x) x_ij) v_ij,  fac = alpha c0 H / xi.   Continuum: nu (lap v + 2 grad div v), nu = fac / 8.
  Morris (1997) form:                           a_i = 2 nu sum_j V W'(r) r / (r^2 + eta^2) v_ij,  eta^2 = 0.0025 H^2.          Continuum: nu lap v (eta -> 0).
For v = v^ exp(i k.x) the rate is a_i = -A(k) v_i with the symbol A(k) = -sum_r w(r) (1 - cos k.r) (a 2x2 matrix for the alpha form).  A shear wave (polarisation e perpendicular to k) decays at e.A e = nu_eff |k|^2.

Long waves: A(k) -> -1/2 sum_r w(r) (r.k)^2 (r (x) r): a FOURTH-rank lattice moment for the alpha form, a SECOND-rank one for Morris.  A square lattice makes every second-rank tensor isotropic, but not the fourth-rank
ones: the alpha form is anisotropic at any k on a square lattice (two shear viscosities, eta_1 for a wave along the lattice axes, eta_2 along the diagonal, nu(theta) = A + B cos 4 theta), the Morris form only at
finite k.  A hexagonal lattice makes fourth-rank tensors isotropic as well: no long-wave anisotropy for either form.

Part 2: the Stokes array with the long-wave anisotropic operator of the alpha form (Fourier-penalisation, stokes_array_ref.py), body force on the fluid only: the K a PERFECT wall closure would give with this bulk operator,
normalised the way periodic_cylinder_array.py normalises it (nu from the axis shear wave).

    python scripts/studies/viscosity_lattice_fourier.py            (table + figure; --stokes adds part 2)
"""
import math
import os
import sys

import numpy as np

XI = {"C2": 2.8213846683502197, "C4": 3.56734561920166}


def W(q, kind):
    """Wendland kernels in 2D with support 1 (normalised to unit integral)."""
    q = np.minimum(q, 1.0)
    if kind == "C2":
        return 7 / math.pi * (1 - q) ** 4 * (1 + 4 * q)
    return 9 / math.pi * (1 - q) ** 6 * (1 + 6 * q + 35 / 3 * q * q)


def dW(q, kind):
    q = np.minimum(q, 1.0)
    if kind == "C2":
        return 7 / math.pi * (-20 * q * (1 - q) ** 3)
    return 9 / math.pi * (-56 / 3 * q * (1 + 5 * q) * (1 - q) ** 5)


def offsets(Hdx, lattice):
    """neighbour offsets r (units of the support H) of a lattice with one particle per area dx^2 (dx = 1 / Hdx), and the particle volume."""
    dx = 1.0 / Hdx
    m = int(math.ceil(2 * Hdx)) + 2
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    if lattice == "square":
        r = np.stack([i.ravel(), j.ravel()], 1) * dx
    else:                                                                           # hexagonal with the same area per particle
        a = dx * math.sqrt(2 / math.sqrt(3))
        r = np.stack([(i + 0.5 * j).ravel() * a, (j * math.sqrt(3) / 2).ravel() * a], 1)
    d = np.linalg.norm(r, axis=1)
    keep = (d > 1e-12) & (d < 1.0)
    return r[keep], dx * dx


def symbol_alpha(k, Hdx, lattice, kind):
    """A(k) [.., 2, 2] of the alpha form with fac = 1 (rate = -A v); continuum shear value 1/8."""
    r, V = offsets(Hdx, lattice)
    d = np.linalg.norm(r, axis=1)
    w = V * dW(d, kind) / d ** 3                                                    # [n]
    rr = r[:, :, None] * r[:, None, :]                                              # [n, 2, 2]
    c = 1 - np.cos(r @ np.asarray(k))
    return -np.tensordot(c * w, rr, axes=([0], [0]))


def symbol_morris(k, Hdx, lattice, kind, eta2=0.0025):
    r, V = offsets(Hdx, lattice)
    d = np.linalg.norm(r, axis=1)
    w = 2 * V * dW(d, kind) * d / (d * d + eta2)
    return -np.sum(w * (1 - np.cos(r @ np.asarray(k))))


def nu_shear(form, theta, kdx, Hdx, lattice, kind):
    """effective shear viscosity / continuum value for a shear wave along theta with |k| dx = kdx."""
    kk = kdx * Hdx                                                                  # |k| in units of 1 / H
    k = kk * np.array([math.cos(theta), math.sin(theta)])
    if form == "alpha":
        e = np.array([-math.sin(theta), math.cos(theta)])
        return float(e @ symbol_alpha(k, Hdx, lattice, kind) @ e) / kk ** 2 / (1 / 8)
    return symbol_morris(k, Hdx, lattice, kind) / kk ** 2


def moments4(Hdx, lattice, kind):
    """long-wave 4th-rank moment of the alpha form, S_abcd = -1/2 sum V W'/r^3 r_a r_b r_c r_d (fac = 1): (S_xxxx, S_xxyy) on the square lattice; continuum: S_xxxx = 3 S_xxyy = 3/8."""
    r, V = offsets(Hdx, lattice)
    d = np.linalg.norm(r, axis=1)
    w = -0.5 * V * dW(d, kind) / d ** 3
    return float(np.sum(w * r[:, 0] ** 4)), float(np.sum(w * r[:, 0] ** 2 * r[:, 1] ** 2))


def table():
    print("shear viscosity / continuum value (long waves, |k| dx = 0.01; measured: shear-wave decay, n = 48, Wendland C2: H = 4 dx axis 0.9579 diag 1.0313, H = 6 dx 0.9844 / 0.9942)")
    print(f"{'kernel':6s} {'H/dx':>5s} | {'alpha sq axis':>13s} {'diag':>7s} {'spread':>7s} | {'alpha hex 0':>11s} {'30deg':>7s} | {'Morris sq axis':>14s} {'diag':>7s} | {'alpha sq @k=2pi/48 axis':>23s} {'diag':>7s}")
    for kind in ("C2", "C4"):
        for Hdx in (3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
            ax, dg = nu_shear("alpha", 0, 0.01, Hdx, "square", kind), nu_shear("alpha", math.pi / 4, 0.01, Hdx, "square", kind)
            h0, h30 = nu_shear("alpha", 0, 0.01, Hdx, "hex", kind), nu_shear("alpha", math.pi / 6, 0.01, Hdx, "hex", kind)
            m0, m45 = nu_shear("morris", 0, 0.01, Hdx, "square", kind), nu_shear("morris", math.pi / 4, 0.01, Hdx, "square", kind)
            kdx = 2 * math.pi / 48
            fa, fd = nu_shear("alpha", 0, kdx, Hdx, "square", kind), nu_shear("alpha", math.pi / 4, kdx, Hdx, "square", kind)
            print(f"{kind:6s} {Hdx:5.1f} | {ax:13.4f} {dg:7.4f} {dg / ax - 1:+7.3f} | {h0:11.4f} {h30:7.4f} | {m0:14.4f} {m45:7.4f} | {fa:23.4f} {fd:7.4f}")


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = np.linspace(0, math.pi / 2, 91)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for Hdx in (3.0, 4.0, 5.0, 6.0, 8.0):
        axs[0].plot(np.degrees(th), [nu_shear("alpha", t, 0.01, Hdx, "square", "C2") for t in th], label=f"H = {Hdx:g} dx")
        axs[1].plot(np.degrees(th), [nu_shear("alpha", t, 0.01, Hdx, "hex", "C2") for t in th], label=f"H = {Hdx:g} dx")
    kdx = np.linspace(0.01, math.pi, 120)
    for Hdx, ls in ((4.0, "-"), (6.0, "--")):
        axs[2].plot(kdx, [nu_shear("alpha", 0, q, Hdx, "square", "C2") for q in kdx], "C0" + ls, label=f"alpha, axis, H = {Hdx:g} dx")
        axs[2].plot(kdx, [nu_shear("alpha", math.pi / 4, q, Hdx, "square", "C2") for q in kdx], "C1" + ls, label=f"alpha, diagonal, H = {Hdx:g} dx")
        axs[2].plot(kdx, [nu_shear("morris", 0, q, Hdx, "square", "C2") for q in kdx], "C2" + ls, label=f"Morris, axis, H = {Hdx:g} dx")
    axs[0].plot([0, 45], [0.9579, 1.0313], "ko", label="measured, H = 4 dx (|k|dx 0.13 axis, 0.19 diag)")
    axs[0].plot([0, 45], [0.9844, 0.9942], "ks", mfc="none", label="measured, H = 6 dx (same k)")
    axs[0].set_title("alpha form, square lattice (long waves)")
    axs[1].set_title("alpha form, hexagonal lattice (long waves)")
    axs[2].set_title("wave-number dependence (square lattice)")
    for a in axs[:2]:
        a.set_xlabel("wave direction (deg)")
        a.set_ylabel("nu_eff / continuum")
        a.axhline(1, color="0.6", lw=0.8)
    axs[2].set_xlabel("|k| dx")
    axs[2].set_ylabel("nu_eff / continuum")
    axs[0].legend(fontsize=7)
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("figure:", path)


def stokes_K(shape, c, N, S4, eta=1e6, tol=1e-10):
    """Stokes array, body force on the fluid only, viscous operator = the long-wave alpha-form operator with the lattice moments S4 = (Sxxxx, Sxxyy) (fac chosen so that the continuum shear viscosity is 1):
    (A u)_a = 8 sum_bcd S_abcd k_c k_d u_b.  Returns F / U (K in units of the continuum viscosity)."""
    Sx4, Sx2y2 = S4
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = 0.35 * 1.5 / N
    if shape == "disk":
        R = math.sqrt(c / math.pi)
        chi = 0.5 * (1 - np.tanh((np.hypot(X - .5, Y - .5) - R) / w))
    else:
        h = 0.5 * math.sqrt(c)
        chi = 0.25 * (1 - np.tanh((np.abs(X - .5) - h) / w)) * (1 - np.tanh((np.abs(Y - .5) - h) / w))
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2s = K2.copy()
    K2s[0, 0] = 1.0
    # A_ab(k) = 8 S_abcd k_c k_d with the cubic tensor S (S_xxxx = S_yyyy, S_xxyy and its permutations, everything else 0)
    Axx = 8 * (Sx4 * KX ** 2 + Sx2y2 * KY ** 2)
    Ayy = 8 * (Sx4 * KY ** 2 + Sx2y2 * KX ** 2)
    Axy = 8 * 2 * Sx2y2 * KX * KY

    def proj(fx, fy):
        d = (KX * fx + KY * fy) / K2s
        return fx - KX * d, fy - KY * d

    def A(u):
        fx, fy = np.fft.fft2(u[0]), np.fft.fft2(u[1])
        lx = np.real(np.fft.ifft2(Axx * fx + Axy * fy)) + eta * chi * u[0]
        ly = np.real(np.fft.ifft2(Axy * fx + Ayy * fy)) + eta * chi * u[1]
        px, py = proj(np.fft.fft2(lx), np.fft.fft2(ly))
        return np.real(np.fft.ifft2(px)), np.real(np.fft.ifft2(py))

    def prec(u):
        s = 1.0 / (K2 + eta * chi.mean() + 1e-30)
        return tuple(np.real(np.fft.ifft2(s * np.fft.fft2(q))) for q in u)

    bx, by = proj(np.fft.fft2(1 - chi), np.zeros((N, N)))
    b = (np.real(np.fft.ifft2(bx)), np.real(np.fft.ifft2(by)))
    u = (np.zeros((N, N)), np.zeros((N, N)))
    r_ = b
    z = prec(r_)
    p = z
    rz = sum((q * bb).sum() for q, bb in zip(r_, z))
    b2 = sum((q * q).sum() for q in b)
    for it in range(40000):
        Ap = A(p)
        al = rz / sum((x_ * y_).sum() for x_, y_ in zip(p, Ap))
        u = (u[0] + al * p[0], u[1] + al * p[1])
        r_ = (r_[0] - al * Ap[0], r_[1] - al * Ap[1])
        if sum((x_ * x_).sum() for x_ in r_) < tol ** 2 * b2:
            break
        z = prec(r_)
        rz2 = sum((x_ * y_).sum() for x_, y_ in zip(r_, z))
        p = (z[0] + rz2 / rz * p[0], z[1] + rz2 / rz * p[1])
        rz = rz2
    U = u[0].mean()
    F = (eta * chi * u[0]).mean()
    return F / U


def stokes(N=256):
    print(f"\nStokes array, long-wave alpha-form operator of the square lattice (Wendland C2), Fourier N = {N}: K_aniso / K_iso, and the ratio periodic_cylinder_array.py would report for a perfect wall")
    print("(it divides by the AXIS shear viscosity: reported ratio = (K_aniso / K_iso) / (nu_axis / nu_continuum))")
    for shape, c in (("disk", 0.1257), ("square", 1 / 9)):
        Kiso = stokes_K(shape, c, N, (3 / 8, 1 / 8))
        for Hdx in (4.0, 6.0):
            S4 = moments4(Hdx, "square", "C2")
            Ka = stokes_K(shape, c, N, S4)
            ax = nu_shear("alpha", 0, 0.01, Hdx, "square", "C2")
            print(f"  {shape:6s} H = {Hdx:g} dx: S_xxxx {S4[0]:.4f} S_xxyy {S4[1]:.4f} (continuum 0.375 / 0.125)  K_aniso / K_iso = {Ka / Kiso:.4f}  nu_axis = {ax:.4f}  reported ratio for an exact wall = {Ka / Kiso / ax:.4f}", flush=True)


if __name__ == "__main__":
    table()
    from warpSPHBoundaries import paths
    d = paths.tmp_dir()
    os.makedirs(d, exist_ok=True)
    figure(os.path.join(d, "viscosity_lattice_fourier.png"))
    if "--stokes" in sys.argv:
        stokes()
