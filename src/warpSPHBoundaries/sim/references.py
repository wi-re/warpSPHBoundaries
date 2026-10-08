"""Closed-form and literature reference solutions of the example flows (notebooks/examples), in one place so that the notebooks, the studies and the tests quote the same numbers.

    tgv_energy(t, nu, k)              Taylor-Green vortex: kinetic energy decay exp(-4 nu k^2 t) of u = U (-cos kx sin ky, sin kx cos ky)
    poiseuille(y, W, f, nu)           plane Poiseuille flow between plates at y = 0, W driven by a body force f: u = f y (W - y) / (2 nu)
    couette_cylinders(r, r1, r2, om)  Taylor-Couette flow, the inner cylinder (radius r1) rotating at om, the outer (r2) at rest: u_theta(r) = A r + B / r, and the torque of the fluid on the inner cylinder
    sangani_acrivos(c)                drag coefficient K = F / (mu U) of a square periodic array of cylinders of area fraction c in Stokes flow, F per unit length for a mean pressure gradient on the whole cell
    slosh_omega(L, depth, mode)       linear natural frequency of the sloshing mode `mode` of a tank of length L and water depth `depth` (rigid walls): omega^2 = g k tanh(k depth), k = mode pi / L
    ritter_front(t, h0, g)            dry-bed dam break (Ritter): the front moves at 2 sqrt(g h0) after the initial collapse of a column of height h0 (frictionless, shallow water; an upper bound for the real front)
    CYLINDER_WAKE                     literature drag and Strouhal number of the unconfined circular cylinder at Re = 20, 40, 100
    hydrostatic(depth, rho, g)        p = rho g depth
"""
import math

import numpy as np


def tgv_energy(t, nu, k=2.0 * math.pi):
    """E(t) / E(0) of the Taylor-Green vortex with wavenumber k in both directions."""
    return np.exp(-4.0 * nu * k * k * np.asarray(t, float))


def poiseuille(y, W, f, nu):
    return f * np.asarray(y, float) * (W - np.asarray(y, float)) / (2.0 * nu)


def couette_cylinders(r, r1, r2, omega, nu=None):
    """u_theta(r) = omega r1^2 (r2^2 / r - r) / (r2^2 - r1^2); with `nu` also the torque (rho = 1) of the fluid on the inner cylinder per unit depth, T = -4 pi nu omega r1^2 r2^2 / (r2^2 - r1^2).
    Returns (u_theta, A, B, torque or None)."""
    r = np.asarray(r, float)
    den = r2 ** 2 - r1 ** 2
    A, B = -omega * r1 ** 2 / den, omega * r1 ** 2 * r2 ** 2 / den
    return A * r + B / r, A, B, (None if nu is None else -4.0 * math.pi * nu * B)


def sangani_acrivos(c):
    """Sangani & Acrivos (1982) / Hasimoto (1959) square array: K = 4 pi / (-1/2 ln c - 0.738 + c - 0.887 c^2 + 2.038 c^3)."""
    return 4.0 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c ** 2 + 2.038 * c ** 3)


def slosh_omega(L, depth, mode=1, g=9.81):
    k = mode * math.pi / L
    return math.sqrt(g * k * math.tanh(k * depth))


def ritter_front(t, h0, g=9.81):
    return 2.0 * math.sqrt(g * h0) * np.asarray(t, float)


def hydrostatic(depth, rho=1.0, g=9.81):
    return rho * g * np.asarray(depth, float)


# unconfined circular cylinder (Dennis & Chang 1970, Fornberg 1980, Tritton 1959, Williamson 1996): drag coefficient (lo, hi), recirculation length / D, lift amplitude, Strouhal number
CYLINDER_WAKE = {
    20: dict(CD=(2.0, 2.09), Lw=(0.91, 0.94)),
    40: dict(CD=(1.50, 1.55), Lw=(2.2, 2.35)),
    100: dict(CD=(1.33, 1.35), CLamp=(0.32, 0.34), St=(0.164, 0.166)),
}
