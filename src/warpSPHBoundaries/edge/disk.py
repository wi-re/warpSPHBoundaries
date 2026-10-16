"""The solid DISK by the divergence theorem on its circle (the edge identity on a curved boundary, `tier4.arc_value` generalised to every channel): a vectorised float64 reference and the builder of the
tables of a disk of ANY radius (docs/disk-element.md).

Geometry (h = 1): the disk of radius R has its centre at (D, 0) relative to the query x (y = x' - x, so the solid is {|y - (D, 0)| < R}); every channel is an integral over the solid of a radial kernel
W(|y|) = (1/pi) P(|y|) (`edge.kernels` pieces).  With the radial primitive  Phi(r) = (M(r) - 1/2) / pi,  M(r) = int_0^r t pi W dt  (M = 1/2 from the support radius on) and the outward normal n = e(phi) of the circle,
dl = R dphi, y = (D + R cos phi, R sin phi):

    lam  = int W            = (1/pi) oint (n . y) M(r) / r^2 dl                                   (regular unsplit value, full circle)
    m1   = int y W          = oint n Phi(r) dl                                                       (div(e_k Phi) = Phi' y_k / r = y_k W)
    g0   = int grad_x W     = - oint n W(r) dl                                                       (grad_x = - grad_y, divergence theorem)
    g1_{dj} = int y_d d_j W = - oint n_j y_d W dl + delta_dj lam

By the reflection symmetry about the axis through x and the centre only five numbers survive: lam, m1x, g0x, g1xx, g1yy (g1xy = g1yx = m1y = g0y = 0).  The integrands vanish outside the support, are smooth inside one kernel
piece and nearly singular where the circle passes close to x (|D - R| small: width |D - R| / sqrt(D R) in phi around phi = pi), so the angle range is cut at the angles where the circle meets a knot radius and
refined geometrically (ratio 4) towards phi = pi; Gauss-Legendre on every panel.  The functions themselves are smooth across |D - R| = 0 (x crossing the surface).

The former compact-potential mass formula split off the winding indicator. Its
singular constant flux is now cancelled analytically, giving the regular
unsplit potential above with a finite origin limit. The compact Phi remains
appropriate for m1. Existing angular panels are retained for the other channels.
Kernel polynomials and radial potentials are converted exactly to Chebyshev
coefficients before rounding, preserving the high-degree registered profiles.
"""
import numpy as np
from fractions import Fraction
from functools import lru_cache

from .kernels import kernel as get_kernel
from .np2d import cheb_coeffs

NODES = 24
MAXPANELS = 26


@lru_cache(maxsize=None)
def _pieces(kname):
    """Compile kernel and radial potentials exactly into a conditioned basis."""
    raw = []
    acc = Fraction(0)
    for lo, hi, c in get_kernel(kname).pieces:
        profile = {n: v for n, v in enumerate(c) if v}
        primitive = {n+2: v/(n+2) for n, v in profile.items()}
        off = acc-sum(v*lo**n for n, v in primitive.items())
        acc = off+sum(v*hi**n for n, v in primitive.items())
        raw.append((lo, hi, profile, primitive, off))
    out = []
    for lo, hi, profile, primitive, off in raw:
        compact = dict(primitive)
        compact[0] = off-acc
        regular = {n: v/(n+2) for n, v in profile.items()}
        compile_ = lambda p: np.array([float(v) for v in cheb_coeffs(p, hi)])
        out.append((float(lo), float(hi), compile_(profile), compile_(compact),
                    compile_(regular), float(off)))
    return out, float(acc)


def kernel_profiles(kname, r):
    """(W(r), Phi(r), M-normalisation) of the registered kernel at radii r (h = 1): W = pi W / pi, Phi = (M(r) - M_s) / pi where M_s = M(1) (1/2 for a normalised kernel; the registered `lw2` / `cone` /
    `w2p5` have other normalisations: the identity holds for the kernel as registered)."""
    pcs, Ms = _pieces(kname)
    r = np.asarray(r, dtype=float)
    W = np.zeros_like(r)
    Phi = np.zeros_like(r)
    for lo, hi, wc, pc, _, _ in pcs:
        sel = (r >= lo) & (r < hi)
        x = 2.0*np.clip(r, 0., hi)/hi-1.0
        W = np.where(sel, np.polynomial.chebyshev.chebval(x, wc), W)
        Phi = np.where(sel, np.polynomial.chebyshev.chebval(x, pc), Phi)
    return W / np.pi, Phi / np.pi, Ms


def _angle(r, D, R):
    """phi in [0, pi] where the circle (centre D, radius R) meets the radius r: cos phi = (r^2 - D^2 - R^2) / (2 D R), clipped (0: the whole circle is within r, pi: none of it)."""
    return np.arccos(np.clip((r * r - D * D - R * R) / (2.0 * D * R), -1.0, 1.0))


def _mass_over_r2(pieces, mass, r):
    """Regular radial primitive M(r)/r², including its finite origin limit.

    Evaluate directly; forming (Phi + mass/pi)/r² would lose the small M(r)
    signal before division. The first piece has zero integration constant.
    """
    value = np.zeros_like(r)
    outside = r >= pieces[-1][1]
    np.divide(mass, r, out=value, where=outside)
    np.divide(value, r, out=value, where=outside)
    for lo, hi, _, _, qc, off in pieces:
        sel = (r >= lo) & (r < hi)
        polynomial = np.polynomial.chebyshev.chebval(2.0*np.clip(r, 0., hi)/hi-1.0, qc)
        correction = np.zeros_like(r)
        if off != 0:
            np.divide(off, r, out=correction, where=sel)
            np.divide(correction, r, out=correction, where=sel)
        value = np.where(sel, correction+polynomial, value)
    return value


def disk_channels(kname, D, R, nodes=NODES):
    """dict(lam, m1x, g0x, g1xx, g1yy) for broadcast D,R, h=1, float64.

    D=0 uses radial symmetry exactly rather than perturbing the query.
    """
    D, R = np.broadcast_arrays(np.asarray(D, dtype=float), np.asarray(R, dtype=float))
    shape = D.shape
    centre = (D == 0).ravel()
    D = np.maximum(D, 1e-12)  # safe placeholder for angular panel construction
    D, R = D.ravel(), R.ravel()
    M = len(D)
    pcs, Ms = _pieces(kname)
    knots = [piece[1] for piece in pcs]
    gx, gw = np.polynomial.legendre.leggauss(nodes)
    # panel breakpoints in phi in [0, pi]: the angles of the knot radii, and a geometric refinement towards pi (the closest approach)
    edges = [np.full(M, np.pi)]
    w = np.maximum(np.abs(D - R) / np.sqrt(D * R), 1e-12)
    for k in range(MAXPANELS):
        edges.append(np.maximum(np.pi - w * 4.0 ** k, 0.0))
    for kn in knots:
        edges.append(_angle(kn, D, R))
    E = np.sort(np.stack(edges, 1), 1)                           # [M, P] ascending; panels [E_k, E_{k+1}]
    lo, hi = E[:, :-1], E[:, 1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    phi = mid[..., None] + half[..., None] * gx                  # [M, P, nodes]
    wt = half[..., None] * gw
    Dm, Rm = D[:, None, None], R[:, None, None]
    # Half-angle forms preserve the small closest-approach geometry near pi.
    # R+D*cos(phi) and D+R*cos(phi) otherwise lose their quadratic terms when
    # cos(phi) rounds to -1, corrupting the flux even at D=R.
    half_cos = np.cos(phi / 2.0)
    yx, yy = (Dm - Rm) + 2.0 * Rm * half_cos**2, Rm * np.sin(phi)
    r = np.hypot(Dm - Rm, 2.0 * np.sqrt(Dm) * np.sqrt(Rm) * half_cos)
    nx, ny = np.cos(phi), np.sin(phi)
    W, Phi, _ = kernel_profiles(kname, r)
    ndoty = (Rm - Dm) + 2.0 * Dm * half_cos**2
    # Unsplit identity: use the regular vector potential y*M(r)/(pi*r²)
    # over the FULL circle. This analytically cancels the indicator against
    # the singular constant-potential flux, including its half-mass boundary
    # value. A narrower quadrature panel cannot reliably resolve that spike
    # when D and R differ by only one floating-point step.
    lam = 2.0 * (wt * ndoty * _mass_over_r2(pcs, Ms, r) / np.pi * Rm).sum((1, 2))
    m1x = 2.0 * (wt * nx * Phi * Rm).sum((1, 2))
    g0x = -2.0 * (wt * nx * W * Rm).sum((1, 2))
    g1xx = -2.0 * (wt * nx * yx * W * Rm).sum((1, 2)) + lam
    g1yy = -2.0 * (wt * ny * yy * W * Rm).sum((1, 2)) + lam
    if centre.any():
        rc = R[centre]
        wc, _, _ = kernel_profiles(kname, rc)
        within = rc < pcs[-1][1]
        mass = np.full_like(rc, 2.0*Ms)
        flux = np.zeros_like(rc)
        mass[within] = 2.0*rc[within]**2*_mass_over_r2(pcs, Ms, rc[within])
        flux[within] = np.pi*rc[within]**2*wc[within]
        lam[centre] = mass
        m1x[centre] = 0.0
        g0x[centre] = 0.0
        g1xx[centre] = mass - flux
        g1yy[centre] = g1xx[centre]
    return {k: v.reshape(shape) for k, v in dict(lam=lam, m1x=m1x, g0x=g0x, g1xx=g1xx, g1yy=g1yy).items()}
