"""The solid DISK by the divergence theorem on its circle (the edge identity on a curved boundary, `tier4.arc_value` generalised to every channel): a vectorised float64 reference and the builder of the
tables of a disk of ANY radius (docs/disk-element.md).

Geometry (h = 1): the disk of radius R has its centre at (D, 0) relative to the query x (y = x' - x, so the solid is {|y - (D, 0)| < R}); every channel is an integral over the solid of a radial kernel
W(|y|) = (1/pi) P(|y|) (`edge.kernels` pieces).  With the radial primitive  Phi(r) = (M(r) - 1/2) / pi,  M(r) = int_0^r t pi W dt  (M = 1/2 from the support radius on) and the outward normal n = e(phi) of the circle,
dl = R dphi, y = (D + R cos phi, R sin phi):

    lam  = int W            = 1[D < R] + (1/pi) oint (n . y) (M(r) - 1/2) / r^2 dl                 (value; the indicator of x in the disk)
    m1   = int y W          = oint n Phi(r) dl                                                       (div(e_k Phi) = Phi' y_k / r = y_k W)
    g0   = int grad_x W     = - oint n W(r) dl                                                       (grad_x = - grad_y, divergence theorem)
    g1_{dj} = int y_d d_j W = - oint n_j y_d W dl + delta_dj lam

By the reflection symmetry about the axis through x and the centre only five numbers survive: lam, m1x, g0x, g1xx, g1yy (g1xy = g1yx = m1y = g0y = 0).  The integrands vanish outside the support, are smooth inside one kernel
piece and nearly singular where the circle passes close to x (|D - R| small: width |D - R| / sqrt(D R) in phi around phi = pi), so the angle range is cut at the angles where the circle meets a knot radius and
refined geometrically (ratio 4) towards phi = pi; Gauss-Legendre on every panel.  The functions themselves are smooth across |D - R| = 0 (x crossing the surface).
"""
import numpy as np

from .kernels import kernel as get_kernel

NODES = 24
MAXPANELS = 22


def _pieces(kname):
    out = []
    acc = 0.0
    for lo, hi, c in get_kernel(kname).pieces:
        c = np.array([float(v) for v in c])                      # pi W = sum c_n r^n on [lo, hi]
        n = np.arange(len(c))
        prim = lambda r: ((c / (n + 2.0)) * r[..., None] ** (n + 2.0)).sum(-1)
        off = acc - float(prim(np.array(float(lo))))
        acc = off + float(prim(np.array(float(hi))))
        out.append((float(lo), float(hi), c, off))
    return out, acc


def kernel_profiles(kname, r):
    """(W(r), Phi(r), M-normalisation) of the registered kernel at radii r (h = 1): W = pi W / pi, Phi = (M(r) - M_s) / pi where M_s = M(1) (1/2 for a normalised kernel; the registered `lw2` / `cone` /
    `w2p5` have other normalisations: the identity holds for the kernel as registered)."""
    pcs, Ms = _pieces(kname)
    r = np.asarray(r, dtype=float)
    W = np.zeros_like(r)
    M = np.full_like(r, Ms)
    for lo, hi, c, off in pcs:
        sel = (r >= lo) & (r < hi)
        n = np.arange(len(c))
        W = np.where(sel, (c * r[..., None] ** n).sum(-1), W)
        M = np.where(sel, off + ((c / (n + 2.0)) * r[..., None] ** (n + 2.0)).sum(-1), M)
    return W / np.pi, (M - Ms) / np.pi, Ms


def _angle(r, D, R):
    """phi in [0, pi] where the circle (centre D, radius R) meets the radius r: cos phi = (r^2 - D^2 - R^2) / (2 D R), clipped (0: the whole circle is within r, pi: none of it)."""
    return np.arccos(np.clip((r * r - D * D - R * R) / (2.0 * D * R), -1.0, 1.0))


def disk_channels(kname, D, R, nodes=NODES):
    """dict(lam, m1x, g0x, g1xx, g1yy) of the solid disk (centre at (D, 0), radius R, h = 1) for arrays D, R (broadcast), float64, any positive R (D = 0 is moved to 1e-12)."""
    D, R = np.broadcast_arrays(np.maximum(np.asarray(D, dtype=float), 1e-12), np.asarray(R, dtype=float))
    shape = D.shape
    D, R = D.ravel(), R.ravel()
    M = len(D)
    pcs, Ms = _pieces(kname)
    knots = [hi for lo, hi, c, off in pcs]
    gx, gw = np.polynomial.legendre.leggauss(nodes)
    # panel breakpoints in phi in [0, pi]: the angles of the knot radii, and a geometric refinement towards pi (the closest approach)
    edges = [np.full(M, np.pi)]
    w = np.maximum(np.abs(D - R) / np.sqrt(D * R), 1e-4)
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
    yx, yy = Dm + Rm * np.cos(phi), Rm * np.sin(phi)
    r = np.sqrt(yx * yx + yy * yy)
    nx, ny = np.cos(phi), np.sin(phi)
    W, Phi, _ = kernel_profiles(kname, r)
    ndoty = nx * yx + ny * yy
    ind = (D < R).astype(float)
    # lam: (1/pi) oint (n . y)(M - Ms) / r^2 dl = oint (n . y) Phi / r^2 dl   (Phi already carries 1/pi)
    lam = ind + 2.0 * (wt * ndoty * Phi / (r * r) * Rm).sum((1, 2))
    m1x = 2.0 * (wt * nx * Phi * Rm).sum((1, 2))
    g0x = -2.0 * (wt * nx * W * Rm).sum((1, 2))
    g1xx = -2.0 * (wt * nx * yx * W * Rm).sum((1, 2)) + lam
    g1yy = -2.0 * (wt * ny * yy * W * Rm).sum((1, 2)) + lam
    return {k: v.reshape(shape) for k, v in dict(lam=lam, m1x=m1x, g0x=g0x, g1xx=g1xx, g1yy=g1yy).items()}
