"""Distance-only half-plane reference: no curvature recovery or T3 selection.

The mass is F0(d/h). Oriented moments additionally need the first derivative
of the SDF. All twelve channels use the existing planar polygon closed forms;
there is no curvature series, Laplacian, quadrature, or fallback in this model.
This offline reference does not change production dispatch.
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np

from ..edge import np2d
from ..edge.kernels import kernel as get_kernel
from .atlas import CHANNELS


@lru_cache(None)
def plane_variance(kernel):
    return sum(Fraction(c)*(hi**(k+4)-lo**(k+4))/Fraction(k+4)
               for lo, hi, coefficients in get_kernel(kernel).pieces for k, c in enumerate(coefficients))


def sample_distance_normal(rep, points, normal_min_norm=1e-10):
    """CPU reference sampler using values and first differences ONLY.

    Does not call SdfRep.signed(), _node_fields(), or the production guard.
    Outside central-stencil coverage, distance and oriented moments are NaN.
    A zero/near-zero gradient leaves mass defined but no direction is invented.
    """
    values = rep.d.detach().cpu().numpy()
    origin = rep.origin.detach().cpu().numpy()
    spacing = rep.spacing
    padded = np.pad(values, 1, mode='edge')
    dx = (padded[2:, 1:-1]-padded[:-2, 1:-1])/(2*spacing)
    dy = (padded[1:-1, 2:]-padded[1:-1, :-2])/(2*spacing)
    fields = np.stack((values, dx, dy))
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(1)
    u = (np.where(finite[:, None], points, origin+spacing)-origin)/spacing
    nx, ny = values.shape
    covered = finite & (u >= 1).all(1) & (u[:, 0] <= nx-2) & (u[:, 1] <= ny-2)
    uc = np.clip(u, 0, np.array(values.shape)-1)
    i = np.minimum(np.floor(uc).astype(int), np.array(values.shape)-2)
    t = uc-i
    out = np.zeros((len(points), 3))
    for a in (0, 1):
        for b in (0, 1):
            weight = (t[:, 0] if a else 1-t[:, 0])*(t[:, 1] if b else 1-t[:, 1])
            out += weight[:, None]*fields[:, i[:, 0]+a, i[:, 1]+b].T
    distance, gradient = out[:, 0].copy(), out[:, 1:]
    norm = np.linalg.norm(gradient, axis=1)
    normal_defined = covered & np.isfinite(gradient).all(1) & (norm > normal_min_norm)
    normal = np.full_like(gradient, np.nan)
    np.divide(gradient, norm[:, None], out=normal, where=normal_defined[:, None])
    distance[~covered] = np.nan
    return dict(distance=distance, normal=normal, gradient_norm=norm,
                normal_defined=normal_defined, distance_defined=covered & np.isfinite(distance))


def halfplane_channels(distance, normal=None, kernel='w4', dtype=np.float64):
    """All twelve unit-support moments of the half-plane n.y <= -d.

    n points into the fluid; y=x'-x. Scalar mass needs no normal. At empty/full
    support, isotropic moments also need no normal. Else an undefined normal
    leaves oriented channels NaN. Rotations use exact half-plane symmetries.
    """
    d = np.asarray(distance, dtype=dtype)
    n = (np.broadcast_to(np.array([0., 1.], dtype=dtype), (len(d), 2)).copy()
         if normal is None else np.asarray(normal, dtype=dtype).copy())
    norm = np.linalg.norm(n, axis=1)
    defined = np.isfinite(n).all(1) & (norm > 0)
    np.divide(n, norm[:, None], out=n, where=defined[:, None])
    n[~defined] = np.nan
    out = np.full((len(d), len(CHANNELS)), np.nan, dtype=dtype)
    known = np.isfinite(d)
    exterior, interior = known & (d >= 1), known & (d <= -1)
    out[exterior] = 0
    var = plane_variance(kernel)
    variance = dtype(var.numerator)/dtype(var.denominator)
    out[interior] = np.array([1, 0, 0, variance, 0, variance, 0, 0, 1, 0, 0, 1], dtype=dtype)
    active = np.flatnonzero(known & (np.abs(d) < 1))
    # Bound working memory and reuse the same axis-aligned half-plane. The
    # rectangle's other three edges lie strictly outside unit support.
    for start in range(0, len(active), 512):
        rows = active[start:start+512]
        q = np.column_stack((np.zeros(len(rows), dtype=dtype), d[rows]))
        vertices = np.broadcast_to(np.array([[-4, -4], [4, -4], [4, 0], [-4, 0]], dtype=dtype), (len(rows), 4, 2))
        moment = lambda alpha: np2d.moment(vertices, q, kernel, alpha, dtype=dtype, unsplit=True)
        mass = moment((0, 0))
        first = moment((0, 1))
        tangential, normal_second = moment((2, 0)), moment((0, 2))
        grad = np2d.grad_moment(vertices, q, kernel, (0, 0), dtype=dtype, unsplit=True)[:, 1]
        covariance = mass-d[rows]*grad
        nn = n[rows, :, None]*n[rows, None, :]
        eye = np.eye(2, dtype=dtype)
        second = tangential[:, None, None]*eye+(normal_second-tangential)[:, None, None]*nn
        cov = mass[:, None, None]*eye+(covariance-mass)[:, None, None]*nn
        out[rows, 0] = mass
        out[rows, 1:3] = first[:, None]*n[rows]
        out[rows, 3:6] = second.reshape(-1, 4)[:, [0, 1, 3]]
        out[rows, 6:8] = grad[:, None]*n[rows]
        out[rows, 8:] = cov.reshape(-1, 4)
    return out


def sampled_distance_only(rep, points, kernel='w4', references=True):
    """Named standalone distance-only scheme; includes an empirical precision check."""
    sampled = sample_distance_normal(rep, points)
    values = halfplane_channels(sampled['distance'], sampled['normal'], kernel)
    same_model = (halfplane_channels(sampled['distance'], sampled['normal'], kernel, np.longdouble)
                  if references else None)
    return dict(sampled, values=values, model_reference=same_model)


def exact_distance_normal(case, points):
    """Geometry-based d/n for separating local-plane bias from SDF reconstruction.

    Uses analytic segment projection or disk distances in extended precision.
    At tied projections with different closest points, or boundary vertices
    with different face normals, no unique normal is claimed.
    """
    p = np.asarray(points, dtype=np.longdouble)
    eps = 128*np.finfo(np.longdouble).eps
    if getattr(case, 'kind', None) == 'disks':
        relative = p[:, None]-np.asarray(case.centres, dtype=np.longdouble)
        radius = np.linalg.norm(relative, axis=2)
        distance = radius-np.longdouble(case.radius)
        chosen = distance.argmin(1)
        ar = np.arange(len(p))
        d = distance[ar, chosen]
        length = radius[ar, chosen]
        n = relative[ar, chosen]/np.where(length > 0, length, 1)[:, None]
        tie = (np.abs(distance-d[:, None]) <= eps*np.maximum(1, np.abs(d))[:, None]).sum(1) > 1
        n[(length == 0) | tie] = np.nan
        return d, n
    if getattr(case, 'kind', None) == 'box':
        (a, b), (c, e) = case.lo, case.hi
        vertices = np.array([[a, b], [c, b], [c, e], [a, e]], dtype=np.longdouble)
    else:
        vertices = np.asarray(case.vertices, dtype=np.longdouble)
    edge = np.roll(vertices, -1, axis=0)-vertices
    relative = p[:, None]-vertices
    t = np.clip((relative*edge).sum(2)/(edge*edge).sum(1), 0, 1)
    r = relative-t[:, :, None]*edge
    lengths = np.linalg.norm(r, axis=2)
    chosen = lengths.argmin(1)
    ar = np.arange(len(p))
    length = lengths[ar, chosen]
    a, b = -relative, np.roll(-relative, -1, axis=1)
    cross = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    winding = (((a[..., 1] <= 0) & (b[..., 1] > 0) & (cross > 0)).sum(1)-
               ((b[..., 1] <= 0) & (a[..., 1] > 0) & (cross < 0)).sum(1))
    d = np.where(winding != 0, -length, length)
    tolerance = eps*np.maximum(1, length)
    tied = np.abs(lengths-length[:, None]) <= tolerance[:, None]
    different = np.linalg.norm(r-r[ar, chosen, None], axis=2) > tolerance[:, None]
    ambiguous = (tied & different).any(1)
    n = r[ar, chosen]/np.where(d != 0, d, 1)[:, None]
    at_boundary = length <= tolerance
    e0 = vertices-vertices[0]
    e1 = np.roll(e0, -1, axis=0)
    orientation = np.sign((e0[:, 0]*e1[:, 1]-e0[:, 1]*e1[:, 0]).sum())
    faces = orientation*np.stack((edge[:, 1], -edge[:, 0]), 1)/np.linalg.norm(edge, axis=1)[:, None]
    n[at_boundary] = faces[chosen[at_boundary]]
    different_faces = np.linalg.norm(faces[None]-faces[chosen, None], axis=2) > eps
    ambiguous |= at_boundary & (tied & different_faces).any(1)
    n[ambiguous] = np.nan
    return d, n
