"""Conservative domain checks for the sampled SDF's bilinearly blended jets.

These bounds apply to the stored/interpolated fields, not to unknown subgrid
geometry or to the truncation error of the curvature expansion.
"""
import torch


def cell_support_limits(nodes, spacing, max_kappa_h, grad_tol, min_denominator,
                        max_variation_h2=None, lap_gradient=None):
    """Maximum admissible h per cell; zero means that the cell is unsafe.

    Bilinear interpolation is a convex combination of four node values. Bound
    d, lap(d), gx and gy independently, then bound the product d*lap(d), the
    gradient norm, and |lap(d)/(1-d*lap(d))| throughout each cell. The interval
    bounds deliberately overestimate the set of possible blended jets.
    """
    corners = torch.stack((nodes[:, :-1, :-1], nodes[:, 1:, :-1],
                           nodes[:, :-1, 1:], nodes[:, 1:, 1:]))
    lo, hi = corners.amin(0), corners.amax(0)
    products = torch.stack((lo[0]*lo[3], lo[0]*hi[3], hi[0]*lo[3], hi[0]*hi[3]))
    # Reserve floating-point slack when bounding the multiply and subtraction.
    eps = 64*torch.finfo(nodes.dtype).eps
    denominator = 1-products.amax(0)-eps*(1+products.abs().amax(0))
    g_lo = torch.where(lo[1:3] > 0, lo[1:3],
                       torch.where(hi[1:3] < 0, -hi[1:3], 0.))
    g_hi = torch.maximum(lo[1:3].abs(), hi[1:3].abs())
    norm_lo = torch.linalg.vector_norm(g_lo, dim=0)
    norm_hi = torch.linalg.vector_norm(g_hi, dim=0)
    finite = torch.isfinite(corners).all(dim=(0, 1)) & torch.isfinite(denominator)
    valid = (finite & (denominator >= min_denominator) &
             (norm_lo >= 1-grad_tol+eps) & (norm_hi <= 1+grad_tol-eps))
    lap_bound = torch.maximum(lo[3].abs(), hi[3].abs())
    # Avoid division by a bad denominator or a zero Laplacian. A flat cell's
    # allowable h is infinite; invalid cells remain zero, not NaN.
    limit = max_kappa_h*denominator/torch.where(lap_bound > 0, lap_bound, 1.)
    limit = torch.where(lap_bound == 0, torch.full_like(limit, float('inf')), limit*(1-eps))
    limit = torch.where(valid, limit.clamp(min=0), torch.zeros_like(limit))
    if max_variation_h2 is not None:
        if lap_gradient is None:
            raise ValueError('curvature-variation bounds require Laplacian-gradient nodes')
        gradients = torch.stack((lap_gradient[:, :-1, :-1], lap_gradient[:, 1:, :-1],
                                 lap_gradient[:, :-1, 1:], lap_gradient[:, 1:, 1:]))
        # |t.grad L| <= ||grad L|| for a unit tangent. Each blended component
        # stays within the node interval; D >= the interval lower bound.
        bound = torch.linalg.vector_norm(gradients.abs().amax(0), dim=0)
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        bound = bound/safe_denominator**3
        variation_limit = torch.sqrt(max_variation_h2/torch.where(bound > 0, bound, torch.ones_like(bound)))
        variation_limit = torch.where(bound == 0, torch.full_like(bound, float('inf')), variation_limit*(1-eps))
        variation_limit = torch.where(valid & torch.isfinite(gradients).all(dim=(0, 1)), variation_limit, torch.zeros_like(limit))
        limit = torch.minimum(limit, variation_limit)
        # Third differences need one more genuine distance node in each
        # direction than the gradient/Laplacian stencil.
        limit[:2, :] = limit[-2:, :] = 0
        limit[:, :2] = limit[:, -2:] = 0
    # Replicate-padding derivatives at outer grid nodes have no stencil
    # outside the sampled box. They cannot justify a local-model acceptance.
    limit[0, :] = 0
    limit[-1, :] = 0
    limit[:, 0] = 0
    limit[:, -1] = 0
    return limit


def square_minima(field):
    """Sparse table: minima of every 2**k square at every grid-cell origin."""
    levels = [field]
    half = 1
    while 2*half <= min(field.shape):
        previous = levels[-1]
        levels.append(torch.minimum(
            torch.minimum(previous[:-half, :-half], previous[half:, :-half]),
            torch.minimum(previous[:-half, half:], previous[half:, half:])))
        half *= 2
    return levels


def support_validity(levels, origin, spacing, positions, supports):
    """Check ALL cells intersecting the axis-aligned support bounding square.

    Isotropic supports give rectangle dimensions differing by at most one
    cell. Four overlapping power-of-two squares therefore cover the complete
    rectangle exactly. The minimum query has O(log(grid size)) tensor work,
    with no per-query scan and no host transfer of particle positions.
    """
    finite = torch.isfinite(positions).all(1) & torch.isfinite(supports) & (supports > 0)
    safe_positions = torch.where(finite[:, None], positions, origin)
    safe_supports = torch.where(finite, supports, torch.zeros_like(supports))
    lower = torch.floor((safe_positions-safe_supports[:, None]-origin)/spacing).long()
    upper = torch.floor((safe_positions+safe_supports[:, None]-origin)/spacing).long()
    shape = levels[0].shape
    inside = ((lower >= 0).all(1) & (upper[:, 0] < shape[0]) & (upper[:, 1] < shape[1]))
    widths = upper-lower+1
    smallest = widths.amin(1).clamp(min=1)
    largest = widths.amax(1)
    limit = torch.zeros_like(supports)
    for k, field in enumerate(levels):
        size = 2**k
        # The size condition also protects the covering argument from extreme
        # coordinate/support arithmetic that loses the isotropic width property.
        # CUDA log2(2**k) can round just below k. Integer selection avoids
        # false rejection of a valid (2**k) by (2**k+1) support rectangle.
        selected = (smallest >= size) & (smallest < 2*size) & (largest <= 2*size) & inside & finite
        i0 = lower[:, 0].clamp(0, field.shape[0]-1)
        j0 = lower[:, 1].clamp(0, field.shape[1]-1)
        i1 = (upper[:, 0]-size+1).clamp(0, field.shape[0]-1)
        j1 = (upper[:, 1]-size+1).clamp(0, field.shape[1]-1)
        bound = torch.minimum(torch.minimum(field[i0, j0], field[i1, j0]),
                              torch.minimum(field[i0, j1], field[i1, j1]))
        limit = torch.where(selected, bound, limit)
    return finite & inside & (supports <= limit)
