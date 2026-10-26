"""One-thread-per-query guarded SDF moments, specialized by requested channels.

The Torch implementation in scene.py remains the numerical oracle. Static
sampled jets, interval minima and coefficient splines are prepared once; queries,
guards, model evaluation and output masks execute in one Warp launch.
"""
from functools import lru_cache

import torch
import warp as wp

from .curvature.wp_func import CurvatureTable, CANONICAL, channel_mask, _moment_function

wp.set_module_options({'enable_backward': False, 'enable_fma': False})
wf = wp.float64


@lru_cache(None)
def _kernel(mask):
    # Mass is required internally by the posterior even when not requested.
    model = _moment_function(mask | 1)

    @wp.kernel
    def evaluate(positions: wp.array2d(dtype=wf), supports: wp.array(dtype=wf),
                 nodes: wp.array3d(dtype=wf), limits: wp.array(dtype=wf), metadata: wp.array2d(dtype=int),
                 ox: wf, oy: wf, spacing: wf, nx: int, ny: int,
                 max_kappa: wf, grad_tol: wf, min_denominator: wf, max_variation: wf,
                 q: wp.array(dtype=wf), f: wp.array2d(dtype=wf), df: wp.array2d(dtype=wf),
                 coefficients: wp.array3d(dtype=wf), cumulative: wp.array3d(dtype=wf), variance: wf,
                 output: wp.array2d(dtype=wf), rejected: wp.array(dtype=wp.bool)):
        row = wp.tid()
        for j in range(12):
            output[row, j] = wf(0.0)
        rejected[row] = True
        px, py, h = positions[row, 0], positions[row, 1], supports[row]
        if not wp.isfinite(px) or not wp.isfinite(py) or not wp.isfinite(h) or h <= wf(0.0):
            return
        ux, uy = (px-ox)/spacing, (py-oy)/spacing
        if not wp.isfinite(ux) or not wp.isfinite(uy):
            return
        # A rejected/out-of-grid query never enters the expansion. Covered
        # full/empty supports retain the normal-free distance-only bypass.
        if ux < wf(1.0) or uy < wf(1.0) or ux > wf(nx-2) or uy > wf(ny-2):
            return
        ix, iy = wp.min(int(wp.floor(ux)), nx-2), wp.min(int(wp.floor(uy)), ny-2)
        tx, ty = ux-wf(ix), uy-wf(iy)
        d, gx, gy, lap, lx, ly = wf(0.0), wf(0.0), wf(0.0), wf(0.0), wf(0.0), wf(0.0)
        for ca in range(2):
            for cb in range(2):
                wx, wy = wf(1.0)-tx, wf(1.0)-ty
                if ca == 1:
                    wx = tx
                if cb == 1:
                    wy = ty
                weight = wx*wy
                d += weight*nodes[0, ix+ca, iy+cb]
                gx += weight*nodes[1, ix+ca, iy+cb]
                gy += weight*nodes[2, ix+ca, iy+cb]
                lap += weight*nodes[3, ix+ca, iy+cb]
                lx += weight*nodes[4, ix+ca, iy+cb]
                ly += weight*nodes[5, ix+ca, iy+cb]
        if not wp.isfinite(d):
            return
        margin = wf(1.4142135623730951)*spacing
        if d >= h+margin:
            rejected[row] = False
            return
        if d <= -h-margin:
            if wp.static(mask & 1):
                output[row, 0] = wf(1.0)
            if wp.static(mask & 8):
                output[row, 3] = variance
            if wp.static(mask & 32):
                output[row, 5] = variance
            if wp.static(mask & 256):
                output[row, 8] = wf(1.0)
            if wp.static(mask & 2048):
                output[row, 11] = wf(1.0)
            rejected[row] = False
            return
        if ux < wf(2.0) or uy < wf(2.0) or ux > wf(nx-3) or uy > wf(ny-3):
            return
        denominator = wf(1.0)-d*lap
        norm = wp.max(wp.sqrt(gx*gx+gy*gy), wf(1.e-300))
        if not wp.isfinite(denominator) or denominator < min_denominator or not wp.isfinite(norm):
            return
        nxn, nyn = gx/norm, gy/norm
        k = lap/denominator*h
        beta = (-nyn*lx+nxn*ly)/(denominator*denominator*denominator)*h*h
        if (not wp.isfinite(k) or not wp.isfinite(beta) or wp.abs(k) > max_kappa
                or wp.abs(norm-wf(1.0)) > grad_tol or wp.abs(beta) > max_variation):
            return
        # Four sparse-table reads cover every support cell. Find the one
        # appropriate scale directly instead of launching work for all levels.
        l0, l1 = wp.floor((px-h-ox)/spacing), wp.floor((py-h-oy)/spacing)
        u0, u1 = wp.floor((px+h-ox)/spacing), wp.floor((py+h-oy)/spacing)
        if l0 < wf(0.0) or l1 < wf(0.0) or u0 >= wf(nx-1) or u1 >= wf(ny-1):
            return
        i0, j0, iend, jend = int(l0), int(l1), int(u0), int(u1)
        w0, w1 = iend-i0+1, jend-j0+1
        size, level = int(1), int(0)
        smallest = wp.min(w0, w1)
        while size*2 <= smallest:
            size *= 2
            level += 1
        if wp.max(w0, w1) > 2*size:
            return
        offset, stride = metadata[level, 0], metadata[level, 2]
        i1, j1 = iend-size+1, jend-size+1
        bound = wp.min(wp.min(limits[offset+i0*stride+j0], limits[offset+i1*stride+j0]),
                       wp.min(limits[offset+i0*stride+j1], limits[offset+i1*stride+j1]))
        if not (h <= bound):
            return
        table = CurvatureTable()
        table.q = q
        table.f = f
        table.df = df
        table.coefficients = coefficients
        table.cumulative = cumulative
        table.variance = variance
        value = model(d/h, wp.vec2d(nxn, nyn), k, beta, table)
        value[6] = value[6]/h
        value[7] = value[7]/h
        eps = wf(1.4210854715202004e-14)
        if not wp.isfinite(value[0]) or value[0] < -eps or value[0] > wf(1.0)+eps:
            return
        for j in range(12):
            if (mask & (1 << j)) != 0 and not wp.isfinite(value[j]):
                return
        for j in range(12):
            if (mask & (1 << j)) != 0:
                output[row, j] = value[j]
        rejected[row] = False

    return evaluate


def _geometry(rep):
    nodes, gradient = rep._node_fields(), rep._laplacianGradient()
    levels = rep._validityLevels()
    key = (nodes.data_ptr(), gradient.data_ptr(), id(levels), str(rep.d.device))
    cached = getattr(rep, '_warpSdf', None)
    if cached is None or cached[0] != key:
        offsets, size = [], 0
        for level in levels:
            offsets.append((size, *level.shape))
            size += level.numel()
        fields = torch.cat((nodes, gradient)).contiguous()
        limits = torch.cat([a.flatten() for a in levels]).contiguous()
        metadata = torch.tensor(offsets, dtype=torch.int32, device=rep.d.device)
        origin = tuple(rep.origin.detach().cpu().tolist())
        cached = (key, fields, limits, metadata, origin)
        rep._warpSdf = cached
    return cached[1:]


def evaluate(rep, positions, supports, table, channels=None):
    """Canonical output; channel numbers follow Scene's legacy 0..11 layout.

    Unrequested outputs are zero; scalar mass is still evaluated internally for
    validity. No normal/geometry estimate or acceptance tolerance is weakened.
    """
    wp.init()
    mask = channel_mask(channels)
    fields, limits, metadata, origin = _geometry(rep)
    cached = getattr(table, '_warpCoefficients', None)
    if cached is None:
        cached = (torch.stack(table._coeff).contiguous(), torch.stack(table._cumulative).contiguous())
        table._warpCoefficients = cached
    positions = positions.to(dtype=torch.float64, device=rep.d.device).contiguous()
    supports = supports.to(dtype=torch.float64, device=rep.d.device).contiguous()
    output = torch.empty((len(positions), 12), dtype=torch.float64, device=positions.device)
    rejected = torch.empty(len(positions), dtype=torch.bool, device=positions.device)
    if len(positions):
        # Respect the active Torch stream, including CUDA graph capture.
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device)) if positions.is_cuda else None
        wp.launch(_kernel(mask), len(positions), inputs=[
            wp.from_torch(positions, dtype=wf), wp.from_torch(supports, dtype=wf),
            wp.from_torch(fields, dtype=wf), wp.from_torch(limits, dtype=wf), wp.from_torch(metadata, dtype=wp.int32),
            origin[0], origin[1], rep.spacing, *rep.d.shape,
            rep.maxKappaH, rep.gradTol, rep.minCurvatureDenominator, rep.maxCurvatureVariationH2,
            wp.from_torch(table.q, dtype=wf), wp.from_torch(table.F, dtype=wf), wp.from_torch(table.dF, dtype=wf),
            wp.from_torch(cached[0], dtype=wf), wp.from_torch(cached[1], dtype=wf), table.variance,
            wp.from_torch(output, dtype=wf), wp.from_torch(rejected, dtype=wp.bool)],
            device=str(positions.device), stream=stream)
    return output, rejected
