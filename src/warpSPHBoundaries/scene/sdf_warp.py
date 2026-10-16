"""One-thread-per-query guarded SDF moments, specialized by requested channels.

The Torch implementation in scene.py remains the numerical oracle. Static
sampled jets, interval minima and coefficient splines are prepared once; queries,
guards, model evaluation and output masks execute in one Warp launch.
"""
from functools import lru_cache

import torch
import warp as wp

wp.set_module_options({'enable_backward': False, 'enable_fma': False})
wf = wp.float64
v12 = wp.types.vector(length=12, dtype=wf)
CANONICAL = (0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5)


def channel_mask(channels):
    requested = set(range(12)) if channels is None else set(map(int, channels))
    if not requested <= set(range(12)):
        raise ValueError('SDF moment channels must be in 0..11')
    return sum(1 << CANONICAL[c] for c in requested)


@wp.func
def _spline(k: int, a: wf, i: int, q: wp.array(dtype=wf),
            f: wp.array2d(dtype=wf), df: wp.array2d(dtype=wf), derivative: bool):
    z = wp.clamp(a, q[0], wf(1.0))
    h = q[i+1]-q[i]
    t = (z-q[i])/h
    t2, t3 = t*t, t*t*t
    s0, s1 = df[k, i]*h, df[k, i+1]*h
    value = (wf(2.0)*t3-wf(3.0)*t2+wf(1.0))*f[k, i]+(t3-wf(2.0)*t2+t)*s0
    value += (-wf(2.0)*t3+wf(3.0)*t2)*f[k, i+1]+(t3-t2)*s1
    slope = wf(0.0)
    if derivative:
        slope = ((wf(6.0)*t2-wf(6.0)*t)*f[k, i]+(wf(3.0)*t2-wf(4.0)*t+wf(1.0))*s0
                 +(-wf(6.0)*t2+wf(6.0)*t)*f[k, i+1]+(wf(3.0)*t2-wf(2.0)*t)*s1)/h
    if a < q[0]:
        value = f[k, 0]+df[k, 0]*(a-q[0])
        slope = df[k, 0]
    return wp.vec2d(value, slope)


@wp.func
def _integral(k: int, power: int, a: wf, i: int, q: wp.array(dtype=wf),
              f: wp.array2d(dtype=wf), df: wp.array2d(dtype=wf),
              coefficients: wp.array3d(dtype=wf), cumulative: wp.array3d(dtype=wf)):
    z = wp.clamp(a, q[0], wf(1.0))
    left, h = q[i], q[i+1]-q[i]
    t = (z-left)/h
    result = cumulative[power, k, i+1]
    tpower = t
    for j in range(4):
        weight = (wf(1.0)-tpower)/wf(j+1)
        if power == 1:
            weight = left*weight+h*(wf(1.0)-tpower*t)/wf(j+2)
        result += h*coefficients[j, k, i]*weight
        tpower *= t
    gap = wp.max(q[0]-a, wf(0.0))
    if power == 0:
        result += f[k, 0]*gap-df[k, 0]*gap*gap/wf(2.0)
    else:
        result += f[k, 0]*(q[0]*gap-gap*gap/wf(2.0))+df[k, 0]*(-q[0]*gap*gap/wf(2.0)+gap*gap*gap/wf(3.0))
    return result


@lru_cache(None)
def _kernel(mask):
    want_first = bool(mask & 6)
    want_second = bool(mask & 56)
    want_gradient = bool(mask & 192)
    want_cov = bool(mask & 3840)

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
        signed_d = d/h
        a = wp.abs(signed_d)
        value = v12(wf(0.0))
        # Same empty/full convention as Tier3.curvature_moments at |q|>=1.
        if signed_d <= wf(-1.0):
            value[0] = wf(1.0)
            value[3] = variance
            value[5] = variance
            value[8] = wf(1.0)
            value[11] = wf(1.0)
        elif signed_d < wf(1.0):
            z = wp.max(a, q[0])
            lo, hi = int(1), q.shape[0]-1
            while lo < hi:
                mid = (lo+hi)//2
                if q[mid] < z:
                    lo = mid+1
                else:
                    hi = mid
            index = lo-1
            f0 = _spline(0, a, index, q, f, df, wp.static(want_gradient or want_cov))
            f1 = _spline(1, a, index, q, f, df, False)[0]
            B0, B2, B4 = -f0[1], -wf(2.0)*f1, wf(0.0)
            plane = f0[0]
            if signed_d < wf(0.0):
                plane = wf(1.0)-plane
            value[0] = plane-k*B2/wf(2.0)+wf(3.0)*signed_d*k*k*B2/wf(8.0)
            tangent_x, tangent_y = -nyn, nxn
            if wp.static(want_first or want_second):
                B4 = -wf(6.0)*_integral(1, 1, a, index, q, f, df, coefficients, cumulative)
            if wp.static(want_first):
                mt = -beta*B4/wf(6.0)
                mn = -B2+k*signed_d*B2/wf(2.0)+k*k*(B4-wf(3.0)*signed_d*signed_d*B2)/wf(8.0)
                value[1] = mt*tangent_x+mn*nxn
                value[2] = mt*tangent_y+mn*nyn
            if wp.static(want_second):
                T = -wf(2.0)*_integral(1, 0, a, index, q, f, df, coefficients, cumulative)
                N = a*a*f0[0]+wf(2.0)*_integral(0, 1, a, index, q, f, df, coefficients, cumulative)
                if signed_d < wf(0.0):
                    T, N = variance-T, variance-N
                stt = T-k*B4/wf(2.0)+wf(5.0)*signed_d*k*k*B4/wf(8.0)
                stn = beta*signed_d*B4/wf(6.0)
                snn = N-k*signed_d*signed_d*B2/wf(2.0)+k*k*(-wf(2.0)*signed_d*B4+wf(3.0)*signed_d*signed_d*signed_d*B2)/wf(8.0)
                value[3] = stt*tangent_x*tangent_x+stn*(wf(2.0)*tangent_x*nxn)+snn*nxn*nxn
                value[4] = stt*tangent_x*tangent_y+stn*(tangent_x*nyn+nxn*tangent_y)+snn*nxn*nyn
                value[5] = stt*tangent_y*tangent_y+stn*(wf(2.0)*tangent_y*nyn)+snn*nyn*nyn
            if wp.static(want_gradient):
                gt = -beta*B2/wf(2.0)
                gn = -B0+k*signed_d*B0/wf(2.0)+wf(3.0)*k*k*(B2-signed_d*signed_d*B0)/wf(8.0)
                value[6] = (gt*tangent_x+gn*nxn)/h
                value[7] = (gt*tangent_y+gn*nyn)/h
            if wp.static(want_cov):
                ctt = plane-wf(3.0)*k*B2/wf(2.0)+wf(15.0)*signed_d*k*k*B2/wf(8.0)
                ctn = beta*signed_d*B2/wf(2.0)
                cnn = plane+signed_d*B0-k*signed_d*signed_d*B0/wf(2.0)+k*k*(-wf(6.0)*signed_d*B2+wf(3.0)*signed_d*signed_d*signed_d*B0)/wf(8.0)
                value[8] = ctt*tangent_x*tangent_x+ctn*(wf(2.0)*tangent_x*nxn)+cnn*nxn*nxn
                value[9] = ctt*tangent_x*tangent_y+ctn*(tangent_x*nyn+nxn*tangent_y)+cnn*nxn*nyn
                value[10] = value[9]
                value[11] = ctt*tangent_y*tangent_y+ctn*(wf(2.0)*tangent_y*nyn)+cnn*nyn*nyn
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
