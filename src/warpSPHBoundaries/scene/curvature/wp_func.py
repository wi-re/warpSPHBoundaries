"""Composable FP64 order-two curvature moments; no sampling or policy decisions.

Use ``moment_function(channels)`` to specialize requested Scene legacy channels.
Arguments are dimensionless signed distance, unit normal, h*kappa and
h**2*kappa_s. Results use canonical unit-support order (see module docs).
Coefficient arrays are frozen inputs. Warp reverse mode differentiates the
executed smooth spline branch; guard/fallback derivatives are a separate contract.
"""
from functools import lru_cache

import warp as wp

wp.set_module_options({'enable_backward': True, 'enable_fma': False})
wf = wp.float64
v12 = wp.types.vector(length=12, dtype=wf)
CANONICAL = (0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5)


def channel_mask(channels):
    requested = set(range(12)) if channels is None else set(map(int, channels))
    if not requested <= set(range(12)):
        raise ValueError('SDF moment channels must be in 0..11')
    return sum(1 << CANONICAL[c] for c in requested)


@wp.struct
class CurvatureTable:
    q: wp.array(dtype=wf)
    f: wp.array2d(dtype=wf)
    df: wp.array2d(dtype=wf)
    coefficients: wp.array3d(dtype=wf)
    cumulative: wp.array3d(dtype=wf)
    variance: wf


@wp.func
def _segment(a: wf, q: wp.array(dtype=wf)):
    # Isolate the integer search from the differentiable arithmetic. Warp's
    # reverse replay does not retain mutated dynamic-loop search variables.
    z = wp.max(a, q[0])
    lo, hi = int(1), q.shape[0]-1
    while lo < hi:
        mid = (lo+hi)//2
        if q[mid] < z:
            lo = mid+1
        else:
            hi = mid
    return lo-1


@wp.func_grad(_segment)
def _segment_grad(a: wf, q: wp.array(dtype=wf), adj_index: int):
    # Integer branch selection has no derivative on an open spline segment.
    # Do not replay the mutating binary search in Warp's generated adjoint.
    pass


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


@wp.func_grad(_spline)
def _spline_grad(k: int, a: wf, i: int, q: wp.array(dtype=wf),
                 f: wp.array2d(dtype=wf), df: wp.array2d(dtype=wf),
                 derivative: bool, adj_value: wp.vec2d):
    # Frozen coefficient table: only differentiate the distance argument.
    first, second = wf(0.0), wf(0.0)
    if a < q[0]:
        first = df[k, 0]
    elif a < wf(1.0):
        h = q[i+1]-q[i]
        t = (a-q[i])/h
        s0, s1 = df[k, i]*h, df[k, i+1]*h
        first = ((wf(6.0)*t*t-wf(6.0)*t)*f[k, i]
                 +(wf(3.0)*t*t-wf(4.0)*t+wf(1.0))*s0
                 +(-wf(6.0)*t*t+wf(6.0)*t)*f[k, i+1]
                 +(wf(3.0)*t*t-wf(2.0)*t)*s1)/h
        if derivative:
            second = ((wf(12.0)*t-wf(6.0))*f[k, i]
                      +(wf(6.0)*t-wf(4.0))*s0
                      +(-wf(12.0)*t+wf(6.0))*f[k, i+1]
                      +(wf(6.0)*t-wf(2.0))*s1)/(h*h)
    wp.adjoint[a] += adj_value[0]*first+adj_value[1]*second


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


@wp.func_grad(_integral)
def _integral_grad(k: int, power: int, a: wf, i: int, q: wp.array(dtype=wf),
                   f: wp.array2d(dtype=wf), df: wp.array2d(dtype=wf),
                   coefficients: wp.array3d(dtype=wf), cumulative: wp.array3d(dtype=wf),
                   adj_value: wf):
    # Fundamental theorem of calculus for the same frozen Hermite spline,
    # including its linear extension below q_min. No loop replay is needed.
    if a < wf(1.0):
        weight = wf(1.0)
        if power == 1:
            weight = a
        wp.adjoint[a] -= adj_value*weight*_spline(k, a, i, q, f, df, False)[0]


@lru_cache(None)
def _moment_function(mask):
    want_first = bool(mask & 6)
    want_second = bool(mask & 56)
    want_gradient = bool(mask & 192)
    want_cov = bool(mask & 3840)

    @wp.func
    def moments(signed_d: wf, normal: wp.vec2d, k: wf, beta: wf, table: CurvatureTable):
        nxn, nyn = normal[0], normal[1]
        value = v12(wf(0.0))
        a = wp.abs(signed_d)
        # Same empty/full convention as Tier3.curvature_moments at |d/h|>=1.
        if signed_d <= wf(-1.0):
            value[0] = wf(1.0)
            value[3] = table.variance
            value[5] = table.variance
            value[8] = wf(1.0)
            value[11] = wf(1.0)
        elif signed_d < wf(1.0):
            index = _segment(a, table.q)
            f0 = _spline(0, a, index, table.q, table.f, table.df, wp.static(want_gradient or want_cov))
            f1 = _spline(1, a, index, table.q, table.f, table.df, False)[0]
            B0, B2, B4 = -f0[1], -wf(2.0)*f1, wf(0.0)
            plane = f0[0]
            if signed_d < wf(0.0):
                plane = wf(1.0)-plane
            value[0] = plane-k*B2/wf(2.0)+wf(3.0)*signed_d*k*k*B2/wf(8.0)
            tangent_x, tangent_y = -nyn, nxn
            if wp.static(want_first or want_second):
                B4 = -wf(6.0)*_integral(1, 1, a, index, table.q, table.f, table.df, table.coefficients, table.cumulative)
            if wp.static(want_first):
                mt = -beta*B4/wf(6.0)
                mn = -B2+k*signed_d*B2/wf(2.0)+k*k*(B4-wf(3.0)*signed_d*signed_d*B2)/wf(8.0)
                value[1] = mt*tangent_x+mn*nxn
                value[2] = mt*tangent_y+mn*nyn
            if wp.static(want_second):
                T = -wf(2.0)*_integral(1, 0, a, index, table.q, table.f, table.df, table.coefficients, table.cumulative)
                N = a*a*f0[0]+wf(2.0)*_integral(0, 1, a, index, table.q, table.f, table.df, table.coefficients, table.cumulative)
                if signed_d < wf(0.0):
                    T, N = table.variance-T, table.variance-N
                stt = T-k*B4/wf(2.0)+wf(5.0)*signed_d*k*k*B4/wf(8.0)
                stn = beta*signed_d*B4/wf(6.0)
                snn = N-k*signed_d*signed_d*B2/wf(2.0)+k*k*(-wf(2.0)*signed_d*B4+wf(3.0)*signed_d*signed_d*signed_d*B2)/wf(8.0)
                value[3] = stt*tangent_x*tangent_x+stn*(wf(2.0)*tangent_x*nxn)+snn*nxn*nxn
                value[4] = stt*tangent_x*tangent_y+stn*(tangent_x*nyn+nxn*tangent_y)+snn*nxn*nyn
                value[5] = stt*tangent_y*tangent_y+stn*(wf(2.0)*tangent_y*nyn)+snn*nyn*nyn
            if wp.static(want_gradient):
                gt = -beta*B2/wf(2.0)
                gn = -B0+k*signed_d*B0/wf(2.0)+wf(3.0)*k*k*(B2-signed_d*signed_d*B0)/wf(8.0)
                value[6] = (gt*tangent_x+gn*nxn)
                value[7] = (gt*tangent_y+gn*nyn)
            if wp.static(want_cov):
                ctt = plane-wf(3.0)*k*B2/wf(2.0)+wf(15.0)*signed_d*k*k*B2/wf(8.0)
                ctn = beta*signed_d*B2/wf(2.0)
                cnn = plane+signed_d*B0-k*signed_d*signed_d*B0/wf(2.0)+k*k*(-wf(6.0)*signed_d*B2+wf(3.0)*signed_d*signed_d*signed_d*B0)/wf(8.0)
                value[8] = ctt*tangent_x*tangent_x+ctn*(wf(2.0)*tangent_x*nxn)+cnn*nxn*nxn
                value[9] = ctt*tangent_x*tangent_y+ctn*(tangent_x*nyn+nxn*tangent_y)+cnn*nxn*nyn
                value[10] = value[9]
                value[11] = ctt*tangent_y*tangent_y+ctn*(wf(2.0)*tangent_y*nyn)+cnn*nyn*nyn
        # Construct a fresh vector: masked constant assignments to an existing
        # vector can retain its old adjoint in Warp's in-place reverse path.
        return v12(
            value[0] if wp.static(bool(mask & 1)) else wf(0.0),
            value[1] if wp.static(bool(mask & 2)) else wf(0.0),
            value[2] if wp.static(bool(mask & 4)) else wf(0.0),
            value[3] if wp.static(bool(mask & 8)) else wf(0.0),
            value[4] if wp.static(bool(mask & 16)) else wf(0.0),
            value[5] if wp.static(bool(mask & 32)) else wf(0.0),
            value[6] if wp.static(bool(mask & 64)) else wf(0.0),
            value[7] if wp.static(bool(mask & 128)) else wf(0.0),
            value[8] if wp.static(bool(mask & 256)) else wf(0.0),
            value[9] if wp.static(bool(mask & 512)) else wf(0.0),
            value[10] if wp.static(bool(mask & 1024)) else wf(0.0),
            value[11] if wp.static(bool(mask & 2048)) else wf(0.0)
        )

    return moments


def moment_function(channels=None):
    """Return a callable @wp.func, specialized by Scene legacy channel IDs.

    Construct once outside a consumer @wp.kernel, then call it per particle or
    interaction. This function does not validate normals or geometric accuracy.
    """
    return _moment_function(channel_mask(channels))
