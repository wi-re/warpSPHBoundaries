"""Toy fp32 P1 FEM-weighted W6 volume integral, entirely inside one kernel.

The only host preparation is an input-independent 1D Gauss rule on [0,1].
One thread consumes one particle/simplex pair and writes one scalar. This is a
fixed-order quadrature experiment, not a production accuracy-controlled backend.
Coordinates use support radius h (not the alternative 2h convention).
"""
from functools import lru_cache

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'max_unroll': 1})


@lru_cache(None)
def tetrahedron_kernel(order=8):
    """Return I = integral_tet W6(|y-x|,h) sum_a lambda_a(y) f_a dy."""
    if not isinstance(order, int) or not 2 <= order <= 64:
        raise ValueError('order must be an integer in [2,64]')
    N = wp.constant(order)

    @wp.kernel
    def tet_p1_w6(
        particle: wp.array(dtype=wp.vec3), support: wp.array(dtype=wp.float32),
        vertices: wp.array2d(dtype=wp.vec3), field: wp.array(dtype=wp.vec4),
        nodes: wp.array(dtype=wp.float32), weights: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        pair = wp.tid()
        inv_h = 1.0 / support[pair]
        a = (vertices[pair, 0] - particle[pair]) * inv_h
        e1 = (vertices[pair, 1] - vertices[pair, 0]) * inv_h
        e2 = (vertices[pair, 2] - vertices[pair, 0]) * inv_h
        e3 = (vertices[pair, 3] - vertices[pair, 0]) * inv_h
        jacobian = wp.abs(wp.dot(e1, wp.cross(e2, e3)))
        f = field[pair]
        total = float(0.0)
        residual = float(0.0)
        for i in range(N):
            u = nodes[i]
            b = 1.0 - u
            base = a + u * e1
            fbase = f[0] + u * (f[1] - f[0])
            wu = weights[i] * b * b
            for j in range(N):
                v = nodes[j]
                c = b * v
                d = b * (1.0 - v)
                line = base + c * e2
                fline = fbase + c * (f[2] - f[0])
                wuv = wu * weights[j] * (1.0 - v)
                for k in range(N):
                    t = d * nodes[k]
                    y = line + t * e3
                    r2 = wp.dot(y, y)
                    if r2 < 1.0:
                        r = wp.sqrt(r2)
                        s = 1.0 - r
                        s2 = s * s
                        s4 = s2 * s2
                        shape = s4 * s4 * (1.0 + r * (8.0 + r * (25.0 + 32.0 * r)))
                        value = wuv * weights[k] * shape * (fline + t * (f[3] - f[0]))
                        next_sum = total + value
                        if wp.abs(total) >= wp.abs(value):
                            residual += (total - next_sum) + value
                        else:
                            residual += (value - next_sum) + total
                        total = next_sum
        output[pair] = (1365.0 / (64.0 * wp.pi)) * jacobian * (total + residual)

    return tet_p1_w6


@lru_cache(None)
def triangle_kernel(order=8):
    """2D area version, with z=0 in input vec3 coordinates."""
    if not isinstance(order, int) or not 2 <= order <= 64:
        raise ValueError('order must be an integer in [2,64]')
    N = wp.constant(order)

    @wp.kernel
    def tri_p1_w6(
        particle: wp.array(dtype=wp.vec3), support: wp.array(dtype=wp.float32),
        vertices: wp.array2d(dtype=wp.vec3), field: wp.array(dtype=wp.vec3),
        nodes: wp.array(dtype=wp.float32), weights: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        pair = wp.tid()
        inv_h = 1.0 / support[pair]
        a = (vertices[pair, 0] - particle[pair]) * inv_h
        e1 = (vertices[pair, 1] - vertices[pair, 0]) * inv_h
        e2 = (vertices[pair, 2] - vertices[pair, 0]) * inv_h
        jacobian = wp.abs(e1[0] * e2[1] - e1[1] * e2[0])
        f = field[pair]
        total = float(0.0)
        residual = float(0.0)
        for i in range(N):
            u = nodes[i]
            b = 1.0 - u
            base = a + u * e1
            fbase = f[0] + u * (f[1] - f[0])
            wu = weights[i] * b
            for j in range(N):
                t = b * nodes[j]
                y = base + t * e2
                r2 = wp.dot(y, y)
                if r2 < 1.0:
                    r = wp.sqrt(r2)
                    s = 1.0 - r
                    s2 = s * s
                    s4 = s2 * s2
                    shape = s4 * s4 * (1.0 + r * (8.0 + r * (25.0 + 32.0 * r)))
                    value = wu * weights[j] * shape * (fbase + t * (f[2] - f[0]))
                    next_sum = total + value
                    if wp.abs(total) >= wp.abs(value):
                        residual += (total - next_sum) + value
                    else:
                        residual += (value - next_sum) + total
                    total = next_sum
        output[pair] = (78.0 / (7.0 * wp.pi)) * jacobian * (total + residual)

    return tri_p1_w6
