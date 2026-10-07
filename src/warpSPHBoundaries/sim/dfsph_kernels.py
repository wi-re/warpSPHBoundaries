"""The omniSPH DFSPH pressure iterate as two fused Warp kernels over the CSR fluid adjacency (docs/plan-next-steps.md "DFSPH track", D2).

One iterate of `DFSPH2D` (`_iterate`) in torch is ~50 small tensor operations, each a few microseconds of GPU time even inside a CUDA graph; here it is two launches, one thread per particle looping over its
neighbours (`offsets[i] .. offsets[i] + counts[i]` of the Verlet list, pairs in CSR order with the stored grad W of the step):

    accel   a_i = - sum_j V_j (p_i / rho_i^2 + p_j / rho_j^2) grad W_ij
                  + sum_b [ -(p_i^+ / rho_i^2 + p_i^+) s_i mu grad lambda_b - (mu int (a1.y) grad W)_b + (1 - theta_b) q_b mu grad lambda_b ]      (wall terms when `wall`)
    update  ks_i = dt^2 sum_j Vt_j (a_i - a_j) . grad W_ij (+ dt^2 a_i . grad lambda),  p_i <- p_i + omega / alpha_i (src_i - ks_i) (clamped at 0 when `clampP`; 0 where the update is not finite),
            res_i = ks_i - src_i

exactly the arithmetic of `DFSPH2D._boundary_accel` / `_fluid_accel` / the iterate (p^+ = max(p, 0) when `clampWall`; theta_b = clip(p^+ / (-q_b), 0, 1) for q_b < 0, else 1, when `clampExcess`).  float64.
"""
import warp as wp


@wp.kernel
def dfsph_accel_kernel(offsets: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int64), gW: wp.array(dtype=wp.vec2d),
                       V: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64), p: wp.array(dtype=wp.float64),
                       wall: int, clampWall: int, clampExcess: int, sClose: wp.array(dtype=wp.float64),
                       gkb: wp.array(dtype=wp.vec2d), out1: wp.array(dtype=wp.vec2d), q: wp.array(dtype=wp.float64), nb: int, N: int,
                       acc: wp.array(dtype=wp.vec2d)):
    i = wp.tid()
    ri = rho[i]
    fi = p[i] / (ri * ri)
    a = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    k0 = offsets[i]
    for k in range(k0, k0 + counts[i]):
        j = wp.int32(nbr[k])
        rj = rho[j]
        a = a - V[j] * (fi + p[j] / (rj * rj)) * gW[k]
    if wall != 0:
        pp = p[i]
        if clampWall != 0:
            pp = wp.max(pp, wp.float64(0.0))
        pfac = (pp / (ri * ri) + pp) * sClose[i]
        for b in range(nb):
            g = gkb[b * N + i]
            o = out1[b * N + i]
            if clampExcess != 0:
                qb = q[b * N + i]
                th = wp.float64(1.0)
                if qb < wp.float64(0.0):
                    th = wp.clamp(pp / wp.max(-qb, wp.float64(1.0e-300)), wp.float64(0.0), wp.float64(1.0))
                o = o - (wp.float64(1.0) - th) * qb * g
            a = a - pfac * g - o
    acc[i] = a


@wp.kernel
def dfsph_update_kernel(offsets: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int64), gW: wp.array(dtype=wp.vec2d),
                        Vt: wp.array(dtype=wp.float64), acc: wp.array(dtype=wp.vec2d), gk: wp.array(dtype=wp.vec2d), wall: int,
                        dt2: wp.array(dtype=wp.float64), omega: wp.float64, alpha: wp.array(dtype=wp.float64), src: wp.array(dtype=wp.float64), clampP: int,
                        p: wp.array(dtype=wp.float64), res: wp.array(dtype=wp.float64)):
    i = wp.tid()
    ai = acc[i]
    s = wp.float64(0.0)
    k0 = offsets[i]
    for k in range(k0, k0 + counts[i]):
        j = wp.int32(nbr[k])
        s = s + Vt[j] * wp.dot(ai - acc[j], gW[k])
    ks = dt2[0] * s
    if wall != 0:
        ks = ks + dt2[0] * wp.dot(ai, gk[i])
    pn = p[i] + omega / alpha[i] * (src[i] - ks)
    if clampP != 0:
        pn = wp.max(pn, wp.float64(0.0))
    bad = wp.abs(alpha[i]) < wp.float64(1.0e-25) or not wp.isfinite(pn) or pn > wp.float64(1.0e25)
    if bad:
        p[i] = wp.float64(0.0)
        res[i] = wp.float64(0.0)
    else:
        p[i] = pn
        res[i] = ks - src[i]
