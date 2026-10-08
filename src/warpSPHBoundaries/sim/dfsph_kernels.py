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


# ---- the compact projection's preconditioned CG (DFSPH2D._solve_compact, projection='compact'): one iteration = four kernels and three reductions (no atomics on the scalars), capturable as a CUDA graph; the scalars live on the device
#      S = [rz, qAq, rz2, rr, active, count, tol2, the last rr]: `active` (1 until the relative residual meets tol2, or a non-finite value appears) multiplies the step, so iterations replayed past convergence are no-ops
#      and `count` is the number of active iterations (the first-converged count for the next solve's check schedule).

@wp.kernel
def cg_matvec_kernel(offsets: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int64), w: wp.array(dtype=wp.float64), D: wp.array(dtype=wp.float64),
                     q: wp.array(dtype=wp.float64), Aq: wp.array(dtype=wp.float64)):
    """Aq = -A q,  A q_i = sum_j w_ij (q_i - q_j) + D_i q_i  (w <= 0, D <= 0 the wall row: -A is positive semi-definite)"""
    i = wp.tid()
    qi = q[i]
    s = D[i] * qi
    k0 = offsets[i]
    for k in range(k0, k0 + counts[i]):
        s = s + w[k] * (qi - q[wp.int32(nbr[k])])
    Aq[i] = -s


@wp.kernel
def cg_update_kernel(S: wp.array(dtype=wp.float64), q: wp.array(dtype=wp.float64), Aq: wp.array(dtype=wp.float64), Minv: wp.array(dtype=wp.float64),
                     p: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.float64)):
    """alpha = active rz / qAq;  p += alpha q;  r -= alpha Aq;  z = M^-1 r  (the dot products r.z, r.r are reduced by the caller)"""
    i = wp.tid()
    alpha = wp.float64(0.0)
    if S[4] > wp.float64(0.5) and S[1] != wp.float64(0.0):
        alpha = S[0] / S[1]
    p[i] = p[i] + alpha * q[i]
    ri = r[i] - alpha * Aq[i]
    r[i] = ri
    z[i] = Minv[i] * ri


@wp.kernel
def cg_direction_kernel(S: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.float64), q: wp.array(dtype=wp.float64)):
    """q = z + (rz2 / rz) q  (active only)"""
    i = wp.tid()
    if S[4] > wp.float64(0.5) and S[0] != wp.float64(0.0):
        q[i] = z[i] + (S[2] / S[0]) * q[i]


@wp.kernel
def cg_scalars_kernel(S: wp.array(dtype=wp.float64)):
    """one thread: count the iteration if it was active, deactivate on convergence / a non-finite value, rz <- rz2 (the dot products S[1..3] are written by reductions each iteration)"""
    S[7] = S[3]
    if S[4] > wp.float64(0.5):
        S[5] = S[5] + wp.float64(1.0)
        if S[3] <= S[6] or not wp.isfinite(S[3]) or not wp.isfinite(S[2]):
            S[4] = wp.float64(0.0)
        else:
            S[0] = S[2]
