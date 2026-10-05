"""FEM nodal weights (stage 1, exact/mpmath): P0..P3 Lagrange fields on a triangle against the SPH kernel.

Field on T:  A(x') = sum_i A_i N_i(x').   Re-expand every basis function about the evaluation point x,

    N_i(x + y) = sum_{|alpha| <= p} B_{alpha i}(x) y^alpha ,      B_{alpha i} = d^alpha N_i(x) / alpha!   (EXACT),

so that  <A>(x) = int_T A W dA = sum_i A_i w_i,   w_i = sum_alpha B_{alpha i} m_alpha,   m_alpha = int_T y^alpha W dA,
and   int_T A grad_x W dA = sum_i A_i G_i,   G_i = sum_alpha B_{alpha i} g_alpha,
      g_alpha = grad_x m_alpha + sum_j alpha_j e_j m_{alpha - e_j}      (grad_x acting on W only).

Basis polynomials are built EXACTLY (Fractions) in barycentric coordinates; lambda_j(x + y) = lambda_j(x) + grad(lambda_j).y;
for rational geometry B is exact rational.  Node ordering (p = 3 shown):
    vertices 0,1,2 ; then for each edge (0,1),(1,2),(2,0): the nodes at 1/3 and 2/3 from the first vertex (P2: the midpoint) ;
    then the centroid (P3 bubble node).  P0: a single constant function (centroid value).
"""
from fractions import Fraction

import mpmath as mp

from . import core, geometry
from .mpq import mpq, to_frac

# ----------------------------------------------------------------------- exact basis
# polynomial in barycentric coordinates: {(a, b, c): coeff}  meaning  l0^a l1^b l2^c


def _pm(P, Q):
    out = {}
    for (e1, c1) in P.items():
        for (e2, c2) in Q.items():
            e = tuple(x + y for x, y in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    return {e: c for e, c in out.items() if c != 0}


def _pa(P, Q, s=1):
    out = dict(P)
    for e, c in Q.items():
        out[e] = out.get(e, 0) + s * c
    return {e: c for e, c in out.items() if c != 0}


def _ps(P, s):
    return {e: c * s for e, c in P.items()}


L = [{(1, 0, 0): Fraction(1)}, {(0, 1, 0): Fraction(1)}, {(0, 0, 1): Fraction(1)}]
ONE = {(0, 0, 0): Fraction(1)}


def _lin(l, c):                      # l - c
    return _pa(l, ONE, -Fraction(c))


def basis(p: int):
    """(nodes, polys): barycentric node coordinates (Fractions) and basis polynomials N_i, exact."""
    if p == 0:
        return [(Fraction(1, 3),) * 3], [ONE]
    if p == 1:
        return [(Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))], [L[0], L[1], L[2]]
    nodes, polys = [], []
    V = [tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)]
    if p == 2:
        for i in range(3):
            nodes.append(V[i])
            polys.append(_pm(L[i], _lin(_ps(L[i], 2), 1)))                         # l_i (2 l_i - 1)
        for (i, j) in [(0, 1), (1, 2), (2, 0)]:
            nodes.append(tuple((V[i][k] + V[j][k]) / 2 for k in range(3)))
            polys.append(_ps(_pm(L[i], L[j]), 4))
        return nodes, polys
    if p == 3:
        for i in range(3):
            nodes.append(V[i])
            polys.append(_ps(_pm(_pm(L[i], _lin(_ps(L[i], 3), 1)), _lin(_ps(L[i], 3), 2)), Fraction(1, 2)))   # l_i(3l_i-1)(3l_i-2)/2
        for (i, j) in [(0, 1), (1, 2), (2, 0)]:
            # node at 1/3 from i towards j: l_i = 2/3, l_j = 1/3 ; basis (9/2) l_i l_j (3 l_i - 1)
            nodes.append(tuple(Fraction(2, 3) * V[i][k] + Fraction(1, 3) * V[j][k] for k in range(3)))
            polys.append(_ps(_pm(_pm(L[i], L[j]), _lin(_ps(L[i], 3), 1)), Fraction(9, 2)))
            nodes.append(tuple(Fraction(1, 3) * V[i][k] + Fraction(2, 3) * V[j][k] for k in range(3)))
            polys.append(_ps(_pm(_pm(L[i], L[j]), _lin(_ps(L[j], 3), 1)), Fraction(9, 2)))
        nodes.append((Fraction(1, 3),) * 3)
        polys.append(_ps(_pm(_pm(L[0], L[1]), L[2]), 27))
        return nodes, polys
    raise ValueError("p must be 0..3")


def eval_bary(P, lam):
    tot = 0
    for (a, b, c), co in P.items():
        tot += co * lam[0] ** a * lam[1] ** b * lam[2] ** c
    return tot


def node_positions(verts, p):
    nodes, _ = basis(p)
    V = [(to_frac(v[0]), to_frac(v[1])) for v in verts]
    return [(sum(n[k] * V[k][0] for k in range(3)), sum(n[k] * V[k][1] for k in range(3))) for n in nodes]


# ----------------------------------------------------------------------- expansion about x
def _ypoly_mul(P, Q):
    out = {}
    for (a1, b1), c1 in P.items():
        for (a2, b2), c2 in Q.items():
            k = (a1 + a2, b1 + b2)
            out[k] = out.get(k, 0) + c1 * c2
    return out


def lambdas_and_grads(verts):
    """exact affine barycentric functions lambda_j(x') = l0_j + g_j . x' of a triangle (any orientation)."""
    (x0, y0), (x1, y1), (x2, y2) = [(to_frac(v[0]), to_frac(v[1])) for v in verts]
    D = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if D == 0:
        raise ValueError("degenerate triangle")
    # lambda_0 = ((x1 y2 - x2 y1) + (y1 - y2) x + (x2 - x1) y) / D  etc. (cyclic)
    out = []
    for (xa, ya), (xb, yb), (xc, yc) in [((x1, y1), (x2, y2), (x0, y0)), ((x2, y2), (x0, y0), (x1, y1)), ((x0, y0), (x1, y1), (x2, y2))]:
        out.append(((xa * yb - xb * ya) / D, (ya - yb) / D, (xb - xa) / D))
    return out                                    # [(const, gx, gy)] for j = 0,1,2


def expansion_coefficients(verts, x, p):
    """EXACT B[i][alpha] (Fractions) for nodes i and |alpha| <= p: N_i(x + y) = sum B y^alpha."""
    lg = lambdas_and_grads(verts)
    xf = (to_frac(x[0]), to_frac(x[1]))
    # lambda_j(x + y) = lam_j(x) + gx y1 + gy y2   as a polynomial in y
    lam_y = [{(0, 0): c + gx * xf[0] + gy * xf[1], (1, 0): gx, (0, 1): gy} for (c, gx, gy) in lg]
    _, polys = basis(p)
    cache = {}

    def power(j, k):
        if (j, k) not in cache:
            cache[(j, k)] = {(0, 0): Fraction(1)} if k == 0 else _ypoly_mul(power(j, k - 1), lam_y[j])
        return cache[(j, k)]
    B = []
    for P in polys:
        acc = {}
        for (a, b, c), co in P.items():
            term = _ypoly_mul(_ypoly_mul(power(0, a), power(1, b)), power(2, c))
            for k, v in term.items():
                acc[k] = acc.get(k, 0) + co * v
        B.append({k: v for k, v in acc.items() if v != 0})
    return B


def alphas(p):
    return [(a, k - a) for k in range(p + 1) for a in range(k + 1)]


# ----------------------------------------------------------------------- weights
def weights(verts, x, kernel, p, h=1, dps=40, grad=False):
    """Nodal weights w_i (value) and, if grad, G_i (2-vectors): int_T A W = sum A_i w_i; int_T A grad_x W = sum A_i G_i."""
    with mp.workdps(dps + core.GUARD):
        prep = geometry.prepare(verts, x, h)
        B = expansion_coefficients(verts, x, p)
        m = {al: core.moment(verts, x, kernel, al, h, dps=dps, prepared=prep) for al in alphas(p)}
        w = [sum((mpq(c) * m[al] for al, c in Bi.items()), mp.mpf(0)) for Bi in B]
        if not grad:
            return w
        mg = {al: core.moment_gradient(verts, x, kernel, al, h, dps=dps, prepared=prep) for al in alphas(p)}
        # g_alpha = grad_x m_alpha + sum_j alpha_j e_j m_{alpha - e_j}
        g = {}
        for al in alphas(p):
            gx, gy = mg[al]
            if al[0]:
                gx = gx + al[0] * m[(al[0] - 1, al[1])]
            if al[1]:
                gy = gy + al[1] * m[(al[0], al[1] - 1)]
            g[al] = (gx, gy)
        G = [(sum((mpq(c) * g[al][0] for al, c in Bi.items()), mp.mpf(0)),
              sum((mpq(c) * g[al][1] for al, c in Bi.items()), mp.mpf(0))) for Bi in B]
        return w, G


def polynomial_field_nodal_values(verts, p, A):
    """nodal values of a polynomial field A(x, y) (callable on Fractions) at the nodes of the P_p element."""
    return [A(*pos) for pos in node_positions(verts, p)]


# ----------------------------------------------------------------------- polynomial fields (tests / projection)
def poly_eval(A, X, Y):
    """A: {(i, j): c} meaning sum c X^i Y^j"""
    return sum(c * X ** i * Y ** j for (i, j), c in A.items())


def poly_shift(A, x):
    """coefficients a_alpha of A(x + y) in y (exact)."""
    from math import comb
    xf = (to_frac(x[0]), to_frac(x[1]))
    out = {}
    for (i, j), c in A.items():
        for a in range(i + 1):
            for b in range(j + 1):
                k = (a, b)
                out[k] = out.get(k, 0) + c * comb(i, a) * comb(j, b) * xf[0] ** (i - a) * xf[1] ** (j - b)
    return {k: v for k, v in out.items() if v != 0}


def poly_grad(A):
    dx = {(i - 1, j): c * i for (i, j), c in A.items() if i}
    dy = {(i, j - 1): c * j for (i, j), c in A.items() if j}
    return dx, dy


def nodal_values(verts, p, A):
    return [poly_eval(A, *pos) for pos in node_positions(verts, p)]


def field_integral(verts, x, kernel, p, nodal, h=1, dps=40):
    """int_T A W dA for the P_p field with nodal values `nodal` (Fractions or mp numbers)."""
    w = weights(verts, x, kernel, p, h, dps)
    with mp.workdps(dps + core.GUARD):
        return sum((mpq(v) * wi for v, wi in zip(nodal, w)), mp.mpf(0))


def field_gradient_integral(verts, x, kernel, p, nodal, h=1, dps=40):
    """int_T A grad_x W dA (grad acting on W only) -> (gx, gy)."""
    w, G = weights(verts, x, kernel, p, h, dps, grad=True)
    with mp.workdps(dps + core.GUARD):
        return (sum((mpq(v) * g[0] for v, g in zip(nodal, G)), mp.mpf(0)),
                sum((mpq(v) * g[1] for v, g in zip(nodal, G)), mp.mpf(0)))
