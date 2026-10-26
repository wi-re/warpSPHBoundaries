"""Exact P0-P3 Lagrange expansions on tetrahedra and embedded triangles.

Nodes are barycentric lattice tuples in descending lexicographic order.
For p>0 each tuple b sums to p, node lambda=b/p, with basis
N_b = product_i product_(k=0)^(b_i-1) (p lambda_i-k)/b_i!.
P0 has one centroid node. Coefficients are exact Fractions even for float
input (the represented binary geometry), before contraction with mp moments.
"""
from fractions import Fraction

import mpmath as mp

from ..edge.mpq import mpq, to_frac
from .reference import ZERO, _poly_mul, dot, sub


def _inverse(matrix):
    size = len(matrix)
    a = [list(row)+[Fraction(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for j in range(size):
        pivot = next((i for i in range(j, size) if a[i][j]), None)
        if pivot is None:
            raise ValueError("degenerate simplex")
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [x/scale for x in a[j]]
        for i in range(size):
            if i != j:
                factor = a[i][j]
                a[i] = [x-factor*y for x, y in zip(a[i], a[j])]
    return [row[size:] for row in a]


def _lattice(total, size):
    if size == 1:
        return [(total,)]
    return [(j,)+b for j in range(total, -1, -1) for b in _lattice(total-j, size-1)]


def expansion(vertices, query, degree):
    """Return physical nodes and exact dictionaries B for N_i(query+y)."""
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be 0..3")
    vertices = [tuple(to_frac(t) for t in v) for v in vertices]
    query = tuple(to_frac(t) for t in query)
    count = len(vertices)
    if count not in (3, 4):
        raise ValueError("expected a triangle or tetrahedron")
    D = [sub(v, vertices[0]) for v in vertices[1:]]
    inv = _inverse([[dot(a, b) for b in D] for a in D])
    grads = [tuple(sum(inv[i][j]*D[j][k] for j in range(count-1)) for k in range(3)) for i in range(count-1)]
    grads.insert(0, tuple(-sum(g[k] for g in grads) for k in range(3)))
    lam = []
    for i, g in enumerate(grads):
        linear = {ZERO: Fraction(i == 0)+dot(g, sub(query, vertices[0]))}
        linear.update({tuple(int(j == k) for j in range(3)): g[k] for k in range(3)})
        lam.append(linear)
    lattice = _lattice(degree, count) if degree else [(0,)*count]
    nodes, coeffs = [], []
    for b in lattice:
        bary = [Fraction(i, degree) for i in b] if degree else [Fraction(1, count)]*count
        nodes.append(tuple(sum(t*v[k] for t, v in zip(bary, vertices)) for k in range(3)))
        poly = {ZERO: Fraction(1)}
        for i, bi in enumerate(b):
            for k in range(bi):
                factor = {a: degree*c/Fraction(k+1) for a, c in lam[i].items()}
                factor[ZERO] -= Fraction(k, k+1)
                poly = _poly_mul(poly, factor)
        coeffs.append({a: c for a, c in poly.items() if c})
    return nodes, coeffs


def weights(element, name="w2", degree=1, h=1, gradient=False):
    """Weights (and optional kernel-gradient weights) in expansion node order."""
    _, coeffs = expansion(element.vertices, element.query, degree)
    alphas = set(a for b in coeffs for a in b)
    moments = {a: element.moment(name, a, h) for a in alphas}
    values = [sum((mpq(c)*moments[a] for a, c in b.items()), mp.mpf(0)) for b in coeffs]
    if not gradient:
        return values
    gradients = {a: element.kernel_gradient(name, a, h) for a in alphas}
    return values, [tuple(sum((mpq(c)*gradients[a][j] for a, c in b.items()), mp.mpf(0))
                          for j in range(3)) for b in coeffs]
