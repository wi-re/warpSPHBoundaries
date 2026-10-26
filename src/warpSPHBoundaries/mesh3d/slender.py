"""Small 3D ball expansion of polynomial kernel moments.

Integrate F over a ball of radius a centred at c:
V_a [F(c) + a^2 Delta F(c)/10 + a^4 Delta^2 F(c)/280 + ...].
The kernel piece must be smooth on the entire ball; crossing a support/knot or
including the radial origin is rejected rather than silently extrapolated.
"""
from math import pi

import numpy as np

from ..curvbound.kernels import KERNELS
from ..edge.kernels import KERNELS as BLOCKS


def integrand_terms(name, field, q):
    """(alpha,power,coefficient) for y^alpha r^power on one kernel piece."""
    shape = KERNELS[name]
    piece = next((c for lo, hi, c in BLOCKS[name].pieces if float(lo) <= q < float(hi)), ())
    coeffs = [float(c*shape.c3_pi/shape.c2_pi)/pi for c in piece]
    zero = (0, 0, 0)
    terms = []
    def add(alpha, power, coefficient):
        if coefficient:
            terms.append((alpha, power, coefficient))
    for p, c in enumerate(coeffs):
        if field == 'm00':
            add(zero, p, c)
        elif field.startswith('m'):
            add(tuple(int(x) for x in field[1:]), p, c)
        elif field.startswith('G'):
            i = 'xyz'.index(field[1])
            add(tuple(int(j == i) for j in range(3)), p-2, -p*c)
        elif field.startswith('g1'):
            i, j = ['xyz'.index(x) for x in field[-2:]]
            add(tuple(int(k == i)+int(k == j) for k in range(3)), p-2, -p*c)
        elif field == 'laplacian':
            add(zero, p-2, p*(p+1)*c)
        elif field.startswith('H'):
            i, j = ['xyz'.index(x) for x in field[-2:]]
            if i == j:
                add(zero, p-2, p*c)
            add(tuple(int(k == i)+int(k == j) for k in range(3)), p-4, p*(p-2)*c)
        else:
            raise ValueError('Unknown moment channel')
    return terms


def laplacian_terms(terms):
    out = []
    for alpha, p, c in terms:
        radial = p*(p+1+2*sum(alpha))*c
        if radial:
            out.append((alpha, p-2, radial))
        for j, a in enumerate(alpha):
            if a >= 2:
                lower = tuple(v-2*int(i == j) for i, v in enumerate(alpha))
                out.append((lower, p, a*(a-1)*c))
    return out


def ball_series(point, radius, name='w4', field='m00', order=4):
    """Normalized unit-support moment about query point; sphere centre is zero.

    Returns NaN outside the smooth-piece validity domain. The radius cutoff
    a<=0.3 is a conservative exploration setting, not an accuracy certificate.
    """
    if order not in (0, 2, 4) or radius <= 0:
        raise ValueError('positive radius and order 0, 2 or 4 required')
    point = np.asarray(point, float)
    q = float(np.linalg.norm(point))
    if q-radius >= 1:
        return 0.
    knots = [0., .5, 1.] if name == 'cubic' else [0., 1.]
    if radius > .3 or q <= radius or any(q-radius <= k <= q+radius for k in knots):
        return float('nan')
    if field == 'grad_norm':
        return float(np.linalg.norm([ball_series(point, radius, name, f'G{a}', order) for a in 'xyz']))
    y = -point
    def evaluate(terms):
        return sum(c*np.prod(y**np.array(alpha))*q**p for alpha, p, c in terms)
    terms = integrand_terms(name, field, q)
    value = evaluate(terms)
    if order >= 2:
        terms = laplacian_terms(terms)
        value += radius**2/10*evaluate(terms)
    if order >= 4:
        terms = laplacian_terms(terms)
        value += radius**4/280*evaluate(terms)
    return 4*pi*radius**3/3*value
