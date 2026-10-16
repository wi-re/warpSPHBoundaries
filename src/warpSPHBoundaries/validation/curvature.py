"""Independent direct shape-Taylor quadrature for all corrected moments."""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq

from .atlas import ALPHAS
from .distance_only import halfplane_channels
from .sdf_accuracy import SmoothCase, gauss, profile_values, rotate_channels
from ..edge.kernels import kernel as get_kernel


def model_reference(distance, normal, curvature, variation=None, kernel='w4', order=128):
    """Integrate f*A + f1²*A_z/2 directly, without reduced B identities.

    The expansion counts k as order one and k_s as order two. This reference
    retains scalar second moments. It does not use production Hermite tables.
    """
    d, n, k = (np.asarray(x, dtype=np.longdouble) for x in (distance, normal, curvature))
    b = np.zeros_like(d) if variation is None else np.asarray(variation, dtype=np.longdouble)
    out = halfplane_channels(d, np.broadcast_to([0., 1.], n.shape), kernel, np.longdouble)
    nodes, weights = gauss(order)
    gradient_powers = ((1, 0), (0, 1), (2, 0), (1, 1), (1, 1), (0, 2))
    for i in range(len(d)):
        if abs(d[i]) >= 1:
            continue
        z = -d[i]
        cuts = [np.longdouble(0)]
        for block in get_kernel(kernel).blocks:
            if float(block.R) > abs(d[i]):
                cuts.append(np.sqrt(np.longdouble(float(block.R))**2-d[i]**2))
        cuts = sorted(set(cuts))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            positive = (lo+hi)/2+(hi-lo)/2*nodes
            t = np.r_[-positive, positive]
            weight = np.tile((hi-lo)/2*weights, 2)
            W, D, H = profile_values(np.sqrt(t*t+z*z), kernel)
            shift = -k[i]*t*t/2-b[i]*t**3/6
            second = k[i]**2*t**4/8
            for channel, (aa, bb) in enumerate((*ALPHAS, *gradient_powers)):
                profile, derivative = (W, -z*D) if channel < 6 else (D, z*H)
                polynomial = t**aa*z**bb
                derivative_polynomial = bb*t**aa*z**(bb-1) if bb else np.zeros_like(t)
                correction = shift*polynomial*profile+second*(derivative_polynomial*profile+polynomial*derivative)
                out[i, channel] += np.sum(weight*correction)
    t = np.stack([-n[:, 1], n[:, 0]], 1)
    for i in range(len(d)):
        out[i] = rotate_channels(out[i], np.stack([t[i], n[i]], 1))
    return out


def exact_variation(case, count):
    """Signed curvature derivative in the production tangent (-ny,nx)."""
    _, normal, tangent, _, variation = case.frame
    production_tangent = np.array([-normal[1], normal[0]])
    return np.full(count, variation*np.dot(tangent, production_tangent))


@dataclass(frozen=True)
class RippleWall(SmoothCase):
    """Small curvature, rapidly varying wall: a deliberate model counterexample.

    The distance uses bracketed global closest-point minimization in a region
    where the stationary-distance equation is strictly monotone.
    """
    name: str = 'rapid-ripple'
    kind: str = 'graph'
    amplitude: float = .001
    frequency: float = 20.

    def height(self, x):
        return self.amplitude*np.sin(self.frequency*x)

    @property
    def frame(self):
        slope = self.amplitude*self.frequency
        scale = np.sqrt(1+slope*slope)
        normal = np.array([-slope, 1.])/scale
        tangent = np.array([1., slope])/scale
        variation = self.amplitude*self.frequency**3/(1+slope*slope)**2
        return np.zeros(2), normal, tangent, 0., variation

    def distance(self, points):
        p = np.asarray(points)@self.matrix
        bound = (np.abs(p[:, 1])+self.amplitude)*self.amplitude*self.frequency
        derivative_lower = 1-(np.abs(p[:, 1])+self.amplitude)*self.amplitude*self.frequency**2
        if not np.all(derivative_lower > 0):
            raise ValueError('ripple fixture distance requires the proven unique-closest-point region')
        lo, hi = p[:, 0]-bound, p[:, 0]+bound
        for _ in range(60):
            x = (lo+hi)/2
            residual = x-p[:, 0]+(self.height(x)-p[:, 1])*self.amplitude*self.frequency*np.cos(self.frequency*x)
            lo, hi = np.where(residual < 0, x, lo), np.where(residual < 0, hi, x)
        x = (lo+hi)/2
        distance = np.sqrt((x-p[:, 0])**2+(self.height(x)-p[:, 1])**2)
        return np.where(p[:, 1] >= self.height(p[:, 0]), distance, -distance)

    def breaks(self, point, radii):
        px, py = point
        cuts = [px-1, px, px+1]
        mesh = np.linspace(px-1, px+1, 4097)
        for r in radii:
            cuts += [px-r, px+r]
            residual = lambda x: (x-px)**2+(self.height(x)-py)**2-r*r
            value = residual(mesh)
            cuts += [brentq(residual, mesh[j], mesh[j+1], xtol=5e-15)
                     for j in np.flatnonzero(value[:-1]*value[1:] < 0)]
        return np.unique(np.clip(cuts, px-1, px+1))
