"""Shared 3D scalar channel metadata and volume-integrand selection."""
import numpy as np

from ..mesh3d.radial import radial_jet

FIELDS3D = {'m00': ('λ · kernel mass', 0, '∫Ω Wₕ(y) dV')}
for a in ((1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)):
    key = 'm'+''.join(str(i) for i in a)
    label = ''.join('xyz'[i]*n for i,n in enumerate(a))
    FIELDS3D[key] = (f'M{label} / h{sum(a) if sum(a)>1 else ""}', sum(a), f'∫Ω y^{a} Wₕ(y) dV / h^{sum(a)}')
for axis in 'xyz':
    FIELDS3D[f'G{axis}'] = (f'h G{axis} · kernel gradient', -1, f'h ∫Ω ∂query,{axis} Wₕ dV')
for a in 'xyz':
    for b in 'xyz':
        FIELDS3D[f'g1{a}{b}'] = (f'g₁ {a}{b} · first gradient moment', 0, f'∫Ω y{a} ∂query,{b} Wₕ dV')
for a,b in ('xx','xy','xz','yy','yz','zz'):
    FIELDS3D[f'H{a}{b}'] = (f'h² H{a}{b} · kernel Hessian', -2, f'h² ∫Ω ∂query,{a}{b} Wₕ dV')
FIELDS3D['laplacian'] = ('h² Δλ · wall Laplacian', -2, 'h² ∫Ω ΔWₕ dV = h² tr H')
FIELDS3D['grad_norm'] = ('|h G| · kernel gradient magnitude', -1, 'h |∫Ω ∇query Wₕ dV|')


def integrand(y, name, field):
    """Dimensionless integrand at source-query offsets y (shape ...,3)."""
    y = np.asarray(y, float)
    q = np.linalg.norm(y, axis=-1)
    w, L, Q, _ = radial_jet(q, name)
    if field == 'm00':
        return w
    if field.startswith('m'):
        alpha = np.array([int(a) for a in field[1:]])
        return np.prod(y**alpha, axis=-1)*w
    if field.startswith('G'):
        return -y[..., 'xyz'.index(field[1])]*L
    if field.startswith('g1'):
        i, j = ['xyz'.index(a) for a in field[-2:]]
        return -y[..., i]*y[..., j]*L
    if field == 'laplacian':
        return 3*L+q*q*Q
    if field.startswith('H'):
        i, j = ['xyz'.index(a) for a in field[-2:]]
        return (L if i == j else 0)+y[..., i]*y[..., j]*Q
    raise ValueError('Select gradient components before taking the vector norm')
