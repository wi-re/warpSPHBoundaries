"""Regular face-to-edge reductions. See docs/derivations/tet-face-edge-chain.md.

All integration uses elementary antiderivatives, never numerical quadrature.
mpmath precision is caller controlled (``with mp.workdps(50): ...``). Geometry
predicates are rational before normalization, including coplanar/collinear
zeros. This is a research reference, not a conditioned floating-point backend.
"""
from functools import lru_cache
from math import comb
from numbers import Integral

import mpmath as mp

from ..curvbound.kernels import KERNELS as SHAPES
from ..edge.kernels import KERNELS as BLOCKS
from ..edge.mpq import mpq, to_frac
from ..edge.primitives import Ig, Sg


ZERO = (0, 0, 0)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _vector(v):
    if len(v) != 3:
        raise ValueError("expected three coordinates")
    return tuple(to_frac(t) for t in v)


def _alpha(alpha):
    alpha = tuple(alpha)
    if len(alpha) != 3 or any(not isinstance(i, Integral) or i < 0 for i in alpha):
        raise ValueError("alpha must be three nonnegative integers")
    return tuple(int(i) for i in alpha)


def _lower(alpha, i, amount=1):
    out = list(alpha)
    out[i] -= amount
    return tuple(out)


def kernel_profile(name, h=1):
    """Physical W_h as tuples (power, cutoff, coefficient), with C3 normalization."""
    h = mpq(to_frac(h))
    if h <= 0:
        raise ValueError("support radius must be positive")
    shape, blocks = SHAPES[name], BLOCKS[name]
    ratio = mpq(shape.c3_pi / shape.c2_pi) / mp.pi
    return tuple((n, mpq(block.R)*h, ratio*mpq(c)/h**(n+3))
                 for block in blocks.blocks for n, c in enumerate(block.coeffs) if c)


def _poly_mul(p, q):
    out = {}
    for a, c in p.items():
        for b, d in q.items():
            k = tuple(x+y for x, y in zip(a, b))
            out[k] = out.get(k, 0) + c*d
    return out


def _K(m, s, w, a):
    """Integral r^m/(s^2+w^2) ds, r^2=s^2+w^2+a^2; w != 0."""
    b = abs(w)
    c = mp.sqrt(w*w+a*a)
    if m == 0:
        return mp.atan(s/b)/b
    if m == -1:
        r = mp.sqrt(s*s+c*c)
        return mp.atan(a*s/(b*r))/(a*b) if a else s/(b*b*r)
    if m == 1:
        return mp.asinh(s/c) + a*a*_K(-1, s, w, a)
    if m >= 2:
        return Ig(m-2, s, c) + a*a*_K(m-2, s, w, a)
    raise ValueError("K supports m >= -1")


class Triangle:
    """An area element with oriented unit normal; scalar area is orientation independent."""

    def __init__(self, vertices, query=(0, 0, 0)):
        if len(vertices) != 3:
            raise ValueError("triangle needs three vertices")
        self.vertices = tuple(_vector(v) for v in vertices)
        self.query = _vector(query)
        p = tuple(sub(v, self.query) for v in self.vertices)
        d = sub(p[1], p[0])
        N = cross(d, sub(p[2], p[0]))
        nlen = mp.sqrt(mpq(dot(N, N)))
        if not nlen:
            raise ValueError("degenerate triangle")
        ell = mp.sqrt(mpq(dot(d, d)))
        self.normal = tuple(mpq(t)/nlen for t in N)
        self.z = mpq(dot(N, p[0]))/nlen
        self.a = abs(self.z)
        self.e1 = tuple(mpq(t)/ell for t in d)
        self.e2 = cross(self.normal, self.e1)
        self.edges = []
        for j in range(3):
            v, q = p[j], p[(j+1) % 3]
            D = sub(q, v)
            length = mp.sqrt(mpq(dot(D, D)))
            t = tuple(mpq(x)/length for x in D)
            mu = cross(t, self.normal)
            # Preserve exact zeros before either square root is introduced.
            w = mpq(dot(cross(D, N), v))/(length*nlen)
            lo, hi = mpq(dot(D, v))/length, mpq(dot(D, q))/length
            self.edges.append((w, lo, hi, dot(mu, self.e1), dot(mu, self.e2),
                               dot(t, self.e1), dot(t, self.e2)))
        # Instance caches do not retain every historical element in global state.
        for method in ("_scalar", "_local", "block_moment"):
            setattr(self, method, lru_cache(maxsize=None)(getattr(self, method)))

    def _scalar(self, n, R):
        """Integral over face of r^n 1[r <= R]; R=None gives full face."""
        a = self.a
        if R is not None and a >= R:
            return mp.mpf(0)
        if n == -2 or n < -3 or (n == -3 and not a):
            raise ValueError("unsupported singular face profile")
        total = mp.mpf(0)
        for w, lo, hi, *_ in self.edges:
            if not w:
                continue
            cuts = [lo, hi]
            if R is not None and R*R > a*a+w*w:
                L = mp.sqrt(R*R-a*a-w*w)
                cuts += [s for s in (-L, L) if lo < s < hi]
            cuts.sort()
            for l, u in zip(cuts, cuts[1:]):
                inside = R is None or ((l+u)/2)**2+w*w+a*a < R*R
                dK0 = _K(0, u, w, a)-_K(0, l, w, a)
                if inside:
                    term = _K(n+2, u, w, a)-_K(n+2, l, w, a)-a**(n+2)*dK0
                else:
                    term = (R**(n+2)-a**(n+2))*dK0
                total += w*term/(n+2)
        return total

    def _line(self, edge, beta, n, R):
        w, lo, hi, mu1, mu2, t1, t2 = edge
        c2 = w*w+self.a*self.a
        if c2 >= R*R:
            return mp.mpf(0)
        L = mp.sqrt(R*R-c2)
        lo, hi = max(lo, -L), min(hi, L)
        if lo >= hi:
            return mp.mpf(0)
        poly = [mp.mpf(1)]
        for b, mu, t in zip(beta, (mu1, mu2), (t1, t2)):
            q = [comb(b, j)*(w*mu)**(b-j)*t**j for j in range(b+1)]
            new = [mp.mpf(0)]*(len(poly)+b)
            for i, v in enumerate(poly):
                for j, v2 in enumerate(q):
                    new[i+j] += v*v2
            poly = new
        c = mp.sqrt(c2)
        return sum(v*(Sg(j, n, hi, c)-Sg(j, n, lo, c))
                   for j, v in enumerate(poly) if v)

    def _local(self, beta, n, R):
        if not any(beta):
            return self._scalar(n, R)
        i = 0 if beta[0] else 1
        b = list(beta)
        b[i] -= 1
        b = tuple(b)
        edge = sum(e[3+i]*(self._line(e, b, n+2, R)-R**(n+2)*self._line(e, b, 0, R))
                   for e in self.edges)/(n+2)
        if b[i]:
            lower = list(b)
            lower[i] -= 1
            lower = tuple(lower)
            edge -= b[i]*(self._local(lower, n+2, R)-R**(n+2)*self._local(lower, 0, R))/(n+2)
        return edge

    def block_moment(self, alpha, n, R):
        """Surface integral y^alpha r^n 1[r<=R], integer n >= -1."""
        alpha = _alpha(alpha)
        if not isinstance(n, Integral) or n < -1 or R <= 0:
            raise ValueError("surface blocks require integer n >= -1 and R > 0")
        poly = {(0, 0): mp.mpf(1)}
        for i, power in enumerate(alpha):
            linear = {(0, 0): self.z*self.normal[i], (1, 0): self.e1[i], (0, 1): self.e2[i]}
            for _ in range(power):
                poly = _poly_mul(poly, linear)
        return sum(c*self._local(b, n, R) for b, c in poly.items() if c)

    def integrate(self, profile, alpha=ZERO):
        alpha = _alpha(alpha)
        return sum((c*self.block_moment(alpha, n, R) for n, R, c in profile), mp.mpf(0))

    def moment(self, name="w2", alpha=ZERO, h=1):
        """Area integral of y^alpha W_h (scales as h^(|alpha|-1))."""
        return self.integrate(kernel_profile(name, h), alpha)

    def kernel_gradient(self, name="w2", alpha=ZERO, h=1):
        """Area integral y^alpha grad_query W_h (kernel derivative only)."""
        alpha = _alpha(alpha)
        L = tuple((n-2, R, c*n) for n, R, c in kernel_profile(name, h) if n)
        return tuple(-self.integrate(L, tuple(a+int(i == j) for i, a in enumerate(alpha))) for j in range(3))

    def nodal_weights(self, name="w2", degree=1, h=1, gradient=False):
        """P0-P3 area FEM weights; see mesh3d.fem for node ordering."""
        from .fem import weights
        return weights(self, name, degree, h, gradient)


class ClosedSurface:
    """Bounded oriented solid. Faces must form a watertight oriented 2-manifold.

    Each edge is checked twice with opposite directions. Positive total signed
    volume is required; nested cavity components must point out of the solid.
    Geometric self intersections/overlap are not detected.
    """

    def __init__(self, vertices, faces, query=(0, 0, 0)):
        self.vertices = tuple(_vector(v) for v in vertices)
        self.query = _vector(query)
        self.indices = tuple(tuple(f) for f in faces)
        if not self.indices:
            raise ValueError("closed surface needs faces")
        edges = {}
        for f in self.indices:
            if len(f) != 3 or len(set(f)) != 3 or any(not isinstance(i, Integral) or i < 0 or i >= len(self.vertices) for i in f):
                raise ValueError("invalid triangle indices")
            for a, b in zip(f, f[1:]+f[:1]):
                key = (min(a, b), max(a, b))
                edges.setdefault(key, []).append(1 if a < b else -1)
        if any(len(e) != 2 or sum(e) != 0 for e in edges.values()):
            raise ValueError("surface must be watertight and consistently oriented")
        # Translate before evaluating the exact volume (also robust at large offsets).
        rel = [sub(v, self.vertices[0]) for v in self.vertices]
        signed6 = sum(dot(rel[a], cross(rel[b], rel[c])) for a, b, c in self.indices)
        if signed6 <= 0:
            raise ValueError("surface must have positive signed volume")
        self.faces = tuple(Triangle([self.vertices[i] for i in f], query) for f in self.indices)
        self.block_moment = lru_cache(maxsize=None)(self.block_moment)

    def block_moment(self, alpha, n, R):
        alpha = _alpha(alpha)
        if not isinstance(n, Integral) or n < 0 or R <= 0:
            raise ValueError("volume blocks require integer n >= 0 and R > 0")
        if not any(alpha):
            # Regular radial flux: no winding indicator or boundary convention.
            return sum(f.z*(f._scalar(n, R)+R**(n+3)*(f._scalar(-3, None)-f._scalar(-3, R)))/(n+3)
                       for f in self.faces if f.z)
        i = next(j for j, a in enumerate(alpha) if a)
        beta = _lower(alpha, i)
        result = sum(f.normal[i]*(f.block_moment(beta, n+2, R)-R**(n+2)*f.block_moment(beta, 0, R))
                     for f in self.faces)/(n+2)
        if beta[i]:
            lower = _lower(beta, i)
            result -= beta[i]*(self.block_moment(lower, n+2, R)-R**(n+2)*self.block_moment(lower, 0, R))/(n+2)
        return result

    def integrate(self, profile, alpha=ZERO):
        alpha = _alpha(alpha)
        return sum((c*self.block_moment(alpha, n, R) for n, R, c in profile), mp.mpf(0))

    def moment(self, name="w2", alpha=ZERO, h=1):
        return self.integrate(kernel_profile(name, h), alpha)

    def moment_gradient(self, name="w2", alpha=ZERO, h=1):
        """Derivative of the complete translated moment, including y^alpha."""
        profile = kernel_profile(name, h)
        alpha = _alpha(alpha)
        return tuple(-sum(f.normal[j]*f.integrate(profile, alpha) for f in self.faces) for j in range(3))

    def kernel_gradient(self, name="w2", alpha=ZERO, h=1):
        """Volume integral y^alpha grad_query W_h; differs from moment_gradient."""
        alpha = _alpha(alpha)
        g = self.moment_gradient(name, alpha, h)
        return tuple(g[j]+(alpha[j]*self.moment(name, _lower(alpha, j), h) if alpha[j] else 0) for j in range(3))

    def laplacian(self, name="w2", alpha=ZERO, h=1):
        """Volume integral y^alpha Delta W_h via Green's second identity."""
        alpha = _alpha(alpha)
        W = kernel_profile(name, h)
        L = tuple((n-2, R, c*n) for n, R, c in W if n)
        value = sum(f.z*f.integrate(L, alpha) for f in self.faces)
        for j, a in enumerate(alpha):
            if a:
                lower = _lower(alpha, j)
                value -= a*sum(f.normal[j]*f.integrate(W, lower) for f in self.faces)
            if a >= 2:
                value += a*(a-1)*self.integrate(W, _lower(alpha, j, 2))
        return value

    def kernel_hessian(self, name="w2", alpha=ZERO, h=1):
        """Volume integral y^alpha Hessian_query W_h, including mixed entries."""
        alpha = _alpha(alpha)
        L = tuple((n-2, R, c*n) for n, R, c in kernel_profile(name, h) if n)
        out = []
        for i in range(3):
            lower_gradient = self.kernel_gradient(name, _lower(alpha, i), h) if alpha[i] else ZERO
            out.append(tuple(sum(f.normal[i]*f.integrate(L, tuple(a+int(k == j) for k, a in enumerate(alpha)))
                                 for f in self.faces) + alpha[i]*lower_gradient[j] for j in range(3)))
        return tuple(out)


class Tetrahedron(ClosedSurface):
    """Single tetrahedral volume; either input orientation is accepted."""

    def __init__(self, vertices, query=(0, 0, 0)):
        if len(vertices) != 4:
            raise ValueError("tetrahedron needs four vertices")
        vertices = tuple(_vector(v) for v in vertices)
        faces = []
        for i in range(4):
            f = [j for j in range(4) if j != i]
            a, b, c = (vertices[j] for j in f)
            side = dot(cross(sub(b, a), sub(c, a)), sub(vertices[i], a))
            if not side:
                raise ValueError("degenerate tetrahedron")
            if side > 0:
                f[1], f[2] = f[2], f[1]
            faces.append(tuple(f))
        super().__init__(vertices, faces, query)

    def nodal_weights(self, name="w2", degree=1, h=1, gradient=False):
        """P0-P3 volume FEM weights; see mesh3d.fem for node ordering."""
        from .fem import weights
        return weights(self, name, degree, h, gradient)


class TetrahedralMesh:
    """Additive tetrahedral volume reference with explicit cell connectivity.

    Cells must have disjoint interiors. Shared faces must use the same vertex
    indices. Boundary extraction checks face incidence and orientation; it does
    not detect geometric overlaps. FEM weights retain each cell's local DOFs.
    """

    def __init__(self, vertices, cells, query=(0, 0, 0)):
        self.vertices = tuple(_vector(v) for v in vertices)
        self.query = _vector(query)
        self.cells = tuple(tuple(c) for c in cells)
        if not self.cells:
            raise ValueError("volume mesh needs tetrahedra")
        if any(len(c) != 4 or len(set(c)) != 4 or any(not isinstance(i, Integral) or i < 0 or i >= len(self.vertices) for i in c) for c in self.cells):
            raise ValueError("invalid tetrahedron indices")
        if len(set(tuple(sorted(c)) for c in self.cells)) != len(self.cells):
            raise ValueError("duplicate tetrahedra")
        self.elements = tuple(Tetrahedron([self.vertices[i] for i in c], query) for c in self.cells)
        face_map = {}
        for cell, tet in zip(self.cells, self.elements):
            for f in tet.indices:
                ids = tuple(cell[i] for i in f)
                face_map.setdefault(tuple(sorted(ids)), []).append(ids)
        boundary_faces = []
        for fs in face_map.values():
            if len(fs) == 1:
                boundary_faces += fs
            elif len(fs) == 2:
                a, b = fs
                if any(a == b[k:]+b[:k] for k in range(3)):
                    raise ValueError("shared tetrahedral face has inconsistent solid orientation")
            else:
                raise ValueError("nonmanifold tetrahedral face")
        self._boundary = ClosedSurface(self.vertices, boundary_faces, query)

    def boundary(self):
        """Equivalent boundary-only solid with the interior triangles removed."""
        return self._boundary

    def block_moment(self, alpha, n, R):
        return sum(t.block_moment(alpha, n, R) for t in self.elements)

    def moment(self, name="w2", alpha=ZERO, h=1):
        return sum(t.moment(name, alpha, h) for t in self.elements)

    def moment_gradient(self, name="w2", alpha=ZERO, h=1):
        gs = [t.moment_gradient(name, alpha, h) for t in self.elements]
        return tuple(sum(g[j] for g in gs) for j in range(3))

    def kernel_gradient(self, name="w2", alpha=ZERO, h=1):
        gs = [t.kernel_gradient(name, alpha, h) for t in self.elements]
        return tuple(sum(g[j] for g in gs) for j in range(3))

    def laplacian(self, name="w2", alpha=ZERO, h=1):
        return sum(t.laplacian(name, alpha, h) for t in self.elements)

    def kernel_hessian(self, name="w2", alpha=ZERO, h=1):
        hs = [t.kernel_hessian(name, alpha, h) for t in self.elements]
        return tuple(tuple(sum(H[i][j] for H in hs) for j in range(3)) for i in range(3))

    def nodal_weights(self, name="w2", degree=1, h=1, gradient=False):
        """Per-cell weights in local node order; global DOF assembly is caller owned."""
        return [t.nodal_weights(name, degree, h, gradient) for t in self.elements]
