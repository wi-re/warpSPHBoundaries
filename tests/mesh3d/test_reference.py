"""Independent checks for the analytic 3D reference; no production accuracy claim."""
from fractions import Fraction as F
from itertools import product
from math import comb, factorial, prod

import mpmath as mp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from warpSPHBoundaries.curvbound import planar3d
from warpSPHBoundaries.curvbound.kernels import KERNELS
from warpSPHBoundaries.edge.mpq import mpq
from warpSPHBoundaries.mesh3d import ClosedSurface, TetrahedralMesh, Tetrahedron, Triangle, kernel_profile
from warpSPHBoundaries.mesh3d.fem import expansion
from warpSPHBoundaries.mesh3d.oracle import constant_overlap
from warpSPHBoundaries.mesh3d.reference import _K


ALL = ("cubic", "w2", "w4", "w6")
UNIT = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBLIQUE = [(F(1, 10), F(1, 5), F(1, 10)), (F(4, 5), 0, F(1, 10)),
           (F(1, 10), F(7, 10), 0), (0, F(1, 5), F(4, 5))]


@pytest.fixture(autouse=True)
def precision():
    with mp.workdps(45):
        yield


def close(a, b, tol="1e-35"):
    assert abs(a-b) < mp.mpf(tol), (a, b, a-b)


@pytest.mark.parametrize("a,w", [(mp.mpf(0), mp.mpf("0.4")), (mp.mpf("0.3"), mp.mpf("-0.2"))])
def test_new_edge_primitives(a, w):
    for m in range(-1, 17):
        for s in (mp.mpf("-0.7"), mp.mpf(0), mp.mpf("0.3")):
            got = mp.diff(lambda t: _K(m, t, w, a), s)
            close(got, (s*s+w*w+a*a)**(mp.mpf(m)/2)/(s*s+w*w))


@pytest.mark.parametrize("name", ALL)
def test_3d_normalization_and_whole_support(name):
    profile = kernel_profile(name)
    close(sum(4*mp.pi*c*R**(n+3)/(n+3) for n, R, c in profile), 1)
    # All four faces lie farther than the unit support from the query.
    verts = [(-6, -6, -6), (18, -6, -6), (-6, 18, -6), (-6, -6, 18)]
    t = Tetrahedron(verts)
    close(t.moment(name), 1)
    for alpha in ((1, 0, 0), (0, 1, 1), (3, 0, 0)):
        close(t.moment(name, alpha), 0)
    second = sum(4*mp.pi*c*R**(n+5)/(3*(n+5)) for n, R, c in profile)
    close(t.moment(name, (2, 0, 0)), second)
    for row in t.kernel_hessian(name):
        for a in row:
            close(a, 0)
    close(t.laplacian(name), 0)
    for j, a in enumerate(t.kernel_gradient(name, (1, 0, 0))):
        close(a, int(j == 0))


def test_tetra_polynomial_moments_exact():
    t = Tetrahedron(UNIT)
    # For constant radial profile: direct barycentric simplex factorial formula.
    for alpha in product(range(5), repeat=3):
        if sum(alpha) > 4:
            continue
        exact = F(prod(factorial(i) for i in alpha), factorial(sum(alpha)+3))
        close(t.block_moment(alpha, 0, mp.mpf(2)), mpq(exact))
    # Even radial powers expand directly into Cartesian monomials, independently.
    for n in (2, 4, 6):
        power = n//2
        exact = F(0)
        for a in range(power+1):
            for b in range(power-a+1):
                c = power-a-b
                exact += F(factorial(power), factorial(a)*factorial(b)*factorial(c))*F(factorial(2*a)*factorial(2*b)*factorial(2*c), factorial(n+3))
        close(t.block_moment((0, 0, 0), n, mp.mpf(2)), mpq(exact))


@pytest.mark.parametrize("name", ALL)
def test_boundary_queries_and_planar_limit(name):
    # Only the face z=0 intersects support, so this is exactly a half-space test.
    verts = [(-10, -10, 0), (10, -10, 0), (0, 10, 0), (0, 0, -10)]
    for d in (F(-1, 4), F(0), F(1, 4), F(1, 2), F(9, 10), F(1), F(11, 10)):
        t = Tetrahedron(verts, (0, 0, d))
        expected = 0 if d >= 1 else (1-planar3d(name, -d) if d < 0 else planar3d(name, d))
        close(t.moment(name), mpq(F(expected)), "1e-32")
    # Vertex, edge, face: support sees respectively octant, quarter-, half-space.
    for q, fraction in [((0, 0, 0), F(1, 8)), ((F(1, 3), 0, 0), F(1, 4)),
                        ((F(1, 3), F(1, 3), 0), F(1, 2))]:
        close(Tetrahedron(UNIT, q).moment(name, h=F(1, 10)), mpq(fraction))
    # Tangency and empty support.
    close(Tetrahedron(UNIT, (-1, 0, 0)).moment(name), 0)
    close(Tetrahedron(UNIT, (-2, 0, 0)).moment(name), 0)


def direct_rule(vertices, query, name, h, order, alpha=(0, 0, 0)):
    """Independent tensor Gauss/Duffy rule on the uncut simplex (sample true W)."""
    v = np.array(vertices, dtype=float)
    query = np.array(query, dtype=float)
    dim = len(v)-1
    nodes, weights = leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    if dim == 3:
        u, v1, w = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        wu, wv, ww = np.meshgrid(weights, weights, weights, indexing="ij")
        bary = np.stack([u, (1-u)*v1, (1-u)*(1-v1)*w], axis=-1)
        jac = abs(np.linalg.det((v[1:]-v[0]).T))*(1-u)**2*(1-v1)*wu*wv*ww
    else:
        u, v1 = np.meshgrid(nodes, nodes, indexing="ij")
        wu, wv = np.meshgrid(weights, weights, indexing="ij")
        bary = np.stack([u, (1-u)*v1], axis=-1)
        jac = np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0]))*(1-u)*wu*wv
    y = v[0]+bary@(v[1:]-v[0])-query
    q = np.linalg.norm(y, axis=-1)/h
    # Evaluate factored shapes, without importing the analytic block compiler.
    if name == "cubic":
        shape = np.maximum(1-q, 0)**3-4*np.maximum(.5-q, 0)**3
        derivative = -3*np.maximum(1-q, 0)**2+12*np.maximum(.5-q, 0)**2
        second = 6*np.maximum(1-q, 0)-24*np.maximum(.5-q, 0)
    else:
        k = KERNELS[name]
        P = sum(float(c)*q**p for p, c in k.poly)
        P1 = sum(float(c)*p*q**(p-1) for p, c in k.poly if p)
        P2 = sum(float(c)*p*(p-1)*q**(p-2) for p, c in k.poly if p > 1)
        t = np.maximum(1-q, 0)
        shape = t**k.n*P
        derivative = -k.n*t**(k.n-1)*P+t**k.n*P1
        second = k.n*(k.n-1)*t**(k.n-2)*P-2*k.n*t**(k.n-1)*P1+t**k.n*P2
    constant = float(KERNELS[name].c3_pi)/np.pi
    monomial = np.prod(y**np.array(alpha), axis=-1)
    measure = jac*monomial
    mass = np.sum(measure*constant/h**3*shape)
    grad = np.array([-np.sum(measure*constant/h**5*derivative/q*y[..., j]) for j in range(3)])
    lap = np.sum(measure*constant/h**5*(second+2*derivative/q))
    return mass, grad, lap


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("element,query,h", [("tet", (-.1, -.2, -.1), 2.),
                                          ("tet", (.2, .3, -.15), .65),
                                          ("tri", (.2, .1, .15), .7)])
def test_direct_simplex_quadrature(name, element, query, h):
    vertices = OBLIQUE if element == "tet" else OBLIQUE[:3]
    e = Tetrahedron(vertices, query) if element == "tet" else Triangle(vertices, query)
    alpha = (1, 0, 1)
    low = direct_rule(vertices, query, name, h, 40, alpha)
    high = direct_rule(vertices, query, name, h, 80, alpha)
    exact = [float(e.moment(name, alpha, h)), np.array(e.kernel_gradient(name, alpha, h), float)]
    # Independent quadrature has limited smoothness at the support/knot circles.
    for a, b, c in zip(exact, high, low):
        assert np.max(np.abs(a-b)) < 2e-7
        assert np.max(np.abs(b-c)) < 2e-6
    if element == "tet":
        assert abs(float(e.laplacian(name, alpha, h))-high[2]) < 2e-5
        assert abs(high[2]-low[2]) < 1e-4


@pytest.mark.parametrize("name", ALL)
def test_derivative_hessian_laplacian_and_scaling(name):
    query = (F(-1, 10), F(1, 5), F(1, 10))
    alpha = (1, 1, 0)
    t = Tetrahedron(OBLIQUE, query)
    step = mp.mpf("1e-9")
    H = t.kernel_hessian(name, alpha)
    close(sum(H[i][i] for i in range(3)), t.laplacian(name, alpha), "1e-32")
    gm = t.moment_gradient(name, alpha)
    for j in range(3):
        ql, qr = list(query), list(query)
        ql[j] = mpq(ql[j])-step
        qr[j] = mpq(qr[j])+step
        left, right = Tetrahedron(OBLIQUE, ql), Tetrahedron(OBLIQUE, qr)
        close((right.moment(name, alpha)-left.moment(name, alpha))/(2*step), gm[j], "1e-16")
        # Differentiate kernel-only moments, correcting translation of y^alpha.
        gl, gr = left.kernel_gradient(name, alpha), right.kernel_gradient(name, alpha)
        correction = t.kernel_gradient(name, tuple(a-int(i == j) for i, a in enumerate(alpha))) if alpha[j] else (0,)*3
        for i in range(3):
            close((gr[i]-gl[i])/(2*step)+alpha[j]*correction[i], H[j][i], "1e-15")
            close(H[i][j], H[j][i], "1e-32")
    s = F(7, 3)
    scaled = Tetrahedron([tuple(s*x for x in v) for v in OBLIQUE], tuple(s*x for x in query))
    close(scaled.moment(name, alpha, s), mpq(s)**sum(alpha)*t.moment(name, alpha))
    for a, b in zip(scaled.kernel_gradient(name, alpha, s), t.kernel_gradient(name, alpha)):
        close(a, mpq(s)**(sum(alpha)-1)*b)
    close(scaled.laplacian(name, alpha, s), mpq(s)**(sum(alpha)-2)*t.laplacian(name, alpha))


def test_triangle_orientation_area_scaling_and_boundary_projection():
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    for q in [(0, 0, 0), (F(1, 3), 0, 0), (F(1, 4), F(1, 4), 0), (0, 0, 1)]:
        f = Triangle(verts, q)
        close(f.block_moment((0, 0, 0), 0, mp.mpf(3)), mp.mpf(1)/2)
        close(f.moment(), Triangle(verts[::-1], q).moment())
    f = Triangle(verts, (0, 0, 0))
    close(f.block_moment((0, 0, 0), 0, mp.mpf("0.1")), mp.pi/400)
    scaled = Triangle([tuple(3*x for x in v) for v in verts])
    close(scaled.moment(h=3), f.moment()/3)


@pytest.mark.parametrize("degree", range(4))
@pytest.mark.parametrize("kind", ["tet", "tri"])
def test_exact_lagrange_basis_and_fem_reproduction(degree, kind):
    vertices = OBLIQUE if kind == "tet" else OBLIQUE[:3]
    query = (F(-1, 10), F(1, 10), F(1, 5))
    e = Tetrahedron(vertices, query) if kind == "tet" else Triangle(vertices, query)
    nodes, polys = expansion(vertices, query, degree)
    for i, node in enumerate(nodes):
        y = tuple(a-b for a, b in zip(node, query))
        for j, poly in enumerate(polys):
            assert sum(c*np.prod([t**k for t, k in zip(y, a)]) for a, c in poly.items()) == int(i == j)
    w, g = e.nodal_weights("w4", degree, gradient=True)
    # Use a degree-p Cartesian field; P0 is constant.
    values = [node[0]**degree for node in nodes]
    expected = sum(comb(degree, j)*mpq(query[0])**(degree-j)*e.moment("w4", (j, 0, 0))
                   for j in range(degree+1))
    close(sum(mpq(v)*a for v, a in zip(values, w)), expected, "1e-32")
    close(sum(w), e.moment("w4"), "1e-32")
    for j in range(3):
        close(sum(row[j] for row in g), e.kernel_gradient("w4")[j], "1e-32")
        expected_gradient = sum(comb(degree, k)*mpq(query[0])**(degree-k)*e.kernel_gradient("w4", (k, 0, 0))[j]
                                for k in range(degree+1))
        close(sum(mpq(v)*row[j] for v, row in zip(values, g)), expected_gradient, "1e-32")


def test_tetrahedral_subdivision_and_boundary_surface_agree():
    q = (F(1, 5), F(1, 10), F(-1, 5))
    whole = Tetrahedron(OBLIQUE, q)
    center = tuple(sum(v[j] for v in OBLIQUE)/4 for j in range(3))
    parts = [Tetrahedron([center]+[OBLIQUE[i] for i in range(4) if i != k], q) for k in range(4)]
    shell = ClosedSurface(OBLIQUE, whole.indices, q)
    for alpha in ((0, 0, 0), (1, 0, 0), (0, 1, 1), (2, 0, 1)):
        close(sum(t.moment("cubic", alpha) for t in parts), whole.moment("cubic", alpha), "1e-32")
        close(shell.moment("cubic", alpha), whole.moment("cubic", alpha))
        for j in range(3):
            close(sum(t.kernel_gradient("cubic", alpha)[j] for t in parts), whole.kernel_gradient("cubic", alpha)[j], "1e-32")


def test_geometry_guards():
    with pytest.raises(ValueError, match="degenerate"):
        Tetrahedron(UNIT[:3]+[UNIT[0]])
    with pytest.raises(ValueError, match="degenerate"):
        Triangle([UNIT[0]]*3)
    t = Tetrahedron(UNIT)
    with pytest.raises(ValueError, match="watertight"):
        ClosedSurface(UNIT, t.indices[:-1])
    with pytest.raises(ValueError, match="oriented"):
        ClosedSurface(UNIT, [tuple(reversed(f)) if i == 0 else f for i, f in enumerate(t.indices)])
    with pytest.raises(ValueError, match="positive signed"):
        ClosedSurface(UNIT, [tuple(reversed(f)) for f in t.indices])
    with pytest.raises(ValueError, match="positive"):
        kernel_profile("w2", 0)


def test_independent_positive_slice_overlap():
    # Thin element plus partial clipping: avoid a signed boundary-flux oracle.
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (F(1, 5), F(1, 5), F(1, 10**8))]
    query, radius = (F(1, 4), F(1, 4), F(-1, 10)), F(2, 5)
    t = Tetrahedron(vertices, query)
    value, estimate = constant_overlap(vertices, query, radius)
    ref = t.block_moment((0, 0, 0), 0, mpq(radius))
    assert abs(value-ref) < mp.mpf("1e-18")
    assert estimate < mp.mpf("1e-17")
    for face in t.faces:
        value, estimate = constant_overlap(face.vertices, query, radius)
        ref = face.block_moment((0, 0, 0), 0, mpq(radius))
        assert abs(value-ref) < mp.mpf("1e-10")


@pytest.mark.parametrize("name", ALL)
def test_nested_cavity_and_rigid_covariance(name):
    outer = Tetrahedron([(-3, -3, -3), (9, -3, -3), (-3, 9, -3), (-3, -3, 9)])
    inner = Tetrahedron(UNIT)
    vertices = list(outer.vertices)+list(inner.vertices)
    faces = list(outer.indices)+[tuple(i+4 for i in reversed(f)) for f in inner.indices]
    query = (F(1, 5), F(1, 5), F(1, 5))
    cavity = ClosedSurface(vertices, faces, query)
    close(cavity.moment(name), 1-Tetrahedron(UNIT, query).moment(name), "1e-32")
    # Rational 90-degree rotation and a very large exact translation.
    def transform(v):
        return (10**9-v[1], -10**9+v[0], 3+v[2])
    t = Tetrahedron(OBLIQUE, query)
    rotated = Tetrahedron([transform(v) for v in OBLIQUE], transform(query))
    close(rotated.moment(name), t.moment(name))
    gx, gy, gz = t.kernel_gradient(name)
    for a, b in zip(rotated.kernel_gradient(name), (-gy, gx, gz)):
        close(a, b)


@pytest.mark.parametrize("name", ("w2", "w4", "w6"))
def test_sphere_mesh_convergence(name):
    from warpSPHBoundaries.curvbound import sphere

    vertices = [(1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.),
                (0., 0., 1.), (0., 0., -1.)]
    faces = []
    for x, y, z in product((0, 1), (2, 3), (4, 5)):
        f = [x, y, z]
        a, b, c = (np.array(vertices[i]) for i in f)
        if np.dot(np.cross(b-a, c-a), a) < 0:
            f.reverse()
        faces.append(tuple(f))
    exact = mpq(sphere(name, F(1, 2), F(1, 5)))
    values = []
    for level in range(3):
        v = [tuple(F(float(t))/2 for t in v) for v in vertices]
        mesh = ClosedSurface(v, faces, (F(7, 10), 0, 0))
        values.append(mesh.moment(name))
        cache, new = {}, []

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                point = np.array(vertices[a])+np.array(vertices[b])
                point /= np.linalg.norm(point)
                cache[key] = len(vertices)
                vertices.append(tuple(float(x) for x in point))
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = new
    assert 0 < values[0] < values[1] < values[2] < exact
    assert exact-values[2] < (exact-values[0])/5


def test_volume_mesh_connectivity_and_boundary_extraction():
    vertices = UNIT+[(1, 1, 1)]
    mesh = TetrahedralMesh(vertices, np.array([(0, 1, 2, 3), (1, 2, 3, 4)]), (F(1, 5), 0, F(-1, 5)))
    assert len(mesh.boundary().faces) == 6
    for alpha in ((0, 0, 0), (1, 1, 0), (2, 1, 0)):
        close(mesh.moment("w6", alpha), mesh.boundary().moment("w6", alpha), "1e-32")
        for a, b in zip(mesh.kernel_gradient("w6", alpha), mesh.boundary().kernel_gradient("w6", alpha)):
            close(a, b, "1e-32")
    close(mesh.laplacian(), mesh.boundary().laplacian(), "1e-32")
    for a, b in zip(mesh.kernel_hessian(), mesh.boundary().kernel_hessian()):
        for x, y in zip(a, b):
            close(x, y, "1e-32")
    for t, w in zip(mesh.elements, mesh.nodal_weights(degree=2)):
        close(sum(w), t.moment(), "1e-32")
    with pytest.raises(ValueError, match="duplicate"):
        TetrahedralMesh(UNIT, [(0, 1, 2, 3), (1, 0, 3, 2)])
    with pytest.raises(ValueError, match="orientation"):
        TetrahedralMesh(UNIT+[(F(1, 10), F(1, 10), F(1, 10))], [(0, 1, 2, 3), (0, 1, 2, 4)])
