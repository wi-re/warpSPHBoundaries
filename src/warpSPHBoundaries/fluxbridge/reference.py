"""Arbitrary-precision fixtures, deliberately independent of the device plan."""
import mpmath as mp
from ..edge import core, geometry
from ..edge.kernels import KERNELS
from ..edge.mpq import mpq
from ..mesh3d.reference import Triangle, Tetrahedron, ClosedSurface, kernel_profile


def evaluate(vertices, query, requests, kernel="w2", h=1, measure="volume", faces=None):
    d = len(query)
    if d == 3:
        el = (Triangle(vertices, query) if measure == "surface" else
              Tetrahedron(vertices, query) if faces is None else ClosedSurface(vertices, faces, query))
        result = []
        for req in requests:
            a, ds = req.alpha, req.derivative
            if not ds:
                v = el.moment(kernel, a, h)
            elif len(ds) == 1:
                v = el.kernel_gradient(kernel, a, h)[ds[0]]
            elif measure == "volume":
                v = el.kernel_hessian(kernel, a, h)[ds[0]][ds[1]]
            else:
                i, j = ds
                q0 = tuple(mpq(v) for v in el.query)
                def shifted(t):
                    q = tuple(q0[k]+(t if k==i else 0) for k in range(3))
                    return Triangle(vertices,q).kernel_gradient(kernel,a,h)[j]
                v = mp.diff(shifted,mp.mpf(0))
                if a[i]:
                    lower = tuple(t-int(k==i) for k,t in enumerate(a))
                    v += a[i]*el.kernel_gradient(kernel,lower,h)[j]
            result.append(v)
        return result
    if measure != "volume":
        raise ValueError("2D uses area/volume, not embedded surface triangles")
    # Boundary differentiation is independent of the new plan interpreter.
    P = geometry.prepare(vertices, query, h)
    blocks = KERNELS[kernel].blocks
    def M(a):
        return core.moment(vertices, query, kernel, a, h)
    def B(a, j, derivative_axis=None):
        acc = mp.mpf(0)
        for e in P.edges:
            for block in blocks:
                R = mpq(block.R)
                chord = geometry.chord(e, R)
                if chord is None:
                    continue
                lo, hi = chord
                b = list(a)
                if derivative_axis is not None:
                    b[derivative_axis] += 1
                poly = core.ypoly(e, b)
                for n,c in enumerate(block.coeffs):
                    if not c or (derivative_axis is not None and not n):
                        continue
                    power = n-2 if derivative_axis is not None else n
                    weight = mpq(c)*n if derivative_axis is not None else mpq(c)
                    acc += e.n[j]*weight*sum(v*core._Sdiff(k,power,lo,hi,e.z) for k,v in enumerate(poly) if v)/mp.pi
        return acc
    out=[]
    for req in requests:
        a, ds = req.alpha, req.derivative
        if not ds:
            v=M(a)
        elif len(ds)==1:
            j=ds[0]
            lower=tuple(t-int(k==j) for k,t in enumerate(a))
            v=-B(a,j)*mpq(h)**(sum(a)-1)+(a[j]*M(lower) if a[j] else 0)
        else:
            i,j=ds
            lower=tuple(t-int(k==i) for k,t in enumerate(a))
            G=evaluate(vertices,query,[type(req)(lower,(j,))],kernel,h)[0] if a[i] else 0
            v=B(a,i,j)*mpq(h)**(sum(a)-2)+a[i]*G
        out.append(v)
    return out
