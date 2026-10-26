"""Versioned, solver-independent Cartesian kernel-integral requests."""
from dataclasses import dataclass
from numbers import Integral

CONTRACT_VERSION = 1
KERNELS = ("w2", "w4", "w6", "cubic")
MAX_DEGREE = 9


@dataclass(frozen=True)
class Request:
    alpha: tuple
    derivative: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "alpha", tuple(self.alpha))
        object.__setattr__(self, "derivative", tuple(self.derivative))
        d = len(self.alpha)
        if d not in (2, 3) or any(not isinstance(a, Integral) or a < 0 for a in self.alpha):
            raise ValueError("alpha must contain two or three nonnegative integers")
        if sum(self.alpha) > MAX_DEGREE:
            raise ValueError("polynomial degree exceeds nine")
        if len(self.derivative) > 2 or any(not isinstance(j, Integral) or j not in range(d) for j in self.derivative):
            raise ValueError("kernel derivatives must have order 0..2 and valid axes")

    @property
    def scale(self):
        return sum(self.alpha)-len(self.derivative)


def monomials(dimension, degree):
    if dimension not in (2, 3) or not isinstance(degree, Integral) or degree not in range(10):
        raise ValueError("dimension must be 2 or 3 and degree 0..9")
    def lattice(d, n):
        if d == 1:
            return [(n,)]
        return [(i,)+b for i in range(n, -1, -1) for b in lattice(d-1, n-i)]
    return tuple(a for n in range(degree+1) for a in lattice(dimension, n))


def baseline(dimension):
    """Stored baseline moments, first gradient moments, plus symmetric Hessian."""
    z = (0,)*dimension
    return tuple(Request(a) for a in monomials(dimension, 2)) + tuple(
        Request(a, (j,)) for a in monomials(dimension, 1) for j in range(dimension)) + tuple(
        Request(z, (i,j)) for i in range(dimension) for j in range(i, dimension))
