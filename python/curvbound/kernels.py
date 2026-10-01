"""Kernel table for the boundary-integral derivations.

Conventions match warpSPHCore (src/warpSPHCore/kernels/kernel.py and
scripts/kernels/kernel_specs.yaml):

    W(r, h) = C_n / h^n * W_hat(q),   q = r / h,   h = support radius,

with W_hat compactly supported on [0, 1].  The kernel shapes are identical
in 2D and 3D; only C_n differs (transcribed from kernel_specs.yaml):

    cubic : W_hat = (1-q)^3 - 4*(1/2-q)_+^3   C2 = 80/(7*pi)  C3 = 16/pi
    w2    : (1-q)^4  (1+4q)                   C2 = 7/pi       C3 = 21/(2*pi)
    w4    : (1-q)^6  (1+6q+35/3*q^2)          C2 = 9/pi       C3 = 495/(32*pi)
    w6    : (1-q)^8  (1+8q+25q^2+32q^3)       C2 = 78/(7*pi)  C3 = 1365/(64*pi)
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Kernel:
    name: str
    kind: str            # "cubic" (knot at q=1/2) or "wendland" (single poly)
    c2_pi: Fraction      # C2 = c2_pi / pi
    c3_pi: Fraction      # C3 = c3_pi / pi
    n: int = 0           # Wendland: power of (1-q)
    poly: tuple = ()     # Wendland: ((power, coeff), ...) of P(q), c in Q

    def w_hat(self, q: Fraction) -> Fraction:
        """Exact kernel shape at a rational q (symbolic checks)."""
        if self.kind == "cubic":
            if q <= Fraction(1, 2):
                return (1 - q) ** 3 - 4 * (Fraction(1, 2) - q) ** 3
            return (1 - q) ** 3
        v = Fraction(0)
        for p, c in self.poly:
            v += c * q ** p
        return (1 - q) ** self.n * v


KERNELS = {
    "cubic": Kernel("cubic", "cubic", Fraction(80, 7), Fraction(16)),
    "w2": Kernel("w2", "wendland", Fraction(7), Fraction(21, 2), 4,
                 ((0, Fraction(1)), (1, Fraction(4)))),
    "w4": Kernel("w4", "wendland", Fraction(9), Fraction(495, 32), 6,
                 ((0, Fraction(1)), (1, Fraction(6)), (2, Fraction(35, 3)))),
    "w6": Kernel("w6", "wendland", Fraction(78, 7), Fraction(1365, 64), 8,
                 ((0, Fraction(1)), (1, Fraction(8)),
                  (2, Fraction(25)), (3, Fraction(32)))),
}


def kernel(name: str) -> Kernel:
    try:
        return KERNELS[name]
    except KeyError:
        raise KeyError(f"unknown kernel {name!r}; expected one of {sorted(KERNELS)}")
