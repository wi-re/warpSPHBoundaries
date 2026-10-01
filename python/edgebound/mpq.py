"""Exact <-> mpmath number helpers (this mpmath build cannot take Fractions).

Rule: rational data stay `Fraction` until the last moment; `mpq` converts
with one correctly rounded division, so no double-precision literal ever
enters a 30-50 digit computation.
"""
from fractions import Fraction

import mpmath as mp


def to_frac(x) -> Fraction:
    """int / Fraction / 'a/b' / '0.25' / float / mpf -> exact Fraction."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x)
    if isinstance(x, mp.mpf):
        if not mp.isfinite(x):
            raise ValueError("non-finite mpf")
        sign, man, exp, _bc = x._mpf_            # man is UNSIGNED, sign is 0/1
        v = Fraction(man << exp) if exp >= 0 else Fraction(man, 1 << (-exp))
        return -v if sign else v
    raise TypeError(f"cannot convert {type(x)} to Fraction")


def mpq(x) -> mp.mpf:
    """Fraction (or anything to_frac accepts) -> mpf at the current precision."""
    f = to_frac(x)
    if f.denominator == 1:
        return mp.mpf(f.numerator)
    return mp.mpf(f.numerator) / mp.mpf(f.denominator)


def sgn(f: Fraction) -> int:
    return (f > 0) - (f < 0)
