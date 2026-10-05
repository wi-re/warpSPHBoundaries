"""Closed-form edge primitives (mpmath).  Verified in `maple/10_edge_primitives.mpl`.

With r = sqrt(s^2 + z^2):

    I_m(s) = int r^m ds         odd antiderivative (I_m(0) = 0)
    J_m(s) = int s r^m ds       r^(m+2)/(m+2)   (J_-2 = ln r)
    S_{j,m}(s) = int s^j r^m ds via s^2 = r^2 - z^2

`I_all` returns I_0..I_M at once (upward recurrence, safe for any z incl. z = 0:
the only place z^2 * asinh(s/|z|) occurs is guarded to 0 at z = 0).
`Ig` supports m < 0 (downward recurrence, divides by z^2: z != 0 only) and is
used ONLY by the far-field cross-check form (b).
"""
from math import comb

import mpmath as mp


def I_all(M: int, s, z):
    """[I_0(s), ..., I_M(s)] for the odd antiderivatives of r^m, m >= 0."""
    zz = z * z
    r2 = s * s + zz
    r = mp.sqrt(r2)
    out = [s]
    if M >= 1:
        tail = zz * mp.asinh(s / abs(z)) if z != 0 else mp.mpf(0)
        out.append((s * r + tail) / 2)
    for m in range(2, M + 1):
        # r^m: even m -> r2**(m//2) exact powers; odd m via r * r2**((m-1)//2)
        rpow = r2 ** (m // 2) if m % 2 == 0 else r * r2 ** ((m - 1) // 2)
        out.append((s * rpow + m * zz * out[m - 2]) / (m + 1))
    return out


def I(m: int, s, z):
    return I_all(m, s, z)[m]


def J(m: int, s, z):
    """int s r^m ds, m >= 0 (even in s)."""
    return (s * s + z * z) ** (mp.mpf(m + 2) / 2) / (m + 2) if m % 2 else (s * s + z * z) ** ((m + 2) // 2) / (m + 2)


def S_all(j: int, m: int, s, z):
    """int s^j r^m ds for m >= 0 (m = n + 2i, i <= j/2 below)."""
    zz = z * z
    if j % 2 == 0:
        h = j // 2
        Is = I_all(m + 2 * h, s, z)
        return sum(comb(h, i) * (-zz) ** (h - i) * Is[m + 2 * i] for i in range(h + 1))
    h = (j - 1) // 2
    return sum(comb(h, i) * (-zz) ** (h - i) * J(m + 2 * i, s, z) for i in range(h + 1))


# ---------------------------------------------------------------- m < 0 (b only)
def Ig(m: int, s, z):
    """I_m for any integer m (z != 0 for m < 0)."""
    if m >= 0:
        return I_all(m, s, z)[m]
    if z == 0:
        raise ZeroDivisionError("negative-power primitives need z != 0")
    zz = z * z
    r2 = s * s + zz
    if m == -1:
        return mp.asinh(s / abs(z))
    if m == -2:
        return mp.atan(s / z) / z
    # downward: I_m = ((m+3) I_{m+2} - s r^(m+2)) / ((m+2) z^2)
    return ((m + 3) * Ig(m + 2, s, z) - s * r2 ** (mp.mpf(m + 2) / 2)) / ((m + 2) * zz)


def Jg(m: int, s, z):
    if m == -2:
        return mp.log(s * s + z * z) / 2
    return (s * s + z * z) ** (mp.mpf(m + 2) / 2) / (m + 2)


def Sg(j: int, m: int, s, z):
    """int s^j r^m ds for any integer m (z != 0 when m + j-dependent terms go negative)."""
    zz = z * z
    if j % 2 == 0:
        h = j // 2
        return sum(comb(h, i) * (-zz) ** (h - i) * Ig(m + 2 * i, s, z) for i in range(h + 1))
    h = (j - 1) // 2
    return sum(comb(h, i) * (-zz) ** (h - i) * Jg(m + 2 * i, s, z) for i in range(h + 1))


def dangle(z, lo, hi):
    """atan(hi/z) - atan(lo/z), computed WITHOUT cancellation for short / far chords:
    the angle between (z, lo) and (z, hi) is atan2(z (hi-lo), z^2 + lo hi).
    Convention at z == 0: 0 (average of the two one-sided limits)."""
    if z == 0:
        return mp.mpf(0)
    return mp.atan2(z * (hi - lo), z * z + lo * hi)
