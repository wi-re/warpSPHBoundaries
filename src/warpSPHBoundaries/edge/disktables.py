"""Tables of the disk element (docs/disk-element.md): the five channels (lam, m1x, g0x, g1xx, g1yy) of the solid disk of ANY radius R (h = 1) at distance D of the query as piecewise tensor-Chebyshev
polynomials, built from the circle-identity reference `edge.disk`, so that one (query, disk) pair costs a table lookup instead of an arc quadrature / a polygon.

Coordinates: delta = D - R (the signed distance of the query to the circle, positive outside).  The channels are analytic in delta except on three lines, where the PANEL BOUNDARIES are placed:
    delta = 0        the query crosses the circle: the kernel's r^3 term (Wendland: 1 - 10 q^2 + 20 q^3 ...) makes the functions C^3 with an eps^4 ln|eps| term;
    D = 1 - R        the far point of the circle meets the support circle (R < 1: the disk stops being inside the support);
    D = 1 + R        the circle leaves the support (end of the table, all channels vanish).
    R > 1: the support ball lies inside the disk for D < R - 1 (lam = 1, g1 = I, m1 = g0 = 0, no table); the table covers delta in [-1, 1] in two panels split at delta = 0, tabulated against s = 1/R
           (s = 0 is the planar half plane, the tier-3 functions).
    R <= 1: two radius branches [0, 1/2] and [1/2, 1] (the lines delta = 0 and D = 1 - R cross at R = 1/2), each with three delta intervals bounded by those lines,
           (R <= 1/2: [-R, 0], [0, 1 - 2R], [1 - 2R, 1];  R >= 1/2: [-R, 1 - 2R], [1 - 2R, 0], [0, 1]), every interval mapped linearly onto [0, 1]; tabulated f / R^2 (every channel is ~ R^2: the point limit is regular).
"""
import numpy as np

from .disk import disk_channels

CH = ("lam", "m1x", "g0x", "g1xx", "g1yy")


class PanelTable:
    """f(a, b) on [0,1]^2 as Chebyshev panels: `pa` x `pb` panels of `na` x `nb` Gauss-Chebyshev nodes; coefficients [pa, pb, na, nb, nch]."""

    def __init__(self, fn, pa, pb, na, nb, nch):
        self.pa, self.pb, self.na, self.nb = pa, pb, na, nb
        xa = np.cos(np.pi * (np.arange(na) + 0.5) / na)          # nodes on [-1, 1]
        xb = np.cos(np.pi * (np.arange(nb) + 0.5) / nb)
        Va = np.polynomial.chebyshev.chebvander(xa, na - 1)
        Vb = np.polynomial.chebyshev.chebvander(xb, nb - 1)
        Ia, Ib = np.linalg.inv(Va), np.linalg.inv(Vb)
        self.coef = np.zeros((pa, pb, na, nb, nch))
        for i in range(pa):
            for j in range(pb):
                a = (i + 0.5 * (xa + 1.0)) / pa
                b = (j + 0.5 * (xb + 1.0)) / pb
                A, B = np.meshgrid(a, b, indexing="ij")
                F = fn(A.ravel(), B.ravel()).reshape(na, nb, nch)
                self.coef[i, j] = np.einsum("ka,abc,lb->klc", Ia, F, Ib)

    def __call__(self, a, b):
        a, b = np.clip(np.asarray(a, float), 0.0, 1.0), np.clip(np.asarray(b, float), 0.0, 1.0)
        i = np.minimum((a * self.pa).astype(int), self.pa - 1)
        j = np.minimum((b * self.pb).astype(int), self.pb - 1)
        ta, tb = 2.0 * (a * self.pa - i) - 1.0, 2.0 * (b * self.pb - j) - 1.0
        Ta = np.polynomial.chebyshev.chebvander(ta, self.na - 1)          # [M, na]
        Tb = np.polynomial.chebyshev.chebvander(tb, self.nb - 1)
        return np.einsum("mk,ml,mklc->mc", Ta, Tb, self.coef[i, j])


def _intervals(R, branch):
    """the delta intervals [lo, hi] (three, ascending) of a radius R <= 1 in the radius branch `branch` (0: R <= 1/2, 1: R >= 1/2)."""
    if branch == 0:
        return [(-R, 0.0 * R), (0.0 * R, 1.0 - 2.0 * R), (1.0 - 2.0 * R, 1.0 + 0.0 * R)]
    return [(-R, 1.0 - 2.0 * R), (1.0 - 2.0 * R, 0.0 * R), (0.0 * R, 1.0 + 0.0 * R)]


class DiskTables:
    """tables of one kernel; `channels(D, R)` is the numpy reference of the Warp lookup."""

    def __init__(self, kname, pu=3, pR=3, nu=12, nR=12, Lpu=6, Lps=3, Lnu=12, Lns=12):
        self.kname = kname
        self.config = (pu, pR, nu, nR, Lpu, Lps, Lnu, Lns)
        self.S = {}
        for branch in (0, 1):
            for k in range(3):
                def f(w, rr, branch=branch, k=k):
                    u = np.sin(0.5 * np.pi * w) ** 2                     # the interval ends are singular (eps^4 ln eps, eps^(11/2)): in w the functions are ~ w^8 ... w^11 there
                    R = (0.5 * rr if branch == 0 else 0.5 + 0.5 * rr)
                    R = np.clip(R, 1e-6, 1.0)
                    lo, hi = _intervals(R, branch)[k]
                    delta = lo + u * (hi - lo)
                    o = disk_channels(kname, np.maximum(R + delta, 1e-12), R)
                    return np.stack([o[c] for c in CH], 1) / (R * R)[:, None]
                self.S[(branch, k)] = PanelTable(f, pu, pR, nu, nR, 5)

        def l_fn(w, s):
            R = 1.0 / np.maximum(s, 1e-9)
            o = disk_channels(kname, R - 1.0 + 2.0 * w, R)
            return np.stack([o[c] for c in CH], 1)
        self.L = PanelTable(l_fn, Lpu, Lps, Lnu, Lns, 5)

    def channels(self, D, R):
        """[M, 5] (lam, m1x, g0x, g1xx, g1yy) at distances D and radii R (h = 1)."""
        D, R = np.broadcast_arrays(np.asarray(D, float), np.asarray(R, float))
        out = np.zeros(D.shape + (5,))
        small = R <= 1.0
        for branch in (0, 1):
            sel = small & ((R <= 0.5) if branch == 0 else (R > 0.5))
            if not sel.any():
                continue
            Rs, Ds = R[sel], D[sel]
            delta = Ds - Rs
            res = np.zeros((len(Rs), 5))
            for k, (lo, hi) in enumerate(_intervals(Rs, branch)):
                m = (delta >= lo) & (delta <= hi) if k == 2 else (delta >= lo) & (delta < hi)
                if not m.any():
                    continue
                u = np.clip((delta[m] - lo[m]) / np.maximum(hi[m] - lo[m], 1e-300), 0.0, 1.0)
                rr = (2.0 * Rs[m]) if branch == 0 else (2.0 * Rs[m] - 1.0)
                res[m] = self.S[(branch, k)](2.0 / np.pi * np.arcsin(np.sqrt(u)), rr) * (Rs[m] ** 2)[:, None]
            out[sel] = res
        big = ~small
        inside = big & (D <= R - 1.0)
        out[inside] = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
        ml = big & (D > R - 1.0) & (D < R + 1.0)
        out[ml] = self.L((D[ml] - R[ml] + 1.0) / 2.0, 1.0 / R[ml])
        return out


# ------------------------------------------------------------------------------------------------ storage: shipped with the package like the tier-3 tables
def _file(kname, config):
    from .. import paths
    return paths.packaged_tables_dir() / ("disk_%s_%s.npz" % (kname, "_".join(str(c) for c in config)))


def load_disk_tables(kname, config=(3, 3, 12, 12, 6, 3, 12, 12)):
    """the `DiskTables` of a registered kernel as flat arrays for the device: (S [6, pu, pR, nu, nR, 5], L [Lpu, Lps, Lnu, Lns, 5], config); read from the package data, else built (about a second per
    kernel; written to the writable table directory, `paths.tables_dir`)."""
    from .. import paths
    f = _file(kname, config)
    if not f.exists():
        f = paths.tables_dir() / f.name
        if not f.exists():
            T = DiskTables(kname, *config)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, S=np.stack([T.S[(b, k)].coef for b in (0, 1) for k in range(3)]), L=T.L.coef, config=np.array(config))
    z = np.load(f)
    return z["S"], z["L"], tuple(int(c) for c in z["config"])
