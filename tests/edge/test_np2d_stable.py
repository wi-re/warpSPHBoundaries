"""Stage 3: float32 conditioning. Exact Chebyshev compile + stable quadrature mode of np2d."""
from fractions import Fraction as F

import numpy as np
import pytest

from edgebound.edge import np2d
from .test_np2d import ELEMENTS, KERNELS, ref, rel_geometry

REPRESENTABLE = [c for c in ELEMENTS if not c["id"].startswith(("small_element-1e6", "small_element-1e3", "tiny_chord-both", "tiny_chord-grazing",
                                                                 "tiny_chord-edge_end-1e-13", "tiny_chord-knot"))]


def test_chebyshev_compile_is_exact_and_well_conditioned():
    """exact rational conversion; amplification sum|a_k| / max|P| ~ 1 (monomial basis: 1e2 .. 2e4)."""
    worst_c, worst_m = 0, 0
    for k in KERNELS:
        for al in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (0, 4), (3, 1)]:
            edges, vals = np2d.compile_moment(k, al)
            profs = [(P, R) for (_, _, R), P in edges.items()] + [(P, R) for R, P in vals.items()] \
                + [({n: c / (n + 2) for n, c in P.items()}, R) for R, P in vals.items()]
            for P, R in profs:
                a = np2d.cheb_coeffs(P, R)
                # exact check at rational points: sum a_k T_k(x) == sum P_n r^n
                for rr in (F(0), F(1, 7), F(1, 3) * F(R), F(R)):
                    x = 2 * rr / F(R) - 1
                    T = [F(1), x]
                    for _ in range(len(a)):
                        T.append(2 * x * T[-1] - T[-2])
                    assert sum(ak * T[i] for i, ak in enumerate(a)) == sum(c * rr ** n for n, c in P.items())
                rs = np.linspace(0, float(R), 1001)
                pmax = np.abs(sum(float(c) * rs ** n for n, c in P.items())).max()
                worst_c = max(worst_c, float(sum(abs(x) for x in a)) / pmax)
                worst_m = max(worst_m, float(sum(abs(c) * F(R) ** n for n, c in P.items())) / pmax)
    assert worst_c < 2.0 and worst_m > 1e4


@pytest.mark.parametrize("k", KERNELS)
def test_float64_stable_quadrature_accuracy(k):
    V, X, H = rel_geometry(REPRESENTABLE)
    st = (8, 6)
    v = np2d.value(V, X, k, H, stable=st)
    g = np2d.gradient(V, X, k, H, stable=st)
    m = np2d.moment(V, X, k, (1, 1), H, stable=st)
    for i, c in enumerate(REPRESENTABLE):
        assert abs(v[i] - ref(c, k, "value")) < 1e-9
        assert np.abs(g[i] - ref(c, k, "grad")).max() * H[i] < 1e-6
        assert abs(m[i] - ref(c, k, "m", "1,1")) / H[i] ** 2 < 1e-9


@pytest.mark.parametrize("k", KERNELS)
def test_float32_stable_meets_float32_tolerances(k):
    """measured (docs/exactness-and-approximations.md): value 1e-7, gradient 3e-7, moments 1e-8 at f32 epsilon 6e-8."""
    V, X, H = rel_geometry(REPRESENTABLE)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    st = (8, 6)
    v = np2d.value(f32(V), f32(X), k, f32(H), dtype=np.float32, stable=st)
    g = np2d.gradient(f32(V), f32(X), k, f32(H), dtype=np.float32, stable=st)
    m = np2d.moment(f32(V), f32(X), k, (1, 1), f32(H), dtype=np.float32, stable=st)
    m4 = np2d.moment(f32(V), f32(X), k, (2, 2), f32(H), dtype=np.float32, stable=st)
    assert v.dtype == np.float32
    for i, c in enumerate(REPRESENTABLE):
        assert abs(float(v[i]) - ref(c, k, "value")) < 5e-7, c["id"]
        assert np.abs(g[i].astype(np.float64) - ref(c, k, "grad")).max() * H[i] < 2e-6, c["id"]
        assert abs(float(m[i]) - ref(c, k, "m", "1,1")) / H[i] ** 2 < 2e-7, c["id"]
        assert abs(float(m4[i]) - ref(c, k, "m", "2,2")) / H[i] ** 4 < 2e-7, c["id"]


def test_float32_stable_beats_float32_closed_form_by_orders_of_magnitude():
    gen = [c for c in REPRESENTABLE if c["id"].startswith("generic")]
    V, X, H = rel_geometry(gen)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    e = {}
    for name, kw in [("closed", {}), ("stable", dict(stable=(6, 5)))]:
        v = np2d.value(f32(V), f32(X), "w6", f32(H), dtype=np.float32, **kw)
        e[name] = max(abs(float(v[i]) - ref(c, "w6", "value")) for i, c in enumerate(gen))
    assert e["stable"] < 1e-6 and e["closed"] > 20 * e["stable"], e


def test_stable_mode_handles_tiny_elements_and_covering_meshes_in_float32():
    from .test_np2d import MESHES
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    c = MESHES[0]
    V, X, H = rel_geometry([dict(polygon=P, x=c["x"], h=c["h"]) for P in c["elements"]])
    for k in KERNELS:
        tot = np2d.value(f32(V), f32(X), k, f32(H), dtype=np.float32, stable=(6, 5)).sum()
        assert abs(float(tot) - 1) < 5e-6
