"""Golden fixtures (tests/fixtures/edge2d_golden.json): internal consistency + independent checks.

Every later backend (numpy/torch/warp) loads this file; here we make sure the file itself is right:
  * regenerating a case reproduces the stored numbers (regression),
  * value / gradient / moments of every element agree with the independent polar oracle,
  * exact expectations hold (covering meshes = rational disk moments, half-plane = lambda_2, ...).
"""
from fractions import Fraction as F

import mpmath as mp
import pytest

import edgebound as eb
from edgebound import fixtures as fx
from edgebound.mpq import mpq

DATA = fx.load()
CASES = DATA["cases"]
ELEMENTS = [c for c in CASES if c["kind"] == "element"]
MESHES = [c for c in CASES if c["kind"] == "mesh"]


def P(rec):
    return [(F(a), F(b)) for a, b in rec]


def num(s):
    return mp.mpf(s) if "/" not in s else mpq(F(s))


def test_meta_and_coverage_of_hard_cases():
    m = DATA["meta"]
    assert m["format"] == "edge2d-golden-v1" and m["kernels"] == ["cubic", "w2", "w4", "w6"]
    tags = {t for c in CASES for t in c["tags"]}
    required = {"x_on_edge", "x_at_vertex", "x_on_edge_line", "z_to_0", "tiny_chord", "small_element", "half_plane",
                "covering_mesh", "engulf", "nonconvex", "clockwise", "h_scaled", "outside_support", "cubic_knot", "generic"}
    assert required <= tags, required - tags
    wheres = {c["where"] for c in ELEMENTS}
    assert wheres == {"inside", "outside", "edge", "vertex"}
    for c in ELEMENTS:
        assert set(c["kernels"]) == set(m["kernels"])
        for k, e in c["kernels"].items():
            assert len(e["grad"]) == 2 and set(e["moments"]) == {f"{a},{b}" for a, b in m["moments"]}


@pytest.mark.parametrize("rec", ELEMENTS, ids=[c["id"] for c in ELEMENTS])
def test_regeneration_reproduces_stored_numbers(rec):
    where, ker = fx.evaluate_element(P(rec["polygon"]), tuple(F(s) for s in rec["x"]), F(rec["h"]))
    assert where == rec["where"]
    for k in ker:
        assert abs(num(ker[k]["value"]) - num(rec["kernels"][k]["value"])) < mp.mpf(10) ** -36
        for key, v in ker[k]["moments"].items():
            ref = num(rec["kernels"][k]["moments"][key])
            assert abs(num(v) - ref) < mp.mpf(10) ** -36 * (1 + abs(ref))


@pytest.mark.parametrize("rec", ELEMENTS, ids=[c["id"] for c in ELEMENTS])
def test_value_gradient_vs_polar_oracle(rec):
    T, x, h = P(rec["polygon"]), tuple(F(s) for s in rec["x"]), F(rec["h"])
    for k in ["cubic", "w4"]:
        e = rec["kernels"][k]
        assert abs(num(e["value"]) - eb.polar_value(T, x, k, h=h)) < mp.mpf(10) ** -28
        g = eb.polar_gradient(T, x, k, h=h)
        assert max(abs(num(e["grad"][0]) - g[0]), abs(num(e["grad"][1]) - g[1])) < mp.mpf(10) ** -28 * (1 + 1 / mpq(h))


SAMPLE = ELEMENTS[::4]


@pytest.mark.parametrize("rec", SAMPLE, ids=[c["id"] for c in SAMPLE])
def test_moments_vs_polar_oracle_sampled(rec):
    T, x, h = P(rec["polygon"]), tuple(F(s) for s in rec["x"]), F(rec["h"])
    for k, al in [("w2", (1, 1)), ("cubic", (2, 1)), ("w6", (0, 4)), ("w4", (3, 0))]:
        ref = eb.polar_moment(T, x, k, al, h=h)
        got = num(rec["kernels"][k]["moments"][f"{al[0]},{al[1]}"])
        assert abs(got - ref) < mp.mpf(10) ** -28 * (1 + abs(ref)), (rec["id"], k, al)


@pytest.mark.parametrize("rec", MESHES, ids=[c["id"] for c in MESHES])
def test_covering_meshes_equal_exact_disk_moments(rec):
    ex = rec["expect"]
    for k in rec["kernels"]:
        assert abs(num(rec["kernels"][k]["value"]) - 1) < mp.mpf(10) ** -35
        assert max(abs(num(g)) for g in rec["kernels"][k]["grad"]) < mp.mpf(10) ** -33
        for key, v in rec["kernels"][k]["moments"].items():
            assert abs(num(v) - num(ex["moments_exact"][k][key])) < mp.mpf(10) ** -34, (rec["id"], k, key)
        # m_alpha of a covering mesh is the full-space moment, independent of x: its x-gradient vanishes
        # (the moment gradient is purely edge-local; the extra -alpha_j m_{alpha-e_j} belongs to grad_x acting on W only)
        for key, v in rec["kernels"][k]["moment_grad"].items():
            assert max(abs(num(v[0])), abs(num(v[1]))) < mp.mpf(10) ** -32, (rec["id"], k, key)


def test_half_plane_expectations():
    hp = [c for c in ELEMENTS if "half_plane" in c["tags"]]
    assert len(hp) >= 15
    for c in hp:
        for k, lam in c["expect"]["half_plane_lambda2"].items():
            assert abs(num(c["kernels"][k]["value"]) - num(lam)) < mp.mpf(10) ** -35, (c["id"], k)


def test_engulf_and_outside_support():
    for c in ELEMENTS:
        if "engulf" in c["tags"]:
            for k, e in c["kernels"].items():
                assert abs(num(e["value"]) - 1) < mp.mpf(10) ** -38
                assert max(abs(num(g)) for g in e["grad"]) < mp.mpf(10) ** -38
                for key, v in e["moments"].items():
                    assert abs(num(v) - num(c["expect"]["moments_exact"][k][key])) < mp.mpf(10) ** -37
        if "outside_support" in c["tags"]:
            for e in c["kernels"].values():
                assert num(e["value"]) == 0 and all(num(v) == 0 for v in e["moments"].values())


def test_inside_complement_symmetry_of_half_plane_moments():
    """x inside vs outside at the same |d|: m_alpha(in) + (-1)^b m_alpha(out) = disk moment (reflection)."""
    by = {c["id"]: c for c in ELEMENTS}
    for ds in ["1/10", "1/2", "9/10"]:
        cin, cout = by[f"half_plane-d=-{ds}"], by[f"half_plane-d={ds}"]
        for k in ["cubic", "w6"]:
            for key in ["1,0", "0,1", "2,0", "1,1", "0,2", "0,3", "2,2"]:
                a, b = map(int, key.split(","))
                disk = mpq(F(fx.disk_moment(eb.KERNELS[k], (a, b))))
                lhs = num(cin["kernels"][k]["moments"][key]) + (-1) ** b * num(cout["kernels"][k]["moments"][key])
                assert abs(lhs - disk) < mp.mpf(10) ** -35
