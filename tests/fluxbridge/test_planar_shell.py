"""Near-roundoff quadrature, tested independently of analytic fp64 accuracy."""
import json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warpSPHBoundaries.fluxbridge import Request, baseline
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator, functions

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
CASES = json.loads((Path(__file__).parents[1]/"fixtures/fluxbridge-planar-v1.json").read_text())["cases"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["w2", "w4", "w6", "cubic"])
@pytest.mark.parametrize("precision,budget", [("float64", 1e-14), ("float32", 1e-5)])
def test_all_degree_nine_jets_against_frozen_reference(device, kernel, precision, budget):
    for c in CASES:
        if c["dimension"] != 3 or c["kernel"] != kernel: continue
        kw = {} if c["half_width"] is None else dict(half_widths=float(c["half_width"]))
        values = []
        for nodes in (16, 24):
            e = PlanarEvaluator(c["kind"], 3, [Request(**r) for r in c["requests"]],
                kernel=kernel, precision=precision, method="shell_quadrature", nodes=nodes, device=device)
            out = e.evaluate(float(c["distance"]), c["normal"], 1, **kw)[0]
            ref = np.array(c["values"], float)
            assert np.isfinite(out).all()
            assert np.max(abs(out-ref)/(1+abs(ref))) < budget
            values.append(out)
        assert np.max(abs(values[0]-values[1])/(1+abs(values[1]))) < budget


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision,tol", [("float64", 1e-14), ("float32", 1e-5)])
def test_units_subsets_empty_contacts_and_reuse(device, precision, tol):
    n = [2/3, -1/3, 2/3]
    for kind in ("plane", "strip"):
        e = PlanarEvaluator(kind, 3, precision=precision, method="shell_quadrature", device=device)
        d = np.array([-.5, 0, 1e-18, -1e-18, .2, 1., 2., -2.])
        kw = {} if kind == "plane" else dict(half_widths=.2)
        b = e.prepare(d, n, 1, **kw)
        b.work.fill_(123); b.facets.fill_(e.api.vec(123)); b.launch(); value = b.numpy()
        b.launch(); np.testing.assert_array_equal(b.numpy(), value)
        scaled = e.evaluate(d*3, n, 3, **({} if kind == "plane" else dict(half_widths=.6)))
        np.testing.assert_allclose(scaled, value*np.array([3.**r.scale for r in e.requests]), atol=tol, rtol=tol)
        np.testing.assert_array_equal(value[-2], 0)
        req = [e.requests[-1], e.requests[0], e.requests[4], e.requests[-1]]
        subset = PlanarEvaluator(kind, 3, req, precision=precision, method="shell_quadrature", device=device)
        np.testing.assert_allclose(subset.evaluate(d, n, 1, **kw), value[:, [-1, 0, 4, -1]], atol=tol, rtol=tol)
        assert e.evaluate([], n, 1, **kw).shape == (0, len(e.requests))
        assert PlanarEvaluator(kind, 3, [], method="shell_quadrature", device=device).evaluate(d, n, 1, **kw).shape == (len(d), 0)


@pytest.mark.parametrize("device", DEVICES)
def test_complete_query_derivatives_and_polynomial_contraction(device):
    e = PlanarEvaluator("strip", 3, kernel="w6", method="shell_quadrature", device=device)
    n = np.array([2/3, -1/3, 2/3]); d = .1; eps = 2e-5
    value = e.evaluate(d, n, 1, half_widths=.2)[0]; ids = {r: i for i, r in enumerate(e.requests)}
    for j in range(3):
        diff = (e.evaluate(d+eps*n[j], n, 1, half_widths=.2)-e.evaluate(d-eps*n[j], n, 1, half_widths=.2))[0]/(2*eps)
        np.testing.assert_allclose(diff[0], value[ids[Request((0,0,0), (j,))]], atol=1e-8, rtol=1e-7)
    # A real polynomial integral can use a sparse degree-nine mask directly.
    req = [Request((0,0,0)), Request((0,0,3)), Request((3,3,3)), Request((2,2,2), (0,2))]
    a = PlanarEvaluator("strip", 3, req, kernel="w6", method="shell_quadrature", device=device)
    reference = PlanarEvaluator("strip", 3, req, kernel="w6", device=device)
    coeff = np.array([[1., .2, -.1, .3], [.4, -.2, .5, 1.]])
    x = a.evaluate(.1, n, 1, half_widths=.2)
    y = reference.evaluate(.1, n, 1, half_widths=.2)
    np.testing.assert_allclose(x@coeff.T, y@coeff.T, atol=1e-10, rtol=1e-10)
    loads = np.array([[.7, -.4]])
    np.testing.assert_allclose(np.sum((x@coeff.T)*loads), np.sum(x*(loads@coeff)), atol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_composed_device_function(device):
    e = PlanarEvaluator("strip", 3, [Request((0,0,0)), Request((0,0,9))], method="shell_quadrature", device=device)
    api = e.api; real, vec, Plan = api.real, api.vec, api.Plan
    element, output = api.element, api.output
    @wp.kernel
    def consumer(plan: Plan, indices: wp.array2d(dtype=int), facets: wp.array2d(dtype=vec),
                 work: wp.array2d(dtype=real), result: wp.array(dtype=real)):
        element(real(-.3), real(.1), vec(real(0), real(0), real(1)), plan, indices, facets, work, 0)
        result[0] = output(plan, work, 0, 0) + real(.2)*output(plan, work, 0, 1)
    b = e.prepare(.1, [0,0,1], 1, half_widths=.2); b.launch(); ref = b.numpy()[0]
    out = wp.empty(1, dtype=real, device=device)
    wp.launch(consumer, 1, inputs=[e.plan, e.indices, b.facets, b.work, out], device=device)
    np.testing.assert_allclose(out.numpy()[0], ref[0]+.2*ref[1], atol=1e-14)


def test_reject_unsupported_contracts():
    for kwargs in (dict(dimension=2), dict(precision="float32_compensated"), dict(nodes=7), dict(nodes=65), dict(kernel="bad")):
        args = dict(kind="plane", dimension=3, method="shell_quadrature"); args.update(kwargs)
        with pytest.raises(ValueError): PlanarEvaluator(**args)
    with pytest.raises(ValueError): functions(2, method="shell_quadrature")


@pytest.mark.parametrize("device", DEVICES)
def test_rotated_degree_nine_against_independent_mesh_reference(device):
    import mpmath as mp
    from warpSPHBoundaries.fluxbridge.planar import FACES
    from warpSPHBoundaries.fluxbridge.reference import evaluate
    normal = [2/3, -1/3, 2/3]
    req = [Request((3,3,3), (0,2)), Request((4,2,3), (2,)), Request((2,3,4))]
    old = PlanarEvaluator("strip", 3, req, kernel="w6")
    b = old.prepare(.1, normal, 1, half_widths=.2); b.launch()
    vertices = [None]*10
    for face, ids in zip(b.facets.numpy(), FACES):
        for point, i in zip(face, ids): vertices[i] = list(point)
    with mp.workdps(70): ref = np.array(evaluate(vertices, [0]*3, req, "w6", faces=FACES), float)
    q = PlanarEvaluator("strip", 3, req, kernel="w6", method="shell_quadrature", device=device)
    np.testing.assert_allclose(q.evaluate(.1, normal, 1, half_widths=.2)[0], ref, atol=1e-14, rtol=1e-14)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA hardware unavailable")
def test_shell_cuda_graph():
    e = PlanarEvaluator("strip", 3, precision="float64", method="shell_quadrature", device="cuda:0")
    b = e.prepare([0, .1, 2.], [0,0,1], 1, half_widths=.2); b.launch(); expected = b.numpy()
    with wp.ScopedCapture(device="cuda:0") as capture: b.launch()
    b.work.fill_(123)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(b.numpy(), expected)
