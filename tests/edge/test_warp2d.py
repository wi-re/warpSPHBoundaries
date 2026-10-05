"""Stage 5: Warp kernels (CPU and CUDA), float64, explicit adjoints, torch bridge."""
import numpy as np
import pytest
import torch
import warp as wp

from edgebound.edge import np2d, torch2d, warp2d
from .test_np2d import ELEMENTS, KERNELS, MOMS, ref, rel_geometry

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
TRI = [c for c in ELEMENTS]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", KERNELS)
def test_fixtures_value_gradient_moments(device, k):
    V, X, H = rel_geometry(TRI)
    val = warp2d.moment(V, X, k, (0, 0), H, device=device)
    gr = warp2d.moment_gradient(V, X, k, (0, 0), H, device=device)
    for i, c in enumerate(TRI):
        assert abs(val[i] - ref(c, k, "value")) < 1e-12, c["id"]
        assert np.abs(gr[i] - ref(c, k, "grad")).max() * H[i] < 1e-11, c["id"]
    for al in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (3, 0), (2, 2), (0, 4)]:
        m = warp2d.moment(V, X, k, al, H, device=device)
        for i, c in enumerate(TRI):
            assert abs(m[i] - ref(c, k, "m", f"{al[0]},{al[1]}")) / H[i] ** sum(al) < 1e-12, (c["id"], al)


@pytest.mark.parametrize("device", DEVICES)
def test_matches_numpy_and_torch_random_batch(device):
    rng = np.random.default_rng(5)
    V = rng.uniform(-1, 1, (500, 3, 2))
    X = rng.uniform(-.4, .4, (500, 2))
    for k in ["cubic", "w6"]:
        tol = 5e-12 if k == "w6" else 1e-13
        assert np.abs(warp2d.moment(V, X, k, (0, 0), device=device) - np2d.value(V, X, k)).max() < tol
        assert np.abs(warp2d.shape_gradient(V, X, k, (1, 1), device=device) - np2d.shape_gradient(V, X, k, (1, 1))).max() < 10 * tol
        assert np.abs(warp2d.moment_gradient(V, X, k, (2, 1), device=device) - np2d.moment_gradient(V, X, k, (2, 1))).max() < 10 * tol


@pytest.mark.parametrize("device", DEVICES)
def test_orientation_and_scaling(device):
    T = np.array([[[-0.3, -0.2], [0.6, -0.1], [0.1, 0.7]]])
    x = np.array([[0.05, 0.1]])
    a = warp2d.shape_gradient(T, x, "w4", device=device)
    b = warp2d.shape_gradient(T[:, ::-1].copy(), x, "w4", device=device)[:, ::-1]
    assert np.abs(a - b).max() < 1e-13
    h = 1.7
    assert np.abs(warp2d.moment(T * h, x * h, "w4", (1, 1), h, device=device) - h ** 2 * warp2d.moment(T, x, "w4", (1, 1), device=device)).max() < 1e-13


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", ["cubic", "w4"])
def test_torch_bridge_custom_adjoint_equals_torch_autograd(device, k):
    rng = np.random.default_rng(9)
    V = rng.uniform(-1, 1, (40, 3, 2))
    X = rng.uniform(-.4, .4, (40, 2))
    for al in [(0, 0), (1, 1), (2, 0)]:
        Vt = torch.tensor(V, requires_grad=True)
        Xt = torch.tensor(X, requires_grad=True)
        w = warp2d.TorchMoment.apply(Vt, Xt, k, al, 1.0, device)
        (w * torch.linspace(0.5, 1.5, 40, dtype=torch.float64)).sum().backward()
        Vt2 = torch.tensor(V, requires_grad=True)
        Xt2 = torch.tensor(X, requires_grad=True)
        w2 = torch2d.moment(Vt2, Xt2, k, al)
        (w2 * torch.linspace(0.5, 1.5, 40, dtype=torch.float64)).sum().backward()
        assert (w.detach() - w2.detach()).abs().max() < 1e-12
        # warp's analytic d/dx of a moment is  -sum_e n int y^alpha W ds (the h^(k-1) moment gradient); torch autograd of m_alpha(x) w.r.t.
        # x equals it (y = x' - x enters W only through |y| ... and y^alpha): compare directly
        assert (Vt.grad - Vt2.grad).abs().max() < 1e-10
        assert (Xt.grad - Xt2.grad).abs().max() < 1e-10


def test_throughput_report(capsys):
    """not a pass/fail benchmark: records the speed so the numbers in the docs can be regenerated (python scripts/bench/warp_bench.py)."""
    rng = np.random.default_rng(1)
    V = rng.uniform(-1, 1, (20000, 3, 2))
    X = rng.uniform(-.4, .4, (20000, 2))
    out = warp2d.moment(V, X, "w4", (0, 0), device=DEVICES[-1])
    assert np.isfinite(out).all()
