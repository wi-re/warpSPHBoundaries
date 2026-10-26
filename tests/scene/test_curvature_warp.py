"""Composable local model: forward, full-channel VJPs and fused-consumer adjoints."""
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.scene.curvature import evaluate, moment_function, prepare_table
from warpSPHBoundaries.scene.curvature.wp_func import CANONICAL, CurvatureTable, wf
from warpSPHBoundaries.scene.implicitBodies import Tier3

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
PROFILES = [None, (0,), (1, 2), (3, 4), (5, 6, 7, 8), (9, 10, 11), ()]


def jets(device, grad=False):
    values = [np.array([-.7231, -.2173, .1371, .5679]),
              np.array([[.6, .8], [-.8, .6], [1., 0.], [0., -1.]]),
              np.array([-.21, 0., .17, .31]), np.array([.12, -.18, .08, 0.])]
    return tuple(torch.tensor(a, dtype=torch.float64, device=device, requires_grad=grad) for a in values)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kernel', ['cubic', 'w2', 'w4', 'w6'])
@pytest.mark.parametrize('profile', PROFILES)
def test_batched_forward_and_every_requested_channel_adjoint(device, kernel, profile):
    table = Tier3(kernel, device)
    inputs = jets(device, True)
    expected = table.curvature_moments(*inputs)
    actual = evaluate(*inputs, table, channels=profile)
    selected = set(range(12)) if profile is None else {CANONICAL[c] for c in profile}
    for j in range(12):
        if j not in selected:
            assert not actual[:, j].any()
            continue
        torch.testing.assert_close(actual[:, j], expected[:, j], atol=3e-13, rtol=3e-12)
        weight = torch.tensor([.3, -.7, 1.1, -.9], dtype=torch.float64, device=device)
        got = torch.autograd.grad((actual[:, j]*weight).sum(), inputs, retain_graph=True)
        want = torch.autograd.grad((expected[:, j]*weight).sum(), inputs, retain_graph=True)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=3e-10)
    # Sparse requests must differentiate to zero through all masked columns.
    masked = [j for j in range(12) if j not in selected]
    if masked:
        got = torch.autograd.grad(actual[:, masked].sum(), inputs)
        assert all(not g.any() for g in got)


@pytest.mark.parametrize('device', DEVICES)
def test_vjp_matches_finite_differences_and_repeated_backward_accumulates_once(device):
    table = Tier3('w4', device)
    inputs = jets(device, True)
    weight = torch.linspace(-.8, 1.1, 48, dtype=torch.float64, device=device).reshape(4, 12)
    loss = (evaluate(*inputs, table)*weight).sum()
    actual = torch.autograd.grad(loss, inputs, retain_graph=True)
    # An independent direction through all local jet inputs, including angle.
    directions = tuple(torch.linspace(-.3, .7, t.numel(), dtype=t.dtype, device=device).reshape(t.shape)
                       for t in inputs)
    epsilon = 2e-6
    def shifted(sign):
        x = tuple(t.detach()+sign*epsilon*v for t, v in zip(inputs, directions))
        return (evaluate(*x, table)*weight).sum().item()
    fd = (shifted(1)-shifted(-1))/(2*epsilon)
    derivative = sum((g*v).sum().item() for g, v in zip(actual, directions))
    assert derivative == pytest.approx(fd, abs=2e-9, rel=2e-8)
    loss.backward(retain_graph=True)
    loss.backward()
    for t, g in zip(inputs, actual):
        torch.testing.assert_close(t.grad, 2*g, atol=3e-12, rtol=3e-11)


plane_moments = moment_function()


@wp.kernel
def plane_consumer(positions: wp.array(dtype=wp.vec2d), supports: wp.array(dtype=wf),
                   offset: wp.array(dtype=wf), angle: wp.array(dtype=wf),
                   table: CurvatureTable, losses: wp.array(dtype=wf)):
    """A consumer fuses geometry, physical scaling and a weighted moment loss."""
    i = wp.tid()
    normal = wp.vec2d(wp.cos(angle[0]), wp.sin(angle[0]))
    h = supports[i]
    distance = (wp.dot(positions[i], normal)-offset[0])/h
    values = plane_moments(distance, normal, wf(0.), wf(0.), table)
    result = wf(0.)
    for j in range(12):
        scale = wf(1.)
        if j == 1 or j == 2:
            scale = h
        elif j >= 3 and j <= 5:
            scale = h*h
        elif j == 6 or j == 7:
            scale = wf(1.)/h
        result += wf(j+1)*values[j]*scale
    losses[i] = result


@pytest.mark.parametrize('device', DEVICES)
def test_external_warp_consumer_position_support_and_boundary_parameter_adjoints(device):
    table = Tier3('w4', device)
    packed = prepare_table(table)
    args = [torch.tensor(a, dtype=torch.float64, device=device, requires_grad=True) for a in
            ([[.23, .07], [-.27, .31], [.48, -.18]], [.8, 1.1, .9], [.03], [.41])]
    arrays = [wp.from_torch(t.detach(), dtype=wp.vec2d if i == 0 else wf, requires_grad=True)
              for i, t in enumerate(args)]
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device != 'cpu' else None
    with wp.ScopedStream(stream):
        output = wp.zeros(3, dtype=wf, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(plane_consumer, 3, inputs=[*arrays, packed, output], device=device)
        tape.backward(grads={output: wp.ones(3, dtype=wf, device=device)})
    normal = torch.stack([args[3][0].cos(), args[3][0].sin()])
    d = ((args[0]*normal).sum(1)-args[2][0])/args[1]
    expected = table.curvature_moments(d, normal.expand(3, 2), torch.zeros_like(d), torch.zeros_like(d))
    h = args[1]
    scale = torch.stack([torch.ones_like(h), h, h, h*h, h*h, h*h, 1/h, 1/h,
                         torch.ones_like(h), torch.ones_like(h), torch.ones_like(h), torch.ones_like(h)], 1)
    weight = torch.arange(1, 13, dtype=torch.float64, device=device)
    expected_loss = (expected*scale*weight).sum(1)
    torch.testing.assert_close(wp.to_torch(output), expected_loss, atol=3e-12, rtol=3e-12)
    gradients = torch.autograd.grad(expected_loss.sum(), args)
    for array, gradient in zip(arrays, gradients):
        torch.testing.assert_close(wp.to_torch(array.grad), gradient, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize('device', DEVICES)
def test_empty_full_support_and_empty_batches_have_constant_adjoint(device):
    table = Tier3('w4', device)
    for n in (0, 4):
        d = torch.tensor([-1.3, -1.1, 1.1, 1.3][:n], dtype=torch.float64, device=device, requires_grad=True)
        normal = torch.tensor([[1., 0.]]*n, dtype=torch.float64, device=device).reshape(n, 2).requires_grad_()
        k = torch.full_like(d, .2, requires_grad=True)
        b = torch.full_like(d, .1, requires_grad=True)
        result = evaluate(d, normal, k, b, table)
        assert result.shape == (n, 12)
        torch.testing.assert_close(result, table.curvature_moments(d, normal, k, b))
        gradients = torch.autograd.grad(result.sum(), (d, normal, k, b))
        assert all(not g.any() for g in gradients)


def test_input_shapes_and_channels_are_checked():
    table = Tier3('w4', 'cpu')
    x = jets('cpu')
    with pytest.raises(ValueError, match='Expected'):
        evaluate(x[0][:, None], *x[1:], table)
    with pytest.raises(ValueError, match='channels'):
        evaluate(*x, table, channels=(12,))


@pytest.mark.parametrize('device', DEVICES)
def test_near_origin_linear_extension_and_near_support_gradients(device):
    table = Tier3('w4', device)
    d = torch.tensor([-1+2e-7, -1.2e-6, -.3e-6, .3e-6, 1.2e-6, 1-2e-7],
                     dtype=torch.float64, device=device, requires_grad=True)
    n = torch.tensor([[.6, .8]]*len(d), dtype=d.dtype, device=device, requires_grad=True)
    k = torch.full_like(d, .1, requires_grad=True)
    b = torch.full_like(d, -.07, requires_grad=True)
    x = (d, n, k, b)
    actual, expected = evaluate(*x, table), table.curvature_moments(*x)
    torch.testing.assert_close(actual, expected, atol=4e-13, rtol=4e-12)
    for j in range(12):
        ga = torch.autograd.grad(actual[:, j].sum(), x, retain_graph=True)
        gb = torch.autograd.grad(expected[:, j].sum(), x, retain_graph=True)
        for a, e in zip(ga, gb):
            torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream/capture validation requires GPU')
def test_nondefault_torch_stream_forward_backward_and_live_graph_inputs():
    table = Tier3('w4', 'cuda:0')
    evaluate(*jets('cuda:0'), table)  # prepare coefficients and compile before capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = jets('cuda:0', True)
        result = evaluate(*x, table)
        actual = torch.autograd.grad(result.square().sum(), x)
        expected_value = table.curvature_moments(*x)
        expected = torch.autograd.grad(expected_value.square().sum(), x)
    torch.cuda.current_stream().wait_stream(stream)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-10)
    detached = tuple(t.detach().clone() for t in x)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            captured = evaluate(*detached, table)
    torch.cuda.current_stream().wait_stream(stream)
    detached[0].add_(.013)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, table.curvature_moments(*detached), atol=3e-13, rtol=3e-12)
