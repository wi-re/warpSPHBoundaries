"""Many local jets against one frozen coefficient table, with a Torch VJP bridge.

No query-time Torch arithmetic beyond conversion/allocation: the same device
function can be called directly by an SPH interaction kernel. This evaluates
the existing local model; sampled geometry preparation and selection are not
part of this differentiable API.
"""
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable
import warp as wp

from .wp_func import CurvatureTable, _moment_function, channel_mask, wf

wp.set_module_options({'enable_backward': True, 'enable_fma': False})


def prepare_table(table):
    """Pack a Tier3 coefficient table for device calls (frozen coefficients).

    Keep the Tier3 owner alive. Reuse the returned struct in repeated launches;
    packing/preparation belongs outside graph capture and timed kernel loops.
    """
    wp.init()
    cached = getattr(table, '_warpCoefficients', None)
    if cached is None:
        cached = (torch.stack(table._coeff).contiguous(), torch.stack(table._cumulative).contiguous())
        table._warpCoefficients = cached
    result = CurvatureTable()
    for name, tensor in zip(('q', 'f', 'df', 'coefficients', 'cumulative'),
                            (table.q, table.F, table.dF, *cached)):
        setattr(result, name, wp.from_torch(tensor.detach(), dtype=wf, requires_grad=False))
    result.variance = table.variance
    return result


@lru_cache(None)
def _batch_kernel(mask):
    model = _moment_function(mask)

    @wp.kernel
    def batch(distance: wp.array(dtype=wf), normal: wp.array(dtype=wp.vec2d),
              curvature: wp.array(dtype=wf), variation: wp.array(dtype=wf),
              table: CurvatureTable, output: wp.array2d(dtype=wf)):
        i = wp.tid()
        value = model(distance[i], normal[i], curvature[i], variation[i], table)
        for j in range(12):
            output[i, j] = value[j]

    return batch


def _launch(inputs, packed, mask, differentiable=False):
    distance, normal, curvature, variation = inputs
    stream = wp.stream_from_torch(torch.cuda.current_stream(distance.device)) if distance.is_cuda else None
    # Gradient zero-initialization must use the same stream as the launch.
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        arrays = [wp.from_torch(t.detach(), dtype=wp.vec2d if i == 1 else wf,
                               requires_grad=differentiable) for i, t in enumerate(inputs)]
        output = torch.empty((len(distance), 12), dtype=torch.float64, device=distance.device)
        out = wp.from_torch(output, dtype=wf, requires_grad=differentiable)
        if len(distance):
            wp.launch(_batch_kernel(mask), len(distance), inputs=[*arrays, packed, out],
                      device=str(distance.device))
    return output, arrays, out


class _Moments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance, normal, curvature, variation, packed, mask):
        ctx.save_for_backward(distance, normal, curvature, variation)
        ctx.packed, ctx.mask = packed, mask
        return _launch((distance, normal, curvature, variation), packed, mask)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        inputs = ctx.saved_tensors
        if not len(inputs[0]):
            return (*(torch.zeros_like(t) for t in inputs), None, None)
        # Recompute the single batch launch; never retain a Tape across user
        # calls. The table is constant and each jet is an independent input.
        stream = wp.stream_from_torch(torch.cuda.current_stream(upstream.device)) if upstream.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            with wp.Tape() as tape:
                _, arrays, out = _launch(inputs, ctx.packed, ctx.mask, differentiable=True)
            tape.backward(grads={out: wp.from_torch(upstream.contiguous(), dtype=wf)})
            gradients = tuple(wp.to_torch(a.grad).clone() for a in arrays)
        return (*gradients, None, None)


def evaluate(distance, normal, curvature, variation, table, channels=None):
    """Evaluate dimensionless order-two moments, optionally with first-order VJPs.

    Inputs: d/h [N], unit normal [N,2], h*kappa [N], h²*kappa_s [N].
    Canonical columns: m00, m10, m01, m20, m11, m02, Gx, Gy,
    g1xx, g1xy, g1yx, g1yy, all at unit support. ``channels`` uses Scene's
    legacy IDs; unrequested columns are zero. Coefficients are frozen FP64.
    Normal components are independent for differentiation; compose a unit
    normal parameterization to obtain geometric angle derivatives.

    Preconditions: finite jets and a unit normal in the partial-support branch.
    This API has no geometry guard/fallback or accuracy guarantee. Derivatives
    hold the active spline/empty/full branch fixed, with no claim at switches;
    higher-order autograd is unsupported. See docs/warp-boundary-functions.md.
    """
    mask = channel_mask(channels)
    device = table.q.device
    inputs = tuple(torch.as_tensor(t, dtype=torch.float64, device=device).contiguous()
                   for t in (distance, normal, curvature, variation))
    n = inputs[0].numel()
    if tuple(t.shape for t in inputs) != ((n,), (n, 2), (n,), (n,)):
        raise ValueError('Expected distance/curvature/variation [N] and normal [N,2]')
    packed = prepare_table(table)
    if any(t.requires_grad for t in inputs):
        return _Moments.apply(*inputs, packed, mask)
    return _launch(inputs, packed, mask)[0]
