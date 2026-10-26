"""Reproduce local-model forward/VJP accuracy and warm batch costs.

Evidence defaults outside the repository. No original-geometry accuracy or
solver-throughput claim follows from this same-model comparison.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from warpSPHBoundaries.scene.curvature import evaluate
from warpSPHBoundaries.scene.implicitBodies import Tier3
from warpSPHBoundaries.validation.paths import evidence_path

NAMES = ('m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'Gx', 'Gy',
         'g1xx', 'g1xy', 'g1yx', 'g1yy')


def run(device, count, repeats, output):
    torch.set_num_threads(1)
    table = Tier3('w4', device)
    generator = torch.Generator().manual_seed(26016)
    angle = torch.rand(count, generator=generator, dtype=torch.float64)*6.28
    inputs = (torch.linspace(-.91, .91, count, dtype=torch.float64),
              torch.stack([angle.cos(), angle.sin()], 1),
              torch.rand(count, generator=generator, dtype=torch.float64)*.6-.3,
              torch.rand(count, generator=generator, dtype=torch.float64)*.4-.2)
    inputs = tuple(t.to(device) for t in inputs)
    actual, reference = evaluate(*inputs, table), table.curvature_moments(*inputs)
    torch.testing.assert_close(actual, reference, atol=5e-13, rtol=5e-12)
    errors = []
    x = tuple(t.detach().requires_grad_() for t in inputs)
    a, r = evaluate(*x, table), table.curvature_moments(*x)
    for j, name in enumerate(NAMES):
        ga = torch.autograd.grad(a[:, j].sum(), x, retain_graph=True)
        gr = torch.autograd.grad(r[:, j].sum(), x, retain_graph=True)
        for u, v in zip(ga, gr):
            torch.testing.assert_close(u, v, atol=5e-10, rtol=5e-9)
        errors.append(dict(moment=name, forward_abs=float((actual[:, j]-reference[:, j]).abs().max()),
                           vjp_abs=[float((u-v).abs().max()) for u, v in zip(ga, gr)]))
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
    def measure(fn):
        for _ in range(3):
            fn()
        sync()
        samples = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter_ns()
            fn()
            sync()
            samples.append((time.perf_counter_ns()-start)/1e6)
        return dict(median_ms=float(np.median(samples)), samples_ms=samples)
    costs = {}
    for name, fn in [('torch', lambda args: table.curvature_moments(*args)),
                     ('warp', lambda args: evaluate(*args, table))]:
        costs[name+'_forward'] = measure(lambda: fn(inputs))
        def backward():
            value = fn(x)
            return torch.autograd.grad(value.sum(), x)
        costs[name+'_forward_vjp'] = measure(backward)
    summary = dict(device=device, count=count, repeats=repeats, kernel='w4', dtype='float64',
                   torch_threads=torch.get_num_threads(), warp_version=wp.__version__,
                   torch_version=torch.__version__, platform=platform.platform(),
                   gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                   scope='one frozen table, independent local jets; wrapper allocations included; warm launches',
                   derivative_inputs=['d/h', 'normal components', 'h*kappa', 'h^2*kappa_s'],
                   errors=errors, costs=costs)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Reusable curvature model: forward and adjoints', '',
             f'Device: {device}; {count} jets; FP64; Wendland C4; {repeats} warm repeats; one CPU thread.', '',
             'Scope: wrapper allocations and launch overhead included. Backward includes a fresh forward',
             'and, for the Warp bridge, recomputation before its VJP. No Scene/guard/fallback costs.', '',
             'This compares the same existing order-two model and frozen spline coefficients.',
             'It does not reduce or bound original-geometry/model error.', '',
             '| Path | Median ms |', '|---|---:|']
    lines += [f'| {k} | {v["median_ms"]:.6f} |' for k, v in costs.items()]
    lines += ['', '| Moment | Forward max abs | VJP max abs (d, n, k, variation) |', '|---|---:|---|']
    lines += [f'| {e["moment"]} | {e["forward_abs"]:.3e} | '+', '.join(f'{v:.3e}' for v in e['vjp_abs'])+' |' for e in errors]
    lines += ['', 'Regression tests additionally cover four kernels, sparse requests, finite differences,',
              'empty/full branches, and an independent Warp plane consumer with position, support,',
              'offset and angle adjoints. These results are empirical checks, not certified bounds.', '',
              'Reproduce with `python scripts/studies/curvature_warp.py --device '+device+'`.', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'device': device, 'output': str(output), 'costs': costs,
                      'max_forward_abs': max(e['forward_abs'] for e in errors),
                      'max_vjp_abs': max(max(e['vjp_abs']) for e in errors)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--count', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=11)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.count < 2 or args.repeats < 1:
        parser.error('count must be at least two and repeats positive')
    output = args.output or evidence_path('curvature-warp-'+args.device.replace(':', '-'))
    run(args.device, args.count, args.repeats, output)
