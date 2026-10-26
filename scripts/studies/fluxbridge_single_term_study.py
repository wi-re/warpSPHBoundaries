"""Accuracy/order/cost experiment for the one-kernel P1 W6 toy integral."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import warp as wp

from fluxbridge_single_term import tetrahedron_kernel, triangle_kernel
from fluxbridge_shell_acceptance import timed

ROOT = Path(__file__).resolve().parents[2]


def rule(n, device):
    x, w = np.polynomial.legendre.leggauss(n)
    return (wp.array((x+1)/2, dtype=wp.float32, device=device),
            wp.array(w/2, dtype=wp.float32, device=device))


def arrays(vertices, query, h, field, device):
    d = np.asarray(query).shape[1]
    vertices = np.pad(np.asarray(vertices, float), ((0, 0), (0, 0), (0, 3-d)))
    query = np.pad(np.asarray(query, float), ((0, 0), (0, 3-d)))
    return [wp.array(query, dtype=wp.vec3, device=device),
            wp.array(h, dtype=wp.float32, device=device),
            wp.array(vertices, dtype=wp.vec3, device=device),
            wp.array(field, dtype=wp.vec3 if d == 2 else wp.vec4, device=device)]


def launch(kernel, inputs, nodes, weights, output, device):
    wp.launch(kernel, output.shape[0], inputs=inputs+[nodes, weights, output], device=device)


def cases(d):
    result = []
    paths = [ROOT/'tests/fixtures/fluxbridge-v1.json']
    if d == 3:
        paths += [ROOT/'tests/fixtures/fluxbridge-tetquad-v1.json', ROOT/'tests/fixtures/fluxbridge-nearplane-v1.json']
    for path in paths:
        for i, c in enumerate(json.loads(path.read_text())['cases']):
            if c['kernel'] != 'w6' or c.get('dimension', 3) != d or c.get('measure', 'volume') != 'volume':
                continue
            if np.asarray(c['vertices']).shape != (d+1, d):
                continue
            c = dict(c, label=c.get('label', f'{path.stem}-{i}'))
            c['field'] = ['.35', '1.1', '-.2'] + (['.7'] if d == 3 else [])
            with mp.workdps(70):
                v = [[mp.mpf(str(t)) for t in p] for p in c['vertices']]
                q = [mp.mpf(str(t)) for t in c['query']]
                f = list(map(mp.mpf, c['field']))
                matrix = mp.matrix([[v[i+1][j]-v[0][j] for j in range(d)] for i in range(d)])
                grad = mp.lu_solve(matrix, mp.matrix([f[i+1]-f[0] for i in range(d)]))
                m = {}
                for request, value in zip(c['requests'], c['values']):
                    if not request['derivative']:
                        m[tuple(request['alpha'])] = mp.mpf(value)
                center = f[0] + sum(grad[j]*(q[j]-v[0][j]) for j in range(d))
                c['reference'] = float(center*m[(0,)*d] + sum(grad[j]*m[tuple(int(k == j) for k in range(d))] for j in range(d)))
            result.append(c)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dimension', type=int, choices=(2, 3), default=3)
    p.add_argument('--orders', default='4,6,8,12,16,24,32')
    p.add_argument('--counts', default='128,8192,32768,131072')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--accuracy-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    orders = list(map(int, a.orders.split(',')))
    counts = list(map(int, a.counts.split(',')))
    if not all(2 <= n <= 64 for n in orders) or not all(1 <= n <= 1048576 for n in counts):
        p.error('invalid bounds')
    if a.profile and (len(orders) != 1 or len(counts) != 1):
        p.error('profile exactly one order and count')
    a.output.mkdir(parents=True, exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock', 'a') as lock:
        if a.device.startswith('cuda'):
            fcntl.flock(lock, fcntl.LOCK_EX)
        wp.init(); device = wp.get_device(a.device)
        d = a.dimension
        make = triangle_kernel if d == 2 else tetrahedron_kernel
        paths = [Path(__file__), ROOT/'scripts/studies/fluxbridge_single_term.py']
        paths += list((ROOT/'tests/fixtures').glob('fluxbridge-*.json'))
        report = dict(device=str(device), device_name=device.name, dimension=d, precision='float32',
            output='Integral over simplex of W6(|y-x|,h) times P1 nodal scalar field',
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            accuracy=[], benchmarks=[], limitations=['Toy fixed-order volume quadrature; no adaptive error certification or production fallback',
                'One thread/pair; all geometry, FEM interpolation, quadrature and summation happen in one kernel',
                'Input-independent 1D Gauss table uploaded outside timing; per-pair sample/profile preparation does not exist',
                'Benchmarks use the old local .2 query / twice-unit simplex scene with varied positions and P1 fields',
                'Unprofiled timings reuse input/output buffers; transfer timing includes new upload and readback; JIT is excluded',
                'No search, assembly, multiple outputs, derivative or high-order FEM contract',
                'Existing high-precision fixture moments independently determine P1 reference; errors include fp32 input rounding'])
        def save():
            (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not a.profile:
            cs = cases(d)
            inputs = arrays([c['vertices'] for c in cs], [c['query'] for c in cs], [c['h'] for c in cs], [c['field'] for c in cs], device)
            out = wp.empty(len(cs), dtype=wp.float32, device=device)
            for n in orders:
                nodes, weights = rule(n, device)
                launch(make(n), inputs, nodes, weights, out, device)
                values = out.numpy()
                for c, value in zip(cs, values):
                    error = float(abs(value-c['reference'])/(1+abs(c['reference'])))
                    report['accuracy'].append(dict(label=c['label'], order=n, samples=n**d,
                        value=float(value), reference=c['reference'], mixed_error=error, passes_1e5=error <= 1e-5))
                save()
                print('accuracy', d, n, 'worst', max(x['mixed_error'] for x in report['accuracy'] if x['order'] == n), flush=True)
        if not a.accuracy_only:
            for n in orders:
                kernel = make(n)
                nodes, weights = rule(n, device)
                for count in counts:
                    rng = np.random.default_rng(852)
                    q = .2+rng.uniform(-.015, .015, (count, d))
                    v = np.broadcast_to(np.vstack([np.zeros(d), 2*np.eye(d)]), (count, d+1, d)).copy()
                    f = np.broadcast_to(np.array([.35, 1.1, -.2]+([.7] if d == 3 else [])), (count, d+1)).copy()
                    h = np.ones(count)
                    inputs = arrays(v, q, h, f, device)
                    out = wp.empty(count, dtype=wp.float32, device=device)
                    action = lambda: launch(kernel, inputs, nodes, weights, out, device)
                    action(); wp.synchronize_device(device)
                    if a.profile:
                        wp.cuda_profiler_start(device); action(); wp.synchronize_device(device); wp.cuda_profiler_stop(device)
                    else:
                        _, warm = timed(action, device, 7, events=device.is_cuda)
                        def transferred():
                            ins = arrays(v, q, h, f, device)
                            dst = wp.empty(count, dtype=wp.float32, device=device)
                            launch(kernel, ins, nodes, weights, dst, device)
                            return dst.numpy()
                        values, complete = timed(transferred, device, 3)
                        assert np.isfinite(values).all()
                        report['benchmarks'].append(dict(order=n, pairs=count, samples_per_pair=n**d,
                            warm=warm, upload_kernel_readback=complete,
                            input_output_bytes=sum(x.capacity for x in inputs)+out.capacity,
                            rule_bytes=nodes.capacity+weights.capacity,
                            warm_pairs_per_second=count*1000/warm['median_ms']))
                    save()
                    print('benchmark', d, n, count, flush=True)
        save()


if __name__ == '__main__':
    main()
