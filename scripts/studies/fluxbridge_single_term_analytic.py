"""Measure the closed-form path for the same single P1 W6 scalar toy term.

No quadrature, no query-dependent integration preparation. The current shared
analytic DAG interpreter and fused field contraction are measured as implemented;
their cost is not an estimate of an optimized straight-line formula.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry, PolynomialField, FieldEvaluator
from fluxbridge_single_term_study import cases, ROOT
from fluxbridge_shell_acceptance import timed


def provider(vertices, values, device, compact, precision):
    vertices = np.asarray(vertices, float); values = np.asarray(values, float)
    d = vertices.shape[1]
    grad = np.linalg.solve(vertices[1:]-vertices[0], values[1:]-values[0])
    coefficients = {(0,)*d: np.array(values[0])}
    for j in range(d):
        coefficients[tuple(int(k == j) for k in range(d))] = np.array(grad[j])
    field = PolynomialField(Geometry([vertices], d), [coefficients])
    return FieldEvaluator(field, kernel='w6', precision=precision, device=device,
        method='analytic', compact=compact, search='device' if device.is_cuda else 'host')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dimension', type=int, choices=(2, 3), default=3)
    p.add_argument('--precision', choices=('float64', 'float32', 'float32_compensated'), default='float64')
    p.add_argument('--counts', default='128,512,8192,131072')
    p.add_argument('--exhaustive-flux', action='store_true')
    p.add_argument('--accuracy-only', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); counts = list(map(int, a.counts.split(',')))
    if not all(1 <= n <= 131072 for n in counts) or (a.profile and len(counts) != 1):
        p.error('invalid study counts')
    a.output.mkdir(parents=True, exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock', 'a') as lock:
        if a.device.startswith('cuda'):
            fcntl.flock(lock, fcntl.LOCK_EX)
        wp.init(); device = wp.get_device(a.device); d = a.dimension
        paths = list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
        paths += [Path(__file__), ROOT/'scripts/studies/fluxbridge_single_term_study.py', ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
        report = dict(device=str(device), device_name=device.name, precision=a.precision, dimension=d,
            compact=not a.exhaustive_flux, accuracy=[], rows=[],
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            limitations=['Same scalar P1 FEM-weighted W6 volume integral as the direct quadrature toy',
                'Existing shared analytic DAG interpreter; not an optimized specialized formula',
                'Warm timings reuse static geometry, field coefficients, plan, adjacency and scratch, recomputing analytic integrals each action',
                'Complete timings include validation, search, allocation, evaluation and readback; static rational plan construction and JIT excluded',
                'GPU clock/cache and desktop activity uncontrolled; profiler timing separate from ordinary timings',
                'FLOP instruction counts exclude division, square root, logarithm and atan2 cost',
                'Raw analytic fp32 remains experimental; no accepted precision tradeoff or error budget changed'])
        def save():
            (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not a.profile:
            for c in cases(d):
                ev = provider(c['vertices'], c['field'], device, report['compact'], a.precision)
                value = float(ev.evaluate([c['query']], c['h'])[0])
                finite = bool(np.isfinite(value))
                error = abs(value-c['reference'])/(1+abs(c['reference'])) if finite else None
                report['accuracy'].append(dict(label=c['label'], value=value if finite else None,
                    finite=finite, reference=c['reference'], mixed_error=error))
            save()
            print('accuracy worst finite', max((x['mixed_error'] for x in report['accuracy'] if x['mixed_error'] is not None), default=None),
                'nonfinite', sum(not x['finite'] for x in report['accuracy']), flush=True)
        if not a.accuracy_only:
            vertices = np.vstack([np.zeros(d), 2*np.eye(d)])
            values = [.35, 1.1, -.2]+([.7] if d == 3 else [])
            ev, static = timed(lambda: provider(vertices, values, device, report['compact'], a.precision), device, 1)
            for n in counts:
                rng = np.random.default_rng(852); q = .2+rng.uniform(-.015, .015, (n, d))
                batch, preparation = timed(lambda: ev.prepare(q, 1), device, 1)
                batch.launch(); wp.synchronize_device(device)
                assert batch.count == n
                plan = ev.evaluator.host_plan
                row = dict(pairs=n, static_setup=static, preparation=preparation, batch_owned_bytes=batch.bytes,
                    scratch_bytes=batch.work.capacity,
                    plan=dict(face_nodes=len(plan.face_nodes), boundary_nodes=len(plan.boundaries), volume_nodes=len(plan.volumes),
                        moment_outputs=len(plan.outputs), terms=sum(map(len, plan.outputs)),
                        face_metadata=[dict(alpha=list(x[0]), radial_power=x[1], radius=str(x[2])) for x in plan.face_nodes]))
                if a.profile:
                    wp.cuda_profiler_start(device); batch.launch(); wp.synchronize_device(device); wp.cuda_profiler_stop(device)
                else:
                    _, row['warm'] = timed(batch.launch, device, 7, events=device.is_cuda)
                    _, row['complete'] = timed(lambda: ev.evaluate(q, 1), device, 3)
                    if device.is_cuda:
                        with wp.ScopedTimer('analytic kernels', print=False, cuda_filter=wp.TIMING_KERNEL) as timer:
                            batch.launch()
                        row['kernel_events'] = [dict(name=r.name, elapsed_ms=r.elapsed) for r in timer.timing_results]
                    assert np.isfinite(batch.numpy()).all()
                report['rows'].append(row); save()
                print(d, a.precision, 'compact', report['compact'], n, row.get('warm'), flush=True)
        save()


if __name__ == '__main__':
    main()
