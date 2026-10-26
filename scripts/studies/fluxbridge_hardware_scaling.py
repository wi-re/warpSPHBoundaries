"""Serialized scaling and externally profileable native fp32 term evaluations.

No operator or resource-budget changes. --profile restricts Nsight collection to
one warmed action via CUDA profiler start/stop; ordinary timings are separate.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry, Evaluator
from fluxbridge_term_throughput import masks, ROOT
from fluxbridge_shell_acceptance import timed


def queries(n, d):
    rng = np.random.default_rng(852)
    q = .2 + rng.uniform(-.015, .015, (n, d))
    q[:, 0] += 4 * (np.arange(n) % 16)
    return q


def activities(call, device):
    # Warp records CUDA events around individual operations, independently of
    # the events enclosing the complete submitted action. Instrumented timings
    # are reported separately, not substituted for ordinary timing samples.
    with wp.ScopedTimer('activities', print=False, cuda_filter=wp.TIMING_ALL) as timer:
        call()
    wp.synchronize_device(device)
    rows = []
    for r in timer.timing_results:
        rows.append(dict(name=r.name, elapsed_ms=r.elapsed, activity=r.filter))
    return rows


def run(a):
    wp.init()
    device = wp.get_device('cuda:0')
    assert device.is_cuda
    a.output.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/studies/fluxbridge_term_throughput.py',
              ROOT/'scripts/studies/fluxbridge_shell_acceptance.py']
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device=device.name, warp=wp.__version__, precision='float32', kernel='w6',
        dimension=a.dimension, nodes=16 if a.dimension == 2 else 8,
        rows=[], source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['One selected volume element per varied query, interior scene; not solver throughput',
            'Static geometry and plans reused; complete action includes preparation, convergence checks and readback',
            'Warm action recomputes integrals; query-dependent preparation reused',
            'Streamed actions preserve the existing per-rule storage cap; chunk size explicitly reported',
            'Individual CUDA activity event timings are instrumented and separate from ordinary samples',
            'Graph replay reported separately; no automatic graph or streaming fallback',
            'Desktop GPU activity is not isolated; profiler replay times are separate'])
    def save():
        (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    d = a.dimension
    geo = Geometry([np.vstack([np.zeros(d), 2*np.eye(d)]) + np.array([4*i]+[0]*(d-1)) for i in range(16)], d)
    for term in a.terms.split(','):
        ev = Evaluator(geo, masks(d)[term], kernel='w6', precision='float32',
            method='conditioned_boundary' if d == 2 else 'tet_quadrature', device=device,
            preparation='host' if d == 2 else 'device', search='device', nodes=report['nodes'])
        # Compile every kernel before external profiling or steady timing.
        first = ev.prepare(queries(16, d), 1)
        first.launch(); wp.synchronize_device(device)
        reference = first.numpy().copy()
        del first
        for n in map(int, a.counts.split(',')):
            q = queries(n, d)
            row = dict(term=term, queries=n, outputs=len(ev.host_plan.requests))
            try:
                b, prep = timed(lambda: ev.prepare(q, 1), device, 1)
            except ValueError as error:
                if 'storage exceeds' not in str(error):
                    raise
                row.update(status='single_batch_rejected', reason=str(error))
                if a.profile:
                    raise
            else:
                assert b.count == n
                b.launch(); wp.synchronize_device(device)
                values = b.numpy()
                err = float(np.max(abs(values[:16]-reference)/(1+abs(reference))))
                assert np.isfinite(values).all() and err <= 1e-5
                row.update(status='passed', batch_owned_bytes=b.bytes, preparation=prep,
                    quadrature=getattr(b, 'quadrature', None), prefix_mixed_error=err)
                if a.profile:
                    wp.cuda_profiler_start(device)
                    if a.profile == 'warm':
                        b.launch()
                    else:
                        ev.evaluate(q, 1)
                    wp.synchronize_device(device)
                    wp.cuda_profiler_stop(device)
                    row['profile_region'] = a.profile
                else:
                    _, row['warm'] = timed(b.launch, device, a.repeats, events=True)
                    with wp.ScopedCapture(device=device) as capture:
                        b.launch()
                    graph_call = lambda: wp.capture_launch(capture.graph)
                    graph_call(); wp.synchronize_device(device)
                    _, row['graph'] = timed(graph_call, device, a.repeats, events=True)
                    row['warm_activities'] = activities(b.launch, device)
                    del capture, graph_call
                del b, values
            if not a.profile:
                chunk = min(n, a.chunk_size) if d == 3 else n
                result, complete = timed(lambda: ev.evaluate_streamed(q, 1, chunk_size=chunk, return_metrics=True), device, a.repeats)
                values, execution = result
                assert execution['total_interactions'] == n
                assert np.isfinite(values).all()
                row.update(complete=complete, execution=execution)
                # One separate instrumented complete call to expose preparation
                # kernels and host gaps, without polluting steady timings.
                if n in (int(a.counts.split(',')[0]), min(512, int(a.counts.split(',')[-1]))):
                    row['complete_activities'] = activities(lambda: ev.evaluate_streamed(q, 1, chunk_size=chunk), device)
                del result, values
            report['rows'].append(row); save()
            print(d, term, n, row['status'], 'warm', row.get('warm', {}).get('median_ms'),
                'complete', row.get('complete', {}).get('median_ms'), flush=True)
            gc.collect()
        del ev
        gc.collect()
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension', type=int, choices=(2, 3), required=True)
    p.add_argument('--counts', default='128,512,2048,8192,32768')
    p.add_argument('--terms', default='occupancy,gradient,first_second,first_gradient,hessian,laplacian_inputs,baseline')
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--profile', choices=('warm', 'complete'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.repeats <= 10 or not 1 <= a.chunk_size <= 512 or any(not 16 <= int(n) <= 1048576 for n in a.counts.split(',')):
        p.error('invalid study bounds')
    if a.profile and (len(a.terms.split(',')) != 1 or len(a.counts.split(',')) != 1):
        p.error('profile one count and term per invocation')
    with open('/tmp/fluxbridge-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(a)
