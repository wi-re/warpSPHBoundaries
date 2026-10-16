"""Warm evaluation costs for the representative curved-wall comparison.

Benchmarks existing implementations, not optimized algorithmic lower bounds.
No compiler/model changes. Default evidence is outside the repository.
"""
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

# Import the package before Core so its documented float64 default is applied.
from warpSPHBoundaries.scene.scene import Body, Scene, SdfRep, SurfaceRep
from warpSPHBoundaries.edge.precision import IS_F64
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         ParticleState, WarpOperation)
from warpSPHBoundaries.edge.disk import disk_channels
from warpSPHBoundaries.validation.paths import evidence_path


def timed(fn, synchronize, repeats, warmups=2):
    result = None
    for _ in range(warmups):
        result = fn()
    synchronize()
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter_ns()
            result = fn()
            synchronize()
            samples.append((time.perf_counter_ns()-start)/1e6)
    finally:
        if was_enabled:
            gc.enable()
    return dict(median_ms=float(np.median(samples)), min_ms=min(samples),
                p10_ms=float(np.quantile(samples, .1)), p90_ms=float(np.quantile(samples, .9)),
                samples_ms=samples), result


def check_moments(result, count):
    mass = torch.zeros(count, dtype=torch.float64)
    first = torch.zeros((count, 2), dtype=torch.float64)
    for entry in result.entries:
        for moments in entry['surface']+entry['implicit']:
            assert moments.m1 is not None
            mass.index_add_(0, moments.q.cpu().long(), moments.lam.detach().cpu())
            first.index_add_(0, moments.q.cpu().long(), moments.m1.detach().cpu())
    assert torch.isfinite(mass).all() and torch.isfinite(first).all()
    return dict(mass_sum=float(mass.sum()), first_sum=float(first[:, 1].sum()))


def gpu_graph_cost(scene, rep, ps, props, device, repeats, channels=None, route='mono'):
    """Separate captured-device study: guarded SDF core / fused polygon path."""
    import warp as wp
    wp.init()
    from warpSPHBoundaries.scene.scene import _sdfCurvature
    from warpSPHBoundaries.scene.fixedadj import fixed_adjacency
    from warpSPHBoundaries.scene.fused import FusedWall, WallOutput
    from warpSPHBoundaries.edge.warpfused import FusedGroup
    if isinstance(rep, SdfRep):
        def evaluate():
            return _sdfCurvature(rep, ps.positions, ps.supports, 'w4', device, channels)
        backend = 'guarded SDF core'
    else:
        outputs = (WallOutput('lam', 0, 'lam'), WallOutput('m1', 0, 'm1'),
                   WallOutput('G', 0, 'g0'), WallOutput('Cov', 0, 'cov'), WallOutput('m2', 0, 'm2'))
        if channels is not None:
            outputs = outputs[:2]
        def evaluate():
            wall = FusedWall(scene, fixed_adjacency(scene, ps, props, 1.),
                             (FusedGroup('w4', channels=channels, route=route),))
            return wall.evaluate(outputs)
        backend = f'FusedWall polygon ({route}), adjacency plus output assembly'
    evaluate()
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with torch.cuda.graph(graph, stream=stream):
            captured = evaluate()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize(device)
    if isinstance(rep, SdfRep):
        assert not captured[1].any()
        assert torch.isfinite(captured[0]).all()
    else:
        assert all(torch.isfinite(v).all() for v in captured.values())
    pilot_start, pilot_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    pilot_start.record()
    graph.replay()
    pilot_end.record()
    pilot_end.synchronize()
    replays = 1 if pilot_start.elapsed_time(pilot_end) > 2. else 10
    samples = []
    for _ in range(max(repeats, 15)):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/replays)
    return dict(backend=backend, median_ms=float(np.median(samples)),
                p10_ms=float(np.quantile(samples, .1)), p90_ms=float(np.quantile(samples, .9)),
                samples_ms=samples, replays_per_sample=replays)


def run(output, device, counts, repeats, sdf_backend='warp', methods=None, skip_arc=False, graph_route='mono'):
    if not IS_F64:
        raise RuntimeError('This benchmark requires float64 kernels: set warpSPHCore_PRECISION=float64 before Python starts')
    output.mkdir(parents=True, exist_ok=True)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.startswith('cuda') else (lambda: None)
    props = OperationProperties(kernel='w4', operation=WarpOperation.Density,
                                gradientMode=GradientScheme.Naive,
                                operationMode=OperationDirection.BoundaryToFluid)
    hardware = dict(platform=platform.platform(), processor=platform.processor(),
                    torch=torch.__version__, numpy=np.__version__,
                    cpu_threads=torch.get_num_threads(), device=device, edge_kernel_precision='float64', sdf_backend=sdf_backend,
                    fused_polygon_route=graph_route)
    if device.startswith('cuda'):
        hardware['gpu'] = torch.cuda.get_device_name(device)
    try:
        hardware['cpu_model'] = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    records, graph_records = [], []
    channel_sets = [('mass_first', (0, 1, 2)), ('all_twelve', None)]
    for count in counts:
        ds = np.resize(np.linspace(-.95, .95, 81), count)
        points = np.column_stack((np.zeros(count), 3+ds))
        x = torch.tensor(points, dtype=torch.float64, device=device)
        one = torch.ones(count, dtype=torch.float64, device=device)
        ps = ParticleState(positions=x, supports=one, masses=one, densities=one,
                           kinds=torch.zeros(count, dtype=torch.int32, device=device))
        for method in methods or ('corrected_sdf', 'polygon32', 'polygon256', 'polygon1024'):
            synchronize()
            start = time.perf_counter_ns()
            if method == 'corrected_sdf':
                # Actual circle SDF sampling is included in representation build.
                rep = SdfRep.fromFunction(lambda p: p.norm(dim=1)-3,
                                         points.min(0)-1.35, points.max(0)+1.35, 1/64, backend=sdf_backend)
            else:
                edges = int(method.removeprefix('polygon'))
                theta = np.pi/2+(np.arange(edges)+.5)*2*np.pi/edges
                rep = SurfaceRep.polygon(3*np.column_stack((np.cos(theta), np.sin(theta))))
            scene = Scene([Body(bodyId=0, reps=[rep])], device)
            synchronize()
            build_ms = (time.perf_counter_ns()-start)/1e6
            start = time.perf_counter_ns()
            adjacency = scene.adjacency(ps, props)
            initial = scene.precompute(adjacency, props)
            synchronize()
            first_ms = (time.perf_counter_ns()-start)/1e6
            check_moments(initial, count)
            pairs = sum(len(topo.e) for body in adjacency.bodies for topo in body.reps if topo is not None)
            for profile, channels in channel_sets:
                adj_timing, _ = timed(lambda: scene.adjacency(ps, props), synchronize, repeats)
                moment_timing, result = timed(lambda: scene.precompute(adjacency, props, channels), synchronize, repeats)
                check = check_moments(result, count)
                total_timing, result = timed(lambda: scene.pairMoments(ps, props, channels), synchronize, repeats)
                repeated_check = check_moments(result, count)
                # GPU edge reductions can differ in the last few bits with
                # pair ordering; compare the measured values, not bit patterns.
                np.testing.assert_allclose(list(repeated_check.values()), list(check.values()), rtol=5e-13, atol=1e-12)
                records.append(dict(method=method, backend='Scene '+device, queries=count, profile=profile,
                                    representation_build_ms=build_ms, first_all_channel_call_ms=first_ms,
                                    edge_query_pairs=pairs, pairs_per_query=pairs/count,
                                    adjacency=adj_timing, moments_cached_adjacency=moment_timing,
                                    total=total_timing, checksum=check))
                print(json.dumps(dict(method=method, count=count, profile=profile,
                                      total_ms=total_timing['median_ms'], moment_ms=moment_timing['median_ms'])), flush=True)
            if device.startswith('cuda'):
                for profile, channels in channel_sets:
                    graph_timing = gpu_graph_cost(scene, rep, ps, props, device, repeats, channels, graph_route)
                    graph_records.append(dict(method=method, queries=count, profile=profile, **graph_timing))
                    print(json.dumps(dict(method=method, count=count, profile=profile, graph_ms=graph_timing['median_ms'])), flush=True)
        # This reference is a CPU NumPy routine, even in a CUDA process. Chunk
        # memory to keep its panel temporaries bounded; no transfers timed.
        if device == 'cpu':
            for nodes in (() if skip_arc else (24, 48)):
                def arc():
                    return [disk_channels('w4', 3+ds[i:i+256], 3, nodes=nodes)
                            for i in range(0, count, 256)]
                timing, values = timed(arc, lambda: None, repeats)
                assert all(np.isfinite(a['m1x']).all() for a in values)
                records.append(dict(method=f'circle_arc{nodes}', backend='NumPy CPU', queries=count,
                                    profile='five_symmetry_channels', total=timing))
                print(json.dumps(dict(method=f'circle_arc{nodes}', count=count, total_ms=timing['median_ms'])), flush=True)
    report = dict(hardware=hardware, repeats=repeats, radius_over_h=3, kernel='w4', sdf_spacing_over_h=1/64,
                  counts=counts, records=records, graph_records=graph_records,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=[
                      'Single CPU thread; synchronous eager calls, CUDA synchronized around each measured call.',
                      'Same 81-point normal cut repeated to the larger batch size; boundary-heavy microbenchmark, not a full simulation.',
                      'Warm geometry caches, SDF derivative/support caches, kernel tables and compiled kernels; compilation and first-call cost excluded from steady timing.',
                      'Total includes rebuilding adjacency; cached-adjacency moments are also reported. No solver contraction, transfer, rendering, or reference construction timed.',
                      'Mass-first requests channels 0,1,2. Both Scene SDF and polygon paths respect requested channels; SDF mass is always evaluated internally for its posterior guard.',
                      'Circle reference computes five symmetry channels using CPU NumPy quadrature, rather than the Scene/Warp backend. Backend and channel count differ.',
                      'Distance-only reference is offline NumPy closed forms, not an optimized Scene scheme; no production speed claim is inferred for it.',
                      'Eager table measures actual Scene local-SDF execution. The separate graph study captures guarded SDF core or FusedWall polygon adjacency plus output assembly; scopes and requested moments are labelled.',
                      'Graph times are CUDA event times, ten replays/sample for a pilot below 2 ms, otherwise one; at least 15 samples. Setup excluded. SDF includes its field-domain guard but not Scene output assembly/fallback. Fused polygon plans also honor requested channels.',
                      'Representation build includes field sampling for this analytic circle only; first-call time includes lazy setup and possibly cached/disk JIT loading, not reproducible fresh-process compilation cost.',
                  ])
    (output/'timings.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Curved-wall compute cost', '', json.dumps(hardware), '',
             f'{repeats} timed repetitions after warm-up; median times in ms. Radius/h=3, Wendland C4, float64.', '',
             '| Queries | Requested channels | Implementation | Adjacency | Moments, cached adjacency | Total with adjacency |',
             '|---:|---|---|---:|---:|---:|']
    for row in records:
        fmt = lambda key: f"{row[key]['median_ms']:.4g}" if key in row else '—'
        lines.append(f"| {row['queries']} | {row['profile']} | {row['method']} | {fmt('adjacency')} | {fmt('moments_cached_adjacency')} | {fmt('total')} |")
    if graph_records:
        lines += ['', '## Captured device paths (different scopes)', '',
                  '| Queries | Method | Requested channels | Captured scope | Median ms |', '|---:|---|---|---|---:|']
        for row in graph_records:
            lines.append(f"| {row['queries']} | {row['method']} | {row['profile']} | {row['backend']} | {row['median_ms']:.4g} |")
    lines += ['', '## Scope and limits', '', *['- '+s for s in report['limitations']], '',
              'Raw timings include every repetition and p10/p90. Setup/first-call timings are retained separately in timings.json.', '',
              'Reproduce with `python scripts/studies/curved_boundary_cost.py --device '+device+'`. Set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the same CPU threading.', '']
    (output/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--counts', type=int, nargs='+', default=[81, 4096])
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--output', type=Path)
    p.add_argument('--sdf-backend', choices=['warp', 'torch'], default='warp')
    p.add_argument('--methods', nargs='+', choices=['corrected_sdf', 'polygon32', 'polygon256', 'polygon1024'])
    p.add_argument('--skip-arc', action='store_true')
    p.add_argument('--graph-route', choices=['mono', 'cheb'], default='mono')
    args = p.parse_args()
    if args.repeats < 3 or min(args.counts) < 1:
        p.error('Use at least three repetitions and positive batch sizes.')
    run(args.output or evidence_path('curved-boundary-cost-fused-f64-'+args.device.split(':')[0]),
        args.device, args.counts, args.repeats, args.sdf_backend, args.methods, args.skip_arc, args.graph_route)
