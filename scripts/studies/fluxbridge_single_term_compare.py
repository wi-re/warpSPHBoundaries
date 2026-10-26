"""Matched scalar P1 integral and metamorphic checks for the toy kernels."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry, PolynomialField, FieldEvaluator
from fluxbridge_single_term import tetrahedron_kernel, triangle_kernel
from fluxbridge_single_term_study import arrays, rule, launch
from fluxbridge_shell_acceptance import timed

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    report = dict(matched=[], properties=[], source_sha256={})
    paths = list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    paths += list((ROOT/'scripts/studies').glob('fluxbridge_single_term*.py'))
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with open('/tmp/fluxbridge-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); wp.init(); device = wp.get_device('cuda:0')
        for d, count in ((2, 8192), (3, 512)):
            v0 = np.vstack([np.zeros(d), 2*np.eye(d)])
            geo = Geometry([v0], d)
            f0 = np.array([.35, 1.1, -.2]+([.7] if d == 3 else []))
            grad = (f0[1:]-f0[0])/2
            coeff = {(0,)*d: np.array(f0[0])}
            for j in range(d):
                coeff[tuple(int(k == j) for k in range(d))] = np.array(grad[j])
            provider = FieldEvaluator(PolynomialField(geo, [coeff]), kernel='w6', precision='float32', device=device,
                method='conditioned_boundary' if d == 2 else 'tet_quadrature', nodes=16 if d == 2 else 8,
                preparation='host' if d == 2 else 'device', search='device')
            rng = np.random.default_rng(852); q = .2+rng.uniform(-.015, .015, (count, d))
            b = provider.prepare(q, 1); b.launch(); reference = b.numpy().reshape(-1)
            _, old_warm = timed(b.launch, device, 7, events=True)
            _, old_complete = timed(lambda: provider.evaluate(q, 1), device, 3)
            make = triangle_kernel if d == 2 else tetrahedron_kernel
            vertices = np.broadcast_to(v0, (count, d+1, d)).copy()
            field = np.broadcast_to(f0, (count, d+1)).copy()
            ins = arrays(vertices, q, np.ones(count), field, device)
            output = wp.empty(count, dtype=wp.float32, device=device)
            for n in (8, 12, 16):
                nodes, weights = rule(n, device)
                action = lambda: launch(make(n), ins, nodes, weights, output, device)
                action(); values = output.numpy()
                error = float(np.max(abs(values-reference)/(1+abs(reference))))
                _, cost = timed(action, device, 7, events=True)
                report['matched'].append(dict(dimension=d, count=count, toy_order=n,
                    native_requested_order=16 if d == 2 else 8, native_warm=old_warm,
                    native_complete=old_complete, toy_warm=cost, mixed_error_vs_native=error,
                    passes_1e5=error <= 1e-5,
                    scope='Same scalar P1 integral; native uses prepared boundary quadrature and moment contraction; toy includes all per-pair preparation in its one launch. Static buffers reused in both warm timings.'))
            # Check orientation, linearity, partition of unity and units in the
            # toy's admitted moderate-size scene, separately from quadrature error.
            n = 16; nodes, weights = rule(n, device)
            def evaluate(v, x, h, f):
                inputs = arrays(v, x, h, f, device)
                result = wp.empty(len(x), dtype=wp.float32, device=device)
                launch(make(n), inputs, nodes, weights, result, device)
                return result.numpy().copy()
            v, x, h, f = vertices[:16], q[:16], np.ones(16), field[:16]
            base = evaluate(v, x, h, f)
            perm = np.array([0, 2, 1]+([3] if d == 3 else []))
            # The Duffy rule's finite-order error can depend on vertex order.
            permuted = evaluate(v[:, perm], x, h, f[:, perm])
            scale = evaluate(v*2, x*2, h*2, f)
            translated = evaluate(v+.5, x+.5, h, f)
            constant = evaluate(v, x, h, np.ones_like(f))
            basis_sum = sum(evaluate(v, x, h, np.broadcast_to(np.eye(d+1)[j], f.shape)) for j in range(d+1))
            defects = {name: float(np.max(abs(value-target)/(1+abs(target)))) for name, value, target in
                [('permutation', permuted, base), ('scaling', scale, base), ('translation', translated, base), ('partition_of_unity', basis_sum, constant)]}
            assert max(defects.values()) <= 1e-5, defects
            report['properties'].append(dict(dimension=d, pairs=16, order=n, defects=defects))
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(d, 'matched and property checks complete', flush=True)


if __name__ == '__main__':
    main()
