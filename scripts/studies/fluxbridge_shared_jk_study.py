"""Input handling, accuracy and CUDA cost of the shared-J/K closed-form P1 toy."""
import argparse
import fcntl
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp
from fluxbridge_shared_jk import functions, prepare_geometry, source
from fluxbridge_single_term_study import cases, ROOT
from fluxbridge_single_term_analytic import provider
from fluxbridge_shell_acceptance import timed


def inputs(api, q, h, field, tet, device):
    q, h, field = np.asarray(q, float), np.asarray(h, float), np.asarray(field, float)
    if h.ndim != 1 or q.shape != (len(h),3) or field.shape != (len(h),4) or not all(np.isfinite(v).all() for v in (q,h,field)) or np.any(h<=0):
        raise ValueError('finite [pair,3] positions, positive supports and [pair,4] nodal fields required')
    tet = np.asarray(tet)
    if tet.shape != h.shape or not np.isfinite(tet).all() or np.any(tet<0) or np.any(tet != np.floor(tet)) or np.any(tet > np.iinfo(np.int32).max):
        raise ValueError('one nonnegative int32 tetrahedron index required per pair')
    return [wp.array(q, dtype=api.vec, device=device),
            wp.array(h, dtype=api.real, device=device),
            wp.array(np.asarray(tet), dtype=int, device=device),
            wp.array(field, dtype=api.field_vec, device=device)]


def evaluate(api, geometry, q, h, field, tet, device):
    arrays = inputs(api, q, h, field, tet, device)
    if np.any(np.asarray(tet) >= geometry.vertices.shape[0]):
        raise ValueError('tetrahedron index outside prepared geometry')
    out = wp.empty(len(q), dtype=api.real, device=device)
    wp.launch(api.kernel, len(q), inputs=arrays+[geometry, out], device=device)
    return out.numpy().copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=('float64', 'float32'), default='float64')
    p.add_argument('--counts', default='128,512,8192,131072,1048576')
    p.add_argument('--accuracy-only', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); counts = list(map(int, a.counts.split(',')))
    if any(not 1 <= n <= 1048576 for n in counts) or (a.profile and len(counts) != 1):
        p.error('invalid counts')
    a.output.mkdir(parents=True, exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock', 'a') as lock:
        if a.device.startswith('cuda'):
            fcntl.flock(lock, fcntl.LOCK_EX)
        wp.init(); device = wp.get_device(a.device); api = functions(a.precision)
        paths = list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))
        paths += [ROOT/'scripts/studies/fluxbridge_single_term_study.py', ROOT/'scripts/studies/fluxbridge_single_term_analytic.py']
        report = dict(precision=a.precision, device=str(device), device_name=device.name,
            accuracy=[], properties={}, rows=[], source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            scope='One scalar integral of W6 times a P1 field, with shared edge J and face-specific combined K recurrences; all per-query work in one kernel',
            limitations=['Prototype, not a production operator replacement', 'Static normals/tangents/outward vectors/FEM gradients prepared once per tet in host fp64 and stored in the selected device dtype',
                'No quadrature, recurrence DAG interpreter or global per-pair integration scratch', 'Field, particle, support and tet index are independently supplied per pair',
                'Raw fp32 is diagnostic; accepted analytic-fp64 policy unchanged', 'Warm times exclude JIT/static geometry and reuse input/output buffers',
                'No search, scatter or global assembly; only a local particle-tet scalar', 'GPU measurements serialized, clocks/caches and desktop activity uncontrolled'])
        def save():
            (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (a.output/'generated-native.cpp').write_text(source(a.precision))
        if not a.profile:
            cs = cases(3)
            geo = prepare_geometry([c['vertices'] for c in cs], device, a.precision)
            values = evaluate(api, geo, [c['query'] for c in cs], [c['h'] for c in cs],
                              [c['field'] for c in cs], np.arange(len(cs)), device)
            for c, value in zip(cs, values):
                finite = bool(np.isfinite(value));error = abs(float(value)-c['reference'])/(1+abs(c['reference'])) if finite else None
                report['accuracy'].append(dict(label=c['label'], value=float(value) if finite else None, reference=c['reference'], mixed_error=error, finite=finite))
            save()
            print('accuracy', [(x['label'], x['mixed_error']) for x in report['accuracy']], flush=True)
            # All vertex orderings, basis partition, units, changing supports,
            # unrelated tetrahedra and independent per-pair fields.
            tet = np.vstack([np.zeros(3), 2*np.eye(3)])
            q = np.array([[.2,.3,.4]]); f = np.array([[.35,1.1,-.2,.7]])
            perms = np.array(list(itertools.permutations(range(4))))
            g = prepare_geometry(tet[perms], device, a.precision)
            permuted = evaluate(api, g, np.repeat(q,24,axis=0), np.ones(24), f[0][perms], np.arange(24), device)
            base = float(permuted[0]); mixed = lambda value,target:float(np.max(abs(np.asarray(value)-target)/(1+abs(target))))
            report['properties']['all_24_vertex_orders'] = mixed(permuted, base)
            g = prepare_geometry([tet, tet*2, tet+.5], device, a.precision)
            transformed = evaluate(api, g, [q[0],q[0]*2,q[0]+.5], [1,2,1], np.repeat(f,3,axis=0), [0,1,2], device)
            report['properties']['units_translation'] = mixed(transformed, base)
            g = prepare_geometry([tet], device, a.precision)
            basis = evaluate(api, g, np.repeat(q,4,axis=0), np.ones(4), np.eye(4), np.zeros(4,int), device)
            constant = evaluate(api, g, q, [1], np.ones((1,4)), [0], device)[0]
            report['properties']['partition_of_unity'] = mixed(basis.sum(), constant)
            report['properties']['field_linearity'] = mixed(np.dot(basis,f[0]),base)
            rng=np.random.default_rng(19); rq=rng.uniform(-.5,1.5,(64,3));rh=rng.uniform(.3,2,64)
            rf=rng.normal(size=(64,4)); rg=prepare_geometry([tet],device,a.precision)
            random=evaluate(api,rg,rq,rh,rf,np.zeros(64,int),device)
            # Independent fixture tests above; this broad changing-input check
            # additionally compares with the existing fp64 analytic interpreter.
            refs=[]
            for x,h,field in zip(rq,rh,rf):
                refs.append(provider(tet,field,device,True,'float64').evaluate([x],h)[0])
            report['properties']['64_varied_positions_supports_fields_vs_native_fp64'] = mixed(random,np.array(refs))
            report['properties']['zero_field'] = mixed(evaluate(api,rg,rq,rh,np.zeros_like(rf),np.zeros(64,int),device),0.)
            report['properties']['empty_output_count'] = int(evaluate(api,rg,np.empty((0,3)),np.empty(0),np.empty((0,4)),np.empty(0,int),device).size)
            invalid = [([q[0]], [0], f, [0]), ([q[0]], [-1], f, [0]),
                       ([[np.nan,0,0]], [1], f, [0]), ([q[0]], [1], [[1,2,3]], [0]),
                       ([q[0]], [1], f, []), ([q[0]], [1], f, [-1]),
                       ([q[0]], [1], f, [.5]), ([q[0]], [1], f, [1])]
            rejected = 0
            for args in invalid:
                try:
                    evaluate(api, rg, *args, device)
                except ValueError:
                    rejected += 1
            report['invalid_input_rejections'] = dict(expected=len(invalid), actual=rejected)
            budget=1e-10 if a.precision=='float64' else 1e-5
            report['validation_budget']=budget
            report['validation_passed']=all(x['finite'] and x['mixed_error']<=budget for x in report['accuracy']) and all(np.isfinite(v) and v<=budget for v in report['properties'].values()) and rejected==len(invalid)
            save()
            if a.precision=='float64' and not report['validation_passed']:
                raise RuntimeError('fp64 accuracy gate failed; see saved report')
        if not a.accuracy_only:
            tet=np.vstack([np.zeros(3),2*np.eye(3)])
            geometry, static = timed(lambda:prepare_geometry([tet],device,a.precision),device,1)
            static_bytes=sum(getattr(geometry,name).capacity for name in ('vertices','normals','tangents','outward','gradients','edges'))
            for n in counts:
                rng=np.random.default_rng(852);q=.2+rng.uniform(-.015,.015,(n,3));h=np.ones(n)
                fields=np.broadcast_to([.35,1.1,-.2,.7],(n,4)).copy();tets=np.zeros(n,int)
                ins=inputs(api,q,h,fields,tets,device);out=wp.empty(n,dtype=api.real,device=device)
                action=lambda:wp.launch(api.kernel,n,inputs=ins+[geometry,out],device=device)
                action();wp.synchronize_device(device)
                row=dict(pairs=n,static_setup=static,static_geometry_bytes=static_bytes,
                    input_output_bytes=sum(x.capacity for x in ins)+out.capacity,global_pair_scratch_bytes=0)
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=device.is_cuda)
                    def transferred():
                        new_inputs=inputs(api,q,h,fields,tets,device);dst=wp.empty(n,dtype=api.real,device=device)
                        wp.launch(api.kernel,n,inputs=new_inputs+[geometry,dst],device=device)
                        return dst.numpy()
                    values,row['upload_kernel_readback']=timed(transferred,device,3)
                    assert np.isfinite(values).all()
                report['rows'].append(row);save();print(n,row.get('warm'),flush=True)


if __name__=='__main__':main()
