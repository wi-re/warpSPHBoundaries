"""Matched two-float/native-fp64 sanity check of the protected W6 core."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_fp64_sanity as impl
import fluxbridge_shared_jk_stability as paired
from fluxbridge_shared_jk_full_study import arrays,evaluate
from fluxbridge_shared_jk_full_precision_study import datasets,metrics
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--accuracy-only',action='store_true')
    p.add_argument('--benchmark-only',action='store_true');p.add_argument('--mixed',action='store_true')
    p.add_argument('--counts',default='1048576');p.add_argument('--profile-variant',choices=impl.VARIANTS)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);counts=list(map(int,a.counts.split(',')))
    if any(n<1 or n>1048576 for n in counts) or (a.profile_variant and len(counts)!=1):p.error('invalid counts')
    if a.device=='cpu' and not a.accuracy_only:p.error('CPU accuracy-only required')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',ROOT/'docs/work/fluxbridge-full-reference-20261026.json']
        paths+=list((ROOT/'tests/fixtures').glob('fluxbridge-*.json'))
        report=dict(device=str(device),device_name=device.name,mixed=a.mixed,block_dim=256,accuracy=[],rows=[],
            scope='Same split-fp32 geometry, fp32 queries/supports, fp32 final accumulators/output and shared buffers; only the protected scalar core/seed implementation changes',
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for variant in impl.VARIANTS:(a.output/(variant+'-native.cpp')).write_text(impl.source(variant))
        if not(a.benchmark_only or a.profile_variant):
            for d in datasets():
                if not d['independent']:continue
                g=paired.prepare_geometry(d['vertices'],device,'floatpair-inline')
                for variant in impl.VARIANTS:
                    api=impl.functions(variant);value=evaluate(api,g,d['q'],d['h'],d['indices'],device)
                    row=dict(variant=variant,dataset=d['name'],metrics=metrics(value,d['expected'],d['labels'],list(range(28))))
                    report['accuracy'].append(row);save();print('accuracy',variant,d['name'],row['metrics']['worst'],flush=True)
            report['passed']=all(x['metrics']['passed'] for x in report['accuracy']);save()
            if not report['passed']:raise RuntimeError('accuracy gate failed')
        if a.accuracy_only:return
        vertices=scene(a.mixed);g=paired.prepare_geometry(vertices,device,'floatpair-inline')
        for count in counts:
            rng=np.random.default_rng(852);ids=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
            if a.mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(count,3)));h=rng.uniform(.3,2,count)
            else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
            args,out=arrays(impl.functions('paired'),g,q,h,ids,device);reference=None
            for variant in ([a.profile_variant] if a.profile_variant else impl.VARIANTS):
                api=impl.functions(variant);action=lambda:wp.launch(api.kernel,count,inputs=args,device=device,block_dim=256)
                action();wp.synchronize_device(device);value=out.numpy()
                if reference is None:reference=value
                error=abs(value.astype(float)-reference)/(1+abs(reference))
                row=dict(variant=variant,pairs=count,input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),
                    kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),finite=bool(np.isfinite(value).all()),
                    worst_difference_vs_first=float(error.max()),failed_outputs_vs_first=int((error>1e-5).sum()))
                if not row['finite'] or row['failed_outputs_vs_first']:raise RuntimeError('matched outputs disagree')
                if a.profile_variant:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=True)
                    with wp.ScopedCapture(device=device) as cap:action()
                    _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                    row['graph_bitwise_equal']=bool(np.array_equal(value,out.numpy()))
                    if not row['graph_bitwise_equal']:raise RuntimeError('graph changed output')
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
