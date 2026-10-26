"""Quantify raw-fp32 full-28 failures against frozen independent references."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_full as full
import fluxbridge_shared_jk_full_precision as precision
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_shared_jk_full_study import evaluate
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_single_term_study import ROOT,cases
from warpSPHBoundaries.fluxbridge import baseline

REQUESTS=baseline(3)
BUDGET=1e-5


def metrics(actual,expected,labels,channels):
    finite=np.isfinite(actual);error=np.where(finite,abs(actual-expected)/(1+abs(expected)),np.inf)
    idx=np.unravel_index(error.argmax(),error.shape)
    finite_errors=error[finite]
    worst=dict(case=labels[idx[0]],channel=int(channels[idx[1]]),actual=float(actual[idx]) if finite[idx] else None,
        reference=float(expected[idx]),absolute_error=float(abs(actual[idx]-expected[idx])) if finite[idx] else None,
        mixed_error=float(error[idx]) if finite[idx] else None)
    return dict(outputs=int(error.size),cases=len(labels),failed_outputs=int(np.sum(error>BUDGET)),
        failed_cases=int(np.sum(np.any(error>BUDGET,axis=1))),nonfinite_outputs=int(np.sum(~finite)),worst=worst,
        mixed_error_quantiles={str(q):float(np.quantile(finite_errors,q)) for q in (0,.5,.9,.99,1)} if finite_errors.size else {},
        passed=bool(np.all(error<=BUDGET)))


def quantized_replay(g32,device):
    """Same rounded geometry coefficients, promoted to double for replay."""
    api=geometry.functions('float64');g=api.Geometry()
    g.data=wp.array(g32.data.numpy().astype(np.float64),dtype=api.vec,device=device)
    g.edges=g32.edges
    return g


def datasets():
    cs=cases(3);expected=[]
    for c in cs:
        table={(tuple(r['alpha']),tuple(r['derivative'])):float(v) for r,v in zip(c['requests'],c['values'])}
        expected.append([table[(r.alpha,r.derivative)] for r in REQUESTS])
    refs=json.loads((ROOT/'docs/work/fluxbridge-full-reference-20261026.json').read_text())
    assert refs['passed']
    result=[]
    for name,rows,values in [('fixtures',cs,expected),('independent_shapes',refs['rows'],[c['values'] for c in refs['rows']])]:
        result.append(dict(name=name,vertices=np.array([c['vertices'] for c in rows]),q=np.array([c['query'] for c in rows]),
            h=np.array([c['h'] for c in rows]),indices=np.arange(len(rows)),labels=[c['label'] for c in rows],expected=np.array(values),independent=True))
    for mixed in (False,True):
        vertices=scene(mixed);n=4096;rng=np.random.default_rng(852)
        indices=rng.integers(0,len(vertices),n,dtype=np.int32) if mixed else np.zeros(n,int)
        if mixed:
            edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
            q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],rng.uniform(-.5,1.5,(n,3)));h=rng.uniform(.3,2,n)
        else:q=.2+rng.uniform(-.015,.015,(n,3));h=np.ones(n)
        result.append(dict(name='mixed_sample' if mixed else 'canonical_sample',vertices=vertices,q=q,h=h,indices=indices,
            labels=list(range(n)),expected=None,independent=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_single_term_study.py',
            ROOT/'docs/work/fluxbridge-full-reference-20261026.json']+list((ROOT/'tests/fixtures').glob('fluxbridge-*.json'))
        report=dict(device=str(device),device_name=device.name,precision='raw float32',budget=BUDGET,rows=[],
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
            scope='Runtime float arithmetic/storage/intrinsics; host-fp64 geometry preparation cast to float; no promoted device accumulator')
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for family in full.FAMILIES:(a.output/(family+'-fp32-native.cpp')).write_text(precision.source('float32',family))
        for data in datasets():
            q,h,ids=data['q'],data['h'],data['indices'];labels=data['labels']
            g64=geometry.prepare_geometry(data['vertices'],device,'float64');g32=geometry.prepare_geometry(data['vertices'],device,'float32')
            expected=data['expected']
            fp64=evaluate(full.functions(),g64,q,h,ids,device)
            if expected is None:expected=fp64
            replay=evaluate(full.functions(),quantized_replay(g32,device),q.astype(np.float32).astype(float),h.astype(np.float32).astype(float),ids,device)
            row=dict(dataset=data['name'],independent_reference=data['independent'],families={},
                fp64_against_reference=metrics(fp64,expected,labels,list(range(28))),
                fp64_replay_quantized_inputs_against_reference=metrics(replay,expected,labels,list(range(28))))
            actual_full=None
            for family,channels in full.FAMILIES.items():
                channels=list(channels);api=precision.functions('float32',family)
                actual=evaluate(api,g32,q,h,ids,device).astype(float)
                row['families'][family]=dict(against_reference=metrics(actual,expected[:,channels],labels,channels),
                    against_fp64_replay=metrics(actual,replay[:,channels],labels,channels))
                if device.is_cuda:row['families'][family]['kernel_properties']=wp.get_cuda_kernel_properties(api.kernel,device)
                if family=='full':actual_full=actual
            row['full_by_family']={name:metrics(actual_full[:,list(channels)],expected[:,list(channels)],labels,list(channels)) for name,channels in full.FAMILIES.items() if name!='full'}
            # Preserve every output and error for post-hoc absolute/relative or
            # sign checks, rather than reporting only aggregate pass/fail.
            np.savez_compressed(a.output/(data['name']+'-outputs.npz'),actual=actual_full,reference=expected,fp64=fp64,replay=replay,q=q,h=h,indices=ids,vertices=data['vertices'])
            report['rows'].append(row);save();print(json.dumps(row,allow_nan=False),flush=True)
        report['independent_passed']=all(r['families']['full']['against_reference']['passed'] for r in report['rows'] if r['independent_reference'])
        save()


if __name__=='__main__':main()
