"""Compare explicit fp32 conditioning variants at the unchanged error gate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_full as fp64
import fluxbridge_shared_jk_layout as geometry
import fluxbridge_shared_jk_stability as impl
from fluxbridge_shared_jk_full_study import evaluate
from fluxbridge_shared_jk_full_precision_study import datasets,metrics
from fluxbridge_single_term_study import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0')
    p.add_argument('--variants',default=','.join(impl.VARIANTS));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();variants=a.variants.split(',')
    if any(v not in impl.VARIANTS for v in variants):p.error('unknown variant')
    a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'docs/work/fluxbridge-full-reference-20261026.json']
        report=dict(device=str(device),budget=1e-5,rows=[],source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for v in variants:(a.output/(v+'-native.cpp')).write_text(impl.source(v))
        for data in datasets():
            expected=data['expected']
            if expected is None:
                g64=geometry.prepare_geometry(data['vertices'],device,'float64')
                expected=evaluate(fp64.functions(),g64,data['q'],data['h'],data['indices'],device)
            for variant in variants:
                g=impl.prepare_geometry(data['vertices'],device,variant)
                api=impl.functions(variant);actual=evaluate(api,g,data['q'],data['h'],data['indices'],device).astype(float)
                row=dict(dataset=data['name'],independent_reference=data['independent'],variant=variant,
                    full=metrics(actual,expected,data['labels'],list(range(28))),
                    by_family={name:metrics(actual[:,list(ids)],expected[:,list(ids)],data['labels'],list(ids)) for name,ids in fp64.FAMILIES.items() if name!='full'})
                if device.is_cuda:row['kernel_properties']=wp.get_cuda_kernel_properties(api.kernel,device)
                np.savez_compressed(a.output/(data['name']+'-'+variant+'.npz'),actual=actual,reference=expected)
                report['rows'].append(row);save()
                print(data['name'],variant,'failed',row['full']['failed_outputs'],'worst',row['full']['worst'],row.get('kernel_properties'),flush=True)


if __name__=='__main__':main()
