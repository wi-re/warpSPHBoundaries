"""Independent-fixture accuracy of the dedicated C2/C4 solid-simplex family."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge.solid import solid_functions,solid_source,prepare_solid_geometry

ROOT=Path(__file__).resolve().parents[2]


def fixture_cases(dimension,kernel):
    files=['fluxbridge-v1.json']+(['fluxbridge-tetquad-v1.json','fluxbridge-nearplane-v1.json'] if dimension==3 else [])
    rows=[]
    for name in files:
        for i,c in enumerate(json.loads((ROOT/'tests/fixtures'/name).read_text())['cases']):
            if c.get('dimension',3)!=dimension or c.get('measure','volume')!='volume' or c['kernel']!=kernel:continue
            if np.shape(c['vertices'])!=(dimension+1,dimension):continue
            table={(tuple(r['alpha']),tuple(r['derivative'])):float(v) for r,v in zip(c['requests'],c['values'])}
            zero=(0,)*dimension;req=[(zero,())]+[(zero,(j,)) for j in range(dimension)]
            for a in range(dimension):
                for b in range(a,dimension):req.append((tuple(int(k==a) for k in range(dimension)),(b,)))
            rows.append(dict(**c,case_label=c.get('label',f'{name}-{i}'),expected=[table[r] for r in req]))
    return rows


def evaluate(api,g,q,h,ids,device):
    q=np.asarray(q);q=np.pad(q,((0,0),(0,3-api.dimension)))
    args=[wp.array(q,dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(ids,dtype=int,device=device),g]
    out=wp.empty((api.channels,len(q)),dtype=api.real,device=device)
    wp.launch(api.kernel,len(q),inputs=args+[out],device=device,block_dim=256)
    return out.numpy().T


def metrics(actual,expected,budget):
    finite=np.isfinite(actual);error=np.where(finite,abs(actual-expected)/(1+abs(expected)),np.inf);idx=np.unravel_index(error.argmax(),error.shape)
    return dict(passed=bool(np.all(error<=budget)),failed_outputs=int(np.sum(error>budget)),nonfinite=int(np.sum(~finite)),
        worst_mixed_error=float(error[idx]) if finite[idx] else None,worst_index=list(map(int,idx)),
        worst_actual=float(actual[idx]) if finite[idx] else None,worst_reference=float(expected[idx]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);p.add_argument('--protected',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'src/warpSPHBoundaries/fluxbridge/solid.py',ROOT/'src/warpSPHBoundaries/fluxbridge/_solid_code.py',ROOT/'src/warpSPHBoundaries/fluxbridge/_solid_pair.py']+list((ROOT/'tests/fixtures').glob('fluxbridge*.json'))
    report=dict(device=a.device,rows=[],source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        for d in (2,3):
            for k in ('w2','w4'):
                cases=fixture_cases(d,k);vertices=np.array([c['vertices'] for c in cases]);q=np.array([c['query'] for c in cases]);h=np.array([c['h'] for c in cases]);ids=np.arange(len(cases));expected=np.array([c['expected'] for c in cases])
                for precision in ('float64','float32'):
                    conditioning='protected' if a.protected and precision=='float32' else 'raw'
                    g=prepare_solid_geometry(vertices,device,precision,conditioning)
                    for bundle in ('density','pressure','lean'):
                        api=solid_functions(d,k,precision,bundle,conditioning=conditioning);actual=evaluate(api,g,q,h,ids,device).astype(float)
                        budget=1e-10 if precision=='float64' else 1e-5;m=metrics(actual,expected[:,:api.channels],budget)
                        row=dict(dimension=d,kernel=k,precision=precision,bundle=bundle,cases=len(cases),budget=budget,conditioning=conditioning,metrics=m)
                        if device.is_cuda:row['kernel_properties']=wp.get_cuda_kernel_properties(api.kernel,device)
                        report['rows'].append(row)
                        np.savez_compressed(a.output/f'{d}-{k}-{precision}-{bundle}.npz',actual=actual,reference=expected[:,:api.channels])
                        (a.output/f'{d}-{k}-{precision}-{bundle}.cpp').write_text(solid_source(d,k,precision,bundle,conditioning))
                        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                        print(d,k,precision,bundle,m,row.get('kernel_properties'),flush=True)
        report['passed']=all(x['metrics']['passed'] for x in report['rows'])
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
