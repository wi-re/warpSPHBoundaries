"""Matched generic/rolling production WGSL dispatches through optional Dawn.

Reports the actual adapter, shader compilation and warm dispatch+readback.
Software adapter timings must not be described as hardware GPU throughput.
The numerical gate compares every value/distance/validity output to the old
shader. Independent float64 gates live in tests/visualizer/test_analytic_gpu.py.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
STATIC=ROOT/'src/warpSPHBoundaries/visualizer/static'
RUNNER=ROOT/'tests/visualizer/js/analytic-gpu-runner.mjs'


def benchmark(resolution=17,repeats=3):
    script=f'''
import {{compileAnalytic2d}} from {json.dumps((STATIC/'analytic2d-plan.js').as_uri())};
import {{compileAnalytic3d}} from {json.dumps((STATIC/'analytic3d-plan.js').as_uri())};
const rows=[];
for(const dimension of [2,3])for(const kernel of ['w2','w4'])for(const field of ['m00','Gx']){{
 const c={{geometry:dimension===2?'triangle':'tetrahedron',radius:1,width:2,height:2,depth:2,h:1,padding:1,quality:1,resolution:{resolution},models:['analytic'],kernel,field}};
 const plan=dimension===2?compileAnalytic2d(c):compileAnalytic3d(c);
 rows.push({{dimension,kernel,field,uniform:[...plan.uniform],table:[...plan.table]}});
}}
console.log(JSON.stringify(rows));
'''
    rows=json.loads(subprocess.check_output(['node','--input-type=module','--eval',script],text=True))
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=subprocess.run(['node',str(RUNNER)],input=json.dumps(dict(rows=rows,benchmark=True,repeats=repeats)),capture_output=True,text=True,timeout=1800)
    if result.returncode:raise RuntimeError(result.stderr)
    raw=json.loads(result.stdout)
    report=dict(adapter=raw['adapter'],resolution=resolution,repeats=repeats,timing='wall-clock dispatch + readback, preallocated buffers; compilation separate',rows=[])
    report['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (STATIC/'analytic-specialize.js',STATIC/'analytic-gpu.js',STATIC/'analytic3d-compute.wgsl',RUNNER)}
    for row,measured in zip(rows,raw['results']):
        a,b=measured['generic']['values'],measured['rolling']['values']
        if len(a)!=len(b):raise AssertionError('Output size changed')
        error=max(abs(x-y)/(1+abs(x)) for x,y in zip(a,b))
        if not error<3e-6:raise AssertionError((row['dimension'],row['kernel'],row['field'],error))
        item={key:row[key] for key in ('dimension','kernel','field')}
        item.update(queries=resolution**row['dimension'],worst_mixed_vs_generic=error)
        for variant in ('generic','rolling'):
            item[variant]={k:v for k,v in measured[variant].items() if k!='values'}
            item[variant]['warm_median_ms']=statistics.median(item[variant]['warm_dispatch_readback_ms'])
        item['speedup']=item['generic']['warm_median_ms']/item['rolling']['warm_median_ms']
        report['rows'].append(item)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution',type=int,default=17)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    result=benchmark(args.resolution,args.repeats)
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:args.output.write_text(text+'\n')
    print(text)
