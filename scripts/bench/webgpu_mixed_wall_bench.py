"""Compare the original mixed-wall implementation to the optimized WebGPU path.

Uses the baseline source directly from Git, actual Dawn/Vulkan execution, and
checks identical prepared sample bits plus all displayed material terms.
Timings include allocation/upload/readback; software adapters are identified.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
STATIC=ROOT/'src/warpSPHBoundaries/visualizer/static'
RUNNER=ROOT/'tests/visualizer/js/mixed-wall-bench.mjs'


def benchmark(repeats=3,baseline_ref='c624db7'):
    revision=subprocess.check_output(['git','rev-parse',baseline_ref],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='mixed-wall-baseline-') as directory:
        baseline=Path(directory)
        for name in ('fluxbridge-plan.js','wall-browser.js'):
            source=subprocess.check_output(['git','show',f'{revision}:src/warpSPHBoundaries/visualizer/static/{name}'],cwd=ROOT,text=True)
            source=source.replace("'./term-engine.js?v=20261022-7'",repr((STATIC/'term-engine.js').as_uri()))
            source=source.replace("'./fluxbridge-gpu.js'",repr((STATIC/'fluxbridge-gpu.js').as_uri()))
            (baseline/name).write_text(source)
        payload=dict(baselineWall=str(baseline/'wall-browser.js'),baselinePlan=str(baseline/'fluxbridge-plan.js'),repeats=repeats)
        with open('/tmp/fluxbridge-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=subprocess.run(['node',str(RUNNER)],input=json.dumps(payload),capture_output=True,text=True,cwd=ROOT,timeout=1800)
        if result.returncode:raise RuntimeError(result.stderr)
        report=json.loads(result.stdout)
    report['baseline_revision']=revision
    report['timing']='wall-clock, with allocation/upload/readback; warm ratios exclude first run'
    report['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [STATIC/name for name in ('fluxbridge-plan.js','fluxbridge-gpu.js','fluxbridge-compute.wgsl','wall-browser.js')]+[RUNNER]}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--baseline-ref',default='c624db7')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    report=json.dumps(benchmark(args.repeats,args.baseline_ref),indent=2,allow_nan=False)
    if args.output:args.output.write_text(report+'\n')
    print(report)
