"""Independent weighted-sphere accuracy and serialized complete native costs."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Request,baseline,SphereShellEvaluator
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    paths += [Path(__file__),ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',
        ROOT/'scripts/make_fluxbridge_sphere_fixtures.py',ROOT/'tests/fixtures/fluxbridge-sphere-v1.json',
        ROOT/'tests/fluxbridge/test_sphere_shell.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def accuracy(device,cases):
    rows=[]
    for precision,budget in [('float32',1e-5),('float64',1e-14)]:
        for kernel in ['w2','w4','w6','cubic']:
            worst=0.;channels=0;promotions=0;orders={}
            for case in cases:
                if case['kernel']!=kernel:continue
                e=SphereShellEvaluator([Request(**r) for r in case['requests']],kernel=kernel,precision=precision,device=device)
                b=e.prepare(float(case['distance']),case['normal'],1,radii=float(case['radius']))
                b.launch();value=b.numpy()[0];ref=np.array(case['values'],float)
                error=float(np.max(abs(value-ref)/(1+abs(ref))))
                if not np.isfinite(value).all() or error>budget:raise RuntimeError((precision,kernel,case['label'],error))
                worst=max(worst,error);channels+=len(value);promotions+=b.quadrature['promotions']
                order=str(b.quadrature['nodes']);orders[order]=orders.get(order,0)+1
            rows.append(dict(precision=precision,kernel=kernel,channels=channels,worst_mixed_error=worst,
                budget=budget,promotions=promotions,accepted_orders=orders,rejected_cases=0,failing_channels=0))
            print('accuracy',precision,kernel,channels,worst,flush=True)
    return rows


def runtime(device,cases):
    full=[Request(**r) for r in cases[0]['requests']]
    rows=[]
    for precision in ['float32','float64']:
        for mask,requests,count in [('baseline',baseline(3),1),('baseline',baseline(3),128),('degree9',full,1),('degree9',full,8)]:
            rng=np.random.default_rng(875);d=rng.uniform(-.3,.3,count);r=np.full(count,.5)
            n=rng.normal(size=(count,3));n/=np.linalg.norm(n,axis=1)[:,None]
            e,static=timed(lambda:SphereShellEvaluator(requests,precision=precision,device=device),device,1)
            b,first=timed(lambda:e.prepare(d,n,1,radii=r),device,1)
            _,warm=timed(b.launch,device,3,events=device.is_cuda)
            _,complete=timed(lambda:e.evaluate(d,n,1,radii=r),device,3)
            graph=None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:b.launch()
                before=b.numpy().copy();wp.capture_launch(capture.graph)
                if not np.array_equal(before,b.numpy()):raise RuntimeError('graph replay mismatch')
                _,graph=timed(lambda:wp.capture_launch(capture.graph),device,3,events=True)
            rows.append(dict(precision=precision,mask=mask,queries=count,outputs=len(requests),kernel='w2',
                static_preparation=static,first_checked_preparation=first,warm_action=warm,
                complete_evaluation=complete,graph_replay=graph,batch_owned_bytes=b.bytes,
                quadrature=b.quadrature,comparison_workspace_scope='a separate higher-order batch plus two normalized host arrays',
                pool_current_bytes=wp.get_mempool_used_mem_current(device) if device.is_cuda else None))
            print('runtime',precision,mask,count,'warm',warm['median_ms'],'complete',complete['median_ms'],flush=True)
    return rows


def compile_only(precision,output):
    import warp._src.build as build
    SphereShellEvaluator(precision=precision).prepare([], [0,0,1],1,radii=.5)
    module=wp.get_module('warpSPHBoundaries.fluxbridge.planar_shell')
    options=dict(module.options);options['block_dim']=256
    code,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
    path=output/f'sphere-{precision}.{extension}';path.write_text(code)
    ptx=output/f'sphere-{precision}.ptx'
    build.build_cuda(str(path),120,str(ptx),pch_dir=None,use_precompiled_headers=False,
        fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
    instructions=[s.strip() for s in ptx.read_text().splitlines() if '.f64' in s and not s.strip().startswith(('.','//'))]
    if precision=='float32' and instructions:raise RuntimeError('unexpected fp64 instructions')
    return dict(precision=precision,arch=120,fp64_instructions=len(instructions),compile_passed=True,execution_tested=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--compile-only',choices=['float32','float64'])
    mode.add_argument('--runtime-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    wp.init();device=wp.get_device(args.device)
    if args.compile_only:
        report=dict(compilation=compile_only(args.compile_only,args.output),source_sha256=sources())
    else:
        cases=json.loads((ROOT/'tests/fixtures/fluxbridge-sphere-v1.json').read_text())['cases']
        with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
            if lock:fcntl.flock(lock,fcntl.LOCK_EX)
            report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),warp=wp.__version__,device=str(device),
                device_name=device.name,cuda_runtime_tested=device.is_cuda,source_sha256=sources(),accuracy=[] if args.runtime_only else accuracy(device,cases),runtime=runtime(device,cases),
                limitations=['explicit spherical geometry and ordinary selected-dtype geometry/evaluation',
                    'two-order moment comparison is finite convergence evidence, not a certified error/positivity bound',
                    'requested radial order 24; comparison/promotion by eight up to 64; unresolved inputs reject',
                    'Cartesian request masks only; no curved-element FEM/shape adjoint or sphere-array search adapter',
                    'owned bytes exclude comparison batch, host convergence arrays, static plans and allocator reservation',
                    'complete evaluation includes allocation, convergence and readback; frozen launch is separate graph-compatible interface',
                    'runtime is one stated kernel; independent four-kernel accuracy is reported separately'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
