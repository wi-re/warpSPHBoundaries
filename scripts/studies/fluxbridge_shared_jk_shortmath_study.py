"""Accuracy and matched CUDA cost of shorter protected elementary seeds."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_shortmath as impl
import fluxbridge_shared_jk_stability as paired
import fluxbridge_shared_jk_full as fp64
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_shared_jk_full_study import arrays,evaluate
from fluxbridge_shared_jk_full_precision_study import datasets,metrics
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed
from fluxbridge_single_term_study import ROOT


def elementary_kernel(variant):
    code=impl.source(variant).split('#define LD')[0]+'''
T a=T(x[0],x[1]),b=T(y[0],y[1]);
T values[2]={logf(T(1)+fabsf(a)),atan2f(b,a)};
for(int j=0;j<2;++j){
 *wp::address(output,row,2*j)=values[j].hi;
 *wp::address(output,row,2*j+1)=values[j].lo;
}
return 0.0f;
'''
    @wp.func_native(code)
    def calc(x:wp.vec2f,y:wp.vec2f,output:wp.array2d(dtype=wp.float32),row:int)->wp.float32: ...
    @wp.kernel(module='unique',module_options={'enable_backward':False,'fuse_fp':True,'max_unroll':1})
    def check(x:wp.array(dtype=wp.vec2f),y:wp.array(dtype=wp.vec2f),output:wp.array2d(dtype=wp.float32)):
        i=wp.tid();ignored=calc(x[i],y[i],output,i)
    return check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--variants',default=','.join(impl.VARIANTS))
    p.add_argument('--accuracy-only',action='store_true');p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--mixed',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--counts',default='1048576');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();variants=a.variants.split(',');counts=list(map(int,a.counts.split(',')))
    if any(v not in impl.VARIANTS for v in variants) or any(n<1 or n>1048576 for n in counts):p.error('invalid variants/counts')
    if a.device=='cpu' and not a.accuracy_only:p.error('CPU accuracy-only required')
    if a.profile and (len(variants)!=1 or len(counts)!=1):p.error('profile requires one variant/count')
    a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',ROOT/'docs/work/fluxbridge-full-reference-20261026.json']
        report=dict(device=str(device),device_name=device.name,mixed=a.mixed,block_dim=256,accuracy=[],elementary=[],rows=[],
            scope='Frozen split-fp32 geometry and protected radial core; only paired log/atan seed polynomials change; fp32 final accumulation/output',
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for variant in variants:(a.output/(variant+'-native.cpp')).write_text(impl.source(variant))
        if not(a.benchmark_only or a.profile):
            rng=np.random.default_rng(761);n=8192
            x=rng.uniform(-2,2,n)*np.exp2(rng.integers(-20,21,n));y=rng.uniform(-2,2,n)*np.exp2(rng.integers(-20,21,n))
            def split(v):
                hi=v.astype(np.float32);return np.stack([hi,(v-hi.astype(float)).astype(np.float32)],axis=1)
            xs,ys=split(x),split(y);x=xs.astype(float).sum(axis=1);y=ys.astype(float).sum(axis=1)
            expected=np.stack([np.log1p(abs(x)),np.arctan2(y,x)],axis=1)
            xx=wp.array(xs,dtype=wp.vec2f,device=device);yy=wp.array(ys,dtype=wp.vec2f,device=device)
            for variant in variants:
                out=wp.empty((n,4),dtype=wp.float32,device=device)
                wp.launch(elementary_kernel(variant),n,inputs=[xx,yy,out],device=device)
                raw=out.numpy().astype(float);actual=raw[:,::2]+raw[:,1::2];error=abs(actual-expected)/(1+abs(expected))
                report['elementary'].append(dict(variant=variant,cases=n,finite=bool(np.isfinite(actual).all()),operations=['log1p_abs','atan2'],worst_mixed_error=error.max(axis=0).tolist()))
            for d in datasets():
                original=d['expected']
                g64=geometry.prepare_geometry(d['vertices'],device,'float64')
                rounded=evaluate(fp64.functions(),g64,d['q'].astype(np.float32).astype(float),d['h'].astype(np.float32).astype(float),d['indices'],device)
                if original is None:original=evaluate(fp64.functions(),g64,d['q'],d['h'],d['indices'],device)
                g=paired.prepare_geometry(d['vertices'],device,'floatpair-inline')
                for variant in variants:
                    actual=evaluate(impl.functions(variant),g,d['q'],d['h'],d['indices'],device)
                    row=dict(variant=variant,dataset=d['name'],independent_reference=d['independent'],
                        against_original=metrics(actual,original,d['labels'],list(range(28))),
                        against_actual_fp32_inputs=metrics(actual,rounded,d['labels'],list(range(28))))
                    if device.is_cuda:row['kernel_properties']=wp.get_cuda_kernel_properties(impl.functions(variant).kernel,device)
                    report['accuracy'].append(row)
                    np.savez_compressed(a.output/(d['name']+'-'+variant+'.npz'),actual=actual,original=original,rounded_reference=rounded)
                    save();print('accuracy',variant,d['name'],'original',row['against_original']['failed_outputs'],row['against_original']['worst']['mixed_error'],'rounded',row['against_actual_fp32_inputs']['failed_outputs'],row['against_actual_fp32_inputs']['worst']['mixed_error'],flush=True)
            report['accepted_variants']=[v for v in variants if all(r['against_actual_fp32_inputs']['passed'] and (not r['independent_reference'] or r['against_original']['passed']) for r in report['accuracy'] if r['variant']==v)]
            save()
        if a.accuracy_only:return
        vertices=scene(a.mixed);g=paired.prepare_geometry(vertices,device,'floatpair-inline')
        for count in counts:
            rng=np.random.default_rng(852);ids=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
            if a.mixed:
                edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
                q=vertices[ids,0]+np.einsum('nij,nj->ni',edges[ids],rng.uniform(-.5,1.5,(count,3)));h=rng.uniform(.3,2,count)
            else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
            args,out=arrays(impl.functions(variants[0]),g,q,h,ids,device);reference=None
            for variant in variants:
                api=impl.functions(variant);action=lambda:wp.launch(api.kernel,count,inputs=args,device=device,block_dim=256)
                action();wp.synchronize_device(device);value=out.numpy()
                if reference is None:reference=value
                error=abs(value.astype(float)-reference)/(1+abs(reference))
                row=dict(variant=variant,pairs=count,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device),
                    input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),finite=bool(np.isfinite(value).all()),
                    worst_difference_vs_first=float(error.max()),failed_outputs_vs_first=int((error>1e-5).sum()))
                if not row['finite'] or row['failed_outputs_vs_first']:raise RuntimeError('output agreement failed')
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:
                    _,row['warm']=timed(action,device,7,events=True)
                    with wp.ScopedCapture(device=device) as cap:action()
                    _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=True)
                    row['graph_bitwise_equal']=bool(np.array_equal(value,out.numpy()))
                    if not row['graph_bitwise_equal']:raise RuntimeError('graph changed output')
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
