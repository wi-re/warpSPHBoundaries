"""Validate/profile early P1-gradient contraction against independent moments."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import warp as wp
import fluxbridge_shared_jk_gradient_contract as impl
from fluxbridge_shared_jk_gradient_study import arrays,fixture
from fluxbridge_single_term_study import cases,ROOT
import fluxbridge_shared_jk_layout as geometry_impl
from fluxbridge_shell_acceptance import timed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--accuracy-only',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--counts',default='512,131072,1048576');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);base=impl.functions()
        api=SimpleNamespace(**{**vars(base),'gradient_kernel':base.contracted_gradient_kernel})
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[a.reference]
        report=dict(device=str(device),accuracy=[],rows=[],source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                    scope='One P1 scalar field times grad_x W, contracted before retaining G/Cov moments')
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (a.output/'generated-native.cpp').write_text(impl.source())
        if not a.profile:
            cs=cases(3);shape=json.loads(a.reference.read_text())
            if shape['worst_convergence']>1e-10:raise RuntimeError('unconverged reference')
            for name,cs,targets in [('fixtures',cs,[fixture(c)[1] for c in cs]),
                                    ('independent_shapes',shape['rows'],[c['basis'] for c in shape['rows']])]:
                g=geometry_impl.prepare_geometry([c['vertices'] for c in cs],device)
                f=np.random.default_rng(319).normal(size=(len(cs),28,4));f[:,0]=1;f[:,1]=0;f[:,2:6]=np.eye(4)
                actual=[]
                for component in range(28):
                    k,args,out=arrays(api,g,[c['query'] for c in cs],[c['h'] for c in cs],np.arange(len(cs)),f[:,component:component+1],'fields',device)
                    wp.launch(k,len(cs),inputs=args,device=device);actual.append(out.numpy().T)
                actual=np.stack(actual,axis=1);expected=np.einsum('nfa,naj->nfj',f,np.asarray(targets))
                error=abs(actual-expected)/(1+abs(expected));index=np.unravel_index(error.argmax(),error.shape)
                row=dict(reference=name,outputs=int(actual.size),worst=float(error.max()),label=cs[index[0]]['label'],
                         component=int(index[1]),axis=int(index[2]),finite=bool(np.isfinite(actual).all()))
                report['accuracy'].append(row);save();print(row,flush=True)
            report['passed']=all(r['finite'] and r['worst']<=1e-10 for r in report['accuracy']);save()
            if not report['passed']:raise RuntimeError('early gradient contraction accuracy failed')
        if not a.accuracy_only:
            g=geometry_impl.prepare_geometry([np.vstack([np.zeros(3),2*np.eye(3)])],device)
            for count in map(int,a.counts.split(',')):
                q=.2+np.random.default_rng(852).uniform(-.015,.015,(count,3));h=np.ones(count);indices=np.zeros(count,int)
                field=np.random.default_rng(317).normal(size=(1,1,4));k,args,out=arrays(api,g,q,h,indices,field,'fields',device)
                action=lambda:wp.launch(k,count,inputs=args,device=device)
                action();wp.synchronize_device(device)
                row=dict(pairs=count,kernel_properties=wp.get_cuda_kernel_properties(k,device) if device.is_cuda else None)
                if a.profile:
                    wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                else:_,row['warm']=timed(action,device,7,events=device.is_cuda)
                report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
