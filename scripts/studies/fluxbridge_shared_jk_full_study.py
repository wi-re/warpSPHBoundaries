"""All-channel CPU/CUDA acceptance, invariants and scaling for the W6 baseline."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import baseline
import fluxbridge_shared_jk_full as impl
import fluxbridge_shared_jk_layout as geometry
from fluxbridge_single_term_study import cases,ROOT
from fluxbridge_shared_jk_consumers import scene
from fluxbridge_shell_acceptance import timed

SHAPES=ROOT/'docs/work/fluxbridge-shared-jk-shapes-20261026.json'
REQUESTS=baseline(3)


def independent(c,order):
    """Factored W6, analytic pointwise gradient/Hessian, independent volume rule."""
    v,q,h=np.asarray(c['vertices']),np.asarray(c['query']),c['h']
    t,w=np.polynomial.legendre.leggauss(order);t,w=(t+1)/2,w/2
    u=t[:,None,None];b=1-u;vv=t[None,:,None];a=b*vv;z=b*(1-vv)*t[None,None,:]
    y=((v[0]-q)/h+u[...,None]*(v[1]-v[0])/h+a[...,None]*(v[2]-v[0])/h+z[...,None]*(v[3]-v[0])/h)
    r=np.sqrt(np.sum(y*y,axis=-1));s=np.maximum(1-r,0)
    weights=w[:,None,None]*w[None,:,None]*w[None,None,:]*b*b*(1-vv)
    weights*=1365/(64*np.pi)*abs(np.linalg.det((v[1:]-v[0]).T))/h**3
    phi=s**8*(1+r*(8+r*(25+32*r)));psi=-22*s**7*(1+r*(7+16*r))
    second=528*s**6*(1+6*r)
    result=[]
    for req in REQUESTS:
        integrand=phi if not req.derivative else (-psi*y[...,req.derivative[0]]/h if len(req.derivative)==1 else
            (psi*int(req.derivative[0]==req.derivative[1])+second*y[...,req.derivative[0]]*y[...,req.derivative[1]])/h**2)
        for axis,power in enumerate(req.alpha):
            if power:integrand=integrand*(h*y[...,axis])**power
        result.append(float(np.sum(weights*integrand)))
    return np.array(result)


def reference(path):
    original=json.loads(SHAPES.read_text())['rows']
    digest=hashlib.sha256(SHAPES.read_bytes()).hexdigest()
    report=json.loads(path.read_text()) if path.exists() else dict(shape_sha256=digest,target=2e-12,rows=[])
    if report['shape_sha256']!=digest:raise RuntimeError('reference input hash changed')
    for c in original[len(report['rows']):]:
        orders=[72,96];old=independent(c,72);value=independent(c,96)
        err=float(np.max(abs(value-old)/(1+abs(value))))
        for n in (128,160,192,224,256):
            if err<=report['target']:break
            old=value;value=independent(c,n);orders.append(n)
            err=float(np.max(abs(value-old)/(1+abs(value))))
        row={k:c[k] for k in ('label','vertices','query','h')}
        row.update(values=value.tolist(),orders=orders,convergence=err)
        report['rows'].append(row);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('reference',c['label'],orders,err,flush=True)
    report['worst_convergence']=max(c['convergence'] for c in report['rows'])
    report['passed']=len(report['rows'])==len(original) and report['worst_convergence']<=report['target']
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['passed']:raise RuntimeError('independent Hessian reference not converged')
    return report


def arrays(api,g,q,h,indices,device):
    q,h,indices=np.asarray(q,float),np.asarray(h,float),np.asarray(indices)
    if h.ndim!=1 or q.shape!=(len(h),3) or indices.shape!=h.shape or not all(np.isfinite(a).all() for a in (q,h,indices)) or np.any(h<=0) or np.any(indices<0) or np.any(indices>=g.data.shape[0]) or np.any(indices!=np.floor(indices)):
        raise ValueError('finite positions, positive supports and valid tet indices required')
    out=wp.empty((api.channels,len(q)),dtype=api.real,device=device)
    args=[wp.array(q,dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(indices,dtype=int,device=device),g,out]
    return args,out


def evaluate(api,g,q,h,indices,device):
    args,out=arrays(api,g,q,h,indices,device)
    wp.launch(api.kernel,len(q),inputs=args,device=device)
    return out.numpy().T.copy()


def transformed(value,R):
    out=value.copy();out[1:4]=R@value[1:4];out[10:13]=R@value[10:13]
    for start,stop in [(4,10),(22,28)]:
        M=np.zeros((3,3))
        for v,(i,j) in zip(value[start:stop],impl.PAIRS):M[i,j]=M[j,i]=v
        M=R@M@R.T;out[start:stop]=[M[i,j] for i,j in impl.PAIRS]
    out[13:22]=(R@value[13:22].reshape(3,3)@R.T).ravel()
    return out


def properties(device):
    import itertools
    api=impl.functions();tet=np.vstack([np.zeros(3),2*np.eye(3)]);q=np.array([.2,.3,.4]);h=.9
    g=geometry.prepare_geometry([tet],device);base=evaluate(api,g,[q],[h],[0],device)[0]
    error=lambda a,b:float(np.max(abs(np.asarray(a)-b)/(1+abs(b))))
    perms=np.array(list(itertools.permutations(range(4))));g=geometry.prepare_geometry(tet[perms],device)
    values=evaluate(api,g,np.repeat([q],24,axis=0),np.full(24,h),np.arange(24),device)
    result=dict(vertex_order=error(values,base))
    R=np.linalg.qr(np.random.default_rng(417).normal(size=(3,3)))[0];shift=np.array([.5,-.8,1.2]);scale=3.
    g=geometry.prepare_geometry([tet+shift,tet*scale,tet@R.T],device)
    values=evaluate(api,g,[q+shift,q*scale,R@q],[h,h*scale,h],[0,1,2],device)
    result['translation']=error(values[0],base);result['units']=error(values[1],base*np.array([scale**r.scale for r in REQUESTS]));result['rotation']=error(values[2],transformed(base,R))
    center=tet.mean(axis=0);children=[np.vstack([center,tet[[j for j in range(4) if j!=i]]]) for i in range(4)]
    g=geometry.prepare_geometry([tet,*children],device)
    values=evaluate(api,g,np.repeat([center],5,axis=0),np.full(5,.8),np.arange(5),device)
    result['subdivision_internal_faces']=error(values[1:].sum(axis=0),values[0])
    result['empty_output_count']=int(evaluate(api,g,np.empty((0,3)),np.empty(0),np.empty(0,int),device).size)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--reference',type=Path)
    p.add_argument('--generate-reference',action='store_true');p.add_argument('--accuracy-only',action='store_true');p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--mixed',action='store_true');p.add_argument('--counts',default='512,8192,131072,1048576')
    p.add_argument('--profile-family',choices=tuple(impl.FAMILIES));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);refs=None
    if a.generate_reference:
        if not a.reference:p.error('reference path required')
        refs=reference(a.reference)
    elif a.reference:
        refs=json.loads(a.reference.read_text())
        if not refs.get('passed'):raise RuntimeError('reference incomplete or unconverged')
    counts=list(map(int,a.counts.split(',')))
    if any(n<1 or n>1048576 for n in counts) or (a.profile_family and len(counts)!=1):p.error('invalid counts')
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[ROOT/'scripts/studies/fluxbridge_single_term_study.py']
        paths+=list((ROOT/'tests/fixtures').glob('fluxbridge-*.json'))
        if a.reference:paths.append(a.reference)
        report=dict(device=str(device),device_name=device.name,precision='float64',mixed=a.mixed,accuracy=[],properties={},rows=[],
            requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in REQUESTS],
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
            scope='Exact baseline ordering: ten value moments, twelve first-gradient moments, six symmetric Hessian moments; one kernel, shared edge stream')
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for family in impl.FAMILIES:(a.output/(family+'-native.cpp')).write_text(impl.source(family))
        if not(a.benchmark_only or a.profile_family):
            cs=cases(3);reference_values=[]
            for c in cs:
                table={(tuple(r['alpha']),tuple(r['derivative'])):float(v) for r,v in zip(c['requests'],c['values'])}
                reference_values.append([table[(r.alpha,r.derivative)] for r in REQUESTS])
            datasets=[('fixtures',cs,np.array(reference_values))]
            if refs:datasets.append(('independent_shapes',refs['rows'],np.array([c['values'] for c in refs['rows']])))
            for label,cs,ref in datasets:
                g=geometry.prepare_geometry([c['vertices'] for c in cs],device)
                for family,indices in impl.FAMILIES.items():
                    api=impl.functions(family);actual=evaluate(api,g,[c['query'] for c in cs],[c['h'] for c in cs],np.arange(len(cs)),device)
                    expected=ref[:,list(indices)];err=abs(actual-expected)/(1+abs(expected));idx=np.unravel_index(err.argmax(),err.shape)
                    row=dict(reference=label,family=family,outputs=int(actual.size),finite=bool(np.isfinite(actual).all()),worst=float(err.max()),label=cs[idx[0]]['label'],channel=int(list(indices)[idx[1]]))
                    report['accuracy'].append(row);save();print(row,flush=True)
            report['properties']=properties(device)
            report['passed']=all(r['finite'] and r['worst']<=1e-10 for r in report['accuracy']) and all(np.isfinite(v) and v<=1e-10 for v in report['properties'].values());save()
            if not report['passed']:raise RuntimeError('full baseline accuracy gate failed')
        if not a.accuracy_only:
            vertices=scene(a.mixed)
            g,report['static_setup']=timed(lambda:geometry.prepare_geometry(vertices,device),device,1)
            report['static_geometry_bytes']=geometry.geometry_bytes(g)
            for count in counts:
                rng=np.random.default_rng(852);indices=rng.integers(0,len(vertices),count,dtype=np.int32) if a.mixed else np.zeros(count,int)
                if a.mixed:
                    edges=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1);q=vertices[indices,0]+np.einsum('nij,nj->ni',edges[indices],rng.uniform(-.5,1.5,(count,3)));h=rng.uniform(.3,2,count)
                else:q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count)
                for family in ([a.profile_family] if a.profile_family else impl.FAMILIES):
                    api=impl.functions(family);args,out=arrays(api,g,q,h,indices,device)
                    action=lambda:wp.launch(api.kernel,count,inputs=args,device=device)
                    action();wp.synchronize_device(device)
                    row=dict(pairs=count,family=family,outputs=api.channels,kernel_properties=wp.get_cuda_kernel_properties(api.kernel,device) if device.is_cuda else None,
                        input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),global_pair_scratch_bytes=0)
                    if a.profile_family:
                        wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                    else:
                        _,row['warm']=timed(action,device,7,events=device.is_cuda)
                        with wp.ScopedCapture(device=device) as cap:action()
                        wp.capture_launch(cap.graph);wp.synchronize_device(device);before=out.numpy()
                        wp.capture_launch(cap.graph);wp.synchronize_device(device);row['graph_bitwise_equal']=bool(np.array_equal(before,out.numpy()))
                        if not row['graph_bitwise_equal']:raise RuntimeError('graph output changed')
                        _,row['graph']=timed(lambda:wp.capture_launch(cap.graph),device,7,events=device.is_cuda)
                        def transferred():
                            new,dst=arrays(api,g,q,h,indices,device);wp.launch(api.kernel,count,inputs=new,device=device);return dst.numpy()
                        _,row['upload_kernel_readback']=timed(transferred,device,3)
                    report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
