"""Independent value/gradient-moment and contracted P1-gradient acceptance."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import mpmath as mp
import numpy as np
import warp as wp
import fluxbridge_shared_jk_gradient as impl
import fluxbridge_shared_jk_layout as geometry_impl
from fluxbridge_single_term_study import cases,ROOT
from fluxbridge_shell_acceptance import timed

SHAPES=ROOT/'docs/work/fluxbridge-shared-jk-shapes-20261026.json'
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]


def independent(c,n):
    v,q,h=np.asarray(c['vertices']),np.asarray(c['query']),c['h']
    t,w=np.polynomial.legendre.leggauss(n);t,w=(t+1)/2,w/2
    u=t[:,None,None];b=1-u;vv=t[None,:,None];a=b*vv;z=b*(1-vv)*t[None,None,:]
    y=((v[0]-q)/h+u[...,None]*(v[1]-v[0])/h+a[...,None]*(v[2]-v[0])/h+z[...,None]*(v[3]-v[0])/h)
    r=np.sqrt(np.sum(y*y,axis=-1));s=np.maximum(1-r,0)
    weights=w[:,None,None]*w[None,:,None]*w[None,None,:]*b*b*(1-vv)
    factor=1365/(64*np.pi)*abs(np.linalg.det((v[1:]-v[0]).T))/h**3
    lam=factor*np.sum(weights*s**8*(1+r*(8+r*(25+32*r))))
    wg=weights*(22/h)*s**7*(1+r*(7+16*r))
    basis=np.array([[factor*np.sum(wg*y[...,j]*f) for j in range(3)] for f in [1-u-a-z,u,a,z]])
    G=basis.sum(axis=0);cov=(v-q).T@basis
    return np.array([lam,*G,*[cov[i,j] for i,j in PAIRS]]),basis


def fixture(c):
    with mp.workdps(70):
        table={(tuple(r['alpha']),tuple(r['derivative'])):mp.mpf(value) for r,value in zip(c['requests'],c['values'])}
        zero=(0,0,0);m=table[(zero,())];G=[table[(zero,(j,))] for j in range(3)]
        cov=mp.matrix([[table[(tuple(int(k==i) for k in range(3)),(j,))] for j in range(3)] for i in range(3)])
        v=[[mp.mpf(str(x)) for x in p] for p in c['vertices']];q=[mp.mpf(str(x)) for x in c['query']]
        matrix=mp.matrix([[v[i+1][j]-v[0][j] for j in range(3)] for i in range(3)])
        basis=[]
        for k in range(4):
            f=[mp.mpf(int(k==j)) for j in range(4)]
            grad=mp.lu_solve(matrix,mp.matrix([f[i+1]-f[0] for i in range(3)]))
            A=f[0]+sum(grad[j]*(q[j]-v[0][j]) for j in range(3))
            basis.append([float(A*G[j]+sum(grad[i]*cov[i,j] for i in range(3))) for j in range(3)])
        return [float(m),*map(float,G),*[float(cov[i,j]) for i,j in PAIRS]],basis


def reference(path):
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['worst_convergence']>1e-10:raise RuntimeError('cached gradient reference not converged')
        return cached
    rows=[]
    for c in json.loads(SHAPES.read_text())['rows']:
        orders=[72,96];values=[independent(c,n) for n in orders]
        def difference():
            return max(float(np.max(abs(values[-2][i]-values[-1][i])/(1+abs(values[-1][i])))) for i in (0,1))
        conv=difference()
        for order in (128,160,192):
            if conv<=1e-10:break
            orders.append(order);values.append(independent(c,order));conv=difference()
        rows.append(dict(label=c['label'],vertices=c['vertices'],query=c['query'],h=c['h'],
                         lean=values[-1][0].tolist(),basis=values[-1][1].tolist(),orders=orders,convergence=conv))
    result=dict(rows=rows,worst_convergence=max(c['convergence'] for c in rows))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    if result['worst_convergence']>1e-10:raise RuntimeError('gradient reference not converged')
    return result


def arrays(api,geometry,q,h,indices,fields,op,device):
    n=len(q);nf=fields.shape[1]
    base=[wp.array(q,dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),wp.array(indices,dtype=int,device=device)]
    out=wp.empty((10 if op=='lean' else 3*nf,n),dtype=api.real,device=device)
    if op=='lean':return api.lean_kernel,base+[geometry,out],out
    return api.gradient_kernel,base+[wp.array(fields,dtype=api.field_vec,device=device),geometry,out],out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--reference',type=Path)
    p.add_argument('--accuracy-only',action='store_true');p.add_argument('--counts',default='512,131072,1048576')
    p.add_argument('--profile-operator',choices=('lean','fields'));p.add_argument('--profile-fields',type=int,default=3)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    refs=reference(a.reference) if a.reference else None
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);api=impl.functions()
        report=dict(device=str(device),precision='float64',accuracy=[],rows=[],
                    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py')},
                    scope='lambda/G/symmetric Cov geometric moments, P1 scalar-field gradient contractions; no solver or full 28-operator contract')
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for op in ('lean','fields'):(a.output/(op+'-native.cpp')).write_text(impl.source(op))
        if not a.profile_operator:
            cs=cases(3)
            datasets=[('fixtures',cs,[fixture(c) for c in cs])]
            if refs:datasets.append(('independent_shapes',refs['rows'],[(c['lean'],c['basis']) for c in refs['rows']]))
            for label,cs,targets in datasets:
                geometry=geometry_impl.prepare_geometry([c['vertices'] for c in cs],device)
                rng=np.random.default_rng(319);fields=rng.normal(size=(len(cs),28,4));fields[:,0]=1;fields[:,1]=0;fields[:,2:6]=np.eye(4)
                for op,nf in [('lean',1),('fields',1),('fields',3),('fields',28)]:
                    k,args,out=arrays(api,geometry,[c['query'] for c in cs],[c['h'] for c in cs],np.arange(len(cs)),fields[:,:nf],op,device)
                    wp.launch(k,len(cs),inputs=args,device=device);actual=out.numpy().T
                    expected=np.array([t[0] for t in targets]) if op=='lean' else np.einsum('nfa,naj->nfj',fields[:,:nf],np.array([t[1] for t in targets])).reshape(len(cs),-1)
                    error=abs(actual-expected)/(1+abs(expected));idx=np.unravel_index(error.argmax(),error.shape)
                    row=dict(reference=label,operator=op,components=nf,outputs=int(actual.size),worst=float(error.max()),label=cs[idx[0]]['label'],channel=int(idx[1]),finite=bool(np.isfinite(actual).all()))
                    report['accuracy'].append(row);save();print(row,flush=True)
            report['passed']=all(r['finite'] and r['worst']<=1e-10 for r in report['accuracy']);save()
            if not report['passed']:raise RuntimeError('gradient accuracy gate failed')
        if not a.accuracy_only:
            vertices=np.array([np.vstack([np.zeros(3),2*np.eye(3)])])
            geometry,report['static_setup']=timed(lambda:geometry_impl.prepare_geometry(vertices,device),device,1)
            report['static_geometry_bytes']=geometry_impl.geometry_bytes(geometry)
            for count in map(int,a.counts.split(',')):
                rng=np.random.default_rng(852);q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count);indices=np.zeros(count,int)
                ops=[(a.profile_operator,a.profile_fields)] if a.profile_operator else [('lean',1),('fields',1),('fields',3),('fields',28)]
                for op,nf in ops:
                    fields=np.random.default_rng(317).normal(size=(1,nf,4));k,args,out=arrays(api,geometry,q,h,indices,fields,op,device)
                    action=lambda:wp.launch(k,count,inputs=args,device=device)
                    action();wp.synchronize_device(device)
                    row=dict(pairs=count,operator=op,components=nf,kernel_properties=wp.get_cuda_kernel_properties(k,device) if device.is_cuda else None,
                             input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),global_pair_scratch_bytes=0)
                    if a.profile_operator:
                        wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                    else:
                        _,row['warm']=timed(action,device,7,events=device.is_cuda)
                        def transferred():
                            kernel,new,dst=arrays(api,geometry,q,h,indices,fields,op,device)
                            wp.launch(kernel,count,inputs=new,device=device);return dst.numpy()
                        _,row['upload_kernel_readback']=timed(transferred,device,3)
                    report['rows'].append(row);save();print(row,flush=True)


if __name__=='__main__':main()
