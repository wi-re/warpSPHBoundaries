"""Validate/measure geometry liveness and bounded caller-specific W6 outputs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import warp as wp

import fluxbridge_shared_jk_layout as impl
from fluxbridge_shared_jk_study import inputs
from fluxbridge_single_term_study import cases,ROOT
from fluxbridge_shell_acceptance import timed

SHAPES=ROOT/'docs/work/fluxbridge-shared-jk-shapes-20261026.json'


def basis_integral(v,q,h,order):
    """Independent CPU reference using the factored radial profile, no J/K."""
    t,w=np.polynomial.legendre.leggauss(order);t,w=(t+1)/2,w/2
    u=t[:,None,None];b=1-u;vv=t[None,:,None];c=b*vv;z=b*(1-vv)*t[None,None,:]
    y=((v[0]-q)/h+u[...,None]*(v[1]-v[0])/h
       +c[...,None]*(v[2]-v[0])/h+z[...,None]*(v[3]-v[0])/h)
    r=np.sqrt(np.sum(y*y,axis=-1));s=np.maximum(1-r,0)
    weighted=(w[:,None,None]*w[None,:,None]*w[None,None,:]*b*b*(1-vv)
              *s**8*(1+r*(8+r*(25+32*r))))
    factor=1365/(64*np.pi)*abs(np.linalg.det((v[1:]-v[0]).T))/h**3
    return factor*np.array([np.sum(weighted*(1-u-c-z)),np.sum(weighted*u),
                            np.sum(weighted*c),np.sum(weighted*z)])


def generate_reference(path):
    rows=[]
    for c in json.loads(SHAPES.read_text())['rows']:
        args=[np.asarray(c[k],float) for k in ('vertices','query')]+[c['h']]
        values=[basis_integral(*args,n) for n in (72,96)]
        rows.append(dict(label=c['label'],vertices=c['vertices'],query=c['query'],h=c['h'],
                         basis=values[-1].tolist(),convergence=float(np.max(abs(values[-1]-values[-2])/(1+abs(values[-1]))))))
    result=dict(orders=[72,96],rows=rows,source_sha256={str(SHAPES.relative_to(ROOT)):hashlib.sha256(SHAPES.read_bytes()).hexdigest()},
                worst_convergence=max(c['convergence'] for c in rows))
    if result['worst_convergence']>1e-10:
        raise RuntimeError('independent basis reference did not converge')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def fixture_basis(c):
    with mp.workdps(70):
        v=mp.matrix([[mp.mpf(str(x)) for x in p] for p in c['vertices']])
        q=mp.matrix([mp.mpf(str(x)) for x in c['query']])
        table={tuple(r['alpha']):mp.mpf(value) for r,value in zip(c['requests'],c['values']) if not r['derivative']}
        matrix=mp.matrix([[v[i+1,j]-v[0,j] for j in range(3)] for i in range(3)])
        result=[]
        for k in range(4):
            f=[mp.mpf(int(k==j)) for j in range(4)]
            grad=mp.lu_solve(matrix,mp.matrix([f[i+1]-f[0] for i in range(3)]))
            center=f[0]+sum(grad[j]*(q[j]-v[0,j]) for j in range(3))
            result.append(float(center*table[(0,0,0)]+sum(grad[j]*table[tuple(int(i==j) for i in range(3))] for j in range(3))))
        return np.array(result)


def launch_inputs(api,q,h,tet,fields,geometry,device,operator):
    ntet=geometry.data.shape[0] if hasattr(geometry,'data') else geometry.vertices.shape[0]
    indices=np.asarray(tet)
    if indices.shape!=(len(q),) or not np.isfinite(indices).all() or np.any(indices<0) or np.any(indices>=ntet) or np.any(indices!=np.floor(indices)):
        raise ValueError('one valid tetrahedron index required per pair')
    if operator=='p1':
        arrays=inputs(api,q,h,fields,tet,device)
        out=wp.empty(len(q),dtype=api.real,device=device)
        return api.kernel,arrays+[geometry,out],out
    q,h=np.asarray(q,float),np.asarray(h,float)
    if h.ndim!=1 or q.shape!=(len(h),3) or not np.isfinite(q).all() or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError('finite [pair,3] positions and positive supports required')
    arrays=[wp.array(q,dtype=api.vec,device=device),wp.array(h,dtype=api.real,device=device),
            wp.array(indices,dtype=int,device=device)]
    if operator=='occupancy':
        out=wp.empty(len(q),dtype=api.real,device=device)
        return api.occupancy_kernel,arrays[:3]+[geometry,out],out
    data=np.asarray(fields,float)
    if data.ndim!=3 or data.shape[0]!=ntet or data.shape[2]!=4 or data.shape[1]<1 or not np.isfinite(data).all():
        raise ValueError('finite [tet,component,4] nodal coefficients required')
    fields_array=wp.array(data,dtype=api.field_vec,device=device)
    out=wp.empty((data.shape[1],len(q)),dtype=api.real,device=device)
    return api.fields_kernel,arrays[:3]+[fields_array,geometry,out],out


def evaluate(api,geometry,cs,fields,device,operator):
    kernel,args,out=launch_inputs(api,[c['query'] for c in cs],[c['h'] for c in cs],
        np.arange(len(cs)),np.asarray(fields),geometry,device,operator)
    wp.launch(kernel,len(cs),inputs=args,device=device)
    values=out.numpy().copy()
    return values.T if operator=='fields' else values


def accuracy(api,layout,reload,device,reference):
    rows=[]
    for name,cs,basis in [('fixtures',cases(3),None),('independent_shapes',reference['rows'],None)]:
        basis=np.asarray([fixture_basis(c) for c in cs]) if name=='fixtures' else np.asarray([c['basis'] for c in cs])
        geometry=impl.prepare_geometry([c['vertices'] for c in cs],device,layout=layout)
        rng=np.random.default_rng(319)
        values=rng.normal(size=(len(cs),28,4));values[:,0]=1;values[:,1]=0
        # Include every individual FEM basis and arbitrary signed components.
        values[:,2:6]=np.eye(4)
        for operator,nfields in [('occupancy',1),('p1',1),('fields',1),('fields',3),('fields',12),('fields',28)]:
            f=values[:,7] if operator=='p1' else values[:,:nfields]
            actual=evaluate(api,geometry,cs,f,device,operator)
            expected=basis.sum(axis=1) if operator=='occupancy' else (np.sum(basis*f,axis=1) if operator=='p1' else np.einsum('na,nka->nk',basis,f))
            error=abs(actual-expected)/(1+abs(expected))
            rows.append(dict(reference=name,operator=operator,components=nfields,cases=len(cs),
                             outputs=int(actual.size),all_finite=bool(np.isfinite(actual).all()),
                             worst_mixed_error=float(error.max())))
    passed=all(r['all_finite'] and r['worst_mixed_error']<=1e-10 for r in rows)
    return dict(passed=passed,budget=1e-10,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=('separate','packed'),default='packed')
    p.add_argument('--reload',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--generate-reference',action='store_true')
    p.add_argument('--accuracy-only',action='store_true')
    p.add_argument('--counts',default='512,131072,1048576')
    p.add_argument('--profile-operator',choices=('p1','occupancy','fields'))
    p.add_argument('--profile-fields',type=int,default=28)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    counts=[int(n) for n in a.counts.split(',')]
    if any(n<1 or n>1048576 for n in counts) or (a.profile_operator and len(counts)!=1):p.error('invalid counts')
    if a.generate_reference:generate_reference(a.reference)
    if not a.reference.exists():p.error('generate the independent basis reference first')
    reference=json.loads(a.reference.read_text())
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);api=impl.functions(layout=a.layout,reload=a.reload)
        paths=list((ROOT/'scripts/studies').glob('fluxbridge_shared_jk*.py'))+[SHAPES,a.reference]
        report=dict(device=str(device),device_name=device.name,layout=a.layout,reload=a.reload,
                    precision='float64',rows=[],properties={},
                    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                    limitations=['Experimental library kernels; no solver integration or 28-output derivative contract',
                        'Fields kernel consumes [tet,component,4] shared P1 coefficients, scalar kernel still accepts per-pair nodal values',
                        'Four moments computed once per pair; one component contracted/stored at a time; no global primitive/moment scratch',
                        'Fields output is component-major [component,pair] for coalesced stores; reference comparisons transpose that view',
                        'Static geometry prepared on host in fp64; JIT and static geometry excluded from warm/transfer timings',
                        'Independent CPU quadrature only generates validation references, never used by the measured kernels',
                        'GPU serialized; clocks/caches/desktop activity uncontrolled'])
        save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for op in ('p1','occupancy','fields'):(a.output/(op+'-native.cpp')).write_text(impl.source(layout=a.layout,reload=a.reload,operator=op))
        if not a.profile_operator:
            report['accuracy']=accuracy(api,a.layout,a.reload,device,reference);save()
            print('accuracy',report['accuracy'],flush=True)
            if not report['accuracy']['passed']:raise RuntimeError('layout accuracy gate failed')
        if not a.accuracy_only:
            tet=np.vstack([np.zeros(3),2*np.eye(3)])
            geometry,setup=timed(lambda:impl.prepare_geometry([tet],device,layout=a.layout),device,1)
            report['static_setup']=setup;report['static_geometry_bytes']=impl.geometry_bytes(geometry)
            ops=[(a.profile_operator,a.profile_fields if a.profile_operator=='fields' else 1)] if a.profile_operator else [('p1',1),('occupancy',1),('fields',1),('fields',3),('fields',12),('fields',28)]
            for count in counts:
                rng=np.random.default_rng(852);q=.2+rng.uniform(-.015,.015,(count,3));h=np.ones(count);indices=np.zeros(count,int)
                for op,nfields in ops:
                    f=(np.broadcast_to([.35,1.1,-.2,.7],(count,4)).copy() if op=='p1'
                       else np.random.default_rng(317).normal(size=(1,nfields,4)))
                    kernel,args,out=launch_inputs(api,q,h,indices,f,geometry,device,op)
                    action=lambda:wp.launch(kernel,count,inputs=args,device=device)
                    action();wp.synchronize_device(device)
                    row=dict(pairs=count,operator=op,components=nfields,
                        input_output_bytes=sum(x.capacity for x in args if isinstance(x,wp.array)),global_pair_scratch_bytes=0)
                    if device.is_cuda:
                        row['kernel_properties']=wp.get_cuda_kernel_properties(kernel,device)
                    if a.profile_operator:
                        wp.cuda_profiler_start(device);action();wp.synchronize_device(device);wp.cuda_profiler_stop(device)
                    else:
                        _,row['warm']=timed(action,device,7,events=device.is_cuda)
                        def transferred():
                            k,new,dst=launch_inputs(api,q,h,indices,f,geometry,device,op)
                            wp.launch(k,count,inputs=new,device=device)
                            return dst.numpy()
                        values,row['upload_kernel_readback']=timed(transferred,device,3)
                        if not np.isfinite(values).all():raise RuntimeError('nonfinite benchmark output')
                    report['rows'].append(row);save();print(op,nfields,count,row.get('warm'),row.get('kernel_properties'),flush=True)


if __name__=='__main__':main()
