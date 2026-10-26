"""Pointwise arithmetic/elementary-function checks for the two-float probe."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import fluxbridge_shared_jk_floatpair as impl

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'max_unroll':1})


def kernel(fast,force_inline):
    code=impl.transform('using T=float;\nT ih=T(1)/h;\nT C=T(1);',True,fast,force_inline)
    header=code[:code.index('T h_pair=T(h);')]
    body=header+'''
T a=T(x[0],x[1]),b=T(y[0],y[1]);
T result[7]={a+b,a-b,a*b,a/b,sqrtf(fabsf(a)),logf(T(1)+fabsf(a)),atan2f(b,a)};
for(int j=0;j<7;++j){
 *wp::address(output,row,2*j)=result[j].hi;
 *wp::address(output,row,2*j+1)=result[j].lo;
}
return 0.0f;
'''

    @wp.func_native(body)
    def arithmetic(x:wp.vec2f,y:wp.vec2f,output:wp.array2d(dtype=wp.float32),row:int)->wp.float32: ...

    @wp.kernel
    def check(x:wp.array(dtype=wp.vec2f),y:wp.array(dtype=wp.vec2f),output:wp.array2d(dtype=wp.float32)):
        i=wp.tid();unused=arithmetic(x[i],y[i],output,i)

    return check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(761);n=8192
    x=rng.uniform(-2,2,n)*np.exp2(rng.integers(-20,21,n));y=rng.uniform(-2,2,n)*np.exp2(rng.integers(-20,21,n))
    def split(v):
        hi=v.astype(np.float32);lo=(v-hi.astype(float)).astype(np.float32);return np.stack([hi,lo],axis=1)
    xs,ys=split(x),split(y);x=xs.astype(float).sum(axis=1);y=ys.astype(float).sum(axis=1)
    reference=np.stack([x+y,x-y,x*y,x/y,np.sqrt(abs(x)),np.log1p(abs(x)),np.arctan2(y,x)],axis=1)
    with open('/tmp/fluxbridge-gpu.lock','a') as lock:
        if a.device.startswith('cuda'):fcntl.flock(lock,fcntl.LOCK_EX)
        wp.init();device=wp.get_device(a.device);xx=wp.array(xs,dtype=wp.vec2f,device=device);yy=wp.array(ys,dtype=wp.vec2f,device=device)
        report=dict(device=str(device),cases=n,budget=2e-12,operations=['add','subtract','multiply','divide','sqrt_abs','log1p_abs','atan2'],rows=[],
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(impl.__file__)]})
        for fast,force in [(False,False),(True,False),(True,True)]:
            output=wp.empty((n,14),dtype=wp.float32,device=device);wp.launch(kernel(fast,force),n,inputs=[xx,yy,output],device=device)
            raw=output.numpy().astype(float);actual=raw[:,::2]+raw[:,1::2];error=abs(actual-reference)/(1+abs(reference))
            row=dict(fast=fast,force_inline=force,finite=bool(np.isfinite(actual).all()),worst_by_operation=error.max(axis=0).tolist(),failed=int((error>2e-12).sum()))
            report['rows'].append(row);print(row,flush=True)
        report['passed']=all(r['finite'] and not r['failed'] for r in report['rows'])
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if not report['passed']:raise RuntimeError('pointwise arithmetic gate failed')


if __name__=='__main__':main()
