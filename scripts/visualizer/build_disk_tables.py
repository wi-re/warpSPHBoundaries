"""Build seven-invariant browser disk tables from split circle-flux quadrature.

Same radius/offset panel maps as edge.disktables; includes second moments.
The shipped tables are approximations, not exact disk formulas. Rebuild offline.
"""
from pathlib import Path
import json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];STATIC=ROOT/'src/warpSPHBoundaries/visualizer/static'
CONFIG=(3,3,12,12,6,3,12,12)

def samples(kernel,D,R):
    script=f"import {{diskChannels}} from '{(STATIC/'browser-compute2d.js').as_uri()}'; let s='';for await(const b of process.stdin)s+=b;console.log(JSON.stringify(JSON.parse(s).map(([D,R])=>{{const v=diskChannels([D,0],R,'{kernel}',32);return [v[0],v[1],v[5],v[3],v[6],v[11],v[8]];}})));"
    shape=D.shape
    raw=subprocess.run(['node','--input-type=module','--eval',script],input=json.dumps(np.column_stack([D.ravel(),R.ravel()]).tolist()),text=True,capture_output=True,check=True)
    return np.array(json.loads(raw.stdout)).reshape(*shape,7)

def panel(kernel,pa,pb,na,nb,fn):
    a=np.cos(np.pi*(np.arange(na)+.5)/na);b=np.cos(np.pi*(np.arange(nb)+.5)/nb)
    IA=np.linalg.inv(np.polynomial.chebyshev.chebvander(a,na-1));IB=np.linalg.inv(np.polynomial.chebyshev.chebvander(b,nb-1))
    A,B=np.meshgrid(np.arange(pa),np.arange(pb),indexing='ij')
    aa=(A[...,None,None]+.5*(a[None,None,:,None]+1))/pa
    bb=(B[...,None,None]+.5*(b[None,None,None,:]+1))/pb
    aa,bb=np.broadcast_arrays(aa,bb);D,R,scale=fn(aa,bb)
    values=samples(kernel,D,R)/scale[...,None]
    return np.einsum('ka,ijabc,lb->ijklc',IA,values,IB)

def main():
    manifest={'config':CONFIG,'channels':['mass','m1n','m2t','m2n','Gn','g1t','g1n'],'source':'32-node split Gauss circle-flux quadrature; tensor Chebyshev interpolation; coefficients float32','kernels':{}}
    for kernel in ['w2','w4','w6','cubic']:
        S=[]
        for branch in (0,1):
            for k in range(3):
                def fn(w,rr,branch=branch,k=k):
                    R=(.5*rr if branch==0 else .5+.5*rr)
                    intervals=[(-R,0*R),(0*R,1-2*R),(1-2*R,1+0*R)] if branch==0 else [(-R,1-2*R),(1-2*R,0*R),(0*R,1+0*R)]
                    lo,hi=intervals[k];u=np.sin(np.pi*w/2)**2;delta=lo+u*(hi-lo)
                    return R+delta,R,R*R
                S.append(panel(kernel,*CONFIG[:4],fn))
        def large(w,s):
            R=1/s;return R-1+2*w,R,np.ones_like(R)
        L=panel(kernel,*CONFIG[4:],large)
        data=np.concatenate([np.stack(S).ravel(),L.ravel()]).astype('<f4')
        filename=f'disk-table-{kernel}.bin';(STATIC/filename).write_bytes(data.tobytes())
        manifest['kernels'][kernel]={'file':filename,'floats':len(data)}
        print(kernel,len(data),'floats',flush=True)
    (STATIC/'disk-tables.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
